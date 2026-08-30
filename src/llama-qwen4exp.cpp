#include "llama-qwen4exp.h"

#include "llama-context.h"
#include "llama-hparams.h"
#include "llama-impl.h"
#include "llama-model.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cstdint>
#include <map>
#include <unordered_set>
#include <utility>
#include <vector>

#if defined(__gnu_linux__)
#include <unistd.h>
#endif

namespace {

struct qwen4exp_base_desc {
    ggml_tensor * state;
    int64_t offset;
    int64_t width;
};

static bool qwen4exp_is_target_context(const llama_context & ctx) {
    return ctx.model.arch == LLM_ARCH_QWEN4EXP &&
            ctx.cparams.mtp_op_type == MTP_OP_NONE &&
            llama_model_mtp_package(&ctx.model) != LLAMA_MTP_PACKAGE_COMPANION;
}

// Per-step checkpoint writers partition every logical step into contiguous
// work units. Keep each node's units local to the threads that fill them. Bind
// only complete pages so a tensor boundary shared with a base shadow remains
// on the node where it was first touched.
static size_t qwen4exp_numa_bind_step_tensor(
        ggml_tensor * tensor,
        int32_t n_steps,
        uint32_t n_threads,
        size_t unit_bytes) {
#if defined(__gnu_linux__)
    if (!ggml_numa_mirror_active() || tensor == nullptr || n_steps <= 0 || n_threads == 0 ||
            unit_bytes == 0 ||
            tensor->data == nullptr || tensor->buffer == nullptr ||
            !ggml_backend_buffer_is_host(tensor->buffer)) {
        return 0;
    }

    const int32_t n_nodes = ggml_numa_node_count();
    const long page_size_value = sysconf(_SC_PAGESIZE);
    const size_t n_bytes = ggml_nbytes(tensor);
    if (n_nodes <= 1 || page_size_value <= 0 || n_bytes % (size_t) n_steps != 0) {
        return 0;
    }

    const size_t page_size = (size_t) page_size_value;
    const size_t step_bytes = n_bytes / (size_t) n_steps;
    if (step_bytes % unit_bytes != 0) {
        return 0;
    }
    const size_t work_units = step_bytes / unit_bytes;
    const size_t units_per_thread = (work_units + n_threads - 1) / n_threads;
    const uintptr_t tensor_begin = (uintptr_t) tensor->data;
    size_t bound_bytes = 0;

    for (int32_t step = 0; step < n_steps; ++step) {
        const size_t step_begin = (size_t) step * step_bytes;
        for (int32_t node = 0; node < n_nodes; ++node) {
            const size_t first_thread =
                    ((size_t) node * n_threads + (size_t) n_nodes - 1) / (size_t) n_nodes;
            const size_t last_thread =
                    ((size_t) (node + 1) * n_threads + (size_t) n_nodes - 1) / (size_t) n_nodes;
            const size_t first_unit = std::min(first_thread * units_per_thread, work_units);
            const size_t last_unit = std::min(last_thread * units_per_thread, work_units);
            uintptr_t begin = tensor_begin + step_begin + first_unit * unit_bytes;
            uintptr_t end = tensor_begin + step_begin + last_unit * unit_bytes;
            const uintptr_t begin_remainder = begin % page_size;
            if (begin_remainder != 0) {
                begin += page_size - begin_remainder;
            }
            end -= end % page_size;
            if (begin >= end) {
                continue;
            }
            ggml_numa_bind((void *) begin, (size_t) (end - begin), node);
            bound_bytes += (size_t) (end - begin);
        }
    }
    return bound_bytes;
#else
    (void) tensor;
    (void) n_steps;
    (void) n_threads;
    (void) unit_bytes;
    return 0;
#endif
}

static size_t qwen4exp_numa_bind_step_checkpoints(
        llama_context & ctx,
        uint32_t n_threads) {
    auto & ckpt = ctx.kv_self.ckpt;
    const llama_hparams & hparams = ctx.model.hparams;
    const size_t head_v = (size_t) hparams.ssm_d_inner / (size_t) hparams.ssm_dt_rank;
    const size_t ssm_unit_bytes = head_v * head_v * sizeof(float);
    const size_t conv_unit_bytes = (size_t) (hparams.ssm_d_conv - 1) * sizeof(float);
    size_t bound_bytes = 0;
    for (const auto & layer : ckpt.per_step_ssm) {
        for (ggml_tensor * tensor : layer) {
            bound_bytes += qwen4exp_numa_bind_step_tensor(
                    tensor, ckpt.per_step_max_allocated - 1, n_threads, ssm_unit_bytes);
        }
    }
    for (const auto & layer : ckpt.per_step_conv) {
        for (ggml_tensor * tensor : layer) {
            bound_bytes += qwen4exp_numa_bind_step_tensor(
                    tensor, ckpt.per_step_max_allocated, n_threads, conv_unit_bytes);
        }
    }
    for (ggml_tensor * tensor : ckpt.qwen4exp_per_step_ple) {
        bound_bytes += qwen4exp_numa_bind_step_tensor(
                tensor, ckpt.qwen4exp_per_step_max_tokens - 1, n_threads, sizeof(float));
    }
    return bound_bytes;
}

static void qwen4exp_numa_ensure_step_checkpoint_layout(llama_context & ctx) {
    auto & ckpt = ctx.kv_self.ckpt;
    const uint32_t n_threads = ctx.cparams.n_threads_batch;
    if (!ggml_numa_mirror_active() || n_threads == 0 ||
            ckpt.qwen4exp_per_step_numa_threads == n_threads) {
        return;
    }
    if (ctx.sched != nullptr) {
        ggml_backend_sched_synchronize(ctx.sched);
    }
    const size_t requested_bytes = qwen4exp_numa_bind_step_checkpoints(ctx, n_threads);
    ckpt.qwen4exp_per_step_numa_threads = n_threads;
    if (requested_bytes > 0) {
        LLAMA_LOG_INFO("%s: requested NUMA-local placement for %8.2f MiB of per-step checkpoint pages "
                "across %d nodes using %u threads\n",
                __func__, requested_bytes / (1024.0 * 1024.0),
                ggml_numa_node_count(), n_threads);
    }
}

static bool qwen4exp_batch_seq_id(
        const llama_batch & batch,
        int32_t i,
        llama_seq_id & seq_id) {
    if (batch.n_seq_id == nullptr && batch.seq_id == nullptr) {
        seq_id = batch.all_seq_id;
        return true;
    }
    if (batch.n_seq_id == nullptr || batch.seq_id == nullptr ||
            batch.n_seq_id[i] != 1 || batch.seq_id[i] == nullptr) {
        return false;
    }
    seq_id = batch.seq_id[i][0];
    return true;
}

static llama_pos qwen4exp_batch_pos(const llama_batch & batch, int32_t i) {
    return batch.pos != nullptr
            ? batch.pos[i]
            : batch.all_pos_0 + (llama_pos) i * batch.all_pos_1;
}

static bool qwen4exp_collect_base_rows(
        llama_context & ctx,
        std::vector<qwen4exp_base_desc> & rows) {
    const llama_hparams & hparams = ctx.model.hparams;
    const auto & s_l = ctx.kv_self.s_l;
    rows.clear();

    for (int32_t il = 0; il < (int32_t) s_l.size(); ++il) {
        ggml_tensor * state = s_l[il];
        if (state == nullptr) {
            continue;
        }
        if (state->type != GGML_TYPE_F32 || state->buffer == nullptr ||
                state->data == nullptr || state->nb[0] != sizeof(float) || state->ne[1] <= 0) {
            LLAMA_LOG_ERROR("%s: invalid Qwen4Exp state tensor in layer %d\n", __func__, il);
            return false;
        }

        const int64_t ple_width = hparams.n_embd_ple_conv(il);
        if (hparams.is_recurrent(il)) {
            if (state->extra == nullptr) {
                rows.push_back({ state, 0, state->ne[0] });
                continue;
            }

            const auto * split = (const ggml_split_tensor_t *) state->extra;
            for (int32_t id = 0; id < split->n_device; ++id) {
                ggml_tensor * split_state = split->splits[id];
                if (split_state == nullptr) {
                    continue;
                }
                if (split_state->type != GGML_TYPE_F32 || split_state->buffer == nullptr ||
                        split_state->data == nullptr || split_state->nb[0] != sizeof(float) ||
                        split_state->ne[1] != state->ne[1]) {
                    LLAMA_LOG_ERROR("%s: invalid split Qwen4Exp state tensor in layer %d device %d\n",
                            __func__, il, id);
                    return false;
                }
                rows.push_back({ split_state, 0, split_state->ne[0] });
            }

            // Split GDN state lives on the device shards, while PLE continues to
            // use the tail of the primary row.
            if (ple_width > 0) {
                if (ple_width > state->ne[0]) {
                    return false;
                }
                rows.push_back({ state, state->ne[0] - ple_width, ple_width });
            }
        } else if (ple_width > 0) {
            if (ple_width != state->ne[0]) {
                LLAMA_LOG_ERROR("%s: invalid PLE-only state width in layer %d\n", __func__, il);
                return false;
            }
            rows.push_back({ state, 0, ple_width });
        }
    }

    return !rows.empty();
}

static ggml_tensor qwen4exp_tensor_slice(
        ggml_tensor * tensor,
        int64_t offset,
        int64_t width,
        int64_t row) {
    ggml_tensor result = *tensor;
    result.ne[0] = width;
    result.ne[1] = result.ne[2] = result.ne[3] = 1;
    result.nb[0] = sizeof(float);
    result.nb[1] = (size_t) width * sizeof(float);
    result.nb[2] = result.nb[3] = result.nb[1];
    result.data = (char *) tensor->data + (size_t) row * tensor->nb[1] +
            (size_t) offset * sizeof(float);
    result.view_src = nullptr;
    result.view_offs = 0;
    result.extra = nullptr;
    return result;
}

static bool qwen4exp_copy_base(llama_context & ctx, bool restore) {
    auto & ckpt = ctx.kv_self.ckpt;
    if (!ckpt.qwen4exp_per_step_allocated ||
            ckpt.qwen4exp_per_step_base_rows.empty() ||
            ckpt.qwen4exp_per_step_seq_id < 0) {
        return false;
    }

    for (const auto & item : ckpt.qwen4exp_per_step_base_rows) {
        if (item.state == nullptr || item.shadow == nullptr || item.width <= 0 ||
                item.offset < 0 || item.offset + item.width > item.state->ne[0] ||
                ckpt.qwen4exp_per_step_seq_id >= item.state->ne[1] ||
                item.state->type != GGML_TYPE_F32 || item.shadow->type != GGML_TYPE_F32 ||
                item.state->buffer == nullptr || item.shadow->buffer == nullptr ||
                item.state->nb[0] != sizeof(float) || item.shadow->nb[0] != sizeof(float) ||
                item.state->data == nullptr || item.shadow->data == nullptr ||
                ggml_nelements(item.shadow) < item.width) {
            return false;
        }
    }

    // Validate every row before the first write so a malformed checkpoint cannot
    // leave only part of the live recurrent state restored.
    ggml_backend_sched_synchronize(ctx.sched);
    std::unordered_set<ggml_tensor *> touched;
    for (const auto & item : ckpt.qwen4exp_per_step_base_rows) {
        ggml_tensor state = qwen4exp_tensor_slice(
                item.state, item.offset, item.width, ckpt.qwen4exp_per_step_seq_id);
        ggml_backend_tensor_copy(restore ? item.shadow : &state, restore ? &state : item.shadow);
        if (restore) {
            touched.insert(item.state);
        }
    }

    for (ggml_tensor * state : touched) {
        ggml_numa_tensor_resync(state);
    }
    return true;
}

static bool qwen4exp_step_slice_valid(
        ggml_tensor * source,
        int64_t source_offset,
        ggml_tensor * state,
        int64_t state_offset,
        int64_t width,
        llama_seq_id seq_id) {
    return source != nullptr && state != nullptr && source->type == GGML_TYPE_F32 &&
            state->type == GGML_TYPE_F32 && source->buffer != nullptr && state->buffer != nullptr &&
            source->data != nullptr && state->data != nullptr &&
            source->nb[0] == sizeof(float) && state->nb[0] == sizeof(float) &&
            width > 0 && source_offset >= 0 &&
            source_offset + width <= ggml_nelements(source) && state_offset >= 0 &&
            state_offset + width <= state->ne[0] && seq_id >= 0 && seq_id < state->ne[1];
}

static bool qwen4exp_copy_step_slice(
        ggml_tensor * source,
        int64_t source_offset,
        ggml_tensor * state,
        int64_t state_offset,
        int64_t width,
        llama_seq_id seq_id) {
    if (!qwen4exp_step_slice_valid(
            source, source_offset, state, state_offset, width, seq_id)) {
        return false;
    }

    ggml_tensor src = qwen4exp_tensor_slice(source, source_offset, width, 0);
    ggml_tensor dst = qwen4exp_tensor_slice(state, state_offset, width, seq_id);
    ggml_backend_tensor_copy(&src, &dst);
    return true;
}

static bool qwen4exp_restore_gdn_step(
        llama_context & ctx,
        int accepted_step,
        std::unordered_set<ggml_tensor *> & touched,
        bool validate_only) {
    auto & kv = ctx.kv_self;
    auto & ckpt = kv.ckpt;
    const llama_hparams & hparams = ctx.model.hparams;
    const int64_t conv_state_dim = ckpt.per_step_conv_state_dim;
    const int64_t ssm_state_dim = ckpt.per_step_ssm_state_size;
    const int32_t d_conv = ckpt.per_step_d_conv;
    if (conv_state_dim <= 0 || ssm_state_dim <= 0 || d_conv <= 1) {
        return false;
    }

    const int32_t num_v_heads = hparams.ssm_dt_rank;
    const int32_t head_v_dim = hparams.ssm_d_inner / num_v_heads;
    for (int32_t il = 0; il < (int32_t) kv.s_l.size(); ++il) {
        ggml_tensor * state = kv.s_l[il];
        if (state == nullptr || !hparams.is_recurrent(il)) {
            continue;
        }
        if (il >= (int32_t) ckpt.per_step_ssm.size() ||
                il >= (int32_t) ckpt.per_step_conv.size()) {
            return false;
        }

        if (state->extra == nullptr) {
            if (ckpt.per_step_ssm[il].size() != 1 || ckpt.per_step_conv[il].size() != 1 ||
                    !(validate_only ? qwen4exp_step_slice_valid(
                        ckpt.per_step_conv[il][0], (int64_t) accepted_step * conv_state_dim,
                        state, 0, conv_state_dim, ckpt.qwen4exp_per_step_seq_id) :
                        qwen4exp_copy_step_slice(
                        ckpt.per_step_conv[il][0], (int64_t) accepted_step * conv_state_dim,
                        state, 0, conv_state_dim, ckpt.qwen4exp_per_step_seq_id)) ||
                    !(validate_only ? qwen4exp_step_slice_valid(
                        ckpt.per_step_ssm[il][0], (int64_t) accepted_step * ssm_state_dim,
                        state, conv_state_dim, ssm_state_dim, ckpt.qwen4exp_per_step_seq_id) :
                        qwen4exp_copy_step_slice(
                        ckpt.per_step_ssm[il][0], (int64_t) accepted_step * ssm_state_dim,
                        state, conv_state_dim, ssm_state_dim, ckpt.qwen4exp_per_step_seq_id))) {
                return false;
            }
            if (!validate_only) {
                touched.insert(state);
            }
            continue;
        }

        const auto * split_state = (const ggml_split_tensor_t *) state->extra;
        const auto * split_out = (const ggml_split_tensor_t *) ctx.model.layers[il].ssm_out->extra;
        if (split_out == nullptr ||
                ckpt.per_step_ssm[il].size() != (size_t) split_state->n_device ||
                ckpt.per_step_conv[il].size() != (size_t) split_state->n_device) {
            return false;
        }
        for (int32_t id = 0; id < split_state->n_device; ++id) {
            ggml_tensor * shard = split_state->splits[id];
            if (shard == nullptr) {
                continue;
            }
            if (id >= split_out->n_device || split_out->splits[id] == nullptr ||
                    split_out->splits[id]->ne[0] % head_v_dim != 0) {
                return false;
            }
            const int32_t nv = split_out->splits[id]->ne[0] / head_v_dim;
            const auto [split_conv_dim, split_ssm_dim] = hparams.n_embd_v_s_dims(nv);
            const int64_t split_conv_state_dim = (int64_t) (d_conv - 1) * split_conv_dim;
            if (!(validate_only ? qwen4exp_step_slice_valid(
                        ckpt.per_step_conv[il][id], (int64_t) accepted_step * split_conv_state_dim,
                        shard, 0, split_conv_state_dim, ckpt.qwen4exp_per_step_seq_id) :
                    qwen4exp_copy_step_slice(
                        ckpt.per_step_conv[il][id], (int64_t) accepted_step * split_conv_state_dim,
                        shard, 0, split_conv_state_dim, ckpt.qwen4exp_per_step_seq_id)) ||
                    !(validate_only ? qwen4exp_step_slice_valid(
                        ckpt.per_step_ssm[il][id], (int64_t) accepted_step * split_ssm_dim,
                        shard, split_conv_state_dim, split_ssm_dim, ckpt.qwen4exp_per_step_seq_id) :
                    qwen4exp_copy_step_slice(
                        ckpt.per_step_ssm[il][id], (int64_t) accepted_step * split_ssm_dim,
                        shard, split_conv_state_dim, split_ssm_dim, ckpt.qwen4exp_per_step_seq_id))) {
                return false;
            }
            if (!validate_only) {
                touched.insert(shard);
            }
        }
    }
    return true;
}

static bool qwen4exp_restore_ple_step(
        llama_context & ctx,
        int accepted_step,
        std::unordered_set<ggml_tensor *> & touched,
        bool validate_only) {
    auto & ckpt = ctx.kv_self.ckpt;
    const llama_hparams & hparams = ctx.model.hparams;
    for (int32_t il = 0; il < (int32_t) ctx.kv_self.s_l.size(); ++il) {
        const int64_t ple_width = hparams.n_embd_ple_conv(il);
        if (ple_width <= 0) {
            continue;
        }
        ggml_tensor * state = ctx.kv_self.s_l[il];
        if (state == nullptr || il >= (int32_t) ckpt.qwen4exp_per_step_ple.size() ||
                !(validate_only ? qwen4exp_step_slice_valid(
                    ckpt.qwen4exp_per_step_ple[il], (int64_t) accepted_step * ple_width,
                    state, state->ne[0] - ple_width, ple_width,
                    ckpt.qwen4exp_per_step_seq_id) :
                qwen4exp_copy_step_slice(
                    ckpt.qwen4exp_per_step_ple[il], (int64_t) accepted_step * ple_width,
                    state, state->ne[0] - ple_width, ple_width,
                    ckpt.qwen4exp_per_step_seq_id))) {
            return false;
        }
        if (!validate_only) {
            touched.insert(state);
        }
    }
    return true;
}

} // namespace

void llama_kv_cache::gpu_checkpoint::release_qwen4exp_per_step() {
    for (ggml_context * ctx : qwen4exp_per_step_ctxs) {
        ggml_free(ctx);
    }
    for (ggml_backend_buffer_t buffer : qwen4exp_per_step_bufs) {
        ggml_backend_buffer_free(buffer);
    }
    qwen4exp_per_step_base_rows.clear();
    qwen4exp_per_step_ple.clear();
    qwen4exp_per_step_ctxs.clear();
    qwen4exp_per_step_bufs.clear();
    qwen4exp_per_step_allocated = false;
    qwen4exp_per_step_saved = false;
    qwen4exp_per_step_captured = false;
    qwen4exp_per_step_invalid = false;
    qwen4exp_per_step_max_tokens = 0;
    qwen4exp_per_step_n_tokens = 0;
    qwen4exp_per_step_numa_threads = 0;
    qwen4exp_per_step_seq_id = -1;
    qwen4exp_per_step_first_pos = -1;
    qwen4exp_per_step_base_bytes = 0;
    qwen4exp_per_step_delta_bytes = 0;
}

bool llama_qwen4exp_spec_ckpt_prepare(llama_context * ctx, int max_tokens) {
    if (ctx == nullptr || ctx->model.arch != LLM_ARCH_QWEN4EXP || max_tokens <= 0) {
        return false;
    }
    if (!qwen4exp_is_target_context(*ctx)) {
        LLAMA_LOG_WARN("%s: Qwen4Exp per-step checkpoints are unsupported on MTP companion contexts\n", __func__);
        return false;
    }
    if ((uint32_t) max_tokens > ctx->cparams.n_ubatch) {
        LLAMA_LOG_WARN("%s: Qwen4Exp per-step checkpoint capacity %d exceeds n_ubatch=%u\n",
                __func__, max_tokens, ctx->cparams.n_ubatch);
        return false;
    }

    auto & ckpt = ctx->kv_self.ckpt;
    if (ckpt.qwen4exp_per_step_allocated && ckpt.qwen4exp_per_step_max_tokens >= max_tokens) {
        return true;
    }

    // A previously built graph can still reference the old per-step tensors.
    // Finish it and evict the cached graph before either generic or
    // Qwen-specific storage is reallocated.
    if (ctx->sched != nullptr) {
        ggml_backend_sched_synchronize(ctx->sched);
        ctx->reset_scheduler();
    }

    const llama_hparams & hparams = ctx->model.hparams;
    if (hparams.ssm_dt_rank <= 0 || hparams.ssm_d_inner % hparams.ssm_dt_rank != 0 ||
            hparams.ssm_d_conv <= 1) {
        return false;
    }
    const int64_t head_v = hparams.ssm_d_inner / hparams.ssm_dt_rank;
    const int64_t key_dim = (int64_t) hparams.ssm_d_state * hparams.ssm_n_group;
    const int64_t value_dim = head_v * hparams.ssm_dt_rank;
    const int64_t conv_dim = 2 * key_dim + value_dim;
    std::vector<qwen4exp_base_desc> base_rows;
    if (!qwen4exp_collect_base_rows(*ctx, base_rows)) {
        return false;
    }

    ckpt.release_qwen4exp_per_step();
    ckpt.per_step_ssm_state_size = head_v * head_v * hparams.ssm_dt_rank;
    ckpt.per_step_conv_state_dim = (hparams.ssm_d_conv - 1) * conv_dim;
    ckpt.per_step_conv_dim = conv_dim;
    ckpt.per_step_d_conv = hparams.ssm_d_conv;
    if (!ctx->kv_self.per_step_alloc(ctx->model, max_tokens)) {
        return false;
    }

    ckpt.qwen4exp_per_step_base_rows.resize(base_rows.size());
    ckpt.qwen4exp_per_step_ple.resize(ctx->kv_self.s_l.size(), nullptr);

    struct alloc_entry {
        bool ple;
        size_t index;
        ggml_tensor * state;
        int64_t width;
    };
    std::map<ggml_backend_buffer_type_t, std::vector<alloc_entry>> entries;
    for (size_t i = 0; i < base_rows.size(); ++i) {
        const qwen4exp_base_desc & row = base_rows[i];
        entries[ggml_backend_buffer_get_type(row.state->buffer)].push_back({ false, i, row.state, row.width });
        ckpt.qwen4exp_per_step_base_bytes += (size_t) row.width * sizeof(float);
    }
    if (max_tokens > 1) {
        for (int32_t il = 0; il < (int32_t) ctx->kv_self.s_l.size(); ++il) {
            ggml_tensor * state = ctx->kv_self.s_l[il];
            const int64_t ple_width = hparams.n_embd_ple_conv(il);
            if (state == nullptr || ple_width <= 0) {
                continue;
            }
            entries[ggml_backend_buffer_get_type(state->buffer)].push_back({ true, (size_t) il, state, ple_width });
            ckpt.qwen4exp_per_step_delta_bytes +=
                    (size_t) (max_tokens - 1) * (size_t) ple_width * sizeof(float);
        }
    }

    for (auto & [buft, group] : entries) {
        ggml_init_params params = {
            /*.mem_size   =*/ group.size() * ggml_tensor_overhead(),
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        ggml_context * graph_ctx = ggml_init(params);
        if (graph_ctx == nullptr) {
            ckpt.release_qwen4exp_per_step();
            return false;
        }

        for (const alloc_entry & entry : group) {
            if (entry.ple) {
                ggml_tensor * steps = ggml_new_tensor_1d(graph_ctx, GGML_TYPE_F32,
                        (int64_t) (max_tokens - 1) * entry.width);
                ggml_format_name(steps, "qwen4exp_per_step_ple_l%zu", entry.index);
                ckpt.qwen4exp_per_step_ple[entry.index] = steps;
            } else {
                ggml_tensor * shadow = ggml_new_tensor_1d(graph_ctx, GGML_TYPE_F32, entry.width);
                ggml_format_name(shadow, "qwen4exp_per_step_base_%zu", entry.index);
                ckpt.qwen4exp_per_step_base_rows[entry.index] = {
                    base_rows[entry.index].state,
                    shadow,
                    base_rows[entry.index].offset,
                    base_rows[entry.index].width,
                };
            }
        }

        ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors_from_buft(graph_ctx, buft);
        if (buffer == nullptr) {
            ggml_free(graph_ctx);
            ckpt.release_qwen4exp_per_step();
            return false;
        }
        ggml_backend_buffer_set_usage(buffer, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
        ggml_backend_buffer_clear(buffer, 0);
        ckpt.qwen4exp_per_step_ctxs.push_back(graph_ctx);
        ckpt.qwen4exp_per_step_bufs.push_back(buffer);
    }

    size_t gdn_bytes = 0;
    for (ggml_backend_buffer_t buffer : ckpt.per_step_bufs) {
        gdn_bytes += ggml_backend_buffer_get_size(buffer);
    }
    ckpt.qwen4exp_per_step_delta_bytes += gdn_bytes;
    ckpt.qwen4exp_per_step_max_tokens = max_tokens;
    ckpt.qwen4exp_per_step_allocated = true;
    LLAMA_LOG_INFO("%s: Qwen4Exp per-step base=%8.2f MiB delta=%8.2f MiB max_tokens=%d\n",
            __func__, ckpt.qwen4exp_per_step_base_bytes / (1024.0 * 1024.0),
            ckpt.qwen4exp_per_step_delta_bytes / (1024.0 * 1024.0), max_tokens);
    return true;
}

bool llama_qwen4exp_spec_ckpt_save(llama_context * ctx, llama_seq_id seq_id) {
    if (ctx == nullptr || !qwen4exp_is_target_context(*ctx)) {
        return false;
    }
    auto & ckpt = ctx->kv_self.ckpt;
    ckpt.qwen4exp_per_step_saved = false;
    ckpt.qwen4exp_per_step_captured = false;
    ckpt.qwen4exp_per_step_invalid = false;
    ckpt.qwen4exp_per_step_n_tokens = 0;
    ckpt.qwen4exp_per_step_first_pos = -1;
    ckpt.qwen4exp_per_step_seq_id = seq_id;
    if (!ckpt.qwen4exp_per_step_allocated || seq_id < 0) {
        ckpt.qwen4exp_per_step_seq_id = -1;
        ckpt.qwen4exp_per_step_first_pos = -1;
        return false;
    }
    qwen4exp_numa_ensure_step_checkpoint_layout(*ctx);
    if (!qwen4exp_copy_base(*ctx, false)) {
        ckpt.qwen4exp_per_step_seq_id = -1;
        ckpt.qwen4exp_per_step_first_pos = -1;
        return false;
    }
    ckpt.qwen4exp_per_step_saved = true;
    ctx->kv_self.save_per_step_ssm = true;
    return true;
}

bool llama_qwen4exp_spec_ckpt_begin_capture(llama_context * ctx, const llama_batch & batch) {
    if (ctx == nullptr || ctx->model.arch != LLM_ARCH_QWEN4EXP) {
        return true;
    }
    auto & ckpt = ctx->kv_self.ckpt;
    if (ckpt.selected_spec_mode != LLAMA_SPEC_CKPT_PER_STEP || !ckpt.qwen4exp_per_step_saved) {
        return true;
    }
    if (ckpt.qwen4exp_per_step_captured) {
        LLAMA_LOG_ERROR("%s: Qwen4Exp checkpoint already has a completed verification batch\n",
                __func__);
        return false;
    }

    ckpt.qwen4exp_per_step_captured = false;
    ckpt.qwen4exp_per_step_invalid = true;
    ckpt.qwen4exp_per_step_n_tokens = 0;
    if (!qwen4exp_is_target_context(*ctx) || batch.n_tokens <= 0 ||
            batch.n_tokens > ckpt.qwen4exp_per_step_max_tokens ||
            (uint32_t) batch.n_tokens > ctx->cparams.n_ubatch) {
        LLAMA_LOG_ERROR("%s: unsupported Qwen4Exp checkpoint verification batch size %d\n",
                __func__, batch.n_tokens);
        return false;
    }

    const llama_pos first_pos = qwen4exp_batch_pos(batch, 0);
    for (int32_t i = 0; i < batch.n_tokens; ++i) {
        llama_seq_id seq_id = -1;
        if (!qwen4exp_batch_seq_id(batch, i, seq_id) ||
                seq_id != ckpt.qwen4exp_per_step_seq_id ||
                qwen4exp_batch_pos(batch, i) != first_pos + i) {
            LLAMA_LOG_ERROR("%s: Qwen4Exp per-step capture requires one saved sequence at contiguous positions\n",
                    __func__);
            return false;
        }
    }

    qwen4exp_numa_ensure_step_checkpoint_layout(*ctx);
    ckpt.qwen4exp_per_step_invalid = false;
    ckpt.qwen4exp_per_step_first_pos = first_pos;
    return true;
}

void llama_qwen4exp_spec_ckpt_finish_capture(llama_context * ctx, int32_t n_tokens) {
    if (ctx == nullptr || ctx->model.arch != LLM_ARCH_QWEN4EXP) {
        return;
    }
    auto & ckpt = ctx->kv_self.ckpt;
    if (ckpt.selected_spec_mode != LLAMA_SPEC_CKPT_PER_STEP || !ckpt.qwen4exp_per_step_saved) {
        return;
    }
    if (ckpt.qwen4exp_per_step_captured) {
        return;
    }
    if (ckpt.qwen4exp_per_step_invalid || n_tokens <= 0 ||
            n_tokens > ckpt.qwen4exp_per_step_max_tokens) {
        ckpt.qwen4exp_per_step_invalid = true;
        ctx->kv_self.save_per_step_ssm = false;
        return;
    }
    ckpt.qwen4exp_per_step_n_tokens = n_tokens;
    ckpt.qwen4exp_per_step_captured = true;
    ctx->kv_self.save_per_step_ssm = false;
}

enum llama_spec_ckpt_restore_result llama_qwen4exp_spec_ckpt_restore(
        llama_context * ctx,
        llama_seq_id seq_id,
        llama_pos n_past,
        int accepted_step) {
    if (ctx == nullptr || !qwen4exp_is_target_context(*ctx)) {
        return LLAMA_SPEC_CKPT_RESTORE_FAILED;
    }
    auto & ckpt = ctx->kv_self.ckpt;
    if (!ckpt.qwen4exp_per_step_allocated || !ckpt.qwen4exp_per_step_saved ||
            ckpt.qwen4exp_per_step_seq_id != seq_id || accepted_step < -1) {
        return LLAMA_SPEC_CKPT_RESTORE_FAILED;
    }

    if (ckpt.qwen4exp_per_step_invalid || ckpt.qwen4exp_per_step_first_pos < 0 ||
            n_past != ckpt.qwen4exp_per_step_first_pos) {
        return LLAMA_SPEC_CKPT_RESTORE_FAILED;
    }
    if (accepted_step < 0) {
        return qwen4exp_copy_base(*ctx, true)
                ? LLAMA_SPEC_CKPT_RESTORE_DIRECT
                : LLAMA_SPEC_CKPT_RESTORE_FAILED;
    }
    if (!ckpt.qwen4exp_per_step_captured ||
            accepted_step >= ckpt.qwen4exp_per_step_n_tokens) {
        return LLAMA_SPEC_CKPT_RESTORE_FAILED;
    }

    // The graph has already left the live state after the final row in place.
    if (accepted_step + 1 == ckpt.qwen4exp_per_step_n_tokens) {
        ggml_backend_sched_synchronize(ctx->sched);
        return LLAMA_SPEC_CKPT_RESTORE_DIRECT;
    }

    ggml_backend_sched_synchronize(ctx->sched);
    std::unordered_set<ggml_tensor *> touched;
    if (!qwen4exp_restore_gdn_step(*ctx, accepted_step, touched, true) ||
            !qwen4exp_restore_ple_step(*ctx, accepted_step, touched, true) ||
            !qwen4exp_restore_gdn_step(*ctx, accepted_step, touched, false) ||
            !qwen4exp_restore_ple_step(*ctx, accepted_step, touched, false)) {
        return LLAMA_SPEC_CKPT_RESTORE_FAILED;
    }
    for (ggml_tensor * state : touched) {
        ggml_numa_tensor_resync(state);
    }
    return LLAMA_SPEC_CKPT_RESTORE_DIRECT;
}

void llama_qwen4exp_spec_ckpt_discard(llama_context * ctx) {
    if (ctx == nullptr || ctx->model.arch != LLM_ARCH_QWEN4EXP) {
        return;
    }
    auto & ckpt = ctx->kv_self.ckpt;
    ckpt.qwen4exp_per_step_saved = false;
    ckpt.qwen4exp_per_step_captured = false;
    ckpt.qwen4exp_per_step_invalid = false;
    ckpt.qwen4exp_per_step_n_tokens = 0;
    ckpt.qwen4exp_per_step_seq_id = -1;
    ckpt.qwen4exp_per_step_first_pos = -1;
    ctx->kv_self.save_per_step_ssm = false;
}

ggml_tensor * llama_qwen4exp_spec_ckpt_ple(llama_context * ctx, int32_t il) {
    if (ctx == nullptr || ctx->model.arch != LLM_ARCH_QWEN4EXP ||
            !ctx->kv_self.save_per_step_ssm) {
        return nullptr;
    }
    const auto & ckpt = ctx->kv_self.ckpt;
    if (!ckpt.qwen4exp_per_step_allocated || !ckpt.qwen4exp_per_step_saved ||
            il < 0 || il >= (int32_t) ckpt.qwen4exp_per_step_ple.size()) {
        return nullptr;
    }
    return ckpt.qwen4exp_per_step_ple[il];
}
