#include "llama.h"

#include "llama-context.h"
#include "llama-model.h"
#include "llama-spec-features-dflash.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#define CHECK(condition) do { \
    if (!(condition)) { \
        std::fprintf(stderr, "check failed at %s:%d: %s\n", __FILE__, __LINE__, #condition); \
        std::abort(); \
    } \
} while (0)

static_assert(sizeof(llama_context_memory_info) == LLAMA_CONTEXT_MEMORY_INFO_STRUCT_SIZE_V1);

static const char * unsupported_backend_name(ggml_backend_t backend) {
    (void) backend;
    return "unsupported-memory-info-test";
}

static void unsupported_backend_free(ggml_backend_t backend) {
    delete backend;
}

static ggml_backend_t make_unsupported_backend() {
    static ggml_guid guid = {
        0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88,
        0x99, 0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xf0, 0x01,
    };

    ggml_backend_t backend = new ggml_backend {};
    backend->guid = &guid;
    backend->iface.get_name = unsupported_backend_name;
    backend->iface.free = unsupported_backend_free;
    return backend;
}

int main() {
    llama_context_memory_info info = {};
    info.struct_size = sizeof(info);

    CHECK(!llama_context_get_memory_info(nullptr, nullptr));
    CHECK(!llama_context_get_memory_info(nullptr, &info));
    CHECK(!llama_supports_full_state_io(nullptr));

    llama_model model = {};
    llama_context ctx(model);

    CHECK(llama_supports_full_state_io(&ctx));
    model.arch = LLM_ARCH_OPENPANGU;
    CHECK(!llama_supports_full_state_io(&ctx));
    ctx.kv_self.row_count.push_back(1);
    model.arch = LLM_ARCH_DEEPSEEK4;
    CHECK(llama_supports_full_state_io(&ctx));
    model.arch = LLM_ARCH_UNKNOWN;
    CHECK(!llama_supports_full_state_io(&ctx));
    // Release the test-only capacity so it does not affect the memory-accounting baseline below.
    std::vector<uint32_t>().swap(ctx.kv_self.row_count);

    llama_model target_a = {};
    llama_model target_b = {};
    llama_model draft = {};
    ggml_tensor target_a_tok = {};
    ggml_tensor target_b_tok = {};
    target_a.tok_embd = &target_a_tok;
    target_b.tok_embd = &target_b_tok;
    draft.arch = LLM_ARCH_DFLASH_DRAFT;
    CHECK(llama_model_share_dflash_io_tensors(&draft, &target_a));
    CHECK(llama_model_dflash_io_mode(&draft, &target_a) == LLAMA_DFLASH_IO_MODE_SHARED);
    CHECK(llama_model_share_dflash_io_tensors(&draft, &target_a));
    target_a.tok_embd = &target_b_tok;
    CHECK(!llama_model_share_dflash_io_tensors(&draft, &target_a));
    target_a.tok_embd = &target_a_tok;
    target_a.output = &target_b_tok;
    CHECK(!llama_model_share_dflash_io_tensors(&draft, &target_a));
    target_a.output = nullptr;
    CHECK(!llama_model_share_dflash_io_tensors(&draft, &target_b));

    llama_context_memory_info undersized = {};
    undersized.struct_size = LLAMA_CONTEXT_MEMORY_INFO_STRUCT_SIZE_V1 - 1;
    undersized.flags = UINT32_MAX;
    CHECK(!llama_context_get_memory_info(&ctx, &undersized));
    CHECK(undersized.struct_size == LLAMA_CONTEXT_MEMORY_INFO_STRUCT_SIZE_V1 - 1);
    CHECK(undersized.flags == UINT32_MAX);

    ctx.backend_cpu = ggml_backend_cpu_init();
    CHECK(ctx.backend_cpu != nullptr);
    ctx.backends.push_back(ctx.backend_cpu);

    ggml_backend_memory_info cpu_info = {};
    CHECK(ggml_backend_get_memory_info(ctx.backend_cpu, &cpu_info));
    CHECK(cpu_info.flags == (GGML_BACKEND_MEMORY_INFO_FLAG_HOST_BYTES_VALID |
                             GGML_BACKEND_MEMORY_INFO_FLAG_DEVICE_BYTES_VALID));
    CHECK(cpu_info.host_bytes > 0);
    CHECK(cpu_info.device_bytes == 0);

    // CPU graph execution retains its largest work arena in the backend instance.
    ggml_init_params compute_params = {
        /*.mem_size   =*/ 1024*1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ false,
    };
    ggml_context * compute_ctx = ggml_init(compute_params);
    CHECK(compute_ctx != nullptr);
    ggml_tensor * compute_input = ggml_new_tensor_1d(compute_ctx, GGML_TYPE_F32, 4096);
    std::memset(compute_input->data, 0, ggml_nbytes(compute_input));
    ggml_tensor * compute_output = ggml_soft_max(compute_ctx, compute_input);
    ggml_cgraph * compute_graph = ggml_new_graph(compute_ctx);
    ggml_build_forward_expand(compute_graph, compute_output);
    CHECK(ggml_backend_graph_compute(ctx.backend_cpu, compute_graph) == GGML_STATUS_SUCCESS);
    ggml_free(compute_ctx);

    ggml_backend_memory_info cpu_info_after_compute = {};
    CHECK(ggml_backend_get_memory_info(ctx.backend_cpu, &cpu_info_after_compute));
    CHECK(cpu_info_after_compute.flags == cpu_info.flags);
    CHECK(cpu_info_after_compute.host_bytes > cpu_info.host_bytes);
    CHECK(cpu_info_after_compute.device_bytes == 0);
    cpu_info = cpu_info_after_compute;

    // The aggregate scheduler query must include its host heap even before a graph is reserved.
    ggml_backend_t scheduler_backends[] = { ctx.backend_cpu };
    ggml_backend_sched_t scheduler =
            ggml_backend_sched_new(scheduler_backends, nullptr, 1, 32, false);
    CHECK(scheduler != nullptr);
    ggml_backend_memory_info scheduler_info = {};
    CHECK(ggml_backend_sched_get_memory_info(scheduler, &scheduler_info));
    CHECK(scheduler_info.flags == (GGML_BACKEND_MEMORY_INFO_FLAG_HOST_BYTES_VALID |
                                   GGML_BACKEND_MEMORY_INFO_FLAG_DEVICE_BYTES_VALID));
    CHECK(scheduler_info.host_bytes > 0);
    CHECK(scheduler_info.device_bytes == 0);
    ggml_backend_sched_free(scheduler);

    llama_context_memory_info baseline = {};
    baseline.struct_size = sizeof(baseline);
    CHECK(llama_context_get_memory_info(&ctx, &baseline));
    CHECK(baseline.flags == (LLAMA_CONTEXT_MEMORY_INFO_FLAG_HOST_BYTES_VALID |
                             LLAMA_CONTEXT_MEMORY_INFO_FLAG_DEVICE_BYTES_VALID |
                             LLAMA_CONTEXT_MEMORY_INFO_FLAG_NUMA_MIRROR_BYTES_VALID));
    CHECK(baseline.host_bytes == sizeof(ctx) + cpu_info.host_bytes +
                                  ctx.backends.capacity()*sizeof(ctx.backends[0]));
    CHECK(baseline.device_bytes == 0);
    CHECK(baseline.numa_mirror_bytes == 0);

    const size_t bufs_capacity_before = ctx.kv_self.bufs.capacity();
    const size_t ctxs_capacity_before = ctx.kv_self.ctxs.capacity();
    const size_t cells_capacity_before = ctx.kv_self.cells.capacity();
    const size_t mirrors_capacity_before = ctx.kv_self.numa_mirror_bufs.capacity();

    ggml_backend_buffer_t buffer =
            ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), 4096);
    CHECK(buffer != nullptr);
    size_t buffer_metadata_size = 0;
    CHECK(ggml_backend_buffer_get_metadata_size(buffer, &buffer_metadata_size));
    CHECK(buffer_metadata_size > 0);
    ctx.kv_self.bufs.push_back(buffer);
    ctx.buf_output = buffer; // The query must not count an aliased buffer twice.

    ggml_init_params ggml_params = {
        /*.mem_size   =*/ 4096,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * metadata = ggml_init(ggml_params);
    CHECK(metadata != nullptr);
    ggml_tensor * mirrored_tensor = ggml_new_tensor_1d(metadata, GGML_TYPE_F32, 1);
    void * mirrored_tensor_node_data[GGML_NUMA_MAX_NODES] = {};
    mirrored_tensor_node_data[0] = reinterpret_cast<void *>(uintptr_t{1});
    ggml_numa_tensor_set_mirror(mirrored_tensor, mirrored_tensor_node_data);
    const size_t tensor_mirror_metadata =
            ggml_numa_tensor_mirror_metadata_size(mirrored_tensor);
    CHECK(tensor_mirror_metadata > 0);
    ctx.kv_self.ctxs.push_back(metadata);
    ctx.kv_self.cells.reserve(3);

    llama_kv_cache::numa_mirror_buffer mirror = {};
    mirror.size = 123;
    mirror.node_base[1] = reinterpret_cast<void *>(uintptr_t{1});
    ctx.kv_self.numa_mirror_bufs.push_back(mirror);

    struct extended_memory_info {
        llama_context_memory_info info;
        uint64_t canary;
    } extended = {};
    extended.info.struct_size = sizeof(extended);
    extended.canary = UINT64_C(0xfedcba9876543210);

    CHECK(llama_context_get_memory_info(&ctx, &extended.info));
    CHECK(extended.info.struct_size == LLAMA_CONTEXT_MEMORY_INFO_STRUCT_SIZE_V1);
    CHECK(extended.info.flags == (LLAMA_CONTEXT_MEMORY_INFO_FLAG_HOST_BYTES_VALID |
                                  LLAMA_CONTEXT_MEMORY_INFO_FLAG_DEVICE_BYTES_VALID |
                                  LLAMA_CONTEXT_MEMORY_INFO_FLAG_NUMA_MIRROR_BYTES_VALID));

    const uint64_t expected_host_delta =
            ggml_backend_buffer_get_size(buffer) + buffer_metadata_size +
            ggml_get_mem_size(metadata) + tensor_mirror_metadata + mirror.size +
            (ctx.kv_self.bufs.capacity() - bufs_capacity_before)*sizeof(ctx.kv_self.bufs[0]) +
            (ctx.kv_self.ctxs.capacity() - ctxs_capacity_before)*sizeof(ctx.kv_self.ctxs[0]) +
            (ctx.kv_self.cells.capacity() - cells_capacity_before)*sizeof(ctx.kv_self.cells[0]) +
            (ctx.kv_self.numa_mirror_bufs.capacity() - mirrors_capacity_before)*
                    sizeof(ctx.kv_self.numa_mirror_bufs[0]);
    CHECK(extended.info.host_bytes == baseline.host_bytes + expected_host_delta);
    CHECK(extended.info.device_bytes == 0);
    CHECK(extended.info.numa_mirror_bytes == mirror.size);
    CHECK(extended.canary == UINT64_C(0xfedcba9876543210));

    // Exercise representative nested capacities from each custom runtime owner. Associative
    // container nodes are intentionally outside the accounting scope; their vector payloads are
    // still included.
    ctx.cparams.devices.emplace_back(64, 'd');
    ctx.kv_self.row_count.reserve(7);
    ctx.kv_self.split_k_l.resize(1);
    ctx.kv_self.split_k_l[0].tensor_splits.reserve(2);
    ctx.kv_self.split_k_l[0].ranges.resize(1);
    ctx.kv_self.split_k_l[0].ranges[0].reserve(3);
    ctx.kv_self.ckpt.dsv4_state_data.resize(1);
    ctx.kv_self.ckpt.dsv4_state_data[0].reserve(17);
    ctx.cvec.tensors.reserve(2);
    ctx.dflash.capture = std::make_unique<llama_context::dflash_runtime::capture_state>();
    ctx.dflash.capture->layer_ids.reserve(3);
    ctx.dflash.capture->layer_rows.resize(1);
    ctx.dflash.capture->layer_rows[0].reserve(4);
    ctx.dsv4.raw.sinfo_write.strm.reserve(2);
    ctx.dsv4.raw.sinfo_write.idxs.resize(1);
    ctx.dsv4.raw.sinfo_write.idxs[0].reserve(5);
    ctx.dsv4.cache.csa_k.reserve(2);
    ctx.inp_qsa.reserve(3);
    ctx.qwen4_mtp_qsa_topk.reserve(7);
    ctx.qwen4_mtp_qsa_valid.reserve(11);
    ctx.swa_compact_buf.reserve(19);
    ctx.embd_seq[42].reserve(6);

    llama_context_memory_info auxiliaries = {};
    auxiliaries.struct_size = sizeof(auxiliaries);
    CHECK(llama_context_get_memory_info(&ctx, &auxiliaries));
    const uint64_t expected_auxiliary_delta =
            ctx.cparams.devices.capacity()*sizeof(ctx.cparams.devices[0]) +
            ctx.cparams.devices[0].capacity() + 1 +
            ctx.kv_self.row_count.capacity()*sizeof(ctx.kv_self.row_count[0]) +
            ctx.kv_self.split_k_l.capacity()*sizeof(ctx.kv_self.split_k_l[0]) +
            ctx.kv_self.split_k_l[0].tensor_splits.capacity()*
                    sizeof(ctx.kv_self.split_k_l[0].tensor_splits[0]) +
            ctx.kv_self.split_k_l[0].ranges.capacity()*
                    sizeof(ctx.kv_self.split_k_l[0].ranges[0]) +
            ctx.kv_self.split_k_l[0].ranges[0].capacity()*
                    sizeof(ctx.kv_self.split_k_l[0].ranges[0][0]) +
            ctx.kv_self.ckpt.dsv4_state_data.capacity()*
                    sizeof(ctx.kv_self.ckpt.dsv4_state_data[0]) +
            ctx.kv_self.ckpt.dsv4_state_data[0].capacity()*
                    sizeof(ctx.kv_self.ckpt.dsv4_state_data[0][0]) +
            ctx.cvec.tensors.capacity()*sizeof(ctx.cvec.tensors[0]) +
            sizeof(*ctx.dflash.capture) +
            ctx.dflash.capture->layer_ids.capacity()*sizeof(ctx.dflash.capture->layer_ids[0]) +
            ctx.dflash.capture->layer_rows.capacity()*sizeof(ctx.dflash.capture->layer_rows[0]) +
            ctx.dflash.capture->layer_rows[0].capacity()*
                    sizeof(ctx.dflash.capture->layer_rows[0][0]) +
            ctx.dsv4.raw.sinfo_write.strm.capacity()*
                    sizeof(ctx.dsv4.raw.sinfo_write.strm[0]) +
            ctx.dsv4.raw.sinfo_write.idxs.capacity()*
                    sizeof(ctx.dsv4.raw.sinfo_write.idxs[0]) +
            ctx.dsv4.raw.sinfo_write.idxs[0].capacity()*
                    sizeof(ctx.dsv4.raw.sinfo_write.idxs[0][0]) +
            ctx.dsv4.cache.csa_k.capacity()*sizeof(ctx.dsv4.cache.csa_k[0]) +
            ctx.inp_qsa.capacity()*sizeof(ctx.inp_qsa[0]) +
            ctx.qwen4_mtp_qsa_topk.capacity()*sizeof(ctx.qwen4_mtp_qsa_topk[0]) +
            ctx.qwen4_mtp_qsa_valid.capacity()*sizeof(ctx.qwen4_mtp_qsa_valid[0]) +
            ctx.swa_compact_buf.capacity()*sizeof(ctx.swa_compact_buf[0]) +
            ctx.embd_seq[42].capacity()*sizeof(ctx.embd_seq[42][0]);
    CHECK(auxiliaries.host_bytes == extended.info.host_bytes + expected_auxiliary_delta);
    CHECK(auxiliaries.device_bytes == extended.info.device_bytes);
    CHECK(auxiliaries.numa_mirror_bytes == extended.info.numa_mirror_bytes);

    // Unsupported backend-private accounting leaves useful lower bounds, but invalidates both
    // aggregate host/device fields. NUMA mirror accounting remains independently valid.
    ggml_backend_t unsupported = make_unsupported_backend();
    ggml_backend_memory_info unsupported_info = {
        UINT32_MAX,
        UINT64_MAX,
        UINT64_MAX,
    };
    CHECK(!ggml_backend_get_memory_info(unsupported, &unsupported_info));
    CHECK(unsupported_info.flags == UINT32_MAX);
    CHECK(unsupported_info.host_bytes == UINT64_MAX);
    CHECK(unsupported_info.device_bytes == UINT64_MAX);
    ctx.backends.push_back(unsupported);

    llama_context_memory_info incomplete = {};
    incomplete.struct_size = sizeof(incomplete);
    CHECK(llama_context_get_memory_info(&ctx, &incomplete));
    CHECK(incomplete.flags == LLAMA_CONTEXT_MEMORY_INFO_FLAG_NUMA_MIRROR_BYTES_VALID);
    CHECK(incomplete.host_bytes >= auxiliaries.host_bytes);
    CHECK(incomplete.device_bytes == 0);
    CHECK(incomplete.numa_mirror_bytes == mirror.size);

    // Avoid freeing the deliberately aliased/fake NUMA test resources twice in ctx's destructor.
    ctx.buf_output = nullptr;
    ctx.kv_self.numa_mirror_bufs.clear();
    return 0;
}
