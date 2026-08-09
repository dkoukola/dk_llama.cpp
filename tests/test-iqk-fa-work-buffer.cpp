#include "ggml.h"
#include "iqk/iqk_mul_mat.h"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace {

size_t work_size(
        int64_t query_heads,
        int64_t kv_heads,
        int64_t k_rows,
        int n_threads,
        int n_swa,
        bool have_mask) {
    ggml_tensor q = {};
    ggml_tensor k = {};
    ggml_tensor v = {};
    ggml_tensor mask = {};
    ggml_tensor dst = {};

    q.type = GGML_TYPE_F32;
    q.ne[0] = 512;
    q.ne[1] = 1;
    q.ne[2] = query_heads;
    q.ne[3] = 1;

    k.type = GGML_TYPE_F16;
    k.ne[0] = 512;
    k.ne[1] = k_rows;
    k.ne[2] = kv_heads;
    k.ne[3] = 1;

    v = k;
    mask.type = GGML_TYPE_F16;
    dst.src[0] = &q;
    dst.src[1] = &k;
    dst.src[2] = &v;
    dst.src[3] = have_mask ? &mask : nullptr;
    dst.op_params[4] = n_swa;
    return iqk_fa_work_buffer_size(&dst, n_threads);
}

void expect_size(
        int64_t query_heads,
        int64_t kv_heads,
        int64_t k_rows,
        int n_threads,
        int n_swa,
        bool have_mask,
        size_t expected) {
    const size_t actual = work_size(
        query_heads, kv_heads, k_rows, n_threads, n_swa, have_mask);
    if (actual != expected) {
        std::fprintf(stderr,
            "unexpected IQK FA work size: heads=%lld/%lld rows=%lld threads=%d swa=%d mask=%d expected=%zu actual=%zu\n",
            (long long) query_heads,
            (long long) kv_heads,
            (long long) k_rows,
            n_threads,
            n_swa,
            have_mask,
            expected,
            actual);
        std::abort();
    }
}

} // namespace

int main() {
    constexpr size_t row8 = (512 + 16)*8*sizeof(float);
    constexpr size_t row2 = (512 + 16)*2*sizeof(float);
    constexpr size_t row64 = (512 + 16)*64*sizeof(float);

    // The failing DSpark shape plans from 768 K rows, then SWA narrows to 512 rows.
    // That changes gcd(24, 64)=8 to gcd(16, 64)=16 and doubles per-thread scratch.
    expect_size(64, 1, 768, 64, 128, true,
        size_t(64)*(512 + 16)*16*sizeof(float));

    // Without runtime narrowing, retain the original 8-query/thread plan.
    expect_size(64, 1, 768, 64, 0, false,
        size_t(64)*(512 + 16)*8*sizeof(float));

    // Non-divisible split-K plans require more result slots than worker threads.
    expect_size(8, 1, 2080, 64, 0, false, 65*row8);
    expect_size(8, 1, 2304, 64, 0, false, 72*row8);

    // Mask narrowing can select a larger scratch plan than the full tensor shape.
    expect_size(8, 1, 4096, 64, 0, true, 95*row8);

    // The generic multi-KV-head split remains valid when a mask narrows to one chunk.
    expect_size(128, 64, 288, 64, 0, true, 64*row2);

    // Bound larger multi-KV-head mask ranges without scanning the whole context.
    expect_size(14, 7, 736, 2, 0, true, 14*row2);

    // The generic split is also reachable at one 32-row chunk.
    expect_size(64, 8, 32, 1, 0, true, 8*row8);
    expect_size(64, 1, 32, 1, 0, true, row64);

    std::puts("IQK flash-attention work-buffer tests passed");
    return 0;
}
