/* Qwen3.8 Flash Next. Included by ds4_cuda.cu so model residency, streams
 * and temporary allocations have the same lifetime as the other CUDA paths. */

#include "ds4_qwen4_vision.h"

namespace qwen4_cuda {

/* Programmatic dependent launch. Each kernel waits for its predecessor
 * before touching memory, then lets its successor be scheduled, so the
 * next kernel's blocks are resident when this one finishes instead of
 * being launched afterwards. Reads and writes keep their order. */
__device__ __forceinline__ void pdl_enter() {
#if __CUDA_ARCH__ >= 900
    cudaGridDependencySynchronize();
    cudaTriggerProgrammaticLaunchCompletion();
#endif
}

/* The two halves of pdl_enter(), for kernels that load weights before the
 * wait.  No kernel in the stream writes weights, so those loads may overlap
 * the predecessor; activations are still read after the wait. */
__device__ __forceinline__ void pdl_wait() {
#if __CUDA_ARCH__ >= 900
    cudaGridDependencySynchronize();
#endif
}

__device__ __forceinline__ void pdl_trigger() {
#if __CUDA_ARCH__ >= 900
    cudaTriggerProgrammaticLaunchCompletion();
#endif
}

/* Weight loads issued before pdl_wait().  The compiler sinks invariant
 * __ldg loads below griddepcontrol.wait; volatile PTX keeps their order.
 * ptxas also hoists the wait to the top of its basic block, so callers keep
 * these loads in a block of their own (behind a branch). */
__device__ __forceinline__ float4 ldg_pre(const float4 *p) {
    float4 v;
    asm volatile("ld.global.nc.v4.f32 {%0, %1, %2, %3}, [%4];"
                 : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w) : "l"(p));
    return v;
}

__device__ __forceinline__ float ldg_pre(const float *p) {
    float v;
    asm volatile("ld.global.nc.f32 %0, [%1];" : "=f"(v) : "l"(p));
    return v;
}

__device__ __forceinline__ float ldg_pre_half(const void *p) {
    unsigned short v;
    asm volatile("ld.global.nc.u16 %0, [%1];" : "=h"(v) : "l"(p));
    return __half2float(__ushort_as_half(v));
}

/* One-shot bulk (TMA) copy of weights into shared memory before the wait.
 * Unlike per-thread loads it holds no load slots of the SM, so the kernel
 * still running there keeps its memory latency.  Thread 0 starts it; all
 * threads wait for phase 0 after a __syncthreads().  The host launches these
 * kernels only on sm_90+ (bulk_supported()). */
__device__ __forceinline__ unsigned smem_addr(const void *p) {
    return (unsigned)__cvta_generic_to_shared(p);
}

__device__ __forceinline__ void bulk_start(uint64_t *bar, unsigned bytes) {
#if __CUDA_ARCH__ >= 900
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(smem_addr(bar)) : "memory");
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
                 :: "r"(smem_addr(bar)), "r"(bytes) : "memory");
#endif
}

__device__ __forceinline__ void bulk_copy(void *dst, const void *src, unsigned bytes, uint64_t *bar) {
#if __CUDA_ARCH__ >= 900
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                 :: "r"(smem_addr(dst)), "l"(src), "r"(bytes), "r"(smem_addr(bar)) : "memory");
#else
    __trap();
#endif
}

__device__ __forceinline__ void bulk_wait(uint64_t *bar) {
#if __CUDA_ARCH__ >= 900
    asm volatile("{\n\t.reg .pred p;\n"
                 "WAIT%=:\n\t"
                 "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], 0;\n\t"
                 "@!p bra WAIT%=;\n}" :: "r"(smem_addr(bar)) : "memory");
#endif
}

/* Dynamic shared memory a launch may request without opting in through
 * cudaFuncSetAttribute(MaxDynamicSharedMemorySize) is 48 KiB less the
 * kernel's static shared memory.  SPLIT_BULK_STATIC_SMEM bounds that of
 * matvec_split_f16_bulk (part[8][ROWS <= 8] floats and an 8-byte barrier,
 * 264 B).  hc_mix_f16_bulk (2056 B static, at most 16 KiB of weights for
 * hc <= 4 and rank <= 512) always fits. */
constexpr unsigned BULK_DYN_SMEM_LIMIT = 48u * 1024u;
constexpr unsigned SPLIT_BULK_STATIC_SMEM = 512u;

static bool bulk_supported() {
    static int on = -1;
    if (on < 0) {
        int device = 0, major = 0;
        on = cudaGetDevice(&device) == cudaSuccess &&
             cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device) == cudaSuccess &&
             major >= 9;
        (void)cudaGetLastError();
    }
    return on;
}

/* Decode rows (a token, or the 2/3-row MTP verify) load the HC, router and
 * gate weights before the dependency wait.  Batched sessions (4+ rows) keep
 * the plain kernels: there the extra shared memory of the bulk copies costs
 * occupancy across several waves.  The tests-only *_ref_tensor entry points
 * run the plain kernels at every T as the byte-exact reference. */
static bool presync_load(unsigned T) { return T <= 3; }

/* Single-token decode lets the HC down projection after hc_norm launch only
 * after the norm reduction (the MTP verify rows gained nothing from it). */
static unsigned hc_norm_late(unsigned T) { return T == 1; }

/* Reports the virtual architecture these kernels were compiled for. */
__global__ void pdl_probe() {}

/* PDL needs an sm_90+ device and kernels compiled for sm_90+: a build for
 * an older target JIT-compiled on a newer GPU has no dependency wait in
 * pdl_enter() and must launch without it. */
static bool pdl_enabled() {
    static int on = -1;   /* DS4_CUDA_NO_PDL=1 launches without it */
    if (on < 0) {
        int device = 0, major = 0;
        cudaFuncAttributes fa;
        on = getenv("DS4_CUDA_NO_PDL") == NULL && cudaGetDevice(&device) == cudaSuccess &&
             cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device) == cudaSuccess &&
             major >= 9 && cudaFuncGetAttributes(&fa, pdl_probe) == cudaSuccess && fa.ptxVersion >= 90;
        (void)cudaGetLastError();
    }
    return on;
}

template<typename... KernelArgs, typename... Args>
static void launch(void (*kernel)(KernelArgs...), dim3 grid, dim3 block, size_t smem, Args&&... args) {
    cudaLaunchConfig_t cfg = {};
    cudaLaunchAttribute attr[1];
    cfg.gridDim = grid;
    cfg.blockDim = block;
    cfg.dynamicSmemBytes = smem;
    cfg.stream = cuda_decode_stream();
    if (pdl_enabled()) {
        attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attr[0].val.programmaticStreamSerializationAllowed = 1;
        cfg.attrs = attr;
        cfg.numAttrs = 1;
    }
    (void)cudaLaunchKernelEx(&cfg, kernel, std::forward<Args>(args)...);
}

__device__ __forceinline__ float sum(float x);
__device__ __forceinline__ float sigmoid(float x);
template<unsigned TYPE>
__device__ __forceinline__ float dot(const char *row, const float *x, unsigned n,
        const uint64_t *grid = NULL, const uint8_t *signs = NULL);
__device__ float injection(const float *inj, unsigned hc, unsigned stream);

struct rope_args { float freq[32], scale; unsigned nrot; };
static float rope_freq[32];
static float rope_scale = 1;
static bool rope_set;

static rope_args rope(unsigned nrot, float base) {
    rope_args r = {};
    r.nrot = nrot;
    r.scale = rope_set ? rope_scale : 1;
    for (unsigned i = 0; i < nrot / 2; i++)
        r.freq[i] = rope_set ? rope_freq[i] : powf(base, -2.0f * i / nrot);
    return r;
}

__device__ void apply_rope(float *row, const uint32_t *pos, rope_args r) {
    const unsigned i = threadIdx.x;
    if (i < r.nrot / 2) {
        const float theta = (float)pos[i % 3] * r.freq[i];
        const float c = cosf(theta) * r.scale, s = sinf(theta) * r.scale;
        const float a = row[i], b = row[i + r.nrot / 2];
        row[i] = a * c - b * s;
        row[i + r.nrot / 2] = a * s + b * c;
    }
    __syncwarp();
}

__device__ __forceinline__ void attn_prep_body(float *qout, float *gate, __half *kc, __half *vc,
        float *iqout, float *ikc, const float *qg, const float *kp, const float *vp,
        const float *iq, const float *ik, const uint32_t *pos3,
        const float *gq, const float *gk, const float *giq,
        unsigned H, unsigned Hkv, unsigned D, unsigned Hi, unsigned Di,
        unsigned pos0, float eps, rope_args rp, unsigned t) {
    const unsigned slot = blockIdx.x, pos = pos0 + t, lane = threadIdx.x;
    __shared__ float row[256];
    if (slot == H + Hkv + Hi) {
        for (unsigned i = lane; i < Di; i += 32) ikc[(uint64_t)pos * Di + i] = ik[(uint64_t)t * Di + i];
        return;
    }
    const bool isq = slot < H, isk = !isq && slot < H + Hkv;
    const unsigned h = isq ? slot : isk ? slot - H : slot - H - Hkv;
    const unsigned dim = isq || isk ? D : Di;
    const float *src = isq ? qg + ((uint64_t)t * H + h) * 2 * D :
                      isk ? kp + ((uint64_t)t * Hkv + h) * D : iq + ((uint64_t)t * Hi + h) * Di;
    const float *gamma = isq ? gq : isk ? gk : giq;
    const unsigned npt = dim / 32;
    float ss = 0;
    for (unsigned i = 0; i < npt; i++) { const float v = src[lane * npt + i]; ss += v * v; }
    const float inv = rsqrtf(sum(ss) / dim + eps);
    for (unsigned i = lane; i < dim; i += 32) row[i] = src[i] * inv * gamma[i];
    __syncwarp();
    apply_rope(row, pos3 + (uint64_t)pos * 4, rp);
    for (unsigned i = lane; i < dim; i += 32) {
        if (isq) {
            qout[((uint64_t)t * H + h) * D + i] = row[i];
            gate[((uint64_t)t * H + h) * D + i] = src[D + i];
        } else if (isk) {
            kc[((uint64_t)pos * Hkv + h) * D + i] = __float2half_rn(row[i]);
            vc[((uint64_t)pos * Hkv + h) * D + i] = __float2half_rn(vp[((uint64_t)t * Hkv + h) * D + i]);
        } else iqout[((uint64_t)t * Hi + h) * Di + i] = row[i];
    }
}

__global__ void attn_prep(float *qout, float *gate, __half *kc, __half *vc,
        float *iqout, float *ikc, const float *qg, const float *kp, const float *vp,
        const float *iq, const float *ik, const uint32_t *pos3,
        const float *gq, const float *gk, const float *giq,
        unsigned H, unsigned Hkv, unsigned D, unsigned Hi, unsigned Di,
        unsigned pos0, float eps, rope_args rp) {
    pdl_enter();
    attn_prep_body(qout,gate,kc,vc,iqout,ikc,qg,kp,vp,iq,ik,pos3,gq,gk,giq,H,Hkv,D,Hi,Di,pos0,eps,rp,blockIdx.y);
}

__device__ __forceinline__ void block_key_body(__half *out, const float *ik, const uint32_t *pos3,
        const float *gamma, unsigned block0, unsigned ratio, unsigned D, float eps, rope_args rp) {
    const unsigned b = block0 + blockIdx.x, lane = threadIdx.x, npt = D / 32;
    __shared__ float row[128];
    float ss = 0, v[4];
    for (unsigned i = 0; i < npt; i++) {
        float a = 0;
        for (unsigned t = 0; t < ratio; t++) a += ik[((uint64_t)b * ratio + t) * D + lane * npt + i];
        v[i] = a / ratio;
        ss += v[i] * v[i];
    }
    const float inv = rsqrtf(sum(ss) / D + eps);
    for (unsigned i = 0; i < npt; i++) row[lane * npt + i] = v[i] * inv * gamma[lane * npt + i];
    __syncwarp();
    apply_rope(row, pos3 + (uint64_t)b * ratio * 4, rp);
    for (unsigned i = lane; i < D; i += 32) out[(uint64_t)b * D + i] = __float2half_rn(row[i]);
}

__global__ void block_key(__half *out, const float *ik, const uint32_t *pos3,
        const float *gamma, unsigned block0, unsigned ratio, unsigned D, float eps, rope_args rp) {
    pdl_enter();
    block_key_body(out,ik,pos3,gamma,block0,ratio,D,eps,rp);
}

__device__ __forceinline__ void idx_score_body(float *out, const float *q, const __half *key,
        unsigned N, unsigned H, unsigned D, unsigned pos0, unsigned ratio, unsigned t) {
    const unsigned b = blockIdx.x * 4 + threadIdx.x / 32, lane = threadIdx.x & 31;
    if (b >= N) return;
    float score = 0;
    if (b >= (pos0 + t + 1) / ratio) score = -3e38f;
    else for (unsigned h = 0; h < H; h++) {
        float a = 0;
        for (unsigned i = lane; i < D; i += 32)
            a += q[((uint64_t)t * H + h) * D + i] * __half2float(key[(uint64_t)b * D + i]);
        score += fmaxf(sum(a), 0);
    }
    if (!lane) out[(uint64_t)t * N + b] = score;
}

__global__ void idx_score(float *out, const float *q, const __half *key,
        unsigned N, unsigned H, unsigned D, unsigned pos0, unsigned ratio) {
    pdl_enter();
    idx_score_body(out,q,key,N,H,D,pos0,ratio,blockIdx.y);
}

__global__ void tile_max(unsigned *out, const float *scores, unsigned N, unsigned tiles) {
    pdl_enter();
    const unsigned tile = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y;
    if (tile >= tiles) return;
    unsigned v = 0;
    for (unsigned i = tile * 8; i < min(N, tile * 8 + 8); i++)
        v = max(v, __float_as_uint(fmaxf(scores[(uint64_t)t * N + i], 0)));
    out[(uint64_t)t * tiles + tile] = v;
}

/* Inclusive prefix sums of a and b over the 256 threads of the block, in
 * thread order: a warp shuffle scan, then the preceding warps' totals. */
__device__ __forceinline__ void block_scan256(unsigned &a, unsigned &b, unsigned (*tot)[8]) {
    const unsigned lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    for (unsigned o = 1; o < 32; o <<= 1) {
        const unsigned na = __shfl_up_sync(0xffffffffu, a, o), nb = __shfl_up_sync(0xffffffffu, b, o);
        if (lane >= o) { a += na; b += nb; }
    }
    if (lane == 31) { tot[0][w] = a; tot[1][w] = b; }
    __syncthreads();
    for (unsigned i = 0; i < w; i++) { a += tot[0][i]; b += tot[1][i]; }
    __syncthreads();
}

/* Exact radix threshold and stable gather, with the same greater-than then
 * equal-score ordering as Metal. No context-dependent candidate truncation.
 * Block-parallel throughout: histogram increments are aggregated per warp
 * (the first pass puts nearly all keys into two or three exponent bins),
 * the digit walk is a suffix scan whose first bin reaching `need` is found
 * with one __syncthreads_count, and the gather offsets are a block scan. */
__device__ __forceinline__ void idx_select_body(int *out, const float *score, unsigned N, unsigned K, unsigned t) {
    const unsigned tid = threadIdx.x, lane = tid & 31, w = tid >> 5;
    __shared__ unsigned hist[256], tot[2][8], threshold, need;
    const float *row = score + (uint64_t)t * N;
    if (!tid) { threshold = 0; need = K; }
    for (unsigned pass = 0; pass < 4; pass++) {
        const unsigned shift = 24 - 8 * pass, mask = pass ? (0xffffffffu << (shift + 8)) : 0;
        hist[tid] = 0;
        __syncthreads();
        const unsigned thr = threshold;
        for (unsigned base = w * 32; base < N; base += 256) {
            const unsigned i = base + lane;
            unsigned bin = 256;
            if (i < N) {
                const unsigned key = __float_as_uint(fmaxf(row[i], 0));
                if ((key & mask) == thr) bin = (key >> shift) & 255;
            }
            const unsigned peers = __match_any_sync(0xffffffffu, bin);
            if (bin < 256 && lane == (unsigned)__ffs(peers) - 1) atomicAdd(hist + bin, (unsigned)__popc(peers));
        }
        __syncthreads();
        /* thread j holds digit 255 - j; incl counts the keys at digits >= it */
        const unsigned h = hist[255 - tid], nd = need;
        unsigned incl = h, unused = 0;
        block_scan256(incl, unused, tot);
        const unsigned first = __syncthreads_count(incl < nd);
        if (first < 256) {
            if (tid == first) { threshold = thr | (255 - tid) << shift; need = nd - (incl - h); }
        } else if (tid == 255) need = nd - incl;
    }
    __syncthreads();
    const unsigned thr = threshold, nd = need;
    const unsigned chunk = (N + 255) / 256, begin = min(N, tid * chunk), end = min(N, begin + chunk);
    unsigned ng = 0, ne = 0;
    for (unsigned i = begin; i < end; i++) {
        const unsigned key = __float_as_uint(fmaxf(row[i], 0));
        ng += key > thr; ne += key == thr;
    }
    unsigned g = ng, e = ne;
    block_scan256(g, e, tot);
    g -= ng; e -= ne;
    for (unsigned i = begin; i < end; i++) {
        const unsigned key = __float_as_uint(fmaxf(row[i], 0));
        if (key > thr) out[(uint64_t)t * K + g++] = i;
        else if (key == thr) { if (e < nd) out[(uint64_t)t * K + K - nd + e] = i; e++; }
    }
}

__global__ void idx_select(int *out, const float *score, unsigned N, unsigned K) {
    pdl_enter();
    idx_select_body(out,score,N,K,blockIdx.x);
}

__device__ __forceinline__ void idx_expand_body(int *out, unsigned *count, const int *blocks,
        unsigned K, unsigned ratio, unsigned pos0, unsigned stride, unsigned t) {
    const unsigned pos = pos0 + t, tail = (pos + 1) / ratio * ratio;
    for (unsigned i = threadIdx.x; i < K * ratio; i += blockDim.x)
        out[(uint64_t)t * stride + i] = blocks[(uint64_t)t * K + i / ratio] * ratio + i % ratio;
    for (unsigned i = tail + threadIdx.x; i <= pos; i += blockDim.x)
        out[(uint64_t)t * stride + K * ratio + i - tail] = i;
    if (!threadIdx.x) count[t] = K * ratio + pos + 1 - tail;
}

__global__ void idx_expand(int *out, unsigned *count, const int *blocks,
        unsigned K, unsigned ratio, unsigned pos0, unsigned stride) {
    pdl_enter();
    idx_expand_body(out,count,blocks,K,ratio,pos0,stride,blockIdx.x);
}

/* Keys a row at pos attends, and the key splits its decode step takes
 * (partial scratch holds up to 64 splits per row). */
__host__ __device__ __forceinline__ unsigned attn_keys(bool sparse, unsigned stride, unsigned pos) {
    return sparse ? stride : pos + 1;
}
__host__ __device__ __forceinline__ unsigned attn_splits(unsigned keys) {
    const unsigned splits = (keys + 31) / 32;
    return splits < 64 ? splits : 64;
}

template<unsigned D>
__device__ __forceinline__ void attention_body(float *out, float *partial, const float *q, const float *gate,
        const __half *kc, const __half *vc, const int *sel, const unsigned *counts,
        unsigned H, unsigned Hkv, unsigned pos0, unsigned stride, bool sparse,
        unsigned splits, unsigned per, float scale, unsigned t) {
    const unsigned h = blockIdx.x * 4 + threadIdx.x / 32, split = blockIdx.z;
    if (h >= H) return;
    const unsigned lane = threadIdx.x & 31, kh = h / (H / Hkv), n = sparse ? counts[t] : pos0 + t + 1;
    float qv[D / 32], acc[D / 32] = {}, m = -3e38f, denom = 0;
    for (unsigned i = 0; i < D / 32; i++) qv[i] = q[((uint64_t)t * H + h) * D + lane + 32 * i] * scale;
    /* Fetch K and V a few keys ahead. The keys are consumed in the same
     * order with the same arithmetic, so the result is unchanged, but the
     * loads of the next keys overlap the softmax update of this one. */
    enum { AHEAD = 4 };
    const unsigned j1 = min(n, (split + 1) * per);
    for (unsigned jb = split * per; jb < j1; jb += AHEAD) {
        unsigned pos[AHEAD];
        __half kr[AHEAD][D / 32], vr[AHEAD][D / 32];
        #pragma unroll
        for (unsigned u = 0; u < AHEAD; u++) {
            pos[u] = jb + u < j1 ? (sparse ? (unsigned)sel[(uint64_t)t * stride + jb + u] : jb + u) : UINT_MAX;
            if (pos[u] <= pos0 + t) {
                #pragma unroll
                for (unsigned i = 0; i < D / 32; i++) {
                    kr[u][i] = kc[((uint64_t)pos[u] * Hkv + kh) * D + lane + 32 * i];
                    vr[u][i] = vc[((uint64_t)pos[u] * Hkv + kh) * D + lane + 32 * i];
                }
            }
        }
        #pragma unroll
        for (unsigned u = 0; u < AHEAD; u++) {
            if (pos[u] > pos0 + t) continue;
            float score = 0;
            #pragma unroll
            for (unsigned i = 0; i < D / 32; i++) score += qv[i] * __half2float(kr[u][i]);
            score = sum(score);
            const float nm = fmaxf(m, score), correction = expf(m - nm), w = expf(score - nm);
            denom = denom * correction + w;
            #pragma unroll
            for (unsigned i = 0; i < D / 32; i++) acc[i] = acc[i] * correction + w * __half2float(vr[u][i]);
            m = nm;
        }
    }
    if (splits == 1) {
        for (unsigned i = 0; i < D / 32; i++) {
            const uint64_t p = ((uint64_t)t * H + h) * D + lane + 32 * i;
            out[p] = (denom > 0 ? acc[i] / denom : 0) * sigmoid(gate[p]);
        }
    } else {
        float *dst = partial + (((uint64_t)t * H + h) * splits + split) * (D + 2);
        if (!lane) { dst[0] = m; dst[1] = denom; }
        for (unsigned i = 0; i < D / 32; i++) dst[2 + lane + 32 * i] = acc[i];
    }
}

/* Row t of a launch is the one-token step at pos0 + t. With partial scratch
 * (decode rows: a token or the MTP verify rows) its keys split by its own key
 * count and its partials sit at a fixed 64-split row stride, so every row
 * rounds exactly like a single-token decode at that position. Without it
 * (prefill) a row takes one split. */
template<unsigned D>
__global__ void attention(float *out, float *partial, const float *q, const float *gate,
        const __half *kc, const __half *vc, const int *sel, const unsigned *counts,
        unsigned H, unsigned Hkv, unsigned pos0, unsigned stride, bool sparse, float scale) {
    pdl_enter();
    const unsigned t = blockIdx.y, keys = attn_keys(sparse,stride,pos0+t), splits = partial ? attn_splits(keys) : 1;
    if (blockIdx.z >= splits) return;
    attention_body<D>(out+(uint64_t)t*H*D,partial ? partial+(uint64_t)t*H*64*(D+2) : nullptr,
        q+(uint64_t)t*H*D,gate+(uint64_t)t*H*D,kc,vc,sparse ? sel+(uint64_t)t*stride : nullptr,
        sparse ? counts+t : nullptr,H,Hkv,pos0+t,stride,sparse,splits,(keys+splits-1)/splits,scale,0);
}

__device__ __forceinline__ void attn_merge_body(float *out, const float *partial, const float *gate,
        unsigned H, unsigned D, unsigned splits, unsigned t) {
    const unsigned h = blockIdx.x, d = threadIdx.x;
    if (d >= D) return;
    const float *p = partial + ((uint64_t)t * H + h) * splits * (D + 2);
    float m = -3e38f, denom = 0, acc = 0;
    for (unsigned s = 0; s < splits; s++) m = fmaxf(m, p[s * (D + 2)]);
    for (unsigned s = 0; s < splits; s++) {
        const float *row = p + s * (D + 2);
        const float w = row[1] > 0 ? expf(row[0] - m) : 0;
        denom += row[1] * w; acc += row[2 + d] * w;
    }
    const uint64_t i = ((uint64_t)t * H + h) * D + d;
    out[i] = (denom > 0 ? acc / denom : 0) * sigmoid(gate[i]);
}

__global__ void attn_merge(float *out, const float *partial, const float *gate,
        unsigned H, unsigned D, unsigned pos0, unsigned stride, bool sparse) {
    pdl_enter();
    const unsigned t = blockIdx.y, splits = attn_splits(attn_keys(sparse,stride,pos0+t));
    if (splits > 1) attn_merge_body(out+(uint64_t)t*H*D,partial+(uint64_t)t*H*64*(D+2),
        gate+(uint64_t)t*H*D,H,D,splits,0);
}

/* The heads in a KV group share the same selected keys. Keep their output
 * accumulators in registers and use tensor cores for both products. Scaled
 * residual components retain the fine part of the FP32 queries/probabilities;
 * K and V are already half, so neither requires further rounding. */
__device__ __forceinline__ void attention_group_body(float *out, float *partial, const float *q, const float *gate,
        const __half *kc, const __half *vc, const int *sel, const unsigned *counts,
        unsigned H, unsigned Hkv, unsigned pos0, unsigned stride, bool sparse, float scale, unsigned splits, unsigned per, unsigned t) {
#if __CUDA_ARCH__ >= 800
    const unsigned D = 256, tid = threadIdx.x, lane = tid&31, warp = tid/32;
    const unsigned kh = blockIdx.x, group = H/Hkv;
    const unsigned qr = tid/16, col = tid%16, h = kh*group+qr;
    const unsigned n = sparse ? counts[t] : pos0+t+1;
    __shared__ __align__(32) __half qh[16][264], ql[16][264], kv[32][264];
    union Scores { float part[2][16][32]; __half prob[2][16][40]; };
    __shared__ __align__(32) Scores scores;
    __shared__ float qs[16], max_score[16], denom[16], correction[16];
    __shared__ unsigned positions[32];
    float mx = 0;
    for (unsigned d = col; d < D; d += 16)
        if (qr < group) mx = fmaxf(mx,fabsf(q[((uint64_t)t*H+h)*D+d]*scale));
    for (unsigned off = 8; off; off /= 2) mx = fmaxf(mx,__shfl_xor_sync(0xffffffff,mx,off,16));
    const int exp = mx > 0 ? max(-120,min(120,(int)((__float_as_uint(mx)>>23)&255)-127)) : 0;
    const float inv = ldexpf(1,-exp);
    if (!col) { qs[qr] = ldexpf(1,exp); max_score[qr] = -3e38f; denom[qr] = 0; }
    for (unsigned d = col; d < D; d += 16) {
        const float v = qr < group ? q[((uint64_t)t*H+h)*D+d]*scale*inv : 0;
        qh[qr][d] = __float2half_rn(v);
        ql[qr][d] = __float2half_rn((v-__half2float(qh[qr][d]))*4096);
    }
    float result[4][4] = {};
    __syncthreads();
    const unsigned end = min(n,(blockIdx.z+1)*per);
    for (unsigned j0 = blockIdx.z*per; j0 < end; j0 += 32) {
        if (tid < 32) {
            const unsigned j = j0+tid;
            const unsigned p = j < end ? (sparse ? (unsigned)sel[(uint64_t)t*stride+j] : j) : UINT_MAX;
            positions[tid] = p <= pos0+t ? p : UINT_MAX;
        }
        __syncthreads();
        for (unsigned i = tid*8; i < 32*D; i += 256*8) {
            const unsigned r = i/D, d = i%D, p = positions[r];
            tt_cp_async_16B(&kv[r][d],kc+((uint64_t)(p == UINT_MAX ? 0 : p)*Hkv+kh)*D+d,p != UINT_MAX);
        }
        tt_cp_async_commit();
        tt_cp_async_wait_group<0>();
        __syncthreads();
        float hi[4] = {}, lo[4] = {};
        const unsigned split = warp/4, key0 = (warp%4)*8;
        for (unsigned k = split*128; k < (split+1)*128; k += 16) {
            uint32_t ah[4], al[4], b[2];
            tt_ldmatrix_x4(ah,&qh[lane%16][k+(lane/16)*8]);
            tt_ldmatrix_x4(al,&ql[lane%16][k+(lane/16)*8]);
            tt_ldmatrix_x2(b,&kv[key0+lane%8][k+((lane%16)/8)*8]);
            tt_mma_m16n8k16_f16_f32(hi,ah,b);
            tt_mma_m16n8k16_f16_f32(lo,al,b);
        }
        #pragma unroll
        for (unsigned i = 0; i < 4; i++)
            scores.part[split][tt_mma_c_i(lane,i)][key0+tt_mma_c_j(lane,i)] = hi[i]+lo[i]*0x1p-12f;
        __syncthreads();
        float prob[2], peak = max_score[qr];
        #pragma unroll
        for (unsigned i = 0; i < 2; i++) {
            const unsigned key = col+i*16;
            prob[i] = positions[key] != UINT_MAX ?
                (scores.part[0][qr][key]+scores.part[1][qr][key])*qs[qr] : -3e38f;
            peak = fmaxf(peak,prob[i]);
        }
        for (unsigned off = 8; off; off /= 2) peak = fmaxf(peak,__shfl_xor_sync(0xffffffff,peak,off,16));
        const float old = expf(max_score[qr]-peak);
        float total = 0;
        #pragma unroll
        for (unsigned i = 0; i < 2; i++) {
            prob[i] = positions[col+i*16] != UINT_MAX ? expf(prob[i]-peak) : 0;
            total += prob[i];
        }
        for (unsigned off = 8; off; off /= 2) total += __shfl_xor_sync(0xffffffff,total,off,16);
        __syncthreads();
        if (!col) {
            correction[qr] = old;
            max_score[qr] = peak;
            denom[qr] = denom[qr]*old+total;
        }
        #pragma unroll
        for (unsigned i = 0; i < 2; i++) {
            const __half p = __float2half_rn(prob[i]);
            scores.prob[0][qr][col+i*16] = p;
            scores.prob[1][qr][col+i*16] = __float2half_rn((prob[i]-__half2float(p))*4096);
        }
        for (unsigned i = tid*8; i < 32*D; i += 256*8) {
            const unsigned r = i/D, d = i%D, p = positions[r];
            tt_cp_async_16B(&kv[r][d],vc+((uint64_t)(p == UINT_MAX ? 0 : p)*Hkv+kh)*D+d,p != UINT_MAX);
        }
        tt_cp_async_commit();
        tt_cp_async_wait_group<0>();
        __syncthreads();
        #pragma unroll
        for (unsigned tile = 0; tile < 4; tile++) {
            float hi[4] = {}, lo[4] = {};
            #pragma unroll
            for (unsigned k = 0; k < 32; k += 16) {
                uint32_t ah[4], al[4], b[2];
                tt_ldmatrix_x4(ah,&scores.prob[0][lane%16][k+(lane/16)*8]);
                tt_ldmatrix_x4(al,&scores.prob[1][lane%16][k+(lane/16)*8]);
                tt_ldmatrix_x2_trans(b,&kv[k+lane%16][warp*32+tile*8]);
                tt_mma_m16n8k16_f16_f32(hi,ah,b);
                tt_mma_m16n8k16_f16_f32(lo,al,b);
            }
            #pragma unroll
            for (unsigned i = 0; i < 4; i++) result[tile][i] =
                result[tile][i]*correction[tt_mma_c_i(lane,i)]+(hi[i]+lo[i]*0x1p-12f);
        }
        __syncthreads();
    }
    #pragma unroll
    for (unsigned tile = 0; tile < 4; tile++) {
        #pragma unroll
        for (unsigned i = 0; i < 4; i++) {
            const unsigned r = tt_mma_c_i(lane,i), d = warp*32+tile*8+tt_mma_c_j(lane,i);
            if (r < group) {
                const uint64_t dst = ((uint64_t)t*H+kh*group+r)*D+d;
                if (splits == 1) out[dst] = (denom[r] > 0 ? result[tile][i]/denom[r] : 0)*sigmoid(gate[dst]);
                else {
                    float *p = partial+(((uint64_t)t*H+kh*group+r)*splits+blockIdx.z)*(D+2);
                    if (!d) { p[0] = max_score[r]; p[1] = denom[r]; }
                    p[2+d] = result[tile][i];
                }
            }
        }
    }
#endif
}

/* Rows and splits as in attention<D>. */
__global__ void attention_group(float *out, float *partial, const float *q, const float *gate,
        const __half *kc, const __half *vc, const int *sel, const unsigned *counts,
        unsigned H, unsigned Hkv, unsigned pos0, unsigned stride, bool sparse, float scale) {
    pdl_enter();
    const unsigned t = blockIdx.y, D = 256, keys = attn_keys(sparse,stride,pos0+t),
        splits = partial ? attn_splits(keys) : 1;
    if (blockIdx.z >= splits) return;
    attention_group_body(out+(uint64_t)t*H*D,partial ? partial+(uint64_t)t*H*64*(D+2) : nullptr,
        q+(uint64_t)t*H*D,gate+(uint64_t)t*H*D,kc,vc,sparse ? sel+(uint64_t)t*stride : nullptr,
        sparse ? counts+t : nullptr,H,Hkv,pos0+t,stride,sparse,scale,splits,(keys+splits-1)/splits,0);
}

static bool tensor(const ds4_gpu_tensor *t, uint64_t bytes) {
    return t && t->ptr && bytes <= t->bytes;
}

static uint64_t row_bytes(uint32_t type, uint64_t n) {
    switch (type) {
    case 0: return n * 4;
    case 1: case 30: return n * 2;
    case 2: return n % 32 ? 0 : n / 32 * 18;
    case 8: return n % 32 ? 0 : n / 32 * 34;
    case 39: return n % 32 ? 0 : n / 32 * 17;
    case 10: return n % 256 ? 0 : n / 256 * 84;
    case 12: return n % 256 ? 0 : n / 256 * 144;
    case 16: return n % 256 ? 0 : n / 256 * 66;
    default: return 0;
    }
}

static const char *weight(const void *map, uint64_t size, uint64_t off, uint64_t bytes) {
    if (!map || !bytes || off > size || bytes > size - off) return NULL;
    return cuda_resolve_weight_ptr(map, off, bytes, 0, "Qwen weights");
}

static int launched(void) { return cuda_ok(cudaGetLastError(), "Qwen kernel"); }

__device__ __forceinline__ float sum(float x) {
    for (int d = 16; d; d >>= 1) x += __shfl_xor_sync(0xffffffff, x, d);
    return x;
}

__device__ __forceinline__ float block_sum(float x, float *shared) {
    x = sum(x);
    if (!(threadIdx.x & 31)) shared[threadIdx.x / 32] = x;
    __syncthreads();
    float result = 0;
    for (unsigned i = 0; i < blockDim.x / 32; i++) result += shared[i];
    __syncthreads();
    return result;
}

__device__ __forceinline__ float sigmoid(float x) {
    const float e = expf(-fabsf(x));
    return x >= 0 ? 1.0f / (1.0f + e) : e / (1.0f + e);
}

__device__ __forceinline__ float silu(float x) { return x * sigmoid(x); }
__device__ __forceinline__ float softplus(float x) {
    return x > 20 ? x : x < -20 ? expf(x) : log1pf(expf(x));
}

/* These readers preserve the GGUF values, including padded Q2_K down rows.
 * Templates remove unused formats from each matrix kernel. */
template<unsigned TYPE>
__device__ __forceinline__ float value(const char *row, unsigned i,
        const uint64_t *grid_table = NULL, const uint8_t *sign_table = NULL) {
    if (TYPE == 0) return ((const float *)row)[i];
    if (TYPE == 1) return __half2float(((const __half *)row)[i]);
    if (TYPE == 30) return __bfloat162float(((const __nv_bfloat16 *)row)[i]);
    if (TYPE == 8) {
        const char *b = row + (i / 32) * 34;
        return __half2float(*(const __half *)b) * (float)((const int8_t *)b)[2 + i % 32];
    }
    if (TYPE == 2) {
        const uint8_t *b = (const uint8_t *)row + (i / 32) * 18;
        return __half2float(*(const __half *)b) *
            (float)((int)((b[2 + i % 16] >> (4 * (i % 32 / 16))) & 15) - 8);
    }
    if (TYPE == 39) {
        const uint8_t *b = (const uint8_t *)row + (i / 32) * 17;
        const unsigned q = (b[1 + i % 16] >> (4 * (i % 32 / 16))) & 15;
        const float levels[8] = {0, .5f, 1, 1.5f, 2, 3, 4, 6};
        const float scale = b[0] == 0 ? 0x1p-127f : __uint_as_float((unsigned)b[0] << 23);
        return (q & 8 ? -levels[q & 7] : levels[q & 7]) * scale;
    }
    if (TYPE == 10) {
        const cuda_block_q2_K *b = (const cuda_block_q2_K *)row + i / 256;
        const unsigned j = i % 256, sc = b->scales[j / 16];
        const unsigned q = (b->qs[j / 128 * 32 + j % 32] >> (2 * (j % 128 / 32))) & 3;
        return dev_f16_to_f32(b->d) * (sc & 15) * q - dev_f16_to_f32(b->dmin) * (sc >> 4);
    }
    if (TYPE == 12) {
        const cuda_block_q4_K *b = (const cuda_block_q4_K *)row + i / 256;
        const unsigned j = i % 256, group = j / 32;
        unsigned sc, mn;
        if (group < 4) { sc = b->scales[group] & 63; mn = b->scales[group + 4] & 63; }
        else {
            sc = (b->scales[group + 4] & 15) | ((b->scales[group - 4] >> 6) << 4);
            mn = (b->scales[group + 4] >> 4) | ((b->scales[group] >> 6) << 4);
        }
        const unsigned q = (b->qs[j / 64 * 32 + j % 32] >> (4 * (group & 1))) & 15;
        return dev_f16_to_f32(b->d) * sc * q - dev_f16_to_f32(b->dmin) * mn;
    }
    if (TYPE == 16) {
        const cuda_block_iq2_xxs *b = (const cuda_block_iq2_xxs *)row + i / 256;
        const unsigned j = i % 256, group = j / 32, sub = j % 32 / 8;
        const uint16_t *p = b->qs + group * 4;
        const unsigned grid_ids = (unsigned)p[0] | ((unsigned)p[1] << 16);
        const unsigned signs_scale = (unsigned)p[2] | ((unsigned)p[3] << 16);
        const unsigned gi = (grid_ids >> (8 * sub)) & 255, si = (signs_scale >> (7 * sub)) & 127;
        const uint64_t grid = grid_table ? grid_table[gi] : cuda_iq2xxs_grid[gi];
        const unsigned signs = sign_table ? sign_table[si] : cuda_ksigns_iq2xs[si];
        float v = (float)((grid >> (8 * (j & 7))) & 255);
        if (signs & (1u << (j & 7))) v = -v;
        return dev_f16_to_f32(b->d) * (.5f + (signs_scale >> 28)) * .25f * v;
    }
    return 0;
}

__device__ __forceinline__ float scalar(const char *row, unsigned i, unsigned type) {
    switch (type) {
    case 0: return value<0>(row, i);
    case 1: return value<1>(row, i);
    case 2: return value<2>(row, i);
    case 8: return value<8>(row, i);
    case 30: return value<30>(row, i);
    case 39: return value<39>(row, i);
    default: return 0;
    }
}

template<unsigned TYPE>
__device__ __forceinline__ float4 value4(const char *row, unsigned i,
        const uint64_t *grid_table, const uint8_t *sign_table) {
    float4 out;
    float *v = (float *)&out;
    if (TYPE == 1) {
        const uint2 bits = *(const uint2 *)(row+i*2);
        const float2 a = __half22float2(*(__half2 *)&bits.x), b = __half22float2(*(__half2 *)&bits.y);
        out = make_float4(a.x,a.y,b.x,b.y);
    } else if (TYPE == 0) {
        out = *(const float4 *)(row+i*4);
    } else if (TYPE == 30) {
        const uint2 bits = *(const uint2 *)(row+i*2);
        out = make_float4(__uint_as_float(bits.x<<16),__uint_as_float(bits.x&0xffff0000u),
                          __uint_as_float(bits.y<<16),__uint_as_float(bits.y&0xffff0000u));
    } else if (TYPE == 8) {
        const char *b = row+(i/32)*34;
        const float scale = __half2float(*(const __half *)b);
        const uint16_t *q = (const uint16_t *)(b+2+i%32);
        out = make_float4((float)(int8_t)q[0]*scale,(float)(int8_t)(q[0]>>8)*scale,
                         (float)(int8_t)q[1]*scale,(float)(int8_t)(q[1]>>8)*scale);
    } else if (TYPE == 39) {
        const uint8_t *b = (const uint8_t *)row+(i/32)*17;
        const float scale = b[0] == 0 ? 0x1p-127f : __uint_as_float((unsigned)b[0]<<23);
        unsigned packed;
        memcpy(&packed,b+1+i%16,4);
        #pragma unroll
        for (unsigned k = 0; k < 4; k++) {
            const unsigned q = (packed>>(k*8+(i%32/16)*4))&15, mag = q&7;
            const float level = mag < 2 ? .5f*mag : __uint_as_float(((mag/2+126)<<23)|((mag&1)<<22));
            v[k] = (q&8 ? -level : level)*scale;
        }
    } else if (TYPE == 16) {
        const cuda_block_iq2_xxs *b = (const cuda_block_iq2_xxs *)row+i/256;
        const unsigned j = i%256, sub = (j%32)/8;
        const uint16_t *p = b->qs+(j/32)*4;
        const unsigned ids = (unsigned)p[0]|((unsigned)p[1]<<16);
        const unsigned ss = (unsigned)p[2]|((unsigned)p[3]<<16);
        const uint64_t grid = grid_table[(ids>>(8*sub))&255];
        const unsigned signs = sign_table[(ss>>(7*sub))&127];
        const float scale = dev_f16_to_f32(b->d)*(.5f+(ss>>28))*.25f;
        /* Apply four signs before conversion. IQ2's nonzero magnitudes keep
         * the packed negations from carrying into neighboring bytes. */
        const unsigned offset = j&7;
        const unsigned bits = (((signs>>offset)&15)*0x00204081u)&0x01010101u;
        const unsigned packed = (((unsigned)(grid>>(8*offset)))^(bits*255u))+bits;
        #pragma unroll
        for (unsigned k = 0; k < 4; k++) {
            v[k] = scale*(float)(int8_t)(packed>>(8*k));
        }
    } else if (TYPE == 10 || TYPE == 12) {
        const unsigned j = i%256;
        unsigned sc, mn, qs, shift;
        float d, dm;
        if (TYPE == 10) {
            const cuda_block_q2_K *b = (const cuda_block_q2_K *)row+i/256;
            const unsigned s = b->scales[j/16];
            sc = s&15; mn = s>>4;
            d = dev_f16_to_f32(b->d); dm = dev_f16_to_f32(b->dmin);
            qs = *(const unsigned *)(b->qs+(j/128)*32+j%32);
            shift = 2*((j%128)/32);
        } else {
            const cuda_block_q4_K *b = (const cuda_block_q4_K *)row+i/256;
            const unsigned group = j/32;
            if (group < 4) { sc = b->scales[group]&63; mn = b->scales[group+4]&63; }
            else {
                sc = (b->scales[group+4]&15)|((b->scales[group-4]>>6)<<4);
                mn = (b->scales[group+4]>>4)|((b->scales[group]>>6)<<4);
            }
            d = dev_f16_to_f32(b->d); dm = dev_f16_to_f32(b->dmin);
            qs = *(const unsigned *)(b->qs+(j/64)*32+j%32);
            shift = 4*(group&1);
        }
        const float scale = d*sc, offset = dm*mn;
        #pragma unroll
        for (unsigned k = 0; k < 4; k++) v[k] = scale*((qs>>(8*k+shift))&(TYPE == 10 ? 3 : 15))-offset;
    } else {
        #pragma unroll
        for (unsigned k = 0; k < 4; k++) v[k] = value<TYPE>(row,i+k,grid_table,sign_table);
    }
    return out;
}

static uint64_t expert_row_bytes(unsigned type, unsigned K) {
    if (type == 10 && K % 32) return 0;
    return row_bytes(type, type == 10 ? ((uint64_t)K + 255) / 256 * 256 : K);
}

__global__ void router(int *selected, float *weights, const float *logits,
        const float *x, const char *gate, float *shared_gate,
        unsigned NE, unsigned NS, unsigned K, unsigned type) {
    pdl_enter();
    const unsigned t = blockIdx.x, tid = threadIdx.x;
    __shared__ float p[512], maxima[256], red[32];
    float mx = -FLT_MAX;
    for (unsigned e = tid; e < NE; e += 256) mx = fmaxf(mx, logits[(uint64_t)t * NE + e]);
    maxima[tid] = mx;
    __syncthreads();
    for (unsigned stride = 128; stride; stride /= 2) {
        if (tid < stride) maxima[tid] = fmaxf(maxima[tid], maxima[tid + stride]);
        __syncthreads();
    }
    mx = maxima[0];
    float ps = 0;
    for (unsigned e = tid; e < NE; e += 256) { p[e] = expf(logits[(uint64_t)t * NE + e] - mx); ps += p[e]; }
    const float total = block_sum(ps, red);
    for (unsigned e = tid; e < NE; e += 256) p[e] /= total;
    if (K) {
        float v = 0;
        for (unsigned i = tid; i < K; i += 256) v += scalar(gate, i, type) * x[(uint64_t)t * K + i];
        v = block_sum(v, red);
        if (!tid) shared_gate[t] = v;
    }
    __syncthreads();
    /* Top-NS by repeated argmax (ties to the lower expert id), done by one
     * warp from registers: NE <= 512 gives 16 candidates per lane, and each
     * pick costs one shuffle reduction instead of a block reduction. */
    if (tid < 32) {
        float v[16];
        #pragma unroll
        for (unsigned j = 0; j < 16; j++) v[j] = tid + 32 * j < NE ? p[tid + 32 * j] : -1;
        for (unsigned s = 0; s < NS; s++) {
            float best = -1;
            unsigned id = UINT_MAX;
            #pragma unroll
            for (unsigned j = 0; j < 16; j++) if (v[j] > best) { best = v[j]; id = tid + 32 * j; }
            for (unsigned off = 16; off; off /= 2) {
                const float ob = __shfl_xor_sync(0xffffffff, best, off);
                const unsigned oi = __shfl_xor_sync(0xffffffff, id, off);
                if (ob > best || (ob == best && oi < id)) { best = ob; id = oi; }
            }
            /* Non-finite router logits leave no candidate; keep the expert
             * index in range rather than handing UINT_MAX to the tables. */
            if (id >= NE) { id = s; best = 0; }
            #pragma unroll
            for (unsigned j = 0; j < 16; j++) if (tid + 32 * j == id) v[j] = -1;
            if (!tid) {
                selected[(uint64_t)t * NS + s] = id;
                weights[(uint64_t)t * NS + s] = best;
            }
        }
    }
    __syncthreads();
    if (tid < NS) {
        float denom = 0;
        for (unsigned s = 0; s < NS; s++) denom += weights[(uint64_t)t * NS + s];
        red[tid] = denom;
    }
    __syncthreads();
    if (tid < NS) weights[(uint64_t)t * NS + tid] /= red[tid];
}

/* router() with an F32 shared gate of K = 256 * KPT. Each thread's KPT gate
 * values are loaded before the dependency wait (the weights are immutable),
 * so the cold 10 KB row no longer arrives from DRAM one dependent load at a
 * time after the router GEMV. Every value rounds as in router(): the gate
 * dot keeps its per-thread order and block_sum, the maximum is exact in any
 * order, and the softmax sum, division and selection are router()'s.
 * Other gate types and widths keep router(). */
template<unsigned KPT>
__global__ void __launch_bounds__(256) router_pre(int *selected, float *weights, const float *logits,
        const float *x, const float *gate, float *shared_gate, unsigned NE, unsigned NS) {
    const unsigned t = blockIdx.x, tid = threadIdx.x;
    float g[KPT];
    #pragma unroll
    for (unsigned j = 0; j < KPT; j++) g[j] = gate[tid + 256 * j];
    pdl_enter();
    __shared__ float p[512], red[32];
    float l[2], xv[KPT];
    #pragma unroll
    for (unsigned j = 0; j < 2; j++)
        l[j] = tid + 256 * j < NE ? logits[(uint64_t)t * NE + tid + 256 * j] : -FLT_MAX;
    #pragma unroll
    for (unsigned j = 0; j < KPT; j++) xv[j] = x[(uint64_t)t * KPT * 256 + tid + 256 * j];
    float sg = 0;
    #pragma unroll
    for (unsigned j = 0; j < KPT; j++) sg += g[j] * xv[j];
    sg = block_sum(sg, red);
    if (!tid) shared_gate[t] = sg;
    float mx = fmaxf(fmaxf(-FLT_MAX, l[0]), l[1]);
    for (int d = 16; d; d >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, d));
    if (!(tid & 31)) red[tid / 32] = mx;
    __syncthreads();
    mx = red[0];
    #pragma unroll
    for (unsigned w = 1; w < 8; w++) mx = fmaxf(mx, red[w]);
    __syncthreads();
    float ps = 0;
    #pragma unroll
    for (unsigned j = 0; j < 2; j++) {
        const unsigned e = tid + 256 * j;
        if (e < NE) { p[e] = expf(l[j] - mx); ps += p[e]; }
    }
    const float total = block_sum(ps, red);
    #pragma unroll
    for (unsigned j = 0; j < 2; j++) if (tid + 256 * j < NE) p[tid + 256 * j] /= total;
    __syncthreads();
    if (tid < 32) {
        float v[16];
        #pragma unroll
        for (unsigned j = 0; j < 16; j++) v[j] = tid + 32 * j < NE ? p[tid + 32 * j] : -1;
        for (unsigned s = 0; s < NS; s++) {
            float best = -1;
            unsigned id = UINT_MAX;
            #pragma unroll
            for (unsigned j = 0; j < 16; j++) if (v[j] > best) { best = v[j]; id = tid + 32 * j; }
            for (unsigned off = 16; off; off /= 2) {
                const float ob = __shfl_xor_sync(0xffffffff, best, off);
                const unsigned oi = __shfl_xor_sync(0xffffffff, id, off);
                if (ob > best || (ob == best && oi < id)) { best = ob; id = oi; }
            }
            if (id >= NE) { id = s; best = 0; }
            #pragma unroll
            for (unsigned j = 0; j < 16; j++) if (tid + 32 * j == id) v[j] = -1;
            if (!tid) {
                selected[(uint64_t)t * NS + s] = id;
                weights[(uint64_t)t * NS + s] = best;
            }
        }
    }
    __syncthreads();
    if (tid < NS) {
        float denom = 0;
        for (unsigned s = 0; s < NS; s++) denom += weights[(uint64_t)t * NS + s];
        red[tid] = denom;
    }
    __syncthreads();
    if (tid < NS) weights[(uint64_t)t * NS + tid] /= red[tid];
}

/* SH is the shared expert's type when known at compile time, or SH_ANY. */
enum : unsigned { SH_ANY = 255 };

template<unsigned TYPE, bool DOWN, unsigned SH>
__global__ void moe_mv(float *out, const float *x, const int *selected,
        const char *w0, const char *w1, const char *sh0, const char *sh1,
        unsigned shared_type, unsigned NE, unsigned NS, unsigned K, unsigned M,
        uint64_t rb, uint64_t srb) {
    pdl_enter();
    const unsigned row = blockIdx.x * 4 + threadIdx.x / 32, slot = blockIdx.y, t = blockIdx.z;
    __shared__ uint64_t grid_table[TYPE == 16 ? 256 : 1];
    __shared__ uint8_t sign_table[TYPE == 16 ? 128 : 1];
    if (TYPE == 16) {
        for (unsigned i = threadIdx.x; i < 256; i += blockDim.x) grid_table[i] = cuda_iq2xxs_grid[i];
        for (unsigned i = threadIdx.x; i < 128; i += blockDim.x) sign_table[i] = cuda_ksigns_iq2xs[i];
        __syncthreads();
    }
    if (row >= M) return;
    const bool shared = slot == NS;
    const unsigned stride = NS + (shared_type != UINT_MAX);
    const uint64_t pair = (uint64_t)t * stride + slot;
    const float *xt = x + (DOWN ? pair : t) * K;
    float a = 0, b = 0;
    if (shared && SH != SH_ANY) {
        /* Same element order as the generic loop; the known type lets the
         * loads of several iterations be in flight together. */
        #pragma unroll 8
        for (unsigned i = threadIdx.x & 31; i < K; i += 32) {
            a += value<SH>(sh0 + row * srb, i) * xt[i];
            if (!DOWN) b += value<SH>(sh1 + row * srb, i) * xt[i];
        }
        a = sum(a); b = sum(b);
    } else if (shared) {
        for (unsigned i = threadIdx.x & 31; i < K; i += 32) {
            a += scalar(sh0 + row * srb, i, shared_type) * xt[i];
            if (!DOWN) b += scalar(sh1 + row * srb, i, shared_type) * xt[i];
        }
        a = sum(a); b = sum(b);
    } else {
        const int e = selected[(uint64_t)t * NS + slot];
        if (e >= 0 && (unsigned)e < NE) {
            const uint64_t off = ((uint64_t)e * M + row) * rb;
            if ((TYPE == 16 || TYPE == 10 || TYPE == 12 || TYPE == 39) && !((uintptr_t)xt&15)) {
                #pragma unroll 4
                for (unsigned i = (threadIdx.x&31)*4; i < K; i += 128) {
                    const float4 xv = *(const float4 *)(xt+i);
                    const float4 av = value4<TYPE>(w0+off,i,grid_table,sign_table);
                    a += av.x*xv.x; a += av.y*xv.y; a += av.z*xv.z; a += av.w*xv.w;
                    if (!DOWN) {
                        const float4 bv = value4<TYPE>(w1+off,i,grid_table,sign_table);
                        b += bv.x*xv.x; b += bv.y*xv.y; b += bv.z*xv.z; b += bv.w*xv.w;
                    }
                }
                a = sum(a); b = sum(b);
            } else {
                a = dot<TYPE>(w0+off,xt,K,grid_table,sign_table);
                if (!DOWN) b = dot<TYPE>(w1+off,xt,K,grid_table,sign_table);
            }
        }
    }
    if (!(threadIdx.x & 31)) out[pair * M + row] = DOWN ? a : silu(a) * b;
}

/* Decode MoE for the Qwen packs (Q4_K gate/up, MXFP4 down, Q8_0 shared
 * expert), laid out for bandwidth. moe_mv decodes four elements per step
 * and re-derives the Q4_K block scales every time, which costs about fifteen
 * instructions per weight and keeps it near 60% of DRAM bandwidth. Here
 * each lane takes eight elements of a Q4_K superblock sharing two scale
 * groups, and the MXFP4 rows (17-byte blocks) are staged in shared memory
 * and decoded through a 16-entry table. Every weight decodes to the same
 * value; the sums are taken in a different order. One kernel serves every
 * row count, so MTP verify rows keep matching single-token decode. */
__device__ __forceinline__ void moe_q8_row(float &a, float &b, const char *r0, const char *r1,
        const float *xt, unsigned K, bool two) {
    #pragma unroll 4
    for (unsigned i = (threadIdx.x & 31) * 4; i < K; i += 128) {
        const float4 v = *(const float4 *)(xt + i);
        const char *q0 = r0 + (i / 32) * 34;
        const uint16_t *p0 = (const uint16_t *)(q0 + 2 + i % 32);
        float p = (float)(int8_t)p0[0] * v.x;
        p += (float)(int8_t)(p0[0] >> 8) * v.y;
        p += (float)(int8_t)p0[1] * v.z;
        p += (float)(int8_t)(p0[1] >> 8) * v.w;
        a += p * __half2float(*(const __half *)q0);
        if (two) {
            const char *q1 = r1 + (i / 32) * 34;
            const uint16_t *p1 = (const uint16_t *)(q1 + 2 + i % 32);
            p = (float)(int8_t)p1[0] * v.x;
            p += (float)(int8_t)(p1[0] >> 8) * v.y;
            p += (float)(int8_t)p1[1] * v.z;
            p += (float)(int8_t)(p1[1] >> 8) * v.w;
            b += p * __half2float(*(const __half *)q1);
        }
    }
}

/* Scales and mins of Q4_K groups 2c and 2c + 1 from the 16-byte header
 * (d, dmin, scales[12]); the same values value<12> derives. */
__device__ __forceinline__ void q4k_group_pair(const uint4 h, unsigned c,
        float &s0, float &o0, float &s1, float &o1) {
    const float d = dev_f16_to_f32((uint16_t)(h.x & 0xffff)), dm = dev_f16_to_f32((uint16_t)(h.x >> 16));
    unsigned sc0, mn0, sc1, mn1;
    if (c < 2) {
        const unsigned lo = h.y >> (16 * c), hi = h.z >> (16 * c);
        sc0 = lo & 63; sc1 = (lo >> 8) & 63; mn0 = hi & 63; mn1 = (hi >> 8) & 63;
    } else {
        const unsigned e = h.w >> (16 * (c - 2)), lo = h.y >> (16 * (c - 2)), hi = h.z >> (16 * (c - 2));
        sc0 = (e & 15) | (((lo >> 6) & 3) << 4);
        mn0 = ((e >> 4) & 15) | (((hi >> 6) & 3) << 4);
        sc1 = ((e >> 8) & 15) | (((lo >> 14) & 3) << 4);
        mn1 = ((e >> 12) & 15) | (((hi >> 14) & 3) << 4);
    }
    s0 = d * sc0; o0 = dm * mn0; s1 = d * sc1; o1 = dm * mn1;
}

__device__ __forceinline__ float q4k_dot8(unsigned q, float s0, float o0, float s1, float o1, float4 x0, float4 x1) {
    float a = (s0 * (q & 15) - o0) * x0.x;
    a += (s0 * ((q >> 8) & 15) - o0) * x0.y;
    a += (s0 * ((q >> 16) & 15) - o0) * x0.z;
    a += (s0 * ((q >> 24) & 15) - o0) * x0.w;
    a += (s1 * ((q >> 4) & 15) - o1) * x1.x;
    a += (s1 * ((q >> 12) & 15) - o1) * x1.y;
    a += (s1 * ((q >> 20) & 15) - o1) * x1.z;
    a += (s1 * (q >> 28) - o1) * x1.w;
    return a;
}

__device__ __forceinline__ void l2_prefetch(const void *p, unsigned bytes) {
#if __CUDA_ARCH__ >= 900
    asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(p), "r"(bytes) : "memory");
#endif
}

/* With pd set, the shared-expert blocks first ask L2 for their own gate and
 * up rows and for a 1/gridDim.x slice of the shared down matrix pd (pdb
 * bytes). They do it before the dependency wait: these are weights, and
 * under programmatic launch this grid is resident while the one-block router
 * runs, when DRAM is otherwise idle. Addresses and sizes are 16-byte
 * multiples (checked by the caller). */
__global__ void moe_gate_up_q4k(float *out, const float *x, const int *selected,
        const char *w0, const char *w1, const char *sh0, const char *sh1,
        unsigned NE, unsigned NS, unsigned stride, unsigned K, unsigned M, uint64_t rb, uint64_t srb,
        const char *pd, uint64_t pdb) {
    if (pd && blockIdx.y == NS && !blockIdx.z && !threadIdx.x && blockIdx.x * 4 < M) {
        const unsigned r0 = blockIdx.x * 4, nr = min(4u, M - r0);
        l2_prefetch(sh0 + r0 * srb, (unsigned)(nr * srb));
        l2_prefetch(sh1 + r0 * srb, (unsigned)(nr * srb));
        const uint64_t chunk = (pdb / gridDim.x + 15) & ~(uint64_t)15, at = blockIdx.x * chunk;
        if (at < pdb) l2_prefetch(pd + at, (unsigned)min(chunk, pdb - at));
    }
    pdl_enter();
    const unsigned row = blockIdx.x * 4 + threadIdx.x / 32, slot = blockIdx.y, t = blockIdx.z, lane = threadIdx.x & 31;
    if (row >= M) return;
    const uint64_t pair = (uint64_t)t * stride + slot;
    const float *xt = x + (uint64_t)t * K;
    float a = 0, b = 0;
    if (slot == NS) {
        moe_q8_row(a, b, sh0 + row * srb, sh1 + row * srb, xt, K, true);
    } else {
        const int e = selected[(uint64_t)t * NS + slot];
        if (e >= 0 && (unsigned)e < NE) {
            const uint64_t off = ((uint64_t)e * M + row) * rb;
            const cuda_block_q4_K *g = (const cuda_block_q4_K *)(w0 + off), *u = (const cuda_block_q4_K *)(w1 + off);
            const unsigned c = lane / 8, q4 = (lane % 8) * 4;
            #pragma unroll 2
            for (unsigned sb = 0; sb < K / 256; sb++) {
                const unsigned qg = *(const unsigned *)(g[sb].qs + c * 32 + q4);
                const unsigned qu = *(const unsigned *)(u[sb].qs + c * 32 + q4);
                const uint4 hg = *(const uint4 *)(g + sb), hu = *(const uint4 *)(u + sb);
                const float *xs = xt + sb * 256 + c * 64 + q4;
                const float4 x0 = *(const float4 *)xs, x1 = *(const float4 *)(xs + 32);
                float s0, o0, s1, o1;
                q4k_group_pair(hg, c, s0, o0, s1, o1);
                a += q4k_dot8(qg, s0, o0, s1, o1, x0, x1);
                q4k_group_pair(hu, c, s0, o0, s1, o1);
                b += q4k_dot8(qu, s0, o0, s1, o1, x0, x1);
            }
        }
    }
    a = sum(a); b = sum(b);
    if (!lane) out[pair * M + row] = silu(a) * b;
}

/* Two rows per warp: an MXFP4 down row is only 340 bytes. */
__global__ void moe_down_mxfp4(float *out, const float *x, const int *selected,
        const char *w0, const char *sh0, unsigned NE, unsigned NS, unsigned stride,
        unsigned K, unsigned M, uint64_t rb, uint64_t srb) {
    pdl_enter();
    extern __shared__ __align__(16) unsigned moe_stage[];
    __shared__ float lut[16];
    if (threadIdx.x < 16) {
        const unsigned mag = threadIdx.x & 7;
        const float level = mag < 2 ? .5f*mag : __uint_as_float(((mag/2+126)<<23)|((mag&1)<<22));
        lut[threadIdx.x] = threadIdx.x & 8 ? -level : level;
    }
    __syncthreads();
    const unsigned lane = threadIdx.x & 31, warp = threadIdx.x / 32, rw = (unsigned)(rb / 4);
    const unsigned row0 = (blockIdx.x * 4 + warp) * 2, slot = blockIdx.y, t = blockIdx.z;
    if (row0 >= M) return;
    const unsigned rows = min(2u, M - row0);
    const uint64_t pair = (uint64_t)t * stride + slot;
    const float *xt = x + pair * K;
    float acc[2] = {};
    if (slot == NS) {
        float unused = 0;
        for (unsigned r = 0; r < rows; r++) moe_q8_row(acc[r], unused, sh0 + (row0 + r) * srb, NULL, xt, K, false);
    } else {
        const int e = selected[(uint64_t)t * NS + slot];
        if (e >= 0 && (unsigned)e < NE) {
            const unsigned *src = (const unsigned *)(w0 + ((uint64_t)e * M + row0) * rb);
            unsigned *stage = moe_stage + warp * (2 * rw + 1);
            const unsigned words = rows * rw;
            for (unsigned k0 = 0; k0 < words; k0 += 128) {
                unsigned r[4];
                #pragma unroll
                for (unsigned k = 0; k < 4; k++) if (k0 + lane + 32 * k < words) r[k] = src[k0 + lane + 32 * k];
                #pragma unroll
                for (unsigned k = 0; k < 4; k++) if (k0 + lane + 32 * k < words) stage[k0 + lane + 32 * k] = r[k];
            }
            __syncwarp();
            const unsigned sub = lane % 4;
            for (unsigned blk = lane / 4; blk < K / 32; blk += 8) {
                const float *xs = xt + blk * 32 + 4 * sub;
                const float4 x0 = *(const float4 *)xs, x1 = *(const float4 *)(xs + 16);
                for (unsigned r = 0; r < rows; r++) {
                    const unsigned *sw = stage + r * rw, at = blk * 17 + 1 + 4 * sub;
                    const unsigned q = __funnelshift_r(sw[at / 4], sw[at / 4 + 1], (at & 3) * 8);
                    const unsigned eb = ((const uint8_t *)sw)[blk * 17];
                    const float scale = eb == 0 ? 0x1p-127f : __uint_as_float(eb << 23);
                    float p = lut[q & 15] * x0.x;
                    p += lut[(q >> 8) & 15] * x0.y;
                    p += lut[(q >> 16) & 15] * x0.z;
                    p += lut[(q >> 24) & 15] * x0.w;
                    p += lut[(q >> 4) & 15] * x1.x;
                    p += lut[(q >> 12) & 15] * x1.y;
                    p += lut[(q >> 20) & 15] * x1.z;
                    p += lut[q >> 28] * x1.w;
                    acc[r] += p * scale;
                }
            }
        }
    }
    for (unsigned r = 0; r < rows; r++) {
        const float v = sum(acc[r]);
        if (!lane) out[pair * M + row0 + r] = v;
    }
}

/* moe_down_mxfp4 for the decode shape (two rows of at most 96 words and at
 * most 24 MXFP4 blocks per row, i.e. K <= 768), with the latency taken out
 * of each warp's path. The activation loads go out first, as they do not
 * depend on the expert id; each lane then loads all of its weight words in
 * one batch instead of two dependent ones; and the shared expert, whose
 * Q8_0 rows are twice as long as a routed row, runs both of its rows in one
 * pass and is dispatched as blockIdx.y == 0, so it lands in the first wave
 * instead of the tail of the second. Each row's arithmetic and lane order
 * are those of moe_down_mxfp4, so the results are identical. */
__global__ void __launch_bounds__(128, 10) moe_down_mxfp4_lat(float *out, const float *x, const int *selected,
        const char *w0, const char *sh0, unsigned NE, unsigned NS, unsigned stride,
        unsigned K, unsigned M, uint64_t rb, uint64_t srb) {
    pdl_enter();
    extern __shared__ __align__(16) unsigned moe_stage[];
    __shared__ float lut[16];
    if (threadIdx.x < 16) {
        const unsigned mag = threadIdx.x & 7;
        const float level = mag < 2 ? .5f*mag : __uint_as_float(((mag/2+126)<<23)|((mag&1)<<22));
        lut[threadIdx.x] = threadIdx.x & 8 ? -level : level;
    }
    __syncthreads();
    const unsigned lane = threadIdx.x & 31, warp = threadIdx.x / 32, rw = (unsigned)(rb / 4);
    const unsigned row0 = (blockIdx.x * 4 + warp) * 2, t = blockIdx.z;
    const unsigned slot = stride > NS ? (blockIdx.y ? blockIdx.y - 1 : NS) : blockIdx.y;
    if (row0 >= M) return;
    const unsigned rows = min(2u, M - row0), words = rows * rw;
    const uint64_t pair = (uint64_t)t * stride + slot;
    const float *xt = x + pair * K;
    float acc[2] = {};
    if (slot == NS) {
        moe_q8_row(acc[0], acc[1], sh0 + row0 * srb, sh0 + (row0 + 1) * srb, xt, K, rows > 1);
    } else {
        const unsigned sub = lane % 4, nb = K / 32;
        float4 xv[3][2];
        #pragma unroll
        for (unsigned j = 0; j < 3; j++) if (lane / 4 + 8 * j < nb) {
            const float *xs = xt + (lane / 4 + 8 * j) * 32 + 4 * sub;
            xv[j][0] = *(const float4 *)xs;
            xv[j][1] = *(const float4 *)(xs + 16);
        }
        const int e = selected[(uint64_t)t * NS + slot];
        if (e >= 0 && (unsigned)e < NE) {
            const unsigned *src = (const unsigned *)(w0 + ((uint64_t)e * M + row0) * rb);
            unsigned r[6];
            #pragma unroll
            for (unsigned k = 0; k < 6; k++) if (lane + 32 * k < words) r[k] = src[lane + 32 * k];
            unsigned *stage = moe_stage + warp * (2 * rw + 1);
            #pragma unroll
            for (unsigned k = 0; k < 6; k++) if (lane + 32 * k < words) stage[lane + 32 * k] = r[k];
            __syncwarp();
            #pragma unroll
            for (unsigned j = 0; j < 3; j++) {
                const unsigned blk = lane / 4 + 8 * j;
                if (blk >= nb) break;
                const float4 x0 = xv[j][0], x1 = xv[j][1];
                for (unsigned rr = 0; rr < rows; rr++) {
                    const unsigned *sw = stage + rr * rw, at = blk * 17 + 1 + 4 * sub;
                    const unsigned q = __funnelshift_r(sw[at / 4], sw[at / 4 + 1], (at & 3) * 8);
                    const unsigned eb = ((const uint8_t *)sw)[blk * 17];
                    const float scale = eb == 0 ? 0x1p-127f : __uint_as_float(eb << 23);
                    float p = lut[q & 15] * x0.x;
                    p += lut[(q >> 8) & 15] * x0.y;
                    p += lut[(q >> 16) & 15] * x0.z;
                    p += lut[(q >> 24) & 15] * x0.w;
                    p += lut[(q >> 4) & 15] * x1.x;
                    p += lut[(q >> 12) & 15] * x1.y;
                    p += lut[(q >> 20) & 15] * x1.z;
                    p += lut[q >> 28] * x1.w;
                    acc[rr] += p * scale;
                }
            }
        }
    }
    for (unsigned rr = 0; rr < rows; rr++) {
        const float v = sum(acc[rr]);
        if (!lane) out[pair * M + row0 + rr] = v;
    }
}

/* A block belongs to a selected (token,slot), but only the first occurrence
 * of that expert runs. Each pass reuses its unpacked weights across four
 * tokens. The per-token K and warp reduction orders match the decode path. */
template<bool DOWN>
__global__ void moe_grouped(float *out, const float *x, const int *selected,
        const int *lists, const int *counts, const char *w0, const char *w1,
        unsigned NE, unsigned NS, unsigned K, unsigned M, unsigned cap, uint64_t rb) {
    pdl_enter();
    extern __shared__ __align__(16) unsigned stage_all[];
    __shared__ float lut[16];
    if (DOWN) {
        if (threadIdx.x < 16) {
            const unsigned mag = threadIdx.x & 7;
            const float v = mag < 2 ? .5f*mag : __uint_as_float(((mag/2+126)<<23)|((mag&1)<<22));
            lut[threadIdx.x] = threadIdx.x & 8 ? -v : v;
        }
        __syncthreads();
    }
    const unsigned lane = threadIdx.x & 31, warp = threadIdx.x / 32;
    const unsigned pair = blockIdx.y, row0 = (blockIdx.x*4+warp)*(DOWN ? 2 : 1);
    if (row0 >= M) return;
    const int e = selected[pair];
    if (e < 0 || (unsigned)e >= NE) {
        if (!lane) for (unsigned r = 0; r < (DOWN ? 2u : 1u) && row0+r < M; r++)
            out[(uint64_t)pair*M+row0+r] = 0;
        return;
    }
    const int *list = lists + (uint64_t)e*cap;
    const unsigned count = min((unsigned)counts[e],cap);
    if (!count || (unsigned)list[0] != pair) return;
    for (unsigned first = 0; first < count; first += 4) {
        const unsigned n = min(4u,count-first);
        float a[4][2] = {}, b[4] = {};
        if (!DOWN) {
            const uint64_t off = ((uint64_t)e*M+row0)*rb;
            const cuda_block_q4_K *g = (const cuda_block_q4_K *)(w0+off), *u = (const cuda_block_q4_K *)(w1+off);
            const unsigned c = lane/8, q4 = (lane%8)*4;
            for (unsigned sb = 0; sb < K/256; sb++) {
                const unsigned qg = *(const unsigned *)(g[sb].qs+c*32+q4), qu = *(const unsigned *)(u[sb].qs+c*32+q4);
                float gs0,go0,gs1,go1,us0,uo0,us1,uo1;
                q4k_group_pair(*(const uint4 *)(g+sb),c,gs0,go0,gs1,go1);
                q4k_group_pair(*(const uint4 *)(u+sb),c,us0,uo0,us1,uo1);
                #pragma unroll
                for (unsigned j = 0; j < 4; j++) if (j < n) {
                    const float *xs = x+(uint64_t)(list[first+j]/NS)*K+sb*256+c*64+q4;
                    const float4 x0 = *(const float4 *)xs, x1 = *(const float4 *)(xs+32);
                    a[j][0] += q4k_dot8(qg,gs0,go0,gs1,go1,x0,x1);
                    b[j] += q4k_dot8(qu,us0,uo0,us1,uo1,x0,x1);
                }
            }
        } else {
            const unsigned rw = rb/4, nr = min(2u,M-row0);
            unsigned *stage = stage_all+warp*(2*rw+1);
            const unsigned *src = (const unsigned *)(w0+((uint64_t)e*M+row0)*rb);
            for (unsigned i = lane; i < nr*rw; i += 32) stage[i] = src[i];
            __syncwarp();
            const unsigned sub = lane%4;
            for (unsigned blk = lane/4; blk < K/32; blk += 8) {
                #pragma unroll
                for (unsigned r = 0; r < 2; r++) if (r < nr) {
                    const unsigned *sw = stage+r*rw, at = blk*17+1+4*sub;
                    const unsigned q = __funnelshift_r(sw[at/4],sw[at/4+1],(at&3)*8);
                    const unsigned eb = ((const uint8_t *)sw)[blk*17];
                    const float scale = eb == 0 ? 0x1p-127f : __uint_as_float(eb<<23);
                    const float v[8] = {lut[q&15],lut[(q>>8)&15],lut[(q>>16)&15],lut[(q>>24)&15],
                        lut[(q>>4)&15],lut[(q>>12)&15],lut[(q>>20)&15],lut[q>>28]};
                    #pragma unroll
                    for (unsigned j = 0; j < 4; j++) if (j < n) {
                        const float *xs = x+(uint64_t)list[first+j]*K+blk*32+4*sub;
                        const float4 x0 = *(const float4 *)xs, x1 = *(const float4 *)(xs+16);
                        float p = v[0]*x0.x;
                        p += v[1]*x0.y; p += v[2]*x0.z; p += v[3]*x0.w;
                        p += v[4]*x1.x; p += v[5]*x1.y; p += v[6]*x1.z; p += v[7]*x1.w;
                        a[j][r] += p*scale;
                    }
                }
            }
        }
        #pragma unroll
        for (unsigned j = 0; j < 4; j++) if (j < n) {
            const float up = DOWN ? 0 : sum(b[j]);
            #pragma unroll
            for (unsigned r = 0; r < (DOWN ? 2u : 1u); r++) if (row0+r < M) {
                const float v = sum(a[j][r]);
                if (!lane) out[(uint64_t)list[first+j]*M+row0+r] = DOWN ? v : silu(v)*up;
            }
        }
        if (DOWN) __syncwarp();
    }
}

static int moe_mv_dispatch(float *out, const float *x, const int *sel,
        const char *w0, const char *w1, const char *s0, const char *s1,
        unsigned type, unsigned st, unsigned NE, unsigned T, unsigned NS, unsigned K, unsigned M, bool down,
        const char *pd = NULL, uint64_t pdb = 0, bool ref = false) {
    const uint64_t rb = expert_row_bytes(type, K), srb = row_bytes(st, K);
    const dim3 grid((M + 3) / 4, NS + (st != UINT_MAX), T);
    const unsigned stride = NS + (st != UINT_MAX);
    const bool xa = !((uintptr_t)x & 15), sha = st == UINT_MAX || (st == 8 && !(K % 128) && !((uintptr_t)s0 & 1) && !(srb & 1));
    if (xa && sha && !down && type == 12 && !(K % 256) && !(rb & 15) &&
        !((uintptr_t)w0 & 15) && !((uintptr_t)w1 & 15)) {
        /* the shared-expert L2 prefetch needs 16-byte rows and ranges */
        const bool pf = pd && st == 8 && !(srb & 15) && !((uintptr_t)s0 & 15) && !((uintptr_t)s1 & 15) &&
            !((uintptr_t)pd & 15) && !(pdb & 15) && 4 * srb < (1u << 20);
        launch(moe_gate_up_q4k, grid, 128, 0, out, x, sel, w0, w1, s0, s1, NE, NS, stride, K, M, rb, srb,
               pf ? pd : (const char *)NULL, pf ? pdb : (uint64_t)0);
        return launched();
    }
    if (xa && sha && down && type == 39 && !(K % 32) && !(rb & 3) && rb <= 4096 &&
        !((uintptr_t)w0 & 3)) {
        const dim3 g2((M + 7) / 8, grid.y, T);
        const size_t smem = (size_t)4 * (2 * (rb / 4) + 1) * 4;
        /* ref (tests) keeps the original kernel as the byte-exact reference. */
        if (rb <= 384 && K <= 768 && !ref)
            launch(moe_down_mxfp4_lat, g2, 128, smem, out, x, sel, w0, s0, NE, NS, stride, K, M, rb, srb);
        else
            launch(moe_down_mxfp4, g2, 128, smem, out, x, sel, w0, s0, NE, NS, stride, K, M, rb, srb);
        return launched();
    }
#define QWEN_MOE_SH(TYPE, SH) \
    if (down) launch(moe_mv<TYPE, true, SH>, grid, 128, 0, out,x,sel,w0,w1,s0,s1,st,NE,NS,K,M,rb,srb); \
    else launch(moe_mv<TYPE, false, SH>, grid, 128, 0, out,x,sel,w0,w1,s0,s1,st,NE,NS,K,M,rb,srb)
#define QWEN_MOE(TYPE) case TYPE: \
    if (st == 8) { QWEN_MOE_SH(TYPE, 8); } else { QWEN_MOE_SH(TYPE, SH_ANY); } break
    switch (type) {
    QWEN_MOE(0); QWEN_MOE(1); QWEN_MOE(2); QWEN_MOE(8); QWEN_MOE(10);
    QWEN_MOE(12); QWEN_MOE(16); QWEN_MOE(30); QWEN_MOE(39);
    default: return 0;
    }
#undef QWEN_MOE
#undef QWEN_MOE_SH
    return launched();
}

__global__ void moe_reduce(float *out, float *R, const float *inj, const float *part,
        const float *weights, const float *gate, const float *shared,
        unsigned NS, unsigned stride, unsigned D, unsigned hc) {
    pdl_enter();
    const unsigned d = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y;
    __shared__ float inject[4];
    if (threadIdx.x < hc) inject[threadIdx.x] = injection(inj + (uint64_t)t * hc * hc * 8, hc, threadIdx.x);
    __syncthreads();
    if (d >= D) return;
    float v = 0;
    for (unsigned s = 0; s < NS; s++) v += weights[(uint64_t)t * NS + s] * part[((uint64_t)t * stride + s) * D + d];
    if (gate) v += sigmoid(gate[t]) * (shared ? shared[(uint64_t)t * D + d] : part[((uint64_t)t * stride + NS) * D + d]);
    out[(uint64_t)t * D + d] = v;
    for (unsigned s = 0; s < hc; s++) R[((uint64_t)t * hc + s) * D + d] += inject[s] * v;
}

__global__ void expert_lists(int *lists, int *counts, const int *selected,
        unsigned pairs, unsigned NE, unsigned cap) {
    pdl_enter();
    __shared__ unsigned counters[512];
    for (unsigned e = threadIdx.x; e < NE; e += blockDim.x) counters[e] = 0;
    __syncthreads();
    for (unsigned p = threadIdx.x; p < pairs; p += blockDim.x) {
        const unsigned e = (unsigned)selected[p];
        if (e < NE) {
            const unsigned i = atomicAdd(counters + e, 1u);
            if (i < cap) lists[(uint64_t)e * cap + i] = p;
        }
    }
    __syncthreads();
    for (unsigned e = threadIdx.x; e < NE; e += blockDim.x) counts[e] = min(counters[e], cap);
}

/* Tiles share dequantized weights across 16 tokens without requantizing the
 * activations. Expert counts determine the work; no expert list is truncated. */
template<unsigned TYPE, bool DOWN, bool EXPERT>
__global__ void matrix(float *out, const float *x, const char *w0, const char *w1,
        const int *lists, const int *counts, unsigned T, unsigned NS, unsigned NO,
        unsigned K, unsigned M, unsigned cap, uint64_t rb) {
    pdl_enter();
    const unsigned r = threadIdx.x % 16, c = threadIdx.x / 16, e = blockIdx.y;
    const unsigned row = blockIdx.x * 16 + r;
    const unsigned count = EXPERT ? (unsigned)counts[e] : T;
    __shared__ float a[16][33], b[16][33], v[16][33];
    for (unsigned t0 = blockIdx.z * 16; t0 < count; t0 += gridDim.z * 16) {
        const unsigned item = t0 + c;
        const int pair = EXPERT && item < count ? lists[(uint64_t)e * cap + item] : (int)item;
        const unsigned tok = EXPERT ? pair / NS : item, slot = EXPERT ? pair % NS : 0;
        float acc = 0, up = 0;
        for (unsigned k0 = 0; k0 < K; k0 += 32) {
            for (unsigned j = threadIdx.x; j < 16 * 32; j += 256) {
                const unsigned rr = j / 32, kk = j % 32, global_row = blockIdx.x * 16 + rr;
                const unsigned logical = k0 + kk;
                const char *wr = w0 + ((uint64_t)(EXPERT ? e : 0) * M + global_row) * rb;
                a[rr][kk] = global_row < M && logical < K ? value<TYPE>(wr, logical) : 0;
                if (EXPERT && !DOWN) {
                    const char *ur = w1 + ((uint64_t)e * M + global_row) * rb;
                    b[rr][kk] = global_row < M && logical < K ? value<TYPE>(ur, logical) : 0;
                }
                const unsigned ti = t0 + rr;
                const int pp = EXPERT && ti < count ? lists[(uint64_t)e * cap + ti] : (int)ti;
                const unsigned tt = EXPERT ? pp / NS : ti, ss = EXPERT ? pp % NS : 0;
                const uint64_t xx = DOWN && EXPERT ? (uint64_t)tt * NO + ss : tt;
                v[rr][kk] = ti < count && logical < K ? x[xx * K + logical] : 0;
            }
            __syncthreads();
            #pragma unroll
            for (unsigned k = 0; k < 32; k++) {
                acc += a[r][k] * v[c][k];
                if (EXPERT && !DOWN) up += b[r][k] * v[c][k];
            }
            __syncthreads();
        }
        if (row < M && item < count) {
            const uint64_t op = EXPERT ? (uint64_t)tok * NO + slot : item;
            out[op * M + row] = EXPERT && !DOWN ? silu(acc) * up : acc;
        }
    }
}

__global__ void expert_tiles(unsigned *prefix, const int *counts, unsigned NE, unsigned nt) {
    pdl_enter();
    unsigned n = 0;
    prefix[0] = 0;
    for (unsigned e = 0; e < NE; e++) {
        n += ((unsigned)counts[e]+nt-1)/nt;
        prefix[e+1] = n;
    }
}

/* Two TF32 components retain the fine part of each FP32 operand. Keep the
 * correction products in their own accumulator: repeatedly adding them to
 * the much larger leading product loses their precision on tensor cores. */
template<unsigned TYPE>
__global__ void matrix_tc(float *out, const float *x, const char *w0,
        unsigned T, unsigned K, unsigned M, uint64_t rb) {
    pdl_enter();
#if __CUDA_ARCH__ >= 800
    namespace wm = nvcuda::wmma;
    const unsigned tid = threadIdx.x, warp = tid/32, r0 = blockIdx.x*64;
    /* Padding separates the banks of adjacent rows in the TF32 fragment
     * loads. The leading planes also hold the completed output tiles. */
    __shared__ __align__(32) float ah[64][36], al[64][36];
    __shared__ __align__(32) float bh[16][36], bl[16][36];
    for (unsigned t0 = blockIdx.z*16; t0 < T; t0 += gridDim.z*16) {
        wm::fragment<wm::accumulator,16,16,8,float> acc, corr, total;
        wm::fill_fragment(acc,0);
        wm::fill_fragment(corr,0);
        wm::fill_fragment(total,0);
        for (unsigned k0 = 0; k0 < K; k0 += 32) {
            for (unsigned j = tid; j < 64*32; j += 128) {
                const unsigned r = j/32, kk = j%32, row = r0+r, col = k0+kk;
                const uint64_t off = (uint64_t)row*rb;
                const float a = row < M && col < K ? value<TYPE>(w0+off,col) : 0;
                ah[r][kk] = wm::__float_to_tf32(a);
                al[r][kk] = wm::__float_to_tf32(a-ah[r][kk]);
            }
            for (unsigned j = tid; j < 16*32; j += 128) {
                const unsigned item = t0+j/32, kk = j%32, col = k0+kk;
                const float b = item < T && col < K ? x[(uint64_t)item*K+col] : 0;
                bh[j/32][kk] = wm::__float_to_tf32(b);
                bl[j/32][kk] = wm::__float_to_tf32(b-bh[j/32][kk]);
            }
            __syncthreads();
            #pragma unroll
            for (unsigned k = 0; k < 32; k += 8) {
                wm::fragment<wm::matrix_a,16,16,8,wm::precision::tf32,wm::row_major> a, alo;
                wm::fragment<wm::matrix_b,16,16,8,wm::precision::tf32,wm::col_major> b, blo;
                wm::load_matrix_sync(a,&ah[warp*16][k],36);
                wm::load_matrix_sync(alo,&al[warp*16][k],36);
                wm::load_matrix_sync(b,&bh[0][k],36);
                wm::load_matrix_sync(blo,&bl[0][k],36);
                wm::mma_sync(corr,alo,blo,corr);
                wm::mma_sync(corr,alo,b,corr);
                wm::mma_sync(corr,a,blo,corr);
                wm::mma_sync(acc,a,b,acc);
            }
            __syncthreads();
            /* Bound tensor-core accumulation depth. CUDA FP32 adds the
             * partial sums, avoiding drift on long HC projection rows. */
            if ((k0+32)%256 == 0 || k0+32 >= K) {
                for (unsigned i = 0; i < acc.num_elements; i++) {
                    total.x[i] += acc.x[i] + corr.x[i];
                }
                wm::fill_fragment(acc,0);
                wm::fill_fragment(corr,0);
            }
        }
        wm::store_matrix_sync(&ah[warp*16][0],total,36,wm::mem_row_major);
        __syncthreads();
        for (unsigned j = tid; j < 64*16; j += 128) {
            const unsigned row = r0+j/16, item = t0+j%16;
            if (row < M && item < T) out[(uint64_t)item*M+row] = ah[j/16][j%16];
        }
        __syncthreads();
    }
#endif
}

__device__ __forceinline__ void mma_tf32(float (&c)[4], const unsigned (&a)[4], const unsigned (&b)[2]) {
#if __CUDA_ARCH__ >= 800
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
#endif
}

template<unsigned TYPE, bool DOWN, unsigned NC>
__global__ void matrix_reg(float *out, const float *x, const char *w0, const char *w1,
        const int *lists, const int *counts, const unsigned *tiles, unsigned NE,
        unsigned NS, unsigned NO, unsigned K, unsigned M, unsigned cap, uint64_t rb) {
    pdl_enter();
#if __CUDA_ARCH__ >= 800
    const unsigned lane = threadIdx.x&31, warp = threadIdx.x/32;
    const unsigned row_tiles = (M+31)/32, job = blockIdx.x/row_tiles;
    if (job >= tiles[NE]) return;
    __shared__ uint64_t grid_table[TYPE == 16 ? 256 : 1];
    __shared__ uint8_t sign_table[TYPE == 16 ? 128 : 1];
    if (TYPE == 16) {
        for (unsigned i = threadIdx.x; i < 256; i += blockDim.x) grid_table[i] = cuda_iq2xxs_grid[i];
        for (unsigned i = threadIdx.x; i < 128; i += blockDim.x) sign_table[i] = cuda_ksigns_iq2xs[i];
        __syncthreads();
    }
    unsigned e = 0, end = NE;
    while (e < end) {
        const unsigned mid = (e+end)/2;
        if (tiles[mid+1] <= job) e = mid+1;
        else end = mid;
    }
    const unsigned count = counts[e], t0 = (job-tiles[e])*16*NC;
    const unsigned r0 = (blockIdx.x%row_tiles)*32+(warp%2)*16, c0 = t0+(warp/2)*8*NC;
    if (r0 >= M || c0 >= count) return;
    float acc[NC][4] = {}, corr[NC][4] = {}, total[NC][4] = {};
    float uacc[NC][4] = {}, ucorr[NC][4] = {}, utotal[NC][4] = {};
    const float *xp[NC];
    #pragma unroll
    for (unsigned nc = 0; nc < NC; nc++) {
        const unsigned item = c0+nc*8+lane/4;
        const unsigned pair = item < count ? lists[(uint64_t)e*cap+item] : 0;
        xp[nc] = x+(DOWN ? (uint64_t)(pair/NS)*NO+pair%NS : pair/NS)*K;
    }
    for (unsigned k0 = 0; k0 < K; k0 += 8) {
        unsigned a[4], al[4], u[4], ul[4], b[2], bl[2];
        #pragma unroll
        for (unsigned i = 0; i < 4; i++) {
            const unsigned r = r0+lane/4+(i%2)*8, k = k0+lane%4+(i/2)*4;
            const uint64_t off = ((uint64_t)e*M+r)*rb;
            const float v = r < M && k < K ? value<TYPE>(w0+off,k,grid_table,sign_table) : 0;
            const float hi = nvcuda::wmma::__float_to_tf32(v);
            a[i] = __float_as_uint(hi);
            al[i] = __float_as_uint(nvcuda::wmma::__float_to_tf32(v-hi));
            if (!DOWN) {
                const float v = r < M && k < K ? value<TYPE>(w1+off,k,grid_table,sign_table) : 0;
                const float hi = nvcuda::wmma::__float_to_tf32(v);
                u[i] = __float_as_uint(hi);
                ul[i] = __float_as_uint(nvcuda::wmma::__float_to_tf32(v-hi));
            }
        }
        #pragma unroll
        for (unsigned nc = 0; nc < NC; nc++) {
            if (c0+nc*8 >= count) continue;
            const unsigned item = c0+nc*8+lane/4;
            #pragma unroll
            for (unsigned i = 0; i < 2; i++) {
                const unsigned k = k0+lane%4+i*4;
                const float v = item < count && k < K ? xp[nc][k] : 0;
                const float hi = nvcuda::wmma::__float_to_tf32(v);
                b[i] = __float_as_uint(hi);
                bl[i] = __float_as_uint(nvcuda::wmma::__float_to_tf32(v-hi));
            }
            mma_tf32(corr[nc],al,bl);
            mma_tf32(corr[nc],al,b);
            mma_tf32(corr[nc],a,bl);
            mma_tf32(acc[nc],a,b);
            if (!DOWN) {
                mma_tf32(ucorr[nc],ul,bl);
                mma_tf32(ucorr[nc],ul,b);
                mma_tf32(ucorr[nc],u,bl);
                mma_tf32(uacc[nc],u,b);
            }
            if ((k0+8)%256 == 0 || k0+8 >= K) {
                #pragma unroll
                for (unsigned i = 0; i < 4; i++) {
                    total[nc][i] += acc[nc][i]+corr[nc][i];
                    acc[nc][i] = corr[nc][i] = 0;
                    if (!DOWN) { utotal[nc][i] += uacc[nc][i]+ucorr[nc][i]; uacc[nc][i] = ucorr[nc][i] = 0; }
                }
            }
        }
    }
    #pragma unroll
    for (unsigned nc = 0; nc < NC; nc++) {
        #pragma unroll
        for (unsigned i = 0; i < 4; i++) {
            const unsigned row = r0+lane/4+(i/2)*8, item = c0+nc*8+(lane%4)*2+(i%2);
            if (row < M && item < count) {
                const unsigned pair = lists[(uint64_t)e*cap+item];
                out[((uint64_t)(pair/NS)*NO+pair%NS)*M+row] = DOWN ? total[nc][i] : silu(total[nc][i])*utotal[nc][i];
            }
        }
    }
#endif
}

/* Match the half operands of the production Metal expert tiles. Products
 * accumulate in FP32; only the matrix operands are rounded to half. */
template<unsigned TYPE, bool DOWN, unsigned NT, unsigned NR = 64>
__global__ void matrix_half_tile(float *out, const float *x, const char *w0, const char *w1,
        const int *lists, const int *counts, const unsigned *tiles, unsigned NE,
        unsigned NS, unsigned NO, unsigned K, unsigned M, unsigned cap, uint64_t rb) {
    pdl_enter();
#if __CUDA_ARCH__ >= 800
    namespace wm = nvcuda::wmma;
    const unsigned tid = threadIdx.x, warp = tid/32, nr = (M+NR-1)/NR, job = blockIdx.x/nr;
    if (job >= tiles[NE]) return;
    unsigned e = 0, end = NE;
    while (e < end) {
        const unsigned mid = (e+end)/2;
        if (tiles[mid+1] <= job) e = mid+1; else end = mid;
    }
    const unsigned count = counts[e], t0 = (job-tiles[e])*NT, r0 = (blockIdx.x%nr)*NR;
    const unsigned wr = warp%(NR/16), wc = warp/(NR/16);
    constexpr unsigned NC = NT/(8/(NR/16));
    union Tile {
        __half ab[(NR+(DOWN ? 0 : NR)+NT)*72];
        float c[NR][NT];
    };
    __shared__ __align__(32) Tile tile;
    __half (*a)[72] = (__half (*)[72])tile.ab;
    __half (*u)[72] = a+NR;
    __half (*b)[72] = a+NR+(DOWN ? 0 : NR);
    __shared__ uint64_t grid_table[TYPE == 16 ? 256 : 1];
    __shared__ uint8_t sign_table[TYPE == 16 ? 128 : 1];
    if (TYPE == 16) {
        for (unsigned i = tid; i < 256; i += 256) grid_table[i] = cuda_iq2xxs_grid[i];
        for (unsigned i = tid; i < 128; i += 256) sign_table[i] = cuda_ksigns_iq2xs[i];
        __syncthreads();
    }
    wm::fragment<wm::accumulator,16,16,16,float> acc[NC/16], up[NC/16];
    #pragma unroll
    for (unsigned j = 0; j < NC/16; j++) {
        wm::fill_fragment(acc[j],0);
        if (!DOWN) wm::fill_fragment(up[j],0);
    }
    for (unsigned k0 = 0; k0 < K; k0 += 64) {
        for (unsigned i = tid*4; i < NR*64; i += 256*4) {
            const unsigned row = r0+i/64, k = k0+i%64;
            const uint64_t off = ((uint64_t)e*M+row)*rb;
            const float4 av = row < M && k < K ? value4<TYPE>(w0+off,k,grid_table,sign_table) : make_float4(0,0,0,0);
            *(__half2 *)&a[i/64][i%64] = __floats2half2_rn(av.x,av.y);
            *(__half2 *)&a[i/64][i%64+2] = __floats2half2_rn(av.z,av.w);
            if (!DOWN) {
                const float4 uv = row < M && k < K ? value4<TYPE>(w1+off,k,grid_table,sign_table) : make_float4(0,0,0,0);
                *(__half2 *)&u[i/64][i%64] = __floats2half2_rn(uv.x,uv.y);
                *(__half2 *)&u[i/64][i%64+2] = __floats2half2_rn(uv.z,uv.w);
            }
        }
        for (unsigned i = tid; i < NT*64; i += 256) {
            const unsigned item = t0+i/64, k = k0+i%64;
            const unsigned pair = item < count ? lists[(uint64_t)e*cap+item] : 0;
            const uint64_t row = DOWN ? (uint64_t)(pair/NS)*NO+pair%NS : pair/NS;
            b[i/64][i%64] = __float2half_rn(item < count && k < K ? x[row*K+k] : 0);
        }
        __syncthreads();
        #pragma unroll
        for (unsigned k = 0; k < 64; k += 16) {
            wm::fragment<wm::matrix_a,16,16,16,__half,wm::row_major> af, uf;
            wm::load_matrix_sync(af,&a[wr*16][k],72);
            if (!DOWN) wm::load_matrix_sync(uf,&u[wr*16][k],72);
            #pragma unroll
            for (unsigned j = 0; j < NC/16; j++) {
                wm::fragment<wm::matrix_b,16,16,16,__half,wm::col_major> bf;
                wm::load_matrix_sync(bf,&b[wc*NC+j*16][k],72);
                wm::mma_sync(acc[j],af,bf,acc[j]);
                if (!DOWN) wm::mma_sync(up[j],uf,bf,up[j]);
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (unsigned j = 0; j < NC/16; j++) {
        if (!DOWN) for (unsigned i = 0; i < acc[j].num_elements; i++)
            acc[j].x[i] = silu(acc[j].x[i])*up[j].x[i];
        wm::store_matrix_sync(&tile.c[wr*16][wc*NC+j*16],acc[j],NT,wm::mem_row_major);
    }
    __syncthreads();
    for (unsigned i = tid; i < NR*NT; i += 256) {
        const unsigned row = r0+i/NT, item = t0+i%NT;
        if (row < M && item < count) {
            const unsigned pair = lists[(uint64_t)e*cap+item];
            out[((uint64_t)(pair/NS)*NO+pair%NS)*M+row] = tile.c[i/NT][i%NT];
        }
    }
#endif
}

static int matrix_dispatch(float *out, const float *x, const char *w0, const char *w1,
        const int *lists, const int *counts, unsigned type, unsigned NE, unsigned T,
        unsigned NS, unsigned NO, unsigned K, unsigned M, unsigned cap, bool down) {
    const dim3 grid((M + 15) / 16, lists ? NE : 1, std::min(8u, (T + 15) / 16));
    const uint64_t rb = expert_row_bytes(type, K);
    if (ds4_cuda_attn_tokentile_arch_ok() && !getenv("DS4_CUDA_NO_TF32")) {
        if (lists && !g_quality_mode && (type == 16 || type == 10 || type == 12 || type == 39)) {
            const unsigned nt = T >= 2048 ? 64 : 32;
            unsigned *tiles = (unsigned *)cuda_tmp_alloc(((uint64_t)NE+1)*4,"Qwen expert tiles");
            if (!tiles) return 0;
            launch(expert_tiles, 1, 1, 0, tiles,counts,NE,nt);
            if (!launched()) return 0;
            /* Wider rows reuse activations; Q2_K down is faster at 64 rows. */
            const unsigned nr = nt == 64 && type != 10 ? 128 : 64;
            const uint64_t blocks = (((uint64_t)T*NS+nt-1)/nt+NE)*((M+nr-1)/nr);
            if (blocks > INT_MAX) return 0;
#define QWEN_HALF(TYPE, DOWN) \
            if (nt == 64) launch(matrix_half_tile<TYPE,DOWN,64,(TYPE == 10 ? 64 : 128)>, blocks, 256, 0, out,x,w0,w1,lists,counts,tiles,NE,NS,NO,K,M,cap,rb); \
            else launch(matrix_half_tile<TYPE,DOWN,32>, blocks, 256, 0, out,x,w0,w1,lists,counts,tiles,NE,NS,NO,K,M,cap,rb)
#define QWEN_HALF_TYPE(TYPE) case TYPE: if (down) { QWEN_HALF(TYPE,true); } else { QWEN_HALF(TYPE,false); } break
            switch (type) { QWEN_HALF_TYPE(16); QWEN_HALF_TYPE(10); QWEN_HALF_TYPE(12); QWEN_HALF_TYPE(39); }
#undef QWEN_HALF_TYPE
#undef QWEN_HALF
            return launched();
        }
        unsigned *tiles = NULL;
        dim3 tcgrid((M+63)/64,1,std::min(8u,(T+15)/16));
        if (lists) {
            tiles = (unsigned *)cuda_tmp_alloc(((uint64_t)NE+1)*4,"Qwen expert tile schedule");
            if (!tiles) return 0;
            launch(expert_tiles, 1, 1, 0, tiles,counts,NE,32);
            if (!launched()) return 0;
            const uint64_t jobs = ((uint64_t)T*NS+31)/32+NE;
            const uint64_t blocks = jobs*((M+31)/32);
            if (blocks > INT_MAX) return 0;
            tcgrid = dim3((unsigned)blocks,1,1);
        }
#define QWEN_TC(TYPE) case TYPE: \
        if (!lists) launch(matrix_tc<TYPE>, tcgrid, 128, 0, out,x,w0,T,K,M,rb); \
        else if (down) launch(matrix_reg<TYPE,true,2>, tcgrid, 128, 0, out,x,w0,w1,lists,counts,tiles,NE,NS,NO,K,M,cap,rb); \
        else launch(matrix_reg<TYPE,false,2>, tcgrid, 128, 0, out,x,w0,w1,lists,counts,tiles,NE,NS,NO,K,M,cap,rb); break
        switch (type) {
        QWEN_TC(0); QWEN_TC(1); QWEN_TC(2); QWEN_TC(8); QWEN_TC(10);
        QWEN_TC(12); QWEN_TC(16); QWEN_TC(30); QWEN_TC(39);
        default: return 0;
        }
#undef QWEN_TC
        return launched();
    }
#define QWEN_MM(TYPE) case TYPE: \
    if (!lists) launch(matrix<TYPE,false,false>, grid, 256, 0, out,x,w0,w1,lists,counts,T,NS,NO,K,M,cap,rb); \
    else if (down) launch(matrix<TYPE,true,true>, grid, 256, 0, out,x,w0,w1,lists,counts,T,NS,NO,K,M,cap,rb); \
    else launch(matrix<TYPE,false,true>, grid, 256, 0, out,x,w0,w1,lists,counts,T,NS,NO,K,M,cap,rb); break
    switch (type) {
    QWEN_MM(0); QWEN_MM(1); QWEN_MM(2); QWEN_MM(8); QWEN_MM(10);
    QWEN_MM(12); QWEN_MM(16); QWEN_MM(30); QWEN_MM(39);
    default: return 0;
    }
#undef QWEN_MM
    return launched();
}

template<unsigned TYPE>
__device__ __forceinline__ float dot(const char *row, const float *x, unsigned n,
        const uint64_t *grid, const uint8_t *signs) {
    float acc = 0;
    if (TYPE == 1 && !(n%4) && !((uintptr_t)x&15) && !((uintptr_t)row&7)) {
        #pragma unroll 4
        for (unsigned i = (threadIdx.x&31)*4; i < n; i += 128) {
            const float4 w = value4<TYPE>(row,i,grid,signs), v = *(const float4 *)(x+i);
            acc += w.x*v.x; acc += w.y*v.y; acc += w.z*v.z; acc += w.w*v.w;
        }
    } else {
        #pragma unroll 8
        for (unsigned i = threadIdx.x & 31; i < n; i += 32) acc += value<TYPE>(row, i, grid, signs) * x[i];
    }
    return sum(acc);
}

template<unsigned TYPE>
__global__ void matvec(float *out, const char *w, const float *x,
                       unsigned K, unsigned M, uint64_t stride) {
    pdl_enter();
    const unsigned row = blockIdx.x * 4 + threadIdx.x / 32, tok = blockIdx.y;
    if (row >= M) return;
    const float v = dot<TYPE>(w + row * stride, x + (uint64_t)tok * K, K);
    if (!(threadIdx.x & 31)) out[(uint64_t)tok * M + row] = v;
}

template<unsigned TYPE, unsigned ROWS>
__global__ void matvec_rows(float *out, const char *w, const float *x,
        unsigned T, unsigned K, unsigned M, uint64_t stride) {
    pdl_enter();
    const unsigned row = blockIdx.x*4+threadIdx.x/32, lane = threadIdx.x&31;
    if (row >= M) return;
    float a[ROWS] = {};
    const char *wr = w+(uint64_t)row*stride;
    if (TYPE == 1 && !(K%4) && !((uintptr_t)x&15) && !((uintptr_t)wr&7)) {
        for (unsigned i = lane*4; i < K; i += 128) {
            const float4 v = value4<TYPE>(wr,i,NULL,NULL);
            #pragma unroll
            for (unsigned t = 0; t < ROWS; t++) if (t < T) {
                const float4 xv = *(const float4 *)(x+(uint64_t)t*K+i);
                a[t] += v.x*xv.x; a[t] += v.y*xv.y; a[t] += v.z*xv.z; a[t] += v.w*xv.w;
            }
        }
    } else {
        for (unsigned i = lane; i < K; i += 32) {
            const float v = value<TYPE>(wr,i);
            #pragma unroll
            for (unsigned t = 0; t < ROWS; t++) if (t < T) a[t] += v*x[(uint64_t)t*K+i];
        }
    }
    #pragma unroll
    for (unsigned t = 0; t < ROWS; t++) if (t < T) {
        const float v = sum(a[t]);
        if (!lane) out[(uint64_t)t*M+row] = v;
    }
}

/* One warp's lane partials of a Q8_0 row: the single-warp Q8 GEMV rows
 * and the GDN projection with its conv epilogue share this loop. */
template<unsigned ROWS>
__device__ __forceinline__ void q8_row_acc(float (&acc)[ROWS], const char *wr, const float *x,
        unsigned T, unsigned K) {
    const unsigned lane = threadIdx.x&31;
    #pragma unroll 4
    for (unsigned i = lane*4; i < K; i += 128) {
        const char *b = wr+(i/32)*34;
        const float scale = __half2float(*(const __half *)b);
        const uint16_t *q = (const uint16_t *)(b+2+i%32);
        const unsigned q01 = q[0], q23 = q[1];
        #pragma unroll
        for (unsigned t = 0; t < ROWS; t++) if (t < T) {
            const float4 v = *(const float4 *)(x+(uint64_t)t*K+i);
            float part = (float)(int8_t)q01*v.x;
            part += (float)(int8_t)(q01>>8)*v.y;
            part += (float)(int8_t)q23*v.z;
            part += (float)(int8_t)(q23>>8)*v.w;
            acc[t] += part*scale;
        }
    }
}

template<unsigned ROWS>
__device__ __forceinline__ void matvec_q8_body(float *out, const char *w, const float *x,
        unsigned T, unsigned K, unsigned M, uint64_t stride, unsigned row) {
    const unsigned lane = threadIdx.x&31;
    if (row >= M) return;
    float acc[ROWS] = {};
    q8_row_acc<ROWS>(acc,w+(uint64_t)row*stride,x,T,K);
    #pragma unroll
    for (unsigned t = 0; t < ROWS; t++) if (t < T) {
        const float v = sum(acc[t]);
        if (!lane) out[(uint64_t)t*M+row] = v;
    }
}

template<unsigned ROWS>
__global__ void matvec_q8(float *out, const char *w, const float *x,
        unsigned T, unsigned K, unsigned M, uint64_t stride) {
    pdl_enter();
    matvec_q8_body<ROWS>(out,w,x,T,K,M,stride,blockIdx.x*4+threadIdx.x/32);
}

/* Decode projections with few output rows (the HC down projections, the
 * router, the linear-attention gates) cannot fill the GPU at one warp per
 * row. Split each row's K across the eight warps of a block instead. The
 * per-row arithmetic is the same for every T, so MTP verify rows keep
 * matching single-token decode. */
/* Lane partials of one warp's split-K part, groups [g0, g1) of 128. */
template<unsigned TYPE, unsigned ROWS>
__device__ __forceinline__ void split_row_acc(float (&acc)[ROWS], const char *wr, const float *x,
        unsigned T, unsigned K, unsigned g0, unsigned g1) {
    const unsigned lane = threadIdx.x & 31;
    #pragma unroll 4
    for (unsigned g = g0; g < g1; g++) {
        const unsigned i = g * 128 + lane * 4;
        if (TYPE == 8) {
            const char *b = wr + (i / 32) * 34;
            const float scale = __half2float(*(const __half *)b);
            const uint16_t *q = (const uint16_t *)(b + 2 + i % 32);
            const unsigned q01 = q[0], q23 = q[1];
            #pragma unroll
            for (unsigned t = 0; t < ROWS; t++) if (t < T) {
                const float4 v = *(const float4 *)(x + (uint64_t)t * K + i);
                float p = (float)(int8_t)q01 * v.x;
                p += (float)(int8_t)(q01 >> 8) * v.y;
                p += (float)(int8_t)q23 * v.z;
                p += (float)(int8_t)(q23 >> 8) * v.w;
                acc[t] += p * scale;
            }
        } else {
            const float4 v = value4<TYPE>(wr, i, NULL, NULL);
            #pragma unroll
            for (unsigned t = 0; t < ROWS; t++) if (t < T) {
                const float4 xv = *(const float4 *)(x + (uint64_t)t * K + i);
                acc[t] += v.x * xv.x; acc[t] += v.y * xv.y; acc[t] += v.z * xv.z; acc[t] += v.w * xv.w;
            }
        }
    }
}

template<unsigned TYPE, unsigned ROWS>
__device__ __forceinline__ void matvec_split_body(float *out, const char *w, const float *x,
        unsigned T, unsigned K, unsigned M, uint64_t stride, unsigned row) {
    const unsigned lane = threadIdx.x & 31, warp = threadIdx.x / 32;
    __shared__ float part[8][ROWS];
    const unsigned groups = K / 128, g0 = groups * warp / 8, g1 = groups * (warp + 1) / 8;
    float acc[ROWS] = {};
    split_row_acc<TYPE,ROWS>(acc, w + (uint64_t)row * stride, x, T, K, g0, g1);
    #pragma unroll
    for (unsigned t = 0; t < ROWS; t++) if (t < T) {
        const float v = sum(acc[t]);
        if (!lane) part[warp][t] = v;
    }
    __syncthreads();
    if (threadIdx.x < ROWS && threadIdx.x < T) {
        float v = 0;
        #pragma unroll
        for (unsigned j = 0; j < 8; j++) v += part[j][threadIdx.x];
        out[(uint64_t)threadIdx.x * M + row] = v;
    }
}

template<unsigned TYPE, unsigned ROWS>
__global__ void matvec_split(float *out, const char *w, const float *x,
        unsigned T, unsigned K, unsigned M, uint64_t stride) {
    pdl_enter();
    matvec_split_body<TYPE,ROWS>(out,w,x,T,K,M,stride,blockIdx.x);
}

/* Pre-wait weight words of the F32 split GEMVs: a float4 of F32, or the
 * uint2 of four BF16 values of the exact BF16 copy (TYPE 30, see
 * bf16_copy), widened exactly as value4<30>. */
template<unsigned TYPE> struct split_pre_word { typedef float4 type; };
template<> struct split_pre_word<30> { typedef uint2 type; };

__device__ __forceinline__ uint2 ldg_pre(const uint2 *p) {
    uint2 v;
    asm volatile("ld.global.nc.v2.u32 {%0, %1}, [%2];" : "=r"(v.x), "=r"(v.y) : "l"(p));
    return v;
}

__device__ __forceinline__ float4 split_pre_value(float4 v) { return v; }

__device__ __forceinline__ float4 split_pre_value(uint2 bits) {
    return make_float4(__uint_as_float(bits.x<<16),__uint_as_float(bits.x&0xffff0000u),
                       __uint_as_float(bits.y<<16),__uint_as_float(bits.y&0xffff0000u));
}

/* The F32 split GEMVs (router, linear-attention alpha/beta: TYPE 0, or
 * TYPE 30 when they read their exact BF16 copies) with each warp's first
 * four weight groups in registers before the dependency wait, which covers
 * K = 2560.  Same groups, order and fused multiply-adds as
 * matvec_split_body. */
template<unsigned TYPE, unsigned ROWS>
__global__ void matvec_split_f32_pre(float *out, const char *w, const float *x,
        unsigned T, unsigned K, unsigned M, uint64_t stride) {
    typedef typename split_pre_word<TYPE>::type word;
    constexpr unsigned PRE = 4;
    const unsigned lane = threadIdx.x & 31, warp = threadIdx.x / 32, row = blockIdx.x;
    const unsigned groups = K / 128, g0 = groups * warp / 8, g1 = groups * (warp + 1) / 8;
    const word *wr = (const word *)(w + (uint64_t)row * stride) + lane;
    word pw[PRE] = {};
    if (g0 + PRE <= g1) {
        #pragma unroll
        for (unsigned p = 0; p < PRE; p++) pw[p] = ldg_pre(wr + (g0 + p) * 32);
    } else {
        #pragma unroll
        for (unsigned p = 0; p < PRE; p++) if (g0 + p < g1) pw[p] = ldg_pre(wr + (g0 + p) * 32);
    }
    pdl_enter();
    __shared__ float part[8][ROWS];
    float acc[ROWS] = {};
    #pragma unroll
    for (unsigned p = 0; p < PRE; p++) if (g0 + p < g1) {
        const unsigned i = (g0 + p) * 128 + lane * 4;
        const float4 v = split_pre_value(pw[p]);
        #pragma unroll
        for (unsigned t = 0; t < ROWS; t++) if (t < T) {
            const float4 xv = *(const float4 *)(x + (uint64_t)t * K + i);
            acc[t] += v.x * xv.x; acc[t] += v.y * xv.y; acc[t] += v.z * xv.z; acc[t] += v.w * xv.w;
        }
    }
    #pragma unroll 4
    for (unsigned g = g0 + PRE; g < g1; g++) {
        const unsigned i = g * 128 + lane * 4;
        const float4 v = split_pre_value(wr[g * 32]);
        #pragma unroll
        for (unsigned t = 0; t < ROWS; t++) if (t < T) {
            const float4 xv = *(const float4 *)(x + (uint64_t)t * K + i);
            acc[t] += v.x * xv.x; acc[t] += v.y * xv.y; acc[t] += v.z * xv.z; acc[t] += v.w * xv.w;
        }
    }
    #pragma unroll
    for (unsigned t = 0; t < ROWS; t++) if (t < T) {
        const float v = sum(acc[t]);
        if (!lane) part[warp][t] = v;
    }
    __syncthreads();
    if (threadIdx.x < ROWS && threadIdx.x < T) {
        float v = 0;
        #pragma unroll
        for (unsigned j = 0; j < 8; j++) v += part[j][threadIdx.x];
        out[(uint64_t)threadIdx.x * M + row] = v;
    }
}

__device__ __forceinline__ float4 half4(uint2 bits) {
    const float2 a = __half22float2(*(__half2 *)&bits.x), b = __half22float2(*(__half2 *)&bits.y);
    return make_float4(a.x, a.y, b.x, b.y);
}

/* The F16 split GEMVs (the HC down projections, 320 rows x 10240) with the
 * block's weight row bulk-copied to shared memory before the dependency
 * wait: its blocks are resident while the HC normalization before it runs
 * and leaves DRAM idle.  K * 2 bytes of dynamic shared memory; same groups,
 * order and fused multiply-adds as matvec_split_body. */
template<unsigned ROWS>
__global__ void matvec_split_f16_bulk(float *out, const char *w, const float *x,
        unsigned T, unsigned K, unsigned M, uint64_t stride) {
    extern __shared__ __align__(16) unsigned char wsm[];
    __shared__ __align__(8) uint64_t bar;
    const unsigned lane = threadIdx.x & 31, warp = threadIdx.x / 32, row = blockIdx.x;
    if (!threadIdx.x) {
        bulk_start(&bar, K * 2);
        bulk_copy(wsm, w + (uint64_t)row * stride, K * 2, &bar);
    }
    pdl_enter();
    __shared__ float part[8][ROWS];
    const unsigned groups = K / 128, g0 = groups * warp / 8, g1 = groups * (warp + 1) / 8;
    float acc[ROWS] = {};
    __syncthreads();
    bulk_wait(&bar);
    #pragma unroll 10
    for (unsigned g = g0; g < g1; g++) {
        const unsigned i = g * 128 + lane * 4;
        const float4 v = half4(*(const uint2 *)(wsm + (uint64_t)i * 2));
        #pragma unroll
        for (unsigned t = 0; t < ROWS; t++) if (t < T) {
            const float4 xv = *(const float4 *)(x + (uint64_t)t * K + i);
            acc[t] += v.x * xv.x; acc[t] += v.y * xv.y; acc[t] += v.z * xv.z; acc[t] += v.w * xv.w;
        }
    }
    #pragma unroll
    for (unsigned t = 0; t < ROWS; t++) if (t < T) {
        const float v = sum(acc[t]);
        if (!lane) part[warp][t] = v;
    }
    __syncthreads();
    if (threadIdx.x < ROWS && threadIdx.x < T) {
        float v = 0;
        #pragma unroll
        for (unsigned j = 0; j < 8; j++) v += part[j][threadIdx.x];
        out[(uint64_t)threadIdx.x * M + row] = v;
    }
}

/* Decode batches of 4..32 rows (several sessions, or sessions with drafts)
 * make the per-row Q8 GEMV issue-bound: each weight feeds up to 32 FP32
 * FMAs and as many activation loads.  Run them on tensor cores instead:
 * int8 weights are exact in FP16, and each activation is split into an
 * FP16 high and low part, so the two products keep about 22 bits of it; the
 * FP32 accumulator takes the Q8 block scale per 32-wide block, as the GEMV
 * does.  Inside a block, lane q of each quad owns elements 8q..8q+7 for
 * both operands (the same permutation of k on both sides), so activations
 * load as one 16-byte vector.  F16 weights (the HC down projections) take
 * the same path with the weight halves loaded as they are.  MTP verify rows
 * (T <= 3) keep the GEMV and with it their match with single-token decode. */
__global__ void split_rows_f16(__half *hi, __half *lo, const float *x, unsigned T, unsigned Tp, unsigned K) {
    pdl_enter();
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (uint64_t)Tp * K) return;
    const float v = i / K < T ? x[i] : 0.f;
    const __half h = __float2half_rn(v);
    hi[i] = h;
    lo[i] = __float2half_rn(v - __half2float(h));
}

__device__ __forceinline__ unsigned i8x2_to_h2(unsigned u16) {
    /* two int8 as (b + 128) in the mantissa of 1024, minus 1152: exact */
    const unsigned u = u16 ^ 0x8080u;
    const unsigned bits = 0x64006400u | (u & 0xffu) | ((u & 0xff00u) << 8);
    __half2 h = *(const __half2 *)&bits;
    h = __hsub2(h, __float2half2_rn(1152.f));
    return *(unsigned *)&h;
}

__device__ __forceinline__ void mma_f16_16816(float (&c)[4], unsigned a0, unsigned a1, unsigned a2, unsigned a3,
                                              unsigned b0, unsigned b1) {
#if __CUDA_ARCH__ >= 800
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#endif
}

struct rows_projection { const char *w; float *out; unsigned M, tiles; };
struct rows_projections { rows_projection p[4]; unsigned n; };

/* Keep the established reduction order and block geometry for one through
 * three tokens. Split-K and ordinary projections have separate launches so
 * wide matrices do not inherit the narrow kernel's register/shared footprint. */
template<unsigned ROWS, bool SPLIT>
__global__ void multi_q8(const __grid_constant__ rows_projections p, const float *x, unsigned T, unsigned K) {
    pdl_enter();
    unsigned b = blockIdx.x, i = 0;
    while (i+1 < p.n && b >= p.p[i].tiles) b -= p.p[i++].tiles;
    const rows_projection w = p.p[i];
    const uint64_t stride = (uint64_t)K/32*34;
    if (SPLIT)
        matvec_split_body<8,ROWS>(w.out,w.w,x,T,K,w.M,stride,b);
    else
        matvec_q8_body<ROWS>(w.out,w.w,x,T,K,w.M,stride,b*4+threadIdx.x/32);
}

/* One block per 16 weight rows; its 16 warps split K; NT tiles of 8 rows.
 * Shared-input projections share one activation pack and one launch: the
 * flattened grid holds only real output tiles of each projection, including
 * unequal pairs.  A grid with gridDim.y > 1 splits K further: slice
 * blockIdx.y writes its partial sums to its own T x M plane of the output,
 * which rows_tc_reduce adds up.  TYPE is 8 (Q8_0) or 1 (F16). */
template<unsigned NT, unsigned TYPE>
__global__ void rows_tc(const __grid_constant__ rows_projections projections, const __half *xh, const __half *xl,
                       unsigned T, unsigned K) {
    pdl_enter();
    enum { SPLIT = 16 };
    __shared__ float red[SPLIT][NT][32][4];
    unsigned tile = blockIdx.x, pi = 0;
    while (pi+1 < projections.n && tile >= projections.p[pi].tiles) tile -= projections.p[pi++].tiles;
    const unsigned M = projections.p[pi].M;
    const char *w = projections.p[pi].w;
    float *out = projections.p[pi].out + (uint64_t)blockIdx.y*T*M;
    const uint64_t stride = TYPE == 8 ? (uint64_t)K/32*34 : (uint64_t)K*2;
    const unsigned lane = threadIdx.x & 31, warp = threadIdx.x / 32, m0 = tile * 16;
    const unsigned g = lane / 4, q8 = 8 * (lane % 4);
    const unsigned r0 = min(m0 + g, M - 1), r1 = min(m0 + g + 8, M - 1);
    const char *w0 = w + (uint64_t)r0 * stride, *w1 = w + (uint64_t)r1 * stride;
    const unsigned nb = K / 32, sp = blockIdx.y*SPLIT+warp, ns = gridDim.y*SPLIT;
    const unsigned b0 = nb * sp / ns, b1 = nb * (sp+1) / ns;
    float acc[NT][4] = {};
    #pragma unroll 4
    for (unsigned b = b0; b < b1; b++) {
        unsigned a00,a01,a02,a03,a10,a11,a12,a13;
        float s0 = 1, s1 = 1;
        if (TYPE == 8) {
            const char *blk0 = w0 + b * 34, *blk1 = w1 + b * 34;
            const uint16_t *p0 = (const uint16_t *)(blk0 + 2 + q8), *p1 = (const uint16_t *)(blk1 + 2 + q8);
            s0 = __half2float(*(const __half *)blk0); s1 = __half2float(*(const __half *)blk1);
            a00=i8x2_to_h2(p0[0]); a01=i8x2_to_h2(p1[0]); a02=i8x2_to_h2(p0[1]); a03=i8x2_to_h2(p1[1]);
            a10=i8x2_to_h2(p0[2]); a11=i8x2_to_h2(p1[2]); a12=i8x2_to_h2(p0[3]); a13=i8x2_to_h2(p1[3]);
        } else {
            const uint4 p0 = *(const uint4 *)(w0+b*64+q8*2), p1 = *(const uint4 *)(w1+b*64+q8*2);
            a00=p0.x; a01=p1.x; a02=p0.y; a03=p1.y;
            a10=p0.z; a11=p1.z; a12=p0.w; a13=p1.w;
        }
        #pragma unroll
        for (unsigned n = 0; n < NT; n++) {
            const uint64_t xo = (uint64_t)(n * 8 + g) * K + b * 32 + q8;
            const uint4 h = *(const uint4 *)(xh + xo), l = *(const uint4 *)(xl + xo);
            float t[4] = {0, 0, 0, 0};
            mma_f16_16816(t, a00, a01, a02, a03, h.x, h.y);
            mma_f16_16816(t, a10, a11, a12, a13, h.z, h.w);
            mma_f16_16816(t, a00, a01, a02, a03, l.x, l.y);
            mma_f16_16816(t, a10, a11, a12, a13, l.z, l.w);
            acc[n][0] += t[0] * s0; acc[n][1] += t[1] * s0;
            acc[n][2] += t[2] * s1; acc[n][3] += t[3] * s1;
        }
    }
    #pragma unroll
    for (unsigned n = 0; n < NT; n++)
        #pragma unroll
        for (unsigned k = 0; k < 4; k++) red[warp][n][lane][k] = acc[n][k];
    __syncthreads();
    for (unsigned idx = threadIdx.x; idx < NT * 128; idx += blockDim.x) {
        const unsigned n = idx / 128, l = (idx / 4) % 32, k = idx % 4;
        float v = 0;
        #pragma unroll
        for (unsigned sp = 0; sp < SPLIT; sp++) v += red[sp][n][l][k];
        const unsigned row = m0 + l / 4 + (k >= 2 ? 8 : 0), tok = n * 8 + 2 * (l % 4) + (k & 1);
        if (row < M && tok < T) out[(uint64_t)tok * M + row] = v;
    }
}

__global__ void rows_tc_reduce(float *out, const float *part, unsigned n, unsigned splits) {
    pdl_enter();
    const unsigned i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i >= n) return;
    float v = 0;
    for (unsigned s = 0; s < splits; s++) v += part[(uint64_t)s*n+i];
    out[i] = v;
}

/* Which 4..32-row projections run on tensor cores.  F16 weights from
 * K = 1024 (the HC down projections; short F16 inputs keep the GEMV).  Q8
 * weights except the output head up to eight rows, which streams faster
 * through the GEMV, and four rows of a long input, where the GEMV keeps
 * up. */
static bool rows_tc_shape(unsigned type, unsigned T, unsigned K, unsigned M) {
    if (T < 4 || T > 32 || K%32 || !ds4_cuda_attn_tokentile_arch_ok()) return false;
    if (type == 1) return T < 32 && K >= 1024;
    return type == 8 && (M < 65536 || T > 8) && (T > 4 || K < 1024);
}

/* A narrow launch (few 16-row tiles) leaves most of the GPU idle, so its
 * K is split over 2 or 4 grid slices whose planes are then added in a
 * fixed order. */
static int rows_tc_dispatch(rows_projections projections, const float *x, unsigned T, unsigned K, unsigned type) {
    const unsigned Tp = (T + 7) / 8 * 8;
    unsigned splits = 1, blocks = 0;
    uint64_t floats = 0;
    for (unsigned i = 0; i < projections.n; i++) { blocks += projections.p[i].tiles; floats += (uint64_t)T*projections.p[i].M; }
    if (K >= 8192 && blocks <= 64) splits = 4;
    else if (type == 8 && K >= 1024 && blocks <= 40) splits = T > 16 ? 4 : 2;
    __half *xh = (__half *)cuda_tmp_alloc((uint64_t)Tp*K*4+(splits > 1 ? floats*splits*4 : 0), "Qwen rows f16 activations");
    if (!xh) return 0;
    __half *xl = xh + (uint64_t)Tp * K;
    rows_projections parts = projections;
    float *scratch = (float *)(xl+(uint64_t)Tp*K);
    if (splits > 1) for (unsigned i = 0; i < projections.n; i++) {
        parts.p[i].out = scratch;
        scratch += (uint64_t)T*projections.p[i].M*splits;
    }
    launch(split_rows_f16, (unsigned)(((uint64_t)Tp * K + 255) / 256), 256, 0, xh, xl, x, T, Tp, K);
#define QWEN_ROWS_TC(N) \
    if (type == 8) launch(rows_tc<N,8>,dim3(blocks,splits),512,0,parts,xh,xl,T,K); \
    else launch(rows_tc<N,1>,dim3(blocks,splits),512,0,parts,xh,xl,T,K)
    switch (Tp / 8) {
    case 1: QWEN_ROWS_TC(1); break;
    case 2: QWEN_ROWS_TC(2); break;
    case 3: QWEN_ROWS_TC(3); break;
    default: QWEN_ROWS_TC(4); break;
    }
#undef QWEN_ROWS_TC
    if (splits > 1) for (unsigned i = 0; i < projections.n; i++)
        launch(rows_tc_reduce,(T*projections.p[i].M+255)/256,256,0,
            projections.p[i].out,parts.p[i].out,T*projections.p[i].M,splits);
    return launched();
}

static int matvec_dispatch(float *out, const char *w, const float *x,
                           unsigned type, unsigned T, unsigned K, unsigned M, bool ref = false) {
    const dim3 grid((M + 3) / 4, T);
    const uint64_t stride = row_bytes(type, K);
    if (!stride) return 0;
    if (T <= 8 && M <= 1536 && K >= 1024 && !(K % 128) && !((uintptr_t)x & 15) &&
        (type == 8 || ((type == 0 || type == 1 || type == 30) && !((uintptr_t)w & 15) && !(stride & 15)))) {
#define QWEN_SPLIT_K(KERNEL, SMEM) \
        if (T == 1) { launch(KERNEL<1>, M, 256, SMEM, out, w, x, T, K, M, stride); } \
        else if (T == 2) { launch(KERNEL<2>, M, 256, SMEM, out, w, x, T, K, M, stride); } \
        else if (T <= 4) { launch(KERNEL<4>, M, 256, SMEM, out, w, x, T, K, M, stride); } \
        else { launch(KERNEL<8>, M, 256, SMEM, out, w, x, T, K, M, stride); }
#define QWEN_SPLIT_PRE(TYPE) \
        if (T == 1) { launch(matvec_split_f32_pre<TYPE, 1>, M, 256, 0, out, w, x, T, K, M, stride); } \
        else if (T == 2) { launch(matvec_split_f32_pre<TYPE, 2>, M, 256, 0, out, w, x, T, K, M, stride); } \
        else if (T <= 4) { launch(matvec_split_f32_pre<TYPE, 4>, M, 256, 0, out, w, x, T, K, M, stride); } \
        else { launch(matvec_split_f32_pre<TYPE, 8>, M, 256, 0, out, w, x, T, K, M, stride); }
#define QWEN_SPLIT(TYPE, N) launch(matvec_split<TYPE, N>, M, 256, 0, out, w, x, T, K, M, stride)
#define QWEN_SPLIT_T(TYPE) \
        if (T == 1) { QWEN_SPLIT(TYPE, 1); } else if (T == 2) { QWEN_SPLIT(TYPE, 2); } \
        else if (T <= 4) { QWEN_SPLIT(TYPE, 4); } else { QWEN_SPLIT(TYPE, 8); }
        const bool pre = !ref && presync_load(T);
        if (type == 0 && pre) { QWEN_SPLIT_PRE(0) }
        else if (type == 0) { QWEN_SPLIT_T(0) }
        else if (type == 30 && pre) { QWEN_SPLIT_PRE(30) }
        else if (type == 30) { QWEN_SPLIT_T(30) }
        else if (type == 1 && pre && bulk_supported() &&
                 (uint64_t)K * 2u + SPLIT_BULK_STATIC_SMEM <= BULK_DYN_SMEM_LIMIT) { QWEN_SPLIT_K(matvec_split_f16_bulk, K * 2) }
        else if (type == 1) { QWEN_SPLIT_T(1) }
        else { QWEN_SPLIT_T(8) }
#undef QWEN_SPLIT_T
#undef QWEN_SPLIT
#undef QWEN_SPLIT_K
#undef QWEN_SPLIT_PRE
        return launched();
    }
    if (type == 8 && T <= 8 && !((uintptr_t)x&15)) {
#define QWEN_Q8_ROWS(N) launch(matvec_q8<N>, (M+3)/4, 128, 0, out,w,x,T,K,M,stride)
        if (T == 1) { QWEN_Q8_ROWS(1); }
        else if (T == 2) { QWEN_Q8_ROWS(2); }
        else if (T <= 4) { QWEN_Q8_ROWS(4); }
        else { QWEN_Q8_ROWS(8); }
#undef QWEN_Q8_ROWS
        return launched();
    }
#define QWEN_MV(TYPE) case TYPE: \
    if (T == 2) launch(matvec_rows<TYPE,2>, (M+3)/4, 128, 0, out,w,x,T,K,M,stride); \
    else if (T > 2 && T <= 4) launch(matvec_rows<TYPE,4>, (M+3)/4, 128, 0, out,w,x,T,K,M,stride); \
    else if (T > 4 && T <= 8) launch(matvec_rows<TYPE,8>, (M+3)/4, 128, 0, out,w,x,T,K,M,stride); \
    else launch(matvec<TYPE>, grid, 128, 0, out,w,x,K,M,stride); break
    switch (type) {
    QWEN_MV(0); QWEN_MV(1); QWEN_MV(2); QWEN_MV(8); QWEN_MV(10);
    QWEN_MV(12); QWEN_MV(16); QWEN_MV(30); QWEN_MV(39);
    default: return 0;
    }
#undef QWEN_MV
    return launched();
}

/* The router and GDN gate projections (ffn_gate_inp, ssm_alpha, ssm_beta)
 * are stored as F32 but hold widened BF16 values: every element's low 16
 * bits are zero. The split GEMV reads them from a BF16 copy made on first
 * use, half the bytes for the same values and the same arithmetic. The
 * conversion checks every element, and a tensor with any nonzero low bits
 * keeps its F32 path. Copies are keyed by the resolved device weight and
 * released with the model's other derived weights; none is made while a
 * decode graph is being captured. */
struct qwen_bf16_copy { uint64_t n; char *dst; };
static std::unordered_map<const char *, qwen_bf16_copy> g_qwen_bf16;
static int *g_qwen_bf16_inexact;

__global__ void f32_to_bf16_exact(unsigned *out, const uint2 *in, uint64_t pairs, int *inexact) {
    for (uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; i < pairs;
         i += (uint64_t)gridDim.x * blockDim.x) {
        const uint2 v = in[i];
        if ((v.x | v.y) & 0xffffu) *inexact = 1;
        out[i] = (v.x >> 16) | (v.y & 0xffff0000u);
    }
}

static void qwen4_bf16_cache_release(void) {
    for (auto &c : g_qwen_bf16) if (c.second.dst) (void)cudaFree(c.second.dst);
    g_qwen_bf16.clear();
}

static const char *bf16_copy(const char *w, uint64_t n) {
    if ((n & 1) || ((uintptr_t)w & 7)) return NULL;
    const auto it = g_qwen_bf16.find(w);
    if (it != g_qwen_bf16.end()) return it->second.n == n ? it->second.dst : NULL;
    if (g_decode_graph_capturing) return NULL;
    const cudaStream_t stream = cuda_decode_stream();
    char *dst = NULL;
    int inexact = 1;
    if ((g_qwen_bf16_inexact || cudaMalloc(&g_qwen_bf16_inexact, sizeof(int)) == cudaSuccess) &&
        cudaMalloc(&dst, n * 2) == cudaSuccess &&
        cudaMemsetAsync(g_qwen_bf16_inexact, 0, sizeof(int), stream) == cudaSuccess) {
        const uint64_t pairs = n / 2;
        f32_to_bf16_exact<<<(unsigned)std::min<uint64_t>((pairs + 255) / 256, 4096), 256, 0, stream>>>(
            (unsigned *)dst, (const uint2 *)w, pairs, g_qwen_bf16_inexact);
        if (cudaGetLastError() != cudaSuccess ||
            cudaMemcpyAsync(&inexact, g_qwen_bf16_inexact, sizeof(int), cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
            cudaStreamSynchronize(stream) != cudaSuccess) inexact = 1;
    }
    (void)cudaGetLastError();
    if (inexact && dst) { (void)cudaFree(dst); dst = NULL; }
    g_qwen_bf16[w] = {n, dst};
    return dst;
}

/* Whether dense_mm_tensor runs this shape through the split GEMV. */
static bool split_shape(unsigned T, unsigned K, unsigned M, const float *x) {
    return T < 32 && M <= 1536 && K >= 1024 && !(K % 128) && !((uintptr_t)x & 15);
}

template<unsigned TYPE>
__global__ void unpack(float *out, const char *w, unsigned K, unsigned M, uint64_t rb) {
    pdl_enter();
    const uint64_t i = (uint64_t)blockIdx.x*blockDim.x + threadIdx.x;
    if (i < (uint64_t)K*M) out[i] = value<TYPE>(w+(i/K)*rb,i%K);
}

/* Power-of-two row scaling protects half range. Preserve the residual too,
 * so dense projections retain substantially more than half precision. */
template<unsigned TYPE>
__global__ void pack_half_components(float *scales, __half *hi, __half *lo,
        const char *x, unsigned K, uint64_t rb) {
    pdl_enter();
    const unsigned row = blockIdx.x, tid = threadIdx.x;
    __shared__ float maxima[256], inv;
    float mx = 0;
    for (unsigned k = tid; k < K; k += 256) mx = fmaxf(mx,fabsf(value<TYPE>(x+(uint64_t)row*rb,k)));
    maxima[tid] = mx;
    __syncthreads();
    for (unsigned d = 128; d; d /= 2) {
        if (tid < d) maxima[tid] = fmaxf(maxima[tid],maxima[tid+d]);
        __syncthreads();
    }
    if (!tid) {
        const int e = maxima[0] > 0 ? max(-120,min(120,(int)((__float_as_uint(maxima[0])>>23)&255)-127)) : 0;
        inv = ldexpf(1,-e); scales[row] = ldexpf(1,e);
    }
    __syncthreads();
    for (unsigned k = tid; k < K; k += 256) {
        const float v = value<TYPE>(x+(uint64_t)row*rb,k)*inv;
        const __half h = __float2half_rn(v);
        hi[(uint64_t)row*K+k] = h;
        lo[(uint64_t)row*K+k] = __float2half_rn((v-__half2float(h))*4096);
    }
}

__global__ void dense_rescale(float *out, const float *scales,
        unsigned M, unsigned N) {
    pdl_enter();
    const uint64_t i = (uint64_t)blockIdx.x*blockDim.x+threadIdx.x;
    if (i < (uint64_t)M*N) out[i] *= scales[i/M];
}

/* Short projections reuse the weights for both activation components.
 * Keep separate FP32 sums and the same scaled residual as the cuBLAS path. */
__global__ void dense_f16_components(float *out, const __half *xh, const __half *xl,
        const __half *w, const float *scales, unsigned T, unsigned K, unsigned M) {
    pdl_enter();
#if __CUDA_ARCH__ >= 800
    namespace wm = nvcuda::wmma;
    const unsigned tid = threadIdx.x, warp = tid/32;
    const unsigned m0 = blockIdx.x*64, t0 = blockIdx.y*64;
    union Tile { __half data[3][64][72]; float result[64][64]; };
    __shared__ __align__(32) Tile tile;
    wm::fragment<wm::accumulator,16,16,16,float> hi[2], lo[2];
    for (unsigned j = 0; j < 2; j++) {
        wm::fill_fragment(hi[j],0);
        wm::fill_fragment(lo[j],0);
    }
    for (unsigned k0 = 0; k0 < K; k0 += 64) {
        for (unsigned i = tid*8; i < 4096; i += 2048) {
            const unsigned r = i/64, k = i%64;
            const bool valid_w = m0+r < M, valid_x = t0+r < T;
            tt_cp_async_16B(&tile.data[0][r][k],w+((uint64_t)(valid_w ? m0+r : 0))*K+k0+k,valid_w);
            tt_cp_async_16B(&tile.data[1][r][k],xh+((uint64_t)(valid_x ? t0+r : 0))*K+k0+k,valid_x);
            tt_cp_async_16B(&tile.data[2][r][k],xl+((uint64_t)(valid_x ? t0+r : 0))*K+k0+k,valid_x);
        }
        tt_cp_async_commit();
        tt_cp_async_wait_group<0>();
        __syncthreads();
        for (unsigned k = 0; k < 64; k += 16) {
            wm::fragment<wm::matrix_a,16,16,16,__half,wm::row_major> a;
            wm::load_matrix_sync(a,&tile.data[0][(warp%4)*16][k],72);
            for (unsigned j = 0; j < 2; j++) {
                wm::fragment<wm::matrix_b,16,16,16,__half,wm::col_major> b, c;
                wm::load_matrix_sync(b,&tile.data[1][(warp/4)*32+j*16][k],72);
                wm::load_matrix_sync(c,&tile.data[2][(warp/4)*32+j*16][k],72);
                wm::mma_sync(hi[j],a,b,hi[j]);
                wm::mma_sync(lo[j],a,c,lo[j]);
            }
        }
        __syncthreads();
    }
    for (unsigned j = 0; j < 2; j++) {
        for (unsigned i = 0; i < hi[j].num_elements; i++) hi[j].x[i] += lo[j].x[i]*0x1p-12f;
        wm::store_matrix_sync(&tile.result[(warp%4)*16][(warp/4)*32+j*16],hi[j],64,wm::mem_row_major);
    }
    __syncthreads();
    for (unsigned i = tid; i < 4096; i += 256) {
        const unsigned m = m0+i%64, t = t0+i/64;
        if (m < M && t < T) out[(uint64_t)t*M+m] = tile.result[i%64][i/64]*scales[t];
    }
#endif
}

static int dense_f16_blas(float *out, const float *x, const __half *w,
        unsigned T, unsigned K, unsigned M) {
    const uint64_t xn = (uint64_t)T*K;
    __half *hi = (__half *)cuda_tmp_alloc((xn+T)*4, "Qwen F16 projection scratch");
    if (!hi) return 0;
    __half *lo = hi+xn;
    float *scales = (float *)(lo+xn);
    launch(pack_half_components<0>, T, 256, 0, scales,hi,lo,(const char *)x,K,(uint64_t)K*4);
    if (!launched()) return 0;
    if (K <= 512 && K%64 == 0 && T >= 32 && ds4_cuda_attn_tokentile_arch_ok()) {
        launch(dense_f16_components, dim3((M+63)/64,(T+63)/64), 256, 0, out,hi,lo,w,scales,T,K,M);
        return launched();
    }
    const float zero = 0, one = 1, low = 0x1p-12f;
    if (!cublas_ok(cublasGemmEx(cuda_cublas_for_tier(0),CUBLAS_OP_T,CUBLAS_OP_N,
                M,T,K,&low,w,CUDA_R_16F,K,lo,CUDA_R_16F,K,&zero,out,CUDA_R_32F,M,
                CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT), "Qwen F16 residual projection") ||
        !cublas_ok(cublasGemmEx(cuda_cublas_for_tier(0),CUBLAS_OP_T,CUBLAS_OP_N,
                M,T,K,&one,w,CUDA_R_16F,K,hi,CUDA_R_16F,K,&one,out,CUDA_R_32F,M,
                CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT), "Qwen F16 leading projection")) return 0;
    launch(dense_rescale, ((uint64_t)M*T+255)/256, 256, 0, out,scales,M,T);
    return launched();
}

__global__ void dense_rescale2(float *out, const float *xs, const float *ws,
        unsigned M, unsigned T, unsigned stride) {
    pdl_enter();
    const uint64_t i = (uint64_t)blockIdx.x*blockDim.x+threadIdx.x;
    if (i < (uint64_t)M*T) {
        const uint64_t dst = (i/M)*stride+i%M;
        out[dst] = out[dst]*xs[i/M]*ws[i%M];
    }
}

static int dense_q8_blas(float *out, const float *x, const char *w,
        unsigned T, unsigned K, unsigned M) {
    const unsigned bound = (unsigned)std::max(UINT64_C(1),(UINT64_C(32)<<20)/((uint64_t)K*4));
    const unsigned tile = std::min(M,bound >= 64 ? bound/64*64 : bound);
    const uint64_t xn = (uint64_t)T*K, wn = (uint64_t)tile*K;
    __half *xh = (__half *)cuda_tmp_alloc((xn+wn+T+tile)*4,"Qwen Q8 projection scratch");
    if (!xh) return 0;
    __half *xl = xh+xn, *wh = xl+xn, *wl = wh+wn;
    float *xs = (float *)(wl+wn), *ws = xs+T;
    launch(pack_half_components<0>, T, 256, 0, xs,xh,xl,(const char *)x,K,(uint64_t)K*4);
    if (!launched()) return 0;
    const float zero = 0, one = 1, low = 0x1p-12f;
    const uint64_t rb = row_bytes(8,K);
    for (unsigned r = 0; r < M; r += tile) {
        const unsigned n = std::min(tile,M-r);
        launch(pack_half_components<8>, n, 256, 0, ws,wh,wl,w+(uint64_t)r*rb,K,rb);
        if (!launched()) return 0;
        if (!cublas_ok(cublasGemmEx(cuda_cublas_for_tier(0),CUBLAS_OP_T,CUBLAS_OP_N,
                n,T,K,&low,wl,CUDA_R_16F,K,xh,CUDA_R_16F,K,&zero,out+r,CUDA_R_32F,M,
                CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT),"Qwen Q8 weight residual") ||
            !cublas_ok(cublasGemmEx(cuda_cublas_for_tier(0),CUBLAS_OP_T,CUBLAS_OP_N,
                n,T,K,&low,wh,CUDA_R_16F,K,xl,CUDA_R_16F,K,&one,out+r,CUDA_R_32F,M,
                CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT),"Qwen Q8 input residual") ||
            !cublas_ok(cublasGemmEx(cuda_cublas_for_tier(0),CUBLAS_OP_T,CUBLAS_OP_N,
                n,T,K,&one,wh,CUDA_R_16F,K,xh,CUDA_R_16F,K,&one,out+r,CUDA_R_32F,M,
                CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT),"Qwen Q8 leading product")) return 0;
        launch(dense_rescale2, ((uint64_t)n*T+255)/256, 256, 0, out+r,xs,ws,n,T,M);
        if (!launched()) return 0;
    }
    return 1;
}

/* Bound the temporary weight expansion instead of retaining an FP32 copy
 * of each projection. cuBLAS uses FP32 math, including FP32 activations. */
static int dense_blas(float *out, const float *x, const char *w,
        unsigned type, unsigned T, unsigned K, unsigned M) {
    const unsigned tile = std::min(M, (unsigned)std::max(UINT64_C(1), (UINT64_C(64)*1024*1024)/((uint64_t)K*4)));
    float *scratch = type ? (float *)cuda_tmp_alloc((uint64_t)tile*K*4, "Qwen dense FP32 tile") : NULL;
    if (type && !scratch) return 0;
    const uint64_t rb = row_bytes(type,K);
    const float alpha = 1, beta = 0;
    for (unsigned r = 0; r < M; r += tile) {
        const unsigned n = std::min(tile,M-r);
        const float *wf = type ? scratch : (const float *)w+(uint64_t)r*K;
        if (type) {
#define QWEN_UNPACK(TYPE) case TYPE: launch(unpack<TYPE>, ((uint64_t)n*K+255)/256, 256, 0, scratch,w+(uint64_t)r*rb,K,n,rb); break
            switch (type) {
            QWEN_UNPACK(1); QWEN_UNPACK(2); QWEN_UNPACK(8); QWEN_UNPACK(10);
            QWEN_UNPACK(12); QWEN_UNPACK(16); QWEN_UNPACK(30); QWEN_UNPACK(39);
            default: return 0;
            }
#undef QWEN_UNPACK
            if (!launched()) return 0;
        }
        if (!cublas_ok(cublasGemmEx(cuda_cublas_for_tier(0),CUBLAS_OP_T,CUBLAS_OP_N,
                n,T,K,&alpha,wf,CUDA_R_32F,K,x,CUDA_R_32F,K,&beta,out+r,CUDA_R_32F,M,
                CUBLAS_COMPUTE_32F_PEDANTIC,CUBLAS_GEMM_DEFAULT), "Qwen FP32 projection")) return 0;
    }
    return 1;
}

__global__ void mtp_stage(float *cat, const float *e, const float *R,
        const float *ge, const float *gh, unsigned E, unsigned hc, float eps) {
    pdl_enter();
    const unsigned row = blockIdx.x, tid = threadIdx.x;
    const bool emb = row == 0;
    const unsigned n = emb ? E : E * hc;
    const float *rs = emb ? e : R;
    __shared__ float red[32];
    float ss = 0;
    for (unsigned i = tid; i < n; i += blockDim.x) ss += rs[i] * rs[i];
    const float inv = rsqrtf(block_sum(ss, red) / n + eps);
    const float *src = emb ? e : R + (uint64_t)(row - 1) * E;
    const float *g = emb ? ge : gh + (uint64_t)(row - 1) * E;
    float *o = cat + (uint64_t)row * 2 * E;
    for (unsigned i = tid; i < E; i += blockDim.x) {
        o[(emb ? 0 : E) + i] = src[i] * inv * g[i];
        o[(emb ? E : 0) + i] = 0;
    }
}

/* mtp_stage for T tokens in one launch, grid (hc + 1, T).  Each thread keeps
 * the element order of mtp_stage's strided loops (same partial sums, same
 * block reduction, same products), so the output is byte-identical; the loads
 * of a chunk are issued before any of them is used, so the reduction pays one
 * memory latency per chunk instead of one per element, and the norm weights
 * (never written on the device) are requested before the dependency wait
 * (ptxas keeps only some of those loads ahead of it).  e and R are written by
 * earlier work on the stream, so they are not __restrict__: a read-only
 * (.nc) load of them could be scheduled above the wait. */
constexpr unsigned MTP_STAGE_CHUNK = 16;

__global__ void mtp_stage_rows(float *__restrict__ cat, const float *e,
        const float *R, const float *__restrict__ ge, const float *__restrict__ gh,
        unsigned E, unsigned hc, float eps) {
    const unsigned row = blockIdx.x, t = blockIdx.y, tid = threadIdx.x, nt = blockDim.x;
    const bool emb = row == 0;
    const float *g = emb ? ge : gh + (uint64_t)(row - 1) * E;
    float gv[MTP_STAGE_CHUNK];
    #pragma unroll
    for (unsigned k = 0; k < MTP_STAGE_CHUNK; k++) {
        const unsigned i = tid + k * nt;
        gv[k] = i < E ? g[i] : 0.0f;
    }
    pdl_enter();
    const unsigned n = emb ? E : E * hc;
    const float *rs = emb ? e + (uint64_t)t * E : R + (uint64_t)t * hc * E;
    __shared__ float red[32];
    float ss = 0;
    for (unsigned base = tid; base < n; base += MTP_STAGE_CHUNK * nt) {
        float v[MTP_STAGE_CHUNK];
        #pragma unroll
        for (unsigned k = 0; k < MTP_STAGE_CHUNK; k++) {
            const unsigned i = base + k * nt;
            v[k] = i < n ? rs[i] : 0.0f;
        }
        /* mtp_stage's loop compiles to one FFMA per element; a lane past
         * the end adds fma(0, 0, ss) == ss (ss is never -0) */
        #pragma unroll
        for (unsigned k = 0; k < MTP_STAGE_CHUNK; k++) ss = fmaf(v[k], v[k], ss);
    }
    const float inv = rsqrtf(block_sum(ss, red) / n + eps);
    const float *src = emb ? rs : rs + (uint64_t)(row - 1) * E;
    float *o = cat + ((uint64_t)t * (hc + 1) + row) * 2 * E;
    for (unsigned base = tid; base < E; base += MTP_STAGE_CHUNK * nt) {
        float sv[MTP_STAGE_CHUNK];
        #pragma unroll
        for (unsigned k = 0; k < MTP_STAGE_CHUNK; k++) {
            const unsigned i = base + k * nt;
            sv[k] = i < E ? src[i] : 0.0f;
        }
        #pragma unroll
        for (unsigned k = 0; k < MTP_STAGE_CHUNK; k++) {
            const unsigned i = base + k * nt;
            if (i < E) {
                const float gk = base == tid ? gv[k] : g[i];
                o[(emb ? 0 : E) + i] = sv[k] * inv * gk;
                o[(emb ? E : 0) + i] = 0;
            }
        }
    }
}

__global__ void mtp_combine(float *out, const float *proj, unsigned E, unsigned hc) {
    pdl_enter();
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < E * hc) out[i] = proj[i % E] + proj[E + i];
}

struct max_pair { float value; int index; };
template<bool FINISH>
__global__ void argmax(int *out, max_pair *scratch, const float *logits, unsigned n, int *host_out) {
    pdl_enter();
    out += blockIdx.y;
    scratch += (uint64_t)blockIdx.y*(FINISH ? n : (n+4095)/4096);
    if (!FINISH) logits += (uint64_t)blockIdx.y*n;
    const unsigned tid = threadIdx.x;
    __shared__ max_pair best[256];
    max_pair v = {-1e30f, 0};
    for (unsigned i = (FINISH ? 0 : blockIdx.x * 4096) + tid;
         i < (FINISH ? n : min(n, (blockIdx.x + 1) * 4096)); i += 256) {
        const max_pair p = FINISH ? scratch[i] : max_pair{logits[i], (int)i};
        if (p.value > v.value || (p.value == v.value && p.index < v.index)) v = p;
    }
    best[tid] = v;
    __syncthreads();
    for (unsigned s = 128; s; s /= 2) {
        if (tid < s) {
            const max_pair p = best[tid + s], q = best[tid];
            if (p.value > q.value || (p.value == q.value && p.index < q.index)) best[tid] = p;
        }
        __syncthreads();
    }
    if (!tid) {
        if (!FINISH) scratch[blockIdx.x] = best[0];
        else {
            *out = best[0].index;
            if (host_out) host_out[blockIdx.y] = best[0].index;
        }
    }
}

/* Copies a span the host wrote into mapped page-locked memory before this
 * launch.  No kernel writes that memory, so it is loaded before the
 * dependency wait (hidden under the predecessor); only the store waits. */
__global__ void stage_host(uint4 *dst, const uint4 *src, unsigned n) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    uint4 a = {};
    if (i < n) a = __ldcv(src + i);
    pdl_enter();
    if (i < n) dst[i] = a;
}

__global__ void vis_patch(float *x, const float *a, const float *b, const float *bias,
        const float *pos, unsigned N, unsigned E) {
    pdl_enter();
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < (uint64_t)N * E) x[i] = a[i] + b[i] + bias[i % E] + pos[i];
}

__global__ void vis_norm(float *out, const float *x, const float *w, const float *b, unsigned E, float eps) {
    pdl_enter();
    const unsigned tid = threadIdx.x;
    const uint64_t base = (uint64_t)blockIdx.x * E;
    __shared__ float red[32];
    float v = 0;
    for (unsigned i = tid; i < E; i += blockDim.x) v += x[base + i];
    const float mean = block_sum(v, red) / E;
    v = 0;
    for (unsigned i = tid; i < E; i += blockDim.x) { const float d = x[base + i] - mean; v += d*d; }
    const float inv = rsqrtf(block_sum(v, red) / E + eps);
    for (unsigned i = tid; i < E; i += blockDim.x) out[base + i] = (x[base + i] - mean) * inv * w[i] + b[i];
}

__global__ void vis_qkv(float *q, float *k, float *v, const float *qkv, const float *bias,
        unsigned H, unsigned D, unsigned grid_w) {
    pdl_enter();
    const unsigned row = blockIdx.x, tid = threadIdx.x, E = H*D, hd = D/2, qd = D/4;
    const unsigned blk = row/4, within = row%4, w2 = grid_w/2;
    const float hp = (blk/w2)*2 + within/2, wp = (blk%w2)*2 + within%2;
    const float *src = qkv + (uint64_t)row*3*E;
    const uint64_t off = (uint64_t)row*E;
    for (unsigned idx = tid; idx < 2*H*hd; idx += blockDim.x) {
        const unsigned part = idx/(H*hd), rem = idx%(H*hd), h = rem/hd, i = rem%hd;
        const unsigned base = part*E + h*D;
        const float x0 = src[base+i] + bias[base+i], x1 = src[base+i+hd] + bias[base+i+hd];
        const unsigned j = i < qd ? i : i-qd;
        const float theta = (i < qd ? hp : wp) * powf(10000.0f, -2.0f*j/hd);
        float s, c; sincosf(theta, &s, &c);
        float *dst = part ? k : q;
        dst[off+h*D+i] = x0*c - x1*s;
        dst[off+h*D+i+hd] = x0*s + x1*c;
    }
    for (unsigned i = tid; i < E; i += blockDim.x) v[off+i] = src[2*E+i] + bias[2*E+i];
}

__global__ void vis_attention(float *out, const float *q, const float *k, const float *v,
        unsigned N, unsigned H, unsigned D) {
    pdl_enter();
    const unsigned row = blockIdx.x, h = blockIdx.y, lane = threadIdx.x, E = H*D;
    const uint64_t qb = (uint64_t)row*E + h*D;
    float query[3] = {}, acc[3] = {};
    for (unsigned j = 0; j < 3; j++) if (lane+j*32 < D) query[j] = q[qb+lane+j*32];
    float mx = -INFINITY, denom = 0;
    for (unsigned p = 0; p < N; p++) {
        const uint64_t kb = (uint64_t)p*E + h*D;
        float score = 0;
        for (unsigned j = 0; j < 3; j++) if (lane+j*32 < D) score += query[j]*k[kb+lane+j*32];
        score = sum(score) * rsqrtf((float)D);
        const float nm = fmaxf(mx, score), old = expf(mx-nm), prob = expf(score-nm);
        denom = denom*old + prob;
        for (unsigned j = 0; j < 3; j++) if (lane+j*32 < D) acc[j] = acc[j]*old + prob*v[kb+lane+j*32];
        mx = nm;
    }
    for (unsigned j = 0; j < 3; j++) if (lane+j*32 < D) out[qb+lane+j*32] = acc[j]/denom;
}

__global__ void vis_add(float *x, const float *add, const float *bias, unsigned N, unsigned E, unsigned mode) {
    pdl_enter();
    const uint64_t i = (uint64_t)blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= (uint64_t)N*E) return;
    if (add) { x[i] += add[i] + bias[i%E]; return; }
    float t = x[i] + bias[i%E];
    if (!mode) t *= .5f * (1 + tanhf(fminf(30, fmaxf(-30, .7978845608f*(t+.044715f*t*t*t)))));
    else if (mode == 1) t *= .5f * (1 + erff(t*.70710678f));
    x[i] = t;
}

/* PRE loads this thread's gamma and injection weights of the first chunk
 * iteration before the dependency wait, so they are not queued behind the
 * weight copies of the HC down projection that follows.  With late, PRE
 * lets that projection launch only after the norm reduction, so its weight
 * copies overlap the second pass instead of both. */
template<unsigned TYPE, bool COMBINE = false, bool PRE = false>
__global__ void hc_norm(float *xn, float *inj, const float *R, const float *gamma,
                        const char *wi, unsigned E, unsigned hc,
                        unsigned ni, float eps, float *next, const float *blk, const float *oldinj,
                        unsigned late) {
    const unsigned stream = blockIdx.x / 8, chunk = blockIdx.x % 8, tok = blockIdx.y;
    const unsigned tid = threadIdx.x, dim = E * hc;
    const unsigned per = (E + 7) / 8, end = min(E, (chunk + 1) * per), first = chunk * per + tid;
    float gp[4] = {}, wp[4][4] = {};
    if (PRE) {
        #pragma unroll
        for (unsigned k = 0; k < 4; k++) {
            const unsigned i = first + k * blockDim.x;
            if (i < end) {
                gp[k] = ldg_pre(gamma + stream * E + i);
                #pragma unroll
                for (unsigned j = 0; j < 4; j++) if (j < ni) {
                    const uint64_t at = (uint64_t)j * dim + stream * E + i;
                    wp[k][j] = TYPE == 1 ? ldg_pre_half(wi + at * 2) :
                               TYPE == 0 ? ldg_pre((const float *)wi + at) : value<TYPE>(wi, at);
                }
            }
        }
        pdl_wait();
        if (!late) pdl_trigger();
    } else pdl_enter();
    const uint64_t base = ((uint64_t)tok * hc + stream) * E;
    __shared__ float red[32];
    __shared__ float add_weight;
    if (COMBINE) {
        if (!tid) add_weight = injection(oldinj + (uint64_t)tok * hc * hc * 8, hc, stream);
        __syncthreads();
    }
    const unsigned nt = blockDim.x;
    /* Load a batch of elements before accumulating them, in the original
     * order and with the same fused multiply-adds as hc_norm_prefill: one
     * outstanding load per iteration left this kernel waiting on L2. */
    float ss = 0;
    for (unsigned i0 = tid; i0 < E; i0 += 8 * nt) {
        float r[8];
        #pragma unroll
        for (unsigned k = 0; k < 8; k++) {
            const unsigned i = i0 + k * nt;
            r[k] = i < E ? R[base + i] : 0;
            if (COMBINE && i < E) r[k] = __fmaf_rn(add_weight, blk[(uint64_t)tok * E + i], r[k]);
        }
        #pragma unroll
        for (unsigned k = 0; k < 8; k++) if (i0 + k * nt < E) ss = __fmaf_rn(r[k], r[k], ss);
    }
    const float inv = rsqrtf(block_sum(ss, red) / E + eps);
    if (PRE && late) pdl_trigger();
    float acc[4] = {};
    for (unsigned i0 = first; i0 < end; i0 += 4 * nt) {
        float r[4], g[4], w[4][4];
        #pragma unroll
        for (unsigned k = 0; k < 4; k++) {
            const unsigned i = i0 + k * nt;
            if (i < end) {
                r[k] = R[base + i];
                if (COMBINE) {
                    r[k] = __fmaf_rn(add_weight, blk[(uint64_t)tok * E + i], r[k]);
                    next[base + i] = r[k];
                }
                if (PRE && i0 == first) {
                    g[k] = gp[k];
                    #pragma unroll
                    for (unsigned j = 0; j < 4; j++) w[k][j] = wp[k][j];
                } else {
                    g[k] = gamma[stream * E + i];
                    #pragma unroll
                    for (unsigned j = 0; j < 4; j++) if (j < ni) w[k][j] = value<TYPE>(wi, j * dim + stream * E + i);
                }
            }
        }
        #pragma unroll
        for (unsigned k = 0; k < 4; k++) {
            const unsigned i = i0 + k * nt;
            if (i >= end) break;
            const float v = r[k] * inv * g[k];
            xn[base + i] = v;
            #pragma unroll
            for (unsigned j = 0; j < 4; j++) if (j < ni) acc[j] = __fmaf_rn(w[k][j], v, acc[j]);
        }
    }
    for (unsigned j = 0; j < ni; j++) {
        const float v = block_sum(acc[j], red);
        if (!tid) inj[((uint64_t)tok * hc * 8 + stream * 8 + chunk) * ni + j] = v;
    }
}

/* Prefill has enough tokens to share one normalization across all eight
 * injection chunks. Each warp preserves the four partial sums of the
 * decode kernel, including their order, without rereading R eight times. */
template<unsigned TYPE>
__global__ void hc_norm_prefill(float *xn, float *inj, const float *R,
        const float *gamma, const char *wi, unsigned E, unsigned hc,
        unsigned ni, float eps) {
    pdl_enter();
    const unsigned stream = blockIdx.x, tok = blockIdx.y;
    const unsigned tid = threadIdx.x, lane = tid & 31, chunk = tid / 32;
    const unsigned dim = E * hc, per = (E + 7) / 8;
    const uint64_t base = ((uint64_t)tok * hc + stream) * E;
    __shared__ float red[32];
    float ss = 0;
    if (tid < 128) for (unsigned i = tid; i < E; i += 128)
        ss += R[base+i] * R[base+i];
    const float inv = rsqrtf(block_sum(ss,red) / E + eps);
    float acc[4][4] = {};
    const unsigned end = min(E,(chunk+1)*per);
    #pragma unroll
    for (unsigned part = 0; part < 4; part++) {
        for (unsigned i = chunk*per+lane+part*32; i < end; i += 128) {
            const float v = R[base+i] * inv * gamma[stream*E+i];
            xn[base+i] = v;
            #pragma unroll
            for (unsigned j = 0; j < 4; j++) if (j < ni)
                acc[j][part] += value<TYPE>(wi,j*dim+stream*E+i) * v;
        }
    }
    #pragma unroll
    for (unsigned j = 0; j < 4; j++) if (j < ni) {
        float v = 0;
        #pragma unroll
        for (unsigned part = 0; part < 4; part++) v += sum(acc[j][part]);
        if (!lane) inj[((uint64_t)tok*hc*8+stream*8+chunk)*ni+j] = v;
    }
}

template<unsigned TYPE>
__global__ void hc_mix(float *out, const float *xn, const float *lo,
                       const char *up, unsigned E, unsigned hc, unsigned rank, uint64_t rb) {
    pdl_enter();
    const unsigned d = blockIdx.x * 4 + threadIdx.x / 32, tok = blockIdx.y;
    __shared__ __align__(16) float activated[512];
    if (rank <= 512) {
        for (unsigned r = threadIdx.x; r < rank; r += blockDim.x)
            activated[r] = silu(lo[(uint64_t)tok*rank+r]/hc);
        __syncthreads();
    }
    if (d >= E) return;
    const unsigned lane = threadIdx.x & 31, stream = lane / 8, l = lane % 8;
    float a = 0;
    if (stream < hc) {
        const char *row = up+((uint64_t)stream*E+d)*rb;
        if (!(rank%4) && rank <= 512 &&
            !(TYPE == 0 ? (uintptr_t)row&15 : TYPE == 1 ? (uintptr_t)row&7 : 0)) {
            for (unsigned r = l*4; r < rank; r += 32) {
                const float4 w = value4<TYPE>(row,r,NULL,NULL), v = *(const float4 *)(activated+r);
                a += w.x*v.x; a += w.y*v.y; a += w.z*v.z; a += w.w*v.w;
            }
        } else {
            for (unsigned r = l; r < rank; r += 8)
                a += value<TYPE>(row,r) *
                    (rank <= 512 ? activated[r] : silu(lo[(uint64_t)tok * rank + r] / hc));
        }
    }
    for (unsigned off = 1; off <= 4; off *= 2) a += __shfl_xor_sync(0xffffffff, a, off);
    float v = stream < hc ? sigmoid(a) * xn[((uint64_t)tok * hc + stream) * E + d] : 0;
    v += __shfl_xor_sync(0xffffffff, v, 8);
    v += __shfl_xor_sync(0xffffffff, v, 16);
    if (!lane) out[(uint64_t)tok * E + d] = v / hc;
}

/* F16 gate/mix with the block's hc streams x 4 rows of up weights
 * bulk-copied to shared memory before the dependency wait, while the HC
 * down projection that produces lo still runs.  Needs E % 4 == 0 and rank
 * % 8 == 0, rank <= 512; same arithmetic as hc_mix's vector path. */
__global__ void hc_mix_f16_bulk(float *out, const float *xn, const float *lo,
                                const char *up, unsigned E, unsigned hc, unsigned rank, uint64_t rb) {
    extern __shared__ __align__(16) unsigned char wsm[];
    __shared__ __align__(8) uint64_t bar;
    const unsigned warp = threadIdx.x / 32, d0 = blockIdx.x * 4, d = d0 + warp, tok = blockIdx.y;
    if (!threadIdx.x) {
        bulk_start(&bar, (unsigned)(hc * 4 * rb));
        for (unsigned s = 0; s < hc; s++)
            bulk_copy(wsm + s * 4 * rb, up + ((uint64_t)s * E + d0) * rb, (unsigned)(4 * rb), &bar);
    }
    pdl_enter();
    __shared__ __align__(16) float activated[512];
    for (unsigned r = threadIdx.x; r < rank; r += blockDim.x)
        activated[r] = silu(lo[(uint64_t)tok*rank+r]/hc);
    __syncthreads();
    bulk_wait(&bar);
    const unsigned lane = threadIdx.x & 31, stream = lane / 8, l = lane % 8;
    float a = 0;
    if (stream < hc) {
        const unsigned char *row = wsm + (stream * 4 + warp) * rb;
        for (unsigned r = l*4; r < rank; r += 32) {
            const float4 w = half4(*(const uint2 *)(row + r * 2)), v = *(const float4 *)(activated+r);
            a += w.x*v.x; a += w.y*v.y; a += w.z*v.z; a += w.w*v.w;
        }
    }
    for (unsigned off = 1; off <= 4; off *= 2) a += __shfl_xor_sync(0xffffffff, a, off);
    float v = stream < hc ? sigmoid(a) * xn[((uint64_t)tok * hc + stream) * E + d] : 0;
    v += __shfl_xor_sync(0xffffffff, v, 8);
    v += __shfl_xor_sync(0xffffffff, v, 16);
    if (!lane) out[(uint64_t)tok * E + d] = v / hc;
}

__device__ float injection(const float *inj, unsigned hc, unsigned stream) {
    float v = 0;
    for (unsigned i = 0; i < hc * 8; i++) v += inj[i * hc + stream];
    return 2 * sigmoid(v / hc);
}

__global__ void hc_combine(float *R, const float *out, const float *inj, unsigned E, unsigned hc) {
    pdl_enter();
    const unsigned d = blockIdx.x * blockDim.x + threadIdx.x, tok = blockIdx.y;
    __shared__ float weights[4];
    if (threadIdx.x < hc) weights[threadIdx.x] = injection(inj + (uint64_t)tok * hc * hc * 8, hc, threadIdx.x);
    __syncthreads();
    if (d >= E) return;
    for (unsigned s = 0; s < hc; s++) R[((uint64_t)tok * hc + s) * E + d] += weights[s] * out[(uint64_t)tok * E + d];
}

__global__ void hc_lo(float *dst, const float *src, uint64_t n, unsigned hc) {
    pdl_enter();
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = silu(src[i] / hc);
}

__global__ void hc_rows(float *dst, const float *up, const float *xn, unsigned E, unsigned hc) {
    pdl_enter();
    const unsigned d = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y;
    if (d >= E) return;
    float v = 0;
    for (unsigned s = 0; s < hc; s++) {
        const uint64_t i = ((uint64_t)t * hc + s) * E + d;
        v += sigmoid(up[i]) * xn[i];
    }
    dst[(uint64_t)t * E + d] = v / hc;
}

} // namespace qwen4_cuda

namespace qwen4_cuda {

struct gdn_row { float *state, *hist, *snap_state, *snap_hist; unsigned row0, n_tok; };
struct attn_row { __half *k, *v; float *ik; __half *block; const uint32_t *pos3; unsigned pos; int sparse; };
static_assert(sizeof(gdn_row) <= 48u, "gdn_row must fit DS4_GPU_QWEN4_GDN_ROW_BYTES (ds4_gpu.h)");
static_assert(sizeof(attn_row) <= 64u, "attn_row must fit DS4_GPU_QWEN4_ATTN_ROW_BYTES (ds4_gpu.h)");
template<typename Row> struct row_batch { Row rows[128]; };
template<typename Row>
__global__ void stage_rows(Row *dst, row_batch<Row> src, unsigned n) {
    pdl_enter();
    if (threadIdx.x < n) dst[threadIdx.x] = src.rows[threadIdx.x];
}

struct row_slots { unsigned slot[128]; };

__global__ void attn_prep_rows(float *q, float *gate, float *iqn, const float *qg,
        const float *kp, const float *vp, const float *iq, const float *ik, const attn_row *rows,
        const float *gq, const float *gk, const float *giq,
        unsigned H, unsigned Hkv, unsigned D, unsigned Hi, unsigned Di, float eps, rope_args rp) {
    pdl_enter();
    const unsigned r = blockIdx.y;
    const attn_row row = rows[r];
    attn_prep_body(q+(uint64_t)r*H*D,gate+(uint64_t)r*H*D,row.k,row.v,iqn+(uint64_t)r*Hi*Di,row.ik,
        qg+(uint64_t)r*H*D*2,kp+(uint64_t)r*Hkv*D,vp+(uint64_t)r*Hkv*D,
        iq+(uint64_t)r*Hi*Di,ik+(uint64_t)r*Di,row.pos3,gq,gk,giq,H,Hkv,D,Hi,Di,row.pos,eps,rp,0);
}

__global__ void block_key_rows(const attn_row *rows, const float *gamma, unsigned ratio,
                              unsigned D, float eps, rope_args rp) {
    pdl_enter();
    const attn_row row = rows[blockIdx.y];
    if ((row.pos+1)%ratio) return;
    block_key_body(row.block,row.ik,row.pos3,gamma,row.pos/ratio,ratio,D,eps,rp);
}

__global__ void idx_score_rows(float *out, const float *q, const attn_row *rows,
                              unsigned stride, unsigned H, unsigned D, unsigned ratio) {
    pdl_enter();
    const unsigned r = blockIdx.y;
    const attn_row row = rows[r];
    if (!row.sparse) return;
    idx_score_body(out+(uint64_t)r*stride,q+(uint64_t)r*H*D,row.block,(row.pos+1)/ratio,H,D,row.pos,ratio,0);
}

__global__ void idx_select_rows(int *out, const float *score, const attn_row *rows,
                               unsigned stride, unsigned K, unsigned ratio) {
    pdl_enter();
    const unsigned r = blockIdx.x;
    const attn_row row = rows[r];
    if (!row.sparse) return;
    idx_select_body(out+(uint64_t)r*K,score+(uint64_t)r*stride,(row.pos+1)/ratio,K,0);
}

__global__ void idx_expand_rows(int *out, unsigned *count, const int *blocks, const attn_row *rows,
                               unsigned K, unsigned ratio, unsigned stride) {
    pdl_enter();
    const unsigned r = blockIdx.x;
    const attn_row row = rows[r];
    if (!row.sparse) return;
    idx_expand_body(out+(uint64_t)r*stride,count+r,blocks+(uint64_t)r*K,K,ratio,row.pos,stride,0);
}

template<unsigned D>
__global__ void attention_rows(float *out, float *part, const float *q, const float *gate,
        const int *sel, const unsigned *counts, const attn_row *rows,
        unsigned H, unsigned Hkv, unsigned stride, float scale) {
    pdl_enter();
    const unsigned r = blockIdx.y;
    const attn_row row = rows[r];
    const unsigned keys = attn_keys(row.sparse,stride,row.pos), splits = attn_splits(keys);
    if (blockIdx.z >= splits) return;
    attention_body<D>(out+(uint64_t)r*H*D,part+(uint64_t)r*H*64*(D+2),
        q+(uint64_t)r*H*D,gate+(uint64_t)r*H*D,row.k,row.v,
        sel ? sel+(uint64_t)r*stride : nullptr,counts ? counts+r : nullptr,
        H,Hkv,row.pos,stride,row.sparse,splits,(keys+splits-1)/splits,scale,0);
}

__global__ void attention_group_rows(float *out, float *part, const float *q, const float *gate,
        const int *sel, const unsigned *counts, const attn_row *rows,
        unsigned H, unsigned Hkv, unsigned stride, float scale) {
    pdl_enter();
    const unsigned r = blockIdx.y, D = 256;
    const attn_row row = rows[r];
    const unsigned keys = attn_keys(row.sparse,stride,row.pos), splits = attn_splits(keys);
    if (blockIdx.z >= splits) return;
    attention_group_body(out+(uint64_t)r*H*D,part+(uint64_t)r*H*64*(D+2),
        q+(uint64_t)r*H*D,gate+(uint64_t)r*H*D,row.k,row.v,
        sel ? sel+(uint64_t)r*stride : nullptr,counts ? counts+r : nullptr,
        H,Hkv,row.pos,stride,row.sparse,scale,splits,(keys+splits-1)/splits,0);
}

__global__ void attn_merge_rows(float *out, const float *part, const float *gate,
        const attn_row *rows, unsigned H, unsigned D, unsigned stride) {
    pdl_enter();
    const unsigned r = blockIdx.y;
    const attn_row row = rows[r];
    const unsigned splits = attn_splits(attn_keys(row.sparse,stride,row.pos));
    if (splits > 1) attn_merge_body(out+(uint64_t)r*H*D,part+(uint64_t)r*H*64*(D+2),
        gate+(uint64_t)r*H*D,H,D,splits,0);
}

__device__ __forceinline__ void conv_body(float *x, float *history, const float *w, unsigned T,
                     unsigned C, unsigned K, bool activate,
                     float *snap, unsigned snap_t, float *snap2, unsigned snap2_t) {
    const unsigned c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    float win[3], taps[4];
    for (unsigned i = 0; i < K - 1; i++) win[i] = history[(uint64_t)i * C + c];
    for (unsigned i = 0; i < K; i++) taps[i] = w[c * K + i];
    for (unsigned t = 0; t < T; t++) {
        const uint64_t pos = (uint64_t)t * C + c;
        const float raw = x[pos];
        float v = taps[K - 1] * raw;
        for (unsigned i = 0; i < K - 1; i++) v += taps[i] * win[i];
        for (unsigned i = 0; i + 2 < K; i++) win[i] = win[i + 1];
        win[K - 2] = raw;
        x[pos] = activate ? silu(v) : v;
        if (snap && t == snap_t) for (unsigned i = 0; i < K - 1; i++) snap[(uint64_t)i * C + c] = win[i];
        if (snap2 && t == snap2_t) for (unsigned i = 0; i < K - 1; i++) snap2[(uint64_t)i * C + c] = win[i];
    }
    for (unsigned i = 0; i < K - 1; i++) history[(uint64_t)i * C + c] = win[i];
}

__global__ void conv(float *x, float *history, const float *w, unsigned T, unsigned C, unsigned K,
        bool activate, float *snap, unsigned snap_t, float *snap2, unsigned snap2_t) {
    pdl_enter();
    conv_body(x,history,w,T,C,K,activate,snap,snap_t,snap2,snap2_t);
}

__global__ void conv_rows(float *x, const float *w, const gdn_row *table,
        unsigned C, unsigned K, bool activate) {
    pdl_enter();
    const gdn_row r = table[blockIdx.y];
    conv_body(x+(uint64_t)r.row0*C,r.hist,w,r.n_tok,C,K,activate,r.n_tok == 2 ? r.snap_hist : nullptr,0,NULL,0);
}

__global__ void conv_slots(float *x, float *history, const float *w, row_slots slots,
        unsigned C, unsigned K, unsigned state_stride, bool activate) {
    pdl_enter();
    const unsigned r = blockIdx.y;
    conv_body(x+(uint64_t)r*C,history+(uint64_t)slots.slot[r]*state_stride,w,1,C,K,activate,NULL,0,NULL,0);
}

/* Shared by gdn_prep and the fused decode projection, so both round alike. */
__device__ __forceinline__ float gdn_decay(float a, float A, float bias) { return expf(A * softplus(a + bias)); }

/* ACT = false: the decode projection already activated alpha/beta. */
template<bool ACT>
__global__ void gdn_prep(float *qkv, float *a, float *b, const float *A, const float *bias,
                         unsigned Hk, unsigned Hv, unsigned D) {
    pdl_enter();
    const unsigned h = blockIdx.x, t = blockIdx.y, lane = threadIdx.x;
    const unsigned C = (2 * Hk + Hv) * D, npt = D / 32;
    float *q = qkv + (uint64_t)t * C + h * D + lane * npt, *k = q + Hk * D;
    float qs = 0, ks = 0;
    for (unsigned i = 0; i < npt; i++) { qs += q[i] * q[i]; ks += k[i] * k[i]; }
    qs = rsqrtf(sum(qs) + 1e-6f) * rsqrtf((float)D);
    ks = rsqrtf(sum(ks) + 1e-6f);
    for (unsigned i = 0; i < npt; i++) { q[i] *= qs; k[i] *= ks; }
    if (ACT && !h) for (unsigned j = lane; j < Hv; j += 32) {
        const uint64_t p = (uint64_t)t * Hv + j;
        a[p] = gdn_decay(a[p], A[j], bias[j]);
        b[p] = sigmoid(b[p]);
    }
}

/* One warp owns a state row across the whole chunk. In particular, MTP
 * snapshots contain the state after the requested token, not the final row. */
template<unsigned ROWS, unsigned D>
__device__ __forceinline__ void gdn_scan_body(float *out, float *state, const float *qkv,
                         const float *a, const float *b, unsigned T, unsigned Hk,
                         unsigned Hv, float *snap, unsigned st,
                         float *snap2, unsigned st2) {
    const unsigned dv = (blockIdx.x * 4 + threadIdx.x / 32)*ROWS, h = blockIdx.y;
    if (dv >= D) return;
    const unsigned npt = D / 32, k0 = (threadIdx.x & 31) * npt, kh = h % Hk;
    const unsigned C = (2 * Hk + Hv) * D;
    const uint64_t idx = ((uint64_t)h * D + dv) * D + k0;
    float s[ROWS][4];
    #pragma unroll
    for (unsigned r = 0; r < ROWS; r++)
        #pragma unroll
        for (unsigned i = 0; i < npt; i++) s[r][i] = state[idx+(uint64_t)r*D+i];
    for (unsigned t = 0; t < T; t++) {
        const float *q = qkv + (uint64_t)t * C + kh * D + k0, *k = q + Hk * D;
        const float decay = a[(uint64_t)t * Hv + h], beta = b[(uint64_t)t * Hv + h];
        #pragma unroll
        for (unsigned r = 0; r < ROWS; r++) {
            const float v = qkv[(uint64_t)t*C+2*Hk*D+h*D+dv+r];
            float u = 0;
            #pragma unroll
            for (unsigned i = 0; i < npt; i++) { s[r][i] *= decay; u += s[r][i]*k[i]; }
            const float delta = (v-sum(u))*beta;
            float o = 0;
            #pragma unroll
            for (unsigned i = 0; i < npt; i++) { s[r][i] += k[i]*delta; o += s[r][i]*q[i]; }
            o = sum(o);
            if (!(threadIdx.x&31)) out[((uint64_t)t*Hv+h)*D+dv+r] = o;
            if (snap && t == st) for (unsigned i = 0; i < npt; i++) snap[idx+(uint64_t)r*D+i] = s[r][i];
            if (snap2 && t == st2) for (unsigned i = 0; i < npt; i++) snap2[idx+(uint64_t)r*D+i] = s[r][i];
        }
    }
    #pragma unroll
    for (unsigned r = 0; r < ROWS; r++)
        #pragma unroll
        for (unsigned i = 0; i < npt; i++) state[idx+(uint64_t)r*D+i] = s[r][i];
}

template<unsigned ROWS, unsigned D>
__global__ void gdn_scan(float *out, float *state, const float *qkv, const float *a, const float *b,
        unsigned T, unsigned Hk, unsigned Hv, float *snap, unsigned st, float *snap2, unsigned st2) {
    pdl_enter();
    gdn_scan_body<ROWS,D>(out,state,qkv,a,b,T,Hk,Hv,snap,st,snap2,st2);
}

template<unsigned D>
__global__ void gdn_scan_rows(float *out, const float *qkv, const float *a, const float *b,
        const gdn_row *table, unsigned Hk, unsigned Hv) {
    pdl_enter();
    const gdn_row r = table[blockIdx.z];
    gdn_scan_body<1,D>(out+(uint64_t)r.row0*Hv*D,r.state,qkv+(uint64_t)r.row0*(2*Hk+Hv)*D,
        a+(uint64_t)r.row0*Hv,b+(uint64_t)r.row0*Hv,r.n_tok,Hk,Hv,r.snap_state,0,NULL,0);
}

template<unsigned D>
__global__ void gdn_scan_slots(float *out, float *state, const float *qkv, const float *a, const float *b,
        row_slots slots, unsigned Hk, unsigned Hv, unsigned state_stride) {
    pdl_enter();
    const unsigned r = blockIdx.z;
    gdn_scan_body<1,D>(out+(uint64_t)r*Hv*D,state+(uint64_t)slots.slot[r]*state_stride,
        qkv+(uint64_t)r*(2*Hk+Hv)*D,a+(uint64_t)r*Hv,b+(uint64_t)r*Hv,1,Hk,Hv,NULL,0,NULL,0);
}

__global__ void gdn_out(float *o, const float *z, const float *w, unsigned H, unsigned D, float eps) {
    pdl_enter();
    const unsigned h = blockIdx.x, t = blockIdx.y, npt = D / 32, k0 = threadIdx.x * npt;
    const uint64_t idx = ((uint64_t)t * H + h) * D + k0;
    float ss = 0;
    for (unsigned i = 0; i < npt; i++) ss += o[idx + i] * o[idx + i];
    const float r = rsqrtf(sum(ss) / D + eps);
    for (unsigned i = 0; i < npt; i++) o[idx + i] = o[idx + i] * r * w[k0 + i] * sigmoid(z[idx + i]);
}

/* The decode (T <= 3: a token, or the MTP verify rows) GDN projections in
 * one launch.  Blocks run, in order:
 * - ab_tiles (0 or 2*Hv) F32 alpha/beta rows, split-K exactly like
 *   matvec_split<0,ROWS> (each of the four warps runs two of its eight
 *   parts, in order), activated as gdn_prep does;
 * - the qkv Q8 rows, one warp per channel; with CONV the warp that computed
 *   channel c also applies its causal conv (K = 4) and SiLU, advances the
 *   history and writes the MTP snapshots, as the conv kernel does (its
 *   compiled chain: fmul by the newest tap, then fma of taps 0..2);
 * - the gate Q8 rows.
 * Every output rounds as with the separate launches, at any T, so verify
 * rows keep matching single-token decode. */
struct gdn_proj_args {
    const char *wqkv, *wz, *walpha, *wbeta;
    float *qkv, *z, *ga, *gb, *hist, *snap, *snap2;
    const float *conv_w, *A, *bias;
    unsigned C, Mz, Hv, ab_tiles, st, st2;
};

template<unsigned ROWS, bool CONV>
__global__ void gdn_proj(const __grid_constant__ gdn_proj_args p, const float *x, unsigned T, unsigned K) {
    pdl_enter();
    const unsigned lane = threadIdx.x & 31, warp = threadIdx.x / 32;
    unsigned b = blockIdx.x;
    if (b < p.ab_tiles) {
        __shared__ float part[8][ROWS];
        const bool beta = b >= p.Hv;
        const unsigned row = beta ? b - p.Hv : b, groups = K / 128;
        const char *wr = (beta ? p.wbeta : p.walpha) + (uint64_t)row * K * 4;
        #pragma unroll 1
        for (unsigned j = warp; j < 8; j += 4) {
            float acc[ROWS] = {};
            split_row_acc<0,ROWS>(acc, wr, x, T, K, groups * j / 8, groups * (j + 1) / 8);
            #pragma unroll
            for (unsigned t = 0; t < ROWS; t++) if (t < T) {
                const float v = sum(acc[t]);
                if (!lane) part[j][t] = v;
            }
        }
        __syncthreads();
        if (threadIdx.x < ROWS && threadIdx.x < T) {
            float v = 0;
            #pragma unroll
            for (unsigned j = 0; j < 8; j++) v += part[j][threadIdx.x];
            const uint64_t o = (uint64_t)threadIdx.x * p.Hv + row;
            if (beta) p.gb[o] = sigmoid(v);
            else p.ga[o] = gdn_decay(v, p.A[row], p.bias[row]);
        }
        return;
    }
    b -= p.ab_tiles;
    const uint64_t stride = (uint64_t)K / 32 * 34;
    const unsigned qkv_tiles = (p.C + 3) / 4;
    if (b >= qkv_tiles) {
        matvec_q8_body<ROWS>(p.z, p.wz, x, T, K, p.Mz, stride, (b - qkv_tiles) * 4 + warp);
        return;
    }
    const unsigned c = b * 4 + warp;
    if (!CONV) {
        matvec_q8_body<ROWS>(p.qkv, p.wqkv, x, T, K, p.C, stride, c);
        return;
    }
    if (c >= p.C) return;
    /* lanes 0..2 fetch the history and lanes 3..6 the taps before the dot */
    float side = 0;
    if (lane < 3) side = p.hist[(uint64_t)lane * p.C + c];
    else if (lane < 7) side = p.conv_w[(uint64_t)c * 4 + lane - 3];
    float acc[ROWS] = {};
    q8_row_acc<ROWS>(acc, p.wqkv + (uint64_t)c * stride, x, T, K);
    float win[3], taps[4];
    #pragma unroll
    for (unsigned i = 0; i < 3; i++) win[i] = __shfl_sync(0xffffffffu, side, i);
    #pragma unroll
    for (unsigned i = 0; i < 4; i++) taps[i] = __shfl_sync(0xffffffffu, side, 3 + i);
    #pragma unroll
    for (unsigned t = 0; t < ROWS; t++) if (t < T) {
        const float raw = sum(acc[t]);
        float v = __fmul_rn(taps[3], raw);
        #pragma unroll
        for (unsigned i = 0; i < 3; i++) v = __fmaf_rn(taps[i], win[i], v);
        win[0] = win[1]; win[1] = win[2]; win[2] = raw;
        if (!lane) {
            p.qkv[(uint64_t)t * p.C + c] = silu(v);
            if (p.snap && t == p.st)
                for (unsigned i = 0; i < 3; i++) p.snap[(uint64_t)i * p.C + c] = win[i];
            if (p.snap2 && t == p.st2)
                for (unsigned i = 0; i < 3; i++) p.snap2[(uint64_t)i * p.C + c] = win[i];
        }
    }
    if (lane < 3) p.hist[(uint64_t)lane * p.C + c] = lane == 0 ? win[0] : lane == 1 ? win[1] : win[2];
}

__global__ void ngram_gate(float *gated, float *normed, const float *R, const float *key,
                           const float *val, const float *gk, const float *gq, const float *gc,
                           unsigned E, unsigned hc, float eps) {
    pdl_enter();
    const unsigned s = blockIdx.x, t = blockIdx.y, tid = threadIdx.x;
    const uint64_t base = ((uint64_t)t * hc + s) * E;
    __shared__ float red[32];
    float sk = 0, sq = 0;
    for (unsigned i = tid; i < E; i += blockDim.x) { sk += key[base + i] * key[base + i]; sq += R[base + i] * R[base + i]; }
    const float ik = rsqrtf(block_sum(sk, red) / E + eps);
    const float iq = rsqrtf(block_sum(sq, red) / E + eps);
    float dot = 0;
    for (unsigned i = tid; i < E; i += blockDim.x)
        dot += (key[base + i] * ik * gk[s * E + i]) * (R[base + i] * iq * gq[s * E + i]);
    const float a = block_sum(dot, red) * rsqrtf((float)E);
    const float mag = sqrtf(fmaxf(fabsf(a), 1e-6f));
    const float gate = sigmoid(a > 0 ? mag : a < 0 ? -mag : 0);
    float ss = 0;
    for (unsigned i = tid; i < E; i += blockDim.x) {
        const float v = gate * val[(uint64_t)t * E + i];
        gated[base + i] = v;
        ss += v * v;
    }
    const float inv = rsqrtf(block_sum(ss, red) / E + eps);
    for (unsigned i = tid; i < E; i += blockDim.x) normed[base + i] = gated[base + i] * inv * gc[s * E + i];
}

__global__ void ngram_conv(float *R, const float *gated, const float *normed, float *history,
                           const char *w, unsigned type, unsigned T, unsigned C, unsigned K,
                           unsigned dilation, float *snap, unsigned st, float *snap2, unsigned st2) {
    pdl_enter();
    const unsigned c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    const unsigned H = (K - 1) * dilation;
    float hist[9], taps[4];
    for (unsigned i = 0; i < H; i++) hist[i] = history[(uint64_t)i * C + c];
    for (unsigned i = 0; i < K; i++) taps[i] = scalar(w, c * K + i, type);
    for (unsigned t = 0; t < T; t++) {
        const uint64_t p = (uint64_t)t * C + c;
        const float cur = normed[p];
        float v = taps[K - 1] * cur;
        for (unsigned i = 0; i < K - 1; i++) v += taps[i] * hist[i * dilation];
        for (unsigned i = 0; i + 1 < H; i++) hist[i] = hist[i + 1];
        hist[H - 1] = cur;
        R[p] += gated[p] + silu(v);
        if (snap && t == st) for (unsigned i = 0; i < H; i++) snap[(uint64_t)i * C + c] = hist[i];
        if (snap2 && t == st2) for (unsigned i = 0; i < H; i++) snap2[(uint64_t)i * C + c] = hist[i];
    }
    for (unsigned i = 0; i < H; i++) history[(uint64_t)i * C + c] = hist[i];
}

} // namespace qwen4_cuda

extern "C" int ds4_gpu_qwen4_conv_stream_tensor(ds4_gpu_tensor *x, ds4_gpu_tensor *history,
        const void *map, uint64_t size, uint64_t offset, uint32_t T, uint32_t C, uint32_t K, bool activate) {
    using namespace qwen4_cuda;
    if (!T || !C || K < 2 || K > 4 || !tensor(x, (uint64_t)T * C * 4) || !tensor(history, (uint64_t)(K - 1) * C * 4)) return 0;
    const char *w = weight(map, size, offset, (uint64_t)C * K * 4);
    if (!w) return 0;
    launch(conv, (C + 255) / 256, 256, 0, (float *)x->ptr, (float *)history->ptr,
        (const float *)w, T, C, K, activate, nullptr, UINT_MAX, nullptr, UINT_MAX);
    return launched();
}

extern "C" int ds4_gpu_qwen4_gdn_prep_tensor(ds4_gpu_tensor *qkv, ds4_gpu_tensor *a, ds4_gpu_tensor *b,
        const void *map, uint64_t size, uint64_t ao, uint64_t bo, uint32_t T, uint32_t Hk, uint32_t Hv, uint32_t D) {
    using namespace qwen4_cuda;
    if (!T || !Hk || !Hv || D < 32 || D > 128 || D % 32 ||
        !tensor(qkv, (uint64_t)T * (2 * Hk + Hv) * D * 4) ||
        !tensor(a, (uint64_t)T * Hv * 4) || !tensor(b, (uint64_t)T * Hv * 4)) return 0;
    const char *A = weight(map, size, ao, (uint64_t)Hv * 4), *bias = weight(map, size, bo, (uint64_t)Hv * 4);
    if (!A || !bias) return 0;
    launch(gdn_prep<true>, dim3(Hk, T), 32, 0, (float *)qkv->ptr,
        (float *)a->ptr, (float *)b->ptr, (const float *)A, (const float *)bias, Hk, Hv, D);
    return launched();
}

extern "C" void ds4_gpu_qwen4_set_verify_rows_exact(bool on) { (void)on; }

extern "C" int ds4_gpu_qwen4_gdn_scan_tensor(ds4_gpu_tensor *out, ds4_gpu_tensor *state,
        const ds4_gpu_tensor *qkv, const ds4_gpu_tensor *a, const ds4_gpu_tensor *b,
        uint32_t T, uint32_t Hk, uint32_t Hv, uint32_t D,
        ds4_gpu_tensor *snap, uint32_t st, ds4_gpu_tensor *snap2, uint32_t st2) {
    using namespace qwen4_cuda;
    const uint64_t bytes = (uint64_t)Hv * D * D * 4;
    if (!T || !Hk || !Hv || Hv % Hk || D < 32 || D > 128 || D % 32 ||
        !tensor(out, (uint64_t)T * Hv * D * 4) || !tensor(state, bytes) ||
        !tensor(qkv, (uint64_t)T * (2 * Hk + Hv) * D * 4) ||
        !tensor(a, (uint64_t)T * Hv * 4) || !tensor(b, (uint64_t)T * Hv * 4) ||
        (snap && !tensor(snap, bytes)) || (snap2 && !tensor(snap2, bytes))) return 0;
#define QWEN_GDN(ROWS, DIM) launch(gdn_scan<ROWS,DIM>, dim3((DIM+4*ROWS-1)/(4*ROWS),Hv), 128, 0, (float *)out->ptr, \
        (float *)state->ptr,(const float *)qkv->ptr,(const float *)a->ptr,(const float *)b->ptr, \
        T,Hk,Hv,snap ? (float *)snap->ptr : nullptr,st,snap2 ? (float *)snap2->ptr : nullptr,st2)
#define QWEN_GDN_DIM(DIM) case DIM: if (T > 8) { QWEN_GDN(4,DIM); } else { QWEN_GDN(1,DIM); } break
    switch (D) { QWEN_GDN_DIM(32); QWEN_GDN_DIM(64); QWEN_GDN_DIM(96); QWEN_GDN_DIM(128); }
#undef QWEN_GDN_DIM
#undef QWEN_GDN
    return launched();
}

extern "C" int ds4_gpu_qwen4_gdn_out_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *z,
        const void *map, uint64_t size, uint64_t off, uint32_t T, uint32_t H, uint32_t D, float eps) {
    using namespace qwen4_cuda;
    const uint64_t n = (uint64_t)T * H * D;
    if (!n || D < 32 || D > 128 || D % 32 || !tensor(out, n * 4) || !tensor(z, n * 4)) return 0;
    const char *w = weight(map, size, off, (uint64_t)D * 4);
    if (!w) return 0;
    launch(gdn_out, dim3(H, T), 32, 0, (float *)out->ptr, (const float *)z->ptr, (const float *)w, H, D, eps);
    return launched();
}

extern "C" int ds4_gpu_qwen4_ple_gate_tensor(ds4_gpu_tensor *gated, ds4_gpu_tensor *normed,
        const ds4_gpu_tensor *R, const ds4_gpu_tensor *key, const ds4_gpu_tensor *val,
        const void *map, uint64_t size, uint64_t ko, uint64_t qo, uint64_t co,
        uint32_t T, uint32_t E, uint32_t hc, float eps) {
    using namespace qwen4_cuda;
    const uint64_t bytes = (uint64_t)T * E * hc * 4, wb = (uint64_t)E * hc * 4;
    if (!T || !E || !hc || hc > 4 || !tensor(gated, bytes) || !tensor(normed, bytes) ||
        !tensor(R, bytes) || !tensor(key, bytes) || !tensor(val, (uint64_t)T * E * 4)) return 0;
    const char *gk = weight(map, size, ko, wb), *gq = weight(map, size, qo, wb), *gc = weight(map, size, co, wb);
    if (!gk || !gq || !gc) return 0;
    launch(ngram_gate, dim3(hc, T), 128, 0, (float *)gated->ptr, (float *)normed->ptr,
        (const float *)R->ptr, (const float *)key->ptr, (const float *)val->ptr,
        (const float *)gk, (const float *)gq, (const float *)gc, E, hc, eps);
    return launched();
}

extern "C" int ds4_gpu_qwen4_ple_conv_tensor(ds4_gpu_tensor *R, const ds4_gpu_tensor *gated,
        const ds4_gpu_tensor *normed, ds4_gpu_tensor *history, const void *map, uint64_t size,
        uint64_t off, uint32_t type, uint32_t T, uint32_t C, uint32_t K, uint32_t dilation,
        ds4_gpu_tensor *snap, uint32_t st, ds4_gpu_tensor *snap2, uint32_t st2) {
    using namespace qwen4_cuda;
    const uint64_t n = (uint64_t)T * C * 4, hb = (uint64_t)(K - 1) * dilation * C * 4;
    if (!T || !C || K < 2 || K > 4 || !dilation || dilation > 3 ||
        (type != 0 && type != 1) || !tensor(R, n) || !tensor(gated, n) || !tensor(normed, n) ||
        !tensor(history, hb) || (snap && !tensor(snap, hb)) || (snap2 && !tensor(snap2, hb))) return 0;
    const char *w = weight(map, size, off, row_bytes(type, (uint64_t)C * K));
    if (!w) return 0;
    launch(ngram_conv, (C + 255) / 256, 256, 0, (float *)R->ptr,
        (const float *)gated->ptr, (const float *)normed->ptr, (float *)history->ptr, w, type, T, C, K,
        dilation, snap ? (float *)snap->ptr : nullptr, st, snap2 ? (float *)snap2->ptr : nullptr, st2);
    return launched();
}

/* ref: the kernels that load their weights after the wait (tests). */
static int qwen4_hc_norm(ds4_gpu_tensor *xn, ds4_gpu_tensor *inj,
        const ds4_gpu_tensor *R, const void *map, uint64_t size, uint64_t go, uint64_t io,
        uint32_t type, uint32_t T, uint32_t E, uint32_t hc, uint32_t ni, float eps, bool ref) {
    using namespace qwen4_cuda;
    const uint64_t n = (uint64_t)T * E * hc;
    if (!T || !E || !hc || hc > 8 || ni > 4 ||
        !tensor(xn, n * 4) || !tensor(R, n * 4) ||
        (ni && !tensor(inj, (uint64_t)T * hc * 8 * ni * 4))) return 0;
    const char *gamma = weight(map, size, go, (uint64_t)E * hc * 4);
    const char *wi = ni ? weight(map, size, io, row_bytes(type, (uint64_t)E * hc) * ni) : gamma;
    if (!gamma || !wi || (type != 0 && type != 1 && type != 8)) return 0;
    if (T > 8) {
#define QWEN_HC_NORM(TYPE) launch(hc_norm_prefill<TYPE>, dim3(hc,T), 256, 0, (float *)xn->ptr, \
        ni ? (float *)inj->ptr : nullptr,(const float *)R->ptr,(const float *)gamma,wi,E,hc,ni,eps)
        if (type == 0) { QWEN_HC_NORM(0); }
        else if (type == 1) { QWEN_HC_NORM(1); }
        else { QWEN_HC_NORM(8); }
#undef QWEN_HC_NORM
        return launched();
    }
#define QWEN_HC_NORM(TYPE, PRE) launch(hc_norm<TYPE, false, PRE>, dim3(hc * 8, T), 128, 0, (float *)xn->ptr, \
        ni ? (float *)inj->ptr : nullptr, (const float *)R->ptr, (const float *)gamma, wi, E, hc, ni, eps, \
        (float *)nullptr, (const float *)nullptr, (const float *)nullptr, hc_norm_late(T))
    const bool pre = !ref && presync_load(T);
    if (type == 0) { if (pre) { QWEN_HC_NORM(0, true); } else { QWEN_HC_NORM(0, false); } }
    else if (type == 1) { if (pre) { QWEN_HC_NORM(1, true); } else { QWEN_HC_NORM(1, false); } }
    else { if (pre) { QWEN_HC_NORM(8, true); } else { QWEN_HC_NORM(8, false); } }
#undef QWEN_HC_NORM
    return launched();
}

extern "C" int ds4_gpu_qwen4_hc_norm_tensor(ds4_gpu_tensor *xn, ds4_gpu_tensor *inj,
        const ds4_gpu_tensor *R, const void *map, uint64_t size, uint64_t go, uint64_t io,
        uint32_t type, uint32_t T, uint32_t E, uint32_t hc, uint32_t ni, float eps) {
    return qwen4_hc_norm(xn, inj, R, map, size, go, io, type, T, E, hc, ni, eps, false);
}

/* Tests only: hc_norm_tensor with the post-wait kernels. */
extern "C" int ds4_gpu_qwen4_hc_norm_ref_tensor(ds4_gpu_tensor *xn, ds4_gpu_tensor *inj,
        const ds4_gpu_tensor *R, const void *map, uint64_t size, uint64_t go, uint64_t io,
        uint32_t type, uint32_t T, uint32_t E, uint32_t hc, uint32_t ni, float eps) {
    return qwen4_hc_norm(xn, inj, R, map, size, go, io, type, T, E, hc, ni, eps, true);
}

static int qwen4_hc_gate_mix(ds4_gpu_tensor *out,
        const ds4_gpu_tensor *xn, const ds4_gpu_tensor *lo, const void *map, uint64_t size,
        uint64_t offset, uint32_t type, uint32_t T, uint32_t E, uint32_t hc, uint32_t rank, bool ref) {
    using namespace qwen4_cuda;
    if (!T || !E || !hc || hc > 4 || !rank || !tensor(out, (uint64_t)T * E * 4) ||
        !tensor(xn, (uint64_t)T * E * hc * 4) || !tensor(lo, (uint64_t)T * rank * 4)) return 0;
    const char *w = weight(map, size, offset, row_bytes(type, rank) * E * hc);
    if (!w || (type != 0 && type != 1 && type != 8)) return 0;
#define QWEN_HC(TYPE) launch(hc_mix<TYPE>, dim3((E + 3) / 4, T), 128, 0, (float *)out->ptr, \
        (const float *)xn->ptr,(const float *)lo->ptr,w,E,hc,rank,row_bytes(TYPE,rank))
    const uint64_t rb = row_bytes(type, rank);
    if (type == 1 && !ref && presync_load(T) && bulk_supported() && !(E % 4) && !(rank % 8) && rank <= 512 &&
        !((uintptr_t)w & 15)) {
        launch(hc_mix_f16_bulk, dim3(E / 4, T), 128, hc * 4 * rb, (float *)out->ptr,
            (const float *)xn->ptr, (const float *)lo->ptr, w, E, hc, rank, rb);
        return launched();
    }
    if (type == 0) { QWEN_HC(0); }
    else if (type == 1) { QWEN_HC(1); }
    else { QWEN_HC(8); }
#undef QWEN_HC
    return launched();
}

extern "C" int ds4_gpu_qwen4_hc_gate_mix_tensor(ds4_gpu_tensor *out,
        const ds4_gpu_tensor *xn, const ds4_gpu_tensor *lo, const void *map, uint64_t size,
        uint64_t offset, uint32_t type, uint32_t T, uint32_t E, uint32_t hc, uint32_t rank) {
    return qwen4_hc_gate_mix(out, xn, lo, map, size, offset, type, T, E, hc, rank, false);
}

/* Tests only: hc_gate_mix_tensor with the post-wait kernels. */
extern "C" int ds4_gpu_qwen4_hc_gate_mix_ref_tensor(ds4_gpu_tensor *out,
        const ds4_gpu_tensor *xn, const ds4_gpu_tensor *lo, const void *map, uint64_t size,
        uint64_t offset, uint32_t type, uint32_t T, uint32_t E, uint32_t hc, uint32_t rank) {
    return qwen4_hc_gate_mix(out, xn, lo, map, size, offset, type, T, E, hc, rank, true);
}

extern "C" int ds4_gpu_qwen4_hc_combine_tensor(ds4_gpu_tensor *R,
        const ds4_gpu_tensor *out, const ds4_gpu_tensor *inj, uint32_t T, uint32_t E, uint32_t hc) {
    using namespace qwen4_cuda;
    if (!T || !E || !hc || hc > 4 || !tensor(R, (uint64_t)T * E * hc * 4) ||
        !tensor(out, (uint64_t)T * E * 4) || !tensor(inj, (uint64_t)T * hc * hc * 8 * 4)) return 0;
    launch(hc_combine, dim3((E + 255) / 256, T), 256, 0,
        (float *)R->ptr, (const float *)out->ptr, (const float *)inj->ptr, E, hc);
    return launched();
}

extern "C" int ds4_gpu_qwen4_hc_lo_act_tensor(ds4_gpu_tensor *dst,
        const ds4_gpu_tensor *src, uint32_t T, uint32_t hc, uint32_t rank) {
    using namespace qwen4_cuda;
    const uint64_t n = (uint64_t)T * rank;
    if (!n || !hc || !tensor(dst, n * 4) || !tensor(src, n * 4)) return 0;
    launch(hc_lo, (n + 255) / 256, 256, 0, (float *)dst->ptr, (const float *)src->ptr, n, hc);
    return launched();
}

extern "C" int ds4_gpu_qwen4_hc_mix_rows_tensor(ds4_gpu_tensor *dst,
        const ds4_gpu_tensor *up, const ds4_gpu_tensor *xn, uint32_t T, uint32_t E, uint32_t hc) {
    using namespace qwen4_cuda;
    if (!T || !E || !hc || hc > 4 || !tensor(dst, (uint64_t)T * E * 4) ||
        !tensor(up, (uint64_t)T * hc * E * 4) || !tensor(xn, (uint64_t)T * hc * E * 4)) return 0;
    launch(hc_rows, dim3((E + 255) / 256, T), 256, 0,
        (float *)dst->ptr, (const float *)up->ptr, (const float *)xn->ptr, E, hc);
    return launched();
}

extern "C" void ds4_gpu_qwen4_set_rope(const float *freq, uint32_t n, float scale) {
    qwen4_cuda::rope_set = freq != NULL;
    qwen4_cuda::rope_scale = freq ? scale : 1;
    memset(qwen4_cuda::rope_freq, 0, sizeof(qwen4_cuda::rope_freq));
    if (freq) memcpy(qwen4_cuda::rope_freq, freq, std::min(n, 32u) * sizeof(float));
}

extern "C" int ds4_gpu_qwen4_attn_prep_tensor(ds4_gpu_tensor *q, ds4_gpu_tensor *gate,
        ds4_gpu_tensor *kc, ds4_gpu_tensor *vc, ds4_gpu_tensor *iqout, ds4_gpu_tensor *ikc,
        const ds4_gpu_tensor *qg, const ds4_gpu_tensor *kp, const ds4_gpu_tensor *vp,
        const ds4_gpu_tensor *iq, const ds4_gpu_tensor *ik, const ds4_gpu_tensor *pos3,
        const void *map, uint64_t size, uint64_t qo, uint64_t ko, uint64_t io,
        uint32_t T, uint32_t H, uint32_t Hkv, uint32_t D, uint32_t nrot,
        uint32_t Hi, uint32_t Di, uint32_t pos0, uint32_t cap, float base, float eps) {
    using namespace qwen4_cuda;
    const uint64_t qb = (uint64_t)T * H * D * 4, kb = (uint64_t)T * Hkv * D * 4, ib = (uint64_t)T * Hi * Di * 4;
    if (!T || !H || !Hkv || H % Hkv || !Hi || D < 32 || D > 256 || D % 32 ||
        Di < 32 || Di > 128 || Di % 32 || nrot > 64 || nrot > D || nrot > Di || nrot % 2 ||
        (uint64_t)pos0 + T > cap || !tensor(q, qb) || !tensor(gate, qb) || !tensor(qg, qb * 2) ||
        !tensor(kp, kb) || !tensor(vp, kb) || !tensor(iq, ib) || !tensor(iqout, ib) ||
        !tensor(ik, (uint64_t)T * Di * 4) || !tensor(ikc, (uint64_t)cap * Di * 4) ||
        !tensor(kc, (uint64_t)cap * Hkv * D * 2) || !tensor(vc, (uint64_t)cap * Hkv * D * 2) ||
        !tensor(pos3, (uint64_t)cap * 16)) return 0;
    const char *gq = weight(map, size, qo, D * 4), *gk = weight(map, size, ko, D * 4), *giq = weight(map, size, io, Di * 4);
    if (!gq || !gk || !giq) return 0;
    launch(attn_prep, dim3(H + Hkv + Hi + 1, T), 32, 0, (float *)q->ptr,
        (float *)gate->ptr, (__half *)kc->ptr, (__half *)vc->ptr, (float *)iqout->ptr, (float *)ikc->ptr,
        (const float *)qg->ptr, (const float *)kp->ptr, (const float *)vp->ptr, (const float *)iq->ptr,
        (const float *)ik->ptr, (const uint32_t *)pos3->ptr, (const float *)gq, (const float *)gk, (const float *)giq,
        H, Hkv, D, Hi, Di, pos0, eps, rope(nrot, base));
    return launched();
}

extern "C" int ds4_gpu_qwen4_idx_block_key_tensor(ds4_gpu_tensor *out,
        const ds4_gpu_tensor *ik, const ds4_gpu_tensor *pos3, const void *map, uint64_t size,
        uint64_t off, uint32_t b0, uint32_t N, uint32_t ratio, uint32_t D, uint32_t nrot,
        float base, float eps) {
    using namespace qwen4_cuda;
    const uint64_t rows = (uint64_t)b0 + N;
    if (!N || !ratio || D < 32 || D > 128 || D % 32 || nrot > D || nrot > 64 || nrot % 2 ||
        !tensor(out, rows * D * 2) || !tensor(ik, rows * ratio * D * 4) || !tensor(pos3, rows * ratio * 16)) return 0;
    const char *w = weight(map, size, off, D * 4);
    if (!w) return 0;
    launch(block_key, N, 32, 0, (__half *)out->ptr, (const float *)ik->ptr,
        (const uint32_t *)pos3->ptr, (const float *)w, b0, ratio, D, eps, rope(nrot, base));
    return launched();
}

extern "C" int ds4_gpu_qwen4_idx_score_tensor(ds4_gpu_tensor *out, ds4_gpu_tensor *tiles,
        const ds4_gpu_tensor *q, const ds4_gpu_tensor *key, uint32_t T, uint32_t N,
        uint32_t H, uint32_t D, uint32_t pos0, uint32_t ratio) {
    using namespace qwen4_cuda;
    const unsigned nt = (N + 7) / 8;
    if (!T || !N || !H || !D || !ratio || !tensor(out, (uint64_t)T * N * 4) ||
        !tensor(q, (uint64_t)T * H * D * 4) || !tensor(key, (uint64_t)N * D * 2) ||
        (tiles && !tensor(tiles, (uint64_t)T * nt * 4))) return 0;
    launch(idx_score, dim3((N + 3) / 4, T), 128, 0, (float *)out->ptr,
        (const float *)q->ptr, (const __half *)key->ptr, N, H, D, pos0, ratio);
    if (!launched()) return 0;
    if (tiles) launch(tile_max, dim3((nt + 255) / 256, T), 256, 0,
        (unsigned *)tiles->ptr, (const float *)out->ptr, N, nt);
    return launched();
}

extern "C" int ds4_gpu_qwen4_idx_select_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *score,
        const ds4_gpu_tensor *tiles, uint32_t N, uint32_t T, uint32_t K) {
    using namespace qwen4_cuda;
    (void)tiles;
    if (!T || !K || K > N || !tensor(out, (uint64_t)T * K * 4) || !tensor(score, (uint64_t)T * N * 4)) return 0;
    launch(idx_select, T, 256, 0, (int *)out->ptr, (const float *)score->ptr, N, K);
    return launched();
}

extern "C" int ds4_gpu_qwen4_idx_expand_tensor(ds4_gpu_tensor *out, ds4_gpu_tensor *count,
        const ds4_gpu_tensor *blocks, uint32_t T, uint32_t K, uint32_t ratio, uint32_t pos0, uint32_t stride) {
    using namespace qwen4_cuda;
    if (!T || !K || !ratio || (uint64_t)K * ratio + ratio - 1 > stride ||
        !tensor(out, (uint64_t)T * stride * 4) || !tensor(count, (uint64_t)T * 4) || !tensor(blocks, (uint64_t)T * K * 4)) return 0;
    launch(idx_expand, T, 256, 0, (int *)out->ptr, (unsigned *)count->ptr,
        (const int *)blocks->ptr, K, ratio, pos0, stride);
    return launched();
}

extern "C" uint64_t ds4_gpu_qwen4_attn_part_floats(uint32_t T, uint32_t H, uint32_t D) {
    return (uint64_t)T * H * 64 * (D + 2);
}

/* ref: the weight as stored (no BF16 copy) and the kernels that load it
 * after the dependency wait, for the tests' reference. */
static int qwen4_dense_mm(ds4_gpu_tensor *out, const ds4_gpu_tensor *x,
        const void *map, uint64_t size, uint64_t off, uint32_t type, uint32_t T, uint32_t K, uint32_t M, bool ref) {
    using namespace qwen4_cuda;
    if (!T || !K || !M || !tensor(x, (uint64_t)T*K*4) || !tensor(out, (uint64_t)T*M*4)) return 0;
    const uint64_t rb = row_bytes(type,K);
    if (!rb) return 0;
    const char *w = weight(map, size, off, rb*M);
    if (!w) return 0;
    if (!ref && type == 0 && split_shape(T,K,M,(const float *)x->ptr))
        if (const char *b = bf16_copy(w,(uint64_t)K*M)) { w = b; type = 30; }
    if (rows_tc_shape(type,T,K,M) && !((uintptr_t)x->ptr & 15) && (type == 8 || !((uintptr_t)w & 15))) {
        rows_projections p = {};
        p.n = 1; p.p[0] = {w,(float *)out->ptr,M,(M+15)/16};
        return rows_tc_dispatch(p,(const float *)x->ptr,T,K,type);
    }
    if (T <= 8) return matvec_dispatch((float *)out->ptr, w, (const float *)x->ptr, type, T, K, M, ref);
    /* Decode batches of 9..31 rows (sessions, or sessions with drafts): the
     * row-batched GEMV in chunks of eight still reads the weights T / 8
     * times, where the tiled GEMM below is sized for prefill and leaves
     * most of the GPU idle at this width. */
    if (T < 32) {
        for (uint32_t t0 = 0; t0 < T; t0 += 8) {
            if (!matvec_dispatch((float *)out->ptr + (uint64_t)t0 * M, w, (const float *)x->ptr + (uint64_t)t0 * K,
                                 type, T - t0 < 8 ? T - t0 : 8, K, M, ref)) return 0;
        }
        return 1;
    }
    if (type == 1 && T >= 32 && T <= INT_MAX && K <= INT_MAX && M <= INT_MAX &&
        g_cublas_ready && !g_quality_mode && !getenv("DS4_CUDA_NO_TF32"))
        return dense_f16_blas((float *)out->ptr,(const float *)x->ptr,(const __half *)w,T,K,M);
    if (type == 8 && T >= 32 && T <= INT_MAX && K <= INT_MAX && M <= INT_MAX &&
        g_cublas_ready && !g_quality_mode && !getenv("DS4_CUDA_NO_TF32"))
        return dense_q8_blas((float *)out->ptr,(const float *)x->ptr,w,T,K,M);
    if (g_cublas_ready && T >= 32 && K <= INT_MAX && M <= INT_MAX && T <= INT_MAX)
        return dense_blas((float *)out->ptr,(const float *)x->ptr,w,type,T,K,M);
    return matrix_dispatch((float *)out->ptr, (const float *)x->ptr, w, NULL, NULL, NULL,
                           type, 1, T, 1, 1, K, M, 0, false);
}

extern "C" int ds4_gpu_qwen4_dense_mm_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *x,
        const void *map, uint64_t size, uint64_t off, uint32_t type, uint32_t T, uint32_t K, uint32_t M) {
    return qwen4_dense_mm(out, x, map, size, off, type, T, K, M, false);
}

/* Tests only: dense_mm_tensor on the weight as stored with the post-wait
 * kernels, the byte-exact reference for the BF16 gate copies and the
 * pre-wait weight loads. */
extern "C" int ds4_gpu_qwen4_dense_mm_ref_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *x,
        const void *map, uint64_t size, uint64_t off, uint32_t type, uint32_t T, uint32_t K, uint32_t M) {
    return qwen4_dense_mm(out, x, map, size, off, type, T, K, M, true);
}

extern "C" int ds4_gpu_qwen4_matmul_q8_0_tensor(ds4_gpu_tensor *out, const void *map,
        uint64_t size, uint64_t off, uint64_t K, uint64_t M, const ds4_gpu_tensor *x, uint64_t T) {
    if (K > UINT_MAX || M > UINT_MAX || T > UINT_MAX) return 0;
    return ds4_gpu_qwen4_dense_mm_tensor(out, x, map, size, off, 8, T, K, M);
}

extern "C" int ds4_gpu_qwen4_matmul_q8_0_weights_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *w,
        uint32_t K, uint32_t M, const ds4_gpu_tensor *x) {
    using namespace qwen4_cuda;
    const uint64_t rb = row_bytes(8, K);
    if (!K || !M || !rb || !tensor(w, rb*M) || !tensor(x, (uint64_t)K*4) || !tensor(out, (uint64_t)M*4)) return 0;
    return matvec_dispatch((float *)out->ptr, (const char *)w->ptr, (const float *)x->ptr, 8, 1, K, M);
}

extern "C" int ds4_gpu_qwen4_multi_gemv_tensor(const ds4_gpu_tensor *x, uint32_t T,
        uint32_t K, uint32_t N, ds4_gpu_tensor *const *outs, const void *map, uint64_t size,
        const uint64_t *offsets, const uint32_t *types, const uint32_t *rows) {
    using namespace qwen4_cuda;
    if (!N || N > 4 || !outs || !offsets || !types || !rows) return 0;
    rows_projections p = {};
    if (T && T <= 3 && K && !(K%128) && tensor(x,(uint64_t)T*K*4) && !((uintptr_t)x->ptr&15)) {
        bool same_q8 = true;
        unsigned blocks[2] = {};
        rows_projections groups[2] = {};
        for (unsigned i = 0; i < N; i++) {
            if (types[i] != 8) { same_q8 = false; break; }
            const char *w = weight(map,size,offsets[i],row_bytes(8,K)*rows[i]);
            if (!w || !rows[i] || !tensor(outs[i],(uint64_t)T*rows[i]*4)) return 0;
            const unsigned group = rows[i] <= 1536 && K >= 1024;
            const unsigned tiles = group ? rows[i] : (rows[i]+3)/4;
            groups[group].p[groups[group].n++] = {w,(float *)outs[i]->ptr,rows[i],tiles};
            blocks[group] += tiles;
        }
        if (same_q8) {
#define QWEN_MULTI(NT) \
            if (groups[0].n) launch(multi_q8<NT,false>,blocks[0],128,0,groups[0],(const float *)x->ptr,T,K); \
            if (groups[1].n) launch(multi_q8<NT,true>,blocks[1],256,0,groups[1],(const float *)x->ptr,T,K)
            if (T == 1) { QWEN_MULTI(1); }
            else if (T == 2) { QWEN_MULTI(2); }
            else { QWEN_MULTI(4); }
#undef QWEN_MULTI
            return launched();
        }
    }
    bool fuse = tensor(x,(uint64_t)T*K*4) && !((uintptr_t)x->ptr&15);
    for (unsigned i = 0; fuse && i < N; i++) {
        fuse = types[i] == types[0] && rows_tc_shape(types[i],T,K,rows[i]);
        if (!fuse) break;
        const char *w = weight(map,size,offsets[i],row_bytes(types[i],K)*rows[i]);
        if (!w || !tensor(outs[i],(uint64_t)T*rows[i]*4)) return 0;
        fuse = types[i] == 8 || !((uintptr_t)w&15);
        p.p[i] = {w,(float *)outs[i]->ptr,rows[i],(rows[i]+15)/16};
    }
    if (fuse) { p.n = N; return rows_tc_dispatch(p,(const float *)x->ptr,T,K,types[0]); }
    for (unsigned i = 0; i < N; i++)
        if (!ds4_gpu_qwen4_dense_mm_tensor(outs[i], x, map, size, offsets[i], types[i], T, K, rows[i])) return 0;
    return 1;
}

extern "C" int ds4_gpu_qwen4_q8_pair_tensor(ds4_gpu_tensor *o0, ds4_gpu_tensor *o1,
        const void *map, uint64_t size, uint64_t w0, uint64_t w1, uint64_t K,
        uint64_t M0, uint64_t M1, const ds4_gpu_tensor *x, uint64_t T) {
    if ((M0 & 1) || (M1 & 1) || K > UINT_MAX || M0 > UINT_MAX || M1 > UINT_MAX || T > UINT_MAX) return 0;
    ds4_gpu_tensor *outs[] = {o0,o1};
    const uint64_t offsets[] = {w0,w1};
    const uint32_t types[] = {8,8}, rows[] = {(uint32_t)M0,(uint32_t)M1};
    return ds4_gpu_qwen4_multi_gemv_tensor(x,T,K,2,outs,map,size,offsets,types,rows);
}

/* pre: an F32 shared gate of K = 2560 takes router_pre<10>. */
static int qwen4_router_topk(ds4_gpu_tensor *sel, ds4_gpu_tensor *weights,
        const ds4_gpu_tensor *logits, const ds4_gpu_tensor *x, const void *map, uint64_t size,
        uint64_t off, uint32_t type, uint32_t K, ds4_gpu_tensor *sg, uint32_t T, uint32_t NE, uint32_t NS,
        bool pre) {
    using namespace qwen4_cuda;
    if (!T || !NE || NE > 512 || !NS || NS > NE || NS > 32 ||
        !tensor(sel, (uint64_t)T*NS*4) || !tensor(weights, (uint64_t)T*NS*4) || !tensor(logits, (uint64_t)T*NE*4)) return 0;
    const char *wg = K ? weight(map, size, off, row_bytes(type, K)) : NULL;
    if (K && (!wg || !tensor(x, (uint64_t)T*K*4) || !tensor(sg, (uint64_t)T*4))) return 0;
    if (pre && K == 2560 && type == 0)
        launch(router_pre<10>, T, 256, 0, (int *)sel->ptr, (float *)weights->ptr, (const float *)logits->ptr,
            (const float *)x->ptr, (const float *)wg, (float *)sg->ptr, NE, NS);
    else
        launch(router, T, 256, 0, (int *)sel->ptr, (float *)weights->ptr, (const float *)logits->ptr,
            K ? (const float *)x->ptr : nullptr, wg, K ? (float *)sg->ptr : nullptr, NE, NS, K, type);
    return launched();
}

extern "C" int ds4_gpu_qwen4_router_topk_tensor(ds4_gpu_tensor *sel, ds4_gpu_tensor *weights,
        const ds4_gpu_tensor *logits, const ds4_gpu_tensor *x, const void *map, uint64_t size,
        uint64_t off, uint32_t type, uint32_t K, ds4_gpu_tensor *sg, uint32_t T, uint32_t NE, uint32_t NS) {
    return qwen4_router_topk(sel, weights, logits, x, map, size, off, type, K, sg, T, NE, NS, true);
}

/* Tests only: the same top-k through router() for every gate, the reference
 * that router_pre must match byte for byte. */
extern "C" int ds4_gpu_qwen4_router_topk_ref_tensor(ds4_gpu_tensor *sel, ds4_gpu_tensor *weights,
        const ds4_gpu_tensor *logits, const ds4_gpu_tensor *x, const void *map, uint64_t size,
        uint64_t off, uint32_t type, uint32_t K, ds4_gpu_tensor *sg, uint32_t T, uint32_t NE, uint32_t NS) {
    return qwen4_router_topk(sel, weights, logits, x, map, size, off, type, K, sg, T, NE, NS, false);
}

/* As ds4_gpu_qwen4_moe_mid_tensor; sdo/sdt name the shared expert's down
 * matrix (K rows of M), which the kernel may prefetch into L2 for the down
 * launch that follows. sdt == UINT32_MAX gives no prefetch. */
extern "C" int ds4_gpu_qwen4_moe_mid_prefetch_tensor(ds4_gpu_tensor *mid, const ds4_gpu_tensor *x,
        const ds4_gpu_tensor *sel, const void *map, uint64_t size, uint64_t go, uint64_t uo,
        uint32_t type, uint32_t NE, uint32_t T, uint32_t NS, uint32_t K, uint32_t M,
        uint64_t sgo, uint64_t suo, uint32_t st, uint64_t sdo, uint32_t sdt) {
    using namespace qwen4_cuda;
    const unsigned NO = NS + (st != UINT_MAX);
    if (!T || !NE || !NS || !K || !M || !tensor(mid, (uint64_t)T*NO*M*4) ||
        !tensor(x, (uint64_t)T*K*4) || !tensor(sel, (uint64_t)T*NS*4)) return 0;
    const uint64_t bytes = expert_row_bytes(type, K)*M*NE, sb = row_bytes(st, K)*M;
    const char *g = weight(map,size,go,bytes), *u = weight(map,size,uo,bytes);
    const char *sg = st != UINT_MAX ? weight(map,size,sgo,sb) : NULL;
    const char *su = st != UINT_MAX ? weight(map,size,suo,sb) : NULL;
    if (!g || !u || (st != UINT_MAX && (!sg || !su))) return 0;
    const uint64_t pdb = st != UINT_MAX && sdt != UINT32_MAX ? row_bytes(sdt, M)*K : 0;
    const char *pd = pdb ? weight(map,size,sdo,pdb) : NULL;
    return moe_mv_dispatch((float *)mid->ptr,(const float *)x->ptr,(const int *)sel->ptr,
                           g,u,sg,su,type,st,NE,T,NS,K,M,false,pd,pd ? pdb : 0);
}

extern "C" int ds4_gpu_qwen4_moe_mid_tensor(ds4_gpu_tensor *mid, const ds4_gpu_tensor *x,
        const ds4_gpu_tensor *sel, const void *map, uint64_t size, uint64_t go, uint64_t uo,
        uint32_t type, uint32_t NE, uint32_t T, uint32_t NS, uint32_t K, uint32_t M,
        uint64_t sgo, uint64_t suo, uint32_t st) {
    return ds4_gpu_qwen4_moe_mid_prefetch_tensor(mid, x, sel, map, size, go, uo, type, NE, T, NS, K, M,
                                                 sgo, suo, st, 0, UINT32_MAX);
}

static int qwen4_moe_down(ds4_gpu_tensor *part, const ds4_gpu_tensor *mid,
        const ds4_gpu_tensor *sel, const void *map, uint64_t size, uint64_t off,
        uint32_t type, uint32_t NE, uint32_t T, uint32_t NS, uint32_t K, uint32_t M,
        uint64_t so, uint32_t st, bool ref) {
    using namespace qwen4_cuda;
    const unsigned NO = NS + (st != UINT_MAX);
    if (!T || !NE || !NS || !K || !M || !tensor(part,(uint64_t)T*NO*M*4) ||
        !tensor(mid,(uint64_t)T*NO*K*4) || !tensor(sel,(uint64_t)T*NS*4)) return 0;
    const char *w = weight(map,size,off,expert_row_bytes(type,K)*M*NE);
    const char *sw = st != UINT_MAX ? weight(map,size,so,row_bytes(st,K)*M) : NULL;
    if (!w || (st != UINT_MAX && !sw)) return 0;
    return moe_mv_dispatch((float *)part->ptr,(const float *)mid->ptr,(const int *)sel->ptr,
                           w,NULL,sw,NULL,type,st,NE,T,NS,K,M,true,NULL,0,ref);
}

extern "C" int ds4_gpu_qwen4_moe_down_tensor(ds4_gpu_tensor *part, const ds4_gpu_tensor *mid,
        const ds4_gpu_tensor *sel, const void *map, uint64_t size, uint64_t off,
        uint32_t type, uint32_t NE, uint32_t T, uint32_t NS, uint32_t K, uint32_t M,
        uint64_t so, uint32_t st) {
    return qwen4_moe_down(part, mid, sel, map, size, off, type, NE, T, NS, K, M, so, st, false);
}

/* Tests only: moe_down_tensor with the original MXFP4 down kernel. */
extern "C" int ds4_gpu_qwen4_moe_down_ref_tensor(ds4_gpu_tensor *part, const ds4_gpu_tensor *mid,
        const ds4_gpu_tensor *sel, const void *map, uint64_t size, uint64_t off,
        uint32_t type, uint32_t NE, uint32_t T, uint32_t NS, uint32_t K, uint32_t M,
        uint64_t so, uint32_t st) {
    return qwen4_moe_down(part, mid, sel, map, size, off, type, NE, T, NS, K, M, so, st, true);
}

extern "C" int ds4_gpu_qwen4_moe_reduce_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *part,
        const ds4_gpu_tensor *weights, const ds4_gpu_tensor *sg, const ds4_gpu_tensor *sh,
        ds4_gpu_tensor *R, const ds4_gpu_tensor *inj, uint32_t T, uint32_t NS, uint32_t stride,
        uint32_t D, uint32_t hc) {
    using namespace qwen4_cuda;
    if (!T || !NS || !D || stride < NS + (sg && !sh) || hc > 4 ||
        !tensor(out,(uint64_t)T*D*4) || !tensor(part,(uint64_t)T*stride*D*4) ||
        !tensor(weights,(uint64_t)T*NS*4) || (sg && !tensor(sg,(uint64_t)T*4)) ||
        (sh && !tensor(sh,(uint64_t)T*D*4)) ||
        (hc && (!tensor(R,(uint64_t)T*hc*D*4) || !tensor(inj,(uint64_t)T*hc*hc*8*4)))) return 0;
    launch(moe_reduce, dim3((D+255)/256,T), 256, 0, (float *)out->ptr,
        hc ? (float *)R->ptr : nullptr, hc ? (const float *)inj->ptr : nullptr, (const float *)part->ptr,
        (const float *)weights->ptr, sg ? (const float *)sg->ptr : nullptr, sh ? (const float *)sh->ptr : nullptr,
        NS,stride,D,hc);
    return launched();
}

extern "C" int ds4_gpu_qwen4_moe_build_lists_tensor(ds4_gpu_tensor *lists, ds4_gpu_tensor *counts,
        const ds4_gpu_tensor *sel, uint32_t T, uint32_t NS, uint32_t NE, uint32_t cap) {
    using namespace qwen4_cuda;
    if (!T || !NS || NS > NE || !NE || NE > 512 || cap < T || (uint64_t)T*NS > INT_MAX ||
        !tensor(lists,(uint64_t)NE*cap*4) || !tensor(counts,(uint64_t)NE*4) || !tensor(sel,(uint64_t)T*NS*4)) return 0;
    launch(expert_lists, 1, 256, 0, (int *)lists->ptr,(int *)counts->ptr,(const int *)sel->ptr,T*NS,NE,cap);
    return launched();
}

static int qwen4_moe_mm(ds4_gpu_tensor *out, const ds4_gpu_tensor *x,
        const ds4_gpu_tensor *lists, const ds4_gpu_tensor *counts, const void *map, uint64_t size,
        uint64_t o0, uint64_t o1, uint32_t type, uint32_t NE, uint32_t T, uint32_t NS,
        uint32_t NO, uint32_t K, uint32_t M, uint32_t cap, bool down) {
    using namespace qwen4_cuda;
    if (!T || !NE || !NS || NS > NE || NO < NS || !K || !M || cap < T ||
        !tensor(out,(uint64_t)T*NO*M*4) || !tensor(x,(uint64_t)T*(down ? NO : 1)*K*4) ||
        !tensor(lists,(uint64_t)NE*cap*4) || !tensor(counts,(uint64_t)NE*4)) return 0;
    const uint64_t bytes = expert_row_bytes(type,K)*M*NE;
    const char *w0 = weight(map,size,o0,bytes), *w1 = down ? NULL : weight(map,size,o1,bytes);
    if (!w0 || (!down && !w1)) return 0;
    return matrix_dispatch((float *)out->ptr,(const float *)x->ptr,w0,w1,(const int *)lists->ptr,
        (const int *)counts->ptr,type,NE,T,NS,NO,K,M,cap,down);
}

static int qwen4_moe_grouped(ds4_gpu_tensor *out, const ds4_gpu_tensor *x, const ds4_gpu_tensor *sel,
        const ds4_gpu_tensor *lists, const ds4_gpu_tensor *counts, uint32_t cap, const void *map, uint64_t size,
        uint64_t o0, uint64_t o1, uint32_t type, uint32_t NE, uint32_t T, uint32_t NS,
        uint32_t K, uint32_t M, bool down) {
    using namespace qwen4_cuda;
    const uint64_t rb = expert_row_bytes(type,K);
    if (!T || !NS || NS > NE || NE > 512 || !M || cap < T || (down ? type != 39 || K%128 : type != 12 || K%256) ||
        !tensor(out,(uint64_t)T*NS*M*4) || !tensor(x,(uint64_t)T*(down ? NS : 1)*K*4) ||
        !tensor(sel,(uint64_t)T*NS*4) || !tensor(lists,(uint64_t)NE*cap*4) || !tensor(counts,(uint64_t)NE*4)) return 0;
    const char *w0 = weight(map,size,o0,rb*M*NE), *w1 = down ? NULL : weight(map,size,o1,rb*M*NE);
    if (!w0 || (!down && !w1)) return 0;
    const dim3 grid((M+(down ? 7 : 3))/(down ? 8 : 4),T*NS);
    if (down) launch(moe_grouped<true>,grid,128,4*(2*(rb/4)+1)*4,(float *)out->ptr,(const float *)x->ptr,
        (const int *)sel->ptr,(const int *)lists->ptr,(const int *)counts->ptr,w0,w1,NE,NS,K,M,cap,rb);
    else launch(moe_grouped<false>,grid,128,0,(float *)out->ptr,(const float *)x->ptr,
        (const int *)sel->ptr,(const int *)lists->ptr,(const int *)counts->ptr,w0,w1,NE,NS,K,M,cap,rb);
    return launched();
}

extern "C" int ds4_gpu_qwen4_moe_mid_grouped_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *x,
        const ds4_gpu_tensor *sel, const ds4_gpu_tensor *lists, const ds4_gpu_tensor *counts, uint32_t cap,
        const void *map, uint64_t size, uint64_t go, uint64_t uo, uint32_t type,
        uint32_t NE, uint32_t T, uint32_t NS, uint32_t K, uint32_t M) {
    return qwen4_moe_grouped(out,x,sel,lists,counts,cap,map,size,go,uo,type,NE,T,NS,K,M,false);
}

extern "C" int ds4_gpu_qwen4_moe_down_grouped_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *x,
        const ds4_gpu_tensor *sel, const ds4_gpu_tensor *lists, const ds4_gpu_tensor *counts, uint32_t cap,
        const void *map, uint64_t size, uint64_t off, uint32_t type,
        uint32_t NE, uint32_t T, uint32_t NS, uint32_t K, uint32_t M) {
    return qwen4_moe_grouped(out,x,sel,lists,counts,cap,map,size,off,0,type,NE,T,NS,K,M,true);
}

extern "C" int ds4_gpu_qwen4_moe_mm_mid_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *x,
        const ds4_gpu_tensor *lists, const ds4_gpu_tensor *counts, const void *map, uint64_t size,
        uint64_t go, uint64_t uo, uint32_t type, uint32_t NE, uint32_t T, uint32_t NS,
        uint32_t NO, uint32_t K, uint32_t M, uint32_t cap) {
    return qwen4_moe_mm(out,x,lists,counts,map,size,go,uo,type,NE,T,NS,NO,K,M,cap,false);
}

extern "C" int ds4_gpu_qwen4_moe_mm_down_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *x,
        const ds4_gpu_tensor *lists, const ds4_gpu_tensor *counts, const void *map, uint64_t size,
        uint64_t off, uint32_t type, uint32_t NE, uint32_t T, uint32_t NS,
        uint32_t NO, uint32_t K, uint32_t M, uint32_t cap) {
    return qwen4_moe_mm(out,x,lists,counts,map,size,off,0,type,NE,T,NS,NO,K,M,cap,true);
}

extern "C" int ds4_gpu_qwen4_gdn_front_tensor(ds4_gpu_tensor *qkv, ds4_gpu_tensor *state,
        const ds4_gpu_tensor *mixed, ds4_gpu_tensor *ga, ds4_gpu_tensor *gb,
        const void *map, uint64_t size, uint64_t co, uint64_t ao, uint64_t bo, uint64_t so, uint64_t dto,
        uint32_t type, uint32_t T, uint32_t Hk, uint32_t Hv, uint32_t D, uint32_t CK, uint32_t K,
        ds4_gpu_tensor *snap, uint32_t st, ds4_gpu_tensor *snap2, uint32_t st2) {
    using namespace qwen4_cuda;
    const uint64_t C = ((uint64_t)2*Hk+Hv)*D, hb = (CK-1)*C*4;
    if (!T || !Hk || !Hv || !D || !K || C > UINT_MAX || CK < 2 || CK > 4 ||
        !tensor(qkv,(uint64_t)T*C*4) || !tensor(state,hb) ||
        (snap && (st >= T || !tensor(snap,hb))) || (snap2 && (st2 >= T || !tensor(snap2,hb)))) return 0;
    const char *w = weight(map,size,co,C*CK*4);
    if (!w || !ds4_gpu_qwen4_dense_mm_tensor(ga,mixed,map,size,ao,type,T,K,Hv) ||
              !ds4_gpu_qwen4_dense_mm_tensor(gb,mixed,map,size,bo,type,T,K,Hv)) return 0;
    launch(conv, (C+255)/256, 256, 0, (float *)qkv->ptr,(float *)state->ptr,(const float *)w,
        T,C,CK,true,snap ? (float *)snap->ptr : nullptr,st,snap2 ? (float *)snap2->ptr : nullptr,st2);
    return launched() && ds4_gpu_qwen4_gdn_prep_tensor(qkv,ga,gb,map,size,so,dto,T,Hk,Hv,D);
}

/* Decode GDN front for T <= 3 (a token or the MTP verify rows), in place of
 * the qkv+gate pair and gdn_front_tensor: the alpha/beta projections (F32)
 * run as extra blocks of the qkv+gate launch and come out activated, and
 * the causal conv (K = 4) runs in the qkv rows' epilogue.  Returns 0
 * without launching when neither applies (the caller runs the separate
 * kernels), 1 when the front ran, -1 on a launch error. */
extern "C" int ds4_gpu_qwen4_gdn_proj_tensor(
        ds4_gpu_tensor *qkv, ds4_gpu_tensor *z, ds4_gpu_tensor *hist, const ds4_gpu_tensor *mixed,
        ds4_gpu_tensor *ga, ds4_gpu_tensor *gb, const void *map, uint64_t size,
        uint64_t qkv_off, uint64_t gate_off, uint64_t conv_off, uint64_t alpha_off, uint64_t beta_off,
        uint64_t ssm_a_off, uint64_t dt_off, uint32_t ab_type,
        uint32_t T, uint32_t Hk, uint32_t Hv, uint32_t D, uint32_t CK, uint32_t K,
        ds4_gpu_tensor *snap, uint32_t st, ds4_gpu_tensor *snap2, uint32_t st2) {
    using namespace qwen4_cuda;
    const uint64_t C = ((uint64_t)2*Hk+Hv)*D, Mz = (uint64_t)Hv*D, rb = row_bytes(8,K), hb = (uint64_t)(CK-1)*C*4;
    /* the qkv/gate rows must be the single-warp Q8 rows multi_q8 runs */
    if (!T || T > 3 || !Hk || !Hv || D < 32 || D > 128 || D%32 || !K || K%128 || !rb || CK < 2 || CK > 4 ||
        C > UINT_MAX || C <= 1536 || Mz <= 1536 ||
        !tensor(mixed,(uint64_t)T*K*4) || ((uintptr_t)mixed->ptr & 15) ||
        !tensor(qkv,(uint64_t)T*C*4) || !tensor(z,(uint64_t)T*Mz*4) || !tensor(hist,hb) ||
        !tensor(ga,(uint64_t)T*Hv*4) || !tensor(gb,(uint64_t)T*Hv*4) ||
        (snap && (st >= T || !tensor(snap,hb))) || (snap2 && (st2 >= T || !tensor(snap2,hb)))) return 0;
    const char *wqkv = weight(map,size,qkv_off,rb*C), *wz = weight(map,size,gate_off,rb*Mz);
    const char *cw = weight(map,size,conv_off,C*CK*4);
    const char *A = weight(map,size,ssm_a_off,(uint64_t)Hv*4), *bias = weight(map,size,dt_off,(uint64_t)Hv*4);
    const uint64_t ab_rb = row_bytes(ab_type,K);
    const char *wa = ab_rb ? weight(map,size,alpha_off,ab_rb*Hv) : NULL;
    const char *wb = ab_rb ? weight(map,size,beta_off,ab_rb*Hv) : NULL;
    if (!wqkv || !wz || !cw || !A || !bias || !wa || !wb) return 0;
    /* only where matvec_dispatch would run alpha/beta as matvec_split<0,ROWS> */
    const bool fold_ab = ab_type == 0 &&
        Hv <= 1536 && K >= 1024 && !((uintptr_t)wa & 15) && !((uintptr_t)wb & 15) && !(ab_rb & 15);
    const bool fold_conv = CK == 4;
    if (!fold_ab && !fold_conv) return 0;
    gdn_proj_args p = {};
    p.wqkv = wqkv; p.wz = wz; p.walpha = wa; p.wbeta = wb;
    p.qkv = (float *)qkv->ptr; p.z = (float *)z->ptr; p.ga = (float *)ga->ptr; p.gb = (float *)gb->ptr;
    p.hist = (float *)hist->ptr;
    p.snap = snap ? (float *)snap->ptr : nullptr; p.snap2 = snap2 ? (float *)snap2->ptr : nullptr;
    p.conv_w = (const float *)cw; p.A = (const float *)A; p.bias = (const float *)bias;
    p.C = (unsigned)C; p.Mz = (unsigned)Mz; p.Hv = Hv; p.ab_tiles = fold_ab ? 2*Hv : 0;
    p.st = snap ? st : UINT_MAX; p.st2 = snap2 ? st2 : UINT_MAX;
    const unsigned grid = p.ab_tiles + (unsigned)((C+3)/4 + (Mz+3)/4);
    const float *x = (const float *)mixed->ptr;
#define QWEN_PROJ(R) if (fold_conv) launch(gdn_proj<R,true>,grid,128,0,p,x,T,K); \
                     else launch(gdn_proj<R,false>,grid,128,0,p,x,T,K)
    if (T == 1) { QWEN_PROJ(1); } else if (T == 2) { QWEN_PROJ(2); } else { QWEN_PROJ(4); }
#undef QWEN_PROJ
    if (!launched()) return -1;
    if (!fold_ab && (!ds4_gpu_qwen4_dense_mm_tensor(ga,mixed,map,size,alpha_off,ab_type,T,K,Hv) ||
                     !ds4_gpu_qwen4_dense_mm_tensor(gb,mixed,map,size,beta_off,ab_type,T,K,Hv))) return -1;
    if (!fold_conv)
        launch(conv, (unsigned)((C+255)/256), 256, 0, (float *)qkv->ptr,(float *)hist->ptr,(const float *)cw,
            T,(unsigned)C,CK,true,snap ? (float *)snap->ptr : nullptr,st,snap2 ? (float *)snap2->ptr : nullptr,st2);
    if (fold_ab) launch(gdn_prep<false>,dim3(Hk,T),32,0,(float *)qkv->ptr,
        (float *)ga->ptr,(float *)gb->ptr,(const float *)A,(const float *)bias,Hk,Hv,D);
    else launch(gdn_prep<true>,dim3(Hk,T),32,0,(float *)qkv->ptr,
        (float *)ga->ptr,(float *)gb->ptr,(const float *)A,(const float *)bias,Hk,Hv,D);
    return launched() ? 1 : -1;
}

extern "C" int ds4_gpu_qwen4_decode_fusions_enabled(void) { return 1; }

static int qwen4_hc_combine_norm(ds4_gpu_tensor *next, const ds4_gpu_tensor *blk,
        const ds4_gpu_tensor *oldinj, ds4_gpu_tensor *xn, ds4_gpu_tensor *inj, const ds4_gpu_tensor *R,
        const void *map, uint64_t size, uint64_t go, uint64_t io, uint32_t type,
        uint32_t T, uint32_t E, uint32_t hc, uint32_t ni, float eps, bool ref) {
    using namespace qwen4_cuda;
    const uint64_t bytes = (uint64_t)T*E*hc*4;
    if (!qwen4_cuda::tensor(next,bytes) || !qwen4_cuda::tensor(R,bytes) || next->ptr == R->ptr ||
        !inj || !oldinj || inj->ptr == oldinj->ptr) return 0;
    if (T <= 8 && T && E && hc && hc <= 4 && ni <= 4 && type == 1) {
        if (!tensor(xn,bytes) || !tensor(blk,(uint64_t)T*E*4) ||
            !tensor(oldinj,(uint64_t)T*hc*hc*8*4) || !tensor(inj,(uint64_t)T*hc*ni*8*4)) return 0;
        const char *gamma = weight(map,size,go,(uint64_t)E*hc*4);
        const char *wi = weight(map,size,io,(uint64_t)E*hc*ni*2);
        if (!gamma || !wi) return 0;
        /* Every chunk reads the old residual, then writes its disjoint piece
         * of next. This folds the copy and combine into normalization without
         * a grid barrier or an in-place read/write race. */
        launch(!ref && presync_load(T) ? hc_norm<1,true,true> : hc_norm<1,true>, dim3(hc*8,T), 128, 0, (float *)xn->ptr,(float *)inj->ptr,
            (const float *)R->ptr,(const float *)gamma,wi,E,hc,ni,eps,
            (float *)next->ptr,(const float *)blk->ptr,(const float *)oldinj->ptr,hc_norm_late(T));
        return launched();
    }
    if (!cuda_ok(cudaMemcpyAsync(next->ptr,R->ptr,bytes,cudaMemcpyDeviceToDevice,cuda_decode_stream()),"Qwen HC copy")) return 0;
    return ds4_gpu_qwen4_hc_combine_tensor(next,blk,oldinj,T,E,hc) &&
           qwen4_hc_norm(xn,inj,next,map,size,go,io,type,T,E,hc,ni,eps,ref);
}

extern "C" int ds4_gpu_qwen4_hc_combine_norm_tensor(ds4_gpu_tensor *next, const ds4_gpu_tensor *blk,
        const ds4_gpu_tensor *oldinj, ds4_gpu_tensor *xn, ds4_gpu_tensor *inj, const ds4_gpu_tensor *R,
        const void *map, uint64_t size, uint64_t go, uint64_t io, uint32_t type,
        uint32_t T, uint32_t E, uint32_t hc, uint32_t ni, float eps) {
    return qwen4_hc_combine_norm(next, blk, oldinj, xn, inj, R, map, size, go, io, type, T, E, hc, ni, eps,
                                    false);
}

/* Tests only: hc_combine_norm_tensor with the post-wait kernels. */
extern "C" int ds4_gpu_qwen4_hc_combine_norm_ref_tensor(ds4_gpu_tensor *next, const ds4_gpu_tensor *blk,
        const ds4_gpu_tensor *oldinj, ds4_gpu_tensor *xn, ds4_gpu_tensor *inj, const ds4_gpu_tensor *R,
        const void *map, uint64_t size, uint64_t go, uint64_t io, uint32_t type,
        uint32_t T, uint32_t E, uint32_t hc, uint32_t ni, float eps) {
    return qwen4_hc_combine_norm(next, blk, oldinj, xn, inj, R, map, size, go, io, type, T, E, hc, ni, eps,
                                    true);
}

extern "C" int ds4_gpu_qwen4_mtp_stage_tensor(ds4_gpu_tensor *cat, const ds4_gpu_tensor *e,
        const ds4_gpu_tensor *R, const void *map, uint64_t size, uint64_t eo, uint64_t ho,
        uint32_t E, uint32_t hc, float eps) {
    using namespace qwen4_cuda;
    if (!E || !hc || hc > 4 || !tensor(cat,(uint64_t)(hc+1)*2*E*4) ||
        !tensor(e,(uint64_t)E*4) || !tensor(R,(uint64_t)hc*E*4)) return 0;
    const char *ge = weight(map,size,eo,(uint64_t)E*4), *gh = weight(map,size,ho,(uint64_t)hc*E*4);
    if (!ge || !gh) return 0;
    launch(mtp_stage, hc+1, 256, 0, (float *)cat->ptr,(const float *)e->ptr,
        (const float *)R->ptr,(const float *)ge,(const float *)gh,E,hc,eps);
    return launched();
}

extern "C" int ds4_gpu_qwen4_mtp_stage_rows_tensor(ds4_gpu_tensor *cat, const ds4_gpu_tensor *e,
        const ds4_gpu_tensor *R, const void *map, uint64_t size, uint64_t eo, uint64_t ho,
        uint32_t T, uint32_t E, uint32_t hc, float eps) {
    using namespace qwen4_cuda;
    if (!T || !E || !hc || hc > 4 || !tensor(cat,(uint64_t)T*(hc+1)*2*E*4) ||
        !tensor(e,(uint64_t)T*E*4) || !tensor(R,(uint64_t)T*hc*E*4)) return 0;
    const char *ge = weight(map,size,eo,(uint64_t)E*4), *gh = weight(map,size,ho,(uint64_t)hc*E*4);
    if (!ge || !gh) return 0;
    launch(mtp_stage_rows, dim3(hc+1, T), 256, 0, (float *)cat->ptr,(const float *)e->ptr,
        (const float *)R->ptr,(const float *)ge,(const float *)gh,E,hc,eps);
    return launched();
}

extern "C" int ds4_gpu_qwen4_mtp_combine_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *proj,
        uint32_t E, uint32_t hc) {
    using namespace qwen4_cuda;
    if (!E || !hc || hc > 4 || !tensor(out,(uint64_t)hc*E*4) || !tensor(proj,(uint64_t)(hc+1)*E*4)) return 0;
    launch(mtp_combine, ((uint64_t)E*hc+255)/256, 256, 0, (float *)out->ptr,(const float *)proj->ptr,E,hc);
    return launched();
}

extern "C" int ds4_gpu_qwen4_argmax_host_tensor(ds4_gpu_tensor *out, ds4_gpu_tensor *scratch,
        const ds4_gpu_tensor *logits, uint32_t N, int32_t *host_out) {
    using namespace qwen4_cuda;
    const uint64_t blocks = ((uint64_t)N+4095)/4096;
    if (!N || !tensor(out,4) || !tensor(logits,(uint64_t)N*4) || !tensor(scratch,blocks*sizeof(max_pair))) return 0;
    launch(argmax<false>, blocks, 256, 0, (int *)out->ptr,(max_pair *)scratch->ptr,(const float *)logits->ptr,N,
           (int *)nullptr);
    launch(argmax<true>, 1, 256, 0, (int *)out->ptr,(max_pair *)scratch->ptr,nullptr,blocks,(int *)host_out);
    return launched();
}

extern "C" int ds4_gpu_qwen4_argmax_rows_host_tensor(ds4_gpu_tensor *out, ds4_gpu_tensor *scratch,
        const ds4_gpu_tensor *logits, uint32_t N, uint32_t T, int32_t *host_out) {
    using namespace qwen4_cuda;
    const uint64_t blocks = ((uint64_t)N+4095)/4096;
    if (!N || !T || T > 65535 || !tensor(out,(uint64_t)T*4) || !tensor(logits,(uint64_t)T*N*4) ||
        !tensor(scratch,(uint64_t)T*blocks*sizeof(max_pair))) return 0;
    launch(argmax<false>,dim3(blocks,T),256,0,(int *)out->ptr,(max_pair *)scratch->ptr,(const float *)logits->ptr,N,
           (int *)nullptr);
    launch(argmax<true>,dim3(1,T),256,0,(int *)out->ptr,(max_pair *)scratch->ptr,nullptr,blocks,(int *)host_out);
    return launched();
}

extern "C" int ds4_gpu_qwen4_argmax_tensor(ds4_gpu_tensor *out, ds4_gpu_tensor *scratch,
        const ds4_gpu_tensor *logits, uint32_t N) {
    return ds4_gpu_qwen4_argmax_host_tensor(out, scratch, logits, N, NULL);
}

extern "C" int ds4_gpu_qwen4_argmax_rows_tensor(ds4_gpu_tensor *out, ds4_gpu_tensor *scratch,
        const ds4_gpu_tensor *logits, uint32_t N, uint32_t T) {
    return ds4_gpu_qwen4_argmax_rows_host_tensor(out, scratch, logits, N, T, NULL);
}

extern "C" int ds4_gpu_qwen4_stage_host_tensor(ds4_gpu_tensor *dst, uint64_t off, const void *src, uint64_t bytes) {
    using namespace qwen4_cuda;
    if (!dst || !src || !bytes || (bytes | off | (uintptr_t)src) % 16u ||
        off > dst->bytes || bytes > dst->bytes - off) return 0;
    const uint64_t n = bytes / 16u;
    if (n > UINT32_MAX - 255u) return 0;
    launch(stage_host, (unsigned)((n + 255u) / 256u), 256, 0,
           (uint4 *)((char *)dst->ptr + off), (const uint4 *)src, (unsigned)n);
    return launched();
}

extern "C" int ds4_gpu_qwen4_vision_encode(float *out, const float *patches, const float *pos_embed,
        uint32_t N, uint32_t grid_w, const void *map, uint64_t size, const ds4_qwen4_vision_weights *w) {
    using namespace qwen4_cuda;
    if (!out || !patches || !pos_embed || !w || !map || !N || N > 65535 || ds4_gpu_commands_active()) return 0;
    const unsigned E = w->n_embd, FF = w->n_ff, H = w->n_head, D = H ? E/H : 0;
    const unsigned P = w->n_patch, M = w->n_merge, O = w->n_out;
    if (M != 2 || !grid_w || grid_w%M || N%grid_w || (N/grid_w)%M ||
        (D != 64 && D != 72) || H*D != E || !FF || !O || !P || P > 32 || E%32) return 0;
    const unsigned IP = 3*P*P, ME = E*M*M, merged = N/(M*M);
    std::vector<ds4_gpu_tensor *> buffers;
    auto alloc = [&](uint64_t n) {
        ds4_gpu_tensor *t = ds4_gpu_tensor_alloc(n*4);
        buffers.push_back(t);
        return t;
    };
    ds4_gpu_tensor *patch = alloc((uint64_t)N*IP), *a0 = alloc((uint64_t)N*E), *a1 = alloc((uint64_t)N*E);
    ds4_gpu_tensor *pos = alloc((uint64_t)N*E), *x = alloc((uint64_t)N*E), *tmp = alloc((uint64_t)N*E);
    ds4_gpu_tensor *qkv = alloc((uint64_t)N*3*E), *q = alloc((uint64_t)N*E), *k = alloc((uint64_t)N*E);
    ds4_gpu_tensor *v = alloc((uint64_t)N*E), *attn = alloc((uint64_t)N*E), *ffn = alloc((uint64_t)N*FF);
    ds4_gpu_tensor *m0 = alloc((uint64_t)merged*ME), *res = alloc((uint64_t)merged*O);
    bool ok = true, active = false;
    for (const auto b : buffers) if (!b) ok = false;
    auto ptr = [](ds4_gpu_tensor *t) { return (float *)t->ptr; };
    auto wf = [&](uint64_t off, unsigned n) { return (const float *)weight(map,size,off,(uint64_t)n*4); };
    auto mm = [&](ds4_gpu_tensor *dst, const ds4_gpu_tensor *src, uint64_t off, unsigned type,
                  unsigned rows, unsigned K, unsigned M) {
        return ds4_gpu_qwen4_dense_mm_tensor(dst,src,map,size,off,type,rows,K,M);
    };
    auto norm = [&](ds4_gpu_tensor *dst, const ds4_gpu_tensor *src, uint64_t wo, uint64_t bo) {
        const float *wgt = wf(wo,E), *bias = wf(bo,E);
        if (!wgt || !bias) return 0;
        launch(vis_norm, N, 256, 0, ptr(dst),(const float *)src->ptr,wgt,bias,E,w->eps);
        return launched();
    };
    auto add = [&](ds4_gpu_tensor *dst, const ds4_gpu_tensor *src, uint64_t bo, unsigned rows,
                   unsigned width, unsigned mode) {
        const float *bias = wf(bo,width);
        if (!bias) return 0;
        launch(vis_add, ((uint64_t)rows*width+255)/256, 256, 0, ptr(dst),
            src ? (const float *)src->ptr : nullptr,bias,rows,width,mode);
        return launched();
    };
    do {
        if (!ok) break;
        ok = ds4_gpu_tensor_write(patch,0,patches,(uint64_t)N*IP*4) &&
             ds4_gpu_tensor_write(pos,0,pos_embed,(uint64_t)N*E*4);
        if (!ok || !(active = ds4_gpu_begin_commands())) { ok = false; break; }
        ok = mm(a0,patch,w->patch_w0,w->patch_type,N,IP,E) && mm(a1,patch,w->patch_w1,w->patch_type,N,IP,E);
        const float *pb = wf(w->patch_b,E);
        if (!ok || !pb) { ok = false; break; }
        launch(vis_patch, ((uint64_t)N*E+255)/256, 256, 0, ptr(x),ptr(a0),ptr(a1),pb,ptr(pos),N,E);
        ok = launched();
        for (unsigned l = 0; ok && l < DS4_QWEN4_VISION_LAYERS; l++) {
            const auto &lw = w->layer[l];
            ok = norm(tmp,x,lw.ln1_w,lw.ln1_b) && mm(qkv,tmp,lw.qkv_w,lw.qkv_type,N,E,3*E);
            const float *qb = wf(lw.qkv_b,3*E);
            if (!ok || !qb) { ok = false; break; }
            launch(vis_qkv, N, 256, 0, ptr(q),ptr(k),ptr(v),ptr(qkv),qb,H,D,grid_w);
            launch(vis_attention, dim3(N,H), 32, 0, ptr(attn),ptr(q),ptr(k),ptr(v),N,H,D);
            ok = launched() && mm(tmp,attn,lw.out_w,lw.out_type,N,E,E) && add(x,tmp,lw.out_b,N,E,2) &&
                 norm(tmp,x,lw.ln2_w,lw.ln2_b) && mm(ffn,tmp,lw.up_w,lw.up_type,N,E,FF) &&
                 add(ffn,NULL,lw.up_b,N,FF,0) && mm(tmp,ffn,lw.down_w,lw.down_type,N,FF,E) &&
                 add(x,tmp,lw.down_b,N,E,2);
        }
        ok = ok && norm(tmp,x,w->post_ln_w,w->post_ln_b) && mm(m0,tmp,w->mm0_w,w->mm0_type,merged,ME,ME) &&
             add(m0,NULL,w->mm0_b,merged,ME,1) && mm(res,m0,w->mm2_w,w->mm2_type,merged,ME,O) &&
             add(res,NULL,w->mm2_b,merged,O,2);
    } while (false);
    if (active && !ds4_gpu_end_commands()) ok = false;
    if (ok) ok = ds4_gpu_tensor_read(res,0,out,(uint64_t)merged*O*4);
    for (auto it = buffers.rbegin(); it != buffers.rend(); ++it) ds4_gpu_tensor_free(*it);
    return ok;
}

extern "C" int ds4_gpu_qwen4_attn_decode_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *gate, const ds4_gpu_tensor *kc, const ds4_gpu_tensor *vc,
        const ds4_gpu_tensor *sel, const ds4_gpu_tensor *count, ds4_gpu_tensor *partial,
        uint32_t T, uint32_t H, uint32_t Hkv, uint32_t D, uint32_t pos0, bool sparse, uint32_t stride, float scale) {
    using namespace qwen4_cuda;
    const uint64_t n = (uint64_t)T * H * D * 4, cb = ((uint64_t)pos0 + T) * Hkv * D * 2;
    if (!T || !H || !Hkv || H % Hkv || (D != 32 && D != 128 && D != 256) ||
        !tensor(out, n) || !tensor(q, n) || !tensor(gate, n) || !tensor(kc, cb) || !tensor(vc, cb) ||
        (sparse && (!stride || !tensor(sel, (uint64_t)T * stride * 4) || !tensor(count, (uint64_t)T * 4)))) return 0;
    /* With partial scratch these are decode rows: row t takes the kernel
     * and key split of the one-token decode at pos0 + t (grouped from 128
     * keys), so MTP verify rows round exactly like plain decode. Prefill
     * batches of 32+ rows take the grouped kernel. Rows [0, g0) run scalar,
     * rows [g0, T) grouped; dense key counts grow with the row. */
    const bool split = partial != nullptr;
    const unsigned max_splits = split ? attn_splits(attn_keys(sparse,stride,pos0+T-1)) : 1;
    if (split && !tensor(partial, ((uint64_t)(T-1)*64+max_splits)*H*(D+2)*4)) return 0;
    const bool group_ok = D == 256 && H/Hkv <= 16 && !((uintptr_t)kc->ptr&15) && !((uintptr_t)vc->ptr&15) &&
        ds4_cuda_attn_tokentile_arch_ok() && !g_quality_mode && !getenv("DS4_QWEN4_NO_ATTN_MM");
    unsigned g0 = T;
    if (group_ok && !split && T >= 32) g0 = 0;
    if (group_ok && split)
        for (g0 = 0; g0 < T && attn_keys(sparse,stride,pos0+g0) < 128; g0++) {}
    float *o = (float *)out->ptr, *part = split ? (float *)partial->ptr : nullptr;
    const float *qp = (const float *)q->ptr, *gp = (const float *)gate->ptr;
    const __half *kp = (const __half *)kc->ptr, *vp = (const __half *)vc->ptr;
    const int *sp = sparse ? (const int *)sel->ptr : nullptr;
    const unsigned *cp = sparse ? (const unsigned *)count->ptr : nullptr;
    if (g0) {
        const dim3 grid((H + 3) / 4, g0, split ? attn_splits(attn_keys(sparse,stride,pos0+g0-1)) : 1);
#define QWEN_ATTN(DIM) launch(attention<DIM>, grid, 128, 0, o, part, qp, gp, kp, vp, sp, cp, H, Hkv, pos0, stride, sparse, scale)
        if (D == 32) { QWEN_ATTN(32); }
        else if (D == 128) { QWEN_ATTN(128); }
        else { QWEN_ATTN(256); }
#undef QWEN_ATTN
    }
    if (g0 < T) {
        const uint64_t r = (uint64_t)g0*H*D;
        launch(attention_group, dim3(Hkv,T-g0,max_splits), 256, 0, o+r,
            part ? part+(uint64_t)g0*H*64*(D+2) : nullptr, qp+r, gp+r, kp, vp,
            sp ? sp+(uint64_t)g0*stride : nullptr, cp ? cp+g0 : nullptr, H, Hkv, pos0+g0, stride, sparse, scale);
    }
    if (!launched()) return 0;
    if (max_splits > 1) launch(attn_merge, dim3(H, T), D, 0, o, part, gp, H, D, pos0, stride, sparse);
    return launched();
}

/* ------------------------------------------------------------------------
 * Qwen decode-batch row APIs.
 *
 * The session-batch driver in ds4.c stacks the rows of several sessions in
 * one arena: the weight products run once over all rows, while the steps
 * that touch a session's own state (convolution history, delta-net state,
 * KV and indexer caches) go through these row APIs. Device descriptors put
 * session index in the grid while retaining each session's recurrence order
 * and private attention scratch. */

/* Mirrors of ds4_gpu_qwen4_attn_row / ds4_gpu_qwen4_gdn_row (ds4_gpu.h). */
typedef struct {
    ds4_gpu_tensor *k_cache, *v_cache, *ik_cache, *block_key;
    const ds4_gpu_tensor *pos3;
    uint32_t pos;
    int use_sel;
} ds4_gpu_qwen4_attn_row;
typedef struct {
    ds4_gpu_tensor *state, *hist, *snap_state, *snap_hist;
    uint32_t row0, n_tok;
} ds4_gpu_qwen4_gdn_row;
static_assert(sizeof(ds4_gpu_qwen4_attn_row) == 48u, "ds4_gpu_qwen4_attn_row must match ds4_gpu.h");
static_assert(sizeof(ds4_gpu_qwen4_gdn_row) == 40u, "ds4_gpu_qwen4_gdn_row must match ds4_gpu.h");

namespace qwen4_cuda {

enum { ROWS_MAX = 128 };

/* The selection and expansion row APIs take no row array: the staged
 * table keeps its row count and block ratio on the host. */
struct staged_attn { const void *table; uint64_t entry0; uint32_t n, ratio; };
static staged_attn g_staged_attn[64];

static staged_attn *find_attn(const ds4_gpu_tensor *table, uint64_t entry0, bool create) {
    staged_attn *free_slot = NULL;
    for (staged_attn &s : g_staged_attn) {
        if (s.table == table && s.entry0 == entry0 && s.n) return &s;
        if (!s.n && !free_slot) free_slot = &s;
    }
    if (!create) return NULL;
    if (!free_slot) free_slot = &g_staged_attn[entry0 % 64];
    free_slot->table = table;
    free_slot->entry0 = entry0;
    return free_slot;
}

} // namespace qwen4_cuda

extern "C" int ds4_gpu_qwen4_attn_rows_stage(ds4_gpu_tensor *table, uint64_t entry0,
        const ds4_gpu_qwen4_attn_row *rows, uint32_t n_rows, uint32_t ratio) {
    using namespace qwen4_cuda;
    if (!rows || !n_rows || n_rows > ROWS_MAX || !ratio) return 0;
    staged_attn *s = find_attn(table, entry0, true);
    s->n = n_rows;
    s->ratio = ratio;
    if (!tensor(table,(entry0+n_rows)*sizeof(attn_row))) return 0;
    row_batch<attn_row> batch = {};
    for (unsigned i = 0; i < n_rows; i++) {
        if (!rows[i].k_cache || !rows[i].v_cache || !rows[i].ik_cache || !rows[i].block_key || !rows[i].pos3) return 0;
        batch.rows[i] = {(__half *)rows[i].k_cache->ptr,(__half *)rows[i].v_cache->ptr,
            (float *)rows[i].ik_cache->ptr,(__half *)rows[i].block_key->ptr,
            (const uint32_t *)rows[i].pos3->ptr,rows[i].pos,rows[i].use_sel};
    }
    launch(stage_rows<attn_row>,1,128,0,(attn_row *)table->ptr+entry0,batch,n_rows);
    return launched();
}

extern "C" int ds4_gpu_qwen4_attn_prep_rows_tensor(
        ds4_gpu_tensor *q_out, ds4_gpu_tensor *gate_out, ds4_gpu_tensor *iq_out,
        const ds4_gpu_tensor *qg, const ds4_gpu_tensor *kproj, const ds4_gpu_tensor *vproj,
        const ds4_gpu_tensor *iq, const ds4_gpu_tensor *ik,
        const ds4_gpu_tensor *table, uint64_t entry0, const ds4_gpu_qwen4_attn_row *rows, uint32_t n_rows,
        const void *map, uint64_t size, uint64_t qo, uint64_t ko, uint64_t io,
        uint32_t H, uint32_t Hkv, uint32_t D, uint32_t nrot, uint32_t Hi, uint32_t Di, float base, float eps) {
    using namespace qwen4_cuda;
    const uint64_t qn = (uint64_t)n_rows*H*D*4, kn = (uint64_t)n_rows*Hkv*D*4, in = (uint64_t)n_rows*Hi*Di*4;
    if (!rows || !n_rows || n_rows > ROWS_MAX || !H || !Hkv || H%Hkv || !Hi ||
        !D || D > 256 || D%32 || !Di || Di > 128 || Di%32 || nrot > D || nrot > Di || nrot > 64 || nrot%2 ||
        !tensor(table,(entry0+n_rows)*sizeof(attn_row)) || !tensor(q_out,qn) || !tensor(gate_out,qn) ||
        !tensor(qg,qn*2) || !tensor(kproj,kn) || !tensor(vproj,kn) || !tensor(iq_out,in) || !tensor(iq,in) ||
        !tensor(ik,(uint64_t)n_rows*Di*4)) return 0;
    for (unsigned i = 0; i < n_rows; i++) {
        const uint64_t cap = (uint64_t)rows[i].pos+1;
        if (!tensor(rows[i].k_cache,cap*Hkv*D*2) || !tensor(rows[i].v_cache,cap*Hkv*D*2) ||
            !tensor(rows[i].ik_cache,cap*Di*4) || !tensor(rows[i].pos3,cap*16)) return 0;
    }
    const float *gq = (const float *)weight(map,size,qo,(uint64_t)D*4), *gk = (const float *)weight(map,size,ko,(uint64_t)D*4),
                *giq = (const float *)weight(map,size,io,(uint64_t)Di*4);
    if (!gq || !gk || !giq) return 0;
    launch(attn_prep_rows,dim3(H+Hkv+Hi+1,n_rows),32,0,(float *)q_out->ptr,(float *)gate_out->ptr,(float *)iq_out->ptr,
        (const float *)qg->ptr,(const float *)kproj->ptr,(const float *)vproj->ptr,(const float *)iq->ptr,(const float *)ik->ptr,
        (const attn_row *)table->ptr+entry0,gq,gk,giq,H,Hkv,D,Hi,Di,eps,rope(nrot,base));
    return launched();
}

extern "C" int ds4_gpu_qwen4_idx_block_key_rows_tensor(
        const ds4_gpu_tensor *table, uint64_t entry0, const ds4_gpu_qwen4_attn_row *rows, uint32_t n_rows,
        const void *map, uint64_t size, uint64_t iko,
        uint32_t ratio, uint32_t Di, uint32_t nrot, float base, float eps) {
    using namespace qwen4_cuda;
    if (!rows || !n_rows || n_rows > ROWS_MAX || !ratio || !Di || Di > 128 || Di%32 ||
        nrot > Di || nrot > 64 || nrot%2 || !tensor(table,(entry0+n_rows)*sizeof(attn_row))) return 0;
    for (unsigned i = 0; i < n_rows; i++) if ((rows[i].pos+1)%ratio == 0) {
        const uint64_t cap = (uint64_t)rows[i].pos+1;
        if (!tensor(rows[i].block_key,cap/ratio*Di*2) || !tensor(rows[i].ik_cache,cap*Di*4) || !tensor(rows[i].pos3,cap*16)) return 0;
    }
    const float *gamma = (const float *)weight(map,size,iko,(uint64_t)Di*4);
    if (!gamma) return 0;
    launch(block_key_rows,dim3(1,n_rows),32,0,(const attn_row *)table->ptr+entry0,gamma,ratio,Di,eps,rope(nrot,base));
    return launched();
}

extern "C" int ds4_gpu_qwen4_idx_score_rows_tensor(
        ds4_gpu_tensor *score, ds4_gpu_tensor *tile_max, const ds4_gpu_tensor *iq,
        const ds4_gpu_tensor *table, uint64_t entry0, const ds4_gpu_qwen4_attn_row *rows, uint32_t n_rows,
        uint32_t n_block_stride, uint32_t Hi, uint32_t Di, uint32_t ratio) {
    using namespace qwen4_cuda;
    (void)tile_max;   /* CUDA selection scans the scores; the tile maxima are Metal's */
    if (!rows || !n_rows || n_rows > ROWS_MAX || !ratio || !Hi || !Di || Di > 128 || Di%32 || !n_block_stride ||
        !tensor(table,(entry0+n_rows)*sizeof(attn_row)) || !tensor(score,(uint64_t)n_rows*n_block_stride*4) ||
        !tensor(iq,(uint64_t)n_rows*Hi*Di*4)) return 0;
    for (unsigned i = 0; i < n_rows; i++) if (rows[i].use_sel) {
        const uint64_t N = ((uint64_t)rows[i].pos+1)/ratio;
        if (N > n_block_stride || !tensor(rows[i].block_key,N*Di*2)) return 0;
    }
    launch(idx_score_rows,dim3((n_block_stride+3)/4,n_rows),128,0,(float *)score->ptr,(const float *)iq->ptr,
        (const attn_row *)table->ptr+entry0,n_block_stride,Hi,Di,ratio);
    return launched();
}

extern "C" int ds4_gpu_qwen4_idx_select_rows_tensor(
        ds4_gpu_tensor *sel, const ds4_gpu_tensor *score, const ds4_gpu_tensor *tile_max,
        const ds4_gpu_tensor *table, uint64_t entry0, const ds4_gpu_qwen4_attn_row *rows, uint32_t n_rows,
        uint32_t n_block_stride, uint32_t top_k) {
    using namespace qwen4_cuda;
    (void)tile_max;
    const staged_attn *meta = find_attn(table,entry0,false);
    if (!meta || !rows || !n_rows || n_rows > meta->n || !top_k ||
        !tensor(table,(entry0+n_rows)*sizeof(attn_row)) || !tensor(score,(uint64_t)n_rows*n_block_stride*4) ||
        !tensor(sel,(uint64_t)n_rows*top_k*4)) return 0;
    for (unsigned i = 0; i < n_rows; i++) if (rows[i].use_sel) {
        const unsigned N = (rows[i].pos+1)/meta->ratio;
        if (top_k > N || N > n_block_stride) return 0;
    }
    launch(idx_select_rows,n_rows,256,0,(int *)sel->ptr,(const float *)score->ptr,
        (const attn_row *)table->ptr+entry0,n_block_stride,top_k,meta->ratio);
    return launched();
}

extern "C" int ds4_gpu_qwen4_idx_expand_rows_tensor(
        ds4_gpu_tensor *sel_tokens, ds4_gpu_tensor *n_sel, const ds4_gpu_tensor *sel_blocks,
        const ds4_gpu_tensor *table, uint64_t entry0, uint32_t n_rows,
        uint32_t n_sel_blocks, uint32_t ratio, uint32_t sel_stride) {
    using namespace qwen4_cuda;
    const staged_attn *meta = find_attn(table,entry0,false);
    if (!meta || !n_rows || n_rows > meta->n || !ratio || !n_sel_blocks ||
        (uint64_t)n_sel_blocks*ratio+ratio > sel_stride || !tensor(table,(entry0+n_rows)*sizeof(attn_row)) ||
        !tensor(sel_tokens,(uint64_t)n_rows*sel_stride*4) || !tensor(n_sel,(uint64_t)n_rows*4) ||
        !tensor(sel_blocks,(uint64_t)n_rows*n_sel_blocks*4)) return 0;
    launch(idx_expand_rows,n_rows,256,0,(int *)sel_tokens->ptr,(unsigned *)n_sel->ptr,(const int *)sel_blocks->ptr,
        (const attn_row *)table->ptr+entry0,n_sel_blocks,ratio,sel_stride);
    return launched();
}

extern "C" int ds4_gpu_qwen4_attn_decode_rows_tensor(
        ds4_gpu_tensor *out, const ds4_gpu_tensor *q, const ds4_gpu_tensor *gate,
        const ds4_gpu_tensor *sel_tokens, const ds4_gpu_tensor *n_sel, ds4_gpu_tensor *part,
        const ds4_gpu_tensor *table, uint64_t entry0, const ds4_gpu_qwen4_attn_row *rows, uint32_t n_rows,
        uint32_t H, uint32_t Hkv, uint32_t D, uint32_t sel_stride, float scale) {
    using namespace qwen4_cuda;
    const uint64_t n = (uint64_t)n_rows*H*D*4;
    if (!rows || !n_rows || n_rows > ROWS_MAX || !H || !Hkv || H%Hkv || (D != 32 && D != 128 && D != 256) ||
        !tensor(out,n) || !tensor(q,n) || !tensor(gate,n) || !tensor(part,(uint64_t)n_rows*H*64*(D+2)*4) ||
        !tensor(table,(entry0+n_rows)*sizeof(attn_row))) return 0;
    unsigned max_splits = 1, min_keys = UINT_MAX;
    bool aligned = true;
    for (unsigned i = 0; i < n_rows; i++) {
        const uint64_t cb = ((uint64_t)rows[i].pos+1)*Hkv*D*2;
        const unsigned keys = attn_keys(rows[i].use_sel,sel_stride,rows[i].pos);
        max_splits = std::max(max_splits,attn_splits(keys));
        min_keys = std::min(min_keys,keys);
        if (!tensor(rows[i].k_cache,cb) || !tensor(rows[i].v_cache,cb) || (rows[i].use_sel &&
            (!sel_stride || !tensor(sel_tokens,(uint64_t)n_rows*sel_stride*4) || !tensor(n_sel,(uint64_t)n_rows*4)))) return 0;
        aligned = aligned && !((uintptr_t)rows[i].k_cache->ptr&15) && !((uintptr_t)rows[i].v_cache->ptr&15);
    }
#define QWEN_ATTN_ROWS(DIM) launch(attention_rows<DIM>,dim3((H+3)/4,n_rows,max_splits),128,0, \
    (float *)out->ptr,(float *)part->ptr,(const float *)q->ptr,(const float *)gate->ptr, \
    sel_tokens ? (const int *)sel_tokens->ptr : nullptr,n_sel ? (const unsigned *)n_sel->ptr : nullptr, \
    (const attn_row *)table->ptr+entry0,H,Hkv,sel_stride,scale)
    if (D == 256 && H/Hkv <= 16 && aligned && ds4_cuda_attn_tokentile_arch_ok() && !g_quality_mode &&
        min_keys >= 128 && !getenv("DS4_QWEN4_NO_ATTN_MM")) {
        launch(attention_group_rows,dim3(Hkv,n_rows,max_splits),256,0,
            (float *)out->ptr,(float *)part->ptr,(const float *)q->ptr,(const float *)gate->ptr,
            sel_tokens ? (const int *)sel_tokens->ptr : nullptr,n_sel ? (const unsigned *)n_sel->ptr : nullptr,
            (const attn_row *)table->ptr+entry0,H,Hkv,sel_stride,scale);
    } else if (D == 32) { QWEN_ATTN_ROWS(32); } else if (D == 128) { QWEN_ATTN_ROWS(128); } else { QWEN_ATTN_ROWS(256); }
#undef QWEN_ATTN_ROWS
    launch(attn_merge_rows,dim3(H,n_rows),D,0,(float *)out->ptr,(const float *)part->ptr,(const float *)gate->ptr,
        (const attn_row *)table->ptr+entry0,H,D,sel_stride);
    return launched();
}

extern "C" int ds4_gpu_qwen4_gdn_rows_stage(ds4_gpu_tensor *table, uint64_t entry0,
        const ds4_gpu_qwen4_gdn_row *rows, uint32_t n_rows) {
    using namespace qwen4_cuda;
    if (!rows || !n_rows || n_rows > ROWS_MAX) return 0;
    for (uint32_t i = 0; i < n_rows; i++) if (!rows[i].n_tok || rows[i].n_tok > 2u) return 0;
    if (!tensor(table,(entry0+n_rows)*sizeof(gdn_row))) return 0;
    row_batch<gdn_row> batch = {};
    for (unsigned i = 0; i < n_rows; i++) {
        if (!rows[i].state || !rows[i].hist) return 0;
        batch.rows[i] = {(float *)rows[i].state->ptr,(float *)rows[i].hist->ptr,
            rows[i].snap_state ? (float *)rows[i].snap_state->ptr : nullptr,
            rows[i].snap_hist ? (float *)rows[i].snap_hist->ptr : nullptr,rows[i].row0,rows[i].n_tok};
    }
    /* Kernel parameters own the host values until consumed. No pageable
     * H2D copy or staging-buffer lifetime fence is needed between layers. */
    launch(stage_rows<gdn_row>,1,128,0,(gdn_row *)table->ptr+entry0,batch,n_rows);
    return launched();
}

extern "C" int ds4_gpu_qwen4_conv_stream_rows2_tensor(
        ds4_gpu_tensor *x, const void *map, uint64_t size, uint64_t woff,
        const ds4_gpu_tensor *table, uint64_t entry0, const ds4_gpu_qwen4_gdn_row *rows, uint32_t n_rows,
        uint32_t n_batch_rows, uint32_t C, uint32_t conv_kernel, uint32_t x_stride, int apply_silu) {
    using namespace qwen4_cuda;
    if (x_stride != C) return 0;
    if (!rows || !n_rows || n_rows > ROWS_MAX || !C || conv_kernel < 2 || conv_kernel > 4 ||
        !tensor(x,(uint64_t)n_batch_rows*C*4) || !tensor(table,(entry0+n_rows)*sizeof(gdn_row))) return 0;
    const uint64_t hb = (uint64_t)(conv_kernel-1)*C*4;
    for (unsigned i = 0; i < n_rows; i++)
        if (rows[i].row0 > n_batch_rows || rows[i].n_tok > n_batch_rows-rows[i].row0 ||
            !tensor(rows[i].hist,hb) || (rows[i].snap_hist && !tensor(rows[i].snap_hist,hb))) return 0;
    const char *w = weight(map,size,woff,(uint64_t)C*conv_kernel*4);
    if (!w) return 0;
    launch(conv_rows,dim3((C+255)/256,n_rows),256,0,(float *)x->ptr,(const float *)w,
        (const gdn_row *)table->ptr+entry0,C,conv_kernel,apply_silu != 0);
    return launched();
}

extern "C" int ds4_gpu_qwen4_gdn_scan_rows2_tensor(
        ds4_gpu_tensor *out, const ds4_gpu_tensor *qkv, const ds4_gpu_tensor *ga, const ds4_gpu_tensor *gb,
        const ds4_gpu_tensor *table, uint64_t entry0, const ds4_gpu_qwen4_gdn_row *rows, uint32_t n_rows,
        uint32_t n_batch_rows, uint32_t K, uint32_t V, uint32_t D, uint32_t qkv_stride, uint32_t out_stride) {
    using namespace qwen4_cuda;
    if (!rows || !n_rows || n_rows > ROWS_MAX || !K || !V || V%K || D < 32 || D > 128 || D%32 ||
        qkv_stride != (2*K+V)*D || out_stride != V*D ||
        !tensor(out,(uint64_t)n_batch_rows*out_stride*4) || !tensor(qkv,(uint64_t)n_batch_rows*qkv_stride*4) ||
        !tensor(ga,(uint64_t)n_batch_rows*V*4) || !tensor(gb,(uint64_t)n_batch_rows*V*4) ||
        !tensor(table,(entry0+n_rows)*sizeof(gdn_row))) return 0;
    const uint64_t sb = (uint64_t)V*D*D*4;
    for (unsigned i = 0; i < n_rows; i++)
        if (rows[i].row0 > n_batch_rows || rows[i].n_tok > n_batch_rows-rows[i].row0 ||
            !tensor(rows[i].state,sb) || (rows[i].snap_state && !tensor(rows[i].snap_state,sb))) return 0;
#define QWEN_SCAN_ROWS(DIM) case DIM: launch(gdn_scan_rows<DIM>,dim3(DIM/4,V,n_rows),128,0, \
    (float *)out->ptr,(const float *)qkv->ptr,(const float *)ga->ptr,(const float *)gb->ptr, \
    (const gdn_row *)table->ptr+entry0,K,V); break
    switch (D) { QWEN_SCAN_ROWS(32); QWEN_SCAN_ROWS(64); QWEN_SCAN_ROWS(96); QWEN_SCAN_ROWS(128); }
#undef QWEN_SCAN_ROWS
    return launched();
}

extern "C" int ds4_gpu_qwen4_conv_stream_rows_tensor(
        ds4_gpu_tensor *x, ds4_gpu_tensor *hist_pool, const void *map, uint64_t size, uint64_t woff,
        const uint32_t *slots, uint32_t n_rows, uint32_t C, uint32_t conv_kernel,
        uint32_t state_stride, uint32_t x_stride, int apply_silu) {
    using namespace qwen4_cuda;
    if (x_stride != C) return 0;
    if (!slots || !n_rows || n_rows > ROWS_MAX || !C || conv_kernel < 2 || conv_kernel > 4 ||
        state_stride < (conv_kernel-1)*C || !tensor(x,(uint64_t)n_rows*C*4)) return 0;
    row_slots rs = {};
    for (unsigned i = 0; i < n_rows; i++) {
        if (!tensor(hist_pool,((uint64_t)slots[i]*state_stride+(conv_kernel-1)*C)*4)) return 0;
        rs.slot[i] = slots[i];
    }
    const char *w = weight(map,size,woff,(uint64_t)C*conv_kernel*4);
    if (!w) return 0;
    launch(conv_slots,dim3((C+255)/256,n_rows),256,0,(float *)x->ptr,(float *)hist_pool->ptr,
        (const float *)w,rs,C,conv_kernel,state_stride,apply_silu != 0);
    return launched();
}

extern "C" int ds4_gpu_qwen4_gdn_scan_rows_tensor(
        ds4_gpu_tensor *out, ds4_gpu_tensor *state_pool,
        const ds4_gpu_tensor *qkv, const ds4_gpu_tensor *ga, const ds4_gpu_tensor *gb,
        const uint32_t *slots, uint32_t n_rows, uint32_t K, uint32_t V, uint32_t D,
        uint32_t state_stride, uint32_t qkv_stride, uint32_t out_stride) {
    using namespace qwen4_cuda;
    if (!slots || !n_rows || n_rows > ROWS_MAX || !K || !V || V%K || D < 32 || D > 128 || D%32 ||
        qkv_stride != (2*K+V)*D || out_stride != V*D || state_stride < V*D*D ||
        !tensor(out,(uint64_t)n_rows*out_stride*4) || !tensor(qkv,(uint64_t)n_rows*qkv_stride*4) ||
        !tensor(ga,(uint64_t)n_rows*V*4) || !tensor(gb,(uint64_t)n_rows*V*4)) return 0;
    row_slots rs = {};
    for (unsigned i = 0; i < n_rows; i++) {
        if (!tensor(state_pool,((uint64_t)slots[i]*state_stride+V*D*D)*4)) return 0;
        rs.slot[i] = slots[i];
    }
#define QWEN_SCAN_SLOTS(DIM) case DIM: launch(gdn_scan_slots<DIM>,dim3(DIM/4,V,n_rows),128,0, \
    (float *)out->ptr,(float *)state_pool->ptr,(const float *)qkv->ptr,(const float *)ga->ptr, \
    (const float *)gb->ptr,rs,K,V,state_stride); break
    switch (D) { QWEN_SCAN_SLOTS(32); QWEN_SCAN_SLOTS(64); QWEN_SCAN_SLOTS(96); QWEN_SCAN_SLOTS(128); }
#undef QWEN_SCAN_SLOTS
    return launched();
}

/* The batch driver's Q8 tile (8 or 16 rows, padding rows included) is the
 * row-batched GEMV in chunks of eight rows on CUDA. */
extern "C" int ds4_gpu_qwen4_batch_mm_q8_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *x,
        const void *map, uint64_t size, uint64_t woff, uint32_t T, uint32_t K, uint32_t M) {
    return ds4_gpu_qwen4_dense_mm_tensor(out, x, map, size, woff, 8, T, K, M);
}
