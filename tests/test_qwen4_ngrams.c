#define DS4_NO_GPU
#ifndef __APPLE__
#include <pthread.h>
static int test_pthread_create(pthread_t *, const pthread_attr_t *, void *(*)(void *), void *);
#define pthread_create test_pthread_create
#endif
#include "../ds4.c"
#ifndef __APPLE__
#undef pthread_create
static int thread_budget = -1;
static int test_pthread_create(pthread_t *thread, const pthread_attr_t *attr,
                               void *(*start)(void *), void *arg) {
    if (!thread_budget) return EAGAIN;
    if (thread_budget > 0) thread_budget--;
    return pthread_create(thread,attr,start,arg);
}
#endif
#include <assert.h>
#include <sys/wait.h>

static void u32(FILE *f, uint32_t n) { assert(fwrite(&n, 4, 1, f) == 1); }
static void u64(FILE *f, uint64_t n) { assert(fwrite(&n, 8, 1, f) == 1); }
static void str(FILE *f, const char *s) { u64(f, strlen(s)); assert(fwrite(s, strlen(s), 1, f) == 1); }

static uint16_t value(size_t row, size_t col) {
    const uint16_t edge[] = {0, 0x8000, 1, 0x8001, 0x007f, 0x0080, 0x3f80, 0xbf80, 0x7f7f};
    return col < sizeof(edge)/sizeof(*edge) ? edge[col] : (uint16_t)(0x3000 + (row*17+col) % 4096);
}

static void fixture(const char *path, bool bad_alignment, uint32_t type) {
    FILE *f = fopen(path, "wb");
    assert(f);
    assert(fwrite("GGUF", 4, 1, f) == 1);
    u32(f, 3); u64(f, 2); u64(f, 1);
    str(f, "general.architecture"); u32(f, 8); str(f, "qwen4exp");
    str(f, "token_embd.weight"); u32(f, 2); u64(f, 1); u64(f, 1); u32(f, 0); u64(f, 0);
    str(f, "per_layer_token_embd.weight"); u32(f, 2); u64(f, 160); u64(f, 1000); u32(f, type);
    uint64_t start = ((uint64_t)ftell(f) + 8 + 31) / 32 * 32;
    u64(f, 65536 - start + (bad_alignment ? 1 : 0));
    assert(!fseek(f, (long)start, SEEK_SET));
    u32(f, 0x3f800000);
    assert(!fseek(f, 65536, SEEK_SET));
    for (size_t r = 0; r < 1000; r++) {
        for (size_t c = 0; c < 160; c++) {
            uint16_t v = value(r, c);
            uint8_t bytes[2] = {v & 255, v >> 8};
            assert(fwrite(bytes, 2, 1, f) == 1);
        }
    }
    if (bad_alignment) fputc(0, f);
    assert(!fclose(f));
}

static uint8_t code8(size_t row, size_t col) {
    const uint8_t edge[] = {0, 0x80, 1, 0x81, 7, 8, 0x7e, 0xfe, 0x38, 0xb8};
    return col < sizeof(edge) ? edge[col] : (uint8_t)((row * 31 + col * 7) % 251);
}

static float scale8(size_t row) { return row == 5 ? 0.0f : 0x1p-12f * (float)(1 + row % 37); }

/* Independent of the runtime's bit construction. */
static float e4m3_ref(uint8_t b) {
    int e = (b >> 3) & 15, mant = b & 7;
    float v = e ? ldexpf((float)(8 + mant), e - 10) : ldexpf((float)mant, -9);
    return b & 128 ? -v : v;
}

static void fixture8(const char *path, const char *encoding, uint32_t type, uint64_t width) {
    FILE *f = fopen(path, "wb");
    assert(f);
    assert(fwrite("GGUF", 4, 1, f) == 1);
    u32(f, 3); u64(f, 2); u64(f, 2);
    str(f, "general.architecture"); u32(f, 8); str(f, "qwen4exp");
    str(f, "qwen4exp.ple.ngram_encoding"); u32(f, 8); str(f, encoding);
    str(f, "token_embd.weight"); u32(f, 2); u64(f, 1); u64(f, 1); u32(f, 0); u64(f, 0);
    str(f, "per_layer_token_embd.weight"); u32(f, 2); u64(f, width); u64(f, 1000); u32(f, type);
    uint64_t start = ((uint64_t)ftell(f) + 8 + 31) / 32 * 32;
    u64(f, 65536 - start);
    assert(!fseek(f, (long)start, SEEK_SET));
    u32(f, 0x3f800000);
    assert(!fseek(f, 65536, SEEK_SET));
    for (size_t r = 0; r < 1000; r++) {
        float scale = scale8(r);
        assert(fwrite(&scale, 4, 1, f) == 1);
        for (size_t c = 0; c < 160; c++) {
            uint8_t v = code8(r, c);
            if (!strcmp(encoding, "e4m3_f32row") && (v & 127) == 127) v ^= 1;
            assert(fwrite(&v, 1, 1, f) == 1);
        }
    }
    /* Room for the mismatched layouts, so they fail on layout rather than size. */
    assert(!ftruncate(fileno(f), 65536 + 330000));
    assert(!fclose(f));
}

static void test_8bit(const char *path, const uint32_t *rows, size_t n, float *out) {
    const char *encodings[] = {"e4m3_f32row", "i8_f32row"};
    for (int ei = 0; ei < 2; ei++) {
        const bool e4m3 = ei == 0;
        fixture8(path, encodings[ei], 24, 164);
        ds4_model m;
        model_open(&m, path, false, false);
        assert(m.size == 65536 && m.ngram_tensor && m.ngram_width == 160 && m.ngram_row_bytes == 164);
        const size_t sizes[] = {1, 16, 4097, n};
        for (size_t ni = 0; ni < sizeof(sizes)/sizeof(*sizes); ni++) {
            memset(out, 0xff, sizes[ni] * 160 * sizeof(*out));
            assert(qwen4_ngram_read(&m, rows, sizes[ni], out));
            for (size_t i = 0; i < sizes[ni]; i++) {
                for (size_t c = 0; c < 160; c++) {
                    uint8_t v = code8(rows[i], c);
                    if (e4m3 && (v & 127) == 127) v ^= 1;
                    float expected = (e4m3 ? e4m3_ref(v) : (float)(int8_t)v) * scale8(rows[i]);
                    assert(out[i*160+c] == expected);
                }
            }
        }
        int fd = open(path, O_WRONLY);
        assert(fd >= 0);
        uint32_t first = 0;
        if (e4m3) {
            uint8_t nan = 0xff;
            assert(pwrite(fd, &nan, 1, 65536 + 4 + 20) == 1);
            assert(!qwen4_ngram_read(&m, &first, 1, out) && errno == EDOM);
            nan = 0;
            assert(pwrite(fd, &nan, 1, 65536 + 4 + 20) == 1);
        }
        const float bad[] = {-1.0f, NAN, INFINITY};
        for (size_t bi = 0; bi < 3; bi++) {
            assert(pwrite(fd, &bad[bi], 4, 65536) == 4);
            assert(!qwen4_ngram_read(&m, &first, 1, out) && errno == EDOM);
        }
        close(fd);
        model_close(&m);
    }
    /* Unknown encodings, BF16 tensors with an 8-bit encoding and oversized rows. */
    for (int bad = 0; bad < 3; bad++) {
        fixture8(path, bad == 0 ? "e5m2_f32row" : "i8_f32row", bad == 1 ? 30 : 24, bad == 2 ? 165 : 164);
        pid_t pid = fork();
        assert(pid >= 0);
        ds4_model m;
        if (!pid) { model_open(&m, path, false, false); _exit(0); }
        int status;
        assert(waitpid(pid, &status, 0) == pid && WIFEXITED(status) && WEXITSTATUS(status) != 0);
    }
}

/* NVFP4 rows: ten E4M3 scales, then 80 code bytes (low nibble first), with
 * one F32 global scale in the metadata (kv: 0 none, 1 the scale, 2 zero). */
static uint8_t scale4(size_t row, size_t g) { return (uint8_t)(g == 3 ? 1 + row % 7 : 0x20 + (row * 13 + g * 5) % 0x40); }
static uint8_t code4(size_t row, size_t b) { return (uint8_t)((row * 37 + b * 11) % 256); }

static void fixture4(const char *path, int kv, uint64_t width) {
    FILE *f = fopen(path, "wb");
    assert(f);
    assert(fwrite("GGUF", 4, 1, f) == 1);
    u32(f, 3); u64(f, 2); u64(f, kv ? 3 : 2);
    str(f, "general.architecture"); u32(f, 8); str(f, "qwen4exp");
    str(f, "qwen4exp.ple.ngram_encoding"); u32(f, 8); str(f, "nvfp4_e4m3x16");
    if (kv) {
        const float g = kv == 1 ? 0x1p-7f * 1.375f : 0.0f;
        str(f, "qwen4exp.ple.ngram_global_scale"); u32(f, 6); assert(fwrite(&g, 4, 1, f) == 1);
    }
    str(f, "token_embd.weight"); u32(f, 2); u64(f, 1); u64(f, 1); u32(f, 0); u64(f, 0);
    str(f, "per_layer_token_embd.weight"); u32(f, 2); u64(f, width); u64(f, 1000); u32(f, 24);
    uint64_t start = ((uint64_t)ftell(f) + 8 + 31) / 32 * 32;
    u64(f, 65536 - start);
    assert(!fseek(f, (long)start, SEEK_SET));
    u32(f, 0x3f800000);
    assert(!fseek(f, 65536, SEEK_SET));
    for (size_t r = 0; r < 1000; r++) {
        for (size_t g = 0; g < 10; g++) { uint8_t v = scale4(r, g); assert(fwrite(&v, 1, 1, f) == 1); }
        for (size_t b = 0; b < 80; b++) { uint8_t v = code4(r, b); assert(fwrite(&v, 1, 1, f) == 1); }
    }
    assert(!ftruncate(fileno(f), 65536 + 100000));
    assert(!fclose(f));
}

static void test_nvfp4(const char *path, const uint32_t *rows, size_t n, float *out) {
    static const float e2m1[16] = {0, .5f, 1, 1.5f, 2, 3, 4, 6, -0.0f, -.5f, -1, -1.5f, -2, -3, -4, -6};
    const float gscale = 0x1p-7f * 1.375f;
    fixture4(path, 1, 90);
    ds4_model m;
    model_open(&m, path, false, false);
    assert(m.size == 65536 && m.ngram_tensor && m.ngram_width == 160 && m.ngram_row_bytes == 90 &&
           m.ngram_encoding == DS4_NGRAM_NVFP4_ROW && m.ngram_global_scale == gscale);
    const size_t sizes[] = {1, 16, 4097, n};
    for (size_t ni = 0; ni < sizeof(sizes)/sizeof(*sizes); ni++) {
        memset(out, 0xff, sizes[ni] * 160 * sizeof(*out));
        assert(qwen4_ngram_read(&m, rows, sizes[ni], out));
        for (size_t i = 0; i < sizes[ni]; i++) {
            for (size_t c = 0; c < 160; c++) {
                const uint8_t b = code4(rows[i], c / 2);
                const float scale = e4m3_ref(scale4(rows[i], c / 16)) * gscale;
                assert(out[i*160+c] == e2m1[c & 1 ? b >> 4 : b & 15] * scale);
            }
        }
    }
    int fd = open(path, O_WRONLY);
    assert(fd >= 0);
    uint32_t first = 0;
    const uint8_t bad[] = {0x7f, 0xff, 0xb8};
    for (size_t bi = 0; bi < 3; bi++) {
        assert(pwrite(fd, &bad[bi], 1, 65536 + 4) == 1);
        assert(!qwen4_ngram_read(&m, &first, 1, out) && errno == EDOM);
    }
    close(fd);
    model_close(&m);
    /* no global scale, a zero one, and rows of the wrong size */
    for (int b = 0; b < 3; b++) {
        fixture4(path, b == 0 ? 0 : b == 1 ? 2 : 1, b == 2 ? 91 : 90);
        pid_t pid = fork();
        assert(pid >= 0);
        if (!pid) { model_open(&m, path, false, false); _exit(0); }
        int status;
        assert(waitpid(pid, &status, 0) == pid && WIFEXITED(status) && WEXITSTATUS(status) != 0);
    }
}

int main(void) {
    /* the fixture goes under TMPDIR when it is set */
    const char *dir = getenv("TMPDIR");
    char path[4096];
    snprintf(path, sizeof(path), "%s/ds4-qwen-ngrams-XXXXXX", dir && dir[0] ? dir : "/tmp");
    int fd = mkstemp(path);
    assert(fd >= 0);
    close(fd);
    fixture(path, false, 30);
    ds4_model m;
    model_open(&m, path, false, false);
    assert(m.size == 65536 && m.file_size == 65536 + 320000);
    assert(m.ngram_fd >= 0 && m.ngram_tensor && m.max_tensor_bytes == 4);
    assert(*(const float *)tensor_data(&m, model_find_tensor(&m, "token_embd.weight")) == 1.0f);
    enum { N = 9001 };
    uint32_t *rows = malloc(N * sizeof(*rows));
    float *out = malloc((size_t)N * 160 * sizeof(*out));
    assert(rows && out);
    for (size_t i = 0; i < N; i++) rows[i] = (i * 173u) % 1000;
    rows[1] = rows[0]; rows[N-1] = 999;
    const size_t sizes[] = {0,16,255,256,257,4095,4096,4097,N};
    for (size_t ni = 0; ni < sizeof(sizes)/sizeof(*sizes); ni++) {
        size_t n = sizes[ni];
        assert(qwen4_ngram_read(&m, rows, n, out));
        for (size_t i = 0; i < n; i++) {
            for (size_t c = 0; c < 160; c++) {
                uint32_t bits, expected = (uint32_t)value(rows[i], c) << 16;
                memcpy(&bits, out + i*160+c, 4);
                assert(bits == expected);
            }
        }
    }
#ifndef __APPLE__
    const int budgets[] = {0,1,7};
    for (size_t bi = 0; bi < sizeof(budgets)/sizeof(*budgets); bi++) {
        qwen4_ngram_pool_free(m.ngram_pool);
        m.ngram_pool = qwen4_ngram_pool_new();
        thread_budget = budgets[bi];
        memset(out,0xff,(size_t)N*160*sizeof(*out));
        assert(qwen4_ngram_read(&m,rows,N,out));
        for (size_t i = 0; i < N; i++) for (size_t c = 0; c < 160; c++) {
            uint32_t bits;
            memcpy(&bits,out+i*160+c,4);
            assert(bits == (uint32_t)value(rows[i],c)<<16);
        }
    }
    thread_budget = -1;
#endif
    uint32_t invalid = 1000;
    assert(!qwen4_ngram_read(&m, &invalid, 1, out) && errno == EINVAL);
    assert(!qwen4_ngram_read(&m, rows, SIZE_MAX, out) && errno == EINVAL);
    assert(!qwen4_ngram_read(&m, NULL, 1, out) && errno == EINVAL);
    fd = open(path, O_WRONLY);
    assert(fd >= 0);
    uint8_t nan[2] = {0xc0, 0x7f};
    assert(pwrite(fd, nan, 2, 65536) == 2);
    uint32_t first = 0;
    assert(!qwen4_ngram_read(&m, &first, 1, out) && errno == EDOM);
    assert(!qwen4_ngram_read(&m, rows, N, out) && errno == EDOM);
    uint16_t zero = 0;
    assert(pwrite(fd,&zero,2,65536) == 2);
    assert(qwen4_ngram_read(&m,rows,N,out));
    assert(!ftruncate(fd, 65536 + 320000 - 1));
    uint32_t last = 999;
    assert(!qwen4_ngram_read(&m, &last, 1, out) && errno == EIO);
    assert(!qwen4_ngram_read(&m,rows,N,out) && errno == EIO);
    close(fd);
    model_close(&m);
    assert(m.ngram_fd == -1 && !m.ngram_tensor);
    for (int bad = 0; bad < 4; bad++) {
        fixture(path, bad == 0, bad == 1 ? 1 : bad == 2 ? 3 : 30);
        if (bad == 3) assert(!truncate(path, 65536 + 320000 - 1));
        pid_t pid = fork();
        assert(pid >= 0);
        if (!pid) { model_open(&m, path, false, false); _exit(0); }
        int status;
        assert(waitpid(pid, &status, 0) == pid && WIFEXITED(status) && WEXITSTATUS(status) != 0);
    }
    test_8bit(path, rows, N, out);
    test_nvfp4(path, rows, N, out);
    free(rows); free(out); unlink(path);
    puts("Qwen BF16, 8-bit and NVFP4 n-grams: disk-only mapping, exact reads, batches and errors OK");
    return 0;
}
