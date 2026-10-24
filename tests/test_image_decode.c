#include "ds4_image.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int hex_fingerprint(const uint8_t *fp, char *out, size_t cap) {
    if (cap < 65) return 0;
    for (int i = 0; i < 32; i++)
        snprintf(out + i * 2, cap - (size_t)i * 2, "%02x", fp[i]);
    return 1;
}

static int check_image(const char *name, const ds4_image *image,
                       uint32_t width, uint32_t height, const char *expected_fp) {
    char got[65] = {0};
    int ok = hex_fingerprint(image->fingerprint, got, sizeof(got)) &&
             image->width == width && image->height == height &&
             strcmp(got, expected_fp) == 0;
    if (!ok) {
        fprintf(stderr, "%s: got %ux%u fp=%s, expected %ux%u fp=%s\n",
                name, image->width, image->height, got,
                width, height, expected_fp);
    }
    return ok;
}

static int check_file(const char *path, uint32_t width, uint32_t height,
                      const char *expected_fp) {
    ds4_image image = {0};
    char error[160] = {0};
    if (!ds4_image_decode_file(&image, path, error, sizeof(error))) {
        fprintf(stderr, "decode failed for %s: %s\n", path, error);
        return 0;
    }
    int ok = check_image(path, &image, width, height, expected_fp);
    ds4_image_free(&image);
    return ok;
}

/* PNG files built here with stored (uncompressed) deflate blocks, and the
 * plain bitwise CRC-32 and per-byte Adler-32, as independent references for
 * the decoder's table CRC and deferred Adler-32. */
typedef struct {
    uint8_t *p;
    size_t n, cap;
} test_buf;

static void buf_put(test_buf *b, const void *data, size_t n) {
    if (b->n + n > b->cap) {
        b->cap = (b->n + n) * 2;
        b->p = realloc(b->p, b->cap);
        if (!b->p) { perror("realloc"); exit(1); }
    }
    memcpy(b->p + b->n, data, n);
    b->n += n;
}

static void buf_be32(test_buf *b, uint32_t v) {
    uint8_t x[4] = {(uint8_t)(v >> 24), (uint8_t)(v >> 16), (uint8_t)(v >> 8), (uint8_t)v};
    buf_put(b, x, 4);
}

static uint32_t crc_bitwise(uint32_t c, const uint8_t *p, size_t n) {
    for (size_t i = 0; i < n; i++) {
        c ^= p[i];
        for (int bit = 0; bit < 8; bit++) c = (c & 1) ? 0xedb88320u ^ (c >> 1) : c >> 1;
    }
    return c;
}

static uint32_t adler_bytewise(const uint8_t *p, size_t n) {
    uint32_t a = 1, b = 0;
    for (size_t i = 0; i < n; i++) {
        a = (a + p[i]) % 65521;
        b = (b + a) % 65521;
    }
    return (b << 16) | a;
}

static void png_chunk(test_buf *b, const char *type, const uint8_t *data, size_t n) {
    buf_be32(b, (uint32_t)n);
    buf_put(b, type, 4);
    if (n) buf_put(b, data, n);
    uint32_t c = crc_bitwise(0xffffffffu, (const uint8_t *)type, 4);
    buf_be32(b, crc_bitwise(c, data, n) ^ 0xffffffffu);
}

/* An 8-bit PNG of colour type ctype with every row filter type in turn
 * (the bytes after the filter byte are arbitrary, so the decoder's
 * unfiltering is covered too), its zlib stream split into IDAT chunks of
 * idat_bytes. */
static test_buf stored_png(uint32_t w, uint32_t h, uint8_t ctype, uint32_t channels,
                           size_t idat_bytes) {
    test_buf raw = {0}, z = {0}, png = {0};
    uint32_t s = 12345u + ctype;
    for (uint32_t y = 0; y < h; y++) {
        uint8_t filter = (uint8_t)(y % 5);
        buf_put(&raw, &filter, 1);
        for (uint32_t x = 0; x < w; x++) {
            for (uint32_t c = 0; c < channels; c++) {
                s = s * 1664525u + 1013904223u;
                uint8_t v = (uint8_t)(x * 3 + y * 5 + c * 40 + (s >> 29));
                buf_put(&raw, &v, 1);
            }
        }
    }
    const uint8_t zhead[2] = {0x78, 0x01};
    buf_put(&z, zhead, 2);
    for (size_t off = 0; off < raw.n;) {
        size_t n = raw.n - off < 65535 ? raw.n - off : 65535;
        uint8_t hdr[5] = {(uint8_t)(off + n == raw.n), (uint8_t)n, (uint8_t)(n >> 8),
                          (uint8_t)~n, (uint8_t)(~n >> 8)};
        buf_put(&z, hdr, 5);
        buf_put(&z, raw.p + off, n);
        off += n;
    }
    buf_be32(&z, adler_bytewise(raw.p, raw.n));

    buf_put(&png, "\x89PNG\r\n\x1a\n", 8);
    uint8_t ihdr[13] = {(uint8_t)(w >> 24), (uint8_t)(w >> 16), (uint8_t)(w >> 8), (uint8_t)w,
                        (uint8_t)(h >> 24), (uint8_t)(h >> 16), (uint8_t)(h >> 8), (uint8_t)h,
                        8, ctype, 0, 0, 0};
    png_chunk(&png, "IHDR", ihdr, sizeof(ihdr));
    if (ctype == 3) {
        uint8_t plte[256 * 3];
        for (int i = 0; i < 256 * 3; i++) plte[i] = (uint8_t)(i * 37 + 11);
        png_chunk(&png, "PLTE", plte, sizeof(plte));
    }
    for (size_t off = 0; off < z.n; off += idat_bytes)
        png_chunk(&png, "IDAT", z.p + off, z.n - off < idat_bytes ? z.n - off : idat_bytes);
    png_chunk(&png, "IEND", NULL, 0);
    free(raw.p);
    free(z.p);
    return png;
}

static int check_stored_png(const char *name, uint32_t w, uint32_t h, uint8_t ctype,
                            uint32_t channels, const char *expected_fp) {
    test_buf png = stored_png(w, h, ctype, channels, 5000);
    ds4_image image = {0};
    char error[160] = {0};
    int ok = ds4_image_decode_memory(&image, png.p, png.n, error, sizeof(error));
    if (!ok) fprintf(stderr, "%s: decode failed: %s\n", name, error);
    else ok = check_image(name, &image, w, h, expected_fp);
    ds4_image_free(&image);

    /* A flipped bit in the first IDAT chunk's CRC, or in the zlib
     * stream's Adler-32 (with its chunk CRC fixed up), must be rejected. */
    size_t idat = 8 + 25 + (ctype == 3 ? 12 + 256 * 3 : 0);
    uint32_t idat_len = ((uint32_t)png.p[idat] << 24) | ((uint32_t)png.p[idat + 1] << 16) |
                        ((uint32_t)png.p[idat + 2] << 8) | png.p[idat + 3];
    png.p[idat + 8 + idat_len] ^= 0x01;
    if (ok && ds4_image_decode_memory(&image, png.p, png.n, error, sizeof(error))) {
        fprintf(stderr, "%s: a corrupted chunk CRC was accepted\n", name);
        ds4_image_free(&image);
        ok = 0;
    }
    png.p[idat + 8 + idat_len] ^= 0x01;

    size_t iend = png.n - 12, last = 0;
    for (size_t pos = 8; pos < iend;) {
        uint32_t n = ((uint32_t)png.p[pos] << 24) | ((uint32_t)png.p[pos + 1] << 16) |
                     ((uint32_t)png.p[pos + 2] << 8) | png.p[pos + 3];
        last = pos;
        pos += 12 + n;
    }
    uint32_t last_len = ((uint32_t)png.p[last] << 24) | ((uint32_t)png.p[last + 1] << 16) |
                        ((uint32_t)png.p[last + 2] << 8) | png.p[last + 3];
    png.p[last + 8 + last_len - 1] ^= 0x01;
    uint32_t c = crc_bitwise(0xffffffffu, png.p + last + 4, 4 + last_len) ^ 0xffffffffu;
    uint8_t *crc = png.p + last + 8 + last_len;
    crc[0] = (uint8_t)(c >> 24); crc[1] = (uint8_t)(c >> 16); crc[2] = (uint8_t)(c >> 8); crc[3] = (uint8_t)c;
    if (ok && ds4_image_decode_memory(&image, png.p, png.n, error, sizeof(error))) {
        fprintf(stderr, "%s: a corrupted Adler-32 was accepted\n", name);
        ds4_image_free(&image);
        ok = 0;
    }
    free(png.p);
    return ok;
}

/* The resized pixels of a preprocessing, recovered as whole levels from the
 * normalized patches (the resize rounds to whole levels), hashed with the
 * output geometry (FNV-1a 64). */
static uint64_t fnv1a(uint64_t h, const void *data, size_t n) {
    const uint8_t *p = data;
    for (size_t i = 0; i < n; i++) { h ^= p[i]; h *= 1099511628211ull; }
    return h;
}

static int check_resize(const char *name, const ds4_image *image, int glm,
                        uint32_t min_tokens, uint32_t max_tokens, uint64_t expected) {
    static const float glm_mean[3] = {0.48145466f, 0.4578275f, 0.40821073f};
    static const float glm_std[3] = {0.26862954f, 0.26130258f, 0.27577711f};
    ds4_image_patches patches;
    char error[160] = {0};
    int ok = glm ? ds4_image_preprocess_glm53(&patches, image, min_tokens, max_tokens, error, sizeof(error))
                 : ds4_image_preprocess_qwen4(&patches, image, min_tokens, max_tokens, error, sizeof(error));
    if (!ok) {
        fprintf(stderr, "%s: preprocessing failed: %s\n", name, error);
        return 0;
    }
    const size_t per_channel = glm ? 2 * 14 * 14 : 16 * 16;
    const size_t values = (size_t)patches.patch_count * 3 * per_channel;
    const uint32_t geometry[5] = {patches.content_width, patches.content_height,
                                  patches.padded_width, patches.padded_height, patches.patch_count};
    uint64_t h = fnv1a(1469598103934665603ull, geometry, sizeof(geometry));
    for (size_t i = 0; i < values && ok; i++) {
        const unsigned c = (unsigned)(i / per_channel % 3);
        const float x = patches.patches[i];
        const float v = glm ? (x * glm_std[c] + glm_mean[c]) * 255.0f : (x * 0.5f + 0.5f) * 255.0f;
        const long level = lrintf(v);
        if (level < 0 || level > 255 || fabsf(v - (float)level) > 0.01f) {
            fprintf(stderr, "%s: value %zu (%g) is not a whole level\n", name, i, (double)v);
            ok = 0;
        }
        const uint8_t byte = (uint8_t)level;
        h = fnv1a(h, &byte, 1);
    }
    ds4_image_patches_free(&patches);
    if (ok && h != expected) {
        fprintf(stderr, "%s: resized pixels hash %016llx, expected %016llx\n",
                name, (unsigned long long)h, (unsigned long long)expected);
        ok = 0;
    }
    return ok;
}

static int synthetic_image(ds4_image *image, uint32_t width, uint32_t height) {
    image->width = width;
    image->height = height;
    image->rgb = malloc((size_t)width * height * 3);
    if (!image->rgb) return 0;
    uint32_t s = 1u;
    for (uint32_t y = 0; y < height; y++) {
        for (uint32_t x = 0; x < width; x++) {
            uint8_t *p = image->rgb + ((size_t)y * width + x) * 3;
            for (unsigned c = 0; c < 3; c++) {
                s = s * 1664525u + 1013904223u;
                p[c] = (uint8_t)((x * 255u / width) + (y * 97u / height) + c * 60u + (s >> 27));
            }
        }
    }
    return 1;
}

static int check_resizes(void) {
    ds4_image big = {0}, small = {0}, maple = {0};
    char error[160] = {0};
    int ok = synthetic_image(&big, 4000, 3000) && synthetic_image(&small, 33, 17) &&
             ds4_image_decode_file(&maple, "tests/vision-fixtures/qwen38/maple.png", error, sizeof(error));
    if (!ok) fprintf(stderr, "resize inputs: %s\n", error[0] ? error : "out of memory");
    if (ok) {
        ok &= check_resize("qwen 4000x3000 -> 1024 tokens", &big, 0, 64, 1024,
                           0x379e0ed18720ff48ull);
        ok &= check_resize("qwen 4000x3000 -> 256 tokens", &big, 0, 64, 256,
                           0x5716f7068cd6139eull);
        ok &= check_resize("glm 4000x3000 -> 1024 tokens", &big, 1, 64, 1024,
                           0x9688fde31f4cd246ull);
        ok &= check_resize("qwen maple -> 64 tokens", &maple, 0, 64, 64,
                           0x006948d979055375ull);
        ok &= check_resize("qwen maple -> 1024 tokens", &maple, 0, 1024, 1024,
                           0x62cfa8d053f702dbull);
        ok &= check_resize("qwen 33x17 -> 64 tokens", &small, 0, 64, 1024,
                           0x2e78ecb6ab510f6dull);
        ok &= check_resize("glm 33x17 upscaled", &small, 1, 64, 1024,
                           0x21846b10db40ed2eull);
    }
    ds4_image_free(&big);
    ds4_image_free(&small);
    ds4_image_free(&maple);
    return ok;
}

int main(void) {
    int fail = 0;
    /* 24x16 grayscale progressive JPEG. Unpatched Iris yields different
     * pixels; patched decode matches libjpeg-turbo djpeg bit-exactly. */
    fail |= !check_file("tests/vision-fixtures/jpeg/prog_ac_refine_zrl_gray.jpg",
                        24u, 16u,
                        "63aae8863e829170c5abc746c90a5ff3ad60e9c4312e2ffa5eefbfeaacd26bfc");
    /* 64x48 4:2:0 progressive JPEG. Unpatched jpeg_load returns NULL.
     * After the ZRL fix plus main's chroma interpolation, decode matches
     * libjpeg-turbo djpeg bit-exactly. */
    fail |= !check_file("tests/vision-fixtures/jpeg/prog_ac_refine_zrl_420.jpg",
                        64u, 48u,
                        "d3be4d7078c41b6589942c82bd622ca8a3ed40adddee11cccf1de9ca1a096ba4");
    /* The fixture PNGs (zlib-compressed, dynamic Huffman) and stored PNGs of
     * every colour type: pixel fingerprints as the bitwise-CRC decoder gave
     * them, and corrupted checksums still rejected. */
    fail |= !check_file("tests/vision-fixtures/glm53/diagram.png", 1200u, 700u,
                        "578b4eec9d5a736c7f840abd34d7114613f51fc124a093eca273ed6e87212f2b");
    fail |= !check_file("tests/vision-fixtures/glm53/screenshot.png", 1400u, 900u,
                        "206eccac40cb6cfe5fbf19398634b5245887b8f44140ea4007fc7e88e54e70b0");
    fail |= !check_file("tests/vision-fixtures/glm53/spatial.png", 1000u, 800u,
                        "9d01968645deea377789b91702e71594fef22fc4a5fd1b3f94c0a4dbac0505c0");
    fail |= !check_file("tests/vision-fixtures/glm53/text.png", 1000u, 800u,
                        "958463719db28f6ddcce69b3b2172c6cfd6373049032f71fba4c053e8c026368");
    fail |= !check_file("tests/vision-fixtures/qwen38/maple.png", 640u, 480u,
                        "6a38295aca22784b852087ae59d62f9ad89dca6d177541c9a51fda67825d3250");
    fail |= !check_file("tests/vision-fixtures/qwen38/orbit.png", 640u, 480u,
                        "122c48d3f42d0bd7d05264b4c513a2fc0a33fcc05aa8a90a69ce05e5ed11e041");
    fail |= !check_stored_png("stored gray", 257u, 131u, 0, 1,
                        "45611f444bdc6fd95663eed4877dfb394f5d9b327228a047953b600a5ba0bd9e");
    fail |= !check_stored_png("stored gray+alpha", 64u, 65u, 4, 2,
                        "3f695ffee9ff217c9f9238fa9abaf959ecfa417b34f91b7cf77e2aaefa38ebc5");
    fail |= !check_stored_png("stored RGB", 333u, 217u, 2, 3,
                        "8cfcf905529b8db13213558182b3ac88e2121c4f62c35de1abd30293d6969bf1");
    fail |= !check_stored_png("stored palette", 200u, 150u, 3, 1,
                        "fe4fb32ea8e630fcc3e3090065194c17f254810e88d86d1169f73ab57bb11a15");
    fail |= !check_stored_png("stored RGBA", 129u, 77u, 6, 4,
                        "67d9da54d4efbc63cd368eb92a5c14b0cf4f36bf334a4f20be028c293981fc22");
    /* Bicubic resize of the Qwen3.8 and GLM-5.3 preprocessing: large
     * downscales (several threads), a fixture, and enlargements. */
    fail |= !check_resizes();
    return fail;
}
