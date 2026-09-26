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
    /* Bicubic resize of the Qwen3.8 and GLM-5.3 preprocessing: large
     * downscales (several threads), a fixture, and enlargements. */
    fail |= !check_resizes();
    return fail;
}
