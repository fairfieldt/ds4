/* Dump the Qwen3.8 vision tower output for one image; tests/qwen4_vision_ref.py
 * compares it against the HF vision model */
#include "ds4.h"

#include <stdio.h>
#include <stdlib.h>

/* ds4.c: writes u32 tokens, dim, grid rows, grid cols, then the merged embeddings as floats */
int ds4_qwen4_vision_dump(const char *vision_path, const char *image_path, const char *out_path,
                          uint32_t min_image_tokens, uint32_t max_image_tokens, uint32_t repeats);

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s mmproj.gguf image out.bin [min_tokens max_tokens [repeats]]\n", argv[0]);
        return 2;
    }
    const uint32_t min_tokens = argc > 4 ? (uint32_t)atoi(argv[4]) : 64u;
    const uint32_t max_tokens = argc > 5 ? (uint32_t)atoi(argv[5]) : 1024u;
    /* repeats > 1 prints the time of every encode (the first includes setup) */
    const uint32_t repeats = argc > 6 ? (uint32_t)atoi(argv[6]) : 1u;
    return ds4_qwen4_vision_dump(argv[1], argv[2], argv[3], min_tokens, max_tokens, repeats) ? 0 : 1;
}
