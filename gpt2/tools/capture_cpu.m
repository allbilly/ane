#import <Foundation/Foundation.h>
#import "weight_loader.h"
#import "decode_cpu.h"

int main(int argc, char **argv) { @autoreleasepool {
    if (argc < 4) return 2;
    OrionGPT2Weights *w = orion_gpt2_weights_load(argv[1]);
    if (!w) return 1;
    int count = argc - 3;
    int *tokens = calloc(count, sizeof(int));
    for (int i = 0; i < count; i++) tokens[i] = atoi(argv[i + 3]);
    float *logits = calloc(50257, sizeof(float));
    orion_gpt2_forward_cpu(w, tokens, count, logits);
    FILE *file = fopen(argv[2], "wb");
    if (!file) return 1;
    fwrite(logits, sizeof(float), 50257, file); fclose(file);
    free(tokens); free(logits); orion_gpt2_weights_free(w);
    return 0;
}}
