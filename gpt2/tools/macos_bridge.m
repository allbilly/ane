// Verification-only adapter: exercise the Python generation flow using Orion
// kernels on macOS. The Linux runtime does not import or compile this file.
#import <Foundation/Foundation.h>
#import "ane_runtime.h"
#import "iosurface_tensor.h"

typedef struct {
    OrionProgram *program;
    IOSurfaceRef input;
    IOSurfaceRef outputs[3];
    int count;
} Bridge;

void *gpt2_mac_open(const char *path) { @autoreleasepool {
    NSString *bundle = @(path);
    NSData *mil = [NSData dataWithContentsOfFile:[bundle stringByAppendingPathComponent:@"model.mil"]];
    NSData *weights = [NSData dataWithContentsOfFile:[bundle stringByAppendingPathComponent:@"weights/packed.bin"]];
    NSDictionary *attrs = [NSDictionary dictionaryWithContentsOfFile:[bundle stringByAppendingPathComponent:@"attributes.plist"]];
    if (!mil || !weights || !attrs || !orion_ane_init()) return NULL;
    int count = (int)[attrs[@"ANEFModelDescription"][@"kANEFModelOutputSymbolsArrayKey"] count];
    if (count != 1 && count != 3) return NULL;
    NSString *text = [[NSString alloc] initWithData:mil encoding:NSUTF8StringEncoding];
    OrionProgram *program = orion_compile_mil(text.UTF8String,
        @{@"@model_path/weights/packed.bin": @{@"offset": @0, @"data": weights}}, path);
    if (!program) return NULL;
    Bridge *bridge = calloc(1, sizeof(Bridge));
    bridge->program = program; bridge->count = count;
    bridge->input = orion_tensor_create(768, 32);
    for (int i = 0; i < count; i++) bridge->outputs[i] = orion_tensor_create(768, 32);
    return bridge;
}}

int gpt2_mac_eval(void *handle, const void *input, void *output) { @autoreleasepool {
    Bridge *bridge = handle;
    orion_tensor_write(bridge->input, input, 768 * 32 * 2);
    if (!orion_eval(bridge->program, &bridge->input, 1, bridge->outputs, bridge->count)) return 1;
    for (int i = 0; i < bridge->count; i++)
        orion_tensor_read(bridge->outputs[i], (char *)output + i * 768 * 32 * 2, 768 * 32 * 2);
    return 0;
}}

void gpt2_mac_close(void *handle) { @autoreleasepool {
    Bridge *bridge = handle;
    for (int i = 0; i < bridge->count; i++) CFRelease(bridge->outputs[i]);
    CFRelease(bridge->input); orion_release_program(bridge->program); free(bridge);
}}
