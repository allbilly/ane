// Numerical fixtures through Orion's macOS ANE runtime, from the same MIL
// and packed weights as the HWX dump. These are not Linux hardware results.
#import <Foundation/Foundation.h>
#import "ane_runtime.h"
#import "iosurface_tensor.h"
#include <math.h>

int main(int argc, char **argv) { @autoreleasepool {
    if (argc != 3) { fprintf(stderr, "usage: capture BUNDLE OUTPUT_DIR\n"); return 2; }
    NSString *bundle = @(argv[1]), *out = @(argv[2]);
    NSFileManager *fm = NSFileManager.defaultManager;
    [fm createDirectoryAtPath:out withIntermediateDirectories:YES attributes:nil error:nil];
    NSData *mil = [NSData dataWithContentsOfFile:[bundle stringByAppendingPathComponent:@"model.mil"]];
    NSData *weights = [NSData dataWithContentsOfFile:[bundle stringByAppendingPathComponent:@"weights/packed.bin"]];
    NSDictionary *attrs = [NSDictionary dictionaryWithContentsOfFile:[bundle stringByAppendingPathComponent:@"attributes.plist"]];
    if (!mil || !weights || !attrs || !orion_ane_init()) return 1;
    NSString *text = [[NSString alloc] initWithData:mil encoding:NSUTF8StringEncoding];
    OrionProgram *program = orion_compile_mil(text.UTF8String,
        @{@"@model_path/weights/packed.bin": @{@"offset": @0, @"data": weights}}, bundle.UTF8String);
    if (!program) return 1;
    NSArray *names = attrs[@"ANEFModelDescription"][@"kANEFModelOutputSymbolsArrayKey"];
    int count = (int)names.count;
    IOSurfaceRef input = orion_tensor_create(768, 32);
    IOSurfaceRef outputs[count];
    // Nonconstant values at every sequence position exercise all strides.
    // Integer-only construction makes the fp16 input reproducible on Linux.
    _Float16 *values = calloc(768 * 32, sizeof(_Float16));
    for (int c = 0; c < 768; c++) for (int s = 0; s < 32; s++)
        values[c * 32 + s] = (_Float16)(((c * 17 + s * 13) % 257 - 128) / 128.0f);
    orion_tensor_write(input, values, 768 * 32 * 2);
    [[NSData dataWithBytes:values length:768 * 32 * 2] writeToFile:[out stringByAppendingPathComponent:@"input.bin"] atomically:YES];
    for (int i = 0; i < count; i++) outputs[i] = orion_tensor_create(768, 32);
    bool ok = orion_eval(program, &input, 1, outputs, count);
    if (ok) for (int i = 0; i < count; i++) {
        orion_tensor_read(outputs[i], values, 768 * 32 * 2);
        for (int j = 0; j < 768 * 32; j++) if (!isfinite((float)values[j])) ok = false;
        NSString *name = [names[i] stringByReplacingOccurrencesOfString:@"@output" withString:@""];
        NSString *path = [out stringByAppendingPathComponent:[name stringByAppendingString:@".bin"]];
        [[NSData dataWithBytes:values length:768 * 32 * 2] writeToFile:path atomically:YES];
    }
    for (int i = 0; i < count; i++) CFRelease(outputs[i]);
    CFRelease(input); free(values); orion_release_program(program);
    fprintf(stderr, "%s: %s (%d outputs)\n", argv[1], ok ? "PASS" : "FAIL", count);
    return ok ? 0 : 1;
}}
