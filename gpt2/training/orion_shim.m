#import <Foundation/Foundation.h>
#import "core/ane_runtime.h"
#import "core/iosurface_tensor.h"
#import "compiler/graph.h"
#import "compiler/codegen.h"
#import "compiler/pipeline.h"
#import "compiler/validate.h"

// External experiment adapter. Orion sources are compiled without modifications.
int training_codegen(OrionGraph *g, const char *destination) {
  @autoreleasepool {
    OrionValidationResult result = orion_graph_validate(g);
    if (!result.valid) { fprintf(stderr, "%s\n", result.message); return -1; }
    orion_pipeline_optimize(g);
    NSString *mil = orion_codegen_mil(g, "main");
    [mil writeToFile:[@(destination) stringByAppendingString:@".raw"] atomically:YES encoding:NSUTF8StringEncoding error:nil];
    // This macOS MIL schema requires rsqrt's epsilon argument. Upstream
    // Orion's generic emitter omits it. Preserve the original beside the MIL.
    mil = [mil stringByReplacingOccurrencesOfString:@" = rsqrt(x="
      withString:@" = rsqrt(epsilon=fp16(0x0.0p+0), x="];
    return [mil writeToFile:@(destination) atomically:YES encoding:NSUTF8StringEncoding error:nil] ? 0 : -2;
  }
}

void *training_open(const char *source, const char *destination) {
  @autoreleasepool {
    if (!orion_ane_init()) return NULL;
    NSString *mil = [NSString stringWithContentsOfFile:@(source) encoding:NSUTF8StringEncoding error:nil];
    OrionProgram *p = orion_compile_mil(mil.UTF8String, @{}, source);
    if (!p) return NULL;
    // The first two fields are documented in Orion core/ane_runtime.m.
    struct ProgramPrefix { void *model; void *tmpDir; };
    NSString *tmp = (__bridge NSString *)((struct ProgramPrefix *)p)->tmpDir;
    NSFileManager *fm = [NSFileManager defaultManager];
    [fm createDirectoryAtPath:@(destination) withIntermediateDirectories:YES attributes:nil error:nil];
    for (NSString *entry in [fm contentsOfDirectoryAtPath:tmp error:nil]) {
      NSString *target = [@(destination) stringByAppendingPathComponent:entry];
      if (![fm fileExistsAtPath:target])
        [fm copyItemAtPath:[tmp stringByAppendingPathComponent:entry] toPath:target error:nil];
    }
    return p;
  }
}
