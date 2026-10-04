// Offline HWX export of the exact emitted training MIL, using Apple's compiler.
#import <Foundation/Foundation.h>
#import <dlfcn.h>
#include <stdio.h>

typedef int (*CompileFn)(NSDictionary *, NSDictionary *, void (^)(unsigned int, NSDictionary *));
int main(int argc, char **argv) {
  @autoreleasepool {
    if (argc != 3) return 2;
    NSString *input = @(argv[1]), *output = @(argv[2]);
    NSFileManager *fm = NSFileManager.defaultManager;
    [fm createDirectoryAtPath:output withIntermediateDirectories:YES attributes:nil error:nil];
    void *library = dlopen("/System/Library/PrivateFrameworks/ANECompiler.framework/ANECompiler", RTLD_NOW);
    CompileFn compile = library ? (CompileFn)dlsym(library, "ANECCompile") : NULL;
    if (!compile) return 1;
    NSDictionary *options = @{@"InputNetworks": @[@{@"NetworkSourceFileName": @"model.mil", @"NetworkSourcePath": [input stringByAppendingString:@"/"]}],
      @"OutputFilePath": [output stringByAppendingString:@"/"], @"OutputFileName": @"model.hwx"};
    NSDictionary *flags = @{@"TargetArchitecture": @"h13g", @"DumpStatusDictionaryToFile": @YES};
    __block unsigned int status = 0;
    int result = compile(options, flags, ^(unsigned int s, NSDictionary *info) {
      status = s;
      if (s) NSLog(@"Compile status %u: %@", s, info);
    });
    if (result || status || ![fm fileExistsAtPath:[output stringByAppendingPathComponent:@"model.hwx"]]) return 1;
    return 0;
  }
}
