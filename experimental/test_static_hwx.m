// Numerical test of a standalone two-input FP16 MUL through _ANEClient.
#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <objc/message.h>
#include <dlfcn.h>
#include <math.h>
#include <string.h>

static id jsonValue(id value) {
    if ([value isKindOfClass:NSDictionary.class]) {
        NSMutableDictionary *result = [NSMutableDictionary dictionary];
        for (id key in value) result[[key description]] = jsonValue(value[key]);
        return result;
    }
    if ([value isKindOfClass:NSArray.class]) {
        NSMutableArray *result = [NSMutableArray array];
        for (id child in value) [result addObject:jsonValue(child)];
        return result;
    }
    if ([value isKindOfClass:NSString.class] || [value isKindOfClass:NSNumber.class] || value == NSNull.null) return value;
    return value ? [value description] : NSNull.null;
}

int main(int argc, char **argv) { @autoreleasepool {
    if (argc != 3 && argc != 4) return 2;
    BOOL loadOnly = argc == 4 && strcmp(argv[3], "--load-only") == 0;
    if (argc == 4 && !loadOnly && strcmp(argv[3], "pattern") != 0) return 2;
    BOOL pattern = argc == 4 && !loadOnly;
    NSString *path = @(argv[1]), *directory = @(argv[2]);
    [NSFileManager.defaultManager createDirectoryAtPath:directory withIntermediateDirectories:YES attributes:nil error:nil];
    NSMutableDictionary *report = [@{@"hwx": path, @"status": @"failed"} mutableCopy];
    void *framework = dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW);
    report[@"framework_loaded"] = @(framework != NULL);
    if (!framework) { const char *failure = dlerror(); report[@"framework_error"] = failure ? @(failure) : @"unknown dlopen error"; }
    Class clientClass = NSClassFromString(@"_ANEClient"), modelClass = NSClassFromString(@"_ANEModel");
    report[@"client_class_available"] = @(clientClass != Nil);
    report[@"model_class_available"] = @(modelClass != Nil);
    NSDictionary *environment = NSProcessInfo.processInfo.environment;
    NSString *modelKey = environment[@"ANE_STATIC_MODEL_KEY"] ?: @"ane_static_mul_check";
    unsigned qos = environment[@"ANE_STATIC_QOS"] ? [environment[@"ANE_STATIC_QOS"] intValue] : 21;
    NSDictionary *loadOptions = @{@"kANEFModelIdentityStrKey": @"ane_static_mul_check", @"kANEFModelType": @"kANEFModelPreCompiled"};
    if (environment[@"ANE_STATIC_LOAD_OPTIONS"]) {
        NSData *optionsData = [NSData dataWithContentsOfFile:environment[@"ANE_STATIC_LOAD_OPTIONS"]];
        id parsed = optionsData ? [NSJSONSerialization JSONObjectWithData:optionsData options:0 error:nil] : nil;
        if (![parsed isKindOfClass:NSDictionary.class]) {
            fprintf(stderr, "ANE_STATIC_LOAD_OPTIONS must name a JSON dictionary\n");
            return 2;
        }
        loadOptions = parsed;
    }
    report[@"model_key"] = modelKey; report[@"qos"] = @(qos); report[@"load_options"] = loadOptions;
    id client = ((id(*)(id,SEL))objc_msgSend)(clientClass, sel_registerName("sharedConnection"));
    id model = ((id(*)(id,SEL,id,id))objc_msgSend)(modelClass, sel_registerName("modelAtURL:key:"), [NSURL fileURLWithPath:path], modelKey);
    report[@"client_available"] = @(client != nil);
    report[@"model_available"] = @(model != nil);
    NSError *error = nil;
    if (environment[@"ANE_STATIC_COMPILE_TYPE"]) {
        NSDictionary *compileOptions = @{@"kANEFModelType": environment[@"ANE_STATIC_COMPILE_TYPE"]};
        BOOL compiled = ((BOOL(*)(id,SEL,id,id,unsigned,NSError**))objc_msgSend)(client,
            sel_registerName("compileModel:options:qos:error:"), model, compileOptions, qos, &error);
        report[@"compile_options"] = compileOptions; report[@"compiled"] = @(compiled);
        report[@"compile_error"] = error ? error.description : NSNull.null;
        error = nil;
    }
    BOOL compileSucceeded = !environment[@"ANE_STATIC_COMPILE_TYPE"] || [report[@"compiled"] boolValue];
    BOOL loaded = compileSucceeded && ((BOOL(*)(id,SEL,id,id,unsigned,NSError**))objc_msgSend)(client,
        sel_registerName("loadModel:options:qos:error:"), model,
        loadOptions, qos, &error);
    report[@"loaded"] = @(loaded); report[@"load_error"] = error ? error.description : NSNull.null;
    if (error) report[@"load_error_details"] = @{@"domain": error.domain, @"code": @(error.code), @"user_info": jsonValue(error.userInfo)};
    NSDictionary *attrs = ((id(*)(id,SEL))objc_msgSend)(model, sel_registerName("modelAttributes"));
    report[@"attributes"] = jsonValue(attrs);
    for (NSString *getter in @[@"modelURL", @"sourceURL", @"cacheURLIdentifier"]) {
        SEL selector = NSSelectorFromString(getter);
        if ([model respondsToSelector:selector])
            report[getter] = jsonValue(((id(*)(id,SEL))objc_msgSend)(model, selector));
    }
    NSMutableArray *surfaces = [NSMutableArray array];
    BOOL passed = NO;
    if (loaded && !loadOnly) {
        NSDictionary *description = attrs[@"ANEFModelDescription"];
        NSDictionary *procedure = [description[@"ANEFModelProcedures"] firstObject];
        NSInteger index = [procedure[@"ANEFModelProcedureID"] integerValue];
        NSDictionary *network = attrs[@"NetworkStatusList"][index];
        NSArray *inputSymbols = description[@"kANEFModelInputSymbolsArrayKey"];
        NSArray *outputSymbols = description[@"kANEFModelOutputSymbolsArrayKey"];
        NSMutableArray *inputs = [NSMutableArray array], *outputs = [NSMutableArray array];
        NSMutableArray *logicalOutputs = [NSMutableArray array];
        for (int role = 0; role < 2; role++) {
            NSArray *symbols = role ? outputSymbols : inputSymbols;
            NSArray *live = role ? network[@"LiveOutputList"] : [network[@"LiveInputList"] arrayByAddingObjectsFromArray:network[@"LiveInputParamList"] ?: @[]];
            for (NSUInteger i = 0; i < symbols.count; i++) {
                NSDictionary *port = nil;
                for (NSDictionary *item in live) if ([item[@"Name"] isEqual:symbols[i]] || [item[@"Symbol"] isEqual:symbols[i]]) { port = item; break; }
                if (![port[@"Type"] isEqual:@"Float16"] || [port[@"Batches"] integerValue] != 1 ||
                    [port[@"Depth"] integerValue] != 1 || [port[@"Interleave"] integerValue] != 1 ||
                    [port[@"Channels"] integerValue] * [port[@"Height"] integerValue] * [port[@"Width"] integerValue] != 64) {
                    report[@"port_error"] = @"expected a single-batch, depth-one 64-value Float16 MUL port";
                    break;
                }
                NSUInteger bytes = [port[@"Batches"] unsignedIntegerValue] * [port[@"BatchStride"] unsignedIntegerValue];
                if (!bytes) bytes = [port[@"Channels"] unsignedIntegerValue] * [port[@"Height"] unsignedIntegerValue] * [port[@"Width"] unsignedIntegerValue] * 2;
                if (!bytes) bytes = 4096;
                NSDictionary *properties = @{(id)kIOSurfaceWidth: @(bytes), (id)kIOSurfaceHeight: @1,
                    (id)kIOSurfaceBytesPerElement: @1, (id)kIOSurfaceBytesPerRow: @(bytes),
                    (id)kIOSurfaceAllocSize: @(bytes), (id)kIOSurfacePixelFormat: @0};
                IOSurfaceRef surface = IOSurfaceCreate((__bridge CFDictionaryRef)properties);
                if (!surface) break;
                [surfaces addObject:(__bridge id)surface];
                IOSurfaceLock(surface, 0, NULL);
                _Float16 *values = IOSurfaceGetBaseAddress(surface);
                for (NSUInteger j = 0; j < bytes / 2; j++) values[j] = role ? (_Float16)NAN : (_Float16)(i ? 3.f : 2.f);
                if (!role && pattern) {
                    NSUInteger cmax = MAX(1, [port[@"Channels"] unsignedIntegerValue]);
                    NSUInteger hmax = MAX(1, [port[@"Height"] unsignedIntegerValue]), wmax = MAX(1, [port[@"Width"] unsignedIntegerValue]);
                    NSUInteger row = [port[@"RowStride"] unsignedIntegerValue], plane = [port[@"PlaneStride"] unsignedIntegerValue], logical = 0;
                    if (!row) row = wmax * 2; if (!plane) plane = hmax * row;
                    for (NSUInteger c = 0; c < cmax; c++) for (NSUInteger h = 0; h < hmax; h++) for (NSUInteger w = 0; w < wmax; w++, logical++) {
                        NSUInteger offset = (c * plane + h * row) / 2 + w;
                        if (offset * 2 < bytes) values[offset] = (_Float16)(i ? (logical % 11 + 1) / 4.f : ((int)(logical % 17) - 8) / 8.f);
                    }
                }
                if (!role) [[NSData dataWithBytes:values length:bytes] writeToFile:[directory stringByAppendingPathComponent:[NSString stringWithFormat:@"input%lu.f16", (unsigned long)i]] atomically:YES];
                IOSurfaceUnlock(surface, 0, NULL);
                id object = ((id(*)(id,SEL,IOSurfaceRef))objc_msgSend)(NSClassFromString(@"_ANEIOSurfaceObject"), sel_registerName("objectWithIOSurface:"), surface);
                [(role ? outputs : inputs) addObject:object];
                if (role) [logicalOutputs addObject:@{@"surface": (__bridge id)surface, @"bytes": @(bytes), @"port": port}];
                CFRelease(surface);
            }
        }
        report[@"input_count"] = @(inputs.count); report[@"output_count"] = @(outputs.count);
        if (inputs.count == 2 && outputs.count == 1) {
            id request = ((id(*)(id,SEL,id,id,id,id,id,id))objc_msgSend)(NSClassFromString(@"_ANERequest"),
                sel_registerName("requestWithInputs:inputIndices:outputs:outputIndices:perfStats:procedureIndex:"),
                inputs, procedure[@"ANEFModelInputSymbolIndexArray"], outputs, procedure[@"ANEFModelOutputSymbolIndexArray"], @[], @(index));
            error = nil;
            BOOL executed = ((BOOL(*)(id,SEL,id,id,id,unsigned,NSError**))objc_msgSend)(client,
                sel_registerName("evaluateWithModel:options:request:qos:error:"), model,
                @{@"kANEFDisableIOFencesUseSharedEventsKey": @0}, request, qos, &error);
            report[@"executed"] = @(executed); report[@"evaluate_error"] = error ? error.description : NSNull.null;
            if (executed) {
                NSDictionary *entry = logicalOutputs[0], *port = entry[@"port"];
                IOSurfaceRef surface = (__bridge IOSurfaceRef)entry[@"surface"];
                IOSurfaceLock(surface, kIOSurfaceLockReadOnly, NULL);
                const _Float16 *data = IOSurfaceGetBaseAddress(surface);
                NSMutableArray *values = [NSMutableArray array];
                NSUInteger channels = MAX(1, [port[@"Channels"] unsignedIntegerValue]);
                NSUInteger height = MAX(1, [port[@"Height"] unsignedIntegerValue]), width = MAX(1, [port[@"Width"] unsignedIntegerValue]);
                NSUInteger row = [port[@"RowStride"] unsignedIntegerValue], plane = [port[@"PlaneStride"] unsignedIntegerValue];
                if (!row) row = width * 2; if (!plane) plane = height * row;
                passed = YES;
                NSUInteger logical = 0;
                for (NSUInteger c = 0; c < channels; c++) for (NSUInteger h = 0; h < height; h++) for (NSUInteger w = 0; w < width; w++, logical++) {
                    NSUInteger offset = (c * plane + h * row) / 2 + w;
                    if (offset * 2 >= [entry[@"bytes"] unsignedIntegerValue]) { passed = NO; continue; }
                    float value = data[offset];
                    [values addObject:isfinite(value) ? @(value) : NSNull.null];
                    float expected = pattern ? (((int)(logical % 17) - 8) / 8.f) * ((logical % 11 + 1) / 4.f) : 6.f;
                    if (value != expected) passed = NO;
                }
                report[@"input_case"] = pattern ? @"nonuniform signed fractional inputs" : @"constant 2 times 3";
                report[@"output_values"] = values;
                passed = passed && values.count == 64;
                [[NSData dataWithBytes:data length:[entry[@"bytes"] unsignedIntegerValue]] writeToFile:[directory stringByAppendingPathComponent:@"output.f16"] atomically:YES];
                IOSurfaceUnlock(surface, kIOSurfaceLockReadOnly, NULL);
            }
        }
    }
    if (loaded) {
        SEL unload = sel_registerName("unloadModel:options:qos:error:");
        if ([client respondsToSelector:unload])
            ((BOOL(*)(id,SEL,id,id,unsigned,NSError**))objc_msgSend)(client, unload, model, @{}, qos, &error);
    }
    if (loadOnly) report[@"scope"] = @"Model-load diagnostic only; no inference or numerical verification";
    report[@"status"] = loadOnly ? (loaded ? @"load_pass" : @"failed") : (passed ? @"pass" : @"failed");
    NSData *json = [NSJSONSerialization dataWithJSONObject:report options:NSJSONWritingPrettyPrinted error:nil];
    [json writeToFile:[directory stringByAppendingPathComponent:@"report.json"] atomically:YES];
    fprintf(stderr, "%s: %s\n", argv[1], [report[@"status"] UTF8String]);
    return (loadOnly ? loaded : passed) ? 0 : 1;
}}
