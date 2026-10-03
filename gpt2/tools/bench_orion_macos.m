// Reference-only benchmark of Orion's native macOS CPU and ANE backends.
// Build/run through bench_orion_macos.py. No CPU fallback is permitted.
#import <Foundation/Foundation.h>
#import "model/weight_loader.h"
#import "model/configs/gpt2_124m.h"
#import "kernels/inference/decode_cpu.h"
#import "kernels/inference/decode_ane.h"
#import "kernels/inference/prefill_ane.h"
#import "kernels/inference/kv_cache.h"
#import "core/ane_runtime.h"
#import "core/ane_program_cache.h"
#import "core/ane_io.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

enum { Steps = 64, Trials = 5, Warmup = 16 };
static const int prompt[] = {15496, 995}; // Hello world

static double milliseconds(void) {
    struct timespec now;
    if (clock_gettime(CLOCK_MONOTONIC, &now)) abort();
    return now.tv_sec * 1000.0 + now.tv_nsec * 1e-6;
}

static int argmax(const float *values, int size) {
    int best = 0;
    for (int i = 0; i < size; ++i) {
        if (!isfinite(values[i])) { fprintf(stderr, "nonfinite logits\n"); exit(1); }
        if (values[i] > values[best]) best = i;
    }
    return best;
}

static void prefill(bool ane, OrionGPT2Weights *weights, OrionKVCache *kv,
                    const char *blobs, float *logits) {
    if (ane) {
        if (!orion_ane_prefill(weights, prompt, 2, &kGPT2_124M, blobs, kv, logits)) {
            fprintf(stderr, "ANE prefill failed; no fallback\n"); exit(1);
        }
    } else orion_gpt2_prefill_kv(weights, prompt, 2, kv, logits);
}

static void step(bool ane, OrionGPT2Weights *weights, OrionKVCache *kv,
                 const char *blobs, int token, float *logits) {
    if (ane) {
        if (!orion_ane_decode_step(weights, kv, token, blobs, logits)) {
            fprintf(stderr, "ANE decode failed; no fallback\n"); exit(1);
        }
    } else orion_gpt2_decode_step(weights, kv, token, logits);
}

static int compareDouble(const void *left, const void *right) {
    const double a = *(const double *)left, b = *(const double *)right;
    return (a > b) - (a < b);
}

static NSDictionary *statistics(const double *samples, int count, bool decode) {
    double sorted[Trials * Steps], total = 0;
    NSMutableArray *raw = [NSMutableArray arrayWithCapacity:count];
    for (int i = 0; i < count; ++i) {
        sorted[i] = samples[i]; total += samples[i]; [raw addObject:@(samples[i])];
    }
    qsort(sorted, count, sizeof(double), compareDouble);
    NSMutableDictionary *result = [@{@"samples": @(count), @"total_ms": @(total), @"mean_ms": @(total / count),
             @"p50_ms": @(sorted[count / 2]), @"p90_ms": @(sorted[count * 9 / 10]),
             @"min_ms": @(sorted[0]), @"max_ms": @(sorted[count - 1]),
             @"raw_ms": raw} mutableCopy];
    if (decode) result[@"decode_steps_per_second"] = @(count * 1000.0 / total);
    return result;
}

static double normalizedRMSE(const float *actual, const float *expected, int count) {
    double error = 0, norm = 0;
    for (int i = 0; i < count; ++i) {
        double delta = actual[i] - expected[i];
        error += delta * delta; norm += (double)expected[i] * expected[i];
    }
    return sqrt(error / fmax(norm, 1e-6));
}

int main(int argc, const char *argv[]) { @autoreleasepool {
    if (argc != 3) { fprintf(stderr, "usage: bench WEIGHT_BLOBS RESULT_JSON\n"); return 64; }
    const char *blobs = argv[1];
    double started = milliseconds();
    OrionGPT2Weights *weights = orion_gpt2_weights_load(blobs);
    if (!weights) return 1;
    double load_ms = milliseconds() - started;
    float *logits = malloc(weights->vocab * sizeof(float));
    float *reference = malloc(Steps * weights->vocab * sizeof(float));
    if (!logits || !reference) return 1;
    int tokens[Steps], expected[Steps];
    float *prompt_reference = malloc(weights->vocab * sizeof(float));
    if (!prompt_reference) return 1;

    // Untimed CPU greedy trace gives both backends identical token/context work.
    fprintf(stderr, "Building the shared 64-step CPU token trace...\n");
    OrionKVCache *kv = orion_kv_cache_create(&kGPT2_124M);
    if (!kv) return 1;
    prefill(false, weights, kv, blobs, logits);
    memcpy(prompt_reference, logits, weights->vocab * sizeof(float));
    for (int i = 0; i < Steps; ++i) {
        tokens[i] = argmax(logits, weights->vocab);
        step(false, weights, kv, blobs, tokens[i], logits);
        expected[i] = argmax(logits, weights->vocab);
        memcpy(reference + i * weights->vocab, logits, weights->vocab * sizeof(float));
    }
    orion_kv_cache_free(kv);

    // Warm both backends in this process. All ANE compilation is outside the
    // measured trials, including capability probes and prompt-prefill programs.
    int compiles_before_warmup = 0;
    double warmup_ms[2];
    for (int backend = 0; backend < 2; ++backend) {
        fprintf(stderr, "Prewarming %s...\n", backend ? "ANE" : "CPU");
        kv = orion_kv_cache_create(&kGPT2_124M);
        if (!kv) return 1;
        started = milliseconds();
        if (backend) {
            if (!orion_ane_init()) return 1;
            compiles_before_warmup = orion_compile_count();
        }
        prefill(backend, weights, kv, blobs, logits);
        for (int i = 0; i < Warmup; ++i) step(backend, weights, kv, blobs, tokens[i], logits);
        warmup_ms[backend] = milliseconds() - started;
        argmax(logits, weights->vocab);
        orion_kv_cache_free(kv);
    }
    const OrionANEIO *policy = orion_ane_io_policy();
    if (!policy || policy->dtype != ORION_IO_FP16 || !policy->pack_weights ||
        orion_io_decode_seq(policy->dtype) != 32) {
        fprintf(stderr, "runtime chose a different I/O layout from the captured M1 FP16/32/packed configuration\n");
        return 1;
    }
    int compiled = orion_compile_count(), cached = orion_cache_size();
    if (cached != 49) { fprintf(stderr, "expected 49 warmed programs; got %d\n", cached); return 1; }

    // Untimed control: identical CPU-created prompt KV, followed by ANE decode.
    // Distinguishes prompt-prefill differences from subsequent fp16 decode drift.
    kv = orion_kv_cache_create(&kGPT2_124M);
    if (!kv) return 1;
    prefill(false, weights, kv, blobs, logits);
    NSMutableArray *control = [NSMutableArray array];
    for (int i = 0; i < 8; ++i) {
        step(true, weights, kv, blobs, tokens[i], logits);
        [control addObject:@{@"step": @(i + 1),
            @"normalized_logit_rmse": @(normalizedRMSE(logits, reference + i * weights->vocab, weights->vocab)),
            @"top1_matches_cpu": @(argmax(logits, weights->vocab) == expected[i])}];
    }
    orion_kv_cache_free(kv);

    double times[2][Trials * Steps], prefill_times[2][Trials];
    int top1_mismatches[2] = {0, 0};
    double max_normalized_rmse[2] = {0, 0}, max_abs[2] = {0, 0};
    double prefill_rmse[2] = {0, 0};
    NSMutableArray *trial_reports = [NSMutableArray array];
    for (int trial = 0; trial < Trials; ++trial) {
        // Alternate which backend runs first to reduce order/thermal bias.
        for (int order = 0; order < 2; ++order) { @autoreleasepool {
            bool ane = (trial + order) % 2;
            fprintf(stderr, "Measured trial %d/%d: %s, 64 steps...\n", trial + 1, Trials, ane ? "ANE" : "CPU");
            kv = orion_kv_cache_create(&kGPT2_124M);
            if (!kv) return 1;
            started = milliseconds();
            prefill(ane, weights, kv, blobs, logits);
            prefill_times[ane][trial] = milliseconds() - started;
            argmax(logits, weights->vocab);
            prefill_rmse[ane] = fmax(prefill_rmse[ane], normalizedRMSE(logits, prompt_reference, weights->vocab));
            for (int i = 0; i < Steps; ++i) {
                started = milliseconds();
                step(ane, weights, kv, blobs, tokens[i], logits);
                times[ane][trial * Steps + i] = milliseconds() - started;
                // All numerical validation is outside the timed interval.
                if (argmax(logits, weights->vocab) != expected[i]) top1_mismatches[ane]++;
                double squared = 0, reference_squared = 0;
                for (int token = 0; token < weights->vocab; ++token) {
                    double value = reference[i * weights->vocab + token];
                    double difference = logits[token] - value;
                    squared += difference * difference; reference_squared += value * value;
                    max_abs[ane] = fmax(max_abs[ane], fabs(difference));
                }
                double relative = sqrt(squared / fmax(reference_squared, 1e-6));
                max_normalized_rmse[ane] = fmax(max_normalized_rmse[ane], relative);
            }
            if (orion_compile_count() != compiled || orion_cache_size() != cached) {
                fprintf(stderr, "compilation/cache change during timed trials\n"); return 1;
            }
            orion_kv_cache_free(kv);
            [trial_reports addObject:@{@"trial": @(trial + 1), @"order": @(order + 1),
                @"backend": ane ? @"ane" : @"cpu", @"prefill_ms": @(prefill_times[ane][trial]),
                @"decode": statistics(times[ane] + trial * Steps, Steps, true)}];
        }}
    }
    NSMutableDictionary *backends = [NSMutableDictionary dictionary];
    for (int backend = 0; backend < 2; ++backend) {
        backends[backend ? @"ane" : @"cpu"] = @{
            @"decode": statistics(times[backend], Trials * Steps, true),
            @"prefill": statistics(prefill_times[backend], Trials, false),
            @"max_prefill_normalized_logit_rmse": @(prefill_rmse[backend]),
            @"top1_mismatches": @(top1_mismatches[backend]),
            @"max_normalized_logit_rmse": @(max_normalized_rmse[backend]),
            @"max_abs_logit_error": @(max_abs[backend])};
    }
    NSMutableArray *trace = [NSMutableArray arrayWithCapacity:Steps];
    for (int i = 0; i < Steps; ++i) [trace addObject:@(tokens[i])];
    NSDictionary *report = @{
        @"benchmark": @"Orion native CPU versus macOS ANE, prewarmed same-process reference",
        @"prompt": @"Hello world", @"prompt_tokens": @[@15496, @995],
        @"steps_per_trial": @(Steps), @"trials_per_backend": @(Trials), @"warmup_decode_steps": @(Warmup),
        @"workload": @"shared CPU greedy token trace, teacher-forced into both backends; fixed 64 steps",
        @"token_trace": trace, @"weights_load_ms_excluded": @(load_ms),
        @"untimed_cpu_prefill_ane_decode_control": control,
        @"parity_diagnostic": @{@"normalized_rmse_threshold": @0.005,
            @"ane_logits_within_threshold": @(max_normalized_rmse[1] <= 0.005),
            @"ane_all_top1_match": @(top1_mismatches[1] == 0),
            @"note": @"Timing remains a fixed teacher-forced workload; logit differences and top1 mismatches are reported, not treated as output equivalence."},
        @"warmup_ms_excluded": @{@"cpu": @(warmup_ms[0]), @"ane": @(warmup_ms[1])},
        @"compile_count_before_warmup": @(compiles_before_warmup),
        @"compile_count_before_measurement": @(compiled), @"compile_count_after_measurement": @(orion_compile_count()),
        @"timed_compiles": @(orion_compile_count() - compiled), @"cached_programs": @(cached),
        @"io": @{@"dtype": @"fp16", @"decode_stride": @32, @"mil_weight_staging": @"packed"},
        @"clock": @"CLOCK_MONOTONIC", @"backend_fallback": @NO,
        @"excluded_from_decode": @[@"compilation", @"weight loading", @"prefill", @"KV allocation",
                                      @"token selection", @"printing", @"numerical validation"],
        @"included_in_decode": @[@"ANE dispatch and IOSurface transfers/allocations", @"CPU attention", @"embeddings and logits"],
        @"backends": backends, @"trials": trial_reports};
    NSError *error = nil;
    NSData *data = [NSJSONSerialization dataWithJSONObject:report options:NSJSONWritingPrettyPrinted error:&error];
    if (!data || ![data writeToFile:@(argv[2]) options:NSDataWritingAtomic error:&error]) {
        fprintf(stderr, "cannot write report: %s\n", error.localizedDescription.UTF8String); return 1;
    }
    for (NSString *backend in @[@"cpu", @"ane"])
        fprintf(stderr, "%s: %.2f decode steps/s, p50 %.2f ms, mean %.2f ms\n", backend.UTF8String,
            [backends[backend][@"decode"][@"decode_steps_per_second"] doubleValue],
            [backends[backend][@"decode"][@"p50_ms"] doubleValue],
            [backends[backend][@"decode"][@"mean_ms"] doubleValue]);
    orion_cache_clear();
    free(logits); free(reference); free(prompt_reference); orion_gpt2_weights_free(weights);
    return 0;
}}
