// Native Orion implementation reference: identical saved HF traces, separate diagnostics.
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

enum { Steps = 64, Trials = 4, Warmup = 16 };
static double milliseconds(void) {
    struct timespec t; if (clock_gettime(CLOCK_MONOTONIC, &t)) abort();
    return t.tv_sec * 1000.0 + t.tv_nsec * 1e-6;
}
static void fail(const char *s) { fprintf(stderr, "%s\n", s); exit(1); }
static NSString *thermal(void) {
    switch (NSProcessInfo.processInfo.thermalState) {
        case NSProcessInfoThermalStateNominal: return @"nominal";
        case NSProcessInfoThermalStateFair: return @"fair";
        case NSProcessInfoThermalStateSerious: return @"serious";
        case NSProcessInfoThermalStateCritical: return @"critical";
    }
    return @"unknown";
}
static int argmax(const float *x, int n) {
    int best = 0;
    for (int i = 0; i < n; i++) {
        if (!isfinite(x[i])) fail("Nonfinite logits");
        if (x[i] > x[best]) best = i;
    }
    return best;
}
static int compareDouble(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}
static NSDictionary *stats(const double *values, int n) {
    double *sorted = malloc(n * sizeof(double)), total = 0;
    if (!sorted || n < 1) fail("Invalid stats allocation");
    NSMutableArray *raw = [NSMutableArray array];
    for (int i = 0; i < n; i++) {
        if (!isfinite(values[i]) || values[i] <= 0) fail("Invalid timing");
        sorted[i] = values[i]; total += values[i]; [raw addObject:@(values[i])];
    }
    qsort(sorted, n, sizeof(double), compareDouble);
    NSDictionary *result = @{@"samples": @(n), @"total_ms": @(total),
        @"mean_ms": @(total/n), @"p50_ms": @(sorted[n/2]), @"p90_ms": @(sorted[n*9/10]),
        @"min_ms": @(sorted[0]), @"max_ms": @(sorted[n-1]),
        @"steps_per_second": @(1000.0*n/total), @"raw_ms": raw};
    free(sorted); return result;
}
static void prefill(bool ane, OrionGPT2Weights *w, OrionKVCache *kv,
                    const char *blobs, const int *prompt, int length, float *logits) {
    if (ane) {
        if (!orion_ane_prefill(w, prompt, length, &kGPT2_124M, blobs, kv, logits))
            fail("ANE prefill failed; no fallback");
    } else orion_gpt2_prefill_kv(w, prompt, length, kv, logits);
    if (kv->current_len != length) fail("Prefill cache length mismatch");
}
static void step(bool ane, OrionGPT2Weights *w, OrionKVCache *kv,
                 const char *blobs, int token, float *logits) {
    int before = kv->current_len;
    if (ane) {
        if (!orion_ane_decode_step(w, kv, token, blobs, logits)) fail("ANE decode failed; no fallback");
    } else orion_gpt2_decode_step(w, kv, token, logits);
    if (kv->current_len != before + 1) fail("Decode cache length mismatch");
}
static OrionKVCache *cache(void) {
    OrionKVCache *kv = orion_kv_cache_create(&kGPT2_124M);
    if (!kv) fail("Cache allocation failed"); return kv;
}
static NSDictionary *diagnostic(const float *x, const float *ref, int n) {
    int chosen = argmax(x, n), expected = argmax(ref, n);
    double pmax = ref[expected], qmax = x[chosen], psum = 0, qsum = 0;
    double err = 0, norm = 0;
    for (int i = 0; i < n; i++) {
        psum += exp((double)ref[i] - pmax); qsum += exp((double)x[i] - qmax);
        double d = (double)x[i] - ref[i]; err += d*d; norm += (double)ref[i]*ref[i];
    }
    double plog = pmax + log(psum), qlog = qmax + log(qsum), kl = 0;
    for (int i = 0; i < n; i++) {
        double lp = (double)ref[i] - plog;
        kl += exp(lp) * (lp - ((double)x[i] - qlog));
    }
    return @{@"top1": @(chosen), @"reference_top1": @(expected),
        @"kl_hf_to_orion_nats": @(fmax(0, kl)),
        @"normalized_logit_rmse": @(sqrt(err/fmax(norm, 1e-12)))};
}

int main(int argc, const char *argv[]) { @autoreleasepool {
    if (argc != 5) { fprintf(stderr, "usage: bench BLOBS TRACE_JSON REFERENCE_F32 RESULT_JSON\n"); return 64; }
    const char *blobs = argv[1];
    NSDictionary *trace = [NSJSONSerialization JSONObjectWithData:[NSData dataWithContentsOfFile:@(argv[2])]
        options:0 error:nil];
    NSData *reference = [NSData dataWithContentsOfFile:@(argv[3]) options:NSDataReadingMappedIfSafe error:nil];
    if (!trace || !reference || [trace[@"vocabulary"] intValue] != 50257) fail("Invalid reference");
    double start = milliseconds(); OrionGPT2Weights *w = orion_gpt2_weights_load(blobs);
    if (!w || w->vocab != 50257) fail("Weight load failed");
    double load_ms = milliseconds() - start;
    float *logits = malloc(w->vocab*sizeof(float)); if (!logits) fail("Logits allocation failed");
    NSMutableArray *cases = [NSMutableArray array];
    for (NSDictionary *item in trace[@"cases"]) { @autoreleasepool {
        int length = (int)[item[@"prompt_ids"] count], prompt[64], tokens[65];
        if ((length != 2 && length != 32 && length != 64) || [item[@"next_tokens"] count] != 65)
            fail("Invalid prompt or token trace");
        for (int i = 0; i < length; i++) prompt[i] = [item[@"prompt_ids"][i] intValue];
        for (int i = 0; i < 65; i++) tokens[i] = [item[@"next_tokens"][i] intValue];
        size_t offset = [item[@"reference_offset"] unsignedLongLongValue];
        if ((offset + 65*w->vocab)*sizeof(float) > reference.length) fail("Reference boundary mismatch");
        const float *ref = (const float *)reference.bytes + offset;
        orion_cache_clear();
        NSMutableDictionary *backends = [NSMutableDictionary dictionary];
        double timing[2][Trials*Steps], prefill_times[2][Trials], warm_ms[2];
        NSMutableArray *trials = [NSMutableArray array];
        // Compile/warm this prompt bucket before separate numerical replays and trials.
        for (int b = 0; b < 2; b++) {
            OrionKVCache *kv = cache(); start = milliseconds();
            if (b && !orion_ane_init()) fail("ANE initialization failed");
            prefill(b, w, kv, blobs, prompt, length, logits);
            for (int i = 0; i < Warmup; i++) step(b, w, kv, blobs, tokens[i], logits);
            warm_ms[b] = milliseconds() - start; orion_kv_cache_free(kv);
        }
        const OrionANEIO *policy = orion_ane_io_policy();
        if (!policy || policy->dtype != ORION_IO_FP16 || !policy->pack_weights ||
            orion_io_decode_seq(policy->dtype) != 32) fail("Unexpected ANE IO policy");
        int compiled = orion_compile_count(), cached = orion_cache_size();
        for (int b = 0; b < 2; b++) {
            OrionKVCache *kv = cache(); NSMutableArray *diag = [NSMutableArray array];
            prefill(b, w, kv, blobs, prompt, length, logits);
            [diag addObject:diagnostic(logits, ref, w->vocab)];
            for (int i = 0; i < Steps; i++) {
                step(b, w, kv, blobs, tokens[i], logits);
                [diag addObject:diagnostic(logits, ref+(i+1)*w->vocab, w->vocab)];
            }
            orion_kv_cache_free(kv); kv = cache();
            prefill(b, w, kv, blobs, prompt, length, logits);
            NSMutableArray *free_tokens = [NSMutableArray array]; bool matches = true;
            for (int i = 0; i < 16; i++) {
                int token = argmax(logits, w->vocab); [free_tokens addObject:@(token)];
                matches = matches && token == tokens[i];
                if (i < 15) step(b, w, kv, blobs, token, logits);
            }
            orion_kv_cache_free(kv);
            int mismatches = 0; double max_kl = 0, max_rmse = 0;
            for (NSDictionary *d in diag) {
                mismatches += ![d[@"top1"] isEqual:d[@"reference_top1"]];
                max_kl = fmax(max_kl, [d[@"kl_hf_to_orion_nats"] doubleValue]);
                max_rmse = fmax(max_rmse, [d[@"normalized_logit_rmse"] doubleValue]);
            }
            backends[b ? @"ane" : @"cpu"] = [@{@"diagnostics": diag,
                @"checked_predictions": @65, @"top1_mismatches": @(mismatches),
                @"max_kl_hf_to_orion_nats": @(max_kl), @"max_normalized_logit_rmse": @(max_rmse),
                @"all_logits_finite": @YES,
                @"strict_trace_choice_and_kl_gate_passed": @(mismatches == 0 && max_kl <= .01),
                @"raw_logit_rmse_gate_passed": @(max_rmse <= .005),
                @"free_greedy_token_ids": free_tokens, @"free_greedy_exact_hf_match": @(matches)} mutableCopy];
        }
        for (int trial = 0; trial < Trials; trial++) {
            for (int order = 0; order < 2; order++) { @autoreleasepool {
                int b = (trial + order) % 2;
                fprintf(stderr, "prompt=%d trial=%d backend=%s\n", length, trial+1, b ? "ane" : "cpu");
                // Two new 16-step requests immediately before each measured request.
                for (int warm = 0; warm < 2; warm++) {
                    OrionKVCache *kv = cache(); prefill(b, w, kv, blobs, prompt, length, logits);
                    for (int i = 0; i < Warmup; i++) step(b, w, kv, blobs, tokens[i], logits);
                    orion_kv_cache_free(kv);
                }
                NSString *initial_thermal = thermal(); OrionKVCache *kv = cache();
                start = milliseconds(); prefill(b, w, kv, blobs, prompt, length, logits);
                prefill_times[b][trial] = milliseconds() - start;
                for (int i = 0; i < Steps; i++) {
                    start = milliseconds(); step(b, w, kv, blobs, tokens[i], logits);
                    timing[b][trial*Steps+i] = milliseconds() - start;
                }
                orion_kv_cache_free(kv);
                if (orion_compile_count() != compiled || orion_cache_size() != cached)
                    fail("Unexpected timed compilation or cache change");
                [trials addObject:@{@"trial": @(trial+1), @"order": @(order+1),
                    @"backend": b ? @"ane" : @"cpu", @"thermal_start": initial_thermal,
                    @"thermal_end": thermal(), @"prefill_prediction_ms": @(prefill_times[b][trial]),
                    @"decode": stats(timing[b]+trial*Steps, Steps)}];
            }}
        }
        for (int b = 0; b < 2; b++) {
            NSMutableDictionary *d = backends[b ? @"ane" : @"cpu"];
            d[@"decode"] = stats(timing[b], Trials*Steps);
            d[@"prefill"] = stats(prefill_times[b], Trials);
        }
        [cases addObject:@{@"prompt_tokens": @(length), @"prompt_ids": item[@"prompt_ids"],
            @"token_trace": [item[@"next_tokens"] subarrayWithRange:NSMakeRange(0, Steps)],
            @"backends": backends, @"trials": trials,
            @"compile_count_before_measurement": @(compiled),
            @"compile_count_after_measurement": @(orion_compile_count()), @"timed_compiles": @0,
            @"cached_programs": @(cached), @"bucket_warmup_ms_excluded": @{@"cpu": @(warm_ms[0]), @"ane": @(warm_ms[1])}}];
        for (NSString *b in @[@"cpu", @"ane"]) fprintf(stderr, "prompt=%d %s %.2f steps/s gate=%s\n",
            length, b.UTF8String, [backends[b][@"decode"][@"steps_per_second"] doubleValue],
            [backends[b][@"strict_trace_choice_and_kl_gate_passed"] boolValue] ? "pass" : "fail");
    }}
    NSDictionary *report = @{@"format_version": @1, @"cases": cases, @"weights_load_ms_excluded": @(load_ms),
        @"backend_fallback": @NO, @"clock": @"CLOCK_MONOTONIC",
        @"io": @{@"dtype": @"fp16", @"decode_stride": @32, @"mil_weight_staging": @"packed"}};
    NSError *error = nil;
    NSData *data = [NSJSONSerialization dataWithJSONObject:report options:NSJSONWritingPrettyPrinted error:&error];
    if (!data || ![data writeToFile:@(argv[4]) options:NSDataWritingAtomic error:&error]) fail("Cannot save report");
    orion_cache_clear(); free(logits); orion_gpt2_weights_free(w); return 0;
}}
