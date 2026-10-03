// Native Core ML benchmark for more-ane-transformers' GPT-2 KV-cache contract.
// Run through bench_coreml_macos.py. CPU_AND_NE excludes GPU, but allows CPU ops.
import CoreML
import Foundation

func ms() -> Double { Double(DispatchTime.now().uptimeNanoseconds) / 1e6 }
func thermal() -> String {
    switch ProcessInfo.processInfo.thermalState {
    case .nominal: return "nominal"
    case .fair: return "fair"
    case .serious: return "serious"
    case .critical: return "critical"
    @unknown default: return "unknown"
    }
}
func stats(_ samples: [Double]) -> [String: Any] {
    let sorted = samples.sorted(), total = samples.reduce(0, +)
    return ["samples": samples.count, "total_ms": total,
            "mean_ms": total / Double(samples.count),
            "p50_ms": sorted[sorted.count / 2], "p90_ms": sorted[sorted.count * 9 / 10],
            "min_ms": sorted.first!, "max_ms": sorted.last!, "raw_ms": samples,
            "steps_per_second": Double(samples.count) * 1000 / total]
}
struct Trace: Decodable {
    struct Case: Decodable {
        let prompt: String
        let prompt_ids: [Int]
        let next_tokens: [Int]
        let reference_offset: Int
        let steps: Int
    }
    let cases: [Case]
    let vocabulary: Int
    let trials: Int
    let warmups: Int
    let warmup_steps: Int
}
struct Diagnostics {
    var count = 0, mismatches = 0
    var maxRMSE = 0.0, maxAbs = 0.0
    var perStep: [[String: Any]] = []
    mutating func check(_ logits: MLMultiArray, _ expected: UnsafePointer<Float>, target: Int) throws -> Int {
        guard logits.count == 50257, logits.strides.last!.intValue == 1 else {
            throw NSError(domain: "benchmark", code: 1, userInfo: [NSLocalizedDescriptionKey: "Unexpected logit layout"])
        }
        var best = 0, bestValue = -Double.infinity, error = 0.0, norm = 0.0, absError = 0.0
        var values = [Double](); values.reserveCapacity(logits.count)
        for i in 0..<logits.count {
            let value: Double
            switch logits.dataType {
            case .float16: value = Double(logits.dataPointer.assumingMemoryBound(to: Float16.self)[i])
            case .float32: value = Double(logits.dataPointer.assumingMemoryBound(to: Float.self)[i])
            default: throw NSError(domain: "benchmark", code: 2, userInfo: [NSLocalizedDescriptionKey: "Unexpected logit dtype"])
            }
            guard value.isFinite else {
                throw NSError(domain: "benchmark", code: 3, userInfo: [NSLocalizedDescriptionKey: "Nonfinite logits"])
            }
            values.append(value)
            if value > bestValue { bestValue = value; best = i }
            let reference = Double(expected[i]), delta = value - reference
            error += delta * delta; norm += reference * reference
            absError = max(absError, abs(delta))
        }
        let rmse = sqrt(error / max(norm, 1e-6))
        let referenceMax = Double(expected[target])
        var sumP = 0.0, sumQ = 0.0, klNumerator = 0.0
        for i in 0..<values.count {
            let p = exp(Double(expected[i]) - referenceMax)
            let q = exp(values[i] - bestValue)
            sumP += p; sumQ += q
            klNumerator += p * (Double(expected[i]) - referenceMax - values[i] + bestValue)
        }
        let kl = max(0, klNumerator / sumP + log(sumQ / sumP))
        count += 1; mismatches += best == target ? 0 : 1
        maxRMSE = max(maxRMSE, rmse); maxAbs = max(maxAbs, absError)
        perStep.append(["argmax": best, "reference_argmax": target,
                        "normalized_logit_rmse": rmse, "max_abs_logit_error": absError,
                        "kl_hf_to_coreml_nats": kl])
        return best
    }
    var json: [String: Any] {
        ["checked_predictions": count, "top1_mismatches": mismatches,
         "max_normalized_logit_rmse": maxRMSE, "max_abs_logit_error": maxAbs,
         "max_kl_hf_to_coreml_nats": perStep.map { $0["kl_hf_to_coreml_nats"] as! Double }.max() ?? 0,
         "all_logits_finite": true, "per_prediction": perStep]
    }
}
final class Runner {
    let model: MLModel
    let ids: MLMultiArray, length: MLMultiArray, zero: MLMultiArray
    let inputLength: Int, context: Int, loadMS: Double
    init(_ url: URL, units: MLComputeUnits) throws {
        let config = MLModelConfiguration(); config.computeUnits = units
        let started = ms()
        model = try MLModel(contentsOf: url, configuration: config)
        loadMS = ms() - started
        let inputs = model.modelDescription.inputDescriptionsByName
        guard Set(inputs.keys) == Set(["input_ids", "full_sequence_length", "kv_cache"]),
              let idSpec = inputs["input_ids"]?.multiArrayConstraint,
              let cacheSpec = inputs["kv_cache"]?.multiArrayConstraint,
              idSpec.shape.count == 2, idSpec.shape[0] == 1,
              idSpec.dataType == .int32, cacheSpec.dataType == .float16,
              cacheSpec.shape.count == 4, cacheSpec.shape[0] == 12,
              cacheSpec.shape[1] == 1, cacheSpec.shape[3] == 1536 else {
            throw NSError(domain: "benchmark", code: 4, userInfo: [NSLocalizedDescriptionKey: "Not the GPT-2 124M KV-cache contract"])
        }
        inputLength = idSpec.shape.last!.intValue
        context = inputLength + cacheSpec.shape[2].intValue
        ids = try MLMultiArray(shape: idSpec.shape, dataType: .int32)
        length = try MLMultiArray(shape: [1], dataType: .int32)
        zero = try MLMultiArray(shape: cacheSpec.shape, dataType: .float16)
        memset(zero.dataPointer, 0, zero.count * MemoryLayout<Float16>.size)
    }
    func predict(_ tokens: [Int], cache: MLMultiArray) throws -> (MLFeatureProvider, Double) {
        let suffix = tokens.suffix(inputLength), pad = inputLength - suffix.count
        let data = ids.dataPointer.assumingMemoryBound(to: Int32.self)
        for i in 0..<pad { data[i] = 50256 }
        for (i, token) in suffix.enumerated() { data[pad + i] = Int32(token) }
        length.dataPointer.assumingMemoryBound(to: Int32.self)[0] = Int32(tokens.count)
        let provider = try MLDictionaryFeatureProvider(dictionary: [
            "input_ids": MLFeatureValue(multiArray: ids),
            "full_sequence_length": MLFeatureValue(multiArray: length),
            "kv_cache": MLFeatureValue(multiArray: cache)])
        let started = ms()
        let output = try model.prediction(from: provider)
        return (output, ms() - started)
    }
    func run(_ trace: Trace.Case, steps: Int, reference: UnsafePointer<Float>, check: Bool) throws -> [String: Any] {
        guard trace.prompt_ids.count <= inputLength,
              trace.prompt_ids.count + steps + 1 <= context else {
            throw NSError(domain: "benchmark", code: 5, userInfo: [NSLocalizedDescriptionKey: "Trace exceeds this benchmark's single-chunk prompt/context"])
        }
        var diag = Diagnostics(), tokens = trace.prompt_ids
        let initialThermal = thermal()
        let (prefill, prefillMS) = try predict(tokens, cache: zero)
        var cache = prefill.featureValue(for: "generation_kv_cache")!.multiArrayValue!
        if check {
            _ = try diag.check(prefill.featureValue(for: "logits")!.multiArrayValue!,
                               reference + trace.reference_offset, target: trace.next_tokens[0])
        }
        var samples: [Double] = []
        for i in 0..<steps {
            tokens.append(trace.next_tokens[i]) // same HF trace for both compute configurations
            try autoreleasepool {
                let (output, elapsed) = try predict(tokens, cache: cache)
                cache = output.featureValue(for: "generation_kv_cache")!.multiArrayValue!
                samples.append(elapsed)
                if check {
                    _ = try diag.check(output.featureValue(for: "logits")!.multiArrayValue!,
                                       reference + trace.reference_offset + (i + 1) * 50257,
                                       target: trace.next_tokens[i + 1])
                }
            }
        }
        return ["prefill_prediction_ms": prefillMS, "decode": stats(samples),
                "diagnostics": diag.json, "thermal_start": initialThermal, "thermal_end": thermal()]
    }
    func greedy(_ trace: Trace.Case, count: Int) throws -> [String: Any] {
        var tokens = trace.prompt_ids, generated: [Int] = [], cache = zero
        for _ in 0..<count {
            try autoreleasepool {
                let (output, _) = try predict(tokens, cache: cache)
                cache = output.featureValue(for: "generation_kv_cache")!.multiArrayValue!
                let logits = output.featureValue(for: "logits")!.multiArrayValue!
                var best = 0, bestValue = -Double.infinity
                for i in 0..<logits.count {
                    let value = logits.dataType == .float16
                        ? Double(logits.dataPointer.assumingMemoryBound(to: Float16.self)[i])
                        : Double(logits.dataPointer.assumingMemoryBound(to: Float.self)[i])
                    guard value.isFinite else { throw NSError(domain: "benchmark", code: 7) }
                    if value > bestValue { best = i; bestValue = value }
                }
                generated.append(best); tokens.append(best)
            }
        }
        let expected = Array(trace.next_tokens.prefix(count))
        let prefix = zip(generated, expected).prefix(while: { $0 == $1 }).count
        return ["prompt": trace.prompt, "generated_token_ids": generated,
                "reference_token_ids": expected, "matched_prefix_tokens": prefix,
                "exact_hf_greedy_match": generated == expected]
    }
}

do {
    guard CommandLine.arguments.count == 5 else {
        throw NSError(domain: "benchmark", code: 64, userInfo: [NSLocalizedDescriptionKey: "usage: bench MODEL.mlmodelc TRACE.json REFERENCE.f32 OUTPUT.json"])
    }
    let args = CommandLine.arguments
    let trace = try JSONDecoder().decode(Trace.self, from: Data(contentsOf: URL(fileURLWithPath: args[2])))
    let referenceData = try Data(contentsOf: URL(fileURLWithPath: args[3]))
    guard trace.vocabulary == 50257, trace.trials > 0, trace.warmups > 0,
          trace.cases.allSatisfy({ $0.next_tokens.count == $0.steps + 1 &&
              ($0.reference_offset + ($0.steps + 1) * 50257) * 4 <= referenceData.count }) else {
        throw NSError(domain: "benchmark", code: 6, userInfo: [NSLocalizedDescriptionKey: "Invalid reference trace"])
    }
    var report: [String: Any] = ["benchmark": "Native CoreML GPT-2, shared HF greedy trace",
                              "thermal_at_start": thermal()]
    try referenceData.withUnsafeBytes { buffer in
        let reference = buffer.bindMemory(to: Float.self).baseAddress!
        let url = URL(fileURLWithPath: args[1])
        let cpu = try Runner(url, units: .cpuOnly)
        let ane = try Runner(url, units: .cpuAndNeuralEngine)
        let runners = ["cpu_only": cpu, "cpu_and_ne": ane]
        var trials: [String: [[String: Any]]] = ["cpu_only": [], "cpu_and_ne": []]
        var warmups: [String: [[String: Any]]] = ["cpu_only": [], "cpu_and_ne": []]
        for backend in ["cpu_only", "cpu_and_ne"] {
            for _ in 0..<trace.warmups {
                let started = ms()
                var run = try runners[backend]!.run(trace.cases[0], steps: trace.warmup_steps,
                                                  reference: reference, check: false)
                run["wall_ms"] = ms() - started
                warmups[backend]!.append(run)
            }
            print("Prewarmed \(backend)")
        }
        var order: [String] = []
        for trial in 0..<trace.trials {
            for backend in (trial % 2 == 0 ? ["cpu_only", "cpu_and_ne"] : ["cpu_and_ne", "cpu_only"]) {
                let run = try runners[backend]!.run(trace.cases[0], steps: trace.cases[0].steps,
                                                  reference: reference, check: true)
                trials[backend]!.append(run); order.append(backend)
                let metric = run["decode"] as! [String: Any]
                print(String(format: "Trial %d %@: %.2f steps/s, p50 %.2f ms, p90 %.2f ms",
                             trial + 1, backend, metric["steps_per_second"] as! Double,
                             metric["p50_ms"] as! Double, metric["p90_ms"] as! Double))
            }
        }
        var backends: [String: Any] = [:]
        for backend in ["cpu_only", "cpu_and_ne"] {
            let runs = trials[backend]!
            let samples = runs.flatMap { ($0["decode"] as! [String: Any])["raw_ms"] as! [Double] }
            var additional: [[String: Any]] = []
            for item in trace.cases.dropFirst() {
                var run = try runners[backend]!.run(item, steps: item.steps, reference: reference, check: true)
                run["prompt"] = item.prompt; additional.append(run)
            }
            backends[backend] = ["model_load_ms": runners[backend]!.loadMS,
                "input_length": runners[backend]!.inputLength, "context_length": runners[backend]!.context,
                "warmups": warmups[backend]!, "trials": runs, "decode": stats(samples),
                "prefill": stats(runs.map { $0["prefill_prediction_ms"] as! Double }),
                "additional_prompt_checks": additional,
                "free_greedy_smoke": try trace.cases.map { try runners[backend]!.greedy($0, count: 16) }]
        }
        report["backends"] = backends; report["trial_order"] = order
    }
    report["thermal_at_end"] = thermal()
    let data = try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
    try data.write(to: URL(fileURLWithPath: args[4]), options: .withoutOverwriting)
} catch {
    fputs("CoreML benchmark failed: \(error)\n", stderr)
    exit(1)
}
