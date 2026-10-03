// Native Core ML benchmark for more-ane-transformers' GPT-2 KV-cache contract.
// Matched-artifact four-configuration benchmark; driven by bench_coreml_fair.py.
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
        let requestStarted = ms()
        memset(zero.dataPointer, 0, zero.count * MemoryLayout<Float16>.size)
        let (prefill, prefillMS) = try predict(tokens, cache: zero)
        var cache = prefill.featureValue(for: "generation_kv_cache")!.multiArrayValue!
        _ = try argmax(prefill.featureValue(for: "logits")!.multiArrayValue!)
        let ttft = ms() - requestStarted
        if check {
            _ = try diag.check(prefill.featureValue(for: "logits")!.multiArrayValue!,
                               reference + trace.reference_offset, target: trace.next_tokens[0])
        }
        var samples: [Double] = [], requestSamples: [Double] = []
        for i in 0..<steps {
            let stepStarted = ms()
            tokens.append(trace.next_tokens[i]) // same HF trace for both compute configurations
            try autoreleasepool {
                let (output, elapsed) = try predict(tokens, cache: cache)
                cache = output.featureValue(for: "generation_kv_cache")!.multiArrayValue!
                _ = try argmax(output.featureValue(for: "logits")!.multiArrayValue!)
                samples.append(elapsed)
                if check {
                    _ = try diag.check(output.featureValue(for: "logits")!.multiArrayValue!,
                                       reference + trace.reference_offset + (i + 1) * 50257,
                                       target: trace.next_tokens[i + 1])
                }
            }
            requestSamples.append(ms() - stepStarted)
        }
        return ["prefill_prediction_ms": prefillMS, "warm_ttft_ms": ttft,
                "request_decode": stats(requestSamples), "decode": stats(samples),
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

func argmax(_ logits: MLMultiArray) throws -> Int {
    guard logits.count == 50257, logits.strides.last!.intValue == 1 else {
        throw NSError(domain: "benchmark", code: 8)
    }
    var best = 0, maximum = -Double.infinity
    for i in 0..<logits.count {
        let value = logits.dataType == .float16
            ? Double(logits.dataPointer.assumingMemoryBound(to: Float16.self)[i])
            : Double(logits.dataPointer.assumingMemoryBound(to: Float.self)[i])
        guard value.isFinite else { throw NSError(domain: "benchmark", code: 9) }
        if value > maximum { best = i; maximum = value }
    }
    return best
}

do {
    guard CommandLine.arguments.count == 5 else { throw NSError(domain: "benchmark", code: 64) }
    let args = CommandLine.arguments
    let trace = try JSONDecoder().decode(Trace.self, from: Data(contentsOf: URL(fileURLWithPath: args[2])))
    let bytes = try Data(contentsOf: URL(fileURLWithPath: args[3]))
    let names = ["cpu_only", "cpu_and_gpu", "cpu_and_ne", "all"]
    let units: [MLComputeUnits] = [.cpuOnly, .cpuAndGPU, .cpuAndNeuralEngine, .all]
    guard trace.trials >= 4, trace.trials % 4 == 0, trace.warmups > 0,
          trace.vocabulary == 50257,
          trace.cases.allSatisfy({ $0.next_tokens.count == $0.steps + 1 &&
            ($0.reference_offset + ($0.steps + 1) * 50257) * 4 <= bytes.count }) else {
        throw NSError(domain: "benchmark", code: 6)
    }
    var report: [String: Any] = ["thermal_at_start": thermal()]
    try bytes.withUnsafeBytes { buffer in
        let reference = buffer.bindMemory(to: Float.self).baseAddress!
        var trialData = trace.cases.map { _ in Dictionary(uniqueKeysWithValues: names.map { ($0, [[String: Any]]()) }) }
        var diagnostics = trace.cases.map { _ in [String: Any]() }
        var smokes = trace.cases.map { _ in [String: Any]() }
        var loads = Dictionary(uniqueKeysWithValues: names.map { ($0, [Double]()) })
        let square = [[0,1,3,2], [1,2,0,3], [2,3,1,0], [3,0,2,1]]
        var order: [[String]] = []
        var inputWindow = 0, context = 0
        for trial in 0..<trace.trials {
            let row = square[trial % 4]
            order.append(row.map { names[$0] })
            for index in row {
                let name = names[index]
                // One resident model. Dispose it before loading the next configuration.
                try autoreleasepool {
                    print("Loading block \(trial + 1) \(name)"); fflush(stdout)
                    let runner = try Runner(URL(fileURLWithPath: args[1]), units: units[index])
                    loads[name]!.append(runner.loadMS)
                    inputWindow = runner.inputLength; context = runner.context
                    for (caseIndex, item) in trace.cases.enumerated() {
                        if trial == 0 {
                            let checked = try runner.run(item, steps: item.steps, reference: reference, check: true)
                            diagnostics[caseIndex][name] = checked["diagnostics"]!
                            smokes[caseIndex][name] = try runner.greedy(item, count: 16)
                        }
                        // Warm each individual timed trial after loading and validation.
                        for _ in 0..<trace.warmups {
                            _ = try runner.run(item, steps: trace.warmup_steps, reference: reference, check: false)
                        }
                        let run = try runner.run(item, steps: item.steps, reference: reference, check: false)
                        trialData[caseIndex][name]!.append(run)
                        let metric = run["decode"] as! [String: Any]
                        print(String(format: "prompt=%d block=%d %@ %.2f steps/s",
                            item.prompt_ids.count, trial + 1, name, metric["steps_per_second"] as! Double))
                        fflush(stdout)
                    }
                }
            }
        }
        var cases: [[String: Any]] = []
        for (caseIndex, item) in trace.cases.enumerated() {
            var backends: [String: Any] = [:]
            for name in names {
                let runs = trialData[caseIndex][name]!
                let engine = runs.flatMap { ($0["decode"] as! [String: Any])["raw_ms"] as! [Double] }
                let request = runs.flatMap { ($0["request_decode"] as! [String: Any])["raw_ms"] as! [Double] }
                backends[name] = ["trials": runs, "decode": stats(engine), "request_decode": stats(request),
                    "prefill": stats(runs.map { $0["prefill_prediction_ms"] as! Double }),
                    "warm_ttft": stats(runs.map { $0["warm_ttft_ms"] as! Double }),
                    "diagnostics": diagnostics[caseIndex][name]!, "free_greedy_smoke": smokes[caseIndex][name]!]
            }
            cases.append(["prompt_tokens": item.prompt_ids.count, "steps": item.steps,
                "trial_order": order, "backends": backends])
        }
        report["cases"] = cases
        report["model_load_ms_excluded"] = loads
        report["input_window"] = inputWindow
        report["context_capacity"] = context
    }
    report["thermal_at_end"] = thermal()
    let data = try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
    try data.write(to: URL(fileURLWithPath: args[4]), options: .withoutOverwriting)
} catch {
    fputs("Fair CoreML benchmark failed: \(error)\n", stderr)
    exit(1)
}
