"""Capture static Core ML MUL compile options, structures, and numerical execution."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import plistlib
import subprocess
import struct
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gpt2.hwx import parse_container, parse_tasks


def structure(path, directory):
    data = path.read_bytes()
    record = dict(path=str(path), sha256=hashlib.sha256(data).hexdigest(), bytes=len(data))
    record["compiler_strings"] = [line for line in subprocess.check_output(["strings", "-a", str(path)], text=True).splitlines()
                                  if any(key in line for key in ("ANEC v", "zin_ane_compiler", "ModuleVersion", "ModuleBundleName"))]
    try:
        container = parse_container(data)
        (directory / "container.json").write_text(json.dumps(container, indent=2) + "\n")
        text = next(s for s in container["segments"] if s["name"] == "__TEXT")
        raw = data[text["fileoff"]:text["fileoff"] + text["filesize"]]
        thread = container["thread"]
        record.update(td_size=thread["td_size"], td_count=thread["td_count"], text_offset=text["fileoff"])
        start = thread["entry"] - text["vmaddr"]
        if 0 <= start <= len(raw) - 44:
            record["entry_words_hex"] = [hex(word) for word in struct.unpack_from("<11I", raw, start)]
        tasks = parse_tasks(raw, thread["td_size"], thread["td_count"])
        for task in tasks: task["registers"] = {hex(k):v for k,v in task["registers"].items()}
        (directory / "tasks.json").write_text(json.dumps(tasks, indent=2) + "\n")
        record["task_structure_sha256"] = hashlib.sha256(json.dumps(tasks, sort_keys=True).encode()).hexdigest()
        record["strict_task_parse"] = "pass"
    except Exception as error:
        record.update(strict_task_parse="unsupported", parse_error=str(error))
    (directory / "structure.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def run(command, log, env=None, cwd=None):
    with log.open("w") as stream:
        result = subprocess.run(command, env=env, cwd=cwd, stdout=stream, stderr=subprocess.STDOUT)
    return dict(command=command, returncode=result.returncode, log=str(log), cwd=str(cwd) if cwd else None)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    root = ROOT
    helper = a.output / "test_static_hwx"
    report = dict(source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=a.source, text=True).strip(), cases=[])
    report["checker_build"] = run(["/usr/bin/clang", "-fobjc-arc", "-O2", str(root / "experimental/test_static_hwx.m"),
                                  "-framework", "Foundation", "-framework", "IOSurface", "-o", str(helper)], a.output / "checker-build.log")
    if report["checker_build"]["returncode"]: raise RuntimeError("static checker build failed")
    source = a.source / "coreml_util.m"
    original = source.read_text()
    changed = original if '@"ANEC_IR_DIR"' in original else original.replace('@"/tmp/espresso_ir_dump/"', '(NSProcessInfo.processInfo.environment[@"ANEC_IR_DIR"] ?: @"/tmp/espresso_ir_dump/")')
    changed = changed.replace('const char* output_base = "/tmp/hwx_output/";', 'const char* output_base = [NSProcessInfo.processInfo.environment[@"ANEC_OUTPUT_ROOT"] UTF8String];')
    changed = changed.replace('flagsDictionary[@"TargetArchitecture"] = @"h13";',
                              'flagsDictionary[@"TargetArchitecture"] = NSProcessInfo.processInfo.environment[@"ANEC_ARCH"] ?: @"h13";\n'
                              '  flagsDictionary[@"DumpStatusDictionaryToFile"] = @YES;\n'
                              '  NSString *optionsFile = NSProcessInfo.processInfo.environment[@"ANEC_OPTIONS_FILE"];\n'
                              '  if (optionsFile) optionsDictionary[@"OptionsFilePath"] = optionsFile;')
    changed = changed.replace('NSDictionary* iDictionary = @{', 'NSMutableDictionary* iDictionary = [@{')
    changed = changed.replace('@"NetworkPlistPath" : temp,\n  };', '@"NetworkPlistPath" : temp,\n  } mutableCopy];\n'
                              '  if ([NSProcessInfo.processInfo.environment[@"ANEC_OPTIONS_PLACEMENT"] isEqual:@"network"])\n'
                              '    iDictionary[@"OptionsFilePath"] = NSProcessInfo.processInfo.environment[@"ANEC_OPTIONS_FILE"];')
    changed = changed.replace('if (optionsFile) optionsDictionary', 'if (optionsFile && ![NSProcessInfo.processInfo.environment[@"ANEC_OPTIONS_PLACEMENT"] isEqual:@"network"]) optionsDictionary')
    source.write_text(changed)
    (a.output / "source.patch").write_text(subprocess.check_output(["git", "diff"], cwd=a.source, text=True))
    report["compiler_build"] = run(["make", "coreml2hwx"], a.output / "compiler-build.log", cwd=a.source)
    if report["compiler_build"]["returncode"]: raise RuntimeError("static compiler build failed")
    import coremltools as ct
    from coremltools.models import datatypes
    from coremltools.models.neural_network import NeuralNetworkBuilder
    builder = NeuralNetworkBuilder([("image", datatypes.Array(64)), ("image2", datatypes.Array(64))],
                                   [("probs", datatypes.Array(64))])
    builder.add_elementwise("multiply", ["image", "image2"], "probs", "MULTIPLY")
    model = a.output / "mul.mlmodel"
    ct.utils.save_spec(builder.spec, str(model))
    import numpy as np
    indices = np.arange(64)
    first, second = ((indices % 17) - 8) / 8, (indices % 11 + 1) / 4
    fixture = a.output / "cpu-reference-inputs.npz"
    np.savez_compressed(fixture, constant_input00=np.full(64, 2, np.float16),
                        constant_input01=np.full(64, 3, np.float16), constant_expected=np.full(64, 6, np.float16),
                        pattern_input00=first.astype(np.float16), pattern_input01=second.astype(np.float16),
                        pattern_expected=(first * second).astype(np.float16))
    report["cpu_reference_fixture"] = dict(path=fixture.name, sha256=hashlib.sha256(fixture.read_bytes()).hexdigest(),
                                           scope="Declared MUL logical inputs and exact mathematical expectations; these are not observed ANE outputs.")
    empty = a.output / "empty-options.plist"
    empty.write_bytes(plistlib.dumps({}))
    system = Path("/System/Library/PrivateFrameworks/VideoProcessing.framework/Versions/A/Resources")
    properties = json.loads((system / "cnn_frame_enhancer_320p.espresso.net").read_text())["properties"]
    candidate = a.output / "observed-system-properties.plist"
    candidate.write_bytes(plistlib.dumps(properties))
    report["candidate_note"] = "Properties copied verbatim from the system Espresso network. Their placement as an options plist is a hypothesis, not a recovered compiler schema."
    report["observed_system_properties"] = properties
    control_dir = a.output / "system-structure"; control_dir.mkdir()
    report["system_structure"] = structure(system / "cnn_frame_enhancer_320p.H13.espresso.hwx", control_dir)
    for arch, label, options, placement in [("h13", "baseline", None, "global"), ("h13g", "baseline", None, "global"),
                                  ("h13", "missing-options", a.output / "absent-options.plist", "global"),
                                  ("h13", "empty-options", empty, "global"),
                                  ("h13", "system-properties", candidate, "global"),
                                  ("h13", "network-missing-options", a.output / "absent-options.plist", "network"),
                                  ("h13", "network-system-properties", candidate, "network")]:
        directory = a.output / f"{arch}-{label}"
        directory.mkdir()
        ir, hwx = directory / "ir", directory / "hwx"
        ir.mkdir(); hwx.mkdir()
        env = dict(os.environ, ANEC_IR_DIR=str(ir) + "/", ANEC_OUTPUT_ROOT=str(hwx) + "/", ANEC_ARCH=arch, ANEC_OPTIONS_PLACEMENT=placement)
        if options is not None: env["ANEC_OPTIONS_FILE"] = str(options)
        task = run([str(a.source / "coreml2hwx"), str(model)], directory / "compile.log", env)
        task.update(architecture=arch, options_case=label, environment={k:v for k,v in env.items() if k.startswith("ANEC_")})
        path = hwx / "mul/model.hwx"
        if path.exists():
            task.update(hwx=str(path), hwx_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), hwx_bytes=path.stat().st_size)
            task["structure"] = structure(path, directory)
            task["execution"] = run([str(helper), str(path), str(directory / "execution")], directory / "execute.log")
            task["pattern_execution"] = run([str(helper), str(path), str(directory / "pattern-execution"), "pattern"], directory / "pattern-execute.log")
        report["cases"].append(task)
        (a.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(task), flush=True)
    for filename in ("hwx/mul.hwx", "hwx/mul_macos14.hwx", "hwx/m1_macOS26/mul.hwx"):
        path = root / filename
        label = filename.replace("/", "-")
        task = run([str(helper), str(path), str(a.output / label)], a.output / (label + ".log"))
        task.update(control=filename, hwx_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        task["structure"] = structure(path, a.output / label)
        task["pattern_execution"] = run([str(helper), str(path), str(a.output / (label + "-pattern")), "pattern"], a.output / (label + "-pattern.log"))
        report["cases"].append(task)
        print(json.dumps(task), flush=True)
    report["options_note"] = "Missing/empty OptionsFilePath are negative controls; successful compilation alone does not prove recognition of a schema. No guessed fields are asserted as valid."
    report["linux_execution"] = "pending native Asahi; macOS runtime acceptance does not establish driver compatibility"
    report["status"] = "completed_attempts"
    report["compiled_variants"] = sum("hwx" in case for case in report["cases"])
    report["numerical_execution_passes"] = sum((case.get("execution", {}).get("returncode") == 0 if "control" not in case else case["returncode"] == 0)
                                              and case.get("pattern_execution", {}).get("returncode") == 0 for case in report["cases"])
    (a.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
