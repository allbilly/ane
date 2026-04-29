"""expt3: Minimal register set per op — line-by-line method.

For each examples_expt/*.py file, comment out one register block at a time,
run the standalone file, check if output is still correct.
If correct → keep commented (UNNEEDED). If wrong → revert (ESSENTIAL).

Multi-line register expressions are handled as single blocks.
State is managed by tracking which blocks are commented, regenerating
the file from scratch each time (no mutable original variable bugs).
"""
import os, sys, re, subprocess, shutil, time

OP_CHECKS = {
    "relu.py": {
        "check_stdout": lambda out: "output = [0. 5. 0. 5." in out,
        "description": "output = [0, 5, 0, 5, ...]",
    },
    "sigmoid.py": {
        "check_stdout": lambda out: "output[0] = 0.95263671875" in out,
        "description": "output[0] = 0.952637",
    },
    "elementwise.py": {
        "check_stdout": lambda out: "output = [5. 5. 5. 5." in out,
        "description": "output = [5, 5, 5, 5, ...]",
    },
}

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXPT_DIR = os.path.join(PROJECT_DIR, "examples_expt")

# Registers that permanently wedge the device when zeroed.
# Skip testing them — they are known ESSENTIAL from expt1/expt2.
POISON_PILL = {
    'SrcFmt', 'DstFmt', 'SrcDMAConfig', 'DstDMAConfig',
    'ConvCfg', 'GroupConvCfg', 'TileCfg', 'Cfg',
    'ChCfg', 'pad0', 'pad1', 'pad2', 'pad3',
    'Cin', 'Cout', 'InDim', 'OutDim',
    'PECfg', 'MACCfg',
    'SourceChannelStride', 'ResultCfg',
    'Srcpad0',  # Secondary DMA config — zeroing wedges device hard
}

PLATFORM_SKIP = {
    'CommonStream', 'SrcStream', 'L2Stream', 'PEStream', 'NEStream', 'DstStream',
    'KernelDMA', 'W0', 'W4', 'W6', 'W8',
}

ZERO_SKIP = {
    'SrcBaseAddr', 'DstBaseAddr', 'SrcGroupStride', 'DstGroupStride',
    'DPE', 'MatrixVectorBias', 'AccBias',
    'Srcpad1', 'Srcpad2', 'Srcpad3', 'Srcpad4', 'Srcpad5', 'Srcpad6', 'Srcpad7', 'Srcpad8',
    'L2pad2', 'L2pad3', 'pad4',
}

def should_skip(reg_name, stripped_text):
    if reg_name in PLATFORM_SKIP:
        return True
    if reg_name in POISON_PILL:
        return True
    if reg_name in ZERO_SKIP and ', 0)' in stripped_text:
        return True
    return False

def find_reg_blocks(filepath):
    """Returns list of (start_line, end_line, reg_name)."""
    lines = open(filepath).read().split('\n')
    blocks = []
    in_btsp = False
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        if 'BTSP_BUF = make_from_segments' in stripped:
            in_btsp = True
        if in_btsp and stripped.startswith('(reg.') and 'stream_header' not in stripped:
            m = re.search(r'reg\.(\w+)', stripped)
            reg_name = m.group(1) if m else "?"
            if should_skip(reg_name, stripped):
                i += 1
                continue
            
            start = i
            paren_depth = stripped.count('(') - stripped.count(')')
            i += 1
            while i < len(lines) and paren_depth > 0:
                s = lines[i].strip()
                if s.startswith('(reg.') and 'stream_header' not in s:
                    m2 = re.search(r'reg\.(\w+)', s)
                    n2 = m2.group(1) if m2 else ""
                    if not should_skip(n2, s):
                        break
                paren_depth += s.count('(') - s.count(')')
                if paren_depth <= 0:
                    i += 1
                    break
                i += 1
            
            blocks.append((start, i - 1, reg_name))
            continue
        i += 1
    return blocks, lines

def apply_comments_to_file(src_filepath, dst_filepath, commented_blocks):
    """Copy src to dst and comment out all lines in commented_blocks."""
    lines = open(src_filepath).read().split('\n')
    for start, end in sorted(commented_blocks):
        for li in range(start, end + 1):
            lines[li] = '# UNNEEDED (expt3): ' + lines[li]
    open(dst_filepath, 'w').write('\n'.join(lines))

def test_file(filename):
    filepath = os.path.join(EXPT_DIR, filename)
    try:
        result = subprocess.run(
            [sys.executable, filepath],
            capture_output=True, text=True, timeout=15,
            cwd=PROJECT_DIR
        )
        stdout = result.stdout
        check = OP_CHECKS[filename]["check_stdout"]
        return check(stdout)
    except subprocess.TimeoutExpired:
        return "HANG"
    except Exception as e:
        return "ERROR"

def run_expt3_for_op(filename):
    src_path = os.path.join(PROJECT_DIR, 'examples', filename)
    dst_path = os.path.join(EXPT_DIR, filename)
    
    print(f"\n{'='*80}")
    print(f"expt3: {filename}  —  line-by-line (fresh copy each test)")
    print(f"{'='*80}")
    
    blocks, _ = find_reg_blocks(src_path)
    print(f"Found {len(blocks)} register blocks to test")
    for start, end, name in blocks:
        print(f"  L{start+1}-L{end+1}: {name}")
    
    # Baseline test
    shutil.copy(src_path, dst_path)
    print("Baseline test... ", end="", flush=True)
    result = test_file(filename)
    if result != True:
        print(f"FAIL ({result}) — aborting")
        return
    print("OK")
    
    results = {"UNNEEDED": [], "ESSENTIAL": [], "HANG": []}
    commented = set()  # set of (start, end) tuples to keep commented
    
    for idx, (start, end, reg_name) in enumerate(blocks):
        print(f"  [{idx+1}/{len(blocks)}] {reg_name}: ", end="", flush=True)
        
        # Build file from scratch: all currently-commented blocks + this test block
        test_comments = commented | {(start, end)}
        apply_comments_to_file(src_path, dst_path, test_comments)
        
        result = test_file(filename)
        
        if result == True:
            print("UNNEEDED")
            results["UNNEEDED"].append(reg_name)
            commented.add((start, end))
        else:
            if result == "HANG":
                print("HANG")
                results["HANG"].append(reg_name)
            else:
                print("ESSENTIAL")
                results["ESSENTIAL"].append(reg_name)
            # Reset device by running the baseline TWICE
            # (first run may wake the device, second confirms it's good)
            shutil.copy(src_path, dst_path)
            test_file(filename)
            time.sleep(0.5)
    
    # Write final state
    apply_comments_to_file(src_path, dst_path, commented)
    
    # Verify final state
    print(f"\n{filename} results:")
    print(f"  UNNEEDED ({len(results['UNNEEDED'])}): {results['UNNEEDED']}")
    print(f"  ESSENTIAL ({len(results['ESSENTIAL'])}): {results['ESSENTIAL']}")
    if results["HANG"]:
        print(f"  HANG ({len(results['HANG'])}): {results['HANG']}")
    
    print(f"Final verification... ", end="", flush=True)
    result = test_file(filename)
    if result == True:
        print("OK")
    else:
        print(f"FAIL ({result})")
        # Debug: get actual output
        import subprocess as sp2
        r2 = sp2.run([sys.executable, dst_path], capture_output=True, text=True, timeout=15, cwd=PROJECT_DIR)
        print(f"  rc: {r2.returncode}")
        print(f"  stdout[:200]: {repr(r2.stdout[:200])}")
        print(f"  stderr[:500]: {repr(r2.stderr[:500])}")
        print("  RESTORING ORIGINAL")
        shutil.copy(src_path, dst_path)
    
    return results

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        run_expt3_for_op(sys.argv[1])
    else:
        for op in ["relu.py", "sigmoid.py", "elementwise.py"]:
            run_expt3_for_op(op)
