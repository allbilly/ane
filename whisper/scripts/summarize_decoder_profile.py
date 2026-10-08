"""Map observed vocabulary/softmax stack subtrees to selected native assembly."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
import subprocess

from whisper.validation import digest

ROLES = ("whisper_profile_vocabulary_token","whisper_profile_vocabulary_prompt",
         "whisper_compute_logprobs","whisper_compute_probs","ggml_compute_forward_flash_attn_ext_f16_one_chunk")


def sample_nodes(text):
    graph = text.split("Call graph:\n",1)[1].split("Total number in stack",1)[0]
    nodes,stack = [],[]
    pattern = r"^([ +!|:]*)(\d+) (.+?)  \(in ([^)]+)\)(.*)$"
    for line in graph.splitlines():
        match = re.match(pattern,line)
        if not match:
            continue
        prefix,count,symbol,image,tail = match.groups()
        depth = len(prefix)
        while stack and nodes[stack[-1]]["depth"] >= depth:
            stack.pop()
        node = dict(count=int(count),symbol=symbol,image=image,tail=tail,depth=depth,
                    parent=stack[-1] if stack else None,children=[])
        index = len(nodes)
        if stack:
            nodes[stack[-1]]["children"].append(index)
        nodes.append(node)
        stack.append(index)
    for node in nodes:
        node["self_count"] = node["count"]-sum(nodes[i]["count"] for i in node["children"])
        if node["self_count"] < 0:
            raise ValueError("sample stack-tree counts are inconsistent")
    return nodes


def images(text):
    result = {}
    for line in text.split("Binary Images:\n",1)[1].splitlines():
        match = re.match(r"\s*(0x[0-9a-f]+) -\s*(0x[0-9a-f]+)\s+\+?(\S+) .*?<([A-F0-9-]+)> (.*)",line)
        if match:
            start,end,name,uuid,path = match.groups()
            result[name] = dict(load_address=start,end_address=end,uuid=uuid,path=path)
    return result


def summarize(text):
    nodes = sample_nodes(text)
    mapped = images(text)
    roles = {role:dict(inclusive_stack_count=0,self_stack_count=0,repeated_frame_count=0,leaves=Counter()) for role in ROLES}
    frames = {}
    for index,node in enumerate(nodes):
        ancestors,index0 = [],index
        while index0 is not None:
            ancestor = nodes[index0]
            ancestors.append(ancestor["symbol"])
            index0 = ancestor["parent"]
        for role in ROLES:
            if role in node["symbol"]:
                field = "repeated_frame_count" if any(role in name for name in ancestors[1:]) else "inclusive_stack_count"
                roles[role][field] += node["count"]
                roles[role]["self_stack_count"] += node["self_count"]
        if not node["self_count"]:
            continue
        for role in ROLES:
            if any(role in name for name in ancestors):
                roles[role]["leaves"][(node["symbol"],node["image"])] += node["self_count"]
        if node["symbol"] == "???":
            continue
        offset = re.search(r" \+ (\d+)(?:[, ]).*?\[(0x[0-9a-f]+)",node["tail"])
        if offset and node["image"] in mapped:
            value = int(offset[2],16)-int(mapped[node["image"]]["load_address"],16)-int(offset[1])
            key = (node["symbol"],node["image"])
            if key in frames and frames[key] != value:
                raise ValueError("sample symbol maps to inconsistent file offsets")
            frames[key] = value
    for data in roles.values():
        data["leaves"] = [dict(symbol=symbol,image=image,self_stack_count=count,
            file_offset=hex(frames[symbol,image]) if (symbol,image) in frames else None)
            for (symbol,image),count in data["leaves"].most_common()]
        if sum(row["self_stack_count"] for row in data["leaves"]) != data["inclusive_stack_count"]:
            raise ValueError("attributed leaf counts do not cover their role subtree")
    return dict(roles=roles,images=mapped)


def selected_stack_paths(text, result):
    nodes = sample_nodes(text)
    lines = [text.split("Call graph:",1)[0],
             "Representative stack paths. Aggregate parent nodes may occur in multiple paths; counts must not be added across these excerpts."]
    for role,data in result["roles"].items():
        for leaf in data["leaves"][:3]:
            candidates = []
            for index,node in enumerate(nodes):
                if (node["symbol"],node["image"]) != (leaf["symbol"],leaf["image"]) or not node["self_count"]:
                    continue
                path,j = [],index
                while j is not None:
                    path.append(nodes[j])
                    j = nodes[j]["parent"]
                if any(role in frame["symbol"] for frame in path):
                    candidates.append((node["self_count"],path))
            _,path = max(candidates,key=lambda item:item[0])
            lines.append("\nRole: "+role+"; leaf: "+leaf["symbol"])
            for depth,node in enumerate(reversed(path)):
                lines.append("  "*depth+str(node["count"])+" "+node["symbol"]+"  (in "+node["image"]+")"+node["tail"])
    lines.append("\nSelected loaded images:")
    for line in text.split("Binary Images:\n",1)[1].splitlines():
        if any(name in line for name in ("benchmark-whisper","libggml-cpu","libggml-blas","libwhisper","libsystem_m.dylib","libBLAS.dylib","libane_e5rt")):
            lines.append(line)
    return "\n".join(lines)+"\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile",type=Path,required=True)
    parser.add_argument("--build",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    raw = json.loads((args.profile/"summary.json").read_text())
    if raw["status"] != "PASS_DIAGNOSTIC" or len(raw["boundaries"]) != 6:
        parser.error("requires completed three-clip native diagnostics")
    for filename,checksum in raw["artifacts"].items():
        if digest(args.profile/filename) != checksum:
            parser.error("profile artifact changed: "+filename)
    args.output.mkdir(parents=True,exist_ok=False)
    report = dict(status="mapped",source_summary_sha256=digest(args.profile/"summary.json"),
        scope="Inclusive and leaf stack counts by diagnostic role. Counts include worker waits, profiling metadata and sampling overhead; they are not CPU-cycle shares or end-to-end wall-time shares.",
        full_logit_gate_nrmse=.005,numerical_failures=raw["numerical_failures"],correctness=raw["correctness"],
        boundaries=raw["boundaries"],profiles={},hashes=raw["hashes"],source_sha256=raw["source_sha256"])
    kernels = {}
    for clip,data in raw["profiles"].items():
        path = args.profile/(clip+"-sample")/"sample.txt"
        result = summarize(path.read_text())
        if any(not role["inclusive_stack_count"] for role in result["roles"].values()):
            raise ValueError("a requested diagnostic role was not observed: "+clip)
        for role in ROLES[:2]:
            for leaf in result["roles"][role]["leaves"]:
                if "tinyBLAS" in leaf["symbol"] and leaf["file_offset"]:
                    kernels[leaf["symbol"]] = leaf["file_offset"]
        result.update(transcriptions=data["transcriptions"],sample_sha256=data["sample_sha256"],environment=data["environment"])
        report["profiles"][clip] = result
        # Keep representative full paths and the relevant loaded-image map.
        text = path.read_text()
        (args.output/(clip+"-selected-stacks.txt")).write_text(selected_stack_paths(text,result))
    driver = args.build/"bin/benchmark-whisper"
    for name,checksum in raw["hashes"]["libraries"].items():
        if digest(args.build/"bin"/name) != checksum:
            parser.error("profiled native library changed: "+name)
    commands = ["target create "+str(driver.resolve()),"image list -u -h -f libggml-cpu.0.dylib libwhisper.1.dylib libsystem_m.dylib"]
    for role in ROLES:
        commands += ["image lookup -n "+role,"disassemble -b -n "+role]
    for symbol,address in sorted(kernels.items()):
        commands += ["image lookup -a "+address,"disassemble -b -a "+address]
    for name in ("ggml_vec_dot_f32","expf","logf"):
        commands += ["image lookup -n "+name,"disassemble -b -n "+name]
    command = ["/usr/bin/lldb","-b"]
    for value in commands:
        command.extend(("-o",value))
    process = subprocess.run(command,capture_output=True,text=True,timeout=60)
    (args.output/"selected-assembly.txt").write_text(process.stdout+process.stderr)
    process.check_returncode()
    report["selected_vocabulary_kernels"] = kernels
    report["collector_source_sha256"] = digest(Path(__file__))
    report["artifacts"] = {p.name:digest(p) for p in args.output.iterdir() if p.is_file()}
    (args.output/"summary.json").write_text(json.dumps(report,indent=2)+"\n")
    print("Mapped decoder profile:",args.output)


if __name__ == "__main__":
    main()
