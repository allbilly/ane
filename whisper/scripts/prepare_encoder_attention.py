"""Patch an isolated shared Mac FP32 worktree for portable batched attention."""
import argparse
from pathlib import Path
import subprocess

from whisper.attention import patch_blas, patch_source
from whisper.native import REVISION, accuracy_patch, blas_profile_patch, instrument_patch
from whisper.scripts.prepare_macos_precision import patch_source as precision_patch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path,required=True)
    args = parser.parse_args()
    source = args.source.resolve()
    if subprocess.check_output(["git","-C",str(source),"rev-parse","HEAD"],text=True).strip() != REVISION:
        parser.error("requires the pinned isolated whisper.cpp worktree")
    changes = []
    for name in ("src/whisper.cpp","ggml/src/ggml-blas/ggml-blas.cpp"):
        path = source/name
        text = path.read_text()
        original = subprocess.check_output(["git","-C",str(source),"show","HEAD:"+name],text=True)
        if name == "src/whisper.cpp":
            before = instrument_patch(accuracy_patch(original))
            if "WhisperMacPrecision" in text:
                before = precision_patch(before)
            after = patch_source(before)
        else:
            before = blas_profile_patch(original)
            after = patch_blas(before)
        if text not in (before,after):
            parser.error("unknown edits in "+str(path))
        changes.append((path,after))
    for path,text in changes:
        if path.read_text() != text:
            path.write_text(text)
    print("Prepared opt-in contiguous FP32 encoder attention:",source)


if __name__ == "__main__":
    main()
