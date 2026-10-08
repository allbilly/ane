#!/usr/bin/env python3
"""Connect regenerated PR encoder packets to an isolated pinned whisper.cpp."""
import argparse
from pathlib import Path
import subprocess

from whisper.native import REVISION, replace_once


def source_patch(text):
    text = replace_once(text, '#include "aneforge/whisper-aneforge.h"',
                        '#include "aneforge/whisper-aneforge.h"\n#include "asahi_full_encoder.h"')
    text = replace_once(text, 'if (const char * aneforge_dir = getenv("ANEFORGE_ENCODER")) {',
                        'if (const char * aneforge_dir = getenv("WHISPER_ASAHI_ENCODER")) {')
    text = replace_once(text, '        WHISPER_LOG_INFO("%s: compiling for the ANE (one time) ...\\n", __func__);',
                        '        WHISPER_LOG_INFO("%s: loading regenerated Asahi packets ...\\n", __func__);')
    anchor = '        WHISPER_LOG_INFO("%s: ANEForge encoder loaded\\n", __func__);'
    text = replace_once(text, anchor, '''        const auto & hp = ctx->model.hparams;
        if (ctx->params.use_gpu || !whisper_asahi_full_model_matches(state->ctx_aneforge,
                hp.n_mels, hp.n_audio_ctx, hp.n_audio_state, hp.n_audio_layer)) {
            WHISPER_LOG_ERROR("%s: Asahi PR encoder requires matching CPU-only model dimensions\\n", __func__);
            whisper_free_state(state);
            return nullptr;
        }
        WHISPER_LOG_INFO("%s: Asahi PR encoder loaded\\n", __func__);''')
    return text


def cmake_patch(text):
    text = replace_once(text, "            aneforge/whisper-aneforge.cpp", '            "${ANE_ROOT}/whisper/asahi_full_encoder.cpp"')
    return text + '''
# Native PR packet replay; preserve upstream CPU math and external-encoder API.
if (NOT ANE_ROOT)
    message(FATAL_ERROR "Set ANE_ROOT to the parent ane repository")
endif()
target_include_directories(whisper PRIVATE "${ANE_ROOT}/whisper")
find_package(OpenMP COMPONENTS CXX)
if (OpenMP_CXX_FOUND)
    target_link_libraries(whisper PRIVATE OpenMP::OpenMP_CXX)
endif()
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Isolated worktree at the pinned revision")
    args = parser.parse_args()
    source = args.source.resolve()
    actual = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if actual != REVISION:
        parser.error("requires isolated whisper.cpp worktree at " + REVISION)
    changes = []
    for name, transform in (("src/whisper.cpp", source_patch), ("src/CMakeLists.txt", cmake_patch)):
        before = subprocess.check_output(["git", "-C", str(source), "show", "HEAD:" + name], text=True)
        after = transform(before)
        path = source / name
        if path.read_text() not in (before, after):
            raise ValueError("refusing to overwrite other edits in " + str(path))
        changes.append((path, after))
    for path, text in changes:
        if path.read_text() != text:
            path.write_text(text)
    print("Prepared native PR encoder adapter:", source)


if __name__ == "__main__":
    main()
