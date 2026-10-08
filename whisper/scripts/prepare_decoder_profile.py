"""Add diagnostic vocabulary stack labels to the pinned shared Mac preparation."""
import argparse
from pathlib import Path
import subprocess

from whisper.native import REVISION, accuracy_patch, instrument_patch, replace_once


def patch_whisper(text):
    text = replace_once(text,"    struct ggml_tensor * logits = ggml_mul_mat(ctx0, model.d_te, cur);",
        '    struct ggml_tensor * logits = ggml_mul_mat(ctx0, model.d_te, cur);\n'
        '    ggml_set_name(logits, "whisper.vocabulary");')
    for name in ("whisper_compute_logprobs","whisper_compute_probs"):
        text = replace_once(text,"static void "+name+"(","static __attribute__((noinline)) void "+name+"(")
    return text


def patch_cpu(text):
    before = "static void ggml_compute_forward(struct ggml_compute_params * params, struct ggml_tensor * tensor) {"
    wrappers = '''// Diagnostic frames: use the same dispatcher/arithmetic, with no tail call.
static void whisper_profile_vocabulary_metadata(const struct ggml_compute_params * params, const struct ggml_tensor * dst) {
    if (params->ith != 0 || !getenv("WHISPER_PROFILE_VOCABULARY")) return;
    const struct ggml_tensor * w = dst->src[0];
    const struct ggml_tensor * x = dst->src[1];
    fprintf(stderr, "VOCABULARY_PROFILE\\t{\\"phase\\":\\"%s\\",\\"m\\":%lld,\\"n\\":%lld,\\"k\\":%lld,"
        "\\"weight\\":\\"%s\\",\\"weight_type\\":\\"%s\\",\\"input_type\\":\\"%s\\",\\"output_type\\":\\"%s\\","
        "\\"threads\\":%d,\\"weight_strides\\":[%zu,%zu,%zu,%zu],\\"input_strides\\":[%zu,%zu,%zu,%zu],"
        "\\"output_strides\\":[%zu,%zu,%zu,%zu]}\\n",
        x->ne[1] == 1 ? "token" : "prompt", (long long)x->ne[1], (long long)w->ne[1], (long long)x->ne[0],
        w->name, ggml_type_name(w->type), ggml_type_name(x->type), ggml_type_name(dst->type), params->nth,
        w->nb[0], w->nb[1], w->nb[2], w->nb[3], x->nb[0], x->nb[1], x->nb[2], x->nb[3],
        dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3]);
}

static __attribute__((noinline)) void whisper_profile_vocabulary_token(const struct ggml_compute_params * params, struct ggml_tensor * dst) {
    whisper_profile_vocabulary_metadata(params, dst);
    ggml_compute_forward_mul_mat(params, dst);
    __asm__ __volatile__("nop" ::: "memory"); // prevent identical-code folding with the prompt frame
}

static __attribute__((noinline)) void whisper_profile_vocabulary_prompt(const struct ggml_compute_params * params, struct ggml_tensor * dst) {
    whisper_profile_vocabulary_metadata(params, dst);
    ggml_compute_forward_mul_mat(params, dst);
    __asm__ __volatile__("" ::: "memory");
}

'''
    text = replace_once(text,before,wrappers+before)
    return replace_once(text,"""        case GGML_OP_MUL_MAT:
            {
                ggml_compute_forward_mul_mat(params, tensor);
            } break;""","""        case GGML_OP_MUL_MAT:
            {
                if (strcmp(tensor->name, "whisper.vocabulary") == 0) {
                    if (tensor->src[1]->ne[1] == 1) whisper_profile_vocabulary_token(params, tensor);
                    else whisper_profile_vocabulary_prompt(params, tensor);
                } else ggml_compute_forward_mul_mat(params, tensor);
            } break;""")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path,required=True)
    args = parser.parse_args()
    source = args.source.resolve()
    if subprocess.check_output(["git","-C",str(source),"rev-parse","HEAD"],text=True).strip() != REVISION:
        parser.error("requires the pinned isolated whisper.cpp worktree")
    changes = []
    for name in ("src/whisper.cpp","ggml/src/ggml-cpu/ggml-cpu.c"):
        path = source/name
        original = subprocess.check_output(["git","-C",str(source),"show","HEAD:"+name],text=True)
        before = instrument_patch(accuracy_patch(original)) if name.endswith("whisper.cpp") else original
        after = patch_whisper(before) if name.endswith("whisper.cpp") else patch_cpu(before)
        if path.read_text() not in (before,after):
            parser.error("unknown source edits in "+str(path))
        changes.append((path,after))
    for path,text in changes:
        if path.read_text() != text:
            path.write_text(text)
    print("Prepared diagnostic vocabulary and probability stack labels:",source)


if __name__ == "__main__":
    main()
