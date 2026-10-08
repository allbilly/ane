"""Portable, opt-in encoder attention graph and strict execution evidence."""
import json
import re

from whisper.native import replace_once


def patch_source(text):
    start = text.index("static struct ggml_cgraph * whisper_build_graph_encoder(")
    end = text.index("// pre-compute cross-attention memory", start)
    graph = text[start:end]
    graph = replace_once(graph, "    const int n_state_head = n_state/n_head;", """    const int n_state_head = n_state/n_head;
    const bool encoder_blas = whisper_encoder_blas_attention_enabled();
    if (encoder_blas && (wctx.params.use_gpu || wctx.itype != GGML_TYPE_F32 ||
        n_ctx != 1500 || n_state != 384 || n_head != 6 || n_layer != 4))
        GGML_ABORT("batched encoder attention requires tiny.en with FP32 CPU host");""")
    graph = replace_once(graph, "            if (wctx.params.flash_attn) {",
                         "            if (wctx.params.flash_attn && !encoder_blas) {")
    graph = replace_once(graph, "                // K * Q\n", """                // BLAS requires contiguous operands, including each head plane.
                if (encoder_blas) {
                    K = ggml_cont(ctx0, K);
                    Q = ggml_cont(ctx0, Q);
                }
                // K * Q
""")
    graph = replace_once(graph, "                struct ggml_tensor * KQ = ggml_mul_mat(ctx0, K, Q);",
        """                struct ggml_tensor * KQ = ggml_mul_mat(ctx0, K, Q);
                if (encoder_blas) ggml_format_name(KQ, "whisper.encoder_attention.%d.qk", il);""")
    graph = replace_once(graph, "                struct ggml_tensor * KQV = ggml_mul_mat(ctx0, V, KQ_soft_max);",
        """                if (encoder_blas) V = ggml_cont(ctx0, V);
                struct ggml_tensor * KQV = ggml_mul_mat(ctx0, V, KQ_soft_max);
                if (encoder_blas) ggml_format_name(KQV, "whisper.encoder_attention.%d.pv", il);""")
    text = text[:start]+graph+text[end:]
    return replace_once(text, '    whisper_trace_tensor("encoder.f32", wstate.embd_enc);',
        '    if (whisper_encoder_blas_attention_enabled())\n'
        '        std::fprintf(stderr, "ENCODER_ATTENTION encoder: layers=4 heads=6 positions=1500 implementation=blas\\n");\n'
        '    whisper_trace_tensor("encoder.f32", wstate.embd_enc);')


def patch_blas(text):
    text = replace_once(text, "const bool profile = whisper_profile_matrix(dst);",
                         "const bool profile = whisper_profile_matrix(dst) || whisper_profile_attention_matrix(dst);")
    return replace_once(text, "if (profile) whisper_profile_matrix_input(src1);",
                         "if (whisper_profile_matrix(dst)) whisper_profile_matrix_input(src1);")


def attention_profiles(log, enabled, encodes=1):
    markers = re.findall(r"ENCODER_ATTENTION encoder: layers=(\d+) heads=(\d+) positions=(\d+) implementation=(\w+)", log)
    records = [json.loads(line.split("\t",1)[1]) for line in log.splitlines()
               if line.startswith("ATTENTION_MATRIX_PROFILE\t")]
    if not enabled:
        if markers or records:
            raise ValueError("flash encoder unexpectedly used batched attention")
        return []
    if markers != [("4","6","1500","blas")]*encodes or len(records) != 8*encodes:
        raise ValueError("missing actual encoder attention execution evidence")
    expected = {f"whisper.encoder_attention.{i}.{kind}" for i in range(4) for kind in ("qk","pv")}
    for offset in range(0,len(records),8):
        if {r["name"] for r in records[offset:offset+8]} != expected:
            raise ValueError("missing or duplicate encoder attention product")
    for row in records:
        dimensions = (1500,1500,64) if row["name"].endswith("qk") else (1500,64,1500)
        if (row["m"],row["n"],row["k"]) != dimensions or row["batch"] != 6:
            raise ValueError("unexpected encoder attention dimensions/head count")
        if (row["backend"] not in ("Accelerate","OpenBLAS") or row["routine"] != "cblas_sgemm" or
                row["requested_threads"] != 4 or (row["backend"] == "OpenBLAS" and row["blas_threads"] != 4)):
            raise ValueError("attention did not execute on the requested BLAS backend")
        if (row["order"] != "row_major" or row["transpose_a"] or not row["transpose_b"] or
                any(row[key] != "f32" for key in ("weight_type","input_type","output_type","gemm_type"))):
            raise ValueError("attention arithmetic/packing changed")
        stages = ("allocate_us","convert_us","thread_setup_us","gemm_us")
        if (any(type(row[key]) is not int or row[key] < 0 for key in (*stages,"total_us")) or
                sum(row[key] for key in stages) != row["total_us"]):
            raise ValueError("invalid attention timing")
    return records


def add_attention_profiles(clips, stderr, enabled):
    sections = stderr.split("BENCH_AUDIO\t")[1:]
    if len(sections) != len(clips):
        raise ValueError("missing attention audio boundaries")
    for index, rows in enumerate(clips.values()):
        number, log = sections[index].split("\n",1)
        if number != str(index):
            raise ValueError("attention audio order changed")
        blocks = {(phase,int(number)):text for phase,number,text in re.findall(
            r"BENCH_BEGIN\t(\w+)\t(\d+)\n(.*?)BENCH_END\t\1\t\2",log,re.S)}
        if len(blocks) != len(rows):
            raise ValueError("attention repetition count changed")
        for row in rows:
            records = attention_profiles(blocks[row["phase"],row["index"]],enabled)
            if enabled:
                row["attention_matrices"] = records
