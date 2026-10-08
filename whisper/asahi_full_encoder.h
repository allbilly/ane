#pragma once
#include <stdint.h>

struct whisper_aneforge_context;

// Called by the isolated whisper.cpp adapter before allocating encoder graphs.
bool whisper_asahi_full_model_matches(const whisper_aneforge_context * ctx,
                                     int n_mels, int n_ctx, int n_state, int n_layers);
