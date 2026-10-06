// Instrumentation only: call the unmodified Uzu CPU decoder and export logits.
// Install as engine/language_model/ane_reference.rs in the pinned Uzu checkout.
use std::{fs::File, io::Write, path::Path};
use half::bf16;
use crate::{
    backends::{cpu::Cpu, common::{BufferMut, BufferRef, Context, CommandBufferEncoding,
        CommandBufferExecutable, CommandBufferPending, gpu_types::trie::TrieNode}},
    encodable_block::batch_topology::BatchTopology,
    engine::language_model::LanguageModel,
};

impl LanguageModel<Cpu> {
    pub fn ane_reference(&self, tokens: &[u32], output: &Path) -> Result<(), Box<dyn std::error::Error>> {
        let context = &self.engine.context;
        let mut state = self.create_empty_state(Some(tokens.len() as u32 + 32), 0)?;
        let mut file = File::create(output)?;
        let mut layer_file = File::create(output.with_extension("layers.bin"))?;
        let capture_all = std::env::var_os("ANE_REFERENCE_ALL_TOKENS").is_some();
        for (position, &token) in tokens.iter().enumerate() {
            state.transformer_state.prepare(position as u32, 1, context)?;
            let mut command = context.create_command_buffer(Some("reference"), None)?;
            let mut input = context.create_buffer(4)?;
            (&mut input).copyin(&[token]);
            let nodes = [TrieNode {trie_start: 0, trie_end: 1, height: 0}];
            let batch = BatchTopology::new(&nodes, true);
            let range = if capture_all || position + 1 == tokens.len() { Some((0..1).into()) } else {None};
            let indices: Vec<u32> = (0..24).collect();
            let capture = if capture_all || position + 1 == tokens.len() {Some(indices.as_slice())} else {None};
            let output = self.decoder.encode(&input, &batch, range, capture, &mut state.transformer_state, &mut command)?;
            state.transformer_state.encode_accept(&[0], &mut command)?;
            command.end_encoding().submit().wait_until_completed()?;
            if let Some(logits) = output.logits {
                for &value in (&logits).as_slice::<bf16>() {
                    file.write_all(&value.to_f32().to_le_bytes())?;
                }
            }
            if let Some(features) = output.hidden_features {
                for feature in &features {
                    for &value in feature.as_slice::<bf16>() {
                        layer_file.write_all(&value.to_f32().to_le_bytes())?;
                    }
                }
            }
            eprintln!("reference {}/{}", position + 1, tokens.len());
        }
        Ok(())
    }
}
