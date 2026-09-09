//! Complete prefill and bounded decode validation across parameters, state edges,
//! artifact emission, and resource binding.

use vyre_model_compiler::config::NamedModelConfig;
use vyre_model_compiler::manifest::{CheckpointManifest, StateEdgeDescriptor};
use vyre_model_compiler::multimodal::{MultimodalConnectorConfig, VisionEncoderConfig};
use vyre_model_compiler::pipeline::ModelCompiler;
use vyre_model_compiler::tokenizer::TokenizerSchema;
use vyre_model_compiler::workload::WorkloadEnvelope;

#[test]
fn prefill_pipeline_validates_parameters_and_emits_artifact() {
    let mut config = NamedModelConfig::DeepSeekV4Flash.config();
    config.num_layers = 2; // Representative 2-layer slice

    let manifest = CheckpointManifest::from_config(&config);
    assert!(
        manifest.validate_against_config(&config).is_ok(),
        "Generated manifest must match model config parameters"
    );

    let workload = WorkloadEnvelope::prefill(1, 16, config.max_seq_len);

    let compiled = ModelCompiler::compile_model(&config, &workload)
        .expect("Prefill compilation must succeed");

    assert!(!compiled.digest().is_empty());
    assert!(compiled.entry_count() > 0);
    assert!(compiled.required_resource_bytes() > 0);

    // Admit artifact and bind resources
    let session = ModelCompiler::admit_model(&compiled, &manifest)
        .expect("Artifact session admission and resource binding must succeed");

    assert!(
        session.len() > 0,
        "Admitted dataset must hold resource buffers"
    );
}

#[test]
fn decode_pipeline_validates_state_edges_and_emits_artifact() {
    let mut config = NamedModelConfig::Llama3_1_8B.config();
    config.num_layers = 2; // Representative 2-layer slice

    let state_edges = StateEdgeDescriptor::for_model(&config, 1);
    assert_eq!(
        state_edges.len(),
        4, // 2 layers * (K cache + V cache)
        "Expected 4 state edges for 2 LLaMA layers"
    );

    let manifest = CheckpointManifest::from_config(&config);
    let workload = WorkloadEnvelope::decode(1, 128, config.max_seq_len);

    let compiled = ModelCompiler::compile_model(&config, &workload)
        .expect("Decode compilation must succeed");

    assert!(!compiled.digest().is_empty());
    assert!(compiled.entry_count() > 0);

    let session = ModelCompiler::admit_model(&compiled, &manifest)
        .expect("Artifact session admission and resource binding must succeed for decode");

    assert!(session.len() > 0);
}

#[test]
fn tokenizer_schemas_validate_vocabulary_bounds() {
    let deepseek_tok = TokenizerSchema::deepseek();
    assert!(deepseek_tok.validate_tokens(&[0, 1, 100, 129_279]).is_ok());
    assert!(deepseek_tok.validate_tokens(&[130_000]).is_err());

    let llama_tok = TokenizerSchema::llama3();
    assert!(llama_tok.validate_tokens(&[128_000, 128_001, 128_009]).is_ok());
    assert!(llama_tok.validate_tokens(&[200_000]).is_err());
}

#[test]
fn multimodal_vision_connector_translates_cleanly() {
    let vision_cfg = VisionEncoderConfig::clip_vit_large_14();
    assert_eq!(vision_cfg.num_patches(true), 257); // 256 patches + 1 CLS token

    let connector = MultimodalConnectorConfig::llava_mlp(1024, 4096);
    assert_eq!(connector.vision_hidden_dim, 1024);
    assert_eq!(connector.text_hidden_dim, 4096);
}
