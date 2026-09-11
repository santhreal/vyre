//! Complete prefill and bounded decode validation across parameters, state edges,
//! artifact emission, resource binding, and semantic oracle execution.

use vyre_model_compiler::config::{ModelConfig, ModelFamily, NamedModelConfig, NormKind};
use vyre_model_compiler::manifest::{
    CheckpointManifest, ManifestError, StateEdgeDescriptor, TensorDescriptor,
};
use vyre_model_compiler::multimodal::{
    MultimodalConnectorConfig, MultimodalError, VisionEncoderConfig,
};
use vyre_model_compiler::pipeline::ModelCompiler;
use vyre_model_compiler::tokenizer::{TokenizerError, TokenizerSchema};
use vyre_model_compiler::translator::{ModelGraphBuilder, TranslationError};
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

    let compiled =
        ModelCompiler::compile_model(&config, &workload).expect("Prefill compilation must succeed");

    // Contract validations
    assert_eq!(
        compiled.envelope.neutral(),
        &compiled.artifact,
        "Artifact envelope must hold neutral artifact"
    );
    assert!(
        compiled.artifact.validate_abi().is_ok(),
        "Artifact ABI must validate"
    );
    assert!(compiled.node_count() > 0, "Artifact must contain nodes");
    assert!(
        compiled.entry_count() > 0,
        "Artifact must contain entry points"
    );
    assert!(
        compiled.required_resource_bytes() > 0,
        "Artifact must declare resource bytes"
    );

    // Determinism
    let compiled_again = ModelCompiler::compile_model(&config, &workload)
        .expect("Deterministic re-compile must succeed");
    assert_eq!(compiled.digest(), compiled_again.digest());

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

    let compiled =
        ModelCompiler::compile_model(&config, &workload).expect("Decode compilation must succeed");

    assert_eq!(compiled.envelope.neutral(), &compiled.artifact);
    assert!(compiled.artifact.validate_abi().is_ok());
    assert!(compiled.node_count() > 0);
    assert!(compiled.entry_count() > 0);

    let session = ModelCompiler::admit_model(&compiled, &manifest)
        .expect("Artifact session admission and resource binding must succeed for decode");

    assert!(session.len() > 0);
}

#[test]
fn independent_reference_driver_oracle_matches_layer_semantics() {
    use vyre_driver_reference::CpuRefEvaluator;
    use vyre_libs::nn::norm::learned_rms_norm;

    let dim = 4u32;
    let eps = 1e-5f32;
    let norm_prog = learned_rms_norm(
        "input",
        "weight",
        "output",
        1,
        dim,
        eps,
        vyre::ir::DataType::F32,
    )
    .expect("learned_rms_norm program must build");

    // Input values: x = [1.0, 2.0, 3.0, 4.0], gamma = [1.0, 1.0, 1.0, 1.0]
    let x_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let gamma_data: Vec<f32> = vec![1.0, 1.0, 1.0, 1.0];

    let x_bytes: &[u8] = bytemuck_cast_slice(&x_data);
    let gamma_bytes: &[u8] = bytemuck_cast_slice(&gamma_data);

    let evaluator = CpuRefEvaluator::default();

    let outputs = evaluator
        .evaluate_default(&norm_prog, &[x_bytes, gamma_bytes])
        .expect("Reference evaluator execution must succeed on neural norm primitive");
    assert_eq!(outputs.len(), 1, "Expected 1 output buffer from RMSNorm");
    let result_f32: &[f32] = bytemuck_cast_slice_f32(&outputs[0]);

    // Independent analytical reference computation:
    // mean_sq = (1^2 + 2^2 + 3^2 + 4^2) / 4 = (1 + 4 + 9 + 16) / 4 = 30 / 4 = 7.5
    // rms = sqrt(7.5 + 1e-5) ~= 2.738613
    // expected = x / rms
    let mean_sq = (1.0f32 + 4.0 + 9.0 + 16.0) / 4.0;
    let rms = (mean_sq + eps).sqrt();
    for i in 0..dim as usize {
        let expected = x_data[i] / rms;
        let diff = (result_f32[i] - expected).abs();
        assert!(
            diff < 1e-4,
            "Oracle differential error at index {i}: got {}, expected {expected}",
            result_f32[i]
        );
    }
}

fn bytemuck_cast_slice<T>(slice: &[T]) -> &[u8] {
    let len = std::mem::size_of_val(slice);
    unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const u8, len) }
}

fn bytemuck_cast_slice_f32(bytes: &[u8]) -> &[f32] {
    let len = bytes.len() / 4;
    unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const f32, len) }
}

#[test]
fn custom_downstream_checkpoints_compile_by_changing_only_manifests() {
    // Construct an arbitrary custom architecture without modifying any compiler internals
    let custom_config = ModelConfig {
        name: "CustomResearch-MoE-16x2B".to_string(),
        family: ModelFamily::DeepSeek,
        vocab_size: 32000,
        hidden_dim: 1024,
        num_layers: 2,
        num_heads: 16,
        head_dim: 64,
        num_kv_heads: 4,
        intermediate_dim: 2048,
        norm_eps: 1e-5,
        max_seq_len: 4096,
        activation: vyre_model_compiler::config::ActivationKind::SwiGlu,
        norm_kind: NormKind::RmsNorm,
        dtype: vyre::ir::DataType::F16,
        moe: Some(vyre_model_compiler::config::MoeConfig {
            num_experts: 16,
            top_k: 2,
            expert_hidden_dim: 1024,
            shared_expert_hidden_dim: Some(1024),
        }),
        mla: None,
    };

    let manifest = CheckpointManifest::from_config(&custom_config);
    assert!(manifest.validate_against_config(&custom_config).is_ok());

    let workload = WorkloadEnvelope::prefill(1, 16, custom_config.max_seq_len);
    let compiled = ModelCompiler::compile_model(&custom_config, &workload)
        .expect("Custom downstream checkpoint must compile cleanly");

    assert_eq!(compiled.envelope.neutral(), &compiled.artifact);
    assert!(compiled.artifact.validate_abi().is_ok());
    assert!(compiled.node_count() > 0);
    assert!(compiled.entry_count() > 0);
}

#[test]
fn unsupported_or_malformed_constructs_fail_with_typed_errors() {
    let valid_config = NamedModelConfig::Llama3_1_8B.config();

    // 1. Missing checkpoint tensor in manifest
    let mut incomplete_manifest = CheckpointManifest::from_config(&valid_config);
    incomplete_manifest
        .tensors
        .remove("model.embed_tokens.weight");
    let manifest_err = incomplete_manifest.validate_against_config(&valid_config);
    assert!(
        matches!(manifest_err, Err(ManifestError::MissingTensor { ref name, .. }) if name == "model.embed_tokens.weight"),
        "Expected MissingTensor error naming 'model.embed_tokens.weight', got {manifest_err:?}"
    );

    // 2. Shape mismatch in checkpoint manifest
    let mut mismatched_manifest = CheckpointManifest::from_config(&valid_config);
    mismatched_manifest.insert(TensorDescriptor::new(
        "model.norm.weight",
        vec![99999], // Wrong dimension
        valid_config.dtype.clone(),
    ));
    let shape_err = mismatched_manifest.validate_against_config(&valid_config);
    assert!(
        matches!(shape_err, Err(ManifestError::ShapeMismatch { ref name, .. }) if name == "model.norm.weight"),
        "Expected ShapeMismatch error naming 'model.norm.weight', got {shape_err:?}"
    );

    // 3. Multimodal invalid image dimensions (200 is not divisible by patch_size 14)
    let vision_cfg = VisionEncoderConfig::clip_vit_large_14();
    let invalid_img_err = vision_cfg.validate_image_shape(224, 200, 3);
    assert!(
        matches!(
            invalid_img_err,
            Err(MultimodalError::IndivisiblePatchSize {
                image_size: 200,
                patch_size: 14
            })
        ),
        "Expected IndivisiblePatchSize error, got {invalid_img_err:?}"
    );

    // 4. Tokenizer out-of-range token ID
    let tok_schema = TokenizerSchema::llama3();
    let tok_err = tok_schema.validate_tokens(&[999_999]);
    assert!(
        matches!(
            tok_err,
            Err(TokenizerError::TokenOutOfRange {
                token_id: 999_999,
                ..
            })
        ),
        "Expected TokenOutOfRange error, got {tok_err:?}"
    );

    // 5. Overflowing batch dimensions during graph building
    let overflow_workload = WorkloadEnvelope::prefill(u32::MAX, u32::MAX, 4096);
    let builder = ModelGraphBuilder::new(&valid_config, &overflow_workload);
    let build_err = builder.build_graph();
    assert!(
        matches!(build_err, Err(TranslationError::InvalidDimensions { ref name, .. }) if name == &valid_config.name),
        "Expected InvalidDimensions error naming '{}', got {build_err:?}",
        valid_config.name
    );
}

#[test]
fn tokenizer_schemas_validate_vocabulary_bounds() {
    let deepseek_tok = TokenizerSchema::deepseek();
    assert!(deepseek_tok.validate_tokens(&[0, 1, 100, 129_279]).is_ok());
    assert!(deepseek_tok.validate_tokens(&[130_000]).is_err());

    let llama_tok = TokenizerSchema::llama3();
    assert!(llama_tok
        .validate_tokens(&[128_000, 128_001, 128_009])
        .is_ok());
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
