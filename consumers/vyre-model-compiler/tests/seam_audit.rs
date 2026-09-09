//! Public seam audit asserting all model families express through domain-neutral constructs.

use vyre_model_compiler::config::{all_named_configs, ModelFamily};
use vyre_model_compiler::pipeline::ModelCompiler;
use vyre_model_compiler::workload::WorkloadEnvelope;

#[test]
fn public_neutral_seam_expresses_every_model_family() {
    let families = [
        ModelFamily::Llama,
        ModelFamily::Mistral,
        ModelFamily::Qwen,
        ModelFamily::DeepSeek,
        ModelFamily::Gemma,
        ModelFamily::Vision,
    ];

    let all_configs = all_named_configs();

    for family in families {
        let matching_config = all_configs
            .iter()
            .find(|c| c.config().family == family)
            .unwrap_or_else(|| panic!("No declared config found for family {family:?}"));

        let mut config = matching_config.config();
        config.num_layers = 1; // 1 layer is sufficient to prove seam expressibility

        let workload = WorkloadEnvelope::prefill(1, 4, config.max_seq_len);

        let compiled = ModelCompiler::compile_model(&config, &workload).unwrap_or_else(|e| {
            panic!(
                "GAP in public neutral seam for family {:?} (config '{}'): {}",
                family,
                config.name,
                e
            )
        });

        assert!(
            !compiled.digest().is_empty(),
            "Family {:?} must emit an artifact with valid digest",
            family
        );
    }
}
