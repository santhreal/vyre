//! Derived coverage test asserting every declared named configuration records
//! an end-to-end compile decision.

use vyre_model_compiler::config::all_named_configs;
use vyre_model_compiler::pipeline::ModelCompiler;
use vyre_model_compiler::workload::WorkloadEnvelope;

#[test]
fn all_declared_named_configurations_compile_end_to_end() {
    let configs = all_named_configs();
    assert!(
        !configs.is_empty(),
        "all_named_configs() must return at least one configuration"
    );

    let mut compile_decisions = Vec::new();

    for named in &configs {
        let mut model_config = named.config();

        // For unit-test execution speed across configurations, scale layer count
        // to a representative 2-layer slice while preserving exact attention, normalization,
        // MoE routing, MLA latents, activation, and dtype contracts.
        model_config.num_layers = 2;

        let workload = WorkloadEnvelope::prefill(1, 8, model_config.max_seq_len);

        let compiled = ModelCompiler::compile_model(&model_config, &workload)
            .unwrap_or_else(|e| panic!("Model '{}' failed end-to-end compile: {e}", named.id()));

        // Contract 1: Artifact envelope holds neutral artifact
        assert_eq!(
            compiled.envelope.neutral(),
            &compiled.artifact,
            "Compiled artifact envelope for '{}' must match neutral artifact",
            named.id()
        );

        // Contract 2: Artifact ABI projection validates
        assert!(
            compiled.artifact.validate_abi().is_ok(),
            "Compiled artifact ABI for '{}' must validate",
            named.id()
        );

        // Contract 3: Structural non-zero counts
        assert!(
            compiled.node_count() > 0,
            "Compiled artifact for '{}' must contain at least one node",
            named.id()
        );
        assert!(
            compiled.entry_count() > 0,
            "Compiled artifact for '{}' must have at least one entry point",
            named.id()
        );
        assert!(
            compiled.required_resource_bytes() > 0,
            "Compiled artifact for '{}' must declare non-zero resource allocations",
            named.id()
        );

        // Contract 4: Determinism
        let compiled_repeat = ModelCompiler::compile_model(&model_config, &workload)
            .expect("Deterministic re-compilation must succeed");
        assert_eq!(
            compiled.digest(),
            compiled_repeat.digest(),
            "Deterministic compilation: repeating compile for '{}' must yield identical digest",
            named.id()
        );

        compile_decisions.push((named.id(), compiled.digest()));
    }

    assert_eq!(
        compile_decisions.len(),
        configs.len(),
        "Every declared named configuration must have an end-to-end compile decision recorded"
    );
}
