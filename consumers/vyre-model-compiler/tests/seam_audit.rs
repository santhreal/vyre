//! Public seam audit asserting all model families express through domain-neutral constructs.

use std::collections::{BTreeMap, BTreeSet};
use vyre_model_compiler::config::{all_model_families, all_named_configs, ModelFamily};
use vyre_model_compiler::pipeline::ModelCompiler;
use vyre_model_compiler::workload::WorkloadEnvelope;

/// Compile-time closure over all [`ModelFamily`] variants.
///
/// An exhaustive `match` without a catch-all (`_`) ensures that adding a new
/// variant to [`ModelFamily`] breaks compilation immediately (E0004) until an
/// explicit handling decision is recorded here.
const fn family_to_dense_index(family: ModelFamily) -> usize {
    match family {
        ModelFamily::Llama => 0,
        ModelFamily::Mistral => 1,
        ModelFamily::Qwen => 2,
        ModelFamily::DeepSeek => 3,
        ModelFamily::Gemma => 4,
        ModelFamily::Vision => 5,
    }
}

#[test]
fn model_family_variant_space_is_closed_and_derived_from_source() {
    let families = all_model_families();
    assert_eq!(
        families,
        ModelFamily::all(),
        "all_model_families() and ModelFamily::all() must agree on the variant slice"
    );

    let mut seen_indices = BTreeSet::new();
    let mut seen_families = BTreeSet::new();

    for &family in families {
        let dense_idx = family_to_dense_index(family);
        let ordinal = family.ordinal();
        assert_eq!(
            dense_idx, ordinal,
            "Dense index and ModelFamily::ordinal() must agree for {family:?}"
        );
        assert!(
            seen_families.insert(family),
            "Duplicate variant in all_model_families(): {family:?}"
        );
        assert!(
            seen_indices.insert(dense_idx),
            "Duplicate dense index {dense_idx} for family {family:?}"
        );
    }

    assert_eq!(
        seen_indices.len(),
        6,
        "Exhaustive closure must account for exactly 6 canonical model families"
    );
}

#[test]
fn public_neutral_seam_expresses_every_model_family_with_strict_contracts() {
    let all_configs = all_named_configs();
    let mut compiled_family_digests = BTreeMap::new();

    for &family in all_model_families() {
        let matching_config = all_configs
            .iter()
            .find(|c| c.config().family == family)
            .unwrap_or_else(|| panic!("No declared config found for family {family:?}"));

        let mut config = matching_config.config();
        config.num_layers = 1; // 1 layer is sufficient to prove seam expressibility

        let workload = WorkloadEnvelope::prefill(1, 4, config.max_seq_len);

        // 1. First compilation
        let compiled = ModelCompiler::compile_model(&config, &workload).unwrap_or_else(|e| {
            panic!(
                "GAP in public neutral seam for family {:?} (config '{}'): {}",
                family, config.name, e
            )
        });

        // Contract 1: Deterministic compilation
        // Compiling the exact same configuration and workload twice must yield identical digests
        let compiled_again = ModelCompiler::compile_model(&config, &workload)
            .expect("Deterministic re-compilation must succeed");
        assert_eq!(
            compiled.digest(),
            compiled_again.digest(),
            "Deterministic compilation contract: compiling same config twice must yield identical digest for {family:?}"
        );

        // Contract 2: Artifact envelope matches neutral artifact
        assert_eq!(
            compiled.envelope.neutral(),
            &compiled.artifact,
            "Emitted artifact envelope must contain the neutral artifact for {family:?}"
        );
        // Contract 3: Artifact ABI validation
        assert!(
            compiled.artifact.validate_abi().is_ok(),
            "Emitted artifact must have a valid ABI projection for {family:?}"
        );

        // Contract 4: Non-zero distinct structural properties
        assert!(
            compiled.node_count() > 0,
            "Emitted artifact for {family:?} must contain at least one node"
        );
        assert!(
            compiled.entry_count() > 0,
            "Emitted artifact for {family:?} must contain at least one ABI entry point"
        );
        assert!(
            compiled.required_resource_bytes() > 0,
            "Emitted artifact for {family:?} must declare positive allocation requirements"
        );

        // Contract 5: Family distinctness
        // Different model families must produce distinct artifact digests
        if let Some(previous_family) = compiled_family_digests.insert(compiled.digest(), family) {
            panic!(
                "Distinctness contract violation: families {previous_family:?} and {family:?} produced identical digest '{}'",
                compiled.digest()
            );
        }
    }

    assert_eq!(
        compiled_family_digests.len(),
        all_model_families().len(),
        "Every model family must produce a unique, distinct compiled artifact digest"
    );
}
