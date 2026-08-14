//! How an honest bench case declares itself: the suites it runs in, its
//! metadata record, and its GPU requirement shape.
//!
//! Every honest case repeated the same three declarations verbatim, and the
//! copies had drifted: one case's suite list omitted `Smoke`, so that case never
//! ran in the smoke suite and a regression in it was invisible to every smoke
//! run. Naming the list once removes the place that drift can live.

use crate::api::case::{
    BenchId, BenchLayer, BenchMetadata, BenchRequirements, DeterminismClass, WorkloadClass,
};
use crate::api::suite::SuiteKind;

/// The suites every honest case runs in.
pub(crate) const HONEST_SUITES: &[SuiteKind] = &[
    SuiteKind::Honest,
    SuiteKind::Deep,
    SuiteKind::Release,
    SuiteKind::Smoke,
];

/// The metadata record for an honest case.
///
/// An honest case is by definition a deterministic honest-layer workload owned
/// by this crate, so only its identity, its prose, and its tags vary.
pub(crate) fn honest_metadata(
    id: BenchId,
    name: &str,
    description: &str,
    tags: &[&str],
) -> BenchMetadata {
    BenchMetadata {
        id,
        name: name.to_string(),
        description: description.to_string(),
        tags: tags.iter().map(|tag| (*tag).to_string()).collect(),
        layer: BenchLayer::Honest,
        workload: WorkloadClass::Honest,
        determinism: DeterminismClass::Deterministic,
        owner_crate: "vyre-bench".to_string(),
    }
}

/// The requirement shape for an honest case: a GPU, no network, and enough
/// device memory to hold the workload.
pub(crate) fn honest_gpu_requirements(min_vram_bytes: u64) -> BenchRequirements {
    BenchRequirements {
        needs_gpu: true,
        needs_network: false,
        min_vram_bytes: Some(min_vram_bytes),
        min_input_bytes: None,
        feature_set: vec![],
    }
}

#[cfg(test)]
mod tests {
    use super::{honest_gpu_requirements, honest_metadata, HONEST_SUITES};
    use crate::api::case::{BenchId, BenchLayer, DeterminismClass, WorkloadClass};
    use crate::api::suite::SuiteKind;

    /// An honest case runs in the smoke suite. One hand-rolled copy of this list
    /// omitted `Smoke`, so `search.binary.u32.1m` was excluded from every smoke
    /// run while its six siblings were included.
    #[test]
    fn honest_suites_include_smoke() {
        assert!(
            HONEST_SUITES.contains(&SuiteKind::Smoke),
            "Fix: an honest case that is not in the smoke suite is never smoke-tested"
        );
        assert!(HONEST_SUITES.contains(&SuiteKind::Honest));
        assert!(HONEST_SUITES.contains(&SuiteKind::Deep));
        assert!(HONEST_SUITES.contains(&SuiteKind::Release));
    }

    /// The classification an honest case carries is fixed; only its prose varies.
    #[test]
    fn metadata_classification_is_fixed() {
        let metadata = honest_metadata(
            BenchId("x.y".to_string()),
            "X Y",
            "does x to y",
            &["honest", "branchy"],
        );

        assert_eq!(metadata.id.0, "x.y");
        assert_eq!(metadata.name, "X Y");
        assert_eq!(metadata.description, "does x to y");
        assert_eq!(metadata.tags, vec!["honest", "branchy"]);
        assert!(matches!(metadata.layer, BenchLayer::Honest));
        assert!(matches!(metadata.workload, WorkloadClass::Honest));
        assert!(matches!(
            metadata.determinism,
            DeterminismClass::Deterministic
        ));
        assert_eq!(metadata.owner_crate, "vyre-bench");
    }

    /// An honest case declares device memory, never a host input floor, and
    /// never a feature gate: it is the plain GPU shape.
    #[test]
    fn gpu_requirements_declare_vram_only() {
        let requirements = honest_gpu_requirements(4_096);

        assert!(requirements.needs_gpu);
        assert!(!requirements.needs_network);
        assert_eq!(requirements.min_vram_bytes, Some(4_096));
        assert_eq!(requirements.min_input_bytes, None);
        assert!(requirements.feature_set.is_empty());
    }
}
