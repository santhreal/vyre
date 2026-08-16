//! Descriptor intent sidecar: the scan-construct mappings intents come from,
//! their validation against a descriptor, and the routing evidence that falls
//! out of a validated set.

use super::{
    DescriptorIntent, DescriptorIntentError, DescriptorIntentEvidence, DescriptorIntentKind,
    DescriptorIntentSet, DescriptorIntentStrategy, IntentAnnotatedDescriptor, KernelDescriptor,
    ScanConstructIntentClass, ScanConstructIntentMapping,
};

/// Current descriptor-intent sidecar schema version.
pub const DESCRIPTOR_INTENT_SCHEMA_VERSION: u32 = 1;

impl DescriptorIntentKind {
    /// Return the backend-neutral routing class for this intent.
    #[must_use]
    pub const fn strategy(self) -> DescriptorIntentStrategy {
        match self {
            Self::LiteralPrefilter => DescriptorIntentStrategy::Prefilter,
            Self::AutomataTransition => DescriptorIntentStrategy::Automata,
            Self::Verifier => DescriptorIntentStrategy::Verifier,
            Self::OutputCompaction => DescriptorIntentStrategy::Compaction,
            Self::RelationSeed => DescriptorIntentStrategy::RelationSeed,
            Self::StreamingState => DescriptorIntentStrategy::Streaming,
        }
    }

    const fn digest_tag(self) -> u8 {
        match self {
            Self::LiteralPrefilter => 1,
            Self::AutomataTransition => 2,
            Self::Verifier => 3,
            Self::OutputCompaction => 4,
            Self::RelationSeed => 5,
            Self::StreamingState => 6,
        }
    }
}

impl ScanConstructIntentMapping {
    /// Return whether this construct requires a verifier fragment.
    #[must_use]
    pub fn requires_verifier_fragment(self) -> bool {
        self.required_intents
            .iter()
            .any(|intent| *intent == DescriptorIntentKind::Verifier)
    }

    /// Return whether this construct includes `class`.
    #[must_use]
    pub fn includes_class(self, class: ScanConstructIntentClass) -> bool {
        self.classes.iter().any(|candidate| *candidate == class)
    }
}

/// Canonical scan-construct intent mappings.
pub const SCAN_CONSTRUCT_INTENT_MAPPINGS: &[ScanConstructIntentMapping] = &[
    ScanConstructIntentMapping {
        construct_id: "regular_exact_core",
        tier: "supported",
        classes: &[
            ScanConstructIntentClass::Literal,
            ScanConstructIntentClass::Automata,
        ],
        required_intents: &[
            DescriptorIntentKind::LiteralPrefilter,
            DescriptorIntentKind::AutomataTransition,
            DescriptorIntentKind::OutputCompaction,
        ],
        verifier_fragment_id: None,
    },
    ScanConstructIntentMapping {
        construct_id: "unsupported_backtracking_constructs",
        tier: "rejected",
        classes: &[ScanConstructIntentClass::Verifier],
        required_intents: &[DescriptorIntentKind::Verifier],
        verifier_fragment_id: Some("unsupported-backtracking-diagnostic"),
    },
    ScanConstructIntentMapping {
        construct_id: "lookaround_prefilter_constructs",
        tier: "approximated",
        classes: &[
            ScanConstructIntentClass::Literal,
            ScanConstructIntentClass::Verifier,
        ],
        required_intents: &[
            DescriptorIntentKind::LiteralPrefilter,
            DescriptorIntentKind::Verifier,
        ],
        verifier_fragment_id: Some("lookaround-verifier"),
    },
    ScanConstructIntentMapping {
        construct_id: "hardware_rule_database_constructs",
        tier: "accelerator-only",
        classes: &[
            ScanConstructIntentClass::ExternalAccelerator,
            ScanConstructIntentClass::Verifier,
        ],
        required_intents: &[
            DescriptorIntentKind::Verifier,
            DescriptorIntentKind::OutputCompaction,
        ],
        verifier_fragment_id: Some("external-rule-database-verifier"),
    },
    ScanConstructIntentMapping {
        construct_id: "capture_extraction_constructs",
        tier: "verifier-required",
        classes: &[ScanConstructIntentClass::Verifier],
        required_intents: &[
            DescriptorIntentKind::Verifier,
            DescriptorIntentKind::OutputCompaction,
        ],
        verifier_fragment_id: Some("capture-extraction-verifier"),
    },
    ScanConstructIntentMapping {
        construct_id: "derivative_regex_constructs",
        tier: "verifier-required",
        classes: &[
            ScanConstructIntentClass::Derivative,
            ScanConstructIntentClass::Verifier,
        ],
        required_intents: &[
            DescriptorIntentKind::AutomataTransition,
            DescriptorIntentKind::Verifier,
            DescriptorIntentKind::OutputCompaction,
        ],
        verifier_fragment_id: Some("derivative-regex-verifier"),
    },
];

/// Find the canonical intent mapping for a scan construct.
#[must_use]
pub fn scan_construct_intent_mapping(
    construct_id: &str,
) -> Option<&'static ScanConstructIntentMapping> {
    SCAN_CONSTRUCT_INTENT_MAPPINGS
        .iter()
        .find(|mapping| mapping.construct_id == construct_id)
}

impl DescriptorIntent {
    /// Create an intent with no optional descriptor references.
    #[must_use]
    pub const fn new(kind: DescriptorIntentKind, section_digest: u64) -> Self {
        Self {
            kind,
            binding_slot: None,
            op_result: None,
            stream_state_bytes: 0,
            relation_arity: 0,
            section_digest,
        }
    }

    /// Attach a binding-slot reference.
    #[must_use]
    pub const fn with_binding_slot(mut self, slot: u32) -> Self {
        self.binding_slot = Some(slot);
        self
    }

    /// Attach an operation-result reference.
    #[must_use]
    pub const fn with_op_result(mut self, result: u32) -> Self {
        self.op_result = Some(result);
        self
    }

    /// Declare persistent stream-state storage.
    #[must_use]
    pub const fn with_stream_state_bytes(mut self, bytes: u32) -> Self {
        self.stream_state_bytes = bytes;
        self
    }

    /// Declare relation arity.
    #[must_use]
    pub const fn with_relation_arity(mut self, arity: u16) -> Self {
        self.relation_arity = arity;
        self
    }
}

impl DescriptorIntentSet {
    /// Create an intent sidecar using the current schema version.
    #[must_use]
    pub fn new(intents: Vec<DescriptorIntent>) -> Self {
        Self {
            schema_version: DESCRIPTOR_INTENT_SCHEMA_VERSION,
            intents,
        }
    }

    /// Validate intent references against a descriptor and return routing
    /// evidence for emitters and benchmark artifacts.
    ///
    /// # Errors
    ///
    /// Returns [`DescriptorIntentError`] when the sidecar is empty,
    /// version-skewed, references missing descriptor slots/results, or omits
    /// required relation/streaming payload metadata.
    pub fn validate_for_descriptor(
        &self,
        descriptor: &KernelDescriptor,
    ) -> Result<DescriptorIntentEvidence, DescriptorIntentError> {
        if self.schema_version != DESCRIPTOR_INTENT_SCHEMA_VERSION {
            return Err(DescriptorIntentError::UnsupportedSchemaVersion {
                expected: DESCRIPTOR_INTENT_SCHEMA_VERSION,
                actual: self.schema_version,
            });
        }
        if self.intents.is_empty() {
            return Err(DescriptorIntentError::EmptyIntentSet);
        }

        let mut evidence = DescriptorIntentEvidence {
            schema_version: DESCRIPTOR_INTENT_SCHEMA_VERSION,
            descriptor_id: descriptor.id.clone(),
            intent_count: self.intents.len(),
            has_literal_prefilter: false,
            has_automata_transition: false,
            has_verifier: false,
            has_output_compaction: false,
            has_relation_seed: false,
            has_streaming_state: false,
            strategy_digest: stable_descriptor_intent_digest(&descriptor.id, &self.intents),
        };

        for intent in &self.intents {
            if intent.section_digest == 0 {
                return Err(DescriptorIntentError::MissingSectionDigest { kind: intent.kind });
            }
            if let Some(slot) = intent.binding_slot {
                let known = descriptor
                    .bindings
                    .slots
                    .iter()
                    .any(|binding| binding.slot == slot);
                if !known {
                    return Err(DescriptorIntentError::UnknownBindingSlot {
                        kind: intent.kind,
                        slot,
                    });
                }
            }
            if let Some(result) = intent.op_result {
                if descriptor.find_op_by_id(result).is_none() {
                    return Err(DescriptorIntentError::UnknownOpResult {
                        kind: intent.kind,
                        result,
                    });
                }
            }

            match intent.kind {
                DescriptorIntentKind::LiteralPrefilter => evidence.has_literal_prefilter = true,
                DescriptorIntentKind::AutomataTransition => {
                    evidence.has_automata_transition = true;
                }
                DescriptorIntentKind::Verifier => evidence.has_verifier = true,
                DescriptorIntentKind::OutputCompaction => evidence.has_output_compaction = true,
                DescriptorIntentKind::RelationSeed => {
                    if intent.relation_arity == 0 {
                        return Err(DescriptorIntentError::MissingRelationArity);
                    }
                    evidence.has_relation_seed = true;
                }
                DescriptorIntentKind::StreamingState => {
                    if intent.stream_state_bytes == 0 {
                        return Err(DescriptorIntentError::MissingStreamingStateBytes);
                    }
                    evidence.has_streaming_state = true;
                }
            }
        }

        Ok(evidence)
    }
}

impl IntentAnnotatedDescriptor {
    /// Construct an intent-annotated descriptor after validating intent
    /// references against the descriptor body and binding layout.
    ///
    /// # Errors
    ///
    /// Returns [`DescriptorIntentError`] when the intent set is incomplete or
    /// references descriptor state that does not exist.
    pub fn try_new(
        descriptor: KernelDescriptor,
        intents: DescriptorIntentSet,
    ) -> Result<Self, DescriptorIntentError> {
        intents.validate_for_descriptor(&descriptor)?;
        Ok(Self {
            descriptor,
            intents,
        })
    }

    /// Preserve descriptor intent through representation canonicalization while
    /// revalidating referenced binding slots or result ids.
    ///
    /// Semantic rewrites occur on `Program` before descriptor construction.
    ///
    /// # Errors
    ///
    /// Returns [`DescriptorIntentError`] when the rewritten descriptor no
    /// longer contains a binding slot or result id referenced by the sidecar.
    pub fn preserve_intents_after_rewrite(
        self,
        descriptor: KernelDescriptor,
    ) -> Result<Self, DescriptorIntentError> {
        Self::try_new(descriptor, self.intents)
    }

    /// Validate and return backend-neutral intent-routing evidence.
    pub fn evidence(&self) -> Result<DescriptorIntentEvidence, DescriptorIntentError> {
        self.intents.validate_for_descriptor(&self.descriptor)
    }
}

impl DescriptorIntentEvidence {
    /// Return whether every scan-pipeline intent is represented.
    #[must_use]
    pub const fn covers_full_scan_pipeline(&self) -> bool {
        self.schema_version == DESCRIPTOR_INTENT_SCHEMA_VERSION
            && self.intent_count > 0
            && self.has_literal_prefilter
            && self.has_automata_transition
            && self.has_verifier
            && self.has_output_compaction
            && self.has_relation_seed
            && self.has_streaming_state
            && self.strategy_digest != 0
    }
}

impl std::fmt::Display for DescriptorIntentError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedSchemaVersion { expected, actual } => write!(
                f,
                "descriptor intent schema version {actual} is unsupported; expected {expected}"
            ),
            Self::EmptyIntentSet => f.write_str("descriptor intent set is empty"),
            Self::MissingSectionDigest { kind } => {
                write!(f, "descriptor intent {kind:?} is missing section_digest")
            }
            Self::UnknownBindingSlot { kind, slot } => write!(
                f,
                "descriptor intent {kind:?} references unknown binding slot {slot}"
            ),
            Self::UnknownOpResult { kind, result } => write!(
                f,
                "descriptor intent {kind:?} references unknown op result {result}"
            ),
            Self::MissingRelationArity => {
                f.write_str("relation-seed descriptor intent is missing relation_arity")
            }
            Self::MissingStreamingStateBytes => {
                f.write_str("streaming-state descriptor intent is missing stream_state_bytes")
            }
        }
    }
}

impl std::error::Error for DescriptorIntentError {}

fn stable_descriptor_intent_digest(descriptor_id: &str, intents: &[DescriptorIntent]) -> u64 {
    const FNV_OFFSET: u64 = 0xcbf29ce484222325;
    const FNV_PRIME: u64 = 0x100000001b3;

    fn mix(mut digest: u64, byte: u8) -> u64 {
        digest ^= u64::from(byte);
        digest.wrapping_mul(FNV_PRIME)
    }

    fn mix_u64(mut digest: u64, value: u64) -> u64 {
        for byte in value.to_le_bytes() {
            digest = mix(digest, byte);
        }
        digest
    }

    let mut digest = FNV_OFFSET;
    for byte in descriptor_id.as_bytes() {
        digest = mix(digest, *byte);
    }
    for intent in intents {
        digest = mix(digest, intent.kind.digest_tag());
        digest = mix_u64(digest, u64::from(intent.binding_slot.unwrap_or(u32::MAX)));
        digest = mix_u64(digest, u64::from(intent.op_result.unwrap_or(u32::MAX)));
        digest = mix_u64(digest, u64::from(intent.stream_state_bytes));
        digest = mix_u64(digest, u64::from(intent.relation_arity));
        digest = mix_u64(digest, intent.section_digest);
    }
    if digest == 0 {
        FNV_OFFSET
    } else {
        digest
    }
}

// Inline: covers the crate-private `covers_full_scan_pipeline` and `evidence`, which no integration test can reach.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::descriptor::test_descriptors::build;
    use crate::descriptor::{KernelOp, KernelOpKind};

    fn literal_op() -> KernelOp {
        KernelOp {
            kind: KernelOpKind::Literal,
            operands: vec![0],
            result: Some(0),
        }
    }

    fn full_scan_intents() -> DescriptorIntentSet {
        DescriptorIntentSet::new(vec![
            DescriptorIntent::new(DescriptorIntentKind::LiteralPrefilter, 11)
                .with_binding_slot(0)
                .with_op_result(0),
            DescriptorIntent::new(DescriptorIntentKind::AutomataTransition, 12)
                .with_binding_slot(0)
                .with_op_result(0),
            DescriptorIntent::new(DescriptorIntentKind::Verifier, 13)
                .with_binding_slot(0)
                .with_op_result(0),
            DescriptorIntent::new(DescriptorIntentKind::OutputCompaction, 14)
                .with_binding_slot(0)
                .with_op_result(0),
            DescriptorIntent::new(DescriptorIntentKind::RelationSeed, 15)
                .with_binding_slot(0)
                .with_relation_arity(2),
            DescriptorIntent::new(DescriptorIntentKind::StreamingState, 16)
                .with_binding_slot(0)
                .with_stream_state_bytes(64),
        ])
    }

    #[test]
    fn descriptor_intents_cover_scan_pipeline_and_route_to_strategies() {
        let d = build(vec![literal_op()], vec![]);
        let intents = full_scan_intents();
        let evidence = intents.validate_for_descriptor(&d).unwrap();

        assert!(evidence.covers_full_scan_pipeline());
        assert_ne!(evidence.strategy_digest, 0);
        assert_eq!(
            DescriptorIntentKind::LiteralPrefilter.strategy(),
            DescriptorIntentStrategy::Prefilter
        );
        assert_eq!(
            DescriptorIntentKind::AutomataTransition.strategy(),
            DescriptorIntentStrategy::Automata
        );
        assert_eq!(
            DescriptorIntentKind::Verifier.strategy(),
            DescriptorIntentStrategy::Verifier
        );
        assert_eq!(
            DescriptorIntentKind::OutputCompaction.strategy(),
            DescriptorIntentStrategy::Compaction
        );
        assert_eq!(
            DescriptorIntentKind::RelationSeed.strategy(),
            DescriptorIntentStrategy::RelationSeed
        );
        assert_eq!(
            DescriptorIntentKind::StreamingState.strategy(),
            DescriptorIntentStrategy::Streaming
        );
    }

    #[test]
    fn scan_construct_intent_mappings_cover_tiers_and_verifier_fragments() {
        let mut tiers = std::collections::BTreeSet::new();
        let mut classes = std::collections::BTreeSet::new();
        for mapping in SCAN_CONSTRUCT_INTENT_MAPPINGS {
            assert!(
                !mapping.required_intents.is_empty(),
                "Fix: scan construct `{}` must map to at least one descriptor intent",
                mapping.construct_id
            );
            tiers.insert(mapping.tier);
            for class in mapping.classes {
                classes.insert(*class);
            }
            if mapping.requires_verifier_fragment() {
                assert!(
                    mapping.verifier_fragment_id.is_some(),
                    "Fix: verifier-required scan construct `{}` must name a verifier fragment",
                    mapping.construct_id
                );
                assert!(
                    mapping
                        .required_intents
                        .contains(&DescriptorIntentKind::Verifier),
                    "Fix: verifier-required scan construct `{}` must require DescriptorIntentKind::Verifier",
                    mapping.construct_id
                );
            }
        }

        for tier in [
            "supported",
            "rejected",
            "approximated",
            "accelerator-only",
            "verifier-required",
        ] {
            assert!(
                tiers.contains(tier),
                "Fix: scan construct intent mappings must cover tier `{tier}`"
            );
        }
        for class in [
            ScanConstructIntentClass::Literal,
            ScanConstructIntentClass::Automata,
            ScanConstructIntentClass::Verifier,
            ScanConstructIntentClass::Derivative,
            ScanConstructIntentClass::ExternalAccelerator,
        ] {
            assert!(
                classes.contains(&class),
                "Fix: scan construct intent mappings must cover class `{class:?}`"
            );
        }
    }

    #[test]
    fn scan_construct_intent_lookup_returns_shared_contract_rows() {
        let lookaround = scan_construct_intent_mapping("lookaround_prefilter_constructs")
            .expect("Fix: lookaround construct mapping must exist");
        assert!(lookaround.includes_class(ScanConstructIntentClass::Literal));
        assert!(lookaround.includes_class(ScanConstructIntentClass::Verifier));
        assert_eq!(lookaround.verifier_fragment_id, Some("lookaround-verifier"));

        let derivative = scan_construct_intent_mapping("derivative_regex_constructs")
            .expect("Fix: derivative construct mapping must exist");
        assert!(derivative.includes_class(ScanConstructIntentClass::Derivative));
        assert!(derivative
            .required_intents
            .contains(&DescriptorIntentKind::AutomataTransition));

        assert!(scan_construct_intent_mapping("unknown_construct").is_none());
    }

    #[test]
    fn descriptor_intents_survive_rewrites_through_envelope() {
        let d = build(vec![literal_op()], vec![]);
        let intents = full_scan_intents();
        let envelope = IntentAnnotatedDescriptor::try_new(d.clone(), intents.clone()).unwrap();
        let rewritten = d.with_id("k.rewritten");

        let preserved = envelope.preserve_intents_after_rewrite(rewritten).unwrap();

        assert_eq!(preserved.intents, intents);
        assert_eq!(preserved.descriptor.id, "k.rewritten");
        assert!(preserved.evidence().unwrap().covers_full_scan_pipeline());
    }

    #[test]
    fn descriptor_intents_reject_invalid_references_and_payloads() {
        let d = build(vec![literal_op()], vec![]);
        let missing_slot = DescriptorIntentSet::new(vec![DescriptorIntent::new(
            DescriptorIntentKind::Verifier,
            17,
        )
        .with_binding_slot(99)]);
        assert_eq!(
            missing_slot.validate_for_descriptor(&d).unwrap_err(),
            DescriptorIntentError::UnknownBindingSlot {
                kind: DescriptorIntentKind::Verifier,
                slot: 99
            }
        );

        let missing_relation_arity = DescriptorIntentSet::new(vec![DescriptorIntent::new(
            DescriptorIntentKind::RelationSeed,
            18,
        )
        .with_binding_slot(0)]);
        assert_eq!(
            missing_relation_arity
                .validate_for_descriptor(&d)
                .unwrap_err(),
            DescriptorIntentError::MissingRelationArity
        );

        let missing_stream_bytes = DescriptorIntentSet::new(vec![DescriptorIntent::new(
            DescriptorIntentKind::StreamingState,
            19,
        )
        .with_binding_slot(0)]);
        assert_eq!(
            missing_stream_bytes
                .validate_for_descriptor(&d)
                .unwrap_err(),
            DescriptorIntentError::MissingStreamingStateBytes
        );
    }
}
