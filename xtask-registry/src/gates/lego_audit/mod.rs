//! `cargo xtask lego-audit`  -  deeper LEGO-block enforcement.
//!
//! Gate 1 (`cargo xtask gate1`) is the floor: loops ≤ 4 AND nodes ≤ 200
//! OR composed_fraction ≥ 60%. That's table stakes. vyre's thesis is
//! composition, so the real measurement is harder.
//!
//! This xtask runs twelve stricter audits:
//!
//! 1. **No-reinvention check**  -  IR fingerprint every op body; any two
//!    ops with >80% fingerprint overlap where one doesn't invoke the
//!    other get flagged as duplication.
//! 2. **Depth-of-composition**  -  Tier 3 operations must place at least 25%
//!    of their nodes under registered children or appear in the explicit
//!    reviewed pure-IR leaf set.
//! 3. **Primitive-coverage**  -  every Tier 2.5 primitive should have
//!    ≥ 2 callers. Orphans are reported as one-release adoption advisories.
//!    Synthetic catalog consumers remain hard failures and never count.
//! 4. **Cross-dialect reach-through**  -  Tier 3 dialects importing
//!    private items from sibling Tier 3 dialects. That coupling
//!    belongs in Tier 2.5; flag it.
//! 5. **Large-file advisory**  -  files over a per-file source-line
//!    review guideline are reported as notes for a split-by-responsibility
//!    review. This is advisory and never fails the audit; the hard size
//!    ceiling is the `file-size` gate.
//! 6. **Composition-chain coverage**  -  every non-leaf registered op must
//!    render at least one child Region. Explicit pure-IR leaves and tiny
//!    operations are exempt.
//! 7. **Trend**  -  compare per-op `composed_fraction` to the previous
//!    tag; fail CI if it regresses. The thesis is "composition gets
//!    deeper over time," not "stagnates."
//! 8. **Composability**  -  flag non-leaf Tier 3 islands with no upstream
//!    caller and no downstream child operations.
//! 9. **Name-stem collision**  -  ≥ 4 ops sharing a leaf-prefix stem
//!    requires a discoverable namespace, merge, or explicit reviewed family.
//! 10. **Operand-shape advisory**  -  identical fingerprint prefixes and
//!     bigram-cosine ≥ 0.55 identify registered operations for semantic review.
//! 12. **Tier claims**  -  a source file may not name a tier its registered
//!     op ids contradict; the tier is derived from the id, never declared.
//!
//! Exit code 0 when every hard check passes. Advisories remain visible.
//! Intended to run in CI after Gate 1.

//! The numbered checks live one per module. A check reads the `OpInfo`
//! summary this module builds and reports through the shared `Report`, so a new
//! check is a module and a call in `run` and nothing else. Helpers more than one
//! check reads stay here.

mod composability;
mod composition_chain;
mod cross_dialect;
mod depth_of_composition;
mod duplicates;
pub(crate) mod exemptions;
mod fingerprint;
mod name_stem;
mod no_reinvention;
mod operand_shape;
mod ops;
mod primitive_coverage;
mod semantic_organization;
mod tier_claim;
mod trend;

use self::composability::*;
use self::composition_chain::*;
use self::cross_dialect::*;
use self::depth_of_composition::*;
use self::exemptions::*;
use self::fingerprint::*;
pub use self::fingerprint::{fingerprint_program, MIN_COMPARABLE_FINGERPRINT_BYTES};
use self::name_stem::*;
use self::no_reinvention::*;
use self::operand_shape::*;
use self::ops::*;
pub(crate) use self::ops::{collect_ops, OpInfo, Tier};
use self::primitive_coverage::*;
use self::semantic_organization::*;
use self::tier_claim::*;
use self::trend::*;

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::io;
use vyre::ir::{Expr, Node, Program};
use xtask::gate::{Coverage, Finding, GateCtx, GateError, Report};

use xtask::gates::dedup_report::{
    duplicate_family_report, duplicate_severity, registered_op_duplicate_family_id,
    registered_op_duplicate_subject, registered_op_owner_lane, structural_similarity,
    DuplicateEvidence, DuplicateFamilyFinding, DuplicateFamilyReport, DuplicateSubject,
};
use xtask::gates::use_paths::{collect_use_paths, is_test_source_path};

/// Check 0: every exemption is live.
pub struct LegoExemptionLiveness;

impl xtask::gate::GateBehavior for LegoExemptionLiveness {
    fn run(&self, _ctx: &GateCtx) -> Result<Report, GateError> {
        let mut report = Report::clean();
        let ops = collect_ops(&mut report);
        report.cover(Coverage::complete("registered operations", ops.len()));
        check_0_every_exemption_is_live(&mut report, &ops);
        Ok(report)
    }
}

/// Check 1: no private reimplementation of registered primitives.
pub struct LegoNoReinvention;

impl xtask::gate::GateBehavior for LegoNoReinvention {
    fn run(&self, _ctx: &GateCtx) -> Result<Report, GateError> {
        let mut report = Report::clean();
        let ops = collect_ops(&mut report);
        report.cover(Coverage::complete("registered operations", ops.len()));
        check_1_no_reinvention(&mut report, &ops);
        Ok(report)
    }
}

/// Check 2: depth of composition.
pub struct LegoCompositionDepth;

impl xtask::gate::GateBehavior for LegoCompositionDepth {
    fn run(&self, _ctx: &GateCtx) -> Result<Report, GateError> {
        let mut report = Report::clean();
        let ops = collect_ops(&mut report);
        report.cover(Coverage::complete("registered operations", ops.len()));
        check_2_depth_of_composition(&mut report, &ops);
        Ok(report)
    }
}

/// Check 3: primitive adoption coverage.
pub struct LegoPrimitiveCoverage;

impl xtask::gate::GateBehavior for LegoPrimitiveCoverage {
    fn run(&self, _ctx: &GateCtx) -> Result<Report, GateError> {
        let mut report = Report::clean();
        let ops = collect_ops(&mut report);
        report.cover(Coverage::complete("registered operations", ops.len()));
        check_3_primitive_coverage(&mut report, &ops);
        Ok(report)
    }
}

/// Check 4: cross-dialect reachthrough.
pub struct LegoCrossDialect;

impl xtask::gate::GateBehavior for LegoCrossDialect {
    fn run(&self, _ctx: &GateCtx) -> Result<Report, GateError> {
        let mut report = Report::clean();
        let ops = collect_ops(&mut report);
        report.cover(Coverage::complete("registered operations", ops.len()));
        check_4_cross_dialect_reachthrough(&mut report);
        Ok(report)
    }
}

/// Check 6: composition chain coverage.
pub struct LegoCompositionChains;

impl xtask::gate::GateBehavior for LegoCompositionChains {
    fn run(&self, _ctx: &GateCtx) -> Result<Report, GateError> {
        let mut report = Report::clean();
        let ops = collect_ops(&mut report);
        report.cover(Coverage::complete("registered operations", ops.len()));
        check_6_composition_chain_coverage(&mut report, &ops);
        Ok(report)
    }
}

/// Check 7: composition trend ratchet.
pub struct LegoTrend;

impl xtask::gate::GateBehavior for LegoTrend {
    fn run(&self, _ctx: &GateCtx) -> Result<Report, GateError> {
        let mut report = Report::clean();
        let ops = collect_ops(&mut report);
        report.cover(Coverage::complete("registered operations", ops.len()));
        check_7_trend(&mut report, &ops);
        Ok(report)
    }
}

/// Check 8: composability contract.
pub struct LegoComposability;

impl xtask::gate::GateBehavior for LegoComposability {
    fn run(&self, _ctx: &GateCtx) -> Result<Report, GateError> {
        let mut report = Report::clean();
        let ops = collect_ops(&mut report);
        report.cover(Coverage::complete("registered operations", ops.len()));
        check_8_composability(&mut report, &ops);
        Ok(report)
    }
}

/// Check 9: name stem collision.
pub struct LegoNameStems;

impl xtask::gate::GateBehavior for LegoNameStems {
    fn run(&self, _ctx: &GateCtx) -> Result<Report, GateError> {
        let mut report = Report::clean();
        let ops = collect_ops(&mut report);
        report.cover(Coverage::complete("registered operations", ops.len()));
        check_9_name_stem_collision(&mut report, &ops);
        Ok(report)
    }
}

/// Check 10: operand shape duplicate.
pub struct LegoOperandShapes;

impl xtask::gate::GateBehavior for LegoOperandShapes {
    fn run(&self, _ctx: &GateCtx) -> Result<Report, GateError> {
        let mut report = Report::clean();
        let ops = collect_ops(&mut report);
        report.cover(Coverage::complete("registered operations", ops.len()));
        check_10_operand_shape_duplicate(&mut report, &ops);
        Ok(report)
    }
}

/// Check 11: semantic organization and file roles.
pub struct LegoSemanticOrganization;

impl xtask::gate::GateBehavior for LegoSemanticOrganization {
    fn run(&self, _ctx: &GateCtx) -> Result<Report, GateError> {
        let mut report = Report::clean();
        let ops = collect_ops(&mut report);
        report.cover(Coverage::complete("registered operations", ops.len()));
        check_semantic_organization(&mut report, &ops);
        Ok(report)
    }
}

/// Check 12: no source file claims a tier its placement contradicts.
pub struct LegoTierClaims;

impl xtask::gate::GateBehavior for LegoTierClaims {
    fn run(&self, _ctx: &GateCtx) -> Result<Report, GateError> {
        let mut report = Report::clean();
        let ops = collect_ops(&mut report);
        report.cover(Coverage::complete("registered operations", ops.len()));
        check_12_tier_claims(&mut report, &ops);
        Ok(report)
    }
}

/// Operations built by hand, for the checks whose rule is about the summary
/// rather than about the registry.
#[cfg(test)]
pub(crate) mod test_ops {
    use super::*;

    pub(crate) fn op(id: &str, tier: Tier, children: &[&str]) -> OpInfo {
        OpInfo {
            id: id.to_string(),
            source_file: String::new(),
            category: None,
            laws: BTreeSet::new(),
            semantic_version: 1,
            tolerance: 0,
            effects: vyre_foundation::operation::OperationEffects::default(),
            capabilities: String::new(),
            required_caps: vyre_foundation::program_caps::RequiredCapabilities::default(),
            callees: BTreeSet::new(),
            program: Program::empty(),
            tier,
            buffer_signature: Vec::new(),
            fingerprint: vec![1; 64],
            semantic_fingerprint: [0; 32],
            own_nodes: 1,
            composed_nodes: 0,
            children: children.iter().map(|child| (*child).to_string()).collect(),
        }
    }

    pub(crate) fn op_with_fingerprint(id: &str, fingerprint: Vec<u8>) -> OpInfo {
        let mut info = op(id, Tier::T3, &[]);
        info.fingerprint = fingerprint;
        info
    }

    pub(crate) fn is_global_reduce_or_indexed_move(id: &str) -> bool {
        matches!(
            id.strip_prefix("vyre-libs::reduce::"),
            Some(
                "all"
                    | "any"
                    | "count"
                    | "count_non_zero"
                    | "max"
                    | "min"
                    | "sum"
                    | "gather"
                    | "scatter"
            )
        )
    }

    pub(crate) fn assert_no_global_reduce_pairs<T>(pairs: &[(T, &OpInfo, &OpInfo)], context: &str) {
        let reduce_pairs: Vec<(&str, &str)> = pairs
            .iter()
            .filter(|(_, a, b)| {
                is_global_reduce_or_indexed_move(&a.id) && is_global_reduce_or_indexed_move(&b.id)
            })
            .map(|(_, a, b)| (a.id.as_str(), b.id.as_str()))
            .collect();
        assert!(reduce_pairs.is_empty(), "{context}: {reduce_pairs:?}");
    }
}
#[cfg(test)]
pub(crate) use self::test_ops::*;
