//! `cargo xtask lego-audit`  -  deeper LEGO-block enforcement.
//!
//! Gate 1 (`cargo xtask gate1`) is the floor: loops ≤ 4 AND nodes ≤ 200
//! OR composed_fraction ≥ 60%. That's table stakes. vyre's thesis is
//! composition, so the real measurement is harder.
//!
//! This xtask runs ten stricter audits:
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
//!
//! Exit code 0 when every hard check passes. Advisories remain visible.
//! Intended to run in CI after Gate 1.

//! The eleven numbered checks live one per module. A check reads the `OpInfo`
//! summary this module builds and reports through the shared `Report`, so a new
//! check is a module and a call in `run` and nothing else. Helpers more than one
//! check reads stay here.

mod composability;
mod composition_chain;
mod cross_dialect;
mod depth_of_composition;
mod duplicates;
mod exemptions;
mod fingerprint;
mod god_files;
mod name_stem;
mod no_reinvention;
mod operand_shape;
mod ops;
mod primitive_coverage;
mod trend;

#[allow(unused_imports)]
use self::composability::*;
#[allow(unused_imports)]
use self::composition_chain::*;
#[allow(unused_imports)]
use self::cross_dialect::*;
#[allow(unused_imports)]
use self::depth_of_composition::*;
#[allow(unused_imports)]
use self::duplicates::*;
#[allow(unused_imports)]
use self::exemptions::*;
#[allow(unused_imports)]
use self::fingerprint::*;
#[allow(unused_imports)]
use self::god_files::*;
#[allow(unused_imports)]
use self::name_stem::*;
#[allow(unused_imports)]
use self::no_reinvention::*;
#[allow(unused_imports)]
use self::operand_shape::*;
#[allow(unused_imports)]
use self::ops::*;
pub(crate) use self::ops::{collect_ops, OpInfo, Tier};
#[allow(unused_imports)]
use self::primitive_coverage::*;
#[allow(unused_imports)]
use self::trend::*;

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::io;
use vyre::ir::{Expr, Node, Program};
use xtask::gate::{Finding, Gate, GateCtx, GateError, Report};

use xtask::gates::dedup_report::{
    duplicate_family_report, duplicate_report_generator_command, duplicate_report_json_path,
    duplicate_severity, registered_op_duplicate_family_id, registered_op_duplicate_subject,
    registered_op_owner_lane, structural_similarity, write_duplicate_report_json,
    DuplicateEvidence, DuplicateFamilyFinding, DuplicateFamilyReport, DuplicateSubject,
};
use xtask::gates::implementation_family::{
    known_distinct_implementation_families, reviewed_distinct_operations,
    same_implementation_family, IMPLEMENTATION_FAMILY_ROWS, REVIEWED_DISTINCT_OPERATIONS,
};
use xtask::gates::use_paths::{collect_use_paths, is_test_source_path};

/// Entry point for the `lego-audit` subcommand.
/// Audits registered composition against the ten LEGO-block laws.
pub struct LegoAudit;

impl Gate for LegoAudit {
    fn name(&self) -> &'static str {
        "lego-audit"
    }

    fn help(&self) -> &'static str {
        "Hold registered composition to the ten composition laws; --write records the composition baseline"
    }

    fn generates(&self) -> bool {
        true
    }

    fn run(&self, ctx: &GateCtx) -> Result<Report, GateError> {
        let mut report = Report::clean();
        let ops = collect_ops(&mut report);
        report.note(format!("{} op(s) audited", ops.len()));
        if ctx.write {
            write_composition_baseline(&ctx.root, &ops).map_err(|error| {
                GateError::new(
                    format!("failed to write the composition baseline: {error}"),
                    "make audits/lego-composition.tsv writable, then run the gate again",
                )
            })?;
            report.note("wrote the composition baseline");
        }
        if let Some(path) = ctx.flag("--duplicate-report-json") {
            let path = duplicate_report_json_path(
                "--duplicate-report-json",
                Some(path),
                "--duplicate-report-json requires a path",
            )
            .map_err(|error| {
                GateError::new(error, "pass a writable path after --duplicate-report-json")
            })?;
            let generator_command = duplicate_report_generator_command("lego-audit", &path);
            let duplicates = lego_duplicate_report(&ops, &generator_command);
            write_duplicate_report_json(&path, &duplicates).map_err(|error| {
                GateError::new(
                    format!(
                        "could not write the duplicate family report `{}`: {error}",
                        path.display()
                    ),
                    "pass a writable path after --duplicate-report-json",
                )
            })?;
            report.note(format!(
                "wrote the duplicate family report to {}",
                path.display()
            ));
        }

        check_0_every_exemption_is_live(&mut report, &ops);
        check_1_no_reinvention(&mut report, &ops);
        check_2_depth_of_composition(&mut report, &ops);
        check_3_primitive_coverage(&mut report, &ops);
        check_4_cross_dialect_reachthrough(&mut report);
        check_5_god_files(&mut report);
        check_6_composition_chain_coverage(&mut report, &ops);
        check_7_trend(&mut report, &ops);
        check_8_composability(&mut report, &ops);
        check_9_name_stem_collision(&mut report, &ops);
        check_10_operand_shape_duplicate(&mut report, &ops);
        Ok(report)
    }
}

/// Enforces canonical primitive adoption and its recorded exceptions.
pub struct PrimitiveAdmissionGate;

impl Gate for PrimitiveAdmissionGate {
    fn name(&self) -> &'static str {
        "primitive-admission-gate"
    }

    fn help(&self) -> &'static str {
        "Enforce canonical primitive adoption and its recorded exceptions"
    }

    fn run(&self, _ctx: &GateCtx) -> Result<Report, GateError> {
        let mut report = Report::clean();
        let ops = collect_ops(&mut report);
        report.note(format!("{} op(s) audited", ops.len()));
        check_3_primitive_coverage(&mut report, &ops);
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
            program: Program::empty(),
            tier,
            buffer_signature: Vec::new(),
            fingerprint: vec![1; 64],
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
}
