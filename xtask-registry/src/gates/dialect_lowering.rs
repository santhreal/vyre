//! `cargo xtask dialect-lowering` - a declared composition that lowers into nothing.
//!
//! A dialect declares `is_composable` on its operation descriptor and registers
//! `build` on the operation registration. These are separate declarations over
//! the same operation id, so an operation can claim it lowers into canonical
//! logical IR while registering nothing that produces that IR.
//!
//! `OperationRegistry::build` already rejects a registration carrying neither a
//! builder nor a signature. The dialect macro always emits a signature, so that
//! check passes for every dialect operation and cannot see this class. An
//! intrinsic is the legitimate case of a signature with no builder: a backend
//! arm emits it rather than composing it. The defect is the operation that
//! declares itself composable and still registers no program.
//!
//! Such an operation reaches the reference interpreter and every backend as an
//! intrinsic that no emitter arm covers, so it fails at lowering on whichever
//! target first selects it rather than at registration.
//!
//! What this gate does not catch: it reads whether a builder is registered, not
//! whether the program that builder returns is a faithful lowering of the
//! declared semantics. `gate1` and `abstraction-gate` judge the body.

use std::collections::BTreeMap;

use xtask::gate::{GateCtx, GateError, Report};

/// One dialect operation's declared composability and its registered lowering.
pub struct DeclaredOp {
    /// Canonical operation identifier, shared by both registrations.
    pub id: String,
    /// Owning dialect identifier.
    pub dialect: String,
    /// Whether the dialect descriptor declares the operation composable.
    pub is_composable: bool,
    /// Whether the operation registration carries a neutral program builder.
    pub lowers: bool,
}

/// Operations that declare a composition and register no program builder.
#[must_use]
pub fn unlowered_compositions(ops: &[DeclaredOp]) -> Vec<&DeclaredOp> {
    ops.iter()
        .filter(|op| op.is_composable && !op.lowers)
        .collect()
}

/// Every registered dialect operation joined to its operation registration.
#[must_use]
pub fn collect_declared_ops() -> Vec<DeclaredOp> {
    let mut lowers: BTreeMap<&'static str, bool> = BTreeMap::new();
    for entry in vyre_registry_link::operation::live_operation_registry().iter() {
        let lowered = lowers.entry(entry.id).or_insert(false);
        *lowered |= entry.build.is_some();
    }
    let mut ops = Vec::new();
    for dialect in vyre_foundation::dialect::DialectRegistry::global().values() {
        for op in dialect.operations {
            ops.push(DeclaredOp {
                id: op.id.to_string(),
                dialect: dialect.id.to_string(),
                is_composable: op.is_composable,
                lowers: lowers.get(op.id).copied().unwrap_or(false),
            });
        }
    }
    ops.sort_by(|left, right| left.id.cmp(&right.id));
    ops
}

/// Entry point for the `dialect-lowering` subcommand.
/// Enforces that every declared composition registers a program builder.
pub struct DialectLowering;

impl xtask::gate::GateBehavior for DialectLowering {
    fn run(&self, _ctx: &GateCtx) -> Result<Report, GateError> {
        let ops = collect_declared_ops();
        let mut report = Report::from_messages(
            unlowered_compositions(&ops)
                .into_iter()
                .map(|op| {
                    format!(
                        "dialect operation `{}` in `{}` declares `is_composable` and registers no program builder",
                        op.id, op.dialect
                    )
                })
                .collect(),
            "register a neutral builder that lowers it into canonical logical IR, or declare it uncomposable and add its reference and backend emitter arms",
        );
        report.cover_complete("registered dialect operations", ops.len());
        Ok(report)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn op(id: &str, is_composable: bool, lowers: bool) -> DeclaredOp {
        DeclaredOp {
            id: id.to_string(),
            dialect: "vyre-test::dialect".to_string(),
            is_composable,
            lowers,
        }
    }

    #[test]
    fn a_composable_operation_without_a_builder_is_reported() {
        let ops = vec![
            op("composed_and_lowered", true, true),
            op("intrinsic_without_builder", false, false),
            op("composed_without_lowering", true, false),
        ];
        let reported: Vec<&str> = unlowered_compositions(&ops)
            .into_iter()
            .map(|entry| entry.id.as_str())
            .collect();
        assert_eq!(reported, vec!["composed_without_lowering"]);
    }

    #[test]
    fn the_live_registry_lowers_every_declared_composition() {
        let ops = collect_declared_ops();
        assert!(
            ops.iter().any(|entry| entry.is_composable),
            "no linked dialect declares a composable operation, so this gate would pass vacuously"
        );
        let missing: Vec<&str> = unlowered_compositions(&ops)
            .into_iter()
            .map(|entry| entry.id.as_str())
            .collect();
        assert_eq!(missing, Vec::<&str>::new());
    }
}
