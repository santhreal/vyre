//! One fixture per `BinOp` variant, so an operator table can be tested
//! against the whole enum instead of against the variants its author
//! remembered.
//!
//! # Why this exists in a shared crate
//!
//! Operand-swap legality is the clearest case. Three tables answered "may the
//! operands of this operator be reordered": the wire canonicalizer, the
//! canonicalize pass, and the CSE key. They named 15, 9 and 11 operators. A
//! program that the canonicalizer reordered was therefore not in the form the
//! pass produced, and CSE missed merges the canonical form had already made
//! identical. `BinOp::operand_swap` is now the single answer, and a suite that
//! holds a consumer to it needs the operator set rather than a remembered
//! subset of it.
//!
//! # How a new variant fails closed
//!
//! The member set is not written here. [`declared_bin_op_variants`] reads the
//! `pub enum BinOp` declaration in `vyre-spec` at run time, and
//! [`assert_covers_every_bin_op_variant`] holds the fixtures to it. Adding a
//! variant to the spec turns every suite built on these fixtures RED until a
//! fixture exists for it.
//!
//! The enumeration reads source as TEXT and never compiles it, so it reports
//! the same variant set whichever features the runner selects. Its failure
//! mode is finding nothing, which is why the assertion refuses a variant set
//! smaller than [`DECLARED_VARIANT_FLOOR`].

use std::collections::BTreeSet;

use vyre_spec::BinOp;

use crate::monorepo::vyre_workspace_root;

/// Fewest `BinOp` variants a working source enumeration can find.
///
/// A scan that matched nothing would report a trivially covered empty set.
/// The floor sits below the current count, so it catches a broken scan without
/// needing an edit every time the operator set grows.
pub const DECLARED_VARIANT_FLOOR: usize = 30;

/// The variant names `vyre-spec` declares for `BinOp`, read from source.
///
/// # Panics
///
/// Panics when the declaration cannot be located or parsed, which is a broken
/// enumeration rather than an empty enum.
#[must_use]
pub fn declared_bin_op_variants() -> BTreeSet<String> {
    let path = vyre_workspace_root().join("vyre-spec/src/bin_op.rs");
    let source = crate::read_source_file_bounded(&path)
        .unwrap_or_else(|err| panic!("Fix: cannot read the BinOp declaration at {path:?}: {err}"));
    let body = crate::braced_body(&source, "pub enum BinOp {").unwrap_or_else(|| {
        panic!("Fix: no `pub enum BinOp` declaration in {path:?}; update this enumeration")
    });
    crate::top_level_variant_names(body)
}

/// One operator per declared `BinOp` variant.
///
/// `Opaque` gets a test-support extension id: an extension operator has no
/// builtin algebra, and the tables under test key off the outer discriminant,
/// so a fixture that varied the payload would test the payload.
#[must_use]
pub fn bin_op_variant_samples() -> Vec<BinOp> {
    vec![
        BinOp::Add,
        BinOp::Sub,
        BinOp::Mul,
        BinOp::Div,
        BinOp::Mod,
        BinOp::WrappingAdd,
        BinOp::WrappingSub,
        BinOp::SaturatingAdd,
        BinOp::SaturatingSub,
        BinOp::SaturatingMul,
        BinOp::MulHigh,
        BinOp::AbsDiff,
        BinOp::Min,
        BinOp::Max,
        BinOp::BitAnd,
        BinOp::BitOr,
        BinOp::BitXor,
        BinOp::Shl,
        BinOp::Shr,
        BinOp::RotateLeft,
        BinOp::RotateRight,
        BinOp::Eq,
        BinOp::Ne,
        BinOp::Lt,
        BinOp::Gt,
        BinOp::Le,
        BinOp::Ge,
        BinOp::And,
        BinOp::Or,
        BinOp::Shuffle,
        BinOp::Ballot,
        BinOp::WaveReduce,
        BinOp::WaveBroadcast,
        BinOp::Opaque(vyre_spec::extension::ExtensionBinOpId::from_name(
            "vyre.test_support.fixture_bin_op",
        )),
    ]
}

/// The declared variant name of `op`, from its `Debug` rendering.
///
/// `Debug` prints the variant name first for every shape a variant can have,
/// and it is derived, so it cannot drift from the declaration the way a
/// hand-written mapping would.
#[must_use]
pub fn variant_name(op: BinOp) -> String {
    let rendered = format!("{op:?}");
    let end = rendered
        .find(|c: char| !c.is_ascii_alphanumeric() && c != '_')
        .unwrap_or(rendered.len());
    rendered[..end].to_string()
}

/// The fixtures name every variant the spec declares, exactly once.
///
/// # Panics
///
/// Panics naming the variants that have no fixture, or the fixtures that name
/// a variant the spec no longer declares.
pub fn assert_covers_every_bin_op_variant(samples: &[BinOp]) {
    let declared = declared_bin_op_variants();
    assert!(
        declared.len() >= DECLARED_VARIANT_FLOOR,
        "Fix: the BinOp source enumeration found only {} variants, below the floor of \
         {DECLARED_VARIANT_FLOOR}; the scan is broken, not the enum",
        declared.len()
    );

    let covered: BTreeSet<String> = samples.iter().copied().map(variant_name).collect();

    let missing: Vec<&String> = declared.difference(&covered).collect();
    assert!(
        missing.is_empty(),
        "Fix: add a BinOp fixture for each of {missing:?} in \
         vyre_test_support::bin_op_variants::bin_op_variant_samples; every table keyed on BinOp \
         is untested for them until you do"
    );

    let unknown: Vec<&String> = covered.difference(&declared).collect();
    assert!(
        unknown.is_empty(),
        "Fix: these fixtures name BinOp variants vyre-spec no longer declares: {unknown:?}"
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_fixture_set_covers_every_declared_bin_op_variant() {
        assert_covers_every_bin_op_variant(&bin_op_variant_samples());
    }

    #[test]
    fn a_missing_fixture_is_named() {
        let mut samples = bin_op_variant_samples();
        samples.retain(|op| !matches!(op, BinOp::Min));
        let failure = std::panic::catch_unwind(move || {
            assert_covers_every_bin_op_variant(&samples);
        })
        .expect_err("a fixture set missing Min must be rejected");
        let message = failure
            .downcast_ref::<String>()
            .map_or_else(String::new, Clone::clone);
        assert!(
            message.contains("\"Min\""),
            "the failure must name the missing variant, got: {message}"
        );
    }

    #[test]
    fn the_source_enumeration_finds_the_operator_set() {
        let declared = declared_bin_op_variants();
        assert!(
            declared.contains("Add") && declared.contains("MulHigh"),
            "the parse must find declared operators, got {declared:?}"
        );
        assert!(
            !declared.contains("Ordered"),
            "the parse must read BinOp, not the OperandSwap declaration above it: {declared:?}"
        );
    }
}
