//! The frozen `vyre_spec` operator variant space, as tables tests can iterate.
//!
//! Four suites needed the same list of every builtin `BinOp`, `UnOp`,
//! `AtomicOp` and `TernaryOp`, and each carried its own copy. The copies had
//! drifted in every direction: the wire sweep in `vyre-spec` was missing six
//! `UnOp` variants and six `BinOp` variants, the random-IR corpus in
//! `vyre-foundation` was missing three `BinOp` variants, and the terminal
//! round trip was missing `RotateLeft`, `RotateRight` and `MulHigh`. A round
//! trip, a wire tag pin and a property that each run over a different subset
//! of the operator space leave gaps nobody chose.
//!
//! Every consumer includes this file with `#[path]`, the same way
//! `tests/support/artifact_fixtures.rs` is shared. The assertion each suite
//! makes about a variant stays in that suite: this file owns only which
//! variants exist.
//!
//! The `DataType` element table is not here. `DataType` fixtures are owned by
//! `vyre_test_support::data_type_variants`, which holds them to the enum
//! declaration in `vyre-spec` at run time, and this file owns the operator
//! enums only.
//!
//! `Opaque` is deliberately absent from every table. It carries an extension
//! id, so each suite draws its own: the property suites want an arbitrary id,
//! the freeze tests want a pinned one.

#![allow(dead_code)]

use std::collections::BTreeSet;

use vyre_spec::{AtomicOp, BinOp, DataType, TernaryOp, UnOp};

/// Every builtin `BinOp`, in wire-tag order.
///
/// The member set is `vyre_test_support::bin_op_variants`, which holds its
/// fixtures to the `BinOp` declaration in `vyre-spec` at run time. A second
/// list here is how one table gains an operator the other does not know
/// about, so this one keeps only the builtins and orders them by wire tag.
pub(crate) fn builtin_bin_ops() -> Vec<BinOp> {
    let mut builtins: Vec<(u8, BinOp)> =
        vyre_test_support::bin_op_variants::bin_op_variant_samples()
            .into_iter()
            .filter_map(|op| op.builtin_wire_tag().map(|tag| (tag, op)))
            .collect();
    builtins.sort_unstable_by_key(|(tag, _)| *tag);
    builtins.into_iter().map(|(_, op)| op).collect()
}

/// Every builtin `UnOp`, in wire-tag order.
pub(crate) fn builtin_un_ops() -> Vec<UnOp> {
    vec![
        UnOp::Negate,
        UnOp::BitNot,
        UnOp::LogicalNot,
        UnOp::Popcount,
        UnOp::Clz,
        UnOp::Ctz,
        UnOp::ReverseBits,
        UnOp::Cos,
        UnOp::Sin,
        UnOp::Abs,
        UnOp::Sqrt,
        UnOp::Floor,
        UnOp::Ceil,
        UnOp::Round,
        UnOp::Trunc,
        UnOp::Sign,
        UnOp::IsNan,
        UnOp::IsInf,
        UnOp::IsFinite,
        UnOp::Exp,
        UnOp::Log,
        UnOp::Log2,
        UnOp::Exp2,
        UnOp::Tan,
        UnOp::Acos,
        UnOp::Asin,
        UnOp::Atan,
        UnOp::Tanh,
        UnOp::Sinh,
        UnOp::Cosh,
        UnOp::InverseSqrt,
        UnOp::Unpack4Low,
        UnOp::Unpack4High,
        UnOp::Unpack8Low,
        UnOp::Unpack8High,
        UnOp::Reciprocal,
    ]
}

/// Every builtin `AtomicOp`, in wire-tag order.
pub(crate) fn builtin_atomic_ops() -> Vec<AtomicOp> {
    vec![
        AtomicOp::Add,
        AtomicOp::Or,
        AtomicOp::And,
        AtomicOp::Xor,
        AtomicOp::Min,
        AtomicOp::Max,
        AtomicOp::Exchange,
        AtomicOp::CompareExchange,
        AtomicOp::CompareExchangeWeak,
        AtomicOp::FetchNand,
        AtomicOp::LruUpdate,
    ]
}

/// Every builtin `TernaryOp`, in wire-tag order.
pub(crate) fn builtin_ternary_ops() -> Vec<TernaryOp> {
    vec![TernaryOp::Fma, TernaryOp::Select]
}

/// The flat `DataType` forms a buffer declaration can carry as its element.
///
/// `vyre_test_support::data_type_elements` owns which flat forms exist, because
/// the IR fixture table is built from the same list; a second copy here is how
/// one table gains an element type the other does not know about.
pub(crate) fn buffer_data_types(element_size: usize) -> Vec<DataType> {
    vyre_test_support::data_type_elements::flat_buffer_element_types(element_size)
}
/// Variant names of `vyre_spec::<enum_name>` as the checked-in public-API
/// snapshot records them, `Opaque` excluded.
///
/// The snapshot is regenerated from rustdoc by
/// `scripts/check_public_api_snapshot.sh` and a byte-stability gate keeps it
/// equal to the crate's real surface, so adding a variant to a frozen operator
/// enum has to land here too. That is what makes the tables above fail closed:
/// a new variant appears in this set, no table lists it, and
/// `vyre-spec/tests/spec_variant_tables_cover_the_frozen_surface.rs` goes red
/// until somebody records a decision for it.
pub(crate) fn public_api_variant_names(enum_name: &str) -> BTreeSet<String> {
    const PACKAGE: &str = "vyre-spec";
    let prefix = format!("pub vyre_spec::{enum_name}::");
    let names: BTreeSet<String> = vyre_test_support::public_api::snapshot_text(PACKAGE)
        .lines()
        .filter_map(|line| line.strip_prefix(&prefix))
        // A field of a struct-like variant reads `Enum::Variant::field: T`, and
        // `Opaque` reads `Opaque(extension::ExtensionUnOpId)`. Neither is a
        // bare variant name.
        .filter(|rest| !rest.contains(':') && !rest.contains('('))
        .filter(|rest| *rest != "Opaque")
        .map(str::to_string)
        .collect();
    assert!(
        !names.is_empty(),
        "Fix: the public-API snapshot at {} lists no `{enum_name}` variants. Refresh it with scripts/check_public_api_snapshot.sh --refresh {PACKAGE}.",
        vyre_test_support::public_api::snapshot_path(PACKAGE).display()
    );
    names
}

/// Every builtin `CollectiveOp`, in wire-tag order.
pub(crate) fn builtin_collective_ops() -> [(vyre_spec::CollectiveOp, u8); 6] {
    [
        (vyre_spec::CollectiveOp::Sum, 0x01),
        (vyre_spec::CollectiveOp::Min, 0x02),
        (vyre_spec::CollectiveOp::Max, 0x03),
        (vyre_spec::CollectiveOp::BitAnd, 0x04),
        (vyre_spec::CollectiveOp::BitOr, 0x05),
        (vyre_spec::CollectiveOp::BitXor, 0x06),
    ]
}
