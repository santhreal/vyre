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
//! `Opaque` is deliberately absent from every table. It carries an extension
//! id, so each suite draws its own: the property suites want an arbitrary id,
//! the freeze tests want a pinned one.

#![allow(dead_code)]

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use vyre_spec::{AtomicOp, BinOp, DataType, TernaryOp, UnOp};

/// Every builtin `BinOp`, in wire-tag order.
pub(crate) fn builtin_bin_ops() -> Vec<BinOp> {
    vec![
        BinOp::Add,
        BinOp::Sub,
        BinOp::Mul,
        BinOp::Div,
        BinOp::Mod,
        BinOp::BitAnd,
        BinOp::BitOr,
        BinOp::BitXor,
        BinOp::Shl,
        BinOp::Shr,
        BinOp::Eq,
        BinOp::Ne,
        BinOp::Lt,
        BinOp::Gt,
        BinOp::Le,
        BinOp::Ge,
        BinOp::And,
        BinOp::Or,
        BinOp::AbsDiff,
        BinOp::Min,
        BinOp::Max,
        BinOp::SaturatingAdd,
        BinOp::SaturatingSub,
        BinOp::SaturatingMul,
        BinOp::Shuffle,
        BinOp::Ballot,
        BinOp::WaveReduce,
        BinOp::WaveBroadcast,
        BinOp::WrappingAdd,
        BinOp::WrappingSub,
        BinOp::RotateLeft,
        BinOp::RotateRight,
        BinOp::MulHigh,
    ]
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
/// `element_size` parameterises `DataType::Array`, which is the one flat form
/// that carries a payload. The nested forms (`Vec`, `TensorShaped`,
/// `SparseBsr`, `Quantized`), `Handle` and `Opaque` are not here: a buffer
/// element table and a cast-target table are different sets, and a suite that
/// needs the nested forms builds them from these leaves.
pub(crate) fn buffer_data_types(element_size: usize) -> Vec<DataType> {
    vec![
        DataType::U8,
        DataType::U16,
        DataType::U32,
        DataType::I8,
        DataType::I16,
        DataType::I32,
        DataType::I64,
        DataType::U64,
        DataType::Vec2U32,
        DataType::Vec4U32,
        DataType::Bool,
        DataType::Bytes,
        DataType::Array { element_size },
        DataType::F16,
        DataType::BF16,
        DataType::F32,
        DataType::F64,
        DataType::Tensor,
    ]
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
    let snapshot_path = vyre_spec_api_snapshot();
    let snapshot = std::fs::read_to_string(&snapshot_path).unwrap_or_else(|error| {
        panic!(
            "Fix: the public-API snapshot at {} must be readable to enumerate {enum_name} variants: {error}",
            snapshot_path.display()
        )
    });
    let prefix = format!("pub vyre_spec::{enum_name}::");
    let names: BTreeSet<String> = snapshot
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
        "Fix: the public-API snapshot at {} lists no `{enum_name}` variants. Refresh it with scripts/check_public_api_snapshot.sh --refresh vyre-spec.",
        snapshot_path.display()
    );
    names
}

fn vyre_spec_api_snapshot() -> PathBuf {
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    manifest
        .ancestors()
        .map(|directory| directory.join("docs/public-api/vyre-spec.txt"))
        .find(|candidate| candidate.is_file())
        .unwrap_or_else(|| {
            panic!(
                "Fix: no docs/public-api/vyre-spec.txt above {}. The variant tables enumerate the frozen operator surface from that snapshot.",
                manifest.display()
            )
        })
}
