//! The literal folder and the reference interpreter answer the same question.
//!
//! WHY this gate exists: `ir_eval` and `ir_inner::model::node_kind` used to
//! carry parallel arithmetic tables for the same operator set, and the tables
//! disagreed per width. The folder folded `AbsDiff`, `And`, `Or`, `RotateLeft`
//! and `RotateRight` on i32, `Mod` on f32, `BitXor` on bool and every
//! transcendental over an integer literal; the interpreter rejected all of
//! them. The interpreter divided f32 by zero to an infinity and shifted i32 by
//! a masked count; the folder declined both. An optimized program and its own
//! CPU reference run therefore computed different answers for the same
//! expression, and nothing was red.
//!
//! The operator set is read out of the enum definitions in `vyre-spec` at run
//! time, so a new `BinOp` or `UnOp` variant turns this suite RED until someone
//! records a decision for it: either both sides handle it at every width, both
//! reject it, or it gets an [`EXCLUDED_BIN_OPS`] / [`EXCLUDED_UN_OPS`] row
//! carrying the reason.
//!
//! What this does NOT catch: a divergence that only shows up on operands other
//! than the fixed probe pair, and any width with no `Expr` literal. `u64` is
//! the second case; [`folder_cannot_produce_a_sixty_four_bit_literal`] pins it.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use vyre::ir::eval::{fold_binary_literal, fold_cast_literal, fold_unary_literal};
use vyre::ir::{BinOp, DataType, Expr, InterpCtx, NodeId, NodeStorage, UnOp, Value};

/// Enum variants below this count mean the source parse broke rather than that
/// the IR shrank, and every per-operator assertion below would be vacuous.
const MIN_BIN_OP_VARIANTS: usize = 15;
/// Same guard for the unary operator set.
const MIN_UN_OP_VARIANTS: usize = 3;

/// Binary operators that evaluate at no width at all, with the reason. Every
/// other parsed operator must answer somewhere, so deleting its row from
/// `vyre_foundation::scalar_ops` turns this suite RED.
const UNEVALUABLE_BIN_OPS: &[(&str, &str)] = &[
    (
        "Opaque",
        "extension-declared operator: its ExtensionBinOpId has no scalar meaning until an \
         extension registers one, so there is no builtin semantics for either side to carry",
    ),
    (
        "Shuffle",
        "subgroup operator: typecheck V097 requires backend subgroup semantics, and its result \
         depends on the invocation's lane neighbours rather than on its two scalar operands",
    ),
    (
        "Ballot",
        "subgroup operator: typecheck V097 requires backend subgroup semantics, and its result \
         depends on the invocation's lane neighbours rather than on its two scalar operands",
    ),
    (
        "WaveReduce",
        "subgroup operator: typecheck V097 requires backend subgroup semantics, and its result \
         depends on the invocation's lane neighbours rather than on its two scalar operands",
    ),
    (
        "WaveBroadcast",
        "subgroup operator: typecheck V097 requires backend subgroup semantics, and its result \
         depends on the invocation's lane neighbours rather than on its two scalar operands",
    ),
];

/// Unary operators that evaluate at no width at all, with the reason.
const UNEVALUABLE_UN_OPS: &[(&str, &str)] = &[(
    "Opaque",
    "extension-declared operator: its ExtensionUnOpId has no scalar meaning until an extension \
     registers one, so there is no builtin semantics for either side to carry",
)];

/// Binary operators defined at exactly one of the two 32-bit integer widths,
/// with the reason the other width is refused. Any other operator must answer
/// at u32 and i32 together, so dropping either row is named by this gate.
const INTEGER_WIDTH_ASYMMETRY_BIN_OPS: &[(&str, &str)] = &[
    (
        "And",
        "typecheck V095 restricts logical And/Or to u32 and bool: signed operands have no `!= 0` \
         lowering the emitters agree on",
    ),
    (
        "Or",
        "typecheck V095 restricts logical And/Or to u32 and bool: signed operands have no `!= 0` \
         lowering the emitters agree on",
    ),
    (
        "AbsDiff",
        "typecheck V086 rejects signed AbsDiff: `i32::MIN.abs_diff(i32::MAX)` has no signed \
         result, and answering it with a u32 literal silently retypes the expression",
    ),
    (
        "RotateLeft",
        "typecheck V094 restricts shifts and rotates to u32 operands",
    ),
    (
        "RotateRight",
        "typecheck V094 restricts shifts and rotates to u32 operands",
    ),
    (
        "MulHigh",
        "multiply-high is the unsigned Granlund-Montgomery primitive: the signed upper half is a \
         different value and no emitter lowers it",
    ),
];

/// Unary operators defined at exactly one of the two 32-bit integer widths.
const INTEGER_WIDTH_ASYMMETRY_UN_OPS: &[(&str, &str)] = &[(
    "LogicalNot",
    "typecheck V100 restricts LogicalNot to u32 and bool",
)];

/// A scalar width that both sides can represent: an `Expr` literal on the
/// folder side and a `Value` variant on the interpreter side.
#[derive(Clone, Copy)]
struct Width {
    name: &'static str,
    left: Probe,
    right: Probe,
}

/// One probe operand, paired as the `Expr` literal and the `Value` the
/// interpreter must see for the same scalar.
#[derive(Clone, Copy)]
enum Probe {
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
}

impl Probe {
    fn expr(self) -> Expr {
        match self {
            Self::U32(value) => Expr::LitU32(value),
            Self::I32(value) => Expr::LitI32(value),
            Self::F32(value) => Expr::LitF32(value),
            Self::Bool(value) => Expr::LitBool(value),
        }
    }

    fn value(self) -> Value {
        match self {
            Self::U32(value) => Value::U32(value),
            Self::I32(value) => Value::I32(value),
            Self::F32(value) => Value::F32(value),
            Self::Bool(value) => Value::Bool(value),
        }
    }
}

/// Probe operands are chosen so no width degenerates: the integers are not
/// equal, not zero and not a power of two, the signed pair is negative over
/// positive so `Div` and `Mod` differ in sign, and the shift/rotate counts stay
/// inside the width.
const WIDTHS: &[Width] = &[
    Width {
        name: "u32",
        left: Probe::U32(0x8000_0003),
        right: Probe::U32(5),
    },
    Width {
        name: "i32",
        left: Probe::I32(-7),
        right: Probe::I32(3),
    },
    Width {
        name: "f32",
        left: Probe::F32(-2.5),
        right: Probe::F32(0.75),
    },
    Width {
        name: "bool",
        left: Probe::Bool(true),
        right: Probe::Bool(false),
    },
];

/// Construct the operator named by a parsed variant name.
///
/// `None` means the name carries a payload this gate cannot synthesize, which
/// is only legal when [`EXCLUDED_BIN_OPS`] records why.
fn bin_op_named(name: &str) -> Option<BinOp> {
    Some(match name {
        "Add" => BinOp::Add,
        "Sub" => BinOp::Sub,
        "Mul" => BinOp::Mul,
        "Div" => BinOp::Div,
        "Mod" => BinOp::Mod,
        "WrappingAdd" => BinOp::WrappingAdd,
        "WrappingSub" => BinOp::WrappingSub,
        "BitAnd" => BinOp::BitAnd,
        "BitOr" => BinOp::BitOr,
        "BitXor" => BinOp::BitXor,
        "Shl" => BinOp::Shl,
        "Shr" => BinOp::Shr,
        "Eq" => BinOp::Eq,
        "Ne" => BinOp::Ne,
        "Lt" => BinOp::Lt,
        "Gt" => BinOp::Gt,
        "Le" => BinOp::Le,
        "Ge" => BinOp::Ge,
        "And" => BinOp::And,
        "Or" => BinOp::Or,
        "AbsDiff" => BinOp::AbsDiff,
        "Min" => BinOp::Min,
        "Max" => BinOp::Max,
        "SaturatingAdd" => BinOp::SaturatingAdd,
        "SaturatingSub" => BinOp::SaturatingSub,
        "SaturatingMul" => BinOp::SaturatingMul,
        "Shuffle" => BinOp::Shuffle,
        "Ballot" => BinOp::Ballot,
        "WaveReduce" => BinOp::WaveReduce,
        "WaveBroadcast" => BinOp::WaveBroadcast,
        "RotateLeft" => BinOp::RotateLeft,
        "RotateRight" => BinOp::RotateRight,
        "MulHigh" => BinOp::MulHigh,
        _ => return None,
    })
}

/// Construct the unary operator named by a parsed variant name.
fn un_op_named(name: &str) -> Option<UnOp> {
    Some(match name {
        "Negate" => UnOp::Negate,
        "BitNot" => UnOp::BitNot,
        "LogicalNot" => UnOp::LogicalNot,
        "Popcount" => UnOp::Popcount,
        "Clz" => UnOp::Clz,
        "Ctz" => UnOp::Ctz,
        "ReverseBits" => UnOp::ReverseBits,
        "Cos" => UnOp::Cos,
        "Sin" => UnOp::Sin,
        "Abs" => UnOp::Abs,
        "Sqrt" => UnOp::Sqrt,
        "Floor" => UnOp::Floor,
        "Ceil" => UnOp::Ceil,
        "Round" => UnOp::Round,
        "Trunc" => UnOp::Trunc,
        "Sign" => UnOp::Sign,
        "IsNan" => UnOp::IsNan,
        "IsInf" => UnOp::IsInf,
        "IsFinite" => UnOp::IsFinite,
        "Exp" => UnOp::Exp,
        "Log" => UnOp::Log,
        "Log2" => UnOp::Log2,
        "Exp2" => UnOp::Exp2,
        "Tan" => UnOp::Tan,
        "Acos" => UnOp::Acos,
        "Asin" => UnOp::Asin,
        "Atan" => UnOp::Atan,
        "Tanh" => UnOp::Tanh,
        "Sinh" => UnOp::Sinh,
        "Cosh" => UnOp::Cosh,
        "InverseSqrt" => UnOp::InverseSqrt,
        "Unpack4Low" => UnOp::Unpack4Low,
        "Unpack4High" => UnOp::Unpack4High,
        "Unpack8Low" => UnOp::Unpack8Low,
        "Unpack8High" => UnOp::Unpack8High,
        "Reciprocal" => UnOp::Reciprocal,
        _ => return None,
    })
}

/// Read the variant names of `pub enum <enum_name>` out of a Rust source file.
///
/// Parsing source rather than listing variants here is the point: the list in
/// this file would go stale in silence, and a stale member list is the same
/// failure as no gate at all.
fn variant_names(source: &Path, enum_name: &str) -> Vec<String> {
    let text = std::fs::read_to_string(source).unwrap_or_else(|error| {
        panic!(
            "Fix: cannot read {} to derive the {enum_name} variant set: {error}",
            source.display()
        )
    });
    let header = format!("pub enum {enum_name} {{");
    let body_start = text.find(&header).unwrap_or_else(|| {
        panic!(
            "Fix: {} no longer declares `{header}`; point this gate at the file that owns {enum_name}.",
            source.display()
        )
    }) + header.len();

    let mut depth = 1usize;
    let mut names = Vec::new();
    for line in text[body_start..].lines() {
        let line = line.trim();
        depth += line.matches('{').count();
        depth -= line.matches('}').count();
        if depth == 0 {
            break;
        }
        if depth != 1 || line.starts_with("//") || line.starts_with("#[") || line.is_empty() {
            continue;
        }
        let name: String = line
            .chars()
            .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
            .collect();
        let terminator = line[name.len()..].chars().next();
        if name.is_empty()
            || !name.starts_with(|c: char| c.is_ascii_uppercase())
            || !matches!(terminator, Some(',' | '(' | '{') | None)
        {
            continue;
        }
        names.push(name);
    }
    names
}

fn spec_source(file: &str) -> std::path::PathBuf {
    vyre_test_support::monorepo::vyre_crate_directory("vyre-spec")
        .join("src")
        .join(file)
}

/// Interpret one binary node through the reference interpreter.
fn interpret_binary(op: BinOp, left: Value, right: Value) -> Result<Value, String> {
    let mut ctx = InterpCtx::default();
    ctx.set(NodeId(0), left);
    ctx.set(NodeId(1), right);
    NodeStorage::BinOp {
        op,
        left: NodeId(0),
        right: NodeId(1),
    }
    .interpret(&mut ctx)
    .map_err(|error| error.to_string())
}

/// Interpret one unary node through the reference interpreter.
fn interpret_unary(op: UnOp, operand: Value) -> Result<Value, String> {
    let mut ctx = InterpCtx::default();
    ctx.set(NodeId(0), operand);
    NodeStorage::UnOp {
        op,
        operand: NodeId(0),
    }
    .interpret(&mut ctx)
    .map_err(|error| error.to_string())
}

/// A folded literal and an interpreted value describe the same scalar.
///
/// f32 is compared by bit pattern so a canonicalization difference (quiet-NaN
/// selection, subnormal flush) is a disagreement rather than a rounding
/// footnote.
fn same_scalar(expr: &Expr, value: Value) -> bool {
    match (expr, value) {
        (Expr::LitU32(folded), Value::U32(interpreted)) => *folded == interpreted,
        (Expr::LitI32(folded), Value::I32(interpreted)) => *folded == interpreted,
        (Expr::LitBool(folded), Value::Bool(interpreted)) => *folded == interpreted,
        (Expr::LitF32(folded), Value::F32(interpreted)) => {
            folded.to_bits() == interpreted.to_bits()
        }
        _ => false,
    }
}

/// Compare the two sides for one (operator, width) pair, returning the
/// disagreement sentence when they differ.
fn disagreement(
    operator: &str,
    width: &str,
    folded: Option<Expr>,
    interpreted: Result<Value, String>,
) -> Option<String> {
    match (folded, interpreted) {
        (Some(folded), Ok(interpreted)) if same_scalar(&folded, interpreted) => None,
        (None, Err(_)) => None,
        (Some(folded), Ok(interpreted)) => Some(format!(
            "`{operator}` at width {width}: folder produced {folded:?} but the interpreter produced {interpreted:?}"
        )),
        (Some(folded), Err(error)) => Some(format!(
            "`{operator}` at width {width}: folder produced {folded:?} but the interpreter rejected it with \"{error}\""
        )),
        (None, Ok(interpreted)) => Some(format!(
            "`{operator}` at width {width}: folder declined to fold but the interpreter produced {interpreted:?}"
        )),
    }
}

fn table_names(table: &[(&str, &str)]) -> BTreeSet<String> {
    table.iter().map(|(name, _)| (*name).to_owned()).collect()
}

/// Assert the three invariants shared by both operator kinds.
///
/// `answered` maps each exercised operator to the widths where both sides
/// produced the same value.
fn assert_support_invariants(
    kind: &str,
    names: &[String],
    answered: &BTreeMap<String, BTreeSet<&'static str>>,
    unevaluable: &[(&str, &str)],
    asymmetric: &[(&str, &str)],
) {
    let unevaluable_names = table_names(unevaluable);
    let asymmetric_names = table_names(asymmetric);

    let silent: Vec<&String> = names
        .iter()
        .filter(|name| {
            !unevaluable_names.contains(*name) && answered.get(*name).is_none_or(BTreeSet::is_empty)
        })
        .collect();
    assert!(
        silent.is_empty(),
        "{kind} variant(s) {silent:?} evaluate at no width and are not recorded in the \
         unevaluable table. Fix: restore the operator's row in vyre_foundation::scalar_ops, or \
         record why it has no scalar semantics."
    );

    let live: Vec<&String> = unevaluable_names
        .iter()
        .filter(|name| answered.get(*name).is_some_and(|w| !w.is_empty()))
        .collect();
    assert!(
        live.is_empty(),
        "{kind} variant(s) {live:?} are recorded as having no scalar semantics but now evaluate. \
         Fix: delete the stale row."
    );

    let mut parity_failures = Vec::new();
    for name in names {
        let Some(widths) = answered.get(name) else {
            continue;
        };
        let unsigned = widths.contains("u32");
        let signed = widths.contains("i32");
        let recorded = asymmetric_names.contains(name);
        if unsigned != signed && !recorded {
            let (has, lacks) = if unsigned {
                ("u32", "i32")
            } else {
                ("i32", "u32")
            };
            parity_failures.push(format!(
                "`{name}` evaluates at {has} but not at {lacks}, and no row records why"
            ));
        }
        if unsigned == signed && recorded {
            parity_failures.push(format!(
                "`{name}` is recorded as defined at only one 32-bit integer width, but it now \
                 evaluates at u32={unsigned} and i32={signed}"
            ));
        }
    }
    assert!(
        parity_failures.is_empty(),
        "{kind} integer-width support changed without a recorded decision:\n{}",
        parity_failures.join("\n")
    );

    let stale: Vec<&String> = unevaluable_names
        .iter()
        .chain(asymmetric_names.iter())
        .filter(|name| !names.contains(name))
        .collect();
    assert!(
        stale.is_empty(),
        "{kind} table row(s) {stale:?} name variants the enum no longer declares. \
         Fix: delete the stale rows."
    );
}

#[test]
fn folder_and_interpreter_agree_on_every_binary_operator_and_width() {
    let names = variant_names(&spec_source("bin_op.rs"), "BinOp");
    assert!(
        names.len() >= MIN_BIN_OP_VARIANTS,
        "Fix: parsed only {} BinOp variants ({names:?}); the source parse is broken and every \
         per-operator assertion below would be vacuous. Repair `variant_names` before trusting \
         a green run.",
        names.len()
    );

    let constructible = table_names(UNEVALUABLE_BIN_OPS);
    let mut answered: BTreeMap<String, BTreeSet<&'static str>> = BTreeMap::new();
    let mut failures = Vec::new();
    let mut unconstructed = Vec::new();

    for name in &names {
        let Some(op) = bin_op_named(name) else {
            if !constructible.contains(name) {
                unconstructed.push(name);
            }
            continue;
        };
        let widths = answered.entry(name.clone()).or_default();
        for width in WIDTHS {
            let folded = fold_binary_literal(&op, &width.left.expr(), &width.right.expr());
            let interpreted = interpret_binary(op, width.left.value(), width.right.value());
            let agreed = matches!((&folded, &interpreted),
                (Some(folded), Ok(interpreted)) if same_scalar(folded, *interpreted));
            if let Some(failure) = disagreement(name, width.name, folded, interpreted) {
                failures.push(failure);
            } else if agreed {
                widths.insert(width.name);
            }
        }
    }

    assert!(
        unconstructed.is_empty(),
        "BinOp variant(s) {unconstructed:?} cannot be constructed by `bin_op_named` and are not \
         recorded as unevaluable. Fix: add the variant to `bin_op_named`, or record why it has no \
         scalar semantics."
    );
    assert!(
        failures.is_empty(),
        "the literal folder and the reference interpreter disagree on {} (operator, width) pair(s). \
         Fix: move the row into vyre_foundation::scalar_ops so one table answers both.\n{}",
        failures.len(),
        failures.join("\n")
    );

    assert_support_invariants(
        "BinOp",
        &names,
        &answered,
        UNEVALUABLE_BIN_OPS,
        INTEGER_WIDTH_ASYMMETRY_BIN_OPS,
    );
}

#[test]
fn folder_and_interpreter_agree_on_every_unary_operator_and_width() {
    let names = variant_names(&spec_source("un_op.rs"), "UnOp");
    assert!(
        names.len() >= MIN_UN_OP_VARIANTS,
        "Fix: parsed only {} UnOp variants ({names:?}); the source parse is broken and every \
         per-operator assertion below would be vacuous. Repair `variant_names` before trusting \
         a green run.",
        names.len()
    );

    let constructible = table_names(UNEVALUABLE_UN_OPS);
    let mut answered: BTreeMap<String, BTreeSet<&'static str>> = BTreeMap::new();
    let mut failures = Vec::new();
    let mut unconstructed = Vec::new();

    for name in &names {
        let Some(op) = un_op_named(name) else {
            if !constructible.contains(name) {
                unconstructed.push(name);
            }
            continue;
        };
        let widths = answered.entry(name.clone()).or_default();
        for width in WIDTHS {
            let folded = fold_unary_literal(&op, &width.left.expr());
            let interpreted = interpret_unary(op.clone(), width.left.value());
            let agreed = matches!((&folded, &interpreted),
                (Some(folded), Ok(interpreted)) if same_scalar(folded, *interpreted));
            if let Some(failure) = disagreement(name, width.name, folded, interpreted) {
                failures.push(failure);
            } else if agreed {
                widths.insert(width.name);
            }
        }
    }

    assert!(
        unconstructed.is_empty(),
        "UnOp variant(s) {unconstructed:?} cannot be constructed by `un_op_named` and are not \
         recorded as unevaluable. Fix: add the variant to `un_op_named`, or record why it has no \
         scalar semantics."
    );
    assert!(
        failures.is_empty(),
        "the literal folder and the reference interpreter disagree on {} (operator, width) pair(s). \
         Fix: move the row into vyre_foundation::scalar_ops so one table answers both.\n{}",
        failures.len(),
        failures.join("\n")
    );

    assert_support_invariants(
        "UnOp",
        &names,
        &answered,
        UNEVALUABLE_UN_OPS,
        INTEGER_WIDTH_ASYMMETRY_UN_OPS,
    );
}

/// `u64` is the one width the interpreter carries that the folder cannot: the
/// expression IR has no 64-bit literal, so a `Value::U64` result has nowhere to
/// land. Adding `Expr::LitU64` without adding a u64 row to [`WIDTHS`] would
/// reopen the divergence this suite closes, so pin the absence.
#[test]
fn folder_cannot_produce_a_sixty_four_bit_literal() {
    for source in [
        Expr::LitU32(1),
        Expr::LitI32(1),
        Expr::LitF32(1.0),
        Expr::LitBool(true),
    ] {
        assert_eq!(
            fold_cast_literal(&DataType::U64, &source),
            None,
            "Fix: the folder gained a 64-bit literal result. Add a u64 row to WIDTHS so the \
             agreement gate covers it."
        );
    }
}
