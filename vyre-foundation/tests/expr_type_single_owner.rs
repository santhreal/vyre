//! Expression type inference has exactly one owner.
//!
//! `vyre_foundation::validate::typecheck::expr_type` answers "what type does
//! this expression have". The optimizer fact cache and the autodiff forward
//! pass used to carry their own `Expr` walkers, and the three answers had
//! drifted: `Expr::BufferRef` was a value in two of them, arithmetic took the
//! left operand's width instead of unifying, and a subgroup shuffle of an f32
//! was typed as a word. A second walker is therefore not a style problem, it is
//! how the optimizer and the validator come to disagree about the same program.
//!
//! Two contracts are asserted here.
//!
//! 1. The owner produces the recorded answer for every `Expr` variant. The
//!    variant space is parsed out of the enum's own source at run time, so a
//!    new variant turns this red until its answer is recorded, and a case for a
//!    variant that no longer exists is red too.
//! 2. No other file under `vyre-foundation/src` defines `fn expr_type`.
//!
//! What it does not catch: a competing walker under a different name, and a
//! wrong-but-recorded answer. The per-variant answers below are the review
//! surface for the second.

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use vyre_foundation::ir::{
    AtomicOp, BufferDecl, DataType, Expr, ExprNode, Ident, MemoryOrdering, Node, Program,
    SubgroupReduceOp, UnOp,
};
use vyre_foundation::optimizer::fact_cache::FactCache;
use vyre_test_support::monorepo::vyre_crate_directory;

/// Smallest plausible size of the `Expr` enum. A parser that silently matches
/// nothing would otherwise report perfect coverage of an empty variant space.
const MIN_EXPR_VARIANTS: usize = 10;

const F32_BUFFER: &str = "buf_f32";
const U32_BUFFER: &str = "buf_u32";
const F32_LOCAL: &str = "src_f32";
const PROBE: &str = "probe";

fn foundation_src() -> PathBuf {
    vyre_crate_directory("vyre-foundation").join("src")
}

/// Variant names of the `Expr` enum, read from the source that declares it.
fn expr_variants_from_source() -> BTreeSet<String> {
    let declaration = foundation_src().join("ir_inner/model/generated.rs");
    let text = fs::read_to_string(&declaration).unwrap_or_else(|error| {
        panic!(
            "cannot read the Expr declaration at {}: {error}. Fix: point this contract at the \
             file that declares the Expr enum.",
            declaration.display()
        )
    });

    let mut variants = BTreeSet::new();
    let mut inside = false;
    for line in text.lines() {
        let trimmed = line.trim();
        if !inside {
            inside = trimmed == "Expr {";
            continue;
        }
        if trimmed == "}" {
            break;
        }
        if trimmed.is_empty() || trimmed.starts_with("//") {
            continue;
        }
        let name: String = trimmed
            .chars()
            .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
            .collect();
        if name.starts_with(|c: char| c.is_ascii_uppercase()) {
            variants.insert(name);
        }
    }

    assert!(
        variants.len() >= MIN_EXPR_VARIANTS,
        "parsed only {} Expr variants from {}, which means the parse broke rather than the enum \
         shrinking. Fix: repair the variant scan before trusting this contract.",
        variants.len(),
        declaration.display()
    );
    variants
}

#[derive(Debug)]
struct GateExtension;

impl ExprNode for GateExtension {
    fn extension_kind(&self) -> &'static str {
        "test.expr_type_single_owner.extension"
    }
    fn debug_identity(&self) -> &str {
        "expr-type-owner-gate"
    }
    fn result_type(&self) -> Option<DataType> {
        Some(DataType::I32)
    }
    fn cse_safe(&self) -> bool {
        true
    }
    fn stable_fingerprint(&self) -> [u8; 32] {
        [7; 32]
    }
    fn validate_extension(&self) -> Result<(), String> {
        Ok(())
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

fn f32_operand() -> Expr {
    Expr::var(F32_LOCAL)
}

/// One representative expression per `Expr` variant, with the type the owner
/// must report for it.
fn recorded_answers() -> Vec<(&'static str, Expr, Option<DataType>)> {
    vec![
        ("LitU32", Expr::u32(7), Some(DataType::U32)),
        ("LitI32", Expr::i32(-7), Some(DataType::I32)),
        ("LitF32", Expr::f32(1.5), Some(DataType::F32)),
        ("LitBool", Expr::bool(true), Some(DataType::Bool)),
        // A local's declared type, not the shape of its initializer.
        ("Var", f32_operand(), Some(DataType::F32)),
        // Names a buffer instead of producing a value. Answering with a type
        // would let it pass an operand typecheck it must never pass.
        (
            "BufferRef",
            Expr::BufferRef {
                buffer: Ident::from(U32_BUFFER),
            },
            None,
        ),
        (
            "Load",
            Expr::load(F32_BUFFER, Expr::u32(0)),
            Some(DataType::F32),
        ),
        (
            "BufLen",
            Expr::BufLen {
                buffer: Ident::from(F32_BUFFER),
            },
            Some(DataType::U32),
        ),
        ("InvocationId", Expr::gid_x(), Some(DataType::U32)),
        (
            "WorkgroupId",
            Expr::WorkgroupId { axis: 0 },
            Some(DataType::U32),
        ),
        ("LocalId", Expr::LocalId { axis: 0 }, Some(DataType::U32)),
        // Arithmetic unifies its operands rather than taking one side's width.
        (
            "BinOp",
            Expr::add(f32_operand(), f32_operand()),
            Some(DataType::F32),
        ),
        (
            "UnOp",
            Expr::UnOp {
                op: UnOp::Sqrt,
                operand: Box::new(f32_operand()),
            },
            Some(DataType::F32),
        ),
        // An operation signature is not visible from the IR alone.
        (
            "Call",
            Expr::Call {
                op_id: Ident::from("unresolved.op"),
                args: vec![Expr::u32(1)],
            },
            None,
        ),
        (
            "Select",
            Expr::Select {
                cond: Box::new(Expr::bool(true)),
                true_val: Box::new(f32_operand()),
                false_val: Box::new(f32_operand()),
            },
            Some(DataType::F32),
        ),
        (
            "Cast",
            Expr::Cast {
                target: DataType::I32,
                value: Box::new(Expr::u32(3)),
            },
            Some(DataType::I32),
        ),
        (
            "Fma",
            Expr::Fma {
                a: Box::new(f32_operand()),
                b: Box::new(f32_operand()),
                c: Box::new(f32_operand()),
            },
            Some(DataType::F32),
        ),
        (
            "Atomic",
            Expr::Atomic {
                op: AtomicOp::Add,
                buffer: Ident::from(U32_BUFFER),
                index: Box::new(Expr::u32(0)),
                expected: None,
                value: Box::new(Expr::u32(1)),
                ordering: MemoryOrdering::Relaxed,
            },
            Some(DataType::U32),
        ),
        (
            "SubgroupBallot",
            Expr::SubgroupBallot {
                cond: Box::new(Expr::bool(true)),
            },
            Some(DataType::U32),
        ),
        // A shuffle moves its operand between lanes; it does not reinterpret
        // it as a word.
        (
            "SubgroupShuffle",
            Expr::SubgroupShuffle {
                value: Box::new(f32_operand()),
                lane: Box::new(Expr::u32(0)),
            },
            Some(DataType::F32),
        ),
        (
            "SubgroupReduce",
            Expr::SubgroupReduce {
                op: SubgroupReduceOp::Add,
                value: Box::new(f32_operand()),
            },
            Some(DataType::F32),
        ),
        (
            "SubgroupLocalId",
            Expr::SubgroupLocalId,
            Some(DataType::U32),
        ),
        ("SubgroupSize", Expr::SubgroupSize, Some(DataType::U32)),
        // The extension declares its own result type.
        (
            "Opaque",
            Expr::Opaque(Arc::new(GateExtension)),
            Some(DataType::I32),
        ),
    ]
}

/// The owner's answer for `probe`, observed through the optimizer fact cache.
fn owner_answer(probe: Expr) -> Option<DataType> {
    let program = Program::wrapped(
        vec![
            BufferDecl::read_write(F32_BUFFER, 0, DataType::F32).with_count(4),
            BufferDecl::read_write(U32_BUFFER, 1, DataType::U32).with_count(4),
        ],
        [1, 1, 1],
        vec![
            Node::let_bind(F32_LOCAL, Expr::f32(1.0)),
            Node::let_bind(PROBE, probe),
        ],
    );
    let cache = FactCache::derive(&program);
    let facts = cache
        .type_map
        .as_ref()
        .expect("FactCache::derive must populate type facts");
    facts.var_types.get(&Ident::from(PROBE)).cloned()
}

#[test]
fn owner_answers_every_expr_variant() {
    let declared = expr_variants_from_source();
    let answers = recorded_answers();
    let covered: BTreeSet<String> = answers
        .iter()
        .map(|(name, ..)| (*name).to_owned())
        .collect();

    let missing: Vec<&String> = declared.difference(&covered).collect();
    assert!(
        missing.is_empty(),
        "Expr variants with no recorded type answer: {missing:?}. Fix: decide what \
         vyre_foundation::validate::typecheck::expr_type must report for each and record it here."
    );
    let stale: Vec<&String> = covered.difference(&declared).collect();
    assert!(
        stale.is_empty(),
        "recorded answers for Expr variants that no longer exist: {stale:?}. Fix: drop the case."
    );

    for (name, probe, expected) in answers {
        assert_eq!(
            owner_answer(probe),
            expected,
            "expr_type reported the wrong type for Expr::{name}. Fix: the owner in \
             vyre_foundation::validate::typecheck must answer every variant, and this answer is \
             the one the validator, the optimizer, and autodiff all read."
        );
    }
}

fn rust_sources(dir: &Path, out: &mut Vec<PathBuf>) {
    let entries = fs::read_dir(dir)
        .unwrap_or_else(|error| panic!("cannot read {}: {error}", dir.display()))
        .filter_map(Result::ok);
    for entry in entries {
        let path = entry.path();
        if path.is_dir() {
            rust_sources(&path, out);
        } else if path.extension().is_some_and(|ext| ext == "rs") {
            out.push(path);
        }
    }
}

#[test]
fn only_one_file_defines_expr_type() {
    let src = foundation_src();
    let mut sources = Vec::new();
    rust_sources(&src, &mut sources);
    assert!(
        !sources.is_empty(),
        "no Rust sources found under {}. Fix: repair the source scan.",
        src.display()
    );

    let mut definitions = Vec::new();
    for path in sources {
        let text = fs::read_to_string(&path)
            .unwrap_or_else(|error| panic!("cannot read {}: {error}", path.display()));
        if text.contains("fn expr_type") {
            definitions.push(
                path.strip_prefix(&src)
                    .unwrap_or(&path)
                    .display()
                    .to_string(),
            );
        }
    }
    definitions.sort();

    assert_eq!(
        definitions,
        vec!["validate/typecheck/expr_type.rs".to_string()],
        "expression type inference must be defined once. Fix: delete the competing walker and \
         route its caller through vyre_foundation::validate::typecheck::expr_type, supplying its \
         environment through the TypeEnv trait."
    );
}
