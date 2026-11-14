//! Deterministic release corpus for the canonical `Program` optimizer.
//!
//! Cases are generated from semantic `Program` IR, exercised through the live
//! registered release scheduler, and summarized for release tooling. Descriptor
//! analyses remain in `vyre-compiler` and are not optimization passes.

use crate::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};

/// Minimum generated cases required by the release optimization gate.
pub const RELEASE_MIN_OPTIMIZATION_CASES: usize = 4_096;

/// The semantic optimization families the release corpus generates, and the
/// set every release consumer of that corpus checks its evidence against.
///
/// The corpus generator and each evidence reader used to restate these eight
/// names, so a family added to the generator was silently uncovered by the
/// gates that were supposed to demand it. They read the list from here.
pub const RELEASE_OPTIMIZATION_FAMILIES: [&str; 8] = [
    "scalar-algebra",
    "strength-reduction",
    "fusion-cse",
    "dead-code",
    "memory-dataflow",
    "loop-transform",
    "control-flow",
    "canonicalization",
];

/// One deterministic semantic optimizer case.
#[derive(Debug, Clone)]
pub struct OptimizationCorpusCase {
    /// Stable case identifier.
    pub id: String,
    /// Semantic family exercised by the case.
    pub family: String,
    /// Runnable semantic program.
    pub program: Program,
}

/// Generate the deterministic semantic release corpus.
#[must_use]
pub fn generate_release_corpus() -> Vec<OptimizationCorpusCase> {
    let mut cases = Vec::with_capacity(RELEASE_MIN_OPTIMIZATION_CASES);
    for seed in 0..512u32 {
        for family in RELEASE_OPTIMIZATION_FAMILIES {
            cases.push(OptimizationCorpusCase {
                id: format!("foundation.optimizer.{family}.{seed:04}"),
                family: family.to_string(),
                program: program_for(family, seed),
            });
        }
    }
    cases
}

fn program_for(family: &str, seed: u32) -> Program {
    let input = BufferDecl::storage("in", 0, BufferAccess::ReadOnly, DataType::U32).with_count(64);
    let scratch =
        BufferDecl::storage("scratch", 1, BufferAccess::ReadWrite, DataType::U32).with_count(64);
    let output = BufferDecl::output("out", 2, DataType::U32).with_count(64);
    let index = Expr::bitand(Expr::gid_x(), Expr::u32(63));
    let loaded = Expr::load("in", index.clone());
    let literal = Expr::u32(seed & 31);
    let body = match family {
        "scalar-algebra" => vec![Node::store(
            "out",
            index,
            Expr::add(Expr::mul(Expr::u32(3), Expr::u32(4)), literal),
        )],
        "strength-reduction" => vec![Node::store(
            "out",
            index,
            Expr::mul(Expr::add(loaded, literal), Expr::u32(8)),
        )],
        "fusion-cse" => vec![
            Node::let_bind("a", Expr::add(loaded.clone(), literal.clone())),
            Node::let_bind("b", Expr::add(loaded, literal)),
            Node::store("out", index, Expr::add(Expr::var("a"), Expr::var("b"))),
        ],
        "dead-code" => vec![
            Node::let_bind("dead", Expr::mul(loaded.clone(), Expr::u32(0))),
            Node::store("out", index, Expr::add(loaded, literal)),
        ],
        "memory-dataflow" => vec![
            Node::store("scratch", index.clone(), loaded),
            Node::store("out", index.clone(), Expr::load("scratch", index)),
        ],
        "loop-transform" => vec![Node::Loop {
            var: "i".into(),
            from: Expr::u32(0),
            to: Expr::u32(1),
            body: vec![Node::store("out", index, Expr::add(loaded, literal))],
        }],
        "control-flow" => vec![Node::if_then_else(
            Expr::bool(true),
            vec![Node::store(
                "out",
                index.clone(),
                Expr::add(loaded, literal),
            )],
            vec![Node::store("out", index, Expr::u32(0))],
        )],
        "canonicalization" => vec![Node::store(
            "out",
            index,
            Expr::add(literal, Expr::add(loaded, Expr::u32(0))),
        )],
        _ => unreachable!("closed semantic corpus family"),
    };
    Program::wrapped(vec![input, scratch, output], [64, 1, 1], body)
        .with_entry_op_id(format!("foundation.optimizer.{family}"))
}
