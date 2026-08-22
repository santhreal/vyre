//! Deterministic release corpus for the canonical `Program` optimizer.
//!
//! Cases are generated from semantic `Program` IR, exercised through the live
//! registered release scheduler, and summarized for release tooling. Descriptor
//! analyses remain in `vyre-lower` and are not optimization passes.

use std::collections::BTreeMap;

use crate::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};
use crate::optimizer::{registered_passes_for_profile, OptimizerProfile, PassScheduler};

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

/// Generated case count for one semantic family.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct OptimizationFamilyCount {
    /// Stable semantic family identifier.
    pub family: String,
    /// Number of generated cases.
    pub cases: usize,
}

/// Machine-readable validation summary for the semantic corpus.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct OptimizationCorpusManifest {
    /// Artifact schema version.
    pub schema_version: u32,
    /// Minimum required case count.
    pub required_min_cases: usize,
    /// Number of generated programs.
    pub generated_cases: usize,
    /// Number of programs accepted by the canonical optimizer.
    pub verified_cases: usize,
    /// Number of programs changed by at least one registered pass.
    pub optimized_cases: usize,
    /// Number of programs that exhausted scheduler convergence.
    pub non_converged_cases: usize,
    /// Total semantic node count before optimization.
    pub total_nodes_before: usize,
    /// Total semantic node count after optimization.
    pub total_nodes_after: usize,
    /// Number of considered pass instances.
    pub pass_instance_count: usize,
    /// Number of pass instances that changed their program.
    pub changed_pass_instances: usize,
    /// Per-family case counts.
    pub families: Vec<OptimizationFamilyCount>,
    /// Release blockers found while validating the corpus.
    pub blockers: Vec<String>,
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

/// Run every generated program through the canonical release scheduler.
#[must_use]
pub fn manifest_for(cases: &[OptimizationCorpusCase]) -> OptimizationCorpusManifest {
    let passes = match registered_passes_for_profile(OptimizerProfile::Release) {
        Ok(passes) => passes,
        Err(error) => {
            return OptimizationCorpusManifest {
                schema_version: 2,
                required_min_cases: RELEASE_MIN_OPTIMIZATION_CASES,
                generated_cases: cases.len(),
                verified_cases: 0,
                optimized_cases: 0,
                non_converged_cases: cases.len(),
                total_nodes_before: 0,
                total_nodes_after: 0,
                pass_instance_count: 0,
                changed_pass_instances: 0,
                families: family_counts(cases),
                blockers: vec![format!(
                    "registered release pass scheduling failed: {error}"
                )],
            };
        }
    };
    let scheduler = match PassScheduler::try_with_passes(passes) {
        Ok(scheduler) => scheduler
            .with_cost_monotone_enforcement(true)
            .with_effect_handler_enforcement(true)
            .with_linear_type_enforcement(true)
            .with_shape_predicate_enforcement(true),
        Err(error) => {
            return OptimizationCorpusManifest {
                schema_version: 2,
                required_min_cases: RELEASE_MIN_OPTIMIZATION_CASES,
                generated_cases: cases.len(),
                verified_cases: 0,
                optimized_cases: 0,
                non_converged_cases: cases.len(),
                total_nodes_before: 0,
                total_nodes_after: 0,
                pass_instance_count: 0,
                changed_pass_instances: 0,
                families: family_counts(cases),
                blockers: vec![format!("release optimizer construction failed: {error}")],
            };
        }
    };

    let mut verified_cases = 0usize;
    let mut optimized_cases = 0usize;
    let mut non_converged_cases = 0usize;
    let mut total_nodes_before = 0usize;
    let mut total_nodes_after = 0usize;
    let mut pass_instance_count = 0usize;
    let mut changed_pass_instances = 0usize;
    let mut blockers = Vec::new();

    for case in cases {
        let nodes_before = case.program.stats().node_count;
        total_nodes_before = total_nodes_before.saturating_add(nodes_before);
        match scheduler.run_with_metrics(case.program.clone()) {
            Ok(report) => {
                verified_cases += 1;
                let changed = report.passes.iter().filter(|metric| metric.changed).count();
                if changed != 0 {
                    optimized_cases += 1;
                }
                pass_instance_count = pass_instance_count.saturating_add(report.passes.len());
                changed_pass_instances = changed_pass_instances.saturating_add(changed);
                total_nodes_after =
                    total_nodes_after.saturating_add(report.program.stats().node_count);
                if let Err(error) = report.program.validate() {
                    blockers.push(format!(
                        "case `{}` produced invalid optimized Program: {error}",
                        case.id
                    ));
                }
            }
            Err(error) => {
                non_converged_cases += 1;
                blockers.push(format!(
                    "case `{}` failed canonical semantic optimization: {error}",
                    case.id
                ));
            }
        }
    }

    OptimizationCorpusManifest {
        schema_version: 2,
        required_min_cases: RELEASE_MIN_OPTIMIZATION_CASES,
        generated_cases: cases.len(),
        verified_cases,
        optimized_cases,
        non_converged_cases,
        total_nodes_before,
        total_nodes_after,
        pass_instance_count,
        changed_pass_instances,
        families: family_counts(cases),
        blockers,
    }
}

fn family_counts(cases: &[OptimizationCorpusCase]) -> Vec<OptimizationFamilyCount> {
    let mut counts = BTreeMap::<String, usize>::new();
    for case in cases {
        *counts.entry(case.family.clone()).or_default() += 1;
    }
    counts
        .into_iter()
        .map(|(family, cases)| OptimizationFamilyCount { family, cases })
        .collect()
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
