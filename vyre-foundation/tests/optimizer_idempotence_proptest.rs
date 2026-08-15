//! Every optimizer entry point must reach a fixed point without changing what
//! the program computes.
//!
//! The entry points are a declared table, not one test each: `canonicalize` and
//! the full registered pipeline owe the same contract, and the two suites that
//! asserted them had drifted into asserting different halves of it -- one
//! compared structural equality after two runs, the other wire bytes after
//! three, and only one checked reference semantics. The table runs the union on
//! every row.
//!
//! The per-pass corpus below is the second half: a fixed point over the whole
//! pipeline does not imply one for each pass in isolation, which is the
//! property the scheduler's convergence loop depends on.

use proptest::prelude::*;
use vyre_foundation::ir::{BufferDecl, DataType, Expr, Node, Program};
use vyre_foundation::optimizer as optimize;
use vyre_foundation::optimizer::passes::algebraic::canonicalize_engine as canonicalize;
use vyre_foundation::optimizer::{
    registered_pass_registrations, PassScheduler, ProgramPassKind, ProgramPassRegistration,
};

#[path = "contract_cases/optimizer_program_corpus.rs"]
mod corpus;

use corpus::{
    assert_fixed_point_and_semantics, canonical_wire, output_only_store, program_strategy,
    program_with_body, test_output_buffer, OptimizerEntryPoint,
};

/// Every optimizer entry point held to the fixed-point contract.
const ENTRY_POINTS: &[OptimizerEntryPoint] = &[
    OptimizerEntryPoint {
        label: "vyre_foundation::optimizer::passes::algebraic::canonicalize_engine::run",
        run: canonicalize::run,
    },
    OptimizerEntryPoint {
        label: "vyre_foundation::optimizer::optimize",
        run: run_full_optimizer,
    },
];

fn run_full_optimizer(program: Program) -> Program {
    optimize::optimize(program).expect("Fix: the registered optimizer must converge")
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 96, .. ProptestConfig::default() })]

    #[test]
    fn every_optimizer_entry_point_is_a_semantics_preserving_fixed_point(
        program in program_strategy(),
    ) {
        for entry in ENTRY_POINTS {
            assert_fixed_point_and_semantics(entry, program.clone())?;
        }
    }
}

#[test]
fn optimize_then_wire_roundtrip_preserves_program_smoke() {
    // Mirrors `optimizer_reference_parity_smoke`  -  enough IR for canonicalize, const fold,
    // CSE/DCE, then full wire round-trip.
    let program = output_only_store(Expr::add(
        Expr::mul(Expr::u32(3), Expr::u32(4)),
        Expr::sub(Expr::u32(10), Expr::u32(2)),
    ));
    let optimized = run_full_optimizer(program);
    let bytes = canonical_wire("vyre_foundation::optimizer::optimize", &optimized);
    let back = Program::from_wire(&bytes).expect("Fix: optimized smoke program must decode");
    assert_eq!(back, optimized);
}

/// Programs that make at least one registered pass fire, so the per-pass fixed
/// point below is measured on passes that actually rewrote something.
fn pass_contract_corpus() -> Vec<Program> {
    let arithmetic = output_only_store(Expr::add(
        Expr::mul(Expr::u32(3), Expr::u32(4)),
        Expr::mul(Expr::gid_x(), Expr::u32(8)),
    ));

    let dead_buffer = Program::wrapped(
        vec![
            BufferDecl::read_write("dead", 0, DataType::U32).with_count(1),
            test_output_buffer(),
        ],
        [1, 1, 1],
        vec![
            Node::store("dead", Expr::u32(0), Expr::u32(99)),
            Node::store("out", Expr::u32(0), Expr::u32(7)),
        ],
    );

    let fusion_candidate = program_with_body(vec![
        Node::let_bind("a", Expr::add(Expr::gid_x(), Expr::u32(1))),
        Node::let_bind("b", Expr::mul(Expr::var("a"), Expr::u32(2))),
        Node::store("out", Expr::u32(0), Expr::var("b")),
    ]);

    let autotune_candidate = Program::wrapped(
        vec![BufferDecl::read_write("out", 0, DataType::U32).with_count(256)],
        [1, 1, 1],
        vec![Node::store("out", Expr::gid_x(), Expr::gid_x())],
    );

    let atomic_condition = Program::wrapped(
        vec![BufferDecl::read_write("out", 0, DataType::U32).with_count(1)],
        [1, 1, 1],
        vec![Node::if_then(
            Expr::atomic_add("out", Expr::u32(0), Expr::u32(1)),
            vec![Node::store("out", Expr::u32(0), Expr::u32(2))],
        )],
    );

    let redundant_store = Program::wrapped(
        vec![BufferDecl::output("out", 0, DataType::U32).with_count(4)],
        [1, 1, 1],
        vec![
            Node::store("out", Expr::u32(0), Expr::u32(1)),
            Node::store("out", Expr::u32(0), Expr::u32(2)),
            Node::let_bind("forwarded", Expr::load("out", Expr::u32(0))),
            Node::store("out", Expr::u32(1), Expr::var("forwarded")),
        ],
    );

    vec![
        arithmetic,
        dead_buffer,
        fusion_candidate,
        autotune_candidate,
        atomic_condition,
        redundant_store,
    ]
}

/// A scheduler holding `registration` and every pass it declares a requirement
/// on, so a pass whose candidate shape another pass produces still gets a
/// program it can rewrite.
fn scheduler_for(registration: &'static ProgramPassRegistration) -> PassScheduler {
    let required = registration.metadata.requires;
    let mut passes: Vec<ProgramPassKind> = registered_pass_registrations()
        .expect("Fix: the registered pass graph must schedule")
        .iter()
        .filter(|candidate| required.contains(&candidate.metadata.name))
        .map(|candidate| ProgramPassKind::from_boxed((candidate.factory)()))
        .collect();
    passes.push(ProgramPassKind::from_boxed((registration.factory)()));
    PassScheduler::with_passes(passes).with_max_iterations(8)
}

/// Every registered pass, run alone, must converge and hold its fixed point.
///
/// The pass set comes from the inventory registry at run time. The list of
/// seven names this replaces named a fifth of the registry and could not go
/// stale loudly: a pass registered after it was written was simply never held
/// to the contract, and nothing said so.
#[test]
fn every_registered_optimizer_pass_converges_and_is_idempotent_on_contract_corpus() {
    let registrations =
        registered_pass_registrations().expect("Fix: the registered pass graph must schedule");
    assert!(
        registrations.len() > 1,
        "Fix: the pass registry reported {} passes, so this gate proves nothing",
        registrations.len()
    );
    let corpus = pass_contract_corpus();

    for registration in registrations.iter() {
        let pass_name = registration.metadata.name;
        for (case_index, program) in corpus.iter().enumerate() {
            let run = |input: Program, stage: &str| {
                scheduler_for(registration)
                    .run(input)
                    .unwrap_or_else(|error| {
                        panic!(
                            "Fix: optimizer pass `{pass_name}` must converge {stage} on contract corpus case {case_index}: {error}"
                        )
                    })
            };
            let once = run(program.clone(), "from the input");
            let twice = run(once.clone(), "after the first run");
            let thrice = run(twice.clone(), "after the second run");

            assert_eq!(
                canonical_wire(pass_name, &once),
                canonical_wire(pass_name, &twice),
                "Fix: optimizer pass `{pass_name}` is not idempotent after convergence on contract corpus case {case_index}"
            );
            assert_eq!(
                canonical_wire(pass_name, &twice),
                canonical_wire(pass_name, &thrice),
                "Fix: optimizer pass `{pass_name}` did not hold its fixed point on contract corpus case {case_index}"
            );
        }
    }
}
