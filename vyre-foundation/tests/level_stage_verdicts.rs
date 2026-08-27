//! The three levels this crate owns accept their own IR and reject broken IR.
//!
//! WHY this suite exists: a level stage that answers `Verified` for everything
//! certifies what it never checked, and reads exactly like a stage that works.
//! Each case below pairs a subject the level must accept with one it must
//! reject, so a verifier replaced by a constant answer turns this suite red.
//!
//! `level_stage_closure` in the linkage owner holds the other half: that every
//! declared level has one stage at all. That question needs a binary linking
//! every crate that owns a level's subject, and this one links three of them.
//!
//! What this does NOT catch: the whole-graph and schedule canonical forms are
//! wire encodings their constructors produce, so no subject of either level
//! exists in a non-canonical form to feed the predicate. The logical level's
//! canonical form is a rewrite, and its rejection is asserted here.

use std::collections::BTreeMap;

use vyre_foundation::ir::{
    BufferAccess, BufferDecl, DataType, Expr, GraphInput, GraphOutput, Node, Program, ProgramGraph,
    ProgramGraphIdentityContext, ShapeDim, ValueContract, ValueLifetime,
};
use vyre_foundation::optimizer::level_contract::{stage_for_level, GraphComposition, LevelVerdict};
use vyre_foundation::schedule::SelectedSchedule;
use vyre_spec::IrLevel;
use vyre_test_support::pass_programs::element_copy;

fn value_contract(access: BufferAccess, lifetime: ValueLifetime) -> ValueContract {
    ValueContract {
        dtype: DataType::F32,
        shape: vec![ShapeDim::Symbol("tokens".into()), ShapeDim::Known(8)],
        access,
        lifetime,
    }
}

fn copy_program() -> Program {
    Program::wrapped(
        vec![
            BufferDecl::storage("input", 0, BufferAccess::ReadOnly, DataType::F32),
            BufferDecl::storage("output", 1, BufferAccess::ReadWrite, DataType::F32),
        ],
        [1, 1, 1],
        vec![element_copy("output", 0, "input", 0)],
    )
}

/// One node consuming one external value and producing one output.
fn one_node_graph() -> ProgramGraph {
    let mut graph = ProgramGraph::new();
    let input = graph
        .add_external_value(
            "input",
            value_contract(BufferAccess::ReadOnly, ValueLifetime::Invocation),
        )
        .expect("Fix: the fixture input must register");
    graph
        .add_node(
            "node.0".to_owned(),
            copy_program(),
            vec![GraphInput {
                buffer: "input".into(),
                value: input,
                contract: value_contract(BufferAccess::ReadOnly, ValueLifetime::Invocation),
            }],
            vec![GraphOutput {
                buffer: "output".into(),
                name: "output".to_owned(),
                contract: value_contract(BufferAccess::ReadWrite, ValueLifetime::Output),
                retained_successor_of: None,
            }],
        )
        .expect("Fix: the fixture node must connect");
    graph
}

fn provenance(bindings: BTreeMap<String, u64>) -> ProgramGraphIdentityContext {
    ProgramGraphIdentityContext {
        artifact_schema_version: 1,
        configuration_digest: [7u8; 32],
        symbolic_bindings: bindings,
        constant_identities: BTreeMap::new(),
    }
}

fn bound_symbols() -> BTreeMap<String, u64> {
    BTreeMap::from([("tokens".to_owned(), 4)])
}

/// A composition with complete provenance verifies; one with a binding the
/// graph never mentions does not.
#[test]
fn whole_graph_stage_rejects_provenance_the_graph_does_not_carry() {
    let stage =
        stage_for_level(IrLevel::WholeGraph).expect("Fix: the whole-graph stage must exist");

    let good = GraphComposition {
        graph: one_node_graph(),
        provenance: provenance(bound_symbols()),
    };
    assert_eq!(
        stage.verify(&good),
        LevelVerdict::Verified,
        "Fix: a composition whose provenance binds every graph symbol must verify"
    );
    assert_eq!(stage.is_canonical(&good), LevelVerdict::Verified);

    let mut extra = bound_symbols();
    extra.insert("absent".to_owned(), 1);
    let bad = GraphComposition {
        graph: one_node_graph(),
        provenance: provenance(extra),
    };
    let verdict = stage.verify(&bad);
    assert!(
        matches!(verdict, LevelVerdict::Rejected(_)),
        "Fix: a binding no graph contract mentions must be rejected, got {verdict:?}"
    );

    let unbound = GraphComposition {
        graph: one_node_graph(),
        provenance: provenance(BTreeMap::new()),
    };
    let verdict = stage.verify(&unbound);
    assert!(
        matches!(verdict, LevelVerdict::Rejected(_)),
        "Fix: an unbound graph symbol must be rejected, got {verdict:?}"
    );
}

/// A valid program verifies; a store to an undeclared buffer does not.
#[test]
fn logical_stage_rejects_a_program_the_validator_rejects() {
    let stage = stage_for_level(IrLevel::Logical).expect("Fix: the logical stage must exist");

    let good = copy_program();
    assert_eq!(
        stage.verify(&good),
        LevelVerdict::Verified,
        "Fix: a valid program must verify at the logical level"
    );

    let bad = Program::wrapped(
        vec![BufferDecl::storage(
            "input",
            0,
            BufferAccess::ReadOnly,
            DataType::F32,
        )],
        [1, 1, 1],
        vec![element_copy("absent", 0, "input", 0)],
    );
    let verdict = stage.verify(&bad);
    assert!(
        matches!(verdict, LevelVerdict::Rejected(_)),
        "Fix: a store to an undeclared buffer must be rejected, got {verdict:?}"
    );
}

/// The logical canonical form is a rewrite, so a program in another form is
/// reported as not canonical.
#[test]
fn logical_stage_reports_a_non_canonical_program() {
    let stage = stage_for_level(IrLevel::Logical).expect("Fix: the logical stage must exist");

    let canonical = copy_program().canonicalized();
    assert_eq!(
        stage.is_canonical(&canonical),
        LevelVerdict::Verified,
        "Fix: a canonicalized program must be reported canonical"
    );

    let swapped = Program::wrapped(
        vec![
            BufferDecl::storage("input", 0, BufferAccess::ReadOnly, DataType::F32),
            BufferDecl::storage("output", 1, BufferAccess::ReadWrite, DataType::F32),
        ],
        [1, 1, 1],
        vec![Node::store(
            "output",
            Expr::u32(0),
            Expr::BinOp {
                op: vyre_foundation::ir::BinOp::Add,
                left: Box::new(Expr::u32(1)),
                right: Box::new(Expr::load("input", Expr::u32(0))),
            },
        )],
    );
    assert_ne!(
        swapped.entry(),
        swapped.canonicalized().entry(),
        "Fix: this fixture is meant to be non-canonical; the canonicalizer changed, not the stage"
    );
    let verdict = stage.is_canonical(&swapped);
    assert!(
        matches!(verdict, LevelVerdict::Rejected(_)),
        "Fix: a program the canonicalizer rewrites is not canonical, got {verdict:?}"
    );
}

/// A replayable schedule verifies; one carrying a version this build does not
/// implement does not.
#[test]
fn schedule_stage_rejects_a_schedule_of_another_version() {
    let stage = stage_for_level(IrLevel::Schedule).expect("Fix: the schedule stage must exist");

    let good = SelectedSchedule::synthetic(2);
    assert_eq!(
        stage.verify(&good),
        LevelVerdict::Verified,
        "Fix: a synthetic schedule must verify at the schedule level"
    );
    assert_eq!(stage.is_canonical(&good), LevelVerdict::Verified);

    let wire = good
        .canonical_wire()
        .expect("Fix: a valid schedule must encode");
    let mut json: serde_json::Value =
        serde_json::from_slice(&wire).expect("Fix: the canonical wire must be JSON");
    json["version"] = serde_json::Value::from(0u16);
    let stale: SelectedSchedule =
        serde_json::from_value(json).expect("Fix: a version edit must still deserialize");
    let verdict = stage.verify(&stale);
    assert!(
        matches!(verdict, LevelVerdict::Rejected(_)),
        "Fix: a schedule of an unimplemented version must be rejected, got {verdict:?}"
    );
}

/// Each of the three stages refuses a subject belonging to another level.
#[test]
fn each_stage_refuses_another_levels_subject() {
    let graph_stage =
        stage_for_level(IrLevel::WholeGraph).expect("Fix: the whole-graph stage must exist");
    let logical_stage =
        stage_for_level(IrLevel::Logical).expect("Fix: the logical stage must exist");
    let schedule_stage =
        stage_for_level(IrLevel::Schedule).expect("Fix: the schedule stage must exist");

    let program = copy_program();
    let schedule = SelectedSchedule::synthetic(1);

    assert_eq!(
        graph_stage.verify(&program),
        LevelVerdict::WrongSubject {
            expected: "GraphComposition"
        }
    );
    assert_eq!(
        logical_stage.verify(&schedule),
        LevelVerdict::WrongSubject {
            expected: "Program"
        }
    );
    assert_eq!(
        schedule_stage.verify(&program),
        LevelVerdict::WrongSubject {
            expected: "SelectedSchedule"
        }
    );
}
