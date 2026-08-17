//! Comprehensive contract tests for Section 182.9.5:
//! Transitive `Expr::Call` effects and capabilities fixed-point closure.

use vyre_foundation::ir::{BufferDecl, DataType, Expr, Node, Program};
use vyre_foundation::operation::{
    CallGraphClosure, OperationEffects, OperationRegistration, OperationRegistry, OperationTier,
};
use vyre_foundation::program_caps::RequiredCapabilities;

// -----------------------------------------------------------------------------
// Fixture program helpers
// -----------------------------------------------------------------------------

/// Leaf program: pure arithmetic add with read-only buffers.
fn make_pure_leaf_program() -> Program {
    Program::wrapped(
        vec![
            BufferDecl::read("in_a", 0, DataType::U32).with_count(4),
            BufferDecl::read("in_b", 1, DataType::U32).with_count(4),
        ],
        [1, 1, 1],
        vec![Node::let_bind(
            "sum",
            Expr::add(Expr::u32(10), Expr::u32(20)),
        )],
    )
}

/// Leaf program with side effects: writes to output buffer and contains an atomic op.
fn make_atomic_write_leaf_program() -> Program {
    Program::wrapped(
        vec![BufferDecl::output("out_buf", 0, DataType::U32).with_count(4)],
        [1, 1, 1],
        vec![
            Node::store("out_buf", Expr::u32(0), Expr::u32(42)),
            Node::let_bind(
                "atm",
                Expr::atomic_add("out_buf", Expr::u32(1), Expr::u32(1)),
            ),
        ],
    )
}

/// Leaf program requiring specialized capabilities: uses f64, trap, and subgroup ops.
fn make_specialized_caps_leaf_program() -> Program {
    Program::wrapped(
        vec![BufferDecl::read("f64_buf", 0, DataType::F64).with_count(4)],
        [32, 1, 1],
        vec![
            Node::let_bind("sub_val", Expr::subgroup_add(Expr::f64(1.0))),
            Node::trap(Expr::u32(0), "arithmetic failure"),
        ],
    )
}

/// Caller program that invokes another operation via `Expr::Call`.
fn make_caller_program(callee_id: &'static str) -> Program {
    Program::wrapped(
        vec![BufferDecl::read("caller_in", 0, DataType::U32).with_count(4)],
        [1, 1, 1],
        vec![Node::let_bind(
            "call_res",
            Expr::call(callee_id, vec![Expr::u32(5)]),
        )],
    )
}

// -----------------------------------------------------------------------------
// Test cases
// -----------------------------------------------------------------------------

/// WHY (182.9.5): Calling an operation transitively propagates its memory
/// and synchronization effects (writes, atomics, synchronization) up the call graph.
#[test]
fn transitive_effects_propagate_across_multi_hop_calls() {
    let reg_c = OperationRegistration::library(
        "vyre-libs::test::closure::leaf_c",
        make_atomic_write_leaf_program,
        None,
        None,
    );
    let reg_b = OperationRegistration::library(
        "vyre-libs::test::closure::intermediate_b",
        || make_caller_program("vyre-libs::test::closure::leaf_c"),
        None,
        None,
    );
    let reg_a = OperationRegistration::library(
        "vyre-libs::test::closure::root_a",
        || make_caller_program("vyre-libs::test::closure::intermediate_b"),
        None,
        None,
    );

    let closure = CallGraphClosure::solve_from_registrations([&reg_a, &reg_b, &reg_c]);

    // 1. Check direct facts vs transitive facts on Leaf C
    let c_direct_eff = closure
        .direct_effects
        .get("vyre-libs::test::closure::leaf_c")
        .unwrap();
    assert!(c_direct_eff.writes, "Leaf C directly writes");
    assert!(c_direct_eff.atomics, "Leaf C directly has atomics");

    // 2. Direct facts on intermediate B and root A do NOT have writes/atomics locally
    let b_direct_eff = closure
        .direct_effects
        .get("vyre-libs::test::closure::intermediate_b")
        .unwrap();
    assert!(
        !b_direct_eff.writes,
        "B directly only has an input buffer, no writes"
    );
    assert!(!b_direct_eff.atomics, "B directly has no atomic nodes");

    let a_direct_eff = closure
        .direct_effects
        .get("vyre-libs::test::closure::root_a")
        .unwrap();
    assert!(
        !a_direct_eff.writes,
        "A directly only has an input buffer, no writes"
    );
    assert!(!a_direct_eff.atomics, "A directly has no atomic nodes");

    // 3. Transitive facts on intermediate B and root A DO contain writes and atomics from C!
    let b_trans_eff = closure
        .transitive_effects("vyre-libs::test::closure::intermediate_b")
        .unwrap();
    assert!(b_trans_eff.writes, "B transitively inherits writes from C");
    assert!(
        b_trans_eff.atomics,
        "B transitively inherits atomics from C"
    );

    let a_trans_eff = closure
        .transitive_effects("vyre-libs::test::closure::root_a")
        .unwrap();
    assert!(
        a_trans_eff.writes,
        "A transitively inherits writes from C across multi-hop"
    );
    assert!(
        a_trans_eff.atomics,
        "A transitively inherits atomics from C across multi-hop"
    );
}

/// WHY (182.9.5): Required capabilities (f64, subgroup ops, traps, async dispatch)
/// propagate transitively from nested callees to parent callers.
#[test]
fn transitive_capabilities_propagate_across_call_graph() {
    let reg_leaf = OperationRegistration::library(
        "vyre-libs::test::closure::caps_leaf",
        make_specialized_caps_leaf_program,
        None,
        None,
    );
    let reg_parent = OperationRegistration::library(
        "vyre-libs::test::closure::caps_parent",
        || make_caller_program("vyre-libs::test::closure::caps_leaf"),
        None,
        None,
    );

    let closure = CallGraphClosure::solve_from_registrations([&reg_leaf, &reg_parent]);

    let parent_direct_caps = closure
        .direct_capabilities
        .get("vyre-libs::test::closure::caps_parent")
        .unwrap();
    assert!(!parent_direct_caps.f64, "Parent directly does not use f64");
    assert!(
        !parent_direct_caps.subgroup_ops,
        "Parent directly does not use subgroup ops"
    );
    assert!(
        !parent_direct_caps.trap,
        "Parent directly does not emit traps"
    );

    let parent_trans_caps = closure
        .transitive_capabilities("vyre-libs::test::closure::caps_parent")
        .unwrap();
    assert!(
        parent_trans_caps.f64,
        "Parent transitively inherits f64 requirement"
    );
    assert!(
        parent_trans_caps.subgroup_ops,
        "Parent transitively inherits subgroup requirement"
    );
    assert!(
        parent_trans_caps.trap,
        "Parent transitively inherits trap propagation requirement"
    );
    assert_eq!(
        parent_trans_caps.max_workgroup_size,
        [32, 1, 1],
        "Parent inherits workgroup size extent"
    );
}

/// WHY (182.9.5): An unresolved or missing callee causes the caller to fail closed,
/// inheriting the strongest applicable effects and capabilities.
#[test]
fn missing_callee_defaults_to_strongest_effects_and_capabilities() {
    let reg_caller = OperationRegistration::library(
        "vyre-libs::test::closure::calls_missing",
        || make_caller_program("external_unknown::missing_operation"),
        None,
        None,
    );

    let closure = CallGraphClosure::solve_from_registrations([&reg_caller]);

    assert!(closure.is_unclosed_or_cyclic("vyre-libs::test::closure::calls_missing"));
    let trans_eff = closure
        .transitive_effects("vyre-libs::test::closure::calls_missing")
        .unwrap();
    assert_eq!(
        trans_eff,
        OperationEffects::ALL,
        "Unresolved callee must cause caller to default to strongest applicable effects"
    );

    let trans_caps = closure
        .transitive_capabilities("vyre-libs::test::closure::calls_missing")
        .unwrap();
    assert_eq!(
        trans_caps,
        RequiredCapabilities::all(),
        "Unresolved callee must cause caller to default to strongest applicable capabilities"
    );
}

/// WHY (182.9.5): A signature-only operation without an explicit semantic contract
/// defaults to strongest effects and capabilities; with an explicit contract, it obeys it.
#[test]
fn signature_only_callee_fails_closed_unless_explicit_contract_closes_it() {
    // 1. Signature-only without explicit contract -> fails closed
    let reg_sig_open = OperationRegistration::new(
        "vyre-libs::test::closure::sig_open",
        OperationTier::Library,
        None,
        None,
        None,
    );
    let reg_caller_open = OperationRegistration::library(
        "vyre-libs::test::closure::caller_sig_open",
        || make_caller_program("vyre-libs::test::closure::sig_open"),
        None,
        None,
    );

    let closure_open =
        CallGraphClosure::solve_from_registrations([&reg_sig_open, &reg_caller_open]);
    assert!(closure_open.is_unclosed_or_cyclic("vyre-libs::test::closure::sig_open"));
    assert_eq!(
        closure_open
            .transitive_effects("vyre-libs::test::closure::caller_sig_open")
            .unwrap(),
        OperationEffects::ALL
    );

    // 2. Signature-only with explicit closed semantic contract -> respected
    let closed_eff = OperationEffects {
        reads: true,
        writes: false,
        atomics: false,
        synchronizes: false,
    };
    let closed_caps = RequiredCapabilities::none();
    let reg_sig_closed = OperationRegistration::new(
        "vyre-libs::test::closure::sig_closed",
        OperationTier::Library,
        None,
        None,
        None,
    )
    .with_explicit_effects(closed_eff)
    .with_explicit_capabilities(closed_caps);

    let reg_caller_closed = OperationRegistration::library(
        "vyre-libs::test::closure::caller_sig_closed",
        || make_caller_program("vyre-libs::test::closure::sig_closed"),
        None,
        None,
    );

    let closure_closed =
        CallGraphClosure::solve_from_registrations([&reg_sig_closed, &reg_caller_closed]);
    assert!(!closure_closed.is_unclosed_or_cyclic("vyre-libs::test::closure::sig_closed"));
    assert_eq!(
        closure_closed
            .transitive_effects("vyre-libs::test::closure::caller_sig_closed")
            .unwrap(),
        closed_eff
    );
    let caller_closed_caps = closure_closed
        .transitive_capabilities("vyre-libs::test::closure::caller_sig_closed")
        .unwrap();
    assert_eq!(
        caller_closed_caps,
        closure_closed
            .direct_capabilities
            .get("vyre-libs::test::closure::caller_sig_closed")
            .copied()
            .unwrap()
            .join(closed_caps)
    );
    assert!(!caller_closed_caps.f16);
    assert!(!caller_closed_caps.trap);
    assert!(!caller_closed_caps.subgroup_ops);
}

/// WHY (182.9.5): Recursive cycles fail closed to strongest effects and capabilities
/// unless an explicit contract closes them.
#[test]
fn recursive_cycles_fail_closed_without_contract() {
    // Cycle A -> B -> A
    let reg_cycle_a = OperationRegistration::library(
        "vyre-libs::test::closure::cycle_a",
        || make_caller_program("vyre-libs::test::closure::cycle_b"),
        None,
        None,
    );
    let reg_cycle_b = OperationRegistration::library(
        "vyre-libs::test::closure::cycle_b",
        || make_caller_program("vyre-libs::test::closure::cycle_a"),
        None,
        None,
    );

    let closure = CallGraphClosure::solve_from_registrations([&reg_cycle_a, &reg_cycle_b]);

    assert!(closure.is_unclosed_or_cyclic("vyre-libs::test::closure::cycle_a"));
    assert!(closure.is_unclosed_or_cyclic("vyre-libs::test::closure::cycle_b"));
    assert_eq!(
        closure
            .transitive_effects("vyre-libs::test::closure::cycle_a")
            .unwrap(),
        OperationEffects::ALL
    );
    assert_eq!(
        closure
            .transitive_effects("vyre-libs::test::closure::cycle_b")
            .unwrap(),
        OperationEffects::ALL
    );
}

/// WHY (182.9.5): A mutation that adds a write, atomic, barrier, trap, or capability
/// only in a nested callee alters the parent record's composite version deterministically.
#[test]
fn nested_callee_mutation_alters_parent_composite_version_and_effects() {
    // Baseline graph: Parent -> Child (pure)
    let reg_child_v1 = OperationRegistration::library(
        "vyre-libs::test::closure::child",
        make_pure_leaf_program,
        None,
        None,
    );
    let reg_parent = OperationRegistration::library(
        "vyre-libs::test::closure::parent",
        || make_caller_program("vyre-libs::test::closure::child"),
        None,
        None,
    );

    let closure_v1 = CallGraphClosure::solve_from_registrations([&reg_parent, &reg_child_v1]);
    let parent_ver_v1 = closure_v1
        .composite_version("vyre-libs::test::closure::parent", 1)
        .unwrap();
    let parent_eff_v1 = closure_v1
        .transitive_effects("vyre-libs::test::closure::parent")
        .unwrap();
    assert!(!parent_eff_v1.writes, "V1 parent has no writes");
    assert!(!parent_eff_v1.atomics, "V1 parent has no atomics");

    // Mutated graph: Child now adds writes and atomics, parent code untouched
    let reg_child_v2 = OperationRegistration::library(
        "vyre-libs::test::closure::child",
        make_atomic_write_leaf_program,
        None,
        None,
    );

    let closure_v2 = CallGraphClosure::solve_from_registrations([&reg_parent, &reg_child_v2]);
    let parent_ver_v2 = closure_v2
        .composite_version("vyre-libs::test::closure::parent", 1)
        .unwrap();
    let parent_eff_v2 = closure_v2
        .transitive_effects("vyre-libs::test::closure::parent")
        .unwrap();
    assert!(parent_eff_v2.writes, "V2 parent now transitively writes");
    assert!(
        parent_eff_v2.atomics,
        "V2 parent now transitively has atomics"
    );

    assert_ne!(
        parent_ver_v1, parent_ver_v2,
        "Nested mutation in callee must alter the parent's composite semantic version"
    );
    assert_ne!(
        closure_v1.closure_identity(),
        closure_v2.closure_identity(),
        "Global closure identity must alter when any callee effect changes"
    );
}

/// WHY (182.9.5): Global process-wide OperationRegistry integrates the call graph closure.
#[test]
fn global_operation_registry_integrates_call_graph_closure() {
    let registry = OperationRegistry::global();
    assert!(
        registry.call_graph_closure_identity() > 0,
        "Closure identity must be non-zero"
    );

    for op in registry.iter() {
        let eff = op.effects();
        assert!(eff.is_some(), "Every registered operation resolves effects");
        let caps = op.required_capabilities();
        assert!(
            caps.is_some(),
            "Every registered operation resolves required capabilities"
        );
        let comp_ver = op.composite_version();
        assert!(comp_ver > 0, "Composite version must be computed");
    }
}

/// WHY (182.9.5): Every registered intrinsic or library operation must have an explicit
/// or derived semantic contract (effects and capabilities) and cannot omit a decision.
#[test]
fn every_registered_operation_has_explicit_or_derived_semantic_contract() {
    let registry = OperationRegistry::global();
    for op in registry.iter() {
        let eff = op.effects().unwrap_or_else(|| {
            panic!("Fix: operation `{}` omitted semantic effects", op.id);
        });
        let caps = op.required_capabilities().unwrap_or_else(|| {
            panic!("Fix: operation `{}` omitted semantic capabilities", op.id);
        });

        if op.id.contains("subgroup") {
            assert!(
                caps.subgroup_ops,
                "Fix: subgroup intrinsic `{}` must declare subgroup_ops capability",
                op.id
            );
        }
        if op.id.contains("barrier") {
            assert!(
                eff.synchronizes,
                "Fix: barrier intrinsic `{}` must declare synchronizes effect",
                op.id
            );
        }
    }
}
