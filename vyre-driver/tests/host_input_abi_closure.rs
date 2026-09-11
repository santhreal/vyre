//! One answer decides which declarations a caller fills.
//!
//! `BufferDecl::consumes_host_input` is the single definition of that answer,
//! and `BindingPlan` is what every backend reads it through. They were separate
//! predicates once, which is how a backend came to accept a zeroed placeholder
//! for a buffer it allocates itself. This sweep walks the whole declaration
//! space and fails when the two disagree anywhere in it.
//!
//! The space is derived from `MemoryKind::ALL` and `BufferAccess::ALL`, both
//! fixed-length arrays in the crates that define those enums, so a tier or an
//! access mode added to either one fails to compile until it is listed and then
//! turns this suite red until a decision is recorded for it. A hardcoded list of
//! variants here would go stale in silence, which is the same failure as having
//! no test.

use vyre_driver::{BindingPlan, BindingRole};
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, MemoryKind, Program};

/// Every declaration the two enums can describe, at both settings of the two
/// flags that suppress host input.
fn declaration_space() -> Vec<BufferDecl> {
    let mut declarations = Vec::new();
    for kind in MemoryKind::ALL {
        for access in BufferAccess::ALL {
            for is_output in [false, true] {
                for pipeline_live_out in [false, true] {
                    // `BufferAccess` is Clone and not Copy, so the outer loop's
                    // binding cannot be moved once per inner iteration.
                    let mut decl = BufferDecl::storage("subject", 0, access.clone(), DataType::F32)
                        .with_count(4)
                        .with_kind(kind)
                        .with_pipeline_live_out(pipeline_live_out);
                    decl.is_output = is_output;
                    declarations.push(decl);
                }
            }
        }
    }
    declarations
}

fn label(decl: &BufferDecl) -> String {
    format!(
        "kind={:?} access={:?} is_output={} pipeline_live_out={}",
        decl.kind, decl.access, decl.is_output, decl.pipeline_live_out
    )
}

/// WHY: a caller passes one buffer per declaration that consumes host input, in
/// declaration order. If the plan admits a slot the predicate rejects, the
/// caller is asked to fill memory the dispatch allocates; if the plan drops a
/// slot the predicate admits, a declared input is silently zero on the device.
/// Both were reachable before the predicate became the only answer.
#[test]
fn the_plan_takes_an_input_slot_for_exactly_the_declarations_that_consume_one() {
    let space = declaration_space();
    assert_eq!(
        space.len(),
        MemoryKind::ALL.len() * BufferAccess::ALL.len() * 4,
        "Fix: the declaration sweep must cover every memory tier and access mode at both flag settings."
    );

    for decl in space {
        let expected = decl.consumes_host_input();
        let program = Program::wrapped(vec![decl.clone()], [4, 1, 1], Vec::new());
        let plan = BindingPlan::build(&program).unwrap_or_else(|error| {
            panic!(
                "Fix: a single-buffer program must plan for {}: {error}",
                label(&decl)
            )
        });

        assert_eq!(
            plan.input_indices.contains(&0),
            expected,
            "Fix: BindingPlan and BufferDecl::consumes_host_input disagree for {}. Change the predicate, not the plan: it is the single definition of which declarations a caller fills.",
            label(&decl)
        );

        let role = plan.bindings[0].role;
        assert_eq!(
            matches!(
                role,
                BindingRole::Input | BindingRole::InputOutput | BindingRole::Uniform
            ),
            expected,
            "Fix: role {role:?} does not carry a host input for {}, but the predicate says it does, or the reverse.",
            label(&decl)
        );
        assert_eq!(
            plan.bindings[0].input_index.is_some(),
            expected,
            "Fix: the input index on a binding must be present for exactly the declarations that consume host input, for {}.",
            label(&decl)
        );
    }
}

/// WHY: a validated input count is what rejects a caller passing a placeholder
/// for a backend-allocated output. The count comes from the plan, so it has to
/// equal the number of declarations the predicate admits for a multi-buffer
/// program, not only for the single-buffer case above.
#[test]
fn the_validated_input_count_equals_the_declarations_that_consume_input() {
    let buffers = vec![
        BufferDecl::storage("read", 0, BufferAccess::ReadOnly, DataType::F32).with_count(4),
        BufferDecl::storage("scratch", 1, BufferAccess::ReadWrite, DataType::F32).with_count(4),
        BufferDecl::workgroup("shared", 4, DataType::F32),
        BufferDecl::output("written", 2, DataType::F32).with_count(4),
    ];
    let expected = buffers
        .iter()
        .filter(|buffer| buffer.consumes_host_input())
        .count();
    let program = Program::wrapped(buffers, [4, 1, 1], Vec::new());
    let plan = BindingPlan::build(&program)
        .expect("Fix: the mixed-role fixture must build a binding plan.");

    assert_eq!(plan.input_indices.len(), expected);

    let sized = vec![vec![0_u8; 16]; expected];
    plan.validate_inputs(&sized)
        .expect("Fix: an input list one entry per consuming declaration must validate.");

    let mut over = sized.clone();
    over.push(vec![0_u8; 16]);
    let over_error = plan
        .validate_inputs(&over)
        .expect_err("Fix: an extra input entry must be rejected, not ignored.");
    assert!(
        over_error.to_string().contains(&format!("{expected}")),
        "Fix: the rejection must state the expected input count, got: {over_error}"
    );

    let under = sized[..expected - 1].to_vec();
    let under_error = plan
        .validate_inputs(&under)
        .expect_err("Fix: a missing input entry must be rejected, not zero-filled.");
    assert!(
        under_error.to_string().contains(&format!("{expected}")),
        "Fix: the rejection must state the expected input count, got: {under_error}"
    );
}

/// The refusal wording every caller of a wrong-length input list reads.
fn both_counts(expected: usize, received: usize) -> String {
    format!("expected {expected} input buffer(s) from Program declarations but received {received}")
}

/// WHY: `BindingPlan` takes an input list through seven public entry points,
/// four that build a plan and three that validate against one. All seven route
/// to one length check, and a caller that reached the wrong one used to get a
/// refusal that named neither count, so a concrete driver restated the
/// assertion for whichever entry point it happened to call. The wording is one
/// contract, so it is pinned once here against every entry point that can
/// produce it.
///
/// What this does not judge: byte lengths per slot. A list of the right length
/// whose entries are too small is a separate refusal.
#[test]
fn every_input_taking_entry_point_refuses_a_wrong_count_naming_both_counts() {
    let buffers = vec![
        BufferDecl::storage("read", 0, BufferAccess::ReadOnly, DataType::F32).with_count(4),
        BufferDecl::storage("scratch", 1, BufferAccess::ReadWrite, DataType::F32).with_count(4),
        BufferDecl::workgroup("shared", 4, DataType::F32),
        BufferDecl::output("written", 2, DataType::F32).with_count(4),
    ];
    let expected = buffers
        .iter()
        .filter(|buffer| buffer.consumes_host_input())
        .count();
    assert!(
        expected > 0,
        "Fix: the fixture must consume at least one host input, or the short case is unreachable."
    );
    let program = Program::wrapped(buffers, [4, 1, 1], Vec::new());
    let plan = BindingPlan::build(&program)
        .expect("Fix: the mixed-role fixture must build a binding plan.");

    let exact = vec![vec![0_u8; 16]; expected];
    let mut over = exact.clone();
    over.push(vec![0_u8; 16]);
    let under = exact[..expected - 1].to_vec();

    for wrong in [&over, &under] {
        let received = wrong.len();
        let borrowed: Vec<&[u8]> = wrong.iter().map(Vec::as_slice).collect();
        let lengths: Vec<usize> = wrong.iter().map(Vec::len).collect();
        let refusals = [
            (
                "BindingPlan::from_program",
                BindingPlan::from_program(&program, wrong).err(),
            ),
            (
                "BindingPlan::from_borrowed_inputs",
                BindingPlan::from_borrowed_inputs(&program, &borrowed).err(),
            ),
            (
                "BindingPlan::from_input_lengths",
                BindingPlan::from_input_lengths(&program, &lengths).err(),
            ),
            (
                "BindingPlan::validate_inputs",
                plan.validate_inputs(wrong).err(),
            ),
            (
                "BindingPlan::validate_borrowed_inputs",
                plan.validate_borrowed_inputs(&borrowed).err(),
            ),
            (
                "BindingPlan::validate_input_byte_lengths",
                plan.validate_input_byte_lengths(&lengths).err(),
            ),
        ];
        for (entry_point, refusal) in refusals {
            let refusal = refusal.unwrap_or_else(|| {
                panic!(
                    "Fix: {entry_point} accepted {received} inputs where {expected} are declared."
                )
            });
            let text = refusal.to_string();
            assert!(
                text.contains(&both_counts(expected, received)),
                "Fix: {entry_point} must state expected {expected} and received {received}, got: {text}"
            );
        }
    }

    plan.validate_inputs(&exact)
        .expect("Fix: an input list one entry per consuming declaration must validate.");
}
