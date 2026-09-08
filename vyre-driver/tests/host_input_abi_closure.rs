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
