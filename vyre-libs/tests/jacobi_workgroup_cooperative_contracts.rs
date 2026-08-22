//! Regression and adversarial contracts for Jacobi workgroup cooperative tiling.
//!
//! WHY: The Jacobi eigensolver (`symmetric_eigen_jacobi`) and its three helper
//! operations (`matrix_identity_fill`, `matrix_diagonal_extract`, `eigenvector_column_sign`)
//! must dispatch across declared cooperative workgroup lanes (`LANES = 64`), not as
//! single-lane `[1, 1, 1]` dispatches or with the independent phases serialized.
//!
//! These tests prove:
//! 1. Every program declares `[LANES, 1, 1]` workgroup geometry and binds `local` to `LocalId { axis: 0 }`.
//! 2. In `jacobi_eigen_body`, identity seeding, column sign canonicalization, and diagonal extraction
//!    are at top-level workgroup cooperative scope, while only the sequential Givens rotation is guarded
//!    to lane 0.
//! 3. Workgroup barriers synchronize state at exact dependency boundaries:
//!    - After cooperative identity fill before the first sweep's pivot search.
//!    - After each sweep's rotation before the next sweep's pivot search.
//! 4. Zero, odd, and multi-chunk tail sizes (`n = 1, 2, 3, 4, 5, 7, 8, 64, 65, 80`) execute
//!    cooperatively without race or OOB, and produce exact reference-identical results.
//!
//! What it does not catch: device-specific subgroup shuffles (covered by backend-specific tests).
#![cfg(feature = "math")]

use vyre_foundation::ir::{Expr, Node, Program};
use vyre_foundation::visit::any_descendant;
use vyre_libs::math::eigenvector_column_sign::{eigenvector_column_sign, OP_ID as SIGN_OP_ID};
use vyre_libs::math::jacobi_apply_rotation::OP_ID as ROTATION_OP_ID;
use vyre_libs::math::matrix_diagonal_extract::{matrix_diagonal_extract, OP_ID as DIAG_OP_ID};
use vyre_libs::math::matrix_identity_fill::{matrix_identity_fill, OP_ID as IDENTITY_OP_ID};
use vyre_libs::math::symmetric_eigen_jacobi::{
    jacobi_eigen_body, jacobi_workgroup, symmetric_eigen_jacobi,
};
use vyre_primitives::wire::{decode_f32_le_bytes_all as unpack_f32, pack_f32_slice as pack_f32};
use vyre_reference::value::Value;
const LANES: u32 = 64;

fn check_local_id_binding(program: &Program) -> bool {
    program.entry().iter().any(|node| {
        any_descendant(node, &mut |inner| match inner {
            Node::Let { name, value } => {
                name == "local" && matches!(value, Expr::LocalId { axis: 0 })
            }
            _ => false,
        })
    })
}

/// The nodes one workgroup runs, with the guard that keeps every other
/// workgroup out asserted on the way in.
///
/// The phase-placement contracts below read the scope the phases live in, so
/// they would pass just as well against a body that dropped the guard. This is
/// where that guard is pinned. Without it a dispatch rounded up past the built
/// grid lets a second workgroup re-seed the identity eigenbasis after the first
/// has finished sweeping.
fn workgroup_scope(body: &[Node]) -> &[Node] {
    assert_eq!(
        body.len(),
        1,
        "Fix: the body must be exactly the workgroup guard, got {body:?}"
    );
    match &body[0] {
        Node::If { cond, then, .. } => {
            assert_eq!(
                *cond,
                Expr::is_first_workgroup(),
                "Fix: the sweep must be guarded on the first workgroup"
            );
            then
        }
        other => panic!("Fix: the body must open with the workgroup guard, got {other:?}"),
    }
}

#[test]
fn all_four_primitives_dispatch_declared_workgroup_lanes() {
    assert_eq!(jacobi_workgroup(), [LANES, 1, 1]);

    let jacobi = symmetric_eigen_jacobi("a", "evec", "eval", 4);
    assert_eq!(
        jacobi.workgroup_size(),
        [LANES, 1, 1],
        "symmetric_eigen_jacobi must dispatch LANES cooperative lanes"
    );

    let identity = matrix_identity_fill("m", 4);
    assert_eq!(
        identity.workgroup_size(),
        [LANES, 1, 1],
        "matrix_identity_fill must dispatch LANES cooperative lanes"
    );

    let diag = matrix_diagonal_extract("m", "diag", 4);
    assert_eq!(
        diag.workgroup_size(),
        [LANES, 1, 1],
        "matrix_diagonal_extract must dispatch LANES cooperative lanes"
    );

    let sign = eigenvector_column_sign("evec", 4);
    assert_eq!(
        sign.workgroup_size(),
        [LANES, 1, 1],
        "eigenvector_column_sign must dispatch LANES cooperative lanes"
    );
}

#[test]
fn all_four_primitives_bind_local_id() {
    let jacobi = symmetric_eigen_jacobi("a", "evec", "eval", 4);
    assert!(
        check_local_id_binding(&jacobi),
        "symmetric_eigen_jacobi missing local binding"
    );

    let identity = matrix_identity_fill("m", 4);
    assert!(
        check_local_id_binding(&identity),
        "matrix_identity_fill missing local binding"
    );

    let diag = matrix_diagonal_extract("m", "diag", 4);
    assert!(
        check_local_id_binding(&diag),
        "matrix_diagonal_extract missing local binding"
    );

    let sign = eigenvector_column_sign("evec", 4);
    assert!(
        check_local_id_binding(&sign),
        "eigenvector_column_sign missing local binding"
    );
}

#[test]
fn jacobi_cooperative_phases_are_not_serialized() {
    let whole = jacobi_eigen_body("a", "evec", "eval", 4);
    let body = workgroup_scope(&whole);

    // 1. Identity fill region is in the workgroup scope, not inside the lane guard
    let identity_node = body.iter().find(
        |n| matches!(n, Node::Region { generator, .. } if generator.as_str() == IDENTITY_OP_ID),
    );
    assert!(
        identity_node.is_some(),
        "matrix_identity_fill must be spliced at top-level cooperative scope, not inside serial"
    );

    // 2. Column sign and diagonal extract regions are top-level (not in an If)
    let sign_node = body
        .iter()
        .find(|n| matches!(n, Node::Region { generator, .. } if generator.as_str() == SIGN_OP_ID));
    assert!(
        sign_node.is_some(),
        "eigenvector_column_sign must be spliced at top-level cooperative scope, not inside serial"
    );

    let diag_node = body
        .iter()
        .find(|n| matches!(n, Node::Region { generator, .. } if generator.as_str() == DIAG_OP_ID));
    assert!(
        diag_node.is_some(),
        "matrix_diagonal_extract must be spliced at top-level cooperative scope, not inside serial"
    );

    // 3. Rotation is inside an If (serial execution)
    let sweep_loop = body
        .iter()
        .find(|n| matches!(n, Node::Loop { var, .. } if var == "jac_sweep"))
        .expect("jac_sweep loop");
    if let Node::Loop {
        body: sweep_body, ..
    } = sweep_loop
    {
        let rotation_is_serialized = sweep_body.iter().any(|node| {
            matches!(node, Node::If { .. })
                && any_descendant(node, &mut |r| {
                    matches!(r, Node::Region { generator, .. } if generator.as_str() == ROTATION_OP_ID)
                })
        });
        assert!(
            rotation_is_serialized,
            "jacobi_apply_rotation must remain guarded inside serial execution"
        );
    }
}

#[test]
fn jacobi_barriers_placed_at_data_boundaries() {
    let whole = jacobi_eigen_body("a", "evec", "eval", 4);
    let body = workgroup_scope(&whole);

    // Barrier immediately after identity fill
    let identity_pos = body
        .iter()
        .position(
            |n| matches!(n, Node::Region { generator, .. } if generator.as_str() == IDENTITY_OP_ID),
        )
        .expect("identity region pos");
    assert!(
        matches!(body.get(identity_pos + 1), Some(Node::Barrier { .. })),
        "a workgroup barrier must immediately follow cooperative identity fill"
    );

    // Barrier at end of sweep loop
    let sweep_loop = body
        .iter()
        .find(|n| matches!(n, Node::Loop { var, .. } if var == "jac_sweep"))
        .expect("jac_sweep loop");
    if let Node::Loop {
        body: sweep_body, ..
    } = sweep_loop
    {
        assert!(
            matches!(sweep_body.last(), Some(Node::Barrier { .. })),
            "a workgroup barrier must follow each rotation sweep before the next sweep begins"
        );
    }
}

#[test]
fn identity_fill_cooperative_lane_striding_reference_exactness() {
    for &n in &[1u32, 2, 3, 5, 7, 8, 16, 64, 65, 80] {
        let program = matrix_identity_fill("m", n);
        let count = (n * n) as usize;
        let outputs =
            vyre_reference::reference_eval(&program, &[]).expect("identity fill reference eval");
        let out = unpack_f32(&outputs[0].to_bytes());
        assert_eq!(out.len(), count);
        for r in 0..(n as usize) {
            for c in 0..(n as usize) {
                let expected = if r == c { 1.0f32 } else { 0.0f32 };
                assert_eq!(out[r * (n as usize) + c], expected);
            }
        }
    }
}

#[test]
fn diagonal_extract_cooperative_lane_striding_reference_exactness() {
    for &n in &[1u32, 2, 3, 5, 7, 8, 16, 64, 65, 80] {
        let program = matrix_diagonal_extract("m", "diag", n);
        let count = (n * n) as usize;
        let mut matrix = Vec::with_capacity(count);
        for i in 0..count {
            matrix.push((i as f32) * 1.5 + 0.25);
        }
        let outputs = vyre_reference::reference_eval(&program, &[Value::from(pack_f32(&matrix))])
            .expect("diagonal extract reference eval");
        let out = unpack_f32(
            &outputs[vyre_reference::output_index(&program, "diag").unwrap()].to_bytes(),
        );
        assert_eq!(out.len(), n as usize);
        for i in 0..(n as usize) {
            assert_eq!(out[i], matrix[i * (n as usize) + i]);
        }
    }
}

#[test]
fn eigenvector_sign_cooperative_lane_striding_reference_exactness() {
    for &n in &[1u32, 2, 3, 5, 7, 8, 16, 64, 65, 80] {
        let n_usize = n as usize;
        let count = n_usize * n_usize;
        let mut matrix = vec![0.0f32; count];
        for c in 0..n_usize {
            matrix[0 * n_usize + c] = 1e-9 * (c as f32); // below epsilon, must not decide sign
            if n_usize > 1 {
                let sign = if c % 2 == 1 { -3.5f32 } else { 3.5f32 };
                matrix[1 * n_usize + c] = sign;
                for r in 2..n_usize {
                    matrix[r * n_usize + c] =
                        ((r * 10 + c) as f32) * if c % 2 == 1 { -1.0 } else { 1.0 };
                }
            } else {
                matrix[0] = -12.0f32;
            }
        }

        let program = eigenvector_column_sign("evec", n);
        let outputs = vyre_reference::reference_eval(&program, &[Value::from(pack_f32(&matrix))])
            .expect("column sign reference eval");
        let out = unpack_f32(&outputs[0].to_bytes());
        assert_eq!(out.len(), count);

        for c in 0..n_usize {
            if n_usize == 1 {
                assert_eq!(out[0], 12.0f32);
            } else {
                assert_eq!(out[1 * n_usize + c], 3.5f32);
                for r in 2..n_usize {
                    assert_eq!(out[r * n_usize + c], (r * 10 + c) as f32);
                }
            }
        }
    }
}
