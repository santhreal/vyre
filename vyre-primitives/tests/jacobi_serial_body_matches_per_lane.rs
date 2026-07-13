//! `jacobi_smooth_step_serial_body` must be BYTE-IDENTICAL to the per-lane `jacobi_smooth_step`.
//!
//! WHY: `amg_v_cycle` composes several Jacobi smoothing steps under a single `InvocationId == 0`
//! lane guard, so it inlines the SERIAL body form instead of the per-lane builder (which writes
//! `x_out[InvocationId]` and, when dispatched with fewer lanes than rows, as the production AMG
//! dispatch does, silently smooths only the first few rows). This test proves the serial form
//! computes exactly what the per-lane form computes when the per-lane form IS given enough lanes,
//! so the serialization preserves the smoother's arithmetic (the essence of the amg fix). The bug
//! it guards against, a serial rewrite that drifts from the real per-lane arithmetic, would
//! otherwise be invisible, because the reference interpreter infers a grid of n² ≥ n from the
//! dense matrix and therefore never reproduces amg's grid=1 under-coverage directly.
#![cfg(feature = "all-lego")]

use std::sync::Arc;

use vyre_foundation::ir::model::expr::Ident;
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Node, Program};
use vyre_primitives::math::multigrid::{jacobi_smooth_step, jacobi_smooth_step_serial_body};
use vyre_primitives::wire::{decode_u32_le_bytes_all as unpack, pack_u32_slice as pack};
use vyre_reference::value::Value;

/// Wrap the serial smoothing body in a runnable single-Region program with the same buffer layout
/// as the per-lane `jacobi_smooth_step`.
fn serial_program(n: u32) -> Program {
    let matrix_cells = n * n;
    Program::wrapped(
        vec![
            BufferDecl::storage("a", 0, BufferAccess::ReadOnly, DataType::U32)
                .with_count(matrix_cells),
            BufferDecl::storage("b", 1, BufferAccess::ReadOnly, DataType::U32).with_count(n),
            BufferDecl::storage("x_in", 2, BufferAccess::ReadOnly, DataType::U32).with_count(n),
            BufferDecl::storage("omega", 3, BufferAccess::ReadOnly, DataType::U32).with_count(1),
            BufferDecl::storage("x_out", 4, BufferAccess::ReadWrite, DataType::U32).with_count(n),
        ],
        [1, 1, 1],
        vec![Node::Region {
            generator: Ident::from("test::jacobi_serial"),
            source_region: None,
            body: Arc::new(jacobi_smooth_step_serial_body(
                "a", "b", "x_in", "omega", "x_out", n, "t",
            )),
        }],
    )
}

fn run(program: &Program, n: u32, a: &[u32], b: &[u32], x_in: &[u32], omega: u32) -> Vec<u32> {
    let outputs = vyre_reference::reference_eval(
        program,
        &[
            Value::from(pack(a)),
            Value::from(pack(b)),
            Value::from(pack(x_in)),
            Value::from(pack(&[omega])),
            Value::from(pack(&vec![0u32; n as usize])),
        ],
    )
    .expect("jacobi reference evaluation must succeed");
    let idx = vyre_reference::output_index(program, "x_out").expect("x_out output");
    unpack(&outputs[idx].to_bytes())[..n as usize].to_vec()
}

#[test]
fn serial_body_matches_per_lane_builder_over_generated_systems() {
    let mut state = 0x2468_ACE0u32;
    let mut next = |s: &mut u32| {
        *s ^= *s << 13;
        *s ^= *s >> 17;
        *s ^= *s << 5;
        *s
    };
    let one: u32 = 1 << 16;
    let mut changed_cases = 0u32;
    for case in 0..200u32 {
        let n = 2 + (next(&mut state) % 5); // 2..=6
        let matrix_cells = (n * n) as usize;
        // Diagonally-dominant fixed-point matrix so the smoother is well-posed: diag in [3,6],
        // off-diagonals small.
        let a: Vec<u32> = (0..matrix_cells)
            .map(|c| {
                let row = c as u32 / n;
                let col = c as u32 % n;
                if row == col {
                    (3 + next(&mut state) % 4) * one // 3.0..=6.0
                } else {
                    (next(&mut state) % (one / 4)) // 0.0..0.25
                }
            })
            .collect();
        let b: Vec<u32> = (0..n).map(|_| next(&mut state) % (4 * one)).collect();
        let x_in: Vec<u32> = (0..n).map(|_| next(&mut state) % (2 * one)).collect();
        let omega = one / 2 + next(&mut state) % (one / 2); // 0.5..1.0

        let serial = run(&serial_program(n), n, &a, &b, &x_in, omega);
        let per_lane = run(
            &jacobi_smooth_step("a", "b", "x_in", "omega", "x_out", n),
            n,
            &a,
            &b,
            &x_in,
            omega,
        );
        if serial != x_in {
            changed_cases += 1;
        }
        assert_eq!(
            serial, per_lane,
            "case {case} (n={n}): serial jacobi body {serial:?} != per-lane builder {per_lane:?} \
The serialized smoother must reproduce the per-lane arithmetic exactly"
        );
    }
    // The per-lane builder covers every row here (reference grid = n² ≥ n), so this asserts the
    // serial form AGREES with it (and that the smoother actually moves x (not a vacuous no-op)).
    assert!(
        changed_cases > 150,
        "only {changed_cases}/200 cases changed x, the smoother is not being exercised"
    );
}

#[test]
fn serial_body_updates_every_row_not_just_the_first() {
    // The precise bug in amg was that a per-lane smoother dispatched with 1 lane updated only row 0.
    // The serial body must update EVERY row. Seed x_in far from the solution so every row moves.
    let one: u32 = 1 << 16;
    let n = 5u32;
    let a: Vec<u32> = (0..(n * n))
        .map(|c| if c / n == c % n { 4 * one } else { one / 8 })
        .collect();
    let b: Vec<u32> = (1..=n).map(|k| k * one).collect(); // 1.0,2.0,3.0,4.0,5.0
    let x_in = vec![0u32; n as usize];
    let omega = one; // 1.0
    let out = run(&serial_program(n), n, &a, &b, &x_in, omega);
    for (row, &v) in out.iter().enumerate() {
        assert_ne!(
            v, 0,
            "row {row} was left un-smoothed (still 0), the serial body must update every row, not \
             only row 0 (the per-lane-at-1-lane bug)"
        );
    }
}
