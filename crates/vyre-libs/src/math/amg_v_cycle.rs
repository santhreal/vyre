//! Algebraic Multigrid (AMG) V-cycle primitive (#P-PRIM-3).
//!
//! Composes the Jacobi smoother (`jacobi_smooth_step_serial_body`) with restriction and
//! prolongation to solve linear systems $Ax = b$ across multiple scales. The whole V-cycle is a
//! single-threaded serial algorithm run under one `LogicalIndex == 0` point guard (see the
//! region body), so it inlines the SERIAL smoother form, not the per-point `jacobi_smooth_step` builder.
//!
//! Sequence (2-level):
//! 1. Pre-smooth: $x = \text{smooth}(A, b, x, \omega)$
//! 2. Restrict: $r = b - Ax$; $b_c = R r$
//! 3. Coarse solve: $x_c = \text{solve}(A_c, b_c)$ (via Jacobi for this primitive)
//! 4. Prolong: $x = x + P x_c$
//! 5. Post-smooth: $x = \text{smooth}(A, b, x, \omega)$

use crate::math::multigrid::jacobi_smooth_step_serial_body;
use std::sync::Arc;
use vyre_foundation::composition::{trap_program, wrap_anonymous_region};
use vyre_foundation::ir::Ident;
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};

/// Op id.
pub const OP_ID: &str = "vyre-libs::math::amg_v_cycle";
const V_CYCLE_PHASE_OP_ID: &str = "vyre-libs::math::amg_v_cycle::v_cycle_phase";

/// Build an AMG V-cycle Program.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn amg_v_cycle(
    a: &str,
    b: &str,
    x: &str,
    r_mat: &str,
    p_mat: &str,
    a_c: &str,
    omega: &str,
    scratch_fine: &str,
    scratch_coarse_b: &str,
    scratch_coarse_x: &str,
    n_fine: u32,
    n_coarse: u32,
) -> Program {
    let trap = |msg: String| trap_program(OP_ID, Some((x, DataType::U32)), msg);
    if n_fine == 0 {
        return trap("Fix: amg_v_cycle requires n_fine > 0, got 0.".into());
    }
    if n_coarse == 0 {
        return trap("Fix: amg_v_cycle requires n_coarse > 0, got 0.".into());
    }
    if n_coarse >= n_fine {
        return trap(format!(
            "Fix: amg_v_cycle requires n_coarse < n_fine, got n_coarse={n_coarse}, n_fine={n_fine}."
        ));
    }
    let Some(fine_cells) = n_fine.checked_mul(n_fine) else {
        return trap(format!(
            "Fix: amg_v_cycle fine matrix cells overflow u32: n_fine={n_fine}."
        ));
    };
    let Some(transfer_cells) = n_fine.checked_mul(n_coarse) else {
        return trap(format!(
            "Fix: amg_v_cycle transfer matrix cells overflow u32: n_fine={n_fine}, n_coarse={n_coarse}."
        ));
    };
    let Some(coarse_cells) = n_coarse.checked_mul(n_coarse) else {
        return trap(format!(
            "Fix: amg_v_cycle coarse matrix cells overflow u32: n_coarse={n_coarse}."
        ));
    };

    let mut nodes = Vec::new();
    let smooth_into = |nodes: &mut Vec<Node>,
                       mat: &str,
                       rhs: &str,
                       sol: &str,
                       tmp: &str,
                       n: u32,
                       tag: &str,
                       copy_var: &str| {
        nodes.extend(jacobi_smooth_step_serial_body(
            mat, rhs, sol, omega, tmp, n, tag,
        ));
        nodes.push(Node::loop_for(
            copy_var,
            Expr::u32(0),
            Expr::u32(n),
            vec![Node::store(
                sol,
                Expr::var(copy_var),
                Expr::load(tmp, Expr::var(copy_var)),
            )],
        ));
    };
    let matvec_loop = |row_var: &str,
                       rows: u32,
                       col_var: &str,
                       cols: u32,
                       acc_var: &str,
                       mat: &str,
                       vec_buf: &str,
                       out_buf: &str,
                       finish: fn(Expr, Expr) -> Expr| {
        Node::loop_for(
            row_var,
            Expr::u32(0),
            Expr::u32(rows),
            vec![
                Node::let_bind(acc_var, Expr::u32(0)),
                Node::loop_for(
                    col_var,
                    Expr::u32(0),
                    Expr::u32(cols),
                    vec![Node::assign(
                        acc_var,
                        Expr::add(
                            Expr::var(acc_var),
                            crate::math::fixed::fixed_mul_16_16_expr(
                                Expr::load(
                                    mat,
                                    Expr::add(
                                        Expr::mul(Expr::var(row_var), Expr::u32(cols)),
                                        Expr::var(col_var),
                                    ),
                                ),
                                Expr::load(vec_buf, Expr::var(col_var)),
                            ),
                        ),
                    )],
                ),
                Node::store(
                    out_buf,
                    Expr::var(row_var),
                    finish(Expr::var(row_var), Expr::var(acc_var)),
                ),
            ],
        )
    };

    smooth_into(&mut nodes, a, b, x, scratch_fine, n_fine, "pre", "__i");
    let b_name = b;
    nodes.push(Node::loop_for(
        "i",
        Expr::u32(0),
        Expr::u32(n_fine),
        vec![
            Node::let_bind("ax_i", Expr::u32(0)),
            Node::loop_for(
                "j",
                Expr::u32(0),
                Expr::u32(n_fine),
                vec![Node::assign(
                    "ax_i",
                    Expr::add(
                        Expr::var("ax_i"),
                        crate::math::fixed::fixed_mul_16_16_expr(
                            Expr::load(
                                a,
                                Expr::add(
                                    Expr::mul(Expr::var("i"), Expr::u32(n_fine)),
                                    Expr::var("j"),
                                ),
                            ),
                            Expr::load(x, Expr::var("j")),
                        ),
                    ),
                )],
            ),
            Node::store(
                scratch_fine,
                Expr::var("i"),
                Expr::sub(Expr::load(b_name, Expr::var("i")), Expr::var("ax_i")),
            ),
        ],
    ));
    nodes.push(matvec_loop(
        "ic",
        n_coarse,
        "jf",
        n_fine,
        "bc_i",
        r_mat,
        scratch_fine,
        scratch_coarse_b,
        |_, acc| acc,
    ));
    nodes.push(Node::store(scratch_coarse_x, Expr::u32(0), Expr::u32(0)));
    for k in 0..4 {
        smooth_into(
            &mut nodes,
            a_c,
            scratch_coarse_b,
            scratch_coarse_x,
            "temp_coarse",
            n_coarse,
            &format!("coarse{k}"),
            "__k",
        );
    }
    let x_name = x;
    nodes.push(Node::loop_for(
        "if",
        Expr::u32(0),
        Expr::u32(n_fine),
        vec![
            Node::let_bind("px_i", Expr::u32(0)),
            Node::loop_for(
                "jc",
                Expr::u32(0),
                Expr::u32(n_coarse),
                vec![Node::assign(
                    "px_i",
                    Expr::add(
                        Expr::var("px_i"),
                        crate::math::fixed::fixed_mul_16_16_expr(
                            Expr::load(
                                p_mat,
                                Expr::add(
                                    Expr::mul(Expr::var("if"), Expr::u32(n_coarse)),
                                    Expr::var("jc"),
                                ),
                            ),
                            Expr::load(scratch_coarse_x, Expr::var("jc")),
                        ),
                    ),
                )],
            ),
            Node::store(
                x_name,
                Expr::var("if"),
                Expr::add(Expr::load(x_name, Expr::var("if")), Expr::var("px_i")),
            ),
        ],
    ));
    smooth_into(&mut nodes, a, b, x, scratch_fine, n_fine, "post", "__m");

    let ro = |name: &str, slot: u32, count: u32| {
        BufferDecl::storage(name, slot, BufferAccess::ReadOnly, DataType::U32).with_count(count)
    };
    let rw = |name: &str, slot: u32, count: u32| {
        BufferDecl::storage(name, slot, BufferAccess::ReadWrite, DataType::U32).with_count(count)
    };
    Program::wrapped(
        vec![
            ro(a, 0, fine_cells),
            ro(b, 1, n_fine),
            rw(x, 2, n_fine),
            ro(r_mat, 3, transfer_cells),
            ro(p_mat, 4, transfer_cells),
            ro(a_c, 5, coarse_cells),
            ro(omega, 6, 1),
            rw(scratch_fine, 7, n_fine),
            rw(scratch_coarse_b, 8, n_coarse),
            rw(scratch_coarse_x, 9, n_coarse),
            rw("temp_coarse", 10, n_coarse),
        ],
        [1, 1, 1],
        vec![wrap_anonymous_region(
            OP_ID,
            vec![Node::Region {
                generator: Ident::from(V_CYCLE_PHASE_OP_ID),
                source_region: Some(Ident::from(OP_ID)),
                // The V-cycle is a SINGLE-THREADED serial algorithm: its smoothing steps are
                // inlined as serial loops (see jacobi_smooth_step_serial_body) and its
                // restriction/prolongation phases are serial loop-nests. The reference/GPU infers
                // the dispatch grid from buffer shapes and the production consumer dispatches
                // ceil(n/256) workgroups of size 1, far fewer lanes than there are rows, so a
                // per-lane body would under-cover the vector. Guard the whole body to
                // `LogicalIndex == 0` so one scheduled point runs the entire serial V-cycle at
                // any grid (the canonical GPU serial-region idiom, cf. matroid, sheaf,
                // path_reconstruct). No GridSync barriers are needed (or safe) under this guard:
                // a single lane sees its own writes sequentially.
                body: Arc::new(vec![Node::if_then(
                    Expr::eq(Expr::LogicalIndex { axis: 0 }, Expr::u32(0)),
                    nodes,
                )]),
            }],
        )],
    )
}

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library_unconstrained(
        OP_ID,
        || amg_v_cycle("a", "b", "x", "r", "p", "ac", "om", "sf", "scb", "scx", 4, 2),
        Some(|| {
            let to_bytes = |words: &[u32]| vyre_primitives::wire::pack_u32_slice(words);
            vec![vec![
                to_bytes(&[0; 16]), // a
                to_bytes(&[0; 4]),  // b
                to_bytes(&[0; 4]),  // x
                to_bytes(&[0; 8]),  // r
                to_bytes(&[0; 8]),  // p
                to_bytes(&[0; 4]),  // ac
                to_bytes(&[0]),     // om
                to_bytes(&[0; 4]),  // sf
                to_bytes(&[0; 2]),  // scb
                to_bytes(&[0; 2]),  // scx
                to_bytes(&[0; 2]),  // temp_coarse
            ]]
        }),
        Some(|| {
            vec![vec![
                vec![0u8; 16], // x (4 u32 words)
                vec![0u8; 16], // sf (4 u32 words)
                vec![0u8; 8],  // scb (2 u32 words)
                vec![0u8; 8],  // scx (2 u32 words)
                vec![0u8; 8],  // temp_coarse (2 u32 words)
            ]]
        }),
    )
    .with_uncharacterized()
}

/// One phase adds one to each of the four fine cells.
const EXPECTED_V_CYCLE_PHASE_BYTES: [u8; 16] = [2, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 0, 5, 0, 0, 0];

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library_unconstrained(
        V_CYCLE_PHASE_OP_ID,
        || {
            Program::wrapped(
                vec![
                    BufferDecl::storage("fine_in", 0, BufferAccess::ReadOnly, DataType::U32)
                        .with_count(4),
                    BufferDecl::output("fine_out", 1, DataType::U32).with_count(4),
                ],
                [1, 1, 1],
                // The serial loop writes all four cells, so one invocation runs it.
                // The backend derives a four-workgroup grid from the output length,
                // and a fusion can widen the arm past one invocation, so the guard
                // names the invocation rather than the workgroup.
                vec![wrap_anonymous_region(
                    V_CYCLE_PHASE_OP_ID,
                    vec![Node::if_then(
                        Expr::is_first_logical_point(),
                        vec![Node::loop_for(
                            "idx",
                            Expr::u32(0),
                            Expr::u32(4),
                            vec![Node::store(
                                "fine_out",
                                Expr::var("idx"),
                                Expr::add(Expr::load("fine_in", Expr::var("idx")), Expr::u32(1)),
                            )],
                        )],
                    )],
                )],
            )
        },
        Some(|| {
            let to_bytes = |words: &[u32]| vyre_primitives::wire::pack_u32_slice(words);
            vec![vec![to_bytes(&[1, 2, 3, 4])]]
        }),
        Some(|| vec![vec![EXPECTED_V_CYCLE_PHASE_BYTES.into()]]),
    )
    .with_no_legal_rewrite()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Default, Clone)]
    struct AmgVcycleScratch {
        inner: vyre_reference::composition_witness::AmgVcycleScratchWitness,
    }

    #[allow(clippy::too_many_arguments)]
    fn try_cpu_ref_into(
        a: &[f64],
        b: &[f64],
        x: &[f64],
        r_mat: &[f64],
        p_mat: &[f64],
        a_c: &[f64],
        omega: f64,
        n_fine: u32,
        n_coarse: u32,
        scratch: &mut AmgVcycleScratch,
        out: &mut Vec<f64>,
    ) -> Result<(), String> {
        vyre_reference::composition_witness::try_amg_v_cycle_witness_with_scratch_into(
            a,
            b,
            x,
            r_mat,
            p_mat,
            a_c,
            omega,
            n_fine,
            n_coarse,
            &mut scratch.inner,
            out,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn try_cpu_ref(
        a: &[f64],
        b: &[f64],
        x: &[f64],
        r_mat: &[f64],
        p_mat: &[f64],
        a_c: &[f64],
        omega: f64,
        n_fine: u32,
        n_coarse: u32,
    ) -> Result<Vec<f64>, String> {
        let mut scratch = AmgVcycleScratch::default();
        let mut out = Vec::new();
        try_cpu_ref_into(
            a,
            b,
            x,
            r_mat,
            p_mat,
            a_c,
            omega,
            n_fine,
            n_coarse,
            &mut scratch,
            &mut out,
        )?;
        Ok(out)
    }

    #[allow(clippy::too_many_arguments)]
    fn cpu_ref(
        a: &[f64],
        b: &[f64],
        x: &[f64],
        r_mat: &[f64],
        p_mat: &[f64],
        a_c: &[f64],
        omega: f64,
        n_fine: u32,
        n_coarse: u32,
    ) -> Vec<f64> {
        vyre_reference::composition_witness::amg_v_cycle_witness(
            a, b, x, r_mat, p_mat, a_c, omega, n_fine, n_coarse,
        )
    }

    fn fixture_4x2() -> (
        [f64; 16],
        [f64; 4],
        [f64; 4],
        [f64; 8],
        [f64; 8],
        [f64; 4],
        f64,
    ) {
        (
            [
                2.0, -1.0, 0.0, 0.0, -1.0, 2.0, -1.0, 0.0, 0.0, -1.0, 2.0, -1.0, 0.0, 0.0, -1.0,
                2.0,
            ],
            [1.0, 0.0, 0.0, 1.0],
            [0.0; 4],
            [1.0, 0.5, 0.0, 0.0, 0.0, 0.5, 1.0, 0.5],
            [1.0, 0.0, 0.5, 0.5, 0.0, 1.0, 0.0, 0.5],
            [2.0, -0.5, -0.5, 2.0],
            2.0 / 3.0,
        )
    }

    #[test]
    fn cpu_ref_identity_holds() {
        let (a, b, x, r_mat, p_mat, a_c, omega) = fixture_4x2();
        let x_out = cpu_ref(&a, &b, &x, &r_mat, &p_mat, &a_c, omega, 4, 2);
        assert_eq!(x_out.len(), 4);
    }

    #[test]
    fn cpu_ref_into_reuses_output_and_scratch_storage() {
        let (a, b, x, r_mat, p_mat, a_c, omega) = fixture_4x2();
        let mut scratch = AmgVcycleScratch::default();
        let mut out = Vec::with_capacity(8);

        try_cpu_ref_into(
            &a,
            &b,
            &x,
            &r_mat,
            &p_mat,
            &a_c,
            omega,
            4,
            2,
            &mut scratch,
            &mut out,
        )
        .unwrap();
        let out_ptr = out.as_ptr();
        let residual_ptr = scratch.inner.residual.as_ptr();
        let first = out.clone();
        out.extend([99.0; 4]);
        try_cpu_ref_into(
            &a,
            &b,
            &x,
            &r_mat,
            &p_mat,
            &a_c,
            omega,
            4,
            2,
            &mut scratch,
            &mut out,
        )
        .unwrap();

        assert_eq!(out, first);
        assert_eq!(out.as_ptr(), out_ptr);
        assert_eq!(scratch.inner.residual.as_ptr(), residual_ptr);
    }

    #[test]
    fn try_cpu_ref_rejects_short_dense_inputs() {
        let err = try_cpu_ref(
            &[1.0],
            &[1.0, 2.0],
            &[0.0, 0.0],
            &[1.0, 0.0],
            &[1.0, 0.0],
            &[1.0],
            1.0,
            2,
            1,
        )
        .unwrap_err();
        assert!(err.contains("buffer `a` is too short"), "{err}");
    }

    #[test]
    fn generated_cpu_ref_matches_reusable_path() {
        for case in 0..24 {
            let n_fine = 3 + (case % 3);
            let n_coarse = 1 + (case % (n_fine - 1));
            let nf = n_fine as usize;
            let nc = n_coarse as usize;
            let mut a = vec![0.0; nf * nf];
            for i in 0..nf {
                a[i * nf + i] = 2.0 + case as f64 * 0.01;
                if i + 1 < nf {
                    a[i * nf + i + 1] = -0.25;
                    a[(i + 1) * nf + i] = -0.25;
                }
            }
            let b: Vec<f64> = (0..nf).map(|i| 1.0 + i as f64 * 0.125).collect();
            let x: Vec<f64> = (0..nf).map(|i| i as f64 * 0.01).collect();
            let mut r_mat = vec![0.0; nc * nf];
            let mut p_mat = vec![0.0; nf * nc];
            for i in 0..nc {
                r_mat[i * nf + (i * nf / nc)] = 1.0;
            }
            for i in 0..nf {
                p_mat[i * nc + (i * nc / nf).min(nc - 1)] = 1.0;
            }
            let mut a_c = vec![0.0; nc * nc];
            for i in 0..nc {
                a_c[i * nc + i] = 1.5 + case as f64 * 0.01;
            }
            let expected = cpu_ref(&a, &b, &x, &r_mat, &p_mat, &a_c, 0.5, n_fine, n_coarse);
            let mut scratch = AmgVcycleScratch::default();
            let mut out = Vec::with_capacity(expected.len() + 3);

            try_cpu_ref_into(
                &a,
                &b,
                &x,
                &r_mat,
                &p_mat,
                &a_c,
                0.5,
                n_fine,
                n_coarse,
                &mut scratch,
                &mut out,
            )
            .unwrap();

            assert_eq!(out.len(), expected.len(), "case {case}");
            for (idx, (&actual, &want)) in out.iter().zip(expected.iter()).enumerate() {
                assert!(
                    (actual - want).abs() < 1e-10,
                    "case {case} idx {idx}: expected {want}, got {actual}"
                );
            }
        }
    }

    #[test]
    fn program_has_correct_buffers() {
        let p = amg_v_cycle(
            "a", "b", "x", "r", "p", "ac", "om", "sf", "scb", "scx", 4, 2,
        );
        assert_eq!(p.buffers().len(), 11);
    }
}
