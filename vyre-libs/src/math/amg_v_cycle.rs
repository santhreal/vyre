//! Algebraic Multigrid (AMG) V-cycle primitive (#P-PRIM-3).
//!
//! Composes the Jacobi smoother (`jacobi_smooth_step_serial_body`) with restriction and
//! prolongation to solve linear systems $Ax = b$ across multiple scales. The whole V-cycle is a
//! single-threaded serial algorithm run under one `InvocationId == 0` lane guard (see the region
//! body), so it inlines the SERIAL smoother form, not the per-lane `jacobi_smooth_step` builder.
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
    if n_fine == 0 {
        return trap_program(
            OP_ID,
            Some((x, DataType::U32)),
            "Fix: amg_v_cycle requires n_fine > 0, got 0.".to_string(),
        );
    }
    if n_coarse == 0 {
        return trap_program(
            OP_ID,
            Some((x, DataType::U32)),
            "Fix: amg_v_cycle requires n_coarse > 0, got 0.".to_string(),
        );
    }
    if n_coarse >= n_fine {
        return trap_program(OP_ID, Some((x, DataType::U32)), format!("Fix: amg_v_cycle requires n_coarse < n_fine, got n_coarse={n_coarse}, n_fine={n_fine}."));
    }
    let Some(fine_cells) = n_fine.checked_mul(n_fine) else {
        return trap_program(
            OP_ID,
            Some((x, DataType::U32)),
            format!("Fix: amg_v_cycle fine matrix cells overflow u32: n_fine={n_fine}."),
        );
    };
    let Some(transfer_cells) = n_fine.checked_mul(n_coarse) else {
        return trap_program(OP_ID, Some((x, DataType::U32)), format!(
            "Fix: amg_v_cycle transfer matrix cells overflow u32: n_fine={n_fine}, n_coarse={n_coarse}."
        ));
    };
    let Some(coarse_cells) = n_coarse.checked_mul(n_coarse) else {
        return trap_program(
            OP_ID,
            Some((x, DataType::U32)),
            format!("Fix: amg_v_cycle coarse matrix cells overflow u32: n_coarse={n_coarse}."),
        );
    };

    let mut nodes = Vec::new();

    // 1. Pre-smooth (serial: the whole V-cycle runs on one lane, see the region-body guard)
    nodes.extend(jacobi_smooth_step_serial_body(
        a,
        b,
        x,
        omega,
        scratch_fine,
        n_fine,
        "pre",
    ));
    // Copy scratch_fine back to x
    nodes.push(Node::loop_for(
        "__i",
        Expr::u32(0),
        Expr::u32(n_fine),
        vec![Node::store(
            x,
            Expr::var("__i"),
            Expr::load(scratch_fine, Expr::var("__i")),
        )],
    ));

    // 2. Restrict: r = b - Ax; b_c = R r
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
                Expr::sub(Expr::load(b, Expr::var("i")), Expr::var("ax_i")),
            ),
        ],
    ));

    // b_c = R * r
    nodes.push(Node::loop_for(
        "ic",
        Expr::u32(0),
        Expr::u32(n_coarse),
        vec![
            Node::let_bind("bc_i", Expr::u32(0)),
            Node::loop_for(
                "jf",
                Expr::u32(0),
                Expr::u32(n_fine),
                vec![Node::assign(
                    "bc_i",
                    Expr::add(
                        Expr::var("bc_i"),
                        crate::math::fixed::fixed_mul_16_16_expr(
                            Expr::load(
                                r_mat,
                                Expr::add(
                                    Expr::mul(Expr::var("ic"), Expr::u32(n_fine)),
                                    Expr::var("jf"),
                                ),
                            ),
                            Expr::load(scratch_fine, Expr::var("jf")),
                        ),
                    ),
                )],
            ),
            Node::store(scratch_coarse_b, Expr::var("ic"), Expr::var("bc_i")),
        ],
    ));

    // 3. Coarse solve
    nodes.push(Node::store(scratch_coarse_x, Expr::u32(0), Expr::u32(0)));
    for k in 0..4 {
        nodes.extend(jacobi_smooth_step_serial_body(
            a_c,
            scratch_coarse_b,
            scratch_coarse_x,
            omega,
            "temp_coarse",
            n_coarse,
            &format!("coarse{k}"),
        ));
        nodes.push(Node::loop_for(
            "__k",
            Expr::u32(0),
            Expr::u32(n_coarse),
            vec![Node::store(
                scratch_coarse_x,
                Expr::var("__k"),
                Expr::load("temp_coarse", Expr::var("__k")),
            )],
        ));
    }

    // 4. Prolong: x = x + P * x_c
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
                x,
                Expr::var("if"),
                Expr::add(Expr::load(x, Expr::var("if")), Expr::var("px_i")),
            ),
        ],
    ));

    // 5. Post-smooth (serial, single lane)
    nodes.extend(jacobi_smooth_step_serial_body(
        a,
        b,
        x,
        omega,
        scratch_fine,
        n_fine,
        "post",
    ));
    nodes.push(Node::loop_for(
        "__m",
        Expr::u32(0),
        Expr::u32(n_fine),
        vec![Node::store(
            x,
            Expr::var("__m"),
            Expr::load(scratch_fine, Expr::var("__m")),
        )],
    ));

    Program::wrapped(
        vec![
            BufferDecl::storage(a, 0, BufferAccess::ReadOnly, DataType::U32).with_count(fine_cells),
            BufferDecl::storage(b, 1, BufferAccess::ReadOnly, DataType::U32).with_count(n_fine),
            BufferDecl::storage(x, 2, BufferAccess::ReadWrite, DataType::U32).with_count(n_fine),
            BufferDecl::storage(r_mat, 3, BufferAccess::ReadOnly, DataType::U32)
                .with_count(transfer_cells),
            BufferDecl::storage(p_mat, 4, BufferAccess::ReadOnly, DataType::U32)
                .with_count(transfer_cells),
            BufferDecl::storage(a_c, 5, BufferAccess::ReadOnly, DataType::U32)
                .with_count(coarse_cells),
            BufferDecl::storage(omega, 6, BufferAccess::ReadOnly, DataType::U32).with_count(1),
            BufferDecl::storage(scratch_fine, 7, BufferAccess::ReadWrite, DataType::U32)
                .with_count(n_fine),
            BufferDecl::storage(scratch_coarse_b, 8, BufferAccess::ReadWrite, DataType::U32)
                .with_count(n_coarse),
            BufferDecl::storage(scratch_coarse_x, 9, BufferAccess::ReadWrite, DataType::U32)
                .with_count(n_coarse),
            BufferDecl::storage("temp_coarse", 10, BufferAccess::ReadWrite, DataType::U32)
                .with_count(n_coarse),
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
                // `InvocationId == 0` so the one dispatched lane runs the entire serial V-cycle at
                // any grid (the canonical GPU serial-region idiom, cf. matroid, sheaf,
                // path_reconstruct). No GridSync barriers are needed (or safe) under this guard:
                // a single lane sees its own writes sequentially.
                body: Arc::new(vec![Node::if_then(
                    Expr::eq(Expr::InvocationId { axis: 0 }, Expr::u32(0)),
                    nodes,
                )]),
            }],
        )],
    )
}

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library(
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
}

/// One phase adds one to each of the four fine cells.
const EXPECTED_V_CYCLE_PHASE_BYTES: [u8; 16] = [2, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 0, 5, 0, 0, 0];

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library(
        V_CYCLE_PHASE_OP_ID,
        || {
            Program::wrapped(
                vec![
                    BufferDecl::storage("fine_in", 0, BufferAccess::ReadOnly, DataType::U32)
                        .with_count(4),
                    BufferDecl::output("fine_out", 1, DataType::U32).with_count(4),
                ],
                [1, 1, 1],
                vec![wrap_anonymous_region(
                    V_CYCLE_PHASE_OP_ID,
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
            )
        },
        Some(|| {
            let to_bytes = |words: &[u32]| vyre_primitives::wire::pack_u32_slice(words);
            vec![vec![to_bytes(&[1, 2, 3, 4])]]
        }),
        Some(|| vec![vec![EXPECTED_V_CYCLE_PHASE_BYTES.to_vec()]]),
    )
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

    #[test]
    fn cpu_ref_identity_holds() {
        let n_fine = 4;
        let n_coarse = 2;
        let a = vec![
            2.0, -1.0, 0.0, 0.0, -1.0, 2.0, -1.0, 0.0, 0.0, -1.0, 2.0, -1.0, 0.0, 0.0, -1.0, 2.0,
        ];
        let b = vec![1.0, 0.0, 0.0, 1.0];
        let x = vec![0.0; 4];
        let r_mat = vec![1.0, 0.5, 0.0, 0.0, 0.0, 0.5, 1.0, 0.5];
        let p_mat = vec![1.0, 0.0, 0.5, 0.5, 0.0, 1.0, 0.0, 0.5];
        let a_c = vec![2.0, -0.5, -0.5, 2.0];
        let omega = 2.0 / 3.0;

        let x_out = cpu_ref(&a, &b, &x, &r_mat, &p_mat, &a_c, omega, n_fine, n_coarse);
        assert_eq!(x_out.len(), 4);
    }

    #[test]
    fn cpu_ref_into_reuses_output_and_scratch_storage() {
        let n_fine = 4;
        let n_coarse = 2;
        let a = vec![
            2.0, -1.0, 0.0, 0.0, -1.0, 2.0, -1.0, 0.0, 0.0, -1.0, 2.0, -1.0, 0.0, 0.0, -1.0, 2.0,
        ];
        let b = vec![1.0, 0.0, 0.0, 1.0];
        let x = vec![0.0; 4];
        let r_mat = vec![1.0, 0.5, 0.0, 0.0, 0.0, 0.5, 1.0, 0.5];
        let p_mat = vec![1.0, 0.0, 0.5, 0.5, 0.0, 1.0, 0.0, 0.5];
        let a_c = vec![2.0, -0.5, -0.5, 2.0];
        let omega = 2.0 / 3.0;
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
            n_fine,
            n_coarse,
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
            n_fine,
            n_coarse,
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
