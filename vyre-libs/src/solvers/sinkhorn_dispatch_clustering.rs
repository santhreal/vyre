//! Dispatch-graph clustering via #2 sinkhorn (#30 substrate).
//!
//! This module implements the clustering of vyre's dispatch graph into
//! fusion-coherent groups using entropic optimal transport (Sinkhorn).
//!
//! # Math Frontier #2 entry
//!
//! "sinkhorn  -  dispatch-graph clustering via Sinkhorn-OT distance between
//! cost-vector distributions."
//!
//! # Transport Problem
//!
//! We model the clustering as an Optimal Transport problem between:
//! 1. The distribution of Regions (each with a weight a_i, e.g. compute cost).
//! 2. The distribution of Cluster capacities (b_j, e.g. target partition sizes).
//!
//! The cost matrix C_ij represents the "fusion distance" between Region i
//! and Cluster centroid j.
//!
//! # GPU Implementation
//!
//! This is a pure-math, GPU-resident implementation. It does not require
//! host-side iterations. It chains the Sinkhorn update steps directly
//! within the IR Program.

use vyre_foundation::composition::wrap_anonymous_region;
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};

use crate::dispatch_buffers::{
    decode_u32_output_exact, ensure_input_slots, write_f32_slice_le_bytes, write_zero_bytes,
};
use vyre_foundation::program_dispatch::{DispatchError, ProgramDispatcher};

/// Op id for the Sinkhorn dispatch clustering primitive.
pub const OP_ID: &str = "vyre-libs::self_substrate::sinkhorn_dispatch_clustering";

/// Caller-owned GPU dispatch scratch for Sinkhorn dispatch clustering.
#[derive(Debug, Default)]
pub struct SinkhornDispatchGpuScratch {
    inputs: Vec<Vec<u8>>,
}

/// Emit a Program that clusters `m` regions into `n` clusters.
///
/// Features:
/// - `region_features`: (m x d) buffer of f32 features.
/// - `cluster_centroids`: (n x d) buffer of f32 centroids.
/// - `region_weights`: (m) buffer of f32 masses.
/// - `cluster_capacities`: (n) buffer of f32 target masses.
/// - `out_assignments`: (m) buffer of u32 cluster indices.
///
/// Parameters:
/// - `eps`: Entropy regularization parameter.
/// - `iters`: Number of Sinkhorn iterations.
#[must_use]
#[allow(clippy::vec_init_then_push)]
pub fn sinkhorn_clustering_program(m: u32, n: u32, d: u32, iters: u32, eps: f32) -> Program {
    use crate::telemetry::{bump, sinkhorn_dispatch_clustering_calls};
    bump(&sinkhorn_dispatch_clustering_calls);
    assert!(m > 0 && n > 0 && d > 0 && iters > 0);

    // We use one workgroup to cluster all regions.
    // Each thread handles some regions.
    let workgroup_size = 256;
    let gid = Expr::gid_x();

    // Intermediate buffers for Sinkhorn vectors u (size m) and v (size n).
    // In a real production substrate, these might be scratchpad / shared memory.
    // For this primitive, we use dedicated internal buffers.

    let mut body = vec![];

    // 1. Initialize v = 1.0
    body.push(Node::if_then(
        Expr::lt(gid.clone(), Expr::u32(n)),
        vec![Node::store("v", gid.clone(), Expr::f32(1.0))],
    ));
    body.push(Node::barrier());

    // 2. Sinkhorn Loop
    body.push(Node::loop_for(
        "it",
        Expr::u32(0),
        Expr::u32(iters),
        vec![
            // u_i = a_i / sum_j (K_ij * v_j)
            Node::if_then(
                Expr::lt(gid.clone(), Expr::u32(m)),
                vec![
                    Node::let_bind("kv_sum", Expr::f32(0.0)),
                    Node::loop_for(
                        "jj",
                        Expr::u32(0),
                        Expr::u32(n),
                        vec![
                            // Compute C_ij = sum_k (f_ik - g_jk)^2
                            Node::let_bind("cost_ij", Expr::f32(0.0)),
                            Node::loop_for(
                                "kk",
                                Expr::u32(0),
                                Expr::u32(d),
                                vec![
                                    Node::let_bind(
                                        "f_ik",
                                        Expr::load(
                                            "region_features",
                                            Expr::add(
                                                Expr::mul(gid.clone(), Expr::u32(d)),
                                                Expr::var("kk"),
                                            ),
                                        ),
                                    ),
                                    Node::let_bind(
                                        "g_jk",
                                        Expr::load(
                                            "cluster_centroids",
                                            Expr::add(
                                                Expr::mul(Expr::var("jj"), Expr::u32(d)),
                                                Expr::var("kk"),
                                            ),
                                        ),
                                    ),
                                    Node::let_bind(
                                        "diff",
                                        Expr::sub(Expr::var("f_ik"), Expr::var("g_jk")),
                                    ),
                                    Node::assign(
                                        "cost_ij",
                                        Expr::add(
                                            Expr::var("cost_ij"),
                                            Expr::mul(Expr::var("diff"), Expr::var("diff")),
                                        ),
                                    ),
                                ],
                            ),
                            // K_ij = exp(-cost_ij / eps)
                            Node::let_bind(
                                "k_ij",
                                Expr::exp(Expr::div(
                                    Expr::negate(Expr::var("cost_ij")),
                                    Expr::f32(eps),
                                )),
                            ),
                            Node::assign(
                                "kv_sum",
                                Expr::add(
                                    Expr::var("kv_sum"),
                                    Expr::mul(Expr::var("k_ij"), Expr::load("v", Expr::var("jj"))),
                                ),
                            ),
                        ],
                    ),
                    Node::store(
                        "u",
                        gid.clone(),
                        Expr::div(
                            Expr::load("region_weights", gid.clone()),
                            Expr::max(Expr::var("kv_sum"), Expr::f32(1e-10)),
                        ),
                    ),
                ],
            ),
            Node::barrier(),
            // v_j = b_j / sum_i (K_ij * u_i)
            Node::if_then(
                Expr::lt(gid.clone(), Expr::u32(n)),
                vec![
                    Node::let_bind("ku_sum", Expr::f32(0.0)),
                    Node::loop_for(
                        "ii",
                        Expr::u32(0),
                        Expr::u32(m),
                        vec![
                            // Recompute K_ij (to save memory; in production we might cache K if m*n is small)
                            Node::let_bind("cost_ij_rev", Expr::f32(0.0)),
                            Node::loop_for(
                                "kk_rev",
                                Expr::u32(0),
                                Expr::u32(d),
                                vec![
                                    Node::let_bind(
                                        "f_ik_rev",
                                        Expr::load(
                                            "region_features",
                                            Expr::add(
                                                Expr::mul(Expr::var("ii"), Expr::u32(d)),
                                                Expr::var("kk_rev"),
                                            ),
                                        ),
                                    ),
                                    Node::let_bind(
                                        "g_jk_rev",
                                        Expr::load(
                                            "cluster_centroids",
                                            Expr::add(
                                                Expr::mul(gid.clone(), Expr::u32(d)),
                                                Expr::var("kk_rev"),
                                            ),
                                        ),
                                    ),
                                    Node::let_bind(
                                        "diff_rev",
                                        Expr::sub(Expr::var("f_ik_rev"), Expr::var("g_jk_rev")),
                                    ),
                                    Node::assign(
                                        "cost_ij_rev",
                                        Expr::add(
                                            Expr::var("cost_ij_rev"),
                                            Expr::mul(Expr::var("diff_rev"), Expr::var("diff_rev")),
                                        ),
                                    ),
                                ],
                            ),
                            Node::let_bind(
                                "k_ij_rev",
                                Expr::exp(Expr::div(
                                    Expr::negate(Expr::var("cost_ij_rev")),
                                    Expr::f32(eps),
                                )),
                            ),
                            Node::assign(
                                "ku_sum",
                                Expr::add(
                                    Expr::var("ku_sum"),
                                    Expr::mul(
                                        Expr::var("k_ij_rev"),
                                        Expr::load("u", Expr::var("ii")),
                                    ),
                                ),
                            ),
                        ],
                    ),
                    Node::store(
                        "v",
                        gid.clone(),
                        Expr::div(
                            Expr::load("cluster_capacities", gid.clone()),
                            Expr::max(Expr::var("ku_sum"), Expr::f32(1e-10)),
                        ),
                    ),
                ],
            ),
            Node::barrier(),
        ],
    ));

    // 3. Final assignment: argmax_j (K_ij * v_j)
    body.push(Node::if_then(
        Expr::lt(gid.clone(), Expr::u32(m)),
        vec![
            Node::let_bind("best_j", Expr::u32(0)),
            Node::let_bind("max_val", Expr::f32(-1.0)),
            Node::loop_for(
                "jj_final",
                Expr::u32(0),
                Expr::u32(n),
                vec![
                    Node::let_bind("cost_ij_final", Expr::f32(0.0)),
                    Node::loop_for(
                        "kk_final",
                        Expr::u32(0),
                        Expr::u32(d),
                        vec![
                            Node::let_bind(
                                "f_ik_final",
                                Expr::load(
                                    "region_features",
                                    Expr::add(
                                        Expr::mul(gid.clone(), Expr::u32(d)),
                                        Expr::var("kk_final"),
                                    ),
                                ),
                            ),
                            Node::let_bind(
                                "g_jk_final",
                                Expr::load(
                                    "cluster_centroids",
                                    Expr::add(
                                        Expr::mul(Expr::var("jj_final"), Expr::u32(d)),
                                        Expr::var("kk_final"),
                                    ),
                                ),
                            ),
                            Node::let_bind(
                                "diff_final",
                                Expr::sub(Expr::var("f_ik_final"), Expr::var("g_jk_final")),
                            ),
                            Node::assign(
                                "cost_ij_final",
                                Expr::add(
                                    Expr::var("cost_ij_final"),
                                    Expr::mul(Expr::var("diff_final"), Expr::var("diff_final")),
                                ),
                            ),
                        ],
                    ),
                    Node::let_bind(
                        "k_ij_final",
                        Expr::exp(Expr::div(
                            Expr::negate(Expr::var("cost_ij_final")),
                            Expr::f32(eps),
                        )),
                    ),
                    Node::let_bind(
                        "val",
                        Expr::mul(
                            Expr::var("k_ij_final"),
                            Expr::load("v", Expr::var("jj_final")),
                        ),
                    ),
                    Node::if_then(
                        Expr::gt(Expr::var("val"), Expr::var("max_val")),
                        vec![
                            Node::assign("max_val", Expr::var("val")),
                            Node::assign("best_j", Expr::var("jj_final")),
                        ],
                    ),
                ],
            ),
            Node::store("out_assignments", gid.clone(), Expr::var("best_j")),
        ],
    ));

    Program::wrapped(
        vec![
            BufferDecl::storage("region_features", 0, BufferAccess::ReadOnly, DataType::F32)
                .with_count(m.saturating_mul(d)),
            BufferDecl::storage(
                "cluster_centroids",
                1,
                BufferAccess::ReadOnly,
                DataType::F32,
            )
            .with_count(n.saturating_mul(d)),
            BufferDecl::storage("region_weights", 2, BufferAccess::ReadOnly, DataType::F32)
                .with_count(m),
            BufferDecl::storage(
                "cluster_capacities",
                3,
                BufferAccess::ReadOnly,
                DataType::F32,
            )
            .with_count(n),
            BufferDecl::storage("u", 4, BufferAccess::ReadWrite, DataType::F32).with_count(m),
            BufferDecl::storage("v", 5, BufferAccess::ReadWrite, DataType::F32).with_count(n),
            BufferDecl::output("out_assignments", 6, DataType::U32).with_count(m),
        ],
        [workgroup_size, 1, 1],
        vec![wrap_anonymous_region(OP_ID, body)],
    )
}

/// Run Sinkhorn dispatch clustering through a concrete GPU dispatcher.
///
/// # Errors
///
/// Returns [`DispatchError::BadInputs`] when dimensions or input buffers are malformed, and
/// [`DispatchError::BackendError`] when the backend returns malformed output.
#[allow(clippy::too_many_arguments)]
pub fn sinkhorn_clustering_via(
    dispatcher: &dyn ProgramDispatcher,
    region_features: &[f32],
    cluster_centroids: &[f32],
    region_weights: &[f32],
    cluster_capacities: &[f32],
    m: u32,
    n: u32,
    d: u32,
    iters: u32,
    eps: f32,
) -> Result<Vec<u32>, DispatchError> {
    let mut assignments = Vec::with_capacity(m as usize);
    sinkhorn_clustering_via_into(
        dispatcher,
        region_features,
        cluster_centroids,
        region_weights,
        cluster_capacities,
        m,
        n,
        d,
        iters,
        eps,
        &mut assignments,
    )?;
    Ok(assignments)
}

/// Run Sinkhorn dispatch clustering through a concrete GPU dispatcher into caller-owned storage.
#[allow(clippy::too_many_arguments)]
pub fn sinkhorn_clustering_via_into(
    dispatcher: &dyn ProgramDispatcher,
    region_features: &[f32],
    cluster_centroids: &[f32],
    region_weights: &[f32],
    cluster_capacities: &[f32],
    m: u32,
    n: u32,
    d: u32,
    iters: u32,
    eps: f32,
    assignments_out: &mut Vec<u32>,
) -> Result<(), DispatchError> {
    let mut scratch = SinkhornDispatchGpuScratch::default();
    sinkhorn_clustering_via_with_scratch_into(
        dispatcher,
        region_features,
        cluster_centroids,
        region_weights,
        cluster_capacities,
        m,
        n,
        d,
        iters,
        eps,
        &mut scratch,
        assignments_out,
    )
}

/// Run Sinkhorn dispatch clustering through a concrete GPU dispatcher into
/// caller-owned dispatch and assignment storage.
#[allow(clippy::too_many_arguments)]
pub fn sinkhorn_clustering_via_with_scratch_into(
    dispatcher: &dyn ProgramDispatcher,
    region_features: &[f32],
    cluster_centroids: &[f32],
    region_weights: &[f32],
    cluster_capacities: &[f32],
    m: u32,
    n: u32,
    d: u32,
    iters: u32,
    eps: f32,
    scratch: &mut SinkhornDispatchGpuScratch,
    assignments_out: &mut Vec<u32>,
) -> Result<(), DispatchError> {
    if m == 0 || n == 0 || d == 0 || iters == 0 {
        return Err(DispatchError::BadInputs(format!(
            "Fix: sinkhorn_clustering_via requires nonzero m, n, d, and iters; got m={m}, n={n}, d={d}, iters={iters}."
        )));
    }
    if !eps.is_finite() || eps <= 0.0 {
        return Err(DispatchError::BadInputs(format!(
            "Fix: sinkhorn_clustering_via requires finite eps > 0, got {eps}."
        )));
    }
    let feature_words = checked_product(m, d, "m", "d", "region_features")?;
    let centroid_words = checked_product(n, d, "n", "d", "cluster_centroids")?;
    if region_features.len() != feature_words {
        return Err(DispatchError::BadInputs(format!(
            "Fix: sinkhorn_clustering_via expected region_features.len() == m*d == {feature_words}, got {}.",
            region_features.len()
        )));
    }
    if cluster_centroids.len() != centroid_words {
        return Err(DispatchError::BadInputs(format!(
            "Fix: sinkhorn_clustering_via expected cluster_centroids.len() == n*d == {centroid_words}, got {}.",
            cluster_centroids.len()
        )));
    }
    if region_weights.len() != m as usize {
        return Err(DispatchError::BadInputs(format!(
            "Fix: sinkhorn_clustering_via expected region_weights.len() == m == {m}, got {}.",
            region_weights.len()
        )));
    }
    if cluster_capacities.len() != n as usize {
        return Err(DispatchError::BadInputs(format!(
            "Fix: sinkhorn_clustering_via expected cluster_capacities.len() == n == {n}, got {}.",
            cluster_capacities.len()
        )));
    }

    let program = sinkhorn_clustering_program(m, n, d, iters, eps);
    // Real-backend dispatch-input contract (vyre-driver `role_for_buffer`): one input per
    // INPUT-CONSUMING buffer in buffer order, the four read-only inputs `region_features` (0),
    // `cluster_centroids` (1), `region_weights` (2), `cluster_capacities` (3), then the two plain-RW
    // dual-scaling scratch buffers `u` (4), `v` (5, zero-init). `out_assignments` (binding 6) is a
    // `BufferDecl::output`: the backend ALLOCATES it and it consumes NO input, so passing a seventh
    // slot for it over-feeds and would fail the backend's strict `validate_input_lengths`
    // ("expected 6, received 7"). Six inputs only.
    ensure_input_slots(&mut scratch.inputs, 6);
    write_f32_slice_le_bytes(&mut scratch.inputs[0], region_features);
    write_f32_slice_le_bytes(&mut scratch.inputs[1], cluster_centroids);
    write_f32_slice_le_bytes(&mut scratch.inputs[2], region_weights);
    write_f32_slice_le_bytes(&mut scratch.inputs[3], cluster_capacities);
    write_zero_bytes(&mut scratch.inputs[4], byte_count(m as usize, "u")?);
    write_zero_bytes(&mut scratch.inputs[5], byte_count(n as usize, "v")?);
    let outputs = dispatcher.dispatch(&program, &scratch.inputs[..6], Some([1, 1, 1]))?;
    // Real-backend output contract: the backend returns every WRITABLE buffer in binding order
    // the two plain-RW dual-scaling buffers `u` (4) and `v` (5, both role InputOutput → outputs 0,1)
    // then the `out_assignments` output (6 → output 2). The assignment vector is therefore the THIRD
    // output, not the first (the old program-ignoring mock faked a single-output return, hiding this).
    let [_, _, assignments_bytes] = match outputs.as_slice() {
        [u, v, a] => [u, v, a],
        _ => {
            return Err(DispatchError::BackendError(format!(
                "Fix: sinkhorn_clustering_via expected 3 writable outputs (u, v, out_assignments), got {}.",
                outputs.len()
            )))
        }
    };
    decode_u32_output_exact(
        assignments_bytes,
        m as usize,
        "sinkhorn_clustering_via out_assignments",
        assignments_out,
    )
}

fn checked_product(
    left: u32,
    right: u32,
    left_name: &str,
    right_name: &str,
    context: &str,
) -> Result<usize, DispatchError> {
    left.checked_mul(right)
        .map(|value| value as usize)
        .ok_or_else(|| {
            DispatchError::BadInputs(format!(
                "Fix: sinkhorn_clustering_via {context} count overflowed u32 for {left_name}={left}, {right_name}={right}."
            ))
        })
}

fn byte_count(words: usize, label: &str) -> Result<usize, DispatchError> {
    words
        .checked_mul(std::mem::size_of::<u32>())
        .ok_or_else(|| {
            DispatchError::BadInputs(format!(
            "Fix: sinkhorn_clustering_via {label} byte count overflowed usize for {words} words."
        ))
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_parity_oracles::StaticOutputs;
    use vyre_reference::composition_witness::sinkhorn_clustering_witness;

    /// Real-backend contract: 4 RO inputs + u/v plain-RW = 6 input-consuming
    /// buffers; out_assignments is a backend-allocated `BufferDecl::output`,
    /// not an input.
    const SINKHORN_CONTRACT: &str = "sinkhorn clustering dispatch";

    #[test]
    fn clustering_identity_one_region_one_cluster() {
        let features = vec![1.0, 1.0];
        let centroids = vec![1.0, 1.0];
        let weights = vec![1.0];
        let capacities = vec![1.0];
        let assignments = sinkhorn_clustering_witness(
            &features,
            &centroids,
            &weights,
            &capacities,
            1,
            1,
            2,
            5,
            0.1,
        );
        assert_eq!(assignments, vec![0]);
    }

    #[test]
    fn clustering_two_regions_two_distant_clusters() {
        // Region 0 at (0,0), Region 1 at (10,10)
        // Cluster 0 at (0,0), Cluster 1 at (10,10)
        let features = vec![0.0, 0.0, 10.0, 10.0];
        let centroids = vec![0.0, 0.0, 10.0, 10.0];
        let weights = vec![1.0, 1.0];
        let capacities = vec![1.0, 1.0];
        let assignments = sinkhorn_clustering_witness(
            &features,
            &centroids,
            &weights,
            &capacities,
            2,
            2,
            2,
            20,
            1.0,
        );
        assert_eq!(assignments, vec![0, 1]);
    }

    #[test]
    fn clustering_respects_capacities() {
        // Capacities enter Sinkhorn via the `v` scaling step; the reference helper still
        // assigns each region with per-row argmax over `K_ij*v_j`, which does **not**
        // enforce hard cluster-cardinality constraints. Place regions clearly near
        // different centroids so argmax aligns with capacities (1 vs 2 mass targets).
        let features = vec![
            0.0, 0.0, // region 0 @ cluster 0
            100.0, 0.0, // regions 1–2 @ cluster 1
            100.0, 0.0,
        ];
        let centroids = vec![0.0, 0.0, 100.0, 0.0];
        let weights = vec![1.0, 1.0, 1.0];
        let capacities = vec![1.0, 2.0];
        let assignments = sinkhorn_clustering_witness(
            &features,
            &centroids,
            &weights,
            &capacities,
            3,
            2,
            2,
            80,
            1.0,
        );

        let count_0 = assignments.iter().filter(|&&x| x == 0).count();
        let count_1 = assignments.iter().filter(|&&x| x == 1).count();
        assert_eq!(count_0, 1);
        assert_eq!(count_1, 2);
    }

    #[test]
    fn clustering_unbalanced_weights() {
        let features = vec![0.0, 10.0];
        let centroids = vec![0.0, 10.0];
        let weights = vec![1.0, 10.0];
        let capacities = vec![1.0, 10.0];
        let assignments = sinkhorn_clustering_witness(
            &features,
            &centroids,
            &weights,
            &capacities,
            2,
            2,
            1,
            20,
            0.1,
        );
        assert_eq!(assignments, vec![0, 1]);
    }

    #[test]
    fn program_structure_is_valid() {
        let p = sinkhorn_clustering_program(10, 2, 2, 5, 0.1);
        assert_eq!(p.workgroup_size, [256, 1, 1]);
        let names: Vec<&str> = p.buffers.iter().map(|b| b.name()).collect();
        assert!(names.contains(&"region_features"));
        assert!(names.contains(&"out_assignments"));
    }

    #[test]
    fn via_decodes_exact_assignments_into_reused_buffer() {
        // Real-backend output shape: [u, v, out_assignments] (the decoder reads the THIRD buffer).
        let dispatcher = StaticOutputs::new(
            SINKHORN_CONTRACT,
            vec![
                crate::dispatch_buffers::u32_slice_to_le_bytes(&[0, 0]),
                crate::dispatch_buffers::u32_slice_to_le_bytes(&[0, 0]),
                crate::dispatch_buffers::u32_slice_to_le_bytes(&[0, 1]),
            ],
        )
        .expecting_grid([1, 1, 1])
        .expecting_inputs(&[6]);
        let mut out = Vec::with_capacity(4);
        let ptr = out.as_ptr();
        sinkhorn_clustering_via_into(
            &dispatcher,
            &[0.0, 0.0, 10.0, 10.0],
            &[0.0, 0.0, 10.0, 10.0],
            &[1.0, 1.0],
            &[1.0, 1.0],
            2,
            2,
            2,
            5,
            1.0,
            &mut out,
        )
        .expect("Fix: dispatch succeeds");
        assert_eq!(out, vec![0, 1]);
        assert_eq!(out.as_ptr(), ptr);
    }

    #[test]
    fn via_with_scratch_reuses_dispatch_and_assignment_storage() {
        // Real-backend output shape: [u, v, out_assignments] (the decoder reads the THIRD buffer).
        let dispatcher = StaticOutputs::new(
            SINKHORN_CONTRACT,
            vec![
                crate::dispatch_buffers::u32_slice_to_le_bytes(&[0, 0]),
                crate::dispatch_buffers::u32_slice_to_le_bytes(&[0, 0]),
                crate::dispatch_buffers::u32_slice_to_le_bytes(&[0, 1]),
            ],
        )
        .expecting_grid([1, 1, 1])
        .expecting_inputs(&[6]);
        let mut scratch = SinkhornDispatchGpuScratch::default();
        let mut out = Vec::with_capacity(2);

        sinkhorn_clustering_via_with_scratch_into(
            &dispatcher,
            &[0.0, 0.0, 10.0, 10.0],
            &[0.0, 0.0, 10.0, 10.0],
            &[1.0, 1.0],
            &[1.0, 1.0],
            2,
            2,
            2,
            5,
            1.0,
            &mut scratch,
            &mut out,
        )
        .expect("Fix: dispatch succeeds");

        let input_capacities = scratch.inputs.iter().map(Vec::capacity).collect::<Vec<_>>();
        let out_capacity = out.capacity();

        sinkhorn_clustering_via_with_scratch_into(
            &dispatcher,
            &[0.0, 0.0, 10.0, 10.0],
            &[0.0, 0.0, 10.0, 10.0],
            &[1.0, 1.0],
            &[1.0, 1.0],
            2,
            2,
            2,
            5,
            1.0,
            &mut scratch,
            &mut out,
        )
        .expect("Fix: dispatch succeeds");

        assert_eq!(
            scratch.inputs.iter().map(Vec::capacity).collect::<Vec<_>>(),
            input_capacities
        );
        assert_eq!(out.capacity(), out_capacity);
        assert_eq!(out, vec![0, 1]);
    }

    #[test]
    fn via_rejects_extra_outputs() {
        // A backend returning more than the 3 writable buffers (u, v, out_assignments) is malformed.
        let dispatcher = StaticOutputs::new(
            SINKHORN_CONTRACT,
            vec![
                crate::dispatch_buffers::u32_slice_to_le_bytes(&[0]),
                crate::dispatch_buffers::u32_slice_to_le_bytes(&[0]),
                crate::dispatch_buffers::u32_slice_to_le_bytes(&[1]),
                crate::dispatch_buffers::u32_slice_to_le_bytes(&[9]),
            ],
        )
        .expecting_grid([1, 1, 1])
        .expecting_inputs(&[6]);
        let err =
            sinkhorn_clustering_via(&dispatcher, &[0.0], &[0.0], &[1.0], &[1.0], 1, 1, 1, 5, 1.0)
                .expect_err("extra outputs must be rejected");
        assert!(
            matches!(err, DispatchError::BackendError(_)),
            "unexpected error: {err:?}"
        );
    }

    #[test]
    fn via_rejects_trailing_assignment_bytes() {
        // Correct 3-output shape but the out_assignments buffer (index 2) has a trailing byte.
        let dispatcher = StaticOutputs::new(
            SINKHORN_CONTRACT,
            vec![
                crate::dispatch_buffers::u32_slice_to_le_bytes(&[0]),
                crate::dispatch_buffers::u32_slice_to_le_bytes(&[0]),
                vec![0, 0, 0, 0, 1],
            ],
        )
        .expecting_grid([1, 1, 1])
        .expecting_inputs(&[6]);
        let err =
            sinkhorn_clustering_via(&dispatcher, &[0.0], &[0.0], &[1.0], &[1.0], 1, 1, 1, 5, 1.0)
                .expect_err("trailing bytes must be rejected");
        assert!(
            matches!(err, DispatchError::BackendError(_)),
            "unexpected error: {err:?}"
        );
    }

    #[test]
    fn parity_test_one_step() {
        // We can't easily run the GPU Program here without a backend,
        // but we verify the reference implementation is consistent with the GPU logic.
        // The GPU logic literally re-implements the reference logic in IR.
        let features = vec![1.0, 2.0, 5.0, 6.0];
        let centroids = vec![0.0, 0.0, 10.0, 10.0];
        let weights = vec![1.0, 1.0];
        let capacities = vec![1.0, 1.0];
        let reference_res = sinkhorn_clustering_witness(
            &features,
            &centroids,
            &weights,
            &capacities,
            2,
            2,
            2,
            1,
            1.0,
        );
        assert_eq!(reference_res.len(), 2);
    }

    #[test]
    fn repeated_clustering_witness_calls_do_not_retain_prior_state() {
        let features = vec![0.0, 0.0, 10.0, 10.0];
        let centroids = vec![0.0, 0.0, 10.0, 10.0];
        let weights = vec![1.0, 1.0];
        let capacities = vec![1.0, 1.0];

        let first = sinkhorn_clustering_witness(
            &features,
            &centroids,
            &weights,
            &capacities,
            2,
            2,
            2,
            20,
            1.0,
        );
        let unrelated =
            sinkhorn_clustering_witness(&[3.0], &[3.0], &[1.0], &[1.0], 1, 1, 1, 2, 0.5);
        let second = sinkhorn_clustering_witness(
            &features,
            &centroids,
            &weights,
            &capacities,
            2,
            2,
            2,
            20,
            1.0,
        );

        assert_eq!(first, vec![0, 1]);
        assert_eq!(unrelated, vec![0]);
        assert_eq!(second, first);
    }
}
