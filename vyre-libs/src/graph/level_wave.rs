//! `level_wave_program`  -  GPU-resident depth-wave dispatcher for
//! bottom-up callee-before-caller computations.
//!
//! Semantically distinct from `fixpoint::persistent_fixpoint`:
//! - **persistent_fixpoint**: re-run a transfer step until convergence.
//!   No depth ordering  -  every lane runs the step every iteration.
//! - **level_wave**: deterministic ordered traversal. Each lane runs
//!   the step only when `current_depth == depth`lane``. Used for
//!   bottom-up summary computations where children must complete
//!   before parents.
//!
//! ## LEGO discipline
//!
//! Composes:
//! - [`crate::graph::toposort::toposort()`]  -  CPU reference for the depth
//!   assignment (caller computes `depth[node]` from the topological
//!   ordering before invoking this primitive).
//! - `Node::Loop` (vyre-foundation IR primitive)  -  outer per-depth
//!   loop.
//! - `Node::Barrier { ordering: vyre_foundation::ir::MemoryOrdering::SeqCst }`  -  synchronisation between depth waves.
//! - `Expr::eq` + `Node::if_then`  -  depth predicate per lane.
//!
//! No new sub-op invented. The caller composes its own per-lane work
//! body; this primitive provides the wave harness.
//!
//! ## Composition contract
//!
//! Caller supplies:
//!
//! - `depth_buf`: per-node depth bitset (u32 per lane). The lane
//!   reads its own depth and gates its work on equality with the
//!   current wave depth.
//! - `step_body`: caller-provided IR body that runs ONE node's work.
//!   Reads `current_depth` and `depth_buf`lane``; the
//!   level_wave_program guards the body in `if depth == current`
//!   already, so the body itself doesn't need to re-check.
//! - `max_depth`: maximum depth value in the topology.
//!
//! Caller receives a `Program` that runs every lane at every depth wave
//! from 0..max_depth. Single-workgroup waves use one compact loop.
//! Multi-workgroup waves expose top-level `GridSync` boundaries so
//! backends without native grid barriers can split the traversal into
//! launch-separated depth waves.

use vyre_foundation::composition::wrap_anonymous_region;

use vyre_foundation::ir::{
    BufferAccess, BufferDecl, DataType, Expr, MemoryOrdering, Node, Program,
};

/// Canonical op id.
pub const OP_ID: &str = "vyre-libs::graph::level_wave";
/// Workgroup shape for per-node depth-wave traversal.
pub const LEVEL_WAVE_WORKGROUP_SIZE: [u32; 3] = [256, 1, 1];

/// Dispatch grid that covers every level-wave lane.
#[must_use]
pub const fn level_wave_dispatch_grid(lane_count: u32) -> [u32; 3] {
    vyre_primitives::lane_grid(lane_count, LEVEL_WAVE_WORKGROUP_SIZE[0])
}

fn depth_wave_body(
    step_body: Vec<Node>,
    depth_buf: &str,
    depth: Expr,
    lane_count: u32,
) -> Vec<Node> {
    let lane = Expr::InvocationId { axis: 0 };
    // The range check MUST control-flow-nest the depth load, not `Expr::and` it:
    // `and(lane < lane_count, load(depth_buf, lane) == depth)` evaluates BOTH operands
    // (the IR has no short-circuit), so `load(depth_buf, lane)` reads `depth_buf` for the
    // whole-workgroup lanes a real GPU fires past `lane_count`: an OOB read the reference
    // silently masks but hardware faults on (the ssa_dominance_scan gather-class bug,
    // BACKLOG BUG-level-wave-depth-guard-eager-oob-load). Nesting means the load only runs
    // when `lane < lane_count`.
    vec![Node::if_then(
        Expr::lt(lane.clone(), Expr::u32(lane_count)),
        vec![Node::if_then(
            Expr::eq(Expr::load(depth_buf, lane), depth),
            step_body,
        )],
    )]
}

/// Build a Program that runs `step_body` per lane in
/// depth-ordered waves.
///
/// Each lane reads `depth_buf[invocation_id]`. The kernel walks
/// `current_depth = 0..max_depth`. At each depth, every lane whose
/// depth equals `current_depth` executes `step_body`. A `Barrier` is
/// emitted between depths so the caller can rely on depth-N effects
/// being globally visible before depth-N+1 begins.
///
/// # Parameters
///
/// - `step_body`: caller's per-lane work body. Free to read/write
///   any buffer the caller declares; it does NOT need to re-check
///   the depth predicate (the wrapper does that).
/// - `depth_buf`: buffer-name holding per-lane depth (u32). Read-only.
/// - `max_depth`: number of waves to execute.
/// - `lane_count`: total number of lanes in the dispatch grid.
#[must_use]
pub fn level_wave_program(
    step_body: Vec<Node>,
    depth_buf: &str,
    max_depth: u32,
    lane_count: u32,
) -> Program {
    level_wave_program_with_buffers(step_body, depth_buf, Vec::new(), max_depth, lane_count)
}

/// Like [`level_wave_program`], but declares `extra_buffers` after the depth
/// buffer so the `step_body` can read/write the caller's own storage.
///
/// `depth_buf` is bound at index 0; every entry in `extra_buffers` MUST carry
/// a distinct binding index `>= 1` (the caller owns the binding layout its
/// `step_body` references). This is the composition point used by
/// depth-ordered evaluators (e.g. `sum_product_evaluate_leveled`) that need
/// their own inputs/outputs visible inside the per-lane wave body while still
/// getting the ONE-PLACE depth-wave harness + inter-wave barriers.
#[must_use]
pub fn level_wave_program_with_buffers(
    step_body: Vec<Node>,
    depth_buf: &str,
    extra_buffers: Vec<BufferDecl>,
    max_depth: u32,
    lane_count: u32,
) -> Program {
    level_wave_program_with_buffers_and_op_id(
        OP_ID,
        step_body,
        depth_buf,
        extra_buffers,
        max_depth,
        lane_count,
    )
}

/// Build a Program that visits every function in callee-before-caller
/// order using GPU-side level-wave dispatch.
///
/// `step_body`: per-function body. Reads/writes any caller-declared
/// buffer via `Expr::InvocationId { axis: 0 }` to address the function
/// being visited.
///
/// `depth_buf`: name of the buffer containing per-function depth in the
/// call graph (leaves at 0).
///
/// `max_depth`: number of waves (i.e., `max(depth) + 1`).
///
/// `function_count`: total functions in the dispatch grid.
#[must_use]
pub fn build_callee_before_caller_program(
    step_body: Vec<Node>,
    depth_buf: &str,
    max_depth: u32,
    function_count: u32,
) -> Program {
    level_wave_program(step_body, depth_buf, max_depth, function_count)
}

/// Like [`build_callee_before_caller_program`], but declares the pass's own
/// per-function DATA buffers after `depth_buf` so the `step_body` can read/write
/// them.
#[must_use]
pub fn build_callee_before_caller_program_with_buffers(
    step_body: Vec<Node>,
    depth_buf: &str,
    extra_buffers: Vec<BufferDecl>,
    max_depth: u32,
    function_count: u32,
) -> Program {
    level_wave_program_with_buffers(
        step_body,
        depth_buf,
        extra_buffers,
        max_depth,
        function_count,
    )
}

/// Same as [`level_wave_program_with_buffers`] with an explicit caller op id.
#[must_use]
pub fn level_wave_program_with_buffers_and_op_id(
    op_id: &str,
    step_body: Vec<Node>,
    depth_buf: &str,
    extra_buffers: Vec<BufferDecl>,
    max_depth: u32,
    lane_count: u32,
) -> Program {
    let body = if lane_count <= LEVEL_WAVE_WORKGROUP_SIZE[0] {
        vec![Node::loop_for(
            "__lw_depth__",
            Expr::u32(0),
            Expr::u32(max_depth),
            {
                let mut loop_body = depth_wave_body(
                    step_body.clone(),
                    depth_buf,
                    Expr::var("__lw_depth__"),
                    lane_count,
                );
                loop_body.push(Node::Barrier {
                    ordering: MemoryOrdering::SeqCst,
                });
                loop_body
            },
        )]
    } else {
        let mut waves = Vec::with_capacity(max_depth.saturating_mul(2) as usize);
        for depth in 0..max_depth {
            waves.extend(depth_wave_body(
                step_body.clone(),
                depth_buf,
                Expr::u32(depth),
                lane_count,
            ));
            if depth + 1 < max_depth {
                waves.push(Node::Barrier {
                    ordering: MemoryOrdering::GridSync,
                });
            }
        }
        waves
    };

    let mut buffers =
        vec![
            BufferDecl::storage(depth_buf, 0, BufferAccess::ReadOnly, DataType::U32)
                .with_count(lane_count),
        ];
    buffers.extend(extra_buffers);

    Program::wrapped(
        buffers,
        LEVEL_WAVE_WORKGROUP_SIZE,
        vec![wrap_anonymous_region(op_id, body)],
    )
}

/// CPU oracle. Iterates depth waves on the host and calls
/// `step_for_lane(lane, depth)` exactly once per (lane, depth ==
/// depth_for_lane`lane`). Used by the conformance harness to verify
/// that the GPU kernel respects the depth ordering.
#[cfg(any(test, feature = "cpu-parity"))]
pub fn cpu_ref<F>(depths: &[u32], max_depth: u32, mut step_for_lane: F)
where
    F: FnMut(u32, u32),
{
    for current_depth in 0..max_depth {
        for (lane_idx, lane_depth) in depths.iter().enumerate() {
            if *lane_depth == current_depth {
                step_for_lane(lane_idx as u32, current_depth);
            }
        }
    }
}

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library(
        OP_ID,
        || {
            level_wave_program_with_buffers(
                vec![Node::store("out", Expr::InvocationId { axis: 0 }, Expr::u32(1))],
                "depths",
                vec![BufferDecl::output("out", 1, DataType::U32).with_count(4)],
                4,
                4,
            )
        },
        Some(|| {
            let to_bytes = vyre_primitives::wire::pack_u32_slice;
            vec![
                vec![to_bytes(&[0, 1, 2, 3])],
                vec![to_bytes(&[0, 0, 0, 0])],
            ]
        }),
        Some(|| {
            let to_bytes = vyre_primitives::wire::pack_u32_slice;
            vec![
                vec![to_bytes(&[1, 1, 1, 1])],
                vec![to_bytes(&[1, 1, 1, 1])],
            ]
        }),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use vyre_foundation::visit::any_descendant;

    fn entry_region_body(program: &Program) -> &[Node] {
        match &program.entry()[0] {
            Node::Region { body, .. } => body.as_slice(),
            other => panic!("expected wrapped level-wave region, got {other:?}"),
        }
    }

    /// True when a grid-wide fence appears anywhere under `nodes`.
    ///
    /// Descent comes from `visit::any_descendant`, the one owner of
    /// which node variants nest. The hand-written match this replaces ended in
    /// `_ => false`, so a fifth body-bearing variant would have made a program
    /// with no reachable fence look correctly synchronized.
    fn contains_grid_sync(nodes: &[Node]) -> bool {
        nodes.iter().any(|node| {
            any_descendant(node, &mut |n| {
                matches!(
                    n,
                    Node::Barrier {
                        ordering: MemoryOrdering::GridSync
                    }
                )
            })
        })
    }

    fn contains_loop(nodes: &[Node]) -> bool {
        nodes
            .iter()
            .any(|node| any_descendant(node, &mut |n| matches!(n, Node::Loop { .. })))
    }

    #[test]
    fn cpu_ref_visits_each_lane_at_its_depth() {
        let depths = vec![0u32, 1, 2, 1, 0];
        let mut visits: Vec<(u32, u32)> = Vec::new();
        cpu_ref(&depths, 3, |lane, depth| visits.push((lane, depth)));
        // Every lane visited exactly once, in depth order.
        assert_eq!(visits.len(), depths.len());
        for (idx, &(lane, depth)) in visits.iter().enumerate() {
            assert_eq!(depth, depths[lane as usize]);
            // Visits are sorted by depth (waves).
            if idx > 0 {
                assert!(depth >= visits[idx - 1].1);
            }
        }
    }

    #[test]
    fn dispatch_grid_packs_lane_count_into_workgroups() {
        assert_eq!(level_wave_dispatch_grid(0), [1, 1, 1]);
        assert_eq!(level_wave_dispatch_grid(1), [1, 1, 1]);
        assert_eq!(level_wave_dispatch_grid(256), [1, 1, 1]);
        assert_eq!(level_wave_dispatch_grid(257), [2, 1, 1]);
        assert_eq!(level_wave_dispatch_grid(1029), [5, 1, 1]);
    }

    #[test]
    fn program_shape_matches_contract() {
        let step = vec![Node::store("out", Expr::u32(0), Expr::u32(1))];
        let program = level_wave_program(step, "depths", 8, 64);
        assert_eq!(program.workgroup_size(), LEVEL_WAVE_WORKGROUP_SIZE);
        assert!(
            program.buffers.iter().any(|b| b.name() == "depths"),
            "depth buffer must be declared"
        );
        assert!(!contains_grid_sync(entry_region_body(&program)));
    }

    #[test]
    fn program_with_buffers_declares_depth_then_caller_buffers() {
        let step = vec![Node::store("out", Expr::u32(0), Expr::u32(1))];
        let extra = vec![
            BufferDecl::storage("kinds", 1, BufferAccess::ReadOnly, DataType::U32).with_count(4),
            BufferDecl::storage("out", 2, BufferAccess::ReadWrite, DataType::U32).with_count(4),
        ];
        let program = level_wave_program_with_buffers(step, "depths", extra, 8, 4);
        let names: Vec<&str> = program.buffers.iter().map(|b| b.name()).collect();
        assert_eq!(
            names,
            vec!["depths", "kinds", "out"],
            "depth buffer is bound first (index 0), then the caller's extra buffers in order"
        );
        // The empty-extra delegation path (`level_wave_program`) must declare only the depth buffer.
        let plain = level_wave_program(
            vec![Node::store("out", Expr::u32(0), Expr::u32(1))],
            "depths",
            8,
            4,
        );
        assert_eq!(
            plain.buffers.len(),
            1,
            "plain level-wave declares only depths"
        );
    }

    #[test]
    fn multi_block_program_uses_top_level_grid_sync_waves() {
        let step = vec![Node::store(
            "out",
            Expr::InvocationId { axis: 0 },
            Expr::u32(1),
        )];
        let program = level_wave_program(step, "depths", 4, LEVEL_WAVE_WORKGROUP_SIZE[0] + 1);
        let body = entry_region_body(&program);
        assert!(contains_grid_sync(body));
        assert!(
            !contains_loop(body),
            "multi-block level-wave must expose GridSync at split-visible depth-wave boundaries"
        );
        assert_eq!(
            body.iter()
                .filter(|node| matches!(
                    node,
                    Node::Barrier {
                        ordering: MemoryOrdering::GridSync,
                    }
                ))
                .count(),
            3
        );
    }
    #[test]
    fn registration_witness_cases_and_abi_alignment() {
        use vyre_foundation::operation::OperationRegistration;
        let entry = inventory::iter::<OperationRegistration>
            .into_iter()
            .find(|op| op.id == OP_ID)
            .expect("level_wave must be registered in inventory");

        let test_inputs = (entry.test_inputs.expect("test_inputs must be declared"))();
        let expected_output = (entry
            .expected_output
            .expect("expected_output must be declared"))();

        assert_eq!(
            test_inputs.len(),
            2,
            "level_wave registration must supply exactly 2 witness input cases"
        );
        assert_eq!(
            expected_output.len(),
            2,
            "level_wave registration must supply matching 2 expected output cases (no case-count divergence)"
        );

        let program = (entry.build.expect("build must be declared"))();
        assert_eq!(program.buffers().len(), 2);
        assert_eq!(program.buffers()[0].name(), "depths");
        assert_eq!(program.buffers()[1].name(), "out");
        for (case_idx, (inputs, expected)) in
            test_inputs.iter().zip(expected_output.iter()).enumerate()
        {
            let mut val_inputs: Vec<vyre_reference::value::Value> = inputs
                .iter()
                .cloned()
                .map(vyre_reference::value::Value::from)
                .collect();
            // out buffer is 4 * u32 (16 bytes) ReadWrite buffer
            if val_inputs.len() < program.buffers().len() {
                val_inputs.push(vyre_reference::value::Value::from(vec![0u8; 16]));
            }
            let outputs: Vec<Vec<u8>> = vyre_reference::reference_eval(&program, &val_inputs)
                .expect("reference eval must succeed for level_wave witness")
                .into_iter()
                .map(|val| val.to_bytes())
                .collect();
            assert_eq!(
                outputs, *expected,
                "reference eval must match expected output for case {case_idx}"
            );
        }
    }

    #[test]
    fn callee_before_caller_builds_nonempty_program() {
        let body = vec![Node::barrier()];
        let program = build_callee_before_caller_program(body, "depths", 4, 16);
        assert_ne!(program.entry().len(), 0);
    }

    #[test]
    fn callee_before_caller_zero_depth_still_builds() {
        let body = vec![Node::barrier()];
        let program = build_callee_before_caller_program(body, "depths", 0, 1);
        assert_eq!(program.workgroup_size(), [256, 1, 1]);
        assert!(!program.buffers().is_empty());
    }

    #[test]
    fn callee_before_caller_commits_children_before_parents() {
        use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr};
        use vyre_reference::reference_eval;
        use vyre_reference::value::Value;

        let t = Expr::InvocationId { axis: 0 };
        let step_body = vec![
            Node::let_bind("c", Expr::load("callee", t.clone())),
            Node::store(
                "out",
                t.clone(),
                Expr::add(Expr::u32(1), Expr::load("out", Expr::var("c"))),
            ),
        ];
        let extra_buffers = vec![
            BufferDecl::storage("callee", 1, BufferAccess::ReadOnly, DataType::U32).with_count(4),
            BufferDecl::storage("out", 2, BufferAccess::ReadWrite, DataType::U32).with_count(4),
        ];
        let program = build_callee_before_caller_program_with_buffers(
            step_body,
            "depths",
            extra_buffers,
            4,
            4,
        );

        let pack = |data: &[u32]| Value::from(vyre_primitives::wire::pack_u32_slice(data));
        let inputs = vec![
            pack(&[0, 1, 2, 3]),
            pack(&[0, 0, 1, 2]),
            pack(&[0, 0, 0, 0]),
        ];
        let results = reference_eval(&program, &inputs).expect("Fix: level-wave pass eval failed");
        let out: Vec<u32> = results[0]
            .to_bytes()
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        assert_eq!(out, vec![1, 2, 3, 4]);
    }
}
