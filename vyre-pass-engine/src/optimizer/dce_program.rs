//! GPU DCE program with early-exit on convergence.
//!
//! `vyre_libs::graph::persistent_bfs` always runs its full
//! `max_iters` loop because its `changed` flag is never reset between
//! iterations. For shallow DAGs (most real Programs) the BFS frontier
//! converges in a handful of hops while the kernel keeps churning
//! through hundreds of no-op iterations.
//!
//! This builder emits a DCE-tailored variant: each iteration zeroes
//! `changed` first, runs the CSR-forward step, and the kernel returns
//! as soon as `changed == 0` after a step. For wide DAGs (diameter ≪
//! `max_iters`) this drops the persistent-loop cost from
//! `O(max_iters)` to `O(actual_diameter)`. For chains
//! (`diameter == n`) it matches the original.
//!
//! Buffer + binding layout matches `persistent_bfs` exactly, including
//! the `converged` word, so the handles can be allocated and dispatched
//! the same way and the outputs line up index-for-index.

use std::sync::Arc;

use vyre_foundation::ir::Ident;
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};
use vyre_libs::bitset::bitset_words;
use vyre_libs::graph::persistent_bfs::{
    BINDING_CHANGED, BINDING_CONVERGED, BINDING_FRONTIER_IN, BINDING_FRONTIER_OUT,
};
use vyre_libs::graph::program_graph::{
    ProgramGraphShape, NAME_EDGE_KIND_MASK, NAME_EDGE_OFFSETS, NAME_EDGE_TARGETS,
};

/// Canonical op id for the optimizer's DCE program.
pub const OP_ID: &str = "vyre-pass-engine::optimizer::dce_program";

/// Workgroup size for the DCE BFS kernels.
///
/// Schedule legality requires one cooperative workgroup because the early-exit
/// flag has workgroup scope. The semantic executor's compiler enforces this
/// constraint; pass-engine callers do not select launch geometry.
///
/// One workgroup suffices, because WORKGROUP 0'S LANES ALONE VISIT EVERY SOURCE:
/// the step strides `src = gid_x() + stride * DCE_WORKGROUP_X` for
/// `stride_count = ceil(node_count / DCE_WORKGROUP_X)` iterations, and the seed
/// strides the frontier words the same way, so lanes `0..DCE_WORKGROUP_X` cover
/// the whole node range by themselves at any width. Verified on device, not
/// assumed: a 2000-node chain pinned to one workgroup reaches all 2000.
///
/// More than one workgroup is UNSOUND, and the reason is not the obvious one. The
/// obvious reading is a lost clear: one lane zeroes `changed[0]` with a plain
/// workgroup-scoped store while another group's `atomic_or` is in flight. That
/// alone would be harmless here, because extra groups are pure duplicates of
/// workgroup 0 and losing a duplicate's flag costs nothing. Acting on that reading
/// produced a measured 4-6x pessimization for a defect that was not the live one.
///
/// The live one is EXCLUSIVE DISCOVERY ATTRIBUTION. Growth is detected by whether
/// THIS lane's `atomic_or` actually flipped the bit (`old & dst_bit == 0`), and
/// exactly one lane in the grid wins any given flip. So when a duplicate group wins
/// it, workgroup 0 sees `old` with the bit already set, never raises
/// `local_changed`, and never sets `changed[0]` for a discovery that really
/// happened. Workgroup 0 can then read 0, record a fixpoint it has not reached, and
/// stop relaxing with the newly discovered node's own edges unexpanded. No other
/// group covers the whole node range, so nobody expands them, and the closure is
/// silently truncated. Coverage being redundant does NOT rescue this: coverage is
/// redundant while attribution is exclusive, because a bit is flipped once.
///
/// Note that the flip test is also what makes the traversal efficient, so this is
/// not a wart to remove. Pinning the grid is the fix, and it is the same fix the
/// tree already applies to scalar reductions that initialize their own output
/// (`reduction_metrics.rs`), for the same reason: some programs cannot be split
/// across unsynchronized workgroups.
///
/// The width is the crate's portable workgroup width, which every sibling pass
/// here already dispatches at. It is not a tuning choice either: one shader
/// dialect this program lowers to caps a workgroup at 256 invocations, so a
/// wider width makes the pass undispatchable on that backend rather than
/// faster. Measured before this was 256: every device end-to-end case of this
/// pass was refused on that target with `workgroup_size axis 0 (requested
/// 1024, max 256)`, so the self-hosted DCE pass had never run there at all.
const DCE_WORKGROUP_X: u32 = super::arena_kernel::WORKGROUP_X;

/// Parallel BFS step with per-thread strided loop. Thread
/// `t = gid_x()` handles sources `t, t + WG, t + 2·WG, …` up to
/// `node_count`. Reads `frontier_out[src/32]`'s bit; if set, walks
/// the source's outgoing CSR edges and atomically ORs each target's
/// frontier bit. Sets outer-scope `local_changed` to 1 whenever a
/// NEW bit is added.
///
/// `allow_mask` filters edges: an edge is followed iff
/// `(kind_mask & allow_mask) != 0`. The DCE caller passes
/// `0xFFFF_FFFF` (any-kind), the generic persistent-BFS caller
/// passes the real allow_mask.
fn parallel_csr_step_per_thread_masked(node_count: u32, allow_mask: u32) -> Vec<Node> {
    let stride_count = node_count.div_ceil(DCE_WORKGROUP_X);
    vec![Node::loop_for(
        "stride",
        Expr::u32(0),
        Expr::u32(stride_count.max(1)),
        vec![
            Node::let_bind(
                "src",
                Expr::add(
                    Expr::gid_x(),
                    Expr::mul(Expr::var("stride"), Expr::u32(DCE_WORKGROUP_X)),
                ),
            ),
            Node::if_then(
                Expr::lt(Expr::var("src"), Expr::u32(node_count)),
                vec![
                    Node::let_bind("src_word_idx", Expr::shr(Expr::var("src"), Expr::u32(5))),
                    Node::let_bind(
                        "src_bit_mask",
                        Expr::shl(Expr::u32(1), Expr::bitand(Expr::var("src"), Expr::u32(31))),
                    ),
                    Node::let_bind(
                        "src_word",
                        Expr::load("frontier_out", Expr::var("src_word_idx")),
                    ),
                    Node::if_then(
                        Expr::ne(
                            Expr::bitand(Expr::var("src_word"), Expr::var("src_bit_mask")),
                            Expr::u32(0),
                        ),
                        vec![
                            Node::let_bind(
                                "edge_start",
                                Expr::load(NAME_EDGE_OFFSETS, Expr::var("src")),
                            ),
                            Node::let_bind(
                                "edge_end",
                                Expr::load(
                                    NAME_EDGE_OFFSETS,
                                    Expr::add(Expr::var("src"), Expr::u32(1)),
                                ),
                            ),
                            Node::loop_for(
                                "e",
                                Expr::var("edge_start"),
                                Expr::var("edge_end"),
                                vec![
                                    Node::let_bind(
                                        "kind_mask",
                                        Expr::load(NAME_EDGE_KIND_MASK, Expr::var("e")),
                                    ),
                                    Node::if_then(
                                        Expr::ne(
                                            Expr::bitand(
                                                Expr::var("kind_mask"),
                                                Expr::u32(allow_mask),
                                            ),
                                            Expr::u32(0),
                                        ),
                                        vec![
                                            Node::let_bind(
                                                "dst",
                                                Expr::load(NAME_EDGE_TARGETS, Expr::var("e")),
                                            ),
                                            Node::if_then(
                                                Expr::lt(Expr::var("dst"), Expr::u32(node_count)),
                                                vec![
                                                    Node::let_bind(
                                                        "dst_word_idx",
                                                        Expr::shr(Expr::var("dst"), Expr::u32(5)),
                                                    ),
                                                    Node::let_bind(
                                                        "dst_bit",
                                                        Expr::shl(
                                                            Expr::u32(1),
                                                            Expr::bitand(
                                                                Expr::var("dst"),
                                                                Expr::u32(31),
                                                            ),
                                                        ),
                                                    ),
                                                    Node::let_bind(
                                                        "old",
                                                        Expr::atomic_or(
                                                            "frontier_out",
                                                            Expr::var("dst_word_idx"),
                                                            Expr::var("dst_bit"),
                                                        ),
                                                    ),
                                                    Node::if_then(
                                                        Expr::eq(
                                                            Expr::bitand(
                                                                Expr::var("old"),
                                                                Expr::var("dst_bit"),
                                                            ),
                                                            Expr::u32(0),
                                                        ),
                                                        vec![Node::assign(
                                                            "local_changed",
                                                            Expr::u32(1),
                                                        )],
                                                    ),
                                                ],
                                            ),
                                        ],
                                    ),
                                ],
                            ),
                        ],
                    ),
                ],
            ),
        ],
    )]
}

/// Build a generic persistent-BFS Program with early-exit and a
/// caller-supplied `allow_mask` for edge filtering.
///
/// Identical buffer layout to `build_dce_bfs_program`; differs only
/// in that the edge-follow check is `(kind_mask & allow_mask) != 0`
/// instead of `kind_mask != 0`. Use this when porting
/// `vyre_libs::graph::persistent_bfs::cpu_ref` to GPU dispatch.
///
/// DISPATCH THIS AS EXACTLY ONE WORKGROUP, the same requirement
/// `build_dce_bfs_program` carries, because it is the same kernel with a
/// different edge filter. See `DCE_WORKGROUP_X`: growth is attributed to the
/// single lane whose `atomic_or` flipped the bit, so across workgroups a
/// duplicate group can win a discovery and leave the group that covers the whole
/// node range reading `changed == 0`, which reports a fixpoint it never reached
/// and truncates the closure. One workgroup's strided lanes already visit every
/// source, so pinning costs no coverage.
#[must_use]
pub fn build_persistent_bfs_program(
    shape: ProgramGraphShape,
    max_iters: u32,
    allow_mask: u32,
) -> Program {
    build_persistent_bfs_program_sticky(shape, max_iters, allow_mask)
}

/// Build a DCE-tailored persistent BFS Program with early-exit.
///
/// Identical buffer layout to `persistent_bfs` (frontier_in,
/// frontier_out, changed, converged, plus the program-graph CSR buffers
/// from `shape.read_only_buffers()`). The kernel:
///
///  1. Seeds `frontier_out` from `frontier_in` and clears `converged`.
///  2. Runs up to `max_iters` BFS steps. Each step:
///     a. Lane 0 zeros `changed[0]`.
///     b. Workgroup barrier.
///     c. CSR forward step; if any node grew its frontier bit, it
///     does `atomic_or(changed, 0, 1)`.
///     d. Workgroup barrier.
///     e. If `changed[0] == 0`, set `converged[0] = 1` and return (no
///     progress this iter ⇒ fixpoint reached; subsequent iters are
///     no-ops).
///  3. Final state lives in `frontier_out`. `converged[0]` is 1 when the
///     loop exited early on a fixpoint and 0 when it burned every
///     iteration while still growing, which means `frontier_out` is a
///     partial closure and must not be trusted as a reachability set.
#[must_use]
pub fn build_dce_bfs_program(shape: ProgramGraphShape, max_iters: u32) -> Program {
    build_persistent_bfs_program_inner(shape, max_iters, u32::MAX)
}

/// Shared implementation for `build_dce_bfs_program` (allow_mask =
/// `u32::MAX`, sticky_changed=false) and `build_persistent_bfs_program`
/// (caller-supplied allow_mask, sticky_changed=true).
///
/// `sticky_changed` controls the semantics of `changed[0]`:
///  - `false` (DCE): `changed[0]` reflects the LAST iter's progress
///    (the kernel zeroes it each iter for early-exit detection). DCE
///    doesn't observe the post-kernel value so this is fine.
///  - `true` (generic persistent BFS): `changed[0]` is sticky-OR'd
///    across all iterations, matching the CPU oracle's contract.
///    The kernel uses an internal scratch slot for the per-iter flag.
fn build_persistent_bfs_program_inner(
    shape: ProgramGraphShape,
    max_iters: u32,
    allow_mask: u32,
) -> Program {
    build_persistent_bfs_program_internal(shape, max_iters, allow_mask, false)
}

fn build_persistent_bfs_program_sticky(
    shape: ProgramGraphShape,
    max_iters: u32,
    allow_mask: u32,
) -> Program {
    build_persistent_bfs_program_internal(shape, max_iters, allow_mask, true)
}

fn build_persistent_bfs_program_internal(
    shape: ProgramGraphShape,
    max_iters: u32,
    allow_mask: u32,
    sticky_changed: bool,
) -> Program {
    let words = bitset_words(shape.node_count);
    let t = Expr::gid_x();

    // For sticky-changed mode, slot 0 = per-iter (zeroed each iter,
    // used for early-exit) and slot 1 = cumulative (sticky-OR'd
    // across all iters, never zeroed). Caller reads slot 1.
    // For DCE mode, only slot 0 is used.
    let mut iter_body: Vec<Node> = vec![
        // Zero `changed[0]` so this iteration's compare starts clean.
        Node::if_then(
            Expr::eq(t.clone(), Expr::u32(0)),
            vec![Node::store("changed", Expr::u32(0), Expr::u32(0))],
        ),
        Node::barrier(),
        Node::let_bind("local_changed", Expr::u32(0)),
        // The `converged` gate is what makes the early exit CHEAP, and it is not
        // redundant with the `Node::Return` below.
        //
        // Read this before changing either one. `Return` nested inside a
        // `Node::Loop` USED to be emitted as nothing by a machine-code emitter, and
        // an earlier version of this comment said so. That is no longer true: the
        // emitter now lowers a nested `Return` to a real exit branch, so the exit
        // is live on device and not only in the reference interpreter. Two
        // consequences, and both matter here.
        //
        // First, the gate stays. It gates the WORK, which the `Return` does not:
        // once the fixpoint is recorded, every later iteration skips the edge walk
        // entirely. Measured on device before the gate existed: a 2000-node
        // star that reaches its fixpoint in 2 iterations cost 2450 ms at a 2000
        // budget against 13 ms at an 8 budget, 183x, because one lane re-walked the
        // hub's 1999 edges 2000 times. The answer was right every time, which is why
        // nothing caught it. That cost is a property of the WALK, so it returns if
        // the gate is removed, whatever `Return` lowers to.
        //
        // Second, a live exit needs a collective proof. The barrier immediately
        // before the exit settles `changed[0]`; every lane then evaluates the
        // same address and returns together or takes the back edge together.
        // V055 derives that uniformity and rejects the lane-dependent twin.
        //
        // The read is deliberately racy and MUST stay benign. `converged` is written
        // by lane 0 with no fence before this read, so a lane may see a stale 0 and
        // do one more pass; that pass adds no bit, because growth is monotone and
        // the fixpoint was already reached. What must NEVER happen is a barrier
        // inside this gate: the barriers stay unconditional at iteration-body level,
        // so a lane reading a stale flag cannot desynchronize the workgroup.
        Node::if_then(
            Expr::eq(Expr::load("converged", Expr::u32(0)), Expr::u32(0)),
            vec![Node::if_then(
                Expr::lt(t.clone(), Expr::u32(shape.node_count)),
                parallel_csr_step_per_thread_masked(shape.node_count, allow_mask),
            )],
        ),
        // OR local_changed into the per-iter early-exit flag.
        Node::if_then(
            Expr::eq(Expr::var("local_changed"), Expr::u32(1)),
            vec![Node::let_bind(
                "_dce_set",
                Expr::atomic_or("changed", Expr::u32(0), Expr::u32(1)),
            )],
        ),
    ];
    if sticky_changed {
        // Mirror the OR into slot 1 (cumulative). slot 1 is never
        // zeroed, so once any iter sets it, it stays 1.
        iter_body.push(Node::if_then(
            Expr::eq(Expr::var("local_changed"), Expr::u32(1)),
            vec![Node::let_bind(
                "_sticky_set",
                Expr::atomic_or("changed", Expr::u32(1), Expr::u32(1)),
            )],
        ));
    }
    iter_body.push(Node::barrier());
    // Early-exit on per-iter fixpoint. Reaching this branch IS convergence: the
    // step added nothing, and growth is monotone, so every later step would add
    // nothing too. Record it before returning so the host can tell a real
    // fixpoint apart from a loop that burned its whole `max_iters` budget while
    // still growing. Without this the two are indistinguishable and a caller
    // silently reasons over a partial closure (Law 10).
    iter_body.push(Node::if_then(
        Expr::eq(Expr::load("changed", Expr::u32(0)), Expr::u32(0)),
        vec![
            Node::if_then(
                Expr::eq(t.clone(), Expr::u32(0)),
                vec![Node::store("converged", Expr::u32(0), Expr::u32(1))],
            ),
            Node::Return,
        ],
    ));

    let entry: Vec<Node> = vec![
        // Seed frontier_out <- frontier_in, strided like the step: a graph with
        // more frontier words than the workgroup has lanes must still be seeded
        // whole, and one lane per word only covers `words <= DCE_WORKGROUP_X`.
        // A truncated seed starts the traversal from a partial frontier and
        // reports a fixpoint over a closure it never reached.
        Node::loop_for(
            "seed_stride",
            Expr::u32(0),
            Expr::u32(words.div_ceil(DCE_WORKGROUP_X).max(1)),
            vec![
                Node::let_bind(
                    "seed_word",
                    Expr::add(
                        t.clone(),
                        Expr::mul(Expr::var("seed_stride"), Expr::u32(DCE_WORKGROUP_X)),
                    ),
                ),
                Node::if_then(
                    Expr::lt(Expr::var("seed_word"), Expr::u32(words)),
                    vec![Node::store(
                        "frontier_out",
                        Expr::var("seed_word"),
                        Expr::load("frontier_in", Expr::var("seed_word")),
                    )],
                ),
            ],
        ),
        // `converged` starts at 0 so a kernel that runs every iteration without
        // reaching a fixpoint leaves it 0. Only the early-exit branch sets it.
        Node::if_then(
            Expr::eq(t.clone(), Expr::u32(0)),
            vec![Node::store("converged", Expr::u32(0), Expr::u32(0))],
        ),
        // Workgroup-scoped, deliberately, and NOT an oversight. The seeding lanes
        // stride within their own workgroup and so all live in workgroup 0, which is
        // also the only workgroup the answer depends on: see `DCE_WORKGROUP_X` for
        // why every other workgroup is a pure duplicate. A non-zero workgroup that
        // enters the traversal reading an unseeded `frontier_out` expands nothing
        // and loses nothing, because it owns no unique work. Escalating this to
        // `MemoryOrdering::GridSync` would order that harmless case at the price of
        // forcing a cooperative launch on every dispatch above one workgroup, which
        // is a real constraint on launch geometry and portability. Measured on
        // device, that fence bought no closure this form does not already reach.
        Node::barrier(),
        // Persistent loop with early-exit.
        Node::loop_for("iter", Expr::u32(0), Expr::u32(max_iters.max(1)), iter_body),
    ];

    let mut buffers = shape.read_only_buffers();
    buffers.push(
        BufferDecl::storage(
            "frontier_in",
            BINDING_FRONTIER_IN,
            BufferAccess::ReadOnly,
            DataType::U32,
        )
        .with_count(words.max(1)),
    );
    buffers.push(
        BufferDecl::storage(
            "frontier_out",
            BINDING_FRONTIER_OUT,
            BufferAccess::ReadWrite,
            DataType::U32,
        )
        .with_count(words.max(1)),
    );
    buffers.push(
        BufferDecl::storage(
            "changed",
            BINDING_CHANGED,
            BufferAccess::ReadWrite,
            DataType::U32,
        )
        .with_count(if sticky_changed { 2 } else { 1 }),
    );
    buffers.push(
        BufferDecl::storage(
            "converged",
            BINDING_CONVERGED,
            BufferAccess::ReadWrite,
            DataType::U32,
        )
        .with_count(1),
    );
    buffers.push(BufferDecl::workgroup(
        "wg_scratch",
        DCE_WORKGROUP_X,
        DataType::U32,
    ));

    // The emitted width MUST be `DCE_WORKGROUP_X`, not a literal that merely
    // matches it today. The step stride, the seed stride and the workgroup
    // scratch are all built from that same const, and the coverage invariant
    // holds only while all of them agree. Pin the width to a literal and raise
    // the const, and the strides start skipping whole lane ranges the emitted
    // launch never has, so the traversal returns a truncated closure.
    Program::wrapped(
        buffers,
        [DCE_WORKGROUP_X, 1, 1],
        vec![Node::Region {
            generator: Ident::from(OP_ID),
            source_region: None,
            body: Arc::new(entry),
        }],
    )
}
