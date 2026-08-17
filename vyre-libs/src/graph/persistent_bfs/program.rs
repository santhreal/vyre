use vyre_foundation::composition::wrap_anonymous_region;

use super::layout::{PersistentBfsBuffers, BATCH_OP_ID, OP_ID, PERSISTENT_BFS_WORKGROUP_SIZE};
use crate::bitset::bitset_words;
use crate::fixpoint::persistent_fixpoint::grid_sync_barrier;
use crate::graph::csr_forward_or_changed::csr_forward_or_changed_parallel_snapshot_child_prefixed_with_active;
use crate::graph::frontier_bits::bind_bit_address;
use crate::graph::persistent_bfs_step::persistent_bfs_step_child_prefixed_with_active;
use crate::graph::program_graph::ProgramGraphShape;
use vyre_foundation::ir::{BufferDecl, DataType, Expr, Node, Program};

/// Build the IR `Program` for persistent BFS.
///
/// The kernel copies `frontier_in` into `frontier_out`, then performs up
/// to `max_iters` forward traversal steps.  The first four iterations are
/// unrolled with inter-step workgroup barriers and a shared `wg_scratch`
/// array; any additional iterations run in a plain bounded loop.
///
/// `changed` is a single u32 word that is set to `1` if *any* step produced
/// a new reachable node.
///
/// `converged` is a single u32 word set to `1` if the frontier reached a
/// fixpoint (a step added nothing) before the `max_iters` budget was exhausted,
/// and `0` if the loop ran all `max_iters` steps while still growing (a partial
/// closure) or `max_iters == 0`. It is the device counterpart of the
/// `vyre-reference` persistent BFS witness and lets a
/// host caller reject an under-approximated frontier loudly instead of silently
/// trusting a closure the kernel never drove to a fixpoint.
#[must_use]
pub fn persistent_bfs(
    shape: ProgramGraphShape,
    frontier_in: &str,
    frontier_out: &str,
    edge_kind_mask: u32,
    max_iters: u32,
) -> Program {
    if shape.node_count > PERSISTENT_BFS_WORKGROUP_SIZE[0] {
        return persistent_bfs_grid_sync_parallel(
            shape,
            frontier_in,
            frontier_out,
            edge_kind_mask,
            max_iters,
            None,
        );
    }
    persistent_bfs_single_workgroup(
        shape,
        frontier_in,
        frontier_out,
        edge_kind_mask,
        max_iters,
        None,
    )
}

/// Build a density-instrumented persistent-BFS `Program`.
///
/// Identical to [`persistent_bfs`] but declares one extra `max_iters`-length u32
/// output buffer, `density_active`, whose entry `i` holds the popcount of the
/// frontier after traversal step `i` (flat once the closure converges, since
/// growth is monotone). A host caller reconstructs every
/// per-iteration frontier-density aggregate from this array plus the seed
/// popcount without a per-step device round-trip. The base [`persistent_bfs`]
/// program is byte-for-byte unchanged, so callers that do not want telemetry
/// (and every other primitive consumer) pay nothing.
#[must_use]
pub fn persistent_bfs_with_density(
    shape: ProgramGraphShape,
    frontier_in: &str,
    frontier_out: &str,
    density_active: &str,
    edge_kind_mask: u32,
    max_iters: u32,
) -> Program {
    if shape.node_count > PERSISTENT_BFS_WORKGROUP_SIZE[0] {
        return persistent_bfs_grid_sync_parallel(
            shape,
            frontier_in,
            frontier_out,
            edge_kind_mask,
            max_iters,
            Some(density_active),
        );
    }
    persistent_bfs_single_workgroup(
        shape,
        frontier_in,
        frontier_out,
        edge_kind_mask,
        max_iters,
        Some(density_active),
    )
}

/// Build a build-time-unrolled popcount of the whole `words`-word frontier
/// bitset (leader lane emits this in the single-workgroup path, where
/// `words <= 8`). Returns the summed set-bit count as one `Expr`.
fn frontier_popcount_expr(frontier_out: &str, words: u32) -> Expr {
    let mut sum = Expr::popcount(Expr::load(frontier_out, Expr::u32(0)));
    for word in 1..words {
        sum = Expr::add(
            sum,
            Expr::popcount(Expr::load(frontier_out, Expr::u32(word))),
        );
    }
    sum
}

fn persistent_bfs_single_workgroup(
    shape: ProgramGraphShape,
    frontier_in: &str,
    frontier_out: &str,
    edge_kind_mask: u32,
    max_iters: u32,
    density_active: Option<&str>,
) -> Program {
    let words = bitset_words(shape.node_count);
    let t = Expr::gid_x();

    let unrolled_iter = |iter: u32| -> Node {
        persistent_bfs_step_child_prefixed_with_active(
            OP_ID,
            shape,
            frontier_out,
            "changed",
            "wg_scratch",
            "wg_active",
            edge_kind_mask,
            &format!("unroll_{iter}"),
        )
    };

    let mut entry: Vec<Node> = vec![
        // Seed frontier_out from frontier_in. Gate on the GLOBAL leader `gid_x()==0`,
        // NOT the per-workgroup leader `local_x()==0`: this is a single-workgroup
        // kernel (node_count <= workgroup size; >256 goes to the grid-sync variant),
        // so its intended dispatch is exactly one workgroup. If a caller over-dispatches
        // it, or a driver rounds up to more than one workgroup, every EXTRA
        // workgroup's `local_x()==0` leader would otherwise RE-SEED `frontier_out` back
        // to `frontier_in`, clobbering the first workgroup's already-expanded frontier
        // (the interpreter runs workgroups in order, so the last re-seed wins → the
        // output collapses to the unexpanded seed). Guarding on `gid_x()==0` makes only
        // the first workgroup seed; extra workgroups are inert (their unrolled steps are
        // already no-ops because `wg_active` is only set to 1 under `gid_x()==0`), so the
        // result is invariant to over-fire (whole-workgroup GPU dispatch never corrupts
        // it). Transparent for the intended single-workgroup dispatch, where
        // `gid_x()==0` and `local_x()==0` are the same lane.
        Node::if_then(
            Expr::eq(t.clone(), Expr::u32(0)),
            vec![Node::loop_for(
                "seed_word_idx",
                Expr::u32(0),
                Expr::u32(words),
                vec![Node::store(
                    frontier_out,
                    Expr::var("seed_word_idx"),
                    Expr::load(frontier_in, Expr::var("seed_word_idx")),
                )],
            )],
        ),
        // Zero the global changed flag.
        Node::if_then(
            Expr::eq(t.clone(), Expr::u32(0)),
            vec![
                Node::store("changed", Expr::u32(0), Expr::u32(0)),
                Node::store("wg_active", Expr::u32(0), Expr::u32(1)),
            ],
        ),
        // Barrier clears fusion hazards from the plain store above before the
        // first atomic access inside the unrolled steps.
        Node::barrier(),
    ];

    // Record the frontier popcount after unrolled step `iter` into
    // `density_active[iter]`. `Node::barrier()` makes every lane's in-place
    // writes from the step visible before the leader popcounts the whole
    // bitset; the barrier and store are emitted ONLY for the density variant,
    // so the base program's IR is unchanged.
    let record_density_after = |step_index: Expr| -> Vec<Node> {
        match density_active {
            Some(density) => vec![
                Node::barrier(),
                Node::if_then(
                    Expr::eq(t.clone(), Expr::u32(0)),
                    vec![Node::store(
                        density,
                        step_index,
                        frontier_popcount_expr(frontier_out, words),
                    )],
                ),
            ],
            None => Vec::new(),
        }
    };

    let unroll_count = max_iters.min(4);
    for iter in 0..unroll_count {
        entry.push(unrolled_iter(iter));
        entry.extend(record_density_after(Expr::u32(iter)));
    }

    let remaining = max_iters.saturating_sub(unroll_count);
    if remaining > 0 {
        let mut loop_body = vec![Node::if_then(
            Expr::ne(Expr::load("wg_active", Expr::u32(0)), Expr::u32(0)),
            vec![
                Node::let_bind("local_changed", Expr::u32(0)),
                Node::if_then(
                    Expr::lt(t.clone(), Expr::u32(shape.node_count)),
                    vec![
                        crate::graph::csr_forward_or_changed::csr_forward_or_changed_child_prefixed(
                            OP_ID,
                            shape,
                            frontier_out,
                            "local_changed",
                            edge_kind_mask,
                            "remaining_csr",
                        ),
                    ],
                ),
                Node::if_then(
                    Expr::eq(t.clone(), Expr::u32(0)),
                    vec![Node::store(
                        "wg_active",
                        Expr::u32(0),
                        Expr::var("local_changed"),
                    )],
                ),
                Node::if_then(
                    Expr::eq(Expr::var("local_changed"), Expr::u32(1)),
                    vec![Node::let_bind(
                        "_",
                        Expr::atomic_or("changed", Expr::u32(0), Expr::u32(1)),
                    )],
                ),
            ],
        )];
        // Record density AFTER the guarded step (unconditionally each loop
        // iteration): when the step is skipped past convergence, `frontier_out`
        // is unchanged, so the flat popcount correctly repeats the converged
        // value. The loop index maps to global step `unroll_count + iter`.
        loop_body.extend(record_density_after(Expr::add(
            Expr::u32(unroll_count),
            Expr::var("iter"),
        )));
        entry.push(Node::loop_for(
            "iter",
            Expr::u32(0),
            Expr::u32(remaining),
            loop_body,
        ));
    }

    // Publish the converged flag from the workgroup leader after the loop.
    // `wg_active` is the monotone workgroup-wide "did the last step add a node"
    // signal: it starts at 1, is rewritten to each step's any-changed reduction,
    // and once a step adds nothing it stays 0 for every later step (reachability
    // growth is monotone). So `wg_active == 0` at the end means a no-change step
    // was observed within budget (converged); `wg_active != 0` means the loop
    // exhausted `max_iters` while still growing, or `max_iters == 0` (in which
    // case `wg_active` is still its seed value 1). The leader that reads it here
    // (`gid_x()==0`) is the same lane that wrote it last, so no barrier is needed.
    entry.push(Node::if_then(
        Expr::eq(t.clone(), Expr::u32(0)),
        vec![Node::store(
            "converged",
            Expr::u32(0),
            Expr::select(
                Expr::eq(Expr::load("wg_active", Expr::u32(0)), Expr::u32(0)),
                Expr::u32(1),
                Expr::u32(0),
            ),
        )],
    ));

    let mut buffers = shape.read_only_buffers();
    PersistentBfsBuffers {
        frontier_in,
        frontier_out,
        frontier_words: words,
        changed: ("changed", 1),
        converged: ("converged", 1),
        density_active: density_active.map(|density| (density, max_iters.max(1))),
    }
    .push_onto(&mut buffers);
    buffers.push(BufferDecl::workgroup("wg_scratch", 256, DataType::U32));
    buffers.push(BufferDecl::workgroup("wg_active", 1, DataType::U32));

    Program::wrapped(
        buffers,
        PERSISTENT_BFS_WORKGROUP_SIZE,
        vec![wrap_anonymous_region(OP_ID, entry)],
    )
}

fn persistent_bfs_grid_sync_parallel(
    shape: ProgramGraphShape,
    frontier_in: &str,
    frontier_out: &str,
    edge_kind_mask: u32,
    max_iters: u32,
    density_active: Option<&str>,
) -> Program {
    let words = bitset_words(shape.node_count);
    let t = Expr::gid_x();
    const GRID_CHANGED_WORDS: u32 = 3;
    const GRID_ACTIVE_BASE: u32 = 1;
    let mut entry: Vec<Node> = vec![
        Node::if_then(
            Expr::lt(t.clone(), Expr::u32(words)),
            vec![Node::store(
                frontier_out,
                t.clone(),
                Expr::load(frontier_in, t.clone()),
            )],
        ),
        Node::if_then(
            Expr::eq(t.clone(), Expr::u32(0)),
            if max_iters > 0 {
                vec![
                    Node::store("changed", Expr::u32(0), Expr::u32(0)),
                    Node::store("changed", Expr::u32(GRID_ACTIVE_BASE), Expr::u32(1)),
                    Node::store("changed", Expr::u32(GRID_ACTIVE_BASE + 1), Expr::u32(0)),
                ]
            } else {
                vec![Node::store("changed", Expr::u32(0), Expr::u32(0))]
            },
        ),
    ];

    if max_iters > 0 {
        entry.push(grid_sync_barrier());
    }
    for iter in 0..max_iters {
        let active_index = GRID_ACTIVE_BASE + (iter & 1);
        let next_active_index = GRID_ACTIVE_BASE + ((iter + 1) & 1);
        entry.push(Node::if_then(
            Expr::eq(t.clone(), Expr::u32(0)),
            vec![Node::store(
                "changed",
                Expr::u32(next_active_index),
                Expr::u32(0),
            )],
        ));
        entry.push(
            csr_forward_or_changed_parallel_snapshot_child_prefixed_with_active(
                OP_ID,
                shape,
                frontier_out,
                "changed",
                Expr::load("changed", Expr::u32(active_index)),
                Expr::u32(next_active_index),
                edge_kind_mask,
                &format!("grid_iter_{iter}"),
            ),
        );
        match density_active {
            // Record `density_active[iter] = popcount(frontier_out)` after this
            // step. The global leader (`gid_x()==0`) serially sums the popcount of
            // every frontier word and stores the total. This is a plain,
            // IDEMPOTENT write (like the `converged` publish): the grid-sync split
            // dispatch re-executes segments to a fixpoint, so an accumulating
            // atomic-add would double-count or land on a zeroed slot, but a
            // recompute-and-store lands the same value on every re-run. Two
            // grid-sync barriers isolate it: one after the step so every
            // workgroup's `frontier_out` write is globally visible before the
            // leader reads it, one after so the leader's reads retire before the
            // next step overwrites `frontier_out`. Emitted only for the density
            // variant, so the base program keeps its single inter-iteration
            // barrier and its ABI.
            Some(density) => {
                entry.push(grid_sync_barrier());
                entry.push(Node::if_then(
                    Expr::eq(t.clone(), Expr::u32(0)),
                    vec![
                        Node::store(density, Expr::u32(iter), Expr::u32(0)),
                        Node::loop_for(
                            &format!("density_word_{iter}"),
                            Expr::u32(0),
                            Expr::u32(words),
                            vec![Node::store(
                                density,
                                Expr::u32(iter),
                                Expr::add(
                                    Expr::load(density, Expr::u32(iter)),
                                    Expr::popcount(Expr::load(
                                        frontier_out,
                                        Expr::var(&format!("density_word_{iter}")),
                                    )),
                                ),
                            )],
                        ),
                    ],
                ));
                entry.push(grid_sync_barrier());
            }
            None => {
                if iter + 1 < max_iters {
                    entry.push(grid_sync_barrier());
                }
            }
        }
    }

    // Publish the converged flag. The ping-ponged active words at
    // `GRID_ACTIVE_BASE + (iter & 1)` each hold "did step `iter` add a node";
    // after the loop, `GRID_ACTIVE_BASE + (max_iters & 1)` is the last step's
    // flag (the `next_active_index` written by the final iteration). A trailing
    // grid-sync barrier makes every workgroup's writes to that flag visible
    // before the global leader reads it. The flag is monotone: once a step adds
    // nothing it stays 0, so `== 0` means a no-change step was reached within
    // budget (converged). With `max_iters == 0` no step runs and the active
    // words are never seeded, so converged is written 0 directly.
    if max_iters > 0 {
        entry.push(grid_sync_barrier());
        let final_active_index = GRID_ACTIVE_BASE + (max_iters & 1);
        entry.push(Node::if_then(
            Expr::eq(t.clone(), Expr::u32(0)),
            vec![Node::store(
                "converged",
                Expr::u32(0),
                Expr::select(
                    Expr::eq(
                        Expr::load("changed", Expr::u32(final_active_index)),
                        Expr::u32(0),
                    ),
                    Expr::u32(1),
                    Expr::u32(0),
                ),
            )],
        ));
    } else {
        entry.push(Node::if_then(
            Expr::eq(t.clone(), Expr::u32(0)),
            vec![Node::store("converged", Expr::u32(0), Expr::u32(0))],
        ));
    }

    let mut buffers = shape.read_only_buffers();
    PersistentBfsBuffers {
        frontier_in,
        frontier_out,
        frontier_words: words,
        changed: (
            "changed",
            if max_iters > 0 { GRID_CHANGED_WORDS } else { 1 },
        ),
        converged: ("converged", 1),
        density_active: density_active.map(|density| (density, max_iters.max(1))),
    }
    .push_onto(&mut buffers);

    Program::wrapped(
        buffers,
        PERSISTENT_BFS_WORKGROUP_SIZE,
        vec![wrap_anonymous_region(OP_ID, entry)],
    )
}

/// Build a batched persistent-BFS Program.
///
/// Frontier buffers are flat `[query][word]` arrays. The launch topology uses
/// `grid.y` for the query and `grid.x` for source-node lanes inside that query.
/// Each expansion pass snapshots active source bits before any lane writes new
/// destination bits, preserving the CPU oracle's one-hop-per-iteration cap.
///
/// # Panics
/// Panics on an invalid flat-frontier shape. Callers that must recover use
/// [`try_persistent_bfs_batch`].
#[must_use]
pub fn persistent_bfs_batch(
    shape: ProgramGraphShape,
    frontier_in: &str,
    frontier_out: &str,
    changed: &str,
    converged: &str,
    query_count: u32,
    edge_kind_mask: u32,
    max_iters: u32,
) -> Program {
    // Fail fast on an invalid flat-frontier shape rather than silently degrading
    // to an inert empty kernel (silent recall loss). Use
    // `try_persistent_bfs_batch` for structured handling.
    try_persistent_bfs_batch(
        shape,
        frontier_in,
        frontier_out,
        changed,
        converged,
        query_count,
        edge_kind_mask,
        max_iters,
    )
    .unwrap_or_else(|error| panic!("{error}"))
}

/// Build a batched persistent-BFS Program with checked flat-frontier sizing.
///
/// `changed` is a per-query u32 array: `changed[q]` is a sticky OR set to `1` if
/// *any* step grew query `q`'s frontier. `converged` is a per-query u32 array:
/// `converged[q]` is `1` iff query `q` reached a fixpoint (a step added nothing)
/// before the `max_iters` budget was exhausted, and `0` if the loop ran all
/// `max_iters` steps while still growing (a partial closure) or `max_iters == 0`.
/// Growth is monotone, so once a step adds nothing every later step adds nothing
/// too; `converged[q]` is therefore exactly "the last iteration added nothing for
/// query `q`". It is the per-query batch counterpart of the single-query
/// [`persistent_bfs`] converged word and lets a host caller reject an
/// under-approximated frontier loudly instead of silently trusting a closure the
/// kernel never drove to a fixpoint.
pub fn try_persistent_bfs_batch(
    shape: ProgramGraphShape,
    frontier_in: &str,
    frontier_out: &str,
    changed: &str,
    converged: &str,
    query_count: u32,
    edge_kind_mask: u32,
    max_iters: u32,
) -> Result<Program, String> {
    try_persistent_bfs_batch_inner(
        shape,
        frontier_in,
        frontier_out,
        changed,
        converged,
        None,
        query_count,
        edge_kind_mask,
        max_iters,
    )
}

/// Build a density-instrumented batched persistent-BFS Program, panicking on an
/// invalid flat-frontier shape.
///
/// The panicking counterpart of [`try_persistent_bfs_batch_with_density`], for
/// callers that stage a statically valid batch. Use the `try_` form for
/// structured error handling.
///
/// # Panics
/// Panics on an invalid flat-frontier shape. Callers that must recover use
/// [`try_persistent_bfs_batch_with_density`].
#[must_use]
pub fn persistent_bfs_batch_with_density(
    shape: ProgramGraphShape,
    frontier_in: &str,
    frontier_out: &str,
    changed: &str,
    converged: &str,
    density_active: &str,
    query_count: u32,
    edge_kind_mask: u32,
    max_iters: u32,
) -> Program {
    try_persistent_bfs_batch_with_density(
        shape,
        frontier_in,
        frontier_out,
        changed,
        converged,
        density_active,
        query_count,
        edge_kind_mask,
        max_iters,
    )
    .unwrap_or_else(|error| panic!("{error}"))
}

/// Build a density-instrumented batched persistent-BFS Program.
///
/// Identical to [`try_persistent_bfs_batch`] but declares one extra
/// `query_count * max_iters` u32 output buffer, `density_active`, laid out
/// per-query: entry `q * max_iters + i` holds the popcount of query `q`'s
/// frontier after traversal step `i` (flat once that query converges, since
/// growth is monotone). A host reconstructs every per-query frontier-density
/// aggregate from this array plus the per-query seed popcount without a per-step
/// device round-trip. The base [`try_persistent_bfs_batch`] program is
/// byte-for-byte unchanged, so callers that do not want telemetry pay nothing.
pub fn try_persistent_bfs_batch_with_density(
    shape: ProgramGraphShape,
    frontier_in: &str,
    frontier_out: &str,
    changed: &str,
    converged: &str,
    density_active: &str,
    query_count: u32,
    edge_kind_mask: u32,
    max_iters: u32,
) -> Result<Program, String> {
    try_persistent_bfs_batch_inner(
        shape,
        frontier_in,
        frontier_out,
        changed,
        converged,
        Some(density_active),
        query_count,
        edge_kind_mask,
        max_iters,
    )
}

fn try_persistent_bfs_batch_inner(
    shape: ProgramGraphShape,
    frontier_in: &str,
    frontier_out: &str,
    changed: &str,
    converged: &str,
    density_active: Option<&str>,
    query_count: u32,
    edge_kind_mask: u32,
    max_iters: u32,
) -> Result<Program, String> {
    let words = bitset_words(shape.node_count).max(1);
    let total_words = checked_batch_frontier_words(words, query_count, BATCH_OP_ID)?;
    let q = Expr::gid_y();
    let base = Expr::mul(q.clone(), Expr::u32(words));
    let lane = Expr::gid_x();
    let uses_grid_sync = persistent_bfs_batch_needs_grid_sync(shape);

    let mut entry: Vec<Node> = vec![
        Node::if_then(
            Expr::lt(lane.clone(), Expr::u32(words)),
            vec![Node::store(
                frontier_out,
                Expr::add(base.clone(), lane.clone()),
                Expr::load(frontier_in, Expr::add(base.clone(), lane.clone())),
            )],
        ),
        Node::if_then(
            Expr::eq(lane, Expr::u32(0)),
            vec![
                Node::store(changed, q.clone(), Expr::u32(0)),
                // Init converged[q] = 0 (not converged). When max_iters == 0 no
                // step runs and this seed value stands, matching the single-query
                // "max_iters == 0 -> converged 0" contract. When max_iters > 0 the
                // per-step reset below overwrites it and the trailing publish sets
                // the real value.
                Node::store(converged, q.clone(), Expr::u32(0)),
            ],
        ),
    ];

    if max_iters > 0 {
        entry.push(persistent_bfs_batch_sync(uses_grid_sync));
    }
    if uses_grid_sync {
        for iter in 0..max_iters {
            entry.extend(persistent_bfs_batch_parallel_step_body(
                shape,
                frontier_out,
                changed,
                converged,
                words,
                edge_kind_mask,
                &format!("batch_grid_iter_{iter}"),
                uses_grid_sync,
            ));
            match density_active {
                // Record per-query density after this step. The step's
                // cross-workgroup writes to each query's frontier must be globally
                // visible before the query leader popcounts (first barrier), and
                // those reads must retire before the next step overwrites the
                // frontier (second barrier). These barriers subsume the base
                // inter-iteration grid-sync barrier, so the base program keeps its
                // single barrier and its ABI.
                Some(density) => {
                    entry.push(grid_sync_barrier());
                    entry.push(persistent_bfs_batch_record_density(
                        density,
                        frontier_out,
                        words,
                        max_iters,
                        Expr::u32(iter),
                        &format!("batch_grid_density_word_{iter}"),
                    ));
                    entry.push(grid_sync_barrier());
                }
                None => {
                    if iter + 1 < max_iters {
                        entry.push(grid_sync_barrier());
                    }
                }
            }
        }
    } else if max_iters > 0 {
        let mut loop_body = persistent_bfs_batch_parallel_step_body(
            shape,
            frontier_out,
            changed,
            converged,
            words,
            edge_kind_mask,
            "batch_loop",
            uses_grid_sync,
        );
        // In the plain-loop path each query lives in one workgroup and the step
        // body ends with a workgroup barrier, so the query leader sees the step's
        // frontier writes here; the density index uses the runtime loop counter.
        if let Some(density) = density_active {
            loop_body.push(persistent_bfs_batch_record_density(
                density,
                frontier_out,
                words,
                max_iters,
                Expr::var("batch_iter"),
                "batch_loop_density_word",
            ));
        }
        entry.push(Node::loop_for(
            "batch_iter",
            Expr::u32(0),
            Expr::u32(max_iters),
            loop_body,
        ));
    }

    // Publish the per-query converged flag. `converged[q]` was reset to 0 at the
    // start of each step and OR'd to 1 by any lane that grew query q, so after the
    // loop it holds "the LAST step added a node for q". Growth is monotone, so a
    // last step that added nothing proves a fixpoint was reached within budget:
    // converged[q] = (converged[q] == 0). The grid-sync path needs one trailing
    // grid-sync barrier so every workgroup's last-step writes are visible before
    // the query leader reads them; the plain-loop path already ran the step body's
    // trailing workgroup barrier on the final iteration (query q lives in one
    // workgroup when node_count <= the workgroup size). With max_iters == 0 no
    // step ran, so the init above already left converged[q] = 0.
    if max_iters > 0 {
        if uses_grid_sync {
            entry.push(grid_sync_barrier());
        }
        entry.push(Node::if_then(
            Expr::eq(Expr::gid_x(), Expr::u32(0)),
            vec![Node::store(
                converged,
                Expr::gid_y(),
                Expr::select(
                    Expr::eq(Expr::load(converged, Expr::gid_y()), Expr::u32(0)),
                    Expr::u32(1),
                    Expr::u32(0),
                ),
            )],
        ));
    }

    let density_words = match density_active {
        Some(_) => Some(
            query_count
                .max(1)
                .checked_mul(max_iters)
                .ok_or_else(|| {
                    format!(
                        "{BATCH_OP_ID} density array words overflow u32: query_count={query_count}, max_iters={max_iters}. Fix: shard the BFS query batch before GPU dispatch."
                    )
                })?
                .max(1),
        ),
        None => None,
    };
    let mut buffers = shape.try_read_only_buffers()?;
    PersistentBfsBuffers {
        frontier_in,
        frontier_out,
        frontier_words: total_words,
        changed: (changed, query_count.max(1)),
        converged: (converged, query_count.max(1)),
        density_active: density_active.zip(density_words),
    }
    .push_onto(&mut buffers);

    Ok(Program::wrapped(
        buffers,
        PERSISTENT_BFS_WORKGROUP_SIZE,
        vec![wrap_anonymous_region(BATCH_OP_ID, entry)],
    ))
}

fn persistent_bfs_batch_needs_grid_sync(shape: ProgramGraphShape) -> bool {
    shape.node_count > PERSISTENT_BFS_WORKGROUP_SIZE[0]
}

/// Record `density_active[q * max_iters + iter] = popcount(query q's frontier)`
/// after a batch step. The per-query leader (`gid_x()==0` on query row `q`)
/// serially sums the popcount of every word in that query's flat bitset region
/// and stores the total. Like the single-query grid-sync path this is a plain,
/// IDEMPOTENT write (recompute-and-store), so it lands the same value even when
/// the grid-sync split re-executes segments to a fixpoint, where an accumulating
/// atomic would double-count. `iter_index` is the step index (a compile-time
/// constant in the grid-sync loop, the runtime loop variable in the plain loop);
/// `word_var` names the private inner loop counter.
fn persistent_bfs_batch_record_density(
    density: &str,
    frontier_out: &str,
    words: u32,
    max_iters: u32,
    iter_index: Expr,
    word_var: &str,
) -> Node {
    let q = Expr::gid_y();
    let base = Expr::mul(q.clone(), Expr::u32(words));
    let density_index = Expr::add(Expr::mul(q, Expr::u32(max_iters)), iter_index);
    Node::if_then(
        Expr::eq(Expr::gid_x(), Expr::u32(0)),
        vec![
            Node::store(density, density_index.clone(), Expr::u32(0)),
            Node::loop_for(
                word_var,
                Expr::u32(0),
                Expr::u32(words),
                vec![Node::store(
                    density,
                    density_index.clone(),
                    Expr::add(
                        Expr::load(density, density_index.clone()),
                        Expr::popcount(Expr::load(
                            frontier_out,
                            Expr::add(base.clone(), Expr::var(word_var)),
                        )),
                    ),
                )],
            ),
        ],
    )
}

fn persistent_bfs_batch_sync(uses_grid_sync: bool) -> Node {
    if uses_grid_sync {
        grid_sync_barrier()
    } else {
        Node::barrier()
    }
}

fn persistent_bfs_batch_parallel_step_body(
    shape: ProgramGraphShape,
    frontier_out: &str,
    changed: &str,
    converged: &str,
    words: u32,
    edge_kind_mask: u32,
    local_prefix: &str,
    uses_grid_sync: bool,
) -> Vec<Node> {
    let local = |name: &str| -> String { format!("{local_prefix}_{name}") };
    let q = Expr::gid_y();
    let base = Expr::mul(q.clone(), Expr::u32(words));
    let src = Expr::gid_x();
    let in_bounds = local("in_bounds");
    let word_idx = local("word_idx");
    let bit_mask = local("bit_mask");
    let src_word = local("src_word");
    let src_active = local("src_active");
    let changed_old = local("changed_old");
    let converged_old = local("converged_old");

    // Neighbor expansion is the ONE canonical CSR edge-scan. This batch step differs
    // from a single-bitset caller on exactly two axes, both supplied here: the frontier
    // word index is offset by this query's `base` (a flat per-query bitset region), and a
    // newly-set bit ORs this query's `changed[q]` slot. The source-activity bit is read
    // and snapshotted before the barrier BELOW (the one-hop-per-iteration guarantee), so
    // this uses the edge-walk-only `csr_edge_expand_nodes`. Output-identical to the former
    // hand-written closure (the `base +` moved from the atomic-OR index onto the word-index
    // bind (same storage slot); locked by the graph oracle/fixpoint matrices).
    let edge_scan = || {
        crate::graph::edge_scan::csr_edge_expand_nodes(
            shape,
            frontier_out,
            src.clone(),
            |word| Expr::add(base.clone(), word),
            || {
                vec![
                    Node::let_bind(
                        changed_old.as_str(),
                        Expr::atomic_or(changed, q.clone(), Expr::u32(1)),
                    ),
                    // Mark THIS step active for query q. `converged[q]` is reset to
                    // 0 at the top of the step, so after the loop it holds "did the
                    // last step grow q"; the trailing publish turns it into the
                    // converged flag. Separate from the sticky `changed[q]` because
                    // convergence needs the last-step state, not the cumulative OR.
                    Node::let_bind(
                        converged_old.as_str(),
                        Expr::atomic_or(converged, q.clone(), Expr::u32(1)),
                    ),
                ]
            },
            edge_kind_mask,
            local_prefix,
        )
    };

    let mut body = vec![
        // Reset this query's per-step active flag before the snapshot barrier
        // below makes it visible to every lane that may OR it. The query leader
        // lane owns the slot. In the plain-loop (non-grid-sync) path query q lives
        // in one workgroup, so a workgroup barrier synchronizes the reset; in the
        // grid-sync path the barrier is grid-wide.
        Node::if_then(
            Expr::eq(src.clone(), Expr::u32(0)),
            vec![Node::store(converged, q.clone(), Expr::u32(0))],
        ),
        Node::let_bind(
            in_bounds.as_str(),
            Expr::lt(src.clone(), Expr::u32(shape.node_count)),
        ),
    ];
    // Snapshot path: the load lands before the barrier and the guard after it, so
    // the address, the load and the test are separate statements rather than one
    // `frontier_bits::when_bit_set` probe. The word index is `select`ed to zero out
    // of bounds so the pre-barrier load stays in range for the tail lanes.
    body.extend(bind_bit_address(
        &src,
        word_idx.as_str(),
        bit_mask.as_str(),
        |word| Expr::select(Expr::var(in_bounds.as_str()), word, Expr::u32(0)),
    ));
    body.extend([
        Node::let_bind(
            src_word.as_str(),
            Expr::load(
                frontier_out,
                Expr::add(base.clone(), Expr::var(word_idx.as_str())),
            ),
        ),
        Node::let_bind(
            src_active.as_str(),
            Expr::select(
                Expr::var(in_bounds.as_str()),
                Expr::bitand(Expr::var(src_word.as_str()), Expr::var(bit_mask.as_str())),
                Expr::u32(0),
            ),
        ),
        persistent_bfs_batch_sync(uses_grid_sync),
        Node::if_then(
            Expr::ne(Expr::var(src_active.as_str()), Expr::u32(0)),
            edge_scan(),
        ),
    ]);
    if !uses_grid_sync {
        body.push(Node::barrier());
    }
    body
}

fn checked_batch_frontier_words(
    words_per_query: u32,
    query_count: u32,
    op_id: &'static str,
) -> Result<u32, String> {
    words_per_query.checked_mul(query_count.max(1)).ok_or_else(|| {
        format!(
            "{op_id} frontier words overflow u32: words_per_query={words_per_query}, query_count={query_count}. Fix: shard the BFS query batch before GPU dispatch."
        )
    })
}
