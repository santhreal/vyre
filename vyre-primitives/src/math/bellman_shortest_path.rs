use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};

#[cfg(test)]
use crate::fixpoint::persistent_fixpoint::{
    declared_words, fixpoint_route, persistent_fixpoint, required_workgroups,
    PERSISTENT_FIXPOINT_WORKGROUP_SIZE,
};
use crate::fixpoint::persistent_fixpoint::{routed_persistent_fixpoint, FixpointState};

/// Canonical op id.
pub const OP_ID: &str = "vyre-primitives::math::bellman_shortest_path";

/// The six buffer bindings one Bellman-Ford shortest-path program declares.
///
/// Every field is a `&str`. Naming each binding at the construction site is what
/// makes a transposition of two of them a diff rather than a silent argument
/// swap: `src`/`dst` reverses every edge, and `dist`/`next_dist` swaps the
/// output half of the ping-pong with the scratch half.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BellmanBuffers<'a> {
    /// `n_edges` edge sources.
    pub src: &'a str,
    /// `n_edges` edge targets.
    pub dst: &'a str,
    /// `n_edges` edge weights.
    pub weight: &'a str,
    /// `n_nodes` distances. Holds the result after the dispatch returns.
    pub dist: &'a str,
    /// `n_nodes` distances, the relaxation scratch half of the ping-pong.
    pub next_dist: &'a str,
    /// Convergence flag. Its width is decided by the routed harness, not the
    /// caller.
    pub changed: &'a str,
}

impl BellmanBuffers<'static> {
    /// The canonical binding names for a Bellman-Ford program.
    ///
    /// A caller with no naming of its own gets one here instead of spelling six
    /// strings, and every program built from it declares the same bindings in the
    /// same order, which is what makes two such programs comparable byte for
    /// byte. This is the only place the six names are spelled, so a transposed
    /// `src`/`dst` or `dist`/`next_dist` is a one-line diff here rather than a
    /// reordering inside a positional list nobody reads.
    pub const CANONICAL: Self = Self {
        src: "src",
        dst: "dst",
        weight: "weight",
        dist: "dist",
        next_dist: "next_dist",
        changed: "changed",
    };

    /// Short binding names, for a caller that only cares about program shape.
    ///
    /// Deliberately disjoint from [`Self::CANONICAL`] so a test that reads a name
    /// back out of a program cannot pass by matching the other set: with one
    /// shared naming, an assertion against `"dist"` would hold whichever record
    /// built the program.
    pub const TERSE: Self = Self {
        src: "s",
        dst: "d",
        weight: "w",
        dist: "di",
        next_dist: "nd",
        changed: "c",
    };
}

/// The graph extents and iteration cap one Bellman-Ford program is built for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BellmanExtents {
    /// Node count, and the length of `dist` and `next_dist`.
    pub n_nodes: u32,
    /// Edge count, and the length of `src`, `dst` and `weight`.
    pub n_edges: u32,
    /// Hard cap on relaxation iterations.
    pub max_iterations: u32,
}

/// Build a fused Bellman-Ford shortest-path Program: relax edges
/// until convergence, all inside ONE GPU dispatch.
///
/// Composes a persistent fixpoint harness over an edge list to perform
/// graph distances without host round-trips.
///
/// # Convergence-flag form
///
/// The launch spans `max(n_nodes, n_edges)` lanes: the relaxation body is
/// gated on `t < n_edges` and the compare plus ping-pong on `t < n_nodes`,
/// and `dispatch_element_count_for_program`
/// (`vyre-driver/src/program_walks/dispatch_params.rs:19`) sizes an
/// atomic-carrying program's launch from its WIDEST declared buffer. This
/// program carries both `atomic_min` on `next_dist` and the harness's
/// `atomic_or`, so that full-span rule applies and the span, not `n_nodes`,
/// selects the harness:
///
/// - span `<= PERSISTENT_FIXPOINT_WORKGROUP_SIZE[0]`: one workgroup covers
///   the launch, so [`persistent_fixpoint`] runs with its single shared
///   `changed[0]` word. That word is cleared by a plain store fenced only by
///   a workgroup-scope barrier; with one group the fence is incidentally
///   grid-wide, so the clear cannot race the `atomic_or` that sets the flag.
/// - span above that width: [`persistent_fixpoint_grid`], which never clears
///   the flag, gives each iteration its own `changed` word, and separates
///   waves with `MemoryOrdering::GridSync`. The single-word form is limited
///   to one workgroup precisely because its clear and its set are unordered
///   across groups: group 0's clear can erase another group's set, that
///   group then reads 0 and returns early with unconverged distances, and
///   the flag the host reads afterwards reports a convergence no group
///   agreed to.
///
/// Invalid dimensions lower to an explicit trap program.
///
/// # Buffers and extents
///
/// Named by [`BellmanBuffers`] and [`BellmanExtents`]. All six binding names are
/// `&str` and all three extents are `u32`, so a positional call of the nine
/// compiled with `src` and `dst` transposed, or `dist` and `next_dist`
/// transposed, and emitted a program that relaxed the graph backwards or wrote
/// its output into the scratch half of the ping-pong.
#[must_use]
pub fn bellman_shortest_path(buffers: BellmanBuffers<'_>, extents: BellmanExtents) -> Program {
    let BellmanBuffers {
        src,
        dst,
        weight,
        dist,
        next_dist,
        changed,
    } = buffers;
    let BellmanExtents {
        n_nodes,
        n_edges,
        max_iterations,
    } = extents;
    if n_nodes == 0 {
        return crate::invalid_output_program(
            OP_ID,
            dist,
            DataType::U32,
            format!("Fix: bellman_shortest_path requires n_nodes > 0, got {n_nodes}."),
        );
    }
    if max_iterations == 0 {
        return crate::invalid_output_program(
            OP_ID,
            dist,
            DataType::U32,
            format!(
                "Fix: bellman_shortest_path requires max_iterations > 0, got {max_iterations}."
            ),
        );
    }

    let transfer_body = bellman_transfer_body(buffers, extents);

    // `n_nodes` alone does NOT decide the harness: see the form note above.
    // The edge buffers are `n_edges` long, so the launch spans
    // `max(n_nodes, n_edges)` lanes and a wide edge list makes the dispatch
    // multi-workgroup even for a node array that fits one group.
    let (inner, route) = routed_persistent_fixpoint(
        transfer_body,
        FixpointState {
            current: dist,
            next: next_dist,
            changed,
            words: n_nodes,
            max_iterations,
        },
        n_nodes.max(n_edges),
    );

    bellman_wrap(&inner, buffers, extents, route.changed_words)
}

/// One edge-relaxation step, the transfer function the convergence harness runs
/// to a fixpoint.
///
/// Lane `t` owns edge `t`, so the work is PARTITIONED by global invocation id
/// rather than chunked: a launch wider than one workgroup really does place edge
/// relaxation in groups above 0, and those groups are the only writers of the
/// slots they own. That is what makes the shared-convergence-word race in
/// [`persistent_fixpoint`] observable here rather than masked.
fn bellman_transfer_body(buffers: BellmanBuffers<'_>, extents: BellmanExtents) -> Vec<Node> {
    let BellmanBuffers {
        src,
        dst,
        weight,
        dist,
        next_dist,
        ..
    } = buffers;
    let BellmanExtents {
        n_nodes, n_edges, ..
    } = extents;
    let t = Expr::InvocationId { axis: 0 };

    vec![Node::if_then(
        Expr::lt(t.clone(), Expr::u32(n_edges)),
        vec![
            Node::let_bind("u", Expr::load(src, t.clone())),
            Node::let_bind("v", Expr::load(dst, t.clone())),
            Node::let_bind("w", Expr::load(weight, t.clone())),
            // `u`/`v` are DATA (edge endpoints loaded from src/dst); nothing validates
            // them `< n_nodes`. The CPU reference SKIPS any edge with an out-of-range
            // endpoint (`if u >= n || v >= n { continue }`), so the GPU MUST gate the
            // dist[u] load AND the next_dist[v] atomic-min on the same bound, otherwise
            // it OOB-loads dist[u] (UB / garbage `du` that can spuriously relax) and
            // OOB atomic-WRITES next_dist[v] (memory corruption on real hardware),
            // diverging from the CPU ref on any malformed edge (gather / test_bit class).
            Node::if_then(
                Expr::and(
                    Expr::lt(Expr::var("u"), Expr::u32(n_nodes)),
                    Expr::lt(Expr::var("v"), Expr::u32(n_nodes)),
                ),
                vec![
                    Node::let_bind("du", Expr::load(dist, Expr::var("u"))),
                    Node::if_then(
                        Expr::ne(Expr::var("du"), Expr::u32(u32::MAX)),
                        vec![
                            Node::let_bind(
                                "alt",
                                Expr::select(
                                    Expr::gt(
                                        Expr::var("w"),
                                        Expr::sub(Expr::u32(u32::MAX), Expr::var("du")),
                                    ),
                                    Expr::u32(u32::MAX),
                                    Expr::add(Expr::var("du"), Expr::var("w")),
                                ),
                            ),
                            Node::let_bind(
                                "_relax",
                                Expr::atomic_min(next_dist, Expr::var("v"), Expr::var("alt")),
                            ),
                        ],
                    ),
                ],
            ),
        ],
    )]
}

/// Wrap a convergence harness in bellman's Region and buffer declarations.
///
/// Single owner of those declarations, so the two routed forms and the
/// single-word form the divergence test builds cannot drift apart in binding
/// order, counts, or access modes.
fn bellman_wrap(
    inner: &Program,
    buffers: BellmanBuffers<'_>,
    extents: BellmanExtents,
    changed_words: u32,
) -> Program {
    let BellmanBuffers {
        src,
        dst,
        weight,
        dist,
        next_dist,
        changed,
    } = buffers;
    let BellmanExtents {
        n_nodes, n_edges, ..
    } = extents;
    super::wrap_fixpoint_program(
        OP_ID,
        inner,
        vec![
            BufferDecl::storage(dist, 0, BufferAccess::ReadWrite, DataType::U32)
                .with_count(n_nodes),
            BufferDecl::storage(next_dist, 1, BufferAccess::ReadWrite, DataType::U32)
                .with_count(n_nodes),
            BufferDecl::storage(changed, 2, BufferAccess::ReadWrite, DataType::U32)
                .with_count(changed_words),
            BufferDecl::storage(src, 3, BufferAccess::ReadOnly, DataType::U32).with_count(n_edges),
            BufferDecl::storage(dst, 4, BufferAccess::ReadOnly, DataType::U32).with_count(n_edges),
            BufferDecl::storage(weight, 5, BufferAccess::ReadOnly, DataType::U32)
                .with_count(n_edges),
        ],
    )
}

/// The pre-routing program: bellman's transfer body on the single-word
/// convergence harness at ANY size, which is exactly what
/// [`bellman_shortest_path`] emitted before the dispatch-span routing landed.
///
/// Exists only so the divergence test can OBSERVE the wrong distances the racing
/// shared flag produces above one workgroup. Production code must never take
/// this path above one workgroup width.
#[cfg(test)]
fn bellman_single_word_harness(buffers: BellmanBuffers<'_>, extents: BellmanExtents) -> Program {
    let transfer_body = bellman_transfer_body(buffers, extents);
    let inner = persistent_fixpoint(
        transfer_body,
        buffers.dist,
        buffers.next_dist,
        buffers.changed,
        extents.n_nodes,
        extents.max_iterations,
    );
    bellman_wrap(&inner, buffers, extents, 1)
}

/// CPU reference.
#[cfg(any(test, feature = "cpu-parity"))]
#[must_use]
pub fn cpu_ref(
    src: &[u32],
    dst: &[u32],
    weight: &[u32],
    dist: &[u32],
    n_nodes: u32,
    max_iterations: u32,
) -> (Vec<u32>, u32) {
    let mut current = Vec::new();
    let mut next = Vec::new();
    let iters = cpu_ref_into(
        src,
        dst,
        weight,
        dist,
        n_nodes,
        max_iterations,
        &mut current,
        &mut next,
    );
    (current, iters)
}

/// CPU reference using caller-owned current and next-distance buffers.
///
/// `current` is overwritten with the final distance vector. `next` is retained
/// as monotone relaxation scratch so repeated parity checks do not allocate
/// fresh `Vec`s or clone the initial distance vector.
#[cfg(any(test, feature = "cpu-parity"))]
#[allow(clippy::too_many_arguments)]
pub fn cpu_ref_into(
    src: &[u32],
    dst: &[u32],
    weight: &[u32],
    dist: &[u32],
    n_nodes: u32,
    max_iterations: u32,
    current: &mut Vec<u32>,
    next: &mut Vec<u32>,
) -> u32 {
    let n = n_nodes as usize;
    let edge_count = src.len().min(dst.len()).min(weight.len());
    current.clear();
    current.resize(n, u32::MAX);
    for (out, &value) in current.iter_mut().zip(dist.iter()) {
        *out = value;
    }
    next.clear();
    next.extend_from_slice(current);
    for iter in 0..max_iterations {
        for i in 0..edge_count {
            let u = src[i] as usize;
            let v = dst[i] as usize;
            if u >= n || v >= n {
                continue;
            }
            let w = weight[i];
            let du = current[u];
            if du != u32::MAX {
                let alt = du.saturating_add(w);
                next[v] = next[v].min(alt);
            }
        }
        if next.as_slice() == current.as_slice() {
            return iter;
        }
        current.copy_from_slice(&next);
    }
    max_iterations
}

#[cfg(feature = "inventory-registry")]
inventory::submit! {
    vyre_foundation::operation::OperationRegistration::primitive(
        OP_ID,
        || {
            bellman_shortest_path(
                BellmanBuffers::CANONICAL,
                BellmanExtents {
                    n_nodes: 4,
                    n_edges: 4,
                    max_iterations: 10,
                },
            )
        },
        Some(|| {
            let to_bytes = |w: &[u32]| crate::wire::pack_u32_slice(w);
            vec![vec![
                to_bytes(&[0, u32::MAX, u32::MAX, u32::MAX]), // dist
                to_bytes(&[0, u32::MAX, u32::MAX, u32::MAX]), // next_dist
                to_bytes(&[0]), // changed
                to_bytes(&[0, 1, 2, 0]), // src
                to_bytes(&[1, 2, 3, 3]), // dst
                to_bytes(&[10, 20, 30, 100]), // weight
            ]]
        }),
        Some(|| {
            let to_bytes = |w: &[u32]| crate::wire::pack_u32_slice(w);
            vec![vec![
                to_bytes(&[0, 10, 30, 60]), // dist
                to_bytes(&[0, 10, 30, 60]), // next_dist
                to_bytes(&[0]),             // changed
            ]]
        }),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    /// The binding names the program tests in this module build against.
    const FIXTURE: BellmanBuffers<'static> = BellmanBuffers::TERSE;

    /// The registry fixture's names, kept distinct so a test that reads them
    /// cannot pass by matching the short set above.
    const VERBOSE_FIXTURE: BellmanBuffers<'static> = BellmanBuffers::CANONICAL;

    /// `BellmanExtents` for `n_nodes` nodes and `n_edges` edges under a cap.
    fn extents(n_nodes: u32, n_edges: u32, max_iterations: u32) -> BellmanExtents {
        BellmanExtents {
            n_nodes,
            n_edges,
            max_iterations,
        }
    }

    /// The emitted binding at `index`, as `(name, count)`.
    fn binding(program: &Program, index: usize) -> (&str, u32) {
        let buffer = &program.buffers()[index];
        (buffer.name(), buffer.count())
    }

    /// Wire encoding of `program`. A debug string would compare formatting.
    fn to_wire(program: &Program) -> Vec<u8> {
        vyre_foundation::serial::wire::encode::to_wire(program)
            .expect("Fix: a bellman program must encode to the wire form.")
    }

    /// Each named field must reach the binding that carries its role.
    ///
    /// All six fields are `&str`, so the type system cannot reject a
    /// transposition; what it can do is make one observable. `n_nodes != n_edges`
    /// so the node-length and edge-length roles cannot be confused by count.
    #[test]
    fn every_named_binding_reaches_the_role_it_names() {
        let program = bellman_shortest_path(FIXTURE, extents(3, 5, 1));

        assert_eq!(binding(&program, 0), (FIXTURE.dist, 3));
        assert_eq!(binding(&program, 1), (FIXTURE.next_dist, 3));
        assert_eq!(binding(&program, 2), (FIXTURE.changed, 1));
        assert_eq!(binding(&program, 3), (FIXTURE.src, 5));
        assert_eq!(binding(&program, 4), (FIXTURE.dst, 5));
        assert_eq!(binding(&program, 5), (FIXTURE.weight, 5));
    }

    /// Transposing two names must change the emitted program.
    ///
    /// Each pair below leaves the argument count and every type valid, so it is
    /// exactly what the old positional call could not catch. `src`/`dst` reverses
    /// every edge and `dist`/`next_dist` swaps the output half of the ping-pong
    /// with the scratch half; both must be visible in the emission.
    #[test]
    fn transposing_two_binding_names_changes_the_wire_encoding() {
        let extents = extents(3, 5, 1);
        let canonical = to_wire(&bellman_shortest_path(FIXTURE, extents));

        for (label, transposed) in [
            (
                "src / dst",
                BellmanBuffers {
                    src: FIXTURE.dst,
                    dst: FIXTURE.src,
                    ..FIXTURE
                },
            ),
            (
                "dist / next_dist",
                BellmanBuffers {
                    dist: FIXTURE.next_dist,
                    next_dist: FIXTURE.dist,
                    ..FIXTURE
                },
            ),
            (
                "weight / src",
                BellmanBuffers {
                    weight: FIXTURE.src,
                    src: FIXTURE.weight,
                    ..FIXTURE
                },
            ),
        ] {
            assert_ne!(
                to_wire(&bellman_shortest_path(transposed, extents)),
                canonical,
                "Fix: transposing {label} must change the emitted program, or the two names are interchangeable and one of them is dead."
            );
        }
    }

    /// Routing through the shared fixpoint owner must not change the emission.
    ///
    /// This op used to re-derive the harness selection and the matching `changed`
    /// width itself. Both sides of the routing threshold, and the flag width each
    /// side needs, are pinned here on the wire encoding so the delegation is
    /// provably behavior-preserving rather than plausibly so.
    #[test]
    fn routing_matches_the_shared_fixpoint_owner_on_both_sides_of_the_threshold() {
        let width = PERSISTENT_FIXPOINT_WORKGROUP_SIZE[0];

        for (n_nodes, n_edges, max_iterations) in [
            (4, 4, 10),
            (width, width, 8),
            (width + 1, width, 8),
            (4, width + 1, 8),
        ] {
            let extents = extents(n_nodes, n_edges, max_iterations);
            let route = fixpoint_route(n_nodes.max(n_edges), max_iterations);
            let program = bellman_shortest_path(FIXTURE, extents);

            assert_eq!(
                binding(&program, 2).1,
                route.changed_words,
                "Fix: the declared convergence-flag width must be the width the routed harness indexes."
            );

            let expected = bellman_wrap(
                &routed_persistent_fixpoint(
                    bellman_transfer_body(FIXTURE, extents),
                    FixpointState {
                        current: FIXTURE.dist,
                        next: FIXTURE.next_dist,
                        changed: FIXTURE.changed,
                        words: n_nodes,
                        max_iterations,
                    },
                    n_nodes.max(n_edges),
                )
                .0,
                FIXTURE,
                extents,
                route.changed_words,
            );
            assert_eq!(
                to_wire(&program),
                to_wire(&expected),
                "Fix: bellman_shortest_path must emit exactly what the routed harness plus its own wrapper produce."
            );
        }
    }

    #[test]
    fn test_cpu_ref_trivial() {
        let src = vec![0];
        let dst = vec![1];
        let weight = vec![5];
        let dist = vec![0, u32::MAX];
        let (final_dist, iters) = cpu_ref(&src, &dst, &weight, &dist, 2, 10);
        assert_eq!(final_dist, vec![0, 5]);
        assert_eq!(iters, 1);
    }

    #[test]
    fn test_cpu_ref_single_node() {
        let dist = vec![0];
        let (final_dist, iters) = cpu_ref(&[], &[], &[], &dist, 1, 10);
        assert_eq!(final_dist, vec![0]);
        assert_eq!(iters, 0);
    }

    #[test]
    fn test_cpu_ref_cycle() {
        let src = vec![0, 1, 2];
        let dst = vec![1, 2, 0];
        let weight = vec![10, 10, 10];
        let dist = vec![0, u32::MAX, u32::MAX];
        let (final_dist, _) = cpu_ref(&src, &dst, &weight, &dist, 3, 10);
        assert_eq!(final_dist, vec![0, 10, 20]);
    }

    #[test]
    fn test_cpu_ref_large_line() {
        let n = 50;
        let mut src = Vec::new();
        let mut dst = Vec::new();
        let mut weight = Vec::new();
        for i in 0..n - 1 {
            src.push(i as u32);
            dst.push((i + 1) as u32);
            weight.push(1);
        }
        let mut dist = vec![u32::MAX; n];
        dist[0] = 0;
        let (final_dist, iters) = cpu_ref(&src, &dst, &weight, &dist, n as u32, n as u32 * 2);
        assert_eq!(final_dist[n - 1], (n - 1) as u32);
        assert_eq!(iters, (n - 1) as u32);
    }

    #[test]
    fn test_cpu_ref_asymmetric() {
        let src = vec![0, 0, 1, 2];
        let dst = vec![1, 3, 3, 3];
        let weight = vec![10, 100, 20, 5];
        let dist = vec![0, u32::MAX, u32::MAX, u32::MAX];
        // 0->3 is 100
        // 0->1->3 is 10+20=30
        let (final_dist, _) = cpu_ref(&src, &dst, &weight, &dist, 4, 10);
        assert_eq!(final_dist[3], 30);
    }

    #[test]
    fn test_cpu_ref_ignores_malformed_edges_and_pads_distances() {
        let src = vec![0, 9, 1];
        let dst = vec![1, 2];
        let weight = vec![5, 99, 7];
        let (final_dist, _) = cpu_ref(&src, &dst, &weight, &[0], 3, 10);
        assert_eq!(final_dist, vec![0, 5, u32::MAX]);
    }

    #[test]
    fn cpu_ref_into_reuses_current_and_next_buffers() {
        let src = vec![0, 1, 2, 0];
        let dst = vec![1, 2, 3, 3];
        let weight = vec![10, 20, 30, 100];
        let dist = vec![0, u32::MAX, u32::MAX, u32::MAX];
        let mut current = Vec::with_capacity(16);
        let mut next = Vec::with_capacity(16);
        current.extend_from_slice(&[99, 98, 97, 96, 95, 94]);
        next.extend_from_slice(&[77, 76, 75, 74, 73, 72]);
        let current_capacity = current.capacity();
        let next_capacity = next.capacity();

        let iters = cpu_ref_into(&src, &dst, &weight, &dist, 4, 10, &mut current, &mut next);

        assert_eq!(current, vec![0, 10, 30, 60]);
        assert!(iters <= 4);
        assert_eq!(current.capacity(), current_capacity);
        assert_eq!(next.capacity(), next_capacity);

        let iters = cpu_ref_into(&[], &[], &[], &[0], 1, 10, &mut current, &mut next);
        assert_eq!(current, vec![0]);
        assert_eq!(next, vec![0]);
        assert_eq!(iters, 0);
        assert_eq!(current.capacity(), current_capacity);
        assert_eq!(next.capacity(), next_capacity);
    }

    #[test]
    fn test_parity_small_graph() {
        let src = vec![0, 1, 2, 0];
        let dst = vec![1, 2, 3, 3];
        let weight = vec![10, 20, 30, 100];
        let dist_init = vec![0, u32::MAX, u32::MAX, u32::MAX];

        let p = bellman_shortest_path(VERBOSE_FIXTURE, extents(4, 4, 10));

        let (expected_dist, _) = cpu_ref(&src, &dst, &weight, &dist_init, 4, 10);

        use vyre_reference::reference_eval;
        use vyre_reference::value::Value;

        let to_value = |data: &[u32]| {
            let bytes = crate::wire::pack_u32_slice(data);
            Value::Bytes(Arc::from(bytes))
        };

        let inputs = vec![
            to_value(&dist_init),
            to_value(&dist_init),
            to_value(&[0]),
            to_value(&src),
            to_value(&dst),
            to_value(&weight),
        ];

        let results = reference_eval(&p, &inputs).expect("Fix: interpreter failed");
        let actual_bytes = results[0].to_bytes();
        let actual_dist: Vec<u32> = actual_bytes
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
            .collect();

        assert_eq!(actual_dist, expected_dist);
    }

    #[test]
    fn program_declares_six_buffers() {
        let p = bellman_shortest_path(FIXTURE, extents(4, 4, 10));
        assert_eq!(p.buffers().len(), 6);
    }

    #[test]
    fn rejects_zero_nodes_with_trap() {
        let p = bellman_shortest_path(FIXTURE, extents(0, 4, 10));
        assert!(p.stats().trap());
    }

    #[test]
    fn rejects_zero_max_iterations_with_trap() {
        let p = bellman_shortest_path(FIXTURE, extents(4, 4, 0));
        assert!(p.stats().trap());
    }

    /// Both ways a Bellman-Ford launch outgrows one workgroup, each registered
    /// against the shared routing contract.
    ///
    /// The span is the widest declared buffer, so it can be widened from either
    /// side: by the node count, which widens `dist`/`next_dist`, or by the edge
    /// count, which widens `src`/`dst`/`weight`. Routing on `n_nodes` alone would
    /// leave 4 nodes and 257 edges on the racing single-word flag while looking
    /// comfortably small, so both sources are members here rather than one.
    fn routed_forms() -> [(&'static str, u32, u32); 2] {
        let width = PERSISTENT_FIXPOINT_WORKGROUP_SIZE[0];
        [
            ("bellman_shortest_path, node-widened span", width + 1, width),
            ("bellman_shortest_path, edge-widened span", 4, width + 1),
        ]
    }

    #[test]
    fn every_routed_form_obeys_the_persistent_fixpoint_routing_contract() {
        let width = PERSISTENT_FIXPOINT_WORKGROUP_SIZE[0];
        for (name, n_nodes, n_edges) in routed_forms() {
            crate::fixpoint::routing_contract::assert_routes_on_dispatch_span(
                &crate::fixpoint::routing_contract::RoutedFixpointOp {
                    name,
                    changed: FIXTURE.changed,
                    at_one_workgroup: &|max_iterations| {
                        bellman_shortest_path(FIXTURE, extents(width, width, max_iterations))
                    },
                    past_one_workgroup: &|max_iterations| {
                        bellman_shortest_path(FIXTURE, extents(n_nodes, n_edges, max_iterations))
                    },
                    grid_harness: &|max_iterations| {
                        crate::fixpoint::routing_contract::bare_grid_harness(
                            FIXTURE.dist,
                            FIXTURE.next_dist,
                            FIXTURE.changed,
                            n_nodes,
                            max_iterations,
                        )
                    },
                },
            );
        }
    }

    /// Run `program` on the reference interpreter and return the final `dist`
    /// vector paired with the final `changed` words. `reversed` steps the
    /// workgroups back to front; both orders are schedules real hardware is free
    /// to pick, because nothing in the IR orders one workgroup against another.
    fn run_bellman(
        program: &Program,
        reversed: bool,
        dist: &[u32],
        src: &[u32],
        dst: &[u32],
        weight: &[u32],
        changed_words: u32,
    ) -> (Vec<u32>, Vec<u32>) {
        use vyre_reference::value::Value;

        let to_value = |data: &[u32]| Value::Bytes(Arc::from(crate::wire::pack_u32_slice(data)));
        let inputs = vec![
            to_value(dist),
            to_value(dist),
            to_value(&vec![0_u32; changed_words as usize]),
            to_value(src),
            to_value(dst),
            to_value(weight),
        ];
        let results = if reversed {
            vyre_reference::reference_eval_lane_reversed(program, &inputs)
        } else {
            vyre_reference::reference_eval(program, &inputs)
        }
        .expect("Fix: the reference interpreter must execute the bellman program.");
        let decode = |value: &vyre_reference::value::Value| -> Vec<u32> {
            value
                .to_bytes()
                .chunks_exact(4)
                .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()))
                .collect()
        };
        (decode(&results[0]), decode(&results[2]))
    }

    /// OBSERVED divergence: the pre-routing single-word harness returns WRONG
    /// distances above one workgroup, it does not merely look unsound.
    ///
    /// 257 nodes, one edge `0 -> 256` of weight 5. The launch is two workgroups
    /// (257 elements over a 256-wide group). Lane 0, in group 0, owns the edge and
    /// relaxes `next_dist[256]`. Lane 256, in group 1, is the ONLY lane whose
    /// compare covers node 256, so it is the only writer of `dist[256]` and the
    /// only lane that can set the convergence flag for that node. Nothing orders
    /// the two groups.
    ///
    /// Step group 1 first and it compares `dist[256]` against a `next_dist[256]`
    /// group 0 has not yet relaxed, sees no change, reads the still-zero shared
    /// flag and returns for good. Group 0 then relaxes `next_dist[256] = 5`, finds
    /// no change among the nodes IT covers, and also returns. `dist[256]` is never
    /// written: the dispatch reports convergence and yields `u32::MAX` where the
    /// answer is 5. Step group 0 first and the same program is correct, which is
    /// the definition of the race.
    #[test]
    fn single_word_harness_returns_wrong_distances_above_one_workgroup() {
        let n_nodes = 257_u32;
        let n_edges = 1_u32;
        let src = vec![0_u32];
        let dst = vec![256_u32];
        let weight = vec![5_u32];
        let mut dist = vec![u32::MAX; n_nodes as usize];
        dist[0] = 0;
        let max_iterations = 4_u32;

        let (expected, _) = cpu_ref(&src, &dst, &weight, &dist, n_nodes, max_iterations);
        assert_eq!(
            expected[256], 5,
            "Fix: the CPU oracle must relax node 256 to 5 over the single edge."
        );

        let unsound =
            bellman_single_word_harness(FIXTURE, extents(n_nodes, n_edges, max_iterations));
        let (reversed, reversed_flag) = run_bellman(&unsound, true, &dist, &src, &dst, &weight, 1);
        assert_eq!(
            reversed[256],
            u32::MAX,
            "Fix: this test exists to record the OBSERVED wrong value the racing shared flag produces; if the single-word harness stops diverging here, re-derive the defect before deleting this test."
        );
        assert_ne!(
            reversed[256], expected[256],
            "Fix: the pre-routing single-word harness diverges from the CPU oracle at node 256."
        );
        // The verdict the host reads back is a LIE, not merely a miss: the flag says
        // converged while node 256 is still unrelaxed. A test that checked only the
        // flag would pass here, which is why the assertion above is on the output.
        assert_eq!(
            reversed_flag[0], 0,
            "Fix: the shared flag must be observed claiming convergence while the state is unconverged, which is what makes the wrong answer silent."
        );

        // Same program, opposite workgroup order: correct. The output depends on
        // the schedule, which is exactly what makes the shared flag a race and not
        // a deterministic bug.
        let (forward, _) = run_bellman(&unsound, false, &dist, &src, &dst, &weight, 1);
        assert_eq!(
            forward[256], 5,
            "Fix: stepping group 0 first must expose the SAME program as correct, proving the divergence is cross-workgroup ordering."
        );
    }

    /// The routed program is correct under BOTH workgroup orders at the size where
    /// the single-word harness diverges, which is the fix working end to end.
    #[test]
    fn grid_routed_bellman_is_order_independent_where_single_word_diverges() {
        let n_nodes = 257_u32;
        let n_edges = 1_u32;
        let src = vec![0_u32];
        let dst = vec![256_u32];
        let weight = vec![5_u32];
        let mut dist = vec![u32::MAX; n_nodes as usize];
        dist[0] = 0;
        let max_iterations = 4_u32;

        let (expected, _) = cpu_ref(&src, &dst, &weight, &dist, n_nodes, max_iterations);
        let routed = bellman_shortest_path(FIXTURE, extents(n_nodes, n_edges, max_iterations));
        assert_eq!(
            declared_words(&routed, FIXTURE.changed),
            max_iterations,
            "Fix: this size must route to the grid harness."
        );

        for reversed in [false, true] {
            let (actual, _) = run_bellman(
                &routed,
                reversed,
                &dist,
                &src,
                &dst,
                &weight,
                max_iterations,
            );
            assert_eq!(
                actual, expected,
                "Fix: the grid-routed bellman program must match the CPU oracle in both workgroup orders (reversed={reversed})."
            );
        }
    }

    /// OBSERVED divergence in the CANONICAL forward workgroup order, at only FOUR
    /// nodes. This is the case a node-count-only threshold would have missed.
    ///
    /// 4 nodes but 257 edges, so the launch is still two workgroups: `vyre-driver`
    /// spans the largest declared binding, and the edge arrays are the largest here.
    /// Lane 256, in group 1, owns edge 256, the ONLY edge that reaches node 3. It
    /// relaxes `next_dist[3] = 7` correctly. But its compare is gated `t < n_nodes`,
    /// which is `256 < 4`, false: group 1 has no lane that publishes ANY node, so it
    /// cannot set the convergence flag and cannot copy `next_dist` into `dist`.
    ///
    /// So group 1's relaxation is only ever published if some group-0 lane runs a
    /// compare AFTER group 1 relaxed. In forward order group 0 finishes first, sees
    /// `next_dist[3]` still unrelaxed, reads a zero flag and retires; node 3 stays
    /// `u32::MAX` instead of 7 and the flag reports convergence. Reverse the order
    /// and group 0's compare runs after the relaxation and the answer is right.
    /// A threshold keyed on `n_nodes` alone leaves exactly this broken.
    #[test]
    fn single_word_harness_loses_far_edge_relaxations_when_edges_exceed_one_workgroup() {
        let n_nodes = 4_u32;
        let n_edges = 257_u32;
        let mut src = vec![0_u32; n_edges as usize];
        let mut dst = vec![1_u32; n_edges as usize];
        let mut weight = vec![1_u32; n_edges as usize];
        src[256] = 0;
        dst[256] = 3;
        weight[256] = 7;
        let mut dist = vec![u32::MAX; n_nodes as usize];
        dist[0] = 0;
        let max_iterations = 4_u32;

        let (expected, _) = cpu_ref(&src, &dst, &weight, &dist, n_nodes, max_iterations);
        assert_eq!(
            expected,
            vec![0, 1, u32::MAX, 7],
            "Fix: the CPU oracle must reach node 3 at cost 7 through edge 256 and leave node 2 unreachable."
        );

        let unsound =
            bellman_single_word_harness(FIXTURE, extents(n_nodes, n_edges, max_iterations));
        assert_eq!(
            required_workgroups(&unsound),
            2,
            "Fix: 257 edges must still span two workgroups even though there are only 4 nodes."
        );

        let (forward, forward_flag) = run_bellman(&unsound, false, &dist, &src, &dst, &weight, 1);
        let (reversed, _) = run_bellman(&unsound, true, &dist, &src, &dst, &weight, 1);

        assert_eq!(
            forward,
            vec![0, 1, u32::MAX, u32::MAX],
            "Fix: this test records the OBSERVED wrong distances the racing shared flag produces in the canonical order; if the single-word harness stops diverging here, re-derive the defect before deleting this test."
        );
        assert_eq!(
            forward_flag[0], 0,
            "Fix: the shared flag must be observed claiming convergence while node 3 is unreachable, which is what makes the wrong answer silent."
        );
        assert_eq!(
            reversed, expected,
            "Fix: stepping group 1 first must match the oracle, proving the forward-order divergence is cross-workgroup ordering and not a wrong fixture."
        );
    }

    /// The routed program is correct under BOTH workgroup orders at the 4-node,
    /// 257-edge shape where the single-word harness diverges in forward order.
    #[test]
    fn grid_routed_bellman_publishes_far_edge_relaxations_in_both_orders() {
        let n_nodes = 4_u32;
        let n_edges = 257_u32;
        let mut src = vec![0_u32; n_edges as usize];
        let mut dst = vec![1_u32; n_edges as usize];
        let mut weight = vec![1_u32; n_edges as usize];
        src[256] = 0;
        dst[256] = 3;
        weight[256] = 7;
        let mut dist = vec![u32::MAX; n_nodes as usize];
        dist[0] = 0;
        let max_iterations = 4_u32;

        let (expected, _) = cpu_ref(&src, &dst, &weight, &dist, n_nodes, max_iterations);
        let routed = bellman_shortest_path(FIXTURE, extents(n_nodes, n_edges, max_iterations));
        assert_eq!(
            declared_words(&routed, FIXTURE.changed),
            max_iterations,
            "Fix: 257 edges must route to the grid harness even at 4 nodes."
        );

        for reversed in [false, true] {
            let (actual, _) = run_bellman(
                &routed,
                reversed,
                &dist,
                &src,
                &dst,
                &weight,
                max_iterations,
            );
            assert_eq!(
                actual, expected,
                "Fix: the grid-routed program must publish edge 256's relaxation in both workgroup orders (reversed={reversed})."
            );
        }
    }
}
