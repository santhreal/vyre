//! Program construction for iterative Sinkhorn balance.

use vyre_foundation::composition::trap_program;
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Node, Program};

use super::{SinkhornBuffers, SinkhornExtents, OP_ID};
#[cfg(test)]
use crate::fixpoint::persistent_fixpoint::persistent_fixpoint;
use crate::fixpoint::persistent_fixpoint::{routed_persistent_fixpoint, FixpointState};
use crate::math::semiring_gemm::{semiring_gemm, Semiring};
use crate::math::sinkhorn::sinkhorn_scale;

/// Sinkhorn full iteration.
///
/// Runs Sinkhorn matrix-scaling iterations to convergence over the bindings
/// [`SinkhornBuffers`] names, at the extents [`SinkhornExtents`] fixes.
///
/// # Convergence-flag form
///
/// The launch spans `m * n` lanes, not `m`:
/// `dispatch_element_count_for_program`
/// (`vyre-driver/src/program_walks/dispatch_params.rs:19`) sizes an
/// atomic-carrying program's launch from its WIDEST declared buffer, this
/// program carries the harness's `atomic_or`, and the widest buffers are the
/// `m * n` kernel matrices `k` and `k_t`. So the cell count, not the scaling
/// vector length, selects the harness:
///
/// - `m * n <= PERSISTENT_FIXPOINT_WORKGROUP_SIZE[0]`: one workgroup covers
///   the launch, so [`persistent_fixpoint`] runs with its single shared
///   `changed[0]` word. That word is cleared by a plain store fenced only by
///   a workgroup-scope barrier; with one group the fence is incidentally
///   grid-wide, so the clear cannot race the `atomic_or` that sets the flag.
/// - `m * n` above that width: [`persistent_fixpoint_grid`], which never
///   clears the flag, gives each iteration its own `changed` word, and
///   separates waves with `MemoryOrdering::GridSync`. The single-word form is
///   limited to one workgroup precisely because its clear and its set are
///   unordered across groups: group 0's clear can erase another group's set,
///   that group then reads 0 and returns early with an unbalanced scaling
///   vector, and the flag the host reads afterwards reports a convergence no
///   group agreed to.
///
/// A `17 x 17` problem is already 289 cells, so this threshold is crossed at
/// modest sizes with both extents far under one workgroup width.
///
/// [`persistent_fixpoint`]: crate::fixpoint::persistent_fixpoint::persistent_fixpoint
/// [`persistent_fixpoint_grid`]: crate::fixpoint::persistent_fixpoint::persistent_fixpoint_grid
#[must_use]
pub fn sinkhorn_iterate(buffers: SinkhornBuffers<'_>, extents: SinkhornExtents) -> Program {
    let SinkhornExtents {
        m,
        n,
        max_iterations,
    } = extents;
    if m == 0 {
        return trap_program(
            OP_ID,
            Some((buffers.u_curr, DataType::U32)),
            "Fix: sinkhorn_iterate requires m > 0, got 0.".to_string(),
        );
    }
    if n == 0 {
        return trap_program(
            OP_ID,
            Some((buffers.u_curr, DataType::U32)),
            "Fix: sinkhorn_iterate requires n > 0, got 0.".to_string(),
        );
    }
    let Some(matrix_cells) = m.checked_mul(n) else {
        return trap_program(
            OP_ID,
            Some((buffers.u_curr, DataType::U32)),
            format!("Fix: sinkhorn_iterate m*n overflows u32: {m}*{n}."),
        );
    };

    let transfer_body = sinkhorn_transfer_body(buffers, extents);

    // `m` alone does NOT decide the harness: see the form note above. `k` and
    // `k_t` are `m * n` long, which dominates `m` and `n` for any non-zero
    // extents, so the launch spans `matrix_cells` lanes and a modest matrix
    // makes the dispatch multi-workgroup while both extents still fit one group.
    let (inner, route) = routed_persistent_fixpoint(
        transfer_body,
        FixpointState {
            current: buffers.u_curr,
            next: buffers.u_next,
            changed: buffers.changed,
            words: m,
            max_iterations,
        },
        matrix_cells,
    );

    sinkhorn_wrap(&inner, buffers, extents, matrix_cells, route.changed_words)
}

/// One full Sinkhorn sweep: `Kv`, then `u`, then `Ktu`, then `v`.
///
/// Every composed pass is PARTITIONED by global invocation id, not chunked: the
/// gemms gate on `t < out_cells` (`m` for `Kv`, `n` for `Ktu`) and the scales on
/// `t < count`, and each lane sums over the shared dimension internally. So the
/// widest lane gate here is `max(m, n)`, which is what decides whether groups
/// above 0 own any state. It is NOT `m * n`: nothing walks the kernel matrices
/// one cell per lane, so a launch made multi-workgroup only by the size of `k`
/// and `k_t` leaves every gate inside group 0.
///
/// Takes the whole binding record even though the sweep never reads `u_curr` or
/// `changed`: those two belong to the convergence harness, and forwarding the
/// record is what keeps the sweep from restating a second copy of the list.
pub(super) fn sinkhorn_transfer_body(
    buffers: SinkhornBuffers<'_>,
    extents: SinkhornExtents,
) -> Vec<Node> {
    let SinkhornBuffers {
        k,
        k_t,
        a,
        b,
        u_next,
        v,
        kv,
        ktu,
        ..
    } = buffers;
    let SinkhornExtents { m, n, .. } = extents;
    let extract_body = |p: Program| -> Vec<Node> {
        let mut body = Vec::new();
        for node in p.entry() {
            if let Node::Region {
                body: region_body, ..
            } = node
            {
                body.extend(region_body.iter().cloned());
            }
        }
        body
    };
    let seq_cst = || Node::Barrier {
        ordering: vyre_foundation::MemoryOrdering::SeqCst,
    };

    let mut transfer_body = Vec::new();

    // 1. Kv = K * v (m x n * n x 1 -> m x 1)
    transfer_body.extend(extract_body(semiring_gemm(
        k,
        v,
        kv,
        m,
        1,
        n,
        Semiring::Real,
    )));
    transfer_body.push(seq_cst());

    // 2. u_next = a ./ Kv
    transfer_body.extend(extract_body(sinkhorn_scale(a, kv, u_next, m)));
    transfer_body.push(seq_cst());

    // 3. Ktu = K_T * u_next (n x m * m x 1 -> n x 1)
    transfer_body.extend(extract_body(semiring_gemm(
        k_t,
        u_next,
        ktu,
        n,
        1,
        m,
        Semiring::Real,
    )));
    transfer_body.push(seq_cst());

    // 4. v = b ./ Ktu
    transfer_body.extend(extract_body(sinkhorn_scale(b, ktu, v, n)));
    transfer_body.push(seq_cst());

    transfer_body
}

/// Wrap a convergence harness in the Sinkhorn Region and buffer declarations.
///
/// Single owner of those ten declarations, so the two routed forms and the
/// single-word form the divergence test builds cannot drift apart in binding
/// order, counts, or access modes.
pub(super) fn sinkhorn_wrap(
    inner: &Program,
    buffers: SinkhornBuffers<'_>,
    extents: SinkhornExtents,
    matrix_cells: u32,
    changed_words: u32,
) -> Program {
    let SinkhornBuffers {
        k,
        k_t,
        a,
        b,
        u_curr,
        u_next,
        v,
        kv,
        ktu,
        changed,
    } = buffers;
    let SinkhornExtents { m, n, .. } = extents;
    crate::math::wrap_fixpoint_program(
        OP_ID,
        inner,
        vec![
            BufferDecl::storage(u_curr, 0, BufferAccess::ReadWrite, DataType::U32).with_count(m),
            BufferDecl::storage(u_next, 1, BufferAccess::ReadWrite, DataType::U32).with_count(m),
            BufferDecl::storage(changed, 2, BufferAccess::ReadWrite, DataType::U32)
                .with_count(changed_words),
            BufferDecl::storage(k, 3, BufferAccess::ReadOnly, DataType::U32)
                .with_count(matrix_cells),
            BufferDecl::storage(k_t, 4, BufferAccess::ReadOnly, DataType::U32)
                .with_count(matrix_cells),
            BufferDecl::storage(a, 5, BufferAccess::ReadOnly, DataType::U32).with_count(m),
            BufferDecl::storage(b, 6, BufferAccess::ReadOnly, DataType::U32).with_count(n),
            BufferDecl::storage(v, 7, BufferAccess::ReadWrite, DataType::U32).with_count(n),
            BufferDecl::storage(kv, 8, BufferAccess::ReadWrite, DataType::U32).with_count(m),
            BufferDecl::storage(ktu, 9, BufferAccess::ReadWrite, DataType::U32).with_count(n),
        ],
    )
}

/// The pre-routing program: the Sinkhorn transfer body on the single-word
/// convergence harness at ANY size, which is exactly what [`sinkhorn_iterate`]
/// emitted before the dispatch-span routing landed.
///
/// Exists only so the divergence test can OBSERVE what the racing shared flag
/// produces above one workgroup. Production code must never take this path above
/// one workgroup width.
#[cfg(test)]
pub(super) fn sinkhorn_single_word_harness(
    buffers: SinkhornBuffers<'_>,
    extents: SinkhornExtents,
) -> Program {
    let matrix_cells = extents
        .m
        .checked_mul(extents.n)
        .expect("Fix: the divergence fixture must use non-overflowing extents.");
    let transfer_body = sinkhorn_transfer_body(buffers, extents);
    let inner = persistent_fixpoint(
        transfer_body,
        buffers.u_curr,
        buffers.u_next,
        buffers.changed,
        extents.m,
        extents.max_iterations,
    );
    sinkhorn_wrap(&inner, buffers, extents, matrix_cells, 1)
}
