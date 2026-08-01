use super::*;
use std::sync::Arc;

#[test]
fn test_sinkhorn_cpu_ref_trivial() {
    let (u, v, _iters) = cpu_ref(
        &[65536],
        &[65536],
        &[65536],
        &[65536],
        &[65536],
        &[65536],
        1,
        1,
        10,
    );
    assert_eq!(u, vec![65536]);
    assert_eq!(v, vec![65536]);
}

#[test]
fn test_sinkhorn_cpu_ref_edge() {
    // u = a / (k * v) = 65536 / (32768 * 65536) = 65536 / 2^31 = 0
    let (u, _, _) = cpu_ref(
        &[32768],
        &[32768],
        &[65536],
        &[65536],
        &[65536],
        &[65536],
        1,
        1,
        10,
    );
    assert_eq!(u, vec![0]);
}

#[test]
fn test_sinkhorn_cpu_ref_normal() {
    let k = vec![65536, 65536, 65536, 65536];
    let k_t = vec![65536, 65536, 65536, 65536];
    let a = vec![32768, 32768];
    let b = vec![32768, 32768];
    let u_c = vec![65536, 65536];
    let v_in = vec![65536, 65536];
    let (u, _v, _) = cpu_ref(&k, &k_t, &a, &b, &u_c, &v_in, 2, 2, 5);
    // Kv = [0, 0] wrapped. u = a/1 = 32768.
    assert_eq!(u, vec![32768, 32768]);
}

#[test]
fn test_sinkhorn_cpu_ref_large() {
    let k = vec![65536; 9];
    let a = vec![65536; 3];
    let b = vec![65536; 3];
    let u_c = vec![65536; 3];
    let v_in = vec![65536; 3];
    let (u, _, _) = cpu_ref(&k, &k, &a, &b, &u_c, &v_in, 3, 3, 5);
    assert_eq!(u.len(), 3);
}

#[test]
fn test_sinkhorn_cpu_ref_asym() {
    let k = vec![65536, 0, 0, 65536, 65536, 65536];
    let k_t = vec![65536, 0, 65536, 0, 65536, 65536];
    let a = vec![32768, 32768, 65536];
    let b = vec![65536, 65536];
    let u_c = vec![65536, 65536, 65536];
    let v_in = vec![65536, 65536];
    let (u, v, _) = cpu_ref(&k, &k_t, &a, &b, &u_c, &v_in, 3, 2, 5);
    assert_eq!(u.len(), 3);
    assert_eq!(v.len(), 2);
}

#[test]
fn test_sinkhorn_cpu_ref_into_reuses_buffers() {
    let k = vec![65536, 65536, 65536, 65536];
    let a = vec![32768, 32768];
    let b = vec![32768, 32768];
    let u_c = vec![65536, 65536];
    let v_in = vec![65536, 65536];
    let mut u = Vec::with_capacity(8);
    let mut v = Vec::with_capacity(8);
    let mut u_old = Vec::with_capacity(8);
    let u_ptr = u.as_ptr();
    let v_ptr = v.as_ptr();
    let old_ptr = u_old.as_ptr();
    let _iters = cpu_ref_into(
        &k, &k, &a, &b, &u_c, &v_in, 2, 2, 5, &mut u, &mut v, &mut u_old,
    );
    assert_eq!(u, vec![32768, 32768]);
    assert_eq!(u.as_ptr(), u_ptr);
    assert_eq!(v.as_ptr(), v_ptr);
    assert_eq!(u_old.as_ptr(), old_ptr);
}

#[test]
fn test_sinkhorn_cpu_ref_into_truncates_stale_buffers() {
    let k = vec![65536, 65536, 65536, 65536];
    let a = vec![32768, 32768];
    let b = vec![32768, 32768];
    let u_c = vec![65536, 65536];
    let v_in = vec![65536, 65536];
    let mut u = Vec::with_capacity(8);
    let mut v = Vec::with_capacity(8);
    let mut u_old = Vec::with_capacity(8);
    u.extend([99u32; 8]);
    v.extend([99u32; 8]);
    u_old.extend([99u32; 8]);
    let u_ptr = u.as_ptr();
    let v_ptr = v.as_ptr();
    let old_ptr = u_old.as_ptr();

    let _iters = try_cpu_ref_into(
        &k, &k, &a, &b, &u_c, &v_in, 2, 2, 5, &mut u, &mut v, &mut u_old,
    )
    .unwrap();

    assert_eq!(u, vec![32768, 32768]);
    assert_eq!(u.as_ptr(), u_ptr);
    assert_eq!(v.as_ptr(), v_ptr);
    assert_eq!(u_old.as_ptr(), old_ptr);
}

#[test]
fn test_sinkhorn_try_cpu_ref_rejects_short_buffers() {
    let err = try_cpu_ref(&[1], &[1], &[1, 1], &[1, 1], &[1, 1], &[1, 1], 2, 2, 1).unwrap_err();
    assert!(err.contains("buffer `k` is too short"), "{err}");
}

#[test]
fn test_sinkhorn_program_parity() {
    let k = vec![1, 1, 1, 1];
    let a = vec![10, 10];
    let b = vec![10, 10];
    let u_c = vec![1, 1];
    let v_in = vec![1, 1];

    let p = sinkhorn_iterate(
        "k", "kt", "a", "b", "uc", "un", "v", "kv", "ktu", "c", 2, 2, 1,
    );

    let (expected_u, _, _) = cpu_ref(&k, &k, &a, &b, &u_c, &v_in, 2, 2, 1);

    use vyre_reference::reference_eval;
    use vyre_reference::value::Value;

    let to_value = |data: &[u32]| {
        let bytes = crate::wire::pack_u32_slice(data);
        Value::Bytes(Arc::from(bytes))
    };

    let inputs = vec![
        to_value(&u_c),
        to_value(&[0_u32, 0]),
        to_value(&[0]),
        to_value(&k),
        to_value(&k),
        to_value(&a),
        to_value(&b),
        to_value(&v_in),
        to_value(&[0_u32, 0]),
        to_value(&[0_u32, 0]),
    ];

    let results = reference_eval(&p, &inputs).expect("Fix: interpreter failed");
    let actual_bytes = results[0].to_bytes();
    let actual_u: Vec<u32> = actual_bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    assert_eq!(actual_u, expected_u);
}

#[test]
fn program_declares_ten_buffers() {
    let p = sinkhorn_iterate(
        "k", "kt", "a", "b", "uc", "un", "v", "kv", "ktu", "c", 2, 2, 5,
    );
    assert_eq!(p.buffers().len(), 10);
}

/// Workgroups a host must launch to cover `program`.
///
/// `sinkhorn_iterate` emits the convergence flag's `atomic_or`, and for an
/// atomic-carrying program `vyre-driver`'s `dispatch_element_count_for_program`
/// spans the LARGEST declared buffer rather than just the output buffer. The
/// widest buffers here are the `m * n` kernel matrices `k` and `k_t`, so the
/// launch width is `m * n` rounded up to whole workgroups, not `m`.
fn required_workgroups(program: &Program) -> u32 {
    let elements = program
        .buffers()
        .iter()
        .map(|buffer| buffer.count())
        .max()
        .unwrap_or(1);
    elements.div_ceil(program.workgroup_size()[0])
}

/// Declared word count of the convergence-flag buffer (always named `"c"` here).
fn changed_words(program: &Program) -> u32 {
    program
        .buffers()
        .iter()
        .find(|buffer| buffer.name() == "c")
        .expect("Fix: sinkhorn_iterate must declare its convergence-flag buffer.")
        .count()
}

/// Locks out the multi-workgroup convergence-flag race.
///
/// `persistent_fixpoint` keeps ONE `changed[0]` word, clears it from global lane 0
/// with a plain store, and orders that clear against every other lane's
/// `atomic_or` with a workgroup-scoped `SeqCst` barrier only. Once the launch spans
/// more than one workgroup nothing orders the clear against the sets: workgroup 0's
/// next clear can erase workgroup 1's set, so workgroup 1 reads 0 and `Return`s
/// with an unbalanced scaling vector, and the post-dispatch flag read reports a
/// convergence verdict no group agreed to. A multi-workgroup build must therefore
/// never be handed one shared cleared word.
#[test]
fn multi_workgroup_sinkhorn_never_shares_one_cleared_convergence_word() {
    let program = sinkhorn_iterate(
        "k", "kt", "a", "b", "uc", "un", "v", "kv", "ktu", "c", 257, 1, 8,
    );

    assert_eq!(
        required_workgroups(&program),
        2,
        "Fix: a 257-element scaling vector over a 256-wide workgroup must need two workgroups."
    );
    assert_eq!(
        changed_words(&program),
        8,
        "Fix: a multi-workgroup sinkhorn dispatch must use the per-iteration convergence-word protocol, not one shared cleared word."
    );
}

/// Grid-wide fences in `nodes`, counted through every nesting construct. The
/// transfer body's own `MemoryOrdering::SeqCst` barriers are workgroup scope and
/// deliberately not counted here.
fn count_grid_sync(nodes: &[Node]) -> usize {
    nodes
        .iter()
        .map(|node| match node {
            Node::Barrier {
                ordering: vyre_foundation::MemoryOrdering::GridSync,
            } => 1,
            Node::If {
                then, otherwise, ..
            } => count_grid_sync(then) + count_grid_sync(otherwise),
            Node::Loop { body, .. } | Node::Block(body) => count_grid_sync(body),
            Node::Region { body, .. } => count_grid_sync(body),
            _ => 0,
        })
        .sum()
}

/// Pins the routing threshold to the declared workgroup width.
///
/// The threshold is `> PERSISTENT_FIXPOINT_WORKGROUP_SIZE[0]`, read from the same
/// constant this program declares as its workgroup size, so the two can never
/// drift apart. At exactly that many kernel cells the launch is one workgroup and
/// the compact single-word protocol is sound, so it stays in use; one cell past it
/// the launch is two workgroups and must switch. An off-by-one here puts a
/// multi-workgroup dispatch back on the racing flag.
#[test]
fn routing_threshold_is_the_declared_workgroup_width() {
    let width = PERSISTENT_FIXPOINT_WORKGROUP_SIZE[0];

    let at_width = sinkhorn_iterate(
        "k", "kt", "a", "b", "uc", "un", "v", "kv", "ktu", "c", width, 1, 8,
    );
    assert_eq!(
        at_width.workgroup_size(),
        PERSISTENT_FIXPOINT_WORKGROUP_SIZE
    );
    assert_eq!(required_workgroups(&at_width), 1);
    assert_eq!(
        changed_words(&at_width),
        1,
        "Fix: a single-workgroup launch must keep the compact one-word convergence flag."
    );

    let past_width = sinkhorn_iterate(
        "k",
        "kt",
        "a",
        "b",
        "uc",
        "un",
        "v",
        "kv",
        "ktu",
        "c",
        width + 1,
        1,
        8,
    );
    assert_eq!(required_workgroups(&past_width), 2);
    assert_eq!(
        changed_words(&past_width),
        8,
        "Fix: one cell past the workgroup width already needs the per-iteration convergence words."
    );
}

/// The routing threshold is the DISPATCH SPAN, not the element count fed to the
/// fixpoint.
///
/// `dispatch_element_count_for_program`
/// (`vyre-driver/src/program_walks/dispatch_params.rs:19`) forces a full span over
/// every non-shared binding once a program contains atomics, and this one holds the
/// harness's `atomic_or`. Full span means the LARGEST declared buffer, and the
/// widest buffers here are the `m * n` kernel matrices, so a `17 x 17` problem is
/// 289 cells and already a two-workgroup dispatch with BOTH extents an order of
/// magnitude under one workgroup width. Routing on `m` alone would leave every
/// such modest matrix on the racing single-word flag: this is the case that makes
/// the defect bite at ordinary sizes rather than large ones.
#[test]
fn modest_square_matrix_with_tiny_extents_still_routes_to_the_grid_form() {
    let program = sinkhorn_iterate(
        "k", "kt", "a", "b", "uc", "un", "v", "kv", "ktu", "c", 17, 17, 8,
    );

    assert_eq!(
        matrix_cells(&program),
        289,
        "Fix: a 17 by 17 kernel is 289 cells, past a 256-wide workgroup."
    );
    assert_eq!(
        required_workgroups(&program),
        2,
        "Fix: 289 kernel cells over a 256-wide workgroup must need two workgroups."
    );
    assert_eq!(
        changed_words(&program),
        8,
        "Fix: the routing threshold must be the dispatch span (max declared buffer), not m."
    );
    assert!(
        count_grid_sync(program.entry()) > 0,
        "Fix: a two-workgroup dispatch must be grid-synchronized whichever buffer widened it."
    );
}

/// Declared cell count of the `k` kernel matrix.
fn matrix_cells(program: &Program) -> u32 {
    program
        .buffers()
        .iter()
        .find(|buffer| buffer.name() == "k")
        .expect("Fix: sinkhorn_iterate must declare its kernel matrix.")
        .count()
}

/// The two routes must not silently converge to the same emission.
///
/// The grid form's soundness IS its `MemoryOrdering::GridSync` fences: they order
/// the per-iteration flag write against every group's read. The single-workgroup
/// form must carry none of them, because emitting one there would impose a
/// cooperative launch on a dispatch that does not need it. If both routes ever emit
/// the same barrier set, the routing has stopped selecting anything.
#[test]
fn grid_route_fences_the_grid_and_single_workgroup_route_does_not() {
    let width = PERSISTENT_FIXPOINT_WORKGROUP_SIZE[0];

    let single = sinkhorn_iterate(
        "k", "kt", "a", "b", "uc", "un", "v", "kv", "ktu", "c", width, 1, 4,
    );
    assert_eq!(
        count_grid_sync(single.entry()),
        0,
        "Fix: a single-workgroup sinkhorn program must not force a cooperative grid launch."
    );

    let grid = sinkhorn_iterate(
        "k", "kt", "a", "b", "uc", "un", "v", "kv", "ktu", "c", 17, 17, 4,
    );
    assert_eq!(
        count_grid_sync(grid.entry()),
        8,
        "Fix: the grid form must fence each of its 4 waves twice, once after the transfer step and once after the compare."
    );
}

/// The grid form indexes `changed[iteration]`, so a one-word buffer there would be
/// an out-of-bounds atomic write on iteration 1. This wrapper re-declares `changed`
/// itself, so its count must equal the count the harness declares, for every
/// iteration budget.
#[test]
fn grid_route_sizes_changed_to_one_word_per_iteration() {
    for max_iterations in [1_u32, 2, 8, 64] {
        let program = sinkhorn_iterate(
            "k",
            "kt",
            "a",
            "b",
            "uc",
            "un",
            "v",
            "kv",
            "ktu",
            "c",
            17,
            17,
            max_iterations,
        );
        let harness = persistent_fixpoint_grid(Vec::new(), "uc", "un", "c", 17, max_iterations);

        assert_eq!(
            changed_words(&program),
            max_iterations,
            "Fix: the grid route needs one convergence word per iteration; {max_iterations} iterations need {max_iterations} words."
        );
        assert_eq!(
            changed_words(&program),
            changed_words(&harness),
            "Fix: this wrapper's `changed` declaration must match persistent_fixpoint_grid's own."
        );
    }
}

/// Sinkhorn fixture whose `u` vector actually EVOLVES across iterations, so an
/// element frozen by an early retire is distinguishable from one that ran to
/// convergence.
///
/// Every kernel cell is 1, so `u[i] = a / (n * v)` and `v[j] = b / (m * u)`. With
/// `a = 4n` and `b = 12m` the sweep walks `u = 4 -> 1 -> 0` before settling. A
/// uniform fixture converges in one step and hides the divergence behind a value
/// that is correct by coincidence, which is exactly how this test first passed
/// when it should not have.
///
/// Returns `(k, a, b, u_curr, v)`.
fn evolving_sinkhorn_fixture(m: u32, n: u32) -> (Vec<u32>, Vec<u32>, Vec<u32>, Vec<u32>, Vec<u32>) {
    let cells = (m as usize) * (n as usize);
    (
        vec![1_u32; cells],
        vec![4 * n; m as usize],
        vec![12 * m; n as usize],
        vec![1_u32; m as usize],
        vec![1_u32; n as usize],
    )
}

/// Run `program` on the reference interpreter and return the final `u_curr`
/// vector paired with the final `changed` words. `reversed` steps the workgroups
/// back to front; both orders are schedules real hardware is free to pick,
/// because nothing in the IR orders one workgroup against another.
#[allow(clippy::too_many_arguments)]
fn run_sinkhorn(
    program: &Program,
    reversed: bool,
    k: &[u32],
    a: &[u32],
    b: &[u32],
    u_curr: &[u32],
    v: &[u32],
    changed_words: u32,
) -> (Vec<u32>, Vec<u32>) {
    use vyre_reference::value::Value;

    let to_value = |data: &[u32]| Value::Bytes(Arc::from(crate::wire::pack_u32_slice(data)));
    let inputs = vec![
        to_value(u_curr),
        to_value(&vec![0_u32; u_curr.len()]),
        to_value(&vec![0_u32; changed_words as usize]),
        to_value(k),
        to_value(k),
        to_value(a),
        to_value(b),
        to_value(v),
        to_value(&vec![0_u32; a.len()]),
        to_value(&vec![0_u32; b.len()]),
    ];
    let results = if reversed {
        vyre_reference::reference_eval_lane_reversed(program, &inputs)
    } else {
        vyre_reference::reference_eval(program, &inputs)
    }
    .expect("Fix: the reference interpreter must execute the sinkhorn program.");
    let decode = |value: &vyre_reference::value::Value| -> Vec<u32> {
        value
            .to_bytes()
            .chunks_exact(4)
            .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()))
            .collect()
    };
    (decode(&results[0]), decode(&results[2]))
}

/// OBSERVED divergence: the pre-routing single-word harness returns a WRONG
/// scaling vector once the LANE GATES, not merely the kernel matrix, exceed one
/// workgroup.
///
/// `m = 257`, `n = 1`. Every gate in the sweep is `t < m` or `t < n`, so at
/// `m = 257` lane 256 sits in group 1 and is the sole owner of `u_curr[256]`:
/// the only lane that computes it, the only lane that can set the convergence
/// flag for it. Nothing orders group 1 against group 0's clear of that flag.
/// Stepping the groups in the two orders the IR permits gives two different
/// answers for element 256, which is the race.
///
/// Note the contrast with `17 x 17`: there the kernel matrix is 289 cells so the
/// launch is also two workgroups, but every gate is `t < 17`, group 1 owns
/// nothing, and the output is correct. The dispatch span enables the second
/// group; only the gate bound decides whether it can corrupt anything.
#[test]
fn single_word_harness_returns_a_wrong_scaling_vector_above_one_workgroup() {
    let m = 257_u32;
    let n = 1_u32;
    let max_iterations = 4_u32;
    let (k, a, b, u_curr, v) = evolving_sinkhorn_fixture(m, n);

    let (expected_u, _, _) = cpu_ref(&k, &k, &a, &b, &u_curr, &v, m, n, max_iterations);

    let unsound = sinkhorn_single_word_harness(
        "k",
        "kt",
        "a",
        "b",
        "uc",
        "un",
        "v",
        "kv",
        "ktu",
        "c",
        m,
        n,
        max_iterations,
    );
    let (forward, _) = run_sinkhorn(&unsound, false, &k, &a, &b, &u_curr, &v, 1);
    let (reversed, reversed_flag) = run_sinkhorn(&unsound, true, &k, &a, &b, &u_curr, &v, 1);
    // Exact observed values, not a shape check. The oracle settles element 256 at
    // 0 after the `4 -> 1 -> 0` walk. Stepping group 0 first reproduces that.
    // Stepping group 1 first freezes element 256 at 4, the value it had after the
    // very first sweep against the initial `v`, because group 1 read the shared
    // flag before group 0 had ever set it and retired for good.
    assert_eq!(
        expected_u[256], 0,
        "Fix: the CPU oracle must settle element 256 at 0 for this fixture."
    );
    assert_eq!(
        forward[256], 0,
        "Fix: stepping group 0 first must expose the SAME program as correct, proving the divergence is cross-workgroup ordering."
    );
    assert_eq!(
        reversed[256],
        4,
        "Fix: this test records the OBSERVED wrong value (4, the first-sweep value) the racing shared flag produces; if the single-word harness stops diverging here, re-derive the defect before deleting this test."
    );
    assert_ne!(
        reversed[256], expected_u[256],
        "Fix: the workgroup order the IR permits must be observed disagreeing with the CPU oracle at element 256."
    );
    assert_eq!(
        reversed_flag[0], 0,
        "Fix: the shared flag must be observed claiming convergence while element 256 is wrong, which is what makes the wrong answer silent."
    );
}

/// The routed program is correct under BOTH workgroup orders at the size where
/// the single-word harness diverges, which is the fix working end to end.
#[test]
fn grid_routed_sinkhorn_is_order_independent_where_single_word_diverges() {
    let m = 257_u32;
    let n = 1_u32;
    let max_iterations = 4_u32;
    let (k, a, b, u_curr, v) = evolving_sinkhorn_fixture(m, n);

    let (expected_u, _, _) = cpu_ref(&k, &k, &a, &b, &u_curr, &v, m, n, max_iterations);
    let routed = sinkhorn_iterate(
        "k",
        "kt",
        "a",
        "b",
        "uc",
        "un",
        "v",
        "kv",
        "ktu",
        "c",
        m,
        n,
        max_iterations,
    );
    assert_eq!(
        changed_words(&routed),
        max_iterations,
        "Fix: this size must route to the grid harness."
    );

    for reversed in [false, true] {
        let (actual, _) = run_sinkhorn(&routed, reversed, &k, &a, &b, &u_curr, &v, max_iterations);
        assert_eq!(
            actual, expected_u,
            "Fix: the grid-routed sinkhorn program must match the CPU oracle in both workgroup orders (reversed={reversed})."
        );
    }
}

/// A `17 x 17` Sinkhorn is CORRECT on the pre-routing single-word harness, which
/// is why the routing threshold and the behavioral threshold are different
/// numbers and must be described separately.
///
/// The kernel matrix is 289 cells, so the launch really is two workgroups and the
/// unsound combination (one shared cleared word, multi-workgroup span) really is
/// present. But every lane gate in the sweep is `t < 17`, so group 1 executes
/// nothing: the clear and every set stay inside group 0 and the workgroup-scope
/// barrier is sufficient by accident. This test exists so nobody upgrades
/// "unsound by construction at `m * n > 256`" into "returns wrong results at
/// `m * n > 256`", which is not true.
#[test]
fn modest_square_matrix_is_correct_on_the_single_word_harness_despite_two_workgroups() {
    let m = 17_u32;
    let n = 17_u32;
    let max_iterations = 4_u32;
    let (k, a, b, u_curr, v) = evolving_sinkhorn_fixture(m, n);

    let (expected_u, _, _) = cpu_ref(&k, &k, &a, &b, &u_curr, &v, m, n, max_iterations);
    let unsound = sinkhorn_single_word_harness(
        "k",
        "kt",
        "a",
        "b",
        "uc",
        "un",
        "v",
        "kv",
        "ktu",
        "c",
        m,
        n,
        max_iterations,
    );

    assert_eq!(
        required_workgroups(&unsound),
        2,
        "Fix: 289 kernel cells must still span two workgroups, so the unsound combination is present."
    );
    for reversed in [false, true] {
        let (actual, _) = run_sinkhorn(&unsound, reversed, &k, &a, &b, &u_curr, &v, 1);
        assert_eq!(
            actual, expected_u,
            "Fix: with every lane gate at `t < 17` the defect is masked and 17x17 must agree with the oracle (reversed={reversed})."
        );
    }
}
