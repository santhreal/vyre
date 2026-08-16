use super::program::{sinkhorn_single_word_harness, sinkhorn_transfer_body, sinkhorn_wrap};
use super::*;
use crate::fixpoint::persistent_fixpoint::{
    declared_words, fixpoint_route, required_workgroups, routed_persistent_fixpoint, FixpointState,
    PERSISTENT_FIXPOINT_WORKGROUP_SIZE,
};
use std::sync::Arc;
use vyre_foundation::ir::Program;

/// The binding names every program test in this module builds against.
const FIXTURE: SinkhornBuffers<'static> = SinkhornBuffers::CANONICAL;

/// `SinkhornExtents` for `m x n` under an iteration cap.
fn extents(m: u32, n: u32, max_iterations: u32) -> SinkhornExtents {
    SinkhornExtents {
        m,
        n,
        max_iterations,
    }
}

/// The emitted binding at `index`, as `(name, count)`.
fn binding(program: &Program, index: usize) -> (&str, u32) {
    let buffer = &program.buffers()[index];
    (buffer.name(), buffer.count())
}

/// Each named field must reach the binding that carries its role.
///
/// All ten fields are `&str`, so the type system cannot reject a transposition;
/// what it can do is make one observable. This pins field to (binding index,
/// declared count) for the whole record, so a swap of any two names moves a name
/// to a binding whose count belongs to the other role and fails here. The counts
/// are deliberately all distinct except where two roles genuinely share a length,
/// and `m != n` so the `m`-length and `n`-length roles cannot be confused.
#[test]
fn every_named_binding_reaches_the_role_it_names() {
    let program = sinkhorn_iterate(FIXTURE, extents(3, 5, 1));

    assert_eq!(binding(&program, 0), (FIXTURE.u_curr, 3));
    assert_eq!(binding(&program, 1), (FIXTURE.u_next, 3));
    assert_eq!(binding(&program, 2), (FIXTURE.changed, 1));
    assert_eq!(binding(&program, 3), (FIXTURE.k, 15));
    assert_eq!(binding(&program, 4), (FIXTURE.k_t, 15));
    assert_eq!(binding(&program, 5), (FIXTURE.a, 3));
    assert_eq!(binding(&program, 6), (FIXTURE.b, 5));
    assert_eq!(binding(&program, 7), (FIXTURE.v, 5));
    assert_eq!(binding(&program, 8), (FIXTURE.kv, 3));
    assert_eq!(binding(&program, 9), (FIXTURE.ktu, 5));
}

/// Transposing two names must change the emitted program.
///
/// The old positional call took the ten names in any order and emitted a program
/// that differed only in which label sat on which binding, which no test read.
/// The crate's own IR parity test passed them in binding order for exactly that
/// reason and named the kernel matrix `u_curr` without failing anything. Each
/// pair below is a swap that leaves the argument COUNT and every type valid, so
/// it is precisely what a positional call could not catch; the wire encoding must
/// differ for every one of them.
#[test]
fn transposing_two_binding_names_changes_the_wire_encoding() {
    let extents = extents(3, 5, 1);
    let canonical = to_wire(&sinkhorn_iterate(FIXTURE, extents));

    let kernel_swapped = SinkhornBuffers {
        k: FIXTURE.k_t,
        k_t: FIXTURE.k,
        ..FIXTURE
    };
    let pingpong_swapped = SinkhornBuffers {
        u_curr: FIXTURE.u_next,
        u_next: FIXTURE.u_curr,
        ..FIXTURE
    };
    let scratch_swapped = SinkhornBuffers {
        kv: FIXTURE.ktu,
        ktu: FIXTURE.kv,
        ..FIXTURE
    };
    let marginal_swapped = SinkhornBuffers {
        a: FIXTURE.b,
        b: FIXTURE.a,
        ..FIXTURE
    };

    for (label, transposed) in [
        ("k / k_t", kernel_swapped),
        ("u_curr / u_next", pingpong_swapped),
        ("kv / ktu", scratch_swapped),
        ("a / b", marginal_swapped),
    ] {
        assert_ne!(
            to_wire(&sinkhorn_iterate(transposed, extents)),
            canonical,
            "Fix: transposing {label} must change the emitted program, or the two names are interchangeable and one of them is dead."
        );
    }
}

/// Wire encoding of `program`, the comparison every collapse in this module is
/// proved on. A debug string would compare formatting rather than emission.
fn to_wire(program: &Program) -> Vec<u8> {
    vyre_foundation::serial::wire::encode::to_wire(program)
        .expect("Fix: a sinkhorn program must encode to the wire form.")
}

/// Routing through the shared fixpoint owner must not change the emission.
///
/// This op used to re-derive the harness selection and the matching `changed`
/// width itself. Both sides of the routing threshold, and the flag width each
/// side needs, are pinned here on the wire encoding so the delegation is provably
/// behavior-preserving rather than plausibly so.
#[test]
fn routing_matches_the_shared_fixpoint_owner_on_both_sides_of_the_threshold() {
    let width = PERSISTENT_FIXPOINT_WORKGROUP_SIZE[0];

    for (m, n, max_iterations) in [(2, 2, 5), (16, 16, 8), (17, 17, 8), (width, 1, 4)] {
        let extents = extents(m, n, max_iterations);
        let matrix_cells = m * n;
        let route = fixpoint_route(matrix_cells, max_iterations);
        let program = sinkhorn_iterate(FIXTURE, extents);

        assert_eq!(
            binding(&program, 2).1,
            route.changed_words,
            "Fix: the declared convergence-flag width must be the width the routed harness indexes."
        );

        let expected = sinkhorn_wrap(
            &routed_persistent_fixpoint(
                sinkhorn_transfer_body(FIXTURE, extents),
                FixpointState {
                    current: FIXTURE.u_curr,
                    next: FIXTURE.u_next,
                    changed: FIXTURE.changed,
                    words: m,
                    max_iterations,
                },
                matrix_cells,
            )
            .0,
            FIXTURE,
            extents,
            matrix_cells,
            route.changed_words,
        );
        assert_eq!(
            to_wire(&program),
            to_wire(&expected),
            "Fix: sinkhorn_iterate must emit exactly what the routed harness plus its own wrapper produce."
        );
    }
}

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

    let p = sinkhorn_iterate(FIXTURE, extents(2, 2, 1));

    let (expected_u, _, _) = cpu_ref(&k, &k, &a, &b, &u_c, &v_in, 2, 2, 1);

    use vyre_reference::reference_eval;
    use vyre_reference::value::Value;

    let to_value = |data: &[u32]| {
        let bytes = vyre_primitives::wire::pack_u32_slice(data);
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
    let p = sinkhorn_iterate(FIXTURE, extents(2, 2, 5));
    assert_eq!(p.buffers().len(), 10);
}

/// Every way a Sinkhorn launch outgrows one workgroup, each registered against
/// the shared routing contract.
///
/// The span is the widest declared buffer, so a long thin scaling vector and a
/// modest square kernel reach it differently: 257 scaling entries widen `u`/`un`
/// directly, while 17 by 17 leaves both scaling vectors inside one workgroup and
/// widens only the 289-cell kernel matrix. Routing on the ping-pong state width
/// would leave the square case on the racing single-word flag.
fn routed_forms() -> [(&'static str, u32, u32); 2] {
    let width = PERSISTENT_FIXPOINT_WORKGROUP_SIZE[0];
    [
        ("sinkhorn_iterate, scaling-widened span", width + 1, 1),
        ("sinkhorn_iterate, kernel-widened span", 17, 17),
    ]
}

#[test]
fn every_routed_form_obeys_the_persistent_fixpoint_routing_contract() {
    let width = PERSISTENT_FIXPOINT_WORKGROUP_SIZE[0];
    for (name, m, n) in routed_forms() {
        crate::fixpoint::routing_contract::assert_routes_on_dispatch_span(
            &crate::fixpoint::routing_contract::RoutedFixpointOp {
                name,
                changed: FIXTURE.changed,
                at_one_workgroup: &|max_iterations| {
                    sinkhorn_iterate(FIXTURE, extents(width, 1, max_iterations))
                },
                past_one_workgroup: &|max_iterations| {
                    sinkhorn_iterate(FIXTURE, extents(m, n, max_iterations))
                },
                grid_harness: &|max_iterations| {
                    crate::fixpoint::routing_contract::bare_grid_harness(
                        FIXTURE.u_curr,
                        FIXTURE.u_next,
                        FIXTURE.changed,
                        m,
                        max_iterations,
                    )
                },
            },
        );
    }
}

type SinkhornFixture = (Vec<u32>, Vec<u32>, Vec<u32>, Vec<u32>, Vec<u32>);

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
fn evolving_sinkhorn_fixture(m: u32, n: u32) -> SinkhornFixture {
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

    let to_value = |data: &[u32]| Value::Bytes(Arc::from(vyre_primitives::wire::pack_u32_slice(data)));
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

    let unsound = sinkhorn_single_word_harness(FIXTURE, extents(m, n, max_iterations));
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
    let routed = sinkhorn_iterate(FIXTURE, extents(m, n, max_iterations));
    assert_eq!(
        declared_words(&routed, FIXTURE.changed),
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
    let unsound = sinkhorn_single_word_harness(FIXTURE, extents(m, n, max_iterations));

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
