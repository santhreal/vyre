//! Every registered single-point program must guard on the logical point.
//!
//! Defect class this closes: a program that keeps a running result in
//! read-write storage and admits that body with `LogicalTileId.x == 0` alone.
//! The guard is exact only while the tile holds one point. Fusion can widen an
//! arm, so the same body then runs once per added point over the same slots, and
//! the result is wrong in a way only a differential sees: the buffer is
//! populated and the shapes match.
//!
//! Measured before this was closed: `nn::top_k` fused behind a 256-wide
//! elementwise arm named one input lane twice, on one host and not another.
//! `math::fft::fft4_complex` and `math::fft::scale_conjugate_inverse` carried
//! the same shape and were fused by two named contracts that passed.
//!
//! The variant space is the registry, walked at run time, so a new operation
//! with this shape turns this red instead of waiting for a differential to
//! catch it on one machine.

use vyre_foundation::execution_plan::fusion::relies_on_single_invocation_workgroup;
use vyre_libs::operation_catalog::all_entries;

/// Invocations in the program's own declared workgroup.
fn declared_invocations(workgroup: [u32; 3]) -> u64 {
    u64::from(workgroup[0]) * u64::from(workgroup[1]) * u64::from(workgroup[2])
}

#[test]
fn a_single_point_program_guards_on_its_point_not_its_tile() {
    let mut offenders = Vec::new();
    let mut checked = 0usize;

    for entry in all_entries() {
        let Some(build) = entry.build else {
            continue;
        };
        let program = build();
        if declared_invocations(program.workgroup_size()) != 1 {
            continue;
        }
        checked += 1;
        if relies_on_single_invocation_workgroup(&program) {
            offenders.push(entry.id);
        }
    }

    assert!(
        checked > 0,
        "Fix: no registered operation declared a single-invocation workgroup, so this contract \
         proved nothing. The catalog projection or the workgroup accessor changed."
    );
    assert!(
        offenders.is_empty(),
        "Fix: {} single-point operation(s) keep a running result in read-write storage under \
         a guard on logical tile identity alone: {}. Guard the body on `LogicalIndex.x == 0`, \
         which selects the same point in the declared geometry and only one point under any wider \
         schedule. {} of {} single-point program(s) checked.",
        offenders.len(),
        offenders.join(", "),
        offenders.len(),
        checked,
    );
}
