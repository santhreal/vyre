//! Every registered single-invocation program must guard on the invocation.
//!
//! Defect class this closes: a program that keeps a running result in
//! read-write storage and admits that body with `workgroup_id.x == 0` alone.
//! The guard is exact only while the workgroup holds one invocation. A fusion
//! widens an arm to the fused workgroup, so the same body then runs once per
//! added invocation over the same slots, and the result is wrong in a way only
//! a differential sees: the buffer is populated and the shapes match.
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
fn a_single_invocation_program_guards_on_its_invocation_not_its_workgroup() {
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
        "Fix: {} single-invocation operation(s) keep a running result in read-write storage under \
         a guard on workgroup identity alone: {}. Guard the body on the invocation as well, the \
         `workgroup_id.x == 0 && local_id.x == 0` form, which is the same lane in the declared \
         geometry and the only lane under any wider one. {} of {} single-invocation program(s) \
         checked.",
        offenders.len(),
        offenders.join(", "),
        offenders.len(),
        checked,
    );
}
