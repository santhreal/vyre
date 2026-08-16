//! A flash-attention plan reports every workgroup buffer its program declares.
//!
//! WHY: `shared_memory_bytes` is the occupancy input the planner rows compare
//! kernels on, so a plan that counts one of its three workgroup buffers makes
//! the kernel look cheaper than it runs. The scalar path reported `q_scratch`
//! alone and omitted `score_tile` and `o_acc`, which stayed invisible while the
//! scalar kernel had no score tile at all and became a 1024-byte understatement
//! the moment the scalar path folded onto the shared online-softmax core.
//!
//! The invariant is stated against the built program rather than against a
//! remembered byte count: whatever buffers a kernel allocates, the plan reports
//! their sum. It does not catch a wrong `count()` on a buffer, since both sides
//! read the same declaration.

mod harness;

use std::collections::BTreeSet;

use vyre_foundation::ir::{BufferAccess, Program};
use vyre_libs::nn::attention::{
    flash_attention, flash_attention_2, plan_flash_attention_scalar, plan_flash_attention_tiled,
    FlashAttentionKernelKind, FlashAttentionWorkPlan,
};

const F32_BYTES: u64 = 4;

/// Bytes of shared memory the program really asks a backend to allocate.
fn declared_workgroup_bytes(program: &Program) -> u64 {
    program
        .buffers()
        .iter()
        .filter(|buffer| buffer.access() == BufferAccess::Workgroup)
        .map(|buffer| u64::from(buffer.count()) * F32_BYTES)
        .sum()
}

fn assert_plan_reports_its_scratch(plan: &FlashAttentionWorkPlan, program: &Program, label: &str) {
    let declared = declared_workgroup_bytes(program);
    assert!(
        declared > 0,
        "{label} declares no workgroup buffer, so this case cannot judge the plan"
    );
    assert_eq!(
        plan.bench_metrics.memory_traffic.shared_memory_bytes, declared,
        "{label} plan reports {} shared bytes and its program declares {declared}",
        plan.bench_metrics.memory_traffic.shared_memory_bytes
    );
}

/// Variant names of `enum FlashAttentionKernelKind`, read from the source that
/// declares them.
///
/// A written list of kernel kinds goes stale in silence, which is the same
/// failure as having no coverage check: a third kernel would ship with its
/// shared-memory accounting unjudged.
fn declared_kernel_kinds() -> BTreeSet<String> {
    harness::declared_enum_variants(
        &harness::crate_file("src/nn/attention/planner.rs"),
        "pub enum FlashAttentionKernelKind {",
    )
}

#[test]
fn scalar_plan_reports_every_workgroup_buffer_its_program_declares() {
    let plan = plan_flash_attention_scalar(9, 7).expect("Fix: scalar plan builds");
    let program = flash_attention("q", "k", "v", "out", 9, 7).expect("Fix: scalar program builds");
    assert_eq!(plan.kernel, FlashAttentionKernelKind::ScalarOnline);
    assert_plan_reports_its_scratch(&plan, &program, "scalar online");
}

#[test]
fn tiled_plan_reports_every_workgroup_buffer_its_program_declares() {
    let plan = plan_flash_attention_tiled(8, 16, 4).expect("Fix: tiled plan builds");
    let program = flash_attention_2("q", "k", "v", "out", 8, 16, 4);
    assert_eq!(plan.kernel, FlashAttentionKernelKind::CooperativeTiled);
    assert_plan_reports_its_scratch(&plan, &program, "cooperative tiled");
}

#[test]
fn every_declared_kernel_kind_has_a_shared_memory_case() {
    let covered: BTreeSet<String> = [
        format!("{:?}", FlashAttentionKernelKind::ScalarOnline),
        format!("{:?}", FlashAttentionKernelKind::CooperativeTiled),
    ]
    .into_iter()
    .collect();
    assert_eq!(
        declared_kernel_kinds(),
        covered,
        "a kernel kind was added or renamed; give it a case above so its plan is judged \
         against the workgroup buffers its program declares"
    );
}
