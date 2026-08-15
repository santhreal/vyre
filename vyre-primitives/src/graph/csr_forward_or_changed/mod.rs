//! CSR frontier expansion over an in-place accumulator bitset.

mod batched_frontier_words;
mod body;
mod cpu_ref;
mod dispatch_plan;
mod launch_plan;
mod layout;
mod plan;
mod program_dispatch;
mod program_parallel;
mod program_parallel_batch;
mod program_parallel_batch_global;
mod program_serial;
mod validate;

#[cfg(feature = "inventory-registry")]
mod registry;

#[cfg(test)]
mod tests;

pub use body::{
    csr_forward_or_changed_body, csr_forward_or_changed_body_prefixed,
    csr_forward_or_changed_child, csr_forward_or_changed_child_prefixed,
};
#[cfg(any(test, feature = "cpu-parity"))]
pub use cpu_ref::{
    cpu_ref, cpu_ref_closure, cpu_ref_closure_into, cpu_ref_closure_into_with_step_hook,
};
pub use launch_plan::CsrForwardOrChangedLaunchPlan;
pub use layout::{
    csr_forward_or_changed_parallel_batch_grid, csr_forward_or_changed_parallel_grid,
    CsrForwardOrChangedProgramKey, CsrForwardOrChangedStaticInputKey,
};
pub use plan::plan_csr_forward_or_changed_launch;
pub use program_dispatch::build_csr_forward_or_changed_dispatch_program;
pub use program_parallel::{
    csr_forward_or_changed_parallel, csr_forward_or_changed_parallel_body_prefixed,
    csr_forward_or_changed_parallel_child_prefixed,
    csr_forward_or_changed_parallel_snapshot_body_prefixed,
    csr_forward_or_changed_parallel_snapshot_child_prefixed,
    csr_forward_or_changed_parallel_snapshot_child_prefixed_with_active,
};
pub use program_parallel_batch::{
    csr_forward_or_changed_parallel_batch, try_csr_forward_or_changed_parallel_batch,
};
pub use program_parallel_batch_global::{
    csr_forward_or_changed_parallel_batch_global,
    csr_forward_or_changed_parallel_batch_global_slot,
    try_csr_forward_or_changed_parallel_batch_global_slot,
};
pub use program_serial::csr_forward_or_changed;
pub use validate::{copy_csr_forward_seed_frontier_into, validate_csr_forward_or_changed_flag};

#[cfg(test)]
pub(crate) use {
    cpu_ref::cpu_ref_into,
    layout::{CsrForwardOrChangedLayout, CSR_FORWARD_OR_CHANGED_PARALLEL_WORKGROUP_SIZE},
    plan::plan_csr_forward_or_changed_dispatch,
    program_parallel_batch_global::try_csr_forward_or_changed_parallel_batch_global_dynamic_slot,
    validate::validate_csr_inputs,
};
