//! Descriptor invariant verifier.
//!
//! Walks a `KernelDescriptor` checking structural invariants that
//! every well-formed descriptor must satisfy:
//!
//! 1. Within each `KernelBody`, `op.result` ids are unique.
//! 2. Operand positions classified as result-id references must
//!    point at a result-id produced in the same body or lexically
//!    captured from a parent structured-control body.
//! 3. Operand positions classified as literal-pool indices must be in
//!    range of the body's `literals` vector.
//! 4. Operand positions classified as child-body indices must be in
//!    range of the body's `child_bodies` vector.
//! 5. Every `KernelOpKind::Literal` op must have at least one operand
//!    (the pool index).
//! 6. Result ids are unique across the whole descriptor, not only
//!    within a body. Backends key their register maps on the raw id, so
//!    reuse between two bodies makes an operand resolve to the wrong
//!    producer.
//!
//! Bodies recurse with lexical scope and loop-carried visibility.
//! `vyre-lower` allocates result ids globally for the descriptor:
//! structured child bodies may reference values available before the
//! child was attached, and parent bodies may reference values assigned
//! by a completed child body.
//!
//! ## Wiring
//!
//! [`crate::lower_verified`] and [`crate::verify_descriptor`] invoke this
//! verifier before emitter handoff. Tests and fuzzers call `verify()` directly
//! to turn malformed descriptors into structured failures.

mod body_walk;
mod error;

use body_walk::{verify_body, verify_result_ids_unique_descriptor_wide};
pub use error::{format_verify_errors, VerifyError, VerifyErrorKind, VerifyResult};

use crate::KernelDescriptor;

/// Verify every structural invariant of a kernel descriptor.
#[must_use]
pub fn verify(desc: &KernelDescriptor) -> VerifyResult {
    use rustc_hash::FxHashSet;
    let mut errors = Vec::new();
    // Dispatch-level checks (don't have a body_path).
    for (axis, &dim) in desc.dispatch.workgroup_size.iter().enumerate() {
        if dim == 0 {
            errors.push(VerifyError {
                body_path: vec![],
                op_index: 0,
                kind: VerifyErrorKind::DispatchZeroDim { axis: axis as u8 },
            });
        }
    }
    // Binding-layout checks: no two slots share `.slot` field; host vs
    // workgroup ranges stay segregated.
    use crate::descriptor::MemoryClass;
    let mut seen_slots: FxHashSet<u32> = FxHashSet::default();
    for s in &desc.bindings.slots {
        if !seen_slots.insert(s.slot) {
            errors.push(VerifyError {
                body_path: vec![],
                op_index: 0,
                kind: VerifyErrorKind::DuplicateBindingSlotId { slot: s.slot },
            });
        }
        let in_workgroup_range = s.slot >= crate::lower::WORKGROUP_SLOT_BASE;
        let is_workgroup_class =
            matches!(s.memory_class, MemoryClass::Shared | MemoryClass::Scratch,);
        if in_workgroup_range && !is_workgroup_class {
            errors.push(VerifyError {
                body_path: vec![],
                op_index: 0,
                kind: VerifyErrorKind::HostBindingInWorkgroupRange { slot: s.slot },
            });
        }
        if !in_workgroup_range && is_workgroup_class {
            errors.push(VerifyError {
                body_path: vec![],
                op_index: 0,
                kind: VerifyErrorKind::WorkgroupBindingInHostRange { slot: s.slot },
            });
        }
    }
    verify_body(
        &desc.body,
        &mut Vec::new(),
        &FxHashSet::default(),
        &mut errors,
    );
    verify_result_ids_unique_descriptor_wide(&desc.body, &mut errors);
    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

// Inline: covers the crate-private `verify` module, which no integration test can reach.
#[cfg(test)]
mod tests;
