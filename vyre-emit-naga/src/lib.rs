#![allow(
    clippy::doc_lazy_continuation,
    clippy::double_must_use,
    clippy::manual_div_ceil,
    clippy::needless_range_loop,
    clippy::collapsible_if,
    clippy::match_like_matches_macro,
    clippy::redundant_closure,
    clippy::too_many_arguments,
    clippy::nonminimal_bool,
    clippy::derivable_impls,
    clippy::unnecessary_lazy_evaluations,
    clippy::needless_lifetimes,
    clippy::bind_instead_of_map,
    clippy::needless_borrows_for_generic_args,
    clippy::map_entry,
    clippy::map_identity,
    clippy::manual_map,
    clippy::match_single_binding,
    clippy::field_reassign_with_default,
    dead_code,
    unused_variables
)]
//! Naga IR emitter for vyre `KernelDescriptor`.
//!
//! Consumes a substrate-neutral `vyre_lower::KernelDescriptor` and
//! produces a `naga::Module`. The emitter owns only Naga construction;
//! descriptor shaping and substrate-neutral analyses stay in
//! `vyre-lower`.

use std::sync::mpsc;
use vyre_lower::KernelDescriptor;

mod emitter;
mod error;
pub mod patterns;
pub mod program;
pub use error::EmitError;

/// Stable diagnostic row emitted when binding a lowered Vyre operation into a
/// Naga module.
#[derive(serde::Serialize, serde::Deserialize, Debug)]
pub struct BindResultEntry {
    /// Stable numeric operation id assigned by Vyre lowering.
    pub vyre_op_id: u32,
    /// Lowered operation kind used by the emitter dispatch table.
    pub op_kind: String,
    /// Naga expression or handle id used for the initial value path.
    pub init_handle: u32,
    /// Scalar kind attached to the initial value when one is known.
    pub init_scalar_kind: Option<String>,
    /// Nesting depth of the child body that produced this bind row.
    pub child_body_depth: usize,
    /// Bit-packed value-type summary observed at the call boundary.
    pub value_types_at_call: Option<u32>,
    /// Human-readable path describing where the value was published.
    pub publish_path: String,
    /// Allocated local type id when the bind operation materialized local storage.
    pub local_allocated_ty: Option<u32>,
}

/// Emit a `naga::Module` from one verified `KernelDescriptor`.
///
/// # Errors
///
/// Returns [`EmitError`] when a binding layout cannot be represented in
/// Naga IR or when the descriptor contains an operation outside this emitter's
/// supported lowering set.
pub fn emit(desc: &KernelDescriptor) -> Result<naga::Module, EmitError> {
    emitter::emit_uncached(desc)
}

/// Emit a Naga module only when `target` supports every descriptor requirement.
///
/// This entry point separates target admission from driver dispatch policy.
/// The returned module is byte-for-byte equivalent to [`emit`] when admitted.
///
/// # Errors
///
/// Returns a stable unsupported-capability error before Naga construction when
/// the descriptor exceeds the supplied subgroup or workgroup capabilities.
pub fn emit_with_capabilities(
    desc: &KernelDescriptor,
    target: &vyre_lower::EmissionTargetCapabilities,
) -> Result<naga::Module, EmitError> {
    let required = vyre_lower::required_subgroup_capabilities(desc);
    if let Some(capability) = target.subgroup.first_missing(required) {
        return Err(EmitError::UnsupportedCapability(capability));
    }
    if let Some(violation) =
        vyre_lower::validate_workgroup_size(desc.dispatch.workgroup_size, target.workgroup)
            .into_iter()
            .next()
    {
        return Err(EmitError::UnsupportedWorkgroup(violation));
    }
    emit(desc)
}

/// Emit many independent verified descriptors exactly as provided.
///
/// Results preserve input order.
#[must_use]
pub fn emit_many(descs: &[KernelDescriptor]) -> Vec<Result<naga::Module, EmitError>> {
    emit_many_with(descs, emit)
}

fn emit_many_with(
    descs: &[KernelDescriptor],
    emit_one: fn(&KernelDescriptor) -> Result<naga::Module, EmitError>,
) -> Vec<Result<naga::Module, EmitError>> {
    if descs.len() <= 1 {
        return descs.iter().map(emit_one).collect();
    }
    let worker_count = std::thread::available_parallelism()
        .map(usize::from)
        .unwrap_or(1)
        .min(descs.len())
        .max(1);
    let chunk_size = descs.len().div_ceil(worker_count);
    let (tx, rx) = mpsc::channel();
    std::thread::scope(|scope| {
        for (chunk_index, chunk) in descs.chunks(chunk_size).enumerate() {
            let tx = tx.clone();
            let start = chunk_index * chunk_size;
            scope.spawn(move || {
                for (offset, desc) in chunk.iter().enumerate() {
                    if tx.send((start + offset, emit_one(desc))).is_err() {
                        break;
                    }
                }
            });
        }
    });
    drop(tx);

    let mut results: Vec<Option<Result<naga::Module, EmitError>>> =
        std::iter::repeat_with(|| None).take(descs.len()).collect();
    for (index, result) in rx {
        if let Some(slot) = results.get_mut(index) {
            *slot = Some(result);
        }
    }
    results
        .into_iter()
        .enumerate()
        .map(|(index, result)| {
            result.unwrap_or_else(|| {
                Err(EmitError::InvalidDescriptor(format!(
                    "parallel Naga emit worker did not return descriptor index {index}. Fix: keep emit_many chunk scheduling and result collection synchronized."
                )))
            })
        })
        .collect()
}

#[cfg(test)]
mod tests;
