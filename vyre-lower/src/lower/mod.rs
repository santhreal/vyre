//! Lower a `vyre_foundation::Program` into a substrate-neutral
//! `KernelDescriptor`.
//!
//! This module is the shared boundary between high-level vyre IR and
//! emitter input. It preserves supported IR semantics in descriptor
//! form and returns an explicit [`LowerError`] when the input is
//! invalid.

mod body_assembly;
mod carrier_names;
mod context;
mod descriptor_metadata;
mod expr_lowering;
mod loop_site;
mod node_lowering;
mod op_emission;
mod scope;

use rustc_hash::{FxHashMap, FxHashSet};
use scope::VarScope;

use crate::descriptor::{BindingLayout, BindingSlot, Dispatch, KernelDescriptor, MemoryClass};
use crate::error::LowerError;
use vyre_foundation::ir::{Ident, Program};

use self::body_assembly::{
    body_contains_trap, empty_body_with_capacity, estimated_root_op_capacity,
};
use self::descriptor_metadata::fingerprint_id;

/// Maximum nested-body depth before lowering refuses with
/// `LowerError::NestingTooDeep`. 64 levels is generous; real programs
/// rarely exceed 10.
const MAX_NESTING_DEPTH: usize = 64;

/// First slot value reserved for `MemoryClass::Shared` / `MemoryClass::Scratch`
/// bindings. Host-bound bindings (`Global`/`Constant`/`Uniform`) use slots
/// 0..WORKGROUP_SLOT_BASE so that backend bind-group layouts (capped at 1000
/// bindings on wgpu) never see a Shared slot. Any rewrite that allocates
/// new Shared/Scratch bindings must seed its `next_slot` cursor at or above
/// this constant to avoid colliding with host slots in `BindingLayout.slots`.
pub(crate) const WORKGROUP_SLOT_BASE: u32 = 1 << 24;

/// Lower a vyre Program to the substrate-neutral kernel descriptor.
///
/// # The only Program-to-emitter boundary
///
/// This is the single place a `Program` becomes something an emitter can read.
/// Every concrete backend consumes only the resulting [`KernelDescriptor`], and
/// `vyre-driver-cuda/src/codegen/descriptor_gate.rs` plus
/// `vyre-driver-wgpu/src/emit/descriptor_gate.rs` exist to keep it that way:
/// backends may analyze or emit descriptors but must not host a parallel
/// Program-to-descriptor lowering, because a second one would let a Program
/// field reach generated code without passing through here.
///
/// That gate makes this function's reads an exhaustive statement of what
/// generated code can depend on. It reads exactly SIX `BufferDecl` fields into
/// the descriptor, and nothing else: `name`, `binding`, `access` (via
/// `binding_visibility`), `kind` (via `memory_class`), `element`, and `count`
/// (as `element_count`). Plus `Program::workgroup_size` and `Program::entry`.
///
/// Anything that keys generated code, most importantly compiled-pipeline cache
/// identity in `Program::try_normalized_cache_digest`, should derive its input
/// set from this list rather than sampling program fields and checking whether
/// the output changed. Sampling is how the ungated `element_count` read in
/// `vyre-emit-ptx`'s `async_copy.rs` was missed: a fixture without an async
/// copy makes a storage buffer's `count` look irrelevant to emitted text.
///
/// # Errors
///
/// Returns [`LowerError`] when the input references undeclared buffers,
/// exceeds the supported structured nesting depth, or uses an IR
/// construct with invalid operands.
pub(crate) fn lower(program: &Program) -> Result<KernelDescriptor, LowerError> {
    let mut ctx = LowerCtx::new(program)?;
    let mut body = empty_body_with_capacity(estimated_root_op_capacity(program));
    ctx.lower_nodes(program.entry(), &mut body, 0)?;
    if body_contains_trap(&body) {
        ctx.add_trap_sidecar_binding()?;
    }

    Ok(KernelDescriptor {
        id: fingerprint_id(program),
        bindings: BindingLayout {
            slots: ctx.bindings,
        },
        dispatch: Dispatch {
            workgroup_size: program.workgroup_size(),
        },
        body,
    })
}

struct LowerCtx {
    bindings: Vec<BindingSlot>,
    buffer_slots: FxHashMap<Ident, u32>,
    slot_memory_classes: FxHashMap<u32, MemoryClass>,
    scope: VarScope,
    next_value: u32,
    /// Stack of "currently active loop carriers"  -  one frame per
    /// enclosing `Node::Loop` we are inside of. An `Assign(name, ..)`
    /// whose `name` is in any active frame commits its new value
    /// directly to the function-local via `LoopCarrierEnd` and then
    /// re-reads via `LoopCarrier`, bypassing the if-then phi-merge
    /// path. The Select-based merge cannot represent the per-iteration
    /// state correctly because the carrier's authoritative storage
    /// lives in the function-local, not in any SSA value.
    active_carriers: Vec<FxHashSet<Ident>>,
}

#[cfg(test)]
mod tests {
    use super::{lower, MAX_NESTING_DEPTH};
    use crate::descriptor::KernelOpKind;

    #[test]
    fn lower_empty_wrapped_program_preserves_region() {
        let program = vyre_foundation::ir::Program::wrapped(vec![], [1, 1, 1], vec![]);
        let desc = lower(&program).unwrap();
        assert_eq!(desc.dispatch.workgroup_size, [1, 1, 1]);
        assert!(desc.bindings.slots.is_empty());
        assert_eq!(desc.body.ops.len(), 1);
        assert!(matches!(desc.body.ops[0].kind, KernelOpKind::Region { .. }));
    }

    #[test]
    fn max_nesting_depth_constant_is_documented() {
        assert_eq!(MAX_NESTING_DEPTH, 64);
    }
}
