//! The target bindings a module's launch stages bytes into.
//!
//! A module's launch reads the bindings its lowered descriptor declares, not the
//! host inputs its neutral `Program` declares. The two lists differ wherever a
//! fused artifact carries a value between modules: an intermediate buffer is
//! written by one module and read by the next, so the reading module's Program
//! marks it as consuming no host input while its kernel loads from it on the
//! first instruction.
//!
//! A materializer that staged only the Program's host inputs launched that
//! module over an allocation nothing had filled. `vyre-libs::security`'s fused
//! flow operations read the stage-one bitset as zero on every case, which is why
//! `sink_intersection` counted nothing and `aliases_dataflow` returned its seed
//! frontier unchanged.

use vyre_foundation::ir::Program;
use vyre_lower::KernelDescriptor;

use crate::envelope::resource_bind_group;
use crate::TargetCompileError;
use crate::{TargetResourceAccess, TargetResourceBinding};

/// One target binding a launch stages bytes into before a module runs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ArtifactInputSlot {
    /// Descriptor binding name, used in rejections.
    pub name: String,
    /// Resource group the payload published this binding under.
    pub group: u32,
    /// Binding slot within `group`.
    pub slot: u32,
    /// Static byte ceiling for the declaration, when it has one.
    pub expected_max: Option<usize>,
    /// What the slot holds at launch when the module produces its own value.
    ///
    /// A slot the target module loads from and this module also writes is
    /// staged, because the emitted binding order carries no gap. Nothing has
    /// written the value yet, so its launch contents are what the dispatch
    /// allocates, which is zero. Allocated once here rather than per launch.
    /// `None` for a runtime-sized declaration, whose byte count is not known
    /// until a caller supplies it.
    pub launch_zeros: Option<Box<[u8]>>,
}

/// The staged input slots of one compiled module, in target binding order.
///
/// The trap sidecar is excluded: it is backend-owned diagnostic storage that
/// carries no artifact value, and each driver allocates it for itself.
///
/// # Errors
///
/// Returns [`TargetCompileError::InvalidArtifact`] when a host-bound descriptor
/// slot has no directional metadata in the payload, or names no buffer in the
/// module's Program.
pub fn staged_input_slots(
    descriptor: &KernelDescriptor,
    resource_bindings: &[TargetResourceBinding],
    program: &Program,
) -> Result<Vec<ArtifactInputSlot>, TargetCompileError> {
    let mut slots = Vec::new();
    for slot in &descriptor.bindings.slots {
        let Some(group) = resource_bind_group(slot.memory_class) else {
            continue;
        };
        if slot.name == vyre_lower::TRAP_SIDECAR_NAME {
            continue;
        }
        let canonical = resource_bindings
            .iter()
            .find(|binding| binding.group == group && binding.slot == slot.slot)
            .ok_or_else(|| {
                TargetCompileError::InvalidArtifact(format!(
                    "target binding `{}` at group {group}, slot {} has no canonical directional metadata",
                    slot.name, slot.slot
                ))
            })?;
        if canonical.access == TargetResourceAccess::WriteOnly {
            continue;
        }
        let buffer = program
            .buffers()
            .iter()
            .find(|buffer| buffer.name() == slot.name)
            .ok_or_else(|| {
                TargetCompileError::InvalidArtifact(format!(
                    "target binding `{}` has no selected Program buffer",
                    slot.name
                ))
            })?;
        let expected_max = usize::try_from(buffer.count())
            .ok()
            .and_then(|count| count.checked_mul(buffer.element().min_bytes()))
            .filter(|_| buffer.count() != 0);
        slots.push(ArtifactInputSlot {
            name: slot.name.clone(),
            group,
            slot: slot.slot,
            expected_max,
            launch_zeros: expected_max.map(|bytes| vec![0_u8; bytes].into_boxed_slice()),
        });
    }
    Ok(slots)
}
