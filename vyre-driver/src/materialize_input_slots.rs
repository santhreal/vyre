//! Staging one module's target-binding inputs out of the execution state.
//!
//! `vyre_megakernel::staged_input_slots` states which bindings a launch fills
//! and in what order; this resolves each one to the artifact value bound to it,
//! which is where a fused artifact's inter-module values live.

use std::collections::BTreeMap;

use vyre_megakernel::{ArtifactInputSlot, ArtifactValueId};

use crate::materialize::InstanceCore;
use crate::BackendError;

/// Borrow one module's staged bytes in the order
/// [`vyre_megakernel::staged_input_slots`] produced.
///
/// A slot this module also writes may be unbound on entry; it launches over the
/// zeros the slot reserved, which is what the allocation holds.
///
/// # Errors
///
/// Returns [`crate::materialize::invalid_module`] when a slot's value is neither
/// bound nor produced by this module, and when a bound value is longer than the
/// slot's static declaration.
pub fn gather_artifact_inputs<'a>(
    core: &'a InstanceCore,
    module_index: usize,
    slots: &'a [ArtifactInputSlot],
    state: &'a BTreeMap<ArtifactValueId, Vec<u8>>,
) -> Result<Vec<&'a [u8]>, BackendError> {
    let produced_here = core
        .module_outputs
        .get(module_index)
        .map_or(&[][..], Vec::as_slice);
    let mut inputs = Vec::with_capacity(slots.len());
    for slot in slots {
        let value = core.value_for_module_slot(
            &core.module_inputs,
            module_index,
            slot.group,
            slot.slot,
            &slot.name,
        )?;
        let bytes = match state.get(&value) {
            Some(bound) => bound.as_slice(),
            None => produced_here
                .contains(&value)
                .then_some(())
                .and_then(|()| slot.launch_zeros.as_deref())
                .ok_or_else(|| {
                    crate::materialize::invalid_module(&format!(
                        "canonical artifact value {} for target binding `{}` is unbound",
                        value.0, slot.name
                    ))
                })?,
        };
        if slot
            .expected_max
            .is_some_and(|expected| bytes.len() > expected)
        {
            let canonical_name = core
                .values
                .iter()
                .find_map(|(name, candidate)| (*candidate == value).then_some(name.as_str()))
                .unwrap_or("<unnamed>");
            return Err(crate::materialize::invalid_module(&format!(
                "canonical artifact value {} (`{canonical_name}`) supplied {} byte(s) to target binding `{}`, whose static limit is {} byte(s)",
                value.0,
                bytes.len(),
                slot.name,
                slot.expected_max.unwrap_or_default(),
            )));
        }
        inputs.push(bytes);
    }
    Ok(inputs)
}
