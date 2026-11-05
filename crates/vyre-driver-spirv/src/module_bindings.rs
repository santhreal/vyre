//! Descriptor bindings an emitted SPIR-V module declares.
//!
//! The driver builds its descriptor set from the neutral `Program`, but the
//! module it dispatches is lowered through `vyre-lower`, which appends a
//! reserved trap-diagnostic binding to any kernel that emits a trap. That
//! binding belongs to no `Program` buffer, so the descriptor set had no entry
//! for it and the shader accessed an unwritten descriptor: on an RTX 4090 every
//! bounds-checked operation returned its inputs unmodified, which the
//! conformance runner read as seven operations computing the wrong answer.
//!
//! The binding numbers are read from the module rather than re-derived by
//! lowering the program a second time. The module is the exact image the device
//! executes, so a set read from it belongs to it by construction, while a second
//! lowering can pick a different slot than the one the module encodes and bind
//! the sidecar over a real buffer.

use std::collections::BTreeMap;

use vyre_driver::BackendError;

/// First word of every well-formed SPIR-V module.
const SPIRV_MAGIC: u32 = 0x0723_0203;
/// Word index the first instruction starts at, past the five-word header.
const HEADER_WORDS: usize = 5;
/// `OpDecorate`.
const OP_DECORATE: u16 = 71;
/// `Decoration::Binding`.
const DECORATION_BINDING: u32 = 33;
/// `Decoration::DescriptorSet`.
const DECORATION_DESCRIPTOR_SET: u32 = 34;

/// One descriptor a module declares.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub(crate) struct ModuleDescriptor {
    /// Descriptor set the variable is decorated with.
    pub(crate) set: u32,
    /// Binding number within that set.
    pub(crate) binding: u32,
}

/// Every descriptor the module declares, ascending by set then binding.
///
/// A variable carries both decorations or neither, so an id decorated with only
/// one of them is a malformed module and is reported rather than bound.
///
/// # Errors
///
/// Returns [`BackendError::InvalidProgram`] when the words are not a SPIR-V
/// module, an instruction claims a length that runs past the end, or a decorated
/// id carries only one of the two descriptor decorations.
pub(crate) fn module_descriptors(words: &[u32]) -> Result<Vec<ModuleDescriptor>, BackendError> {
    if words.first().copied() != Some(SPIRV_MAGIC) || words.len() < HEADER_WORDS {
        return Err(BackendError::InvalidProgram {
            fix: "Fix: SPIR-V module does not start with the SPIR-V header. Emit the module through vyre-emit-spirv before Vulkan dispatch.".to_string(),
        });
    }

    let mut sets: BTreeMap<u32, u32> = BTreeMap::new();
    let mut bindings: BTreeMap<u32, u32> = BTreeMap::new();
    let mut index = HEADER_WORDS;
    while index < words.len() {
        let instruction = words[index];
        let length = (instruction >> 16) as usize;
        let opcode = (instruction & 0xFFFF) as u16;
        if length == 0 || index + length > words.len() {
            return Err(BackendError::InvalidProgram {
                fix: format!(
                    "Fix: SPIR-V instruction at word {index} declares a {length}-word length that runs past the {}-word module. Re-emit the module through vyre-emit-spirv.",
                    words.len()
                ),
            });
        }
        if opcode == OP_DECORATE && length >= 4 {
            let target = words[index + 1];
            match words[index + 2] {
                DECORATION_BINDING => {
                    bindings.insert(target, words[index + 3]);
                }
                DECORATION_DESCRIPTOR_SET => {
                    sets.insert(target, words[index + 3]);
                }
                _ => {}
            }
        }
        index += length;
    }

    let mut descriptors = Vec::with_capacity(bindings.len());
    for (target, binding) in bindings {
        let set = sets.remove(&target).ok_or_else(|| BackendError::InvalidProgram {
            fix: format!(
                "Fix: SPIR-V id {target} is decorated Binding {binding} with no DescriptorSet, so the driver cannot place it in a descriptor set. Re-emit the module through vyre-emit-spirv."
            ),
        })?;
        descriptors.push(ModuleDescriptor { set, binding });
    }
    if let Some((target, set)) = sets.into_iter().next() {
        return Err(BackendError::InvalidProgram {
            fix: format!(
                "Fix: SPIR-V id {target} is decorated DescriptorSet {set} with no Binding, so the driver cannot place it in a descriptor set. Re-emit the module through vyre-emit-spirv."
            ),
        });
    }
    descriptors.sort_unstable();
    Ok(descriptors)
}

// Inline: `module_descriptors` is crate-private, so no integration test can
// reach the parse it performs.
#[cfg(test)]
mod tests {
    use super::*;

    /// Build a module header plus the given instruction words.
    fn module(instructions: &[u32]) -> Vec<u32> {
        let mut words = vec![SPIRV_MAGIC, 0x0001_0300, 0, 32, 0];
        words.extend_from_slice(instructions);
        words
    }

    fn decorate(target: u32, decoration: u32, literal: u32) -> [u32; 4] {
        [
            (4 << 16) | u32::from(OP_DECORATE),
            target,
            decoration,
            literal,
        ]
    }

    #[test]
    fn reads_every_declared_descriptor() {
        let mut instructions = Vec::new();
        instructions.extend_from_slice(&decorate(7, DECORATION_DESCRIPTOR_SET, 0));
        instructions.extend_from_slice(&decorate(7, DECORATION_BINDING, 2));
        instructions.extend_from_slice(&decorate(9, DECORATION_BINDING, 0));
        instructions.extend_from_slice(&decorate(9, DECORATION_DESCRIPTOR_SET, 0));
        assert_eq!(
            module_descriptors(&module(&instructions))
                .expect("Fix: a well-formed module must parse"),
            vec![
                ModuleDescriptor { set: 0, binding: 0 },
                ModuleDescriptor { set: 0, binding: 2 },
            ]
        );
    }

    #[test]
    fn skips_decorations_that_are_not_descriptor_placement() {
        let mut instructions = Vec::new();
        // Decoration::Block == 2, carried by the wrapper struct type.
        instructions.extend_from_slice(&decorate(5, 2, 0));
        instructions.extend_from_slice(&decorate(7, DECORATION_DESCRIPTOR_SET, 1));
        instructions.extend_from_slice(&decorate(7, DECORATION_BINDING, 4));
        assert_eq!(
            module_descriptors(&module(&instructions))
                .expect("Fix: a well-formed module must parse"),
            vec![ModuleDescriptor { set: 1, binding: 4 }]
        );
    }

    #[test]
    fn a_module_with_no_descriptors_declares_none() {
        assert!(module_descriptors(&module(&[]))
            .expect("Fix: a module with no decorations must parse")
            .is_empty());
    }

    #[test]
    fn rejects_words_that_are_not_a_spirv_module() {
        let error = module_descriptors(&[0, 0, 0, 0, 0])
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("SPIR-V header"),
            "Fix: a non-SPIR-V word list must be refused by header, got: {error}"
        );
    }

    #[test]
    fn rejects_an_instruction_length_past_the_module_end() {
        let error = module_descriptors(&module(&[(9 << 16) | u32::from(OP_DECORATE), 1, 2, 3]))
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("runs past"),
            "Fix: an over-long instruction must be refused, got: {error}"
        );
    }

    #[test]
    fn rejects_a_zero_length_instruction() {
        let error = module_descriptors(&module(&[0])).unwrap_err().to_string();
        assert!(
            error.contains("0-word length"),
            "Fix: a zero-length instruction must be refused instead of looping, got: {error}"
        );
    }

    #[test]
    fn rejects_a_binding_with_no_descriptor_set() {
        let error = module_descriptors(&module(&decorate(7, DECORATION_BINDING, 3)))
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("no DescriptorSet"),
            "Fix: a half-decorated variable must be refused, got: {error}"
        );
    }

    #[test]
    fn rejects_a_descriptor_set_with_no_binding() {
        let error = module_descriptors(&module(&decorate(7, DECORATION_DESCRIPTOR_SET, 3)))
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("no Binding"),
            "Fix: a half-decorated variable must be refused, got: {error}"
        );
    }
}
