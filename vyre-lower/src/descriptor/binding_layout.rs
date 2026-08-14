//! Kernel-boundary binding layout behavior: the reserved trap sidecar
//! binding, plus memory-class and visibility predicates.

use super::{BindingVisibility, MemoryClass};

/// Reserved binding name for trap diagnostics.
pub const TRAP_SIDECAR_NAME: &str = "__vyre_descriptor_trap_sidecar";
/// Number of words in the trap-diagnostic sidecar.
pub const TRAP_SIDECAR_WORDS: u32 = 4;

impl MemoryClass {
    /// True iff this memory class is visible across workgroups
    /// (Global, Constant). Shared and Scratch are workgroup-local.
    #[must_use]
    pub fn is_global_visibility(self) -> bool {
        matches!(self, Self::Global | Self::Constant)
    }

    /// True iff this memory class can be written by the kernel.
    /// Constant is read-only; the rest are writable.
    #[must_use]
    pub fn is_writable(self) -> bool {
        !matches!(self, Self::Constant)
    }
}

impl BindingVisibility {
    /// True iff the binding can be read by the kernel
    /// (`ReadOnly` or `ReadWrite`).
    #[must_use]
    pub fn is_readable(self) -> bool {
        matches!(self, Self::ReadOnly | Self::ReadWrite)
    }

    /// True iff the binding can be written by the kernel
    /// (`WriteOnly` or `ReadWrite`).
    #[must_use]
    pub fn is_writable(self) -> bool {
        matches!(self, Self::WriteOnly | Self::ReadWrite)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::descriptor::BindingSlot;
    use vyre_foundation::ir::DataType;

    #[test]
    fn memory_class_predicates() {
        assert!(MemoryClass::Global.is_global_visibility());
        assert!(MemoryClass::Constant.is_global_visibility());
        assert!(!MemoryClass::Shared.is_global_visibility());
        assert!(!MemoryClass::Scratch.is_global_visibility());

        assert!(MemoryClass::Global.is_writable());
        assert!(MemoryClass::Shared.is_writable());
        assert!(MemoryClass::Scratch.is_writable());
        assert!(!MemoryClass::Constant.is_writable());
    }

    #[test]
    fn binding_visibility_readable_writable() {
        assert!(BindingVisibility::ReadOnly.is_readable());
        assert!(!BindingVisibility::ReadOnly.is_writable());
        assert!(!BindingVisibility::WriteOnly.is_readable());
        assert!(BindingVisibility::WriteOnly.is_writable());
        assert!(BindingVisibility::ReadWrite.is_readable());
        assert!(BindingVisibility::ReadWrite.is_writable());
    }

    #[test]
    fn binding_carries_full_data_type() {
        // Confirm a parametric DataType (Vec) round-trips through binding.
        let b = BindingSlot {
            slot: 5,
            element_type: DataType::Vec {
                element: Box::new(DataType::F32),
                count: 4,
            },
            element_count: Some(64),
            memory_class: MemoryClass::Global,
            visibility: BindingVisibility::ReadWrite,
            name: "v4f32".into(),
        };
        let json = serde_json::to_string(&b).unwrap();
        let parsed: BindingSlot = serde_json::from_str(&json).unwrap();
        assert_eq!(b, parsed);
    }
}
