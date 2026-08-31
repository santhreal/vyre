//! The substructural discipline a buffer binding declares.

/// Linear-type discipline for a buffer binding.
///
/// Vyre's IR is moving from an unrestricted-by-default world toward
/// a substructural type system: a buffer can be marked `Linear`
/// (must be used exactly once on each path through the Program),
/// `Affine` (used at most once  -  drops are fine), `Relevant`
/// (used at least once), or `Unrestricted` (the historical default).
/// The type-checker pass (P-1.0-V2.2) verifies these assertions
/// before lowering; backends that hit a violation reject the
/// program at validation time instead of producing wrong code.
///
/// `Unrestricted` is the safe default when authoring a `BufferDecl`
/// for back-compat  -  every existing program continues to type-check.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[non_exhaustive]
pub enum LinearType {
    /// Use exactly once on every path. Forbids both drop-without-use
    /// and double-use.
    Linear,
    /// Use at most once on every path. Allows drop-without-use,
    /// forbids double-use.
    Affine,
    /// Use at least once on every path. Forbids drop-without-use,
    /// allows double-use.
    Relevant,
    /// No discipline applied. Default for back-compat with the
    /// pre-V2.x IR.
    #[default]
    Unrestricted,
}

impl LinearType {
    /// Whether this discipline forbids dropping a buffer without
    /// using it (`Linear` or `Relevant`).
    #[must_use]
    #[inline]
    pub const fn forbids_drop(self) -> bool {
        matches!(self, Self::Linear | Self::Relevant)
    }

    /// Whether this discipline forbids using a buffer more than once
    /// (`Linear` or `Affine`).
    #[must_use]
    #[inline]
    pub const fn forbids_reuse(self) -> bool {
        matches!(self, Self::Linear | Self::Affine)
    }
}

#[cfg(test)]
mod linear_type_tests {
    use super::*;
    use crate::ir_inner::model::op_signature::DataType;
    use crate::ir_inner::model::program::BufferDecl;

    #[test]
    fn default_is_unrestricted() {
        let buf = BufferDecl::read("a", 0, DataType::U32);
        assert_eq!(buf.linear_type(), LinearType::Unrestricted);
        assert!(!LinearType::Unrestricted.forbids_drop());
        assert!(!LinearType::Unrestricted.forbids_reuse());
    }

    #[test]
    fn linear_forbids_both() {
        assert!(LinearType::Linear.forbids_drop());
        assert!(LinearType::Linear.forbids_reuse());
    }

    #[test]
    fn affine_forbids_only_reuse() {
        assert!(!LinearType::Affine.forbids_drop());
        assert!(LinearType::Affine.forbids_reuse());
    }

    #[test]
    fn relevant_forbids_only_drop() {
        assert!(LinearType::Relevant.forbids_drop());
        assert!(!LinearType::Relevant.forbids_reuse());
    }

    #[test]
    fn with_linear_type_is_round_trip() {
        for lt in [
            LinearType::Linear,
            LinearType::Affine,
            LinearType::Relevant,
            LinearType::Unrestricted,
        ] {
            let buf = BufferDecl::read("a", 0, DataType::U32).with_linear_type(lt);
            assert_eq!(buf.linear_type(), lt);
        }
    }

    #[test]
    fn workgroup_constructor_defaults_to_unrestricted() {
        let buf = BufferDecl::workgroup("scratch", 64, DataType::U32);
        assert_eq!(buf.linear_type(), LinearType::Unrestricted);
    }

    #[test]
    fn static_byte_len_uses_packed_subbyte_width() {
        let buf = BufferDecl::read("packed_i4", 0, DataType::I4).with_count(3);
        assert_eq!(
            buf.static_byte_len()
                .expect("Fix: packed I4 byte length must compute"),
            Some(2)
        );
    }

    #[test]
    fn static_byte_len_marks_runtime_sized_buffers_dynamic() {
        let zero_count = BufferDecl::read("dynamic_count", 0, DataType::U32);
        assert_eq!(
            zero_count
                .static_byte_len()
                .expect("Fix: zero-count buffer must be representable"),
            None
        );

        let dynamic_element = BufferDecl::read("tensor", 0, DataType::Tensor).with_count(4);
        assert_eq!(
            dynamic_element
                .static_byte_len()
                .expect("Fix: runtime-sized element must be representable"),
            None
        );
    }
}
