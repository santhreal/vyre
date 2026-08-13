//! Fluent construction for [`KernelDescriptor`] values.
//!
//! Descriptors are deep struct literals: a binding layout, a dispatch shape,
//! and a body tree whose ops carry a `Vec<u32>` operand list and an
//! `Option<u32>` result. Written out by hand the shape dwarfs the two or
//! three fields that actually vary, and every crate that builds a descriptor
//! ends up carrying its own copy of the same scaffolding.
//!
//! This module is the ONE owner of that scaffolding. It is backend-neutral by
//! construction: it names only `vyre-lower`'s own descriptor vocabulary and
//! adds no target, dialect, or artifact concepts.
//!
//! Result ids, literal-pool indices, and child-body indices stay explicit.
//! The builder removes punctuation, not meaning, so a descriptor written
//! through it is byte-identical to the struct literal it replaces.
//!
//! ```
//! use vyre_foundation::ir::DataType;
//! use vyre_lower::descriptor_builder::{body, descriptor, effect, global_rw, lit, op};
//! use vyre_lower::{KernelOpKind, LiteralValue};
//!
//! let desc = descriptor("copy")
//!     .slot(global_rw(0, DataType::U32, "buf"))
//!     .dispatch(64, 1, 1)
//!     .body(
//!         body()
//!             .literals([LiteralValue::U32(0), LiteralValue::U32(7)])
//!             .op(lit(0, 0))
//!             .op(lit(1, 1))
//!             .op(effect(KernelOpKind::StoreGlobal, [0, 0, 1]))
//!             .op(op(KernelOpKind::LoadGlobal, [0, 0], 2)),
//!     )
//!     .build();
//! assert_eq!(desc.body.ops.len(), 4);
//! ```

use vyre_foundation::ir::DataType;

use crate::{
    BindingLayout, BindingSlot, BindingVisibility, Dispatch, KernelBody, KernelDescriptor, KernelOp,
    KernelOpKind, LiteralValue, MemoryClass,
};

/// An op that produces `result`.
#[must_use]
pub fn op(kind: KernelOpKind, operands: impl Into<Vec<u32>>, result: u32) -> KernelOp {
    KernelOp {
        kind,
        operands: operands.into(),
        result: Some(result),
    }
}

/// An op that produces no value.
#[must_use]
pub fn effect(kind: KernelOpKind, operands: impl Into<Vec<u32>>) -> KernelOp {
    KernelOp {
        kind,
        operands: operands.into(),
        result: None,
    }
}

/// A [`KernelOpKind::Literal`] op reading `pool_index` into `result`.
#[must_use]
pub fn lit(pool_index: u32, result: u32) -> KernelOp {
    op(KernelOpKind::Literal, [pool_index], result)
}

/// A binding slot with every field named.
#[must_use]
pub fn slot(
    index: u32,
    element_type: DataType,
    memory_class: MemoryClass,
    visibility: BindingVisibility,
    name: &str,
) -> BindingSlot {
    BindingSlot {
        slot: index,
        element_type,
        element_count: None,
        memory_class,
        visibility,
        name: name.into(),
    }
}

/// A runtime-sized read-write global binding.
#[must_use]
pub fn global_rw(index: u32, element_type: DataType, name: &str) -> BindingSlot {
    slot(
        index,
        element_type,
        MemoryClass::Global,
        BindingVisibility::ReadWrite,
        name,
    )
}

/// A runtime-sized read-only global binding.
#[must_use]
pub fn global_ro(index: u32, element_type: DataType, name: &str) -> BindingSlot {
    slot(
        index,
        element_type,
        MemoryClass::Global,
        BindingVisibility::ReadOnly,
        name,
    )
}

/// A runtime-sized write-only global binding.
#[must_use]
pub fn global_wo(index: u32, element_type: DataType, name: &str) -> BindingSlot {
    slot(
        index,
        element_type,
        MemoryClass::Global,
        BindingVisibility::WriteOnly,
        name,
    )
}

/// A fixed-size read-write workgroup-shared binding.
#[must_use]
pub fn shared_rw(index: u32, element_type: DataType, count: u32, name: &str) -> BindingSlot {
    slot(
        index,
        element_type,
        MemoryClass::Shared,
        BindingVisibility::ReadWrite,
        name,
    )
    .with_count(count)
}

/// Element-count refinement for a slot produced by this module.
pub trait SlotCount {
    /// Set the element count, replacing the runtime-sized default.
    #[must_use]
    fn with_count(self, count: u32) -> Self;
}

impl SlotCount for BindingSlot {
    fn with_count(mut self, count: u32) -> Self {
        self.element_count = Some(count);
        self
    }
}

/// Accumulates one [`KernelBody`].
#[derive(Debug, Default, Clone)]
pub struct KernelBodyBuilder {
    ops: Vec<KernelOp>,
    child_bodies: Vec<KernelBody>,
    literals: Vec<LiteralValue>,
}

/// Start an empty body.
#[must_use]
pub fn body() -> KernelBodyBuilder {
    KernelBodyBuilder::default()
}

impl KernelBodyBuilder {
    /// Append one op.
    #[must_use]
    pub fn op(mut self, op: KernelOp) -> Self {
        self.ops.push(op);
        self
    }

    /// Append several ops.
    #[must_use]
    pub fn ops(mut self, ops: impl IntoIterator<Item = KernelOp>) -> Self {
        self.ops.extend(ops);
        self
    }

    /// Append one child body. Its index is the current child count.
    #[must_use]
    pub fn child(mut self, child: impl Into<KernelBody>) -> Self {
        self.child_bodies.push(child.into());
        self
    }

    /// Append several child bodies in order.
    #[must_use]
    pub fn children<B: Into<KernelBody>>(mut self, children: impl IntoIterator<Item = B>) -> Self {
        self.child_bodies
            .extend(children.into_iter().map(Into::into));
        self
    }

    /// Append one literal-pool entry. Its index is the current pool length.
    #[must_use]
    pub fn literal(mut self, value: LiteralValue) -> Self {
        self.literals.push(value);
        self
    }

    /// Append several literal-pool entries in order.
    #[must_use]
    pub fn literals(mut self, values: impl IntoIterator<Item = LiteralValue>) -> Self {
        self.literals.extend(values);
        self
    }

    /// Finish the body.
    #[must_use]
    pub fn build(self) -> KernelBody {
        KernelBody {
            ops: self.ops,
            child_bodies: self.child_bodies,
            literals: self.literals,
        }
    }
}

impl From<KernelBodyBuilder> for KernelBody {
    fn from(builder: KernelBodyBuilder) -> Self {
        builder.build()
    }
}

/// Accumulates one [`KernelDescriptor`].
#[derive(Debug, Clone)]
pub struct KernelDescriptorBuilder {
    id: String,
    slots: Vec<BindingSlot>,
    dispatch: Dispatch,
    body: KernelBody,
}

/// Start a descriptor with no bindings, a single-invocation dispatch, and an
/// empty body.
#[must_use]
pub fn descriptor(id: &str) -> KernelDescriptorBuilder {
    KernelDescriptorBuilder {
        id: id.to_owned(),
        slots: Vec::new(),
        dispatch: Dispatch::new(1, 1, 1),
        body: KernelBody {
            ops: Vec::new(),
            child_bodies: Vec::new(),
            literals: Vec::new(),
        },
    }
}

impl KernelDescriptorBuilder {
    /// Append one binding slot.
    #[must_use]
    pub fn slot(mut self, slot: BindingSlot) -> Self {
        self.slots.push(slot);
        self
    }

    /// Append several binding slots in order.
    #[must_use]
    pub fn slots(mut self, slots: impl IntoIterator<Item = BindingSlot>) -> Self {
        self.slots.extend(slots);
        self
    }

    /// Set the workgroup dimensions.
    #[must_use]
    pub fn dispatch(mut self, x: u32, y: u32, z: u32) -> Self {
        self.dispatch = Dispatch::new(x, y, z);
        self
    }

    /// Set the top-level body.
    #[must_use]
    pub fn body(mut self, body: impl Into<KernelBody>) -> Self {
        self.body = body.into();
        self
    }

    /// Set the top-level body's ops, keeping its children and literals.
    #[must_use]
    pub fn ops(mut self, ops: impl IntoIterator<Item = KernelOp>) -> Self {
        self.body.ops.extend(ops);
        self
    }

    /// Finish the descriptor.
    #[must_use]
    pub fn build(self) -> KernelDescriptor {
        KernelDescriptor {
            id: self.id,
            bindings: BindingLayout { slots: self.slots },
            dispatch: self.dispatch,
            body: self.body,
        }
    }
}

impl From<KernelDescriptorBuilder> for KernelDescriptor {
    fn from(builder: KernelDescriptorBuilder) -> Self {
        builder.build()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builder_output_matches_the_struct_literal_it_replaces() {
        let built = descriptor("k")
            .slot(global_rw(0, DataType::U32, "buf"))
            .dispatch(64, 1, 1)
            .body(
                body()
                    .literals([LiteralValue::U32(0), LiteralValue::U32(7)])
                    .op(lit(0, 0))
                    .op(lit(1, 1))
                    .op(effect(KernelOpKind::StoreGlobal, [0, 0, 1])),
            )
            .build();

        let literal = KernelDescriptor {
            id: "k".into(),
            bindings: BindingLayout {
                slots: vec![BindingSlot {
                    slot: 0,
                    element_type: DataType::U32,
                    element_count: None,
                    memory_class: MemoryClass::Global,
                    visibility: BindingVisibility::ReadWrite,
                    name: "buf".into(),
                }],
            },
            dispatch: Dispatch::new(64, 1, 1),
            body: KernelBody {
                ops: vec![
                    KernelOp {
                        kind: KernelOpKind::Literal,
                        operands: vec![0],
                        result: Some(0),
                    },
                    KernelOp {
                        kind: KernelOpKind::Literal,
                        operands: vec![1],
                        result: Some(1),
                    },
                    KernelOp {
                        kind: KernelOpKind::StoreGlobal,
                        operands: vec![0, 0, 1],
                        result: None,
                    },
                ],
                child_bodies: vec![],
                literals: vec![LiteralValue::U32(0), LiteralValue::U32(7)],
            },
        };

        assert_eq!(built, literal);
    }

    #[test]
    fn child_body_index_is_the_append_order() {
        let built = descriptor("nested")
            .body(
                body()
                    .op(effect(KernelOpKind::StructuredIfThenElse, [0, 0, 1]))
                    .child(body().op(lit(0, 10)))
                    .child(body().op(lit(0, 20))),
            )
            .build();
        assert_eq!(built.body.child_bodies.len(), 2);
        assert_eq!(built.body.child_bodies[0].ops[0].result, Some(10));
        assert_eq!(built.body.child_bodies[1].ops[0].result, Some(20));
    }

    #[test]
    fn defaults_are_empty_body_no_bindings_single_invocation() {
        let built = descriptor("empty").build();
        assert!(built.bindings.slots.is_empty());
        assert!(built.body.ops.is_empty());
        assert!(built.body.child_bodies.is_empty());
        assert!(built.body.literals.is_empty());
        assert_eq!(built.dispatch, Dispatch::new(1, 1, 1));
    }

    #[test]
    fn shared_slot_carries_its_element_count() {
        let s = shared_rw(1, DataType::F32, 64, "tile");
        assert_eq!(s.element_count, Some(64));
        assert_eq!(s.memory_class, MemoryClass::Shared);
        assert_eq!(global_ro(0, DataType::F32, "g").element_count, None);
    }
}
