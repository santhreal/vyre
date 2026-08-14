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

use vyre_foundation::ir::{BinOp, DataType};

use crate::{
    BindingLayout, BindingSlot, BindingVisibility, Dispatch, EmissionTargetCapabilities, KernelBody,
    KernelDescriptor, KernelOp, KernelOpKind, LiteralValue, MemoryClass, SubgroupCapabilities,
    WorkgroupLimits,
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

/// A binary-arithmetic op over two value ids.
#[must_use]
pub fn binop(kind: BinOp, lhs: u32, rhs: u32, result: u32) -> KernelOp {
    op(KernelOpKind::BinOpKind(kind), [lhs, rhs], result)
}

/// A load of `slot` at element `index`.
#[must_use]
pub fn load_global(slot: u32, index: u32, result: u32) -> KernelOp {
    op(KernelOpKind::LoadGlobal, [slot, index], result)
}

/// A store of `value` into `slot` at element `index`.
#[must_use]
pub fn store_global(slot: u32, index: u32, value: u32) -> KernelOp {
    effect(KernelOpKind::StoreGlobal, [slot, index, value])
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

/// Invocation ceiling used by [`permissive_workgroup_limits`].
const PERMISSIVE_MAX_INVOCATIONS: u32 = 1024;

/// Per-axis ceiling used by [`permissive_workgroup_limits`].
const PERMISSIVE_MAX_SIZE: [u32; 3] = [1024, 1024, 64];

/// Workgroup limits with every axis and the invocation product named.
#[must_use]
pub fn workgroup_limits(max_size: [u32; 3], max_invocations: u32) -> WorkgroupLimits {
    WorkgroupLimits {
        max_size,
        max_invocations,
    }
}

/// Workgroup limits wide enough that an ordinary fixture dispatch does not
/// violate them, so a test that is not about limits does not have to pick any.
#[must_use]
pub fn permissive_workgroup_limits() -> WorkgroupLimits {
    workgroup_limits(PERMISSIVE_MAX_SIZE, PERMISSIVE_MAX_INVOCATIONS)
}

/// Every subgroup feature supported.
#[must_use]
pub fn all_subgroup_capabilities() -> SubgroupCapabilities {
    SubgroupCapabilities {
        basic: true,
        ballot: true,
        shuffle: true,
        arithmetic: true,
    }
}

/// Emission target capabilities from explicit workgroup limits and subgroup
/// support.
#[must_use]
pub fn emission_target(
    workgroup: WorkgroupLimits,
    subgroup: SubgroupCapabilities,
) -> EmissionTargetCapabilities {
    EmissionTargetCapabilities {
        workgroup,
        subgroup,
    }
}

/// A target that admits any ordinary dispatch but supports no subgroup
/// feature, which is the shape a subgroup-rejection test needs.
#[must_use]
pub fn target_without_subgroups() -> EmissionTargetCapabilities {
    emission_target(
        permissive_workgroup_limits(),
        SubgroupCapabilities::default(),
    )
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

    /// The two capability fixtures must sit on opposite sides of the admission
    /// gate for every feature, or a rejection test written against them proves
    /// nothing about the gate.
    #[test]
    fn the_capability_fixtures_bracket_the_subgroup_admission_gate() {
        let all = all_subgroup_capabilities();
        assert_eq!(all.count(), 4);
        assert!(all.first_missing(all).is_none());

        let none = target_without_subgroups().subgroup;
        assert!(!none.any());
        assert_eq!(none.first_missing(all), Some("subgroup.basic"));
    }

    #[test]
    fn permissive_limits_admit_a_full_invocation_workgroup_and_reject_one_over() {
        let limits = permissive_workgroup_limits();
        assert!(crate::validate_workgroup_size([1024, 1, 1], limits).is_empty());
        assert!(!crate::validate_workgroup_size([1025, 1, 1], limits).is_empty());
    }
}
