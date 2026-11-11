//! Stable fingerprints and descriptor-layout sharing for binding plans.

use crate::binding::{BindingPlan, BindingRole};

/// Stable fingerprint of a binding set's *layout*  -  the parts that
/// determine whether two `BindingPlan`s can share a backend bind
/// group layout / descriptor set.
///
/// Two plans with the same [`BindingSetFingerprint`] can reuse the
/// same `portable::BindGroupLayout` or native descriptor set across
/// consecutive dispatches, skipping the layout-rebind cost. The
/// hot-path perf snapshot puts binding rebind at ~20% of warm
/// dispatch time on attention/softmax/reduce shapes.
///
/// Layout (this fingerprint) is distinct from contents (which
/// `program_vsa_fingerprint` covers)  -  two dispatches of the same
/// kernel on different input buffers share a layout fingerprint but
/// differ in their content fingerprint.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct BindingSetFingerprint {
    /// Per-binding layout slot: `(binding_index, role, element_size)`.
    /// Ordered by `binding_index` for deterministic equality.
    pub slots: Vec<(u32, BindingRole, usize)>,
}

impl BindingSetFingerprint {
    /// Derive the layout fingerprint from a `BindingPlan`. Stable
    /// across runs and across machines (no random salts).
    #[must_use]
    pub fn from_plan(plan: &BindingPlan) -> Self {
        let mut slots: Vec<(u32, BindingRole, usize)> = plan
            .bindings
            .iter()
            .map(|b| (b.binding, b.role, b.element_size))
            .collect();
        slots.sort_by_key(|(idx, _, _)| *idx);
        Self { slots }
    }
}

/// True when two binding plans can share a backend bind group
/// layout / descriptor set. This is the N7 merge predicate; a
/// driver maintains a cache keyed by [`BindingSetFingerprint`] and
/// reuses the cached layout when this returns `true`.
#[must_use]
pub fn binding_plans_share_layout(a: &BindingPlan, b: &BindingPlan) -> bool {
    BindingSetFingerprint::from_plan(a) == BindingSetFingerprint::from_plan(b)
}

/// Backend-neutral descriptor/bind-group layout slot.
///
/// Concrete drivers own target-specific object creation, but the
/// fingerprint used to decide whether a descriptor layout is reusable is
/// shared here so portable/native/secondary do not grow separate cache-key rules.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BackendLayoutSlot {
    /// Target descriptor group/set.
    pub group: u32,
    /// Binding index inside the descriptor group/set.
    pub binding: u32,
    /// Descriptor memory class.
    pub class: BackendLayoutClass,
    /// Whether storage descriptors are read-only.
    pub read_only: bool,
    /// Element size in bytes when statically known.
    pub element_size: usize,
}

/// Backend-neutral descriptor memory class.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BackendLayoutClass {
    /// Read-write or read-only storage buffer.
    Storage,
    /// Uniform/constant buffer.
    Uniform,
}

/// Stable descriptor-layout fingerprint for backend object caches.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct BackendLayoutFingerprint {
    /// Canonical slots sorted by `(group, binding)`.
    pub slots: Vec<BackendLayoutSlot>,
}

impl BackendLayoutFingerprint {
    /// Build a deterministic fingerprint from unsorted layout slots.
    #[must_use]
    pub fn new(mut slots: Vec<BackendLayoutSlot>) -> Self {
        slots.sort_by_key(|slot| (slot.group, slot.binding));
        Self { slots }
    }
}

#[cfg(test)]
mod n7_tests {
    use super::*;
    use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Program};

    fn add_one_program() -> Program {
        Program::wrapped(
            vec![
                BufferDecl::storage("input", 0, BufferAccess::ReadOnly, DataType::U32)
                    .with_count(16),
                BufferDecl::output("out", 1, DataType::U32).with_count(16),
            ],
            [16, 1, 1],
            vec![],
        )
    }

    fn add_one_program_different_input_count() -> Program {
        // Same binding shape (slot 0 ReadOnly, slot 1 output, both
        // U32), different element_count. Layout fingerprint must match;
        // content fingerprint will not.
        Program::wrapped(
            vec![
                BufferDecl::storage("input", 0, BufferAccess::ReadOnly, DataType::U32)
                    .with_count(64),
                BufferDecl::output("out", 1, DataType::U32).with_count(64),
            ],
            [16, 1, 1],
            vec![],
        )
    }

    fn different_layout_program() -> Program {
        // Three bindings instead of two  -  must NOT share layout.
        Program::wrapped(
            vec![
                BufferDecl::storage("a", 0, BufferAccess::ReadOnly, DataType::U32).with_count(16),
                BufferDecl::storage("b", 1, BufferAccess::ReadOnly, DataType::U32).with_count(16),
                BufferDecl::output("out", 2, DataType::U32).with_count(16),
            ],
            [16, 1, 1],
            vec![],
        )
    }

    #[test]
    fn same_layout_with_different_element_counts_shares_fingerprint() {
        let a = BindingPlan::build(&add_one_program()).unwrap();
        let b = BindingPlan::build(&add_one_program_different_input_count()).unwrap();
        assert!(
            binding_plans_share_layout(&a, &b),
            "plans with same (binding, role, element_size) tuples must share layout"
        );
    }

    #[test]
    fn different_binding_count_does_not_share_layout() {
        let a = BindingPlan::build(&add_one_program()).unwrap();
        let b = BindingPlan::build(&different_layout_program()).unwrap();
        assert!(
            !binding_plans_share_layout(&a, &b),
            "plans with different binding count must not share layout"
        );
    }

    #[test]
    fn fingerprint_is_stable_across_repeated_builds() {
        let a = BindingPlan::build(&add_one_program()).unwrap();
        let b = BindingPlan::build(&add_one_program()).unwrap();
        assert_eq!(
            BindingSetFingerprint::from_plan(&a),
            BindingSetFingerprint::from_plan(&b),
            "repeated build of the same Program must produce identical fingerprints"
        );
    }

    #[test]
    fn fingerprint_slots_are_sorted_by_binding_index() {
        let plan = BindingPlan::build(&add_one_program()).unwrap();
        let fp = BindingSetFingerprint::from_plan(&plan);
        let indices: Vec<u32> = fp.slots.iter().map(|(i, _, _)| *i).collect();
        assert_eq!(indices, [0, 1], "slots must be sorted by binding index");
    }

    #[test]
    fn backend_layout_fingerprint_sorts_slots() {
        let a = BackendLayoutFingerprint::new(vec![
            BackendLayoutSlot {
                group: 1,
                binding: 4,
                class: BackendLayoutClass::Storage,
                read_only: false,
                element_size: 4,
            },
            BackendLayoutSlot {
                group: 0,
                binding: 1,
                class: BackendLayoutClass::Uniform,
                read_only: true,
                element_size: 4,
            },
        ]);
        let b = BackendLayoutFingerprint::new(vec![
            BackendLayoutSlot {
                group: 0,
                binding: 1,
                class: BackendLayoutClass::Uniform,
                read_only: true,
                element_size: 4,
            },
            BackendLayoutSlot {
                group: 1,
                binding: 4,
                class: BackendLayoutClass::Storage,
                read_only: false,
                element_size: 4,
            },
        ]);
        assert_eq!(a, b);
    }
}
