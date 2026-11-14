//! Stable descriptor-layout fingerprints for backend object caches.

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
