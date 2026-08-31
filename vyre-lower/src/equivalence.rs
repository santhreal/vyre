//! Effect equivalence across the lowering boundary.
//!
//! Descriptor verification proves the lowered form is well shaped: every
//! operand resolves, every slot is declared, every declaration is statable. It
//! cannot notice that a store the program performs is missing, because a
//! descriptor with one fewer store is as well shaped as one with it. That
//! failure reaches a device as a buffer nobody wrote and a result that looks
//! like a numerical bug.
//!
//! The observable effect of a kernel on caller-visible storage is a fact both
//! representations state. This module reads it from each side, keyed by binding
//! name, and compares the two before the kernel leaves the lowering boundary.
//! The semantic side comes from the one owner of "what does this statement do
//! to a buffer" in `vyre-foundation`; the physical side comes from the operand
//! classification table, so a new op kind that addresses storage is classified
//! in one place rather than here.
//!
//! Three directions are checked, and they are deliberately not symmetric:
//!
//! - A write is exact. A store to caller-visible storage is never dead, so a
//!   write the program performs must appear physically, and a write the program
//!   does not perform must not.
//! - A read-modify-write is exact, for the same reason in both directions: an
//!   atomic cannot be dropped, and inventing one changes what other invocations
//!   observe.
//! - A read is one-directional: the physical side may read no binding the
//!   program does not read, while a read the program performs may legitimately
//!   disappear, because a value nothing consumes is eliminable and elimination
//!   runs on the descriptor.
//!
//! Workgroup-scoped storage is out of scope here. A shared region is introduced
//! by lowering and by schedule lowering, so it has no counterpart on the
//! semantic side by construction, and the storage layout is what states it.

use rustc_hash::FxHashMap;

use vyre_foundation::ir::Program;
use vyre_foundation::optimizer::program_soa::{BufferRefKind, ProgramFacts};

use crate::descriptor::{KernelBody, KernelDescriptor, KernelOpKind, MemoryClass};
use crate::lower::descriptor_metadata::memory_class;
use crate::lower::WORKGROUP_SLOT_BASE;
use crate::operand_class::{classify_operand, OperandClass};

/// What one kernel does to one caller-visible binding.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct BindingEffects {
    /// The binding's data or length is read.
    pub read: bool,
    /// The binding is written.
    pub written: bool,
    /// The binding is read and written by one indivisible operation.
    pub atomic: bool,
}

impl BindingEffects {
    const READ: Self = Self {
        read: true,
        written: false,
        atomic: false,
    };
    const WRITTEN: Self = Self {
        read: false,
        written: true,
        atomic: false,
    };
    const ATOMIC: Self = Self {
        read: true,
        written: true,
        atomic: true,
    };

    fn absorb(&mut self, other: Self) {
        self.read |= other.read;
        self.written |= other.written;
        self.atomic |= other.atomic;
    }
}

/// Effects on caller-visible storage, keyed by binding name.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct EffectSignature {
    bindings: FxHashMap<String, BindingEffects>,
}

/// Why a lowered kernel does not perform the effects its program states.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EquivalenceError {
    /// The program writes a binding the lowered kernel never writes.
    WriteLost {
        /// Binding name.
        binding: String,
    },
    /// The lowered kernel writes a binding the program never writes.
    WriteInvented {
        /// Binding name.
        binding: String,
    },
    /// The program combines into a binding atomically and the lowered kernel
    /// does not, or the reverse.
    AtomicDisagreement {
        /// Binding name.
        binding: String,
        /// Whether the program states the read-modify-write.
        semantic: bool,
    },
    /// The lowered kernel reads a binding the program never reads.
    ReadInvented {
        /// Binding name.
        binding: String,
    },
}

impl std::fmt::Display for EquivalenceError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::WriteLost { binding } => write!(
                formatter,
                "the program writes `{binding}` and the lowered kernel does not. Fix: keep the store in the neutral lowering mapping for that node"
            ),
            Self::WriteInvented { binding } => write!(
                formatter,
                "the lowered kernel writes `{binding}` and the program does not. Fix: write only the storage the program's own statements write"
            ),
            Self::AtomicDisagreement { binding, semantic } => write!(
                formatter,
                "`{binding}` is combined atomically by the {} side only. Fix: lower a read-modify-write to a read-modify-write",
                if *semantic { "semantic" } else { "physical" }
            ),
            Self::ReadInvented { binding } => write!(
                formatter,
                "the lowered kernel reads `{binding}` and the program does not. Fix: read only the storage the program's own expressions read"
            ),
        }
    }
}

impl std::error::Error for EquivalenceError {}

impl EffectSignature {
    /// Effects on caller-visible storage stated by a semantic program.
    ///
    /// Workgroup-scoped and scratch storage is skipped, decided by the same
    /// buffer-to-memory-class mapping the descriptor lowering uses, so the two
    /// sides cover the same set of bindings by construction rather than by a
    /// second reading of the declaration.
    ///
    /// `IndirectCount` is a read of the count buffer: the dispatch consumes the
    /// value rather than producing it. The dependency-ordering walk classifies
    /// it more conservatively, which is correct there and would report an
    /// invented write here.
    #[must_use]
    pub fn from_program(program: &Program) -> Self {
        let mut visible: FxHashMap<&str, bool> = FxHashMap::default();
        for buffer in program.buffers() {
            let host = !matches!(
                memory_class(buffer),
                Ok(MemoryClass::Shared | MemoryClass::Scratch)
            );
            visible.insert(buffer.name(), host);
        }
        let mut signature = Self::default();
        let facts = ProgramFacts::build_cached(program);
        for (_, name, kind) in facts.buffer_refs() {
            if !visible.get(name.as_str()).copied().unwrap_or(true) {
                continue;
            }
            let effects = match kind {
                BufferRefKind::Read | BufferRefKind::IndirectCount | BufferRefKind::AsyncSource => {
                    BindingEffects::READ
                }
                BufferRefKind::Write | BufferRefKind::AsyncDestination => BindingEffects::WRITTEN,
                BufferRefKind::Atomic(_) => BindingEffects::ATOMIC,
            };
            signature.absorb(name.as_str(), effects);
        }
        signature
    }

    /// Effects on caller-visible storage stated by a lowered descriptor.
    ///
    /// A workgroup-scoped binding is skipped: it is storage lowering created,
    /// so no semantic buffer names it.
    #[must_use]
    pub fn from_descriptor(descriptor: &KernelDescriptor) -> Self {
        let mut names: FxHashMap<u32, &str> = FxHashMap::default();
        for slot in &descriptor.bindings.slots {
            if slot.slot < WORKGROUP_SLOT_BASE {
                names.insert(slot.slot, slot.name.as_str());
            }
        }
        let mut signature = Self::default();
        collect_body(&descriptor.body, &names, &mut signature);
        signature
    }

    fn absorb(&mut self, binding: &str, effects: BindingEffects) {
        self.bindings
            .entry(binding.to_owned())
            .or_default()
            .absorb(effects);
    }

    /// Effects stated for one binding, absent when the kernel does not name it.
    #[must_use]
    pub fn binding(&self, name: &str) -> Option<BindingEffects> {
        self.bindings.get(name).copied()
    }

    /// Number of bindings the signature states an effect on.
    #[must_use]
    pub fn len(&self) -> usize {
        self.bindings.len()
    }

    /// Whether the kernel performs no effect on caller-visible storage.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.bindings.is_empty()
    }
}

/// Compare what a program states against what its lowered form performs.
///
/// `ignored` names bindings lowering introduces at the kernel boundary and no
/// program declares, such as a diagnostic sidecar.
///
/// # Errors
///
/// Returns every disagreement found, in binding-name order, so one report names
/// the whole difference rather than the first field of it.
pub fn check_effects(
    semantic: &EffectSignature,
    physical: &EffectSignature,
    ignored: &[&str],
) -> Result<(), Vec<EquivalenceError>> {
    let mut names: Vec<&String> = semantic
        .bindings
        .keys()
        .chain(physical.bindings.keys())
        .filter(|name| !ignored.contains(&name.as_str()))
        .collect();
    names.sort_unstable();
    names.dedup();

    let mut errors = Vec::new();
    for name in names {
        let stated = semantic.bindings.get(name).copied().unwrap_or_default();
        let performed = physical.bindings.get(name).copied().unwrap_or_default();
        if stated.written && !performed.written {
            errors.push(EquivalenceError::WriteLost {
                binding: name.clone(),
            });
        }
        if performed.written && !stated.written {
            errors.push(EquivalenceError::WriteInvented {
                binding: name.clone(),
            });
        }
        if stated.atomic != performed.atomic {
            errors.push(EquivalenceError::AtomicDisagreement {
                binding: name.clone(),
                semantic: stated.atomic,
            });
        }
        if performed.read && !stated.read {
            errors.push(EquivalenceError::ReadInvented {
                binding: name.clone(),
            });
        }
    }
    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

fn collect_body(body: &KernelBody, names: &FxHashMap<u32, &str>, signature: &mut EffectSignature) {
    for op in &body.ops {
        for (position, operand) in op.operands.iter().copied().enumerate() {
            if classify_operand(&op.kind, position) != OperandClass::BindingSlot {
                continue;
            }
            let Some(name) = names.get(&operand).copied() else {
                continue;
            };
            signature.absorb(name, slot_effects(&op.kind, position));
        }
    }
    for child in &body.child_bodies {
        collect_body(child, names, signature);
    }
}

/// What one op does to the binding named at `position`.
///
/// A transfer names two bindings, so the position decides the direction: the
/// first is the source it reads and the second the destination it writes.
fn slot_effects(kind: &KernelOpKind, position: usize) -> BindingEffects {
    match kind {
        KernelOpKind::Atomic { .. } => BindingEffects::ATOMIC,
        KernelOpKind::StoreGlobal
        | KernelOpKind::StoreShared
        | KernelOpKind::VectorStoreGlobal { .. } => BindingEffects::WRITTEN,
        KernelOpKind::AsyncLoad(_) | KernelOpKind::AsyncStore(_) => {
            if position == 0 {
                BindingEffects::READ
            } else {
                BindingEffects::WRITTEN
            }
        }
        _ => BindingEffects::READ,
    }
}
