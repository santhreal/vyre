//! Semantic compilation unit subjected to verification.

use std::sync::Arc;
use crate::ir_inner::model::program::Program;
use crate::memory_model::{AtomicOrdering, CollectiveGroup};
use crate::types::contract::NumericalContract;
use crate::types::shape::{ShapeConstraint, ShapeInterner};
use crate::types::SemanticType;

/// Semantic module representing a high-level compilation unit with types,
/// constraints, and effect declarations.
#[derive(Clone, Debug, Default)]
pub struct SemanticModule {
    /// Unique module identifier.
    pub name: String,
    /// Executable IR program representation.
    pub program: Option<Program>,
    /// Closed orthogonal semantic types defined in this module.
    pub types: Vec<SemanticType>,
    /// Symbolic shape constraints to prove.
    pub shape_constraints: Vec<ShapeConstraint>,
    /// Symbolic shape interner associated with this module.
    pub shape_interner: Arc<ShapeInterner>,
    /// Registered entry points.
    pub entry_points: Vec<String>,
    /// Explicit atomic orderings declared in the module.
    pub atomic_effects: Vec<AtomicOrdering>,
    /// Declared collective communication groups.
    pub collective_groups: Vec<CollectiveGroup>,
    /// Numerical accuracy contract.
    pub numeric_contract: Option<NumericalContract>,
}

impl SemanticModule {
    /// Create a new empty semantic module.
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            program: None,
            types: Vec::new(),
            shape_constraints: Vec::new(),
            shape_interner: Arc::new(ShapeInterner::new()),
            entry_points: Vec::new(),
            atomic_effects: Vec::new(),
            collective_groups: Vec::new(),
            numeric_contract: None,
        }
    }

    /// Construct a semantic module from an existing [`Program`].
    #[must_use]
    pub fn from_program(name: impl Into<String>, program: Program) -> Self {
        Self {
            name: name.into(),
            program: Some(program),
            types: Vec::new(),
            shape_constraints: Vec::new(),
            shape_interner: Arc::new(ShapeInterner::new()),
            entry_points: vec!["main".into()],
            atomic_effects: Vec::new(),
            collective_groups: Vec::new(),
            numeric_contract: None,
        }
    }

    /// Attach a semantic type declaration.
    #[must_use]
    pub fn with_type(mut self, ty: SemanticType) -> Self {
        self.types.push(ty);
        self
    }

    /// Attach a shape constraint.
    #[must_use]
    pub fn with_shape_constraint(mut self, constraint: ShapeConstraint) -> Self {
        self.shape_constraints.push(constraint);
        self
    }

    /// Attach an atomic effect.
    #[must_use]
    pub fn with_atomic_effect(mut self, effect: AtomicOrdering) -> Self {
        self.atomic_effects.push(effect);
        self
    }

    /// Attach a collective communication group.
    #[must_use]
    pub fn with_collective_group(mut self, group: CollectiveGroup) -> Self {
        self.collective_groups.push(group);
        self
    }

    /// Set the numerical contract.
    #[must_use]
    pub fn with_numeric_contract(mut self, contract: NumericalContract) -> Self {
        self.numeric_contract = Some(contract);
        self
    }

    /// Compute a stable Blake3 identity digest for this module.
    #[must_use]
    pub fn compute_identity(&self) -> String {
        let mut hasher = blake3::Hasher::new();
        hasher.update(self.name.as_bytes());
        for ep in &self.entry_points {
            hasher.update(ep.as_bytes());
        }
        if let Some(prog) = &self.program {
            for buf in prog.buffers() {
                hasher.update(buf.name().as_bytes());
                hasher.update(&buf.count().to_le_bytes());
            }
        }
        for eff in &self.atomic_effects {
            hasher.update(&[eff.wire_tag()]);
        }
        for cg in &self.collective_groups {
            hasher.update(&[cg.wire_tag()]);
        }
        hasher.finalize().to_hex().to_string()
    }
}
