//! Formal dominance and SSA use-def verification for Region SSA.

use super::builder::DominanceError;
use super::{
    Block, RegionFunction, RegionKind, RegionOp, RegionOpKind, StructuredRegion, Terminator,
    ValueId,
};
use rustc_hash::{FxHashMap, FxHashSet};

/// Verifier checking that dominance holds strictly for every use in a [`RegionFunction`].
#[derive(Debug, Default)]
pub struct DominanceVerifier {
    defined_values: FxHashSet<ValueId>,
    value_def_site: FxHashMap<ValueId, String>,
}

impl DominanceVerifier {
    /// Create a new verifier.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Verify a complete function for dominance and SSA invariants.
    pub fn verify_function(&mut self, func: &RegionFunction) -> Result<(), DominanceError> {
        let mut dominating = FxHashSet::default();

        // 1. Function parameters are dominating everywhere
        for param in &func.params {
            self.register_def(param.id, "function_param")?;
            dominating.insert(param.id);
        }

        // 2. Verify all basic blocks
        for block in &func.blocks {
            self.verify_block(block, &mut dominating)?;
        }

        Ok(())
    }

    fn register_def(&mut self, val: ValueId, site: &str) -> Result<(), DominanceError> {
        if !self.defined_values.insert(val) {
            return Err(DominanceError::UseBeforeDef(val));
        }
        self.value_def_site.insert(val, site.to_string());
        Ok(())
    }

    fn verify_block(
        &mut self,
        block: &Block,
        dominating: &mut FxHashSet<ValueId>,
    ) -> Result<(), DominanceError> {
        let mut block_dominating = dominating.clone();

        // Block arguments are dominating within this block
        for arg in &block.args {
            self.register_def(arg.id, "block_arg")?;
            block_dominating.insert(arg.id);
        }

        for op in &block.ops {
            self.verify_op(op, &mut block_dominating)?;
        }

        self.verify_terminator(&block.terminator, &block_dominating)?;

        Ok(())
    }

    fn verify_op(
        &mut self,
        op: &RegionOp,
        dominating: &mut FxHashSet<ValueId>,
    ) -> Result<(), DominanceError> {
        // First check that all input operands are dominating
        match &op.kind {
            RegionOpKind::Constant(_) | RegionOpKind::CoordinateQuery { .. } => {}
            RegionOpKind::Unary { input, .. } => {
                self.check_use(*input, dominating)?;
            }
            RegionOpKind::Binary { left, right, .. } => {
                self.check_use(*left, dominating)?;
                self.check_use(*right, dominating)?;
            }
            RegionOpKind::Ternary { a, b, c, .. } => {
                self.check_use(*a, dominating)?;
                self.check_use(*b, dominating)?;
                self.check_use(*c, dominating)?;
            }
            RegionOpKind::Cast { input, .. } => {
                self.check_use(*input, dominating)?;
            }
            RegionOpKind::View { input, .. } => {
                self.check_use(*input, dominating)?;
            }
            RegionOpKind::Call { args, .. } => {
                for arg in args {
                    self.check_use(*arg, dominating)?;
                }
            }
            RegionOpKind::BufferLoad { index, .. } => {
                self.check_use(*index, dominating)?;
            }
            RegionOpKind::BufferStore { index, value, .. } => {
                self.check_use(*index, dominating)?;
                self.check_use(*value, dominating)?;
            }
            RegionOpKind::BufferAlloc { .. } => {}
            RegionOpKind::Region(region) => {
                self.verify_region(region, dominating)?;
            }
            RegionOpKind::Custom { operands, .. } => {
                for opnd in operands {
                    self.check_use(*opnd, dominating)?;
                }
            }
        }

        // Then define results in the dominating scope
        for res in &op.results {
            self.register_def(res.id, "op_result")?;
            dominating.insert(res.id);
        }

        Ok(())
    }

    fn verify_region(
        &mut self,
        region: &StructuredRegion,
        outer_dominating: &FxHashSet<ValueId>,
    ) -> Result<(), DominanceError> {
        // Regional operands must dominate from outside
        match &region.kind {
            RegionKind::Map { inputs, .. } => {
                for inp in inputs {
                    self.check_use(*inp, outer_dominating)?;
                }
            }
            RegionKind::Reduce { input, neutral, .. } => {
                self.check_use(*input, outer_dominating)?;
                self.check_use(*neutral, outer_dominating)?;
            }
            RegionKind::Scan { input, neutral, .. } => {
                self.check_use(*input, outer_dominating)?;
                self.check_use(*neutral, outer_dominating)?;
            }
            RegionKind::Recurrence { init_state, .. } => {
                for s in init_state {
                    self.check_use(*s, outer_dominating)?;
                }
            }
            RegionKind::Condition { cond } => {
                self.check_use(*cond, outer_dominating)?;
            }
            RegionKind::Contraction { left, right, .. } => {
                self.check_use(*left, outer_dominating)?;
                self.check_use(*right, outer_dominating)?;
            }
            RegionKind::Stencil { input, .. } => {
                self.check_use(*input, outer_dominating)?;
            }
            RegionKind::Gather {
                source, indices, ..
            } => {
                self.check_use(*source, outer_dominating)?;
                self.check_use(*indices, outer_dominating)?;
            }
            RegionKind::Scatter {
                target,
                indices,
                updates,
                ..
            } => {
                self.check_use(*target, outer_dominating)?;
                self.check_use(*indices, outer_dominating)?;
                self.check_use(*updates, outer_dominating)?;
            }
            RegionKind::Segmented {
                input,
                segment_offsets,
            } => {
                self.check_use(*input, outer_dominating)?;
                self.check_use(*segment_offsets, outer_dominating)?;
            }
            RegionKind::Custom { operands, .. } => {
                for opnd in operands {
                    self.check_use(*opnd, outer_dominating)?;
                }
            }
        }

        // Inner scope inherits outer dominating values plus region entry arguments
        let mut inner_dominating = outer_dominating.clone();
        for arg in &region.entry_args {
            self.register_def(arg.id, "region_entry_arg")?;
            inner_dominating.insert(arg.id);
        }

        // Verify all blocks in the region
        for block in &region.blocks {
            self.verify_block(block, &mut inner_dominating)?;
        }

        Ok(())
    }

    fn verify_terminator(
        &self,
        term: &Terminator,
        dominating: &FxHashSet<ValueId>,
    ) -> Result<(), DominanceError> {
        match term {
            Terminator::Yield { values } | Terminator::Return { values } => {
                for v in values {
                    self.check_use(*v, dominating)?;
                }
            }
            Terminator::Branch { args, .. } => {
                for arg in args {
                    self.check_use(*arg, dominating)?;
                }
            }
            Terminator::CondBranch {
                cond,
                true_args,
                false_args,
                ..
            } => {
                self.check_use(*cond, dominating)?;
                for arg in true_args {
                    self.check_use(*arg, dominating)?;
                }
                for arg in false_args {
                    self.check_use(*arg, dominating)?;
                }
            }
            Terminator::Trap { .. } | Terminator::Unreachable => {}
        }
        Ok(())
    }

    fn check_use(
        &self,
        val: ValueId,
        dominating: &FxHashSet<ValueId>,
    ) -> Result<(), DominanceError> {
        if !dominating.contains(&val) {
            return Err(DominanceError::OutOfScopeValue(val));
        }
        Ok(())
    }
}

/// Verify that a RegionFunction strictly satisfies dominance and SSA use-def invariants.
pub fn verify_dominance(func: &RegionFunction) -> Result<(), DominanceError> {
    let mut verifier = DominanceVerifier::new();
    verifier.verify_function(func)
}
