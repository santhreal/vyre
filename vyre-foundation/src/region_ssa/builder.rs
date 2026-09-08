//! Builder interface enforcing dominance by construction for Region SSA.

use rustc_hash::FxHashSet;
use vyre_spec::{BinOp, CombineKind, DataType, TernaryOp, UnOp};

use super::{
    Block, BlockArg, BlockId, EffectToken, RegionFunction, RegionId, RegionKind,
    RegionOp, RegionOpKind, ScalarLiteral, StructuredRegion, Terminator, ValueId,
    ViewOp,
};

/// Error produced when a builder operation violates dominance or typing invariants.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum DominanceError {
    /// An operand value is used outside its dominating scope.
    #[error("Value {0} is out of scope or used without a dominating definition")]
    OutOfScopeValue(ValueId),
    /// An operand value is used before its definition within the same block.
    #[error("Value {0} is used before its definition")]
    UseBeforeDef(ValueId),
    /// Block identifier does not exist.
    #[error("Block {0} does not exist in function")]
    NoSuchBlock(BlockId),
    /// Region identifier does not exist.
    #[error("Region {0} does not exist")]
    NoSuchRegion(RegionId),
    /// Terminator mismatch or missing yield.
    #[error("Terminator invariant violation: {0}")]
    InvalidTerminator(String),
    /// Yielded value count or type mismatch.
    #[error("Yield mismatch in region {0}: expected {1} values, got {2}")]
    YieldCountMismatch(RegionId, usize, usize),
}

/// Scope context tracking dominating definitions at the current insertion point.
#[derive(Debug, Clone, Default)]
pub struct ScopeContext {
    /// Set of ValueIds defined in ancestor and dominating scopes.
    dominating_values: FxHashSet<ValueId>,
    /// Stack of regional scopes for lexical containment.
    region_stack: Vec<FxHashSet<ValueId>>,
}

impl ScopeContext {
    /// Create a new empty scope context.
    #[must_use]
    pub fn new() -> Self {
        Self {
            dominating_values: FxHashSet::default(),
            region_stack: Vec::new(),
        }
    }

    /// Introduce a newly defined ValueId into the current dominating scope.
    pub fn define(&mut self, val: ValueId) {
        self.dominating_values.insert(val);
        if let Some(top) = self.region_stack.last_mut() {
            top.insert(val);
        }
    }

    /// Check whether a value is visible in the current dominating scope.
    #[must_use]
    pub fn is_visible(&self, val: ValueId) -> bool {
        self.dominating_values.contains(&val)
    }

    /// Push a new nested region scope.
    pub fn push_region(&mut self) {
        self.region_stack.push(FxHashSet::default());
    }

    /// Pop a nested region scope, removing all inner definitions from dominating scope.
    pub fn pop_region(&mut self) {
        if let Some(inner_defs) = self.region_stack.pop() {
            for val in inner_defs {
                self.dominating_values.remove(&val);
            }
        }
    }
}

/// Scoped builder for constructing Region SSA functions and blocks.
pub struct RegionBuilder {
    next_value_id: u32,
    next_block_id: u32,
    next_region_id: u32,
    next_op_id: u32,
    scope: ScopeContext,
    function: RegionFunction,
    current_block_id: BlockId,
}

impl RegionBuilder {
    /// Start building a new function with the given name and signature.
    pub fn new_function(
        name: impl Into<String>,
        param_types: Vec<(DataType, Option<String>)>,
        return_types: Vec<DataType>,
    ) -> Self {
        let mut builder = Self {
            next_value_id: 0,
            next_block_id: 0,
            next_region_id: 0,
            next_op_id: 0,
            scope: ScopeContext::new(),
            function: RegionFunction {
                name: name.into(),
                shape_params: Vec::new(),
                type_params: Vec::new(),
                effect_params: Vec::new(),
                params: Vec::new(),
                return_types,
                entry_block: BlockId(0),
                blocks: Vec::new(),
            },
            current_block_id: BlockId(0),
        };

        // Create parameters
        for (ty, name_hint) in param_types {
            let val = builder.alloc_value();
            builder.scope.define(val);
            builder.function.params.push(BlockArg::new(val, ty, name_hint));
        }

        // Create entry block
        let entry_id = builder.alloc_block_id();
        builder.function.entry_block = entry_id;
        builder.current_block_id = entry_id;
        builder.function.blocks.push(Block {
            id: entry_id,
            args: Vec::new(),
            ops: Vec::new(),
            terminator: Terminator::Unreachable,
        });

        builder
    }

    /// Allocate a fresh ValueId.
    pub fn alloc_value(&mut self) -> ValueId {
        let val = ValueId(self.next_value_id);
        self.next_value_id += 1;
        val
    }

    /// Allocate a fresh BlockId.
    pub fn alloc_block_id(&mut self) -> BlockId {
        let id = BlockId(self.next_block_id);
        self.next_block_id += 1;
        id
    }

    /// Allocate a fresh RegionId.
    pub fn alloc_region_id(&mut self) -> RegionId {
        let id = RegionId(self.next_region_id);
        self.next_region_id += 1;
        id
    }

    /// Create a new basic block with block arguments and switch insertion point to it.
    pub fn create_block(&mut self, arg_types: Vec<(DataType, Option<String>)>) -> BlockId {
        let block_id = self.alloc_block_id();
        let mut args = Vec::with_capacity(arg_types.len());
        for (ty, hint) in arg_types {
            let val = self.alloc_value();
            self.scope.define(val);
            args.push(BlockArg::new(val, ty, hint));
        }
        self.function.blocks.push(Block {
            id: block_id,
            args,
            ops: Vec::new(),
            terminator: Terminator::Unreachable,
        });
        self.current_block_id = block_id;
        block_id
    }

    /// Set the current insertion block.
    pub fn set_current_block(&mut self, block_id: BlockId) -> Result<(), DominanceError> {
        if self.function.get_block(block_id).is_none() {
            return Err(DominanceError::NoSuchBlock(block_id));
        }
        self.current_block_id = block_id;
        Ok(())
    }

    fn check_operand(&self, val: ValueId) -> Result<(), DominanceError> {
        if !self.scope.is_visible(val) {
            return Err(DominanceError::OutOfScopeValue(val));
        }
        Ok(())
    }

    fn emit_op_internal(
        &mut self,
        kind: RegionOpKind,
        results: Vec<(DataType, Option<String>)>,
    ) -> Result<Vec<ValueId>, DominanceError> {
        let op_id = self.next_op_id;
        self.next_op_id += 1;

        let mut res_args = Vec::with_capacity(results.len());
        let mut res_vals = Vec::with_capacity(results.len());
        for (ty, hint) in results {
            let val = self.alloc_value();
            self.scope.define(val);
            res_args.push(BlockArg::new(val, ty, hint));
            res_vals.push(val);
        }

        let op = RegionOp {
            id: op_id,
            results: res_args,
            kind,
            location: None,
        };

        let block = self
            .function
            .get_block_mut(self.current_block_id)
            .ok_or(DominanceError::NoSuchBlock(self.current_block_id))?;
        block.ops.push(op);

        Ok(res_vals)
    }

    /// Emit a scalar constant operation.
    pub fn emit_constant(&mut self, lit: ScalarLiteral) -> Result<ValueId, DominanceError> {
        let ty = lit.data_type();
        let res = self.emit_op_internal(RegionOpKind::Constant(lit), vec![(ty, None)])?;
        Ok(res[0])
    }

    /// Emit an element-wise binary arithmetic operation.
    pub fn emit_binary(
        &mut self,
        op: BinOp,
        left: ValueId,
        right: ValueId,
        result_type: DataType,
    ) -> Result<ValueId, DominanceError> {
        self.check_operand(left)?;
        self.check_operand(right)?;
        let res = self.emit_op_internal(
            RegionOpKind::Binary { op, left, right },
            vec![(result_type, None)],
        )?;
        Ok(res[0])
    }

    /// Emit an element-wise unary arithmetic operation.
    pub fn emit_unary(
        &mut self,
        op: UnOp,
        input: ValueId,
        result_type: DataType,
    ) -> Result<ValueId, DominanceError> {
        self.check_operand(input)?;
        let res = self.emit_op_internal(
            RegionOpKind::Unary { op, input },
            vec![(result_type, None)],
        )?;
        Ok(res[0])
    }

    /// Emit a ternary operation (e.g. Select).
    pub fn emit_ternary(
        &mut self,
        op: TernaryOp,
        a: ValueId,
        b: ValueId,
        c: ValueId,
        result_type: DataType,
    ) -> Result<ValueId, DominanceError> {
        self.check_operand(a)?;
        self.check_operand(b)?;
        self.check_operand(c)?;
        let res = self.emit_op_internal(
            RegionOpKind::Ternary { op, a, b, c },
            vec![(result_type, None)],
        )?;
        Ok(res[0])
    }

    /// Emit a type conversion cast.
    pub fn emit_cast(
        &mut self,
        input: ValueId,
        target_type: DataType,
    ) -> Result<ValueId, DominanceError> {
        self.check_operand(input)?;
        let res = self.emit_op_internal(
            RegionOpKind::Cast { input, target_type: target_type.clone() },
            vec![(target_type, None)],
        )?;
        Ok(res[0])
    }

    /// Emit a non-materializing semantic view transformation.
    pub fn emit_view(
        &mut self,
        input: ValueId,
        view: ViewOp,
        result_type: DataType,
    ) -> Result<ValueId, DominanceError> {
        self.check_operand(input)?;
        let res = self.emit_op_internal(
            RegionOpKind::View { input, view },
            vec![(result_type, None)],
        )?;
        Ok(res[0])
    }

    /// Emit a coordinate query (e.g. global ID).
    pub fn emit_coordinate_query(
        &mut self,
        name: impl Into<String>,
        result_type: DataType,
    ) -> Result<ValueId, DominanceError> {
        let res = self.emit_op_internal(
            RegionOpKind::CoordinateQuery { name: name.into() },
            vec![(result_type, None)],
        )?;
        Ok(res[0])
    }

    /// Emit a semantic buffer load.
    pub fn emit_buffer_load(
        &mut self,
        buffer: impl Into<String>,
        index: ValueId,
        element_type: DataType,
        effect_in: Option<EffectToken>,
    ) -> Result<ValueId, DominanceError> {
        self.check_operand(index)?;
        let res = self.emit_op_internal(
            RegionOpKind::BufferLoad {
                buffer: buffer.into(),
                index,
                effect_in,
            },
            vec![(element_type, None)],
        )?;
        Ok(res[0])
    }

    /// Emit a semantic buffer store.
    pub fn emit_buffer_store(
        &mut self,
        buffer: impl Into<String>,
        index: ValueId,
        value: ValueId,
        effect_in: Option<EffectToken>,
    ) -> Result<(), DominanceError> {
        self.check_operand(index)?;
        self.check_operand(value)?;
        self.emit_op_internal(
            RegionOpKind::BufferStore {
                buffer: buffer.into(),
                index,
                value,
                effect_in,
            },
            Vec::new(),
        )?;
        Ok(())
    }

    /// Build a parallel map region with explicit yields.
    pub fn build_map_region<F>(
        &mut self,
        inputs: Vec<ValueId>,
        domain_shape: Vec<u64>,
        input_elem_types: Vec<DataType>,
        yield_types: Vec<DataType>,
        body: F,
    ) -> Result<Vec<ValueId>, DominanceError>
    where
        F: FnOnce(&mut Self, Vec<ValueId>) -> Result<Vec<ValueId>, DominanceError>,
    {
        for inp in &inputs {
            self.check_operand(*inp)?;
        }

        let region_id = self.alloc_region_id();
        self.scope.push_region();

        let mut entry_args = Vec::with_capacity(input_elem_types.len());
        let mut entry_vals = Vec::with_capacity(input_elem_types.len());
        for ty in input_elem_types {
            let val = self.alloc_value();
            self.scope.define(val);
            entry_args.push(BlockArg::new(val, ty, None));
            entry_vals.push(val);
        }

        // Inner entry block
        let inner_block_id = self.alloc_block_id();
        let saved_block = self.current_block_id;
        self.current_block_id = inner_block_id;
        self.function.blocks.push(Block {
            id: inner_block_id,
            args: Vec::new(),
            ops: Vec::new(),
            terminator: Terminator::Unreachable,
        });

        let mut inner_blocks = Vec::new();
        // Build inner body
        let yielded_vals = body(self, entry_vals)?;

        // Close inner block with explicit yield
        if yielded_vals.len() != yield_types.len() {
            return Err(DominanceError::YieldCountMismatch(
                region_id,
                yield_types.len(),
                yielded_vals.len(),
            ));
        }
        for val in &yielded_vals {
            self.check_operand(*val)?;
        }

        if let Some(cur) = self.function.get_block_mut(self.current_block_id) {
            cur.terminator = Terminator::Yield {
                values: yielded_vals,
            };
        }

        // Drain inner blocks from function that belong to the region
        let mut idx = 0;
        while idx < self.function.blocks.len() {
            if self.function.blocks[idx].id == inner_block_id {
                let blk = self.function.blocks.remove(idx);
                inner_blocks.push(blk);
            } else {
                idx += 1;
            }
        }

        self.scope.pop_region();
        self.current_block_id = saved_block;

        let region = StructuredRegion {
            id: region_id,
            kind: RegionKind::Map {
                inputs,
                domain_shape,
            },
            entry_args,
            blocks: inner_blocks,
            yielded_types: yield_types.clone(),
        };

        let result_tuples = yield_types.into_iter().map(|ty| (ty, None)).collect();
        let results = self.emit_op_internal(
            RegionOpKind::Region(Box::new(region)),
            result_tuples,
        )?;

        Ok(results)
    }

    /// Build an associative reduction region.
    pub fn build_reduce_region<F>(
        &mut self,
        input: ValueId,
        axis: usize,
        neutral: ValueId,
        combine: CombineKind,
        elem_type: DataType,
        body: F,
    ) -> Result<ValueId, DominanceError>
    where
        F: FnOnce(&mut Self, ValueId, ValueId) -> Result<ValueId, DominanceError>,
    {
        self.check_operand(input)?;
        self.check_operand(neutral)?;

        let region_id = self.alloc_region_id();
        self.scope.push_region();

        let acc_val = self.alloc_value();
        let elt_val = self.alloc_value();
        self.scope.define(acc_val);
        self.scope.define(elt_val);

        let entry_args = vec![
            BlockArg::new(acc_val, elem_type.clone(), Some("acc".into())),
            BlockArg::new(elt_val, elem_type.clone(), Some("elt".into())),
        ];

        let inner_block_id = self.alloc_block_id();
        let saved_block = self.current_block_id;
        self.current_block_id = inner_block_id;

        self.function.blocks.push(Block {
            id: inner_block_id,
            args: Vec::new(),
            ops: Vec::new(),
            terminator: Terminator::Unreachable,
        });

        let mut inner_blocks = Vec::new();
        let combined = body(self, acc_val, elt_val)?;
        self.check_operand(combined)?;

        if let Some(cur) = self.function.get_block_mut(self.current_block_id) {
            cur.terminator = Terminator::Yield {
                values: vec![combined],
            };
        }

        let mut idx = 0;
        while idx < self.function.blocks.len() {
            if self.function.blocks[idx].id == inner_block_id {
                let blk = self.function.blocks.remove(idx);
                inner_blocks.push(blk);
            } else {
                idx += 1;
            }
        }
        self.scope.pop_region();
        self.current_block_id = saved_block;

        let region = StructuredRegion {
            id: region_id,
            kind: RegionKind::Reduce {
                input,
                axis,
                neutral,
                combine,
            },
            entry_args,
            blocks: inner_blocks,
            yielded_types: vec![elem_type.clone()],
        };

        let results = self.emit_op_internal(
            RegionOpKind::Region(Box::new(region)),
            vec![(elem_type, None)],
        )?;

        Ok(results[0])
    }

    /// Build a bounded recurrence / fixpoint loop region with loop-carried state and explicit yields.
    pub fn build_recurrence_region<F>(
        &mut self,
        init_state: Vec<ValueId>,
        state_types: Vec<DataType>,
        trip_count_bound: u64,
        body: F,
    ) -> Result<Vec<ValueId>, DominanceError>
    where
        F: FnOnce(&mut Self, ValueId, Vec<ValueId>) -> Result<Vec<ValueId>, DominanceError>,
    {
        for s in &init_state {
            self.check_operand(*s)?;
        }

        let region_id = self.alloc_region_id();
        self.scope.push_region();

        let iv_val = self.alloc_value();
        self.scope.define(iv_val);
        let mut entry_args = vec![BlockArg::new(iv_val, DataType::U32, Some("iv".into()))];
        let mut state_vals = Vec::with_capacity(state_types.len());

        for (i, ty) in state_types.iter().enumerate() {
            let s_val = self.alloc_value();
            self.scope.define(s_val);
            entry_args.push(BlockArg::new(s_val, ty.clone(), Some(format!("state_{i}"))));
            state_vals.push(s_val);
        }

        let inner_block_id = self.alloc_block_id();
        let saved_block = self.current_block_id;
        self.current_block_id = inner_block_id;
        self.function.blocks.push(Block {
            id: inner_block_id,
            args: Vec::new(),
            ops: Vec::new(),
            terminator: Terminator::Unreachable,
        });

        let mut inner_blocks = Vec::new();
        let next_states = body(self, iv_val, state_vals)?;
        if next_states.len() != state_types.len() {
            return Err(DominanceError::YieldCountMismatch(
                region_id,
                state_types.len(),
                next_states.len(),
            ));
        }
        for s in &next_states {
            self.check_operand(*s)?;
        }

        if let Some(cur) = self.function.get_block_mut(self.current_block_id) {
            cur.terminator = Terminator::Yield {
                values: next_states,
            };
        }

        let mut idx = 0;
        while idx < self.function.blocks.len() {
            if self.function.blocks[idx].id == inner_block_id {
                let blk = self.function.blocks.remove(idx);
                inner_blocks.push(blk);
            } else {
                idx += 1;
            }
        }
        self.current_block_id = saved_block;

        let region = StructuredRegion {
            id: region_id,
            kind: RegionKind::Recurrence {
                init_state,
                trip_count_bound,
            },
            entry_args,
            blocks: inner_blocks,
            yielded_types: state_types.clone(),
        };

        let result_tuples = state_types.into_iter().map(|ty| (ty, None)).collect();
        let results = self.emit_op_internal(
            RegionOpKind::Region(Box::new(region)),
            result_tuples,
        )?;

        Ok(results)
    }

    /// Terminate current block with Return.
    pub fn terminate_return(&mut self, values: Vec<ValueId>) -> Result<(), DominanceError> {
        for val in &values {
            self.check_operand(*val)?;
        }
        let block = self
            .function
            .get_block_mut(self.current_block_id)
            .ok_or(DominanceError::NoSuchBlock(self.current_block_id))?;
        block.terminator = Terminator::Return { values };
        Ok(())
    }

    /// Terminate current block with unconditional Branch.
    pub fn terminate_branch(
        &mut self,
        target: BlockId,
        args: Vec<ValueId>,
    ) -> Result<(), DominanceError> {
        for arg in &args {
            self.check_operand(*arg)?;
        }
        let block = self
            .function
            .get_block_mut(self.current_block_id)
            .ok_or(DominanceError::NoSuchBlock(self.current_block_id))?;
        block.terminator = Terminator::Branch { target, args };
        Ok(())
    }

    /// Terminate current block with CondBranch.
    pub fn terminate_cond_branch(
        &mut self,
        cond: ValueId,
        true_dest: BlockId,
        true_args: Vec<ValueId>,
        false_dest: BlockId,
        false_args: Vec<ValueId>,
    ) -> Result<(), DominanceError> {
        self.check_operand(cond)?;
        for arg in &true_args {
            self.check_operand(*arg)?;
        }
        for arg in &false_args {
            self.check_operand(*arg)?;
        }
        let block = self
            .function
            .get_block_mut(self.current_block_id)
            .ok_or(DominanceError::NoSuchBlock(self.current_block_id))?;
        block.terminator = Terminator::CondBranch {
            cond,
            true_dest,
            true_args,
            false_dest,
            false_args,
        };
        Ok(())
    }

    /// Finalize and return the built RegionFunction.
    pub fn build(self) -> Result<RegionFunction, DominanceError> {
        // Run dominance verification to certify the finished function
        super::dominance::verify_dominance(&self.function)?;
        Ok(self.function)
    }
}
