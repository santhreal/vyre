//! SSA optimization passes running directly on Region SSA.

use rustc_hash::{FxHashMap, FxHashSet};
use vyre_spec::{BinOp, DataType, UnOp};

use super::{
    RegionFunction, RegionModule, RegionOp, RegionOpKind, ScalarLiteral, Terminator, ValueId,
};

/// Tracks ValueId mapping across optimization passes, certifying value stability.
#[derive(Debug, Clone, Default)]
pub struct ValueRemap {
    mapping: FxHashMap<ValueId, ValueId>,
}

impl ValueRemap {
    /// Create a new empty value remap tracker.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Record that `old_val` was remapped to `new_val`.
    pub fn insert(&mut self, old_val: ValueId, new_val: ValueId) {
        self.mapping.insert(old_val, new_val);
    }

    /// Resolve the current active ValueId for an original ValueId.
    #[must_use]
    pub fn resolve(&self, val: ValueId) -> ValueId {
        let mut curr = val;
        while let Some(&next) = self.mapping.get(&curr) {
            curr = next;
        }
        curr
    }

    /// Check whether a value ID remained stable (unaltered) across transformations.
    #[must_use]
    pub fn is_stable(&self, val: ValueId) -> bool {
        !self.mapping.contains_key(&val) || self.resolve(val) == val
    }
}

/// Constant propagation and algebraic folding over Region SSA.
#[derive(Debug, Default)]
pub struct RegionSsaConstProp;

impl RegionSsaConstProp {
    /// Run constant propagation over a [`RegionFunction`], returning the transformed function and remap.
    #[must_use]
    pub fn run_function(func: &RegionFunction) -> (RegionFunction, ValueRemap) {
        let mut remap = ValueRemap::new();
        let mut known_constants: FxHashMap<ValueId, ScalarLiteral> = FxHashMap::default();
        let mut new_func = func.clone();

        for block in &mut new_func.blocks {
            let mut new_ops = Vec::with_capacity(block.ops.len());
            for op in block.ops.drain(..) {
                let transformed_op = Self::fold_op(op, &mut known_constants, &mut remap);
                new_ops.push(transformed_op);
            }
            block.ops = new_ops;
        }

        (new_func, remap)
    }

    fn fold_op(
        mut op: RegionOp,
        known_constants: &mut FxHashMap<ValueId, ScalarLiteral>,
        remap: &mut ValueRemap,
    ) -> RegionOp {
        match &mut op.kind {
            RegionOpKind::Constant(lit) => {
                if let Some(res) = op.results.first() {
                    known_constants.insert(res.id, lit.clone());
                }
            }
            RegionOpKind::Binary {
                op: binop,
                left,
                right,
            } => {
                *left = remap.resolve(*left);
                *right = remap.resolve(*right);
                if let (Some(l_lit), Some(r_lit)) =
                    (known_constants.get(left), known_constants.get(right))
                {
                    if let Some(folded) = Self::fold_binary(*binop, l_lit, r_lit) {
                        if let Some(res) = op.results.first() {
                            known_constants.insert(res.id, folded.clone());
                        }
                        op.kind = RegionOpKind::Constant(folded);
                    }
                }
            }
            RegionOpKind::Unary { op: unop, input } => {
                *input = remap.resolve(*input);
                if let Some(in_lit) = known_constants.get(input) {
                    if let Some(folded) = Self::fold_unary(unop.clone(), in_lit) {
                        if let Some(res) = op.results.first() {
                            known_constants.insert(res.id, folded.clone());
                        }
                        op.kind = RegionOpKind::Constant(folded);
                    }
                }
            }
            RegionOpKind::Cast { input, target_type } => {
                *input = remap.resolve(*input);
                if let Some(in_lit) = known_constants.get(input) {
                    if let Some(folded) = Self::fold_cast(in_lit, target_type.clone()) {
                        if let Some(res) = op.results.first() {
                            known_constants.insert(res.id, folded.clone());
                        }
                        op.kind = RegionOpKind::Constant(folded);
                    }
                }
            }
            RegionOpKind::BufferLoad { index, .. } => {
                *index = remap.resolve(*index);
            }
            RegionOpKind::BufferStore { index, value, .. } => {
                *index = remap.resolve(*index);
                *value = remap.resolve(*value);
            }
            _ => {}
        }
        op
    }

    fn fold_binary(
        op: BinOp,
        left: &ScalarLiteral,
        right: &ScalarLiteral,
    ) -> Option<ScalarLiteral> {
        match (left, right) {
            (ScalarLiteral::U32(a), ScalarLiteral::U32(b)) => match op {
                BinOp::Add => Some(ScalarLiteral::U32(a.wrapping_add(*b))),
                BinOp::Sub => Some(ScalarLiteral::U32(a.wrapping_sub(*b))),
                BinOp::Mul => Some(ScalarLiteral::U32(a.wrapping_mul(*b))),
                BinOp::Div => {
                    if *b != 0 {
                        Some(ScalarLiteral::U32(a / b))
                    } else {
                        None
                    }
                }
                BinOp::BitAnd => Some(ScalarLiteral::U32(a & b)),
                BinOp::BitOr => Some(ScalarLiteral::U32(a | b)),
                BinOp::BitXor => Some(ScalarLiteral::U32(a ^ b)),
                BinOp::Shl => Some(ScalarLiteral::U32(a.wrapping_shl(*b))),
                BinOp::Shr => Some(ScalarLiteral::U32(a.wrapping_shr(*b))),
                BinOp::Eq => Some(ScalarLiteral::Bool(a == b)),
                BinOp::Ne => Some(ScalarLiteral::Bool(a != b)),
                BinOp::Lt => Some(ScalarLiteral::Bool(a < b)),
                BinOp::Le => Some(ScalarLiteral::Bool(a <= b)),
                BinOp::Gt => Some(ScalarLiteral::Bool(a > b)),
                BinOp::Ge => Some(ScalarLiteral::Bool(a >= b)),
                _ => None,
            },
            (ScalarLiteral::I32(a), ScalarLiteral::I32(b)) => match op {
                BinOp::Add => Some(ScalarLiteral::I32(a.wrapping_add(*b))),
                BinOp::Sub => Some(ScalarLiteral::I32(a.wrapping_sub(*b))),
                BinOp::Mul => Some(ScalarLiteral::I32(a.wrapping_mul(*b))),
                BinOp::Eq => Some(ScalarLiteral::Bool(a == b)),
                BinOp::Ne => Some(ScalarLiteral::Bool(a != b)),
                _ => None,
            },
            _ => None,
        }
    }

    fn fold_unary(op: UnOp, input: &ScalarLiteral) -> Option<ScalarLiteral> {
        match input {
            ScalarLiteral::U32(v) => match op {
                UnOp::BitNot => Some(ScalarLiteral::U32(!*v)),
                UnOp::Negate => Some(ScalarLiteral::U32(v.wrapping_neg())),
                _ => None,
            },
            ScalarLiteral::Bool(b) => match op {
                UnOp::LogicalNot => Some(ScalarLiteral::Bool(!*b)),
                _ => None,
            },
            _ => None,
        }
    }

    fn fold_cast(input: &ScalarLiteral, target: DataType) -> Option<ScalarLiteral> {
        match (input, target) {
            (ScalarLiteral::U32(v), DataType::I32) => Some(ScalarLiteral::I32(*v as i32)),
            (ScalarLiteral::I32(v), DataType::U32) => Some(ScalarLiteral::U32(*v as u32)),
            (ScalarLiteral::U32(v), DataType::F32) => Some(ScalarLiteral::F32(*v as f32)),
            _ => None,
        }
    }
}

/// Dead code elimination over unused SSA operations.
#[derive(Debug, Default)]
pub struct RegionSsaDce;

impl RegionSsaDce {
    /// Run dead code elimination on a [`RegionFunction`].
    #[must_use]
    pub fn run_function(func: &RegionFunction) -> RegionFunction {
        let mut used_values: FxHashSet<ValueId> = FxHashSet::default();

        // 1. Mark values used in terminators and side-effecting ops
        for block in &func.blocks {
            Self::mark_terminator_uses(&block.terminator, &mut used_values);
            for op in &block.ops {
                if Self::is_side_effecting(&op.kind) {
                    Self::mark_op_uses(op, &mut used_values);
                }
            }
        }

        // 2. Iteratively mark transitively used values
        let mut changed = true;
        while changed {
            changed = false;
            for block in &func.blocks {
                for op in &block.ops {
                    let produces_used = op.results.iter().any(|res| used_values.contains(&res.id));
                    if produces_used {
                        let before_len = used_values.len();
                        Self::mark_op_uses(op, &mut used_values);
                        if used_values.len() > before_len {
                            changed = true;
                        }
                    }
                }
            }
        }

        // 3. Filter out unused pure ops
        let mut new_func = func.clone();
        for block in &mut new_func.blocks {
            block.ops.retain(|op| {
                Self::is_side_effecting(&op.kind)
                    || op.results.iter().any(|res| used_values.contains(&res.id))
            });
        }

        new_func
    }

    fn is_side_effecting(kind: &RegionOpKind) -> bool {
        matches!(
            kind,
            RegionOpKind::BufferStore { .. }
                | RegionOpKind::Call { .. }
                | RegionOpKind::Region { .. }
                | RegionOpKind::Custom { .. }
        )
    }

    fn mark_op_uses(op: &RegionOp, used: &mut FxHashSet<ValueId>) {
        match &op.kind {
            RegionOpKind::Unary { input, .. }
            | RegionOpKind::Cast { input, .. }
            | RegionOpKind::View { input, .. } => {
                used.insert(*input);
            }
            RegionOpKind::Binary { left, right, .. } => {
                used.insert(*left);
                used.insert(*right);
            }
            RegionOpKind::Ternary { a, b, c, .. } => {
                used.insert(*a);
                used.insert(*b);
                used.insert(*c);
            }
            RegionOpKind::Call { args, .. } => {
                for arg in args {
                    used.insert(*arg);
                }
            }
            RegionOpKind::BufferLoad { index, .. } => {
                used.insert(*index);
            }
            RegionOpKind::BufferStore { index, value, .. } => {
                used.insert(*index);
                used.insert(*value);
            }
            _ => {}
        }
    }

    fn mark_terminator_uses(term: &Terminator, used: &mut FxHashSet<ValueId>) {
        match term {
            Terminator::Yield { values } | Terminator::Return { values } => {
                for v in values {
                    used.insert(*v);
                }
            }
            Terminator::Branch { args, .. } => {
                for arg in args {
                    used.insert(*arg);
                }
            }
            Terminator::CondBranch {
                cond,
                true_args,
                false_args,
                ..
            } => {
                used.insert(*cond);
                for arg in true_args {
                    used.insert(*arg);
                }
                for arg in false_args {
                    used.insert(*arg);
                }
            }
            _ => {}
        }
    }
}

/// Pipeline running SSA optimizer passes over a [`RegionModule`].
#[derive(Debug, Default)]
pub struct RegionSsaOptimizer;

impl RegionSsaOptimizer {
    /// Optimize a [`RegionModule`] using constant propagation and dead code elimination.
    #[must_use]
    pub fn optimize_module(module: &RegionModule) -> (RegionModule, ValueRemap) {
        let mut new_module = module.clone();
        let mut total_remap = ValueRemap::new();

        new_module.functions = new_module
            .functions
            .iter()
            .map(|func| {
                let (f_const, remap) = RegionSsaConstProp::run_function(func);
                for (old_v, new_v) in remap.mapping {
                    total_remap.insert(old_v, new_v);
                }
                RegionSsaDce::run_function(&f_const)
            })
            .collect();

        (new_module, total_remap)
    }
}
