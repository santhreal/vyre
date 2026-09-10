//! Typed region-based SSA intermediate representation.
//!
//! Replaces the statement-based semantic level with:
//! - Stable, typed `ValueId` values
//! - Block arguments instead of phi nodes
//! - Explicit `Yield` terminators for regions and blocks
//! - Immutable definitions and dominance by construction
//! - First-class generic modules and functions with shape, type, and effect parameters
//! - Structured regions for parallel map, reduction, scan, recurrence, and bounded control
//! - Non-materializing semantic views (slice, permute, broadcast, reshape, pad)

pub(crate) mod builder;
pub(crate) mod dominance;
pub(crate) mod lower;
pub(crate) mod opt;

pub use builder::{DominanceError, RegionBuilder, ScopeContext};
pub use dominance::{verify_dominance, DominanceVerifier};
pub use lower::{lower_program_to_region_ssa, lower_region_ssa_to_program, RegionSsaError};
pub use opt::{RegionSsaConstProp, RegionSsaDce, RegionSsaOptimizer, ValueRemap};

use crate::extension::ExtensionCatalogBundle;
use serde::{Deserialize, Serialize};
use std::fmt;
use vyre_spec::ExtensionIdentity;
use vyre_spec::{BinOp, CombineKind, DataType, SideEffectClass, TernaryOp, UnOp};

/// Stable identifier of an immutable SSA value.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Serialize, Deserialize)]
pub struct ValueId(pub u32);

impl fmt::Display for ValueId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "v{}", self.0)
    }
}

/// Identifier of a basic block within a function or region.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Serialize, Deserialize)]
pub struct BlockId(pub u32);

impl fmt::Display for BlockId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "bb{}", self.0)
    }
}

/// Identifier of a structured nested region.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Serialize, Deserialize)]
pub struct RegionId(pub u32);

impl fmt::Display for RegionId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "region_{}", self.0)
    }
}

/// Typed block argument in SSA form.
#[derive(Clone, PartialEq, Eq, Hash, Debug, Serialize, Deserialize)]
pub struct BlockArg {
    /// SSA Value identifier.
    pub id: ValueId,
    /// Data type of the value.
    pub ty: DataType,
    /// Optional human-readable name hint.
    pub name_hint: Option<String>,
}

impl BlockArg {
    /// Construct a new typed block argument.
    #[must_use]
    pub fn new(id: ValueId, ty: DataType, name_hint: Option<String>) -> Self {
        Self { id, ty, name_hint }
    }
}

/// Generic shape parameter declared on a function or region.
#[derive(Clone, PartialEq, Eq, Hash, Debug, Serialize, Deserialize)]
pub struct ShapeParam {
    /// Symbolic parameter name (e.g. `$N`).
    pub name: String,
    /// Expected rank constraint, if any.
    pub rank: Option<usize>,
}

/// Generic type parameter declared on a function or region.
#[derive(Clone, PartialEq, Eq, Hash, Debug, Serialize, Deserialize)]
pub struct TypeParam {
    /// Type parameter name (e.g. `$T`).
    pub name: String,
    /// Required trait or concept bounds.
    pub bounds: Vec<String>,
}

/// Generic effect parameter declared on a function or region.
#[derive(Clone, PartialEq, Eq, Hash, Debug, Serialize, Deserialize)]
pub struct EffectParam {
    /// Effect parameter name.
    pub name: String,
    /// Ceiling side-effect class.
    pub effect_class: SideEffectClass,
}

/// Identifier of an explicit effect and ordering token in a region function.
///
/// Names one ordering edge between region ops. The token record carrying an
/// effect kind and its obligation is `memory_model::obligations::EffectToken`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Serialize, Deserialize)]
pub struct EffectTokenId(pub u32);

/// Scalar literal constant.
///
/// The one literal value space of this crate's IR. Statement IR (`ir::Expr`)
/// carries only the 32-bit and boolean widths, so lowering a `U64`, `I64` or
/// `F64` literal to it goes through [`Self::to_expr`], which rejects a value
/// that does not survive the narrowing.
#[derive(Clone, Copy, PartialEq, Debug, Serialize, Deserialize)]
pub enum ScalarLiteral {
    /// 32-bit unsigned integer.
    U32(u32),
    /// 32-bit signed integer.
    I32(i32),
    /// 64-bit unsigned integer.
    U64(u64),
    /// 64-bit signed integer.
    I64(i64),
    /// 32-bit IEEE float.
    F32(f32),
    /// 64-bit IEEE float.
    F64(f64),
    /// Boolean proposition.
    Bool(bool),
}

impl Eq for ScalarLiteral {}

impl std::hash::Hash for ScalarLiteral {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        match self {
            Self::U32(v) => (0u8, *v).hash(state),
            Self::I32(v) => (1u8, *v).hash(state),
            Self::U64(v) => (2u8, *v).hash(state),
            Self::I64(v) => (3u8, *v).hash(state),
            Self::F32(v) => (4u8, v.to_bits()).hash(state),
            Self::F64(v) => (5u8, v.to_bits()).hash(state),
            Self::Bool(v) => (6u8, *v).hash(state),
        }
    }
}

impl ScalarLiteral {
    /// Infer the primitive data type of this literal.
    #[must_use]
    pub const fn data_type(&self) -> DataType {
        match self {
            Self::U32(_) => DataType::U32,
            Self::I32(_) => DataType::I32,
            Self::U64(_) => DataType::U64,
            Self::I64(_) => DataType::I64,
            Self::F32(_) => DataType::F32,
            Self::F64(_) => DataType::F64,
            Self::Bool(_) => DataType::Bool,
        }
    }
}

/// Non-materializing semantic view over a tensor/buffer value.
#[derive(Clone, PartialEq, Eq, Hash, Debug, Serialize, Deserialize)]
pub enum ViewOp {
    /// Slices along an axis with start, length, and stride.
    Slice {
        /// Dimension index.
        axis: usize,
        /// Start offset.
        start: u64,
        /// Slice length.
        length: u64,
        /// Step stride.
        stride: u64,
    },
    /// Permutes dimensions by axis permutation index list.
    Permute {
        /// Permuted axes.
        axes: Vec<usize>,
    },
    /// Broadcasts leading dimensions to target shape.
    Broadcast {
        /// Target shape extents.
        target_shape: Vec<u64>,
    },
    /// Reshapes without data movement.
    Reshape {
        /// New shape extents.
        new_shape: Vec<u64>,
    },
    /// Symmetric or asymmetric padding.
    Pad {
        /// Low padding per dimension.
        low: Vec<u64>,
        /// High padding per dimension.
        high: Vec<u64>,
    },
}

/// Structured region kind describing high-level parallel, recurrence, or control semantics.
#[derive(Clone, PartialEq, Eq, Hash, Debug, Serialize, Deserialize)]
pub enum RegionKind {
    /// Parallel map over an iteration domain.
    Map {
        /// Input value operands mapped over.
        inputs: Vec<ValueId>,
        /// Domain extents.
        domain_shape: Vec<u64>,
    },
    /// Associative reduction across an axis.
    Reduce {
        /// Input value being reduced.
        input: ValueId,
        /// Reduction axis index.
        axis: usize,
        /// Neutral identity value.
        neutral: ValueId,
        /// Combine operator kind.
        combine: CombineKind,
    },
    /// Inclusive or exclusive prefix scan.
    Scan {
        /// Input value scanned.
        input: ValueId,
        /// Scan axis.
        axis: usize,
        /// Neutral identity value.
        neutral: ValueId,
        /// Whether the scan is inclusive.
        is_inclusive: bool,
    },
    /// Bounded recurrence / fixpoint loop.
    Recurrence {
        /// Initial loop-carried state values.
        init_state: Vec<ValueId>,
        /// Validated upper bound on loop trip count.
        trip_count_bound: u64,
    },
    /// Structured conditional branch.
    Condition {
        /// Boolean condition value.
        cond: ValueId,
    },
    /// Generalized tensor contraction.
    Contraction {
        /// Left tensor operand.
        left: ValueId,
        /// Right tensor operand.
        right: ValueId,
        /// Axes contracted.
        contracted_axes: Vec<usize>,
    },
    /// Stencil / neighborhood window computation.
    Stencil {
        /// Input tensor operand.
        input: ValueId,
        /// Window shape.
        window_shape: Vec<u64>,
        /// Strides per dimension.
        strides: Vec<u64>,
    },
    /// Non-mutating gather indexing.
    Gather {
        /// Source tensor.
        source: ValueId,
        /// Index tensor.
        indices: ValueId,
        /// Axis gathered along.
        axis: usize,
    },
    /// Non-mutating scatter update yielding a new tensor.
    Scatter {
        /// Base target tensor.
        target: ValueId,
        /// Indices tensor.
        indices: ValueId,
        /// Updates tensor.
        updates: ValueId,
        /// Axis scattered along.
        axis: usize,
    },
    /// Segmented / ragged iteration.
    Segmented {
        /// Input tensor.
        input: ValueId,
        /// Segment offsets.
        segment_offsets: ValueId,
    },
    /// Declarative extension region.
    Custom {
        /// Extension identity.
        extension_id: ExtensionIdentity,
        /// Operands passed to region.
        operands: Vec<ValueId>,
    },
}

/// A structured region holding child basic blocks and explicit yields.
#[derive(Clone, PartialEq, Eq, Hash, Debug, Serialize, Deserialize)]
pub struct StructuredRegion {
    /// Region identifier.
    pub id: RegionId,
    /// Semantic kind of region.
    pub kind: RegionKind,
    /// Block arguments introduced at region entry.
    pub entry_args: Vec<BlockArg>,
    /// Child basic blocks within this region.
    pub blocks: Vec<Block>,
    /// Result types yielded by the region.
    pub yielded_types: Vec<DataType>,
}

/// Specific operation kind in Region SSA.
#[derive(Clone, PartialEq, Eq, Hash, Debug, Serialize, Deserialize)]
pub enum RegionOpKind {
    /// Scalar literal constant.
    Constant(ScalarLiteral),
    /// Unary arithmetic / bitwise operation.
    Unary {
        /// Unary operator.
        op: UnOp,
        /// Input value.
        input: ValueId,
    },
    /// Binary arithmetic / bitwise operation.
    Binary {
        /// Binary operator.
        op: BinOp,
        /// Left input value.
        left: ValueId,
        /// Right input value.
        right: ValueId,
    },
    /// Ternary select / FMA operation.
    Ternary {
        /// Ternary operator.
        op: TernaryOp,
        /// First operand.
        a: ValueId,
        /// Second operand.
        b: ValueId,
        /// Third operand.
        c: ValueId,
    },
    /// Type conversion cast.
    Cast {
        /// Input value.
        input: ValueId,
        /// Target data type.
        target_type: DataType,
    },
    /// Non-materializing semantic view transformation.
    View {
        /// Source value.
        input: ValueId,
        /// View operation.
        view: ViewOp,
    },
    /// Invocation / workgroup coordinate query.
    CoordinateQuery {
        /// Query name (e.g. "global_id_x", "local_id_x", "workgroup_id_x").
        name: String,
    },
    /// Pure function call.
    Call {
        /// Target function name.
        callee: String,
        /// Arguments passed.
        args: Vec<ValueId>,
    },
    /// Semantic buffer read with effect token.
    BufferLoad {
        /// Buffer name.
        buffer: String,
        /// Linear or element index.
        index: ValueId,
        /// Ordering effect token.
        effect_in: Option<EffectTokenId>,
    },
    /// Semantic buffer write yielding a new effect token.
    BufferStore {
        /// Buffer name.
        buffer: String,
        /// Linear or element index.
        index: ValueId,
        /// Value written.
        value: ValueId,
        /// Ordering effect token.
        effect_in: Option<EffectTokenId>,
    },
    /// Allocation of semantic buffer.
    BufferAlloc {
        /// Buffer name.
        name: String,
        /// Allocation size in bytes.
        size_bytes: u64,
        /// Element data type.
        element_type: DataType,
    },
    /// Structured nested region yielding SSA results.
    Region(Box<StructuredRegion>),
    /// Declarative extension operation.
    Custom {
        /// Extension identity.
        extension_id: ExtensionIdentity,
        /// Input operands.
        operands: Vec<ValueId>,
    },
}

/// An immutable SSA operation within a basic block.
#[derive(Clone, PartialEq, Eq, Hash, Debug, Serialize, Deserialize)]
pub struct RegionOp {
    /// Operation identifier within the block.
    pub id: u32,
    /// Results produced by this operation.
    pub results: Vec<BlockArg>,
    /// Operation semantics.
    pub kind: RegionOpKind,
    /// Optional source location or diagnostic tag.
    pub location: Option<String>,
}

/// Required terminator ending every basic block.
#[derive(Clone, PartialEq, Eq, Hash, Debug, Serialize, Deserialize)]
pub enum Terminator {
    /// Explicit yield of result values from a region or block.
    Yield {
        /// Yielded SSA values.
        values: Vec<ValueId>,
    },
    /// Direct unconditional branch to a successor block with arguments.
    Branch {
        /// Target block ID.
        target: BlockId,
        /// Arguments passed to target block.
        args: Vec<ValueId>,
    },
    /// Conditional branch to true/false successor blocks with arguments.
    CondBranch {
        /// Condition boolean value.
        cond: ValueId,
        /// Target block if true.
        true_dest: BlockId,
        /// Arguments passed if true.
        true_args: Vec<ValueId>,
        /// Target block if false.
        false_dest: BlockId,
        /// Arguments passed if false.
        false_args: Vec<ValueId>,
    },
    /// Return from enclosing function.
    Return {
        /// Returned SSA values.
        values: Vec<ValueId>,
    },
    /// Abort execution with a trap code.
    Trap {
        /// Trap code.
        code: u32,
        /// Diagnostic message.
        message: String,
    },
    /// Unreachable marker for dead control flow paths.
    Unreachable,
}

/// A basic block consisting of block arguments, sequential ops, and a terminator.
#[derive(Clone, PartialEq, Eq, Hash, Debug, Serialize, Deserialize)]
pub struct Block {
    /// Block identifier.
    pub id: BlockId,
    /// Block arguments received upon entry.
    pub args: Vec<BlockArg>,
    /// Sequence of operations executed in this block.
    pub ops: Vec<RegionOp>,
    /// Required terminating transfer.
    pub terminator: Terminator,
}

/// First-class function in Region SSA.
#[derive(Clone, PartialEq, Eq, Debug, Serialize, Deserialize)]
pub struct RegionFunction {
    /// Function name.
    pub name: String,
    /// Generic shape parameters.
    pub shape_params: Vec<ShapeParam>,
    /// Generic type parameters.
    pub type_params: Vec<TypeParam>,
    /// Generic effect parameters.
    pub effect_params: Vec<EffectParam>,
    /// Function parameters as SSA block arguments.
    pub params: Vec<BlockArg>,
    /// Return types.
    pub return_types: Vec<DataType>,
    /// Entry basic block ID.
    pub entry_block: BlockId,
    /// All basic blocks in the function.
    pub blocks: Vec<Block>,
}

impl RegionFunction {
    /// Find a block by its ID.
    #[must_use]
    pub fn get_block(&self, id: BlockId) -> Option<&Block> {
        self.blocks.iter().find(|b| b.id == id)
    }

    /// Find a block mutably by its ID.
    pub fn get_block_mut(&mut self, id: BlockId) -> Option<&mut Block> {
        self.blocks.iter_mut().find(|b| b.id == id)
    }

    /// Lookup the DataType of any SSA ValueId defined within the function.
    #[must_use]
    pub fn value_type(&self, val: ValueId) -> Option<DataType> {
        for p in &self.params {
            if p.id == val {
                return Some(p.ty.clone());
            }
        }
        for b in &self.blocks {
            for arg in &b.args {
                if arg.id == val {
                    return Some(arg.ty.clone());
                }
            }
            for op in &b.ops {
                for res in &op.results {
                    if res.id == val {
                        return Some(res.ty.clone());
                    }
                }
                if let RegionOpKind::Region(r) = &op.kind {
                    if let Some(ty) = Self::value_type_in_region(r, val) {
                        return Some(ty);
                    }
                }
            }
        }
        None
    }

    fn value_type_in_region(r: &StructuredRegion, val: ValueId) -> Option<DataType> {
        for arg in &r.entry_args {
            if arg.id == val {
                return Some(arg.ty.clone());
            }
        }
        for b in &r.blocks {
            for arg in &b.args {
                if arg.id == val {
                    return Some(arg.ty.clone());
                }
            }
            for op in &b.ops {
                for res in &op.results {
                    if res.id == val {
                        return Some(res.ty.clone());
                    }
                }
                if let RegionOpKind::Region(inner) = &op.kind {
                    if let Some(ty) = Self::value_type_in_region(inner, val) {
                        return Some(ty);
                    }
                }
            }
        }
        None
    }
}

/// Global declaration within a module.
#[derive(Clone, PartialEq, Eq, Hash, Debug, Serialize, Deserialize)]
pub struct GlobalDecl {
    /// Global symbol name.
    pub name: String,
    /// Data type.
    pub ty: DataType,
    /// Size in bytes.
    pub size_bytes: u64,
    /// Buffer binding or memory space.
    pub binding: Option<u32>,
}

/// First-class module containing functions, globals, and declarative extension catalog.
#[derive(Clone, PartialEq, Eq, Debug, Serialize, Deserialize)]
pub struct RegionModule {
    /// Module name.
    pub name: String,
    /// Functions contained in this module.
    pub functions: Vec<RegionFunction>,
    /// Global memory declarations.
    pub globals: Vec<GlobalDecl>,
    /// Attached extension catalog bundle.
    pub catalog: Option<ExtensionCatalogBundle>,
}

impl RegionModule {
    /// Create a new empty Region SSA module.
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            functions: Vec::new(),
            globals: Vec::new(),
            catalog: None,
        }
    }

    /// Find a function by name.
    #[must_use]
    pub fn get_function(&self, name: &str) -> Option<&RegionFunction> {
        self.functions.iter().find(|f| f.name == name)
    }

    /// Add a function to the module.
    pub fn add_function(&mut self, function: RegionFunction) {
        self.functions.push(function);
    }
}
