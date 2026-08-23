// Expression nodes  -  produce values.
//
// Every expression evaluates to a typed value. Expressions are pure:
// they read state but do not modify it.

use crate::ir_inner::model::op_signature::{DataType, SubgroupReduceOp};
use std::fmt;
use std::sync::Arc;

mod ident;

pub use ident::Ident;

/// An expression that produces a value.
///
/// # Examples
///
/// ```
/// use vyre::ir::Expr;
///
/// let lit = Expr::u32(42);
/// let var = Expr::var("x");
/// let add = Expr::add(lit, var);
/// ```
pub use crate::ir_inner::model::generated::Expr;

/// Public contract for downstream expression extension nodes.
///
/// Extension nodes are intentionally opaque to core. A downstream crate owns
/// the semantic payload and provides the stable metadata core needs for
/// validation, debug output, equality, and CSE identity. Backends that
/// understand the extension can downcast through their own wrapper type before
/// constructing target code; backends that do not understand it must reject it
/// with an actionable error.
pub trait ExprNode: fmt::Debug + Send + Sync + 'static {
    /// Stable extension namespace, for example `my_backend.tensor.shuffle`.
    fn extension_kind(&self) -> &'static str;

    /// Human-readable identity used in diagnostics and debug logs.
    fn debug_identity(&self) -> &str;

    /// Static result type produced by this expression.
    fn result_type(&self) -> Option<DataType>;

    /// Whether CSE may treat this extension as a pure, repeatable expression.
    fn cse_safe(&self) -> bool;

    /// Stable, content-addressed identity for equality and optimizer keys.
    fn stable_fingerprint(&self) -> [u8; 32];

    /// Validate extension-local invariants.
    ///
    /// # Errors
    ///
    /// The returned error must explain the bad invariant and include `Fix:`.
    fn validate_extension(&self) -> Result<(), String>;

    /// Downcast to Any to allow backend-specific dispatch from opaque payloads.
    fn as_any(&self) -> &dyn std::any::Any;

    /// Serialize the extension payload into stable bytes used by the wire
    /// encoder's `Expr::Opaque` path (tag `0x80`). Default: empty payload  -
    /// suitable for extensions that carry no state beyond their type
    /// identity. Extensions with state must override this to emit the exact
    /// bytes `wire_payload`'s matching `OpaqueExprResolver` will consume.
    ///
    /// The payload contract is endian-fixed: any numeric field wider than
    /// one byte MUST be written with `to_le_bytes`, and the matching decoder
    /// MUST reconstruct it with `from_le_bytes`. Host-endian encodings such as
    /// `to_ne_bytes` are forbidden because the wire format must stay
    /// byte-identical across architectures.
    ///
    /// Extension authors should use [`crate::opaque_payload::endian::LeBytesWriter`] when
    /// building payloads because it makes the required endianness explicit in the type.
    ///
    /// Literal extensions that encode regex payloads must also canonicalize
    /// inline flag prefixes before emitting bytes. For example, `(?mi)` and
    /// `(?im)` are the same semantic payload and MUST serialize to the same
    /// flag ordering.
    fn wire_payload(&self) -> Vec<u8> {
        Vec::new()
    }
}

impl Expr {
    /// Load from buffer at index.
    ///
    /// # Examples
    ///
    /// ```
    /// use vyre::ir::Expr;
    /// let _ = Expr::load("a", Expr::u32(0));
    /// ```
    #[must_use]
    #[inline]
    pub fn load(buffer: impl Into<Ident>, index: Self) -> Self {
        Self::Load {
            buffer: buffer.into(),
            index: Box::new(index),
        }
    }

    /// Buffer element count.
    ///
    /// # Examples
    ///
    /// ```
    /// use vyre::ir::Expr;
    /// let _ = Expr::buf_len("a");
    /// ```
    #[must_use]
    #[inline]
    pub fn buf_len(buffer: impl Into<Ident>) -> Self {
        Self::BufLen {
            buffer: buffer.into(),
        }
    }

    /// Name a whole buffer as an argument to a composite op.
    ///
    /// This is not a value: it has no type and only call-argument position
    /// accepts it. Inlining rebinds the callee's matching parameter onto
    /// this buffer, so a callee that reads `table[i]` ends up reading the
    /// caller's buffer at the same index. To read one element instead, use
    /// [`Expr::load`].
    ///
    /// ```
    /// use vyre::ir::Expr;
    /// let _ = Expr::call("dialect::lookup", vec![Expr::buffer_ref("table"), Expr::u32(3)]);
    /// ```
    #[must_use]
    #[inline]
    pub fn buffer_ref(buffer: impl Into<Ident>) -> Self {
        Self::BufferRef {
            buffer: buffer.into(),
        }
    }

    /// Schedule-free logical domain index for one algorithm axis.
    ///
    /// This expression is valid in semantic library programs. Selected-schedule
    /// lowering maps it to a physical invocation identity before emission.
    #[must_use]
    #[inline]
    pub const fn logical_index(axis: u8) -> Self {
        Self::LogicalIndex { axis }
    }

    /// Whether the current logical point is the first point on axis zero.
    #[must_use]
    #[inline]
    pub fn is_first_logical_point() -> Self {
        Self::eq(Self::LogicalIndex { axis: 0 }, Self::u32(0))
    }

    /// Schedule-free logical tile coordinate for one algorithm axis.
    #[must_use]
    #[inline]
    pub const fn logical_tile_index(axis: u8) -> Self {
        Self::LogicalTileId { axis }
    }

    /// Schedule-free coordinate within one logical cooperative tile.
    #[must_use]
    #[inline]
    pub const fn logical_within_tile_index(axis: u8) -> Self {
        Self::LogicalWithinTileId { axis }
    }

    /// Whether the current logical tile is the first tile on axis zero.
    #[must_use]
    #[inline]
    pub fn is_first_logical_tile() -> Self {
        Self::eq(Self::LogicalTileId { axis: 0 }, Self::u32(0))
    }

    /// `global_invocation_id.x`
    #[must_use]
    #[inline]
    pub fn gid_x() -> Self {
        Self::InvocationId { axis: 0 }
    }

    /// `global_invocation_id.y`
    #[must_use]
    #[inline]
    pub fn gid_y() -> Self {
        Self::InvocationId { axis: 1 }
    }

    /// `global_invocation_id.z`
    #[must_use]
    #[inline]
    pub fn gid_z() -> Self {
        Self::InvocationId { axis: 2 }
    }

    /// `workgroup_id.x`
    #[must_use]
    #[inline]
    pub fn workgroup_x() -> Self {
        Self::WorkgroupId { axis: 0 }
    }

    /// `workgroup_id.y`
    #[must_use]
    #[inline]
    pub fn workgroup_y() -> Self {
        Self::WorkgroupId { axis: 1 }
    }

    /// `workgroup_id.z`
    #[must_use]
    #[inline]
    pub fn workgroup_z() -> Self {
        Self::WorkgroupId { axis: 2 }
    }

    /// Predicate `workgroup_id.x == 0`: the canonical "first parallel region only" guard.
    ///
    /// Single-workgroup kernels (scalar reductions, workgroup-local tree reductions, and any
    /// kernel meant to run in exactly one parallel region) gate their body on this so a dispatch
    /// of more than one workgroup leaves the extra regions as no-ops instead of double-counting or
    /// racing on the shared output. Prefer this over re-spelling `eq(WorkgroupId{axis:0}, 0)`
    /// inline so the "first workgroup" contract has one owner.
    #[must_use]
    #[inline]
    pub fn is_first_workgroup() -> Self {
        Self::eq(Self::WorkgroupId { axis: 0 }, Self::u32(0))
    }

    /// Predicate `workgroup_id.x == 0 && local_id.x == 0`: the one invocation
    /// that owns a serial body.
    ///
    /// A serial scan that keeps its running result in read-write storage must
    /// run exactly once. [`is_first_workgroup`](Self::is_first_workgroup)
    /// expresses that only while the workgroup holds one invocation, and a
    /// fusion widens an arm to the fused workgroup, where the same body runs
    /// once per added invocation over the same slots. This predicate is the
    /// same lane in a one-wide geometry and the only lane in any other, so a
    /// program that means "one invocation" says so and stays fusable.
    #[must_use]
    #[inline]
    pub fn is_first_invocation() -> Self {
        Self::and(
            Self::is_first_workgroup(),
            Self::eq(Self::LocalId { axis: 0 }, Self::u32(0)),
        )
    }

    /// `local_invocation_id.x`
    #[must_use]
    #[inline]
    pub fn local_x() -> Self {
        Self::LocalId { axis: 0 }
    }

    /// `subgroup_invocation_id` (lane index within subgroup).
    #[must_use]
    #[inline]
    pub fn subgroup_local_id() -> Self {
        Self::SubgroupLocalId
    }

    /// `subgroup_size` (number of lanes per subgroup).
    #[must_use]
    #[inline]
    pub fn subgroup_size() -> Self {
        Self::SubgroupSize
    }

    /// `local_invocation_id.y`
    #[must_use]
    #[inline]
    pub fn local_y() -> Self {
        Self::LocalId { axis: 1 }
    }

    /// `local_invocation_id.z`
    #[must_use]
    #[inline]
    pub fn local_z() -> Self {
        Self::LocalId { axis: 2 }
    }

    /// Substrate-neutral alias for [`workgroup_x`](Self::workgroup_x).
    ///
    /// "Parallel region" is the vocabulary used in the public `vyre` facade.
    /// Concrete drivers translate this concept into target vocabulary at the
    /// lowering boundary.
    #[must_use]
    #[inline]
    pub fn parallel_region_x() -> Self {
        Self::WorkgroupId { axis: 0 }
    }

    /// Substrate-neutral alias for [`workgroup_y`](Self::workgroup_y).
    #[must_use]
    #[inline]
    pub fn parallel_region_y() -> Self {
        Self::WorkgroupId { axis: 1 }
    }

    /// Substrate-neutral alias for [`workgroup_z`](Self::workgroup_z).
    #[must_use]
    #[inline]
    pub fn parallel_region_z() -> Self {
        Self::WorkgroupId { axis: 2 }
    }

    /// Substrate-neutral alias for [`local_x`](Self::local_x).
    #[must_use]
    #[inline]
    pub fn invocation_local_x() -> Self {
        Self::LocalId { axis: 0 }
    }

    /// Substrate-neutral alias for [`local_y`](Self::local_y).
    #[must_use]
    #[inline]
    pub fn invocation_local_y() -> Self {
        Self::LocalId { axis: 1 }
    }

    /// Substrate-neutral alias for [`local_z`](Self::local_z).
    #[must_use]
    #[inline]
    pub fn invocation_local_z() -> Self {
        Self::LocalId { axis: 2 }
    }

    /// Conditional select.
    #[must_use]
    #[inline]
    pub fn select(cond: Self, true_val: Self, false_val: Self) -> Self {
        Self::Select {
            cond: Box::new(cond),
            true_val: Box::new(true_val),
            false_val: Box::new(false_val),
        }
    }

    /// Subgroup reduction across the active subgroup with the given operator.
    #[must_use]
    #[inline]
    pub fn subgroup_reduce(op: SubgroupReduceOp, value: Self) -> Self {
        Self::SubgroupReduce {
            op,
            value: Box::new(value),
        }
    }

    /// Subgroup sum reduction across the active subgroup.
    #[must_use]
    #[inline]
    pub fn subgroup_add(value: Self) -> Self {
        Self::subgroup_reduce(SubgroupReduceOp::Add, value)
    }

    /// Subgroup product reduction across the active subgroup.
    #[must_use]
    #[inline]
    pub fn subgroup_mul(value: Self) -> Self {
        Self::subgroup_reduce(SubgroupReduceOp::Mul, value)
    }

    /// Subgroup minimum reduction across the active subgroup.
    #[must_use]
    #[inline]
    pub fn subgroup_min(value: Self) -> Self {
        Self::subgroup_reduce(SubgroupReduceOp::Min, value)
    }

    /// Subgroup maximum reduction across the active subgroup.
    #[must_use]
    #[inline]
    pub fn subgroup_max(value: Self) -> Self {
        Self::subgroup_reduce(SubgroupReduceOp::Max, value)
    }

    /// Subgroup bitwise-AND reduction across the active subgroup.
    #[must_use]
    #[inline]
    pub fn subgroup_and(value: Self) -> Self {
        Self::subgroup_reduce(SubgroupReduceOp::And, value)
    }

    /// Subgroup bitwise-OR reduction across the active subgroup.
    #[must_use]
    #[inline]
    pub fn subgroup_or(value: Self) -> Self {
        Self::subgroup_reduce(SubgroupReduceOp::Or, value)
    }

    /// Subgroup bitwise-XOR reduction across the active subgroup.
    #[must_use]
    #[inline]
    pub fn subgroup_xor(value: Self) -> Self {
        Self::subgroup_reduce(SubgroupReduceOp::Xor, value)
    }

    /// Subgroup shuffle: broadcast `value` from the given lane id to
    /// every active lane in the subgroup.
    #[must_use]
    #[inline]
    pub fn subgroup_shuffle(value: Self, lane: Self) -> Self {
        Self::SubgroupShuffle {
            value: Box::new(value),
            lane: Box::new(lane),
        }
    }

    /// Subgroup ballot: gather the boolean predicate `cond` across
    /// the active subgroup into a single bitmask.
    #[must_use]
    #[inline]
    pub fn subgroup_ballot(cond: Self) -> Self {
        Self::SubgroupBallot {
            cond: Box::new(cond),
        }
    }

    /// Named variable reference.
    #[must_use]
    #[inline]
    pub fn var(name: impl Into<Ident>) -> Self {
        Self::Var(name.into())
    }

    /// Unsigned 32-bit literal.
    #[must_use]
    #[inline]
    pub fn u32(value: u32) -> Self {
        Self::LitU32(value)
    }

    /// Signed 32-bit literal.
    #[must_use]
    #[inline]
    pub fn i32(value: i32) -> Self {
        Self::LitI32(value)
    }

    /// 32-bit floating-point literal.
    #[must_use]
    #[inline]
    pub fn f32(value: f32) -> Self {
        Self::LitF32(value)
    }

    /// Boolean literal.
    #[must_use]
    #[inline]
    pub fn bool(value: bool) -> Self {
        Self::LitBool(value)
    }

    /// Operation call by stable operation ID.
    #[must_use]
    #[inline]
    pub fn call(op_id: impl Into<Ident>, args: Vec<Self>) -> Self {
        Self::Call {
            op_id: op_id.into(),
            args,
        }
    }

    /// Fused multiply-add `a * b + c` (f32).
    #[must_use]
    #[inline]
    pub fn fma(a: Self, b: Self, c: Self) -> Self {
        Self::Fma {
            a: Box::new(a),
            b: Box::new(b),
            c: Box::new(c),
        }
    }

    /// Cast a value to `target`.
    #[must_use]
    #[inline]
    pub fn cast(target: DataType, value: Self) -> Self {
        Self::Cast {
            target,
            value: Box::new(value),
        }
    }

    /// Wrap a downstream extension expression node.
    #[must_use]
    #[inline]
    pub fn opaque(node: impl ExprNode) -> Self {
        Self::Opaque(Arc::new(node))
    }

    /// Wrap a shared downstream extension expression node.
    #[must_use]
    #[inline]
    pub fn opaque_arc(node: Arc<dyn ExprNode>) -> Self {
        Self::Opaque(node)
    }
}

mod atomics;
mod builders;

#[cfg(test)]
mod tests {
    use super::Expr;

    #[test]
    fn expr_size_is_bounded() {
        let size = std::mem::size_of::<Expr>();
        assert!(
            size <= 128,
            "Expr grew to {size} bytes. Fix: box the largest variant before adding more fields."
        );
    }
}
