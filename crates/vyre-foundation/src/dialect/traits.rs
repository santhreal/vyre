//! Core traits for declarative and versioned dialects.

use crate::dialect::descriptor::{DialectDescriptor, DialectOpDescriptor};
use crate::dialect::version::DialectVersionError;
use crate::ir::Expr;
use crate::validate::ValidationError;

/// Trait implemented by every declarative dialect.
pub trait Dialect: Send + Sync + 'static {
    /// Associated operation enumeration representing every variant of this dialect.
    type Op: DialectOp;

    /// Dialect descriptor describing metadata, schema version, and operations.
    fn descriptor() -> &'static DialectDescriptor;

    /// Current schema version of the dialect.
    #[inline]
    fn version() -> u32 {
        Self::descriptor().version
    }

    /// Stable dialect namespace identifier.
    #[inline]
    fn id() -> &'static str {
        Self::descriptor().id
    }

    /// Enumerate all operation descriptors in this dialect.
    #[inline]
    fn ops() -> &'static [DialectOpDescriptor] {
        Self::descriptor().operations
    }

    /// Validate version compatibility for this dialect.
    #[inline]
    fn validate_version(requested: u32) -> Result<(), DialectVersionError> {
        crate::dialect::version::validate_dialect_version(Self::descriptor(), requested)
    }

    /// Try matching an operation identifier to a dialect operation variant.
    fn match_op_id(op_id: &str) -> Option<Self::Op>;
}

/// Trait implemented by every dialect operation enum variant.
pub trait DialectOp: Copy + Eq + core::fmt::Debug + Send + Sync + 'static {
    /// Stable operation identifier string.
    fn op_id(self) -> &'static str;

    /// Short operation name.
    fn op_name(self) -> &'static str;

    /// Version where this operation was introduced.
    fn introduced_version(self) -> u32;

    /// Operation descriptor.
    fn descriptor(self) -> &'static DialectOpDescriptor;

    /// Whether this operation is a composition over existing IR (true) or intrinsic (false).
    fn is_composable(self) -> bool;
}

/// Trait implemented by visitors that traverse dialect operations.
pub trait DialectVisitor<D: Dialect> {
    /// Return type of the visitor.
    type Output;

    /// Visit a specific operation in the dialect.
    fn visit_op(&mut self, op: D::Op, args: &[Expr]) -> Self::Output;
}

/// Trait for pattern matching dialect operations inside IR expressions.
pub trait DialectMatcher<D: Dialect> {
    /// Match an `Expr::Call` against this dialect.
    fn match_call<'a>(expr: &'a Expr) -> Option<(D::Op, &'a [Expr])> {
        if let Expr::Call { op_id, args } = expr {
            D::match_op_id(op_id.as_str()).map(|matched_op| (matched_op, args.as_slice()))
        } else {
            None
        }
    }
}

/// Validation hook for dialect operations.
pub trait DialectValidator<D: Dialect> {
    /// Validate dialect call arguments and version.
    fn validate_dialect_call(
        op: D::Op,
        args: &[Expr],
        target_version: u32,
        errors: &mut Vec<ValidationError>,
    );
}

/// Endian-fixed serialization and deserialization for dialect operation tokens.
pub trait DialectCodec<D: Dialect> {
    /// Encode a dialect operation variant into a 32-bit discriminator token.
    fn encode_token(op: D::Op) -> u32;

    /// Decode a 32-bit discriminator token into a dialect operation variant.
    fn decode_token(token: u32) -> Option<D::Op>;
}
