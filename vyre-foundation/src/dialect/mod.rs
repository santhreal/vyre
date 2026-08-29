//! Declarative, versioned dialect definitions and generation framework.
//!
//! A dialect definition in vyre is a single declarative source of truth that generates:
//! 1. Typed operation enumerations and AST markers.
//! 2. Dialect descriptors and registration records.
//! 3. Exhaustive visitor traits and traversal dispatchers.
//! 4. Endian-fixed binary serialization and deserialization.
//! 5. Semantic validation hooks and version compatibility checks.
//! 6. Pattern matching and rewrite matchers.
//! 7. Actionable diagnostics with `Fix:` recommendations.
//! 8. Closure rosters and compile-time exhaustive variant enforcement.
//! 9. External schema, field, and resource ABI translation validation.

pub mod descriptor;
pub mod schema;
pub mod traits;
pub mod version;

pub use descriptor::{
    DialectDescriptor, DialectDescriptorRegistration, DialectOpDescriptor, DialectRegistry,
};
pub use schema::{
    validate_node_fields, validate_node_resources, validate_schema_identity, ExternalSchemaNode,
    FieldContract, FieldType, ResourceAbi, ResourceBinding, SchemaTranslationError,
};
pub use traits::{
    Dialect, DialectCodec, DialectMatcher, DialectOp, DialectValidator, DialectVisitor,
};
pub use version::{validate_dialect_version, validate_op_version, DialectVersionError};

#[cfg(test)]
mod tests;

/// Declarative macro to generate a complete versioned dialect module from a single definition.
#[macro_export]
macro_rules! define_dialect {
    (
        $(#[$dialect_meta:meta])*
        dialect: $dialect_id:literal,
        name: $dialect_name:ident,
        visitor: $visitor_trait:ident,
        version: $version:literal,
        min_supported_version: $min_supported_version:literal,
        tier: $tier:expr,
        category: $category:literal,
        summary: $summary:literal,

        operations: [
            $(
                {
                    $(#[$op_meta:meta])*
                    op: $op_variant:ident,
                    discriminant: $op_disc:literal,
                    name: $op_name_str:literal,
                    id: $op_id:literal,
                    version: $op_version:literal,
                    summary: $op_summary:literal,
                    signature: $op_sig:expr,
                    is_composable: $is_composable:literal
                    $(, build: $op_build:expr )?
                    $(, test_inputs: $op_inputs:expr )?
                    $(, expected_output: $op_expected:expr )?
                    $(, call_builder: $call_builder:ident )?
                    $(, fields: $op_fields:expr )?
                    $(, resource_abi: $op_resource_abi:expr )?
                    $(,)?
                }
            ),* $(,)?
        ]
    ) => {
        $(#[$dialect_meta])*
        pub mod $dialect_name {
            use super::*;

            /// Dialect identifier string.
            pub const DIALECT_ID: &str = $dialect_id;
            /// Current schema version.
            pub const SCHEMA_VERSION: u32 = $version;
            /// Minimum supported schema version.
            pub const MIN_SUPPORTED_VERSION: u32 = $min_supported_version;
            /// Dialect taxonomy category.
            pub const CATEGORY: &str = $category;
            /// Dialect summary description.
            pub const SUMMARY: &str = $summary;

            /// Enumeration of every operation declared in this dialect.
            #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
            #[repr(u32)]
            pub enum Op {
                $(
                    $(#[$op_meta])*
                    #[doc = $op_summary]
                    $op_variant = $op_disc,
                )*
            }

            impl Op {
                /// Return the stable operation identifier string.
                #[must_use]
                pub const fn op_id(self) -> &'static str {
                    match self {
                        $(
                            Self::$op_variant => $op_id,
                        )*
                    }
                }

                /// Return the short operation name.
                #[must_use]
                pub const fn op_name(self) -> &'static str {
                    match self {
                        $(
                            Self::$op_variant => $op_name_str,
                        )*
                    }
                }

                /// Return the version where this operation was introduced.
                #[must_use]
                pub const fn introduced_version(self) -> u32 {
                    match self {
                        $(
                            Self::$op_variant => $op_version,
                        )*
                    }
                }

                /// Return the numeric discriminant for wire serialization.
                #[must_use]
                pub const fn discriminant(self) -> u32 {
                    self as u32
                }

                /// Decode a numeric discriminant into an operation variant.
                #[must_use]
                pub const fn from_discriminant(tag: u32) -> Option<Self> {
                    match tag {
                        $(
                            $op_disc => Some(Self::$op_variant),
                        )*
                        _ => None,
                    }
                }

                /// Return whether this operation is composable (true) or intrinsic (false).
                #[must_use]
                pub const fn is_composable(self) -> bool {
                    match self {
                        $(
                            Self::$op_variant => $is_composable,
                        )*
                    }
                }

                /// Return the metadata descriptor for this operation.
                #[must_use]
                pub fn descriptor(self) -> &'static $crate::dialect::DialectOpDescriptor {
                    match self {
                        $(
                            Self::$op_variant => {
                                static DESC: $crate::dialect::DialectOpDescriptor =
                                    $crate::dialect::DialectOpDescriptor {
                                        id: $op_id,
                                        dialect: $dialect_id,
                                        name: $op_name_str,
                                        version: $op_version,
                                        signature: &$op_sig,
                                        is_composable: $is_composable,
                                        summary: $op_summary,
                                    };
                                &DESC
                            }
                        )*
                    }
                }

                /// Return the declared field contract for external schema translation.
                #[must_use]
                pub fn declared_fields(self) -> &'static [$crate::dialect::FieldContract] {
                    match self {
                        $(
                            Self::$op_variant => {
                                let fields: &'static [$crate::dialect::FieldContract] = &[];
                                $( let fields = $op_fields; )?
                                fields
                            }
                        )*
                    }
                }

                /// Return the declared resource ABI for external translation.
                #[must_use]
                pub fn resource_abi(self) -> &'static $crate::dialect::ResourceAbi {
                    match self {
                        $(
                            Self::$op_variant => {
                                static DEFAULT_ABI: $crate::dialect::ResourceAbi = $crate::dialect::ResourceAbi::EMPTY;
                                let abi = &DEFAULT_ABI;
                                $( let abi = &$op_resource_abi; )?
                                abi
                            }
                        )*
                    }
                }
            }

            impl $crate::dialect::DialectOp for Op {
                #[inline]
                fn op_id(self) -> &'static str {
                    self.op_id()
                }

                #[inline]
                fn op_name(self) -> &'static str {
                    self.op_name()
                }

                #[inline]
                fn introduced_version(self) -> u32 {
                    self.introduced_version()
                }

                #[inline]
                fn descriptor(self) -> &'static $crate::dialect::DialectOpDescriptor {
                    self.descriptor()
                }

                #[inline]
                fn is_composable(self) -> bool {
                    self.is_composable()
                }
            }

            /// Every operation identifier declared by this dialect in declaration order.
            pub static ALL_OP_IDS: &[&str] = &[
                $(
                    $op_id,
                )*
            ];

            /// Operation descriptors declared by this dialect.
            pub static OPERATIONS: &[$crate::dialect::DialectOpDescriptor] = &[
                $(
                    $crate::dialect::DialectOpDescriptor {
                        id: $op_id,
                        dialect: $dialect_id,
                        name: $op_name_str,
                        version: $op_version,
                        signature: &$op_sig,
                        is_composable: $is_composable,
                        summary: $op_summary,
                    },
                )*
            ];

            /// Static dialect descriptor record.
            pub static DESCRIPTOR: $crate::dialect::DialectDescriptor =
                $crate::dialect::DialectDescriptor {
                    id: $dialect_id,
                    name: stringify!($dialect_name),
                    version: $version,
                    min_supported_version: $min_supported_version,
                    tier: $tier,
                    category: $category,
                    operations: OPERATIONS,
                    summary: $summary,
                };

            /// Dialect marker type.
            #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
            pub struct DialectMarker;

            impl $crate::dialect::Dialect for DialectMarker {
                type Op = Op;

                #[inline]
                fn descriptor() -> &'static $crate::dialect::DialectDescriptor {
                    &DESCRIPTOR
                }

                #[inline]
                fn match_op_id(op_id: &str) -> Option<Self::Op> {
                    match_op_id(op_id)
                }
            }

            /// Look up an operation variant by its fully qualified operation identifier.
            #[must_use]
            pub fn match_op_id(op_id: &str) -> Option<Op> {
                match op_id {
                    $(
                        $op_id => Some(Op::$op_variant),
                    )*
                    _ => None,
                }
            }

            /// Pattern match an `Expr::Call` to extract a dialect operation variant and arguments.
            #[must_use]
            pub fn match_call<'a>(expr: &'a $crate::ir::Expr) -> Option<(Op, &'a [$crate::ir::Expr])> {
                if let $crate::ir::Expr::Call { op_id, args } = expr {
                    match_op_id(op_id.as_str()).map(|matched| (matched, args.as_slice()))
                } else {
                    None
                }
            }

            /// Visitor trait with an exhaustive method for every operation in this dialect.
            #[allow(non_snake_case)]
            pub trait $visitor_trait {
                /// Result type of the visitor.
                type Output;

                $(
                    /// Visit the corresponding operation.
                    fn $op_variant(&mut self, args: &[$crate::ir::Expr]) -> Self::Output;
                )*
            }

            /// Dispatch a visitor over an operation variant and argument list.
            pub fn dispatch_visitor<V: $visitor_trait>(
                op: Op,
                args: &[$crate::ir::Expr],
                visitor: &mut V,
            ) -> V::Output {
                match op {
                    $(
                        Op::$op_variant => visitor.$op_variant(args),
                    )*
                }
            }

            /// Validate an operation invocation against dialect schema and version.
            pub fn validate_call(
                op: Op,
                args: &[$crate::ir::Expr],
                target_version: u32,
                errors: &mut Vec<$crate::validate::ValidationError>,
            ) {
                let desc = op.descriptor();
                if let Err(err) = $crate::dialect::version::validate_op_version(desc, target_version) {
                    errors.push($crate::validate::err(
                        "V023",
                        $crate::validate::ValidationPhase::Expression,
                        $crate::validate::ValidationLocation::Program,
                        err.to_string(),
                        "Fix: update the dialect reference version or raise target schema version."
                    ));
                    return;
                }
                let expected = desc.signature.inputs.len();
                if args.len() != expected {
                    errors.push($crate::validate::err(
                        "V020",
                        $crate::validate::ValidationPhase::Expression,
                        $crate::validate::ValidationLocation::Program,
                        format!("call `{}` has {} arguments but signature expects {expected}", desc.id, args.len()),
                        format!("Fix: pass exactly {expected} arguments in the order declared by the op signature")
                    ));
                }
            }

            /// Validate an external schema node mapping into this dialect.
            ///
            /// # Errors
            ///
            /// Returns [`$crate::dialect::SchemaTranslationError`] if the node is unmapped,
            /// contains unknown/duplicate fields, is missing required fields, or has an incomplete resource roster.
            pub fn validate_external_node(
                node: &$crate::dialect::ExternalSchemaNode,
            ) -> Result<Op, $crate::dialect::SchemaTranslationError> {
                let Some(op) = match_op_id(&node.op_name) else {
                    return Err($crate::dialect::SchemaTranslationError::UnmappedNode {
                        dialect: $dialect_id,
                        node_op: node.op_name.clone(),
                    });
                };
                $crate::dialect::validate_node_fields(
                    $dialect_id,
                    &node.op_name,
                    &node.raw_fields,
                    op.declared_fields(),
                )?;
                $crate::dialect::validate_node_resources(
                    $dialect_id,
                    &node.op_name,
                    &node.bound_resources,
                    op.resource_abi(),
                )?;
                Ok(op)
            }

            /// Encode an operation variant into binary format.
            pub fn encode_op<W: std::io::Write>(op: Op, writer: &mut W) -> std::io::Result<()> {
                let tag = op.discriminant();
                writer.write_all(&tag.to_le_bytes())
            }

            /// Decode an operation variant from binary format.
            pub fn decode_op<R: std::io::Read>(reader: &mut R) -> std::io::Result<Option<Op>> {
                let mut bytes = [0u8; 4];
                reader.read_exact(&mut bytes)?;
                let tag = u32::from_le_bytes(bytes);
                Ok(Op::from_discriminant(tag))
            }

            $(
                $(
                    /// Construct a call expression for this operation.
                    #[must_use]
                    pub fn $call_builder(args: Vec<$crate::ir::Expr>) -> $crate::ir::Expr {
                        $crate::ir::Expr::call($op_id, args)
                    }
                )?
            )*

            // Inventory registrations
            inventory::submit! {
                $crate::dialect::DialectDescriptorRegistration {
                    descriptor: &DESCRIPTOR,
                }
            }

            $(
                inventory::submit! {
                    $crate::operation::OperationRegistration {
                        id: $op_id,
                        semantic_version: $op_version,
                        signature: Some($op_sig),
                        tier: $tier,
                        category: Some($category),
                        build: {
                            #[allow(unused_mut)]
                            let mut selected: Option<fn() -> $crate::ir::Program> = None;
                            $( selected = Some($op_build); )?
                            selected
                        },
                        test_inputs: {
                            #[allow(unused_mut)]
                            let mut selected: Option<$crate::operation::OperationFixtures> = None;
                            $( selected = Some($op_inputs); )?
                            selected
                        },
                        expected_output: {
                            #[allow(unused_mut)]
                            let mut selected: Option<$crate::operation::OperationFixtures> = None;
                            $( selected = Some($op_expected); )?
                            selected
                        },
                        laws: &[],
                        numeric: $crate::numeric::NumericContract::EXACT,
                        geometry_requirements: $crate::GeometryRequirements::agnostic(),
                        source_file: file!(),
                        explicit_effects: None,
                        explicit_capabilities: None,
                    }
                }
            )*
        }
    };
}
