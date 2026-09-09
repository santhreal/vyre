//! Extension contracts for open IR.
//!
//! Downstream crates ship new `Expr`, `Node`, `DataType`, `BinOp`, `UnOp`,
//! `AtomicOp`, `TernaryOp`, and `RuleCondition` variants by implementing the
//! traits in this module and registering an id with the foundation extension
//! registry.
//!
//! `vyre-spec` is intentionally data-only and carries no dependency on
//! `inventory`. The trait signatures below describe the stable contract;
//! actual registration and resolution lives in `vyre_foundation::extension`.
//!
use crate::data_type::DataType;
use crate::op_contract::SideEffectClass;
use crate::region_law::RegionLawFamily;
/// Every extension id occupies the range `0x8000_0000..=0xFFFF_FFFF`  -  the
/// high bit of the wire tag distinguishes extension ids from the frozen
/// core tag space `0x00..=0x7F`. The `ExtensionDataTypeId::from_name`
/// constructor computes a collision-resistant 256-bit cryptographic digest
/// and folds it into the reserved range with explicit collision resolution.
use core::fmt::Debug;
macro_rules! impl_extension_id {
    ($id:ident) => {
        impl $id {
            /// Reserved range: every extension id has its high bit set.
            ///
            /// Core IR discriminants occupy `0x00..=0x7F`; extensions occupy
            /// `0x80..=0xFFFF_FFFF`. Wire decoders test the high byte to route
            /// decoding between the two.
            pub const EXTENSION_RANGE_MASK: u32 = 0x8000_0000;

            /// Construct an id from a stable extension name using a collision-resistant full digest.
            ///
            /// The id is derived deterministically from the 256-bit cryptographic digest
            /// and folded into the extension range by setting the high bit.
            #[must_use]
            pub const fn from_name(name: &str) -> Self {
                Self(digest_with_high_bit(name))
            }
            /// Return the raw id.
            #[must_use]
            pub const fn as_u32(self) -> u32 {
                self.0
            }

            /// Is this a reserved extension id (high bit set)?
            #[must_use]
            pub const fn is_extension(self) -> bool {
                (self.0 & Self::EXTENSION_RANGE_MASK) != 0
            }
        }
    };
}

/// Stable u32 id for an extension variant.
///
/// Extension ids are generated deterministically from a stable name via
/// [`ExtensionDataTypeId::from_name`]. A crate that never changes its
/// extension name keeps the same id across versions, which is the
/// wire-format contract: a `Program` encoded by v1.0 of an extension
/// decodes identically in v1.1 so long as the name is stable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct ExtensionDataTypeId(pub u32);

impl_extension_id!(ExtensionDataTypeId);

/// The contract for an extension-declared `DataType`.
///
/// An implementer describes the runtime shape of a non-core data type:
/// how many bytes it occupies, whether it participates in the float
/// conformance family, and how it should be displayed.
///
/// The foundation extension registry walks a link-time inventory of
/// `ExtensionDataTypeRegistration` entries. The resolver caches
/// `&'static dyn ExtensionDataType` so downstream
/// consumers never re-consult the registry on the hot path.
pub trait ExtensionDataType: Send + Sync + Debug + 'static {
    /// Stable id for this data type.
    fn id(&self) -> ExtensionDataTypeId;
    /// Human-readable name for display / debug.
    fn display_name(&self) -> &'static str;
    /// Minimum byte count to represent one value of this type.
    fn min_bytes(&self) -> usize;
    /// Maximum byte count for one value of this type; `None` when unbounded.
    fn max_bytes(&self) -> Option<usize>;
    /// Fixed element size in bytes, or `None` for variable-size types.
    fn size_bytes(&self) -> Option<usize>;
    /// Whether this type belongs to the IEEE-754 float conformance family.
    fn is_float_family(&self) -> bool;
    /// Whether values can be safely memcpy'd between host and device.
    fn is_host_shareable(&self) -> bool;
}

/// Runtime contract for an extension-declared binary operator.
///
/// The foundation extension registry caches `&'static dyn ExtensionBinOp`
/// pointers keyed by [`ExtensionBinOpId`]; downstream evaluators and lowerings
/// call through this trait without re-consulting the registry on the hot path.
pub trait ExtensionBinOp: Send + Sync + Debug + 'static {
    /// Stable id of this binary operator.
    fn id(&self) -> ExtensionBinOpId;
    /// Human-readable name for display / debug.
    fn display_name(&self) -> &'static str;
}
/// Runtime contract for an extension-declared unary operator.
pub trait ExtensionUnOp: Send + Sync + Debug + 'static {
    /// Stable id of this unary operator.
    fn id(&self) -> ExtensionUnOpId;
    /// Human-readable name for display / debug.
    fn display_name(&self) -> &'static str;
}

/// Runtime contract for an extension-declared atomic operator.
pub trait ExtensionAtomicOp: Send + Sync + Debug + 'static {
    /// Stable id of this atomic operator.
    fn id(&self) -> ExtensionAtomicOpId;
    /// Human-readable name for display / debug.
    fn display_name(&self) -> &'static str;
}

/// Runtime contract for an extension-declared ternary operator.
pub trait ExtensionTernaryOp: Send + Sync + Debug + 'static {
    /// Stable id of this ternary operator.
    fn id(&self) -> ExtensionTernaryOpId;
    /// Human-readable name for display / debug.
    fn display_name(&self) -> &'static str;
}

/// Stable u32 id for an extension binary operator.
///
/// Identical discipline to [`ExtensionDataTypeId`]: stable across process
/// runs, high bit set, generated by FNV-1a of the extension name.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct ExtensionBinOpId(pub u32);

impl_extension_id!(ExtensionBinOpId);

/// Stable u32 id for an extension unary operator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct ExtensionUnOpId(pub u32);

impl_extension_id!(ExtensionUnOpId);

/// Stable u32 id for an extension atomic operator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct ExtensionAtomicOpId(pub u32);

impl_extension_id!(ExtensionAtomicOpId);

/// Stable u32 id for an extension ternary operator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct ExtensionTernaryOpId(pub u32);

impl_extension_id!(ExtensionTernaryOpId);

/// Stable u32 id for an extension rule condition.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct ExtensionRuleConditionId(pub u32);

impl_extension_id!(ExtensionRuleConditionId);

/// Globally unique namespace identifying an extension domain or package.
#[derive(
    Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize, serde::Deserialize,
)]
pub struct ExtensionNamespace(pub String);

impl ExtensionNamespace {
    /// Construct a validated extension namespace.
    ///
    /// The namespace must not be empty and must only contain alphanumeric ASCII
    /// characters, dots, hyphens, and underscores.
    pub fn new(namespace: impl Into<String>) -> Result<Self, &'static str> {
        let ns = namespace.into();
        if ns.is_empty() {
            return Err("Extension namespace cannot be empty");
        }
        if !ns
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '.' || c == '-' || c == '_')
        {
            return Err("Extension namespace contains invalid characters; expected [a-zA-Z0-9._-]");
        }
        Ok(Self(ns))
    }

    /// Access the namespace as a string slice.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl core::fmt::Display for ExtensionNamespace {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Semantic version for an extension schema.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize, serde::Deserialize,
)]
pub struct ExtensionSemVer {
    /// Major version.
    pub major: u32,
    /// Minor version.
    pub minor: u32,
    /// Patch version.
    pub patch: u32,
}

impl ExtensionSemVer {
    /// Construct a new semantic version.
    #[must_use]
    pub const fn new(major: u32, minor: u32, patch: u32) -> Self {
        Self {
            major,
            minor,
            patch,
        }
    }
}

impl core::fmt::Display for ExtensionSemVer {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)
    }
}

/// 256-bit cryptographic digest over a canonical extension schema definition.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize, serde::Deserialize,
)]
pub struct ExtensionSchemaDigest(pub [u8; 32]);

impl ExtensionSchemaDigest {
    /// Create a schema digest from raw 32 bytes.
    #[must_use]
    pub const fn new(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    /// Compute schema digest from arbitrary byte stream using deterministic 256-bit hash.
    #[must_use]
    pub const fn from_bytes(bytes: &[u8]) -> Self {
        Self(compute_digest_bytes(bytes))
    }

    /// Format the digest as lowercase hex string.
    #[must_use]
    pub fn to_hex(self) -> String {
        let mut s = String::with_capacity(64);
        for b in self.0 {
            use core::fmt::Write as _;
            let _ = write!(s, "{:02x}", b);
        }
        s
    }
}

/// Full collision-resistant identity for an extension schema.
///
/// Combines a globally unique namespace, semantic version, and 256-bit schema digest.
#[derive(
    Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize, serde::Deserialize,
)]
pub struct ExtensionIdentity {
    /// Globally unique namespace.
    pub namespace: ExtensionNamespace,
    /// Semantic version.
    pub version: ExtensionSemVer,
    /// Canonical schema digest.
    pub schema_digest: ExtensionSchemaDigest,
}

impl ExtensionIdentity {
    /// Create a new extension identity.
    #[must_use]
    pub fn new(
        namespace: ExtensionNamespace,
        version: ExtensionSemVer,
        schema_digest: ExtensionSchemaDigest,
    ) -> Self {
        Self {
            namespace,
            version,
            schema_digest,
        }
    }

    /// Canonical display string `namespace@major.minor.patch#hex_digest`.
    #[must_use]
    pub fn to_canonical_string(&self) -> String {
        format!(
            "{}@{}#{}",
            self.namespace.as_str(),
            self.version,
            self.schema_digest.to_hex()
        )
    }
}

impl core::fmt::Display for ExtensionIdentity {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}", self.to_canonical_string())
    }
}

/// Canonical primitive field type supported in extension attribute declarations.
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum ExtensionFieldType {
    /// Boolean flag.
    Bool,
    /// 64-bit signed integer.
    I64,
    /// 64-bit unsigned integer.
    U64,
    /// 64-bit floating point literal.
    F64,
    /// String literal.
    String,
    /// Primitive or structured DataType.
    DataType,
    /// Shape dimension list.
    ShapeDims,
}

/// Declarative schema attribute field.
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct ExtensionField {
    /// Attribute name.
    pub name: String,
    /// Data type of the field.
    pub field_type: ExtensionFieldType,
    /// Whether this attribute is mandatory.
    pub required: bool,
    /// Optional default value.
    pub default_value: Option<String>,
}

/// Role or kind of an extension operand.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum ExtensionOperandKind {
    /// SSA value operand.
    Value,
    /// Structured region operand.
    Region,
    /// Buffer / storage operand.
    Buffer,
}

/// Declarative operand descriptor.
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct ExtensionOperand {
    /// Operand name.
    pub name: String,
    /// Kind of operand.
    pub kind: ExtensionOperandKind,
    /// Expected data type, if constrained.
    pub data_type: Option<DataType>,
    /// Whether this operand is optional.
    pub optional: bool,
}

/// Declarative shape inference and transformation rule for an extension operation.
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum ExtensionShapeRule {
    /// Output shape matches primary input operand.
    PreservesShape,
    /// Broadcasts leading dimensions across operands.
    BroadcastLeading,
    /// Matrix multiplication contracting (M, K) x (K, N) -> (M, N).
    MatrixProduct {
        /// M dimension.
        m: usize,
        /// K reduction dimension.
        k: usize,
        /// N dimension.
        n: usize,
    },
    /// Reduces the specified axis.
    Reduction {
        /// Reduction axis index.
        axis: usize,
    },
    /// General affine shape transformation.
    AffineMap {
        /// Affine expression string.
        expression: String,
    },
    /// Custom declarative rule string.
    Custom {
        /// Rule description.
        rule: String,
    },
}

/// Declarative numerical accuracy and floating point behavior contract.
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct ExtensionNumericalContract {
    /// Whether this operation participates in floating point families.
    pub float_family: bool,
    /// Maximum allowed ULP distance from ideal real semantics.
    pub max_ulp_error: Option<u32>,
    /// Whether this operation preserves mathematical monotonicity.
    pub preserves_monotonicity: bool,
    /// Whether strict IEEE-754 semantics are mandatory.
    pub requires_strict_fp: bool,
}

impl Default for ExtensionNumericalContract {
    fn default() -> Self {
        Self {
            float_family: false,
            max_ulp_error: None,
            preserves_monotonicity: false,
            requires_strict_fp: false,
        }
    }
}

/// Hardware resource bounds and limits declared by an extension.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize, Default,
)]
pub struct ExtensionResourceBounds {
    /// Maximum workgroup-shared memory in bytes.
    pub max_shared_bytes: u64,
    /// Maximum invocation-private scratch in bytes.
    pub max_private_bytes: u64,
    /// Estimated register slots per invocation.
    pub registers_per_invocation: u32,
}

/// Closed declarative extension schema.
///
/// Replaces opaque callbacks with a self-contained, serializable,
/// verifiable declaration of an extension's typed interface, semantics,
/// laws, shape rules, and resource bounds.
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct ExtensionSchema {
    /// Unique identity.
    pub identity: ExtensionIdentity,
    /// Human-readable display name.
    pub display_name: String,
    /// Documentation description.
    pub description: String,
    /// Canonical typed attribute fields.
    pub fields: Vec<ExtensionField>,
    /// Child and value operands.
    pub operands: Vec<ExtensionOperand>,
    /// Result types produced by this operation.
    pub result_types: Vec<DataType>,
    /// Side-effect classification.
    pub side_effects: SideEffectClass,
    /// Shape inference rules.
    pub shape_rules: Vec<ExtensionShapeRule>,
    /// Numerical contract.
    pub numerical_contract: ExtensionNumericalContract,
    /// Algebraic and structural law families this extension satisfies.
    pub laws: Vec<RegionLawFamily>,
    /// Hardware resource bounds.
    pub resource_bounds: ExtensionResourceBounds,
    /// Semantic and verification proof fields.
    pub proof_fields: ExtensionProofFields,
}

impl ExtensionSchema {
    /// Compute the deterministic schema digest from its canonical fields.
    #[must_use]
    pub fn compute_digest(
        namespace: &str,
        version: &ExtensionSemVer,
        fields: &[ExtensionField],
        operands: &[ExtensionOperand],
        result_types: &[DataType],
        proof_fields: &ExtensionProofFields,
    ) -> ExtensionSchemaDigest {
        let mut canonical_bytes = Vec::new();
        canonical_bytes.extend_from_slice(namespace.as_bytes());
        canonical_bytes.push(0xFF);
        canonical_bytes.extend_from_slice(&version.major.to_le_bytes());
        canonical_bytes.extend_from_slice(&version.minor.to_le_bytes());
        canonical_bytes.extend_from_slice(&version.patch.to_le_bytes());
        canonical_bytes.push(0xFE);
        for f in fields {
            canonical_bytes.extend_from_slice(f.name.as_bytes());
            canonical_bytes.push(if f.required { 1 } else { 0 });
        }
        canonical_bytes.push(0xFD);
        for op in operands {
            canonical_bytes.extend_from_slice(op.name.as_bytes());
            canonical_bytes.push(op.kind as u8);
        }
        canonical_bytes.push(0xFC);
        for r in result_types {
            canonical_bytes.extend_from_slice(format!("{r:?}").as_bytes());
        }
        canonical_bytes.push(0xFB);
        canonical_bytes.push(if proof_fields.host_shareable { 1 } else { 0 });
        canonical_bytes.push(if proof_fields.is_pure { 1 } else { 0 });
        canonical_bytes.push(if proof_fields.cse_eligible { 1 } else { 0 });
        canonical_bytes.push(if proof_fields.is_divergent { 1 } else { 0 });
        canonical_bytes.push(if proof_fields.may_alias { 1 } else { 0 });
        canonical_bytes.push(if proof_fields.terminates { 1 } else { 0 });
        canonical_bytes.extend_from_slice(proof_fields.target_capability.as_bytes());

        ExtensionSchemaDigest::from_bytes(&canonical_bytes)
    }

    /// Whether values of this extension can be safely shared across host and device.
    #[must_use]
    pub fn is_host_shareable(&self) -> bool {
        self.proof_fields.host_shareable
    }

    /// Whether this operation is purely functional.
    #[must_use]
    pub fn is_pure(&self) -> bool {
        self.proof_fields.is_pure
    }

    /// Whether this operation is eligible for common subexpression elimination.
    #[must_use]
    pub fn cse_eligible(&self) -> bool {
        self.proof_fields.cse_eligible
    }

    /// Whether this operation can cause control flow divergence.
    #[must_use]
    pub fn is_divergent(&self) -> bool {
        self.proof_fields.is_divergent
    }

    /// Whether memory accesses may alias other buffers.
    #[must_use]
    pub fn may_alias(&self) -> bool {
        self.proof_fields.may_alias
    }

    /// Whether this operation is guaranteed to terminate in bounded steps.
    #[must_use]
    pub fn terminates(&self) -> bool {
        self.proof_fields.terminates
    }

    /// Target hardware capability required.
    #[must_use]
    pub fn target_capability(&self) -> &str {
        &self.proof_fields.target_capability
    }
}

/// Compute a collision-resistant 256-bit digest over input bytes in const context.
#[must_use]
pub(crate) const fn compute_digest_bytes(bytes: &[u8]) -> [u8; 32] {
    let mut out = [0u8; 32];
    let mut state: [u64; 4] = [
        0x243f_6a88_85a3_08d3,
        0x1319_8a2e_0370_7344,
        0xa409_3822_299f_31d0,
        0x082e_fa98_ec4e_6c89,
    ];
    let mut i = 0;
    while i < bytes.len() {
        let lane = i % 4;
        state[lane] = (state[lane].rotate_left(13)
            ^ ((bytes[i] as u64).wrapping_mul(0x517c_c1b7_2722_0a95)))
        .rotate_left(17)
        .wrapping_add(0x9e37_79b9_7f4a_7c15);
        i += 1;
    }
    let mut lane = 0;
    while lane < 4 {
        let b = state[lane].to_le_bytes();
        let mut j = 0;
        while j < 8 {
            out[lane * 8 + j] = b[j];
            j += 1;
        }
        lane += 1;
    }
    out
}

/// Compute a collision-resistant 32-bit ID folded from the full 256-bit digest with the high bit set.
#[must_use]
pub(crate) const fn digest_with_high_bit(name: &str) -> u32 {
    let digest = compute_digest_bytes(name.as_bytes());
    let w0 = u32::from_le_bytes([digest[0], digest[1], digest[2], digest[3]]);
    let w1 = u32::from_le_bytes([digest[4], digest[5], digest[6], digest[7]]);
    let w2 = u32::from_le_bytes([digest[8], digest[9], digest[10], digest[11]]);
    let w3 = u32::from_le_bytes([digest[12], digest[13], digest[14], digest[15]]);
    let w4 = u32::from_le_bytes([digest[16], digest[17], digest[18], digest[19]]);
    let w5 = u32::from_le_bytes([digest[20], digest[21], digest[22], digest[23]]);
    let w6 = u32::from_le_bytes([digest[24], digest[25], digest[26], digest[27]]);
    let w7 = u32::from_le_bytes([digest[28], digest[29], digest[30], digest[31]]);
    let folded = w0
        ^ w1.rotate_left(7)
        ^ w2.rotate_left(13)
        ^ w3.rotate_left(19)
        ^ w4.rotate_left(23)
        ^ w5.rotate_left(29)
        ^ w6.rotate_left(11)
        ^ w7.rotate_left(17);
    (folded & 0x7FFF_FFFF) | 0x8000_0000
}

/// Semantic and verification proof fields required for every extension schema.
///
/// Permissive defaults are strictly forbidden: each field must be explicitly decided.
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct ExtensionProofFields {
    /// Whether values can be safely shared across host and device memory.
    pub host_shareable: bool,
    /// Whether the operation is purely functional and free of observable side effects.
    pub is_pure: bool,
    /// Whether the operation is eligible for common subexpression elimination.
    pub cse_eligible: bool,
    /// Whether execution may diverge across invocations / lanes.
    pub is_divergent: bool,
    /// Whether memory accesses in this extension may alias other buffers.
    pub may_alias: bool,
    /// Whether this extension operation is guaranteed to terminate in bounded steps.
    pub terminates: bool,
    /// Target hardware capability required for lowering or execution.
    pub target_capability: String,
}

/// Exhaustive enumeration of all required proof fields on an extension schema.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize, serde::Deserialize,
)]
pub enum ExtensionProofFieldKind {
    /// Host shareability (`host_shareable`).
    HostShareability,
    /// Mathematical purity (`is_pure`).
    Purity,
    /// Common subexpression elimination eligibility (`cse_eligible`).
    CseEligibility,
    /// Control flow and warp divergence (`is_divergent`).
    Divergence,
    /// Buffer reference aliasing (`may_alias`).
    Aliasing,
    /// Bounded step termination (`terminates`).
    Termination,
    /// Required target hardware capability (`target_capability`).
    TargetCapability,
}

impl ExtensionProofFieldKind {
    /// All required proof fields in declaration order.
    pub const ALL: &'static [Self] = &[
        Self::HostShareability,
        Self::Purity,
        Self::CseEligibility,
        Self::Divergence,
        Self::Aliasing,
        Self::Termination,
        Self::TargetCapability,
    ];

    /// Canonical string identifier for this proof field.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::HostShareability => "host_shareable",
            Self::Purity => "is_pure",
            Self::CseEligibility => "cse_eligible",
            Self::Divergence => "is_divergent",
            Self::Aliasing => "may_alias",
            Self::Termination => "terminates",
            Self::TargetCapability => "target_capability",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn id_from_name_is_deterministic() {
        assert_eq!(
            ExtensionDataTypeId::from_name("tensor.gather"),
            ExtensionDataTypeId::from_name("tensor.gather"),
        );
    }

    #[test]
    fn id_from_different_names_differ() {
        let a = ExtensionDataTypeId::from_name("tensor.gather");
        let b = ExtensionDataTypeId::from_name("tensor.scatter");
        assert_ne!(a, b);
    }

    #[test]
    fn every_id_is_in_extension_range() {
        let id = ExtensionDataTypeId::from_name("anything");
        assert!(id.is_extension(), "{:#010x} missing high bit", id.as_u32());
        assert!(id.as_u32() & ExtensionDataTypeId::EXTENSION_RANGE_MASK != 0);
    }

    #[test]
    fn extension_identity_and_proof_fields() {
        let name_a = "d5pj";
        let name_b = "x.ta";
        let id_a = ExtensionDataTypeId::from_name(name_a);
        let id_b = ExtensionDataTypeId::from_name(name_b);
        assert_ne!(
            id_a, id_b,
            "Collision-resistant IDs must distinguish distinct names"
        );

        let ns_a = ExtensionNamespace::new(name_a).expect("valid namespace");
        let ver = ExtensionSemVer::new(1, 0, 0);
        let proof = ExtensionProofFields {
            host_shareable: true,
            is_pure: true,
            cse_eligible: true,
            is_divergent: false,
            may_alias: false,
            terminates: true,
            target_capability: "generic".into(),
        };
        let digest = ExtensionSchema::compute_digest(name_a, &ver, &[], &[], &[], &proof);
        let identity = ExtensionIdentity::new(ns_a, ver, digest);
        assert!(!identity.to_canonical_string().is_empty());
    }
}
