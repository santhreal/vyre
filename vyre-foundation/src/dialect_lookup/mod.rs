//! Operation signature schema.

/// Attribute value type declared by an operation schema.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum AttrType {
    /// Unsigned 32-bit integer.
    U32,
    /// Signed 32-bit integer.
    I32,
    /// IEEE-754 binary32.
    F32,
    /// Boolean.
    Bool,
    /// Opaque byte string.
    Bytes,
    /// UTF-8 string.
    String,
    /// Enumerated string value.
    Enum(&'static [&'static str]),
    /// Unknown extension attribute.
    Unknown,
}

/// Attribute schema entry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AttrSchema {
    /// Attribute name.
    pub name: &'static str,
    /// Attribute value type.
    pub ty: AttrType,
    /// Optional default value.
    pub default: Option<&'static str>,
}

/// Typed input or output parameter.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TypedParam {
    /// Parameter name.
    pub name: &'static str,
    /// Stable type spelling.
    pub ty: &'static str,
}

/// Operation signature contract.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Signature {
    /// Input parameters.
    pub inputs: &'static [TypedParam],
    /// Output parameters.
    pub outputs: &'static [TypedParam],
    /// Attribute parameters.
    pub attrs: &'static [AttrSchema],
    /// True when this operation may read byte buffers.
    pub bytes_extraction: bool,
}

impl Signature {
    /// Construct a signature for an operation that performs byte extraction.
    #[must_use]
    pub const fn bytes_extractor(
        inputs: &'static [TypedParam],
        outputs: &'static [TypedParam],
        attrs: &'static [AttrSchema],
    ) -> Self {
        Self {
            inputs,
            outputs,
            attrs,
            bytes_extraction: true,
        }
    }
}

#[cfg(test)]
#[path = "../../tests/internal/dialect_lookup/mod.rs"]
mod tests;
