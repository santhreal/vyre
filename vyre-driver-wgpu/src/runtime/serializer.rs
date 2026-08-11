//! Runtime wire-format serialization for multi-part programs.

/// Owner-local runtime framing failure.
#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
pub enum SerializerError {
    /// Input bytes do not satisfy the framing contract.
    #[error("invalid runtime frame: {message}")]
    InvalidFrame {
        /// Actionable framing failure.
        message: String,
    },
    /// Frame encoding, decoding, sizing, or allocation failed.
    #[error("runtime frame serialization failed: {message}")]
    Serialization {
        /// Actionable serialization failure.
        message: String,
    },
}

/// Runtime framing result.
pub type SerializerResult<T, E = SerializerError> = Result<T, E>;

pub use decode_parts::decode_parts;
pub use encode_parts::{encode_parts, MAX_SERIALIZED_PART_BYTES};

/// Runtime frame decoder.
pub mod decode_parts;
/// Runtime frame encoder and serializer limits.
pub mod encode_parts;
