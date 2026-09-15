//! Which calibration a quantized buffer was produced under.

use serde::{Deserialize, Serialize};

use crate::hashing::update_length_delimited_field;

/// Version of the calibration identity shape.
pub const CALIBRATION_IDENTITY_VERSION: u32 = 1;

/// The calibration or conversion a quantized buffer was produced under.
///
/// Scales and zero points are data. Two runs that calibrate the same weights on
/// different inputs produce different data for the same storage bytes, and a
/// compile under one is not the compile under the other. The identity is a
/// digest of that data rather than a name a producer chose, so a renamed
/// payload with the same content is one identity and an edited payload under
/// the same name is another.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Deserialize, Serialize)]
pub struct CalibrationIdentity {
    /// Version of the identity shape.
    pub version: u32,
    /// BLAKE3 digest of the calibration payload.
    pub digest: [u8; 32],
}

impl CalibrationIdentity {
    /// The identity of `payload`.
    ///
    /// The version is hashed as its own field, so raising
    /// [`CALIBRATION_IDENTITY_VERSION`] changes every identity rather than
    /// leaving an old digest readable as a new one.
    #[must_use]
    pub fn of(payload: &[u8]) -> Self {
        let mut hasher = blake3::Hasher::new();
        update_length_delimited_field(
            &mut hasher,
            b"vyre-calibration",
            &CALIBRATION_IDENTITY_VERSION.to_le_bytes(),
        );
        update_length_delimited_field(&mut hasher, b"payload", payload);
        Self {
            version: CALIBRATION_IDENTITY_VERSION,
            digest: *hasher.finalize().as_bytes(),
        }
    }

    /// Whether `payload` is the payload this identity was taken over.
    #[must_use]
    pub fn authenticates(&self, payload: &[u8]) -> bool {
        *self == Self::of(payload)
    }

    /// The digest as lowercase hex.
    #[must_use]
    pub fn hex(&self) -> String {
        blake3::Hash::from(self.digest).to_hex().to_string()
    }
}
