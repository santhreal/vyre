//! One frame around every versioned byte product this crate emits.
//!
//! A frame is a four-byte magic, a little-endian schema version, a little-endian
//! body length, the canonical body, and a domain-separated digest of it. Every
//! product that crosses a process boundary (the neutral artifact, the envelope
//! around it, one target payload) uses this layout, so the layout is written
//! once: a second copy of it is a second reader of the same bytes, and the two
//! drift on error paths long before they drift on the happy path.

use crate::{failure, CompileError, CompilerFailureKind};

/// Magic, schema version, and body length, in that order.
pub(crate) const HEADER_BYTES: usize = 10;
/// Width of the trailing content digest.
pub(crate) const DIGEST_BYTES: usize = 32;

/// How one product frames its bytes and names its failures.
pub(crate) struct Frame {
    /// Four bytes that identify the product.
    pub(crate) magic: &'static [u8; 4],
    /// Schema this build reads and writes.
    pub(crate) version: u16,
    /// Digest domain separator, so identical bodies of different products differ.
    pub(crate) domain: &'static [u8],
    /// Diagnostic path root, such as `artifact`.
    pub(crate) path: &'static str,
    /// Failure for bytes carrying another schema version.
    pub(crate) version_skew: CompilerFailureKind,
    /// Failure for a body that does not match its recorded identity.
    pub(crate) digest_mismatch: CompilerFailureKind,
}

/// Frame around the neutral artifact.
pub(crate) const ARTIFACT: Frame = Frame {
    magic: b"VMK0",
    version: crate::ARTIFACT_SCHEMA_VERSION,
    domain: b"vyre-megakernel-artifact-v4\0",
    path: "artifact",
    version_skew: CompilerFailureKind::VersionSkew,
    digest_mismatch: CompilerFailureKind::DigestMismatch,
};

/// Frame around the envelope that carries an artifact and its target payloads.
pub(crate) const ENVELOPE: Frame = Frame {
    magic: b"VME0",
    version: crate::ARTIFACT_ENVELOPE_SCHEMA_VERSION,
    domain: b"vyre-megakernel-envelope-v2\0",
    path: "envelope",
    version_skew: CompilerFailureKind::VersionSkew,
    digest_mismatch: CompilerFailureKind::DigestMismatch,
};

/// Frame around one target payload attachment.
pub(crate) const TARGET_PAYLOAD: Frame = Frame {
    magic: b"VTP0",
    version: crate::TARGET_PAYLOAD_SCHEMA_VERSION,
    domain: b"vyre-megakernel-target-payload-v3\0",
    path: "target_payload",
    version_skew: CompilerFailureKind::TargetPayloadVersionSkew,
    digest_mismatch: CompilerFailureKind::TargetPayloadDigestMismatch,
};

/// Every frame this crate emits, so a rule can be proved against all of them.
pub(crate) const FRAMES: &[&Frame] = &[&ARTIFACT, &ENVELOPE, &TARGET_PAYLOAD];

/// Framed bytes and the identity stamped into them.
pub(crate) struct Framed {
    /// Complete frame, ready to write.
    pub(crate) bytes: Vec<u8>,
    /// Digest of the body, as it appears in the trailer.
    pub(crate) digest: [u8; DIGEST_BYTES],
}

/// One authenticated frame, borrowed from the bytes it was read out of.
#[derive(Debug)]
pub(crate) struct Decoded<'a> {
    /// Schema version the frame declares, already checked against the reader.
    pub(crate) version: u16,
    /// Canonical body between the header and the digest.
    pub(crate) body: &'a [u8],
    /// Digest carried by the frame, already checked against the body.
    pub(crate) digest: [u8; DIGEST_BYTES],
}

impl Frame {
    /// Digest of one canonical body under this product's domain and version.
    pub(crate) fn digest(&self, version: u16, body: &[u8]) -> [u8; DIGEST_BYTES] {
        let mut hasher = blake3::Hasher::new();
        hasher.update(self.domain);
        hasher.update(&version.to_le_bytes());
        hasher.update(&(body.len() as u64).to_le_bytes());
        hasher.update(body);
        *hasher.finalize().as_bytes()
    }

    /// Frame one canonical body, reporting the identity it was stamped with.
    pub(crate) fn encode(&self, version: u16, body: &[u8]) -> Result<Framed, CompileError> {
        let body_len = u32::try_from(body.len()).map_err(|_| {
            failure(
                CompilerFailureKind::ResourceOverflow,
                format!("{}.body", self.path),
                "canonical body exceeds the u32 framing limit",
                "reduce or detach the framed bytes",
            )
        })?;
        let capacity = HEADER_BYTES
            .checked_add(body.len())
            .and_then(|len| len.checked_add(DIGEST_BYTES))
            .ok_or_else(|| {
                failure(
                    CompilerFailureKind::ResourceOverflow,
                    self.path,
                    "framed length overflowed addressable memory",
                    "reduce or detach the framed bytes",
                )
            })?;
        let digest = self.digest(version, body);
        let mut bytes = Vec::with_capacity(capacity);
        bytes.extend_from_slice(self.magic);
        bytes.extend_from_slice(&version.to_le_bytes());
        bytes.extend_from_slice(&body_len.to_le_bytes());
        bytes.extend_from_slice(body);
        bytes.extend_from_slice(&digest);
        Ok(Framed { bytes, digest })
    }

    /// Read one frame, rejecting foreign, truncated, skewed, or corrupt bytes.
    pub(crate) fn decode<'a>(&self, bytes: &'a [u8]) -> Result<Decoded<'a>, CompileError> {
        if bytes.len() < HEADER_BYTES + DIGEST_BYTES {
            return Err(failure(
                CompilerFailureKind::MalformedArtifact,
                format!("{}.header", self.path),
                "framed bytes are shorter than the fixed header and digest",
                "supply one complete canonical frame",
            ));
        }
        if &bytes[..4] != self.magic {
            return Err(failure(
                CompilerFailureKind::MalformedArtifact,
                format!("{}.magic", self.path),
                format!(
                    "framing magic is not {}",
                    String::from_utf8_lossy(self.magic)
                ),
                "supply bytes emitted for this artifact layer",
            ));
        }
        let version = u16::from_le_bytes([bytes[4], bytes[5]]);
        if version != self.version {
            return Err(failure(
                self.version_skew,
                format!("{}.schema_version", self.path),
                format!("schema {version} is unsupported; expected {}", self.version),
                "recompile or re-materialize with a compatible schema version",
            ));
        }
        // The length check above proves bytes 6..10 exist.
        let body_len = u32::from_le_bytes([bytes[6], bytes[7], bytes[8], bytes[9]]) as usize;
        let expected_len = HEADER_BYTES
            .checked_add(body_len)
            .and_then(|len| len.checked_add(DIGEST_BYTES))
            .ok_or_else(|| {
                failure(
                    CompilerFailureKind::MalformedArtifact,
                    format!("{}.body_length", self.path),
                    "framed body length overflowed addressable memory",
                    "supply bounded canonical bytes",
                )
            })?;
        if bytes.len() != expected_len {
            return Err(failure(
                CompilerFailureKind::MalformedArtifact,
                format!("{}.body_length", self.path),
                format!(
                    "framing declares {expected_len} bytes but received {}",
                    bytes.len()
                ),
                "supply exactly one complete canonical frame",
            ));
        }
        let body = &bytes[HEADER_BYTES..HEADER_BYTES + body_len];
        let digest: [u8; DIGEST_BYTES] =
            bytes[HEADER_BYTES + body_len..].try_into().map_err(|_| {
                failure(
                    self.digest_mismatch,
                    format!("{}.digest", self.path),
                    "framed trailer is not one digest wide",
                    "supply exactly one complete canonical frame",
                )
            })?;
        if self.digest(version, body) != digest {
            return Err(failure(
                self.digest_mismatch,
                format!("{}.digest", self.path),
                "framed body does not match its content identity",
                "discard the corrupted bytes and regenerate them",
            ));
        }
        Ok(Decoded {
            version,
            body,
            digest,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn code_of(error: &CompileError) -> String {
        error.diagnostic.code.as_str().to_string()
    }

    fn path_of(error: &CompileError) -> String {
        error
            .diagnostic
            .location
            .as_ref()
            .and_then(|location| location.path.clone())
            .unwrap_or_default()
    }

    /// WHY: three products share this layout, and before they shared this code
    /// each carried its own reader. Every rejection is proved for every frame in
    /// `FRAMES` rather than for the artifact alone, so adding a product cannot
    /// pick up a reader that accepts truncated, foreign, skewed, or corrupt
    /// bytes. What this does not catch: a frame declared and never added to
    /// `FRAMES`, which the distinct-identity assertions below cannot see either.
    #[test]
    fn every_frame_rejects_truncation_foreign_magic_skew_and_tampering() {
        for frame in FRAMES {
            let framed = frame
                .encode(frame.version, b"body-bytes")
                .unwrap_or_else(|error| panic!("{} must frame a body: {error}", frame.path));
            assert_eq!(framed.bytes.len(), HEADER_BYTES + 10 + DIGEST_BYTES);

            let decoded = frame
                .decode(&framed.bytes)
                .unwrap_or_else(|error| panic!("{} must read its own frame: {error}", frame.path));
            assert_eq!(decoded.body, b"body-bytes");
            assert_eq!(decoded.version, frame.version);
            assert_eq!(decoded.digest, framed.digest);

            let short = frame
                .decode(&framed.bytes[..HEADER_BYTES + DIGEST_BYTES - 1])
                .expect_err("a frame shorter than its fixed parts is unreadable");
            assert_eq!(path_of(&short), format!("{}.header", frame.path));

            let mut foreign = framed.bytes.clone();
            foreign[..4].copy_from_slice(b"XXXX");
            let foreign = frame
                .decode(&foreign)
                .expect_err("another product's bytes are not this product's");
            assert_eq!(path_of(&foreign), format!("{}.magic", frame.path));

            let mut skewed = framed.bytes.clone();
            skewed[4..6].copy_from_slice(&frame.version.wrapping_add(1).to_le_bytes());
            let skewed = frame
                .decode(&skewed)
                .expect_err("another schema version is not readable");
            assert_eq!(code_of(&skewed), frame.version_skew.as_str());
            assert_eq!(path_of(&skewed), format!("{}.schema_version", frame.path));

            let mut long = framed.bytes.clone();
            long[6..10].copy_from_slice(&11u32.to_le_bytes());
            let long = frame
                .decode(&long)
                .expect_err("a declared length that misses the trailer is unreadable");
            assert_eq!(path_of(&long), format!("{}.body_length", frame.path));

            let mut tampered = framed.bytes.clone();
            tampered[HEADER_BYTES] ^= 1;
            let tampered = frame
                .decode(&tampered)
                .expect_err("a mutated body does not match its identity");
            assert_eq!(code_of(&tampered), frame.digest_mismatch.as_str());
            assert_eq!(path_of(&tampered), format!("{}.digest", frame.path));

            let mut restamped = framed.bytes.clone();
            let trailer = restamped.len() - DIGEST_BYTES;
            restamped[trailer] ^= 1;
            let restamped = frame
                .decode(&restamped)
                .expect_err("a rewritten trailer does not authenticate the body");
            assert_eq!(code_of(&restamped), frame.digest_mismatch.as_str());
        }
    }

    /// WHY: the digest domain is what stops one product's bytes from
    /// authenticating as another's, and the magic is what stops them from being
    /// read at all. Both must be unique per frame, and an empty body must still
    /// produce distinct identities.
    #[test]
    fn no_two_frames_share_a_magic_a_domain_or_a_digest() {
        for (index, frame) in FRAMES.iter().enumerate() {
            for other in &FRAMES[index + 1..] {
                assert_ne!(frame.magic, other.magic, "{} magic", frame.path);
                assert_ne!(frame.domain, other.domain, "{} domain", frame.path);
                assert_ne!(
                    frame.digest(1, b""),
                    other.digest(1, b""),
                    "{} and {} must separate their digest domains",
                    frame.path,
                    other.path
                );
            }
        }
    }
}
