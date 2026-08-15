//! Wire-format envelope constants.

/// Leading magic bytes for the vyre IR wire format.
///
/// The `VYRE` tag identifies the stable schema; the version byte pair
/// [`WIRE_FORMAT_VERSION`] immediately after the magic identifies which
/// schema version the payload follows.
pub const MAGIC: &[u8; 4] = b"VYRE";

/// Current wire-format schema version. Incremented on any
/// breaking schema change (variant added/removed, field reorder, new
/// non-additive framing). Audit L.1.47: the previous format lacked a
/// version field entirely; any schema drift produced arbitrary
/// parse errors with no way to tell caller "you're on a newer
/// version than this decoder knows about".
///
/// Rev 6 records three `BufferDecl` fields the encoder previously dropped:
/// `linear_type`, `bytes_extraction`, and `shape_predicate`. Before rev 6 a
/// round trip silently reset all three to their defaults, which changed a
/// program's meaning (losing `bytes_extraction` flips a V013 validation
/// verdict) and left `Program::fingerprint` blind to them, so two programs
/// differing only in one of the three shared a cache identity.
///
/// Rev 5 adds expression tag 22, `Expr::BufferRef`. Rev 4 preserves
/// program-level composition-safety flags in metadata so parser/stateful
/// kernels do not become fusible after wire round trip. Rev 3 introduces:
/// structured version-mismatch errors (see
/// [`crate::error::IrError::VersionMismatch`]) and a reserved
/// dialect-manifest section after the header for rev-3+ readers. Rev
/// 2 was never released; versions go 1 to 3 directly.
pub const WIRE_FORMAT_VERSION: u16 = 6;

/// Oldest schema version this decoder reads.
///
/// Rev 6 appends three fields INSIDE each buffer record, so a rev-5 reader
/// cannot read rev-6 bytes. The reverse direction is preserved deliberately:
/// the rev-6 decoder gates those three reads on `version >= 6` and applies the
/// historical defaults for older payloads, so rev-4 and rev-5 blobs still
/// load.
///
/// Be precise about what that guarantee is NOT. A rev-4 or rev-5 blob written
/// from a program that DID carry a linear type, a bytes-extraction opt-in, or a
/// shape predicate lost that information at encode time, permanently. No
/// decoder can recover a field the writer never emitted. The guarantee is
/// faithfulness to what those bytes actually say, not recovery of what the
/// program once was.
///
/// Rev 5 only appended an expression tag, so rev-4 bytes decode under the
/// rev-5 rules unchanged. Anything older predates the metadata layout and
/// genuinely cannot be read.
pub const MIN_SUPPORTED_WIRE_FORMAT_VERSION: u16 = 4;

/// Maximum nesting depth accepted for a `ShapePredicate` on the wire.
///
/// `ShapePredicate::And`, `Or` and `Not` are recursive, so an untrusted blob
/// could otherwise nest them deeply enough to overflow the decoder stack.
/// Both the encoder and the decoder read this one constant, so the two sides
/// cannot drift into disagreeing about what is representable.
pub const MAX_SHAPE_PREDICATE_DEPTH: usize = 32;

/// Whether this decoder can read a payload stamped with `version`.
///
/// The one place the accepted range is decided. Three decode paths check the
/// header version, and while each spelled the comparison itself, widening the
/// range for rev 5 fixed two of them and left the third rejecting rev 4.
#[must_use]
#[inline]
pub fn wire_format_version_is_supported(version: u16) -> bool {
    (MIN_SUPPORTED_WIRE_FORMAT_VERSION..=WIRE_FORMAT_VERSION).contains(&version)
}
