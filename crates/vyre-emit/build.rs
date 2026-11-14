//! Stamp the digest of this crate's source into `VYRE_EMIT_LOWERING_DIGEST`.

#[path = "src/source_digest.rs"]
#[allow(dead_code)]
mod source_digest;

/// # Panics
///
/// Panics when `CARGO_MANIFEST_DIR` is absent or the package cannot be digested.
/// A build that continued without the digest would pin the previous value and
/// serve artifacts the previous emitter produced.
fn main() {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR")
        .expect("Fix: CARGO_MANIFEST_DIR names the package root; restore it before building.");
    let directives = source_digest::cargo_directives(
        std::path::Path::new(&manifest_dir),
        "VYRE_EMIT_LOWERING_DIGEST",
        source_digest::LOWERING_DIGEST_DOMAIN,
    )
    .unwrap_or_else(|error| panic!("{error}"));
    print!("{directives}");
}
