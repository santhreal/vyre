//! Stamp a digest of this emitter's own source into the crate.
//!
//! The CUDA module cache keys on the program and on the PTX it compiled. Without
//! this emitter's own identity in that key, a PTX fix is answered by the module
//! the previous emitter produced and never reaches the device. The digest
//! stamped below changes whenever this crate's source or manifest changes.
//!
//! `vyre_foundation::source_digest` owns the walk, the hash, and the rerun
//! triggers. This script names the package, the variable, and the domain.

/// Stamp the digest of this package into `VYRE_PTX_LOWERING_DIGEST`.
///
/// # Panics
///
/// Panics when `CARGO_MANIFEST_DIR` is absent or when the package cannot be
/// digested. A build script has no caller to hand an error to, and a build that
/// continued without the digest would pin the previous value and serve a module
/// the previous emitter compiled.
fn main() {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR")
        .expect("Fix: CARGO_MANIFEST_DIR names the package root; restore it before building.");
    let directives = vyre_foundation::source_digest::cargo_directives(
        std::path::Path::new(&manifest_dir),
        "VYRE_PTX_LOWERING_DIGEST",
        b"vyre-emit-ptx-lowering-digest-v1\0",
    )
    .unwrap_or_else(|error| panic!("{error}"));
    print!("{directives}");
}
