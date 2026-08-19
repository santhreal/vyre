//! Stamp a digest of this emitter's own source into the crate.
//!
//! A pipeline cache that keys on the program alone serves a pipeline compiled
//! by an older emitter: the program is unchanged, so the key is unchanged, and
//! an emitter fix never reaches the device. The lowering-contract label the
//! driver mixes in covers that only while someone remembers to edit it. The
//! digest stamped below changes whenever this crate's source or manifest
//! changes, so a cache that mixes it in invalidates on every emitter edit.
//!
//! `vyre_foundation::source_digest` owns the walk, the hash, and the rerun
//! triggers. This script names the package, the variable, and the domain.

fn main() {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR")
        .expect("Fix: CARGO_MANIFEST_DIR names the package root; restore it before building.");
    let directives = vyre_foundation::source_digest::cargo_directives(
        std::path::Path::new(&manifest_dir),
        "VYRE_NAGA_LOWERING_DIGEST",
        b"vyre-emit-naga-lowering-digest-v1\0",
    )
    .unwrap_or_else(|error| panic!("{error}"));
    print!("{directives}");
}
