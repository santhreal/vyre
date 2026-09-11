//! The stamped lowering digest tracks this emitter's source.
//!
//! Pipeline caches mix [`vyre_emit_naga::LOWERING_DIGEST`] into their keys so an
//! emitter change cannot be answered by a pipeline the previous emitter
//! compiled. That holds only while the stamp is live: a build the script no
//! longer reruns for pins the old value and the caches go back to serving stale
//! shaders. This recomputes the digest from the tree on disk and compares.
//!
//! What it does not catch: a cache that never mixes the digest in. The driver
//! crates own that side. `vyre-foundation` owns whether the digest covers the
//! tree.

#[test]
fn the_stamped_lowering_digest_matches_the_emitter_source_on_disk() {
    let package_root = vyre_test_support::monorepo::vyre_crate_directory(env!("CARGO_PKG_NAME"));
    let recomputed = vyre_foundation::source_digest::source_tree_digest(
        &package_root,
        b"vyre-emit-naga-lowering-digest-v1\0",
    )
    .expect("Fix: the emitter source must digest; the build script reads the same tree.");

    assert_eq!(
        vyre_emit_naga::LOWERING_DIGEST,
        recomputed,
        "Fix: the stamped WGSL lowering digest is not the digest of the emitter source on disk, \
         so pipeline caches will answer an emitter change with a pipeline the previous emitter \
         compiled. Rebuild so build.rs reruns."
    );
}
