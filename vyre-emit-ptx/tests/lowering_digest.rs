//! The stamped lowering digest tracks this emitter's source.
//!
//! The CUDA PTX caches mix [`vyre_emit_ptx::LOWERING_DIGEST`] into their keys so
//! an emitter change cannot be answered by PTX the previous emitter wrote. That
//! holds only while the stamp is live: a build the script no longer reruns for
//! pins the old value and the caches go back to serving stale PTX. This
//! recomputes the digest from the tree on disk and compares.
//!
//! What it does not catch: a cache that never mixes the digest in. The driver
//! crates own that side. `vyre-foundation` owns whether the digest covers the
//! tree.

#[test]
fn the_stamped_lowering_digest_matches_the_emitter_source_on_disk() {
    let package_root = vyre_test_support::monorepo::vyre_crate_directory(env!("CARGO_PKG_NAME"));
    let recomputed = vyre_foundation::source_digest::source_tree_digest(
        &package_root,
        b"vyre-emit-ptx-lowering-digest-v1\0",
    )
    .expect("Fix: the emitter source must digest; the build script reads the same tree.");

    assert_eq!(
        vyre_emit_ptx::LOWERING_DIGEST,
        recomputed,
        "Fix: the stamped PTX lowering digest is not the digest of the emitter source on disk, \
         so the PTX source cache will answer an emitter change with text the previous emitter \
         wrote. Rebuild so build.rs reruns."
    );
}
