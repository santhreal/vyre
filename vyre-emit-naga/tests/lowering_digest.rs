//! The stamped lowering digest tracks this emitter's source.
//!
//! Pipeline caches mix [`vyre_emit_naga::LOWERING_DIGEST`] into their keys so an
//! emitter change cannot be answered by a pipeline the previous emitter
//! compiled. That holds only while the stamp is live: a digest that misses a
//! source file, or a build the script no longer reruns for, pins the old value
//! and the caches go back to serving stale shaders. This recomputes the digest
//! from the tree on disk and compares.
//!
//! What it does not catch: a cache that never mixes the digest in. The driver
//! crates own that side.

use std::path::{Path, PathBuf};

const FNV_OFFSET_BASIS: u128 = 0x6c62_272e_07bb_0142_62b8_2175_6295_c58d;
const FNV_PRIME: u128 = 0x0000_0000_0100_0000_0000_0000_0000_013b;
const DOMAIN: &[u8] = b"vyre-emit-naga-lowering-digest-v1\0";

#[test]
fn the_stamped_lowering_digest_matches_the_emitter_source_on_disk() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let source_dir = root.join("src");
    let mut files = collect_files(&source_dir, &source_dir);
    assert!(
        files.iter().any(|(relative, _)| relative == "lib.rs"),
        "Fix: the digest walk found no crate root, so it is not walking the emitter."
    );
    files.sort_by(|left, right| left.0.cmp(&right.0));

    let mut hash = update(FNV_OFFSET_BASIS, DOMAIN);
    for (relative, path) in &files {
        let bytes = std::fs::read(path).expect("Fix: an emitter source file must be readable.");
        hash = update(hash, relative.as_bytes());
        hash = update(hash, b"\0");
        hash = update(hash, &(bytes.len() as u64).to_le_bytes());
        hash = update(hash, &bytes);
    }
    hash = update(hash, b"\0manifest\0");
    let manifest =
        std::fs::read(root.join("Cargo.toml")).expect("Fix: the manifest must be readable.");
    hash = update(hash, &manifest);

    assert_eq!(
        vyre_emit_naga::LOWERING_DIGEST,
        format!("{hash:032x}"),
        "Fix: the stamped WGSL lowering digest is not the digest of the emitter source on disk, \
         so pipeline caches will answer an emitter change with a pipeline the previous emitter \
         compiled. Rebuild so build.rs reruns, and confirm it hashes every file under src."
    );
}

fn collect_files(root: &Path, dir: &Path) -> Vec<(String, PathBuf)> {
    let mut out = Vec::new();
    let entries = std::fs::read_dir(dir).expect("Fix: the emitter source directory must exist.");
    for entry in entries {
        let entry = entry.expect("Fix: an emitter source entry must be readable.");
        let path = entry.path();
        if path.is_dir() {
            out.extend(collect_files(root, &path));
            continue;
        }
        let relative = path
            .strip_prefix(root)
            .expect("Fix: a walked path must sit under src.")
            .to_str()
            .expect("Fix: an emitter source path must be UTF-8.")
            .replace('\\', "/");
        out.push((relative, path));
    }
    out
}

fn update(mut hash: u128, bytes: &[u8]) -> u128 {
    for byte in bytes {
        hash ^= u128::from(*byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}
