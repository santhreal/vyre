//! Stamp a digest of this emitter's own source into the crate.
//!
//! A pipeline cache that keys on the program alone serves a pipeline compiled
//! by an older emitter: the program is unchanged, so the key is unchanged, and
//! an emitter fix never reaches the device. The lowering-contract label the
//! driver mixes in covers that only while someone remembers to edit it. The
//! digest below changes whenever this crate's source or manifest changes, so a
//! cache that mixes it in invalidates on every emitter edit.
//!
//! Reads only this crate's own directory, because the workspace root is not
//! available inside the crates.io tarball.

use std::path::{Path, PathBuf};

const MAX_DIGEST_INPUT_BYTES: u64 = 64 * 1024 * 1024;
const FNV_OFFSET_BASIS: u128 = 0x6c62_272e_07bb_0142_62b8_2175_6295_c58d;
const FNV_PRIME: u128 = 0x0000_0000_0100_0000_0000_0000_0000_013b;
const DOMAIN: &[u8] = b"vyre-emit-ptx-lowering-digest-v1\0";

fn fail(message: impl std::fmt::Display) -> ! {
    eprintln!("Fix: {message}");
    std::process::exit(1);
}

fn main() {
    let manifest_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| {
        fail("CARGO_MANIFEST_DIR missing; restore this invariant before continuing.")
    }));
    let source_dir = manifest_dir.join("src");
    let manifest_path = manifest_dir.join("Cargo.toml");

    let mut files = Vec::new();
    collect_files(&source_dir, &source_dir, &mut files);
    files.sort_by(|left, right| left.0.cmp(&right.0));

    let mut total = 0u64;
    let mut hash = update(FNV_OFFSET_BASIS, DOMAIN);
    for (relative, path) in &files {
        let bytes = read_bounded(path, &mut total);
        hash = update(hash, relative.as_bytes());
        hash = update(hash, b"\0");
        hash = update(hash, &(bytes.len() as u64).to_le_bytes());
        hash = update(hash, &bytes);
    }
    hash = update(hash, b"\0manifest\0");
    hash = update(hash, &read_bounded(&manifest_path, &mut total));

    println!("cargo:rerun-if-changed={}", source_dir.display());
    println!("cargo:rerun-if-changed={}", manifest_path.display());
    println!("cargo:rustc-env=VYRE_PTX_LOWERING_DIGEST={hash:032x}");
}

fn collect_files(root: &Path, dir: &Path, out: &mut Vec<(String, PathBuf)>) {
    let entries = std::fs::read_dir(dir)
        .unwrap_or_else(|error| fail(format!("failed to read {}: {error}", dir.display())));
    for entry in entries {
        let entry =
            entry.unwrap_or_else(|error| fail(format!("failed to read {}: {error}", dir.display())));
        let path = entry.path();
        let kind = entry
            .file_type()
            .unwrap_or_else(|error| fail(format!("failed to stat {}: {error}", path.display())));
        if kind.is_dir() {
            collect_files(root, &path, out);
            continue;
        }
        let relative = path
            .strip_prefix(root)
            .unwrap_or_else(|error| {
                fail(format!(
                    "{} escaped {}: {error}",
                    path.display(),
                    root.display()
                ))
            })
            .to_str()
            .unwrap_or_else(|| fail(format!("{} is not valid UTF-8", path.display())))
            .replace('\\', "/");
        out.push((relative, path));
    }
}

fn read_bounded(path: &Path, total: &mut u64) -> Vec<u8> {
    let bytes = std::fs::read(path)
        .unwrap_or_else(|error| fail(format!("failed to read {}: {error}", path.display())));
    *total = total.saturating_add(bytes.len() as u64);
    if *total > MAX_DIGEST_INPUT_BYTES {
        fail(format!(
            "emitter source exceeds the {MAX_DIGEST_INPUT_BYTES} byte build-script cap at {}",
            path.display()
        ));
    }
    bytes
}

fn update(mut hash: u128, bytes: &[u8]) -> u128 {
    for byte in bytes {
        hash ^= u128::from(*byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}
