//! Submission-bundle packager.
//!
//! Takes an authenticated [`ArtifactEnvelope`] plus uncompressed weight bytes
//! and a launcher source tree and writes the on-disk submission directory:
//!
//! ```text
//! <bundle_dir>/
//! ├── manifest.json
//! ├── kernel.<ext>.lzma         (LZMA-compressed kernel bytes)
//! ├── weights.brotli            (Brotli-11-compressed weight bytes)
//! ├── pgolf-launcher/           (Rust launcher crate source)
//! │   ├── Cargo.toml
//! │   ├── .cargo/config.toml
//! │   └── src/{main.rs,artifact.rs,...}
//! └── README.md
//! ```
//!
//! The launcher source is shipped *unbuilt* by default. Submission
//! packaging compiles it once on the target hardware (5090 / H100) and
//! ships the static binary in place of the source tree.

use std::collections::BTreeMap;
use std::fs;
use std::io::{Cursor, Write};
use std::path::{Path, PathBuf};

fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    let result = hasher.finalize();
    result.iter().map(|b| format!("{:02x}", b)).collect()
}
use thiserror::Error;

use crate::artifact::Target;
use crate::launcher::{emit_launcher_rust, LauncherError, LauncherOpts};
use crate::manifest::Manifest;
use crate::VERSION;
use vyre_megakernel::{ArtifactEnvelope, TargetPayload};

const METRIC_RECORD_WORDS: u32 = 8;

/// Files written for one deployable artifact envelope.
#[derive(Debug, Clone)]
pub struct DeploymentBundle {
    /// Files written to disk relative to bundle root, with absolute paths.
    pub files: Vec<PathBuf>,
}

/// Error variants returned by [`bundle`].
#[derive(Debug, Error)]
pub enum BundleError {
    /// I/O while writing files.
    #[error("vyre-aot bundle: i/o error: {0}")]
    Io(#[from] std::io::Error),

    /// JSON serialization of the manifest failed.
    #[error("vyre-aot bundle: manifest serialization: {0}")]
    Json(#[from] serde_json::Error),

    /// Canonical artifact envelope validation failed.
    #[error("vyre-aot bundle: canonical artifact: {0}")]
    CanonicalArtifact(#[from] vyre_megakernel::CompileError),

    /// LZMA compression failed.
    #[error("vyre-aot bundle: lzma error: {0}")]
    Lzma(String),

    /// Brotli compression failed.
    #[error("vyre-aot bundle: brotli error: {0}")]
    Brotli(String),

    /// Launcher generation failed.
    #[error(transparent)]
    Launcher(#[from] LauncherError),

    /// Artifact ABI cannot be represented by the emitted launcher contract.
    #[error("vyre-aot bundle: invalid artifact: {0}")]
    InvalidArtifact(String),
}

/// Write the full bundle.
///
/// `weights` is the uncompressed weight bytes (bytes the launcher will
/// upload to the `params` device buffer after Brotli decompression).
pub fn bundle(
    out_dir: &Path,
    envelope: &ArtifactEnvelope,
    target: Target,
    weights: &[u8],
    artifact_name: &str,
    launcher_opts: &LauncherOpts,
    notes: &str,
) -> Result<DeploymentBundle, BundleError> {
    validate_artifact_for_bundle(envelope, target, weights)?;
    let launcher_tree: BTreeMap<PathBuf, String> =
        emit_launcher_rust(envelope, target, launcher_opts)?;
    fs::create_dir_all(out_dir)?;
    let mut written =
        write_package_files(out_dir, envelope, target, weights, artifact_name, notes)?;

    // 5. Write launcher source tree.
    let launcher_root = out_dir.join(&launcher_opts.crate_name);
    written.reserve(launcher_tree.len() + 1);
    for (rel, contents) in launcher_tree {
        let abs = launcher_root.join(rel);
        if let Some(parent) = abs.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(&abs, contents)?;
        written.push(abs);
    }

    // 6. Write top-level README.
    let readme = format!(
        "# {artifact_name}\n\n\
        Self-contained vyre-aot bundle.\n\n\
        ## Build the launcher\n\n\
        ```\n\
        cd {crate_name}\n\
        ./cargo_full build --release\n\
        ```\n\n\
        ## Run\n\n\
        ```\n\
        {crate_name}/target/release/{crate_name} <bundle_dir>\n\
        ```\n",
        crate_name = launcher_opts.crate_name,
    );
    let readme_path = out_dir.join("README.md");
    fs::write(&readme_path, readme)?;

    written.push(readme_path);

    Ok(DeploymentBundle { files: written })
}

/// Write the canonical envelope, weights, and manifest without generating a launcher.
pub fn package_artifact(
    out_dir: &Path,
    envelope: &ArtifactEnvelope,
    target: Target,
    weights: &[u8],
    artifact_name: &str,
    notes: &str,
) -> Result<DeploymentBundle, BundleError> {
    validate_artifact_for_bundle(envelope, target, weights)?;
    fs::create_dir_all(out_dir)?;
    Ok(DeploymentBundle {
        files: write_package_files(out_dir, envelope, target, weights, artifact_name, notes)?,
    })
}

fn write_package_files(
    out_dir: &Path,
    envelope: &ArtifactEnvelope,
    target: Target,
    weights: &[u8],
    artifact_name: &str,
    notes: &str,
) -> Result<Vec<PathBuf>, BundleError> {
    let envelope_bytes = envelope.to_bytes()?;
    let envelope_compressed = lzma_compress(&envelope_bytes)?;
    let envelope_filename = "artifact.vmk.lzma".to_string();
    let envelope_path = out_dir.join(&envelope_filename);
    fs::write(&envelope_path, &envelope_compressed)?;

    let weights_compressed = brotli_compress(weights)?;
    let weights_filename = "weights.brotli".to_string();
    let weights_path = out_dir.join(&weights_filename);
    fs::write(&weights_path, &weights_compressed)?;

    let target_payload = target_payload(envelope, target)?;
    let manifest = Manifest {
        schema: Manifest::SCHEMA_VERSION.to_string(),
        aot_version: VERSION.to_string(),
        artifact_name: artifact_name.to_string(),
        target,
        envelope_file: envelope_filename,
        envelope_compression: "lzma".to_string(),
        envelope_sha256_hex: sha256_hex(&envelope_bytes),
        neutral_artifact_digest_hex: digest_hex(envelope.neutral().digest()),
        target_payload_digest_hex: digest_hex(target_payload.digest()),
        weights_file: weights_filename,
        weights_compression: "brotli-11".to_string(),
        weights_sha256_hex: sha256_hex(weights),
        notes: notes.to_string(),
    };
    let manifest_path = out_dir.join("manifest.json");
    fs::write(&manifest_path, serde_json::to_vec_pretty(&manifest)?)?;
    Ok(vec![envelope_path, weights_path, manifest_path])
}

/// Read and authenticate a packaged canonical artifact envelope.
pub fn read_bundle_artifact(
    bundle_dir: &Path,
) -> Result<(Manifest, ArtifactEnvelope), BundleError> {
    let manifest_bytes = fs::read(bundle_dir.join("manifest.json"))?;
    let manifest: Manifest = serde_json::from_slice(&manifest_bytes)?;
    if manifest.schema != Manifest::SCHEMA_VERSION {
        return Err(BundleError::InvalidArtifact(format!(
            "manifest schema `{}` is incompatible; expected `{}`",
            manifest.schema,
            Manifest::SCHEMA_VERSION
        )));
    }
    if manifest.envelope_compression != "lzma" {
        return Err(BundleError::InvalidArtifact(format!(
            "envelope compression `{}` is unsupported; expected `lzma`",
            manifest.envelope_compression
        )));
    }
    let compressed = fs::read(bundle_dir.join(&manifest.envelope_file))?;
    let envelope_bytes = lzma_decompress(&compressed)?;
    if sha256_hex(&envelope_bytes) != manifest.envelope_sha256_hex {
        return Err(BundleError::InvalidArtifact(
            "canonical envelope SHA-256 does not match manifest identity".to_string(),
        ));
    }
    let envelope = ArtifactEnvelope::from_bytes(&envelope_bytes)?;
    if digest_hex(envelope.neutral().digest()) != manifest.neutral_artifact_digest_hex {
        return Err(BundleError::InvalidArtifact(
            "neutral artifact digest does not match manifest identity".to_string(),
        ));
    }
    if digest_hex(target_payload(&envelope, manifest.target)?.digest())
        != manifest.target_payload_digest_hex
    {
        return Err(BundleError::InvalidArtifact(
            "target payload digest does not match manifest identity".to_string(),
        ));
    }
    Ok((manifest, envelope))
}

fn validate_artifact_for_bundle(
    envelope: &ArtifactEnvelope,
    target: Target,
    weights: &[u8],
) -> Result<(), BundleError> {
    let payload = target_payload(envelope, target)?;
    if payload.bytes().is_empty() {
        return Err(BundleError::InvalidArtifact(
            "target payload bytes are empty".to_string(),
        ));
    }
    let entry = payload.entries().first().ok_or_else(|| {
        BundleError::InvalidArtifact("target payload has no entry metadata".to_string())
    })?;
    if entry.resource_bindings.is_empty() {
        return Err(BundleError::InvalidArtifact(
            "target entry has no canonical resource bindings".to_string(),
        ));
    }
    let neutral = envelope.neutral();
    let geometry = neutral
        .geometry()
        .iter()
        .find(|geometry| geometry.node == entry.node)
        .ok_or_else(|| {
            BundleError::InvalidArtifact(
                "target entry is not associated with canonical neutral geometry".to_string(),
            )
        })?;
    validate_axes("workgroup_size", geometry.workgroup_size)?;
    validate_axes("grid_size", entry.grid_size)?;
    validate_resource_bindings(envelope, payload)?;
    validate_weight_payload_fits_first_finite_resource(envelope, payload, weights)
}

fn validate_axes(label: &str, axes: [u32; 3]) -> Result<(), BundleError> {
    if let Some(axis) = axes.iter().position(|extent| *extent == 0) {
        return Err(BundleError::InvalidArtifact(format!(
            "{label} axis {axis} is zero; explicit positive geometry is required"
        )));
    }
    u64::from(axes[0])
        .checked_mul(u64::from(axes[1]))
        .and_then(|xy| xy.checked_mul(u64::from(axes[2])))
        .ok_or_else(|| {
            BundleError::InvalidArtifact(format!(
                "{label} {axes:?} overflows u64; shard the AOT dispatch"
            ))
        })?;
    Ok(())
}

fn validate_resource_bindings(
    envelope: &ArtifactEnvelope,
    payload: &TargetPayload,
) -> Result<(), BundleError> {
    let neutral = envelope.neutral();
    let entry = &payload.entries()[0];
    let mut metrics_resources = 0_usize;
    for binding in &entry.resource_bindings {
        let resource = neutral
            .resources()
            .iter()
            .find(|resource| resource.value == binding.resource)
            .ok_or_else(|| {
                BundleError::InvalidArtifact(format!(
                    "binding slot {} names missing canonical resource {}",
                    binding.slot, binding.resource.0
                ))
            })?;
        if resource.name == "metrics" {
            metrics_resources += 1;
            let element_size = if resource.element_count == 0 {
                0
            } else {
                resource.byte_count / resource.element_count
            };
            if element_size != 4 {
                return Err(BundleError::InvalidArtifact(format!(
                    "metrics resource has {element_size} bytes per element; expected 4"
                )));
            }
            if resource.element_count < u64::from(METRIC_RECORD_WORDS) {
                return Err(BundleError::InvalidArtifact(format!(
                    "metrics resource has {} word(s); expected at least {METRIC_RECORD_WORDS}",
                    resource.element_count
                )));
            }
        }
    }
    if metrics_resources > 1 {
        return Err(BundleError::InvalidArtifact(format!(
            "target entry binds {metrics_resources} metrics resources; expected at most one"
        )));
    }
    Ok(())
}

fn validate_weight_payload_fits_first_finite_resource(
    envelope: &ArtifactEnvelope,
    payload: &TargetPayload,
    weights: &[u8],
) -> Result<(), BundleError> {
    let entry = &payload.entries()[0];
    let first_binding = entry
        .resource_bindings
        .iter()
        .min_by_key(|binding| binding.slot)
        .expect("validated non-empty target resource bindings");
    let first = envelope
        .neutral()
        .resources()
        .iter()
        .find(|resource| resource.value == first_binding.resource)
        .expect("canonical target payload validation guarantees resource association");
    if first.element_count == 0 {
        return Ok(());
    }
    let weight_bytes = u64::try_from(weights.len()).map_err(|error| {
        BundleError::InvalidArtifact(format!("weights payload length cannot fit u64: {error}"))
    })?;
    if weight_bytes > first.byte_count {
        return Err(BundleError::InvalidArtifact(format!(
            "weights payload has {weight_bytes} byte(s) but first canonical resource `{}` declares {} byte(s)",
            first.name, first.byte_count
        )));
    }
    Ok(())
}

fn target_payload(
    envelope: &ArtifactEnvelope,
    target: Target,
) -> Result<&TargetPayload, BundleError> {
    let identity = target.aot_target_id();
    let mut matches = envelope
        .target_payloads()
        .iter()
        .filter(|payload| payload.format().identity() == identity);
    let payload = matches.next().ok_or_else(|| {
        BundleError::InvalidArtifact(format!(
            "artifact envelope has no `{identity}` target payload"
        ))
    })?;
    if matches.next().is_some() {
        return Err(BundleError::InvalidArtifact(format!(
            "artifact envelope has multiple `{identity}` target payload versions"
        )));
    }
    Ok(payload)
}

fn digest_hex(digest: vyre_megakernel::Digest) -> String {
    digest
        .as_bytes()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn lzma_compress(input: &[u8]) -> Result<Vec<u8>, BundleError> {
    let mut out = Vec::with_capacity(input.len() / 2);
    let mut cursor = Cursor::new(input);
    lzma_rs::lzma_compress(&mut cursor, &mut out)
        .map_err(|e| BundleError::Lzma(format!("{e:?}")))?;
    Ok(out)
}

fn lzma_decompress(input: &[u8]) -> Result<Vec<u8>, BundleError> {
    let mut output = Vec::new();
    let mut cursor = Cursor::new(input);
    lzma_rs::lzma_decompress(&mut cursor, &mut output)
        .map_err(|error| BundleError::Lzma(format!("{error:?}")))?;
    Ok(output)
}

fn brotli_compress(input: &[u8]) -> Result<Vec<u8>, BundleError> {
    let mut out = Vec::with_capacity(input.len() / 2);
    let params = brotli::enc::BrotliEncoderParams {
        quality: 11,
        ..Default::default()
    };
    {
        let mut writer = brotli::CompressorWriter::with_params(&mut out, 4096, &params);
        writer
            .write_all(input)
            .map_err(|e| BundleError::Brotli(format!("{e}")))?;
        writer
            .flush()
            .map_err(|e| BundleError::Brotli(format!("{e}")))?;
    }
    Ok(out)
}
