//! Rust launcher source generator.
//!
//! `vyre-aot` owns the target-neutral package shell. Concrete driver crates
//! own every target-specific launcher file through `vyre-driver`'s AOT
//! launcher registry.

use std::collections::BTreeMap;
use std::path::PathBuf;

use thiserror::Error;
use vyre_driver::{AotLauncherRequest, LauncherDependency};

use crate::artifact::{registration, TargetId};
use vyre_megakernel::ArtifactEnvelope;

/// Options controlling launcher emission.
#[derive(Debug, Clone)]
pub struct LauncherOpts {
    /// Crate name (becomes the binary name on disk).
    pub crate_name: String,
    /// Whether to enable target-owned multi-rank collective support.
    pub include_collectives: bool,
    /// Whether to enable a built-in TTT loop (eval-time SGD over chunks).
    /// If `false`, eval is delegated to a separate launcher invocation.
    pub include_ttt_loop: bool,
}

impl Default for LauncherOpts {
    fn default() -> Self {
        Self {
            crate_name: "pgolf-launcher".to_string(),
            include_collectives: true,
            include_ttt_loop: false,
        }
    }
}

/// Error variants returned by [`emit_launcher_rust`].
#[derive(Debug, Error)]
pub enum LauncherError {
    /// No linked driver owns launcher generation for the artifact target.
    #[error("vyre-aot launcher: target {0} has no linked launcher emitter")]
    TargetNotEnabled(TargetId),

    /// The target-owned launcher emitter rejected the request.
    #[error("vyre-aot launcher: target emitter failed: {0}")]
    Backend(String),

    /// The envelope does not carry exactly one payload for the selected target.
    #[error("vyre-aot launcher: invalid artifact envelope: {0}")]
    InvalidArtifact(String),
}

/// Emit the Rust launcher source tree.
///
/// Returns a map from relative file path to file contents. The caller writes
/// each entry to disk under the launcher crate root.
///
/// # Errors
///
/// Returns [`LauncherError::TargetNotEnabled`] when the artifact target has
/// no linked launcher emitter, or [`LauncherError::Backend`] when that emitter
/// rejects the request.
pub fn emit_launcher_rust(
    envelope: &ArtifactEnvelope,
    selected_target: TargetId,
    opts: &LauncherOpts,
) -> Result<BTreeMap<PathBuf, String>, LauncherError> {
    let registration = registration(&selected_target)
        .map_err(|_| LauncherError::TargetNotEnabled(selected_target.clone()))?;
    let target = registration
        .payload_format
        .ok_or_else(|| LauncherError::TargetNotEnabled(selected_target.clone()))?;
    let matching_payloads = envelope
        .target_payloads()
        .iter()
        .filter(|payload| payload.format().identity() == target)
        .count();
    if matching_payloads != 1 {
        return Err(LauncherError::InvalidArtifact(format!(
            "expected one `{target}` payload, found {matching_payloads}"
        )));
    }
    let request = AotLauncherRequest {
        target: selected_target.clone(),
        crate_name: &opts.crate_name,
        include_collectives: opts.include_collectives,
        include_ttt_loop: opts.include_ttt_loop,
    };

    let target_files =
        vyre_driver::emit_aot_launcher_target(&selected_target, &request).map_err(|error| {
            match error {
                vyre_driver::BackendError::UnsupportedFeature { .. } => {
                    LauncherError::TargetNotEnabled(selected_target.clone())
                }
                other => LauncherError::Backend(other.to_string()),
            }
        })?;

    let mut tree = target_files.files;
    tree.insert(
        PathBuf::from("Cargo.toml"),
        emit_launcher_cargo_toml(opts, &target_files.dependencies),
    );
    tree.insert(
        PathBuf::from(".cargo/config.toml"),
        EMIT_CARGO_BUILD_CONFIG.to_string(),
    );
    tree.insert(
        PathBuf::from("src/artifact.rs"),
        EMIT_ARTIFACT_LOADER.to_string(),
    );
    tree.insert(PathBuf::from("README.md"), emit_launcher_readme(opts));

    Ok(tree)
}

fn emit_launcher_cargo_toml(opts: &LauncherOpts, deps: &[LauncherDependency]) -> String {
    let mut dependencies = String::from(
        r#"[dependencies]
serde = { version = "1", features = ["derive"] }
serde_json = "1"
brotli = "8"
lzma-rs = "0.3"
sha2 = "0.10"
"#,
    );
    dependencies.push_str("vyre-megakernel = \"=");
    dependencies.push_str(env!("CARGO_PKG_VERSION"));
    dependencies.push_str("\"\n");
    for dep in deps {
        dependencies.push_str(dep.name);
        dependencies.push_str(" = ");
        dependencies.push_str(dep.spec);
        dependencies.push('\n');
    }

    format!(
        r#"[package]
name = "{name}"
version = "0.1.0"
edition = "2021"
publish = false

{dependencies}
[profile.release]
opt-level = "z"
lto = "fat"
codegen-units = 1
panic = "abort"
strip = "symbols"
overflow-checks = false
debug-assertions = false
incremental = false

[[bin]]
name = "{name}"
path = "src/main.rs"
"#,
        name = opts.crate_name,
        dependencies = dependencies,
    )
}

const EMIT_CARGO_BUILD_CONFIG: &str = r#"# Size-optimized launcher build flags.
[build]
rustflags = ["-C", "relocation-model=pic", "-C", "embed-bitcode=yes"]
"#;

const EMIT_ARTIFACT_LOADER: &str = include_str!("../templates/artifact.rs.tmpl");

fn emit_launcher_readme(opts: &LauncherOpts) -> String {
    let collective_status = if opts.include_collectives {
        "enabled"
    } else {
        "disabled"
    };
    format!(
        r#"# {name}

Auto-generated by `vyre-aot`. Self-contained target launcher for a vyre-emitted
persistent kernel.

## Build

```
./cargo_full build --release
```

The release binary is heavily size-optimized:
`opt-level=z`, `lto=fat`, `codegen-units=1`, `panic=abort`, `strip=symbols`.

## Run

```
{name} <bundle_dir>
```

Reads `manifest.json`, the authenticated compiler envelope, and
`weights.brotli` from the bundle directory. Projects target bytes and ABI
bindings from the envelope before allocation and submission.

Target-owned collective support is {collective_status} in this launcher.
"#,
        name = opts.crate_name,
        collective_status = collective_status,
    )
}

// Inline: covers the private `emit_launcher_cargo_toml`, which no integration test can reach.
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn emitted_launcher_declares_sha2_for_runtime_integrity_checks() {
        let opts = LauncherOpts::default();
        let cargo = emit_launcher_cargo_toml(&opts, &[]);

        assert!(
            cargo.contains("sha2 = \"0.10\""),
            "Fix: generated launchers must include sha2 so runtime artifact integrity checks compile."
        );
        assert!(cargo.contains(&format!(
            "vyre-megakernel = \"={}\"",
            env!("CARGO_PKG_VERSION")
        )));
    }
}
