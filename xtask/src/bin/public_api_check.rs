//! Run `cargo_full public-api` against every workspace crate and diff the
//! result against the committed snapshots under `xtask/public_api/`.
//!
//! Run via `cargo_full run -p xtask --bin public_api_check`. The binary
//! exits non-zero when any crate's public-API surface drifts from its
//! frozen snapshot, which is the publish-floor invariant.

use std::fs;
use std::io::{self, Read};
use std::path::Path;
use std::process::Command;

const MAX_PUBLIC_API_CHECK_TEXT_BYTES: u64 = 4_194_304;

const CRATES: &[&str] = &[
    "vyre-spec",
    "vyre-foundation",
    "vyre-primitives",
    "vyre-libs",
    "vyre-driver",
    "vyre-driver-wgpu",
    "vyre-driver-spirv",
    "vyre-runtime",
    "vyre-frontend-c",
    "vyre-intrinsics",
    "vyre-reference",
    "security-analysis-consumer",
    "surge",
    "surge-source",
];

fn main() {
    let mut args = std::env::args().skip(1);
    let is_update = args.next().as_deref() == Some("--update");

    let mut failed = false;

    let root = Path::new("../../..");
    let cargo_runner = std::env::var("VYRE_CARGO_RUNNER").unwrap_or_else(|_| "cargo_full".into());

    for crate_name in CRATES {
        let output = match Command::new(&cargo_runner)
            .arg("public-api")
            .arg("-p")
            .arg(crate_name)
            .output()
        {
            Ok(output) => output,
            Err(error) => {
                eprintln!(
                    "Fix: failed to execute `{cargo_runner} public-api -p {crate_name}`: {error}"
                );
                failed = true;
                continue;
            }
        };

        if !output.status.success() {
            eprintln!(
                "Failed to generate public API for {}: {}",
                crate_name,
                String::from_utf8_lossy(&output.stderr)
            );
            failed = true;
            continue;
        }

        let new_api = match String::from_utf8(output.stdout) {
            Ok(api) => api,
            Err(error) => {
                eprintln!("Fix: public API output for {crate_name} was not UTF-8: {error}");
                failed = true;
                continue;
            }
        };

        let md_path = match find_crate_dir(crate_name, root) {
            Ok(Some(p)) => p.join("PUBLIC_API.md"),
            Ok(None) => {
                eprintln!("Could not find dir for crate {}", crate_name);
                failed = true;
                continue;
            }
            Err(error) => {
                eprintln!("Fix: failed while locating crate {crate_name}: {error}");
                failed = true;
                continue;
            }
        };

        if is_update {
            if let Err(error) = fs::write(&md_path, new_api) {
                eprintln!("Fix: failed to write `{}`: {error}", md_path.display());
                failed = true;
                continue;
            }
            println!("Updated {}", md_path.display());
        } else {
            let old_api = match read_text_bounded(&md_path) {
                Ok(api) => api,
                Err(error) => {
                    eprintln!(
                        "Fix: failed to read public API snapshot `{}`: {error}",
                        md_path.display()
                    );
                    failed = true;
                    continue;
                }
            };
            if new_api != old_api {
                eprintln!("Public API drifted for crate {}. Fix: run `cargo_full run --bin xtask -- public-api-update` to regenerate.", crate_name);
                failed = true;
            } else {
                println!("{} API matches snapshot.", crate_name);
            }
        }
    }

    if failed && !is_update {
        std::process::exit(1);
    }
}

fn find_crate_dir(name: &str, root: &Path) -> Result<Option<std::path::PathBuf>, String> {
    for entry in walkdir::WalkDir::new(root)
        .into_iter()
        .filter_map(|e| e.ok())
    {
        if entry.path().components().any(|c| c.as_os_str() == "target") {
            continue;
        }
        if entry.file_name() == "Cargo.toml" {
            let content = read_text_bounded(entry.path())
                .map_err(|error| format!("{}: {error}", entry.path().display()))?;
            if content.contains(&format!("name = \"{}\"", name)) {
                return Ok(entry.path().parent().map(Path::to_path_buf));
            }
        }
    }
    Ok(None)
}

fn read_text_bounded(path: &Path) -> io::Result<String> {
    let mut reader = fs::File::open(path)?.take(MAX_PUBLIC_API_CHECK_TEXT_BYTES.saturating_add(1));
    let mut text = String::new();
    reader.read_to_string(&mut text)?;
    if text.len() as u64 > MAX_PUBLIC_API_CHECK_TEXT_BYTES {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "{} exceeds {MAX_PUBLIC_API_CHECK_TEXT_BYTES} byte public API check read cap",
                path.display()
            ),
        ));
    }
    Ok(text)
}
