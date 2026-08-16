//! Backend parity suites, and the host each one needs to prove anything.
//!
//! Both suites judge a backend against the reference, and both used to be
//! reachable only through a shell script that a workflow called by path. A
//! suite reached that way is registered nowhere: it has no baseline, the sweep
//! cannot see it, and the assertions the script carried live in bash.
//!
//! Each gate has a cheap half the sweep runs everywhere, and an expensive half
//! a workflow runs on the host that has the hardware or the validator. The
//! cheap half is the part that rots: a renamed target, a suite no longer behind
//! its feature, a crate whose parity target was deleted. The expensive half
//! fails closed when the host cannot prove anything, because reporting a clean
//! backend on a machine with no device is the defect these gates exist to
//! prevent.

use std::collections::BTreeSet;
use std::path::Path;
use std::process::Command;

use crate::gate::{Finding, Gate, GateCtx, GateError, Report};
use crate::gates::scan::Tree;

/// Crate whose emitted SPIR-V is validated.
const SPIRV_CRATE: &str = "vyre-driver-spirv";

/// Feature the validated target is registered behind.
const SPIRV_FEATURE: &str = "spirv-val";

/// Validator binary, and the fragment a test source names it by.
const SPIRV_VALIDATOR: &str = "spirv-val";

/// Crate whose parity is proved against a live device.
const CUDA_CRATE: &str = "vyre-driver-cuda";

/// Target-name fragment that marks a reference-parity target.
const PARITY: &str = "gpu_parity";

/// Emitted SPIR-V is validated, not just shaped.
pub struct SpirvParity;

impl Gate for SpirvParity {
    fn name(&self) -> &'static str {
        "spirv-parity"
    }

    fn help(&self) -> &'static str {
        "Hold the SPIR-V parity suite behind the validator feature; --validate installs nothing and runs it"
    }

    fn run(&self, ctx: &GateCtx) -> Result<Report, GateError> {
        let tree = Tree::open(&ctx.root)?;
        let mut report = Report::clean();
        let manifest_path = format!("{SPIRV_CRATE}/Cargo.toml");
        let manifest = tree.read_toml(&manifest_path)?;

        let declares_feature = manifest
            .get("features")
            .and_then(toml::Value::as_table)
            .is_some_and(|table| table.contains_key(SPIRV_FEATURE));
        if !declares_feature {
            report.find(Finding::in_file(
                manifest_path.clone(),
                format!("`{SPIRV_CRATE}` declares no `{SPIRV_FEATURE}` feature, so the validated target cannot be registered behind one"),
                format!("declare the `{SPIRV_FEATURE}` feature; a default test run has to skip the target rather than compile it and pass without validating"),
            ));
        }

        let gated = gated_targets(&manifest, SPIRV_FEATURE);
        for path in tree.paths() {
            let Some(target) = test_target(path, SPIRV_CRATE) else {
                continue;
            };
            let text = tree.read(path)?;
            if !text.contains(SPIRV_VALIDATOR) {
                continue;
            }
            if !gated.contains(&target) {
                report.find(Finding::in_file(
                    path.clone(),
                    format!(
                        "`{target}` validates with `{SPIRV_VALIDATOR}` and is not registered behind `required-features = [\"{SPIRV_FEATURE}\"]`, so a host without the validator runs it and decides for itself what to do"
                    ),
                    format!("register the target with required-features = [\"{SPIRV_FEATURE}\"], so the run that cannot validate does not happen at all"),
                ));
            }
        }
        if gated.is_empty() {
            report.find(Finding::in_file(
                manifest_path,
                format!("no [[test]] entry requires `{SPIRV_FEATURE}`, so the validated suite runs nowhere"),
                "register the parity target behind the validator feature",
            ));
        }

        if !ctx.has("--validate") {
            report.note(format!(
                "{} validated target(s) behind `{SPIRV_FEATURE}`",
                gated.len()
            ));
            return Ok(report);
        }

        let version = tool_version(SPIRV_VALIDATOR, &["--version"]).map_err(|reason| {
            GateError::new(
                reason,
                "install spirv-tools (Debian/Ubuntu: apt-get install -y spirv-tools; macOS: brew install spirv-tools); an unvalidated blob with a correct header is what this gate catches",
            )
        })?;
        report.note(format!("validator: {version}"));
        for target in &gated {
            let failed = !cargo_test(
                &ctx.root,
                SPIRV_CRATE,
                &["--features", SPIRV_FEATURE, "--test", target],
            )?;
            if failed {
                report.find(Finding::new(
                    format!("`{target}` failed against {SPIRV_VALIDATOR}"),
                    "fix the emission the validator rejected",
                ));
            }
        }
        Ok(report)
    }
}

/// CUDA parity is proved on a live device, or not at all.
pub struct CudaParity;

impl Gate for CudaParity {
    fn name(&self) -> &'static str {
        "cuda-parity"
    }

    fn help(&self) -> &'static str {
        "Hold the CUDA driver to tracked parity targets; --device runs the crate's suite on a live NVIDIA GPU"
    }

    fn run(&self, ctx: &GateCtx) -> Result<Report, GateError> {
        let tree = Tree::open(&ctx.root)?;
        let mut report = Report::clean();
        let targets: BTreeSet<String> = tree
            .paths()
            .iter()
            .filter_map(|path| test_target(path, CUDA_CRATE))
            .collect();

        if targets.is_empty() {
            report.find(Finding::new(
                format!("no tracked test target under `{CUDA_CRATE}/tests`, so the device job would report a clean backend with nothing to run"),
                "restore the parity suite; a device gate that runs nothing is the defect it guards",
            ));
            return Ok(report);
        }
        let parity = targets
            .iter()
            .filter(|target| target.contains(PARITY))
            .count();
        if parity == 0 {
            report.find(Finding::new(
                format!(
                    "none of the {} tracked `{CUDA_CRATE}` test targets is a `{PARITY}` target",
                    targets.len()
                ),
                "reference parity against the live device is the evidence this gate exists to produce",
            ));
        }

        if !ctx.has("--device") {
            report.note(format!(
                "{} tracked test target(s), {parity} of them {PARITY}",
                targets.len()
            ));
            return Ok(report);
        }

        let device = tool_version(
            "nvidia-smi",
            &["--query-gpu=name,driver_version", "--format=csv,noheader"],
        )
        .map_err(|reason| {
            GateError::new(
                reason,
                "repair CUDA driver visibility on this host; CUDA parity is not skipped on a fleet that has the device",
            )
        })?;
        report.note(format!("device: {device}"));
        // The crate's documented test command, so this gate and the testing
        // guide cannot disagree about what proving the backend means.
        if !cargo_test(&ctx.root, CUDA_CRATE, &[])? {
            report.find(Finding::new(
                format!(
                    "the {} tracked `{CUDA_CRATE}` test target(s) did not all pass on the live device",
                    targets.len()
                ),
                "fix the parity failure the suite reported",
            ));
        }
        Ok(report)
    }
}

/// The test target a tracked path names, for one crate's own `tests/`.
///
/// A nested support module is not a target, and a stray source elsewhere in the
/// crate is not one either.
fn test_target(path: &Path, crate_dir: &str) -> Option<String> {
    let text = path.to_str()?;
    let rest = text.strip_prefix(crate_dir)?.strip_prefix("/tests/")?;
    let name = rest.strip_suffix(".rs")?;
    (!name.contains('/')).then(|| name.to_string())
}

/// Every `[[test]]` target the manifest registers behind `feature`.
fn gated_targets(manifest: &toml::Table, feature: &str) -> BTreeSet<String> {
    let Some(entries) = manifest.get("test").and_then(toml::Value::as_array) else {
        return BTreeSet::new();
    };
    entries
        .iter()
        .filter(|entry| {
            entry
                .get("required-features")
                .and_then(toml::Value::as_array)
                .is_some_and(|features| {
                    features.iter().any(|value| value.as_str() == Some(feature))
                })
        })
        .filter_map(|entry| {
            entry
                .get("name")
                .and_then(toml::Value::as_str)
                .map(str::to_string)
        })
        .collect()
}

/// What a required host tool reports about itself, or why it cannot be used.
fn tool_version(tool: &str, args: &[&str]) -> Result<String, String> {
    let output = Command::new(tool)
        .args(args)
        .output()
        .map_err(|error| format!("`{tool}` is required and could not be run: {error}"))?;
    if !output.status.success() {
        return Err(format!(
            "`{tool}` ran and reported nothing usable: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    Ok(String::from_utf8_lossy(&output.stdout)
        .lines()
        .next()
        .unwrap_or_default()
        .trim()
        .to_string())
}

/// Run one crate's tests, answering whether they passed.
fn cargo_test(root: &Path, package: &str, extra: &[&str]) -> Result<bool, GateError> {
    let status = crate::cargo_runner::command(root)
        .args(["test", "-p", package])
        .args(extra)
        .args(["--", "--nocapture"])
        .status()
        .map_err(|error| {
            GateError::new(
                format!("cannot run cargo test for `{package}`: {error}"),
                "install a cargo the gate can start, or set CARGO to one",
            )
        })?;
    Ok(status.success())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// WHY: the historical defect is a suite that runs without its validator
    /// and decides for itself to pass. The registration behind the feature is
    /// what stops the run, so reading it wrong reopens the class. The reader is
    /// crate-private, so no integration test reaches it.
    #[test]
    fn only_a_target_requiring_the_feature_counts_as_gated() {
        let manifest: toml::Table = toml::from_str(
            "[[test]]\nname = \"spirv_parity\"\nrequired-features = [\"spirv-val\"]\n\n[[test]]\nname = \"dispatch\"\n\n[[test]]\nname = \"other\"\nrequired-features = [\"unrelated\"]\n",
        )
        .expect("table");
        let gated = gated_targets(&manifest, "spirv-val");
        assert_eq!(gated.len(), 1);
        assert!(gated.contains("spirv_parity"));
    }

    /// WHY: a support module under `tests/` is not a cargo target, and passing
    /// one to `--test` fails the run for a reason that has nothing to do with
    /// the backend.
    #[test]
    fn a_support_module_is_not_a_test_target() {
        assert_eq!(
            test_target(
                Path::new("vyre-driver-cuda/tests/int4_quantized_gpu_parity.rs"),
                "vyre-driver-cuda"
            ),
            Some("int4_quantized_gpu_parity".to_string())
        );
        assert_eq!(
            test_target(
                Path::new("vyre-driver-cuda/tests/support/mod.rs"),
                "vyre-driver-cuda"
            ),
            None
        );
        assert_eq!(
            test_target(
                Path::new("vyre-driver-spirv/src/lib.rs"),
                "vyre-driver-spirv"
            ),
            None
        );
    }
}
