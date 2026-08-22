//! Device-wide inclusive u32 scan, measured against CUB's `DeviceScan`.
//!
//! Every other baseline in this harness is a CPU library, so a GPU composition
//! was only ever compared against a host implementation of the same primitive.
//! That answers whether the device is faster than a CPU, which for a scan is
//! not in doubt, and never answers whether the composition is worth using
//! instead of the vendor's own device scan. `BaselineClass::GpuSota` existed
//! for this and nothing constructed it.
//!
//! CUB is header-only C++ and needs nvcc. The baseline is therefore compiled at
//! measurement time rather than by a build script: a build script that needs a
//! CUDA toolkit would impose one on every CPU-only build of this workspace for
//! a benchmark those builds never run. The compiled binary is cached under the
//! system temporary directory, keyed by the source digest and the target
//! architecture, so repeated samples pay nvcc once.
//!
//! Absence of nvcc on a host that reached this case is a configuration failure
//! and is reported as one. The case requires a GPU, so a host without a device
//! never selects it in the first place.

use std::path::{Path, PathBuf};
use std::process::Command;

use crate::api::case::{
    prepared_as, BaselineClass, BaselineTarget, BenchCase, BenchContext, BenchError, BenchId,
    BenchLayer, BenchMetadata, BenchRequirements, BenchRun, Correctness, DeterminismClass,
    PerformanceContract, PreparedCase, WorkloadClass,
};
use crate::api::metric::{BenchMetrics, MetricPoint};
use crate::api::suite::SuiteKind;
use vyre_foundation::ir::Program;

/// Elements scanned. One mebi-element is large enough that CUB reaches its
/// multi-pass path and small enough to stay inside any device this harness
/// runs on.
const ELEMENTS: u32 = 1 << 20;

/// Warmups and samples the CUB run is given, matching the benchmark floors the
/// vyre side is held to so the two numbers are comparable.
const WARMUPS: u32 = 300;
const SAMPLES: u32 = 30;

/// The CUB program, compiled at measurement time.
const BASELINE_SOURCE: &str = include_str!("../../baselines/cub_inclusive_scan.cu");

/// File name the source is written to before nvcc sees it.
const BASELINE_SOURCE_NAME: &str = "cub_inclusive_scan.cu";

pub struct CubScanBench;

/// This case's name in a prepared-payload diagnostic.
pub(crate) const CUB_SCAN_CASE: &str = "cases::cub_scan::CubScanBench";

/// What the CUB run reported.
struct CubMeasurement {
    /// Wrapping u32 sum over the scanned buffer.
    checksum: u32,
    /// Per-sample device milliseconds, in sample order.
    samples_ms: Vec<f64>,
    /// Device name CUB measured on.
    device: String,
    /// Compute capability CUB measured on, as `major.minor`.
    compute_capability: String,
    /// CUB version as `CUB_VERSION`.
    version: u32,
}

impl CubMeasurement {
    /// Median device nanoseconds. Median rather than mean, because a scan this
    /// short is dominated by an occasional scheduling outlier and the mean
    /// reports the outlier rather than the kernel.
    fn median_device_ns(&self) -> u64 {
        let mut ordered = self.samples_ms.clone();
        ordered.sort_by(f64::total_cmp);
        let middle = ordered[ordered.len() / 2];
        (middle * 1.0e6).max(0.0) as u64
    }

    /// The capability spelled the way nvcc spells it, `12.0` as `120`.
    fn architecture(&self) -> String {
        self.compute_capability.replace('.', "")
    }
}

impl BenchCase for CubScanBench {
    fn id(&self) -> BenchId {
        BenchId("foundation.scan.inclusive.u32.1m.cub".to_string())
    }

    fn metadata(&self) -> BenchMetadata {
        BenchMetadata {
            id: self.id(),
            name: "Inclusive u32 scan vs CUB DeviceScan".to_string(),
            description: format!(
                "Device-wide inclusive prefix sum over {ELEMENTS} u32 elements, against \
                 cub::DeviceScan::InclusiveSum on the same device"
            ),
            tags: vec![
                "scan".to_string(),
                "gpu-baseline".to_string(),
                "memory-bound".to_string(),
            ],
            layer: BenchLayer::Foundation,
            workload: WorkloadClass::Micro,
            determinism: DeterminismClass::Deterministic,
            owner_crate: "vyre-bench".to_string(),
        }
    }

    fn suites(&self) -> &'static [SuiteKind] {
        &[SuiteKind::Deep, SuiteKind::Gpu]
    }

    fn requirements(&self) -> BenchRequirements {
        BenchRequirements {
            needs_gpu: true,
            needs_network: false,
            min_vram_bytes: Some(u64::from(ELEMENTS) * 4 * 4),
            min_input_bytes: None,
            feature_set: vec![],
        }
    }

    fn performance_contract(&self) -> Option<PerformanceContract> {
        Some(PerformanceContract {
            primitive: format!("u32 inclusive scan, {ELEMENTS} elements"),
            baselines: vec![BaselineTarget {
                name: "cub::DeviceScan::InclusiveSum".to_string(),
                crate_name: "cub".to_string(),
                class: BaselineClass::GpuSota,
                // Deliberately not a speedup demand. CUB is the vendor's tuned
                // device scan and the composition is not expected to beat it;
                // the contract records that the comparison is measured, and the
                // recorded ratio is what a regression is judged against. A
                // floor above parity here would be an aspiration pinned as a
                // gate, which is how a benchmark ends up with a raised ceiling.
                min_speedup_x: 0.0,
                backend_ids: vec!["cuda".to_string()],
            }],
        })
    }

    fn prepare(&self, _ctx: &mut BenchContext) -> Result<PreparedCase, BenchError> {
        Ok(Box::new(vyre_libs::math::scan::scan_prefix_sum(
            "input", "output", ELEMENTS,
        )))
    }

    fn program<'a>(&self, prepared: &'a PreparedCase) -> Option<&'a Program> {
        prepared_as::<Program>(prepared, CUB_SCAN_CASE).ok()
    }

    fn run(
        &self,
        ctx: &mut BenchContext,
        prepared: &mut PreparedCase,
    ) -> Result<BenchRun, BenchError> {
        let program = prepared_as::<Program>(prepared, CUB_SCAN_CASE)?.clone();
        let inputs = vec![vyre_primitives::wire::pack_u32_slice(&host_input())];
        let input_bytes = inputs.iter().map(Vec::len).sum::<usize>() as u64;

        let timed = ctx
            .dispatch_timed(&program, &inputs, &ctx.dispatch_config)
            .map_err(|error| BenchError::BackendFailed(error.to_string()))?;
        let scanned_bytes = timed.outputs.last().map_or(0, Vec::len) as u64;
        let checksum = checksum_of(&timed.outputs)?;

        let cub = measure_cub()?;

        Ok(BenchRun {
            metrics: BenchMetrics {
                wall_ns: Some(timed.wall_ns),
                dispatch_ns: timed.device_ns,
                input_bytes: Some(input_bytes),
                output_bytes: Some(scanned_bytes),
                custom: vec![MetricPoint {
                    name: "elements".to_string(),
                    value: u64::from(ELEMENTS),
                }],
                ..BenchMetrics::default()
            },
            baseline_metrics: Some(BenchMetrics {
                wall_ns: Some(cub.median_device_ns()),
                dispatch_ns: Some(cub.median_device_ns()),
                input_bytes: Some(input_bytes),
                output_bytes: Some(scanned_bytes),
                custom: vec![
                    MetricPoint {
                        name: "cub_version".to_string(),
                        value: u64::from(cub.version),
                    },
                    MetricPoint {
                        name: "cub_samples".to_string(),
                        value: cub.samples_ms.len() as u64,
                    },
                ],
                ..BenchMetrics::default()
            }),
            outputs: vec![u32::to_le_bytes(checksum).to_vec()],
            baseline_outputs: Some(vec![u32::to_le_bytes(cub.checksum).to_vec()]),
        })
    }

    fn verify(&self, _ctx: &mut BenchContext, run: &BenchRun) -> Result<Correctness, BenchError> {
        // Compared as checksums rather than buffers. CUB owns its own device
        // allocation and never hands the scanned buffer back to this process,
        // so a byte comparison would need a second megabyte copied across the
        // process boundary to prove what one wrapping sum already proves.
        run.verify_exact_outputs()
    }

    fn bytes_touched(&self, _prepared: &PreparedCase) -> (u64, u64) {
        let bytes = u64::from(ELEMENTS) * 4;
        (bytes, bytes)
    }
}

/// The same deterministic input `cub_inclusive_scan.cu` builds.
fn host_input() -> Vec<u32> {
    (0..ELEMENTS).map(|index| (index % 7) + 1).collect()
}

/// Wrapping u32 sum over the SCANNED buffer.
///
/// The last buffer, not every buffer. Above the single-workgroup width the scan
/// is a fused three-pass chain, and its per-element partials and per-block
/// totals are declared live-out so the passes can hand them to each other. A
/// dispatch therefore returns several buffers and only the last is the scan.
/// Summing all of them compares vyre's intermediates against CUB's answer,
/// which reports a correctness violation for a scan that is exactly right.
fn checksum_of(outputs: &[Vec<u8>]) -> Result<u32, BenchError> {
    let scanned = outputs.last().ok_or_else(|| {
        BenchError::BackendFailed(
            "Fix: the scan dispatch returned no output buffers; the case cannot check an answer \
             that was not produced."
                .to_string(),
        )
    })?;
    Ok(scanned.chunks_exact(4).fold(0_u32, |sum, word| {
        sum.wrapping_add(u32::from_le_bytes([word[0], word[1], word[2], word[3]]))
    }))
}

/// Compile the CUB baseline if it is not already built, then run it.
fn measure_cub() -> Result<CubMeasurement, BenchError> {
    let architecture = device_architecture()?;
    let binary = compiled_baseline(&architecture)?;
    let output = Command::new(&binary)
        .arg(ELEMENTS.to_string())
        .arg(WARMUPS.to_string())
        .arg(SAMPLES.to_string())
        .output()
        .map_err(|error| {
            BenchError::BackendFailed(format!(
                "Fix: the CUB baseline at {} did not run: {error}",
                binary.display()
            ))
        })?;
    if !output.status.success() {
        return Err(BenchError::BackendFailed(format!(
            "Fix: the CUB baseline exited {}: {}",
            output.status,
            String::from_utf8_lossy(&output.stderr).trim()
        )));
    }
    let measurement = parse_measurement(&String::from_utf8_lossy(&output.stdout))?;
    // The binary is cached across runs, so the one thing that cannot be
    // assumed is that it ran on the device the cache key named. A cached
    // binary on the wrong device still prints a plausible measurement, which
    // is the whole reason the key exists; checking it is what turns the key
    // from a hope into a fact.
    if measurement.architecture() != architecture {
        return Err(BenchError::BackendFailed(format!(
            "Fix: the CUB baseline reports compute capability {} on `{}`, and this host reports \
             sm_{architecture}. Delete {} and let the baseline rebuild.",
            measurement.compute_capability,
            measurement.device,
            binary.display()
        )));
    }
    Ok(measurement)
}

/// Read the one JSON line the baseline prints.
fn parse_measurement(stdout: &str) -> Result<CubMeasurement, BenchError> {
    let document: serde_json::Value = serde_json::from_str(stdout.trim()).map_err(|error| {
        BenchError::BackendFailed(format!(
            "Fix: the CUB baseline printed something other than its measurement line: {error}"
        ))
    })?;
    let field = |name: &str| -> Result<&serde_json::Value, BenchError> {
        document.get(name).ok_or_else(|| {
            BenchError::BackendFailed(format!(
                "Fix: the CUB baseline measurement is missing `{name}`"
            ))
        })
    };
    let samples_ms = field("samples_ms")?
        .as_array()
        .ok_or_else(|| BenchError::BackendFailed("Fix: `samples_ms` is not an array".to_string()))?
        .iter()
        .map(|sample| {
            sample.as_f64().ok_or_else(|| {
                BenchError::BackendFailed("Fix: a `samples_ms` entry is not a number".to_string())
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    if samples_ms.is_empty() {
        return Err(BenchError::BackendFailed(
            "Fix: the CUB baseline recorded no samples".to_string(),
        ));
    }
    Ok(CubMeasurement {
        checksum: u32::try_from(field("checksum")?.as_u64().unwrap_or(u64::MAX))
            .map_err(|_| BenchError::BackendFailed("Fix: `checksum` is not a u32".to_string()))?,
        samples_ms,
        device: field("device")?.as_str().unwrap_or_default().to_string(),
        compute_capability: field("compute_capability")?
            .as_str()
            .unwrap_or_default()
            .to_string(),
        version: u32::try_from(field("cub_version")?.as_u64().unwrap_or(0)).unwrap_or(0),
    })
}

/// Path to the compiled baseline, compiling it on first use.
///
/// Keyed by the source digest and the architecture, so editing the `.cu` or
/// moving to another device rebuilds instead of running a stale binary. That is
/// the same failure mode a shared cargo target directory produces, and it is
/// worse here because the stale binary still prints a plausible measurement.
fn compiled_baseline(architecture: &str) -> Result<PathBuf, BenchError> {
    let digest = blake3::hash(BASELINE_SOURCE.as_bytes()).to_hex();
    let directory = std::env::temp_dir().join(format!(
        "vyre-cub-baseline-{}-{architecture}",
        &digest[..16]
    ));
    let binary = directory.join("cub_inclusive_scan");
    if binary.exists() {
        return Ok(binary);
    }
    std::fs::create_dir_all(&directory).map_err(|error| {
        BenchError::BackendFailed(format!(
            "Fix: cannot create {} for the CUB baseline: {error}",
            directory.display()
        ))
    })?;
    let source = directory.join(BASELINE_SOURCE_NAME);
    std::fs::write(&source, BASELINE_SOURCE).map_err(|error| {
        BenchError::BackendFailed(format!("Fix: cannot write {} : {error}", source.display()))
    })?;
    compile(&source, &binary, architecture)?;
    Ok(binary)
}

/// Run nvcc over the baseline source.
fn compile(source: &Path, binary: &Path, architecture: &str) -> Result<(), BenchError> {
    let output = Command::new("nvcc")
        .arg("-O3")
        .arg("-std=c++17")
        .arg(format!("-arch=sm_{architecture}"))
        .arg("-o")
        .arg(binary)
        .arg(source)
        .output()
        .map_err(|error| {
            BenchError::BackendFailed(format!(
                "Fix: nvcc is required to measure the CUB baseline and did not run: {error}. \
                 Install the CUDA toolkit on this runner; a missing toolkit is a configuration \
                 failure, not a reason to report a scan with no baseline."
            ))
        })?;
    if !output.status.success() {
        return Err(BenchError::BackendFailed(format!(
            "Fix: nvcc failed to build the CUB baseline: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        )));
    }
    Ok(())
}

/// Compute capability of the measured device, as nvcc spells it.
///
/// Taken from the driver rather than hardcoded: a binary built for another
/// architecture still runs through JIT and reports a number that reads as a
/// measurement of this device while being a measurement of a translated
/// kernel.
fn device_architecture() -> Result<String, BenchError> {
    let output = Command::new("nvidia-smi")
        .args(["--query-gpu=compute_cap", "--format=csv,noheader"])
        .output()
        .map_err(|error| {
            BenchError::BackendFailed(format!(
                "Fix: cannot read the device compute capability for the CUB baseline: {error}"
            ))
        })?;
    let text = String::from_utf8_lossy(&output.stdout);
    let capability = text.lines().next().unwrap_or_default().trim();
    let (major, minor) = capability.split_once('.').ok_or_else(|| {
        BenchError::BackendFailed(format!(
            "Fix: the device reported compute capability `{capability}`, which is not `major.minor`"
        ))
    })?;
    Ok(format!("{major}{minor}"))
}

inventory::submit! {
    &CubScanBench as &'static dyn BenchCase
}

#[cfg(test)]
mod tests {
    use super::*;

    /// WHY: the host fill and the `.cu` fill must be the same sequence or the
    /// checksums differ for a reason that has nothing to do with either scan.
    /// The `.cu` computes `(index % 7) + 1`, and this pins the Rust side to it.
    #[test]
    fn the_host_input_matches_the_sequence_the_baseline_builds() {
        let input = host_input();
        assert_eq!(input.len(), ELEMENTS as usize);
        assert_eq!(&input[..8], &[1, 2, 3, 4, 5, 6, 7, 1]);
        assert!(BASELINE_SOURCE.contains("(index % 7u) + 1u"));
    }

    /// WHY: the baseline is a subprocess, so its output is parsed rather than
    /// returned. A parser that accepts a truncated line reports whatever
    /// happened to be on stdout as a measurement.
    #[test]
    fn a_measurement_line_parses_into_samples_and_a_checksum() {
        let measurement = parse_measurement(
            r#"{"elements":4,"checksum":10,"device":"d","compute_capability":"12.0",
                "cub_version":200700,"samples_ms":[0.5,0.25,0.75]}"#,
        )
        .expect("the line parses");
        assert_eq!(measurement.checksum, 10);
        assert_eq!(measurement.version, 200_700);
        assert_eq!(measurement.device, "d");
        assert_eq!(measurement.compute_capability, "12.0");
        assert_eq!(measurement.architecture(), "120");
        assert_eq!(measurement.median_device_ns(), 500_000);
    }

    /// WHY: every one of these was a way for a missing measurement to read as a
    /// zero-cost baseline, which would report an infinite speedup.
    #[test]
    fn an_incomplete_measurement_line_is_rejected() {
        for line in [
            "",
            "not json",
            r#"{"checksum":1,"device":"d","compute_capability":"1.0","cub_version":1}"#,
            r#"{"checksum":1,"device":"d","compute_capability":"1.0","cub_version":1,"samples_ms":[]}"#,
            r#"{"checksum":1,"device":"d","compute_capability":"1.0","cub_version":1,"samples_ms":["x"]}"#,
            r#"{"checksum":1,"device":"d","cub_version":1,"samples_ms":[0.5]}"#,
        ] {
            assert!(parse_measurement(line).is_err(), "accepted `{line}`");
        }
    }

    /// WHY: the median must be the middle sample, not the mean and not the
    /// first. A scan this short is dominated by occasional scheduling outliers,
    /// and the first sample is reliably one of them.
    #[test]
    fn the_reported_time_is_the_median_sample() {
        let measurement = CubMeasurement {
            checksum: 0,
            samples_ms: vec![9.0, 0.001, 0.002, 0.003, 0.004],
            device: String::new(),
            compute_capability: "12.0".to_string(),
            version: 0,
        };
        assert_eq!(measurement.median_device_ns(), 3_000);
    }

    /// WHY: two separate ways this checksum reported a correct scan as wrong.
    /// It must wrap the way the `.cu` wraps rather than saturate or panic in
    /// debug, and it must read only the scanned buffer: the fused multi-block
    /// chain declares its partials and block totals live-out, so a dispatch
    /// hands back intermediates alongside the answer and summing all of them
    /// compares vyre's scratch against CUB's result.
    #[test]
    fn the_checksum_wraps_and_reads_only_the_scanned_buffer() {
        let intermediate = vec![7_u32, 9, 11]
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .collect::<Vec<u8>>();
        let scanned = [u32::MAX, 2]
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .collect::<Vec<u8>>();
        assert_eq!(
            checksum_of(&[intermediate, scanned]).expect("outputs are present"),
            1,
            "Fix: the checksum must wrap over the last buffer alone; folding the live-out \
             intermediates in reports a correct scan as a correctness violation."
        );
    }

    /// WHY: no buffers means nothing was produced. Folding an empty iterator
    /// yields 0, which is a plausible checksum and would make an unproduced
    /// answer read as a wrong one instead of a missing one.
    #[test]
    fn a_dispatch_that_produced_nothing_is_an_error_not_a_zero_checksum() {
        assert!(checksum_of(&[]).is_err());
    }
}
