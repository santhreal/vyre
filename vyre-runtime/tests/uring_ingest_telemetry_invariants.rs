//! Public invariant tests for NVMe to GPU ingest telemetry.

use vyre_runtime::uring::{NativeReadPath, NvmeGpuIngestTelemetry};

fn completed_registered() -> NvmeGpuIngestTelemetry {
    NvmeGpuIngestTelemetry {
        submitted_bytes: 4096,
        completed_bytes: 4096,
        submitted_reads: 4,
        completed_reads: 4,
        registered_mapped_read_submissions: 4,
        gpudirect_nvme_submissions: 0,
        cpu_bounce_bytes: 0,
        failed_completions: 0,
    }
}

fn completed_gpudirect() -> NvmeGpuIngestTelemetry {
    NvmeGpuIngestTelemetry {
        submitted_bytes: 8192,
        completed_bytes: 8192,
        submitted_reads: 2,
        completed_reads: 2,
        registered_mapped_read_submissions: 0,
        gpudirect_nvme_submissions: 2,
        cpu_bounce_bytes: 0,
        failed_completions: 0,
    }
}

#[test]
fn completed_zero_copy_validation_accepts_each_native_path() {
    completed_registered()
        .validate_completed_zero_copy(NativeReadPath::RegisteredMappedRead)
        .expect("Fix: registered mapped read telemetry should validate.");
    completed_gpudirect()
        .validate_completed_zero_copy(NativeReadPath::GpuDirectNvmePassthrough)
        .expect("Fix: GPUDirect NVMe telemetry should validate.");
}

#[test]
fn completed_zero_copy_validation_rejects_cpu_bounce_bytes() {
    let mut telemetry = completed_registered();
    telemetry.cpu_bounce_bytes = 1;
    assert!(
        telemetry
            .validate_completed_zero_copy(NativeReadPath::RegisteredMappedRead)
            .is_err(),
        "Fix: zero-copy telemetry must reject any CPU bounce byte."
    );
}

#[test]
fn completed_zero_copy_validation_rejects_failed_or_inflight_reads() {
    let mut failed = completed_registered();
    failed.failed_completions = 1;
    assert!(
        failed
            .validate_completed_zero_copy(NativeReadPath::RegisteredMappedRead)
            .is_err(),
        "Fix: completed telemetry must reject failed completions."
    );

    let mut inflight = completed_registered();
    inflight.completed_reads = 3;
    assert!(
        inflight
            .validate_completed_zero_copy(NativeReadPath::RegisteredMappedRead)
            .is_err(),
        "Fix: completed telemetry must reject inflight reads."
    );
}

#[test]
fn completed_zero_copy_validation_rejects_byte_and_read_mismatch() {
    let mut bytes = completed_registered();
    bytes.completed_bytes = 2048;
    assert!(
        bytes
            .validate_completed_zero_copy(NativeReadPath::RegisteredMappedRead)
            .is_err(),
        "Fix: completed telemetry must reject partial byte accounting."
    );

    let mut reads = completed_registered();
    reads.completed_reads = 5;
    assert!(
        reads
            .validate_completed_zero_copy(NativeReadPath::RegisteredMappedRead)
            .is_err(),
        "Fix: completed telemetry must reject impossible read accounting."
    );
}

#[test]
fn completed_zero_copy_validation_rejects_path_mixing() {
    let mut registered = completed_registered();
    registered.gpudirect_nvme_submissions = 1;
    assert!(
        registered
            .validate_completed_zero_copy(NativeReadPath::RegisteredMappedRead)
            .is_err(),
        "Fix: registered mapped telemetry must reject mixed GPUDirect submissions."
    );

    let mut gpudirect = completed_gpudirect();
    gpudirect.registered_mapped_read_submissions = 1;
    assert!(
        gpudirect
            .validate_completed_zero_copy(NativeReadPath::GpuDirectNvmePassthrough)
            .is_err(),
        "Fix: GPUDirect telemetry must reject mixed registered mapped submissions."
    );
}
