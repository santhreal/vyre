//! `VYRE_CUDA_DISPATCH_PHASE_PROBE` emits one consistent phase record per
//! timed dispatch.
//!
//! The probe reads its switch once per process, so the parent test runs a
//! helper test of this binary in a child process with the switch set and parses
//! the child's stderr. Both timed entry points that emit a record are driven:
//! `CudaBackend::dispatch_resident_timed`, which lowers and keys the program
//! per call, and the compiled pipeline's resident entry point, which reuses
//! what it compiled.
//!
//! Both are driven on the registered device generation, whose timed-dispatch
//! counter is the record's `seq`, so all records form one sequence. Each record
//! is checked for the exact field set, a strictly increasing sequence, counts
//! that match the dispatched program, a nonzero kernel window, a named total
//! equal to the sum of its phases, and a residue equal to the
//! wall less that total. Not covered here: whether a phase's duration is
//! charged to the right phase, which only a timing comparison could show.

#![cfg(feature = "device-tests")]

use std::collections::BTreeMap;
use std::process::Command;

use crate::harness::{add_one_program, resident_compiled_cuda_outputs};
use vyre_driver::DispatchConfig;

const HELPER_FLAG: &str = "VYRE_CUDA_PHASE_PROBE_HELPER";
const RECORD: &str = "vyre-cuda-dispatch-phase";
const DIRECT_DISPATCHES: usize = 3;
const COMPILED_DISPATCHES: usize = 2;

const PHASES: [&str; 9] = [
    "prepare_ns",
    "ptx_ns",
    "modkey_ns",
    "stage_ns",
    "resolve_ns",
    "lease_ns",
    "launch_ns",
    "release_ns",
    "readback_ns",
];

const FIELDS: [&str; 29] = [
    "seq",
    "nodes",
    "ptx_bytes",
    "buffers",
    "bindings",
    "fixpoint",
    "grid_blocks",
    "wall_ns",
    "enqueue_ns",
    "wait_ns",
    "device_window_ns",
    "kernel_ns",
    "prepare_ns",
    "ptx_ns",
    "modkey_ns",
    "stage_ns",
    "resolve_ns",
    "lease_ns",
    "launch_ns",
    "release_ns",
    "readback_ns",
    "sub_ptx_digest_ns",
    "sub_ptx_vsa_ns",
    "sub_barrier_reset_ns",
    "sub_launch_call_ns",
    "named_host_ns",
    "unattributed_ns",
    "ptx_cache_hits",
    "ptx_cache_misses",
];

fn expected_fields() -> Vec<&'static str> {
    let mut fields = FIELDS.to_vec();
    fields.sort_unstable();
    fields
}

fn input_bytes() -> Vec<u8> {
    (0..8u32)
        .flat_map(|value| (value * 7).to_le_bytes())
        .collect()
}

/// libtest names a test by its module path inside the `all_tests` binary,
/// without the crate segment `module_path!()` carries.
fn helper_test_filter() -> String {
    let module = module_path!();
    let module = module.split_once("::").map_or(module, |(_, rest)| rest);
    format!("{module}::dispatch_phase_probe_helper_process")
}

#[test]
fn dispatch_phase_probe_helper_process() {
    if std::env::var_os(HELPER_FLAG).is_none() {
        return;
    }
    let program = add_one_program();
    let input = input_bytes();
    let backend = vyre_driver_cuda::registered_device()
        .expect("Fix: the registered CUDA device must be acquirable");
    let handles = [
        backend
            .allocate_resident(input.len())
            .expect("Fix: input allocation"),
        backend
            .allocate_resident(input.len())
            .expect("Fix: output allocation"),
    ];
    backend
        .upload_resident(handles[0], &input)
        .expect("Fix: input upload");
    for _ in 0..DIRECT_DISPATCHES {
        backend
            .dispatch_resident_timed(&program, &handles, &DispatchConfig::default())
            .expect("Fix: direct resident timed dispatch must succeed");
    }
    for handle in handles {
        backend.free_resident(handle).expect("Fix: resident free");
    }
    for _ in 0..COMPILED_DISPATCHES {
        resident_compiled_cuda_outputs(&program, &[input.clone()], "dispatch_phase_probe");
    }
}

fn records(stderr: &str) -> Vec<BTreeMap<String, u64>> {
    stderr
        .lines()
        .filter_map(|line| line.strip_prefix(RECORD))
        .map(|rest| {
            rest.split_whitespace()
                .map(|pair| {
                    let (name, value) = pair
                        .split_once('=')
                        .unwrap_or_else(|| panic!("Fix: probe field `{pair}` has no value"));
                    let value = value
                        .parse::<u64>()
                        .unwrap_or_else(|_| panic!("Fix: probe field `{pair}` is not a u64"));
                    (name.to_string(), value)
                })
                .collect()
        })
        .collect()
}

#[test]
fn every_timed_dispatch_emits_one_consistent_phase_record() {
    let output = Command::new(std::env::current_exe().expect("Fix: test binary path must exist"))
        .arg("--exact")
        .arg(helper_test_filter())
        .arg("--nocapture")
        .arg("--test-threads=1")
        .env(HELPER_FLAG, "1")
        .env("VYRE_CUDA_DISPATCH_PHASE_PROBE", "1")
        .output()
        .expect("Fix: the helper test process must launch");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "Fix: helper process failed with {}:\n{stderr}",
        output.status
    );
    let records = records(&stderr);
    assert_eq!(
        records.len(),
        DIRECT_DISPATCHES + COMPILED_DISPATCHES,
        "Fix: every timed dispatch must emit exactly one record:\n{stderr}"
    );
    let expected = expected_fields();
    let program = add_one_program();
    let mut previous_seq = 0;
    for (index, record) in records.iter().enumerate() {
        let names: Vec<&str> = record.keys().map(String::as_str).collect();
        assert_eq!(names, expected, "record {index} field set");
        let field = |name: &str| record[name];
        assert!(
            field("seq") > previous_seq,
            "record {index} sequence must increase"
        );
        previous_seq = field("seq");
        assert!(field("nodes") > 0, "record {index} counts no IR nodes");
        assert!(field("ptx_bytes") > 0, "record {index} counts no PTX");
        assert_eq!(field("buffers"), program.buffers().len() as u64);
        assert_eq!(field("bindings"), program.buffers().len() as u64);
        assert!(
            field("grid_blocks") >= 1,
            "record {index} has an empty grid"
        );
        assert!(
            field("kernel_ns") > 0,
            "record {index} kernel window not charged"
        );
        assert!(
            field("launch_ns") > 0,
            "record {index} launch loop not charged"
        );
        assert!(
            field("sub_launch_call_ns") <= field("launch_ns"),
            "record {index}: the launch call is nested inside the launch loop"
        );
        let named: u64 = PHASES.iter().map(|phase| field(phase)).sum();
        assert_eq!(field("named_host_ns"), named, "record {index} named total");
        assert_eq!(
            field("unattributed_ns"),
            field("wall_ns").saturating_sub(named),
            "record {index} residue"
        );
    }
    let direct = &records[..DIRECT_DISPATCHES];
    assert!(
        direct
            .iter()
            .all(|record| record["ptx_ns"] > 0 && record["prepare_ns"] > 0),
        "Fix: the direct entry point must charge its prepare and PTX phases"
    );
    assert!(
        direct[DIRECT_DISPATCHES - 1]["ptx_cache_hits"] > direct[0]["ptx_cache_hits"],
        "Fix: a repeated program must hit the PTX source cache"
    );
}
