//! Shared reading of the float lowering decision ledger.
//!
//! The ledger is `vyre-driver/float-lowering-decisions.toml`: one row per
//! backend stating what it does with each `FloatLoweringMode`. Two suites read
//! it. The closure suite runs in every lane and asks whether a row exists for
//! every backend and mode this build carries. The device suite runs only where
//! a device does and asks whether the shipped backend answers what its row
//! claims, so the two share this reader rather than parsing the same file twice.

#![forbid(unsafe_code)]

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

use vyre_driver::BackendError;
use vyre_foundation::fp_parity::FloatLoweringMode;
use vyre_registry_link::backend::{linked_backend_sources, live_backend_registry};

/// What a row claims a backend does with one mode.
///
/// A backend whose answer is a property of its emitter is `Always` or `Never`
/// and the row states which. A backend whose answer is a property of the device
/// in front of it cannot be pinned to either: the wgpu driver hands the platform
/// a contraction-free module and the platform compiles it again, so whether the
/// strict mode survives is measured on the adapter. `Measured` records that the
/// decision is taken at run time, which is a different claim from "either
/// answer is acceptable": the strict-dispatch contract below still holds the
/// backend to matching the oracle or refusing by name.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Lowering {
    /// The row claims the backend lowers this mode on every device.
    Always,
    /// The row claims the backend never lowers this mode.
    Never,
    /// The row claims the answer is taken from the adapter at run time.
    Measured,
}

impl Lowering {
    /// Whether `answered` is admissible under this claim.
    pub fn admits(self, answered: bool) -> bool {
        match self {
            Self::Always => answered,
            Self::Never => !answered,
            Self::Measured => true,
        }
    }
}

/// One backend's row: which modes it lowers, and why.
pub struct Decision {
    /// What the row claims for each mode, keyed by `cache_label`.
    pub lowers: BTreeMap<String, Lowering>,
    /// Why the row claims it. An empty reason is a finding.
    pub reason: String,
}

/// Where the ledger is, resolved from the checkout the test binary runs in.
pub fn ledger_path() -> PathBuf {
    structure_gate::workspace_root().join("vyre-driver/float-lowering-decisions.toml")
}

/// The ledger as one row per backend id.
pub fn read_ledger() -> BTreeMap<String, Decision> {
    let path = ledger_path();
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("Fix: cannot read {}: {error}", path.display()));
    let table: toml::Value = toml::from_str(&text)
        .unwrap_or_else(|error| panic!("Fix: cannot parse {}: {error}", path.display()));
    let rows = table
        .as_table()
        .unwrap_or_else(|| panic!("Fix: {} must be a table of backend rows", path.display()));
    rows.iter()
        .map(|(backend, row)| {
            let row = row.as_table().unwrap_or_else(|| {
                panic!(
                    "Fix: {} row `{backend}` must be a table of mode decisions",
                    path.display()
                )
            });
            let reason = row
                .get("reason")
                .and_then(toml::Value::as_str)
                .unwrap_or_else(|| {
                    panic!(
                        "Fix: {} row `{backend}` must state a `reason` for its decisions",
                        path.display()
                    )
                })
                .to_string();
            let lowers = row
                .iter()
                .filter(|(key, _)| key.as_str() != "reason")
                .map(|(mode, value)| {
                    let lowers = match value {
                        toml::Value::Boolean(true) => Lowering::Always,
                        toml::Value::Boolean(false) => Lowering::Never,
                        toml::Value::String(word) if word == "measured" => Lowering::Measured,
                        other => panic!(
                            "Fix: {} row `{backend}` key `{mode}` must be `true`, `false`, or the \
                             string \"measured\", not {other}",
                            path.display()
                        ),
                    };
                    (mode.clone(), lowers)
                })
                .collect();
            (backend.clone(), Decision { lowers, reason })
        })
        .collect()
}

/// Every backend id this build can be asked about: registered here, or owned by
/// a driver crate this build links whose registration is compiled out.
pub fn backends_needing_a_decision() -> BTreeSet<String> {
    let mut ids: BTreeSet<String> = linked_backend_sources()
        .iter()
        .map(|source| source.backend_id.to_string())
        .collect();
    for registration in live_backend_registry().expect("the backend registry must be readable") {
        ids.insert(registration.id.to_string());
    }
    ids
}

/// Little-endian bytes of an f32 slice, as a dispatch input buffer.
pub fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// The first lane two answers disagree on, as bits and as values.
///
/// A strict mismatch is a rounding question, and "different bits" does not say
/// whether one rounding was lost to a contraction, a subnormal was flushed, or
/// an approximate instruction survived the expansion. The offending lane and
/// its two words separate those without another run.
pub fn first_divergence(expected: &[Vec<u8>], actual: &[Vec<u8>]) -> String {
    if expected.len() != actual.len() {
        return format!(
            " oracle produced {} buffer(s) and the backend {}",
            expected.len(),
            actual.len()
        );
    }
    for (index, (oracle, device)) in expected.iter().zip(actual).enumerate() {
        if oracle.len() != device.len() {
            return format!(
                " buffer {index}: oracle {} byte(s), backend {} byte(s)",
                oracle.len(),
                device.len()
            );
        }
        for (lane, (left, right)) in oracle
            .chunks_exact(4)
            .zip(device.chunks_exact(4))
            .enumerate()
        {
            if left == right {
                continue;
            }
            let oracle_bits = u32::from_le_bytes([left[0], left[1], left[2], left[3]]);
            let device_bits = u32::from_le_bytes([right[0], right[1], right[2], right[3]]);
            return format!(
                " buffer {index} lane {lane}: oracle 0x{oracle_bits:08x} ({}) backend \
                 0x{device_bits:08x} ({})",
                f32::from_bits(oracle_bits),
                f32::from_bits(device_bits)
            );
        }
    }
    String::from(" no differing lane, so the buffers differ in trailing bytes")
}

/// What one dispatch under `mode` proved about a backend, as a finding line or
/// nothing.
///
/// Two suites ask this question. The closure suite asks it for the strict mode
/// in every lane, and the device suite asks it for every mode where a device
/// admits a backend. Each one classified the four outcomes itself, and a
/// classifier written twice is free to hold one suite to naming the backend and
/// the other to naming only the mode, which is how a refusal that names neither
/// passes on the lane that happens to run it.
///
/// `named_operation` is the operation the program is built around when the mode
/// blocks it, and `None` when the mode admits every operation.
pub fn honored_or_refused_by_name(
    backend_id: &str,
    honors: bool,
    mode: FloatLoweringMode,
    named_operation: Option<&str>,
    outcome: &Result<Vec<Vec<u8>>, BackendError>,
) -> Option<String> {
    let label = mode.cache_label();
    match outcome {
        Ok(_) if honors => None,
        Ok(_) => Some(format!(
            "  backend `{backend_id}` returned Ok for mode `{label}` but honors_float_lowering \
             returned false"
        )),
        Err(error) if honors => Some(format!(
            "  backend `{backend_id}` failed dispatch for honored mode `{label}`: {error}"
        )),
        Err(error) => {
            let message = error.to_string();
            let mut absent = Vec::new();
            if !message.contains(label) {
                absent.push("the mode");
            }
            if !message.contains(backend_id) {
                absent.push("the backend");
            }
            if let Some(operation) = named_operation {
                if !message.contains(operation) {
                    absent.push("the operation");
                }
            }
            if !message.contains("Fix:") {
                absent.push("a Fix:");
            }
            if absent.is_empty() {
                return None;
            }
            Some(format!(
                "  backend `{backend_id}` refused unhonored mode `{label}` without naming {}: \
                 {message}",
                absent.join(", ")
            ))
        }
    }
}
