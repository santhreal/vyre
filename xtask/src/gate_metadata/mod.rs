//! Authoritative metadata for every registered gate.
//!
//! The implementation registry owns executable code. This table owns the
//! authoritative facts the runner must validate before execution: stable name,
//! help text, owner package, area membership, authoritative subject class,
//! exact generated paths, prerequisites, and mutation-proof method.
//! Registry and descriptor agreement is checked in both directions. Named subsets
//! are derived from `areas`; no second list of gate names exists.

use std::collections::BTreeSet;

pub use crate::gate_proof_validation::{validate_all_descriptors, validate_proof_symbol};
use crate::gate::{GateDescriptor, ResourceClass};

pub mod artifacts;
pub mod descriptors_a_g;
pub mod descriptors_h_p;
pub mod descriptors_q_z;

pub use artifacts::{FROZEN_CONTRACT_ARTIFACTS, PUBLIC_API_ARTIFACTS, TESTING_GUIDE_ARTIFACTS};
use descriptors_a_g::GATES_A_G;
use descriptors_h_p::GATES_H_P;
use descriptors_q_z::GATES_Q_Z;

const fn concat_descriptors(
    a: &[GateDescriptor; 52],
    b: &[GateDescriptor; 59],
    c: &[GateDescriptor; 36],
) -> [GateDescriptor; 147] {
    let mut out = [GateDescriptor {
        name: "",
        help: "",
        package: "xtask",
        areas: &[],
        subject: "",
        inputs: &[],
        artifacts: &[],
        prerequisites: &[],
        resource_class: ResourceClass::Cpu,
        proof: "",
    }; 147];
    let mut i = 0;
    while i < a.len() {
        out[i] = a[i];
        i += 1;
    }
    let mut j = 0;
    while j < b.len() {
        out[a.len() + j] = b[j];
        j += 1;
    }
    let mut k = 0;
    while k < c.len() {
        out[a.len() + b.len() + k] = c[k];
        k += 1;
    }
    out
}

/// Complete array holding all registered gate descriptors.
pub static GATE_METADATA_ARRAY: [GateDescriptor; 147] = concat_descriptors(&GATES_A_G, &GATES_H_P, &GATES_Q_Z);

/// Every gate descriptor, sorted by gate name.
pub static GATE_METADATA: &[GateDescriptor] = &GATE_METADATA_ARRAY;

/// All gate names declared in `GATE_METADATA`, sorted.
#[must_use]
pub fn all_gate_names() -> Vec<&'static str> {
    GATE_METADATA.iter().map(|d| d.name).collect()
}
/// Authoritative descriptor for `gate_name`, or `None` if it is not registered.
#[must_use]
pub fn descriptor(gate_name: &str) -> Option<&'static GateDescriptor> {
    GATE_METADATA.iter().find(|d| d.name == gate_name)
}

/// Authoritative descriptor for `gate_name`.
///
/// # Panics
///
/// Panics when `gate_name` is not in `GATE_METADATA`.
#[must_use]
pub fn descriptor_by_name(gate_name: &str) -> &'static GateDescriptor {
    descriptor(gate_name).unwrap_or_else(|| panic!("gate `{gate_name}` is not in GATE_METADATA"))
}

/// All gate names owned by `package`, sorted.
#[must_use]
pub fn owned_by(package: &str) -> Vec<&'static str> {
    GATE_METADATA
        .iter()
        .filter(|d| d.package == package)
        .map(|d| d.name)
        .collect()
}

/// All distinct area names declared across all gate descriptors, sorted.
#[must_use]
pub fn areas() -> Vec<&'static str> {
    let mut set = BTreeSet::new();
    for row in GATE_METADATA {
        for area in row.areas {
            set.insert(*area);
        }
    }
    set.into_iter().collect()
}

/// All gate names belonging to `area`, sorted.
#[must_use]
pub fn gates_in_area(area: &str) -> Vec<&'static str> {
    GATE_METADATA
        .iter()
        .filter(|d| d.areas.contains(&area))
        .map(|d| d.name)
        .collect()
}

/// Exact generated artifact paths with their owning gate names.
#[must_use]
pub fn generated_artifacts() -> Vec<(&'static str, &'static str)> {
    let mut pairs = Vec::new();
    for d in GATE_METADATA {
        for art in d.artifacts {
            pairs.push((*art, d.name));
        }
    }
    pairs.sort_unstable_by_key(|(art, _)| *art);
    pairs
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::{BTreeMap, BTreeSet};

    /// WHY: Section 182.2 requires every gate metadata entry to be sorted by name.
    #[test]
    fn metadata_is_sorted_by_name() {
        let names: Vec<&str> = GATE_METADATA.iter().map(|d| d.name).collect();
        let mut sorted = names.clone();
        sorted.sort_unstable();
        assert_eq!(
            names, sorted,
            "GATE_METADATA must be sorted alphabetically by name"
        );
    }

    /// WHY: Section 182.2 requires every gate metadata entry to have a non-empty name and unique entry.
    #[test]
    fn every_gate_name_is_unique() {
        let mut names = BTreeSet::new();
        for d in GATE_METADATA {
            assert!(
                names.insert(d.name),
                "duplicate gate name in GATE_METADATA: {}",
                d.name
            );
        }
    }

    /// WHY: Section 182.2.1 requires every descriptor to declare valid fields without provisional values.
    #[test]
    fn every_metadata_entry_passes_validation() {
        for d in GATE_METADATA {
            let failures = d.failures();
            assert!(
                failures.is_empty(),
                "gate `{}` has invalid metadata: {:?}",
                d.name,
                failures
            );
        }
    }

    /// WHY: Section 182.5.3 requires exact 1-to-1 generated artifact ownership without duplicate owners.
    #[test]
    fn artifact_paths_are_uniquely_owned() {
        let mut owners: BTreeMap<&str, &str> = BTreeMap::new();
        for (artifact, gate) in generated_artifacts() {
            if let Some(existing) = owners.insert(artifact, gate) {
                panic!(
                    "artifact `{artifact}` is declared by both `{existing}` and `{gate}`: each artifact must have exactly 1 owning gate"
                );
            }
        }
    }

    /// WHY: Section 182 requires every gate descriptor in GATE_METADATA to carry a unique mutation proof identity.
    #[test]
    fn every_descriptor_proof_is_unique() {
        let mut proofs = BTreeSet::new();
        for d in GATE_METADATA {
            assert!(
                proofs.insert(d.proof),
                "duplicate proof identity in GATE_METADATA for gate `{}`: `{}`",
                d.name,
                d.proof
            );
        }
    }
}

