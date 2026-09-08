//! Declarative DAG schema and execution kernel for registered gates.
//!
//! Owns DAG construction, dependency cycle detection, topological execution ordering,
//! prerequisite validation, content-addressed cache key derivation, and change-impact
//! execution scheduling.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::fmt;
use std::path::Path;

use crate::gate::{GateDescriptor, RegisteredGate};

/// Errors encountered while validating or ordering the gate DAG.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum DagError {
    /// A gate declares a prerequisite that does not exist in the registry.
    MissingPrerequisite {
        /// Gate declaring the missing prerequisite.
        gate: &'static str,
        /// Name of the missing prerequisite.
        prerequisite: &'static str,
    },
    /// A dependency cycle was detected among registered gates.
    DependencyCycle {
        /// Names of gates involved in the cycle.
        cycle: Vec<&'static str>,
    },
    /// A gate declares a prerequisite on itself.
    SelfDependency {
        /// Name of the gate.
        gate: &'static str,
    },
}

impl fmt::Display for DagError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingPrerequisite { gate, prerequisite } => {
                write!(
                    f,
                    "gate `{gate}` declares prerequisite `{prerequisite}`, which is not registered"
                )
            }
            Self::DependencyCycle { cycle } => {
                write!(
                    f,
                    "dependency cycle detected in gate DAG: {}",
                    cycle.join(" -> ")
                )
            }
            Self::SelfDependency { gate } => {
                write!(f, "gate `{gate}` declares a dependency on itself")
            }
        }
    }
}

impl std::error::Error for DagError {}

/// One node in the declarative gate DAG.
#[derive(Clone, Debug)]
pub struct DagNode {
    /// Authoritative metadata descriptor.
    pub descriptor: &'static GateDescriptor,
    /// Gates this gate directly depends on.
    pub prerequisites: Vec<&'static str>,
    /// Gates that depend on this gate.
    pub dependents: Vec<&'static str>,
}

/// The declarative DAG holding all registered gates and their dependencies.
#[derive(Clone, Debug, Default)]
pub struct GateDag {
    nodes: BTreeMap<&'static str, DagNode>,
}

impl GateDag {
    /// Build a DAG from a slice of static gate descriptors.
    ///
    /// # Errors
    ///
    /// Returns `DagError` if any prerequisite is missing or if a cycle is detected.
    pub fn from_descriptors(descriptors: &'static [GateDescriptor]) -> Result<Self, DagError> {
        let mut nodes = BTreeMap::new();
        let known_names: BTreeSet<&'static str> = descriptors.iter().map(|d| d.name).collect();

        for desc in descriptors {
            if desc.prerequisites.contains(&desc.name) {
                return Err(DagError::SelfDependency { gate: desc.name });
            }
            for prereq in desc.prerequisites {
                if !known_names.contains(prereq) {
                    return Err(DagError::MissingPrerequisite {
                        gate: desc.name,
                        prerequisite: prereq,
                    });
                }
            }
            nodes.insert(
                desc.name,
                DagNode {
                    descriptor: desc,
                    prerequisites: desc.prerequisites.to_vec(),
                    dependents: Vec::new(),
                },
            );
        }

        // Build inverse edges (dependents)
        for desc in descriptors {
            for prereq in desc.prerequisites {
                if let Some(node) = nodes.get_mut(prereq) {
                    node.dependents.push(desc.name);
                }
            }
        }

        let dag = Self { nodes };
        // Validate acyclicity
        let _ = dag.topological_order()?;
        Ok(dag)
    }

    /// Build a DAG from a slice of registered gates.
    ///
    /// # Errors
    ///
    /// Returns `DagError` if any prerequisite is missing or if a cycle is detected.
    pub fn from_registry(gates: &[RegisteredGate]) -> Result<Self, DagError> {
        let mut descriptors = Vec::with_capacity(gates.len());
        for gate in gates {
            descriptors.push(*gate.descriptor());
        }
        let static_slice: &'static [GateDescriptor] = descriptors.leak();
        Self::from_descriptors(static_slice)
    }

    /// Return the node corresponding to `gate_name`, if registered.
    #[must_use]
    pub fn get(&self, gate_name: &str) -> Option<&DagNode> {
        self.nodes.get(gate_name)
    }

    /// Total number of gates in the DAG.
    #[must_use]
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Whether the DAG is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Return all gate names in the DAG in alphabetical order.
    #[must_use]
    pub fn gate_names(&self) -> Vec<&'static str> {
        self.nodes.keys().copied().collect()
    }

    /// Return the topological ordering of gate execution.
    ///
    /// If gate A depends on gate B, B is guaranteed to precede A in the returned order.
    ///
    /// # Errors
    ///
    /// Returns `DagError::DependencyCycle` if a cycle exists.
    pub fn topological_order(&self) -> Result<Vec<&'static str>, DagError> {
        let mut in_degrees: BTreeMap<&'static str, usize> = BTreeMap::new();
        for (name, node) in &self.nodes {
            in_degrees.insert(*name, node.prerequisites.len());
        }

        let mut queue: VecDeque<&'static str> = VecDeque::new();
        for (name, deg) in &in_degrees {
            if *deg == 0 {
                queue.push_back(*name);
            }
        }

        let mut ordered = Vec::with_capacity(self.nodes.len());
        while let Some(current) = queue.pop_front() {
            ordered.push(current);
            if let Some(node) = self.nodes.get(current) {
                for dependent in &node.dependents {
                    if let Some(deg) = in_degrees.get_mut(dependent) {
                        *deg -= 1;
                        if *deg == 0 {
                            queue.push_back(dependent);
                        }
                    }
                }
            }
        }

        if ordered.len() != self.nodes.len() {
            // Find nodes involved in cycle
            let mut cycle_nodes = Vec::new();
            for (name, deg) in in_degrees {
                if deg > 0 {
                    cycle_nodes.push(name);
                }
            }
            return Err(DagError::DependencyCycle { cycle: cycle_nodes });
        }

        Ok(ordered)
    }

    /// Validate the DAG structure, inputs, and prerequisites against the workspace root.
    #[must_use]
    pub fn validate(&self, root: &Path) -> Vec<String> {
        let mut failures = Vec::new();

        // 1. Check topological ordering and cycles
        if let Err(err) = self.topological_order() {
            failures.push(format!("DAG ordering error: {err}"));
        }

        // 2. Check each node
        for (name, node) in &self.nodes {
            let desc = node.descriptor;
            if desc.name != *name {
                failures.push(format!(
                    "gate `{name}` node key does not match descriptor name `{}`",
                    desc.name
                ));
            }
            // Check prerequisites exist
            for prereq in &node.prerequisites {
                if !self.nodes.contains_key(prereq) {
                    failures.push(format!(
                        "gate `{name}` declares missing prerequisite `{prereq}`"
                    ));
                }
            }
            // Check declared inputs exist if specified
            for input in desc.inputs {
                let input_path = root.join(input);
                if !input_path.exists() {
                    // Could be a relative path or glob
                    failures.push(format!(
                        "gate `{name}` declares input path `{input}`, which does not exist in workspace root",
                    ));
                }
            }
        }

        failures
    }

    /// Check if a gate is skippable because its cache key matches the previous run.
    #[must_use]
    pub fn is_skippable(&self, gate_name: &str, root: &Path, recorded_cache_key: Option<&str>) -> bool {
        let Some(node) = self.nodes.get(gate_name) else {
            return false;
        };
        let current_key = node.descriptor.compute_cache_key(root);
        recorded_cache_key == Some(&current_key)
    }
}

/// Execution status of one gate in a DAG run.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum GateExecutionStatus {
    /// Gate ran and passed (findings <= baseline).
    Passed {
        /// Number of findings reported.
        findings: usize,
        /// Computed cache key.
        cache_key: String,
    },
    /// Gate ran and failed (findings > baseline or execution error).
    Failed {
        /// Number of findings or error message.
        error: String,
    },
    /// Gate was skipped because its inputs and descriptor are unchanged.
    SkippedUnchanged {
        /// Matched cache key.
        cache_key: String,
    },
    /// Gate was skipped because one of its prerequisites failed.
    SkippedPrerequisiteFailed {
        /// Name of the prerequisite that failed.
        failed_prerequisite: &'static str,
    },
}

/// Execution options for running a selection of gates through the DAG kernel.
#[derive(Clone, Debug, Default)]
pub struct DagExecutionOptions {
    /// Allow skipping gates whose inputs are unchanged since their last recorded run.
    pub skip_unchanged: bool,
    /// Recorded cache keys from previous clean runs: gate_name -> cache_key.
    pub cache_keys: BTreeMap<String, String>,
}

/// Summary result of a DAG execution pass.
#[derive(Clone, Debug, Default)]
pub struct DagExecutionReport {
    /// Outcome per gate name.
    pub outcomes: BTreeMap<&'static str, GateExecutionStatus>,
    /// Any fatal failure messages.
    pub failures: Vec<String>,
}

impl DagExecutionReport {
    /// Whether all executed gates succeeded or were cleanly skipped.
    #[must_use]
    pub fn is_success(&self) -> bool {
        self.failures.is_empty()
            && self.outcomes.values().all(|s| {
                matches!(
                    s,
                    GateExecutionStatus::Passed { .. } | GateExecutionStatus::SkippedUnchanged { .. }
                )
            })
    }
}

/// Execute a list of gates through the DAG kernel enforcing prerequisite ordering and caching.
pub fn execute_dag(
    root: &Path,
    dag: &GateDag,
    gates: &[RegisteredGate],
    options: &DagExecutionOptions,
) -> DagExecutionReport {
    let mut report = DagExecutionReport::default();
    let order = match dag.topological_order() {
        Ok(o) => o,
        Err(err) => {
            report.failures.push(format!("DAG execution aborted: {err}"));
            return report;
        }
    };

    let selected_names: BTreeSet<&str> = gates.iter().map(RegisteredGate::name).collect();

    for gate_name in order {
        if !selected_names.contains(gate_name) {
            continue;
        }
        let Some(node) = dag.get(gate_name) else {
            continue;
        };

        // 1. Check prerequisites
        let mut failed_prereq: Option<&'static str> = None;
        for prereq in &node.prerequisites {
            if let Some(status) = report.outcomes.get(prereq) {
                if !matches!(
                    status,
                    GateExecutionStatus::Passed { .. }
                        | GateExecutionStatus::SkippedUnchanged { .. }
                ) {
                    failed_prereq = Some(prereq);
                    break;
                }
            }
        }

        if let Some(failed) = failed_prereq {
            report.outcomes.insert(
                gate_name,
                GateExecutionStatus::SkippedPrerequisiteFailed {
                    failed_prerequisite: failed,
                },
            );
            continue;
        }

        // 2. Check if skippable
        let cache_key = node.descriptor.compute_cache_key(root);
        if options.skip_unchanged {
            if let Some(recorded) = options.cache_keys.get(gate_name) {
                if *recorded == cache_key {
                    report.outcomes.insert(
                        gate_name,
                        GateExecutionStatus::SkippedUnchanged {
                            cache_key: cache_key.clone(),
                        },
                    );
                    continue;
                }
            }
        }

        // 3. Find registered gate and execute
        let Some(registered) = gates.iter().find(|g| g.name() == gate_name) else {
            continue;
        };

        let ctx = crate::gate::GateCtx::new(root.to_path_buf(), Vec::new());
        match registered.run(&ctx) {
            Ok(gate_report) => {
                let found = gate_report.count();
                if found == 0 {
                    report.outcomes.insert(
                        gate_name,
                        GateExecutionStatus::Passed {
                            findings: 0,
                            cache_key,
                        },
                    );
                } else {
                    report.outcomes.insert(
                        gate_name,
                        GateExecutionStatus::Failed {
                            error: format!("reported {found} finding(s)"),
                        },
                    );
                }
            }
            Err(err) => {
                report.outcomes.insert(
                    gate_name,
                    GateExecutionStatus::Failed {
                        error: err.to_string(),
                    },
                );
            }
        }
    }

    report
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gate::ResourceClass;

    const DUMMY_GATE_A: GateDescriptor = GateDescriptor {
        name: "gate-a",
        help: "Gate A help",
        package: "xtask",
        areas: &["contract-rules"],
        subject: "test subject",
        inputs: &[],
        artifacts: &[],
        prerequisites: &[],
        resource_class: ResourceClass::Cpu,
        proof: "crate::gate_dag::tests::dag_validation_detects_cycles_and_missing_prerequisites",
    };

    const DUMMY_GATE_B: GateDescriptor = GateDescriptor {
        name: "gate-b",
        help: "Gate B help",
        package: "xtask",
        areas: &["contract-rules"],
        subject: "test subject",
        inputs: &[],
        artifacts: &[],
        prerequisites: &["gate-a"],
        resource_class: ResourceClass::Cpu,
        proof: "crate::gate_dag::tests::dag_validation_detects_cycles_and_missing_prerequisites",
    };

    const DUMMY_GATE_C: GateDescriptor = GateDescriptor {
        name: "gate-c",
        help: "Gate C help",
        package: "xtask",
        areas: &["contract-rules"],
        subject: "test subject",
        inputs: &[],
        artifacts: &[],
        prerequisites: &["gate-b"],
        resource_class: ResourceClass::Cpu,
        proof: "crate::gate_dag::tests::dag_validation_detects_cycles_and_missing_prerequisites",
    };

    #[test]
    fn topological_sort_orders_prerequisites_first() {
        static GATES: [GateDescriptor; 3] = [DUMMY_GATE_C, DUMMY_GATE_B, DUMMY_GATE_A];
        let dag = GateDag::from_descriptors(&GATES).expect("DAG should build");
        let order = dag.topological_order().expect("Topological sort should succeed");

        let pos_a = order.iter().position(|&x| x == "gate-a").unwrap();
        let pos_b = order.iter().position(|&x| x == "gate-b").unwrap();
        let pos_c = order.iter().position(|&x| x == "gate-c").unwrap();

        assert!(pos_a < pos_b, "gate-a must come before gate-b");
        assert!(pos_b < pos_c, "gate-b must come before gate-c");
    }

    #[test]
    fn dag_detects_dependency_cycle() {
        const CYCLE_A: GateDescriptor = GateDescriptor {
            name: "cycle-a",
            help: "Cycle A",
            package: "xtask",
            areas: &["contract-rules"],
            subject: "test",
            inputs: &[],
            artifacts: &[],
            prerequisites: &["cycle-b"],
            resource_class: ResourceClass::Cpu,
            proof: "crate::gate_dag::tests::dag_validation_detects_cycles_and_missing_prerequisites",
        };
        const CYCLE_B: GateDescriptor = GateDescriptor {
            name: "cycle-b",
            help: "Cycle B",
            package: "xtask",
            areas: &["contract-rules"],
            subject: "test",
            inputs: &[],
            artifacts: &[],
            prerequisites: &["cycle-a"],
            resource_class: ResourceClass::Cpu,
            proof: "crate::gate_dag::tests::dag_validation_detects_cycles_and_missing_prerequisites",
        };
        static CYCLE_GATES: [GateDescriptor; 2] = [CYCLE_A, CYCLE_B];
        let res = GateDag::from_descriptors(&CYCLE_GATES);
        assert!(res.is_err());
        assert!(matches!(res.unwrap_err(), DagError::DependencyCycle { .. }));
    }

    #[test]
    fn dag_detects_missing_prerequisites() {
        const ORPHAN: GateDescriptor = GateDescriptor {
            name: "orphan-gate",
            help: "Orphan gate",
            package: "xtask",
            areas: &["contract-rules"],
            subject: "test",
            inputs: &[],
            artifacts: &[],
            prerequisites: &["non-existent-gate"],
            resource_class: ResourceClass::Cpu,
            proof: "crate::gate_dag::tests::dag_validation_detects_cycles_and_missing_prerequisites",
        };
        static ORPHAN_GATES: [GateDescriptor; 1] = [ORPHAN];
        let res = GateDag::from_descriptors(&ORPHAN_GATES);
        assert!(res.is_err());
        assert!(matches!(
            res.unwrap_err(),
            DagError::MissingPrerequisite { .. }
        ));
    }

    #[test]
    fn dag_validation_detects_cycles_and_missing_prerequisites() {
        // Closure proof test: validates that live GATE_METADATA is valid DAG
        let dag = GateDag::from_descriptors(crate::gate_metadata::GATE_METADATA)
            .expect("Live GATE_METADATA must be a valid acyclic DAG");
        let root = crate::checkout::checkout_root();
        let failures = dag.validate(&root);
        assert!(failures.is_empty(), "DAG validation failed: {failures:?}");
    }
}
