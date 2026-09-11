//! Whole-composition validation, liveness, and reusable allocation analysis.

use std::collections::BTreeSet;

use thiserror::Error;

use super::program_graph::{
    GraphNodeId, GraphValueId, LivenessInterval, ProgramGraph, ValueContract, ValueLifetime,
};

/// Allocation decision for one graph value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GraphBufferAllocation {
    /// Connected value.
    pub value: GraphValueId,
    /// Inclusive producer-to-last-consumer interval.
    pub interval: LivenessInterval,
    /// Reusable invocation-local slot, or `None` for dedicated storage.
    pub reusable_slot: Option<u32>,
}

/// Validated schedule and memory facts shared by planners and runtimes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProgramGraphAnalysis {
    /// Canonical topological schedule.
    pub schedule: Vec<GraphNodeId>,
    /// One allocation decision per graph value.
    pub allocations: Vec<GraphBufferAllocation>,
    /// Number of invocation-local slots required by interval coloring.
    pub reusable_slot_count: u32,
}

/// First structural counterexample found during whole-graph analysis.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ProgramGraphAnalysisError {
    /// Node vector position and canonical identity disagree.
    #[error(
        "node position {position} carries identity {actual:?}; expected GraphNodeId({position})"
    )]
    NodeIdentity {
        /// Vector position.
        position: u32,
        /// Stored identity.
        actual: GraphNodeId,
    },
    /// Value vector position and canonical identity disagree.
    #[error(
        "value position {position} carries identity {actual:?}; expected GraphValueId({position})"
    )]
    ValueIdentity {
        /// Vector position.
        position: u32,
        /// Stored identity.
        actual: GraphValueId,
    },
    /// Node input points outside the value table.
    #[error("node {node:?} input references missing value {value:?}")]
    MissingInput {
        /// Consuming node.
        node: GraphNodeId,
        /// Missing value.
        value: GraphValueId,
    },
    /// Input producer is not earlier in the topological schedule.
    #[error("node {node:?} consumes {value:?} from non-preceding producer {producer:?}")]
    NonTopologicalInput {
        /// Consuming node.
        node: GraphNodeId,
        /// Connected value.
        value: GraphValueId,
        /// Invalid producer.
        producer: GraphNodeId,
    },
    /// Value consumer metadata omits its connected node.
    #[error("value {value:?} omits connected consumer {node:?}")]
    MissingConsumer {
        /// Connected value.
        value: GraphValueId,
        /// Missing consumer.
        node: GraphNodeId,
    },
    /// Consumer identity is invalid or duplicated.
    #[error("value {value:?} has invalid or duplicate consumer {consumer:?}")]
    InvalidConsumer {
        /// Connected value.
        value: GraphValueId,
        /// Invalid consumer.
        consumer: GraphNodeId,
    },
    /// Consumer type or shape differs from its value.
    #[error("node {node:?} input {value:?} changes dtype, shape, or lifetime")]
    InputContract {
        /// Consuming node.
        node: GraphNodeId,
        /// Connected value.
        value: GraphValueId,
    },
    /// Node output ids and output bindings have different lengths.
    #[error("node {node:?} has {ids} output ids but {ports} output ports")]
    OutputArity {
        /// Producing node.
        node: GraphNodeId,
        /// Value identity count.
        ids: usize,
        /// Port count.
        ports: usize,
    },
    /// Produced value metadata disagrees with its output port.
    #[error("node {node:?} output {value:?} disagrees with its typed port")]
    OutputContract {
        /// Producing node.
        node: GraphNodeId,
        /// Produced value.
        value: GraphValueId,
    },
    /// State successor is missing, unconsumed, or not type preserving.
    #[error("state value {value:?} has invalid predecessor {prior:?}")]
    StateTransition {
        /// Produced state value.
        value: GraphValueId,
        /// Invalid predecessor.
        prior: GraphValueId,
    },
    /// Graph size exceeds stable u32 identities.
    #[error("graph analysis identity exceeds u32")]
    IdentityOverflow,
}

impl ProgramGraph {
    /// Validate the complete composition and derive one reusable allocation plan.
    pub fn analyze(&self) -> Result<ProgramGraphAnalysis, ProgramGraphAnalysisError> {
        validate_graph(self)?;
        let schedule = self.nodes().iter().map(|node| node.id).collect();
        let intervals = self.liveness_intervals();
        let mut allocations = intervals
            .iter()
            .copied()
            .map(|interval| GraphBufferAllocation {
                value: interval.value,
                interval,
                reusable_slot: None,
            })
            .collect::<Vec<_>>();

        let mut invocation = allocations
            .iter()
            .enumerate()
            .filter(|(_, allocation)| {
                self.values()[allocation.value.0 as usize].contract.lifetime
                    == ValueLifetime::Invocation
            })
            .map(|(index, allocation)| (index, allocation.interval))
            .collect::<Vec<_>>();
        invocation
            .sort_unstable_by_key(|(_, interval)| (interval.start, interval.end, interval.value.0));
        let mut slot_ends = Vec::<usize>::new();
        for (allocation_index, interval) in invocation {
            let slot = slot_ends.iter().position(|end| *end < interval.start);
            let slot = match slot {
                Some(slot) => slot,
                None => {
                    slot_ends.push(0);
                    slot_ends.len() - 1
                }
            };
            slot_ends[slot] = interval.end;
            allocations[allocation_index].reusable_slot =
                Some(u32::try_from(slot).map_err(|_| ProgramGraphAnalysisError::IdentityOverflow)?);
        }
        Ok(ProgramGraphAnalysis {
            schedule,
            allocations,
            reusable_slot_count: u32::try_from(slot_ends.len())
                .map_err(|_| ProgramGraphAnalysisError::IdentityOverflow)?,
        })
    }
}

fn validate_graph(graph: &ProgramGraph) -> Result<(), ProgramGraphAnalysisError> {
    for (position, value) in graph.values().iter().enumerate() {
        let position =
            u32::try_from(position).map_err(|_| ProgramGraphAnalysisError::IdentityOverflow)?;
        if value.id != GraphValueId(position) {
            return Err(ProgramGraphAnalysisError::ValueIdentity {
                position,
                actual: value.id,
            });
        }
        let mut consumers = BTreeSet::new();
        for consumer in &value.consumers {
            if consumer.0 as usize >= graph.nodes().len() || !consumers.insert(*consumer) {
                return Err(ProgramGraphAnalysisError::InvalidConsumer {
                    value: value.id,
                    consumer: *consumer,
                });
            }
        }
        if let Some(prior_id) = value.retained_successor_of {
            let prior = graph.values().get(prior_id.0 as usize).ok_or(
                ProgramGraphAnalysisError::StateTransition {
                    value: value.id,
                    prior: prior_id,
                },
            )?;
            let producer = value
                .producer
                .ok_or(ProgramGraphAnalysisError::StateTransition {
                    value: value.id,
                    prior: prior_id,
                })?;
            let same_retained = prior.contract == value.contract
                && value.contract.lifetime == ValueLifetime::Retained;
            let caller_output_transition = prior.contract.lifetime == ValueLifetime::Retained
                && value.contract.lifetime == ValueLifetime::Output
                && prior.contract.dtype == value.contract.dtype
                && prior.contract.shape == value.contract.shape
                && prior.contract.access == value.contract.access
                && graph
                    .nodes()
                    .get(producer.0 as usize)
                    .and_then(|node| {
                        node.output_ports
                            .iter()
                            .find(|port| port.name == value.name)
                            .and_then(|port| {
                                node.program
                                    .buffers()
                                    .iter()
                                    .find(|buffer| buffer.name() == port.buffer)
                            })
                    })
                    .is_some_and(|buffer| buffer.is_output());
            if (!same_retained && !caller_output_transition) || !prior.consumers.contains(&producer)
            {
                return Err(ProgramGraphAnalysisError::StateTransition {
                    value: value.id,
                    prior: prior_id,
                });
            }
        }
    }

    for (position, node) in graph.nodes().iter().enumerate() {
        let position =
            u32::try_from(position).map_err(|_| ProgramGraphAnalysisError::IdentityOverflow)?;
        if node.id != GraphNodeId(position) {
            return Err(ProgramGraphAnalysisError::NodeIdentity {
                position,
                actual: node.id,
            });
        }
        for input in &node.inputs {
            let value = graph.values().get(input.value.0 as usize).ok_or(
                ProgramGraphAnalysisError::MissingInput {
                    node: node.id,
                    value: input.value,
                },
            )?;
            if value.producer.is_some_and(|producer| producer >= node.id) {
                return Err(ProgramGraphAnalysisError::NonTopologicalInput {
                    node: node.id,
                    value: input.value,
                    producer: value.producer.unwrap_or(node.id),
                });
            }
            if !value.consumers.contains(&node.id) {
                return Err(ProgramGraphAnalysisError::MissingConsumer {
                    value: input.value,
                    node: node.id,
                });
            }
            if !input_contract_matches(&value.contract, &input.contract) {
                return Err(ProgramGraphAnalysisError::InputContract {
                    node: node.id,
                    value: input.value,
                });
            }
        }
        if node.outputs.len() != node.output_ports.len() {
            return Err(ProgramGraphAnalysisError::OutputArity {
                node: node.id,
                ids: node.outputs.len(),
                ports: node.output_ports.len(),
            });
        }
        for (id, port) in node.outputs.iter().zip(&node.output_ports) {
            let value = graph.values().get(id.0 as usize).ok_or(
                ProgramGraphAnalysisError::OutputContract {
                    node: node.id,
                    value: *id,
                },
            )?;
            if value.producer != Some(node.id)
                || value.name != port.name
                || value.contract != port.contract
                || value.retained_successor_of != port.retained_successor_of
            {
                return Err(ProgramGraphAnalysisError::OutputContract {
                    node: node.id,
                    value: *id,
                });
            }
        }
    }
    Ok(())
}

fn input_contract_matches(actual: &ValueContract, expected: &ValueContract) -> bool {
    actual.dtype == expected.dtype
        && actual.shape == expected.shape
        && actual.lifetime == expected.lifetime
}
