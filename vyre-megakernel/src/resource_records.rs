//! Resource, ABI, and materialization records derived from one validated graph.

use std::collections::BTreeMap;

use vyre_foundation::ir::{BufferAccess, ProgramGraph, ProgramGraphValue, ShapeDim, ValueLifetime};

use crate::error::{failure, overflow, CompileError, CompilerFailureKind};
use crate::identity::{ArtifactNodeId, ArtifactValueId, FusionGroupId};
use crate::schema::{
    AbiAccess, ArtifactAbi, EntryAbiRecord, MaterializationReason, MaterializationRecord,
    ResourceAbiRecord, ResourceEnvelope, ResourceLifetime, ResourceRecord,
};

pub(crate) fn build_abi(graph: &ProgramGraph) -> Result<ArtifactAbi, CompileError> {
    let resources = graph
        .values()
        .iter()
        .map(|value| {
            let access = match value.contract.access.clone() {
                BufferAccess::ReadOnly => AbiAccess::ReadOnly,
                BufferAccess::WriteOnly => AbiAccess::WriteOnly,
                BufferAccess::ReadWrite => AbiAccess::ReadWrite,
                BufferAccess::Uniform => AbiAccess::Uniform,
                unsupported => {
                    return Err(failure(
                        CompilerFailureKind::InvalidProgram,
                        format!("request.graph.values[{}].contract.access", value.id.0),
                        format!("access {unsupported:?} has no artifact ABI representation"),
                        "lower workgroup/private resources inside the node Program",
                    ))
                }
            };
            Ok(ResourceAbiRecord {
                slot: value.id.0,
                value: ArtifactValueId(value.id.0),
                dtype: value.contract.dtype.clone(),
                access,
            })
        })
        .collect::<Result<Vec<_>, CompileError>>()?;
    let entries = graph
        .nodes()
        .iter()
        .map(|node| {
            let mut inputs = Vec::new();
            let mut outputs = Vec::new();
            for buffer in node.program.buffers() {
                let is_in = matches!(
                    buffer.access(),
                    BufferAccess::ReadOnly | BufferAccess::ReadWrite | BufferAccess::Uniform
                );
                let is_out = matches!(
                    buffer.access(),
                    BufferAccess::WriteOnly | BufferAccess::ReadWrite
                ) || buffer.is_output()
                    || buffer.pipeline_live_out;

                if is_in {
                    if let Some(input) = node.inputs.iter().find(|i| i.buffer == buffer.name()) {
                        inputs.push(ArtifactValueId(input.value.0));
                    }
                }
                if is_out {
                    if let Some(pos) = node
                        .output_ports
                        .iter()
                        .position(|o| o.buffer == buffer.name())
                    {
                        if let Some(output_id) = node.outputs.get(pos) {
                            outputs.push(ArtifactValueId(output_id.0));
                        }
                    }
                }
            }
            EntryAbiRecord {
                node: ArtifactNodeId(node.id.0),
                inputs,
                outputs,
            }
        })
        .collect();
    Ok(ArtifactAbi { resources, entries })
}

pub(crate) fn build_materializations(
    graph: &ProgramGraph,
    node_groups: &[FusionGroupId],
    stages: &[u32],
) -> Vec<MaterializationRecord> {
    let mut records = Vec::new();
    for value in graph.values() {
        let Some(producer) = value.producer else {
            continue;
        };
        let producer_node = ArtifactNodeId(producer.0);
        let producer_group = node_groups[producer_node.0 as usize];
        let producer_stage = stages[producer_group.0 as usize];
        let cross_group = value.consumers.iter().any(|consumer| {
            let consumer_node = ArtifactNodeId(consumer.0);
            node_groups[consumer_node.0 as usize] != producer_group
        });
        let reason = match value.contract.lifetime {
            ValueLifetime::Output => Some(MaterializationReason::Output),
            ValueLifetime::Retained => Some(MaterializationReason::Retained),
            _ if cross_group => Some(MaterializationReason::CrossGroupUse),
            _ => None,
        };
        if let Some(reason) = reason {
            records.push(MaterializationRecord {
                value: ArtifactValueId(value.id.0),
                producer: producer_group,
                stage: producer_stage,
                reason,
            });
        }
    }
    records.sort_by_key(|record| (record.value, record.reason as u8));
    records
}

/// Exact element count of one graph value under validated bindings.
fn value_element_count(
    value: &ProgramGraphValue,
    bindings: &BTreeMap<String, u64>,
) -> Result<u64, CompileError> {
    let mut element_count = 1u64;
    for dim in &value.contract.shape {
        let extent = match dim {
            ShapeDim::Known(extent) => *extent,
            ShapeDim::Symbol(symbol) => *bindings.get(symbol).ok_or_else(|| {
                failure(
                    CompilerFailureKind::MissingSymbol,
                    format!("graph.values[{}].shape", value.name),
                    format!("symbolic extent `{symbol}` has no exact binding"),
                    "bind every symbolic graph dimension before compilation",
                )
            })?,
        };
        element_count = element_count.checked_mul(extent).ok_or_else(|| {
            overflow(
                format!("graph.values[{}].shape", value.name),
                "shape element count exceeds u64",
            )
        })?;
    }
    Ok(element_count)
}

/// Exact packed byte length of one graph value under validated bindings.
///
/// The cost model prices materialized traffic in bytes and the resource records
/// report the same number, so both read it here.
pub(crate) fn value_byte_count(
    value: &ProgramGraphValue,
    bindings: &BTreeMap<String, u64>,
) -> Result<u64, CompileError> {
    let element_count = value_element_count(value, bindings)?;
    let host_count = usize::try_from(element_count).map_err(|_| {
        overflow(
            format!("graph.values[{}].shape", value.name),
            "shape element count exceeds addressable packed-size input",
        )
    })?;
    let byte_count = value
        .contract
        .dtype
        .packed_size_bytes(host_count)
        .map_err(|message| overflow(format!("graph.values[{}].dtype", value.name), message))?
        .ok_or_else(|| {
            failure(
                CompilerFailureKind::UnsizedResource,
                format!("graph.values[{}].dtype", value.name),
                "value representation has no fixed packed byte size",
                "resolve the representation to a fixed-width typed value before compilation",
            )
        })?;
    u64::try_from(byte_count).map_err(|_| {
        overflow(
            format!("graph.values[{}]", value.name),
            "packed byte count exceeds u64",
        )
    })
}

pub(crate) fn build_resources(
    graph: &ProgramGraph,
    bindings: &BTreeMap<String, u64>,
    node_groups: &[FusionGroupId],
    stages: &[u32],
) -> Result<(Vec<ResourceRecord>, ResourceEnvelope), CompileError> {
    let final_stage = stages.iter().copied().max().unwrap_or(0);
    let mut resources = Vec::with_capacity(graph.values().len());
    for value in graph.values() {
        let element_count = value_element_count(value, bindings)?;
        let byte_count = value_byte_count(value, bindings)?;
        let producer_stage = value.producer.map_or(0, |producer| {
            stages[node_groups[producer.0 as usize].0 as usize]
        });
        let mut last_stage = value
            .consumers
            .iter()
            .map(|consumer| stages[node_groups[consumer.0 as usize].0 as usize])
            .max()
            .unwrap_or(producer_stage);
        if matches!(
            value.contract.lifetime,
            ValueLifetime::Output | ValueLifetime::Retained
        ) {
            last_stage = last_stage.max(final_stage);
        }
        resources.push(ResourceRecord {
            value: ArtifactValueId(value.id.0),
            name: value.name.clone(),
            element_count,
            byte_count,
            lifetime: match value.contract.lifetime {
                ValueLifetime::Constant => ResourceLifetime::Constant,
                ValueLifetime::Invocation => ResourceLifetime::Invocation,
                ValueLifetime::Retained => ResourceLifetime::Retained,
                ValueLifetime::Output => ResourceLifetime::Output,
            },
            retained_predecessor: value.retained_successor_of.map(|id| ArtifactValueId(id.0)),
            first_stage: producer_stage,
            last_stage,
        });
    }
    resources.sort_by_key(|resource| resource.value);
    let total_bytes = resources.iter().try_fold(0u64, |total, resource| {
        total.checked_add(resource.byte_count).ok_or_else(|| {
            overflow(
                "artifact.resource_envelope.total_bytes",
                "resource sum exceeds u64",
            )
        })
    })?;
    let mut peak_live_bytes = 0u64;
    for stage in 0..=final_stage {
        let live = resources
            .iter()
            .filter(|resource| resource.first_stage <= stage && stage <= resource.last_stage)
            .try_fold(0u64, |total, resource| {
                total.checked_add(resource.byte_count).ok_or_else(|| {
                    overflow(
                        "artifact.resource_envelope.peak_live_bytes",
                        "live resource sum exceeds u64",
                    )
                })
            })?;
        peak_live_bytes = peak_live_bytes.max(live);
    }
    Ok((
        resources,
        ResourceEnvelope {
            total_bytes,
            peak_live_bytes,
        },
    ))
}
