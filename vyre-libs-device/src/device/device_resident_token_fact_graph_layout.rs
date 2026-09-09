//! Device-resident token/fact graph layout, allocation envelopes, and out-degree profiling.

use vyre_foundation::composition::wrap_anonymous_region;
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};

use super::*;

/// Compute the resident byte envelope and out-degree skew profile for a graph.
pub fn plan_device_resident_token_fact_graph_layout(
    graph: &DeviceResidentTokenFactGraph,
    node_record_bytes: u64,
    edge_record_bytes: u64,
) -> Result<DeviceResidentTokenFactGraphLayout, DeviceResidentTokenFactGraphError> {
    let mut scratch = DeviceResidentTokenFactGraphScratch::new();
    plan_device_resident_token_fact_graph_layout_with_scratch(
        graph,
        node_record_bytes,
        edge_record_bytes,
        &mut scratch,
    )
}

/// Compute the resident byte envelope reusing caller-owned profiling scratch.
pub fn plan_device_resident_token_fact_graph_layout_with_scratch(
    graph: &DeviceResidentTokenFactGraph,
    node_record_bytes: u64,
    edge_record_bytes: u64,
    scratch: &mut DeviceResidentTokenFactGraphScratch,
) -> Result<DeviceResidentTokenFactGraphLayout, DeviceResidentTokenFactGraphError> {
    if node_record_bytes == 0 {
        return Err(DeviceResidentTokenFactGraphError::ZeroRecordWidth {
            field: "node_record_bytes",
        });
    }
    if edge_record_bytes == 0 {
        return Err(DeviceResidentTokenFactGraphError::ZeroRecordWidth {
            field: "edge_record_bytes",
        });
    }
    let node_count = u64::try_from(graph.node_ids.len()).map_err(|_| {
        DeviceResidentTokenFactGraphError::ByteCountOverflow {
            field: "node count",
        }
    })?;
    let edge_count = u64::try_from(graph.column_indices.len()).map_err(|_| {
        DeviceResidentTokenFactGraphError::ByteCountOverflow {
            field: "edge count",
        }
    })?;
    let (max_out_degree, top_out_degree_prefix_sums) =
        csr_out_degree_profile(graph, edge_count, &mut scratch.row_degrees)?;
    let node_bytes = checked_mul(node_count, node_record_bytes, "node bytes")?;
    let edge_bytes = checked_mul(edge_count, edge_record_bytes, "edge bytes")?;
    let resident_without_payload = checked_add(node_bytes, edge_bytes, "node plus edge bytes")?;
    let resident_bytes = checked_add(
        resident_without_payload,
        graph.payload_bytes,
        "resident bytes",
    )?;

    Ok(DeviceResidentTokenFactGraphLayout {
        node_count,
        edge_count,
        max_out_degree,
        top_out_degree_prefix_sums,
        node_record_bytes,
        edge_record_bytes,
        node_bytes,
        edge_bytes,
        payload_bytes: graph.payload_bytes,
        resident_bytes,
    })
}

/// Canonical op id for device-resident token/fact graph traversal.
pub const OP_ID: &str = "vyre-libs::device::device_resident_token_fact_graph";

/// Build a Program that dispatches device-resident token/fact graph traversal.
pub fn device_resident_token_fact_graph_program(
    nodes: &[TokenFactNode],
    edges: &[TokenFactEdge],
    payload_bytes: u64,
    node_record_bytes: u64,
    edge_record_bytes: u64,
    row_offsets_buf: &str,
    column_indices_buf: &str,
    out_buf: &str,
) -> Result<Program, DeviceResidentTokenFactGraphError> {
    let graph = plan_device_resident_token_fact_graph(nodes, edges, payload_bytes)?;
    let layout =
        plan_device_resident_token_fact_graph_layout(&graph, node_record_bytes, edge_record_bytes)?;
    let node_count = u32::try_from(layout.node_count)
        .map_err(|_| DeviceResidentTokenFactGraphError::CsrIndexOverflow)?;
    let edge_count = u32::try_from(layout.edge_count)
        .map_err(|_| DeviceResidentTokenFactGraphError::CsrIndexOverflow)?;
    let t = Expr::LogicalIndex { axis: 0 };
    let body = vec![Node::store(
        out_buf,
        t.clone(),
        Expr::load(column_indices_buf, t.clone()),
    )];
    Ok(Program::wrapped(
        vec![
            BufferDecl::storage(row_offsets_buf, 0, BufferAccess::ReadOnly, DataType::U32)
                .with_count(node_count.saturating_add(1)),
            BufferDecl::storage(column_indices_buf, 1, BufferAccess::ReadOnly, DataType::U32)
                .with_count(edge_count),
            BufferDecl::storage(out_buf, 2, BufferAccess::ReadWrite, DataType::U32)
                .with_count(edge_count),
        ],
        [256, 1, 1],
        vec![wrap_anonymous_region(
            OP_ID,
            vec![Node::if_then(Expr::lt(t, Expr::u32(edge_count)), body)],
        )],
    ))
}

fn checked_mul(
    left: u64,
    right: u64,
    field: &'static str,
) -> Result<u64, DeviceResidentTokenFactGraphError> {
    left.checked_mul(right)
        .ok_or(DeviceResidentTokenFactGraphError::ByteCountOverflow { field })
}

fn checked_add(
    left: u64,
    right: u64,
    field: &'static str,
) -> Result<u64, DeviceResidentTokenFactGraphError> {
    left.checked_add(right)
        .ok_or(DeviceResidentTokenFactGraphError::ByteCountOverflow { field })
}

/// Validate the CSR rows and report the top out-degrees the profile needs.
///
/// Degrees are collected once into a reusable `u32` buffer, and only when a
/// graph has more rows than the highest reported rank is the buffer reduced to
/// that prefix. Sorting the retained prefix is then the whole cost. A heap kept
/// per row instead paid a sift for every row of every graph.
fn csr_out_degree_profile(
    graph: &DeviceResidentTokenFactGraph,
    edge_count: u64,
    row_degrees: &mut Vec<u32>,
) -> Result<(u64, [u64; TOKEN_FACT_DEGREE_PROFILE_BUCKETS]), DeviceResidentTokenFactGraphError> {
    let expected_row_offsets = graph.node_ids.len().checked_add(1).ok_or(
        DeviceResidentTokenFactGraphError::ByteCountOverflow {
            field: "row offset count",
        },
    )?;
    if graph.row_offsets.len() != expected_row_offsets {
        return Err(DeviceResidentTokenFactGraphError::InvalidCsrShape {
            field: "row_offsets length",
        });
    }
    let declared_edges = u64::from(*graph.row_offsets.last().ok_or(
        DeviceResidentTokenFactGraphError::InvalidCsrShape {
            field: "row_offsets terminator",
        },
    )?);
    if declared_edges != edge_count {
        return Err(DeviceResidentTokenFactGraphError::InvalidCsrShape {
            field: "row_offsets edge count",
        });
    }

    row_degrees.clear();
    row_degrees.reserve(graph.node_ids.len());
    let mut max_out_degree = 0_u32;
    for row in graph.row_offsets.windows(2) {
        let start = row[0];
        let end = row[1];
        if end < start {
            return Err(DeviceResidentTokenFactGraphError::InvalidCsrShape {
                field: "row_offsets ordering",
            });
        }
        let degree = end - start;
        max_out_degree = max_out_degree.max(degree);
        row_degrees.push(degree);
    }
    if row_degrees.len() > TOKEN_FACT_DEGREE_PROFILE_MAX_RANK {
        row_degrees.select_nth_unstable_by(TOKEN_FACT_DEGREE_PROFILE_MAX_RANK - 1, |lhs, rhs| {
            rhs.cmp(lhs)
        });
        row_degrees.truncate(TOKEN_FACT_DEGREE_PROFILE_MAX_RANK);
    }
    row_degrees.sort_unstable_by(|lhs, rhs| rhs.cmp(lhs));

    let mut prefix_sum = 0_u64;
    let mut prefix_sums = [0_u64; TOKEN_FACT_DEGREE_PROFILE_BUCKETS];
    let mut bucket = 0_usize;
    for (index, degree) in row_degrees.iter().enumerate() {
        prefix_sum = checked_add(prefix_sum, u64::from(*degree), "top out-degree prefix sum")?;
        let rank = u64::try_from(index + 1).map_err(|_| {
            DeviceResidentTokenFactGraphError::ByteCountOverflow {
                field: "out-degree profile rank",
            }
        })?;
        while bucket < TOKEN_FACT_DEGREE_PROFILE_BUCKETS
            && rank >= TOKEN_FACT_DEGREE_PROFILE_RANKS[bucket]
        {
            prefix_sums[bucket] = prefix_sum;
            bucket += 1;
        }
    }
    while bucket < TOKEN_FACT_DEGREE_PROFILE_BUCKETS {
        prefix_sums[bucket] = prefix_sum;
        bucket += 1;
    }
    Ok((u64::from(max_out_degree), prefix_sums))
}
