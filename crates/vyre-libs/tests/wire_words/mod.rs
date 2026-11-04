//! Oracle helpers the remaining `vyre-libs` contract tests build their buffers
//! from.
//!
//! `vyre_primitives::wire` owns the little-endian packers and decoders and
//! `vyre_test_support::fixed_point` owns the deterministic word generator, so
//! the two names below are aliases onto those owners rather than a second copy.
//! What is defined here is the pair of multi-domain reference runs the facade
//! still proves: a toposort witness that maps its oracle's error text onto the
//! graph domain's error type, and a full matroid-intersection dispatch that
//! spans the graph and math-kernel domains.
//!
//! This module is compiled once per including test binary, and no binary uses
//! every helper. Each unused-in-this-binary helper is live in a sibling binary,
//! so `dead_code` here reports the inclusion shape rather than an item with no
//! caller. An `expect` cannot state that: the lint fires in some binaries and
//! not others, and the fulfilled half would fail
//! `unfulfilled_lint_expectations`.
#![allow(dead_code, unused_imports)]

use vyre_reference::value::Value;

pub(crate) use vyre_primitives::wire::decode_u32_le_bytes_all as decode_u32_words;
pub(crate) use vyre_primitives::wire::pack_u32_slice as u32_bytes;
pub(crate) use vyre_test_support::fixed_point::xorshift32 as next_u32;

#[cfg(feature = "graph")]
pub(crate) fn toposort(
    node_count: u32,
    edges: &[(u32, u32)],
) -> Result<Vec<u32>, vyre_libs::graph::toposort::ToposortError> {
    for (edge_idx, &(from, to)) in edges.iter().enumerate() {
        if from >= node_count {
            return Err(vyre_libs::graph::toposort::ToposortError::UnknownNode {
                edge: edge_idx,
                node: from,
            });
        }
        if to >= node_count {
            return Err(vyre_libs::graph::toposort::ToposortError::UnknownNode {
                edge: edge_idx,
                node: to,
            });
        }
    }
    vyre_reference::composition_witness::toposort_witness(node_count, edges).map_err(|err| {
        if let Some(rest) = err.strip_prefix("Cycle detected involving node ") {
            if let Ok(node) = rest.parse::<u32>() {
                return vyre_libs::graph::toposort::ToposortError::Cycle { node };
            }
        }
        vyre_libs::graph::toposort::ToposortError::InconsistentState { message: err }
    })
}

#[cfg(all(feature = "math-kernels", feature = "graph"))]
pub(crate) fn matroid_intersection_eval(
    exchange_adj: &[u32],
    sources: &[u32],
    sinks: &[u32],
    set_x: &[u32],
    n: u32,
    max_augmentations: u32,
    min_dispatch: u32,
) -> Vec<u32> {
    use vyre_libs::graph::matroid_intersection_full::matroid_intersection_full;
    use vyre_primitives::wire::{decode_u32_le_bytes_all as unpack, pack_u32_slice as pack};
    let program = matroid_intersection_full(
        "exchange_adj",
        "sources",
        "sinks",
        "set_x",
        "parent",
        "frontier",
        "next_frontier",
        "visited",
        "any_change",
        "path_out",
        "path_len",
        n,
        max_augmentations,
    );
    let zeros_n = vec![0u32; n as usize];
    let zero1 = vec![0u32];
    let outputs = vyre_reference::ReferenceRequest::standard(
        &program,
        &[
            Value::from(pack(exchange_adj)),
            Value::from(pack(sources)),
            Value::from(pack(sinks)),
            Value::from(pack(set_x)),
            Value::from(pack(&zeros_n)),
            Value::from(pack(&zeros_n)),
            Value::from(pack(&zeros_n)),
            Value::from(pack(&zeros_n)),
            Value::from(pack(&zero1)),
            Value::from(pack(&zeros_n)),
            Value::from(pack(&zero1)),
            Value::from(pack(&zero1)),
        ],
    )
    .with_min_dispatch_elements(min_dispatch)
    .outputs()
    .expect("matroid_intersection_full reference evaluation must succeed");
    let index = vyre_reference::output_index(&program, "set_x")
        .expect("matroid_intersection_full must declare output set_x");
    unpack(&outputs[index].to_bytes())[..n as usize].to_vec()
}

#[cfg(feature = "go-parser")]
pub(crate) mod go;
