//! WHY: the cooperative tile scaffold is named, not positional. Every body
//! built on it reads `local`, `local_row`, `local_col`, `tile_row_base`,
//! `tile_col_base`, `row` and `col` by name, so dropping or renaming one
//! binding turns into an undeclared-variable failure inside whichever body
//! happens to run, far from the change. This pins the roster and the launch
//! arithmetic the roster is derived from.
//!
//! It does not check the values the bindings take on a device. Parity for
//! that is owned by the tiled matmul contract tests in `vyre-libs-math`.

use vyre_foundation::ir::Node;
use vyre_libs_builder::builder::matrix_tile::{
    bind_output_tile_coordinates, output_tile_shape, padded_tile_lane_count, MatrixShape,
    OutputTileCoordNames, TileShape,
};
use vyre_libs_builder::plumbing::operand::tensor_ref::TensorRefError;

fn bound_names(nodes: &[Node]) -> Vec<String> {
    nodes
        .iter()
        .filter_map(|node| match node {
            Node::Let { name, .. } => Some(name.as_str().to_string()),
            _ => None,
        })
        .collect()
}

#[test]
fn the_tile_scaffold_binds_every_name_a_body_reads() {
    let nodes = bind_output_tile_coordinates(
        MatrixShape {
            m: 40,
            k: 24,
            n: 48,
        },
        TileShape {
            k_tile: 8,
            out_rows: 8,
            out_cols: 16,
            x_lanes: 16,
            y_lanes: 4,
            lanes: 64,
            a_values: 64,
            b_values: 128,
        },
        OutputTileCoordNames {
            lane_row: "local_row",
            lane_col: "local_col",
            row: "row",
            col: "col",
        },
    );

    assert_eq!(
        bound_names(&nodes),
        vec![
            "local",
            "tile_block",
            "tile_cols",
            "tile_row_base",
            "tile_col_base",
            "local_row",
            "local_col",
            "row",
            "col",
        ],
        "the scaffold roster changed; every cooperative body reads these by name"
    );
    assert_eq!(
        nodes.len(),
        9,
        "the scaffold emits bindings only, so a non-binding node is a leak"
    );
}

#[test]
fn a_caller_chooses_the_output_coordinate_names() {
    let nodes = bind_output_tile_coordinates(
        MatrixShape { m: 4, k: 4, n: 4 },
        TileShape {
            k_tile: 4,
            out_rows: 2,
            out_cols: 2,
            x_lanes: 2,
            y_lanes: 1,
            lanes: 4,
            a_values: 8,
            b_values: 8,
        },
        OutputTileCoordNames {
            lane_row: "frag_row",
            lane_col: "frag_col",
            row: "out_row",
            col: "out_col",
        },
    );

    let names = bound_names(&nodes);
    assert_eq!(&names[5..], ["frag_row", "frag_col", "out_row", "out_col"]);
}

#[test]
fn a_degenerate_workgroup_still_owns_one_lane() {
    assert_eq!(output_tile_shape([0, 0, 0]).expect("clamped"), (1, 1, 1));
    assert_eq!(output_tile_shape([16, 4, 2]).expect("shape"), (16, 8, 128));
}

#[test]
fn a_partially_covered_matrix_pads_to_whole_tiles() {
    // 40 rows over 8-row tiles is exact; 47 columns over 16 needs three tiles,
    // so the launch declares 48 columns' worth of lanes.
    assert_eq!(
        padded_tile_lane_count(40, 47, 8, 16, 128).expect("lane count"),
        5 * 3 * 128
    );
}

#[test]
fn a_launch_too_large_for_u32_is_refused_rather_than_wrapped() {
    let error = padded_tile_lane_count(u32::MAX, u32::MAX, 1, 1, 1024)
        .expect_err("the lane count leaves u32");
    assert!(
        matches!(error, TensorRefError::ElementCountOverflow { ref name, .. } if name == "matmul_tiled_launch_lanes"),
        "expected the launch lane count to name itself, got {error:?}"
    );
}
