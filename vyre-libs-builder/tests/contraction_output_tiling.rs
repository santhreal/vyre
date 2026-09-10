//! What a row-batched contraction assigns to one invocation.
//!
//! The row-batched program assigned one output element per invocation and
//! looped the whole contraction dimension, so the physical work split was a
//! constant inside the builder and no device fact could change it. These cases
//! pin the replacement: the output tile is a function of the declared extents
//! and the stated per-invocation register budget, the untiled program stays
//! reachable for facts that admit no tile, and a geometry that has no program
//! for a stated tiling is rejected before any candidate is ranked.
//!
//! What they do not catch: which candidate is faster on a device. That is a
//! measurement, and these cases only pin the operand traffic the two programs
//! declare per output element.

use vyre_foundation::ir::{Expr, Node, Program};
use vyre_foundation::visit::{for_each_expr, for_each_node};
use vyre_libs_builder::builder::gemm::{
    ContractionComposer, ContractionOutputTile, ContractionTiling,
};
use vyre_libs_builder::plumbing::operand::tensor_ref::{TensorRef, TensorRefError};
use vyre_foundation::validate::BackendCapabilities;
use vyre_megakernel::DeviceFacts;
use vyre_spec::DataType;

const ROWS: u32 = 64;
const IN_DIM: u32 = 32;
const OUT_DIM: u32 = 48;

/// Facts stating a per-invocation register budget and a workgroup limit.
fn facts_with_register_budget(registers: u32) -> DeviceFacts {
    DeviceFacts::new(BackendCapabilities::NONE, 256).with_occupancy(registers, 0)
}

/// A row-batched projection over [`ROWS`] x [`IN_DIM`] x [`OUT_DIM`].
fn composer() -> ContractionComposer {
    ContractionComposer::batched_rows(
        "vyre-libs::builder::gemm::row_tiling_case",
        TensorRef::new("x", DataType::F32, vec![ROWS, IN_DIM]),
        TensorRef::new("w", DataType::F32, vec![IN_DIM, OUT_DIM]),
        None,
        TensorRef::new("out", DataType::F32, vec![ROWS, OUT_DIM]),
        ROWS,
        IN_DIM,
        OUT_DIM,
        DataType::F32,
        false,
    )
}

/// Stores one invocation of `program` performs.
fn store_count(program: &Program) -> usize {
    let mut stores = 0;
    for_each_node(&program.entry, |node| {
        if matches!(node, Node::Store { .. }) {
            stores += 1;
        }
    });
    stores
}

/// Loads of `buffer` one contraction step of `program` performs.
fn load_count(program: &Program, buffer: &str) -> usize {
    let mut loads = 0;
    for_each_expr(&program.entry, |expr| {
        if let Expr::Load { buffer: name, .. } = expr {
            if name.as_ref() == buffer {
                loads += 1;
            }
        }
    });
    loads
}

#[test]
fn an_output_tile_is_a_function_of_the_stated_register_budget() {
    // A budget that holds nothing beyond the body's own live scalars admits no
    // tile, so the untiled candidate is what the caller keeps.
    for registers in [0, 1, ContractionOutputTile::register_footprint(1, 1) - 1] {
        assert_eq!(
            ContractionOutputTile::derive(ROWS, OUT_DIM, &facts_with_register_budget(registers)),
            None,
            "Fix: a {registers}-register budget states no room for a tile and must admit none"
        );
    }

    let mut derived = Vec::new();
    for registers in [12, 20, 32, 64, 128, 256] {
        let facts = facts_with_register_budget(registers);
        let tile = ContractionOutputTile::derive(ROWS, OUT_DIM, &facts)
            .unwrap_or_else(|| panic!("Fix: a {registers}-register budget must admit a tile"));

        assert!(
            ContractionOutputTile::register_footprint(tile.rows, tile.columns) <= registers,
            "Fix: {tile:?} must fit the {registers}-register budget it was derived from"
        );
        assert!(
            tile.rows >= ROWS
                || ContractionOutputTile::register_footprint(tile.rows + 1, tile.columns)
                    > registers,
            "Fix: {tile:?} must be maximal in rows for a {registers}-register budget"
        );
        assert!(
            tile.columns >= OUT_DIM
                || ContractionOutputTile::register_footprint(tile.rows, tile.columns + 1)
                    > registers,
            "Fix: {tile:?} must be maximal in columns for a {registers}-register budget"
        );
        derived.push((registers, tile));
    }

    // A tile read off a constant would be the same tile at every budget.
    let distinct: std::collections::BTreeSet<(u32, u32)> = derived
        .iter()
        .map(|(_, tile)| (tile.rows, tile.columns))
        .collect();
    assert!(
        distinct.len() > 1,
        "Fix: the output tile must be derived from the stated budget, not a constant; got {derived:?}"
    );

    // A larger budget never derives a smaller tile.
    for pair in derived.windows(2) {
        let [(low, small), (high, large)] = pair else {
            unreachable!("windows(2) yields pairs")
        };
        assert!(
            ContractionOutputTile::register_footprint(large.rows, large.columns)
                >= ContractionOutputTile::register_footprint(small.rows, small.columns),
            "Fix: a {high}-register budget must not derive a smaller tile than a {low}-register one"
        );
    }
}

#[test]
fn a_declared_extent_bounds_the_tile_it_derives() {
    // A geometry one element wide in both extents has no tile to accumulate,
    // however large the stated budget is.
    assert_eq!(
        ContractionOutputTile::derive(1, 1, &facts_with_register_budget(4096)),
        None,
        "Fix: a single-element output states no tile"
    );

    let tile = ContractionOutputTile::derive(3, 2, &facts_with_register_budget(4096))
        .expect("a 3x2 output under an ample budget must admit a tile");
    assert_eq!(
        (tile.rows, tile.columns),
        (3, 2),
        "Fix: the tile must be bounded by the declared extents, not by the budget alone"
    );
}

#[test]
fn stated_facts_move_the_row_batched_contraction_off_one_output_per_invocation() {
    let untiled = composer()
        .build()
        .expect("the untiled row-batched contraction must build");
    assert_eq!(
        store_count(&untiled),
        1,
        "Fix: the untiled candidate assigns one output element per invocation"
    );
    assert_eq!(load_count(&untiled, "x"), 1);
    assert_eq!(load_count(&untiled, "w"), 1);

    let facts = facts_with_register_budget(64);
    let tile = ContractionOutputTile::derive(ROWS, OUT_DIM, &facts)
        .expect("a 64-register budget must admit a tile");
    let tiled = composer()
        .with_device_facts(facts)
        .build()
        .expect("the register-tiled row-batched contraction must build");

    let outputs_per_invocation = (tile.rows * tile.columns) as usize;
    assert!(
        outputs_per_invocation > 1,
        "Fix: stated facts must move the contraction off one output per invocation"
    );
    assert_eq!(
        store_count(&tiled),
        outputs_per_invocation,
        "Fix: one invocation must store every element of the tile it accumulated"
    );

    // Operand traffic per output element is the whole reason a tile exists: one
    // staged left value serves a tile row and one staged right value serves a
    // tile column, so the tile reads `rows + columns` elements per contraction
    // step where the untiled program reads two per output.
    let tiled_loads = load_count(&tiled, "x") + load_count(&tiled, "w");
    assert_eq!(
        tiled_loads,
        (tile.rows + tile.columns) as usize,
        "Fix: a contraction step must stage one value per tile row and per tile column"
    );
    let untiled_loads_per_output = 2.0;
    let tiled_loads_per_output = tiled_loads as f64 / outputs_per_invocation as f64;
    assert!(
        tiled_loads_per_output < untiled_loads_per_output,
        "Fix: the tile must read fewer operand elements per output element than the untiled program; got {tiled_loads_per_output} against {untiled_loads_per_output}"
    );
}

#[test]
fn facts_that_admit_no_tile_leave_the_untiled_candidate_selected() {
    // A caller with no device states no budget, and the candidate that stands
    // is the one every target can run.
    let neutral = composer()
        .with_device_facts(DeviceFacts::unknown())
        .build()
        .expect("device-neutral facts must still build the row-batched contraction");
    assert_eq!(
        store_count(&neutral),
        1,
        "Fix: facts admitting no tile must leave the untiled candidate selected"
    );

    let untiled = composer().build().expect("the untiled candidate must build");
    assert_eq!(
        neutral, untiled,
        "Fix: device-neutral facts must select exactly the untiled candidate"
    );
}

#[test]
fn every_tiling_states_whether_the_row_batched_geometry_has_a_program_for_it() {
    let tilings = [
        ContractionTiling::Linear {
            workgroup_size: [64, 1, 1],
        },
        ContractionTiling::RegisterTiled {
            rows: 4,
            columns: 4,
            workgroup_size: [64, 1, 1],
        },
        ContractionTiling::CooperativeShared {
            tile: 16,
            a_tile_name: "a_shared".to_string(),
            b_tile_name: "b_shared".to_string(),
        },
        ContractionTiling::Block1D { tile: 8 },
    ];

    for tiling in tilings {
        // The match has no catch-all, so an added tiling fails to compile until
        // a decision is recorded for the row-batched geometry.
        let expected_stores: Option<usize> = match &tiling {
            ContractionTiling::Linear { .. } => Some(1),
            ContractionTiling::RegisterTiled { rows, columns, .. } => {
                Some((rows * columns) as usize)
            }
            ContractionTiling::CooperativeShared { .. } | ContractionTiling::Block1D { .. } => None,
        };

        let built = composer().with_tiling(tiling.clone()).build();

        let Some(expected_stores) = expected_stores else {
            let error = built.expect_err(&format!(
                "Fix: {tiling:?} has no row-batched program and must be rejected"
            ));
            assert!(
                matches!(error, TensorRefError::UnsupportedTiling { .. }),
                "Fix: {tiling:?} must be rejected as an unsupported tiling, got {error}"
            );
            continue;
        };

        let program = built.unwrap_or_else(|error| panic!("Fix: {tiling:?} must build: {error}"));
        assert_eq!(
            store_count(&program),
            expected_stores,
            "Fix: {tiling:?} must store exactly the outputs one invocation accumulates"
        );
    }
}
