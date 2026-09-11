//! One binary for every integration test in this crate.

#[path = "contract_cases/visual_compositions__conformance_semantics.rs"]
mod visual_compositions_conformance_semantics;

#[path = "contract_cases/visual_compositions__cell_grid.rs"]
mod visual_compositions_cell_grid;

#[path = "contract_cases/visual_compositions__default_params.rs"]
mod visual_compositions_default_params;

#[path = "contract_cases/visual_compositions__glyph_grid.rs"]
mod visual_compositions_glyph_grid;

#[path = "contract_cases/visual_compositions__program_has_correct_buffers.rs"]
mod visual_compositions_program_has_correct_buffers;

#[path = "visual_compositions.rs"]
mod visual_compositions;
