//! Comprehensive test suite for vyre-libs visual compositions.
//!
//! Tests validate:
//! - **Identity transforms**: confirm no-op when params are neutral
//! - **Program structure**: verify buffer declarations and region tagging
//! - **Edge cases**: zero-radius, 1-pixel, max-radius
//! - **Algebraic properties**: energy conservation, symmetry, commutativity
//! - **Pixel math correctness**: fixed-point arithmetic, clamp boundaries

#![allow(deprecated)]

#[cfg(feature = "visual")]
#[path = "contract_cases/visual_compositions__cell_grid.rs"]
mod visual_compositions_cell_grid;
#[cfg(feature = "visual")]
#[path = "contract_cases/visual_compositions__default_params.rs"]
mod visual_compositions_default_params;
#[cfg(feature = "visual")]
#[path = "contract_cases/visual_compositions__glyph_grid.rs"]
mod visual_compositions_glyph_grid;
#[cfg(feature = "visual")]
#[path = "contract_cases/visual_compositions__program_has_correct_buffers.rs"]
mod visual_compositions_program_has_correct_buffers;
