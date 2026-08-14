//! PTX text emission.
//!
//! Owns the entry point that turns one verified `KernelDescriptor` into PTX
//! module text, and the module tree that does it. Each submodule is named for
//! the PTX concept it owns; none of them owns descriptor shaping, which stays
//! in `vyre-lower`.

mod async_copy;
mod atomic;
mod binop;
mod body;
mod cast;
mod coercion;
mod const_strength_reduction;
mod context;
mod control;
mod dispatch;
mod grid_barrier;
mod memory;
mod mma;
mod module;
mod operand_decode;
mod operand_use_scan;
mod param_identifier;
pub(crate) mod schedule;
mod state;
mod store_guard;
mod subgroup;
mod text_capacity;
mod type_suffix;
mod unop;
mod value_bindings;
mod vector;

use module::ModuleBuilder;
use state::BodyCtx;
use text_capacity::estimated_module_text_capacity;
use vyre_lower::KernelDescriptor;

use crate::{EmitError, PtxEmitOptions};

pub(crate) fn emit_text(
    desc: &KernelDescriptor,
    options: PtxEmitOptions,
) -> Result<String, EmitError> {
    let mut module = ModuleBuilder::new(options, estimated_module_text_capacity(desc));
    module.write_preamble();
    module.write_entry_point(desc)?;
    Ok(module.finish())
}
