//! GPU composition library for vyre IR.
//!
//! Public entry points build validated [`vyre_foundation::ir::Program`] and
//! [`vyre_foundation::ir::ProgramGraph`] compositions across graph, pattern,
//! parsing, text, math, neural, hash, decode, bitset, reduction, fixpoint,
//! visual, device, encoding, analysis, reasoning, scheduling, and solver
//! operations.

pub mod builder;
pub mod plumbing;
pub mod prelude;

pub use plumbing::host::dispatch_buffers;
pub use plumbing::operand::buffer_names;
pub use plumbing::operand::tensor_ref::{TensorRef, TensorRefError};

#[cfg(feature = "telemetry")]
pub use plumbing::host::telemetry;
pub use plumbing::registration::{contracts, operation_catalog};

/// Reference the operation catalog so the linker retains every active domain's
/// `inventory` registrations, and return the count of registered library
/// operations.
#[inline(never)]
#[must_use]
pub fn link_anchor() -> usize {
    let _ = plumbing::registration::operation_catalog::link_anchor();
    operation_catalog::library_entries().count()
}

#[cfg(feature = "geom")]
pub mod geom;
#[cfg(any(feature = "math", feature = "math-kernels", feature = "math-dialect"))]
pub mod math;
#[cfg(feature = "opt")]
pub mod opt;
#[cfg(feature = "representation")]
pub mod representation;

#[cfg(feature = "llm")]
pub mod llm;
#[cfg(any(feature = "nn", feature = "nn-kernels", feature = "safetensors"))]
pub mod nn;

#[cfg(feature = "graph")]
pub mod graph;
#[cfg(feature = "graph")]
pub mod graph_compositions;
#[cfg(feature = "topology")]
pub mod topology;

#[cfg(feature = "nfa")]
pub mod nfa;
#[cfg(any(feature = "pattern", feature = "pattern-kernels"))]
pub mod pattern;

#[cfg(feature = "decode")]
pub mod decode;
#[cfg(feature = "hash")]
pub mod hash;
#[cfg(feature = "text")]
pub mod text;

#[cfg(any(
    feature = "parsing",
    feature = "parsing-kernels",
    feature = "go-parser",
    feature = "python-parser"
))]
pub mod parsing;

#[cfg(feature = "vfs")]
pub mod vfs;
#[cfg(feature = "visual")]
pub mod visual;

#[cfg(feature = "bitset")]
pub mod bitset;
#[cfg(feature = "fixpoint")]
pub mod fixpoint;
#[cfg(feature = "logical")]
pub mod logical;
#[cfg(feature = "reduce")]
pub mod reduce;

#[cfg(feature = "analysis")]
pub mod analysis;
#[cfg(feature = "device")]
pub mod device;
#[cfg(feature = "encoding")]
pub mod encoding;
#[cfg(feature = "reasoning")]
pub mod reasoning;
#[cfg(feature = "scheduling")]
pub mod scheduling;
#[cfg(feature = "solvers")]
pub mod solvers;
