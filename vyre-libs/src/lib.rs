//! # vyre-libs: curated consumer facade for the composition library family
//!
//! Each public function returns a [`vyre_foundation::ir::Program`] built from
//! existing IR. Compositions are partitioned into substantive domain packages
//! under the `vyre-libs` ownership family.

pub use vyre_libs_builder::builder;
pub use vyre_libs_builder::plumbing;
pub use vyre_libs_builder::plumbing::host::dispatch_buffers;
pub use vyre_libs_builder::plumbing::operand::buffer_names;
pub use vyre_libs_builder::plumbing::operand::tensor_ref::{TensorRef, TensorRefError};
pub use vyre_libs_builder::prelude;

pub use vyre_libs_builder::builder::*;
#[cfg(feature = "telemetry")]
pub use vyre_libs_builder::plumbing::host::telemetry;
pub use vyre_libs_builder::plumbing::registration::signatures::*;
pub use vyre_libs_builder::plumbing::registration::{contracts, operation_catalog};

/// Reference every feature-selected domain crate so the linker retains its
/// operation registrations, and report how many library operations the active
/// feature set registers.
///
/// A domain crate's anchor returns nothing. Registrations are link-time records
/// in one process-wide registry, so the count is read from that registry once.
#[must_use]
pub fn link_anchor() -> usize {
    vyre_libs_builder::link_anchor();
    #[cfg(feature = "bitset")]
    vyre_libs_bitset::link_anchor();
    #[cfg(feature = "reduce")]
    vyre_libs_reduce::link_anchor();
    #[cfg(feature = "fixpoint")]
    vyre_libs_fixpoint::link_anchor();
    #[cfg(any(feature = "math", feature = "math-kernels", feature = "math-dialect"))]
    vyre_libs_math::link_anchor();
    #[cfg(any(feature = "nn", feature = "nn-kernels"))]
    vyre_libs_nn::link_anchor();
    #[cfg(feature = "graph")]
    vyre_libs_graph::link_anchor();
    #[cfg(any(feature = "pattern", feature = "pattern-kernels"))]
    vyre_libs_pattern::link_anchor();
    #[cfg(feature = "hash")]
    vyre_libs_hash::link_anchor();
    #[cfg(feature = "text")]
    vyre_libs_text::link_anchor();
    #[cfg(feature = "decode")]
    vyre_libs_decode::link_anchor();
    #[cfg(any(feature = "parsing", feature = "parsing-kernels"))]
    vyre_libs_parsing::link_anchor();
    #[cfg(feature = "security")]
    vyre_libs_security::link_anchor();
    #[cfg(feature = "visual")]
    vyre_libs_visual::link_anchor();
    #[cfg(feature = "rule")]
    vyre_libs_rule::link_anchor();
    #[cfg(feature = "vfs")]
    vyre_libs_vfs::link_anchor();
    #[cfg(feature = "device")]
    vyre_libs_device::link_anchor();
    #[cfg(feature = "solvers")]
    vyre_libs_solvers::link_anchor();
    #[cfg(feature = "encoding")]
    vyre_libs_encoding::link_anchor();
    #[cfg(feature = "analysis")]
    vyre_libs_analysis::link_anchor();
    #[cfg(feature = "reasoning")]
    vyre_libs_reasoning::link_anchor();
    #[cfg(feature = "scheduling")]
    vyre_libs_scheduling::link_anchor();
    operation_catalog::library_entries().count()
}

#[cfg(feature = "geom")]
pub use vyre_libs_math::geom;
#[cfg(any(feature = "math", feature = "math-kernels", feature = "math-dialect"))]
pub use vyre_libs_math::math;
#[cfg(feature = "opt")]
pub use vyre_libs_math::opt;
#[cfg(feature = "representation")]
pub use vyre_libs_math::representation;

#[cfg(feature = "llm")]
pub use vyre_libs_nn::llm;
#[cfg(any(feature = "nn", feature = "nn-kernels"))]
pub use vyre_libs_nn::nn;

#[cfg(feature = "graph")]
pub use vyre_libs_graph::graph;
#[cfg(feature = "graph")]
pub use vyre_libs_graph::graph_compositions;
#[cfg(feature = "topology")]
pub use vyre_libs_graph::topology;

#[cfg(feature = "nfa")]
pub use vyre_libs_pattern::nfa;
#[cfg(any(feature = "pattern", feature = "pattern-kernels"))]
pub use vyre_libs_pattern::pattern;

#[cfg(feature = "decode")]
pub use vyre_libs_decode::decode;
#[cfg(feature = "hash")]
pub use vyre_libs_hash::hash;
#[cfg(feature = "text")]
pub use vyre_libs_text::text;

#[cfg(any(feature = "parsing", feature = "parsing-kernels"))]
pub use vyre_libs_parsing::parsing;

#[cfg(feature = "label")]
pub use vyre_libs_security::label;
#[cfg(feature = "predicate")]
pub use vyre_libs_security::predicate;
#[cfg(feature = "security")]
pub use vyre_libs_security::security;

#[cfg(feature = "rule")]
pub use vyre_libs_rule::rule;
#[cfg(feature = "vfs")]
pub use vyre_libs_vfs::vfs;
#[cfg(feature = "visual")]
pub use vyre_libs_visual::visual;

#[cfg(feature = "bitset")]
pub use vyre_libs_bitset::bitset;
#[cfg(feature = "logical")]
pub use vyre_libs_bitset::logical;
#[cfg(feature = "fixpoint")]
pub use vyre_libs_fixpoint::fixpoint;
#[cfg(feature = "reduce")]
pub use vyre_libs_reduce::reduce;

#[cfg(feature = "analysis")]
pub use vyre_libs_analysis::analysis;
#[cfg(feature = "device")]
pub use vyre_libs_device::device;
#[cfg(feature = "encoding")]
pub use vyre_libs_encoding::encoding;
#[cfg(feature = "reasoning")]
pub use vyre_libs_reasoning::reasoning;
#[cfg(feature = "scheduling")]
pub use vyre_libs_scheduling::scheduling;
#[cfg(feature = "solvers")]
pub use vyre_libs_solvers::solvers;
