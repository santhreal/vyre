//! # vyre-libs: curated consumer facade for the composition library family
//!
//! Each public function returns a [`vyre_foundation::ir::Program`] built from
//! existing IR. Compositions are partitioned into substantive domain packages
//! under the `vyre-libs` ownership family.

pub use vyre_libs_builder::prelude;
pub use vyre_libs_builder::builder;
pub use vyre_libs_builder::plumbing;
pub use vyre_libs_builder::plumbing::host::dispatch_buffers;

pub use vyre_libs_builder::builder::*;
pub use vyre_libs_builder::plumbing::registration::{contracts, operation_catalog};
pub use vyre_libs_builder::plumbing::registration::signatures::*;
#[cfg(feature = "telemetry")]
pub use vyre_libs_builder::plumbing::host::telemetry;

/// Ensure all feature-selected library operation registrations are retained by the linker.
#[must_use]
pub fn link_anchor() -> usize {
    let mut count = vyre_libs_builder::link_anchor();
    #[cfg(feature = "bitset")]
    { count += vyre_libs_bitset::link_anchor(); }
    #[cfg(feature = "reduce")]
    { count += vyre_libs_reduce::link_anchor(); }
    #[cfg(feature = "fixpoint")]
    { count += vyre_libs_fixpoint::link_anchor(); }
    #[cfg(any(feature = "math", feature = "math-kernels", feature = "math-dialect"))]
    { count += vyre_libs_math::link_anchor(); }
    #[cfg(any(feature = "nn", feature = "nn-kernels"))]
    { count += vyre_libs_nn::link_anchor(); }
    #[cfg(feature = "graph")]
    { count += vyre_libs_graph::link_anchor(); }
    #[cfg(any(feature = "pattern", feature = "pattern-kernels"))]
    { count += vyre_libs_pattern::link_anchor(); }
    #[cfg(feature = "hash")]
    { count += vyre_libs_hash::link_anchor(); }
    #[cfg(feature = "text")]
    { count += vyre_libs_text::link_anchor(); }
    #[cfg(feature = "decode")]
    { count += vyre_libs_decode::link_anchor(); }
    #[cfg(any(feature = "parsing", feature = "parsing-kernels"))]
    { count += vyre_libs_parsing::link_anchor(); }
    #[cfg(feature = "security")]
    { count += vyre_libs_security::link_anchor(); }
    #[cfg(feature = "visual")]
    { count += vyre_libs_visual::link_anchor(); }
    #[cfg(feature = "rule")]
    { count += vyre_libs_rule::link_anchor(); }
    #[cfg(feature = "vfs")]
    { count += vyre_libs_vfs::link_anchor(); }
    #[cfg(feature = "device")]
    { count += vyre_libs_device::link_anchor(); }
    #[cfg(feature = "solvers")]
    { count += vyre_libs_solvers::link_anchor(); }
    #[cfg(feature = "encoding")]
    { count += vyre_libs_encoding::link_anchor(); }
    #[cfg(feature = "analysis")]
    { count += vyre_libs_analysis::link_anchor(); }
    #[cfg(feature = "reasoning")]
    { count += vyre_libs_reasoning::link_anchor(); }
    #[cfg(feature = "scheduling")]
    { count += vyre_libs_scheduling::link_anchor(); }
    count
}

#[cfg(any(feature = "math", feature = "math-kernels", feature = "math-dialect"))]
pub use vyre_libs_math::math;
#[cfg(feature = "geom")]
pub use vyre_libs_math::geom;
#[cfg(feature = "opt")]
pub use vyre_libs_math::opt;
#[cfg(feature = "representation")]
pub use vyre_libs_math::representation;

#[cfg(any(feature = "nn", feature = "nn-kernels"))]
pub use vyre_libs_nn::nn;
#[cfg(feature = "llm")]
pub use vyre_libs_nn::llm;

#[cfg(feature = "graph")]
pub use vyre_libs_graph::graph;
#[cfg(feature = "topology")]
pub use vyre_libs_graph::topology;
#[cfg(feature = "graph")]
pub use vyre_libs_graph::graph_compositions;

#[cfg(any(feature = "pattern", feature = "pattern-kernels"))]
pub use vyre_libs_pattern::pattern;
#[cfg(feature = "nfa")]
pub use vyre_libs_pattern::nfa;

#[cfg(feature = "hash")]
pub use vyre_libs_hash::hash;
#[cfg(feature = "text")]
pub use vyre_libs_text::text;
#[cfg(feature = "decode")]
pub use vyre_libs_decode::decode;

#[cfg(any(feature = "parsing", feature = "parsing-kernels"))]
pub use vyre_libs_parsing::parsing;

#[cfg(feature = "security")]
pub use vyre_libs_security::security;
#[cfg(feature = "predicate")]
pub use vyre_libs_security::predicate;
#[cfg(feature = "label")]
pub use vyre_libs_security::label;

#[cfg(feature = "visual")]
pub use vyre_libs_visual::visual;
#[cfg(feature = "rule")]
pub use vyre_libs_rule::rule;
#[cfg(feature = "vfs")]
pub use vyre_libs_vfs::vfs;

#[cfg(feature = "bitset")]
pub use vyre_libs_bitset::bitset;
#[cfg(feature = "logical")]
pub use vyre_libs_bitset::logical;
#[cfg(feature = "reduce")]
pub use vyre_libs_reduce::reduce;
#[cfg(feature = "fixpoint")]
pub use vyre_libs_fixpoint::fixpoint;

#[cfg(feature = "device")]
pub use vyre_libs_device::device;
#[cfg(feature = "solvers")]
pub use vyre_libs_solvers::solvers;
#[cfg(feature = "encoding")]
pub use vyre_libs_encoding::encoding;
#[cfg(feature = "analysis")]
pub use vyre_libs_analysis::analysis;
#[cfg(feature = "reasoning")]
pub use vyre_libs_reasoning::reasoning;
#[cfg(feature = "scheduling")]
pub use vyre_libs_scheduling::scheduling;
