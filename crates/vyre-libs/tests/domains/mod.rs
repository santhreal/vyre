//! Domain integration test suites folded into the single vyre-libs test binary.

#[cfg(all(feature = "analysis", feature = "encoding"))]
pub mod analysis;
#[cfg(all(feature = "bitset", feature = "logical"))]
pub mod bitset;
#[cfg(all(
    feature = "builder",
    feature = "builder-ops",
    feature = "cat-a-builder-options",
    feature = "telemetry"
))]
pub mod builder;
#[cfg(all(feature = "decode", feature = "parsing"))]
pub mod decode;
#[cfg(feature = "encoding")]
pub mod encoding;
#[cfg(feature = "fixpoint")]
pub mod fixpoint;
#[cfg(all(
    feature = "graph",
    feature = "topology",
    feature = "graph-dispatch",
    feature = "dense-reachability",
    feature = "impact-mask",
    feature = "interactive-graphics"
))]
pub mod graph;
#[cfg(all(feature = "hash", feature = "crypto-blake3"))]
pub mod hash;
#[cfg(all(
    feature = "math",
    feature = "geom",
    feature = "opt",
    feature = "representation"
))]
pub mod math;
#[cfg(all(
    feature = "nn",
    feature = "nn-inference",
    feature = "nn-linear-4bit",
    feature = "llm"
))]
pub mod nn;
#[cfg(feature = "parsing")]
pub mod parsing;
#[cfg(all(
    feature = "pattern",
    feature = "pattern-nfa",
    feature = "pattern-regex"
))]
pub mod pattern;
#[cfg(feature = "reasoning")]
pub mod reasoning;
#[cfg(feature = "reduce")]
pub mod reduce;
#[cfg(feature = "safetensors")]
pub mod safetensors;
#[cfg(feature = "scheduling")]
pub mod scheduling;
#[cfg(feature = "solvers")]
pub mod solvers;
#[cfg(feature = "text")]
pub mod text;
#[cfg(feature = "visual")]
pub mod visual;
