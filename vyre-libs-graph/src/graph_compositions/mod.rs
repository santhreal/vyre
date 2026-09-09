//! Representative multi-domain whole-graph compositions.
//!
//! Three unrelated domains compile through the identical production path:
//! [`build_dense_neural_pipeline`] is dense numerical work,
//! [`build_csr_graph_traversal_pipeline`] is irregular relational dataflow,
//! and [`build_streaming_parser_pipeline`] is streaming parsing.
//!
//! Each domain has its own file and no public path of its own, so a caller
//! writes one name per composition.

mod csr_graph_traversal;
mod dense_neural_pipeline;
#[cfg(feature = "visual")]
mod interactive_graphics_pipeline;
mod streaming_parser_pipeline;

pub use csr_graph_traversal::build_csr_graph_traversal_pipeline;
pub use dense_neural_pipeline::build_dense_neural_pipeline;
#[cfg(feature = "visual")]
pub use interactive_graphics_pipeline::{
    build_interactive_graphics_pipeline, InteractiveGraphicsPipelineParams,
};
pub use streaming_parser_pipeline::build_streaming_parser_pipeline;
