//! Representative multi-domain whole-graph compositions.
//!
//! BACKLOG row 48 requires representative complete graphs from at least three
//! unrelated domains compiling through the identical production path.
//!
//! - [`dense_neural_pipeline`]: Domain 1 (Dense Numerical / Neural Multi-Layer Linear Graph)
//! - [`csr_graph_traversal`]: Domain 2 (Graph / Irregular / Relational Dataflow)
//! - [`streaming_parser_pipeline`]: Domain 3 (Parsing / Streaming / Security Dataflow)

pub mod csr_graph_traversal;
pub mod dense_neural_pipeline;
pub mod streaming_parser_pipeline;

pub use csr_graph_traversal::build_csr_graph_traversal_pipeline;
pub use dense_neural_pipeline::build_dense_neural_pipeline;
pub use streaming_parser_pipeline::build_streaming_parser_pipeline;
