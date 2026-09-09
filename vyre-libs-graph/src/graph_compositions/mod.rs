//! Representative multi-domain whole-graph compositions.

mod csr_graph_traversal;
mod dense_neural_pipeline;
mod streaming_parser_pipeline;

pub use csr_graph_traversal::build_csr_graph_traversal_pipeline;
pub use dense_neural_pipeline::build_dense_neural_pipeline;
pub use streaming_parser_pipeline::build_streaming_parser_pipeline;
