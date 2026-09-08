//! Connected model configurations and reusable ProgramGraph builders.

mod dense_gated_mlp;

pub use dense_gated_mlp::{dense_gated_mlp_graph, DenseGatedMlpError, DenseGatedMlpSpec};
