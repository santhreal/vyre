//! End-to-end Rust nano-subset compilation pipeline.
//!
//! This is the thin orchestration layer.  All heavy lifting (lexing,
//! parsing, analysis) lives in `vyre-libs` or `weir`.

use vyre::{DispatchConfig, VyreBackend};
use vyre_libs::parsing::rust::lex::lexer::plan::RustLexerPlan;

use crate::RustFrontendError;

/// Configuration for the Rust frontend pipeline.
#[derive(Debug, Clone)]
pub struct RustPipelineConfig {
    /// Whether to attempt GPU lexing.
    pub gpu_lex: bool,
    /// Whether to validate against `rustc_lexer` (oracle mode).
    pub validate: bool,
}

impl Default for RustPipelineConfig {
    fn default() -> Self {
        Self {
            gpu_lex: true,
            validate: cfg!(debug_assertions),
        }
    }
}

/// Pipeline state holder.  Construct once, compile many files.
pub struct RustPipeline {
    config: RustPipelineConfig,
    lex_plan: RustLexerPlan,
}

impl RustPipeline {
    /// Create a new pipeline with the given config.
    pub fn new(config: RustPipelineConfig) -> Self {
        Self {
            config,
            lex_plan: RustLexerPlan::new(),
        }
    }

    /// Compile a single translation unit.
    ///
    /// TODO(v0.0.1): only lex + parse.  v0.1.0 adds name resolution,
    /// type checking, and borrow checking.
    pub fn compile_unit(&self, source: &[u8]) -> Result<(), RustFrontendError> {
        let _ = self.lex_plan.build(); // placeholder: ensure plan is constructible

        let summary = crate::api::parse_rust_bytes(source)?;

        // TODO: lower to Vyre IR
        let _ = summary;

        Ok(())
    }
}
