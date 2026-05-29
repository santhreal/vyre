//! Pipeline orchestration: lex -> parse -> resolve -> typeck -> borrow -> lower.

use vyre_libs::parsing::rust::lex::lexer::plan::RustLexerPlan;

use crate::RustFrontendError;

pub mod borrow_stage;
pub mod lexer_dispatch;
pub mod lower_stage;
pub mod parse_stage;
pub mod resolve_stage;
pub mod typeck_stage;

/// Configuration for the Rust frontend pipeline.
#[derive(Debug, Clone)]
pub struct RustPipelineConfig {
    /// Whether to attempt GPU lexing. Off by default: GPU lexer dispatch is not
    /// wired yet and fails loudly when enabled.
    pub gpu_lex: bool,
    /// Whether to run borrow checking. Off by default: borrow checking is not
    /// wired yet and fails loudly when enabled.
    pub borrow_check: bool,
    /// Whether to lower to Vyre IR. Off by default: lowering is not wired yet
    /// and fails loudly when enabled.
    pub lower: bool,
}

impl Default for RustPipelineConfig {
    fn default() -> Self {
        // The working envelope today is CPU lex + parse. GPU lexing, borrow
        // checking, and lowering are unwired and fail loudly, so they are
        // opt-in: the default pipeline reaches the meaningful boundary (name
        // resolution is not wired) rather than dying at the GPU probe.
        Self {
            gpu_lex: false,
            borrow_check: false,
            lower: false,
        }
    }
}

/// Pipeline state holder. Construct once, compile many files.
pub struct RustPipeline {
    config: RustPipelineConfig,
    lex_plan: RustLexerPlan,
}

impl RustPipeline {
    /// Create a new pipeline.
    pub fn new(config: RustPipelineConfig) -> Self {
        Self {
            config,
            lex_plan: RustLexerPlan::new(),
        }
    }

    /// Run the full pipeline on a single source buffer.
    ///
    /// CPU lex + parse always run. Resolution and type checking are attempted
    /// next; they currently return a loud error until the `vyre-libs` sema
    /// substrate is implemented. Borrow checking and lowering are gated on the
    /// config and likewise fail loudly until wired, so a caller never receives
    /// a success that skipped a requested stage.
    pub fn compile_unit(&self, source: &[u8]) -> Result<CompilationUnit, RustFrontendError> {
        let tokens = self::lexer_dispatch::lex(source, &self.config, &self.lex_plan)?;
        let module = self::parse_stage::parse(source, &tokens)?;
        let resolved = self::resolve_stage::resolve(&module)?;
        let typed = self::typeck_stage::typeck(&resolved)?;
        let verified = if self.config.borrow_check {
            self::borrow_stage::borrow_check(&typed)?
        } else {
            typed
        };
        let program = if self.config.lower {
            Some(self::lower_stage::lower(&verified)?)
        } else {
            None
        };

        Ok(CompilationUnit {
            token_count: tokens.len(),
            module: verified,
            program,
        })
    }
}

/// Result of compiling one translation unit.
#[derive(Debug, Clone)]
pub struct CompilationUnit {
    /// Number of tokens lexed.
    pub token_count: usize,
    /// Parsed and verified module.
    pub module: vyre_libs::parsing::rust::parse::Module,
    /// Lowered Vyre program (if lowering was enabled).
    pub program: Option<vyre::ir::Program>,
}
