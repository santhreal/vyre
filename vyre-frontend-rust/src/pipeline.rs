//! Pipeline orchestration: lex → parse → resolve → typeck → borrow → lower.

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
    /// Whether to attempt GPU lexing.
    pub gpu_lex: bool,
    /// Whether to validate against rustc (oracle mode).
    pub validate: bool,
    /// Whether to run borrow checking.
    pub borrow_check: bool,
    /// Whether to lower to Vyre IR.
    pub lower: bool,
}

impl Default for RustPipelineConfig {
    fn default() -> Self {
        Self {
            gpu_lex: true,
            validate: cfg!(debug_assertions),
            borrow_check: true,
            lower: true,
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
    pub fn compile_unit(&self, source: &[u8]) -> Result<CompilationUnit, RustFrontendError> {
        // Stage 1: Lex
        let tokens = self::lexer_dispatch::lex(source, &self.config, &self.lex_plan)?;

        // Stage 2: Parse
        let module = self::parse_stage::parse(source, &tokens)?;

        // Stage 3: Name resolution (placeholder)
        let resolved = self::resolve_stage::resolve(&module)?;

        // Stage 4: Type check (placeholder)
        let typed = self::typeck_stage::typeck(&resolved)?;

        // Stage 5: Borrow check via Weir (placeholder)
        let verified = if self.config.borrow_check {
            self::borrow_stage::borrow_check(&typed)?
        } else {
            typed
        };

        // Stage 6: Lower to Vyre IR (placeholder)
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
