//! Hostile compilation quota enforcement and diagnostic redaction.

use serde::{Deserialize, Serialize};

use super::error::SecurityError;

/// Bounded resource quotas for untrusted IR compilation.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Deserialize, Serialize)]
pub struct CompilationQuota {
    /// Maximum allowed IR nodes.
    pub max_nodes: usize,
    /// Maximum AST/expression nesting depth.
    pub max_depth: usize,
    /// Maximum compilation wall-clock duration in nanoseconds.
    pub max_compile_time_ns: u64,
    /// Maximum intermediate memory allocation in bytes.
    pub max_memory_bytes: usize,
}

impl Default for CompilationQuota {
    fn default() -> Self {
        Self {
            max_nodes: 50_000,
            max_depth: 128,
            max_compile_time_ns: 5_000_000_000,  // 5.0 seconds
            max_memory_bytes: 128 * 1024 * 1024, // 128 MB
        }
    }
}

/// Active compiler budget enforcer tracking resource limits during lowering and optimization.
#[derive(Clone, Debug)]
pub struct CompilationBudgetEnforcer {
    quota: CompilationQuota,
    start_time_ns: u64,
    current_nodes: usize,
    current_depth: usize,
}

impl CompilationBudgetEnforcer {
    /// Initialize a budget tracker with given quota and start timestamp.
    pub fn new(quota: CompilationQuota, start_time_ns: u64) -> Self {
        Self {
            quota,
            start_time_ns,
            current_nodes: 0,
            current_depth: 0,
        }
    }

    /// Record processing of a single node, failing if node budget is exceeded.
    pub fn increment_node_count(&mut self) -> Result<(), SecurityError> {
        self.current_nodes += 1;
        if self.current_nodes > self.quota.max_nodes {
            return Err(SecurityError::QuotaExceeded {
                resource: "ir_nodes".to_string(),
                current: self.current_nodes as u64,
                limit: self.quota.max_nodes as u64,
            });
        }
        Ok(())
    }

    /// Push nesting depth scope.
    pub fn enter_scope(&mut self) -> Result<(), SecurityError> {
        self.current_depth += 1;
        if self.current_depth > self.quota.max_depth {
            return Err(SecurityError::QuotaExceeded {
                resource: "ast_depth".to_string(),
                current: self.current_depth as u64,
                limit: self.quota.max_depth as u64,
            });
        }
        Ok(())
    }

    /// Pop nesting depth scope.
    pub fn exit_scope(&mut self) {
        self.current_depth = self.current_depth.saturating_sub(1);
    }

    /// Check elapsed compilation time against budget.
    pub fn check_time(&self, current_time_ns: u64) -> Result<(), SecurityError> {
        let elapsed = current_time_ns.saturating_sub(self.start_time_ns);
        if elapsed > self.quota.max_compile_time_ns {
            return Err(SecurityError::QuotaExceeded {
                resource: "compile_time_ns".to_string(),
                current: elapsed,
                limit: self.quota.max_compile_time_ns,
            });
        }
        Ok(())
    }
}

/// Redactor for diagnostics and traces to prevent secret values or raw pointers from leaking.
pub struct RedactedDiagnostic;

impl RedactedDiagnostic {
    /// Sanitize diagnostic text, redacting potential pointer hex addresses and raw literal tokens.
    pub fn sanitize_text(text: &str) -> String {
        let mut result = String::with_capacity(text.len());
        for word in text.split_whitespace() {
            if !result.is_empty() {
                result.push(' ');
            }
            if word.starts_with("0x")
                && word.len() > 8
                && word[2..].chars().all(|c| c.is_ascii_hexdigit())
            {
                result.push_str("[REDACTED_ADDR]");
            } else if word.starts_with("secret:") || word.starts_with("key:") {
                result.push_str("[REDACTED_SECRET]");
            } else {
                result.push_str(word);
            }
        }
        result
    }
}
