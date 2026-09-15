//! Deterministic compilation budget accounting for optimizer passes and search.
//!
//! A compile budget bounds CPU-work, pass transform steps, memory allocations,
//! code-size, and measurement steps deterministically. When a budget ceiling is
//! overrun, compilation terminates with an actionable bound failure reporting the
//! budget in the error message, rather than degrading silently or emitting partial output.

use super::pass_result::RefusalReason;

/// Deterministic compilation budget bounds and accounting.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompileBudget {
    /// Maximum allowable CPU-work steps (e.g. node visits, pass invocations).
    pub max_cpu_steps: u64,
    /// Maximum allowable pass transformation / rewrite steps.
    pub max_transform_steps: u64,
    /// Maximum estimated memory allocation bytes.
    pub max_memory_bytes: u64,
    /// Maximum allowable program code-size (node and expr count).
    pub max_code_size: u64,
    /// Maximum measurement or candidate evaluation steps.
    pub max_measurement_steps: u64,

    /// CPU-work steps consumed so far.
    pub consumed_cpu_steps: u64,
    /// Transformation steps applied so far.
    pub consumed_transform_steps: u64,
    /// Estimated memory bytes consumed so far.
    pub consumed_memory_bytes: u64,
    /// Peak code size observed.
    pub consumed_code_size: u64,
    /// Measurement steps consumed so far.
    pub consumed_measurement_steps: u64,
}

impl Default for CompileBudget {
    fn default() -> Self {
        Self::unbounded()
    }
}

impl CompileBudget {
    /// Create an unbounded compilation budget.
    #[must_use]
    pub const fn unbounded() -> Self {
        Self {
            max_cpu_steps: u64::MAX,
            max_transform_steps: u64::MAX,
            max_memory_bytes: u64::MAX,
            max_code_size: u64::MAX,
            max_measurement_steps: u64::MAX,
            consumed_cpu_steps: 0,
            consumed_transform_steps: 0,
            consumed_memory_bytes: 0,
            consumed_code_size: 0,
            consumed_measurement_steps: 0,
        }
    }

    /// Builder: set maximum CPU steps.
    #[must_use]
    pub const fn with_max_cpu_steps(mut self, steps: u64) -> Self {
        self.max_cpu_steps = steps;
        self
    }

    /// Builder: set maximum transformation steps.
    #[must_use]
    pub const fn with_max_transform_steps(mut self, steps: u64) -> Self {
        self.max_transform_steps = steps;
        self
    }

    /// Builder: set maximum memory bytes.
    #[must_use]
    pub const fn with_max_memory_bytes(mut self, bytes: u64) -> Self {
        self.max_memory_bytes = bytes;
        self
    }

    /// Builder: set maximum code size.
    #[must_use]
    pub const fn with_max_code_size(mut self, size: u64) -> Self {
        self.max_code_size = size;
        self
    }

    /// Builder: set maximum measurement steps.
    #[must_use]
    pub const fn with_max_measurement_steps(mut self, steps: u64) -> Self {
        self.max_measurement_steps = steps;
        self
    }

    /// Charge CPU work steps.
    pub fn charge_cpu_steps(&mut self, steps: u64) -> Result<(), RefusalReason> {
        self.consumed_cpu_steps = self.consumed_cpu_steps.saturating_add(steps);
        if self.consumed_cpu_steps > self.max_cpu_steps {
            return Err(RefusalReason::BudgetExceeded {
                resource: "cpu_steps",
                budget: self.max_cpu_steps,
                consumed: self.consumed_cpu_steps,
            });
        }
        Ok(())
    }

    /// Charge transformation steps.
    pub fn charge_transform_steps(&mut self, steps: u64) -> Result<(), RefusalReason> {
        self.consumed_transform_steps = self.consumed_transform_steps.saturating_add(steps);
        if self.consumed_transform_steps > self.max_transform_steps {
            return Err(RefusalReason::BudgetExceeded {
                resource: "transform_steps",
                budget: self.max_transform_steps,
                consumed: self.consumed_transform_steps,
            });
        }
        Ok(())
    }

    /// Charge memory bytes.
    pub fn charge_memory_bytes(&mut self, bytes: u64) -> Result<(), RefusalReason> {
        self.consumed_memory_bytes = self.consumed_memory_bytes.saturating_add(bytes);
        if self.consumed_memory_bytes > self.max_memory_bytes {
            return Err(RefusalReason::BudgetExceeded {
                resource: "memory_bytes",
                budget: self.max_memory_bytes,
                consumed: self.consumed_memory_bytes,
            });
        }
        Ok(())
    }

    /// Check code size.
    pub fn check_code_size(&mut self, size: u64) -> Result<(), RefusalReason> {
        if size > self.consumed_code_size {
            self.consumed_code_size = size;
        }
        if size > self.max_code_size {
            return Err(RefusalReason::BudgetExceeded {
                resource: "code_size",
                budget: self.max_code_size,
                consumed: size,
            });
        }
        Ok(())
    }

    /// Charge measurement steps.
    pub fn charge_measurement_steps(&mut self, steps: u64) -> Result<(), RefusalReason> {
        self.consumed_measurement_steps = self.consumed_measurement_steps.saturating_add(steps);
        if self.consumed_measurement_steps > self.max_measurement_steps {
            return Err(RefusalReason::BudgetExceeded {
                resource: "measurement_steps",
                budget: self.max_measurement_steps,
                consumed: self.consumed_measurement_steps,
            });
        }
        Ok(())
    }
}
