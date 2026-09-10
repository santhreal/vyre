//! One-expression entry point into the canonical evaluator.
//!
//! A sweep over `Expr` variants wants one value out of one expression, not a
//! whole dispatch. Routing that through a program means storing the value into
//! a declared buffer, and a buffer coerces the value to its element type, so a
//! predicate that evaluates to [`Value::Bool`] would be read back as a word and
//! the sweep would stop being able to see the difference.
//!
//! The crate answered that with a second expression evaluator over its own
//! invocation and memory types. Two evaluators for one `Expr` is two answers to
//! what a node means, and a differential oracle with two answers cannot say
//! which one a backend must match. This module is an entry point rather than an
//! evaluator: it builds the one-lane state the canonical evaluator already
//! takes and calls [`super::hashmap::eval_expr_public`], so every arm is
//! evaluated exactly once, in one place.

use rustc_hash::FxHashMap;
use vyre_foundation::ir::{Expr, Program};

use crate::oob::Buffer;
use crate::value::Value;
use crate::workgroup::InvocationIds;
use crate::ReferenceError;

use super::hashmap::memory::HashmapMemory;

/// Storage the canonical evaluator reads for a one-expression evaluation.
///
/// Holds declared storage buffers by name. Workgroup-scoped buffers are
/// allocated by the dispatch driver, so a single expression evaluated outside a
/// dispatch has none.
#[derive(Default)]
pub struct ReferenceMemory {
    storage: FxHashMap<String, Buffer>,
}

impl ReferenceMemory {
    /// Storage with no buffers bound.
    #[must_use]
    pub fn empty() -> Self {
        Self {
            storage: FxHashMap::default(),
        }
    }

    /// Bind one storage buffer under `name`.
    #[must_use]
    pub fn with_storage(mut self, name: &str, buffer: Buffer) -> Self {
        self.storage.insert(name.to_string(), buffer);
        self
    }

    /// The bytes currently backing the storage buffer named `name`.
    ///
    /// # Errors
    /// Returns a [`ReferenceError::missing_value`] naming `name` when no buffer
    /// is bound under it, rather than an empty byte vector that reads as a
    /// buffer of zeroes.
    pub fn storage_bytes(&self, name: &str) -> Result<Vec<u8>, ReferenceError> {
        self.storage
            .get(name)
            .map(|buffer| buffer.clone().into_bytes())
            .ok_or_else(|| {
                ReferenceError::missing_value(format!(
                    "no storage buffer `{name}` is bound in this reference memory. Fix: bind it with `ReferenceMemory::with_storage` before reading it back."
                ))
            })
    }
}

/// Evaluate one `Expr` for one invocation through the canonical evaluator.
///
/// `program` supplies the entry node slice the lane's frame stack starts on, so
/// an expression that resolves a callee or a buffer sees the same program the
/// canonical dispatch would.
///
/// # Errors
/// Every failure the canonical evaluator returns for that expression, and a
/// [`ReferenceError::budget_exhaustion`] when the expression exceeds the armed
/// work ceiling.
pub fn reference_eval_expr(
    program: &Program,
    memory: &mut ReferenceMemory,
    ids: InvocationIds,
    expr: &Expr,
) -> Result<Value, ReferenceError> {
    let budget = super::step_budget::arm(program);
    let mut hashmap_memory = HashmapMemory::new(std::mem::take(&mut memory.storage));
    let result = super::hashmap::eval_expr_public(expr, ids, program.entry(), &mut hashmap_memory);
    memory.storage = hashmap_memory.into_storage();
    drop(budget);
    result
}
