//! Adversarial search for inputs that falsify a law.
//!
//! A law is a claim; a counterexample is the evidence against it. The
//! generator and the record it produces are kept apart from the law taxonomy
//! so a new search strategy does not touch the law definitions.

use alloc::string::String;
use alloc::vec::Vec;

/// Adversarial counterexample generator for validating and falsifying law hypotheses.
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct CounterexampleGenerator {
    /// Generator strategy name.
    pub name: String,
    /// Deterministic pseudo-random seed.
    pub seed: u64,
    /// Maximum sample attempts before certifying absence of counterexamples in search space.
    pub max_attempts: usize,
}

impl CounterexampleGenerator {
    /// Construct a counterexample generator with explicit parameters.
    #[must_use]
    pub fn new(name: impl Into<String>, seed: u64, max_attempts: usize) -> Self {
        Self {
            name: name.into(),
            seed,
            max_attempts,
        }
    }

    /// Construct a canonical deterministic generator with default search parameters.
    #[must_use]
    pub fn deterministic(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            seed: 0x5EED_C0DE,
            max_attempts: 1024,
        }
    }

    /// Generate a deterministic stream of adversarial input tuples for an operation of given arity.
    #[must_use]
    pub fn generate_adversarial_inputs(&self, arity: usize) -> Vec<Vec<u64>> {
        let mut results = Vec::with_capacity(self.max_attempts.min(256));
        let corner_cases: &[u64] = &[
            0,
            1,
            2,
            u64::MAX,
            u64::MAX - 1,
            u32::MAX as u64,
            (u32::MAX as u64) + 1,
            0x8000_0000,
            0x7FFF_FFFF,
            0x5555_5555_5555_5555,
            0xAAAA_AAAA_AAAA_AAAA,
        ];
        if arity == 1 {
            for &c in corner_cases {
                results.push(alloc::vec![c]);
            }
        } else if arity == 2 {
            for &c1 in corner_cases {
                for &c2 in corner_cases {
                    results.push(alloc::vec![c1, c2]);
                    if results.len() >= self.max_attempts {
                        return results;
                    }
                }
            }
        } else {
            let mut tuple = alloc::vec![0u64; arity];
            for &c in corner_cases {
                tuple.fill(c);
                results.push(tuple.clone());
            }
        }

        let mut state = self.seed ^ 0x9E37_79B9_7F4A_7C15;
        while results.len() < self.max_attempts {
            let mut tuple = Vec::with_capacity(arity);
            for _ in 0..arity {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                tuple.push(state);
            }
            results.push(tuple);
        }
        results
    }

    /// Search for a counterexample that falsifies the given predicate `predicate(&inputs) -> bool`.
    /// Returns `Some(LawCounterexample)` if a violating input is found.
    pub fn find_counterexample<F>(
        &self,
        arity: usize,
        mut predicate: F,
    ) -> Option<LawCounterexample>
    where
        F: FnMut(&[u64]) -> bool,
    {
        let inputs_stream = self.generate_adversarial_inputs(arity);
        for inputs in inputs_stream {
            if !predicate(&inputs) {
                return Some(LawCounterexample {
                    description: alloc::format!(
                        "counterexample found by generator `{}` at input {:?}",
                        self.name,
                        inputs
                    ),
                    inputs,
                    observed: None,
                    expected: None,
                });
            }
        }
        None
    }
}

impl Default for CounterexampleGenerator {
    fn default() -> Self {
        Self::deterministic("canonical-adversarial-generator")
    }
}

/// Counterexample discovered by validation or metamorphic testing.
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct LawCounterexample {
    /// Human-readable explanation.
    pub description: String,
    /// Concrete input tuple that falsified the law.
    pub inputs: Vec<u64>,
    /// Observed output value, if applicable.
    pub observed: Option<u64>,
    /// Expected output value, if applicable.
    pub expected: Option<u64>,
}
