//! CPU reference execution contract for operation types.

use crate::ir_inner::model::program::Program;
pub use vyre_spec::CpuFn;

/// CPU reference implementation for an operation.
pub trait CpuOp {
    /// Execute one flat byte payload and append the byte output to `output`.
    fn cpu(input: &[u8], output: &mut Vec<u8>);
}

/// Marker trait for Category A operations with an executable IR program.
pub trait CategoryAOp {
    /// Build the canonical Category A IR program.
    fn program() -> Program;
}

/// Failing CPU adapter for intrinsics whose existing reference accepts structured buffers.
///
/// This is the explicit reference-oracle sentinel for Category C ops whose
/// typed CPU reference is intentionally not exposed through the flat ABI. The
/// function clears the output buffer and returns no flat result. Runtime
/// dispatchers must reject this sentinel through [`is_cpu_reference_sentinel`]
/// before invocation so callers cannot consume an empty byte vector as a valid
/// CPU reference result.
///
/// Each op can register its own CPU ref via `vyre-reference`, and
/// `DialectRegistry::get_lowering(ReferenceBackend)` dispatches to it
/// directly rather than going through this sentinel.
///
/// Category C registrations store this private function through
/// [`SENTINEL_CPU_REF`]; dispatchers refuse that identity before invocation.
#[inline(never)]
fn structured_intrinsic_cpu(input: &[u8], output: &mut Vec<u8>) {
    let _ = input;
    output.clear();
    // Keep this body from compiling to the same instructions as anyone else's.
    //
    // The sentinel's whole identity is its address, so two functions sharing an
    // address are the same function as far as `is_cpu_reference_sentinel` can
    // tell. `output.clear()` is about as generic as a `CpuFn` body gets, and an
    // op with a genuinely trivial CPU reference writes exactly that. Merged
    // onto this address, that op is then read as the sentinel and refused,
    // which is a real op made undispatchable by a coincidence of codegen.
    //
    // Observing a private static the rest of the crate never touches makes the
    // body unique, so nothing can be folded onto it. See BACKLOG.md R75 for the
    // structural fix, which is to stop identifying the sentinel by address.
    std::hint::black_box(&SENTINEL_BODY_MARKER);
}

/// Private, referenced only by the sentinel body, so that body is unique.
static SENTINEL_BODY_MARKER: u8 = 0xA7;

/// The one pointer value every sentinel slot is filled from.
///
/// Reading the sentinel through a `static` rather than naming the function at
/// each site is what makes [`is_cpu_reference_sentinel`] reliable. The address
/// of a function is not guaranteed to be unique: with more than one codegen
/// unit the compiler may materialize a second copy or a local thunk, and
/// `fn_addr_eq` then compares two different addresses for the same function
/// and answers `false`. A `static` holds a single pointer value resolved once,
/// so a producer that stores `SENTINEL_CPU_REF` and a consumer that compares
/// against `SENTINEL_CPU_REF` are comparing the same bits by construction.
///
/// This is not a style preference. The comparison sits in front of a refusal:
/// when it wrongly answers `false` the dispatcher stops refusing and INVOKES
/// the sentinel, which clears the output and returns `Ok(())`, handing the
/// caller an empty byte vector that looks like a successful CPU reference
/// result. That is precisely the fail-open the sentinel exists to prevent.
pub static SENTINEL_CPU_REF: CpuFn = structured_intrinsic_cpu;

/// True when `structured_intrinsic_cpu` is set as an op's CPU lowering.
///
/// Conformance tooling uses this to flag operations that still expose only the
/// structured-reference sentinel, so parity status is recorded explicitly
/// instead of pretending a flat CPU adapter exists.
#[must_use]
pub fn is_cpu_reference_sentinel(f: CpuFn) -> bool {
    std::ptr::fn_addr_eq(f, SENTINEL_CPU_REF)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The sentinel read through the static is recognised as the sentinel.
    ///
    /// This is the identity every producer must use. `LoweringTable::empty()`
    /// stores `SENTINEL_CPU_REF`, and the dispatcher's refusal reads it back
    /// through this predicate, so if the pair ever stops agreeing the refusal
    /// turns into an invocation that returns an empty output as `Ok`.
    #[test]
    fn the_sentinel_static_is_recognised_as_the_sentinel() {
        assert!(is_cpu_reference_sentinel(SENTINEL_CPU_REF));
    }

    /// A pointer round-tripped through a `CpuFn` binding still matches.
    ///
    /// Storing the static in a struct field and reading it back is exactly
    /// what `LoweringTable` does. Copying the value must not change it.
    #[test]
    fn the_sentinel_survives_being_stored_and_read_back() {
        let stored: CpuFn = SENTINEL_CPU_REF;
        let copied = stored;
        assert!(is_cpu_reference_sentinel(copied));
    }

    /// Naming the function directly also matches, within this crate.
    ///
    /// Kept as a canary rather than as the contract: this is the comparison
    /// that is allowed to be unreliable across codegen units, and it is the
    /// form the predicate used to be written in. If it ever fails while
    /// `the_sentinel_static_is_recognised_as_the_sentinel` still passes, that
    /// is confirmation the static indirection is load-bearing, not decoration.
    #[test]
    fn is_cpu_reference_sentinel_detects_structured_intrinsic() {
        assert!(is_cpu_reference_sentinel(structured_intrinsic_cpu));
    }

    #[test]
    fn is_cpu_reference_sentinel_rejects_other_fn() {
        #[allow(clippy::ptr_arg)] // Must match `CpuFn` (`&mut Vec<u8>`), not `&mut [u8]`.
        fn custom_cpu(_input: &[u8], _output: &mut Vec<u8>) {}
        assert!(!is_cpu_reference_sentinel(custom_cpu));
    }

    /// A second empty-bodied CpuFn is not the sentinel.
    ///
    /// The predicate must key on identity, not on the function doing nothing.
    /// An op with a genuinely trivial CPU reference must stay dispatchable.
    #[test]
    fn another_do_nothing_cpu_fn_is_not_the_sentinel() {
        #[allow(clippy::ptr_arg)]
        fn also_clears(_input: &[u8], output: &mut Vec<u8>) {
            output.clear();
        }
        assert!(!is_cpu_reference_sentinel(also_clears));
    }

    /// An empty lowering table carries the sentinel; a populated one does not.
    ///
    /// Ties the producer and the predicate together in one assertion, which is
    /// the pairing the dispatcher's refusal actually depends on.
    #[test]
    fn an_empty_lowering_table_carries_the_sentinel_and_a_populated_one_does_not() {
        #[allow(clippy::ptr_arg)]
        fn real_cpu_ref(input: &[u8], output: &mut Vec<u8>) {
            output.extend_from_slice(input);
        }

        let empty = crate::dispatch::dialect_lookup::LoweringTable::empty();
        assert!(
            is_cpu_reference_sentinel(empty.cpu_ref),
            "LoweringTable::empty must leave the sentinel in the reference slot"
        );

        let populated = crate::dispatch::dialect_lookup::LoweringTable::new(real_cpu_ref);
        assert!(
            !is_cpu_reference_sentinel(populated.cpu_ref),
            "a table built with a real CPU reference must not look like the sentinel"
        );
    }

    #[test]
    fn structured_intrinsic_clears_output_without_flat_result() {
        let mut output = vec![1, 2, 3];
        structured_intrinsic_cpu(b"input", &mut output);
        assert!(output.is_empty());
    }
}
