//! The published call shapes of a CSR closure.
//!
//! Iterating a one-step CSR traversal to a fixpoint is one algorithm with
//! several call shapes: allocate the two frontier buffers or borrow the
//! caller's, observe every attempted step or not, and report a malformed graph
//! or panic on it. Only the hooked driver carries semantics; the outer shapes
//! are argument plumbing. Each closure op used to retype that plumbing, which
//! is how one op gains a bound, an attribute, or a fix that the others miss.
//!
//! Every shape takes [`crate::graph::csr_closure_inputs::CsrClosureInputs`] plus
//! the seed frontier, so the argument list is two names instead of seven
//! positions and no pair of CSR slices can transpose at a call site.
//!
//! Every macro takes the documentation for each shape it publishes, so an op
//! keeps documenting its own closure while the argument list is stated once.
//!
//! [`define_csr_closure_entry_points`] is exported because a composition that
//! observes a primitive closure publishes the same two shapes over the same
//! argument list. A second copy of that list in a consuming crate is how a
//! reference facade drifts into a second implementation of the fixpoint it is
//! only supposed to count.

/// Publish the allocating and borrowing shapes of an infallible CSR closure
/// over `hooked`, the driver that owns the fixpoint.
///
/// `step_hook` is the observer the borrowing shape passes to `hooked`: a no-op
/// for a primitive's own reference, a counter for a composition facade. It is
/// expanded inside the generated body, so a consuming crate must spell it with
/// fully qualified paths.
#[macro_export]
macro_rules! define_csr_closure_entry_points {
    (
        allocating: $alloc:ident { $(#[$alloc_doc:meta])* },
        borrowing: $into:ident { $(#[$into_doc:meta])* },
        hooked: $hooked:path,
        step_hook: $step_hook:expr,
    ) => {
        $(#[$alloc_doc])*
        #[must_use]
        #[cfg(any(test, feature = "cpu-parity"))]
        pub fn $alloc(
            inputs: $crate::graph::csr_closure_inputs::CsrClosureInputs<'_>,
            seed: &[u32],
        ) -> Vec<u32> {
            let mut current = Vec::new();
            let mut next = Vec::new();
            $into(inputs, seed, &mut current, &mut next);
            current
        }

        $(#[$into_doc])*
        #[cfg(any(test, feature = "cpu-parity"))]
        pub fn $into(
            inputs: $crate::graph::csr_closure_inputs::CsrClosureInputs<'_>,
            seed: &[u32],
            current: &mut Vec<u32>,
            next: &mut Vec<u32>,
        ) {
            $hooked(inputs, seed, current, next, $step_hook);
        }
    };
}

pub(crate) use define_csr_closure_entry_points;

/// Publish the allocating and borrowing shapes of a fallible CSR closure over
/// `hooked`, the driver that owns the fixpoint.
macro_rules! define_try_csr_closure_entry_points {
    (
        allocating: $alloc:ident { $(#[$alloc_doc:meta])* },
        borrowing: $into:ident { $(#[$into_doc:meta])* },
        hooked: $hooked:ident,
        step_hook: $step_hook:expr,
    ) => {
        $(#[$alloc_doc])*
        #[cfg(any(test, feature = "cpu-parity"))]
        pub fn $alloc(
            inputs: $crate::graph::csr_closure_inputs::CsrClosureInputs<'_>,
            seed: &[u32],
        ) -> Result<Vec<u32>, String> {
            let mut current = Vec::new();
            let mut next = Vec::new();
            $into(inputs, seed, &mut current, &mut next)?;
            Ok(current)
        }

        $(#[$into_doc])*
        #[cfg(any(test, feature = "cpu-parity"))]
        pub fn $into(
            inputs: $crate::graph::csr_closure_inputs::CsrClosureInputs<'_>,
            seed: &[u32],
            current: &mut Vec<u32>,
            next: &mut Vec<u32>,
        ) -> Result<(), String> {
            $hooked(inputs, seed, current, next, $step_hook)
        }
    };
}

pub(crate) use define_try_csr_closure_entry_points;

/// Publish the panicking mirror of a fallible CSR closure trio.
///
/// A malformed graph is a caller contract violation, so these shapes exist for
/// callers that would only unwrap. `diagnostic` prefixes the message the
/// fallible arm produced.
macro_rules! define_panicking_csr_closure_entry_points {
    (
        allocating: $alloc:ident from $try_alloc:ident { $(#[$alloc_doc:meta])* },
        borrowing: $into:ident from $try_into:ident { $(#[$into_doc:meta])* },
        hooked: $hooked:ident from $try_hooked:ident { $(#[$hooked_doc:meta])* },
        diagnostic: $label:literal,
        hook_bound: $($bound:tt)*
    ) => {
        $(#[$alloc_doc])*
        #[must_use]
        #[cfg(any(test, feature = "cpu-parity"))]
        pub fn $alloc(
            inputs: $crate::graph::csr_closure_inputs::CsrClosureInputs<'_>,
            seed: &[u32],
        ) -> Vec<u32> {
            $try_alloc(inputs, seed).unwrap_or_else(|err| panic!("{} {err}", $label))
        }

        $(#[$into_doc])*
        #[cfg(any(test, feature = "cpu-parity"))]
        pub fn $into(
            inputs: $crate::graph::csr_closure_inputs::CsrClosureInputs<'_>,
            seed: &[u32],
            current: &mut Vec<u32>,
            next: &mut Vec<u32>,
        ) {
            $try_into(inputs, seed, current, next)
                .unwrap_or_else(|err| panic!("{} {err}", $label));
        }

        $(#[$hooked_doc])*
        #[cfg(any(test, feature = "cpu-parity"))]
        pub fn $hooked<F>(
            inputs: $crate::graph::csr_closure_inputs::CsrClosureInputs<'_>,
            seed: &[u32],
            current: &mut Vec<u32>,
            next: &mut Vec<u32>,
            mut on_step: F,
        ) where
            F: $($bound)*,
        {
            $try_hooked(inputs, seed, current, next, &mut on_step)
                .unwrap_or_else(|err| panic!("{} {err}", $label));
        }
    };
}

pub(crate) use define_panicking_csr_closure_entry_points;
