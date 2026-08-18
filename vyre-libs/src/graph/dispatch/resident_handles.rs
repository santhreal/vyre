//! Shared resident-handle utilities for graph dispatch wrappers.

use vyre_foundation::program_dispatch::{DispatchError, ProgramDispatcher};

/// Free each resident handle at most once while still attempting every unique
/// handle after the first backend failure.
pub(crate) fn free_unique_resident_handles(
    dispatcher: &dyn ProgramDispatcher,
    handles: &[u64],
    _context: &'static str,
) -> Result<(), DispatchError> {
    let mut seen = Vec::with_capacity(handles.len());
    let mut first_err = None;
    for &handle in handles {
        if seen.contains(&handle) {
            continue;
        }
        seen.push(handle);
        if let Err(err) = dispatcher.free_resident(handle) {
            if first_err.is_none() {
                first_err = Some(err);
            }
        }
    }
    match first_err {
        Some(err) => Err(err),
        None => Ok(()),
    }
}

macro_rules! impl_resident_graph_accessors {
    ($graph:ident) => {
        impl $graph {
            /// Number of graph nodes.
            #[must_use]
            pub fn node_count(&self) -> u32 {
                self.node_count
            }

            /// Number of physical or logical CSR edges.
            #[must_use]
            pub fn edge_count(&self) -> u32 {
                self.edge_count
            }

            /// Largest CSR row degree.
            #[must_use]
            pub fn max_row_degree(&self) -> u32 {
                self.max_row_degree
            }

            /// Number of rows at or above the resident mixed-split high-degree threshold.
            #[must_use]
            pub fn high_degree_source_count(&self) -> u32 {
                self.high_degree_source_count
            }

            /// Number of u32 words in each frontier bitset.
            #[must_use]
            pub fn words(&self) -> usize {
                self.words
            }
        }
    };
}
pub(crate) use impl_resident_graph_accessors;

macro_rules! impl_resident_graph_free {
    ($graph:ident, $label:literal) => {
        impl $graph {
            /// Free graph-resident buffers.
            ///
            /// # Errors
            ///
            /// Returns the first backend free failure after attempting all handles.
            pub fn free(
                self,
                dispatcher: &dyn vyre_foundation::program_dispatch::ProgramDispatcher,
            ) -> Result<(), vyre_foundation::program_dispatch::DispatchError> {
                $crate::graph::dispatch::resident_handles::free_unique_resident_handles(
                    dispatcher,
                    &self.handles,
                    $label,
                )
            }
        }
    };
}
pub(crate) use impl_resident_graph_free;

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;
    use vyre_foundation::ir::Program;

    #[derive(Default)]
    struct RecordingFreeDispatcher {
        freed: RefCell<Vec<u64>>,
        fail_on: Option<u64>,
    }

    impl ProgramDispatcher for RecordingFreeDispatcher {
        fn dispatch(
            &self,
            _program: &Program,
            _inputs: &[Vec<u8>],
            _grid_override: Option<[u32; 3]>,
        ) -> Result<Vec<Vec<u8>>, DispatchError> {
            Err(DispatchError::Rejected(
                "Fix: resident handle tests should not dispatch programs.".to_string(),
            ))
        }

        fn free_resident(&self, handle: u64) -> Result<(), DispatchError> {
            self.freed.borrow_mut().push(handle);
            if self.fail_on == Some(handle) {
                return Err(DispatchError::BackendError(format!(
                    "Fix: injected resident free failure for handle {handle}."
                )));
            }
            Ok(())
        }
    }

    #[test]
    fn generated_free_unique_resident_handles_dedupes_and_preserves_order() {
        let dispatcher = RecordingFreeDispatcher::default();

        free_unique_resident_handles(&dispatcher, &[7, 9, 7, 11, 9], "test graph")
            .expect("Fix: deduped resident handle free should succeed");

        assert_eq!(dispatcher.freed.borrow().as_slice(), &[7, 9, 11]);
    }

    #[test]
    fn generated_free_unique_resident_handles_attempts_after_first_failure() {
        let dispatcher = RecordingFreeDispatcher {
            freed: RefCell::new(Vec::new()),
            fail_on: Some(9),
        };

        let error = free_unique_resident_handles(&dispatcher, &[7, 9, 11], "test graph")
            .expect_err("Fix: injected resident handle free failure must surface");

        assert!(
            error.to_string().contains("handle 9"),
            "first resident free error must be returned"
        );
        assert_eq!(dispatcher.freed.borrow().as_slice(), &[7, 9, 11]);
    }
}
