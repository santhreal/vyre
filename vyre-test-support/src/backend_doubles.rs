//! The dispatch-capable backend double that produces no outputs.
//!
//! A gate that asks which backend selection chose needs a registered backend
//! whose dispatch succeeds, and never asks what it computed. That is one
//! `dispatch_borrowed` returning an empty output list, and it was written out
//! once per suite that needed it: `vyre-driver`'s fixture module and both
//! timing doubles in its resident-sequence unit tests. Each copy restates the
//! dispatch signature, so a change to the trait has to be applied once per
//! copy, and a copy that starts returning values turns a selection gate into a
//! host-arithmetic gate without saying so.
//!
//! A double that needs no other method uses [`NoOutputBackend`]. One that adds
//! timing or resident behaviour writes its own `impl` and expands
//! [`no_output_dispatch_borrowed`] inside it.

use vyre_driver::{BackendError, DispatchConfig, VyreBackend};
use vyre_foundation::ir::Program;

/// The `dispatch_borrowed` a backend double that produces no outputs writes.
///
/// Expands inside an `impl VyreBackend for _` block. The signature names
/// `Program`, `DispatchConfig` and `BackendError` unqualified, because the
/// unit tests inside `vyre-driver` link a different instance of that crate
/// than this one does and a path through here would name incompatible types.
/// The calling module imports the three names it already needs for the trait.
#[macro_export]
macro_rules! no_output_dispatch_borrowed {
    () => {
        fn dispatch_borrowed(
            &self,
            _program: &Program,
            _inputs: &[&[u8]],
            _config: &DispatchConfig,
        ) -> ::core::result::Result<::std::vec::Vec<::std::vec::Vec<u8>>, BackendError> {
            ::core::result::Result::Ok(::std::vec::Vec::new())
        }
    };
}

/// A backend that dispatches successfully and returns no outputs, under `id`.
///
/// Returning no outputs is deliberate: a suite that registers this asks which
/// backend was selected, never what it produced, and a fixture that produced
/// values would be host arithmetic in a driver test.
pub struct NoOutputBackend(pub &'static str);

impl vyre_driver::sealed::Sealed for NoOutputBackend {}

impl VyreBackend for NoOutputBackend {
    fn id(&self) -> &'static str {
        self.0
    }

    no_output_dispatch_borrowed!();
}
