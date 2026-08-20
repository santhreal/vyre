//! The one recording `ProgramDispatcher` the resident CSR queue tests use.
//!
//! WHY: the single-step and batch resident suites each kept their own recorder
//! with the same five `ProgramDispatcher` arms and differently spelled fields,
//! so a contract asserted against one recorder said nothing about the other and
//! a change to the resident ABI had two places to land. Both suites include
//! this file with `#[path]`, so the recording behaviour is one definition and
//! the field names mean the same thing in every assertion.
//!
//! It records what the tests assert on and nothing else: the byte length of
//! every resident allocation, the bytes of every plain upload, the handles and
//! grid overrides of every sequenced step, and the order handles are freed in.
//! `dispatch` refuses, because a resident test that reaches the non-resident
//! path is testing the wrong path.

use std::cell::{Cell, RefCell};
use vyre_foundation::ir::Program;
use vyre_foundation::program_dispatch::{
    DispatchError, ProgramDispatcher, ResidentDispatchStep, ResidentReadRange,
};

#[derive(Default)]
pub(super) struct RecordingResidentDispatcher {
    /// Last handle handed out. Handles start at 1 so 0 is never a valid one.
    pub(super) next_handle: Cell<u64>,
    /// Byte length of every `alloc_resident`, in call order.
    pub(super) allocs: RefCell<Vec<usize>>,
    /// Bytes of every `upload_resident_many` payload, in call order.
    pub(super) uploads: RefCell<Vec<Vec<u8>>>,
    /// Uploaded handles per sequenced call.
    pub(super) upload_handles: RefCell<Vec<Vec<u64>>>,
    /// Handles bound by each step of each sequenced call.
    pub(super) step_handles: RefCell<Vec<Vec<Vec<u64>>>>,
    /// Grid override of each step of each sequenced call.
    pub(super) step_grids: RefCell<Vec<Vec<Option<[u32; 3]>>>>,
    /// Freed handles, in the order they were released.
    pub(super) freed: RefCell<Vec<u64>>,
}

impl ProgramDispatcher for RecordingResidentDispatcher {
    fn dispatch(
        &self,
        _program: &Program,
        _inputs: &[Vec<u8>],
        _grid_override: Option<[u32; 3]>,
    ) -> Result<Vec<Vec<u8>>, DispatchError> {
        Err(DispatchError::Rejected(
            "Fix: resident queue tests must not reach the non-resident dispatch path.".to_string(),
        ))
    }

    fn alloc_resident(&self, byte_len: usize) -> Result<u64, DispatchError> {
        self.allocs.borrow_mut().push(byte_len);
        let handle = self.next_handle.get() + 1;
        self.next_handle.set(handle);
        Ok(handle)
    }

    fn upload_resident_many(&self, uploads: &[(u64, &[u8])]) -> Result<(), DispatchError> {
        self.uploads
            .borrow_mut()
            .extend(uploads.iter().map(|(_, bytes)| bytes.to_vec()));
        Ok(())
    }

    fn upload_resident_many_sequence_read_ranges_into(
        &self,
        uploads: &[(u64, &[u8])],
        steps: &[ResidentDispatchStep<'_>],
        read_ranges: &[ResidentReadRange],
        outputs: &mut Vec<Vec<u8>>,
    ) -> Result<(), DispatchError> {
        self.upload_handles
            .borrow_mut()
            .push(uploads.iter().map(|(handle, _)| *handle).collect());
        self.step_handles
            .borrow_mut()
            .push(steps.iter().map(|step| step.handle_ids.to_vec()).collect());
        self.step_grids
            .borrow_mut()
            .push(steps.iter().map(|step| step.grid_override).collect());
        outputs.clear();
        outputs.extend(read_ranges.iter().map(|range| vec![0u8; range.byte_len]));
        Ok(())
    }

    fn free_resident(&self, handle: u64) -> Result<(), DispatchError> {
        self.freed.borrow_mut().push(handle);
        Ok(())
    }
}
