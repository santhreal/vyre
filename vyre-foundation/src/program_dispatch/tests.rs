use super::*;
use std::cell::{Cell, RefCell};

struct RangedReadDispatcher {
    buffers: Vec<(u64, Vec<u8>)>,
    read_calls: Cell<usize>,
    batched_handles: RefCell<Vec<u64>>,
}

impl ProgramDispatcher for RangedReadDispatcher {
    fn dispatch(
        &self,
        _program: &Program,
        _inputs: &[Vec<u8>],
        _grid_override: Option<[u32; 3]>,
    ) -> Result<Vec<Vec<u8>>, DispatchError> {
        Err(DispatchError::Rejected(
            "Fix: ranged-read test dispatcher does not implement dispatch.".to_string(),
        ))
    }

    fn read_resident(&self, handle: u64) -> Result<Vec<u8>, DispatchError> {
        self.read_calls.set(self.read_calls.get() + 1);
        self.buffers
            .iter()
            .find(|(candidate, _)| *candidate == handle)
            .map(|(_, bytes)| bytes.clone())
            .ok_or_else(|| {
                DispatchError::BadInputs(format!(
                    "Fix: test dispatcher missing resident handle {handle}."
                ))
            })
    }

    fn read_resident_many(&self, handles: &[u64]) -> Result<Vec<Vec<u8>>, DispatchError> {
        self.batched_handles.borrow_mut().extend_from_slice(handles);
        handles
            .iter()
            .map(|&handle| self.read_resident(handle))
            .collect()
    }
}

struct IntoOverrideDispatcher {
    many_calls: Cell<usize>,
    range_calls: Cell<usize>,
}

impl ProgramDispatcher for IntoOverrideDispatcher {
    fn dispatch(
        &self,
        _program: &Program,
        _inputs: &[Vec<u8>],
        _grid_override: Option<[u32; 3]>,
    ) -> Result<Vec<Vec<u8>>, DispatchError> {
        Err(DispatchError::Rejected(
            "Fix: into-override test dispatcher does not implement dispatch.".to_string(),
        ))
    }

    fn upload_resident_many_sequence_read_many_into(
        &self,
        uploads: &[(u64, &[u8])],
        _steps: &[ResidentDispatchStep<'_>],
        read_handles: &[u64],
        outputs: &mut Vec<Vec<u8>>,
    ) -> Result<(), DispatchError> {
        self.many_calls.set(self.many_calls.get() + 1);
        outputs.clear();
        outputs.push(vec![uploads.len() as u8, read_handles.len() as u8]);
        Ok(())
    }

    fn upload_resident_many_sequence_read_ranges_into(
        &self,
        uploads: &[(u64, &[u8])],
        _steps: &[ResidentDispatchStep<'_>],
        read_ranges: &[ResidentReadRange],
        outputs: &mut Vec<Vec<u8>>,
    ) -> Result<(), DispatchError> {
        self.range_calls.set(self.range_calls.get() + 1);
        outputs.clear();
        outputs.push(vec![uploads.len() as u8, read_ranges.len() as u8]);
        Ok(())
    }
}

struct FailingAllocDispatcher {
    next_handle: Cell<u64>,
    fail_at_call: usize,
    allocations: RefCell<Vec<usize>>,
    freed: RefCell<Vec<u64>>,
}

impl FailingAllocDispatcher {
    fn new(first_handle: u64, fail_at_call: usize) -> Self {
        Self {
            next_handle: Cell::new(first_handle),
            fail_at_call,
            allocations: RefCell::new(Vec::new()),
            freed: RefCell::new(Vec::new()),
        }
    }
}

impl ProgramDispatcher for FailingAllocDispatcher {
    fn dispatch(
        &self,
        _program: &Program,
        _inputs: &[Vec<u8>],
        _grid_override: Option<[u32; 3]>,
    ) -> Result<Vec<Vec<u8>>, DispatchError> {
        Err(DispatchError::Rejected(
            "Fix: failing allocation test dispatcher does not implement dispatch.".to_string(),
        ))
    }

    fn alloc_resident(&self, byte_len: usize) -> Result<u64, DispatchError> {
        let call = self.allocations.borrow().len();
        self.allocations.borrow_mut().push(byte_len);
        if call == self.fail_at_call {
            return Err(DispatchError::BackendError(
                "Fix: injected optimizer resident allocation failure".to_string(),
            ));
        }
        let handle = self.next_handle.get();
        self.next_handle.set(handle + 1);
        Ok(handle)
    }

    fn free_resident(&self, handle: u64) -> Result<(), DispatchError> {
        self.freed.borrow_mut().push(handle);
        Ok(())
    }
}

/// Value-returning resident helpers must preserve a dispatcher's fused
/// caller-owned implementation. Falling back to upload, dispatch, then
/// read introduces two extra host fences on CUDA.
#[test]
fn resident_value_wrappers_route_through_into_overrides() {
    let dispatcher = IntoOverrideDispatcher {
        many_calls: Cell::new(0),
        range_calls: Cell::new(0),
    };
    let payload = [0xA5_u8];
    let uploads = [(7_u64, payload.as_slice())];
    let handles = [11_u64];
    let ranges = [ResidentReadRange {
        handle_id: 13,
        byte_offset: 2,
        byte_len: 3,
    }];

    assert_eq!(
        dispatcher
            .upload_resident_many_sequence_read_many(&uploads, &[], &handles)
            .expect("Fix: fused whole-buffer convenience dispatch should succeed"),
        vec![vec![1, 1]]
    );
    assert_eq!(
        dispatcher
            .dispatch_resident_sequence_read_many(&[], &handles)
            .expect("Fix: fused whole-buffer dispatch/read convenience should succeed"),
        vec![vec![0, 1]]
    );
    assert_eq!(
        dispatcher
            .upload_resident_many_sequence_read_ranges(&uploads, &[], &ranges)
            .expect("Fix: fused ranged convenience dispatch should succeed"),
        vec![vec![1, 1]]
    );
    assert_eq!(
        dispatcher
            .dispatch_resident_sequence_read_ranges(&[], &ranges)
            .expect("Fix: fused ranged dispatch/read convenience should succeed"),
        vec![vec![0, 1]]
    );
    assert_eq!(
        dispatcher.many_calls.get(),
        2,
        "Fix: both whole-buffer value wrappers must route through the fused into override."
    );
    assert_eq!(
        dispatcher.range_calls.get(),
        2,
        "Fix: both ranged value wrappers must route through the fused into override."
    );
}

#[test]
fn generated_fill_upload_staging_preserves_fill_then_upload_order() {
    let host_payload = [0xA5_u8, 0x5A];
    let mut staged = Vec::new();

    with_staged_fill_uploads(
        &[(7, 3, 0x11), (9, 2, 0x22)],
        &[(13, host_payload.as_slice())],
        "test fill payloads",
        "test combined uploads",
        |uploads| {
            for &(handle, bytes) in uploads {
                staged.push((handle, bytes.to_vec()));
            }
            Ok(())
        },
    )
    .expect("Fix: shared resident fill staging should succeed");

    assert_eq!(
        staged,
        vec![
            (7, vec![0x11, 0x11, 0x11]),
            (9, vec![0x22, 0x22]),
            (13, host_payload.to_vec()),
        ],
        "resident fill staging must preserve device-fill uploads before caller uploads"
    );
}

#[test]
fn resident_grouped_allocation_rolls_back_partial_handles() {
    let dispatcher = FailingAllocDispatcher::new(90, 2);

    let err = dispatcher
        .alloc_resident_many(&[4, 8, 12])
        .expect_err("Fix: injected grouped allocation failure should surface");

    assert!(
        matches!(err, DispatchError::BackendError(message) if message.contains("injected optimizer resident allocation failure"))
    );
    assert_eq!(dispatcher.allocations.borrow().as_slice(), &[4, 8, 12]);
    assert_eq!(
        dispatcher.freed.borrow().as_slice(),
        &[90, 91],
        "Fix: grouped resident allocation must free every prior handle on failure."
    );
}

#[test]
fn ranged_readback_deduplicates_full_buffer_reads_by_handle() {
    let dispatcher = RangedReadDispatcher {
        buffers: vec![(7, (0u8..32).collect()), (9, (100u8..132).collect())],
        read_calls: Cell::new(0),
        batched_handles: RefCell::new(Vec::new()),
    };

    let outputs = dispatcher
        .read_resident_ranges(&[
            ResidentReadRange {
                handle_id: 7,
                byte_offset: 4,
                byte_len: 4,
            },
            ResidentReadRange {
                handle_id: 9,
                byte_offset: 2,
                byte_len: 3,
            },
            ResidentReadRange {
                handle_id: 7,
                byte_offset: 12,
                byte_len: 5,
            },
        ])
        .expect("Fix: ranged readback must succeed for in-bounds dedup keys; return Err on overlap violations - deduplicated ranged readback must succeed");

    assert_eq!(
        outputs,
        vec![
            vec![4, 5, 6, 7],
            vec![102, 103, 104],
            vec![12, 13, 14, 15, 16]
        ]
    );
    assert_eq!(
        dispatcher.read_calls.get(),
        2,
        "Fix: default ranged readback must read each unique resident handle once, not once per range."
    );
    assert_eq!(
        dispatcher.batched_handles.borrow().as_slice(),
        &[7, 9],
        "Fix: default ranged readback must preserve first-seen handle order for batched backend overrides."
    );
}

#[test]
fn generated_ranged_readbacks_deduplicate_handles_without_reordering_ranges() {
    let dispatcher = RangedReadDispatcher {
        buffers: (0..8u64)
            .map(|handle| {
                (
                    handle,
                    (0..64u8)
                        .map(|byte| byte.wrapping_add((handle as u8).wrapping_mul(17)))
                        .collect::<Vec<_>>(),
                )
            })
            .collect(),
        read_calls: Cell::new(0),
        batched_handles: RefCell::new(Vec::new()),
    };
    let ranges = (0..2048usize)
        .map(|case| ResidentReadRange {
            handle_id: ((case.wrapping_mul(5).wrapping_add(case / 11)) % 8) as u64,
            byte_offset: (case.wrapping_mul(7)) % 48,
            byte_len: (case % 16) + 1,
        })
        .collect::<Vec<_>>();

    let outputs = dispatcher
        .read_resident_ranges(&ranges)
        .expect("Fix: generated matrix fixtures must stay in-bounds; fix fixture or return Err - generated ranged readback matrix must succeed");

    assert_eq!(outputs.len(), ranges.len());
    for (range, output) in ranges.iter().zip(outputs.iter()) {
        let full = dispatcher
            .buffers
            .iter()
            .find(|(handle, _)| *handle == range.handle_id)
            .map(|(_, bytes)| bytes.as_slice())
            .expect("Fix: replace expect with fallible API or document caller precondition; panic only on programmer error - generated range uses known handle");
        assert_eq!(
            output.as_slice(),
            &full[range.byte_offset..range.byte_offset + range.byte_len],
            "generated range must preserve caller range order and byte-exact slices"
        );
    }
    assert_eq!(
        dispatcher.read_calls.get(),
        8,
        "Fix: generated ranged readback matrix must issue one full read per unique handle."
    );
}

/// The dispatch ABI returns writable storage buffers in declared order. Workgroup
/// scratch is not a dispatch output, and admitting it would shift every consumer's
/// output index by one without any visible error.
#[test]
fn declared_dispatch_outputs_are_the_writable_storage_buffers_in_order() {
    use crate::ir::{BufferAccess, BufferDecl, DataType};

    let program = Program::wrapped(
        vec![
            BufferDecl::storage("ro_in", 0, BufferAccess::ReadOnly, DataType::U32),
            BufferDecl::storage("frontier_out", 1, BufferAccess::ReadWrite, DataType::U32)
                .with_count(4),
            BufferDecl::workgroup("wg_scratch", 256, DataType::U32),
            BufferDecl::storage("changed", 2, BufferAccess::ReadWrite, DataType::U32).with_count(1),
            BufferDecl::storage("sink", 3, BufferAccess::WriteOnly, DataType::U32).with_count(1),
        ],
        [1, 1, 1],
        Vec::new(),
    );

    let names: Vec<&str> = declared_dispatch_outputs(&program)
        .iter()
        .map(|decl| decl.name())
        .collect();
    assert_eq!(names, vec!["frontier_out", "changed", "sink"]);
}

/// A program that declares no writable buffer has no dispatch outputs. Returning a
/// non-empty list there would be the oracle inventing a layout, which is the defect
/// this derivation exists to prevent.
#[test]
fn declared_dispatch_outputs_is_empty_when_nothing_is_writable() {
    use crate::ir::{BufferAccess, BufferDecl, DataType};

    let program = Program::wrapped(
        vec![
            BufferDecl::storage("ro_in", 0, BufferAccess::ReadOnly, DataType::U32),
            BufferDecl::workgroup("wg_scratch", 64, DataType::U32),
        ],
        [1, 1, 1],
        Vec::new(),
    );
    assert!(declared_dispatch_outputs(&program).is_empty());
}
