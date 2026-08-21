//! Resident buffer lifetime: transfers, ranged batch views, resident and
//! sequenced dispatch, and handle release on shutdown.

#![cfg(feature = "device-tests")]

use crate::*;

use super::fixtures::word_to_word;
use vyre_driver::{DispatchConfig, ResidentDispatchStep, ResidentReadRange};
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};

#[test]
fn apple_resident_transfers_cover_full_range_batch_and_stale_handles() {
    let backend = acquire().expect(
        "Fix: Apple Metal builds must acquire the system default MTLDevice before resident transfers.",
    );
    let first = backend
        .allocate_resident(8)
        .expect("Fix: native Metal must allocate resident buffers.");
    let second = backend
        .allocate_resident(4)
        .expect("Fix: native Metal must allocate multiple resident buffers.");

    backend
        .upload_resident(&first, &[1, 2, 3, 4])
        .expect("Fix: native Metal full resident upload must accept bounded payloads.");
    assert_eq!(
        backend
            .download_resident(&first)
            .expect("Fix: native Metal resident download must return logical allocation bytes."),
        vec![1, 2, 3, 4, 0, 0, 0, 0],
        "Fix: full resident upload must zero-pad unwritten allocation bytes."
    );

    backend
        .upload_resident_at(&first, 4, &[5, 6, 7, 8])
        .expect("Fix: native Metal resident ranged upload must write subranges.");
    assert_eq!(
        backend
            .download_resident_range(&first, 2, 4)
            .expect("Fix: native Metal resident ranged download must read subranges."),
        vec![3, 4, 5, 6]
    );

    backend
        .upload_resident_many(&[(&first, &[9, 8, 7, 6, 5, 4, 3, 2]), (&second, &[1, 2])])
        .expect("Fix: native Metal resident batch upload must validate and stage every handle.");
    backend
        .upload_resident_at_many(&[(&first, 0, &[10, 11, 12, 13]), (&second, 2, &[3, 4])])
        .expect(
            "Fix: native Metal resident ranged batch upload must validate and stage every range.",
        );
    let mut first_range = Vec::new();
    let mut second_range = Vec::new();
    backend
        .download_resident_ranges_into(
            &[(&first, 0, 4), (&second, 0, 4)],
            &mut [&mut first_range, &mut second_range],
        )
        .expect("Fix: native Metal resident batch ranged download must fill caller-owned buffers.");
    assert_eq!(first_range, vec![10, 11, 12, 13]);
    assert_eq!(second_range, vec![1, 2, 3, 4]);

    backend
        .free_resident(second.clone())
        .expect("Fix: native Metal resident free must release live handles.");
    let stale = backend
        .download_resident(&second)
        .expect_err("Fix: native Metal must reject stale resident handles after free.");
    assert!(
        stale.to_string().contains("stale resident handle"),
        "Fix: stale resident diagnostics must name the handle lifetime problem: {stale}"
    );
    backend
        .free_resident(first)
        .expect("Fix: native Metal resident free must release each live handle exactly once.");
}

#[test]
fn apple_resident_transfer_range_errors_are_actionable() {
    let backend = acquire().expect(
        "Fix: Apple Metal builds must acquire the system default MTLDevice before resident transfer negative tests.",
    );
    let resource = backend
        .allocate_resident(4)
        .expect("Fix: native Metal must allocate resident buffers before negative range checks.");

    let oversized_upload = backend
        .upload_resident(&resource, &[1, 2, 3, 4, 5])
        .expect_err(
            "Fix: native Metal must reject full resident uploads larger than the allocation.",
        );
    assert!(
        oversized_upload
            .to_string()
            .contains("requested byte range [0..5) from allocation 4"),
        "Fix: oversized resident upload error must name the invalid range and allocation: {oversized_upload}"
    );

    let ranged_upload = backend
        .upload_resident_at(&resource, 3, &[9, 9])
        .expect_err(
            "Fix: native Metal must reject ranged resident uploads that cross allocation end.",
        );
    assert!(
        ranged_upload
            .to_string()
            .contains("requested byte range [3..5) from allocation 4"),
        "Fix: out-of-bounds resident ranged upload must name the invalid range and allocation: {ranged_upload}"
    );

    let ranged_download = backend.download_resident_range(&resource, 2, 3).expect_err(
        "Fix: native Metal must reject ranged resident downloads that cross allocation end.",
    );
    assert!(
        ranged_download
            .to_string()
            .contains("requested byte range [2..5) from allocation 4"),
        "Fix: out-of-bounds resident ranged download must name the invalid range and allocation: {ranged_download}"
    );

    let mut only_output = Vec::new();
    let count_mismatch = backend
        .download_resident_ranges_into(
            &[(&resource, 0, 1), (&resource, 1, 1)],
            &mut [&mut only_output],
        )
        .expect_err("Fix: native Metal must reject resident range/output count mismatches.");
    assert!(
        count_mismatch
            .to_string()
            .contains("matching range/output counts"),
        "Fix: resident batch download count mismatch must be actionable: {count_mismatch}"
    );

    backend
        .free_resident(resource)
        .expect("Fix: native Metal must free resident negative-test handles.");
}

#[test]
fn apple_resident_ranged_batch_download_fuses_views_and_preflights_outputs() {
    let backend = acquire().expect(
        "Fix: Apple Metal builds must acquire the system default MTLDevice before fused resident batch readback tests.",
    );
    let resource = backend
        .allocate_resident(16)
        .expect("Fix: native Metal must allocate resident buffers before fused readback.");
    let bytes = (0u8..16u8).collect::<Vec<_>>();
    backend
        .upload_resident(&resource, &bytes)
        .expect("Fix: native Metal must upload resident bytes before fused readback.");

    let mut first = vec![0xaa; 2];
    let first_capacity = first.capacity();
    let mut overlap = vec![0xbb; 1];
    let mut empty = vec![0xcc; 3];
    let mut tail = Vec::with_capacity(32);
    let tail_capacity = tail.capacity();
    backend
        .download_resident_ranges_into(
            &[
                (&resource, 0, 6),
                (&resource, 4, 6),
                (&resource, 10, 0),
                (&resource, 10, 4),
            ],
            &mut [&mut first, &mut overlap, &mut empty, &mut tail],
        )
        .expect("Fix: native Metal must materialize overlapping and empty views from one fused resident batch plan.");

    assert_eq!(first, vec![0, 1, 2, 3, 4, 5]);
    assert_eq!(overlap, vec![4, 5, 6, 7, 8, 9]);
    assert_eq!(
        empty,
        Vec::<u8>::new(),
        "Fix: zero-byte resident batch views must clear stale caller output bytes."
    );
    assert_eq!(tail, vec![10, 11, 12, 13]);
    assert!(
        first.capacity() >= first_capacity && tail.capacity() >= tail_capacity,
        "Fix: fused Metal resident readback must preserve reusable caller output capacity."
    );

    let mut valid_output = vec![0xdd, 0xee];
    let mut invalid_output = vec![0xff];
    let before_valid = valid_output.clone();
    let before_invalid = invalid_output.clone();
    let error = backend
        .download_resident_ranges_into(
            &[(&resource, 0, 2), (&resource, 15, 4)],
            &mut [&mut valid_output, &mut invalid_output],
        )
        .expect_err("Fix: native Metal fused resident batch readback must reject invalid ranges before mutating outputs.");
    assert!(
        error
            .to_string()
            .contains("requested byte range [15..19) from allocation 16"),
        "Fix: fused resident readback range errors must name the invalid range and allocation: {error}"
    );
    assert_eq!(
        valid_output, before_valid,
        "Fix: fused resident batch download must not mutate earlier outputs when a later range fails validation."
    );
    assert_eq!(
        invalid_output, before_invalid,
        "Fix: fused resident batch download must not mutate the invalid output slot."
    );

    backend
        .free_resident(resource)
        .expect("Fix: native Metal must free fused readback resident handles.");
}

#[test]
fn apple_resident_dispatch_uses_binding_order_handles_and_persists_output() {
    let program = word_to_word("input", "out", |input| Expr::add(input, Expr::u32(1)));

    let backend = acquire().expect(
        "Fix: Apple Metal builds must acquire the system default MTLDevice before resident dispatch.",
    );
    let input = backend
        .allocate_resident(4)
        .expect("Fix: native Metal must allocate a resident input handle.");
    let output = backend
        .allocate_resident(4)
        .expect("Fix: native Metal must allocate a resident output handle.");
    backend
        .upload_resident(&input, &41u32.to_le_bytes())
        .expect("Fix: native Metal must upload resident input bytes before dispatch.");

    let timed = backend
        .dispatch_resident_timed(
            &program,
            &[input.clone(), output.clone()],
            &DispatchConfig::default(),
        )
        .expect(
            "Fix: native Metal resident dispatch must bind resources in Program binding order.",
        );
    assert_eq!(timed.outputs, vec![42u32.to_le_bytes().to_vec()]);
    assert!(
        timed.enqueue_ns.is_some() && timed.wait_ns.is_some() && timed.wall_ns > 0,
        "Fix: Metal resident timed dispatch must report host timing fields."
    );
    assert_eq!(
        backend
            .download_resident(&output)
            .expect("Fix: resident output must remain readable after dispatch."),
        42u32.to_le_bytes().to_vec()
    );

    backend
        .free_resident(input)
        .expect("Fix: native Metal must free resident input handles.");
    backend
        .free_resident(output)
        .expect("Fix: native Metal must free resident output handles.");
}

#[test]
fn apple_resident_dispatch_resource_errors_are_actionable() {
    let program = word_to_word("input", "out", |input| Expr::add(input, Expr::u32(1)));

    let backend = acquire().expect(
        "Fix: Apple Metal builds must acquire the system default MTLDevice before resident dispatch negative tests.",
    );
    let input = backend
        .allocate_resident(4)
        .expect("Fix: native Metal must allocate resident input before negative dispatch checks.");
    backend
        .upload_resident(&input, &10u32.to_le_bytes())
        .expect("Fix: native Metal must upload resident input before negative dispatch checks.");

    let wrong_count = backend
        .dispatch_resident_timed(
            &program,
            std::slice::from_ref(&input),
            &DispatchConfig::default(),
        )
        .expect_err("Fix: native Metal resident dispatch must reject missing output resources.");
    assert!(
        wrong_count
            .to_string()
            .contains("expected 2 resource(s) in binding order but received 1"),
        "Fix: resident dispatch wrong-count error must name expected and received resource counts: {wrong_count}"
    );

    let stale_output = backend
        .allocate_resident(4)
        .expect("Fix: native Metal must allocate a resident output before stale-handle checks.");
    backend
        .free_resident(stale_output.clone())
        .expect("Fix: native Metal must free resident output before stale-handle checks.");
    let stale_resources = [input.clone(), stale_output];
    let stale_error = backend
        .dispatch_resident_timed(&program, &stale_resources, &DispatchConfig::default())
        .expect_err("Fix: native Metal resident dispatch must reject stale output handles.");
    assert!(
        stale_error.to_string().contains("stale handle"),
        "Fix: resident dispatch stale-handle error must name the handle lifetime problem: {stale_error}"
    );

    let undersized_output = backend.allocate_resident(0).expect(
        "Fix: native Metal must allow zero-byte logical resident allocations for boundary testing.",
    );
    let undersized_resources = [input.clone(), undersized_output.clone()];
    let undersized_error = backend
        .dispatch_resident_timed(&program, &undersized_resources, &DispatchConfig::default())
        .expect_err("Fix: native Metal resident dispatch must reject undersized output handles.");
    assert!(
        undersized_error
            .to_string()
            .contains("requires 4 byte(s)")
            && undersized_error.to_string().contains("has 0"),
        "Fix: resident dispatch undersized-output error must name required and actual byte counts: {undersized_error}"
    );

    backend
        .free_resident(input)
        .expect("Fix: native Metal must free resident negative-dispatch input handles.");
    backend
        .free_resident(undersized_output)
        .expect("Fix: native Metal must free resident negative-dispatch output handles.");
}

#[test]
fn apple_resident_sequence_dispatches_ordered_steps_and_reads_ranges() {
    let double_program = word_to_word("input", "mid", |input| Expr::mul(input, Expr::u32(2)));
    let add_program = word_to_word("mid", "out", |mid| Expr::add(mid, Expr::u32(7)));

    let backend = acquire().expect(
        "Fix: Apple Metal builds must acquire the system default MTLDevice before resident sequence dispatch.",
    );
    let seed = backend
        .allocate_resident(4)
        .expect("Fix: native Metal must allocate resident sequence input.");
    let mid = backend
        .allocate_resident(4)
        .expect("Fix: native Metal must allocate resident sequence handoff.");
    let out = backend
        .allocate_resident(4)
        .expect("Fix: native Metal must allocate resident sequence output.");
    backend
        .upload_resident(&seed, &16u32.to_le_bytes())
        .expect("Fix: native Metal must upload resident sequence seed bytes.");

    let first_resources = [seed.clone(), mid.clone()];
    let second_resources = [mid.clone(), out.clone()];
    let steps = [
        ResidentDispatchStep {
            program: &double_program,
            resources: &first_resources,
            grid_override: None,
            workgroup_override: None,
        },
        ResidentDispatchStep {
            program: &add_program,
            resources: &second_resources,
            grid_override: None,
            workgroup_override: None,
        },
    ];
    let read_ranges = [ResidentReadRange {
        resource: &out,
        byte_offset: 0,
        byte_len: 4,
    }];
    let mut readback = Vec::new();

    let timing = backend
        .dispatch_resident_sequence_read_ranges_timed_into(
            &steps,
            &read_ranges,
            &mut [&mut readback],
        )
        .expect("Fix: native Metal must execute ordered resident sequences through the public backend API.");

    assert_eq!(
        readback,
        39u32.to_le_bytes().to_vec(),
        "Fix: resident sequence readback must observe step-2 output fed by step-1 resident handoff."
    );
    assert_eq!(
        backend
            .download_resident(&mid)
            .expect("Fix: resident sequence handoff must remain readable."),
        32u32.to_le_bytes().to_vec(),
        "Fix: resident sequence must persist intermediate output in the handoff handle."
    );
    assert!(
        timing.wall_ns > 0 && timing.enqueue_ns.is_some() && timing.wait_ns.is_some(),
        "Fix: resident sequence timing must preserve Metal host enqueue/wait evidence."
    );
    assert_eq!(
        timing.device_ns, None,
        "Fix: Metal resident sequence must not fake device timing until native counters are wired."
    );

    backend
        .free_resident(seed)
        .expect("Fix: native Metal must free resident sequence seed handles.");
    backend
        .free_resident(mid)
        .expect("Fix: native Metal must free resident sequence handoff handles.");
    backend
        .free_resident(out)
        .expect("Fix: native Metal must free resident sequence output handles.");
}

#[test]
fn apple_repeated_resident_sequence_updates_read_write_handle() {
    let increment_program = Program::wrapped(
        vec![
            BufferDecl::storage("state", 0, BufferAccess::ReadWrite, DataType::U32)
                .with_count(1)
                .with_output_byte_range(0..4),
        ],
        [1, 1, 1],
        vec![Node::store(
            "state",
            Expr::u32(0),
            Expr::add(Expr::load("state", Expr::u32(0)), Expr::u32(1)),
        )],
    );

    let backend = acquire().expect(
        "Fix: Apple Metal builds must acquire the system default MTLDevice before repeated resident sequence dispatch.",
    );
    let state = backend
        .allocate_resident(4)
        .expect("Fix: native Metal must allocate repeated resident sequence state.");
    backend
        .upload_resident(&state, &5u32.to_le_bytes())
        .expect("Fix: native Metal must upload repeated resident sequence state bytes.");

    let step_resources = [state.clone()];
    let repeated_steps = [ResidentDispatchStep {
        program: &increment_program,
        resources: &step_resources,
        grid_override: None,
        workgroup_override: None,
    }];
    let read_ranges = [ResidentReadRange {
        resource: &state,
        byte_offset: 0,
        byte_len: 4,
    }];
    let mut readback = Vec::new();

    backend
        .dispatch_resident_repeated_sequence_read_ranges_into(
            &[],
            &repeated_steps,
            3,
            &read_ranges,
            &mut [&mut readback],
        )
        .expect("Fix: native Metal must execute repeated resident sequences through the public backend API.");

    assert_eq!(
        readback,
        8u32.to_le_bytes().to_vec(),
        "Fix: repeated resident sequence must preserve state across repeated read-write dispatches."
    );
    assert_eq!(
        backend
            .download_resident(&state)
            .expect("Fix: repeated resident sequence state must remain readable."),
        8u32.to_le_bytes().to_vec(),
        "Fix: repeated resident sequence must persist the final state in the resident handle."
    );

    backend
        .free_resident(state)
        .expect("Fix: native Metal must free repeated resident sequence state handles.");
}

#[test]
fn apple_shutdown_clears_resident_handles() {
    let backend = acquire().expect(
        "Fix: Apple Metal builds must acquire the system default MTLDevice before shutdown testing.",
    );
    let resource = backend
        .allocate_resident(4)
        .expect("Fix: native Metal must allocate a resident handle before shutdown.");
    backend
        .upload_resident(&resource, &[1, 2, 3, 4])
        .expect("Fix: native Metal must upload resident bytes before shutdown.");

    backend
        .shutdown()
        .expect("Fix: native Metal shutdown must clear backend-owned resources.");
    let error = backend
        .download_resident(&resource)
        .expect_err("Fix: resident handles must be invalid after Metal shutdown clears resources.");
    assert!(
        error.to_string().contains("stale resident handle"),
        "Fix: post-shutdown resident use must fail closed as a stale handle: {error}"
    );
}
