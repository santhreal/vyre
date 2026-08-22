//! Layout-parameter contracts for the `scan::hit_buffer` append/compact pair.
//!
//! WHY: `emit_hit_with_layout` and `compact_hits_with_layout` own the hit-buffer
//! IR; `emit_hit` and `compact_hits` are default-argument wrappers over them.
//! Only the wrappers were ever named by a test or an `inventory::submit!` block,
//! so the two owners carried no coverage at all. Everything the owners add over
//! the wrappers is exactly the part the wrappers cannot express: a lane count
//! that differs from the hit capacity, and a backing hit-buffer size that
//! differs from the reported live-length clamp. Swapping those parameters, or
//! collapsing the two capacities onto one, would have shipped green.
//!
//! This closes the class for every layout parameter the two owners expose, on
//! the declaration side (what the program asks the device to allocate) and on
//! the value side (what the program actually writes).
//!
//! What this does not catch: a semantic change applied identically to the
//! wrapper and the owner, which the delegation check below would still accept.

#![forbid(unsafe_code)]

use vyre_libs::pattern::{
    compact_hits, compact_hits_with_layout, emit_hit, emit_hit_with_layout, HIT_BUFFER_LIVE_LENGTH,
    HIT_BUFFER_OVERFLOW_COUNT,
};
use vyre_primitives::wire::{decode_u32_le_bytes_all, pack_u32_slice};
use vyre_reference::value::Value;

const RULE_ID: &str = "rule_id";
const FILE_ID: &str = "file_id";
const SPAN_START: &str = "span_start";
const SPAN_LEN: &str = "span_len";
const OUT_HITS: &str = "out_hits";
const OUT_CURSOR: &str = "out_cursor";

/// Declared element count of the named buffer.
fn declared_count(program: &vyre_foundation::ir::Program, name: &str) -> u32 {
    program
        .buffers()
        .iter()
        .find(|decl| decl.name() == name)
        .unwrap_or_else(|| panic!("Fix: program must declare {name}"))
        .count
}

fn words(value: &Value) -> Vec<u32> {
    decode_u32_le_bytes_all(&value.to_bytes())
}

/// Run an emit program over `supplied` per-lane tuples. The reference
/// interpreter refuses a buffer smaller than the declared count, so `supplied`
/// is at least `lane_count`; supplying more is how the run-time lane bound is
/// distinguished from the declared one. Returns `(hits, cursor, overflow)`.
fn run_emit(lane_count: u32, max_hits: u32, supplied: u32) -> (Vec<u32>, u32, u32) {
    assert!(
        supplied >= lane_count,
        "the interpreter refuses a buffer below the declared element count"
    );
    let program = emit_hit_with_layout(
        RULE_ID, FILE_ID, SPAN_START, SPAN_LEN, OUT_HITS, OUT_CURSOR, lane_count, max_hits,
    );
    let rule_ids: Vec<u32> = (0..supplied).map(|lane| 700 + lane).collect();
    let file_ids: Vec<u32> = (0..supplied).map(|lane| 800 + lane).collect();
    let starts: Vec<u32> = (0..supplied).map(|lane| 900 + lane).collect();
    let lens: Vec<u32> = (0..supplied).map(|lane| 1 + lane).collect();
    let outputs = vyre_reference::reference_eval(
        &program,
        &[
            Value::from(pack_u32_slice(&rule_ids)),
            Value::from(pack_u32_slice(&file_ids)),
            Value::from(pack_u32_slice(&starts)),
            Value::from(pack_u32_slice(&lens)),
            Value::from(pack_u32_slice(&[0])),
            Value::from(pack_u32_slice(&[0])),
        ],
    )
    .expect("Fix: emit_hit_with_layout must execute on the reference interpreter");
    assert_eq!(
        outputs.len(),
        3,
        "emit publishes the hit buffer, the cursor, and the overflow counter"
    );
    (
        words(&outputs[0]),
        words(&outputs[1])[0],
        words(&outputs[2])[0],
    )
}

/// Run a compact program over `supplied_hits` tuples and a cursor value.
/// `supplied_hits` is at least `hit_capacity` for the same reason.
fn run_compact(hit_capacity: u32, max_capacity: u32, supplied_hits: u32, cursor: u32) -> u32 {
    assert!(
        supplied_hits >= hit_capacity,
        "the interpreter refuses a buffer below the declared element count"
    );
    let program = compact_hits_with_layout(OUT_HITS, OUT_CURSOR, hit_capacity, max_capacity);
    let hits: Vec<u32> = (0..supplied_hits * 4).collect();
    let outputs = vyre_reference::reference_eval(
        &program,
        &[
            Value::from(pack_u32_slice(&hits)),
            Value::from(pack_u32_slice(&[cursor])),
        ],
    )
    .expect("Fix: compact_hits_with_layout must execute on the reference interpreter");
    assert_eq!(outputs.len(), 1, "compact publishes only the live length");
    words(&outputs[0])[0]
}

/// The wrappers are the owners at the documented default layout and nothing
/// else. Goes red if a wrapper grows a private copy of the IR, or if either
/// default drifts away from the four-lane / four-hit shape the registered
/// operation fixture is sized for.
#[test]
fn default_wrappers_are_the_owners_at_the_documented_default_layout() {
    assert_eq!(
        emit_hit(RULE_ID, FILE_ID, SPAN_START, SPAN_LEN, OUT_HITS, OUT_CURSOR).fingerprint(),
        emit_hit_with_layout(RULE_ID, FILE_ID, SPAN_START, SPAN_LEN, OUT_HITS, OUT_CURSOR, 4, 4,)
            .fingerprint(),
        "emit_hit must be emit_hit_with_layout at four lanes and four hits"
    );
    assert_eq!(
        compact_hits(OUT_HITS, OUT_CURSOR, 6).fingerprint(),
        compact_hits_with_layout(OUT_HITS, OUT_CURSOR, 6, 6).fingerprint(),
        "compact_hits must be compact_hits_with_layout with one shared capacity"
    );
}

/// Lane count sizes the four per-lane inputs; hit capacity sizes the append
/// buffer as four words per tuple. Goes red if the two are transposed or if
/// either is applied to the other's buffers.
#[test]
fn emit_sizes_lane_inputs_and_the_hit_buffer_independently() {
    let program = emit_hit_with_layout(
        RULE_ID, FILE_ID, SPAN_START, SPAN_LEN, OUT_HITS, OUT_CURSOR, 6, 2,
    );
    for lane_input in [RULE_ID, FILE_ID, SPAN_START, SPAN_LEN] {
        assert_eq!(
            declared_count(&program, lane_input),
            6,
            "{lane_input} carries one element per lane"
        );
    }
    assert_eq!(
        declared_count(&program, OUT_HITS),
        8,
        "the append buffer holds four words per hit tuple"
    );
    assert_eq!(declared_count(&program, OUT_CURSOR), 1);
    assert_eq!(declared_count(&program, HIT_BUFFER_OVERFLOW_COUNT), 1);
}

/// Hit capacity is a real bound at run time, not just an allocation hint: the
/// lanes past it are dropped, the cursor stops at the capacity, and the drop
/// count is published. Goes red if the capacity stops gating the stores, or if
/// the overflow counter is written when nothing overflowed.
#[test]
fn emit_truncates_at_the_hit_capacity_and_publishes_the_drop_count() {
    let (hits, cursor, overflow) = run_emit(6, 2, 6);
    assert_eq!(
        hits,
        vec![700, 800, 900, 1, 701, 801, 901, 2],
        "only the tuples that fit are appended, in lane order"
    );
    assert_eq!(cursor, 2, "the cursor saturates at the hit capacity");
    assert_eq!(overflow, 4, "the four lanes that did not fit are counted");

    let (hits, cursor, overflow) = run_emit(3, 8, 3);
    assert_eq!(
        hits[..12].to_vec(),
        vec![700, 800, 900, 1, 701, 801, 901, 2, 702, 802, 902, 3],
        "a capacity above the lane count appends every tuple"
    );
    assert_eq!(
        hits[12..].iter().copied().max(),
        Some(0),
        "the capacity past the supplied lanes is left untouched"
    );
    assert_eq!(cursor, 3, "the cursor reports the live tuple count");
    assert_eq!(overflow, 0, "nothing is dropped, so nothing is counted");

    // The lane bound is read from the supplied buffer, not from the declared
    // count, so a caller that hands over more lanes than it declared gets them
    // emitted rather than silently dropped.
    let (_, cursor, overflow) = run_emit(2, 8, 5);
    assert_eq!(cursor, 5, "every supplied lane is live");
    assert_eq!(overflow, 0);
}

/// The backing hit-buffer size and the reported live-length clamp are separate
/// parameters. The first is a declaration, the second is baked into the
/// published value. Goes red the moment one is used for the other.
#[test]
fn compact_separates_the_backing_capacity_from_the_live_length_clamp() {
    let program = compact_hits_with_layout(OUT_HITS, OUT_CURSOR, 8, 2);
    assert_eq!(
        declared_count(&program, OUT_HITS),
        32,
        "the backing buffer is sized by the hit capacity, four words per tuple"
    );
    assert_eq!(declared_count(&program, OUT_CURSOR), 1);
    assert_eq!(declared_count(&program, HIT_BUFFER_LIVE_LENGTH), 1);
    assert_eq!(
        declared_count(
            &compact_hits_with_layout(OUT_HITS, OUT_CURSOR, 2, 8),
            OUT_HITS
        ),
        8,
        "the live-length clamp must not size the backing buffer"
    );

    assert_eq!(
        run_compact(8, 2, 8, 5),
        2,
        "the max-capacity clamp wins when it is the smallest bound"
    );
    assert_eq!(
        run_compact(8, 9, 8, 5),
        5,
        "the cursor wins when it is the smallest bound"
    );
    assert_eq!(
        run_compact(3, 9, 3, 7),
        3,
        "the backing hit buffer wins when it is the smallest bound"
    );
    // `buffer_cap` is read from the supplied buffer, not from the declared
    // count, so a larger supply raises the bound.
    assert_eq!(
        run_compact(3, 9, 6, 7),
        6,
        "a supply above the declared capacity is what bounds the live length"
    );
}
