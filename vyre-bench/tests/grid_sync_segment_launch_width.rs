//! Acceptance and regression contract: a grid-sync segment launches its own domain.
//!
//! WHY: a program that splits at a grid-sync fence gives every segment the whole
//! program's buffer table, so one resident resource slice binds to every segment.
//! The dispatch width was the widest binding in that table, so the pass that
//! scans a 4096-element block-total buffer launched one lane per element of the
//! 1048576-element input beside it: 1048576 lanes for a 4096-lane domain, 256
//! times over. Measured on an RTX 4090, that pass ran 22.0 us while moving 22 KB
//! of device memory, 0.10 percent of peak bandwidth, and two of the five segments
//! of a one-mebi-element scan were launched that way.
//!
//! The rule under test: a segment's dispatch width never exceeds the widest
//! non-shared buffer the segment references. A buffer no statement of the
//! segment names is touched by no lane of it, so its declared width states
//! nothing about how many lanes the segment needs.
//!
//! Both the segment list and the per-segment buffer widths are derived from the
//! split at run time, so a change in how the scan decomposes, or a new pass in
//! the chain, is covered without editing this file.
//!
//! This proves nothing about how fast any segment runs, and nothing about the
//! widths of a program that carries no fence: a program whose every declared
//! buffer is referenced is decided exactly as it was before.

use vyre_driver::{dispatch_element_count_for_program, grid_sync::try_split_on_grid_sync, BindingPlan};
use vyre_foundation::ir::{BufferAccess, Program};

/// Elements the chain decomposes into more than one pass at.
const ELEMENTS: u32 = 1 << 20;

/// Widest non-shared buffer `program` names in a statement.
fn widest_referenced(program: &Program) -> u32 {
    let referenced = vyre_foundation::visit::referenced_buffers(program);
    program
        .buffers()
        .iter()
        .filter(|buffer| buffer.access() != BufferAccess::Workgroup)
        .filter(|buffer| referenced.iter().any(|name| name.as_ref() == buffer.name()))
        .map(|buffer| buffer.count())
        .max()
        .unwrap_or(0)
}

/// Widest non-shared buffer `program` declares, referenced or not.
fn widest_declared(program: &Program) -> u32 {
    program
        .buffers()
        .iter()
        .filter(|buffer| buffer.access() != BufferAccess::Workgroup)
        .map(|buffer| buffer.count())
        .max()
        .unwrap_or(0)
}

#[test]
fn no_grid_sync_segment_launches_wider_than_the_buffers_it_references() {
    let program = vyre_libs::math::scan::scan_prefix_sum("input", "output", ELEMENTS);
    let segments = try_split_on_grid_sync(&program).expect("the scan chain splits at its fences");
    assert!(
        segments.len() > 1,
        "this contract reads the per-segment width, and a chain that did not split carries one"
    );

    let mut narrowed = 0usize;
    for (index, segment) in segments.iter().enumerate() {
        let plan = BindingPlan::build(segment).expect("a split segment plans its bindings");
        let width = dispatch_element_count_for_program(segment, &plan.bindings);
        let referenced = widest_referenced(segment);

        assert!(
            width <= referenced,
            "segment {index} launches {width} lanes over a domain of {referenced}: \
             every lane above {referenced} touches no buffer the segment names"
        );
        if referenced < widest_declared(segment) {
            narrowed += 1;
        }
    }

    assert!(
        narrowed > 0,
        "no segment referenced fewer buffers than the table it inherited, so this run \
         proved nothing; the split stopped carrying the whole program's table"
    );
}
