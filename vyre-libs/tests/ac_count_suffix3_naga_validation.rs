//! Validate that the suffix3 AC count prefilter lowers cleanly through Naga.

use std::collections::BTreeMap;
use vyre_libs::pattern::classic_ac::{
    build_ac_bounded_count_suffix3_prefilter_program, classic_ac_compile,
};

use vyre_foundation::ir::ProgramGraph;
use vyre_foundation::logical::LogicalProgramGraph;
use vyre_foundation::schedule::SelectedSchedule;

#[test]
fn ac_count_suffix3_prefilter_naga_validates() {
    let patterns: [&[u8]; 5] = [b"AKIA", b"ghp_", b"password=", b"BEGIN", b"unsafe {"];
    let ac = classic_ac_compile(&patterns);
    let program = build_ac_bounded_count_suffix3_prefilter_program(&ac.dfa);
    let buffers = program
        .buffers()
        .iter()
        .cloned()
        .map(|buffer| {
            if buffer.name() == "haystack" {
                buffer.with_count(1)
            } else {
                buffer
            }
        })
        .collect();
    let program = program.with_rewritten_buffers(buffers);

    let graph = ProgramGraph::from_program("ac_count_suffix3_prefilter", program.clone())
        .expect("suffix3 AC count must form a valid graph");
    let logical = LogicalProgramGraph::validate(&graph, &BTreeMap::new())
        .expect("suffix3 AC count must have a valid logical domain");
    let schedule = SelectedSchedule::from_logical(&logical);
    let phase = schedule.phases[0].id;
    let lowered = vyre_lower::lower_scheduled(&program, &schedule, phase)
        .expect("suffix3 AC count must lower through a selected schedule");
    let module =
        vyre_emit_naga::emit(lowered.descriptor()).expect("suffix3 AC count must emit to Naga");
    let validation = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module);

    if let Err(error) = validation {
        panic!("suffix3 AC count Naga validation failed: {error:?}");
    }
}
