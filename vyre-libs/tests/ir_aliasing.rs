//! IR aliasing regression tests.

#![cfg(feature = "c-parser")]
use std::collections::HashSet;

use vyre::ir::{BufferAccess, Program};
use vyre_foundation::composition::tag_program;
use vyre_libs::decode::{base64_decode, hex_decode, inflate_stored_block};
use vyre_primitives::parsing::core_delimiter_match::core_delimiter_match;

fn rebind_program(program: &Program, binding_base: u32) -> Program {
    let mut next_binding = binding_base;
    let mut buffers = program.buffers().to_vec();
    for buffer in &mut buffers {
        if buffer.access() != BufferAccess::Workgroup {
            buffer.binding = next_binding;
            next_binding += 1;
        }
        buffer.is_output = false;
    }
    Program::wrapped(buffers, program.workgroup_size(), program.entry().to_vec())
}

fn combine_programs(programs: &[Program]) -> Program {
    let mut buffers = Vec::new();
    let mut entry = Vec::new();
    let mut binding_base = 0_u32;

    for program in programs {
        let rebound = rebind_program(program, binding_base);
        binding_base += rebound
            .buffers()
            .iter()
            .filter(|buffer| buffer.access() != BufferAccess::Workgroup)
            .count() as u32;
        buffers.extend(rebound.buffers().iter().cloned());
        entry.extend(rebound.entry().iter().cloned());
    }

    Program::wrapped(buffers, [1, 1, 1], entry)
}

fn assert_unique_buffer_names(program: &Program) {
    let unique = program
        .buffers()
        .iter()
        .map(|buffer| buffer.name().to_string())
        .collect::<HashSet<_>>();
    assert_eq!(unique.len(), program.buffers().len());
}

#[test]
fn fused_decode_programs_keep_generic_buffers_disjoint() {
    let combined = combine_programs(&[
        base64_decode("input", "decoded", 16),
        hex_decode("input", "decoded", 16),
        inflate_stored_block("input", "decoded", 16),
    ]);

    assert_unique_buffer_names(&combined);
    let errors = vyre::validate(&combined);
    assert!(errors.is_empty(), "{errors:#?}");
}

#[test]
fn duplicate_self_exclusive_parser_regions_fail_validation() {
    let op_id = vyre_primitives::parsing::core_delimiter_match::OP_ID;
    let scanner = core_delimiter_match("tok_types_a", "tok_depths_a", 8, 12, 13);
    assert!(
        scanner.is_non_composable_with_self(),
        "the delimiter scanner shares scratch depth state; it must advertise self-exclusivity"
    );

    // A Tier-3 composition boundary carries that flag into the region name, which
    // is what survives fusion into one program.
    let combined = combine_programs(&[
        tag_program(op_id, scanner.clone()),
        tag_program(op_id, scanner),
    ]);

    let errors = vyre::validate(&combined);
    assert!(
        errors.iter().any(|error| error
            .message()
            .contains("marked non-composable with itself")),
        "{errors:#?}"
    );
}
