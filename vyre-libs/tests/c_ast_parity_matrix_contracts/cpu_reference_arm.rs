//! The C-AST parity matrix run on the CPU reference interpreter.
//!
//! Every family's `CASES` table is evaluated through the same four stages, the
//! same programs and the same comparisons the driver crates' GPU arms use. Only
//! the dispatch differs: [`CpuRefBackend`] interprets the `Program` in process
//! instead of submitting it to a device.
//!
//! # Why this arm exists
//!
//! A parity failure used to be observable only on a machine with a working
//! adapter, so the matrix could not be exercised while the one device was busy,
//! and a case that no arm named at all (`gnu_restrict_qualifier`) looked exactly
//! like a case that passed. Dispatching the same programs on the reference
//! interpreter separates "the kernel disagrees with the oracle" from "this
//! backend disagrees with the kernel": a divergence here is in the program, and
//! a divergence only on a device is in that device's lowering.

use vyre::ir::Program;
use vyre_driver::VyreBackend;
use vyre_driver_reference::CpuRefBackend;

use crate::c_frontend::parity_matrix::{assert_family_parity, ParityArm};
use crate::{
    declaration_advanced_constructs, declarator_matrix_constructs, semantic_gap_constructs,
};

/// The CPU reference interpreter as a parity arm.
struct CpuRefArm;

impl ParityArm for CpuRefArm {
    fn dispatch(
        &self,
        context: &'static str,
        program: Program,
        inputs: Vec<Vec<u8>>,
    ) -> Vec<Vec<u8>> {
        let borrowed: Vec<&[u8]> = inputs.iter().map(Vec::as_slice).collect();
        CpuRefBackend
            .dispatch_borrowed(&program, &borrowed, &Default::default())
            .unwrap_or_else(|error| {
                panic!(
                    "{context}: cpu-ref dispatch failed: {error}. Fix: the program and its input \
                     buffer order are owned by c_frontend::parity_matrix::program; correct them \
                     there so every arm sees the fix."
                )
            })
    }
}

#[test]
fn cpu_reference_parity_declarator_matrix_cases() {
    assert_family_parity(&CpuRefArm, declarator_matrix_constructs::CASES);
}

#[test]
fn cpu_reference_parity_declaration_advanced_cases() {
    assert_family_parity(&CpuRefArm, declaration_advanced_constructs::CASES);
}

#[test]
fn cpu_reference_parity_semantic_gap_cases() {
    assert_family_parity(&CpuRefArm, semantic_gap_constructs::CASES);
}
