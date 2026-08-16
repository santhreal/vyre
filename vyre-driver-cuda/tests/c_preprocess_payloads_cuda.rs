//! Live CUDA coverage for C preprocessor directive payload extraction.
//!
//! This drives the real tokenization and directive-payload pipeline through
//! raw U8 source buffers, including the fused define/include/undef parse path.

#![cfg(test)]

#[path = "harness/c_preprocess_oracles.rs"]
mod c_preprocess_oracles;
mod harness;

use c_preprocess_oracles::{CudaOracle, ReferenceOracle};
use harness::with_live_backend;
use vyre_libs::parsing::c::preprocess::gpu_pipeline::{
    gpu_extract_directive_payloads, gpu_tokenize_and_classify, DirectivePayload, ProgramOracle,
};

fn payloads(
    dispatcher: &dyn ProgramOracle,
    source: &[u8],
    macros: &[&[u8]],
) -> Vec<DirectivePayload> {
    let classified = gpu_tokenize_and_classify(dispatcher, source)
        .unwrap_or_else(|error| panic!("Fix: C payload tokenization failed: {error}"));
    gpu_extract_directive_payloads(dispatcher, &classified, macros)
        .unwrap_or_else(|error| panic!("Fix: C payload extraction failed: {error}"))
}

fn meaningful_payload_count(payloads: &[DirectivePayload]) -> usize {
    payloads
        .iter()
        .filter(|payload| !matches!(payload, DirectivePayload::None))
        .count()
}

#[test]
fn cuda_c_preprocess_payloads_match_reference() {
    with_live_backend("c preprocess directive payloads", |backend| {
        let cuda_dispatcher = CudaOracle(backend);
        let reference_dispatcher = ReferenceOracle;
        let source = br#"
#define FOO 42
#define MAX(a,b) ((a)>(b)?(a):(b))
#include <stdio.h>
#include_next <linux/compiler.h>
#undef FOO
#ifdef ENABLED
#endif
#ifndef MISSING
#endif
#if defined(ENABLED) && (3 + 4) > 1
#elif 0
#else
#endif
"#;
        let macros: [&[u8]; 1] = [b"ENABLED"];
        let expected = payloads(&reference_dispatcher, source, &macros);
        let actual = payloads(&cuda_dispatcher, source, &macros);
        assert_eq!(
            actual, expected,
            "Fix: CUDA directive payload extraction must match reference output byte-for-byte."
        );
        assert!(
            meaningful_payload_count(&actual) >= 12,
            "Fix: CUDA payload test must cover define, include, undef, ifdef, ifndef, if, elif, else, and endif rows."
        );
    });
}
