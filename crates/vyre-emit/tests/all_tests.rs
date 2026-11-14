//! One test binary for every emitter. The target suites cross-check each other
//! (PTX against Naga, SPIR-V against PTX), so the binary needs every target.
#![cfg(all(
    feature = "naga",
    feature = "ptx",
    feature = "spirv",
    feature = "metal"
))]

mod cross_emitter_parity;
mod emitted_artifact_byte_stability;
mod metal;
mod naga;
mod ptx;
mod source_tree_digest;
mod spirv;
