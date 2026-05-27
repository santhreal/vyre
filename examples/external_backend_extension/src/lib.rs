//! External backend extension compatibility probe.
//!
//! `VyreBackend` is intentionally sealed in the current public API. This
//! standalone example documents the boundary an external backend crate can
//! exercise today without depending on workspace internals: build public IR,
//! serialize it, and carry backend metadata that can later be wired into an
//! official extension hook.

use vyre::ir::{BufferDecl, DataType, Expr, Node, Program};

/// Metadata an out-of-tree backend crate can publish without importing private
/// driver internals.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExternalBackendManifest {
    /// Stable backend identifier.
    pub id: &'static str,
    /// Backend implementation version.
    pub version: &'static str,
    /// Human-readable execution target.
    pub target: &'static str,
}

/// Return the example backend manifest.
#[must_use]
pub fn manifest() -> ExternalBackendManifest {
    ExternalBackendManifest {
        id: "example.external.backend",
        version: env!("CARGO_PKG_VERSION"),
        target: "documentation-only sealed-backend probe",
    }
}

/// Build a tiny public IR program through the same surface a downstream
/// backend crate would receive from callers.
#[must_use]
pub fn build_probe_program() -> Program {
    Program::wrapped(
        vec![BufferDecl::output("out", 0, DataType::U32).with_count(1)],
        [1, 1, 1],
        vec![Node::store(
            "out",
            Expr::u32(0),
            Expr::add(Expr::u32(40), Expr::u32(2)),
        )],
    )
}

/// Serialize the probe program so external crates can test wire compatibility
/// without implementing the sealed backend trait.
///
/// # Errors
///
/// Returns the public wire encoder error when the probe program cannot be
/// serialized.
pub fn probe_wire() -> Result<Vec<u8>, String> {
    build_probe_program()
        .to_wire()
        .map_err(|error| error.to_string())
}
