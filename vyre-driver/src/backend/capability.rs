//! Backend capability traits.

use super::VyreBackend;
use std::collections::HashSet;
use vyre_foundation::ir::OpId;

/// Minimal backend identity and capability contract.
pub trait Backend: Send + Sync {
    /// Stable backend identifier.
    fn id(&self) -> &'static str;
    /// Backend implementation version.
    fn version(&self) -> &'static str;
    /// Operation ids this backend can execute without further lowering.
    fn supported_ops(&self) -> &HashSet<OpId>;
}

impl<T: VyreBackend + ?Sized> Backend for T {
    fn id(&self) -> &'static str {
        VyreBackend::id(self)
    }

    fn version(&self) -> &'static str {
        VyreBackend::version(self)
    }

    fn supported_ops(&self) -> &HashSet<OpId> {
        VyreBackend::supported_ops(self)
    }
}
