//! Where each production backend registration executes the program it is given.
//!
//! WHY: "vyre never runs a user program on the CPU" is a claim about the
//! backend registry, and the registry is a run-time set of `&'static str` ids
//! submitted through `inventory` by whichever driver crates a binary links. A
//! test that reads that set and rejects the ids it recognizes as host paths
//! certifies nothing: the substring gate it replaces admitted any id that did
//! not contain `ref` or `cpu`, so a registration called `host-interp` passed.
//!
//! This is the decision ledger instead. One variant per backend the production
//! registry is allowed to hold, and one exhaustive match from that variant to
//! the domain its dispatch runs in. The two halves fail in opposite
//! directions, which is what closes the class:
//!
//! - A registration whose id is not in the ledger has no recorded decision, so
//!   [`ProductionBackend::from_registry_id`] returns `None` and the reader
//!   fails. Re-registering the reference oracle lands here.
//! - A variant added to the ledger without a decision does not compile, because
//!   [`ProductionBackend::registry_id`], [`ProductionBackend::driver_crate`]
//!   and [`ProductionBackend::execution_domain`] each match exhaustively with
//!   no catch-all arm.
//!
//! The ledger's crate set is judged against `vyre_registry_link::backend::DECLARED_SOURCES`
//! by its readers rather than being trusted, and that constant is itself
//! derived from the tree at run time by
//! `every_crate_that_submits_a_backend_registration_is_declared_here`. A driver
//! crate added tomorrow therefore reaches this ledger without anyone
//! remembering it exists.
//!
//! Not proven here: that a crate carrying [`ExecutionDomain::Device`] contains
//! no host arithmetic of its own. That is a property of the dependency graph,
//! and `vyre-driver/tests/no_backend_crate_links_host_arithmetic.rs` owns it.

/// Where a backend registration executes the program dispatched to it.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ExecutionDomain {
    /// Dispatch leaves the host: the program runs on an accelerator reached
    /// through a vendor driver or a portable graphics API.
    Device,
    /// Dispatch evaluates the program in host memory on this process's CPU.
    /// No production registration may carry this domain.
    Host,
}

/// One backend registration the production registry is allowed to hold.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum ProductionBackend {
    /// CUDA driver, PTX payloads.
    Cuda,
    /// Metal driver, MSL payloads.
    Metal,
    /// Vulkan driver, SPIR-V payloads.
    Spirv,
    /// Portable graphics API driver, WGSL payloads.
    Wgpu,
}

impl ProductionBackend {
    /// Every recorded decision.
    pub const ALL: &'static [Self] = &[Self::Cuda, Self::Metal, Self::Spirv, Self::Wgpu];

    /// Backend id this registration submits into the registry.
    #[must_use]
    pub const fn registry_id(self) -> &'static str {
        match self {
            Self::Cuda => "cuda",
            Self::Metal => "metal",
            Self::Spirv => "spirv",
            Self::Wgpu => "wgpu",
        }
    }

    /// Workspace member that submits this registration.
    #[must_use]
    pub const fn driver_crate(self) -> &'static str {
        match self {
            Self::Cuda => "vyre-driver-cuda",
            Self::Metal => "vyre-driver-metal",
            Self::Spirv => "vyre-driver-spirv",
            Self::Wgpu => "vyre-driver-wgpu",
        }
    }

    /// Domain a dispatch through this registration executes in.
    #[must_use]
    pub const fn execution_domain(self) -> ExecutionDomain {
        match self {
            Self::Cuda | Self::Metal | Self::Spirv | Self::Wgpu => ExecutionDomain::Device,
        }
    }

    /// The recorded decision for `id`, or `None` when the registry reported a
    /// backend nobody decided anything about.
    #[must_use]
    pub fn from_registry_id(id: &str) -> Option<Self> {
        Self::ALL
            .iter()
            .copied()
            .find(|backend| backend.registry_id() == id)
    }

    /// Every decided backend id.
    #[must_use]
    pub fn registry_ids() -> Vec<&'static str> {
        Self::ALL
            .iter()
            .copied()
            .map(Self::registry_id)
            .collect::<Vec<_>>()
    }

    /// Every decided driver crate.
    #[must_use]
    pub fn driver_crates() -> Vec<&'static str> {
        Self::ALL
            .iter()
            .copied()
            .map(Self::driver_crate)
            .collect::<Vec<_>>()
    }
}

/// The recorded decision for a registry id, or a failure naming the id.
///
/// The message states the two ways to make the run green, because an id with no
/// decision is either a new device backend somebody has to record or a host
/// path somebody has to stop registering, and the reader cannot tell which.
///
/// # Errors
/// Returns the failure text when `id` carries no recorded decision.
pub fn decision_for(id: &str) -> Result<ProductionBackend, String> {
    ProductionBackend::from_registry_id(id).ok_or_else(|| {
        format!(
            "backend `{id}` is in the production backend registry and no execution-domain \
             decision is recorded for it. Fix: if it dispatches to a device, add its variant to \
             `vyre_test_support::backend_execution_domain::ProductionBackend` and record \
             `ExecutionDomain::Device` for it; if it evaluates the program on the host, delete \
             its registration and reach it through the oracle seam instead. Decided ids: {:?}",
            ProductionBackend::registry_ids()
        )
    })
}

/// Assert `id` names a registration whose dispatch leaves the host.
///
/// # Panics
/// Panics when `id` carries no recorded decision, or when its recorded domain
/// is [`ExecutionDomain::Host`].
pub fn assert_dispatch_leaves_the_host(id: &str) {
    let backend = match decision_for(id) {
        Ok(backend) => backend,
        Err(failure) => panic!("{failure}"),
    };
    assert_eq!(
        backend.execution_domain(),
        ExecutionDomain::Device,
        "Fix: backend `{id}`, registered by `{}`, evaluates the program on the host, so linking \
         its crate puts a CPU execution route behind the production dispatch seam. Delete the \
         registration and reach the interpreter through the oracle seam instead.",
        backend.driver_crate()
    );
}
