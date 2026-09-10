//! Source-derived host support matrix and canonical byte order.
//!
//! The cell space is closed by the compiler. [`HostOs`] and [`HostArch`] carry
//! no open variant, every classification below matches them exhaustively with
//! no catch-all arm, and `current()` refuses to build on a target none of them
//! names. Adding an operating system or an architecture therefore stops this
//! crate compiling until somebody states what is claimed for the new cell.

use core::fmt;
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Canonical schema version for `PlatformSupportMatrix`.
///
/// Version 3 replaced the flat supported-host list with tiered cells and
/// dropped the open `Other` variants that let an unnamed host be recorded as
/// supported. A version 1 or 2 payload states claims this reader no longer
/// makes and is rejected.
pub const PLATFORM_SUPPORT_MATRIX_SCHEMA_VERSION: u32 = 3;

/// Minimum Rust toolchain, read from the manifest that declares it.
pub const CANONICAL_RUST_VERSION: &str = env!("CARGO_PKG_RUST_VERSION");

// Program identity counters, interned program ids and the substrate query
// cache are all `AtomicU64`. A target without a native 64-bit atomic fails
// today with a dozen unresolved-import errors spread across six modules,
// which names neither the requirement nor the remedy.
#[cfg(not(target_has_atomic = "64"))]
compile_error!(
    "vyre-foundation requires a native 64-bit atomic and this target has none. \
     Fix: build for a host whose architecture provides one. 32-bit PowerPC, MIPS \
     and RISC-V hosts are excluded from the published support matrix for this \
     reason and no cell claims them."
);

/// Supported host operating systems.
///
/// Closed on purpose. A host this enum cannot name is rejected at compile
/// time, because a build that records itself as `Other` claims support for a
/// cell no run ever covered.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum HostOs {
    /// Linux.
    Linux,
    /// macOS.
    MacOS,
    /// Microsoft Windows.
    Windows,
    /// FreeBSD.
    FreeBsd,
    /// Android.
    Android,
    /// iOS.
    IOs,
}

impl HostOs {
    /// Every declared operating system, in declaration order.
    pub const ALL: &'static [Self] = &[
        Self::Linux,
        Self::MacOS,
        Self::Windows,
        Self::FreeBsd,
        Self::Android,
        Self::IOs,
    ];

    /// The operating system this build targets.
    pub const fn current() -> Self {
        #[cfg(target_os = "linux")]
        {
            Self::Linux
        }
        #[cfg(target_os = "macos")]
        {
            Self::MacOS
        }
        #[cfg(target_os = "windows")]
        {
            Self::Windows
        }
        #[cfg(target_os = "freebsd")]
        {
            Self::FreeBsd
        }
        #[cfg(target_os = "android")]
        {
            Self::Android
        }
        #[cfg(target_os = "ios")]
        {
            Self::IOs
        }
        #[cfg(not(any(
            target_os = "linux",
            target_os = "macos",
            target_os = "windows",
            target_os = "freebsd",
            target_os = "android",
            target_os = "ios"
        )))]
        {
            compile_error!(
                "unsupported target_os; add the operating system to vyre_foundation::platform::HostOs and state its support tier, or build for a named host"
            );
        }
    }

    /// Stable identifier used in generated documents.
    #[must_use]
    pub const fn id(self) -> &'static str {
        match self {
            Self::Linux => "linux",
            Self::MacOS => "macos",
            Self::Windows => "windows",
            Self::FreeBsd => "freebsd",
            Self::Android => "android",
            Self::IOs => "ios",
        }
    }
}

impl fmt::Display for HostOs {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.id())
    }
}

/// Supported host CPU architectures.
///
/// Closed for the same reason as [`HostOs`]. Pointer width and byte order are
/// facts of the architecture, so each variant answers for both rather than
/// letting a caller pair a width with an architecture that does not have it.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum HostArch {
    /// x86_64 / AMD64.
    X86_64,
    /// 32-bit x86.
    X86,
    /// ARM64 / AArch64.
    AArch64,
    /// ARMv7, 32-bit little-endian.
    Armv7,
    /// RISC-V, 64-bit.
    RiscV64,
    /// 32-bit PowerPC, big-endian.
    PowerPc,
    /// IBM Z, 64-bit big-endian.
    S390x,
    /// WebAssembly, 32-bit.
    Wasm32,
}

impl HostArch {
    /// Every declared architecture, in declaration order.
    pub const ALL: &'static [Self] = &[
        Self::X86_64,
        Self::X86,
        Self::AArch64,
        Self::Armv7,
        Self::RiscV64,
        Self::PowerPc,
        Self::S390x,
        Self::Wasm32,
    ];

    /// The architecture this build targets.
    pub const fn current() -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            Self::X86_64
        }
        #[cfg(target_arch = "x86")]
        {
            Self::X86
        }
        #[cfg(target_arch = "aarch64")]
        {
            Self::AArch64
        }
        #[cfg(target_arch = "arm")]
        {
            Self::Armv7
        }
        #[cfg(target_arch = "riscv64")]
        {
            Self::RiscV64
        }
        #[cfg(target_arch = "powerpc")]
        {
            Self::PowerPc
        }
        #[cfg(target_arch = "s390x")]
        {
            Self::S390x
        }
        #[cfg(target_arch = "wasm32")]
        {
            Self::Wasm32
        }
        #[cfg(not(any(
            target_arch = "x86_64",
            target_arch = "x86",
            target_arch = "aarch64",
            target_arch = "arm",
            target_arch = "riscv64",
            target_arch = "powerpc",
            target_arch = "s390x",
            target_arch = "wasm32"
        )))]
        {
            compile_error!(
                "unsupported target_arch; add the architecture to vyre_foundation::platform::HostArch with its pointer width and byte order, or build for a named host"
            );
        }
    }

    /// Stable identifier used in generated documents.
    #[must_use]
    pub const fn id(self) -> &'static str {
        match self {
            Self::X86_64 => "x86_64",
            Self::X86 => "x86",
            Self::AArch64 => "aarch64",
            Self::Armv7 => "armv7",
            Self::RiscV64 => "riscv64",
            Self::PowerPc => "powerpc",
            Self::S390x => "s390x",
            Self::Wasm32 => "wasm32",
        }
    }

    /// Pointer width of this architecture.
    #[must_use]
    pub const fn pointer_width(self) -> PointerWidth {
        match self {
            Self::X86_64 | Self::AArch64 | Self::RiscV64 | Self::S390x => PointerWidth::Bits64,
            Self::X86 | Self::Armv7 | Self::PowerPc | Self::Wasm32 => PointerWidth::Bits32,
        }
    }

    /// Byte order of this architecture.
    #[must_use]
    pub const fn endianness(self) -> Endianness {
        match self {
            Self::X86_64
            | Self::X86
            | Self::AArch64
            | Self::Armv7
            | Self::RiscV64
            | Self::Wasm32 => Endianness::LittleEndian,
            Self::PowerPc | Self::S390x => Endianness::BigEndian,
        }
    }
}

impl fmt::Display for HostArch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.id())
    }
}

/// Pointer width in bits.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PointerWidth {
    /// 32-bit pointer width.
    Bits32,
    /// 64-bit pointer width.
    Bits64,
}

impl PointerWidth {
    /// Pointer width this build targets.
    pub const fn current() -> Self {
        #[cfg(target_pointer_width = "64")]
        {
            Self::Bits64
        }
        #[cfg(target_pointer_width = "32")]
        {
            Self::Bits32
        }
        #[cfg(not(any(target_pointer_width = "64", target_pointer_width = "32")))]
        {
            compile_error!(
                "unsupported target_pointer_width; vyre requires 64-bit or 32-bit pointer width"
            );
        }
    }

    /// Stable identifier used in generated documents.
    #[must_use]
    pub const fn id(self) -> &'static str {
        match self {
            Self::Bits32 => "32",
            Self::Bits64 => "64",
        }
    }
}

impl fmt::Display for PointerWidth {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Bits32 => f.write_str("32-bit"),
            Self::Bits64 => f.write_str("64-bit"),
        }
    }
}

/// Byte order.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Endianness {
    /// Little-endian, the canonical wire order.
    LittleEndian,
    /// Big-endian.
    BigEndian,
}

impl Endianness {
    /// Byte order this build targets.
    pub const fn current() -> Self {
        #[cfg(target_endian = "little")]
        {
            Self::LittleEndian
        }
        #[cfg(target_endian = "big")]
        {
            Self::BigEndian
        }
        #[cfg(not(any(target_endian = "little", target_endian = "big")))]
        {
            compile_error!(
                "unsupported target_endian; vyre requires little-endian or big-endian byte order"
            );
        }
    }

    /// Stable identifier used in generated documents.
    #[must_use]
    pub const fn id(self) -> &'static str {
        match self {
            Self::LittleEndian => "little_endian",
            Self::BigEndian => "big_endian",
        }
    }
}

impl fmt::Display for Endianness {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LittleEndian => f.write_str("little-endian"),
            Self::BigEndian => f.write_str("big-endian"),
        }
    }
}

/// What a build claims for one host cell.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum HostSupportTier {
    /// The compiler runs here and drives a device here.
    ///
    /// A claim at this tier needs a run on the cell itself, not a
    /// cross-compilation of it.
    Runtime,
    /// Host arithmetic, canonical encoding and decoding are claimed here.
    ///
    /// Nothing about device execution or performance is claimed. This is the
    /// tier of the cells that exist to prove the identity bytes do not move
    /// with pointer width or byte order.
    Encoding,
    /// Nothing is claimed for this cell.
    Excluded,
}

impl HostSupportTier {
    /// Stable identifier used in generated documents.
    #[must_use]
    pub const fn id(self) -> &'static str {
        match self {
            Self::Runtime => "runtime",
            Self::Encoding => "encoding",
            Self::Excluded => "excluded",
        }
    }
}

impl fmt::Display for HostSupportTier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.id())
    }
}

/// One host execution cell.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Deserialize, Serialize)]
pub struct HostCell {
    /// Host operating system.
    pub os: HostOs,
    /// Host CPU architecture.
    pub arch: HostArch,
    /// Pointer width, as the architecture defines it.
    pub pointer_width: PointerWidth,
    /// Byte order, as the architecture defines it.
    pub endianness: Endianness,
}

impl HostCell {
    /// The cell for one operating system and architecture.
    #[must_use]
    pub const fn new(os: HostOs, arch: HostArch) -> Self {
        Self {
            os,
            arch,
            pointer_width: arch.pointer_width(),
            endianness: arch.endianness(),
        }
    }

    /// The cell this build targets.
    #[must_use]
    pub const fn current() -> Self {
        Self {
            os: HostOs::current(),
            arch: HostArch::current(),
            pointer_width: PointerWidth::current(),
            endianness: Endianness::current(),
        }
    }

    /// The target triple prefix a reader recognizes this cell by.
    #[must_use]
    pub fn label(&self) -> String {
        format!("{}/{}", self.os.id(), self.arch.id())
    }
}

impl fmt::Display for HostCell {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} on {} ({}, {})",
            self.os, self.arch, self.pointer_width, self.endianness
        )
    }
}

/// Rejection of a host cell, or of a matrix payload describing one.
#[derive(Clone, Debug, Eq, PartialEq, Error)]
pub enum UnsupportedPlatformError {
    /// The cell is named by the matrix but nothing is claimed for it.
    #[error(
        "host cell {cell} is excluded from every support claim. Fix: run vyre on a runtime-tier cell, or add a support tier for this cell in vyre_foundation::platform and record a verifying run for it in docs/generated/platform-support-matrix.toml."
    )]
    ExcludedHost {
        /// The rejected cell.
        cell: HostCell,
    },
    /// The cell carries an encoding claim, and the caller asked for runtime.
    #[error(
        "host cell {cell} is claimed at the {tier} tier only: canonical encoding and decoding are proven here, device execution is not. Fix: compile artifacts here and execute them on a runtime-tier cell, or record a runtime verification for this cell in docs/generated/platform-support-matrix.toml."
    )]
    EncodingOnlyHost {
        /// The rejected cell.
        cell: HostCell,
        /// The tier the cell does carry.
        tier: HostSupportTier,
    },
    /// The payload was written against different rules.
    #[error(
        "platform matrix schema version {found} is not {expected}. Fix: regenerate the payload with `cargo xtask platform-support-matrix --write`; a stale payload states claims this reader does not make."
    )]
    StaleSchemaVersion {
        /// Version this reader accepts.
        expected: u32,
        /// Version the payload carries.
        found: u32,
    },
    /// The payload could not be represented or parsed.
    #[error("platform matrix serialization failed: {0}")]
    Serialization(String),
}

/// The host cells a build claims, and the byte order every persisted payload
/// is written in.
///
/// Backend support is not stated here. Which backends a build carries is
/// answered by the registrations linked into it, so a second roster in a
/// substrate-neutral crate would name concrete backends and go stale against
/// the one that decides.
#[derive(Clone, Debug, Eq, PartialEq, Deserialize, Serialize)]
pub struct PlatformSupportMatrix {
    /// Schema version for fail-closed validation.
    pub schema_version: u32,
    /// Minimum Rust toolchain every cell requires.
    pub rust_version: String,
    /// Byte order every persisted payload is written in.
    pub canonical_endianness: Endianness,
    /// Every declared cell with its tier, in `(os, arch)` declaration order.
    pub cells: Vec<(HostCell, HostSupportTier)>,
}

impl PlatformSupportMatrix {
    /// The declared matrix, built from the closed cell space.
    #[must_use]
    pub fn canonical() -> Self {
        let mut cells = Vec::with_capacity(HostOs::ALL.len() * HostArch::ALL.len());
        for &os in HostOs::ALL {
            for &arch in HostArch::ALL {
                cells.push((HostCell::new(os, arch), Self::tier(os, arch)));
            }
        }
        Self {
            schema_version: PLATFORM_SUPPORT_MATRIX_SCHEMA_VERSION,
            rust_version: CANONICAL_RUST_VERSION.to_string(),
            canonical_endianness: Endianness::LittleEndian,
            cells,
        }
    }

    /// The tier claimed for one cell.
    ///
    /// The match is exhaustive over both enums with no catch-all arm. Adding
    /// an operating system or an architecture fails to compile here, which is
    /// the only mechanism that forces a decision at the moment the cell
    /// appears rather than years later through a wrong claim.
    #[must_use]
    pub const fn tier(os: HostOs, arch: HostArch) -> HostSupportTier {
        match (os, arch) {
            // Desktop and server cells the compiler runs and dispatches on.
            (HostOs::Linux | HostOs::MacOS | HostOs::Windows, HostArch::X86_64 | HostArch::AArch64) => {
                HostSupportTier::Runtime
            }
            // Cells that exist to prove the canonical identity bytes do not
            // move with pointer width or byte order. Each is reached by
            // cross-compilation, and the big-endian one is executed under a
            // user-mode emulator.
            (HostOs::Linux, HostArch::X86 | HostArch::S390x) => HostSupportTier::Encoding,
            // Architectures with a Linux toolchain but no verifying run.
            // 32-bit PowerPC additionally has no native 64-bit atomic, which
            // the compile-time check above rejects outright.
            (
                HostOs::Linux,
                HostArch::Armv7 | HostArch::PowerPc | HostArch::RiscV64 | HostArch::Wasm32,
            ) => HostSupportTier::Excluded,
            // Mobile and BSD hosts. No lane holds one, so nothing is claimed.
            (HostOs::Android | HostOs::IOs | HostOs::FreeBsd, _) => HostSupportTier::Excluded,
            // Remaining desktop pairings that no toolchain targets.
            (
                HostOs::MacOS | HostOs::Windows,
                HostArch::X86
                | HostArch::Armv7
                | HostArch::RiscV64
                | HostArch::PowerPc
                | HostArch::S390x
                | HostArch::Wasm32,
            ) => HostSupportTier::Excluded,
        }
    }

    /// Cells claimed at the runtime tier.
    #[must_use]
    pub fn runtime_cells(&self) -> Vec<HostCell> {
        self.cells
            .iter()
            .filter(|(_, tier)| *tier == HostSupportTier::Runtime)
            .map(|(cell, _)| *cell)
            .collect()
    }

    /// Cells claimed at the encoding tier.
    #[must_use]
    pub fn encoding_cells(&self) -> Vec<HostCell> {
        self.cells
            .iter()
            .filter(|(_, tier)| *tier == HostSupportTier::Encoding)
            .map(|(cell, _)| *cell)
            .collect()
    }

    /// Reject a cell that does not carry a runtime claim.
    ///
    /// # Errors
    ///
    /// Returns [`UnsupportedPlatformError::EncodingOnlyHost`] for a cell that
    /// is proven for encoding only, and
    /// [`UnsupportedPlatformError::ExcludedHost`] for a cell nothing claims.
    pub fn require_runtime(&self, cell: &HostCell) -> Result<(), UnsupportedPlatformError> {
        match Self::tier(cell.os, cell.arch) {
            HostSupportTier::Runtime => Ok(()),
            tier @ HostSupportTier::Encoding => {
                Err(UnsupportedPlatformError::EncodingOnlyHost { cell: *cell, tier })
            }
            HostSupportTier::Excluded => Err(UnsupportedPlatformError::ExcludedHost { cell: *cell }),
        }
    }

    /// Reject the running host when it carries no runtime claim.
    ///
    /// # Errors
    ///
    /// Returns the same rejections as [`Self::require_runtime`].
    pub fn validate_active_environment(&self) -> Result<HostCell, UnsupportedPlatformError> {
        let current = HostCell::current();
        self.require_runtime(&current)?;
        Ok(current)
    }

    /// Render the matrix as TOML.
    ///
    /// # Errors
    ///
    /// Returns [`UnsupportedPlatformError::Serialization`] when the record
    /// cannot be represented.
    pub fn to_toml(&self) -> Result<String, UnsupportedPlatformError> {
        toml::to_string_pretty(self)
            .map_err(|error| UnsupportedPlatformError::Serialization(error.to_string()))
    }

    /// Parse a matrix payload, rejecting a version written under other rules.
    ///
    /// # Errors
    ///
    /// Returns [`UnsupportedPlatformError::StaleSchemaVersion`] for a payload
    /// from another schema, and [`UnsupportedPlatformError::Serialization`]
    /// when the payload cannot be parsed.
    pub fn from_toml(payload: &str) -> Result<Self, UnsupportedPlatformError> {
        let matrix: Self = toml::from_str(payload)
            .map_err(|error| UnsupportedPlatformError::Serialization(error.to_string()))?;
        if matrix.schema_version != PLATFORM_SUPPORT_MATRIX_SCHEMA_VERSION {
            return Err(UnsupportedPlatformError::StaleSchemaVersion {
                expected: PLATFORM_SUPPORT_MATRIX_SCHEMA_VERSION,
                found: matrix.schema_version,
            });
        }
        Ok(matrix)
    }
}
