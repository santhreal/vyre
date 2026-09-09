//! Source-derived host support matrix and canonical byte order.

use core::fmt;
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Canonical schema version for `PlatformSupportMatrix`.
///
/// Version 2 dropped the backend roster the record used to carry. A version 1
/// payload names backends the reader no longer decides and is rejected.
pub const PLATFORM_SUPPORT_MATRIX_SCHEMA_VERSION: u32 = 2;

/// Supported host operating systems.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum HostOs {
    /// Linux (Ubuntu, Debian, RHEL, Fedora, Alpine).
    Linux,
    /// macOS (Darwin, Apple Silicon & Intel).
    MacOS,
    /// Microsoft Windows (x86_64, aarch64).
    Windows,
    /// FreeBSD.
    FreeBsd,
    /// Android.
    Android,
    /// iOS.
    IOs,
    /// Other or custom operating system.
    Other(String),
}

impl HostOs {
    /// Return the currently running host operating system.
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
            Self::Other(String::new())
        }
    }
}

impl fmt::Display for HostOs {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Linux => write!(f, "linux"),
            Self::MacOS => write!(f, "macos"),
            Self::Windows => write!(f, "windows"),
            Self::FreeBsd => write!(f, "freebsd"),
            Self::Android => write!(f, "android"),
            Self::IOs => write!(f, "ios"),
            Self::Other(name) => write!(f, "other({name})"),
        }
    }
}

/// Supported host CPU architectures.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum HostArch {
    /// x86_64 / AMD64.
    X86_64,
    /// ARM64 / AArch64.
    AArch64,
    /// RISC-V 64-bit.
    RiscV64,
    /// ARMv7 (32-bit).
    Armv7,
    /// WebAssembly 32-bit.
    Wasm32,
    /// Other or custom architecture.
    Other(String),
}

impl HostArch {
    /// Return the current host CPU architecture.
    pub const fn current() -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            Self::X86_64
        }
        #[cfg(target_arch = "aarch64")]
        {
            Self::AArch64
        }
        #[cfg(target_arch = "riscv64")]
        {
            Self::RiscV64
        }
        #[cfg(target_arch = "arm")]
        {
            Self::Armv7
        }
        #[cfg(target_arch = "wasm32")]
        {
            Self::Wasm32
        }
        #[cfg(not(any(
            target_arch = "x86_64",
            target_arch = "aarch64",
            target_arch = "riscv64",
            target_arch = "arm",
            target_arch = "wasm32"
        )))]
        {
            Self::Other(String::new())
        }
    }
}

impl fmt::Display for HostArch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::X86_64 => write!(f, "x86_64"),
            Self::AArch64 => write!(f, "aarch64"),
            Self::RiscV64 => write!(f, "riscv64"),
            Self::Armv7 => write!(f, "armv7"),
            Self::Wasm32 => write!(f, "wasm32"),
            Self::Other(name) => write!(f, "other({name})"),
        }
    }
}

/// Pointer width in bits.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PointerWidth {
    /// 32-bit pointer width (4-byte pointers).
    Bits32,
    /// 64-bit pointer width (8-byte pointers).
    Bits64,
}

impl PointerWidth {
    /// Return current pointer width.
    pub const fn current() -> Self {
        if core::mem::size_of::<usize>() == 8 {
            Self::Bits64
        } else {
            Self::Bits32
        }
    }
}

/// Endianness byte order.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Endianness {
    /// Little-endian (canonical wire format).
    LittleEndian,
    /// Big-endian.
    BigEndian,
}

impl Endianness {
    /// Return current host endianness.
    pub const fn current() -> Self {
        #[cfg(target_endian = "little")]
        {
            Self::LittleEndian
        }
        #[cfg(target_endian = "big")]
        {
            Self::BigEndian
        }
    }
}

/// Hardware capability profile description.
#[derive(Clone, Debug, Eq, Hash, PartialEq, Deserialize, Serialize)]
pub struct DeviceCapabilityProfile {
    /// Subgroup width in lanes (32 or 64 on shipped devices).
    pub subgroup_size: u32,
    /// Maximum workgroup invocations.
    pub max_invocations: u32,
    /// Shared memory capacity in kilobytes per block.
    pub shared_memory_kb: u32,
    /// Native 16-bit floating point math support (fp16).
    pub fp16_support: bool,
    /// Native Brain Float 16 support (bf16).
    pub bf16_support: bool,
    /// Tensor core / matrix hardware instructions.
    pub tensor_cores: bool,
}

/// Host execution cell representation.
#[derive(Clone, Debug, Eq, Hash, PartialEq, Deserialize, Serialize)]
pub struct HostCell {
    /// Host operating system.
    pub os: HostOs,
    /// Host CPU architecture.
    pub arch: HostArch,
    /// Pointer width in bits.
    pub pointer_width: PointerWidth,
    /// Byte endianness.
    pub endianness: Endianness,
    /// Minimum supported Rust toolchain version.
    pub rust_version: String,
}

impl HostCell {
    /// Probe the currently running host cell.
    pub fn current() -> Self {
        Self {
            os: HostOs::current(),
            arch: HostArch::current(),
            pointer_width: PointerWidth::current(),
            endianness: Endianness::current(),
            rust_version: "1.80.0".to_string(),
        }
    }
}

/// Error returned when an unsupported platform or host cell is evaluated.
#[derive(Debug, Error)]
pub enum UnsupportedPlatformError {
    /// Unsupported host operating system or architecture.
    #[error("unsupported host target cell: {os} on {arch} ({pointer_width:?}, {endianness:?})")]
    UnsupportedHost {
        /// Target OS.
        os: HostOs,
        /// Target CPU arch.
        arch: HostArch,
        /// Pointer width.
        pointer_width: PointerWidth,
        /// Byte order.
        endianness: Endianness,
    },
    /// Stale schema version.
    #[error("stale platform matrix schema version: expected {expected}, found {found}")]
    StaleSchemaVersion {
        /// Expected version.
        expected: u32,
        /// Found version.
        found: u32,
    },
    /// Serialization error.
    #[error("serialization error: {0}")]
    Serialization(String),
}

/// The host cells a build is supported on, and the byte order every persisted
/// payload is written in.
///
/// Backend support is not stated here. Which backends a build carries is
/// answered by the registrations linked into it, so a second roster in a
/// substrate-neutral crate would name concrete backends and go stale against
/// the one that decides.
#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
pub struct PlatformSupportMatrix {
    /// Schema version for fail-closed validation.
    pub schema_version: u32,
    /// Supported host execution cells.
    pub supported_hosts: Vec<HostCell>,
    /// Required canonical endianness for persistent wire payloads.
    pub canonical_endianness: Endianness,
}

impl PlatformSupportMatrix {
    /// Generate the canonical declared support matrix.
    pub fn canonical() -> Self {
        let supported_hosts = vec![
            HostCell {
                os: HostOs::Linux,
                arch: HostArch::X86_64,
                pointer_width: PointerWidth::Bits64,
                endianness: Endianness::LittleEndian,
                rust_version: "1.80.0".to_string(),
            },
            HostCell {
                os: HostOs::Linux,
                arch: HostArch::AArch64,
                pointer_width: PointerWidth::Bits64,
                endianness: Endianness::LittleEndian,
                rust_version: "1.80.0".to_string(),
            },
            HostCell {
                os: HostOs::MacOS,
                arch: HostArch::AArch64,
                pointer_width: PointerWidth::Bits64,
                endianness: Endianness::LittleEndian,
                rust_version: "1.80.0".to_string(),
            },
            HostCell {
                os: HostOs::MacOS,
                arch: HostArch::X86_64,
                pointer_width: PointerWidth::Bits64,
                endianness: Endianness::LittleEndian,
                rust_version: "1.80.0".to_string(),
            },
            HostCell {
                os: HostOs::Windows,
                arch: HostArch::X86_64,
                pointer_width: PointerWidth::Bits64,
                endianness: Endianness::LittleEndian,
                rust_version: "1.80.0".to_string(),
            },
            HostCell {
                os: HostOs::Windows,
                arch: HostArch::AArch64,
                pointer_width: PointerWidth::Bits64,
                endianness: Endianness::LittleEndian,
                rust_version: "1.80.0".to_string(),
            },
        ];

        Self {
            schema_version: PLATFORM_SUPPORT_MATRIX_SCHEMA_VERSION,
            supported_hosts,
            canonical_endianness: Endianness::LittleEndian,
        }
    }

    /// Check whether a given host cell is officially supported.
    pub fn is_supported(&self, host: &HostCell) -> Result<(), UnsupportedPlatformError> {
        let found = self.supported_hosts.iter().any(|supported| {
            supported.os == host.os
                && supported.arch == host.arch
                && supported.pointer_width == host.pointer_width
                && supported.endianness == host.endianness
        });

        if found {
            Ok(())
        } else {
            Err(UnsupportedPlatformError::UnsupportedHost {
                os: host.os.clone(),
                arch: host.arch.clone(),
                pointer_width: host.pointer_width,
                endianness: host.endianness,
            })
        }
    }

    /// Validate that the currently running host environment is supported.
    pub fn validate_active_environment(&self) -> Result<HostCell, UnsupportedPlatformError> {
        let current = HostCell::current();
        self.is_supported(&current)?;
        Ok(current)
    }

    /// Serialize matrix to TOML.
    pub fn to_toml(&self) -> Result<String, UnsupportedPlatformError> {
        toml::to_string_pretty(self)
            .map_err(|e| UnsupportedPlatformError::Serialization(e.to_string()))
    }

    /// Deserialize matrix from TOML, validating schema version fail-closed.
    pub fn from_toml(toml_str: &str) -> Result<Self, UnsupportedPlatformError> {
        let matrix: Self = toml::from_str(toml_str)
            .map_err(|e| UnsupportedPlatformError::Serialization(e.to_string()))?;
        if matrix.schema_version != PLATFORM_SUPPORT_MATRIX_SCHEMA_VERSION {
            return Err(UnsupportedPlatformError::StaleSchemaVersion {
                expected: PLATFORM_SUPPORT_MATRIX_SCHEMA_VERSION,
                found: matrix.schema_version,
            });
        }
        Ok(matrix)
    }
}
