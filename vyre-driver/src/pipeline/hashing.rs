//! Stable hashing helpers for compiled-pipeline cache identity.

use crate::backend::DispatchConfig;
use vyre_foundation::ir::Program;

/// Return the normalized program digest used by backend pipeline caches.
///
/// Thin forwarder: the digest and its per-`Program` memo are owned by
/// [`vyre_foundation::ir::Program::try_normalized_cache_digest`], so the
/// algorithm has exactly one implementation and is computed at most once per
/// program value instead of once per dispatch.
///
/// # Errors
///
/// Returns when the program contains an IR type or node shape that cannot be
/// serialized into stable cache identity. Dispatch admission should surface the
/// error rather than panic or generate a lossy cache key.
pub fn try_normalized_program_cache_digest(program: &Program) -> Result<[u8; 32], String> {
    program.try_normalized_cache_digest()
}

/// Return the normalized program digest used by backend pipeline caches.
#[must_use]
pub fn normalized_program_cache_digest(program: &Program) -> [u8; 32] {
    try_normalized_program_cache_digest(program).unwrap_or([0u8; 32])
}

/// Append dispatch policy fields that alter generated backend code to a cache
/// hasher.
pub fn update_dispatch_policy_cache_hash(hasher: &mut blake3::Hasher, config: &DispatchConfig) {
    hasher.update(b"ulp\0");
    match config.ulp_budget {
        Some(ulp) => {
            hasher.update(&[1, ulp]);
        }
        None => {
            hasher.update(&[0, 0]);
        }
    };
    hasher.update(b"\0wg\0");
    match config.launch_workgroup() {
        Some(workgroup) => {
            hasher.update(&[1]);
            for axis in workgroup {
                hasher.update(&axis.to_le_bytes());
            }
        }
        None => {
            hasher.update(&[0]);
        }
    };
    // Two rounding modes over one program are two different modules, so the
    // mode and the contract it belongs to are both in the key. The contract
    // version is what makes an artifact cached before the mode existed miss
    // instead of being served to a strict-mode dispatch.
    hasher.update(b"\0float\0");
    hasher.update(config.float_lowering.cache_label().as_bytes());
    hasher.update(&vyre_foundation::fp_parity::FLOAT_LOWERING_CONTRACT_VERSION.to_le_bytes());
    // The ceiling decides whether the launch is folded across grid axes, and a
    // folded launch is emitted with a grid-linearized element index, so two
    // ceilings over one program are two different modules.
    hasher.update(b"\0axes\0");
    match config.max_workgroups_per_axis {
        Some(limits) => {
            hasher.update(&[1]);
            for axis in limits {
                hasher.update(&axis.to_le_bytes());
            }
        }
        None => {
            hasher.update(&[0]);
        }
    };
}

/// Return the dispatch-policy digest used inside backend cache keys.
///
/// This keeps policy serialization single-sourced while letting backend cache
/// identities use the shared tuple-boundary-preserving key envelope instead of
/// owning a second ad hoc hasher sequence.
#[must_use]
pub fn dispatch_policy_cache_digest(config: &DispatchConfig) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    update_dispatch_policy_cache_hash(&mut hasher, config);
    *hasher.finalize().as_bytes()
}

/// Human-readable dispatch policy fingerprint for cache metadata.
#[must_use]
pub fn dispatch_policy_cache_string(config: &DispatchConfig) -> String {
    // "ulp=" (4) + max u8 decimal (3) + ":wg=" (4) + workgroup repr
    // (~32) + ":float=" (7) + mode label (~11) ≈ 96 bytes worst case; pre-size
    // so the push_str calls do not realloc.
    let mut policy = String::with_capacity(96);
    policy.push_str("ulp=");
    push_debug_option_u8(&mut policy, config.ulp_budget);
    policy.push_str(":wg=");
    push_debug_option_workgroup(&mut policy, config.launch_workgroup());
    policy.push_str(":float=");
    policy.push_str(config.float_lowering.cache_label());
    policy
}

/// Hex-encode bytes using lowercase ASCII.
#[must_use]
pub fn hex_encode(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    push_lower_hex(bytes, &mut out);
    out
}

/// Append the lowercase-hex encoding of `bytes` to `out`. Single owner of the
/// lowercase-hex nibble loop for the whole driver.
pub fn push_lower_hex(bytes: &[u8], out: &mut String) {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    for &byte in bytes {
        out.push(HEX[(byte >> 4) as usize] as char);
        out.push(HEX[(byte & 0x0f) as usize] as char);
    }
}

/// Hex-encode the first eight bytes of a 32-byte digest for compact ids.
#[must_use]
pub fn hex_short(bytes: &[u8; 32]) -> String {
    hex_encode(&bytes[..8])
}

/// Stable device fingerprint for persistent pipeline caches.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PipelineDeviceFingerprint {
    /// Vendor identifier.
    pub vendor: u32,
    /// Device identifier.
    pub device: u32,
    /// Cryptographic digest of driver/runtime revision text.
    pub driver_digest: [u8; 32],
}

impl PipelineDeviceFingerprint {
    /// Build a fingerprint from numeric identifiers and revision text.
    #[must_use]
    pub fn from_parts(vendor: u32, device: u32, revision: &str, revision_extra: &str) -> Self {
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"vyre-pipeline-device-fingerprint-v1\0");
        hasher.update(revision.as_bytes());
        hasher.update(b"\0extra\0");
        hasher.update(revision_extra.as_bytes());
        Self {
            vendor,
            device,
            driver_digest: *hasher.finalize().as_bytes(),
        }
    }

    /// Compose a cache key from canonical program digest and device identity.
    #[must_use]
    pub fn cache_key(self, program_digest: [u8; 32]) -> [u8; 32] {
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"vyre-disk-pipeline-cache-key-v1\0program\0");
        hasher.update(&program_digest);
        hasher.update(b"\0vendor\0");
        hasher.update(&self.vendor.to_le_bytes());
        hasher.update(b"\0device\0");
        hasher.update(&self.device.to_le_bytes());
        hasher.update(b"\0driver\0");
        hasher.update(&self.driver_digest);
        *hasher.finalize().as_bytes()
    }
}

pub(super) fn push_debug_option_u8(out: &mut String, value: Option<u8>) {
    match value {
        Some(value) => {
            out.push_str("Some(");
            push_decimal_u8(out, value);
            out.push(')');
        }
        None => out.push_str("None"),
    }
}

pub(super) fn push_debug_option_workgroup(out: &mut String, value: Option<[u32; 3]>) {
    match value {
        Some([x, y, z]) => {
            out.push_str("Some([");
            push_decimal_u32(out, x);
            out.push_str(", ");
            push_decimal_u32(out, y);
            out.push_str(", ");
            push_decimal_u32(out, z);
            out.push_str("])");
        }
        None => out.push_str("None"),
    }
}

pub(super) fn push_decimal_u8(out: &mut String, value: u8) {
    push_decimal_u32(out, u32::from(value));
}

pub(super) fn push_decimal_u32(out: &mut String, value: u32) {
    let mut buf = [0_u8; 10];
    let mut n = value;
    let mut i = buf.len();
    if n == 0 {
        out.push('0');
        return;
    }
    while n > 0 {
        i -= 1;
        buf[i] = b'0' + (n % 10) as u8;
        n /= 10;
    }
    for &digit in &buf[i..] {
        out.push(digit as char);
    }
}

// Inline: covers `push_decimal_u32`, which no integration test can name.
#[cfg(test)]
mod tests {
    use super::{
        dispatch_policy_cache_digest, dispatch_policy_cache_string, hex_encode, push_decimal_u32,
        push_lower_hex, update_dispatch_policy_cache_hash,
    };
    use crate::backend::DispatchConfig;
    use vyre_foundation::fp_parity::FloatLoweringMode;

    #[test]
    fn hex_encode_and_push_lower_hex_agree_on_known_bytes() {
        assert_eq!(hex_encode(&[0x00, 0xff, 0x1a, 0x0f]), "00ff1a0f");
        let mut out = String::from("k=");
        push_lower_hex(&[0xde, 0xad, 0xbe, 0xef], &mut out);
        assert_eq!(out, "k=deadbeef");
    }

    /// WHY: a rounding mode changes the emitted module, so two modes over one
    /// program are two cache entries. The sweep runs the whole
    /// `FloatLoweringMode` roster rather than the pair that existed when this
    /// was written: a mode added without reaching the key would serve
    /// contracted arithmetic to a strict request out of the disk cache.
    #[test]
    fn every_float_lowering_mode_is_its_own_cache_identity() {
        let policy_of = |mode: FloatLoweringMode| {
            let mut config = DispatchConfig::default();
            config.float_lowering = mode;
            (
                dispatch_policy_cache_digest(&config),
                dispatch_policy_cache_string(&config),
            )
        };
        let mut seen: Vec<([u8; 32], String)> = Vec::new();
        for &mode in FloatLoweringMode::EVERY {
            let (digest, string) = policy_of(mode);
            assert!(
                string.contains(mode.cache_label()),
                "Fix: the policy fingerprint must name the float lowering mode; `{string}` omits `{}`.",
                mode.cache_label()
            );
            for (other_digest, other_string) in &seen {
                assert_ne!(
                    *other_digest, digest,
                    "Fix: mix the float lowering mode into the dispatch policy digest; `{}` collides with `{string}`.",
                    other_string
                );
                assert_ne!(
                    *other_string, string,
                    "Fix: give every float lowering mode its own policy fingerprint."
                );
            }
            seen.push((digest, string));
        }
        assert_eq!(seen.len(), FloatLoweringMode::EVERY.len());
    }

    #[test]
    fn push_decimal_u32_renders_boundaries() {
        let mut out = String::new();
        push_decimal_u32(&mut out, 0);
        push_decimal_u32(&mut out, 42);
        push_decimal_u32(&mut out, u32::MAX);
        assert_eq!(out, "0424294967295");
    }

    #[test]
    fn dispatch_policy_cache_digest_matches_shared_hasher_for_generated_configs() {
        for case in 0..4096u32 {
            let mut config = DispatchConfig::default();
            if case & 1 != 0 {
                config.ulp_budget = Some((case as u8).wrapping_mul(17).wrapping_add(1));
            }
            if case & 2 != 0 {
                config.workgroup_override = Some([
                    1 + (case & 255),
                    1 + ((case.rotate_left(7) >> 3) & 31),
                    1 + ((case.rotate_right(5) >> 2) & 7),
                ]);
            }

            let mut hasher = blake3::Hasher::new();
            update_dispatch_policy_cache_hash(&mut hasher, &config);
            assert_eq!(
                dispatch_policy_cache_digest(&config),
                *hasher.finalize().as_bytes(),
                "Fix: dispatch-policy digest must stay single-sourced through update_dispatch_policy_cache_hash for generated case {case}."
            );
        }
    }

    /// WHY: a frozen launch selects the workgroup a backend compiles against, so
    /// two artifacts differing only in that workgroup must not share a pipeline
    /// cache entry. A key that reads only the tuner override would collide them
    /// and run the second launch on the first launch's kernel.
    #[test]
    fn a_frozen_launch_workgroup_separates_pipeline_cache_keys() {
        let launch_of = |workgroup: [u32; 3]| {
            let mut config = DispatchConfig::default();
            config.launch = Some(
                crate::launch_directive::LaunchDirective::stated(workgroup, [8, 1, 1], 0)
                    .expect("the stated fixture launch is positive"),
            );
            config
        };
        let narrow = dispatch_policy_cache_digest(&launch_of([64, 1, 1]));
        let wide = dispatch_policy_cache_digest(&launch_of([256, 1, 1]));
        assert_ne!(
            narrow, wide,
            "Fix: a frozen launch workgroup must reach the pipeline cache key."
        );

        // The same kernel, stated through either authority, is one cache entry:
        // the grid is a launch argument and does not change generated code.
        let mut overridden = DispatchConfig::default();
        overridden.workgroup_override = Some([64, 1, 1]);
        assert_eq!(dispatch_policy_cache_digest(&overridden), narrow);
    }
}
