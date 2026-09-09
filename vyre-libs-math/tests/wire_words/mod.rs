//! Wire helpers for tests.
#![allow(dead_code, unused_imports, unused_variables)]

pub(crate) struct Lcg(pub(crate) u64);

impl Lcg {
    pub(crate) fn new(seed: u64) -> Self {
        Self(seed)
    }

    pub(crate) fn next_u32(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }

    pub(crate) fn below(&mut self, n: u32) -> u32 {
        if n == 0 {
            0
        } else {
            self.next_u32() % n
        }
    }
}

pub(crate) use vyre_primitives::wire::decode_u32_le_bytes_all as decode_u32_words;
pub(crate) use vyre_primitives::wire::pack_u32_slice as u32_bytes;

pub(crate) fn lcg_u32(seed: u32, len: usize) -> Vec<u32> {
    let mut state = seed;
    (0..len)
        .map(|idx| {
            state = state
                .wrapping_mul(1_664_525)
                .wrapping_add(1_013_904_223)
                .wrapping_add(idx as u32);
            state
        })
        .collect()
}

pub(crate) fn ramp(count: usize, start: u32, step: u32) -> Vec<u32> {
    (0..count)
        .map(|i| start.wrapping_add((i as u32).wrapping_mul(step)))
        .collect()
}

pub(crate) fn alternating(count: usize, a: u32, b: u32) -> Vec<u32> {
    (0..count).map(|i| if i % 2 == 0 { a } else { b }).collect()
}
pub(crate) fn prefix_scan_cpu_ref(
    input: &[u32],
    kind: vyre_libs_math::math::prefix_scan::ScanKind,
) -> Vec<u32> {
    match kind {
        vyre_libs_math::math::prefix_scan::ScanKind::InclusiveSum => {
            vyre_reference::composition_witness::inclusive_prefix_sum_witness(input)
        }
        vyre_libs_math::math::prefix_scan::ScanKind::ExclusiveSum => {
            vyre_reference::composition_witness::exclusive_prefix_sum_witness(input)
        }
    }
}
