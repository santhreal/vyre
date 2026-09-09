//! Wire helpers for hash tests.
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

pub(crate) fn lcg_u32(count: usize, seed: u64) -> Vec<u32> {
    let mut rng = Lcg::new(seed);
    (0..count).map(|_| rng.next_u32()).collect()
}

pub(crate) fn ramp(count: usize, start: u32, step: u32) -> Vec<u32> {
    (0..count)
        .map(|i| start.wrapping_add((i as u32).wrapping_mul(step)))
        .collect()
}

pub(crate) fn alternating(count: usize, a: u32, b: u32) -> Vec<u32> {
    (0..count).map(|i| if i % 2 == 0 { a } else { b }).collect()
}

pub(crate) fn hostile_bytes(seed: u32) -> Vec<u8> {
    let len = 1 + (seed as usize % 512);
    let mut v = Vec::with_capacity(len);
    let mut s = seed as u64 ^ 0xDEAD_BEEF_CAFE_BABE;
    for _ in 0..len {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
        v.push(s as u8);
    }
    v
}

pub(crate) fn oracle_blake3_g(
    state: &mut [u32; 16],
    a: usize,
    b: usize,
    c: usize,
    d: usize,
    mx: u32,
    my: u32,
) {
    state[a] = state[a].wrapping_add(state[b]).wrapping_add(mx);
    state[d] = (state[d] ^ state[a]).rotate_right(16);
    state[c] = state[c].wrapping_add(state[d]);
    state[b] = (state[b] ^ state[c]).rotate_right(12);
    state[a] = state[a].wrapping_add(state[b]).wrapping_add(my);
    state[d] = (state[d] ^ state[a]).rotate_right(8);
    state[c] = state[c].wrapping_add(state[d]);
    state[b] = (state[b] ^ state[c]).rotate_right(7);
}
