//! Hash oracles for this crate's tests.

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
