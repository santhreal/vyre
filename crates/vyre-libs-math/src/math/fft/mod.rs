//! Fast Fourier Transform sub-dialect.
//!
//! FFT convolution for large kernels. `fft_radix2_complex` owns the transform
//! for every power-of-two N. `fft4_complex` is a fixed-size entry point that
//! builds through it at N=4, and `fft_convolve_circular_complex` is the
//! circular convolution wrapper.
//!
//! Complex values are represented as interleaved (re, im) pairs in
//! a length-`2 * N` F32 buffer. `fft4_complex` consumes a length-8
//! buffer (4 complex values) and produces a length-8 output (4
//! complex frequency bins).
//!
//! ## Why 4-point first
//!
//! The 4-point FFT is the smallest non-trivial DFT: it has
//! distinct twiddle factors (`W_4 = 1, -i, -1, i`) and exercises
//! every code path of the radix-2 butterfly (real-axis combine,
//! imaginary-axis combine, sign flip, cross-axis swap). A working
//! 4-point reference unblocks the recursive caller for N=8, 16,
//! ... powers of two; convolution then composes forward FFTs,
//! pointwise complex multiply, and inverse FFT.

pub(crate) mod complex_length;
pub(crate) mod convolution;
pub(crate) mod fft4;
pub(crate) mod fft_radix2;

pub use convolution::fft_convolve_circular_complex;
pub use fft4::fft4_complex;
pub use fft_radix2::fft_radix2_complex;
