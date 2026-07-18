//! Assembly inspection probes for the hot arithmetic paths.
//!
//! Each probe is `#[no_mangle]` + `#[inline(never)]` so it appears under a
//! stable name in the generated assembly. `scripts/check_asm.sh` builds this
//! example with `--emit asm` and verifies the codegen expectations, most
//! importantly that **re-scaling by the compile-time 10^DIGITS constant
//! compiles to multiply sequences with no division instructions at all**.
//!
//! Inspect manually with:
//!
//! ```text
//! cargo rustc --release --example asm_probe -- --emit asm -C codegen-units=1
//! less target/release/examples/asm_probe-*.s
//! ```

use fin_decimal::{Amount64, Amount128, Amount256, I256, Rate64, Rate128, Rounding};
use std::hint::black_box;

// ---- multiplication: re-scale by constant 10^4, must be division-free ----

#[unsafe(no_mangle)]
#[inline(never)]
pub fn probe_amount64_mul(a: Amount64, b: Amount64) -> Amount64 {
    a * b
}

#[unsafe(no_mangle)]
#[inline(never)]
pub fn probe_amount128_mul(a: Amount128, b: Amount128) -> Amount128 {
    a * b
}

#[unsafe(no_mangle)]
#[inline(never)]
pub fn probe_amount256_mul(a: Amount256, b: Amount256) -> Amount256 {
    a * b
}

// ---- cross-scale multiplication: re-scale by constant 10^8 ----

#[unsafe(no_mangle)]
#[inline(never)]
pub fn probe_amount64_mul_rate(a: Amount64, r: Rate64) -> Amount64 {
    a.mul_rounded(r, Rounding::HalfUp)
}

#[unsafe(no_mangle)]
#[inline(never)]
pub fn probe_amount128_mul_rate(a: Amount128, r: Rate128) -> Amount128 {
    a.mul_rounded(r, Rounding::HalfUp)
}

// ---- square root: constant re-scale + u128::isqrt, must be division-free ----

#[unsafe(no_mangle)]
#[inline(never)]
pub fn probe_amount64_sqrt(a: Amount64) -> Amount64 {
    a.sqrt_rounded(Rounding::HalfUp)
}

// ---- rounding: divide/multiply by constant 10^4, must be division-free ----

#[unsafe(no_mangle)]
#[inline(never)]
pub fn probe_amount128_round(a: Amount128) -> Amount128 {
    a.round_to(Rounding::HalfEven)
}

#[unsafe(no_mangle)]
#[inline(never)]
pub fn probe_amount256_round(a: Amount256) -> Amount256 {
    a.round_to(Rounding::HalfEven)
}

#[unsafe(no_mangle)]
#[inline(never)]
pub fn probe_amount128_trunc(a: Amount128) -> Amount128 {
    a.trunc()
}

#[unsafe(no_mangle)]
#[inline(never)]
pub fn probe_amount256_trunc(a: Amount256) -> Amount256 {
    a.trunc()
}

#[unsafe(no_mangle)]
#[inline(never)]
pub fn probe_amount128_fract(a: Amount128) -> Amount128 {
    a.fract()
}

// ---- division by a runtime decimal: division instructions are expected,
// ---- but never a call to the compiler's 128-bit builtins (__udivti3) ----

#[unsafe(no_mangle)]
#[inline(never)]
pub fn probe_amount64_div(a: Amount64, b: Amount64) -> Amount64 {
    a / b
}

#[unsafe(no_mangle)]
#[inline(never)]
pub fn probe_amount128_div(a: Amount128, b: Amount128) -> Amount128 {
    a / b
}

#[unsafe(no_mangle)]
#[inline(never)]
pub fn probe_amount256_div(a: Amount256, b: Amount256) -> Amount256 {
    a / b
}

#[unsafe(no_mangle)]
#[inline(never)]
pub fn probe_amount128_rem(a: Amount128, b: Amount128) -> Amount128 {
    a % b
}

#[unsafe(no_mangle)]
#[inline(never)]
pub fn probe_amount64_ratio(a: Amount64, b: Amount64) -> Rate64 {
    a.div_rounded_to::<8>(b, Rounding::HalfUp)
}

#[unsafe(no_mangle)]
#[inline(never)]
pub fn probe_amount128_div_int(a: Amount128, n: i64) -> Amount128 {
    a.div_int_rounded(n, Rounding::HalfUp)
}

#[unsafe(no_mangle)]
#[inline(never)]
pub fn probe_amount64_mul_div(a: Amount64, b: Amount64, c: Amount64) -> Amount64 {
    a.mul_div_rounded(b, c, Rounding::HalfUp)
}

fn main() {
    // Call every probe so all of them are codegen'd.
    let a64 = black_box(Amount64::from(3));
    let b64 = black_box(Amount64::from_bits(10007));
    let a128 = black_box(Amount128::from(3));
    let b128 = black_box(Amount128::from_bits(10007));
    let a256 = black_box(Amount256::from(3));
    let b256 = black_box(Amount256::from_bits(I256::from_i128(10007)));

    let r64 = black_box(Rate64::from_bits(10007));
    let r128 = black_box(Rate128::from_bits(10007));

    println!("{}", black_box(probe_amount64_mul(a64, b64)));
    println!("{}", black_box(probe_amount128_mul(a128, b128)));
    println!("{}", black_box(probe_amount256_mul(a256, b256)));
    println!("{}", black_box(probe_amount64_mul_rate(a64, r64)));
    println!("{}", black_box(probe_amount128_mul_rate(a128, r128)));
    println!("{}", black_box(probe_amount128_round(a128)));
    println!("{}", black_box(probe_amount256_round(a256)));
    println!("{}", black_box(probe_amount128_trunc(a128)));
    println!("{}", black_box(probe_amount256_trunc(a256)));
    println!("{}", black_box(probe_amount128_fract(a128)));
    println!("{}", black_box(probe_amount64_div(a64, b64)));
    println!("{}", black_box(probe_amount128_div(a128, b128)));
    println!("{}", black_box(probe_amount256_div(a256, b256)));
    println!("{}", black_box(probe_amount128_rem(a128, b128)));
    println!("{}", black_box(probe_amount64_ratio(a64, b64)));
    println!(
        "{}",
        black_box(probe_amount128_div_int(a128, black_box(10007)))
    );
    println!("{}", black_box(probe_amount64_sqrt(a64)));
    println!("{}", black_box(probe_amount64_mul_div(a64, b64, b64)));
}
