//! Multi-word integer primitives backing the wide decimal types
//! ([`Decimal128`](crate::Decimal128) and [`Decimal256`](crate::Decimal256)).
//!
//! Values are held as little-endian arrays of 64-bit limbs. Two division
//! flavors exist:
//!
//! * [`div_words_by_pow10`] / [`div_rem_u128_pow10`] divide by the
//!   compile-time constant 10^DIGITS via Möller–Granlund reciprocal
//!   multiplication, with the reciprocal computed at compile time: no
//!   division instructions and no `__udivti3` on any architecture. (LLVM
//!   does NOT strength-reduce 128-bit division by a constant, so this must
//!   be done by hand; `scripts/check_asm.sh` verifies it stays that way.)
//!   All decimal re-scaling — the hot path — goes through these, and they
//!   are `const fn`, so whole expressions can be evaluated at compile time.
//! * For runtime divisors the key primitive is [`div_2by1`]: a 128-bit-by-
//!   64-bit division with a 64-bit quotient. On `x86_64` with the `asm`
//!   feature it is the native `div` instruction; the portable fallback is
//!   Knuth's long division on 32-bit half-digits, substantially cheaper than
//!   the compiler's full 128-by-128 `__udivti3`. Word-based long division and
//!   Knuth's algorithm D are built on top of it.

use crate::AmountErrorKind;
use crate::AmountSign;
use crate::Rounding;
use core::cmp::Ordering;

/// Computes 10^pow as u64. Valid for `pow <= 19` (10^19 < 2^64).
///
/// Call sites that need the value folded into generated code must wrap the
/// call in an inline `const { .. }` block: without it, a cross-crate call
/// can survive into codegen and defeat constant propagation entirely.
#[inline]
pub(crate) const fn upow10(pow: u32) -> u64 {
    const P10: [u64; 20] = [
        1,
        10,
        100,
        1_000,
        10_000,
        100_000,
        1_000_000,
        10_000_000,
        100_000_000,
        1_000_000_000,
        10_000_000_000,
        100_000_000_000,
        1_000_000_000_000,
        10_000_000_000_000,
        100_000_000_000_000,
        1_000_000_000_000_000,
        10_000_000_000_000_000,
        100_000_000_000_000_000,
        1_000_000_000_000_000_000,
        10_000_000_000_000_000_000,
    ];
    P10[pow as usize]
}

/// Divide `(hi << 64) | lo` by `d`. Requires `hi < d` so the quotient fits in
/// a single limb. Returns `(quotient, remainder)`.
#[cfg(all(target_arch = "x86_64", feature = "asm"))]
#[inline]
pub(crate) fn div_2by1(hi: u64, lo: u64, d: u64) -> (u64, u64) {
    debug_assert!(hi < d);
    let q: u64;
    let r: u64;
    unsafe {
        core::arch::asm!(
            "div {d}",
            d = in(reg) d,
            inout("rax") lo => q,
            inout("rdx") hi => r,
            options(pure, nomem, nostack)
        );
    }
    (q, r)
}

/// Divide the 128-bit value `(n1 << 64) | n0` by `d`, which must be
/// normalized (top bit set) with `n1 < d`. Returns `(quotient, remainder)`.
///
/// Knuth long division on 32-bit half-digits (the classic `udiv_qrnnd`
/// construction); avoids the compiler's 128-by-128 division builtin.
#[cfg(any(not(target_arch = "x86_64"), not(feature = "asm")))]
fn udiv_qrnnd(n1: u64, n0: u64, d: u64) -> (u64, u64) {
    debug_assert!(d >= 1 << 63 && n1 < d);
    let d_high = d >> 32;
    let d_low = d & 0xffff_ffff;

    let mut q_high = n1 / d_high;
    let mut r_high = n1 - q_high * d_high;
    let m = q_high * d_low;
    r_high = (r_high << 32) | (n0 >> 32);
    // Fine tune the estimate. `wrapping_add(d)` may overflow; when it does the
    // logical value of r_high exceeds any m, detected via `r_high >= d`.
    if r_high < m {
        q_high -= 1;
        r_high = r_high.wrapping_add(d);
        if r_high >= d && r_high < m {
            q_high -= 1;
            r_high = r_high.wrapping_add(d);
        }
    }
    r_high = r_high.wrapping_sub(m);

    let mut q_low = r_high / d_high;
    let mut r_low = r_high - q_low * d_high;
    let m = q_low * d_low;
    r_low = (r_low << 32) | (n0 & 0xffff_ffff);
    if r_low < m {
        q_low -= 1;
        r_low = r_low.wrapping_add(d);
        if r_low >= d && r_low < m {
            q_low -= 1;
            r_low = r_low.wrapping_add(d);
        }
    }
    r_low = r_low.wrapping_sub(m);

    ((q_high << 32) | q_low, r_low)
}

/// Divide `(hi << 64) | lo` by `d`. Requires `hi < d` so the quotient fits in
/// a single limb. Returns `(quotient, remainder)`.
#[cfg(any(not(target_arch = "x86_64"), not(feature = "asm")))]
#[inline]
pub(crate) fn div_2by1(hi: u64, lo: u64, d: u64) -> (u64, u64) {
    debug_assert!(hi < d);
    if hi == 0 {
        return (lo / d, lo % d);
    }
    let s = d.leading_zeros();
    if s == 0 {
        udiv_qrnnd(hi, lo, d)
    } else {
        let (q, r) = udiv_qrnnd((hi << s) | (lo >> (64 - s)), lo << s, d << s);
        (q, r >> s)
    }
}

/// In-place long division of `u` by the single word `d`; `u` becomes the
/// quotient. Returns the remainder.
#[inline]
pub(crate) fn div_words_by_word(u: &mut [u64], d: u64) -> u64 {
    debug_assert!(d != 0);
    let mut rem: u64 = 0;
    for x in u.iter_mut().rev() {
        // Leading limbs smaller than the divisor produce a zero quotient limb
        // and pass straight into the remainder; real values rarely fill all
        // limbs, so this skips most divisions.
        if rem == 0 && *x < d {
            rem = *x;
            *x = 0;
            continue;
        }
        let (q, r) = div_2by1(rem, *x, d);
        *x = q;
        rem = r;
    }
    rem
}

/// Per-scale constants for reciprocal division by 10^DIGITS: the normalized
/// divisor (top bit set), its normalization shift, and the Möller–Granlund
/// reciprocal `floor((2^128 - 1) / dn) - 2^64`. Evaluated at compile time
/// (the 128-bit division here is const-folded, it never reaches codegen).
const fn pow10_norm(digits: u32) -> (u64, u32, u64) {
    let d = upow10(digits);
    let s = d.leading_zeros();
    let dn = d << s;
    let v = (u128::MAX / (dn as u128) - (1u128 << 64)) as u64;
    (dn, s, v)
}

/// Möller–Granlund 2-by-1 division by a normalized divisor `dn` (top bit
/// set) with the precomputed reciprocal `v`: one 64x64->128 multiply plus
/// fixups, no division instruction. Requires `u1 < dn`.
#[inline]
const fn div_2by1_recip(u1: u64, u0: u64, dn: u64, v: u64) -> (u64, u64) {
    // (q1, q0) estimate = v * u1 + (u1:u0); cannot overflow u128 since
    // u1 <= dn - 1 and v + 2^64 = floor((2^128 - 1) / dn).
    let q = (v as u128) * (u1 as u128) + (((u1 as u128) << 64) | u0 as u128);
    let mut q1 = ((q >> 64) as u64).wrapping_add(1);
    let q0 = q as u64;
    let mut r = u0.wrapping_sub(q1.wrapping_mul(dn));
    // First fixup, branchless: a data-dependent branch here would mispredict.
    let over = (r > q0) as u64;
    q1 = q1.wrapping_sub(over);
    r = r.wrapping_add(dn & over.wrapping_neg());
    // Second fixup is almost never taken; keep it a predictable branch.
    if r >= dn {
        q1 += 1;
        r -= dn;
    }
    (q1, r)
}

/// One step of long division by 10^DIGITS: divides `(rem << 64) | limb` for
/// `rem < 10^DIGITS`, normalizing into the reciprocal division above. All
/// per-scale constants fold at monomorphization time.
#[inline]
const fn div_2by1_pow10<const DIGITS: u8>(rem: u64, limb: u64) -> (u64, u64) {
    let (dn, s, v) = const { pow10_norm(DIGITS as u32) };
    // (rem:limb) << s; for s == 0 (DIGITS = 19) the shifts fold away.
    let u1 = if s == 0 {
        rem
    } else {
        (rem << s) | (limb >> (64 - s))
    };
    let (q, r) = div_2by1_recip(u1, limb << s, dn, v);
    (q, r >> s)
}

/// In-place long division of `u` by the compile-time constant 10^DIGITS;
/// `u` becomes the quotient. Returns the remainder.
///
/// Every step is a reciprocal multiplication with compile-time constants —
/// no division instructions and no `__udivti3` on any architecture. (LLVM
/// does NOT strength-reduce 128-bit division by a constant, so writing
/// `u128 / const` would lower to a libcall; `scripts/check_asm.sh` guards
/// against regressing this.) Leading limbs smaller than the divisor are
/// skipped entirely, which covers typical financial magnitudes.
#[inline]
pub(crate) const fn div_words_by_pow10<const DIGITS: u8>(u: &mut [u64]) -> u64 {
    if DIGITS == 0 {
        return 0; // dividing by 1
    }
    let d = const { upow10(DIGITS as u32) };
    let mut rem: u64 = 0;
    let mut i = u.len();
    while i > 0 {
        i -= 1;
        // Leading limbs smaller than the divisor produce a zero quotient limb
        // and pass straight into the remainder.
        if rem == 0 && u[i] < d {
            rem = u[i];
            u[i] = 0;
            continue;
        }
        let (q, r) = div_2by1_pow10::<DIGITS>(rem, u[i]);
        u[i] = q;
        rem = r;
    }
    rem
}

/// Divides a full-range `u128` by the compile-time constant 10^DIGITS,
/// returning `(quotient, remainder)`, using the same reciprocal
/// multiplication as [`div_words_by_pow10`] — division-instruction-free.
#[inline]
pub(crate) const fn div_rem_u128_pow10<const DIGITS: u8>(n: u128) -> (u128, u64) {
    if DIGITS == 0 {
        return (n, 0);
    }
    let d = const { upow10(DIGITS as u32) };
    let hi = (n >> 64) as u64;
    let lo = n as u64;
    let (q1, r1) = if hi < d {
        (0, hi) // covers the common case of values below 2^64
    } else {
        div_2by1_pow10::<DIGITS>(0, hi)
    };
    let (q0, r) = div_2by1_pow10::<DIGITS>(r1, lo);
    (((q1 as u128) << 64) | q0 as u128, r)
}

/// `acc = acc * mul + add`, in place. Returns `true` if a carry overflowed
/// out of the limb array.
#[inline]
pub(crate) const fn mul_add_word(acc: &mut [u64], mul: u64, add: u64) -> bool {
    let mut carry: u128 = add as u128;
    let mut i = 0;
    while i < acc.len() {
        let v = (acc[i] as u128) * (mul as u128) + carry;
        acc[i] = v as u64;
        carry = v >> 64;
        i += 1;
    }
    carry != 0
}

/// `c = a * b` (schoolbook). Requires `c.len() >= a.len() + b.len()`; `c` is
/// fully overwritten (excess top limbs are zeroed).
#[inline]
pub(crate) const fn mul_words(c: &mut [u64], a: &[u64], b: &[u64]) {
    debug_assert!(c.len() >= a.len() + b.len());
    let mut i = 0;
    while i < c.len() {
        c[i] = 0;
        i += 1;
    }
    let mut i = 0;
    while i < a.len() {
        // c[i..i+b.len()] += a[i] * b, tracking the carry into c[i+b.len()].
        let mut carry: u128 = 0;
        let mut j = 0;
        while j < b.len() {
            let v = (a[i] as u128) * (b[j] as u128) + (c[i + j] as u128) + carry;
            c[i + j] = v as u64;
            carry = v >> 64;
            j += 1;
        }
        c[i + b.len()] = carry as u64;
        i += 1;
    }
}

/// Number of significant limbs (at least 1, even for zero).
#[inline]
pub(crate) fn sig_limbs(l: &[u64]) -> usize {
    let mut n = l.len();
    while n > 1 && l[n - 1] == 0 {
        n -= 1;
    }
    n
}

/// True if every limb is zero.
#[inline]
pub(crate) const fn is_zero(l: &[u64]) -> bool {
    let mut i = 0;
    while i < l.len() {
        if l[i] != 0 {
            return false;
        }
        i += 1;
    }
    true
}

/// Lexicographic comparison of equal-length limb arrays as unsigned integers.
#[inline]
pub(crate) fn cmp_words(a: &[u64], b: &[u64]) -> Ordering {
    debug_assert_eq!(a.len(), b.len());
    for i in (0..a.len()).rev() {
        match a[i].cmp(&b[i]) {
            Ordering::Equal => {}
            o => return o,
        }
    }
    Ordering::Equal
}

/// Shifts left by one bit in place; returns the bit shifted out of the top.
#[inline]
pub(crate) fn shl1(l: &mut [u64]) -> u64 {
    let mut carry = 0u64;
    for x in l.iter_mut() {
        let next = *x >> 63;
        *x = (*x << 1) | carry;
        carry = next;
    }
    carry
}

/// Maximum dividend size (in limbs) supported by [`div_knuth`].
pub(crate) const KNUTH_MAX_M: usize = 8;

/// Knuth's algorithm D: divides `u` (m limbs) by `v` (n limbs, `2 <= n <= m`,
/// top limb of `v` non-zero). Writes the quotient into `q` (needs `m - n + 1`
/// limbs; excess limbs are zeroed) and the remainder into `r` (needs `n`
/// limbs; excess limbs are zeroed).
///
/// Direct port of the classic 32-bit `bn_div_ex` construction, widened to
/// 64-bit limbs with `u128`/`i128` intermediates.
#[inline]
pub(crate) fn div_knuth(q: &mut [u64], r: &mut [u64], u: &[u64], v: &[u64]) {
    let m = u.len();
    let n = v.len();
    debug_assert!((2..=m).contains(&n), "use div_words_by_word for n == 1");
    debug_assert!(v[n - 1] != 0);
    debug_assert!(m <= KNUTH_MAX_M);
    debug_assert!(q.len() > m - n && r.len() >= n);

    let mut un = [0u64; KNUTH_MAX_M + 1]; // normalized u, gains one limb
    let mut vn = [0u64; KNUTH_MAX_M]; // normalized v

    // Normalize so the divisor's top bit is set (required by div_2by1's
    // estimate quality guarantees).
    let s = v[n - 1].leading_zeros();
    if s == 0 {
        vn[..n].copy_from_slice(v);
        un[..m].copy_from_slice(u);
        un[m] = 0;
    } else {
        for i in (1..n).rev() {
            vn[i] = (v[i] << s) | (v[i - 1] >> (64 - s));
        }
        vn[0] = v[0] << s;
        un[m] = u[m - 1] >> (64 - s);
        for i in (1..m).rev() {
            un[i] = (u[i] << s) | (u[i - 1] >> (64 - s));
        }
        un[0] = u[0] << s;
    }

    for x in q.iter_mut() {
        *x = 0;
    }

    const B: u128 = 1u128 << 64;
    let vtop = vn[n - 1];
    let vnext = vn[n - 2];

    // Main loop, reducing un digit by digit from the top.
    let mut j = m - n;
    loop {
        debug_assert!(un[j + n] <= vtop);
        // Estimate the quotient digit from the top two dividend limbs.
        let (mut qhat, mut rhat): (u128, u128) = if un[j + n] == vtop {
            // The generic path needs un[j+n] < vtop; here the division splits
            // exactly: (vtop*B + lo) / vtop == B + lo/vtop, rem lo%vtop.
            (
                B + (un[j + n - 1] / vtop) as u128,
                (un[j + n - 1] % vtop) as u128,
            )
        } else {
            let (q0, r0) = div_2by1(un[j + n], un[j + n - 1], vtop);
            (q0 as u128, r0 as u128)
        };
        // Fine tune: while the estimate is too large by the next-limb test.
        // The `qhat >= B` arm must be checked first so the multiplication
        // below never overflows u128.
        loop {
            if qhat >= B || qhat * (vnext as u128) > ((rhat << 64) | un[j + n - 2] as u128) {
                qhat -= 1;
                rhat += vtop as u128;
                if rhat < B {
                    continue;
                }
            }
            break;
        }

        // Multiply and subtract: un[j..j+n+1] -= qhat * vn.
        let mut qd = qhat as u64;
        let mut borrow: i128 = 0;
        for i in 0..n {
            let p = (qd as u128) * (vn[i] as u128);
            let t = (un[i + j] as i128) - borrow - ((p as u64) as i128);
            un[i + j] = t as u64;
            borrow = ((p >> 64) as i128) - (t >> 64);
        }
        let t = (un[j + n] as i128) - borrow;
        un[j + n] = t as u64;

        // If we borrowed past the top, the estimate was one too large:
        // add the divisor back and adjust.
        if t < 0 {
            qd -= 1;
            let mut carry: u128 = 0;
            for i in 0..n {
                let sum = un[i + j] as u128 + vn[i] as u128 + carry;
                un[i + j] = sum as u64;
                carry = sum >> 64;
            }
            un[j + n] = un[j + n].wrapping_add(carry as u64);
        }
        q[j] = qd;

        if j == 0 {
            break;
        }
        j -= 1;
    }

    // Denormalize the remainder (shift right by s bits).
    if s == 0 {
        r[..n].copy_from_slice(&un[..n]);
    } else {
        for i in 0..n - 1 {
            r[i] = (un[i] >> s) | (un[i + 1] << (64 - s));
        }
        r[n - 1] = un[n - 1] >> s;
    }
    for x in r.iter_mut().skip(n) {
        *x = 0;
    }
}

/// Multiplies two sign-magnitude decimals and re-scales by 10^DIGITS with
/// the given rounding mode. This is THE multiplication core shared by every
/// backing width: `W` is the magnitude size in limbs (1 for `Decimal`, 2 for
/// `Decimal128`, 4 for `Decimal256`) and `W2` must equal `2 * W` (passed
/// separately because stable Rust cannot compute const-generic expressions).
///
/// Returns `None` on overflow of the symmetric range: thanks to `MIN ==
/// -MAX` on every backing, "fits" is uniformly "top bit of the top limb is
/// clear", regardless of sign.
#[inline]
pub(crate) const fn dec_mul<const DIGITS: u8, const W: usize, const W2: usize>(
    a_neg: bool,
    a_mag: &[u64; W],
    b_neg: bool,
    b_mag: &[u64; W],
    mode: Rounding,
) -> Option<(bool, [u64; W])> {
    debug_assert!(W2 == 2 * W);
    // Single-limb operands take the two-limb core: this replaces the
    // WxW-limb schoolbook product with a 2x2 one for typical magnitudes.
    // Only 1-limb operands qualify because there the narrow core provably
    // never overflows (product < 2^128, re-scaled result < 2^115 for
    // DIGITS >= 1), so no work is ever thrown away; the rare DIGITS = 0
    // overflow falls through and recomputes wide.
    if W > 2 && is_zero(a_mag.split_at(1).1) && is_zero(b_mag.split_at(1).1) {
        let a2 = [a_mag[0], 0];
        let b2 = [b_mag[0], 0];
        if let Some((n2, m2)) = dec_mul::<DIGITS, 2, 4>(a_neg, &a2, b_neg, &b2, mode) {
            let mut mag = [0u64; W];
            mag[0] = m2[0];
            mag[1] = m2[1];
            return Some((n2, mag));
        }
    }
    let neg = a_neg != b_neg;
    let mut prod = [0u64; W2];
    mul_words(&mut prod, a_mag, b_mag);
    // HalfUp (the operators' default) folds its rounding into the division:
    // adding half the divisor before truncating rounds half away from zero
    // with no data-dependent branch. `mode` is statically known at almost
    // every call site, so this `if` disappears after inlining.
    let half_up = matches!(mode, Rounding::HalfUp);
    if half_up {
        // The product is at most (2^(64W-1) - 1)^2: no carry out of W2 limbs.
        mul_add_word(&mut prod, 1, const { upow10(DIGITS as u32) / 2 });
    }
    let rem = div_words_by_pow10::<DIGITS>(&mut prod);
    if !is_zero(prod.split_at(W).1) {
        return None;
    }
    let mut mag = [0u64; W];
    let mut i = 0;
    while i < W {
        mag[i] = prod[i];
        i += 1;
    }
    if !half_up
        && round_up_by_cmp(
            cmp_twice_rem_u64(rem, const { upow10(DIGITS as u32) }),
            rem == 0,
            mag[0] & 1 != 0,
            neg,
            mode,
        )
        && mul_add_word(&mut mag, 1, 1)
    {
        return None;
    }
    if mag[W - 1] >> 63 != 0 {
        return None;
    }
    Some((neg, mag))
}

/// Divides two sign-magnitude decimals (scaling the dividend by 10^DIGITS
/// first) with the given rounding mode. The division core shared by every
/// backing width; `WP1` must equal `W + 1` (the scaled dividend gains one
/// limb). Single-limb divisors take word division (through [`div_2by1`], so
/// the `asm` feature applies); wider divisors take Knuth's algorithm D over
/// significant limbs only.
///
/// Returns `None` if `b` is zero or on overflow of the symmetric range.
#[inline]
pub(crate) fn dec_div<const DIGITS: u8, const W: usize, const WP1: usize>(
    a_neg: bool,
    a_mag: &[u64; W],
    b_neg: bool,
    b_mag: &[u64; W],
    mode: Rounding,
) -> Option<(bool, [u64; W])> {
    debug_assert!(WP1 == W + 1);
    if is_zero(b_mag) {
        return None;
    }
    // Same narrow-operand tiering as dec_mul; `b` is known non-zero here, so
    // `None` from the narrow core only means quotient overflow: fall through.
    if W > 2 && is_zero(a_mag.split_at(2).1) && is_zero(b_mag.split_at(2).1) {
        let a2 = [a_mag[0], a_mag[1]];
        let b2 = [b_mag[0], b_mag[1]];
        if let Some((n2, m2)) = dec_div::<DIGITS, 2, 3>(a_neg, &a2, b_neg, &b2, mode) {
            let mut mag = [0u64; W];
            mag[0] = m2[0];
            mag[1] = m2[1];
            return Some((n2, mag));
        }
    }
    let neg = a_neg != b_neg;
    let mut num = [0u64; WP1];
    mul_words(&mut num, a_mag, &[const { upow10(DIGITS as u32) }]);

    let n = sig_limbs(b_mag);
    let mut q = [0u64; WP1];
    let (rem_cmp, rem_zero) = if n == 1 {
        // Single-limb divisor: word-based long division. For W == 1 this is
        // the only possible path and the rest folds away.
        let d = b_mag[0];
        let rem = div_words_by_word(&mut num, d);
        q = num;
        (cmp_twice_rem_u64(rem, d), rem == 0)
    } else {
        let mut r = [0u64; W];
        let m = sig_limbs(&num);
        if m < n {
            // Fewer numerator limbs than divisor limbs: quotient is zero and
            // the whole numerator is the remainder.
            r[..m].copy_from_slice(&num[..m]);
        } else {
            div_knuth(&mut q[..m - n + 1], &mut r[..n], &num[..m], &b_mag[..n]);
        }
        // Order 2 * rem against the divisor.
        let mut r2 = [0u64; W];
        r2[..n].copy_from_slice(&r[..n]);
        let carry = shl1(&mut r2[..n]);
        let cmp = if carry != 0 {
            Ordering::Greater
        } else {
            cmp_words(&r2[..n], &b_mag[..n])
        };
        (cmp, is_zero(&r[..n]))
    };
    if q[W] != 0 {
        return None;
    }
    let mut mag = [0u64; W];
    mag.copy_from_slice(&q[..W]);
    if round_up_by_cmp(rem_cmp, rem_zero, mag[0] & 1 != 0, neg, mode)
        && mul_add_word(&mut mag, 1, 1)
    {
        return None;
    }
    if mag[W - 1] >> 63 != 0 {
        return None;
    }
    Some((neg, mag))
}

/// Decides whether a truncated magnitude should be rounded up (away from the
/// integer's current value on the magnitude scale), given the ordering of
/// `2 * remainder` against the divisor.
///
/// The `Down`/`Up` modes are directional (floor/ceil), matching
/// `mul_rounded`/`div_rounded` on [`Decimal`](crate::Decimal): flooring a
/// negative result means rounding its magnitude up.
#[inline]
pub(crate) const fn round_up_by_cmp(
    twice_rem_vs_div: Ordering,
    rem_is_zero: bool,
    quo_is_odd: bool,
    is_neg: bool,
    mode: Rounding,
) -> bool {
    if rem_is_zero {
        return false;
    }
    match mode {
        Rounding::HalfEven => {
            matches!(twice_rem_vs_div, Ordering::Greater)
                || (matches!(twice_rem_vs_div, Ordering::Equal) && quo_is_odd)
        }
        Rounding::HalfUp => !matches!(twice_rem_vs_div, Ordering::Less),
        Rounding::HalfDown => matches!(twice_rem_vs_div, Ordering::Greater),
        Rounding::Down => is_neg,
        Rounding::Up => !is_neg,
    }
}

/// Orders `2 * rem` against `d` without overflow, given `rem < d`.
#[inline]
pub(crate) const fn cmp_twice_rem_u64(rem: u64, d: u64) -> Ordering {
    debug_assert!(rem < d);
    let other = d - rem;
    if rem < other {
        Ordering::Less
    } else if rem > other {
        Ordering::Greater
    } else {
        Ordering::Equal
    }
}

/// Converts a decimal string into a sign and a 4-limb magnitude scaled by
/// 10^`scale`, rounding excess fractional digits with `mode`.
///
/// The limb-based counterpart of
/// [`parse_decimal_i64_rounded`](crate::parse_decimal_i64_rounded), with the
/// same grammar, rounding semantics (first dropped digit + sticky flag +
/// parity for `HalfEven`) and error kinds. Range checks against a concrete
/// backing type are the caller's job; `Overflow` is returned only when the
/// scaled magnitude exceeds 256 bits.
pub(crate) const fn parse_decimal_mag_rounded<const W: usize>(
    src: &str,
    scale: u8,
    mode: Rounding,
) -> Result<(bool, [u64; W]), AmountErrorKind> {
    let src: &[u8] = src.as_bytes();
    let scale = scale as i64;
    debug_assert!(scale <= 19);
    // `acc` holds all kept digits (integer digits plus at most `scale`
    // fractional digits) as one magnitude; trailing zeros are appended at the
    // end, so no separate integer part is needed.
    let mut acc = [0u64; W];
    // Kept digits are buffered 19 at a time in a single word, so the wide
    // accumulator is touched once per 19 digits instead of once per digit.
    let mut chunk: u64 = 0;
    let mut chunk_len: u32 = 0;
    let mut point: i64 = 0; // flag and counter for decimal point digits
    let mut digit = false; // true if there was any digit
    let mut round_digit: u64 = 0; // first fractional digit dropped beyond `scale`
    let mut sticky = false; // any non-zero digit beyond `round_digit`

    if src.is_empty() {
        return Err(AmountErrorKind::Empty);
    }

    let (sign, digits): (bool, &[u8]) = match src[0] {
        b'+' => (false, src.split_at(1).1),
        b'-' => (true, src.split_at(1).1),
        _ => (false, src),
    };

    let mut di = 0;
    while di < digits.len() {
        let s = digits[di];
        di += 1;
        match s {
            b'.' if point > 0 => {
                // should be just one decimal point
                return Err(AmountErrorKind::InvalidDigit);
            }
            b'.' => point = 1,
            c @ b'0'..=b'9' => {
                digit = true;
                let x = (c - b'0') as u64;
                if point <= scale {
                    chunk = chunk * 10 + x;
                    chunk_len += 1;
                    if chunk_len == 19 {
                        if mul_add_word(&mut acc, upow10(19), chunk) {
                            return Err(AmountErrorKind::Overflow);
                        }
                        chunk = 0;
                        chunk_len = 0;
                    }
                } else if point == scale + 1 {
                    // first digit beyond the resolution of the type
                    round_digit = x;
                } else if x != 0 {
                    // any further non-zero digit sets the sticky flag
                    sticky = true;
                }
                if point > 0 {
                    point += 1; // count digits after the decimal point
                }
            }
            _ => return Err(AmountErrorKind::InvalidDigit),
        }
    }
    if !digit {
        return Err(AmountErrorKind::Empty);
    }
    // Flush the partial chunk before scaling and rounding (the HalfEven
    // parity below inspects the fully-accumulated magnitude).
    if chunk_len > 0 && mul_add_word(&mut acc, upow10(chunk_len), chunk) {
        return Err(AmountErrorKind::Overflow);
    }

    if point == 0 {
        point = 1; // no decimal point is the same as a trailing one
    }
    // Append zeros so the magnitude carries exactly `scale` fractional digits
    // (when the input had more than `scale` of them, nothing is missing).
    if point <= scale && mul_add_word(&mut acc, upow10((scale + 1 - point) as u32), 0) {
        return Err(AmountErrorKind::Overflow);
    }

    let round_up = match mode {
        Rounding::HalfUp => round_digit >= 5,
        Rounding::HalfDown => round_digit > 5 || (round_digit == 5 && sticky),
        Rounding::HalfEven => round_digit > 5 || (round_digit == 5 && (sticky || acc[0] & 1 != 0)),
        Rounding::Down => false,
        Rounding::Up => round_digit != 0 || sticky,
    };
    if round_up && mul_add_word(&mut acc, 1, 1) {
        return Err(AmountErrorKind::Overflow);
    }

    Ok((sign, acc))
}

/// ASCII digit pairs `"00"`..`"99"` (most significant digit first), so digit
/// extraction below can take two digits per strength-reduced division by 100,
/// halving the serial divide chain that dominates formatting latency.
const DIGIT_PAIRS: [u8; 200] = {
    let mut t = [0u8; 200];
    let mut i = 0;
    while i < 100 {
        t[2 * i] = b'0' + (i / 10) as u8;
        t[2 * i + 1] = b'0' + (i % 10) as u8;
        i += 1;
    }
    t
};

/// Formats a scaled decimal magnitude into `buf`, mirroring the semantics of
/// [`str_i64`](crate::str_i64): trailing fractional zeros are trimmed when no
/// precision is given, an explicit precision rounds (half away from zero) or
/// zero-pads, and the string is built right-to-left in `buf`.
pub(crate) fn str_mag<'a>(
    mag: &[u64],
    neg: bool,
    frac_digits: usize,
    precision: Option<usize>,
    sign: AmountSign,
    buf: &'a mut [u8],
) -> Option<&'a str> {
    debug_assert!(frac_digits <= 19 && mag.len() <= 4);

    // Fast path: a single-limb magnitude with default precision — every
    // i64-backed value and typical wide-type magnitudes. It reads `mag`
    // directly, so the common case skips the working-copy memcpy entirely,
    // and writes digits straight into `buf` right-to-left: no intermediate
    // digit buffer and no second pass. At most 21 digits, the point and a
    // sign are written (23 bytes), so with `buf.len() >= 24` the positions
    // never underflow. (The `== 1` guard also proves `mag` is non-empty.)
    if precision.is_none() && buf.len() >= 24 && sig_limbs(mag) == 1 {
        let mut pos = buf.len();
        let mut v = mag[0];
        let is_zero_val = v == 0;
        // Trim trailing fractional zeros, then emit the remaining fraction.
        let mut fd = frac_digits;
        while fd > 0 && v % 10 == 0 {
            v /= 10;
            fd -= 1;
        }
        if fd > 0 {
            while fd >= 2 {
                let p = (v % 100) as usize * 2;
                v /= 100;
                pos -= 2;
                buf[pos] = DIGIT_PAIRS[p];
                buf[pos + 1] = DIGIT_PAIRS[p + 1];
                fd -= 2;
            }
            if fd > 0 {
                pos -= 1;
                buf[pos] = (v % 10) as u8 + b'0';
                v /= 10;
            }
            pos -= 1;
            buf[pos] = b'.';
        }
        // Integer part, at least one digit (possibly zero).
        while v >= 100 {
            let p = (v % 100) as usize * 2;
            v /= 100;
            pos -= 2;
            buf[pos] = DIGIT_PAIRS[p];
            buf[pos + 1] = DIGIT_PAIRS[p + 1];
        }
        if v >= 10 {
            let p = v as usize * 2;
            pos -= 2;
            buf[pos] = DIGIT_PAIRS[p];
            buf[pos + 1] = DIGIT_PAIRS[p + 1];
        } else {
            pos -= 1;
            buf[pos] = v as u8 + b'0';
        }
        match (sign, neg && !is_zero_val) {
            (AmountSign::None, _) => {}
            (_, true) => {
                pos -= 1;
                buf[pos] = b'-';
            }
            (AmountSign::Always, false) if !is_zero_val => {
                pos -= 1;
                buf[pos] = b'+';
            }
            _ => {}
        }
        // All bytes written are ASCII digits, '.', '+' or '-'.
        return core::str::from_utf8(&buf[pos..]).ok();
    }

    // Remaining paths mutate the magnitude, so take a working copy. One spare
    // limb: rounding up at a precision boundary can carry.
    let mut work = [0u64; 5];
    work[..mag.len()].copy_from_slice(mag);

    if let Some(precision) = precision {
        // If the requested precision cannot fit, bail out immediately.
        if precision >= buf.len() {
            return None;
        }
        // Round to the requested precision first (half away from zero), then
        // scale back so digit extraction below stays uniform.
        if precision < frac_digits {
            let round_scale = upow10((frac_digits - precision) as u32);
            let dropped = div_words_by_word(&mut work, round_scale);
            if dropped.wrapping_shl(1) >= round_scale {
                mul_add_word(&mut work, 1, 1);
            }
            mul_add_word(&mut work, round_scale, 0);
        }
    }

    // Extract ASCII digits, least significant first. While the value spans
    // several limbs, full 19-digit chunks come off with one reciprocal
    // division by 10^19 per pass; the last limb's digits come straight off
    // that word. The value is < 10^78 (and each chunk emits real positional
    // digits), so `nd` stays well within the 96-digit buffer.
    let mut digits = [b'0'; 96];
    let mut nd = 0usize;
    while sig_limbs(&work) > 1 {
        let mut rem = div_words_by_pow10::<19>(&mut work);
        // Exactly 19 digits: nine pairs, then the leftover top digit
        // (rem < 10^19 / 100^9 = 10).
        for _ in 0..9 {
            let p = (rem % 100) as usize * 2;
            rem /= 100;
            digits[nd] = DIGIT_PAIRS[p + 1];
            digits[nd + 1] = DIGIT_PAIRS[p];
            nd += 2;
        }
        digits[nd] = rem as u8 + b'0';
        nd += 1;
    }
    let mut rem = work[0];
    while rem >= 100 {
        let p = (rem % 100) as usize * 2;
        rem /= 100;
        digits[nd] = DIGIT_PAIRS[p + 1];
        digits[nd + 1] = DIGIT_PAIRS[p];
        nd += 2;
    }
    if rem >= 10 {
        let p = rem as usize * 2;
        digits[nd] = DIGIT_PAIRS[p + 1];
        digits[nd + 1] = DIGIT_PAIRS[p];
        nd += 2;
    } else if rem != 0 {
        digits[nd] = rem as u8 + b'0';
        nd += 1;
    }
    // Guarantee at least one integer digit (possibly zero).
    if nd < frac_digits + 1 {
        nd = frac_digits + 1;
    }

    // Compose right-to-left: [pad zeros][fraction]['.'][integer][sign].
    let mut pos = buf.len();
    macro_rules! push {
        ($c:expr) => {{
            if pos == 0 {
                return None;
            }
            pos -= 1;
            buf[pos] = $c;
        }};
    }

    // Number of fractional digits to print.
    let print_frac = match precision {
        Some(p) => p,
        None => {
            // Trim trailing zeros of the fractional part.
            let mut skip = 0;
            while skip < frac_digits && digits[skip] == b'0' {
                skip += 1;
            }
            frac_digits - skip
        }
    };

    if print_frac > 0 {
        // Zero-pad on the right if the precision exceeds the scale.
        for _ in frac_digits..print_frac {
            push!(b'0');
        }
        let start = frac_digits - print_frac.min(frac_digits);
        for digit in digits.iter().take(frac_digits).skip(start) {
            push!(*digit);
        }
        push!(b'.');
    }

    for digit in digits.iter().take(nd).skip(frac_digits) {
        push!(*digit);
    }

    let is_zero_val = is_zero(mag);
    match (sign, neg && !is_zero_val) {
        (AmountSign::None, _) => {}
        (_, true) => push!(b'-'),
        (AmountSign::Always, false) if !is_zero_val => push!(b'+'),
        _ => {}
    }

    // All bytes written are ASCII digits, '.', '+' or '-'.
    core::str::from_utf8(&buf[pos..]).ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::AmountSign;

    /// 10^19, the largest power of ten that fits in a u64.
    const POW10_19: u64 = upow10(19);

    /// xorshift64* pseudo-random generator for deterministic tests.
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x >> 12;
            x ^= x << 25;
            x ^= x >> 27;
            self.0 = x;
            x.wrapping_mul(0x2545F4914F6CDD1D)
        }
    }

    #[test]
    fn test_div_2by1() {
        // Edge and random cases, checked against native u128 division.
        let mut rng = Rng(0x12345678DEADBEEF);
        let check = |hi: u64, lo: u64, d: u64| {
            let n = ((hi as u128) << 64) | lo as u128;
            let (q, r) = div_2by1(hi, lo, d);
            assert_eq!(q as u128, n / d as u128, "q for {n} / {d}");
            assert_eq!(r as u128, n % d as u128, "r for {n} / {d}");
        };
        check(0, 0, 1);
        check(0, u64::MAX, 1);
        check(0, u64::MAX, u64::MAX);
        check(u64::MAX - 1, u64::MAX, u64::MAX);
        check(1, 0, 2);
        check(1, 1, 2);
        check(0x7FFF_FFFF_FFFF_FFFF, u64::MAX, 1 << 63);
        check(POW10_19 - 1, u64::MAX, POW10_19);
        for _ in 0..20000 {
            let d = rng.next() | 1;
            let hi = rng.next() % d;
            let lo = rng.next();
            check(hi, lo, d);
            // small divisors too
            let ds = (rng.next() % 1000) + 1;
            check(rng.next() % ds, rng.next(), ds);
        }
    }

    #[test]
    fn test_div_words_by_pow10() {
        // Cross-check the constant-divisor path against the generic word
        // division for several scales, sizes and magnitude shapes.
        fn check<const DIGITS: u8>(rng: &mut Rng) {
            let d = upow10(DIGITS as u32);
            for shape in 0..4 {
                // 1, 2, 3 or 4 significant limbs
                let mut u = [0u64; 4];
                for x in u.iter_mut().take(shape + 1) {
                    *x = rng.next();
                }
                let mut a = u;
                let ra = div_words_by_word(&mut a, d);
                let mut b = u;
                let rb = div_words_by_pow10::<DIGITS>(&mut b);
                assert_eq!((a, ra), (b, rb), "DIGITS={DIGITS} u={u:?}");
                // Also with an 8-limb buffer as used by dec256_mul.
                let mut w = [0u64; 8];
                w[..4].copy_from_slice(&u);
                let mut wa = w;
                let ra = div_words_by_word(&mut wa, d);
                let mut wb = w;
                let rb = div_words_by_pow10::<DIGITS>(&mut wb);
                assert_eq!((wa, ra), (wb, rb), "DIGITS={DIGITS} w={w:?}");
            }
        }
        let mut rng = Rng(0x9E3779B97F4A7C15);
        for _ in 0..2000 {
            check::<0>(&mut rng);
            check::<1>(&mut rng);
            check::<4>(&mut rng);
            check::<8>(&mut rng);
            check::<19>(&mut rng);
        }
        // Zero value stays zero on the fast path.
        let mut z = [0u64; 4];
        assert_eq!(div_words_by_pow10::<4>(&mut z), 0);
        assert_eq!(z, [0u64; 4]);
    }

    #[test]
    fn test_div_words_by_word() {
        let mut u = [u64::MAX, u64::MAX, u64::MAX, u64::MAX];
        let rem = div_words_by_word(&mut u, 10);
        // (2^256 - 1) % 10 == 5
        assert_eq!(rem, 5);
        // Rebuild: q * 10 + 5 == 2^256 - 1.
        assert!(!mul_add_word(&mut u, 10, 5));
        assert_eq!(u, [u64::MAX; 4]);
    }

    #[test]
    fn test_mul_words() {
        let mut c = [0u64; 4];
        mul_words(&mut c, &[u64::MAX, u64::MAX], &[u64::MAX, u64::MAX]);
        // (2^128 - 1)^2 = 2^256 - 2^129 + 1
        assert_eq!(c, [1, 0, u64::MAX - 1, u64::MAX]);

        let mut rng = Rng(0xC0FFEE);
        for _ in 0..5000 {
            let a = ((rng.next() as u128) << 64) | rng.next() as u128;
            let b = rng.next() as u128;
            let mut c = [0u64; 3];
            mul_words(&mut c, &[a as u64, (a >> 64) as u64], &[b as u64]);
            // Verify low 128 bits against native u128 multiplication.
            let lo = a.wrapping_mul(b);
            assert_eq!(c[0], lo as u64);
            assert_eq!(c[1], (lo >> 64) as u64);
        }
    }

    #[test]
    fn test_div_knuth_reconstruct() {
        // Validate q*v + r == u and r < v over random shapes.
        let mut rng = Rng(0xFEEDFACE);
        for iter in 0..20000 {
            let n = 2 + (rng.next() as usize) % 3; // 2..=4 divisor limbs
            let m = n + (rng.next() as usize) % (5 - n) + 1; // n+1..=5
            let mut u = [0u64; 5];
            let mut v = [0u64; 4];
            for x in u.iter_mut().take(m) {
                *x = rng.next();
            }
            for x in v.iter_mut().take(n) {
                *x = rng.next();
            }
            if v[n - 1] == 0 {
                v[n - 1] = 1 + (rng.next() >> 32);
            }
            // Sometimes force interesting top-limb relationships.
            if iter % 7 == 0 {
                u[m - 1] = v[n - 1];
            }
            if iter % 11 == 0 {
                v[n - 1] = 1 << 63;
            }

            let mut q = [0u64; 4];
            let mut r = [0u64; 4];
            div_knuth(&mut q[..m - n + 1], &mut r[..n], &u[..m], &v[..n]);

            // r < v
            assert_eq!(cmp_words(&r[..n], &v[..n]), Ordering::Less);
            // q * v + r == u  (reconstruct in 9 limbs)
            let mut prod = [0u64; 9];
            mul_words(&mut prod[..m + 1], &q[..m - n + 1], &v[..n]);
            let mut carry: u128 = 0;
            for i in 0..9 {
                let add = if i < n { r[i] as u128 } else { 0 };
                let s = prod[i] as u128 + add + carry;
                prod[i] = s as u64;
                carry = s >> 64;
            }
            assert_eq!(&prod[..m], &u[..m], "reconstruction failed");
            assert!(is_zero(&prod[m..]));
        }
    }

    #[test]
    fn test_div_knuth_exact_and_edges() {
        // 2^192 / 2^128 == 2^64
        let u = [0u64, 0, 0, 1];
        let v = [0u64, 0, 1];
        let mut q = [0u64; 2];
        let mut r = [0u64; 3];
        div_knuth(&mut q, &mut r, &u, &v);
        assert_eq!(q, [0, 1]);
        assert!(is_zero(&r));

        // Divisor with top bit set (s == 0 path).
        let u = [5u64, 7, 9];
        let v = [1u64, 1 << 63];
        let mut q = [0u64; 2];
        let mut r = [0u64; 2];
        div_knuth(&mut q, &mut r, &u, &v);
        let mut prod = [0u64; 4];
        mul_words(&mut prod, &q, &v);
        let mut carry: u128 = 0;
        for i in 0..4 {
            let add = if i < 2 { r[i] as u128 } else { 0 };
            let s = prod[i] as u128 + add + carry;
            prod[i] = s as u64;
            carry = s >> 64;
        }
        assert_eq!(&prod[..3], &u);
        assert_eq!(prod[3], 0);
    }

    #[test]
    fn test_parse_mag() {
        use crate::Rounding::*;
        assert_eq!(
            parse_decimal_mag_rounded::<4>("1.0001", 4, HalfUp),
            Ok((false, [10001, 0, 0, 0]))
        );
        assert_eq!(
            parse_decimal_mag_rounded::<4>("-1.00005", 4, HalfUp),
            Ok((true, [10001, 0, 0, 0]))
        );
        assert_eq!(
            parse_decimal_mag_rounded::<4>("1.00005", 4, HalfEven),
            Ok((false, [10000, 0, 0, 0]))
        );
        assert_eq!(
            parse_decimal_mag_rounded::<4>("", 4, HalfUp),
            Err(AmountErrorKind::Empty)
        );
        assert_eq!(
            parse_decimal_mag_rounded::<4>("1.2.3", 4, HalfUp),
            Err(AmountErrorKind::InvalidDigit)
        );
        // A magnitude needing more than one limb:
        // 2^128 == 340282366920938463463374607431768211456
        let (neg, mag) =
            parse_decimal_mag_rounded::<4>("340282366920938463463374607431768211456", 0, HalfUp)
                .unwrap();
        assert!(!neg);
        assert_eq!(mag, [0, 0, 1, 0]);
        // Too many digits for 256 bits.
        let huge = "9".repeat(80);
        assert_eq!(
            parse_decimal_mag_rounded::<4>(&huge, 0, HalfUp),
            Err(AmountErrorKind::Overflow)
        );
    }

    #[test]
    fn test_str_mag() {
        extern crate std;
        let mut buf = [0u8; 128];
        let m = |v: u64| [v, 0, 0, 0];
        assert_eq!(
            str_mag(&m(10000), false, 4, None, AmountSign::Negative, &mut buf),
            Some("1")
        );
        assert_eq!(
            str_mag(&m(10001), false, 4, None, AmountSign::Negative, &mut buf),
            Some("1.0001")
        );
        assert_eq!(
            str_mag(&m(10001), true, 4, None, AmountSign::Negative, &mut buf),
            Some("-1.0001")
        );
        assert_eq!(
            str_mag(&m(10050), false, 4, Some(2), AmountSign::Negative, &mut buf),
            Some("1.01")
        );
        assert_eq!(
            str_mag(&m(10000), false, 4, Some(5), AmountSign::Negative, &mut buf),
            Some("1.00000")
        );
        assert_eq!(
            str_mag(&m(10000), false, 4, None, AmountSign::Always, &mut buf),
            Some("+1")
        );
        assert_eq!(
            str_mag(&m(0), false, 4, None, AmountSign::Always, &mut buf),
            Some("0")
        );
        // 2^128 with scale 4: 34028236692093846346337460743176821.1456
        assert_eq!(
            str_mag(
                &[0, 0, 1, 0],
                false,
                4,
                None,
                AmountSign::Negative,
                &mut buf
            ),
            Some("34028236692093846346337460743176821.1456")
        );
        // 20-digit single-limb magnitude: the direct path, no chunk division.
        assert_eq!(
            str_mag(&m(u64::MAX), false, 4, None, AmountSign::Negative, &mut buf),
            Some("1844674407370955.1615")
        );
        // 2^64, the smallest two-limb magnitude: one 19-digit chunk plus a
        // single leading digit.
        assert_eq!(
            str_mag(
                &[0, 1, 0, 0],
                false,
                0,
                None,
                AmountSign::Negative,
                &mut buf
            ),
            Some("18446744073709551616")
        );
        // 10^20 = [7766279631452241920, 5]: the low chunk is all zeros, so
        // the chunk path must still emit its 19 positional zero digits.
        assert_eq!(
            str_mag(
                &[7766279631452241920, 5, 0, 0],
                false,
                0,
                None,
                AmountSign::Negative,
                &mut buf
            ),
            Some("100000000000000000000")
        );
        // Round-trip with the parser.
        let s = "-1234567890123456789012345678901234567890123456789012345.67891";
        let (neg, mag) = parse_decimal_mag_rounded::<4>(s, 5, crate::Rounding::HalfUp).unwrap();
        assert_eq!(
            str_mag(&mag, neg, 5, None, AmountSign::Negative, &mut buf),
            Some(s)
        );
        // Buffers under the fast-path minimum fall back to the general path
        // with identical output, and still reject values that do not fit.
        let mut small = [0u8; 8];
        assert_eq!(
            str_mag(&m(10001), true, 4, None, AmountSign::Negative, &mut small),
            Some("-1.0001")
        );
        assert_eq!(
            str_mag(
                &m(u64::MAX),
                false,
                4,
                None,
                AmountSign::Negative,
                &mut small
            ),
            None
        );
    }
}
