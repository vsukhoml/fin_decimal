//! 128-bit backed decimal fixed-point type.
//!
//! [`Decimal128`] follows the same power-of-10 scaling design as
//! [`Decimal`](crate::Decimal), widened to an `i128` backing. Addition and
//! subtraction stay native integer operations; multiplication and division go
//! through a 256-bit intermediate built from 64-bit limbs and are re-scaled
//! with a single word division (the divisor 10^DIGITS always fits in 64
//! bits), using [`div_2by1`] — the native `x86_64` `div` instruction with the
//! `asm` feature, or a fast portable long division otherwise.

use crate::AmountErrorKind;
use crate::Rounding;
use crate::ipow10_i128;
use crate::limbs::{
    cmp_twice_rem_u64, dec_div, dec_mul, dec_mul_div, div_knuth, div_rem_u128_pow10,
    div_words_by_word, isqrt_words, mul_add_word, mul_words, parse_decimal_mag_rounded,
    round_up_by_cmp, upow10,
};
use core::cmp::Ordering;
use core::ops::*;

/// Decimal fixed-point number backed by a 128-bit signed integer, with
/// `DIGITS` fractional decimal digits (`DIGITS <= 19`).
///
/// Same decimal semantics as [`Decimal`](crate::Decimal), with roughly twice
/// the range: about ±1.7 * 10^(38 - DIGITS).
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Decimal128<const DIGITS: u8>(pub i128);

/// Amount128 is a 128-bit integer based decimal type with 4 decimal digits precision.
pub type Amount128 = Decimal128<4>;
/// Rate128 is a 128-bit integer based decimal type with 8 decimal digits precision.
pub type Rate128 = Decimal128<8>;

/// Splits an i128 into a sign and a two-limb magnitude.
#[inline]
const fn i128_sign_mag(v: i128) -> (bool, [u64; 2]) {
    let m = v.unsigned_abs();
    (v < 0, [m as u64, (m >> 64) as u64])
}

/// Rebuilds an i128 from a sign and a two-limb magnitude with the top bit
/// clear (guaranteed by the shared cores' symmetric-range check).
#[inline]
const fn i128_from_sign_mag(neg: bool, mag: [u64; 2]) -> i128 {
    let v = ((mag[0] as u128) | ((mag[1] as u128) << 64)) as i128;
    if neg { v.wrapping_neg() } else { v }
}

/// Truncated remainder `a % b` (`b != 0`) on sign-magnitude limbs. A native
/// 128-bit `%` is a `__modti3` libcall on every architecture — no platform
/// has a 128-by-128 division primitive — so this goes through the crate's
/// word division ([`div_2by1`](crate::limbs::div_2by1) steps) instead, like
/// [`Decimal256`](crate::Decimal256)'s remainder.
fn i128_rem(a: i128, b: i128) -> i128 {
    let neg = a < 0;
    let am = a.unsigned_abs();
    let bm = b.unsigned_abs();
    let r = if bm >> 64 == 0 {
        let mut q = [am as u64, (am >> 64) as u64];
        div_words_by_word(&mut q, bm as u64) as u128
    } else if am < bm {
        am // the divisor is larger, so the dividend is the remainder
    } else {
        let mut q = [0u64];
        let mut r = [0u64; 2];
        div_knuth(
            &mut q,
            &mut r,
            &[am as u64, (am >> 64) as u64],
            &[bm as u64, (bm >> 64) as u64],
        );
        ((r[1] as u128) << 64) | r[0] as u128
    };
    // r < |b| <= 2^127 (and in the pass-through branch |a| < |b|), so the
    // remainder always fits an i128 with either sign.
    if neg { -(r as i128) } else { r as i128 }
}

impl<const DIGITS: u8> Decimal128<DIGITS> {
    /// The decimal scale: the number of fractional digits.
    pub const SCALE: i32 = DIGITS as i32;

    /// The multiplier used to scale values up, as a single 64-bit word.
    /// Constrains `DIGITS <= 19` at compile time.
    pub(crate) const SCALE_U64: u64 = upow10(DIGITS as u32);

    /// The multiplier used to scale values up. Equal to 10^DIGITS.
    pub const SCALE_INT: i128 = Self::SCALE_U64 as i128;

    /// Half of the scaling factor, used for rounding.
    pub const SCALE_INT_HALF: i128 = Self::SCALE_INT / 2;

    /// The largest value that can be represented by this type.
    pub const MAX: Self = Decimal128::<DIGITS>(i128::MAX);
    /// The smallest value that can be represented by this type.
    pub const MIN: Self = Decimal128::<DIGITS>(-i128::MAX); // make MIN symmetric

    /// Constant equal to '1'
    pub const ONE: Self = Decimal128::<DIGITS>(Self::SCALE_INT);
    /// Constant equal to '-1'
    pub const MINUS_ONE: Self = Decimal128::<DIGITS>(-Self::SCALE_INT);
    /// Constant equal to '0'
    pub const ZERO: Self = Decimal128::<DIGITS>(0);

    /// The smallest integer value that can be represented by this type.
    pub const INT_MIN: i128 = (i128::MIN + 1) / Self::SCALE_INT;
    /// The largest integer value that can be represented by this type.
    pub const INT_MAX: i128 = i128::MAX / Self::SCALE_INT;

    /// The multiplier used to scale values up, as f64.
    pub const SCALE_F64: f64 = Self::SCALE_INT as f64;

    /// The smallest f64 value that can be represented.
    pub const F64_MIN: f64 = (i128::MIN + 1) as f64 / Self::SCALE_F64;
    /// The largest f64 value that can be represented.
    pub const F64_MAX: f64 = i128::MAX as f64 / Self::SCALE_F64;

    /// Constructs a new decimal integer with value 0.
    ///
    /// # Examples
    /// ```rust
    /// use fin_decimal::Amount128;
    /// let i = Amount128::new();
    /// assert_eq!(i, Amount128::ZERO);
    /// ```
    #[inline]
    pub fn new() -> Self {
        Decimal128::<DIGITS>(0)
    }

    /// Tries to convert a f32 to a Decimal128.
    #[inline]
    pub fn from_f32(val: f32) -> Result<Self, AmountErrorKind> {
        Self::from_f64(val as f64)
    }

    /// Tries to convert a f64 to a Decimal128.
    #[inline]
    pub fn from_f64(val: f64) -> Result<Self, AmountErrorKind> {
        if (Self::F64_MIN..=Self::F64_MAX).contains(&val) {
            Ok(Decimal128::<DIGITS>((val * Self::SCALE_F64) as i128))
        } else {
            Err(AmountErrorKind::Overflow)
        }
    }

    /// Tries to convert an i128 to a Decimal128.
    #[inline]
    pub const fn from_i128(val: i128) -> Result<Self, AmountErrorKind> {
        if (val <= Self::INT_MAX) && (val >= Self::INT_MIN) {
            Ok(Decimal128::<DIGITS>(val * Self::SCALE_INT))
        } else {
            Err(AmountErrorKind::Overflow)
        }
    }

    /// Tries to convert an i64 to a Decimal128.
    #[inline]
    pub const fn from_i64(val: i64) -> Result<Self, AmountErrorKind> {
        Self::from_i128(val as i128)
    }

    /// Converts the Decimal128 back into an f64.
    #[inline]
    pub fn to_f64(self) -> f64 {
        self.0 as f64 / Self::SCALE_F64
    }

    /// Converts the Decimal128 back into an i128 (truncating the fractional part).
    #[inline]
    pub const fn to_i128(self) -> i128 {
        self.0 / Self::SCALE_INT
    }

    /// Returns the raw backing value as `i128`.
    ///
    /// This is the unscaled mantissa: the stored integer such that the value
    /// equals `mantissa * 10^(-SCALE)`. For this type the backing already is
    /// `i128`, so the conversion is free.
    ///
    /// # Examples
    /// ```
    /// use fin_decimal::Amount128;
    /// let a = Amount128::from(3); // SCALE = 4
    /// assert_eq!(a.mantissa(), 30000);
    /// ```
    #[inline]
    pub const fn mantissa(self) -> i128 {
        self.0
    }

    /// Decomposes the value into `(mantissa, exponent)` such that the value
    /// equals `mantissa * 10^exponent`. The exponent is always `-SCALE`.
    ///
    /// # Examples
    /// ```
    /// use fin_decimal::Amount128;
    /// let a = Amount128::from(3); // SCALE = 4
    /// assert_eq!(a.to_decimal_parts(), (30000, -4));
    /// ```
    #[inline]
    pub const fn to_decimal_parts(self) -> (i128, i32) {
        (self.0, -Self::SCALE)
    }

    /// Builds a value from `(mantissa, exponent)`, where the represented
    /// number is `mantissa * 10^exponent`.
    ///
    /// Same contract as [`Decimal::from_decimal_parts`]: scaling up is exact
    /// (or [`Overflow`](AmountErrorKind::Overflow)); scaling down is
    /// exact-or-error, dropping surplus trailing zeros exactly and returning
    /// [`Inexact`](AmountErrorKind::Inexact) for non-zero dropped digits.
    ///
    /// [`Decimal::from_decimal_parts`]: crate::Decimal::from_decimal_parts
    ///
    /// # Examples
    /// ```
    /// use fin_decimal::{Amount128, AmountErrorKind};
    /// assert_eq!(Amount128::from_decimal_parts(123, -2), Ok(Amount128::from_bits(12300)));
    /// let a = Amount128::from(7);
    /// let (m, e) = a.to_decimal_parts();
    /// assert_eq!(Amount128::from_decimal_parts(m, e), Ok(a));
    /// assert_eq!(Amount128::from_decimal_parts(1, -5), Err(AmountErrorKind::Inexact));
    /// ```
    pub const fn from_decimal_parts(
        mantissa: i128,
        exponent: i32,
    ) -> Result<Self, AmountErrorKind> {
        if mantissa == 0 {
            return Ok(Decimal128::<DIGITS>(0));
        }
        let shift = exponent as i64 + Self::SCALE as i64;
        let scaled = if shift >= 0 {
            // Scale up: mantissa * 10^shift.
            let factor = match ipow10_i128(shift) {
                Some(f) => f,
                None => return Err(AmountErrorKind::Overflow),
            };
            match mantissa.checked_mul(factor) {
                Some(v) => v,
                None => return Err(AmountErrorKind::Overflow),
            }
        } else {
            // Scale down: mantissa / 10^(-shift), but only if exact.
            let divisor = match ipow10_i128(-shift) {
                Some(d) => d,
                None => return Err(AmountErrorKind::Inexact),
            };
            if mantissa % divisor != 0 {
                return Err(AmountErrorKind::Inexact);
            }
            mantissa / divisor
        };
        Ok(Decimal128::<DIGITS>(scaled))
    }

    /// Builds a value from `(mantissa, exponent)`, rounding to this type's
    /// scale with the given [`Rounding`] mode when the input carries more
    /// fractional digits than can be held exactly. Never returns
    /// [`Inexact`](AmountErrorKind::Inexact); only
    /// [`Overflow`](AmountErrorKind::Overflow) can fail.
    ///
    /// # Examples
    /// ```
    /// use fin_decimal::{Amount128, Rounding};
    /// assert_eq!(
    ///     Amount128::from_decimal_parts_rounded(123456, -5, Rounding::HalfUp),
    ///     Ok(Amount128::from_bits(12346)),
    /// );
    /// ```
    pub const fn from_decimal_parts_rounded(
        mantissa: i128,
        exponent: i32,
        mode: Rounding,
    ) -> Result<Self, AmountErrorKind> {
        if mantissa == 0 {
            return Ok(Decimal128::<DIGITS>(0));
        }
        let shift = exponent as i64 + Self::SCALE as i64;
        let scaled = if shift >= 0 {
            let factor = match ipow10_i128(shift) {
                Some(f) => f,
                None => return Err(AmountErrorKind::Overflow),
            };
            match mantissa.checked_mul(factor) {
                Some(v) => v,
                None => return Err(AmountErrorKind::Overflow),
            }
        } else {
            let is_neg = mantissa < 0;
            let mag = mantissa.unsigned_abs();
            let divisor = match ipow10_i128(-shift) {
                Some(d) => d as u128,
                // |value| < 0.5 ULP: rounds to 0, except `Up`.
                None => {
                    let one = match mode {
                        Rounding::Up => {
                            if is_neg {
                                -1
                            } else {
                                1
                            }
                        }
                        _ => 0,
                    };
                    return Ok(Decimal128::<DIGITS>(one));
                }
            };
            let quo = mag / divisor;
            let rem = mag % divisor;
            // The divisor is a power of ten >= 10, hence always even.
            let half = divisor / 2;
            let round_up = match mode {
                Rounding::HalfEven => rem > half || (rem == half && !quo.is_multiple_of(2)),
                Rounding::HalfUp => rem >= half,
                Rounding::HalfDown => rem > half,
                Rounding::Down => false,
                Rounding::Up => rem != 0,
            };
            let q = if round_up { quo + 1 } else { quo };
            if q > i128::MAX as u128 {
                return Err(AmountErrorKind::Overflow);
            }
            if is_neg { -(q as i128) } else { q as i128 }
        };
        Ok(Decimal128::<DIGITS>(scaled))
    }

    /// Parses a decimal string, rounding any fractional digits beyond this
    /// type's scale with the given [`Rounding`] mode.
    ///
    /// # Examples
    /// ```
    /// use fin_decimal::{Amount128, Rounding};
    /// assert_eq!(
    ///     Amount128::from_str_rounded("1.23455", Rounding::HalfEven),
    ///     Ok(Amount128::from_bits(12346)),
    /// );
    /// ```
    pub const fn from_str_rounded(src: &str, mode: Rounding) -> Result<Self, AmountErrorKind> {
        let (neg, mag) = match parse_decimal_mag_rounded::<2>(src, DIGITS, mode) {
            Ok(v) => v,
            Err(e) => return Err(e),
        };
        if mag[1] >> 63 != 0 {
            return Err(AmountErrorKind::Overflow);
        }
        Ok(Decimal128::<DIGITS>(i128_from_sign_mag(neg, mag)))
    }

    /// Parses a decimal string with [`Rounding::HalfUp`], panicking on invalid
    /// input — intended for compile-time constants, where the panic becomes a
    /// compile error.
    ///
    /// # Examples
    /// ```
    /// use fin_decimal::Amount128;
    /// const FEE: Amount128 = Amount128::from_str_const("0.0035");
    /// assert_eq!(FEE, Amount128::from_bits(35));
    /// ```
    pub const fn from_str_const(src: &str) -> Self {
        match Self::from_str_rounded(src, Rounding::HalfUp) {
            Ok(v) => v,
            Err(_) => panic!("invalid decimal literal"),
        }
    }

    /// Sign and 4-limb magnitude (zero-extended), for the shared
    /// formatting code in [`common`](crate::common).
    #[inline]
    pub(crate) const fn sign_mag4(self) -> (bool, [u64; 4]) {
        let (neg, m) = i128_sign_mag(self.0);
        (neg, [m[0], m[1], 0, 0])
    }

    /// Computes the absolute value of self.
    #[inline]
    pub const fn abs(self) -> Self {
        Decimal128::<DIGITS>(self.0.abs())
    }

    /// Checked addition. Computes `self + rhs`, returning `None` if overflow occurred.
    #[inline]
    pub const fn checked_add(self, rhs: Self) -> Option<Self> {
        match self.0.checked_add(rhs.0) {
            Some(v) => Some(Decimal128::<DIGITS>(v)),
            None => None,
        }
    }

    /// Checked subtraction. Computes `self - rhs`, returning `None` if overflow occurred.
    #[inline]
    pub const fn checked_sub(self, rhs: Self) -> Option<Self> {
        match self.0.checked_sub(rhs.0) {
            Some(v) => Some(Decimal128::<DIGITS>(v)),
            None => None,
        }
    }

    /// Checked multiplication. Computes `self * rhs`, returning `None` if overflow occurred.
    #[inline]
    pub const fn checked_mul(self, rhs: Self) -> Option<Self> {
        let (an, am) = i128_sign_mag(self.0);
        let (bn, bm) = i128_sign_mag(rhs.0);
        match dec_mul::<DIGITS, 2, 4>(an, &am, bn, &bm, Rounding::HalfUp) {
            Some((neg, mag)) => Some(Decimal128::<DIGITS>(i128_from_sign_mag(neg, mag))),
            None => None,
        }
    }

    /// Checked division. Computes `self / rhs`, returning `None` if `rhs == 0`
    /// or the division results in overflow.
    #[inline]
    pub fn checked_div(self, rhs: Self) -> Option<Self> {
        let (an, am) = i128_sign_mag(self.0);
        let (bn, bm) = i128_sign_mag(rhs.0);
        dec_div::<DIGITS, 2, 3>(an, &am, bn, &bm, Rounding::HalfUp)
            .map(|(neg, mag)| Decimal128::<DIGITS>(i128_from_sign_mag(neg, mag)))
    }

    /// Checked remainder. Computes `self % rhs`, returning `None` if
    /// `rhs == 0`. (The remainder itself cannot overflow: its magnitude
    /// never exceeds the divisor's.)
    #[inline]
    pub fn checked_rem(self, rhs: Self) -> Option<Self> {
        if rhs.0 == 0 {
            None
        } else {
            Some(Decimal128::<DIGITS>(i128_rem(self.0, rhs.0)))
        }
    }

    /// Takes the reciprocal (inverse) of a number, 1/x.
    ///
    /// # Panics
    /// Panics if `self` is zero (like core's `/`);
    /// [`checked_recip`](Self::checked_recip) is the non-panicking form.
    #[inline]
    pub fn recip(self) -> Self {
        match self.checked_recip() {
            Some(v) => v,
            // `checked_recip` can only fail for zero on this backing: the
            // reciprocal's mantissa is at most 10^(2 * DIGITS) <= 10^38,
            // inside the range.
            None => panic!("attempt to divide by zero"),
        }
    }

    /// Checked reciprocal: `None` if `self` is zero. (The result cannot
    /// overflow on this backing: its mantissa is at most
    /// `10^(2 * DIGITS) <= 10^38`, inside the `i128` range.)
    #[inline]
    pub fn checked_recip(self) -> Option<Self> {
        Self::ONE.checked_div(self)
    }

    /// Signed fractional remainder, `self.0 % 10^DIGITS`.
    ///
    /// Computed on the magnitude via [`div_rem_u128_pow10`] because a direct
    /// 128-bit `%` by a constant is not strength-reduced by the compiler (it
    /// lowers to a `__umodti3` call).
    #[inline]
    const fn frac_rem(self) -> i128 {
        let (_, r) = div_rem_u128_pow10::<DIGITS>(self.0.unsigned_abs());
        if self.0 < 0 { -(r as i128) } else { r as i128 }
    }

    /// Returns the integer part of a number.
    ///
    /// # Examples
    /// ```
    /// use fin_decimal::Amount128;
    /// assert_eq!(Amount128::from_f64(3.7).unwrap().trunc(), Amount128::from(3));
    /// assert_eq!(Amount128::from_f64(-3.7).unwrap().trunc(), Amount128::from(-3));
    /// ```
    #[inline]
    pub const fn trunc(self) -> Self {
        Decimal128::<DIGITS>(self.0 - self.frac_rem())
    }

    /// Returns the largest integer less than or equal to a number.
    ///
    /// # Examples
    /// ```
    /// use fin_decimal::Amount128;
    /// assert_eq!(Amount128::from_f64(-3.7).unwrap().floor(), Amount128::from(-4));
    /// ```
    ///
    /// # Panics
    /// Panics if rounding at the very edge of the range overflows (see
    /// [`round_to`](Self::round_to)).
    #[inline]
    pub const fn floor(self) -> Self {
        self.round_to(Rounding::Down)
    }

    /// Returns the smallest integer greater than or equal to a number.
    ///
    /// # Examples
    /// ```
    /// use fin_decimal::Amount128;
    /// assert_eq!(Amount128::from_f64(3.01).unwrap().ceil(), Amount128::from(4));
    /// assert_eq!(Amount128::from_f64(-3.7).unwrap().ceil(), Amount128::from(-3));
    /// ```
    ///
    /// # Panics
    /// Panics if rounding at the very edge of the range overflows (see
    /// [`round_to`](Self::round_to)).
    #[inline]
    pub const fn ceil(self) -> Self {
        self.round_to(Rounding::Up)
    }

    /// Returns the nearest integer to a number. Rounds half-way cases away
    /// from `0.0`.
    ///
    /// # Examples
    /// ```
    /// use fin_decimal::Amount128;
    /// assert_eq!(Amount128::from_f64(3.5).unwrap().round(), Amount128::from(4));
    /// assert_eq!(Amount128::from_f64(-3.5).unwrap().round(), Amount128::from(-4));
    /// ```
    ///
    /// # Panics
    /// Panics if rounding at the very edge of the range overflows (see
    /// [`round_to`](Self::round_to)).
    #[inline]
    pub const fn round(self) -> Self {
        self.round_to(Rounding::HalfUp)
    }

    /// Explicitly rounds the value to an integer using the specified rounding mode.
    ///
    /// # Panics
    /// Panics if rounding up at the very edge of the range overflows.
    pub const fn round_to(self, mode: Rounding) -> Self {
        match self.checked_round_to(mode) {
            Some(v) => v,
            None => panic!("attempt to round with overflow"),
        }
    }

    /// Checked form of [`round_to`](Self::round_to): `None` if rounding away
    /// from zero at the very edge of the range overflows.
    pub const fn checked_round_to(self, mode: Rounding) -> Option<Self> {
        // Magnitude-based, like the other rounding paths: a direct 128-bit
        // `%` by a constant would lower to a `__umodti3` call.
        let neg = self.0 < 0;
        let (q, rem) = div_rem_u128_pow10::<DIGITS>(self.0.unsigned_abs());
        if rem == 0 {
            return Some(self);
        }
        let up = round_up_by_cmp(
            cmp_twice_rem_u64(rem, const { upow10(DIGITS as u32) }),
            false,
            q & 1 != 0,
            neg,
            mode,
        );
        // Rebuild (q + up) * 10^DIGITS; q * 10^DIGITS <= |self.0|, so the
        // u128 arithmetic cannot overflow.
        let scaled = (q + up as u128) * const { upow10(DIGITS as u32) } as u128;
        if scaled > i128::MAX as u128 {
            return None;
        }
        let v = scaled as i128;
        Some(Decimal128::<DIGITS>(if neg { v.wrapping_neg() } else { v }))
    }

    /// Multiply by another decimal, explicitly applying the given rounding mode.
    ///
    /// The right-hand side may have a different scale: the exact product is
    /// rounded once to `self`'s scale, so an amount times a high-precision
    /// rate needs no intermediate conversion.
    ///
    /// Usable in const contexts, so multiplication chains can be evaluated at
    /// compile time.
    ///
    /// # Panics
    /// Panics if the result overflows (like core integer arithmetic in debug
    /// builds); [`checked_mul_rounded`](Self::checked_mul_rounded) is the
    /// non-panicking form.
    pub const fn mul_rounded<const RHS_DIGITS: u8>(
        self,
        rhs: Decimal128<RHS_DIGITS>,
        mode: Rounding,
    ) -> Self {
        match self.checked_mul_rounded(rhs, mode) {
            Some(v) => v,
            None => panic!("attempt to multiply with overflow"),
        }
    }

    /// Checked form of [`mul_rounded`](Self::mul_rounded): `None` if the
    /// result overflows.
    #[inline]
    pub const fn checked_mul_rounded<const RHS_DIGITS: u8>(
        self,
        rhs: Decimal128<RHS_DIGITS>,
        mode: Rounding,
    ) -> Option<Self> {
        let (an, am) = i128_sign_mag(self.0);
        let (bn, bm) = i128_sign_mag(rhs.0);
        match dec_mul::<RHS_DIGITS, 2, 4>(an, &am, bn, &bm, mode) {
            Some((neg, mag)) => Some(Decimal128::<DIGITS>(i128_from_sign_mag(neg, mag))),
            None => None,
        }
    }

    /// Divide by another decimal, explicitly applying the given rounding mode.
    ///
    /// # Panics
    /// Panics if `rhs` is zero (like core's `/`) or the result overflows;
    /// [`checked_div_rounded`](Self::checked_div_rounded) is the
    /// non-panicking form.
    pub fn div_rounded(self, rhs: Self, mode: Rounding) -> Self {
        self.div_rounded_to::<DIGITS>(rhs, mode)
    }

    /// Checked form of [`div_rounded`](Self::div_rounded): `None` if `rhs`
    /// is zero or the result overflows.
    #[inline]
    pub fn checked_div_rounded(self, rhs: Self, mode: Rounding) -> Option<Self> {
        self.checked_div_rounded_to::<DIGITS>(rhs, mode)
    }

    /// Divide by another decimal of the same scale, producing the quotient at
    /// an explicitly chosen scale with the given rounding mode. The exact
    /// quotient is rounded once at `TO_DIGITS` digits, so a proportion of two
    /// amounts can be taken directly as a higher-precision rate.
    ///
    /// # Panics
    /// Panics if `rhs` is zero (like core's `/`) or the result overflows;
    /// [`checked_div_rounded_to`](Self::checked_div_rounded_to) is the
    /// non-panicking form.
    pub fn div_rounded_to<const TO_DIGITS: u8>(
        self,
        rhs: Self,
        mode: Rounding,
    ) -> Decimal128<TO_DIGITS> {
        if rhs.0 == 0 {
            panic!("attempt to divide by zero");
        }
        match self.checked_div_rounded_to::<TO_DIGITS>(rhs, mode) {
            Some(v) => v,
            None => panic!("attempt to divide with overflow"),
        }
    }

    /// Checked form of [`div_rounded_to`](Self::div_rounded_to): `None` if
    /// `rhs` is zero or the result overflows.
    #[inline]
    pub fn checked_div_rounded_to<const TO_DIGITS: u8>(
        self,
        rhs: Self,
        mode: Rounding,
    ) -> Option<Decimal128<TO_DIGITS>> {
        let (an, am) = i128_sign_mag(self.0);
        let (bn, bm) = i128_sign_mag(rhs.0);
        dec_div::<TO_DIGITS, 2, 3>(an, &am, bn, &bm, mode)
            .map(|(neg, mag)| Decimal128::<TO_DIGITS>(i128_from_sign_mag(neg, mag)))
    }

    /// Divide by an integer, explicitly applying the given rounding mode: the
    /// exact quotient is rounded once at the type's own scale. Equivalent to
    /// `div_rounded` by `Decimal128::from(n)` but skips re-scaling the
    /// dividend: the divisor is a single word, so this is plain word division
    /// (a native `i128 / i64` would lower to a `__udivti3` call).
    ///
    /// # Panics
    /// Panics if `n` is zero (like core's `/`);
    /// [`checked_div_int_rounded`](Self::checked_div_int_rounded) is the
    /// non-panicking form. The result itself cannot overflow: its magnitude
    /// never exceeds `self`'s (rounding up only happens for `|n| >= 2`).
    pub fn div_int_rounded(self, n: i64, mode: Rounding) -> Self {
        match self.checked_div_int_rounded(n, mode) {
            Some(v) => v,
            None => panic!("attempt to divide by zero"),
        }
    }

    /// Checked form of [`div_int_rounded`](Self::div_int_rounded): `None`
    /// only if `n` is zero (the result cannot overflow).
    #[inline]
    pub fn checked_div_int_rounded(self, n: i64, mode: Rounding) -> Option<Self> {
        if n == 0 {
            return None;
        }
        let (an, mut mag) = i128_sign_mag(self.0);
        let neg = an != (n < 0);
        let d = n.unsigned_abs();
        let r = div_words_by_word(&mut mag, d);
        if round_up_by_cmp(cmp_twice_rem_u64(r, d), r == 0, mag[0] & 1 != 0, neg, mode) {
            mul_add_word(&mut mag, 1, 1);
        }
        Some(Decimal128::<DIGITS>(i128_from_sign_mag(neg, mag)))
    }

    /// Square root with the given rounding mode: `None` for a negative
    /// value. (Unlike the i64 backing, overflow is impossible here: the
    /// root of even `MAX`'s scaled mantissa stays below `2^96`.)
    ///
    /// The exact root of `self` is rounded once at the type's own scale:
    /// the result is the correctly rounded decimal square root, computed as
    /// the integer square root of `mantissa * 10^DIGITS`. For the half
    /// modes a tie cannot occur (`(s + 1/2)^2` is never an integer), so
    /// `HalfUp`, `HalfDown` and `HalfEven` always agree here.
    ///
    /// Not usable in const contexts (unlike the i64 backing's): the wide
    /// core's Newton steps divide by a runtime divisor, which is where the
    /// `asm` feature applies.
    pub fn checked_sqrt_rounded(self, mode: Rounding) -> Option<Self> {
        if self.0 < 0 {
            return None;
        }
        let mag = self.0.unsigned_abs();
        let mut n = [0u64; 4];
        mul_words(
            &mut n[..3],
            &[mag as u64, (mag >> 64) as u64],
            &[const { upow10(DIGITS as u32) }],
        );
        let (root, rem_zero, rem_gt_root) = isqrt_words::<4, 2, 4>(&n);
        let up = match mode {
            Rounding::Down => false,
            Rounding::Up => !rem_zero,
            // Half modes, tie-free: round up iff n is past (s + 1/2)^2.
            _ => rem_gt_root,
        };
        let s = ((root[0] as u128) | ((root[1] as u128) << 64)) + up as u128;
        Some(Decimal128::<DIGITS>(s as i128))
    }

    /// Square root, explicitly applying the given rounding mode. See
    /// [`checked_sqrt_rounded`](Self::checked_sqrt_rounded).
    ///
    /// # Examples
    /// ```
    /// use fin_decimal::{Amount128, Rounding};
    /// let two = Amount128::from(2);
    /// assert_eq!(two.sqrt_rounded(Rounding::HalfUp), Amount128::from_str_const("1.4142"));
    /// assert_eq!(two.sqrt_rounded(Rounding::Up), Amount128::from_str_const("1.4143"));
    /// ```
    ///
    /// # Panics
    /// Panics if `self` is negative;
    /// [`checked_sqrt_rounded`](Self::checked_sqrt_rounded) is the
    /// non-panicking form.
    pub fn sqrt_rounded(self, mode: Rounding) -> Self {
        match self.checked_sqrt_rounded(mode) {
            Some(v) => v,
            // None only for a negative input: the root cannot overflow.
            None => panic!("argument of integer square root cannot be negative"),
        }
    }

    /// Square root, rounded half-up at the type's own scale (the operators'
    /// default rounding). See
    /// [`checked_sqrt_rounded`](Self::checked_sqrt_rounded).
    ///
    /// # Examples
    /// ```
    /// use fin_decimal::Amount128;
    /// assert_eq!(Amount128::from_str_const("2.25").sqrt(), Amount128::from_str_const("1.5"));
    /// ```
    ///
    /// # Panics
    /// Panics if `self` is negative;
    /// [`checked_sqrt`](Self::checked_sqrt) is the non-panicking form.
    #[inline]
    pub fn sqrt(self) -> Self {
        self.sqrt_rounded(Rounding::HalfUp)
    }

    /// Checked form of [`sqrt`](Self::sqrt) (half-up rounding), mirroring
    /// core's `checked_isqrt` on signed integers: `None` for a negative
    /// value.
    #[inline]
    pub fn checked_sqrt(self) -> Option<Self> {
        self.checked_sqrt_rounded(Rounding::HalfUp)
    }

    /// `self * b / c` on the exact 256-bit product with a single rounding at
    /// the end, or `None` if `c` is zero or the result overflows.
    ///
    /// The muldiv/pro-rata primitive: allocating an amount by the proportion
    /// `b / c` this way avoids the double rounding of first taking the ratio
    /// and then multiplying by it. `b` and `c` share a scale — any scale,
    /// since it cancels in `b / c` — and the result keeps `self`'s scale.
    pub fn checked_mul_div_rounded<const S: u8>(
        self,
        b: Decimal128<S>,
        c: Decimal128<S>,
        mode: Rounding,
    ) -> Option<Self> {
        let (an, am) = i128_sign_mag(self.0);
        let (bn, bm) = i128_sign_mag(b.0);
        let (cn, cm) = i128_sign_mag(c.0);
        dec_mul_div::<2, 4>(an, &am, bn, &bm, cn, &cm, mode)
            .map(|(neg, mag)| Decimal128::<DIGITS>(i128_from_sign_mag(neg, mag)))
    }

    /// `self * b / c` on the exact 256-bit product with a single rounding at
    /// the end. See
    /// [`checked_mul_div_rounded`](Self::checked_mul_div_rounded).
    ///
    /// # Examples
    /// ```
    /// use fin_decimal::{Amount128, Rounding};
    /// // Allocate a total pro rata by 1/3 without an intermediate rate.
    /// let total = Amount128::from_str_const("100.00");
    /// let share = total.mul_div_rounded(Amount128::from(1), Amount128::from(3), Rounding::HalfUp);
    /// assert_eq!(share, Amount128::from_str_const("33.3333"));
    /// ```
    ///
    /// # Panics
    /// Panics if `c` is zero (like core's `/`) or the result overflows;
    /// [`checked_mul_div_rounded`](Self::checked_mul_div_rounded) is the
    /// non-panicking form.
    pub fn mul_div_rounded<const S: u8>(
        self,
        b: Decimal128<S>,
        c: Decimal128<S>,
        mode: Rounding,
    ) -> Self {
        if c.0 == 0 {
            panic!("attempt to divide by zero");
        }
        // The 2W-limb product is exact, so overflow is always quotient-driven.
        match self.checked_mul_div_rounded(b, c, mode) {
            Some(v) => v,
            None => panic!("attempt to divide with overflow"),
        }
    }

    /// Returns the fractional part of a number.
    #[inline]
    pub const fn fract(self) -> Self {
        // Via the magnitude reciprocal path: a direct `%` by the scale
        // constant would lower to a `__umodti3` call.
        Decimal128::<DIGITS>(self.frac_rem())
    }

    /// Returns `true` if `self` is positive and `false` if the number is zero or negative.
    #[inline]
    pub const fn is_positive(self) -> bool {
        self.0 > 0
    }

    /// Returns `true` if `self` is negative and `false` if the number is zero or positive.
    #[inline]
    pub const fn is_negative(self) -> bool {
        self.0 < 0
    }

    /// Returns a number that represents the sign of `self`.
    pub const fn signum(self) -> Self {
        match (self.0 < 0, self.0 > 0) {
            (true, _) => Self::MINUS_ONE,
            (false, false) => Self::ZERO,
            (_, _) => Self::ONE,
        }
    }

    /// Raw transmutation to the backing scaled integer.
    #[inline]
    pub const fn to_bits(self) -> i128 {
        self.0
    }

    /// Raw transmutation from a backing scaled integer.
    ///
    /// # Examples
    /// ```
    /// use fin_decimal::Amount128;
    /// assert_eq!(Amount128::from_bits(25000), Amount128::from_f64(2.5).unwrap());
    /// ```
    #[inline]
    pub const fn from_bits(v: i128) -> Self {
        Decimal128::<DIGITS>(v)
    }

    /// Return the memory representation of this value as a byte array in
    /// big-endian (network) byte order.
    #[inline]
    pub const fn to_be_bytes(self) -> [u8; 16] {
        self.0.to_be_bytes()
    }

    /// Return the memory representation of this value as a byte array in
    /// little-endian byte order.
    #[inline]
    pub const fn to_le_bytes(self) -> [u8; 16] {
        self.0.to_le_bytes()
    }

    /// Create a value from its representation as a byte array in big endian.
    #[inline]
    pub const fn from_be_bytes(bytes: [u8; 16]) -> Self {
        Decimal128::<DIGITS>(i128::from_be_bytes(bytes))
    }

    /// Create a value from its representation as a byte array in little endian.
    #[inline]
    pub const fn from_le_bytes(bytes: [u8; 16]) -> Self {
        Decimal128::<DIGITS>(i128::from_le_bytes(bytes))
    }

    /// Raises a number to an integer power. Usable in const contexts.
    ///
    /// # Panics
    /// Panics if an intermediate product overflows;
    /// [`checked_powi`](Self::checked_powi) is the non-panicking form.
    #[inline]
    pub const fn powi(self, exp: u32) -> Self {
        match self.checked_powi(exp) {
            Some(v) => v,
            None => panic!("attempt to multiply with overflow"),
        }
    }

    /// Checked form of [`powi`](Self::powi): `None` if an intermediate
    /// product overflows. Each step rounds `HalfUp` at the type's scale,
    /// like `powi`.
    pub const fn checked_powi(self, mut exp: u32) -> Option<Self> {
        let mut base = self;
        let mut acc = Self::ONE;
        while exp > 1 {
            if (exp & 1) == 1 {
                acc = match base.checked_mul_rounded(acc, Rounding::HalfUp) {
                    Some(v) => v,
                    None => return None,
                };
            }
            exp /= 2;
            base = match base.checked_mul_rounded(base, Rounding::HalfUp) {
                Some(v) => v,
                None => return None,
            };
        }
        // The final bit of the exponent is handled separately: squaring the
        // base afterwards is unnecessary and could overflow needlessly.
        if exp == 1 {
            acc = match base.checked_mul_rounded(acc, Rounding::HalfUp) {
                Some(v) => v,
                None => return None,
            };
        }
        Some(acc)
    }

    /// Restrict a value to a certain interval.
    #[inline]
    pub fn clamp(self, min: Self, max: Self) -> Self {
        if self.0 < min.0 {
            min
        } else if self.0 > max.0 {
            max
        } else {
            self
        }
    }

    /// Returns the minimum of the two numbers.
    #[inline]
    pub fn min(self, other: Self) -> Self {
        if self <= other { self } else { other }
    }

    /// Returns the maximum of the two numbers.
    #[inline]
    pub fn max(self, other: Self) -> Self {
        if self >= other { self } else { other }
    }
}

impl<const DIGITS: u8> From<i32> for Decimal128<DIGITS> {
    #[inline]
    fn from(item: i32) -> Self {
        Decimal128::<DIGITS>(item as i128 * Self::SCALE_INT)
    }
}

impl<const DIGITS: u8> From<i64> for Decimal128<DIGITS> {
    #[inline]
    fn from(item: i64) -> Self {
        Decimal128::<DIGITS>(item as i128 * Self::SCALE_INT)
    }
}

impl<const DIGITS: u8> From<i128> for Decimal128<DIGITS> {
    /// Converts an i128, saturating to `MIN`/`MAX` when out of range.
    #[inline]
    fn from(item: i128) -> Self {
        match Self::from_i128(item) {
            Ok(v) => v,
            Err(_) if item < 0 => Self::MIN,
            Err(_) => Self::MAX,
        }
    }
}

impl<const DIGITS: u8> From<f64> for Decimal128<DIGITS> {
    /// Converts an f64, saturating to `MIN`/`MAX` when out of range.
    #[inline]
    fn from(item: f64) -> Self {
        if (item < Self::F64_MAX) && (item > Self::F64_MIN) {
            Decimal128::<DIGITS>((item * Self::SCALE_F64) as i128)
        } else if item < Self::F64_MIN {
            Self::MIN
        } else {
            Self::MAX
        }
    }
}

impl<const DIGITS: u8> From<f32> for Decimal128<DIGITS> {
    #[inline]
    fn from(item: f32) -> Self {
        Self::from(item as f64)
    }
}

impl<const DIGITS: u8> PartialOrd<i64> for Decimal128<DIGITS> {
    #[inline]
    fn partial_cmp(&self, other: &i64) -> Option<Ordering> {
        PartialOrd::partial_cmp(&self.0, &(*other as i128 * Self::SCALE_INT))
    }
}

impl<const DIGITS: u8> PartialEq<i64> for Decimal128<DIGITS> {
    #[inline]
    fn eq(&self, other: &i64) -> bool {
        self.0 == *other as i128 * Self::SCALE_INT
    }
}

impl<const DIGITS: u8> Neg for Decimal128<DIGITS> {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self::Output {
        Decimal128::<DIGITS>(-self.0)
    }
}

impl<const DIGITS: u8> Add for Decimal128<DIGITS> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        Decimal128::<DIGITS>(self.0 + rhs.0)
    }
}

impl<const DIGITS: u8> Add<i64> for Decimal128<DIGITS> {
    type Output = Self;
    #[inline]
    fn add(self, other: i64) -> Self {
        Decimal128::<DIGITS>(self.0 + other as i128 * Self::SCALE_INT)
    }
}

impl<const DIGITS: u8> Add<i32> for Decimal128<DIGITS> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: i32) -> Self {
        Decimal128::<DIGITS>(self.0 + (rhs as i128) * Self::SCALE_INT)
    }
}

impl<const DIGITS: u8> Sub for Decimal128<DIGITS> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Decimal128::<DIGITS>(self.0 - rhs.0)
    }
}

impl<const DIGITS: u8> Sub<i64> for Decimal128<DIGITS> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: i64) -> Self {
        Decimal128::<DIGITS>(self.0 - rhs as i128 * Self::SCALE_INT)
    }
}

impl<const DIGITS: u8> Sub<i32> for Decimal128<DIGITS> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: i32) -> Self {
        Decimal128::<DIGITS>(self.0 - (rhs as i128) * Self::SCALE_INT)
    }
}

impl<const DIGITS: u8> Mul for Decimal128<DIGITS> {
    type Output = Self;
    /// Multiplies with decimal `HalfUp` rounding (half away from zero).
    ///
    /// # Panics
    /// Panics if the result overflows.
    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        self.mul_rounded(rhs, Rounding::HalfUp)
    }
}

impl<const DIGITS: u8> Mul<i64> for Decimal128<DIGITS> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: i64) -> Self {
        Decimal128::<DIGITS>(self.0 * rhs as i128)
    }
}

impl<const DIGITS: u8> Mul<i32> for Decimal128<DIGITS> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: i32) -> Self {
        Decimal128::<DIGITS>(self.0 * (rhs as i128))
    }
}

impl<const DIGITS: u8> Div for Decimal128<DIGITS> {
    type Output = Self;
    /// Divides with decimal `HalfUp` rounding (half away from zero).
    ///
    /// # Panics
    /// Panics if `rhs` is zero or the result overflows.
    #[inline]
    fn div(self, rhs: Self) -> Self {
        self.div_rounded(rhs, Rounding::HalfUp)
    }
}

impl<const DIGITS: u8> Rem for Decimal128<DIGITS> {
    type Output = Self;
    /// # Panics
    /// Panics if `rhs` is zero.
    #[inline]
    fn rem(self, rhs: Self) -> Self {
        match self.checked_rem(rhs) {
            Some(v) => v,
            None => panic!("attempt to calculate the remainder with a divisor of zero"),
        }
    }
}

crate::common::impl_decimal_common!(Decimal128, "Decimal128");

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;
    use core::str::FromStr;
    use std::format;
    use std::string::ToString;

    #[test]
    fn test_basic_ops() {
        let a = Amount128::from(2);
        let b = Amount128::from_f64(0.5).unwrap();
        assert_eq!(a.0, 20000);
        assert_eq!(b.0, 5000);
        assert_eq!((a + b).0, 25000);
        assert_eq!((a - b).0, 15000);
        assert_eq!((a * b).0, 10000);
        assert_eq!((a / b).0, 40000);
        assert_eq!((-a).0, -20000);
        assert_eq!((a % b).0, 0);
    }

    #[test]
    fn test_mul_rounding_matches_decimal64() {
        // Mirror Decimal<4> test vectors.
        assert_eq!(
            Decimal128::<4>(10001) * Decimal128::<4>(10001),
            Decimal128::<4>(10002)
        );
        assert_eq!(
            Decimal128::<4>(11004) * Decimal128::<4>(10015),
            Decimal128::<4>(11021)
        );
        assert_eq!(
            Decimal128::<4>(11004) * Decimal128::<4>(-10015),
            Decimal128::<4>(-11021)
        );
    }

    #[test]
    fn test_div_rounding_matches_decimal64() {
        assert_eq!(
            Decimal128::<4>(10000) / Decimal128::<4>(110000),
            Decimal128::<4>(909)
        );
        assert_eq!(
            Decimal128::<4>(10000) / Decimal128::<4>(130000),
            Decimal128::<4>(769)
        );
        assert_eq!(
            Decimal128::<4>(10000) / Decimal128::<4>(-130000),
            Decimal128::<4>(-769)
        );
        assert_eq!(
            Decimal128::<4>(-10000) / Decimal128::<4>(130000),
            Decimal128::<4>(-769)
        );
        assert_eq!(
            Decimal128::<4>(-10000) / Decimal128::<4>(-130000),
            Decimal128::<4>(769)
        );
        assert_eq!(
            Decimal128::<4>(10000) / Decimal128::<4>(180000),
            Decimal128::<4>(556)
        );
        assert_eq!(
            Decimal128::<4>(-10000) / Decimal128::<4>(180000),
            Decimal128::<4>(-556)
        );
    }

    #[test]
    fn test_beyond_64_bit_range() {
        // Values that overflow Amount64 are exact here.
        let big = Amount128::from(10_000_000_000_000_000i64); // 10^16
        assert_eq!(big.0, 100_000_000_000_000_000_000i128);
        assert_eq!((big * big).0, 10i128.pow(36));
        assert_eq!(
            format!("{}", big * big),
            "100000000000000000000000000000000"
        );
        // Division with a two-limb divisor.
        let one = Amount128::from(1);
        let r = one / big;
        assert_eq!(r.0, 0); // 10^-16 rounds to 0 at 4 digits
        let r = big / Amount128::from(3);
        assert_eq!(r.0, 33_333_333_333_333_333_333i128);
    }

    #[test]
    fn test_checked_math() {
        let max = Amount128::MAX;
        assert_eq!(max.checked_add(Amount128::from(1)), None);
        assert_eq!(max.checked_mul(Amount128::from(2)), None);
        assert_eq!(Amount128::from(2).checked_div(Amount128::ZERO), None);
        assert_eq!(
            Amount128::from(6).checked_div(Amount128::from(2)),
            Some(Amount128::from(3))
        );
        assert_eq!(
            Amount128::from(2).checked_mul(Amount128::from(3)),
            Some(Amount128::from(6))
        );
    }

    #[test]
    fn test_rounding_modes() {
        let a = Amount128::from_f64(1.5).unwrap();
        assert_eq!(a.round_to(Rounding::HalfUp), Amount128::from(2));
        assert_eq!(a.round_to(Rounding::HalfDown), Amount128::from(1));
        assert_eq!(a.round_to(Rounding::HalfEven), Amount128::from(2));
        assert_eq!(a.round_to(Rounding::Down), Amount128::from(1));
        assert_eq!(a.round_to(Rounding::Up), Amount128::from(2));

        // mul_rounded ties: 0.5 * 1.0001 = 0.50005
        let h = Amount128::from_f64(0.5).unwrap();
        let x = Amount128::from_bits(10001);
        assert_eq!(h.mul_rounded(x, Rounding::HalfUp).0, 5001);
        assert_eq!(h.mul_rounded(x, Rounding::HalfDown).0, 5000);
        assert_eq!(h.mul_rounded(x, Rounding::HalfEven).0, 5000);
        assert_eq!(h.mul_rounded(x, Rounding::Down).0, 5000);
        assert_eq!(h.mul_rounded(x, Rounding::Up).0, 5001);
        // Negative: floor/ceil are directional.
        assert_eq!((-h).mul_rounded(x, Rounding::Down).0, -5001);
        assert_eq!((-h).mul_rounded(x, Rounding::Up).0, -5000);
        assert_eq!((-h).mul_rounded(x, Rounding::HalfUp).0, -5001);
    }

    #[test]
    fn test_div_rounded_modes() {
        let one = Amount128::from(1);
        let three = Amount128::from(3);
        assert_eq!(one.div_rounded(three, Rounding::Down).0, 3333);
        assert_eq!(one.div_rounded(three, Rounding::Up).0, 3334);
        assert_eq!((-one).div_rounded(three, Rounding::Down).0, -3334);
        assert_eq!((-one).div_rounded(three, Rounding::Up).0, -3333);
        assert_eq!(one.div_rounded(three, Rounding::HalfUp).0, 3333);
    }

    #[test]
    fn test_from_str_and_display() {
        assert_eq!(Amount128::from_str("1.0001").unwrap().0, 10001);
        assert_eq!(Amount128::from_str("-1.0001").unwrap().0, -10001);
        assert_eq!(Amount128::from_str("1.00005").unwrap().0, 10001);
        assert_eq!(
            Amount128::from_str_rounded("1.00005", Rounding::HalfEven)
                .unwrap()
                .0,
            10000
        );
        assert_eq!(Amount128::from_str(""), Err(AmountErrorKind::Empty));
        assert_eq!(
            Amount128::from_str("1.2.3"),
            Err(AmountErrorKind::InvalidDigit)
        );

        // Max value round-trips through Display/FromStr.
        let s = Amount128::MAX.to_string();
        assert_eq!(s, "17014118346046923173168730371588410.5727");
        assert_eq!(Amount128::from_str(&s).unwrap(), Amount128::MAX);
        let s = Amount128::MIN.to_string();
        assert_eq!(Amount128::from_str(&s).unwrap(), Amount128::MIN);
        // Just past the range.
        assert_eq!(
            Amount128::from_str("17014118346046923173168730371588410.5728"),
            Err(AmountErrorKind::Overflow)
        );

        assert_eq!(&format!("{}", Decimal128::<4>(10000)), "1");
        assert_eq!(&format!("{:+}", Decimal128::<4>(10000)), "+1");
        assert_eq!(&format!("{:4.2}", Decimal128::<4>(10000)), "1.00");
        assert_eq!(&format!("{:03}", Decimal128::<4>(10000)), "001");
        assert_eq!(&format!("{}", Decimal128::<4>(1)), "0.0001");
        assert_eq!(&format!("{}", Decimal128::<4>(-10001)), "-1.0001");
        assert_eq!(&format!("{}", Decimal128::<4>(0)), "0");
    }

    #[test]
    fn test_decimal_parts() {
        let a = Amount128::from(3);
        assert_eq!(a.mantissa(), 30000);
        assert_eq!(a.to_decimal_parts(), (30000, -4));
        for raw in [0i128, 1, -1, 12345, i128::MAX, -i128::MAX] {
            let d = Decimal128::<4>(raw);
            let (m, e) = d.to_decimal_parts();
            assert_eq!(Amount128::from_decimal_parts(m, e), Ok(d));
        }
        assert_eq!(
            Amount128::from_decimal_parts(1, -5),
            Err(AmountErrorKind::Inexact)
        );
        assert_eq!(
            Amount128::from_decimal_parts(12300, -5),
            Ok(Decimal128::<4>(1230))
        );
        assert_eq!(
            Amount128::from_decimal_parts_rounded(123456, -5, Rounding::HalfUp),
            Ok(Decimal128::<4>(12346))
        );
        assert_eq!(
            Amount128::from_decimal_parts(i128::MAX, 1),
            Err(AmountErrorKind::Overflow)
        );
    }

    #[test]
    fn test_misc() {
        assert_eq!(
            Amount128::from_f64(3.7).unwrap().trunc(),
            Amount128::from(3)
        );
        assert_eq!(
            Amount128::from_f64(-3.7).unwrap().floor(),
            Amount128::from(-4)
        );
        assert_eq!(Amount128::from_f64(3.2).unwrap().ceil(), Amount128::from(4));
        assert_eq!(
            Amount128::from_f64(-3.5).unwrap().round(),
            Amount128::from(-4)
        );
        assert_eq!(Amount128::from(10).signum(), Amount128::ONE);
        assert_eq!(Amount128::from(-10).signum(), Amount128::MINUS_ONE);
        assert_eq!(Amount128::from(0).signum(), Amount128::ZERO);
        assert_eq!(Amount128::from(2).recip().0, 5000);
        assert_eq!(Amount128::from(2).powi(10), Amount128::from(1024));
        assert_eq!(
            Amount128::from(5).clamp(Amount128::ZERO, Amount128::ONE),
            Amount128::ONE
        );
        assert!(Amount128::from(1) == 1i64);
        assert!(Amount128::from(2) > 1i64);
        let v: Amount128 = [1, 2, 3].iter().map(|&x| Amount128::from(x)).sum();
        assert_eq!(v, Amount128::from(6));
        let bytes = Amount128::from(1).to_le_bytes();
        assert_eq!(Amount128::from_le_bytes(bytes), Amount128::from(1));
        assert_eq!(Rate128::from_f64(1.12345678).unwrap().0, 112345678);
        assert_eq!(
            &format!("{}", Rate128::from_f64(1.12345678).unwrap()),
            "1.12345678"
        );
    }

    #[test]
    fn test_const_eval() {
        const A: Amount128 = Amount128::from_str_const("123456789012345.6789");
        const B: Amount128 = A.mul_rounded(A, Rounding::HalfUp);
        const C: Amount128 = A.round_to(Rounding::HalfEven);
        const D: Amount128 = A.trunc();
        const E: Amount128 = Amount128::from_str_const("1.01").powi(12);
        const F: Option<Amount128> = A.checked_mul(A);
        const G: Amount128 = A.fract();
        const H: Amount128 = Amount128::from_bits(-15000).abs();
        const I: Option<Amount128> = Amount128::ONE.checked_add(Amount128::ONE);
        const J: Option<Amount128> = Amount128::MIN.checked_sub(Amount128::ONE);
        const K: Amount128 = Amount128::from_str_const("-3.25").floor();
        const L: Amount128 = Amount128::from_str_const("3.25").ceil();
        const L2: Amount128 = Amount128::from_str_const("-3.25").ceil();
        const N2: Option<Amount128> = Amount128::MAX.checked_powi(2);
        const N3: Option<Amount128> = Amount128::MAX.checked_round_to(Rounding::Up);
        const M: Result<Amount128, AmountErrorKind> =
            Amount128::from_decimal_parts_rounded(12345, -5, Rounding::HalfEven);
        const N: Result<Amount128, AmountErrorKind> = Amount128::from_i128(7);

        assert_eq!(H, Amount128::from_bits(15000));
        assert_eq!(I, Some(Amount128::from(2)));
        assert_eq!(J, None);
        assert_eq!(K, Amount128::from(-4));
        assert_eq!(L, Amount128::from(4));
        assert_eq!(L2, Amount128::from(-3));
        assert_eq!(N2, None);
        assert_eq!(N3, None);
        assert_eq!(M, Ok(Amount128::from_bits(1234)));
        assert_eq!(N, Ok(Amount128::from(7)));

        let a = Amount128::from_str("123456789012345.6789").unwrap();
        assert_eq!(A, a);
        assert_eq!(B, a.mul_rounded(a, Rounding::HalfUp));
        assert_eq!(C, a.round_to(Rounding::HalfEven));
        assert_eq!(D, a.trunc());
        assert_eq!(E, Amount128::from_str("1.01").unwrap().powi(12));
        assert_eq!(F, a.checked_mul(a));
        assert_eq!(G, a.fract());
        assert_eq!(G.0, 6789);
    }

    /// The limb-based remainder must agree with the compiler's native i128
    /// `%` across one- and two-limb divisors and all sign combinations.
    #[test]
    fn test_rem_wide() {
        let mut state = 0x9E3779B97F4A7C15u64;
        let mut next = move || {
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            state.wrapping_mul(0x2545F4914F6CDD1D)
        };

        // A signed value whose magnitude spans a random number of bits.
        let mut rnd = move || {
            let sh = (next() as u32) % 128;
            let neg = next() & 1 != 0;
            let m = ((((next() as u128) << 64) | next() as u128) >> sh) as i128;
            if neg { m.wrapping_neg() } else { m }
        };
        for _ in 0..20000 {
            let a = rnd();
            let b = rnd();
            if b == 0 {
                continue;
            }
            assert_eq!(
                (Decimal128::<4>(a) % Decimal128::<4>(b)).0,
                a % b,
                "{a} % {b}"
            );
        }

        // Extremes: i128::MIN on either side, and |a| < |b|.
        let (min, three) = (i128::MIN, 3i128);
        assert_eq!(
            (Decimal128::<4>(min) % Decimal128::<4>(three)).0,
            min % three
        );
        assert_eq!(
            (Decimal128::<4>(i128::MIN) % Decimal128::<4>(i128::MIN)).0,
            0
        );
        assert_eq!((Decimal128::<4>(5) % Decimal128::<4>(i128::MIN)).0, 5);
        assert_eq!((Decimal128::<4>(-5) % Decimal128::<4>(i128::MIN)).0, -5);
        assert_eq!((Decimal128::<4>(-5) % Decimal128::<4>(5)).0, 0);
    }

    #[test]
    #[should_panic(expected = "divisor of zero")]
    fn test_rem_by_zero() {
        let _ = Amount128::from(1) % Amount128::ZERO;
    }

    #[test]
    fn test_new_and_conversions() {
        assert_eq!(Amount128::new(), Amount128::ZERO);

        assert_eq!(Amount128::from_f32(1.5), Ok(Amount128::from_bits(15000)));
        assert_eq!(
            Amount128::from_f32(f32::MAX),
            Err(AmountErrorKind::Overflow)
        );
        assert_eq!(Amount128::from_f64(1e35), Err(AmountErrorKind::Overflow));
        assert_eq!(Amount128::from_f64(-1e35), Err(AmountErrorKind::Overflow));

        assert_eq!(Amount128::from_i128(7), Ok(Amount128::from(7)));
        assert_eq!(
            Amount128::from_i128(i128::MAX),
            Err(AmountErrorKind::Overflow)
        );
        assert_eq!(
            Amount128::from_i128(i128::MIN),
            Err(AmountErrorKind::Overflow)
        );
        assert_eq!(Amount128::from_i64(-7), Ok(Amount128::from(-7)));

        assert_eq!(Amount128::from(2).to_f64(), 2.0);
        assert_eq!(Amount128::from_bits(-2500).to_f64(), -0.25);
        // to_i128 truncates toward zero.
        assert_eq!(Amount128::from_f64(3.7).unwrap().to_i128(), 3);
        assert_eq!(Amount128::from_f64(-3.7).unwrap().to_i128(), -3);
    }

    #[test]
    fn test_abs_checked_add_sub() {
        assert_eq!(Amount128::from(-3).abs(), Amount128::from(3));
        assert_eq!(Amount128::from(3).abs(), Amount128::from(3));

        assert_eq!(
            Amount128::from(2).checked_add(Amount128::from(3)),
            Some(Amount128::from(5))
        );
        assert_eq!(
            Amount128::from(3).checked_sub(Amount128::from(1)),
            Some(Amount128::from(2))
        );
        assert_eq!(Amount128::MIN.checked_sub(Amount128::from(1)), None);
    }

    #[test]
    fn test_from_decimal_parts_range_errors() {
        // Scale-up factor 10^44 exceeds i128: Overflow.
        assert_eq!(
            Amount128::from_decimal_parts(1, 40),
            Err(AmountErrorKind::Overflow)
        );
        // Scale-down divisor 10^46 exceeds i128: any non-zero mantissa is Inexact.
        assert_eq!(
            Amount128::from_decimal_parts(1, -50),
            Err(AmountErrorKind::Inexact)
        );
    }

    #[test]
    fn test_from_decimal_parts_rounded_branches() {
        use crate::Rounding::*;
        // Zero mantissa short-circuits regardless of exponent.
        assert_eq!(
            Amount128::from_decimal_parts_rounded(0, 1000, HalfUp),
            Ok(Amount128::ZERO)
        );
        // Scale up exactly.
        assert_eq!(
            Amount128::from_decimal_parts_rounded(123, -2, HalfUp),
            Ok(Amount128::from_bits(12300))
        );
        // Scale-up overflow: factor out of range, then product out of range.
        assert_eq!(
            Amount128::from_decimal_parts_rounded(1, 40, HalfUp),
            Err(AmountErrorKind::Overflow)
        );
        assert_eq!(
            Amount128::from_decimal_parts_rounded(i128::MAX, 0, HalfUp),
            Err(AmountErrorKind::Overflow)
        );
        // |value| < 0.5 ULP (divisor 10^56 exceeds i128): rounds to zero,
        // except `Up`, which yields one signed ULP.
        assert_eq!(
            Amount128::from_decimal_parts_rounded(7, -60, Up),
            Ok(Amount128::from_bits(1))
        );
        assert_eq!(
            Amount128::from_decimal_parts_rounded(-7, -60, Up),
            Ok(Amount128::from_bits(-1))
        );
        assert_eq!(
            Amount128::from_decimal_parts_rounded(7, -60, HalfUp),
            Ok(Amount128::ZERO)
        );
        // 1.2345 at one dropped digit: rem == half, quotient 1234 is even.
        assert_eq!(
            Amount128::from_decimal_parts_rounded(12345, -5, HalfEven),
            Ok(Amount128::from_bits(1234))
        );
        assert_eq!(
            Amount128::from_decimal_parts_rounded(12345, -5, HalfDown),
            Ok(Amount128::from_bits(1234))
        );
        assert_eq!(
            Amount128::from_decimal_parts_rounded(12345, -5, Down),
            Ok(Amount128::from_bits(1234))
        );
        assert_eq!(
            Amount128::from_decimal_parts_rounded(12345, -5, Up),
            Ok(Amount128::from_bits(1235))
        );
        // Odd quotient tie rounds up under HalfEven; above-half rounds up.
        assert_eq!(
            Amount128::from_decimal_parts_rounded(12355, -5, HalfEven),
            Ok(Amount128::from_bits(1236))
        );
        assert_eq!(
            Amount128::from_decimal_parts_rounded(12346, -5, HalfEven),
            Ok(Amount128::from_bits(1235))
        );
        // Sign is carried through the magnitude rounding.
        assert_eq!(
            Amount128::from_decimal_parts_rounded(-12345, -5, Up),
            Ok(Amount128::from_bits(-1235))
        );
    }

    #[test]
    fn test_from_str_const_runtime() {
        assert_eq!(
            Amount128::from_str_const("0.0035"),
            Amount128::from_bits(35)
        );
    }

    #[test]
    #[should_panic(expected = "invalid decimal literal")]
    fn test_from_str_const_invalid() {
        let _ = Amount128::from_str_const("1.2.3");
    }

    #[test]
    fn test_floor_ceil_round_branches() {
        // Non-negative floor takes the plain truncation branch.
        assert_eq!(
            Amount128::from_f64(3.7).unwrap().floor(),
            Amount128::from(3)
        );
        assert_eq!(Amount128::from(-4).floor(), Amount128::from(-4));
        // Negative ceil truncates toward +inf (mathematical ceiling).
        assert_eq!(
            Amount128::from_f64(-3.2).unwrap().ceil(),
            Amount128::from(-3)
        );
        assert_eq!(Amount128::from(5).ceil(), Amount128::from(5));
        // ceil(x) == -floor(-x)
        let x = Amount128::from_f64(-3.2).unwrap();
        assert_eq!(x.ceil(), -(-x).floor());
        // Positive half rounds away from zero.
        assert_eq!(
            Amount128::from_f64(3.5).unwrap().round(),
            Amount128::from(4)
        );
        assert_eq!(
            Amount128::from_f64(3.4).unwrap().round(),
            Amount128::from(3)
        );
    }

    #[test]
    #[should_panic(expected = "attempt to round with overflow")]
    fn test_round_to_overflow() {
        // MAX has fractional digits .5727; rounding up exceeds the range.
        let _ = Amount128::MAX.round_to(Rounding::Up);
    }

    #[test]
    #[should_panic(expected = "attempt to multiply with overflow")]
    fn test_mul_rounded_overflow() {
        let _ = Amount128::MAX.mul_rounded(Amount128::from(2), Rounding::HalfUp);
    }

    #[test]
    fn test_checked_rounded_forms() {
        assert_eq!(
            Amount128::MAX.checked_mul_rounded(Amount128::from(2), Rounding::HalfUp),
            None
        );
        assert_eq!(
            Amount128::from(3).checked_mul_rounded(Amount128::from(2), Rounding::HalfUp),
            Some(Amount128::from(6))
        );
        assert_eq!(
            Amount128::from(1).checked_div_rounded(Amount128::from(3), Rounding::Up),
            Some(Amount128::from_bits(3334))
        );
        assert_eq!(
            Amount128::from(1).checked_div_rounded(Amount128::ZERO, Rounding::HalfUp),
            None
        );
    }

    #[test]
    #[should_panic(expected = "attempt to divide by zero")]
    fn test_div_rounded_by_zero() {
        let _ = Amount128::from(1).div_rounded(Amount128::ZERO, Rounding::HalfUp);
    }

    #[test]
    #[should_panic(expected = "attempt to divide with overflow")]
    fn test_div_rounded_overflow() {
        // MAX / 0.5 doubles the largest value.
        let _ = Amount128::MAX.div_rounded(Amount128::from_bits(5000), Rounding::HalfUp);
    }

    #[test]
    fn test_mul_div_rounded() {
        use crate::Rounding::*;
        // The product needs the full 256-bit intermediate: 10^30 * 7 / 9 at
        // scale 4 has mantissa 7 * 10^34 / 9, far beyond i64's muldiv.
        let a = Amount128::from_str_const("1000000000000000000000000000000"); // 10^30
        assert_eq!(
            a.mul_div_rounded(Amount128::from(7), Amount128::from(9), HalfUp),
            Amount128::from_str_const("777777777777777777777777777777.7778")
        );
        assert_eq!(
            a.mul_div_rounded(Amount128::from(7), Amount128::from(9), Down),
            Amount128::from_str_const("777777777777777777777777777777.7777")
        );

        // Multi-limb divisor takes the Knuth path; exact quotient.
        let b = Amount128::from_str_const("100000000000000000000"); // 10^20
        let c = Amount128::from_str_const("10000000000000000000000000"); // 10^25
        assert_eq!(
            a.mul_div_rounded(b, c, HalfUp),
            Amount128::from_str_const("10000000000000000000000000") // 10^25
        );

        // Product narrower than the divisor: quotient 0, remainder rounds.
        let one = Amount128::from(1);
        assert_eq!(one.mul_div_rounded(one, c, Down), Amount128::ZERO);
        assert_eq!(
            one.mul_div_rounded(one, c, Up),
            Amount128::from_str_const("0.0001")
        );

        // Exact tie: 0.0007 * 1 / 2 = 0.00035; quotient mantissa 3 is odd.
        let t = Amount128::from_str_const("0.0007");
        let two = Amount128::from(2);
        assert_eq!(t.mul_div_rounded(one, two, HalfUp).0, 4);
        assert_eq!(t.mul_div_rounded(one, two, HalfEven).0, 4);
        assert_eq!(t.mul_div_rounded(one, two, HalfDown).0, 3);
        assert_eq!(t.mul_div_rounded(one, two, Down).0, 3);
        assert_eq!(t.mul_div_rounded(one, two, Up).0, 4);

        // Signs: a single flip negates, a double flip cancels. Down is
        // directional (floor), so a negative -0.00035 floors to -0.0004.
        assert_eq!((-t).mul_div_rounded(one, two, Down).0, -4);
        assert_eq!(t.mul_div_rounded(-one, two, Down).0, -4);
        assert_eq!((-t).mul_div_rounded(-one, two, Down).0, 3);
        assert_eq!((-t).mul_div_rounded(one, two, Up).0, -3);

        // Checked form: zero divisor and overflow both yield None.
        assert_eq!(
            t.checked_mul_div_rounded(one, Amount128::ZERO, HalfUp),
            None
        );
        assert_eq!(
            Amount128::MAX.checked_mul_div_rounded(two, one, HalfUp),
            None
        );
    }

    #[test]
    fn test_sqrt() {
        use crate::Rounding::*;
        // Exact squares round-trip; zero is exact.
        let big = Amount128::from_str_const("1000000000000000"); // 10^15
        assert_eq!(big.mul_rounded(big, Down).sqrt(), big);
        assert_eq!(
            Amount128::from_str_const("2.25").sqrt(),
            Amount128::from_str_const("1.5")
        );
        assert_eq!(Amount128::ZERO.sqrt(), Amount128::ZERO);
        // Rounding modes (tie-free: all half modes agree).
        let two = Amount128::from(2);
        assert_eq!(two.sqrt_rounded(Down), Amount128::from_str_const("1.4142"));
        assert_eq!(
            two.sqrt_rounded(HalfEven),
            Amount128::from_str_const("1.4142")
        );
        assert_eq!(two.sqrt_rounded(Up), Amount128::from_str_const("1.4143"));
        // MAX takes the multi-limb Newton path; pinned against an
        // independently computed root.
        assert_eq!(
            Amount128::MAX.sqrt_rounded(Down),
            Amount128::from_str_const("130438178253327822.1234")
        );
        assert_eq!(
            Amount128::MAX.sqrt(),
            Amount128::from_str_const("130438178253327822.1235")
        );
        // Negative input.
        assert_eq!(Amount128::from(-1).checked_sqrt(), None);
        assert_eq!(Amount128::from(-1).checked_sqrt_rounded(HalfEven), None);
    }

    #[test]
    #[should_panic(expected = "argument of integer square root cannot be negative")]
    fn test_sqrt_negative_panics() {
        let _ = Amount128::from(-1).sqrt();
    }

    #[test]
    fn test_checked_rem_powi_recip_round_to() {
        use crate::Rounding::*;
        let a = Amount128::from_str_const("7.5");
        let b = Amount128::from(2);
        assert_eq!(a.checked_rem(b), Some(Amount128::from_str_const("1.5")));
        assert_eq!(a.checked_rem(Amount128::ZERO), None);

        assert_eq!(b.checked_powi(10), Some(Amount128::from(1024)));
        assert_eq!(Amount128::MAX.checked_powi(2), None);
        // x^2 fits but x^3 does not: exp = 3 overflows at the final odd-bit
        // multiply, exp = 7 at the in-loop odd-bit multiply.
        let big = Amount128::from(10i128.pow(12));
        assert_eq!(big.checked_powi(3), None);
        assert_eq!(big.checked_powi(7), None);

        assert_eq!(b.checked_recip(), Some(Amount128::from_str_const("0.5")));
        assert_eq!(Amount128::ZERO.checked_recip(), None);
        // The reciprocal cannot overflow on this backing: even the smallest
        // positive value at the maximum scale stays inside i128.
        assert!(Decimal128::<19>::from_bits(1).checked_recip().is_some());

        assert_eq!(a.checked_round_to(Down), Some(Amount128::from(7)));
        assert_eq!(Amount128::MAX.checked_round_to(Up), None);
    }

    #[test]
    #[should_panic(expected = "attempt to calculate the remainder")]
    fn test_rem_by_zero_panics() {
        let _ = Amount128::from(1) % Amount128::ZERO;
    }

    #[test]
    #[should_panic(expected = "attempt to multiply with overflow")]
    fn test_powi_overflow_panics() {
        let _ = Amount128::MAX.powi(2);
    }

    #[test]
    #[should_panic(expected = "attempt to divide by zero")]
    fn test_recip_zero_panics() {
        let _ = Amount128::ZERO.recip();
    }

    #[test]
    #[should_panic(expected = "attempt to round with overflow")]
    fn test_ceil_edge_overflow_panics() {
        // MAX has a non-zero fraction, so ceiling steps past the range; this
        // must panic in every build profile, never wrap.
        let _ = Amount128::MAX.ceil();
    }

    #[test]
    #[should_panic(expected = "attempt to round with overflow")]
    fn test_floor_edge_overflow_panics() {
        let _ = Amount128::MIN.floor();
    }

    #[test]
    #[should_panic(expected = "attempt to divide by zero")]
    fn test_mul_div_rounded_by_zero() {
        let _ = Amount128::from(1).mul_div_rounded(
            Amount128::from(1),
            Amount128::ZERO,
            Rounding::HalfUp,
        );
    }

    #[test]
    #[should_panic(expected = "attempt to divide with overflow")]
    fn test_mul_div_rounded_overflow() {
        let _ = Amount128::MAX.mul_div_rounded(
            Amount128::from(2),
            Amount128::from(1),
            Rounding::HalfUp,
        );
    }

    #[test]
    #[should_panic(expected = "attempt to divide by zero")]
    fn test_div_int_rounded_by_zero() {
        let _ = Amount128::from(1).div_int_rounded(0, Rounding::HalfUp);
    }

    #[test]
    fn test_checked_div_int_rounded_zero() {
        assert_eq!(
            Amount128::from(1).checked_div_int_rounded(0, Rounding::HalfUp),
            None
        );
        assert_eq!(
            Amount128::from(1).checked_div_int_rounded(3, Rounding::Down),
            Some(Amount128::from_bits(3333))
        );
    }

    #[test]
    fn test_sign_predicates_bits_bytes() {
        assert!(Amount128::from(1).is_positive());
        assert!(!Amount128::ZERO.is_positive());
        assert!(!Amount128::from(-1).is_positive());
        assert!(Amount128::from(-1).is_negative());
        assert!(!Amount128::ZERO.is_negative());
        assert!(!Amount128::from(1).is_negative());

        assert_eq!(Amount128::from_bits(123).to_bits(), 123);

        let be = Amount128::from(1).to_be_bytes();
        assert_eq!(&be[14..], &[0x27, 0x10]); // 10000 == 0x2710
        assert_eq!(Amount128::from_be_bytes(be), Amount128::from(1));
        assert_eq!(
            Amount128::from_be_bytes(Amount128::from(-1).to_be_bytes()),
            Amount128::from(-1)
        );
    }

    #[test]
    fn test_clamp_min_max() {
        let (lo, hi) = (Amount128::ZERO, Amount128::ONE);
        assert_eq!(Amount128::from(-5).clamp(lo, hi), lo);
        assert_eq!(
            Amount128::from_bits(5000).clamp(lo, hi),
            Amount128::from_bits(5000)
        );

        let (a, b) = (Amount128::from(1), Amount128::from(2));
        assert_eq!(a.min(b), a);
        assert_eq!(b.min(a), a);
        assert_eq!(a.max(b), b);
        assert_eq!(b.max(a), b);
    }

    #[test]
    fn test_from_saturating() {
        // From<i128> saturates out-of-range values.
        assert_eq!(Amount128::from(5i128), Amount128::from(5));
        assert_eq!(Amount128::from(i128::MAX), Amount128::MAX);
        assert_eq!(Amount128::from(i128::MIN), Amount128::MIN);
        // From<f64> saturates likewise.
        assert_eq!(Amount128::from(2.5f64), Amount128::from_bits(25000));
        assert_eq!(Amount128::from(1e40f64), Amount128::MAX);
        assert_eq!(Amount128::from(-1e40f64), Amount128::MIN);
        // From<f32> goes through f64.
        assert_eq!(Amount128::from(1.5f32), Amount128::from_bits(15000));
    }

    #[test]
    fn test_int_operand_ops() {
        let a = Amount128::from(10);
        assert_eq!(a + 5i64, Amount128::from(15));
        assert_eq!(a + 5i32, Amount128::from(15));
        assert_eq!(a - 3i64, Amount128::from(7));
        assert_eq!(a - 3i32, Amount128::from(7));
        assert_eq!(a * 2i64, Amount128::from(20));
        assert_eq!(a * -2i32, Amount128::from(-20));
    }

    #[test]
    fn test_assign_ops() {
        let mut v = Amount128::from(10);
        v += Amount128::from(2);
        assert_eq!(v, Amount128::from(12));
        v -= Amount128::from(4);
        assert_eq!(v, Amount128::from(8));
        v *= Amount128::from(2);
        assert_eq!(v, Amount128::from(16));
        v /= Amount128::from(4);
        assert_eq!(v, Amount128::from(4));
        v %= Amount128::from(3);
        assert_eq!(v, Amount128::from(1));
    }

    #[test]
    fn test_from_str_and_iter_traits() {
        // From<&str> panics on bad input, parses good input.
        assert_eq!(Amount128::from("1.5"), Amount128::from_bits(15000));

        let arr = [Amount128::from(1), Amount128::from(2), Amount128::from(3)];
        // By-reference Sum/Product over an iterator of &Amount128.
        let s: Amount128 = arr.iter().sum();
        assert_eq!(s, Amount128::from(6));
        let p: Amount128 = arr.iter().product();
        assert_eq!(p, Amount128::from(6));
        // By-value Product (owned iterator).
        let p: Amount128 = arr.iter().copied().product();
        assert_eq!(p, Amount128::from(6));
        // Empty iterators yield the identities.
        assert_eq!([].iter().sum::<Amount128>(), Amount128::ZERO);
        assert_eq!([].iter().product::<Amount128>(), Amount128::ONE);
    }

    #[test]
    #[should_panic(expected = "called `Result::unwrap()` on an `Err` value")]
    fn test_from_str_ref_invalid() {
        let _ = Amount128::from("not a number");
    }

    #[test]
    fn test_display_precision_fallback() {
        let x = Amount128::from_bits(15000);
        assert_eq!(format!("{x:.2}"), "1.50");
        // Requested precision cannot fit the 128-byte format buffer.
        assert_eq!(format!("{x:.200}"), "Amount::ERROR");
    }

    /// Differential test: for values within the i64 range, Decimal128 must
    /// agree with Decimal (the i64-backed type) on every operation and
    /// rounding mode.
    #[test]
    fn test_differential_vs_decimal64() {
        use crate::Decimal;
        use crate::Rounding::*;

        let mut state = 0x853C49E6748FEA9Bu64;
        let mut next = move || {
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            state.wrapping_mul(0x2545F4914F6CDD1D)
        };

        for _ in 0..20000 {
            // Keep magnitudes below ~2^30 so i64 multiplication cannot overflow.
            let a = (next() as i64) >> 34;
            let b = (next() as i64) >> 34;
            let da = Decimal::<4>(a);
            let db = Decimal::<4>(b);
            let wa = Decimal128::<4>(a as i128);
            let wb = Decimal128::<4>(b as i128);

            assert_eq!((da + db).0 as i128, (wa + wb).0);
            assert_eq!((da - db).0 as i128, (wa - wb).0);
            assert_eq!((da * db).0 as i128, (wa * wb).0, "mul {a} * {b}");
            assert_eq!(format!("{da}"), format!("{wa}"));
            assert_eq!(format!("{da:.2}"), format!("{wa:.2}"));

            // The integer-rounding helpers: ceil/floor/round delegate to
            // round_to per backing; trunc/fract stay hand-rolled.
            assert_eq!(da.ceil().0 as i128, wa.ceil().0, "ceil {a}");
            assert_eq!(da.floor().0 as i128, wa.floor().0, "floor {a}");
            assert_eq!(da.round().0 as i128, wa.round().0, "round {a}");
            assert_eq!(da.trunc().0 as i128, wa.trunc().0, "trunc {a}");
            assert_eq!(da.fract().0 as i128, wa.fract().0, "fract {a}");

            // Cross-scale operands (an 8-digit rate) for mul_rounded.
            let rb = Decimal::<8>(b);
            let wrb = Decimal128::<8>(b as i128);
            for mode in [HalfUp, HalfDown, HalfEven, Down, Up] {
                assert_eq!(
                    da.mul_rounded(db, mode).0 as i128,
                    wa.mul_rounded(wb, mode).0,
                    "mul_rounded {a} * {b}"
                );
                assert_eq!(
                    da.mul_rounded(rb, mode).0 as i128,
                    wa.mul_rounded(wrb, mode).0,
                    "mul_rounded cross-scale {a} * {b}"
                );
                assert_eq!(
                    da.round_to(mode).0 as i128,
                    wa.round_to(mode).0,
                    "round_to {a}"
                );
                if b != 0 {
                    assert_eq!(
                        da.div_rounded(db, mode).0 as i128,
                        wa.div_rounded(wb, mode).0,
                        "div_rounded {a} / {b}"
                    );
                    assert_eq!(
                        da.div_rounded_to::<8>(db, mode).0 as i128,
                        wa.div_rounded_to::<8>(wb, mode).0,
                        "div_rounded_to::<8> {a} / {b}"
                    );
                    assert_eq!(
                        da.div_int_rounded(b, mode).0 as i128,
                        wa.div_int_rounded(b, mode).0,
                        "div_int_rounded {a} / {b}"
                    );
                    assert_eq!(
                        da.checked_mul_div_rounded(da, db, mode)
                            .map(|v| v.0 as i128),
                        wa.checked_mul_div_rounded(wa, wb, mode).map(|v| v.0),
                        "mul_div ({a} * {a}) / {b}"
                    );
                    assert_eq!(
                        da.abs().sqrt_rounded(mode).0 as i128,
                        wa.abs().sqrt_rounded(mode).0,
                        "sqrt |{a}|"
                    );
                }
            }
            if b != 0 {
                assert_eq!((da / db).0 as i128, (wa / wb).0, "div {a} / {b}");
                assert_eq!((da % db).0 as i128, (wa % wb).0, "rem {a} % {b}");
                assert_eq!(
                    da.checked_div(db).map(|v| v.0 as i128),
                    wa.checked_div(wb).map(|v| v.0),
                    "checked_div {a} / {b}"
                );
            }
        }
    }
}
