//! 256-bit backed decimal fixed-point type.
//!
//! [`Decimal256`] extends the power-of-10 scaling design to a 256-bit signed
//! backing integer ([`I256`]). Addition and subtraction are two native
//! add/adc (sub/sbb) pairs; multiplication produces a 512-bit intermediate
//! from 64-bit limbs and re-scales with a single word-based long division
//! (10^DIGITS always fits one limb); division scales the dividend up and
//! uses Knuth's algorithm D with the same 128-by-64 division primitive.

use crate::AmountErrorKind;
use crate::AmountSign;
use crate::Rounding;
use crate::limbs::{
    cmp_twice_rem_u64, dec_div, dec_mul, div_knuth, div_words_by_pow10, div_words_by_word,
    mul_add_word, parse_decimal_mag_rounded, round_up_by_cmp, sig_limbs, str_mag, upow10,
};
use core::cmp::Ordering;
use core::fmt;
use core::ops::*;

/// A 256-bit signed integer in two's complement form, used as the backing
/// storage of [`Decimal256`]. Only the operations the decimal type needs are
/// provided; it is not a general-purpose big integer.
#[derive(Clone, Copy, PartialEq, Eq, Default, Hash)]
pub struct I256 {
    lo: u128,
    hi: i128,
}

impl I256 {
    /// Constant equal to zero.
    pub const ZERO: Self = I256 { lo: 0, hi: 0 };
    /// The largest representable value, 2^255 - 1.
    pub const MAX: Self = I256 {
        lo: u128::MAX,
        hi: i128::MAX,
    };
    /// The smallest value used by the decimal types: `-MAX` (symmetric range).
    pub const MIN: Self = I256 {
        lo: 1,
        hi: i128::MIN,
    };

    /// Sign-extends an `i128` into an `I256`.
    #[inline]
    pub const fn from_i128(v: i128) -> Self {
        I256 {
            lo: v as u128,
            hi: v >> 127,
        }
    }

    /// Returns `true` if the value is negative.
    #[inline]
    pub const fn is_negative(self) -> bool {
        self.hi < 0
    }

    /// Returns `true` if the value is zero.
    #[inline]
    pub const fn is_zero(self) -> bool {
        self.hi == 0 && self.lo == 0
    }

    /// Two's complement negation (wrapping).
    #[inline]
    pub const fn wrapping_neg(self) -> Self {
        let lo = (!self.lo).wrapping_add(1);
        let hi = (!self.hi).wrapping_add((lo == 0) as i128);
        I256 { lo, hi }
    }

    /// Checked addition, `None` on signed overflow.
    #[inline]
    pub const fn checked_add(self, rhs: Self) -> Option<Self> {
        let (lo, carry) = self.lo.overflowing_add(rhs.lo);
        let (hi, o1) = self.hi.overflowing_add(rhs.hi);
        let (hi, o2) = hi.overflowing_add(carry as i128);
        if o1 != o2 {
            None
        } else {
            Some(I256 { lo, hi })
        }
    }

    /// Checked subtraction, `None` on signed overflow.
    #[inline]
    pub const fn checked_sub(self, rhs: Self) -> Option<Self> {
        let (lo, borrow) = self.lo.overflowing_sub(rhs.lo);
        let (hi, o1) = self.hi.overflowing_sub(rhs.hi);
        let (hi, o2) = hi.overflowing_sub(borrow as i128);
        if o1 != o2 {
            None
        } else {
            Some(I256 { lo, hi })
        }
    }

    /// Decomposes into a sign and a 4-limb little-endian magnitude.
    #[inline]
    pub(crate) const fn to_sign_mag(self) -> (bool, [u64; 4]) {
        let neg = self.hi < 0;
        let v = if neg { self.wrapping_neg() } else { self };
        (
            neg,
            [
                v.lo as u64,
                (v.lo >> 64) as u64,
                v.hi as u64,
                ((v.hi as u128) >> 64) as u64,
            ],
        )
    }

    /// Builds a value from a sign and a 4-limb magnitude. Returns `None` if
    /// the magnitude exceeds 2^255 - 1 (the symmetric range bound).
    #[inline]
    pub(crate) const fn from_sign_mag(neg: bool, mag: [u64; 4]) -> Option<Self> {
        if mag[3] >> 63 != 0 {
            return None;
        }
        let v = I256 {
            lo: (mag[0] as u128) | ((mag[1] as u128) << 64),
            hi: ((mag[2] as u128) | ((mag[3] as u128) << 64)) as i128,
        };
        Some(if neg { v.wrapping_neg() } else { v })
    }

    /// Returns the memory representation as 32 bytes in little-endian order.
    #[inline]
    pub const fn to_le_bytes(self) -> [u8; 32] {
        let lo = self.lo.to_le_bytes();
        let hi = self.hi.to_le_bytes();
        let mut out = [0u8; 32];
        let mut i = 0;
        while i < 16 {
            out[i] = lo[i];
            out[i + 16] = hi[i];
            i += 1;
        }
        out
    }

    /// Creates a value from its 32-byte little-endian representation.
    #[inline]
    pub const fn from_le_bytes(bytes: [u8; 32]) -> Self {
        let mut lo = [0u8; 16];
        let mut hi = [0u8; 16];
        let mut i = 0;
        while i < 16 {
            lo[i] = bytes[i];
            hi[i] = bytes[i + 16];
            i += 1;
        }
        I256 {
            lo: u128::from_le_bytes(lo),
            hi: i128::from_le_bytes(hi),
        }
    }
}

impl Ord for I256 {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        match self.hi.cmp(&other.hi) {
            Ordering::Equal => self.lo.cmp(&other.lo),
            o => o,
        }
    }
}

impl PartialOrd for I256 {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl fmt::Debug for I256 {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut buf = [0u8; 96];
        let (neg, mag) = self.to_sign_mag();
        match str_mag(&mag, neg, 0, None, AmountSign::Negative, &mut buf) {
            Some(s) => f.write_str(s),
            None => f.write_str("I256::ERROR"),
        }
    }
}

/// Decimal fixed-point number backed by a 256-bit signed integer, with
/// `DIGITS` fractional decimal digits (`DIGITS <= 19`).
///
/// Same decimal semantics as [`Decimal`](crate::Decimal) and
/// [`Decimal128`](crate::Decimal128), with a range of about
/// ±5.8 * 10^(76 - DIGITS).
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Decimal256<const DIGITS: u8>(pub I256);

/// Amount256 is a 256-bit integer based decimal type with 4 decimal digits precision.
pub type Amount256 = Decimal256<4>;
/// Rate256 is a 256-bit integer based decimal type with 8 decimal digits precision.
pub type Rate256 = Decimal256<8>;

/// Remainder of the backing integers (sign follows the dividend, like `%` on
/// primitive integers). `b` must be non-zero.
#[inline]
fn i256_rem(a: I256, b: I256) -> I256 {
    let (an, am) = a.to_sign_mag();
    let (_, bm) = b.to_sign_mag();
    let n = sig_limbs(&bm);
    let mut r = [0u64; 4];
    if n == 1 {
        let mut q = am;
        r[0] = div_words_by_word(&mut q, bm[0]);
    } else {
        let m = sig_limbs(&am);
        if m < n {
            // Fewer dividend limbs than divisor limbs: |a| < |b|, so the
            // remainder is the dividend itself.
            r[..m].copy_from_slice(&am[..m]);
        } else {
            let mut q = [0u64; 3];
            div_knuth(&mut q[..m - n + 1], &mut r[..n], &am[..m], &bm[..n]);
        }
    }
    // The remainder magnitude is < |b| <= MAX, so this cannot fail.
    I256::from_sign_mag(an, r).unwrap()
}

impl<const DIGITS: u8> Decimal256<DIGITS> {
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
    pub const MAX: Self = Decimal256::<DIGITS>(I256::MAX);
    /// The smallest value that can be represented by this type (`-MAX`).
    pub const MIN: Self = Decimal256::<DIGITS>(I256::MIN);

    /// Constant equal to '1'
    pub const ONE: Self = Decimal256::<DIGITS>(I256::from_i128(Self::SCALE_INT));
    /// Constant equal to '-1'
    pub const MINUS_ONE: Self = Decimal256::<DIGITS>(I256::from_i128(-Self::SCALE_INT));
    /// Constant equal to '0'
    pub const ZERO: Self = Decimal256::<DIGITS>(I256::ZERO);

    /// The multiplier used to scale values up, as f64.
    pub const SCALE_F64: f64 = Self::SCALE_INT as f64;

    /// 2^128 as f64, used to assemble/split wide float conversions.
    const TWO_POW_128: f64 = 340282366920938463463374607431768211456.0;

    /// The largest f64 value that can be represented, about 5.8e76 / 10^DIGITS.
    pub const F64_MAX: f64 =
        ((i128::MAX as f64) * Self::TWO_POW_128 + (u128::MAX as f64)) / Self::SCALE_F64;
    /// The smallest f64 value that can be represented.
    pub const F64_MIN: f64 = -Self::F64_MAX;

    /// Constructs a new decimal integer with value 0.
    ///
    /// # Examples
    /// ```rust
    /// use fin_decimal::Amount256;
    /// let i = Amount256::new();
    /// assert_eq!(i, Amount256::ZERO);
    /// ```
    #[inline]
    pub fn new() -> Self {
        Self::ZERO
    }

    /// Tries to convert a f32 to a Decimal256.
    #[inline]
    pub fn from_f32(val: f32) -> Result<Self, AmountErrorKind> {
        Self::from_f64(val as f64)
    }

    /// Tries to convert a f64 to a Decimal256.
    pub fn from_f64(val: f64) -> Result<Self, AmountErrorKind> {
        if !(Self::F64_MIN..=Self::F64_MAX).contains(&val) {
            return Err(AmountErrorKind::Overflow);
        }
        let x = val * Self::SCALE_F64;
        // Values within i128 take the exact native cast; wider ones are split
        // into 2^128-sized halves (f64 has only 53 mantissa bits, so the
        // split loses nothing that the input still carried).
        const TWO_POW_127: f64 = 170141183460469231731687303715884105728.0;
        if (-TWO_POW_127..TWO_POW_127).contains(&x) {
            Ok(Decimal256::<DIGITS>(I256::from_i128(x as i128)))
        } else {
            let two128 = Self::TWO_POW_128;
            // floor(x / 2^128) without std: the cast truncates toward zero,
            // so step down once when the remainder comes out negative.
            let mut hi = (x / two128) as i128;
            let mut lo = x - (hi as f64) * two128;
            if lo < 0.0 {
                hi -= 1;
                lo += two128;
            }
            Ok(Decimal256::<DIGITS>(I256 { lo: lo as u128, hi }))
        }
    }

    /// Converts an i128 to a Decimal256. Always exact: every i128 fits.
    #[inline]
    pub const fn from_i128(val: i128) -> Self {
        let neg = val < 0;
        let m = val.unsigned_abs();
        let mut mag = [m as u64, (m >> 64) as u64, 0, 0];
        // |i128| * 10^19 < 2^191: never overflows four limbs.
        mul_add_word(&mut mag, Self::SCALE_U64, 0);
        match I256::from_sign_mag(neg, mag) {
            Some(v) => Decimal256::<DIGITS>(v),
            None => unreachable!(),
        }
    }

    /// Converts an i64 to a Decimal256. Always exact.
    #[inline]
    pub const fn from_i64(val: i64) -> Self {
        Self::from_i128(val as i128)
    }

    /// Converts the Decimal256 back into an f64 (approximate for values wider
    /// than the f64 mantissa).
    pub fn to_f64(self) -> f64 {
        ((self.0.hi as f64) * Self::TWO_POW_128 + self.0.lo as f64) / Self::SCALE_F64
    }

    /// Converts the Decimal256 back into an i128, truncating the fractional
    /// part and saturating to `i128::MIN`/`i128::MAX` when out of range.
    pub const fn to_i128(self) -> i128 {
        let (neg, mag) = self.0.to_sign_mag();
        let mut q = mag;
        div_words_by_pow10::<DIGITS>(&mut q);
        if q[2] != 0 || q[3] != 0 || ((q[0] as u128) | ((q[1] as u128) << 64)) > i128::MAX as u128 {
            return if neg { i128::MIN } else { i128::MAX };
        }
        let v = ((q[0] as u128) | ((q[1] as u128) << 64)) as i128;
        if neg { -v } else { v }
    }

    /// Parses a decimal string, rounding any fractional digits beyond this
    /// type's scale with the given [`Rounding`] mode.
    ///
    /// # Examples
    /// ```
    /// use fin_decimal::{Amount256, Rounding};
    /// assert_eq!(
    ///     Amount256::from_str_rounded("1.00005", Rounding::HalfEven),
    ///     Ok(Amount256::from(1)),
    /// );
    /// ```
    pub const fn from_str_rounded(src: &str, mode: Rounding) -> Result<Self, AmountErrorKind> {
        let (neg, mag) = match parse_decimal_mag_rounded::<4>(src, DIGITS, mode) {
            Ok(v) => v,
            Err(e) => return Err(e),
        };
        match I256::from_sign_mag(neg, mag) {
            Some(v) => Ok(Decimal256::<DIGITS>(v)),
            None => Err(AmountErrorKind::Overflow),
        }
    }

    /// Parses a decimal string with [`Rounding::HalfUp`], panicking on invalid
    /// input — intended for compile-time constants, where the panic becomes a
    /// compile error.
    ///
    /// # Examples
    /// ```
    /// use fin_decimal::Amount256;
    /// const BUDGET: Amount256 = Amount256::from_str_const("1000000000000000000000000.25");
    /// assert_eq!(BUDGET.to_string(), "1000000000000000000000000.25");
    /// ```
    pub const fn from_str_const(src: &str) -> Self {
        match Self::from_str_rounded(src, Rounding::HalfUp) {
            Ok(v) => v,
            Err(_) => panic!("invalid decimal literal"),
        }
    }

    /// Sign and 4-limb magnitude, for the shared formatting code in
    /// [`common`](crate::common).
    #[inline]
    pub(crate) const fn sign_mag4(self) -> (bool, [u64; 4]) {
        self.0.to_sign_mag()
    }

    /// Computes the absolute value of self.
    #[inline]
    pub const fn abs(self) -> Self {
        if self.0.is_negative() {
            Decimal256::<DIGITS>(self.0.wrapping_neg())
        } else {
            self
        }
    }

    /// Checked addition. Computes `self + rhs`, returning `None` if overflow occurred.
    #[inline]
    pub const fn checked_add(self, rhs: Self) -> Option<Self> {
        match self.0.checked_add(rhs.0) {
            Some(v) => Some(Decimal256::<DIGITS>(v)),
            None => None,
        }
    }

    /// Checked subtraction. Computes `self - rhs`, returning `None` if overflow occurred.
    #[inline]
    pub const fn checked_sub(self, rhs: Self) -> Option<Self> {
        match self.0.checked_sub(rhs.0) {
            Some(v) => Some(Decimal256::<DIGITS>(v)),
            None => None,
        }
    }

    /// Checked multiplication. Computes `self * rhs`, returning `None` if overflow occurred.
    #[inline]
    pub const fn checked_mul(self, rhs: Self) -> Option<Self> {
        let (an, am) = self.0.to_sign_mag();
        let (bn, bm) = rhs.0.to_sign_mag();
        match dec_mul::<DIGITS, 4, 8>(an, &am, bn, &bm, Rounding::HalfUp) {
            Some((neg, mag)) => match I256::from_sign_mag(neg, mag) {
                Some(v) => Some(Decimal256::<DIGITS>(v)),
                None => unreachable!(),
            },
            None => None,
        }
    }

    /// Checked division. Computes `self / rhs`, returning `None` if `rhs == 0`
    /// or the division results in overflow.
    #[inline]
    pub fn checked_div(self, rhs: Self) -> Option<Self> {
        let (an, am) = self.0.to_sign_mag();
        let (bn, bm) = rhs.0.to_sign_mag();
        dec_div::<DIGITS, 4, 5>(an, &am, bn, &bm, Rounding::HalfUp)
            .map(|(neg, mag)| Decimal256::<DIGITS>(I256::from_sign_mag(neg, mag).unwrap()))
    }

    /// Takes the reciprocal (inverse) of a number, 1/x.
    #[inline]
    pub fn recip(self) -> Self {
        Self::ONE / self
    }

    /// Returns the integer part of a number (rounding toward zero).
    ///
    /// # Examples
    /// ```
    /// use fin_decimal::Amount256;
    /// assert_eq!(Amount256::from_f64(-3.7).unwrap().trunc(), Amount256::from(-3));
    /// ```
    pub const fn trunc(self) -> Self {
        let (neg, mag) = self.0.to_sign_mag();
        let mut q = mag;
        div_words_by_pow10::<DIGITS>(&mut q);
        // q * scale <= mag: cannot overflow or exceed range.
        mul_add_word(&mut q, Self::SCALE_U64, 0);
        match I256::from_sign_mag(neg, q) {
            Some(v) => Decimal256::<DIGITS>(v),
            None => unreachable!(),
        }
    }

    /// Returns the fractional part of a number.
    #[inline]
    pub const fn fract(self) -> Self {
        // |fract| < 1: the subtraction cannot overflow.
        match self.0.checked_sub(self.trunc().0) {
            Some(v) => Decimal256::<DIGITS>(v),
            None => unreachable!(),
        }
    }

    /// Returns the largest integer less than or equal to a number.
    ///
    /// # Examples
    /// ```
    /// use fin_decimal::Amount256;
    /// assert_eq!(Amount256::from_f64(-3.7).unwrap().floor(), Amount256::from(-4));
    /// ```
    #[inline]
    pub const fn floor(self) -> Self {
        self.round_to(Rounding::Down)
    }

    /// Returns the smallest integer greater than or equal to a number.
    ///
    /// # Examples
    /// ```
    /// use fin_decimal::Amount256;
    /// assert_eq!(Amount256::from_f64(3.01).unwrap().ceil(), Amount256::from(4));
    /// ```
    #[inline]
    pub const fn ceil(self) -> Self {
        self.round_to(Rounding::Up)
    }

    /// Returns the nearest integer to a number. Rounds half-way cases away
    /// from `0.0`.
    ///
    /// # Examples
    /// ```
    /// use fin_decimal::Amount256;
    /// assert_eq!(Amount256::from_f64(-3.5).unwrap().round(), Amount256::from(-4));
    /// ```
    #[inline]
    pub const fn round(self) -> Self {
        self.round_to(Rounding::HalfUp)
    }

    /// Explicitly rounds the value to an integer using the specified rounding mode.
    ///
    /// # Panics
    /// Panics if rounding up at the very edge of the range overflows.
    pub const fn round_to(self, mode: Rounding) -> Self {
        let (neg, mag) = self.0.to_sign_mag();
        let mut q = mag;
        let rem = div_words_by_pow10::<DIGITS>(&mut q);
        if rem == 0 {
            return self;
        }
        if round_up_by_cmp(
            cmp_twice_rem_u64(rem, Self::SCALE_U64),
            false,
            q[0] & 1 != 0,
            neg,
            mode,
        ) {
            mul_add_word(&mut q, 1, 1);
        }
        let overflow = mul_add_word(&mut q, Self::SCALE_U64, 0);
        match (overflow, I256::from_sign_mag(neg, q)) {
            (false, Some(v)) => Decimal256::<DIGITS>(v),
            _ => panic!("attempt to round with overflow"),
        }
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
    /// Panics if the result overflows.
    pub const fn mul_rounded<const RHS_DIGITS: u8>(
        self,
        rhs: Decimal256<RHS_DIGITS>,
        mode: Rounding,
    ) -> Self {
        let (an, am) = self.0.to_sign_mag();
        let (bn, bm) = rhs.0.to_sign_mag();
        match dec_mul::<RHS_DIGITS, 4, 8>(an, &am, bn, &bm, mode) {
            Some((neg, mag)) => match I256::from_sign_mag(neg, mag) {
                Some(v) => Decimal256::<DIGITS>(v),
                None => unreachable!(),
            },
            None => panic!("attempt to multiply with overflow"),
        }
    }

    /// Divide by another decimal, explicitly applying the given rounding mode.
    ///
    /// # Panics
    /// Panics if `rhs` is zero or the result overflows.
    pub fn div_rounded(self, rhs: Self, mode: Rounding) -> Self {
        self.div_rounded_to::<DIGITS>(rhs, mode)
    }

    /// Divide by another decimal of the same scale, producing the quotient at
    /// an explicitly chosen scale with the given rounding mode. The exact
    /// quotient is rounded once at `TO_DIGITS` digits, so a proportion of two
    /// amounts can be taken directly as a higher-precision rate.
    ///
    /// # Panics
    /// Panics if `rhs` is zero or the result overflows.
    pub fn div_rounded_to<const TO_DIGITS: u8>(
        self,
        rhs: Self,
        mode: Rounding,
    ) -> Decimal256<TO_DIGITS> {
        if rhs.0.is_zero() {
            panic!("Can't divide by zero");
        }
        let (an, am) = self.0.to_sign_mag();
        let (bn, bm) = rhs.0.to_sign_mag();
        match dec_div::<TO_DIGITS, 4, 5>(an, &am, bn, &bm, mode) {
            Some((neg, mag)) => Decimal256::<TO_DIGITS>(I256::from_sign_mag(neg, mag).unwrap()),
            None => panic!("attempt to divide with overflow"),
        }
    }

    /// Divide by an integer, explicitly applying the given rounding mode: the
    /// exact quotient is rounded once at the type's own scale. The divisor is
    /// a single word, so this is plain word division over the limbs.
    ///
    /// # Panics
    /// Panics if `n` is zero. The result itself cannot overflow: its magnitude
    /// never exceeds `self`'s (rounding up only happens for `|n| >= 2`).
    pub fn div_int_rounded(self, n: i64, mode: Rounding) -> Self {
        if n == 0 {
            panic!("Can't divide by zero");
        }
        let (an, mut mag) = self.0.to_sign_mag();
        let neg = an != (n < 0);
        let d = n.unsigned_abs();
        let r = div_words_by_word(&mut mag, d);
        if round_up_by_cmp(cmp_twice_rem_u64(r, d), r == 0, mag[0] & 1 != 0, neg, mode) {
            mul_add_word(&mut mag, 1, 1);
        }
        Decimal256::<DIGITS>(I256::from_sign_mag(neg, mag).unwrap())
    }

    /// Returns `true` if `self` is positive and `false` if the number is zero or negative.
    #[inline]
    pub const fn is_positive(self) -> bool {
        !self.0.is_negative() && !self.0.is_zero()
    }

    /// Returns `true` if `self` is negative and `false` if the number is zero or positive.
    #[inline]
    pub const fn is_negative(self) -> bool {
        self.0.is_negative()
    }

    /// Returns a number that represents the sign of `self`.
    pub const fn signum(self) -> Self {
        if self.0.is_negative() {
            Self::MINUS_ONE
        } else if self.0.is_zero() {
            Self::ZERO
        } else {
            Self::ONE
        }
    }

    /// Raw transmutation to the backing scaled integer.
    #[inline]
    pub const fn to_bits(self) -> I256 {
        self.0
    }

    /// Raw transmutation from a backing scaled integer.
    #[inline]
    pub const fn from_bits(v: I256) -> Self {
        Decimal256::<DIGITS>(v)
    }

    /// Return the memory representation of this value as a byte array in
    /// little-endian byte order.
    #[inline]
    pub const fn to_le_bytes(self) -> [u8; 32] {
        self.0.to_le_bytes()
    }

    /// Create a value from its representation as a byte array in little endian.
    #[inline]
    pub const fn from_le_bytes(bytes: [u8; 32]) -> Self {
        Decimal256::<DIGITS>(I256::from_le_bytes(bytes))
    }

    /// Raises a number to an integer power. Usable in const contexts.
    pub const fn powi(self, mut exp: u32) -> Self {
        let mut base = self;
        let mut acc = Self::ONE;
        while exp > 1 {
            if (exp & 1) == 1 {
                acc = base.mul_rounded(acc, Rounding::HalfUp);
            }
            exp /= 2;
            base = base.mul_rounded(base, Rounding::HalfUp);
        }
        if exp == 1 {
            acc = base.mul_rounded(acc, Rounding::HalfUp);
        }
        acc
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

impl<const DIGITS: u8> From<i32> for Decimal256<DIGITS> {
    #[inline]
    fn from(item: i32) -> Self {
        Self::from_i128(item as i128)
    }
}

impl<const DIGITS: u8> From<i64> for Decimal256<DIGITS> {
    #[inline]
    fn from(item: i64) -> Self {
        Self::from_i128(item as i128)
    }
}

impl<const DIGITS: u8> From<i128> for Decimal256<DIGITS> {
    #[inline]
    fn from(item: i128) -> Self {
        Self::from_i128(item)
    }
}

impl<const DIGITS: u8> From<f64> for Decimal256<DIGITS> {
    /// Converts an f64, saturating to `MIN`/`MAX` when out of range.
    #[inline]
    fn from(item: f64) -> Self {
        match Self::from_f64(item) {
            Ok(v) => v,
            Err(_) if item < 0.0 => Self::MIN,
            Err(_) => Self::MAX,
        }
    }
}

impl<const DIGITS: u8> From<f32> for Decimal256<DIGITS> {
    #[inline]
    fn from(item: f32) -> Self {
        Self::from(item as f64)
    }
}

impl<const DIGITS: u8> PartialOrd<i64> for Decimal256<DIGITS> {
    #[inline]
    fn partial_cmp(&self, other: &i64) -> Option<Ordering> {
        PartialOrd::partial_cmp(&self.0, &Self::from_i64(*other).0)
    }
}

impl<const DIGITS: u8> PartialEq<i64> for Decimal256<DIGITS> {
    #[inline]
    fn eq(&self, other: &i64) -> bool {
        self.0 == Self::from_i64(*other).0
    }
}

impl<const DIGITS: u8> Neg for Decimal256<DIGITS> {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self::Output {
        Decimal256::<DIGITS>(self.0.wrapping_neg())
    }
}

impl<const DIGITS: u8> Add for Decimal256<DIGITS> {
    type Output = Self;
    /// # Panics
    /// Panics if the result overflows.
    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        match self.0.checked_add(rhs.0) {
            Some(v) => Decimal256::<DIGITS>(v),
            None => panic!("attempt to add with overflow"),
        }
    }
}

impl<const DIGITS: u8> Add<i64> for Decimal256<DIGITS> {
    type Output = Self;
    #[inline]
    fn add(self, other: i64) -> Self {
        self + Self::from_i64(other)
    }
}

impl<const DIGITS: u8> Add<i32> for Decimal256<DIGITS> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: i32) -> Self {
        self + Self::from_i128(rhs as i128)
    }
}

impl<const DIGITS: u8> Sub for Decimal256<DIGITS> {
    type Output = Self;
    /// # Panics
    /// Panics if the result overflows.
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        match self.0.checked_sub(rhs.0) {
            Some(v) => Decimal256::<DIGITS>(v),
            None => panic!("attempt to subtract with overflow"),
        }
    }
}

impl<const DIGITS: u8> Sub<i64> for Decimal256<DIGITS> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: i64) -> Self {
        self - Self::from_i64(rhs)
    }
}

impl<const DIGITS: u8> Sub<i32> for Decimal256<DIGITS> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: i32) -> Self {
        self - Self::from_i128(rhs as i128)
    }
}

impl<const DIGITS: u8> Mul for Decimal256<DIGITS> {
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

impl<const DIGITS: u8> Mul<i64> for Decimal256<DIGITS> {
    type Output = Self;
    /// # Panics
    /// Panics if the result overflows.
    #[inline]
    fn mul(self, rhs: i64) -> Self {
        let (neg, mut mag) = self.0.to_sign_mag();
        let overflow = mul_add_word(&mut mag, rhs.unsigned_abs(), 0);
        match (overflow, I256::from_sign_mag(neg != (rhs < 0), mag)) {
            (false, Some(v)) => Decimal256::<DIGITS>(v),
            _ => panic!("attempt to multiply with overflow"),
        }
    }
}

impl<const DIGITS: u8> Mul<i32> for Decimal256<DIGITS> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: i32) -> Self {
        self * (rhs as i64)
    }
}

impl<const DIGITS: u8> Div for Decimal256<DIGITS> {
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

impl<const DIGITS: u8> Rem for Decimal256<DIGITS> {
    type Output = Self;
    /// # Panics
    /// Panics if `rhs` is zero.
    #[inline]
    fn rem(self, rhs: Self) -> Self {
        if rhs.0.is_zero() {
            panic!("attempt to calculate the remainder with a divisor of zero");
        }
        Decimal256::<DIGITS>(i256_rem(self.0, rhs.0))
    }
}

crate::common::impl_decimal_common!(Decimal256, "Decimal256");

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;
    use core::str::FromStr;
    use std::format;
    use std::string::ToString;

    fn raw(v: i128) -> Amount256 {
        Amount256::from_bits(I256::from_i128(v))
    }

    #[test]
    fn test_i256_basics() {
        let one = I256::from_i128(1);
        let minus_one = I256::from_i128(-1);
        assert!(minus_one.is_negative());
        assert!(!one.is_negative());
        assert_eq!(one.wrapping_neg(), minus_one);
        assert_eq!(minus_one.wrapping_neg(), one);
        assert_eq!(I256::ZERO.wrapping_neg(), I256::ZERO);
        assert_eq!(I256::MAX.wrapping_neg(), I256::MIN);

        // Carry across the 128-bit boundary.
        let max_lo = I256 {
            lo: u128::MAX,
            hi: 0,
        };
        let sum = max_lo.checked_add(one).unwrap();
        assert_eq!(sum, I256 { lo: 0, hi: 1 });
        assert_eq!(sum.checked_sub(one).unwrap(), max_lo);

        // Overflow detection at the very top.
        assert_eq!(I256::MAX.checked_add(one), None);
        assert_eq!(I256::MIN.checked_sub(I256::from_i128(2)), None);
        assert!(I256::MAX.checked_add(minus_one).is_some());

        // Ordering.
        assert!(minus_one < I256::ZERO);
        assert!(I256::MIN < minus_one);
        assert!(one < I256::MAX);
        assert!(
            I256 { lo: 0, hi: 1 }
                > I256 {
                    lo: u128::MAX,
                    hi: 0
                }
        );

        // Sign/magnitude round trip.
        for v in [0i128, 1, -1, i128::MAX, -i128::MAX, 12345, -98765] {
            let x = I256::from_i128(v);
            let (neg, mag) = x.to_sign_mag();
            assert_eq!(I256::from_sign_mag(neg, mag), Some(x));
        }
        let (neg, mag) = I256::MAX.to_sign_mag();
        assert!(!neg);
        assert_eq!(I256::from_sign_mag(neg, mag), Some(I256::MAX));
        assert_eq!(I256::from_sign_mag(false, [0, 0, 0, 1 << 63]), None);

        // Byte round trip.
        let x = I256::from_i128(-123456789);
        assert_eq!(I256::from_le_bytes(x.to_le_bytes()), x);

        assert_eq!(format!("{:?}", I256::from_i128(-42)), "-42");
    }

    #[test]
    fn test_basic_ops() {
        let a = Amount256::from(2);
        let b = Amount256::from_f64(0.5).unwrap();
        assert_eq!(a, raw(20000));
        assert_eq!(b, raw(5000));
        assert_eq!(a + b, raw(25000));
        assert_eq!(a - b, raw(15000));
        assert_eq!(a * b, raw(10000));
        assert_eq!(a / b, raw(40000));
        assert_eq!(-a, raw(-20000));
        assert_eq!(a % b, raw(0));
        assert_eq!(raw(25000) % raw(20000), raw(5000));
        assert_eq!(raw(-25000) % raw(20000), raw(-5000));
    }

    #[test]
    fn test_mul_div_match_decimal64_vectors() {
        assert_eq!(raw(10001) * raw(10001), raw(10002));
        assert_eq!(raw(11004) * raw(10015), raw(11021));
        assert_eq!(raw(11004) * raw(-10015), raw(-11021));
        assert_eq!(raw(10000) / raw(110000), raw(909));
        assert_eq!(raw(10000) / raw(130000), raw(769));
        assert_eq!(raw(10000) / raw(-130000), raw(-769));
        assert_eq!(raw(-10000) / raw(130000), raw(-769));
        assert_eq!(raw(-10000) / raw(-130000), raw(769));
        assert_eq!(raw(10000) / raw(180000), raw(556));
        assert_eq!(raw(-10000) / raw(-180000), raw(556));
    }

    #[test]
    fn test_beyond_128_bit_range() {
        // 10^36 as an Amount256: overflows Amount64 and pushes Amount128.
        let big = Amount256::from(10i128.pow(36));
        assert_eq!((big * big).to_string(), "1".to_string() + &"0".repeat(72));
        // Knuth-path division: divisor uses all four limbs.
        let q = (big * big) / big;
        assert_eq!(q, big);
        let r = (big * big + Amount256::from(7)) % (big * big);
        assert_eq!(r, Amount256::from(7));
        // 1 / 3 at scale 4 still rounds correctly.
        assert_eq!(Amount256::from(1) / Amount256::from(3), raw(3333));
        // MAX round-trips through Display/FromStr.
        let s = Amount256::MAX.to_string();
        assert_eq!(Amount256::from_str(&s).unwrap(), Amount256::MAX);
        let s = Amount256::MIN.to_string();
        assert_eq!(Amount256::from_str(&s).unwrap(), Amount256::MIN);

        // Wide multiplication result verified against an independently
        // computed value: (2^126 / 10^4) * (2^126 / 10^4) needs > 128 bits.
        let a = raw(i128::MAX / 2 + 1); // 2^126, value 2^126 / 10^4
        let prod = a * a;
        // value = 2^252 / 10^8, mantissa rounded HalfUp at 4 digits.
        assert_eq!(prod / a, a);
        assert_eq!(
            prod.to_string(),
            "72370055773322622139731865630429942408293740416025352524660990004945.706"
        );
    }

    #[test]
    fn test_checked_math() {
        assert_eq!(Amount256::MAX.checked_add(raw(1)), None);
        // Like the i64-backed type, checked math permits the asymmetric
        // two's-complement minimum (MIN - 1); one step further overflows.
        assert!(Amount256::MIN.checked_sub(raw(1)).is_some());
        assert_eq!(Amount256::MIN.checked_sub(raw(2)), None);
        assert_eq!(Amount256::MAX.checked_mul(Amount256::from(2)), None);
        assert_eq!(Amount256::from(2).checked_div(Amount256::ZERO), None);
        assert_eq!(
            Amount256::from(6).checked_div(Amount256::from(2)),
            Some(Amount256::from(3))
        );
        assert_eq!(
            Amount256::MAX.checked_div(raw(5000)), // MAX / 0.5 overflows
            None
        );
    }

    #[test]
    fn test_rounding() {
        let a = Amount256::from_f64(1.5).unwrap();
        assert_eq!(a.round_to(Rounding::HalfUp), Amount256::from(2));
        assert_eq!(a.round_to(Rounding::HalfDown), Amount256::from(1));
        assert_eq!(a.round_to(Rounding::HalfEven), Amount256::from(2));
        assert_eq!(a.round_to(Rounding::Down), Amount256::from(1));
        assert_eq!(a.round_to(Rounding::Up), Amount256::from(2));
        let b = Amount256::from_f64(-1.5).unwrap();
        assert_eq!(b.round_to(Rounding::HalfUp), Amount256::from(-2));
        assert_eq!(b.round_to(Rounding::Down), Amount256::from(-2));
        assert_eq!(b.round_to(Rounding::Up), Amount256::from(-1));

        assert_eq!(
            Amount256::from_f64(-3.7).unwrap().trunc(),
            Amount256::from(-3)
        );
        assert_eq!(
            Amount256::from_f64(-3.7).unwrap().floor(),
            Amount256::from(-4)
        );
        assert_eq!(Amount256::from_f64(3.2).unwrap().ceil(), Amount256::from(4));
        assert_eq!(
            Amount256::from_f64(-3.5).unwrap().round(),
            Amount256::from(-4)
        );
        assert_eq!(Amount256::from_f64(3.25).unwrap().fract(), raw(2500));

        // mul_rounded ties.
        let h = raw(5000); // 0.5
        let x = raw(10001);
        assert_eq!(h.mul_rounded(x, Rounding::HalfUp), raw(5001));
        assert_eq!(h.mul_rounded(x, Rounding::HalfDown), raw(5000));
        assert_eq!(h.mul_rounded(x, Rounding::HalfEven), raw(5000));
        assert_eq!((-h).mul_rounded(x, Rounding::Down), raw(-5001));
        assert_eq!((-h).mul_rounded(x, Rounding::Up), raw(-5000));

        let one = Amount256::from(1);
        let three = Amount256::from(3);
        assert_eq!(one.div_rounded(three, Rounding::Down), raw(3333));
        assert_eq!(one.div_rounded(three, Rounding::Up), raw(3334));
        assert_eq!((-one).div_rounded(three, Rounding::Down), raw(-3334));
        assert_eq!((-one).div_rounded(three, Rounding::Up), raw(-3333));
    }

    #[test]
    fn test_from_str_and_display() {
        assert_eq!(Amount256::from_str("1.0001").unwrap(), raw(10001));
        assert_eq!(Amount256::from_str("-1.0001").unwrap(), raw(-10001));
        assert_eq!(Amount256::from_str("1.00005").unwrap(), raw(10001));
        assert_eq!(Amount256::from_str(""), Err(AmountErrorKind::Empty));
        assert_eq!(Amount256::from_str("x"), Err(AmountErrorKind::InvalidDigit));
        // A 70-digit value round-trips exactly.
        let s = "-1234567890123456789012345678901234567890123456789012345678901234567890.5";
        let v = Amount256::from_str(s).unwrap();
        assert_eq!(v.to_string(), s);
        // Too large.
        let huge = "9".repeat(80);
        assert_eq!(Amount256::from_str(&huge), Err(AmountErrorKind::Overflow));

        assert_eq!(&format!("{}", raw(10000)), "1");
        assert_eq!(&format!("{:+}", raw(10000)), "+1");
        assert_eq!(&format!("{:4.2}", raw(10000)), "1.00");
        assert_eq!(&format!("{}", raw(1)), "0.0001");
        assert_eq!(&format!("{}", raw(-10001)), "-1.0001");
        assert_eq!(&format!("{}", Amount256::ZERO), "0");
        assert_eq!(&format!("{:.2}", raw(10050)), "1.01");
    }

    #[test]
    fn test_conversions() {
        assert_eq!(Amount256::from(2.5f64), raw(25000));
        assert_eq!(Amount256::from(-2.5f32), raw(-25000));
        assert_eq!(Amount256::from(3i32), raw(30000));
        assert_eq!(
            Amount256::from_f64(f64::MAX),
            Err(AmountErrorKind::Overflow)
        );
        assert_eq!(Amount256::from(f64::MAX), Amount256::MAX);
        assert_eq!(Amount256::from(f64::MIN), Amount256::MIN);

        // Wide but f64-exact value: 2^140.
        let x = 1.3937965749081639e42f64;
        let v = Amount256::from_f64(x).unwrap();
        assert!((v.to_f64() - x).abs() / x < 1e-15);

        // to_i128 truncation and saturation.
        assert_eq!(Amount256::from_f64(-3.7).unwrap().to_i128(), -3);
        let big = Amount256::from(10i128.pow(36));
        assert_eq!((big * big).to_i128(), i128::MAX);
        assert_eq!((-(big * big)).to_i128(), i128::MIN);

        // i128 conversions are exact over the whole i128 range.
        let v = Amount256::from(i128::MAX);
        assert_eq!(v.to_i128(), i128::MAX);
        assert_eq!(v.to_string(), i128::MAX.to_string());

        // Bytes round trip.
        let v = Amount256::from(-42);
        assert_eq!(Amount256::from_le_bytes(v.to_le_bytes()), v);
    }

    #[test]
    fn test_misc() {
        assert_eq!(Amount256::from(10).signum(), Amount256::ONE);
        assert_eq!(Amount256::from(-10).signum(), Amount256::MINUS_ONE);
        assert_eq!(Amount256::from(0).signum(), Amount256::ZERO);
        assert!(Amount256::from(-10).is_negative());
        assert!(Amount256::from(10).is_positive());
        assert!(!Amount256::ZERO.is_positive());
        assert_eq!(Amount256::from(-3).abs(), Amount256::from(3));
        assert_eq!(Amount256::from(2).recip(), raw(5000));
        assert_eq!(Amount256::from(2).powi(100).to_string(), {
            // 2^100
            "1267650600228229401496703205376"
        });
        assert_eq!(
            Amount256::from(5).clamp(Amount256::ZERO, Amount256::ONE),
            Amount256::ONE
        );
        assert!(Amount256::from(1) == 1i64);
        assert!(Amount256::from(2) > 1i64);
        let v: Amount256 = [1, 2, 3].iter().map(|&x| Amount256::from(x)).sum();
        assert_eq!(v, Amount256::from(6));
        let p: Amount256 = [2, 3, 4].iter().map(|&x| Amount256::from(x)).product();
        assert_eq!(p, Amount256::from(24));
        assert_eq!(Amount256::from(7) * 3i64, Amount256::from(21));
        assert_eq!(Amount256::from(7) * -3i32, Amount256::from(-21));
        assert_eq!(Amount256::from(7) + 3i64, Amount256::from(10));
        assert_eq!(Amount256::from(7) - 3i64, Amount256::from(4));
        assert_eq!(
            Rate256::from_f64(1.12345678).unwrap().to_string(),
            "1.12345678"
        );
    }

    #[test]
    fn test_const_eval() {
        const A: Amount256 = Amount256::from_str_const(
            "123456789012345678901234567890123456789012345678901234.5678",
        );
        const B: Amount256 = A.mul_rounded(Amount256::from_str_const("1.0001"), Rounding::HalfUp);
        const C: Amount256 = A.round_to(Rounding::HalfEven);
        const D: Amount256 = A.trunc();
        const E: Amount256 = Amount256::from_str_const("2").powi(100);
        const F: Option<Amount256> = A.checked_mul(A); // overflows -> None

        let a = Amount256::from_str("123456789012345678901234567890123456789012345678901234.5678")
            .unwrap();
        assert_eq!(A, a);
        assert_eq!(
            B,
            a.mul_rounded(Amount256::from_str("1.0001").unwrap(), Rounding::HalfUp)
        );
        assert_eq!(C, a.round_to(Rounding::HalfEven));
        assert_eq!(D, a.trunc());
        assert_eq!(E.to_string(), "1267650600228229401496703205376");
        assert_eq!(F, None);
    }

    /// Differential test: for values within the i128 range, Decimal256 must
    /// agree with Decimal128 on every operation and rounding mode. Operands
    /// go well past 64 bits so the Knuth division path is exercised.
    #[test]
    fn test_differential_vs_decimal128() {
        use crate::Decimal128;
        use crate::Rounding::*;

        let mut state = 0xDA3E39CB94B95BDBu64;
        let mut next = move || {
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            state.wrapping_mul(0x2545F4914F6CDD1D)
        };

        for i in 0..20000 {
            // Alternate magnitudes: up to ~2^60 (products stay well inside
            // i128) and small ones, so both division paths are hit.
            let shift = if i % 3 == 0 { 40 } else { 4 };
            let a = ((((next() as u128) << 64) | next() as u128) >> (68 + shift)) as i128
                * if next() & 1 == 0 { 1 } else { -1 };
            let b = ((((next() as u128) << 64) | next() as u128) >> (68 + shift)) as i128
                * if next() & 1 == 0 { 1 } else { -1 };
            let da = Decimal128::<4>(a);
            let db = Decimal128::<4>(b);
            let wa = Decimal256::<4>(I256::from_i128(a));
            let wb = Decimal256::<4>(I256::from_i128(b));
            fn to128<const D: u8>(v: Decimal256<D>) -> i128 {
                let (neg, mag) = v.0.to_sign_mag();
                assert_eq!(mag[2] | mag[3], 0);
                let m = ((mag[0] as u128) | ((mag[1] as u128) << 64)) as i128;
                if neg { -m } else { m }
            }

            assert_eq!((da + db).0, to128(wa + wb));
            assert_eq!((da - db).0, to128(wa - wb));
            assert_eq!((da * db).0, to128(wa * wb), "mul {a} * {b}");
            assert_eq!(format!("{da}"), format!("{wa}"));
            assert_eq!(format!("{da:.2}"), format!("{wa:.2}"));

            // Cross-scale operands (an 8-digit rate) for mul_rounded.
            let rb = Decimal128::<8>(b);
            let wrb = Decimal256::<8>(I256::from_i128(b));
            for mode in [HalfUp, HalfDown, HalfEven, Down, Up] {
                assert_eq!(
                    da.mul_rounded(db, mode).0,
                    to128(wa.mul_rounded(wb, mode)),
                    "mul_rounded {a} * {b}"
                );
                assert_eq!(
                    da.mul_rounded(rb, mode).0,
                    to128(wa.mul_rounded(wrb, mode)),
                    "mul_rounded cross-scale {a} * {b}"
                );
                assert_eq!(
                    da.round_to(mode).0,
                    to128(wa.round_to(mode)),
                    "round_to {a}"
                );
                if b != 0 {
                    assert_eq!(
                        da.div_rounded(db, mode).0,
                        to128(wa.div_rounded(wb, mode)),
                        "div_rounded {a} / {b}"
                    );
                    assert_eq!(
                        da.div_rounded_to::<8>(db, mode).0,
                        to128(wa.div_rounded_to::<8>(wb, mode)),
                        "div_rounded_to::<8> {a} / {b}"
                    );
                    if let Ok(n) = i64::try_from(b) {
                        assert_eq!(
                            da.div_int_rounded(n, mode).0,
                            to128(wa.div_int_rounded(n, mode)),
                            "div_int_rounded {a} / {n}"
                        );
                    }
                }
            }
            if b != 0 {
                assert_eq!((da / db).0, to128(wa / wb), "div {a} / {b}");
                assert_eq!((da % db).0, to128(wa % wb), "rem {a} % {b}");
            }
        }
    }
}
