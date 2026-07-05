//!
//! A Fixed-point decimal implementation written in Rust suitable
//! for a wide range of financial calculations that require significant
//! integral and fractional digits to follow decimal arithmetic rounding.
//!
//! The binary representation consists of a 64 bit integer number,
//! multiplied by power of 10 - default 10000 for 'Amount64' type.
//!
//! Such implementation results in highly efficient addition and subtraction
//! which are implemented as native operations on i64. These operations also
//! don't result in any rounding errors.
//!
//! Multiplication and division are also relatively efficient and implemented
//! via operations on i128 type internally. On some platforms (like x86-64) where
//! 64 x 64 bit multiplication with 128 bit result and division of 128 bit integer
//! by 64 bit are implemented as native instructions, performance penalty compared
//! to regular i64 division and multiplication is negligible.
//!
//! While 4 fractional decimal digits handles most of cases for accounting and tax
//! computations, in some cases like exchange rates higher precision is desirable.
//! To address this, a sibling type 'Rate' is introduced with 8 fractional digits.
//!
//! ## Wider backings: 128-bit and 256-bit
//!
//! When the ~±922 trillion range of `Amount64` is not enough, the same decimal
//! semantics are available on wider backing integers:
//!
//! * [`Amount128`] / [`Rate128`] ([`Decimal128`]): backed by `i128`, range about
//!   ±1.7 * 10^34 at 4 digits.
//! * [`Amount256`] / [`Rate256`] ([`Decimal256`]): backed by a 256-bit integer
//!   ([`I256`]), range about ±5.8 * 10^72 at 4 digits.
//!
//! Addition and subtraction remain native integer operations (one or two
//! add/adc instructions). Multiplication builds the double-width intermediate
//! from 64-bit limbs and re-scales it by the compile-time constant 10^DIGITS
//! using Möller–Granlund reciprocal multiplication with a compile-time
//! reciprocal — no division instructions and no compiler builtins on that
//! path, on any architecture (`scripts/check_asm.sh` verifies the generated
//! code). Division by another decimal (a runtime divisor) uses a
//! 128-bit-by-64-bit division primitive — the native `div` instruction on
//! `x86_64` with the `asm` feature, or a portable long division on 32-bit
//! half-digits — and Knuth's algorithm D over the significant limbs for
//! divisors wider than one limb.
//!
//! Key operations (`checked_mul`, `mul_rounded`, `round_to`, `trunc`, `powi`,
//! `from_str_rounded`, `from_str_const`, ...) are `const fn`, so rates, fees
//! and whole derived constants can be evaluated at compile time:
//!
//! ```rust
//! use fin_decimal::{Amount64, Rounding};
//! const PRICE: Amount64 = Amount64::from_str_const("19.99");
//! const RATE: Amount64 = Amount64::from_str_const("0.0825");
//! const TAX: Amount64 = PRICE.mul_rounded(RATE, Rounding::HalfEven);
//! ```
//!
//! The `*` and `/` operators of every backing panic on overflow regardless of
//! build profile (they share one arithmetic core); use
//! `checked_mul`/`checked_div` to get `None` instead.
//!
//! ## Usage
//!
//! The stable version of rust requires you to create a Decimal number
//! using one of its convenience methods.
//!
//! ```rust
//! use fin_decimal::Amount64;
//! use core::str::FromStr;
//!
//! // Using an integer number.
//! let from_int = Amount64::from(3); // 3.0000
//!
//! // Using a floating point number.
//! let from_f64 = Amount64::from(2.02f64); // 2.0200
//!
//! // From a string representation
//! let from_string = Amount64::from_str("2.02").unwrap(); // 2.0200
//!
//! // Using the `Into` trait
//! let my_int : Amount64 = 3i32.into();
//! ```
//!

//#![cfg_attr(feature = "no_std", no_std)]
//#![cfg_attr(feature = "asm", feature(llvm_asm))]
#![crate_name = "fin_decimal"]
#![crate_type = "lib"]
#![no_std]
#![deny(unconditional_recursion)]
#![warn(missing_docs)]
#![warn(clippy::all)]
//#![feature(llvm_asm)]
//#![feature(maybe_uninit_slice)]
//#![feature(const_fn)]
//#![feature(const_fn_union)]
//#![feature(rustc_attrs)]
//#![feature(half_open_range_patterns)]
//#![feature(exclusive_range_pattern)]
//#![feature(const_fn_floating_point_arithmetic)]
//#![feature(asm)]

use core::cmp::*;
use core::default::*;
use core::f64;
use core::fmt::*;
use core::hash::Hash;
use core::marker::*;
use core::ops::*;
use core::option::Option;
use core::result::Result;

use core::*;

mod common;
mod decimal128;
mod decimal256;
mod limbs;

pub use decimal128::{Amount128, Decimal128, Rate128};
pub use decimal256::{Amount256, Decimal256, I256, Rate256};

/// Enum to store the various types of errors that can cause parsing an integer to fail.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AmountErrorKind {
    /// Value being parsed doesn't have any digits (could have sign and point though)
    ///
    /// Among other causes, this variant will be constructed when parsing an empty string.
    Empty,
    /// Contains an invalid digit.
    ///
    /// Among other causes, this variant will be constructed when parsing a string that
    /// contains a letter.
    InvalidDigit,
    /// Value is too large to store in target integer type.
    Overflow,
    /// Value cannot be represented exactly at the target decimal scale.
    ///
    /// Constructed when a value has more fractional digits than the target type
    /// can hold, so converting it would silently drop precision (e.g. building a
    /// 4-digit `Amount64` from `(mantissa: 1, exponent: -5)`).
    Inexact,
}

impl AmountErrorKind {
    /// Returns the kind of the error.
    pub fn kind(&self) -> &AmountErrorKind {
        self
    }
    #[doc(hidden)]
    pub fn __description(&self) -> &str {
        match self {
            AmountErrorKind::Empty => "cannot parse integer from empty string",
            AmountErrorKind::InvalidDigit => "invalid symbol found in string",
            AmountErrorKind::Overflow => "number too large to fit in target type",
            AmountErrorKind::Inexact => "value cannot be represented exactly at the target scale",
        }
    }
}

impl fmt::Display for AmountErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        core::fmt::Display::fmt(&self.__description(), f)
    }
}

/// Splits an i64 into a sign and a one-limb magnitude for the shared
/// sign-magnitude cores in [`limbs`].
#[inline]
const fn i64_sign_mag(v: i64) -> (bool, [u64; 1]) {
    (v < 0, [v.unsigned_abs()])
}

/// Rebuilds an i64 from a sign and a one-limb magnitude with the top bit
/// clear (guaranteed by the shared cores' symmetric-range check).
#[inline]
const fn i64_from_sign_mag(neg: bool, mag: [u64; 1]) -> i64 {
    let v = mag[0] as i64;
    if neg { v.wrapping_neg() } else { v }
}

#[inline]
// Compute pow of 10 for values 0..19 (which fits in i64 range)
// Returns 1 if pow is negative and 0 if result will overflow
const fn ipow10(pow: i64) -> i64 {
    const P10: [i64; 18] = [
        10,
        100,
        1000,
        10000,
        100000,
        1000000,
        10000000,
        100000000,
        1000000000,
        10000000000,
        100000000000,
        1000000000000,
        10000000000000,
        100000000000000,
        1000000000000000,
        10000000000000000,
        100000000000000000,
        1000000000000000000,
    ];
    match (pow <= 0, pow > (P10.len() as i64)) {
        (true, _) => 1,
        (_, true) => 0, // overflow
        (_, _) => P10[pow as usize - 1],
    }
}

#[inline]
// Compute 10^pow as i128 for non-negative `pow`.
// Returns `None` if `pow` is negative or the result overflows i128.
const fn ipow10_i128(pow: i64) -> Option<i128> {
    if pow < 0 {
        return None;
    }
    let mut result: i128 = 1;
    let mut i = 0i64;
    while i < pow {
        result = match result.checked_mul(10) {
            Some(v) => v,
            None => return None,
        };
        i += 1;
    }
    Some(result)
}

#[test]
fn test_ipow10() {
    assert_eq!(ipow10(0), 1);
    assert_eq!(ipow10(-1), 1);
    assert_eq!(ipow10(19), 0); // check overflow

    assert_eq!(ipow10(1), 10);
    assert_eq!(ipow10(4), 10000);

    let mut p: i64 = 1;
    for x in 0..18i64 {
        assert_eq!(ipow10(x), p);
        p *= 10;
    }
}
/// Defines how to display the sign of the parsed number.
pub enum AmountSign {
    /// Omit the sign entirely.
    None,
    /// Display only the negative sign.
    Negative,
    /// Always display the sign (+ or -).
    Always,
}

/// Converts an i64 to a fixed-point string representation, optionally padded.
///
/// Thin wrapper over the shared limb formatter: same trailing-zero trimming
/// and half-away-from-zero precision rounding, no `unsafe`.
pub fn str_i64(
    num: i64,
    frac_digit: usize,
    precision: Option<usize>,
    sign: AmountSign,
    buf: &mut [u8],
) -> Option<&str> {
    limbs::str_mag(
        &[num.unsigned_abs()],
        num < 0,
        frac_digit,
        precision,
        sign,
        buf,
    )
}

#[test]
fn test_istr() {
    let mut buf = [0u8; 3 * mem::size_of::<i64>()];

    assert_eq!(
        str_i64(10000, 4, None, AmountSign::Negative, &mut buf),
        Some("1")
    );
    assert_eq!(
        str_i64(10001, 4, None, AmountSign::Negative, &mut buf),
        Some("1.0001")
    );
    assert_eq!(
        str_i64(10010, 4, None, AmountSign::Negative, &mut buf),
        Some("1.001")
    );
    assert_eq!(
        str_i64(10100, 4, None, AmountSign::Negative, &mut buf),
        Some("1.01")
    );
    assert_eq!(
        str_i64(11000, 4, None, AmountSign::Negative, &mut buf),
        Some("1.1")
    );
    assert_eq!(
        str_i64(101000000, 8, None, AmountSign::Negative, &mut buf),
        Some("1.01")
    );

    assert_eq!(
        str_i64(10000, 4, Some(4), AmountSign::Negative, &mut buf),
        Some("1.0000")
    );

    assert_eq!(
        str_i64(10000, 4, Some(5), AmountSign::Negative, &mut buf),
        Some("1.00000")
    );

    assert_eq!(
        str_i64(10001, 4, Some(5), AmountSign::Negative, &mut buf),
        Some("1.00010")
    );

    assert_eq!(
        str_i64(10000, 4, Some(3), AmountSign::Negative, &mut buf),
        Some("1.000")
    );
    assert_eq!(
        str_i64(10000, 4, Some(2), AmountSign::Negative, &mut buf),
        Some("1.00")
    );
    assert_eq!(
        str_i64(10100, 4, Some(2), AmountSign::Negative, &mut buf),
        Some("1.01")
    );
    assert_eq!(
        str_i64(10050, 4, Some(2), AmountSign::Negative, &mut buf),
        Some("1.01")
    );
    assert_eq!(
        str_i64(10050, 4, Some(1), AmountSign::Negative, &mut buf),
        Some("1.0")
    );
    assert_eq!(
        str_i64(-10050, 4, Some(1), AmountSign::Negative, &mut buf),
        Some("-1.0")
    );
    assert_eq!(
        str_i64(-10050, 4, Some(1), AmountSign::None, &mut buf),
        Some("1.0")
    );
    assert_eq!(
        str_i64(-10050, 4, Some(1), AmountSign::Always, &mut buf),
        Some("-1.0")
    );
    assert_eq!(
        str_i64(10050, 4, Some(1), AmountSign::Always, &mut buf),
        Some("+1.0")
    );

    let mut small_buf = [0u8; 5];
    assert_eq!(
        str_i64(10050, 4, Some(3), AmountSign::Negative, &mut small_buf),
        Some("1.005")
    );
    assert_eq!(
        str_i64(10050, 4, Some(4), AmountSign::Negative, &mut small_buf),
        None
    );
    assert_eq!(
        str_i64(100500, 4, Some(3), AmountSign::Negative, &mut small_buf),
        None
    );
}

/// Converts a string in base 10 to a fixed-point scaled value.
/// Can be used with non-default scale to handle higher precision
/// exchange rates and other scenarios with longer fractional parts.
///
/// Fractional digits beyond `scale` are dropped with `Rounding::HalfUp`. Use
/// [`parse_decimal_i64_rounded`] to choose a different rounding mode.
pub fn parse_decimal_i64(src: &str, scale: u8) -> Result<i64, AmountErrorKind> {
    parse_decimal_i64_rounded(src, scale, Rounding::HalfUp)
}

/// Converts a string in base 10 to a fixed-point scaled value, rounding any
/// fractional digits beyond `scale` with the given [`Rounding`] mode.
///
/// Inputs that carry more precision than `scale` can hold are accepted (never an
/// error) and rounded; trailing zeros are dropped exactly. The rounding is
/// "correct" in the IEEE sense: it inspects the first dropped digit together with
/// a sticky flag (whether any further digit is non-zero) and, for `HalfEven`, the
/// parity of the retained value.
pub const fn parse_decimal_i64_rounded(
    src: &str,
    scale: u8,
    mode: Rounding,
) -> Result<i64, AmountErrorKind> {
    // Delegates to the shared limb parser at one-limb width; only the
    // symmetric i64 range check lives here.
    let (neg, mag) = match limbs::parse_decimal_mag_rounded::<1>(src, scale, mode) {
        Ok(v) => v,
        Err(e) => return Err(e),
    };
    if mag[0] > i64::MAX as u64 {
        return Err(AmountErrorKind::Overflow);
    }
    Ok(i64_from_sign_mag(neg, mag))
}

/// `Rounding` represents the different strategies that can be used.
///
/// `Rounding::HalfEven` - Rounds toward the nearest even number, e.g. 5.5 -> 6, 4.5 -> 4
/// `Rounding::HalfUp` - Rounds up if the value >= 5, otherwise rounds down, e.g. 6.5 -> 7 (default),
/// `Rounding::HalfDown` - Rounds down if the value =< 5, otherwise rounds up, e.g.
/// 4.5 -> 4, 4.51 -> 5,  1.4999 -> 1
/// `Rounding::Down` - Always round down.
/// `Rounding::Up` - Always round up.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Rounding {
    /// Rounds toward the nearest even number, e.g. 5.5 -> 6, 4.5 -> 4
    HalfEven,
    /// Rounds up if the value >= 5, otherwise rounds down, e.g. 6.5 -> 7 (default)
    HalfUp,
    /// Rounds down if the value <= 5, otherwise rounds up, e.g. 4.5 -> 4, 4.51 -> 5
    HalfDown,
    /// Always rounds down.
    Down,
    /// Always rounds up.
    Up,
}

/// Amount64 type implements decimal fixed-point arithmetic for financial computations.
/// It is implemented to be as efficient as possible with most common add/sub operations
/// to be native binary add/sub.
/// Actual decimal processing is needed for multiplication and division where rounding
/// should follow specific rules.
/// Number of decimal points is chosen to be 4 - this seems to be enough for most use cases
/// except for exchange rates where sometimes up to 8 decimal digits is required
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Decimal<const DIGITS: u8>(pub i64);

/// Amount64 is a 64-bit integer based decimal type with 4 decimal digits precision.
pub type Amount64 = Decimal<4>;
/// Rate64 is a 64-bit integer based decimal type with 8 decimal digits precision.
pub type Rate64 = Decimal<8>;

impl<const DIGITS: u8> Decimal<DIGITS> {
    /// The decimal scale: the number of fractional digits.
    ///
    /// A stable name for the scale that is independent of the const-generic
    /// `DIGITS` parameter. It survives the eventual move off the const-generic
    /// form (when `Decimal` becomes a trait or a wider concrete type, `DIGITS`
    /// disappears but `SCALE` stays).
    pub const SCALE: i32 = DIGITS as i32;

    /// The multiplier used to scale values up. Equal to 10^DIGITS.
    pub const SCALE_INT: i64 = ipow10(DIGITS as i64);

    /// The largest value that can be represented by this type.
    pub const MAX: Self = Decimal::<DIGITS>(i64::MAX);
    /// The smallest value that can be represented by this type.
    pub const MIN: Self = Decimal::<DIGITS>(-i64::MAX); // make MIN symmetric

    /// Constant equal to '1'
    pub const ONE: Self = Decimal::<DIGITS>(Self::SCALE_INT);
    /// Constant equal to '-1'
    pub const MINUS_ONE: Self = Decimal::<DIGITS>(-Self::SCALE_INT);
    /// Constant equal to '0'
    pub const ZERO: Self = Decimal::<DIGITS>(0);

    /// The smallest integer value that can be represented by this type.
    pub const INT_MIN: i64 = (i64::MIN + 1) / Self::SCALE_INT;
    /// The largest integer value that can be represented by this type.
    pub const INT_MAX: i64 = i64::MAX / Self::SCALE_INT;

    /// The smallest f64 value that can be represented.
    pub const F64_MIN: f64 = (i64::MIN + 1) as f64 / Self::SCALE_INT as f64; // -922337203685477.5807f64;
    /// The largest f64 value that can be represented.
    pub const F64_MAX: f64 = i64::MAX as f64 / Self::SCALE_INT as f64; //922337203685477.5807f64;

    /// Half of the scaling factor, used for rounding.
    pub const SCALE_INT_HALF: i64 = Self::SCALE_INT / 2;

    /// The multiplier used to scale values up, as f64.
    pub const SCALE_F64: f64 = Self::SCALE_INT as f64;

    /// Scale factor for 1/100 of unit.
    pub const SCALE_INT_100: i64 = Self::SCALE_INT / 100;
    /// Half of scale factor for 1/100 of unit.
    pub const SCALE_INT_HALF_100: i64 = Self::SCALE_INT_100 / 2;

    /// Constructs a new decimal integer with value 0.
    ///
    /// # Examples
    /// ```rust
    /// use fin_decimal::Amount64;
    /// let i = Amount64::new();
    /// ```
    #[inline]
    pub fn new() -> Self {
        Decimal::<DIGITS>(0)
    }

    /// Tries to convert a f32 to a Decimal.
    #[inline]
    pub fn from_f32(val: f32) -> Result<Self, AmountErrorKind> {
        Self::from_f64(val as f64)
    }

    /// Tries to convert a f64 to a Decimal.
    #[inline]
    pub fn from_f64(val: f64) -> Result<Self, AmountErrorKind> {
        if (Self::F64_MIN..=Self::F64_MAX).contains(&val) {
            Ok(Decimal::<DIGITS>((val * Self::SCALE_F64) as i64))
        } else {
            Err(AmountErrorKind::Overflow)
        }
    }

    /// Tries to convert an i64 to a Decimal.
    #[inline]
    pub const fn from_i64(val: i64) -> Result<Self, AmountErrorKind> {
        if (val <= Self::INT_MAX) && (val >= Self::INT_MIN) {
            Ok(Decimal::<DIGITS>(val * Self::SCALE_INT))
        } else {
            Err(AmountErrorKind::Overflow)
        }
    }

    /// Converts the Decimal back into an f64.
    #[inline]
    pub fn to_f64(self) -> f64 {
        // use division as 0.0001 doesn't have exact representation in f64
        self.0 as f64 / Self::SCALE_F64
    }

    /// Converts the Decimal back into an i64 (truncating the fractional part).
    #[inline]
    pub const fn to_i64(self) -> i64 {
        self.0 / Self::SCALE_INT
    }

    /// Returns the raw backing value widened to `i128`.
    ///
    /// This is the unscaled mantissa: the stored integer such that the value
    /// equals `mantissa * 10^(-SCALE)`. The return type is deliberately `i128`,
    /// not the current `i64` backing, so this signature stays stable when the
    /// backing integer widens (`i64` → `i128` → …).
    ///
    /// # Examples
    /// ```
    /// use fin_decimal::Amount64;
    /// let a = Amount64::from(3); // SCALE = 4
    /// assert_eq!(a.mantissa(), 30000);
    /// ```
    #[inline]
    pub const fn mantissa(self) -> i128 {
        self.0 as i128
    }

    /// Decomposes the value into `(mantissa, exponent)` such that the value
    /// equals `mantissa * 10^exponent`.
    ///
    /// The exponent is always `-SCALE`. This is the symmetric counterpart of
    /// [`from_decimal_parts`](Self::from_decimal_parts) and lets serialization
    /// codecs encode without reaching into the backing field directly.
    ///
    /// # Examples
    /// ```
    /// use fin_decimal::Amount64;
    /// let a = Amount64::from(3); // SCALE = 4
    /// assert_eq!(a.to_decimal_parts(), (30000, -4));
    /// ```
    #[inline]
    pub const fn to_decimal_parts(self) -> (i128, i32) {
        (self.0 as i128, -Self::SCALE)
    }

    /// Builds a value from `(mantissa, exponent)`, where the represented number
    /// is `mantissa * 10^exponent`.
    ///
    /// The value is rescaled to this type's fixed [`SCALE`](Self::SCALE):
    /// * If it needs to be scaled up (more fractional capacity than the input),
    ///   the mantissa is multiplied by the appropriate power of ten; an
    ///   [`Overflow`](AmountErrorKind::Overflow) is returned if the result does
    ///   not fit the backing integer.
    /// * If it needs to be scaled down (the input carries more fractional digits
    ///   than this type can hold), the conversion is **exact-or-error**: it
    ///   succeeds only when the dropped digits are all zero, otherwise an
    ///   [`Inexact`](AmountErrorKind::Inexact) is returned.
    ///
    /// This is the inverse of [`to_decimal_parts`](Self::to_decimal_parts) and
    /// the single place where the backing-int range check lives, so widening the
    /// backing only relaxes the check here and leaves callers untouched.
    ///
    /// # Examples
    /// ```
    /// use fin_decimal::{Amount64, AmountErrorKind};
    /// // 1.23 expressed as 123 * 10^-2, retargeted to SCALE = 4.
    /// assert_eq!(Amount64::from_decimal_parts(123, -2), Ok(Amount64::from(123) / Amount64::from(100)));
    /// // Round-trip of an existing value.
    /// let a = Amount64::from(7);
    /// let (m, e) = a.to_decimal_parts();
    /// assert_eq!(Amount64::from_decimal_parts(m, e), Ok(a));
    /// // More fractional digits than a 4-digit type can hold: rejected.
    /// assert_eq!(Amount64::from_decimal_parts(1, -5), Err(AmountErrorKind::Inexact));
    /// ```
    pub const fn from_decimal_parts(
        mantissa: i128,
        exponent: i32,
    ) -> Result<Self, AmountErrorKind> {
        // Zero is exactly representable at any exponent; short-circuit before
        // computing powers of ten that may overflow for extreme exponents.
        if mantissa == 0 {
            return Ok(Decimal::<DIGITS>(0));
        }
        // Decimal places to shift the mantissa by to reach this type's scale.
        // Widen to i64 so the addition cannot overflow regardless of `exponent`.
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
                // 10^(-shift) exceeds i128, yet the mantissa is non-zero here:
                // the value carries far more precision than this type can hold.
                None => return Err(AmountErrorKind::Inexact),
            };
            if mantissa % divisor != 0 {
                return Err(AmountErrorKind::Inexact);
            }
            mantissa / divisor
        };
        // Fit into the backing integer.
        if scaled < i64::MIN as i128 || scaled > i64::MAX as i128 {
            return Err(AmountErrorKind::Overflow);
        }
        Ok(Decimal::<DIGITS>(scaled as i64))
    }

    /// Builds a value from `(mantissa, exponent)`, rounding to this type's scale
    /// with the given [`Rounding`] mode when the input carries more fractional
    /// digits than can be held exactly.
    ///
    /// This is the deliberately-inexact counterpart of
    /// [`from_decimal_parts`](Self::from_decimal_parts): it never returns
    /// [`Inexact`](AmountErrorKind::Inexact). The only failure mode is
    /// [`Overflow`](AmountErrorKind::Overflow), when the value does not fit the
    /// backing integer. Scaling *up* is always exact, so the rounding mode only
    /// matters when digits are dropped.
    ///
    /// # Examples
    /// ```
    /// use fin_decimal::{Amount64, Rounding};
    /// // 1.23456 has more than 4 fractional digits; round half-up to 1.2346.
    /// assert_eq!(
    ///     Amount64::from_decimal_parts_rounded(123456, -5, Rounding::HalfUp),
    ///     Ok(Amount64::from(12346) / Amount64::from(10000)),
    /// );
    /// // Trailing zeros are dropped exactly regardless of mode.
    /// assert_eq!(
    ///     Amount64::from_decimal_parts_rounded(1_230_000, -6, Rounding::Down),
    ///     Amount64::from_decimal_parts(1_230_000, -6),
    /// );
    /// ```
    pub const fn from_decimal_parts_rounded(
        mantissa: i128,
        exponent: i32,
        mode: Rounding,
    ) -> Result<Self, AmountErrorKind> {
        if mantissa == 0 {
            return Ok(Decimal::<DIGITS>(0));
        }
        let shift = exponent as i64 + Self::SCALE as i64;
        let scaled = if shift >= 0 {
            // Scale up: always exact.
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
            // `unsigned_abs` avoids the `i128::MIN` overflow that `abs` would hit.
            let mag = mantissa.unsigned_abs();
            let divisor = match ipow10_i128(-shift) {
                Some(d) => d as u128,
                // 10^(-shift) > i128::MAX >= mag, so |value| < 0.5 ULP: rounds to
                // 0, except `Up` rounds away from zero to one ULP.
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
                    return Ok(Decimal::<DIGITS>(one));
                }
            };
            let quo = mag / divisor;
            let rem = mag % divisor;
            // The divisor is a power of ten >= 10, hence always even: `half` is an
            // exact tie threshold.
            let half = divisor / 2;
            let round_up = match mode {
                Rounding::HalfEven => rem > half || (rem == half && !quo.is_multiple_of(2)),
                Rounding::HalfUp => rem >= half,
                Rounding::HalfDown => rem > half,
                Rounding::Down => false,
                Rounding::Up => rem != 0,
            };
            let q = if round_up { quo + 1 } else { quo } as i128;
            if is_neg { -q } else { q }
        };
        if scaled < i64::MIN as i128 || scaled > i64::MAX as i128 {
            return Err(AmountErrorKind::Overflow);
        }
        Ok(Decimal::<DIGITS>(scaled as i64))
    }

    /// Parses a decimal string, rounding any fractional digits beyond this
    /// type's scale with the given [`Rounding`] mode.
    ///
    /// Like [`FromStr`], but with explicit control over how excess precision is
    /// rounded. [`FromStr`]/[`From<&str>`](From) use [`Rounding::HalfUp`].
    ///
    /// # Examples
    /// ```
    /// use fin_decimal::{Amount64, Rounding};
    /// // Five fractional digits into a 4-digit type, rounded to even.
    /// assert_eq!(
    ///     Amount64::from_str_rounded("1.23455", Rounding::HalfEven),
    ///     Ok(Amount64::from(12346) / Amount64::from(10000)),
    /// );
    /// assert_eq!(
    ///     Amount64::from_str_rounded("1.23465", Rounding::HalfEven),
    ///     Ok(Amount64::from(12346) / Amount64::from(10000)),
    /// );
    /// ```
    pub const fn from_str_rounded(src: &str, mode: Rounding) -> Result<Self, AmountErrorKind> {
        match parse_decimal_i64_rounded(src, DIGITS, mode) {
            Ok(v) => Ok(Decimal::<DIGITS>(v)),
            Err(e) => Err(e),
        }
    }

    /// Parses a decimal string with [`Rounding::HalfUp`], panicking on invalid
    /// input — intended for compile-time constants, where the panic becomes a
    /// compile error.
    ///
    /// # Examples
    /// ```
    /// use fin_decimal::Amount64;
    /// use core::str::FromStr;
    /// const PRICE: Amount64 = Amount64::from_str_const("19.99");
    /// assert_eq!(Ok(PRICE), Amount64::from_str("19.99"));
    /// ```
    pub const fn from_str_const(src: &str) -> Self {
        match Self::from_str_rounded(src, Rounding::HalfUp) {
            Ok(v) => v,
            Err(_) => panic!("invalid decimal literal"),
        }
    }

    /// Computes the absolute value of self.
    /// # Examples
    ///
    /// ```
    /// use fin_decimal::Amount64;
    /// let f = Amount64::from(3);
    /// let g = Amount64::from(-4);
    ///
    /// assert_eq!(f.abs(), 3);
    /// assert_eq!(g.abs(), 4);
    /// ```
    #[inline]
    pub const fn abs(self) -> Self {
        Decimal::<DIGITS>(self.0.abs())
    }

    /// Checked integer addition. Computes `self + rhs`, returning `None` if overflow occurred.
    #[inline]
    pub const fn checked_add(self, rhs: Self) -> Option<Self> {
        match self.0.checked_add(rhs.0) {
            Some(v) => Some(Decimal::<DIGITS>(v)),
            None => None,
        }
    }

    /// Checked integer subtraction. Computes `self - rhs`, returning `None` if overflow occurred.
    #[inline]
    pub const fn checked_sub(self, rhs: Self) -> Option<Self> {
        match self.0.checked_sub(rhs.0) {
            Some(v) => Some(Decimal::<DIGITS>(v)),
            None => None,
        }
    }

    /// Sign and 4-limb magnitude (zero-extended), for the shared
    /// formatting code in [`common`](crate::common).
    #[inline]
    pub(crate) const fn sign_mag4(self) -> (bool, [u64; 4]) {
        (self.0 < 0, [self.0.unsigned_abs(), 0, 0, 0])
    }

    /// Checked integer multiplication. Computes `self * rhs`, returning `None` if overflow occurred.
    #[inline]
    pub const fn checked_mul(self, rhs: Self) -> Option<Self> {
        let (an, am) = i64_sign_mag(self.0);
        let (bn, bm) = i64_sign_mag(rhs.0);
        match limbs::dec_mul::<DIGITS, 1, 2>(an, &am, bn, &bm, Rounding::HalfUp) {
            Some((neg, mag)) => Some(Decimal::<DIGITS>(i64_from_sign_mag(neg, mag))),
            None => None,
        }
    }

    /// Checked integer division. Computes `self / rhs`, returning `None` if `rhs == 0` or the division results in overflow.
    #[inline]
    pub fn checked_div(self, rhs: Self) -> Option<Self> {
        let (an, am) = i64_sign_mag(self.0);
        let (bn, bm) = i64_sign_mag(rhs.0);
        limbs::dec_div::<DIGITS, 1, 2>(an, &am, bn, &bm, Rounding::HalfUp)
            .map(|(neg, mag)| Decimal::<DIGITS>(i64_from_sign_mag(neg, mag)))
    }

    /// Takes the reciprocal (inverse) of a number, 1/x.
    #[inline]
    pub fn recip(self) -> Self {
        Self::ONE.div_rounded(self, Rounding::HalfUp)
    }

    /// Returns the integer part of a number.
    ///
    /// # Examples
    ///
    /// ```
    /// use fin_decimal::Amount64;
    /// let f = Amount64::from_f64(3.7_f64).unwrap();
    /// let g = Amount64::from_f64(3.0_f64).unwrap();
    /// let h = Amount64::from_f64(-3.7_f64).unwrap();
    ///
    /// assert_eq!(f.trunc(), 3.0);
    /// assert_eq!(g.trunc(), 3.0);
    /// assert_eq!(h.trunc(), -3.0);
    /// ```
    #[inline]
    pub fn trunc(self) -> Self {
        Decimal::<DIGITS>(self.0 - self.0 % Self::SCALE_INT)
    }

    /// Returns the largest integer less than or equal to a number.
    ///
    /// # Examples
    ///
    /// ```
    /// use fin_decimal::Amount64;
    /// let f = Amount64::from_f64(3.7_f64).unwrap();
    /// let g = Amount64::from_f64(3.0_f64).unwrap();
    /// let h = Amount64::from_f64(-3.7_f64).unwrap();
    ///
    /// assert_eq!(f.floor(), 3);
    /// assert_eq!(g.floor(), 3);
    /// assert_eq!(h.floor(), -4);
    /// ```
    pub fn floor(self) -> Self {
        let frac = self.0 % Self::SCALE_INT;

        if self.0 < 0 && frac != 0 {
            Decimal::<DIGITS>(self.0 - frac - Self::SCALE_INT)
        } else {
            Decimal::<DIGITS>(self.0 - frac)
        }
    }

    /// Returns the smallest integer greater than or equal to a number.
    ///
    /// # Examples
    ///
    /// ```
    /// use fin_decimal::Amount64;
    /// let f = Amount64::from_f64(3.01_f64).unwrap();
    /// let g = Amount64::from_f64(4.0_f64).unwrap();
    ///
    /// assert_eq!(f.ceil(), 4);
    /// assert_eq!(g.ceil(), 4);
    /// ```
    pub fn ceil(self) -> Self {
        let mut frac = self.0 % Self::SCALE_INT;
        if frac != 0 {
            if self.0 < 0 {
                frac += Self::SCALE_INT
            } else {
                frac -= Self::SCALE_INT
            }
            Decimal::<DIGITS>(self.0 - frac)
        } else {
            self
        }
    }

    /// Returns the nearest integer to a number. Round half-way cases away from
    /// `0.0`.
    ///
    /// # Examples
    ///
    /// ```
    /// use fin_decimal::Amount64;
    /// let f = Amount64::from(3.3_f64);
    /// let g = Amount64::from(-3.3_f64);
    ///
    /// assert_eq!(f.round(), 3);
    /// assert_eq!(g.round(), -3);
    /// ```
    pub fn round(self) -> Self {
        let mut frac = self.0 % Self::SCALE_INT;

        // check if rounding is needed
        if frac >= Self::SCALE_INT_HALF {
            frac -= Self::SCALE_INT
        } else if frac <= -Self::SCALE_INT_HALF {
            frac += Self::SCALE_INT
        }

        Decimal::<DIGITS>(self.0 - frac)
    }

    /// Rounding to 1/100th (0.01) half-way cases away from `0.0`.
    ///
    /// # Examples
    ///
    /// ```
    /// use fin_decimal::Amount64;
    /// assert_eq!(Amount64::from(3.356_f64).round100(), Amount64::from(3.36_f64));
    /// assert_eq!(Amount64::from(3.354_f64).round100(), Amount64::from(3.35_f64));
    /// ```
    pub fn round100(self) -> Self {
        let mut frac = self.0 % Self::SCALE_INT_100;

        // check if rounding is needed
        if frac >= Self::SCALE_INT_HALF_100 {
            frac -= Self::SCALE_INT_100
        } else if frac <= -Self::SCALE_INT_HALF_100 {
            frac += Self::SCALE_INT_100
        }
        Decimal::<DIGITS>(self.0 - frac)
    }

    /// Explicitly rounds the value using the specified rounding mode.
    pub const fn round_to(self, mode: Rounding) -> Self {
        let frac = self.0 % Decimal::<DIGITS>::SCALE_INT;
        if frac == 0 {
            return self;
        }

        let int_part = self.0 - frac;

        let mut add_one = false;
        let mut sub_one = false;

        match mode {
            Rounding::HalfEven => {
                let half = Decimal::<DIGITS>::SCALE_INT_HALF;
                let is_even = (int_part / Decimal::<DIGITS>::SCALE_INT) % 2 == 0;

                if frac > half {
                    add_one = true;
                } else if frac < -half {
                    sub_one = true;
                } else if frac == half {
                    if !is_even {
                        add_one = true;
                    }
                } else if frac == -half && !is_even {
                    sub_one = true;
                }
            }
            Rounding::HalfUp => {
                let half = Decimal::<DIGITS>::SCALE_INT_HALF;
                if frac >= half {
                    add_one = true;
                } else if frac <= -half {
                    sub_one = true;
                }
            }
            Rounding::HalfDown => {
                let half = Decimal::<DIGITS>::SCALE_INT_HALF;
                if frac > half {
                    add_one = true;
                } else if frac < -half {
                    sub_one = true;
                }
            }
            Rounding::Down => {
                if frac < 0 {
                    sub_one = true;
                }
            }
            Rounding::Up => {
                if frac > 0 {
                    add_one = true;
                }
            }
        }

        let mut res = int_part;
        if add_one {
            res += Decimal::<DIGITS>::SCALE_INT;
        }
        if sub_one {
            res -= Decimal::<DIGITS>::SCALE_INT;
        }
        Decimal::<DIGITS>(res)
    }

    /// Multiply by another decimal, explicitly applying the given rounding mode.
    ///
    /// Usable in const contexts, so multiplication chains can be evaluated at
    /// compile time.
    ///
    /// # Panics
    /// Panics if the result overflows.
    pub const fn mul_rounded(self, rhs: Self, mode: Rounding) -> Self {
        let (an, am) = i64_sign_mag(self.0);
        let (bn, bm) = i64_sign_mag(rhs.0);
        match limbs::dec_mul::<DIGITS, 1, 2>(an, &am, bn, &bm, mode) {
            Some((neg, mag)) => Decimal::<DIGITS>(i64_from_sign_mag(neg, mag)),
            None => panic!("attempt to multiply with overflow"),
        }
    }

    /// Divide by another decimal, explicitly applying the given rounding mode.
    ///
    /// # Panics
    /// Panics if `rhs` is zero or the result overflows.
    pub fn div_rounded(self, rhs: Self, mode: Rounding) -> Self {
        if rhs.0 == 0 {
            panic!("Can't divide by zero");
        }
        let (an, am) = i64_sign_mag(self.0);
        let (bn, bm) = i64_sign_mag(rhs.0);
        match limbs::dec_div::<DIGITS, 1, 2>(an, &am, bn, &bm, mode) {
            Some((neg, mag)) => Decimal::<DIGITS>(i64_from_sign_mag(neg, mag)),
            None => panic!("attempt to divide with overflow"),
        }
    }

    #[inline]
    /// Returns the fractional part of a number.
    ///
    /// # Examples
    ///
    /// ```
    /// use fin_decimal::Amount64;
    /// let x = Amount64::from(3.6_f64);
    /// let y = Amount64::from(-3.6_f64);
    /// let abs_difference_x = (x.fract() - 0.6).abs();
    /// let abs_difference_y = (y.fract() - (-0.6)).abs();
    ///
    /// assert!(abs_difference_x < 1e-10);
    /// assert!(abs_difference_y < 1e-10);
    /// ```
    pub const fn fract(self) -> Self {
        // fractional part would be a reminder of division by the scaler
        Decimal::<DIGITS>(self.0 % Self::SCALE_INT)
    }

    /// Returns `true` if `self` is positive and `false` if the number is zero or negative.
    /// # Examples
    /// Basic usage:
    /// ```
    /// use fin_decimal::Amount64;
    /// assert_eq!(Amount64::from(10).is_positive(), true);
    /// assert_eq!(Amount64::from(-10).is_positive(), false);
    /// ```
    #[inline]
    pub const fn is_positive(self) -> bool {
        self.0 > 0
    }

    /// Returns `true` if `self` is negative and `false` if the number is zero or positive.
    /// # Examples
    /// Basic usage:
    /// ```
    /// use fin_decimal::Amount64;
    /// assert_eq!(Amount64::from(10).is_negative(), false);
    /// assert_eq!(Amount64::from(-10).is_negative(), true);
    /// ```
    #[inline]
    pub const fn is_negative(self) -> bool {
        self.0 < 0
    }

    /// Returns a number that represents the sign of `self`.
    ///
    /// - `1.0` if the number is positive
    /// - `-1.0` if the number is negative
    /// - `0` if the number is `0`
    ///
    /// # Examples
    /// ```
    /// use fin_decimal::Amount64;
    /// assert_eq!(Amount64::from(10).signum(), Amount64::ONE);
    /// assert_eq!(Amount64::from(-10).signum(), Amount64::MINUS_ONE);
    /// assert_eq!(Amount64::from(0).signum(), Amount64::ZERO);
    /// ```
    pub fn signum(self) -> Self {
        match (self.0 < 0, self.0 > 0) {
            (true, _) => Self::MINUS_ONE,
            (false, false) => Self::ZERO,
            (_, _) => Self::ONE,
        }
    }

    /// Raw transmutation to u64.
    /// This is currently identical to transmute::<f64, u64>(self) on all platforms.
    /// See from_bits for some discussion of the portability of this operation
    /// (there are almost no issues).  Note that this function is distinct from as casting,
    /// which attempts to preserve the numeric value, and not the bitwise value.
    /// # Examples
    /// ```
    /// use fin_decimal::Amount64;
    /// assert!(Amount64::from(1.0).to_bits() != 1.0 as u64); // to_bits() is not casting!
    /// assert_eq!(Amount64::from(2.5).to_bits(), (Amount64::SCALE_INT*2 + Amount64::SCALE_INT_HALF) as u64);
    /// ```
    #[inline]
    pub const fn to_bits(self) -> u64 {
        self.0 as u64
    }

    /// Raw transmutation from u64.
    /// This is currently identical to transmute::<u64, f64>(v) on all platforms.
    /// # Examples
    /// ```
    /// use fin_decimal::Amount64;
    /// assert_eq!(Amount64::from_bits((Amount64::SCALE_INT*2 + Amount64::SCALE_INT_HALF) as u64), Amount64::from(2.5));
    /// ```
    #[inline]
    pub const fn from_bits(v: u64) -> Self {
        // SAFETY: `u64` is a plain old datatype so we can always transmute from it
        // It turns out the safety issues with sNaN were overblown! Hooray!
        Decimal::<DIGITS>(v as i64)
    }

    /// Reverses the byte order of the Amount64. Primary use case - serialization.
    /// ```
    /// use fin_decimal::Amount64;
    /// let n = Amount64::from_bits(0x1234567890123456u64);
    /// let m = n.swap_bytes();
    /// assert_eq!(m.to_bits(), 0x5634129078563412);
    /// ```
    #[inline]
    pub const fn swap_bytes(self) -> Self {
        Decimal::<DIGITS>(self.0.swap_bytes())
    }

    /// Return the memory representation of this integer as a byte array in big-endian (network) byte order.
    /// # Examples
    /// Basic usage:
    /// ```
    /// use fin_decimal::Amount64;
    /// let bytes = Amount64::from_bits(0x1234567890123456u64).to_be_bytes();
    /// assert_eq!(bytes, [0x12, 0x34, 0x56, 0x78, 0x90, 0x12, 0x34, 0x56]);
    /// ```
    #[inline]
    pub const fn to_be_bytes(self) -> [u8; mem::size_of::<i64>()] {
        self.0.to_be().to_ne_bytes()
    }

    /// Return the memory representation of this integer as a byte array in little-endian byte order.
    /// # Examples
    /// Basic usage:
    /// ```
    /// use fin_decimal::Amount64;
    /// let bytes = Amount64::from_bits(0x1234567890123456u64).to_le_bytes();
    /// assert_eq!(bytes, [0x56, 0x34, 0x12, 0x90, 0x78, 0x56, 0x34, 0x12]);
    /// ```
    #[inline]
    pub const fn to_le_bytes(self) -> [u8; mem::size_of::<i64>()] {
        self.0.to_le().to_ne_bytes()
    }

    /// Return the memory representation of this integer as a byte array in native byte order.
    /// As the target platform's native endianness is used, portable code should use to_be_bytes
    /// or to_le_bytes, as appropriate, instead.
    /// # Examples
    /// Basic usage:
    /// ```
    /// use fin_decimal::Amount64;
    /// let bytes = Amount64::from_bits(0x1234567890123456u64).to_le_bytes();
    /// assert_eq!(bytes,
    /// if cfg!(target_endian = "big") {
    ///        [0x12, 0x34, 0x56, 0x78, 0x90, 0x12, 0x34, 0x56]
    ///     } else {
    ///        [0x56, 0x34, 0x12, 0x90, 0x78, 0x56, 0x34, 0x12]
    ///     }
    /// );
    /// ```
    #[inline]
    pub fn to_ne_bytes(self) -> [u8; mem::size_of::<i64>()] {
        #[repr(C)]
        union Bytes<const DIGITS: u8> {
            val: core::mem::ManuallyDrop<Decimal<DIGITS>>,
            bytes: [u8; mem::size_of::<i64>()],
        }
        // SAFETY: integers are plain old datatypes so we can always transmute them to
        // arrays of bytes
        unsafe {
            Bytes {
                val: core::mem::ManuallyDrop::new(self),
            }
            .bytes
        }
    }

    /// Create an integer value from its memory representation as a byte array in native endianness.
    /// As the target platform's native endianness is used, portable code likely wants to use from_be_bytes
    /// or from_le_bytes, as appropriate instead.
    ///
    /// # Examples
    /// Basic usage:
    /// ```
    /// use fin_decimal::Amount64;
    /// let value = Amount64::from_ne_bytes(if cfg!(target_endian = "big") {
    ///                 [0x12, 0x34, 0x56, 0x78, 0x90, 0x12, 0x34, 0x56]
    ///             } else {
    ///                 [0x56, 0x34, 0x12, 0x90, 0x78, 0x56, 0x34, 0x12]
    ///             });
    /// assert_eq!(value,Amount64::from_bits(0x1234567890123456u64));
    /// ```
    #[inline]
    pub fn from_ne_bytes(bytes: [u8; mem::size_of::<i64>()]) -> Self {
        #[repr(C)]
        union Bytes<const DIGITS: u8> {
            val: core::mem::ManuallyDrop<Decimal<DIGITS>>,
            bytes: [u8; mem::size_of::<i64>()],
        }
        // SAFETY: integers are plain old datatypes so we can always transmute to them
        unsafe { core::mem::ManuallyDrop::into_inner(Bytes { bytes }.val) }
    }

    /// Create an integer value from its representation as a byte array in big endian.
    ///
    /// # Examples
    /// Basic usage:
    /// ```
    /// use fin_decimal::Amount64;
    /// let value = Amount64::from_be_bytes([0x12, 0x34, 0x56, 0x78, 0x90, 0x12, 0x34, 0x56]);
    /// assert_eq!(value,Amount64::from_bits(0x1234567890123456u64));
    /// ```
    #[inline]
    pub fn from_be_bytes(bytes: [u8; mem::size_of::<i64>()]) -> Self {
        Self::from_bits(u64::from_be_bytes(bytes))
    }

    /// Create an integer value from its representation as a byte array in little endian.
    ///
    /// # Examples
    /// Basic usage:
    /// ```
    /// use fin_decimal::Amount64;
    /// let value = Amount64::from_le_bytes([0x56, 0x34, 0x12, 0x90, 0x78, 0x56, 0x34, 0x12]);
    /// assert_eq!(value,Amount64::from_bits(0x1234567890123456u64));
    /// ```
    #[inline]
    pub fn from_le_bytes(bytes: [u8; mem::size_of::<i64>()]) -> Self {
        Self::from_bits(u64::from_ne_bytes(bytes))
    }

    /// Raises a number to an integer power.
    ///
    /// # Examples
    ///
    /// ```
    /// use fin_decimal::Amount64;
    /// let x = Amount64::from(2.0_f64);
    /// let abs_difference = (x.powi(2) - (x * x)).abs();
    ///
    /// assert!(abs_difference < 1e-10);
    /// ```
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

        // Deal with the final bit of the exponent separately, since
        // squaring the base afterwards is not necessary and may cause a
        // needless overflow.
        if exp == 1 {
            acc = base.mul_rounded(acc, Rounding::HalfUp);
        }

        acc
    }

    //pub const fn div_euclid(self, rhs: Self) -> Self {
    //    let q = self / rhs;
    //    if self % rhs < 0 {
    //        return if rhs > 0 { q - 1 } else { q + 1 }
    //    }
    //    q
    //}

    /// Restrict a value to a certain interval.
    /// Returns max if self is greater than max, and min if self is less than min.
    /// Otherwise this returns self.
    /// # Examples
    ///
    /// ```
    /// use fin_decimal::Amount64;
    /// assert_eq!(Amount64::from(2).clamp(Amount64::MINUS_ONE,Amount64::ONE), Amount64::ONE);
    /// assert_eq!(Amount64::from(0).clamp(Amount64::MINUS_ONE,Amount64::ONE), Amount64::ZERO);
    /// ```
    #[inline]
    pub fn clamp(self, min: Self, max: Self) -> Self {
        if self.0 < min.0 {
            Decimal::<DIGITS>(min.0)
        } else if self.0 > max.0 {
            Decimal::<DIGITS>(max.0)
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

impl<const DIGITS: u8> PartialOrd<i64> for Decimal<DIGITS> {
    #[inline]
    fn partial_cmp(&self, other: &i64) -> Option<Ordering> {
        PartialOrd::partial_cmp(&self.0, &(*other * Self::SCALE_INT))
    }
}

impl<const DIGITS: u8> PartialOrd<f64> for Decimal<DIGITS> {
    #[inline]
    fn partial_cmp(&self, other: &f64) -> Option<Ordering> {
        PartialOrd::partial_cmp(&(self.0 as f64), &(*other * Self::SCALE_F64))
    }
}

impl<const DIGITS: u8> PartialOrd<Decimal<DIGITS>> for i64 {
    #[inline]
    fn partial_cmp(&self, other: &Decimal<DIGITS>) -> Option<Ordering> {
        PartialOrd::partial_cmp(&(self * Decimal::<DIGITS>::SCALE_INT), &other.0)
    }
}

impl<const DIGITS: u8> PartialEq<i64> for Decimal<DIGITS> {
    #[inline]
    fn eq(&self, other: &i64) -> bool {
        self.0 == *other * Self::SCALE_INT
    }
}

impl<const DIGITS: u8> PartialEq<f64> for Decimal<DIGITS> {
    #[inline]
    fn eq(&self, other: &f64) -> bool {
        self.0 as f64 == *other * Self::SCALE_F64
    }
}

impl<const DIGITS: u8> PartialEq<Decimal<DIGITS>> for i64 {
    #[inline]
    fn eq(&self, other: &Decimal<DIGITS>) -> bool {
        *self * Decimal::<DIGITS>::SCALE_INT == other.0
    }
}

impl<const DIGITS: u8> PartialEq<Decimal<DIGITS>> for f64 {
    #[inline]
    fn eq(&self, other: &Decimal<DIGITS>) -> bool {
        *self * Amount64::SCALE_F64 == other.0 as f64
    }
}

impl<const DIGITS: u8> From<i32> for Decimal<DIGITS> {
    #[inline]
    fn from(item: i32) -> Self {
        Decimal::<DIGITS>(item as i64 * Self::SCALE_INT)
    }
}

impl<const DIGITS: u8> From<i64> for Decimal<DIGITS> {
    #[inline]
    fn from(item: i64) -> Self {
        Decimal::<DIGITS>(item * Self::SCALE_INT)
    }
}

impl<const DIGITS: u8> From<f64> for Decimal<DIGITS> {
    #[inline]
    fn from(item: f64) -> Self {
        if (item < Self::F64_MAX) && (item > Self::F64_MIN) {
            Decimal::<DIGITS>((item * Self::SCALE_F64) as i64)
        } else if item < Self::F64_MIN {
            Self::MIN
        } else {
            Self::MAX
        }
    }
}

impl<const DIGITS: u8> From<f32> for Decimal<DIGITS> {
    #[inline]
    fn from(item: f32) -> Self {
        Self::from(item as f64)
    }
}

impl<const DIGITS: u8> Neg for Decimal<DIGITS> {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self::Output {
        Decimal::<DIGITS>(-self.0)
    }
}

impl<const DIGITS: u8> Add for Decimal<DIGITS> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        Decimal::<DIGITS>(self.0 + rhs.0)
    }
}

impl<const DIGITS: u8> Add<i64> for Decimal<DIGITS> {
    type Output = Self;
    #[inline]
    fn add(self, other: i64) -> Self {
        Decimal::<DIGITS>(self.0 + other * Self::SCALE_INT)
    }
}

impl<const DIGITS: u8> Add<f64> for Decimal<DIGITS> {
    type Output = Self;
    #[inline]
    fn add(self, other: f64) -> Self {
        Decimal::<DIGITS>(self.0 + (other * Self::SCALE_F64) as i64)
    }
}

impl<const DIGITS: u8> Add<i32> for Decimal<DIGITS> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: i32) -> Self {
        Decimal::<DIGITS>(self.0 + (rhs as i64) * Self::SCALE_INT)
    }
}

impl<const DIGITS: u8> Add<Decimal<DIGITS>> for i64 {
    type Output = Self;
    #[allow(clippy::suspicious_arithmetic_impl)]
    #[inline]
    fn add(self, other: Decimal<DIGITS>) -> Self {
        let mut quo = other.0 / Decimal::<DIGITS>::SCALE_INT;
        let rem = other.0 % Decimal::<DIGITS>::SCALE_INT;
        if rem >= Decimal::<DIGITS>::SCALE_INT_HALF {
            quo += 1; // make sure works for negative
        } else if rem <= -Decimal::<DIGITS>::SCALE_INT_HALF {
            quo -= 1;
        }
        self + quo
    }
}

impl<const DIGITS: u8> Add<Decimal<DIGITS>> for f64 {
    type Output = Self;
    #[inline]
    fn add(self, other: Decimal<DIGITS>) -> Self {
        self + other.to_f64()
    }
}

impl<const DIGITS: u8> Sub for Decimal<DIGITS> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Decimal::<DIGITS>(self.0 - rhs.0)
    }
}

impl<const DIGITS: u8> Sub<i64> for Decimal<DIGITS> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: i64) -> Self {
        Decimal::<DIGITS>(self.0 - rhs * Self::SCALE_INT)
    }
}

impl<const DIGITS: u8> Sub<i32> for Decimal<DIGITS> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: i32) -> Self {
        Decimal::<DIGITS>(self.0 - (rhs as i64) * Self::SCALE_INT)
    }
}

impl<const DIGITS: u8> Sub<f64> for Decimal<DIGITS> {
    type Output = Self;
    #[inline]
    fn sub(self, other: f64) -> Self {
        Decimal::<DIGITS>(self.0 - (other * Self::SCALE_F64) as i64)
    }
}

impl<const DIGITS: u8> Sub<Decimal<DIGITS>> for i64 {
    type Output = Self;
    #[allow(clippy::suspicious_arithmetic_impl)]
    #[inline]
    fn sub(self, other: Decimal<DIGITS>) -> Self {
        let mut quo = other.0 / Decimal::<DIGITS>::SCALE_INT;
        let rem = other.0 % Decimal::<DIGITS>::SCALE_INT;
        if rem >= Decimal::<DIGITS>::SCALE_INT_HALF {
            quo += 1; // make sure works for negative
        } else if rem <= -Decimal::<DIGITS>::SCALE_INT_HALF {
            quo -= 1;
        }
        self - quo
    }
}

impl<const DIGITS: u8> Sub<Decimal<DIGITS>> for f64 {
    type Output = Self;
    #[inline]
    fn sub(self, other: Decimal<DIGITS>) -> Self {
        self - other.to_f64()
    }
}

impl<const DIGITS: u8> Mul for Decimal<DIGITS> {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Decimal<DIGITS>) -> Self::Output {
        self.mul_rounded(rhs, Rounding::HalfUp)
    }
}

impl<const DIGITS: u8> Mul<i64> for Decimal<DIGITS> {
    type Output = Decimal<DIGITS>;

    #[inline]
    fn mul(self, rhs: i64) -> Decimal<DIGITS> {
        Decimal::<DIGITS>(self.0 * rhs)
    }
}

impl<const DIGITS: u8> Mul<i32> for Decimal<DIGITS> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: i32) -> Self {
        Decimal::<DIGITS>(self.0 * (rhs as i64))
    }
}

impl<const DIGITS: u8> Mul<Decimal<DIGITS>> for i64 {
    type Output = i64;

    #[inline]
    fn mul(self, rhs: Decimal<DIGITS>) -> i64 {
        (Decimal::<DIGITS>(self) * rhs).0
    }
}

impl<const DIGITS: u8> Div for Decimal<DIGITS> {
    type Output = Self;

    #[inline]
    fn div(self, rhs: Self) -> Self {
        self.div_rounded(rhs, Rounding::HalfUp)
    }
}

impl<const DIGITS: u8> Rem for Decimal<DIGITS> {
    type Output = Self;

    #[inline]
    fn rem(self, rhs: Self) -> Self {
        Decimal::<DIGITS>(self.0 % rhs.0)
    }
}

crate::common::impl_decimal_common!(Decimal, "Decimal");

#[cfg(test)]
mod tests {
    extern crate std;
    use crate::Amount64;
    use crate::AmountErrorKind;
    use crate::Decimal;
    use core::str::FromStr;
    use std::format;

    #[test]
    fn test_decimal_parts_roundtrip() {
        // mantissa() / to_decimal_parts() expose the raw scaled value.
        let a = Amount64::from(3);
        assert_eq!(a.mantissa(), 30000);
        assert_eq!(a.to_decimal_parts(), (30000, -4));
        assert_eq!(Amount64::SCALE, 4);
        assert_eq!(Decimal::<8>::SCALE, 8);
        assert_eq!(Decimal::<0>::SCALE, 0);

        // Round-trip through (mantissa, exponent) for several values.
        for raw in [0i64, 1, -1, 12345, -98765, i64::MAX, -i64::MAX] {
            let d = Decimal::<4>(raw);
            let (m, e) = d.to_decimal_parts();
            assert_eq!(Amount64::from_decimal_parts(m, e), Ok(d));
        }
    }

    #[test]
    fn test_from_decimal_parts_scaling() {
        // Scale up: 1.23 == 123 * 10^-2 retargeted to 4 digits == 12300.
        assert_eq!(
            Amount64::from_decimal_parts(123, -2),
            Ok(Decimal::<4>(12300))
        );
        // Exponent equal to scale: integer scaling.
        assert_eq!(Amount64::from_decimal_parts(7, 0), Ok(Decimal::<4>(70000)));
        // Positive exponent scales further up.
        assert_eq!(
            Amount64::from_decimal_parts(5, 2),
            Ok(Decimal::<4>(5_000_000))
        );
        // Zero with any exponent is always representable.
        assert_eq!(Amount64::from_decimal_parts(0, -100), Ok(Decimal::<4>(0)));
        assert_eq!(Amount64::from_decimal_parts(0, 100), Ok(Decimal::<4>(0)));

        // Exact scale-down: surplus trailing zeros are droppable.
        // 1230000 * 10^-6 == 1.23 == 12300 * 10^-4.
        assert_eq!(
            Amount64::from_decimal_parts(1_230_000, -6),
            Ok(Decimal::<4>(12300))
        );
    }

    #[test]
    fn test_from_decimal_parts_inexact() {
        // More fractional digits than a 4-digit type can hold.
        assert_eq!(
            Amount64::from_decimal_parts(1, -5),
            Err(AmountErrorKind::Inexact)
        );
        assert_eq!(
            Amount64::from_decimal_parts(12345, -5),
            Err(AmountErrorKind::Inexact)
        );
        // Huge negative exponent with non-zero mantissa: inexact, not a panic.
        assert_eq!(
            Amount64::from_decimal_parts(1, -1000),
            Err(AmountErrorKind::Inexact)
        );

        // Trailing zeros after the representable digits are NOT inexact: the
        // dropped digits are all zero, so the value is exact.
        assert_eq!(
            Amount64::from_decimal_parts(12300, -5),
            Ok(Decimal::<4>(1230)) // 0.12300 -> 0.1230
        );
        assert_eq!(
            Amount64::from_decimal_parts(120_000, -5),
            Ok(Decimal::<4>(12000)) // 1.20000 -> 1.2000
        );
        // Many surplus zeros are still exact.
        assert_eq!(
            Amount64::from_decimal_parts(50_000_000, -10),
            Ok(Decimal::<4>(50)) // 0.0050000000 -> 0.0050
        );
    }

    #[test]
    fn test_from_decimal_parts_rounded() {
        use crate::Rounding::*;
        // 1.23456 -> 4 digits. Round digit is 6 (> half).
        for mode in [HalfEven, HalfUp, HalfDown, Up] {
            assert_eq!(
                Amount64::from_decimal_parts_rounded(123456, -5, mode),
                Ok(Decimal::<4>(12346))
            );
        }
        assert_eq!(
            Amount64::from_decimal_parts_rounded(123456, -5, Down),
            Ok(Decimal::<4>(12345))
        );

        // Exact half: 1.23455 -> last kept digit decides HalfEven/HalfDown.
        assert_eq!(
            Amount64::from_decimal_parts_rounded(123455, -5, HalfUp),
            Ok(Decimal::<4>(12346))
        );
        assert_eq!(
            Amount64::from_decimal_parts_rounded(123455, -5, HalfDown),
            Ok(Decimal::<4>(12345))
        );
        assert_eq!(
            Amount64::from_decimal_parts_rounded(123455, -5, HalfEven),
            Ok(Decimal::<4>(12346)) // 12345 is odd -> round to even 12346
        );
        assert_eq!(
            Amount64::from_decimal_parts_rounded(123465, -5, HalfEven),
            Ok(Decimal::<4>(12346)) // 12346 is even -> stays
        );

        // Negative values round away from / toward zero symmetrically.
        assert_eq!(
            Amount64::from_decimal_parts_rounded(-123456, -5, HalfUp),
            Ok(Decimal::<4>(-12346))
        );
        assert_eq!(
            Amount64::from_decimal_parts_rounded(-123456, -5, Down),
            Ok(Decimal::<4>(-12345))
        );
        assert_eq!(
            Amount64::from_decimal_parts_rounded(-123451, -5, Up),
            Ok(Decimal::<4>(-12346))
        );

        // Exact inputs are unaffected by the mode.
        assert_eq!(
            Amount64::from_decimal_parts_rounded(1_230_000, -6, Down),
            Ok(Decimal::<4>(12300))
        );

        // Magnitude below half a ULP: rounds to zero, except `Up`.
        assert_eq!(
            Amount64::from_decimal_parts_rounded(1, -1000, HalfUp),
            Ok(Decimal::<4>(0))
        );
        assert_eq!(
            Amount64::from_decimal_parts_rounded(1, -1000, Up),
            Ok(Decimal::<4>(1))
        );
        assert_eq!(
            Amount64::from_decimal_parts_rounded(-1, -1000, Up),
            Ok(Decimal::<4>(-1))
        );

        // Overflow still reported when the rounded value won't fit.
        assert_eq!(
            Amount64::from_decimal_parts_rounded(i128::MAX, 0, HalfUp),
            Err(AmountErrorKind::Overflow)
        );
    }

    #[test]
    fn test_parse_rounded() {
        use crate::Rounding::*;
        use crate::parse_decimal_i64_rounded;

        // Default parse / FromStr keep HalfUp behavior.
        assert_eq!(Amount64::from_str("1.23455"), Ok(Decimal::<4>(12346)));

        // HalfEven uses the sticky bit and parity of the kept digit.
        assert_eq!(parse_decimal_i64_rounded("1.23455", 4, HalfEven), Ok(12346));
        assert_eq!(parse_decimal_i64_rounded("1.23465", 4, HalfEven), Ok(12346));
        // 1.234551 is past the half -> always up regardless of parity.
        assert_eq!(
            parse_decimal_i64_rounded("1.234551", 4, HalfEven),
            Ok(12346)
        );
        // HalfDown drops an exact half but rounds 5-with-remainder up.
        assert_eq!(parse_decimal_i64_rounded("1.23455", 4, HalfDown), Ok(12345));
        assert_eq!(
            parse_decimal_i64_rounded("1.234551", 4, HalfDown),
            Ok(12346)
        );

        // Trailing zeros never change the value or carry into rounding.
        assert_eq!(parse_decimal_i64_rounded("1.23450000", 4, Up), Ok(12345));
        assert_eq!(parse_decimal_i64_rounded("1.2345", 4, Up), Ok(12345));
        // Any non-zero dropped digit rounds away from zero under `Up`.
        assert_eq!(parse_decimal_i64_rounded("1.23451", 4, Up), Ok(12346));
        assert_eq!(parse_decimal_i64_rounded("-1.23451", 4, Up), Ok(-12346));
        assert_eq!(parse_decimal_i64_rounded("1.23459", 4, Down), Ok(12345));

        // Rounding can carry into the integer part.
        assert_eq!(parse_decimal_i64_rounded("0.99995", 4, HalfUp), Ok(10000));
    }

    #[test]
    fn test_from_decimal_parts_overflow() {
        // Scaled value exceeds the i64 backing.
        assert_eq!(
            Amount64::from_decimal_parts(i128::MAX, 0),
            Err(AmountErrorKind::Overflow)
        );
        // Power of ten itself overflows i128.
        assert_eq!(
            Amount64::from_decimal_parts(1, 1000),
            Err(AmountErrorKind::Overflow)
        );
        // Just past the backing range.
        assert_eq!(
            Amount64::from_decimal_parts(i64::MAX as i128 + 1, -4),
            Err(AmountErrorKind::Overflow)
        );
    }

    #[test]
    fn test_add() {
        assert_eq!(
            Decimal::<4>(10000) + Decimal::<4>(20000),
            Decimal::<4>(30000)
        );
        assert_eq!(
            Decimal::<4>(10000) + Decimal::<4>(20001),
            Decimal::<4>(30001)
        );
        assert_eq!(
            Decimal::<4>(10000) + Decimal::<4>(-20000),
            Decimal::<4>(-10000)
        );
        assert_eq!(
            Decimal::<4>(9223372036854775806) + Decimal::<4>(1),
            Decimal::<4>(9223372036854775807)
        );
    }

    #[test]
    fn test_trunc() {
        assert_eq!(Decimal::<4>(10000).trunc(), Decimal::<4>(10000));
        assert_eq!(Decimal::<4>(10001).trunc(), Decimal::<4>(10000));
        assert_eq!(Decimal::<4>(19999).trunc(), Decimal::<4>(10000));
        assert_eq!(Decimal::<4>(199999).trunc(), Decimal::<4>(190000));

        assert_eq!(Decimal::<4>(-10000).trunc(), Decimal::<4>(-10000));
        assert_eq!(Decimal::<4>(-10001).trunc(), Decimal::<4>(-10000));
        assert_eq!(Decimal::<4>(-19999).trunc(), Decimal::<4>(-10000));
        assert_eq!(Decimal::<4>(-199999).trunc(), Decimal::<4>(-190000));
    }

    #[test]
    fn test_floor() {
        assert_eq!(Decimal::<4>(10000).floor(), Decimal::<4>(10000));
        assert_eq!(Decimal::<4>(10001).floor(), Decimal::<4>(10000));
        assert_eq!(Decimal::<4>(19999).floor(), Decimal::<4>(10000));
        assert_eq!(Decimal::<4>(199999).floor(), Decimal::<4>(190000));

        assert_eq!(Decimal::<4>(-10000).floor(), Decimal::<4>(-10000));
        assert_eq!(Decimal::<4>(-10001).floor(), Decimal::<4>(-20000));
        assert_eq!(Decimal::<4>(-19999).floor(), Decimal::<4>(-20000));
        assert_eq!(Decimal::<4>(-199999).floor(), Decimal::<4>(-200000));
    }

    #[test]
    fn test_round() {
        assert_eq!(Decimal::<4>(10000).round(), Decimal::<4>(10000));
        assert_eq!(Decimal::<4>(10001).round(), Decimal::<4>(10000));
        assert_eq!(Decimal::<4>(19999).round(), Decimal::<4>(20000));
        assert_eq!(Decimal::<4>(199999).round(), Decimal::<4>(200000));

        assert_eq!(Decimal::<4>(-10000).round(), Decimal::<4>(-10000));
        assert_eq!(Decimal::<4>(-10001).round(), Decimal::<4>(-10000));
        assert_eq!(Decimal::<4>(-19999).round(), Decimal::<4>(-20000));
        assert_eq!(Decimal::<4>(-199999).round(), Decimal::<4>(-200000));
    }

    #[test]
    fn test_round100() {
        assert_eq!(Decimal::<4>::SCALE_INT_HALF_100, 50);

        assert_eq!(Decimal::<4>(10000).round100(), Decimal::<4>(10000));
        assert_eq!(Decimal::<4>(10001).round100(), Decimal::<4>(10000));
        assert_eq!(Decimal::<4>(19999).round100(), Decimal::<4>(20000));
        assert_eq!(Decimal::<4>(199999).round100(), Decimal::<4>(200000));

        assert_eq!(Decimal::<4>(-10000).round100(), Decimal::<4>(-10000));
        assert_eq!(Decimal::<4>(-10001).round100(), Decimal::<4>(-10000));
        assert_eq!(Decimal::<4>(-19999).round100(), Decimal::<4>(-20000));
        assert_eq!(Decimal::<4>(-199999).round100(), Decimal::<4>(-200000));

        assert_eq!(Decimal::<4>(-10050).round100(), Decimal::<4>(-10100));
        assert_eq!(Decimal::<4>(10050).round100(), Decimal::<4>(10100));

        assert_eq!(Decimal::<4>(10049).round100(), Decimal::<4>(10000));
    }

    #[test]
    fn test_sub() {
        assert_eq!(
            Decimal::<4>(10000) - Decimal::<4>(20000),
            Decimal::<4>(-10000)
        );
        assert_eq!(
            Decimal::<4>(10000) - Decimal::<4>(20001),
            Decimal::<4>(-10001)
        );
        assert_eq!(
            Decimal::<4>(10000) - Decimal::<4>(-20000),
            Decimal::<4>(30000)
        );
        assert_eq!(
            Decimal::<4>(9223372036854775807) - Decimal::<4>(1),
            Decimal::<4>(9223372036854775806)
        );
    }

    // test operations with integers
    #[test]
    fn test_add_int() {
        assert_eq!(Decimal::<4>(10000) + 2, Decimal::<4>(30000));
        assert_eq!(Decimal::<4>(10001) + 2, Decimal::<4>(30001));
        assert_eq!(Decimal::<4>(10000) + (-2), Decimal::<4>(-10000));
        assert_eq!(
            Decimal::<4>(9223372036854765806) + 1,
            Decimal::<4>(9223372036854775806)
        );
    }

    // test multiplication of scaled
    #[test]
    fn test_mul() {
        assert_eq!(
            Decimal::<4>(10000) * Decimal::<4>(10000),
            Decimal::<4>(10000)
        );
        assert_eq!(
            Decimal::<4>(10000) * Decimal::<4>(-11111),
            Decimal::<4>(-11111)
        );
        assert_eq!(
            Decimal::<4>(10001) * Decimal::<4>(10001),
            Decimal::<4>(10002)
        );
        assert_eq!(
            Decimal::<4>(9223372036854775807) * Decimal::<4>(10000),
            Decimal::<4>(9223372036854775807)
        );
        assert_eq!(
            Decimal::<4>(11004) * Decimal::<4>(10015),
            Decimal::<4>(11021)
        );
        assert_eq!(
            Decimal::<4>(11004) * Decimal::<4>(-10015),
            Decimal::<4>(-11021)
        );
    }

    #[test]
    fn test_div() {
        assert_eq!(
            Decimal::<4>(10000) / Decimal::<4>(10000),
            Decimal::<4>(10000)
        );
        assert_eq!(
            Decimal::<4>(10000) / Decimal::<4>(20000),
            Decimal::<4>(5000)
        );
        assert_eq!(
            Decimal::<4>(10000) / Decimal::<4>(5000),
            Decimal::<4>(20000)
        );

        assert_eq!(
            Decimal::<4>(10001) / Decimal::<4>(10001),
            Decimal::<4>(10000)
        );
        assert_eq!(
            Decimal::<4>(9223372036854775807) / Decimal::<4>(10000),
            Decimal::<4>(9223372036854775807)
        );

        assert_eq!(
            Decimal::<4>(10000) / Decimal::<4>(110000),
            Decimal::<4>(909)
        );
        assert_eq!(
            Decimal::<4>(10000) / Decimal::<4>(110000),
            Decimal::<4>(909)
        );

        // test roundings to be 'decimal'
        assert_eq!(
            Decimal::<4>(10000) / Decimal::<4>(130000),
            Decimal::<4>(769)
        );
        assert_eq!(
            Decimal::<4>(10000) / Decimal::<4>(-130000),
            Decimal::<4>(-769)
        );
        assert_eq!(
            Decimal::<4>(-10000) / Decimal::<4>(-130000),
            Decimal::<4>(769)
        );
        assert_eq!(
            Decimal::<4>(-10000) / Decimal::<4>(130000),
            Decimal::<4>(-769)
        );

        assert_eq!(
            Decimal::<4>(10000) / Decimal::<4>(180000),
            Decimal::<4>(556)
        );
        assert_eq!(
            Decimal::<4>(10000) / Decimal::<4>(-180000),
            Decimal::<4>(-556)
        );
        assert_eq!(
            Decimal::<4>(-10000) / Decimal::<4>(180000),
            Decimal::<4>(-556)
        );
        assert_eq!(
            Decimal::<4>(-10000) / Decimal::<4>(-180000),
            Decimal::<4>(556)
        );
    }

    #[test]
    fn test_recip() {
        assert_eq!(Decimal::<4>(10000).recip(), Decimal::<4>(10000));
        assert_eq!(Decimal::<4>(9223372036854775807).recip(), Decimal::<4>(0));

        assert_eq!(Decimal::<4>(110000).recip(), Decimal::<4>(909));
        assert_eq!(Decimal::<4>(-110000).recip(), Decimal::<4>(-909));

        // test roundings to be 'decimal'
        assert_eq!(Decimal::<4>(130000).recip(), Decimal::<4>(769));
        assert_eq!(Decimal::<4>(-130000).recip(), Decimal::<4>(-769));

        assert_eq!(Decimal::<4>(180000).recip(), Decimal::<4>(556));
        assert_eq!(Decimal::<4>(-180000).recip(), Decimal::<4>(-556));
    }

    #[test]
    fn test_misc() {
        let a = Amount64::from_f32(1.0).unwrap();
        let b = Amount64::from_f32(0.5).unwrap();
        assert_eq!(a.0, 10000);
        assert_eq!(b.0, 5000);

        let c = a / b;
        assert_eq!(c.0, 20000);
        let d = a * b;
        assert_eq!(d.0, 5000);
        let e = b - 1i64;
        assert_eq!(e.0, -5000);
    }
    #[test]
    fn test_from_str() {
        assert_eq!(Amount64::from_str(""), Err(AmountErrorKind::Empty));
        assert_eq!(Amount64::from_str("+"), Err(AmountErrorKind::Empty));
        assert_eq!(Amount64::from_str("-"), Err(AmountErrorKind::Empty));
        assert_eq!(Amount64::from_str("-."), Err(AmountErrorKind::Empty));
        assert_eq!(Amount64::from_str("+."), Err(AmountErrorKind::Empty));
        assert_eq!(Amount64::from_str("-+"), Err(AmountErrorKind::InvalidDigit));

        assert_eq!(Amount64::from_str("1").unwrap().0, 10000);
        assert_eq!(Amount64::from_str("+1").unwrap().0, 10000);
        assert_eq!(Amount64::from_str("1.").unwrap().0, 10000);
        assert_eq!(Amount64::from_str("1.0").unwrap().0, 10000);
        assert_eq!(Amount64::from_str("+1.0").unwrap().0, 10000);
        assert_eq!(Amount64::from_str("1.00").unwrap().0, 10000);
        assert_eq!(Amount64::from_str("1.000").unwrap().0, 10000);
        assert_eq!(Amount64::from_str("1.0000").unwrap().0, 10000);
        assert_eq!(Amount64::from_str("1.00000").unwrap().0, 10000);
        assert_eq!(Amount64::from_str("01.00000").unwrap().0, 10000);
        assert_eq!(Amount64::from_str("-1").unwrap().0, -10000);
        assert_eq!(Amount64::from_str("-1.").unwrap().0, -10000);
        assert_eq!(Amount64::from_str("-1.0").unwrap().0, -10000);
        assert_eq!(Amount64::from_str("-1.00").unwrap().0, -10000);
        assert_eq!(Amount64::from_str("-1.000").unwrap().0, -10000);
        assert_eq!(Amount64::from_str("-1.0000").unwrap().0, -10000);
        assert_eq!(Amount64::from_str("-1.00000").unwrap().0, -10000);
        assert_eq!(Amount64::from_str("-01.00000").unwrap().0, -10000);
        assert_eq!(Amount64::from_str("1.5").unwrap().0, 15000);
        assert_eq!(Amount64::from_str("1.05").unwrap().0, 10500);
        assert_eq!(Amount64::from_str("1.005").unwrap().0, 10050);
        assert_eq!(Amount64::from_str("1.0005").unwrap().0, 10005);
        assert_eq!(Amount64::from_str("1.00005").unwrap().0, 10001);
        assert_eq!(Amount64::from_str("1.00004").unwrap().0, 10000);
        assert_eq!(Amount64::from_str("1.00006").unwrap().0, 10001);
        assert_eq!(
            Amount64::from_str("922337203685477.5807").unwrap().0,
            9223372036854775807
        );
        assert_eq!(
            Amount64::from_str("-922337203685477.5807").unwrap().0,
            -9223372036854775807
        );
        assert_eq!(
            Amount64::from_str("922337203685477.5808"),
            Err(AmountErrorKind::Overflow)
        );
        assert_eq!(
            Amount64::from_str("-922337203685477.5808"),
            Err(AmountErrorKind::Overflow)
        );
    }
    #[test]
    fn test_display() {
        assert_eq!(&format!("{}", Decimal::<4>(10000)), "1");
        assert_eq!(&format!("{:+}", Decimal::<4>(10000)), "+1");
        assert_eq!(&format!("{:2}", Decimal::<4>(10000)), " 1");
        assert_eq!(&format!("{:3}", Decimal::<4>(10000)), "  1");
        assert_eq!(&format!("{:4.2}", Decimal::<4>(10000)), "1.00");
        assert_eq!(&format!("{:T>4.2}", Decimal::<4>(10000)), "1.00");
        assert_eq!(&format!("{:<3}", Decimal::<4>(10000)), "1  ");
        assert_eq!(&format!("{:03}", Decimal::<4>(10000)), "001");
        assert_eq!(&format!("{:+2}", Decimal::<4>(10000)), "+1");
        assert_eq!(&format!("{:+05}", Decimal::<4>(10000)), "+0001");
        assert_eq!(&format!("{}", Decimal::<4>(1)), "0.0001");
        assert_eq!(&format!("{}", Decimal::<4>(10)), "0.001");
        assert_eq!(&format!("{}", Decimal::<4>(-10000)), "-1");
        assert_eq!(&format!("{:+}", Decimal::<4>(-10000)), "-1");
        assert_eq!(&format!("{}", Decimal::<4>(10001)), "1.0001");
        assert_eq!(&format!("{}", Decimal::<4>(-10001)), "-1.0001");
        // Regression: zero used to render as "-0" (pad_integral was handed
        // `num > 0` as the non-negative flag).
        assert_eq!(&format!("{}", Decimal::<4>(0)), "0");
        assert_eq!(&format!("{:+}", Decimal::<4>(0)), "+0"); // matches i32 behavior
    }

    #[test]
    fn test_rate64() {
        use crate::Rate64;
        let rate = Rate64::from(1.12345678);
        assert_eq!(&format!("{}", rate), "1.12345678");
        assert_eq!(rate.0, 112345678);
    }

    #[test]
    fn test_rounding_modes() {
        use crate::Rounding;
        let a = Amount64::from(1.5);
        assert_eq!(a.round_to(Rounding::HalfUp), Amount64::from(2));
        assert_eq!(a.round_to(Rounding::HalfDown), Amount64::from(1));
        assert_eq!(a.round_to(Rounding::HalfEven), Amount64::from(2));
        assert_eq!(a.round_to(Rounding::Down), Amount64::from(1));
        assert_eq!(a.round_to(Rounding::Up), Amount64::from(2));
    }

    #[test]
    fn test_checked_math() {
        let max = Amount64::MAX;
        assert_eq!(max.checked_add(Amount64::from(1)), None);
        assert_eq!(
            Amount64::from(2).checked_sub(Amount64::from(1)),
            Some(Amount64::from(1))
        );
        assert_eq!(Amount64::from(2).checked_div(Amount64::from(0)), None);
        // Regression: checked_div used to return the remainder, not the quotient.
        assert_eq!(
            Amount64::from(6).checked_div(Amount64::from(2)),
            Some(Amount64::from(3))
        );
        assert_eq!(
            Amount64::from(1).checked_div(Amount64::from(3)),
            Some(Decimal::<4>(3333))
        );
        assert_eq!(
            Amount64::from(-1).checked_div(Amount64::from(3)),
            Some(Decimal::<4>(-3333))
        );
        assert_eq!(max.checked_mul(Amount64::from(2)), None);
    }

    #[test]
    fn test_const_eval() {
        use crate::Rounding;
        // Whole expressions evaluate at compile time.
        const PRICE: Amount64 = Amount64::from_str_const("19.99");
        const RATE: Amount64 = Amount64::from_str_const("0.0825");
        const TAX: Amount64 = PRICE.mul_rounded(RATE, Rounding::HalfEven);
        const DOUBLED: Option<Amount64> = PRICE.checked_mul(Amount64::from_str_const("2"));
        const GROWTH: Amount64 = Amount64::from_str_const("1.05").powi(10);
        const ROUNDED: Amount64 = TAX.round_to(Rounding::HalfUp);

        assert_eq!(PRICE.0, 199_900);
        // Compile-time results match the runtime paths exactly.
        let price = Amount64::from_str("19.99").unwrap();
        let rate = Amount64::from_str("0.0825").unwrap();
        assert_eq!(TAX, price.mul_rounded(rate, Rounding::HalfEven));
        assert_eq!(DOUBLED, price.checked_mul(Amount64::from(2)));
        assert_eq!(GROWTH, Amount64::from_str("1.05").unwrap().powi(10));
        assert_eq!(ROUNDED, TAX.round_to(Rounding::HalfUp));
        assert_eq!(
            Amount64::from_str_rounded("1.00005", Rounding::HalfEven),
            Ok(Decimal::<4>(10000))
        );
    }

    #[test]
    fn test_div_half_ties() {
        // Exact half ties round away from zero under the default HalfUp:
        // 0.0001 / 2 = 0.00005 -> 0.0001.
        assert_eq!(Decimal::<4>(1) / Decimal::<4>(20000), Decimal::<4>(1));
        assert_eq!(Decimal::<4>(-1) / Decimal::<4>(20000), Decimal::<4>(-1));
        assert_eq!(
            Decimal::<4>(1).checked_div(Decimal::<4>(20000)),
            Some(Decimal::<4>(1))
        );
        // recip: 1 / 20000.0 = 0.00005 -> 0.0001.
        assert_eq!(Decimal::<4>(200000000).recip(), Decimal::<4>(1));

        // HalfEven with an odd divisor: rem == (d-1)/2 is *below* the half,
        // so 0.0001 / 0.0003 = 0.3333... stays 0.3333.
        use crate::Rounding;
        assert_eq!(
            Decimal::<4>(1).div_rounded(Decimal::<4>(3), Rounding::HalfEven),
            Decimal::<4>(3333)
        );
        // Even divisor exact tie obeys the parity of the quotient.
        // 0.3333 / 2 = 0.16665: quotient 1666 is even -> stays.
        assert_eq!(
            Decimal::<4>(3333).div_rounded(Decimal::<4>(20000), Rounding::HalfEven),
            Decimal::<4>(1666)
        );
        // 0.3335 / 2 = 0.16675: quotient 1667 is odd -> rounds to 1668.
        assert_eq!(
            Decimal::<4>(3335).div_rounded(Decimal::<4>(20000), Rounding::HalfEven),
            Decimal::<4>(1668)
        );
    }

    #[cfg(feature = "ufmt")]
    #[test]
    fn test_ufmt() {
        struct StringWriter(std::string::String);
        impl ufmt::uWrite for StringWriter {
            type Error = core::convert::Infallible;
            fn write_str(&mut self, s: &str) -> Result<(), Self::Error> {
                self.0.push_str(s);
                Ok(())
            }
        }

        let mut w1 = StringWriter(std::string::String::new());
        let val1 = Decimal::<4>(10000);
        ufmt::uwrite!(&mut w1, "{}", val1).unwrap();
        assert_eq!(w1.0, "1");

        let mut w2 = StringWriter(std::string::String::new());
        let val2 = Decimal::<4>(-10001);
        ufmt::uwrite!(&mut w2, "{}", val2).unwrap();
        assert_eq!(w2.0, "-1.0001");

        let mut w3 = StringWriter(std::string::String::new());
        ufmt::uwrite!(&mut w3, "{:?}", val2).unwrap();
        assert_eq!(w3.0, "Decimal(-10001)");
    }
}
