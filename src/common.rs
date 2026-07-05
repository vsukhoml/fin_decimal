//! Trait implementations shared verbatim by the three decimal backings.
//!
//! [`impl_decimal_common!`] stamps out, for each type, every impl whose body
//! is identical across `Decimal`, `Decimal128` and `Decimal256`. A type only
//! has to supply two hooks:
//!
//! * `sign_mag4(self) -> (bool, [u64; 4])` — sign and 4-limb magnitude
//!   (narrow backings zero-extend), feeding the shared limb formatter;
//! * `from_str_rounded(&str, Rounding) -> Result<Self, AmountErrorKind>`.
//!
//! Arithmetic stays out of this macro on purpose: it lives in the width-
//! generic cores in [`limbs`](crate::limbs), and the operators' per-type
//! semantics (overflow behavior of `+`/`-`) remain visible in each module.

macro_rules! impl_decimal_common {
    ($Ty:ident, $name:literal) => {
        impl<const DIGITS: u8> core::str::FromStr for $Ty<DIGITS> {
            type Err = crate::AmountErrorKind;
            /// Converts a string in base 10, rounding fractional digits beyond
            /// the scale with [`Rounding::HalfUp`](crate::Rounding).
            ///
            /// Accepts `Sign? ( Digit+ | Digit+ '.' Digit* | Digit* '.' Digit+ )`;
            /// leading/trailing whitespace or other symbols are an error.
            fn from_str(src: &str) -> Result<Self, Self::Err> {
                Self::from_str_rounded(src, crate::Rounding::HalfUp)
            }
        }

        impl<const DIGITS: u8> From<&str> for $Ty<DIGITS> {
            /// # Panics
            /// Panics if the string is not a valid in-range decimal.
            fn from(src: &str) -> Self {
                <Self as core::str::FromStr>::from_str(src).unwrap()
            }
        }

        impl<const DIGITS: u8> core::ops::AddAssign for $Ty<DIGITS> {
            #[inline]
            fn add_assign(&mut self, rhs: Self) {
                *self = *self + rhs;
            }
        }

        impl<const DIGITS: u8> core::ops::SubAssign for $Ty<DIGITS> {
            #[inline]
            fn sub_assign(&mut self, rhs: Self) {
                *self = *self - rhs;
            }
        }

        impl<const DIGITS: u8> core::ops::MulAssign for $Ty<DIGITS> {
            #[inline]
            fn mul_assign(&mut self, rhs: Self) {
                *self = *self * rhs;
            }
        }

        impl<const DIGITS: u8> core::ops::DivAssign for $Ty<DIGITS> {
            #[inline]
            fn div_assign(&mut self, rhs: Self) {
                *self = *self / rhs;
            }
        }

        impl<const DIGITS: u8> core::ops::RemAssign for $Ty<DIGITS> {
            #[inline]
            fn rem_assign(&mut self, rhs: Self) {
                *self = *self % rhs;
            }
        }

        impl<const DIGITS: u8> core::iter::Sum for $Ty<DIGITS> {
            fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
                iter.fold(Self::ZERO, |a, b| a + b)
            }
        }

        impl<'a, const DIGITS: u8> core::iter::Sum<&'a $Ty<DIGITS>> for $Ty<DIGITS> {
            fn sum<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
                iter.fold(Self::ZERO, |a, b| a + *b)
            }
        }

        impl<const DIGITS: u8> core::iter::Product for $Ty<DIGITS> {
            fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
                iter.fold(Self::ONE, |a, b| a * b)
            }
        }

        impl<'a, const DIGITS: u8> core::iter::Product<&'a $Ty<DIGITS>> for $Ty<DIGITS> {
            fn product<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
                iter.fold(Self::ONE, |a, b| a * *b)
            }
        }

        impl<const DIGITS: u8> core::fmt::Display for $Ty<DIGITS> {
            fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
                // Largest value is 78 digits + point + sign; explicit
                // precision may pad further within the buffer.
                let mut buf = [0u8; 128];
                let (neg, mag) = self.sign_mag4();
                match crate::limbs::str_mag(
                    &mag,
                    neg,
                    DIGITS as usize,
                    f.precision(),
                    crate::AmountSign::None,
                    &mut buf,
                ) {
                    Some(s) => f.pad_integral(!neg, "", s),
                    _ => f.write_str("Amount::ERROR"),
                }
            }
        }

        #[cfg(feature = "serde")]
        impl<const DIGITS: u8> serde::Serialize for $Ty<DIGITS> {
            /// Serializes as a decimal string to survive transports that
            /// would round-trip numbers through 64-bit floats.
            fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
            where
                S: serde::Serializer,
            {
                serializer.collect_str(self)
            }
        }

        #[cfg(feature = "serde")]
        impl<'de, const DIGITS: u8> serde::Deserialize<'de> for $Ty<DIGITS> {
            fn deserialize<Deser>(deserializer: Deser) -> Result<Self, Deser::Error>
            where
                Deser: serde::Deserializer<'de>,
            {
                struct DecimalVisitor<const D: u8>;

                impl<'de, const D: u8> serde::de::Visitor<'de> for DecimalVisitor<D> {
                    type Value = $Ty<D>;

                    fn expecting(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
                        f.write_str("a string representation of a decimal number")
                    }

                    fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
                    where
                        E: serde::de::Error,
                    {
                        core::str::FromStr::from_str(value).map_err(serde::de::Error::custom)
                    }
                }

                deserializer.deserialize_str(DecimalVisitor::<DIGITS>)
            }
        }

        #[cfg(feature = "ufmt")]
        impl<const DIGITS: u8> ufmt::uDisplay for $Ty<DIGITS> {
            fn fmt<W>(&self, f: &mut ufmt::Formatter<'_, W>) -> Result<(), W::Error>
            where
                W: ufmt::uWrite + ?Sized,
            {
                let mut buf = [0u8; 128];
                let (neg, mag) = self.sign_mag4();
                match crate::limbs::str_mag(
                    &mag,
                    neg,
                    DIGITS as usize,
                    None,
                    crate::AmountSign::Negative,
                    &mut buf,
                ) {
                    Some(s) => f.write_str(s),
                    None => f.write_str("Amount::ERROR"),
                }
            }
        }

        #[cfg(feature = "ufmt")]
        impl<const DIGITS: u8> ufmt::uDebug for $Ty<DIGITS> {
            fn fmt<W>(&self, f: &mut ufmt::Formatter<'_, W>) -> Result<(), W::Error>
            where
                W: ufmt::uWrite + ?Sized,
            {
                // Prints the raw backing integer, like the derived Debug.
                let mut buf = [0u8; 96];
                let (neg, mag) = self.sign_mag4();
                f.write_str(concat!($name, "("))?;
                match crate::limbs::str_mag(
                    &mag,
                    neg,
                    0,
                    None,
                    crate::AmountSign::Negative,
                    &mut buf,
                ) {
                    Some(s) => f.write_str(s)?,
                    None => f.write_str("ERROR")?,
                }
                f.write_str(")")
            }
        }
    };
}

pub(crate) use impl_decimal_common;
