//! Performance measurements across operand and divisor sizes.
//!
//! Self-contained (no external bench harness, in keeping with the crate's
//! minimal-dependency policy). Run with:
//!
//! ```text
//! cargo bench
//! cargo bench --features asm
//! ```
//!
//! Cases are grouped by type and sized by the number of 64-bit limbs the
//! operand magnitudes occupy, since that is what selects the internal code
//! path (two-limb re-scale fast path, word division, Knuth algorithm D).

use fin_decimal::{Amount64, Amount128, Amount256, I256, Rounding};
use std::fmt::Write as _;
use std::hint::black_box;
use std::time::Instant;

/// xorshift64* pseudo-random generator: deterministic, no dependencies.
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
    /// A value with exactly `bits` significant bits (top bit set).
    fn bits(&mut self, bits: u32) -> u128 {
        assert!(bits >= 1 && bits <= 127);
        let top = 1u128 << (bits - 1);
        let mask = top - 1;
        top | (((self.next() as u128) << 64 | self.next() as u128) & mask)
    }
    /// An I256 magnitude with exactly `bits` significant bits.
    fn bits256(&mut self, bits: u32) -> I256 {
        assert!(bits >= 1 && bits <= 254);
        let mut bytes = [0u8; 32];
        for b in bytes.iter_mut().take((bits as usize).div_ceil(8)) {
            *b = self.next() as u8;
        }
        let mut v = I256::from_le_bytes(bytes);
        // clear everything at and above `bits`, then set the top bit
        let keep = |x: I256, bits: u32| {
            let mut le = x.to_le_bytes();
            for (i, b) in le.iter_mut().enumerate() {
                let bit0 = 8 * i as u32;
                if bit0 >= bits {
                    *b = 0;
                } else if bit0 + 8 > bits {
                    *b &= (1u16 << (bits - bit0)) as u8 - 1;
                }
            }
            le[(bits as usize - 1) / 8] |= 1 << ((bits - 1) % 8);
            I256::from_le_bytes(le)
        };
        v = keep(v, bits);
        v
    }
}

const PAIRS: usize = 256;
const N: usize = 400_000;

fn bench<R>(name: &str, mut op: impl FnMut(usize) -> R) {
    for i in 0..(N / 8) {
        black_box(op(i));
    }
    let t = Instant::now();
    for i in 0..N {
        black_box(op(i));
    }
    let ns = t.elapsed().as_nanos() as f64 / N as f64;
    println!("  {name:<50}{ns:>9.2} ns/op");
}

fn pairs64(rng: &mut Rng, bits_a: u32, bits_b: u32) -> Vec<(Amount64, Amount64)> {
    (0..PAIRS)
        .map(|_| {
            (
                Amount64::from_bits(rng.bits(bits_a) as u64),
                Amount64::from_bits(rng.bits(bits_b) as u64),
            )
        })
        .collect()
}

fn pairs128(rng: &mut Rng, bits_a: u32, bits_b: u32) -> Vec<(Amount128, Amount128)> {
    (0..PAIRS)
        .map(|_| {
            (
                Amount128::from_bits(rng.bits(bits_a) as i128),
                Amount128::from_bits(rng.bits(bits_b) as i128),
            )
        })
        .collect()
}

fn pairs256(rng: &mut Rng, bits_a: u32, bits_b: u32) -> Vec<(Amount256, Amount256)> {
    (0..PAIRS)
        .map(|_| {
            (
                Amount256::from_bits(rng.bits256(bits_a)),
                Amount256::from_bits(rng.bits256(bits_b)),
            )
        })
        .collect()
}

/// Fixed-capacity `fmt::Write` sink so Display benches measure formatting,
/// not the allocator.
struct Sink {
    buf: [u8; 128],
    len: usize,
}
impl std::fmt::Write for Sink {
    fn write_str(&mut self, s: &str) -> std::fmt::Result {
        self.buf[self.len..self.len + s.len()].copy_from_slice(s.as_bytes());
        self.len += s.len();
        Ok(())
    }
}

fn main() {
    let mut rng = Rng(0x243F6A8885A308D3);

    println!("fin_decimal performance ({} iterations/case)", N);
    #[cfg(feature = "asm")]
    println!("features: asm");
    #[cfg(not(feature = "asm"))]
    println!("features: (default)");

    // ---------------- Amount64 ----------------
    println!("\nAmount64 (i64 backing, scale 10^4):");
    {
        // typical money: hundreds to millions of units
        let money = pairs64(&mut rng, 30, 30);
        // large: near the safe multiplication bound (raw ~2^36; the product
        // 2^72 / 10^4 must stay below i64::MAX)
        let large = pairs64(&mut rng, 36, 36);
        bench("add", |i| {
            let (a, b) = money[i % PAIRS];
            a + b
        });
        bench("mul  1w x 1w (money)", |i| {
            let (a, b) = money[i % PAIRS];
            a * b
        });
        bench("mul  1w x 1w (large)", |i| {
            let (a, b) = large[i % PAIRS];
            a * b
        });
        bench("div  by ~unit divisor", |i| {
            let (a, _) = large[i % PAIRS];
            a / Amount64::from_bits(10007)
        });
        bench("div  large / large", |i| {
            let (a, b) = large[i % PAIRS];
            a / b
        });
        bench("round_to HalfEven", |i| {
            let (a, _) = large[i % PAIRS];
            a.round_to(Rounding::HalfEven)
        });
    }

    // ---------------- Amount128 ----------------
    println!("\nAmount128 (i128 backing, scale 10^4):");
    {
        let m1x1 = pairs128(&mut rng, 40, 40); // both one limb
        let m2x1 = pairs128(&mut rng, 80, 40); // two limbs x one limb
        let m2x2 = pairs128(&mut rng, 69, 69); // both two limbs (max safe)
        let d_big = pairs128(&mut rng, 90, 60); // divisor one large limb
        let d_2w = pairs128(&mut rng, 90, 100); // divisor two limbs -> Knuth
        bench("add", |i| {
            let (a, b) = m1x1[i % PAIRS];
            a + b
        });
        bench("mul  1w x 1w (money)", |i| {
            let (a, b) = m1x1[i % PAIRS];
            a * b
        });
        bench("mul  2w x 1w", |i| {
            let (a, b) = m2x1[i % PAIRS];
            a * b
        });
        bench("mul  2w x 2w", |i| {
            let (a, b) = m2x2[i % PAIRS];
            a * b
        });
        bench("div  by ~unit divisor (1w)", |i| {
            let (a, _) = m2x1[i % PAIRS];
            a / Amount128::from_bits(10007)
        });
        bench("div  by large 1w divisor", |i| {
            let (a, b) = d_big[i % PAIRS];
            a / b
        });
        bench("div  by 2w divisor (Knuth)", |i| {
            let (a, b) = d_2w[i % PAIRS];
            a / b
        });
        bench("rem  2w % 1w", |i| {
            let (a, b) = d_big[i % PAIRS];
            a % b
        });
        bench("rem  2w % 2w", |i| {
            let (a, b) = d_2w[i % PAIRS];
            a % b
        });
        bench("round_to HalfEven", |i| {
            let (a, _) = m2x1[i % PAIRS];
            a.round_to(Rounding::HalfEven)
        });
    }

    // ---------------- Amount256 ----------------
    println!("\nAmount256 (I256 backing, scale 10^4):");
    {
        let m1x1 = pairs256(&mut rng, 40, 40);
        let m2x2 = pairs256(&mut rng, 100, 100);
        let m4x1 = pairs256(&mut rng, 220, 30);
        let d1 = pairs256(&mut rng, 250, 40); // 1-limb divisor
        let d2 = pairs256(&mut rng, 250, 100); // 2-limb divisor
        let d3 = pairs256(&mut rng, 250, 170); // 3-limb divisor
        let d4 = pairs256(&mut rng, 250, 230); // 4-limb divisor
        bench("add", |i| {
            let (a, b) = m1x1[i % PAIRS];
            a + b
        });
        bench("mul  1w x 1w (money)", |i| {
            let (a, b) = m1x1[i % PAIRS];
            a * b
        });
        bench("mul  2w x 2w", |i| {
            let (a, b) = m2x2[i % PAIRS];
            a * b
        });
        bench("mul  4w x 1w", |i| {
            let (a, b) = m4x1[i % PAIRS];
            a * b
        });
        bench("div  by ~unit divisor (1w)", |i| {
            let (a, _) = m1x1[i % PAIRS];
            a / Amount256::from_bits(I256::from_i128(10007))
        });
        bench("div  4w / 1w", |i| {
            let (a, b) = d1[i % PAIRS];
            a / b
        });
        bench("div  4w / 2w (Knuth)", |i| {
            let (a, b) = d2[i % PAIRS];
            a / b
        });
        bench("div  4w / 3w (Knuth)", |i| {
            let (a, b) = d3[i % PAIRS];
            a / b
        });
        bench("div  4w / 4w (Knuth)", |i| {
            let (a, b) = d4[i % PAIRS];
            a / b
        });
        bench("round_to HalfEven", |i| {
            let (a, _) = m4x1[i % PAIRS];
            a.round_to(Rounding::HalfEven)
        });
    }

    // ---------------- string conversions ----------------
    println!("\nString conversions:");
    {
        use std::str::FromStr;
        let mut sink = Sink {
            buf: [0u8; 128],
            len: 0,
        };
        let v64 = Amount64::from_str("1234567.8901").unwrap();
        let v128 = Amount128::from_str("123456789012345678901234567890.1234").unwrap();
        let s256 = "12345678901234567890123456789012345678901234567890123456789012345678.9012";
        let v256 = Amount256::from_str(s256).unwrap();
        bench("parse   Amount64  \"1234567.8901\"", |_| {
            Amount64::from_str(black_box("1234567.8901"))
        });
        bench("parse   Amount128 (34 digits)", |_| {
            Amount128::from_str(black_box("123456789012345678901234567890.1234"))
        });
        bench("parse   Amount256 (72 digits)", |_| {
            Amount256::from_str(black_box(s256))
        });
        bench("display Amount64", |_| {
            sink.len = 0;
            write!(sink, "{}", black_box(v64)).unwrap();
            sink.len
        });
        bench("str_i64 (formatter core, no fmt machinery)", |_| {
            let mut buf = [0u8; 32];
            fin_decimal::str_i64(
                black_box(12345678901),
                4,
                None,
                fin_decimal::AmountSign::Negative,
                &mut buf,
            )
            .unwrap()
            .len()
        });
        bench("display Amount128 (34 digits)", |_| {
            sink.len = 0;
            write!(sink, "{}", black_box(v128)).unwrap();
            sink.len
        });
        bench("display Amount256 (72 digits)", |_| {
            sink.len = 0;
            write!(sink, "{}", black_box(v256)).unwrap();
            sink.len
        });
    }
}
