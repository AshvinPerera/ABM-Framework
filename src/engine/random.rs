use std::cell::Cell;
use std::thread_local;


thread_local! {static TL_RNG: Cell<u64> = Cell::new(0x9E37_79B9_7F4A_7C15);}

#[inline]
pub fn tl_rand_u64() -> u64 {
    TL_RNG.with(|c| {
        let mut x = c.get();
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        c.set(x);
        x.wrapping_mul(0x2545F4914F6CDD1D)
    })
}