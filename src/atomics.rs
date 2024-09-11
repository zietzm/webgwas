use std::sync::atomic::{AtomicU32, Ordering};

pub struct AtomicF32 {
    storage: AtomicU32,
}

impl AtomicF32 {
    pub fn new(value: f32) -> Self {
        let as_u32 = value.to_bits();
        Self {
            storage: AtomicU32::new(as_u32),
        }
    }
    pub fn store(&self, value: f32) {
        let as_u32 = value.to_bits();
        self.storage.store(as_u32, Ordering::Relaxed);
    }
    pub fn load(&self, order: Ordering) -> f32 {
        let as_u32 = self.storage.load(order);
        f32::from_bits(as_u32)
    }
    pub fn add(&self, value: f32) {
        let as_f32 = self.load(Ordering::Relaxed);
        let value = as_f32 + value;
        self.store(value);
    }
}
