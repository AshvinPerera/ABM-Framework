#![cfg(test)]
#![allow(dead_code)]

use std::sync::{Arc, Mutex};

use abm_framework::engine::{
    systems::{System, SystemBackend, AccessSets},
    component::component_id_of,
    manager::ECSReference,
    error::ECSResult,
};

use crate::sugarscape::components::*;


/// Deterministic RNG

#[inline]
fn rng_next_u32(state: &mut u64) -> u32 {
    let mut x = *state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *state = x;
    ((x.wrapping_mul(0x2545F4914F6CDD1D)) >> 32) as u32
}

#[inline]
pub fn rng_range(state: &mut u64, n: u32) -> u32 {
    if n <= 1 {
        0
    } else {
        rng_next_u32(state) % n
    }
}

/// Sugarscape terrain

#[inline]
pub fn sugar_capacity_hills(x: i32, y: i32, w: i32, h: i32) -> f32 {
    let (cx1, cy1) = (w / 4, h / 4);
    let (cx2, cy2) = (3 * w / 4, 3 * h / 4);

    let d1 = (x - cx1).abs() + (y - cy1).abs();
    let d2 = (x - cx2).abs() + (y - cy2).abs();
    let d = d1.min(d2) as f32;

    (10.0 - 0.2 * d).max(1.0)
}

/// CPU Grid

#[derive(Clone, Copy)]
struct Cell {
    current: f32,
    capacity: f32,
    occupied: bool,
}

pub struct Grid {
    pub w: i32,
    pub h: i32,
    cells: Vec<Cell>,
}

impl Grid {
    pub fn new(w: i32, h: i32) -> Self {
        let mut cells = Vec::with_capacity((w * h) as usize);

        for y in 0..h {
            for x in 0..w {
                let cap = sugar_capacity_hills(x, y, w, h);
                cells.push(Cell {
                    current: cap,
                    capacity: cap,
                    occupied: false,
                });
            }
        }

        Self { w, h, cells }
    }

    #[inline]
    fn idx(&self, x: i32, y: i32) -> usize {
        (y * self.w + x) as usize
    }

    #[inline]
    fn in_bounds(&self, x: i32, y: i32) -> bool {
        x >= 0 && x < self.w && y >= 0 && y < self.h
    }

    pub fn clear_occupancy(&mut self) {
        for c in &mut self.cells {
            c.occupied = false;
        }
    }

    pub fn set_occupant(&mut self, x: i32, y: i32) {
        let i = self.idx(x, y);
        self.cells[i].occupied = true;
    }

    pub fn is_free(&self, x: i32, y: i32) -> bool {
        !self.cells[self.idx(x, y)].occupied
    }

    pub fn sugar_at(&self, x: i32, y: i32) -> f32 {
        self.cells[self.idx(x, y)].current
    }

    pub fn harvest(&mut self, x: i32, y: i32) -> f32 {
        let i = self.idx(x, y);
        let s = self.cells[i].current;
        self.cells[i].current = 0.0;
        s
    }

    pub fn regrow(&mut self, rate: f32) {
        for c in &mut self.cells {
            c.current = (c.current + rate).min(c.capacity);
        }
    }
}

/// Sugar regrowth system (CPU)

pub struct SugarRegrowthSystem {
    pub grid: Arc<Mutex<Grid>>,
    pub rate: f32,
}

impl System for SugarRegrowthSystem {
    fn id(&self) -> u16 { 1 }

    fn backend(&self) -> SystemBackend {
        SystemBackend::CPU
    }

    fn access(&self) -> AccessSets {
        AccessSets::default()
    }

    fn run(&self, _: ECSReference<'_>) -> ECSResult<()> {
        let mut grid = self.grid.lock().unwrap();
        grid.regrow(self.rate);
        Ok(())
    }
}

/// Move + Harvest system (CPU)
/// -----------------------------------------------------------
pub struct MoveAndHarvestSystem {
    pub grid: Arc<Mutex<Grid>>,
}

impl System for MoveAndHarvestSystem {
    fn id(&self) -> u16 { 2 }

    fn backend(&self) -> SystemBackend {
        SystemBackend::CPU
    }

    fn access(&self) -> AccessSets {
        let mut a = AccessSets::default();

        a.read.set(component_id_of::<Vision>().unwrap());

        a.write.set(component_id_of::<Position>().unwrap());
        a.write.set(component_id_of::<Sugar>().unwrap());
        a.write.set(component_id_of::<RNG>().unwrap());
        a.write.set(component_id_of::<Alive>().unwrap());

        a
    }

    fn run(&self, ecs: ECSReference<'_>) -> ECSResult<()> {
        // ---------------- Clear occupancy ----------------
        {
            let mut grid = self.grid.lock().unwrap();
            grid.clear_occupancy();
        }

        // ---------------- Mark occupancy ----------------
        {
            let grid = self.grid.clone();
            let q = ecs.query()?
                .read::<Position>()?
                .read::<Alive>()?
                .build()?;

            ecs.for_each_read2::<Position, Alive>(q, move |pos, alive| {
                if alive.0 == 0 {
                    return;
                }

                let mut g = grid.lock().unwrap();
                if g.in_bounds(pos.x, pos.y) {
                    g.set_occupant(pos.x, pos.y);
                }
            })?;
        }

        // ---------------- Move + harvest ----------------
        let grid = self.grid.clone();

        let q = ecs.query()?
            .read::<Vision>()?
            .write::<Position>()?
            .write::<Sugar>()?
            .write::<RNG>()?
            .write::<Alive>()?
            .build()?;

        ecs.for_each_abstraction(q, move |reads, writes| unsafe {
            let vision =
                abm_framework::engine::storage::cast_slice::<Vision>(reads[0].as_ptr(), reads[0].len());

            let pos =
                abm_framework::engine::storage::cast_slice_mut::<Position>(writes[0].as_mut_ptr(), writes[0].len());

            let sugar =
                abm_framework::engine::storage::cast_slice_mut::<Sugar>(writes[1].as_mut_ptr(), writes[1].len());

            let rng =
                abm_framework::engine::storage::cast_slice_mut::<RNG>(writes[2].as_mut_ptr(), writes[2].len());

            let alive =
                abm_framework::engine::storage::cast_slice_mut::<Alive>(writes[3].as_mut_ptr(), writes[3].len());

            let dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)];

            for i in 0..vision.len() {
                if alive[i].0 == 0 {
                    continue;
                }

                let (x0, y0) = (pos[i].x, pos[i].y);
                let mut g = grid.lock().unwrap();

                if !g.in_bounds(x0, y0) {
                    continue;
                }

                let v = vision[i].0.max(1).min(50);
                let mut best = -1.0;
                let mut best_xy = (x0, y0);

                for (dx, dy) in dirs {
                    for s in 1..=v {
                        let nx = x0 + dx * s;
                        let ny = y0 + dy * s;

                        if !g.in_bounds(nx, ny) || !g.is_free(nx, ny) {
                            break;
                        }

                        let sc = g.sugar_at(nx, ny);
                        if sc > best {
                            best = sc;
                            best_xy = (nx, ny);
                        }
                    }
                }

                pos[i].x = best_xy.0;
                pos[i].y = best_xy.1;

                sugar[i].0 += g.harvest(best_xy.0, best_xy.1);
                rng_next_u32(&mut rng[i].state);
            }
        })?;

        Ok(())
    }
}
