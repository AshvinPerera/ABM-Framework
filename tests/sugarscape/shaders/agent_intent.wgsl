struct Params {
    entity_len: u32,
    _p0: u32,
    _p1: u32,
    _p2: u32,
};

@group(0) @binding(0) var<storage, read> vision : array<i32>;
@group(0) @binding(1) var<storage, read> pos : array<vec2<i32>>;
@group(0) @binding(2) var<storage, read> alive : array<u32>;
@group(0) @binding(3) var<uniform> params : Params;

@group(1) @binding(0) var<storage, read_write> sugar : array<f32>;
@group(1) @binding(1) var<storage, read> capacity : array<f32>;
@group(1) @binding(2) var<storage, read_write> occupancy : array<u32>;
@group(1) @binding(3) var<storage, read_write> agent_target : array<u32>;

const GRID_W : i32 = 200;
const GRID_H : i32 = 200;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let i = gid.x;
    if (i >= params.entity_len) { return; }
    if (alive[i] == 0u) { return; }

    let p = pos[i];
    let v = max(vision[i], 0);

    if (p.x < 0 || p.y < 0 || p.x >= GRID_W || p.y >= GRID_H) {
        agent_target[i] = 0u;
        return;
    }

    var best_idx : i32 = p.y * GRID_W + p.x;
    var best_val : f32 = sugar[best_idx];

    for (var dx = -v; dx <= v; dx = dx + 1) {
        for (var dy = -v; dy <= v; dy = dy + 1) {
            let nx = p.x + dx;
            let ny = p.y + dy;
            if (nx < 0 || ny < 0 || nx >= GRID_W || ny >= GRID_H) { continue; }

            let idx = ny * GRID_W + nx;
            let s = sugar[idx];
            if (s > best_val) {
                best_val = s;
                best_idx = idx;
            }
        }
    }

    agent_target[i] = u32(best_idx);
}
