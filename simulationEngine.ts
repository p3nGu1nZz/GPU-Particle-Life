import { SimulationParams, RuleMatrix, ColorDefinition } from './types';

// WebGPU Type Polyfills
type GPUDevice = any;
type GPUCanvasContext = any;
type GPURenderPipeline = any;
type GPUComputePipeline = any;
type GPUBuffer = any;
type GPUBindGroup = any;
type GPUTexture = any;
declare var GPUBufferUsage: any;
declare var GPUTextureUsage: any;

// Utility to pack float to half-float (f16) for initial CPU upload
// Returns 16-bit integer representation
const toHalf = (val: number) => {
    const floatView = new Float32Array(1);
    const int32View = new Int32Array(floatView.buffer);
    floatView[0] = val;
    const x = int32View[0];
    const bits = (x >> 16) & 0x8000; /* Get the sign */
    let m = (x >> 12) & 0x07ff; /* Keep one extra bit for rounding */
    const e = (x >> 23) & 0xff; /* Using only the exponent */
    if (e < 103) return bits; /* Check for too small exponent (0) */
    if (e > 142) { /* Check for too large exponent (infinity) */
        return bits | 0x7c00;
    }
    if (e < 113) { /* Subnormal */
            m |= 0x0800;
            return bits | ((m >> (114 - e)) + ((m >> (113 - e)) & 1));
    }
    return bits | ((e - 112) << 10) | (m >> 1) + (m & 1);
};

// WGSL Shaders

const FADE_SHADER = `
@vertex
fn vs_main(@builtin(vertex_index) vIdx: u32) -> @builtin(position) vec4f {
    var pos = array<vec2f, 6>(
        vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(-1.0, 1.0),
        vec2f(-1.0, 1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0)
    );
    return vec4f(pos[vIdx], 0.0, 1.0);
}

@fragment
fn fs_main() -> @location(0) vec4f {
    // Black with alpha 0.15 gives ~85% retention per frame (trails)
    return vec4f(0.0, 0.0, 0.0, 0.15); 
}
`;

const COMPUTE_SHADER = `
// Packed Structures Optimization:
//
// ParticlesA (Hot Loop Buffer - Reduced to 4 bytes):
//   u32 containing:
//   - Bits 0-11 : Position X (12-bit quantized 0..4095)
//   - Bits 12-23: Position Y (12-bit quantized 0..4095)
//   - Bits 24-31: Type Index (8-bit 0..255)
//
// ParticlesB (State Buffer - Expanded to 16 bytes to preserve precision):
//   vec4u containing:
//   .x = High Precision Pos (pack2x16float) - Source of truth for integration
//   .y = Velocity (pack2x16float)
//   .z = State (Density f16, Age f16)
//   .w = Padding / Reserved

struct Params {
    width: f32,
    height: f32,
    friction: f32,
    dt: f32,
    rMax: f32,
    forceFactor: f32,
    minDist: f32,
    count: f32,
    size: f32,
    opacity: f32,
    numTypes: f32,
    time: f32,     
    growth: f32,
    mouseX: f32,
    mouseY: f32,
    mouseType: f32,
    temperature: f32, // Thermal Energy
    pad0: f32,
    pad1: f32,
    pad2: f32,
};

@group(0) @binding(0) var<storage, read_write> particlesA: array<u32>;
@group(0) @binding(1) var rulesTex: texture_2d<f32>; 
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(4) var<storage, read_write> particlesB: array<vec4u>;

const BLOCK_SIZE = 256u;
const MAX_TYPES = 64u; 

// Unpacked cache in Shared Memory (~3KB total)
// Stores fully expanded float positions and int types for fast inner-loop access
var<workgroup> tile_pos: array<vec2f, BLOCK_SIZE>;
var<workgroup> tile_type: array<u32, BLOCK_SIZE>;

// Faster hash for inner loops
fn fast_hash(seed: vec2f) -> f32 {
    return fract(sin(dot(seed, vec2f(12.9898, 78.233))) * 43758.5453);
}

fn rand2(seed: vec2f) -> vec2f {
    let t = fast_hash(seed);
    let u = fast_hash(seed + vec2f(1.0, 0.0));
    return vec2f(t, u) * 2.0 - 1.0;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3u, @builtin(local_invocation_id) local_id: vec3u) {
    let index = global_id.x;
    let particleCount = u32(params.count);

    let rMax = params.rMax;
    let rMaxSq = rMax * rMax; 
    let rMaxInv = 1.0 / rMax; 
    
    let rMin = params.minDist; 
    let safeRMin = max(0.001, rMin);
    
    let baseForceFactor = params.forceFactor;
    let growthEnabled = params.growth > 0.5;
    
    let bioSeed = params.time + f32(index) * 0.123;
    let randomVal = fast_hash(vec2f(bioSeed, f32(index)));
    
    // --- Load Self Data ---
    // We load high-precision position from B for physics stability
    // We load Type from A
    var myPos = vec2f(0.0);
    var myType = 0u;
    
    if (index < particleCount) {
        let packedB = particlesB[index];
        myPos = unpack2x16float(packedB.x);
        
        let packedA = particlesA[index];
        myType = (packedA >> 24u) & 0xFFu;
    }

    var force = vec2f(0.0, 0.0);
    var newColor = f32(myType); 
    var localDensity = 0.0; 
    var nearNeighbors = 0.0;
    var foreignNeighbors = 0.0;

    // --- Optimization: Rule Caching ---
    var myRules: array<f32, 64>;
    let numTypes = u32(params.numTypes);
    
    for (var t = 0u; t < MAX_TYPES; t++) {
        if (t < numTypes) {
             myRules[t] = textureLoad(rulesTex, vec2u(t, myType), 0).r;
        } else {
             myRules[t] = 0.0;
        }
    }

    // --- Mouse Interaction ---
    if (params.mouseType != 0.0) {
        let mousePos = vec2f(params.mouseX, params.mouseY);
        let rawD = myPos - mousePos;
        let dm = fract(rawD * 0.5 + 0.5) * 2.0 - 1.0;
        let distM = length(dm);
        
        if (distM < 0.4) { 
             let mForce = (1.0 - distM / 0.4) * 10.0; 
             force += (dm / (distM + 0.001)) * mForce * params.mouseType;
        }
    }

    // --- Tiled N-Body Calculation ---
    for (var i = 0u; i < particleCount; i += BLOCK_SIZE) {
        let tile_idx = i + local_id.x;
        
        if (tile_idx < particleCount) {
            let packed = particlesA[tile_idx];
            
            // UNPACK ONCE during load:
            // Extract 12-bit positions and 8-bit type
            let nx = f32(packed & 0xFFFu);
            let ny = f32((packed >> 12u) & 0xFFFu);
            let nType = (packed >> 24u) & 0xFFu;
            
            // Normalize 0..4095 -> -1..1
            tile_pos[local_id.x] = vec2f(nx, ny) * (1.0 / 4095.0) * 2.0 - 1.0;
            tile_type[local_id.x] = nType;
        } else {
            // Sentinel: Position far outside range
            tile_pos[local_id.x] = vec2f(-1000.0, -1000.0);
            tile_type[local_id.x] = 0u;
        }

        workgroupBarrier(); 

        if (index < particleCount) {
            let limit = min(BLOCK_SIZE, particleCount - i);
            
            for (var j = 0u; j < BLOCK_SIZE; j++) {
                if (j >= limit) { break; } 

                // FAST READ: No bitwise ops or casts in the hot loop
                let otherPos = tile_pos[j];
                let nType = tile_type[j];

                let rawD = otherPos - myPos;
                let d = fract(rawD * 0.5 + 0.5) * 2.0 - 1.0;
                let d2 = dot(d, d);

                if (d2 > rMaxSq || d2 < 0.000001) {
                    continue;
                }

                let invDist = inverseSqrt(d2);
                let dist = d2 * invDist; 
                
                localDensity += (1.0 - dist * rMaxInv);

                // Check for metabolic conditions (Growth/Decay/Differentiation)
                // Expanded radius for "Social" sensing (diversity checks)
                if (growthEnabled && dist < 0.15) {
                    nearNeighbors += 1.0;
                    
                    if (nType > 0u && nType != myType) {
                        foreignNeighbors += 1.0;
                    }

                    if (myType == 0u && nType > 0u && dist < 0.1) {
                         // Density-dependent growth (Infection)
                         // If area is not too crowded and we get lucky
                         if (localDensity < 20.0 && randomVal < 0.003) {
                             newColor = f32(nType);
                         }
                    }
                }

                var f = 0.0;
                
                if (dist < safeRMin) {
                    let repulse = 1.0 - (dist / safeRMin);
                    f = -20.0 * repulse * repulse;
                } else {
                    let q = (dist - safeRMin) / (rMax - safeRMin);
                    let envelope = 4.0 * q * (1.0 - q);
                    f = myRules[nType] * envelope;
                }
                
                force += d * (invDist * f * baseForceFactor);
            }
        }
        workgroupBarrier();
    }

    if (index < particleCount) {
        // Load B state (16 bytes read - OK for once per thread)
        let packedB = particlesB[index];
        var vel = unpack2x16float(packedB.y);
        let state = unpack2x16float(packedB.z);
        var density = state.x;
        var age = state.y;

        // --- Metabolic Logic ---
        if (growthEnabled && myType > 0u) {
            // 1. Decay Rules (Overcrowding / Isolation)
            let overCrowded = nearNeighbors > 45.0; 
            let lonely = nearNeighbors < 1.0; 
            
            if ((overCrowded || lonely) && randomVal < 0.0005) {
                newColor = 0.0; 
            }
            
            // Random entropy death (Old Age)
            if (randomVal > 0.99995) {
                newColor = 0.0;
            }

            // 2. Differentiation (Specialization)
            let diversity = foreignNeighbors / (nearNeighbors + 0.1);
            
            // Rule: If mature (age > 5s) and surrounded by identical clones (low diversity),
            // differentiate to the next type to form complex tissues (e.g., skin layers).
            if (age > 5.0 && diversity < 0.15 && nearNeighbors > 5.0) {
                 // Use a secondary hash to ensure independent probability
                 let diffChance = fast_hash(vec2f(randomVal, age));
                 if (diffChance < 0.005) {
                     let numTypes = u32(params.numTypes);
                     // Cycle to next type (skipping Food/0)
                     let nextType = (myType % (numTypes - 1u)) + 1u;
                     newColor = f32(nextType);
                     // Reset age to indicate new cell state
                     age = 0.0; 
                 }
            }
        }

        // --- Physics Dynamics ---
        let crowding = smoothstep(5.0, 50.0, localDensity);
        var frict = params.friction;
        frict = mix(frict, frict * 0.5, crowding);

        if (params.temperature > 0.001) {
             let seed = myPos + vec2f(params.time * 60.0, f32(index) * 0.7123);
             let randDir = rand2(seed); 
             force += randDir * params.temperature;
        }

        let fLen = length(force);
        let maxForce = 50.0; 
        if (fLen > maxForce) {
            force = (force / fLen) * maxForce;
        }

        vel += force * params.dt;
        vel *= frict; 
        myPos += vel * params.dt;
        myPos = fract(myPos * 0.5 + 0.5) * 2.0 - 1.0;
        
        let typeChanged = abs(newColor - f32(myType)) > 0.1;

        if (!typeChanged) {
            age += params.dt;
        } else {
            age = 0.0; 
        }
        density = localDensity;

        // --- Write Back ---
        
        // 1. Pack B (High Precision State)
        let newPackedB = vec4u(
            pack2x16float(myPos),      // x: High Prec Pos
            pack2x16float(vel),        // y: Velocity
            pack2x16float(vec2f(density, age)), // z: State
            0u                         // w: Pad
        );
        particlesB[index] = newPackedB;

        // 2. Pack A (Low Precision Neighbor Data)
        // Map -1..1 to 0..1, then scale to 0..4095
        let normPos = clamp(myPos * 0.5 + 0.5, vec2f(0.0), vec2f(1.0));
        let uPos = vec2u(normPos * 4095.0);
        let uType = u32(newColor) & 0xFFu;
        
        let newPackedA = uPos.x | (uPos.y << 12u) | (uType << 24u);
        particlesA[index] = newPackedA;
    }
}
`;

const RENDER_SHADER = `
struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) color: vec4f,
    @location(1) uv: vec2f,
    @location(2) speed: f32,
    @location(3) extra: vec2f, // x: Density, y: Age
};

struct Params {
    width: f32,
    height: f32,
    friction: f32,
    dt: f32,
    rMax: f32,
    forceFactor: f32,
    minDist: f32,
    count: f32,
    size: f32,
    opacity: f32, 
    numTypes: f32,
    time: f32,
    growth: f32,
    mouseX: f32,
    mouseY: f32,
    mouseType: f32,
    temperature: f32,
};

struct Color {
    r: f32, g: f32, b: f32, a: f32
};

@group(0) @binding(0) var<storage, read> particlesA: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read> colors: array<Color>;
@group(0) @binding(4) var<storage, read> particlesB: array<vec4u>;

@vertex
fn vs_main(
    @builtin(vertex_index) vIdx: u32,
    @builtin(instance_index) iIdx: u32
) -> VertexOutput {
    // Access High Precision Position (B)
    let packedB = particlesB[iIdx];
    let pos = unpack2x16float(packedB.x);
    let vel = unpack2x16float(packedB.y);
    let state = unpack2x16float(packedB.z);
    let density = state.x;
    let age = state.y;

    // Access Type (A)
    let packedA = particlesA[iIdx];
    let typeIdx = (packedA >> 24u) & 0xFFu;
    
    // --- Screen Dimensions & Culling ---
    // Ensure we never divide by zero or very small numbers
    let screenW = max(1.0, params.width);
    let screenH = max(1.0, params.height);
    
    // Frustum culling margin (approximate max particle size in NDC)
    let maxParticleSizePx = params.size * 8.0; 
    let maxExtentX = (maxParticleSizePx / screenW) * 2.0;
    let maxExtentY = (maxParticleSizePx / screenH) * 2.0;
    
    // Cull if outside view (plus margin)
    if (abs(pos.x) > (1.0 + maxExtentX) || abs(pos.y) > (1.0 + maxExtentY)) {
        var cullOut: VertexOutput;
        cullOut.position = vec4f(-2.0, -2.0, 2.0, 1.0); 
        return cullOut;
    }

    // --- Color & Alpha Logic ---
    let maxType = i32(params.numTypes) - 1;
    let cType = clamp(i32(typeIdx), 0, maxType);
    let colorData = colors[cType];
    
    var baseAlpha = 1.0;
    if (cType == 0) { baseAlpha = 0.3; } // Food is transparent
    
    let growthFactor = min(1.0, age * 2.0); 
    
    // Calculate maximum potential alpha to determine visibility
    let maxAlpha = params.opacity * colorData.a * baseAlpha * growthFactor;
    
    if (maxAlpha < 0.005) {
        var cullOut: VertexOutput;
        cullOut.position = vec4f(-2.0, -2.0, 2.0, 1.0);
        return cullOut;
    }

    // --- Dynamic Sizing ---
    // Calculate visible radius based on alpha falloff to reduce overdraw
    // pow(0.01 / maxAlpha, 0.4) estimates where the gaussian dropoff hits ~1% opacity
    let safeAlpha = max(maxAlpha, 0.001);
    let cutoff = pow(0.01 / safeAlpha, 0.4);
    let visibleRadius = clamp(1.0 - cutoff, 0.2, 1.0);

    // Calculate final pixel size
    let quadSizePx = params.size * 4.0 * growthFactor * visibleRadius;
    
    // Don't render sub-pixel particles
    if (quadSizePx < 0.5) {
        var cullOut: VertexOutput;
        cullOut.position = vec4f(-2.0, -2.0, 2.0, 1.0);
        return cullOut;
    }
    
    // --- Geometry Construction ---
    var offsets = array<vec2f, 6>(
        vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(-1.0, 1.0),
        vec2f(-1.0, 1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0)
    );
    let rawOffset = offsets[vIdx]; // [-1, 1]
    
    // Velocity Deformation (Stretch/Squash)
    let speed = length(vel);
    var dir = vec2f(1.0, 0.0);
    if (speed > 0.001) {
        dir = vel / speed;
    }
    let perp = vec2f(-dir.y, dir.x);
    
    let stretch = clamp(1.0 + speed * 1.5, 1.0, 2.5);
    let squash = 1.0 / sqrt(stretch); 
    
    // Apply rotation to the unit offset
    let rotOffset = (dir * rawOffset.x * stretch) + (perp * rawOffset.y * squash);
    
    // Scale to pixels
    let finalOffsetPx = rotOffset * quadSizePx;

    // Convert pixels to NDC [2.0/W, 2.0/H]
    let offsetNDC = finalOffsetPx * vec2f(2.0 / screenW, 2.0 / screenH);
    // Correct Y aspect is implicit because screenH is used directly

    var output: VertexOutput;
    output.position = vec4f(pos + offsetNDC, 0.5, 1.0);
    output.uv = rawOffset * visibleRadius; // Pass correct UV scale for fragment shader
    output.speed = speed;
    output.color = vec4f(colorData.r, colorData.g, colorData.b, baseAlpha);
    output.extra = vec2f(density, age);
    
    return output;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4f {
    let uv = input.uv;
    let d2 = dot(uv, uv);
    if (d2 > 1.0) { discard; }

    let r = sqrt(d2); 
    
    let density = input.extra.x;
    let age = input.extra.y;
    let baseColor = input.color.rgb;

    let edgeWidth = 0.15;
    let edgeStart = 1.0 - edgeWidth;
    let membrane = smoothstep(edgeStart - 0.05, edgeStart, r) * smoothstep(1.0, 0.95, r);
    
    let body = smoothstep(1.0, 0.2, r);
    
    let stressFactor = smoothstep(5.0, 40.0, density);

    let pulseSpeed = 2.0 + (stressFactor * 8.0);
    let pulsePhase = sin(params.time * pulseSpeed);
    
    let pulseAmp = 0.05 + (stressFactor * 0.15); 
    
    let nucleusBaseSize = 0.25;
    let currentNucleusSize = nucleusBaseSize + (pulsePhase * pulseAmp);
    
    let nucleus = smoothstep(currentNucleusSize + 0.15, currentNucleusSize, r);

    let maturity = smoothstep(0.0, 2.0, age);
    let matureBodyColor = mix(vec3f(0.95), baseColor, 0.6 + 0.4 * maturity);

    let stressColor = mix(vec3f(1.0), vec3f(1.0, 0.4, 0.1), stressFactor); 
    let nucleusColor = mix(vec3f(1.0), stressColor, 0.8 + 0.2 * pulsePhase);

    var finalColor = matureBodyColor * body * 0.7; 
    finalColor += baseColor * membrane * 1.8;
    finalColor = mix(finalColor, nucleusColor, nucleus * 0.9);

    let speed = min(input.speed, 3.0);
    finalColor += vec3f(0.1) * speed; 

    let alpha = params.opacity * input.color.a * smoothstep(1.0, 0.85, r);
    
    if (alpha < 0.01) { discard; }

    return vec4f(finalColor * alpha, alpha);
}
`;

export class SimulationEngine {
    private canvas: HTMLCanvasElement;
    private device: GPUDevice | null = null;
    private context: GPUCanvasContext | null = null;
    
    private pipelineAdditive: GPURenderPipeline | null = null;
    private pipelineNormal: GPURenderPipeline | null = null;
    private pipelineFade: GPURenderPipeline | null = null;
    
    private computePipeline: GPUComputePipeline | null = null;
    
    // Buffer A: u32 (4 bytes). [Pos 12b | Pos 12b | Type 8b]
    private particleBufferA: GPUBuffer | null = null;
    // Buffer B: vec4u (16 bytes). [Pos 32b | Vel 32b | State 32b | Pad 32b]
    private particleBufferB: GPUBuffer | null = null;

    private paramsBuffer: GPUBuffer | null = null;
    private rulesTexture: GPUTexture | null = null;
    private colorBuffer: GPUBuffer | null = null;
    
    private renderTarget: GPUTexture | null = null;
    private depthTexture: GPUTexture | null = null; // Depth Buffer
    private textureNeedsClear: boolean = true;
    
    private computeBindGroup: GPUBindGroup | null = null;
    private renderBindGroupAdditive: GPUBindGroup | null = null;
    private renderBindGroupNormal: GPUBindGroup | null = null;
    
    private paramsData: Float32Array | null = null;
    private particleCount: number = 0;
    private numTypes: number = 4;
    private trailsEnabled: boolean = false;
    private dpiScale: number = 1.0;
    private blendMode: 'additive' | 'normal' = 'additive';
    
    private windowWidth: number = 100;
    private windowHeight: number = 100;
    
    private animationId: number = 0;
    private isPaused: boolean = false;
    private onFpsUpdate?: (fps: number) => void;
    private lastFpsTime: number = 0;
    private frameCount: number = 0;
    private adapterInfo: string = "";

    // Mouse Tracking
    private mouseX: number = 0;
    private mouseY: number = 0;
    private mouseInteractionType: number = 0; 
    
    private readonly MAX_RULE_TEXTURE_SIZE = 64;

    constructor(canvas: HTMLCanvasElement, onFpsUpdate?: (fps: number) => void) {
        this.canvas = canvas;
        this.onFpsUpdate = onFpsUpdate;
        this.setupInputHandlers();
    }

    private setupInputHandlers() {
        const updateMouse = (e: MouseEvent) => {
            const rect = this.canvas.getBoundingClientRect();
            const x = (e.clientX - rect.left) / rect.width;
            const y = (e.clientY - rect.top) / rect.height;
            this.mouseX = x * 2.0 - 1.0;
            this.mouseY = -(y * 2.0 - 1.0); 
        };

        this.canvas.addEventListener('mousemove', (e) => updateMouse(e));
        this.canvas.addEventListener('mousedown', (e) => {
            updateMouse(e);
            if (e.button === 0) this.mouseInteractionType = 1.0;
            else if (e.button === 2) this.mouseInteractionType = -1.0;
        });
        this.canvas.addEventListener('mouseup', () => this.mouseInteractionType = 0);
        this.canvas.addEventListener('mouseleave', () => this.mouseInteractionType = 0);
        this.canvas.addEventListener('contextmenu', (e) => e.preventDefault());
    }

    async init(params: SimulationParams, rules: RuleMatrix, colors: ColorDefinition[]) {
        if (!(navigator as any).gpu) throw new Error("WebGPU not supported");
        
        const adapter = await (navigator as any).gpu.requestAdapter({
            powerPreference: params.gpuPreference
        });
        
        if (!adapter) throw new Error("No GPU Adapter found");
        
        if (adapter.requestAdapterInfo) {
            const info = await adapter.requestAdapterInfo();
            this.adapterInfo = info.device || info.description || "";
            console.log("Using GPU:", this.adapterInfo);
        }

        this.device = await adapter.requestDevice();
        this.context = this.canvas.getContext('webgpu') as GPUCanvasContext;
        
        this.particleCount = params.particleCount;
        this.numTypes = params.numTypes;
        this.trailsEnabled = params.trails;
        this.dpiScale = params.dpiScale;
        this.blendMode = params.blendMode;

        this.resize(this.canvas.clientWidth || window.innerWidth, this.canvas.clientHeight || window.innerHeight);

        this.createParticleBuffers();
        this.createRulesTexture(rules);
        this.createParamsBuffer(params);
        this.createColorBuffer(colors);

        const computeModule = this.device.createShaderModule({ code: COMPUTE_SHADER });
        this.computePipeline = this.device.createComputePipeline({
            layout: 'auto',
            compute: { module: computeModule, entryPoint: 'main' },
        });

        const renderModule = this.device.createShaderModule({ code: RENDER_SHADER });
        const format = (navigator as any).gpu.getPreferredCanvasFormat();
        
        // Depth Stencil State Config
        const depthStencilState = {
            format: 'depth24plus',
            depthWriteEnabled: false, // Read-only for particles (additive/translucent)
            depthCompare: 'less-equal',
        };

        this.pipelineAdditive = this.device.createRenderPipeline({
            layout: 'auto',
            vertex: { module: renderModule, entryPoint: 'vs_main' },
            fragment: {
                module: renderModule,
                entryPoint: 'fs_main',
                targets: [{
                    format,
                    blend: {
                        color: { srcFactor: 'one', dstFactor: 'one', operation: 'add' },
                        alpha: { srcFactor: 'one', dstFactor: 'one', operation: 'add' }
                    }
                }],
            },
            depthStencil: depthStencilState,
            primitive: { topology: 'triangle-list' },
        });

        this.pipelineNormal = this.device.createRenderPipeline({
            layout: 'auto',
            vertex: { module: renderModule, entryPoint: 'vs_main' },
            fragment: {
                module: renderModule,
                entryPoint: 'fs_main',
                targets: [{
                    format,
                    blend: {
                        color: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
                        alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' }
                    }
                }],
            },
            depthStencil: depthStencilState,
            primitive: { topology: 'triangle-list' },
        });
        
        const fadeModule = this.device.createShaderModule({ code: FADE_SHADER });
        this.pipelineFade = this.device.createRenderPipeline({
            layout: 'auto',
            vertex: { module: fadeModule, entryPoint: 'vs_main' },
            fragment: {
                module: fadeModule,
                entryPoint: 'fs_main',
                targets: [{
                    format,
                    blend: {
                        color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha', operation: 'add' },
                        alpha: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha', operation: 'add' }
                    }
                }],
            },
            // FIX: Must contain depthStencil state matching the render pass attachment
            depthStencil: {
                format: 'depth24plus',
                depthWriteEnabled: false,
                depthCompare: 'always', 
            },
            primitive: { topology: 'triangle-list' },
        });

        this.updateBindGroups();
        this.start();
        
        return this.adapterInfo;
    }

    public destroy() {
        this.stop();
        if (this.device) {
            this.device.destroy();
            this.device = null;
        }
        if (this.renderTarget) {
            this.renderTarget.destroy();
            this.renderTarget = null;
        }
        if (this.depthTexture) {
            this.depthTexture.destroy();
            this.depthTexture = null;
        }
        if (this.rulesTexture) {
            this.rulesTexture.destroy();
            this.rulesTexture = null;
        }
        if (this.particleBufferA) {
            this.particleBufferA.destroy();
            this.particleBufferA = null;
        }
        if (this.particleBufferB) {
            this.particleBufferB.destroy();
            this.particleBufferB = null;
        }
    }

    private createParticleBuffers() {
        if (!this.device) return;
        const count = this.particleCount;
        
        // Buffer A: 4 bytes per particle (u32)
        // [12-bit X | 12-bit Y | 8-bit Type]
        const dataA = new Uint32Array(count); 
        
        // Buffer B: 16 bytes per particle (vec4u)
        // [Pos(32b) | Vel(32b) | State(32b) | Pad(32b)]
        const dataB = new Uint32Array(count * 4); 
        
        for(let i=0; i<count; i++) {
            const rx = Math.random() * 2 - 1;
            const ry = Math.random() * 2 - 1;
            
            // A - Quantized
            const uX = Math.floor((rx * 0.5 + 0.5) * 4095.0);
            const uY = Math.floor((ry * 0.5 + 0.5) * 4095.0);
            const type = Math.floor(Math.random() * this.numTypes);
            
            dataA[i] = (uX & 0xFFF) | ((uY & 0xFFF) << 12) | ((type & 0xFF) << 24);

            // B - High Precision
            const px = toHalf(rx);
            const py = toHalf(ry);
            const packedPos = (py << 16) | px;

            const vx = toHalf(0);
            const vy = toHalf(0);
            const packedVel = (vy << 16) | vx;
            
            const state = toHalf(0);
            const packedState = (state << 16) | state; // density=0, age=0

            dataB[i*4 + 0] = packedPos; 
            dataB[i*4 + 1] = packedVel; 
            dataB[i*4 + 2] = packedState; 
            dataB[i*4 + 3] = 0; // Padding
        }

        // Buffer A
        this.particleBufferA = this.device.createBuffer({
            size: dataA.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
            mappedAtCreation: true,
        });
        new Uint32Array(this.particleBufferA.getMappedRange()).set(dataA);
        this.particleBufferA.unmap();

        // Buffer B
        this.particleBufferB = this.device.createBuffer({
            size: dataB.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            mappedAtCreation: true,
        });
        new Uint32Array(this.particleBufferB.getMappedRange()).set(dataB);
        this.particleBufferB.unmap();
    }

    private createRulesTexture(rules: RuleMatrix) {
        if (!this.device) return;

        if (!this.rulesTexture) {
            this.rulesTexture = this.device.createTexture({
                size: [this.MAX_RULE_TEXTURE_SIZE, this.MAX_RULE_TEXTURE_SIZE],
                format: 'r8snorm', 
                usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
            });
        }
        
        this.uploadRulesToTexture(rules);
    }

    private uploadRulesToTexture(rules: RuleMatrix) {
        if (!this.device || !this.rulesTexture) return;

        const numTypes = rules.length;
        if (numTypes === 0) return;
        
        const safeNumTypes = Math.min(numTypes, this.MAX_RULE_TEXTURE_SIZE);
        const bytesPerRow = safeNumTypes; 
        const bufferSize = bytesPerRow * safeNumTypes;
        
        const data = new Int8Array(bufferSize);

        for (let row = 0; row < safeNumTypes; row++) {
            const rowOffset = row * bytesPerRow;
            for (let col = 0; col < safeNumTypes; col++) {
                let val = Math.max(-1, Math.min(1, rules[row][col]));
                data[rowOffset + col] = Math.round(val * 127);
            }
        }

        this.device.queue.writeTexture(
            { texture: this.rulesTexture },
            data,
            { bytesPerRow: bytesPerRow, rowsPerImage: safeNumTypes },
            { width: safeNumTypes, height: safeNumTypes }
        );
    }

    private createColorBuffer(colors: ColorDefinition[]) {
        if (!this.device) return;
        const data = new Float32Array(colors.length * 4);
        for(let i=0; i<colors.length; i++) {
            data[i*4 + 0] = colors[i].r / 255.0;
            data[i*4 + 1] = colors[i].g / 255.0;
            data[i*4 + 2] = colors[i].b / 255.0;
            data[i*4 + 3] = 1.0; 
        }
        this.colorBuffer = this.device.createBuffer({
            size: data.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            mappedAtCreation: true,
        });
        new Float32Array(this.colorBuffer.getMappedRange()).set(data);
        this.colorBuffer.unmap();
    }

    private createParamsBuffer(params: SimulationParams) {
        if (!this.device) return;
        const scaledWidth = this.windowWidth * this.dpiScale;
        const scaledHeight = this.windowHeight * this.dpiScale;

        const data = new Float32Array([
            scaledWidth,
            scaledHeight,
            params.friction,
            params.dt,
            params.rMax,
            params.forceFactor,
            params.minDistance,
            params.particleCount,
            params.particleSize,
            params.baseColorOpacity,
            params.numTypes,
            0.0, // time
            params.growth ? 1.0 : 0.0,
            0.0, // mouseX
            0.0, // mouseY
            0.0, // mouseType
            params.temperature,
            0.0, 0.0, 0.0 // Padding
        ]);
        this.paramsData = data;

        this.paramsBuffer = this.device.createBuffer({
            size: data.byteLength,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            mappedAtCreation: true,
        });
        new Float32Array(this.paramsBuffer.getMappedRange()).set(data);
        this.paramsBuffer.unmap();
    }

    private updateBindGroups() {
        if (!this.device || !this.computePipeline || !this.pipelineAdditive || !this.particleBufferA || !this.particleBufferB || !this.rulesTexture) return;

        this.computeBindGroup = this.device.createBindGroup({
            layout: this.computePipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.particleBufferA } },
                { binding: 1, resource: this.rulesTexture.createView() },
                { binding: 2, resource: { buffer: this.paramsBuffer } },
                { binding: 4, resource: { buffer: this.particleBufferB } },
            ],
        });

        const renderEntries = [
            { binding: 0, resource: { buffer: this.particleBufferA } },
            { binding: 2, resource: { buffer: this.paramsBuffer } },
            { binding: 3, resource: { buffer: this.colorBuffer } },
            { binding: 4, resource: { buffer: this.particleBufferB } },
        ];

        this.renderBindGroupAdditive = this.device.createBindGroup({
            layout: this.pipelineAdditive.getBindGroupLayout(0),
            entries: renderEntries,
        });

        this.renderBindGroupNormal = this.device.createBindGroup({
            layout: this.pipelineNormal!.getBindGroupLayout(0),
            entries: renderEntries,
        });
    }

    public updateColors(colors: ColorDefinition[]) {
        if(!this.device) return;
        this.createColorBuffer(colors);
        this.updateBindGroups();
    }

    public updateParams(params: SimulationParams) {
        if (!this.device || !this.paramsBuffer || !this.paramsData) return;
        
        let needsBindGroupUpdate = false;
        if (params.particleCount !== this.particleCount) {
             this.particleCount = params.particleCount;
             this.createParticleBuffers();
             needsBindGroupUpdate = true;
        }

        this.numTypes = params.numTypes;
        this.trailsEnabled = params.trails;
        this.blendMode = params.blendMode;

        if (params.dpiScale !== this.dpiScale) {
            this.dpiScale = params.dpiScale;
            this.resize(this.windowWidth, this.windowHeight);
        }

        this.paramsData[0] = this.windowWidth * this.dpiScale;
        this.paramsData[1] = this.windowHeight * this.dpiScale;
        this.paramsData[2] = params.friction;
        this.paramsData[3] = params.dt;
        this.paramsData[4] = params.rMax;
        this.paramsData[5] = params.forceFactor;
        this.paramsData[6] = params.minDistance;
        this.paramsData[7] = params.particleCount;
        this.paramsData[8] = params.particleSize;
        this.paramsData[9] = params.baseColorOpacity;
        this.paramsData[10] = params.numTypes;
        this.paramsData[12] = params.growth ? 1.0 : 0.0;
        this.paramsData[16] = params.temperature;
        
        this.device.queue.writeBuffer(this.paramsBuffer, 0, this.paramsData);
        if(needsBindGroupUpdate) this.updateBindGroups();
    }

    public updateRules(rules: RuleMatrix) {
        if (!this.device || !this.rulesTexture) return;
        this.uploadRulesToTexture(rules);
    }

    public resize(width: number, height: number) {
        if (!this.device) return;
        this.windowWidth = width;
        this.windowHeight = height;
        const scaledWidth = Math.floor(width * this.dpiScale);
        const scaledHeight = Math.floor(height * this.dpiScale);
        this.canvas.width = scaledWidth;
        this.canvas.height = scaledHeight;
        const format = (navigator as any).gpu.getPreferredCanvasFormat();
        
        this.context?.configure({
            device: this.device,
            format: format,
            alphaMode: 'premultiplied',
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_DST 
        });

        // 1. Recreate Color Target
        if (this.renderTarget) this.renderTarget.destroy();
        this.renderTarget = this.device.createTexture({
            size: [scaledWidth, scaledHeight],
            format: format,
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC
        });

        // 2. Recreate Depth Target
        if (this.depthTexture) this.depthTexture.destroy();
        this.depthTexture = this.device.createTexture({
            size: [scaledWidth, scaledHeight],
            format: 'depth24plus',
            usage: GPUTextureUsage.RENDER_ATTACHMENT,
        });

        this.textureNeedsClear = true; 
        
        if (this.paramsData && this.paramsBuffer) {
            this.paramsData[0] = scaledWidth;
            this.paramsData[1] = scaledHeight;
            this.device.queue.writeBuffer(this.paramsBuffer, 0, this.paramsData);
        }
    }

    public reset() {
        if (!this.device || !this.particleBufferA || !this.particleBufferB) return;
        const count = this.particleCount;
        
        const dataA = new Uint32Array(count);
        const dataB = new Uint32Array(count * 4);

        for(let i=0; i<count; i++) {
             const rx = Math.random() * 2 - 1;
             const ry = Math.random() * 2 - 1;
             
             // A
             const uX = Math.floor((rx * 0.5 + 0.5) * 4095.0);
             const uY = Math.floor((ry * 0.5 + 0.5) * 4095.0);
             const type = Math.floor(Math.random() * this.numTypes);
             
             dataA[i] = (uX & 0xFFF) | ((uY & 0xFFF) << 12) | ((type & 0xFF) << 24);
 
             // B
             const px = toHalf(rx);
             const py = toHalf(ry);
             const packedPos = (py << 16) | px;
 
             const vx = toHalf(0);
             const vy = toHalf(0);
             const packedVel = (vy << 16) | vx;
             
             const state = toHalf(0);
             const packedState = (state << 16) | state; // density=0, age=0
 
             dataB[i*4 + 0] = packedPos; 
             dataB[i*4 + 1] = packedVel; 
             dataB[i*4 + 2] = packedState; 
             dataB[i*4 + 3] = 0; 
        }

        this.device.queue.writeBuffer(this.particleBufferA, 0, dataA);
        this.device.queue.writeBuffer(this.particleBufferB, 0, dataB);
        this.textureNeedsClear = true;
    }

    public setPaused(paused: boolean) {
        this.isPaused = paused;
    }

    private loop = (timestamp: number) => {
        this.animationId = requestAnimationFrame(this.loop);
        
        if (this.isPaused || !this.device || !this.context || !this.computePipeline || !this.pipelineAdditive || !this.renderTarget || !this.pipelineFade) return;

        if (this.paramsData && this.paramsBuffer) {
            this.paramsData[11] = timestamp / 1000.0;
            this.paramsData[13] = this.mouseX;
            this.paramsData[14] = this.mouseY;
            this.paramsData[15] = this.mouseInteractionType;
            this.device.queue.writeBuffer(this.paramsBuffer, 0, this.paramsData);
        }

        const commandEncoder = this.device.createCommandEncoder();

        const computePass = commandEncoder.beginComputePass();
        computePass.setPipeline(this.computePipeline);
        computePass.setBindGroup(0, this.computeBindGroup!);
        computePass.dispatchWorkgroups(Math.ceil(this.particleCount / 256));
        computePass.end();

        const currentTexture = this.context.getCurrentTexture();
        
        const pipeline = this.blendMode === 'normal' ? this.pipelineNormal! : this.pipelineAdditive!;
        const bindGroup = this.blendMode === 'normal' ? this.renderBindGroupNormal! : this.renderBindGroupAdditive!;

        // RENDER OPTIMIZATION: Branch logic based on trails requirement
        if (this.trailsEnabled) {
             // Trails enabled: Render to intermediate texture (load previous frame), then copy to screen
            const shouldClear = this.textureNeedsClear;
            this.textureNeedsClear = false;

            const renderPass = commandEncoder.beginRenderPass({
                colorAttachments: [{
                    view: this.renderTarget.createView(),
                    clearValue: { r: 0.0, g: 0.0, b: 0.0, a: 1.0 },
                    loadOp: shouldClear ? 'clear' : 'load',
                    storeOp: 'store',
                }],
                depthStencilAttachment: {
                    view: this.depthTexture!.createView(),
                    depthClearValue: 1.0,
                    depthLoadOp: 'clear',
                    depthStoreOp: 'discard',
                }
            });
            
            // 1. Draw Fade Quad (Dims the previous frame)
            renderPass.setPipeline(this.pipelineFade);
            renderPass.draw(6);

            // 2. Draw Particles
            renderPass.setPipeline(pipeline);
            renderPass.setBindGroup(0, bindGroup);
            renderPass.draw(6, this.particleCount);
            renderPass.end();

            commandEncoder.copyTextureToTexture(
                { texture: this.renderTarget },
                { texture: currentTexture },
                [this.canvas.width, this.canvas.height]
            );
        } else {
            // Trails disabled (High Performance): Render directly to swap chain with clear
            const renderPass = commandEncoder.beginRenderPass({
                colorAttachments: [{
                    view: currentTexture.createView(),
                    clearValue: { r: 0.0, g: 0.0, b: 0.0, a: 1.0 },
                    loadOp: 'clear',
                    storeOp: 'store',
                }],
                depthStencilAttachment: {
                    view: this.depthTexture!.createView(),
                    depthClearValue: 1.0,
                    depthLoadOp: 'clear',
                    depthStoreOp: 'discard',
                }
            });

            renderPass.setPipeline(pipeline);
            renderPass.setBindGroup(0, bindGroup);
            renderPass.draw(6, this.particleCount);
            renderPass.end();

            // If we switch back to trails later, ensure the texture is cleared first
            this.textureNeedsClear = true;
        }

        this.device.queue.submit([commandEncoder.finish()]);

        this.frameCount++;
        if (timestamp - this.lastFpsTime >= 1000) {
            if (this.onFpsUpdate) this.onFpsUpdate(this.frameCount);
            this.frameCount = 0;
            this.lastFpsTime = timestamp;
        }
    }

    public start() {
        if (this.animationId) cancelAnimationFrame(this.animationId);
        this.lastFpsTime = performance.now();
        this.loop(this.lastFpsTime);
    }

    public stop() {
        cancelAnimationFrame(this.animationId);
    }
}