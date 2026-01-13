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
struct Particle {
    pos: vec2f,
    vel: vec2f,
    color: f32,
    pad0: f32, // Used for "Stress" tracking
    pad1: f32,
    pad2: f32,
};

// Aligned to 16 floats + 4 floats padding = 20 floats (80 bytes)
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

@group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
// Use Texture for Rules Lookup (r8snorm gives free normalization from -128..127 to -1.0..1.0)
@group(0) @binding(1) var rulesTex: texture_2d<f32>; 
@group(0) @binding(2) var<uniform> params: Params;

const BLOCK_SIZE = 256u;
const MAX_TYPES = 64u; 

var<workgroup> tile_pos: array<vec2f, BLOCK_SIZE>;
var<workgroup> tile_type: array<u32, BLOCK_SIZE>;

// Faster hash for inner loops (lower quality but sufficient for variation)
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

    // Loop Invariants & Optimization Constants
    let rMax = params.rMax;
    let rMaxSq = rMax * rMax; 
    let rMaxInv = 1.0 / rMax; 
    
    // "Stiff Core" Physics settings
    // A smaller safeRMin creates "tighter" lattices
    let rMin = params.minDist; 
    let safeRMin = max(0.001, rMin);
    
    let baseForceFactor = params.forceFactor;
    let growthEnabled = params.growth > 0.5;
    
    // Seed for biological functions
    let bioSeed = params.time + f32(index) * 0.123;
    let randomVal = fast_hash(vec2f(bioSeed, f32(index)));
    
    var myPos = vec2f(0.0);
    var myColor = 0.0;
    
    if (index < particleCount) {
        let p = particles[index];
        myPos = p.pos;
        myColor = p.color;
    }

    let myType = u32(round(myColor));
    var force = vec2f(0.0, 0.0);
    var newColor = myColor; 
    var localDensity = 0.0; 
    var nearNeighbors = 0.0;

    // --- Optimization: Rule Caching ---
    var myRules: array<f32, 64>;
    let numTypes = u32(params.numTypes);
    
    // Load rules into registers
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
            let p = particles[tile_idx];
            tile_pos[local_id.x] = p.pos;
            tile_type[local_id.x] = u32(round(p.color)); 
        } else {
            tile_pos[local_id.x] = vec2f(-1000.0, -1000.0); // Sentinel
            tile_type[local_id.x] = 0u;
        }

        workgroupBarrier(); 

        if (index < particleCount) {
            let limit = min(BLOCK_SIZE, particleCount - i);
            
            for (var j = 0u; j < BLOCK_SIZE; j++) {
                if (j >= limit) { break; } 

                let otherPos = tile_pos[j];
                let rawD = otherPos - myPos;
                let d = fract(rawD * 0.5 + 0.5) * 2.0 - 1.0;
                let d2 = dot(d, d);

                // Early Exit
                if (d2 > rMaxSq || d2 < 0.000001) {
                    continue;
                }

                let invDist = inverseSqrt(d2);
                let dist = d2 * invDist; // = sqrt(d2)
                
                // Density for Crowd Control
                localDensity += (1.0 - dist * rMaxInv);

                // Metabolic Neighbor Counting (Very close)
                if (growthEnabled && dist < 0.05) {
                    nearNeighbors += 1.0;
                    
                    // Feeding Logic:
                    // If I am Food (Type 0) and neighbor is Life (Type > 0), I might become Life.
                    // This creates growth at the edges of organisms.
                    if (myType == 0u && tile_type[j] > 0u) {
                        // Growth probability check
                        if (randomVal < 0.005) {
                            newColor = f32(tile_type[j]);
                        }
                    }
                }

                var f = 0.0;
                
                if (dist < safeRMin) {
                    // STIFF CORE PHYSICS
                    // Instead of linear repulsion, use a power curve to create "solid" particles
                    // This prevents particles from merging into a single point, enabling "tissues"
                    let repulse = 1.0 - (dist / safeRMin);
                    f = -20.0 * repulse * repulse; // Strong exponential repulsion
                } else {
                    // BONDING PHYSICS
                    // Use a curve that supports equilibrium at specific distances
                    let q = (dist - safeRMin) / (rMax - safeRMin);
                    
                    // Standard envelope: 4 * q * (1-q)
                    // We modify it to maintain force longer to create "chains"
                    let envelope = 4.0 * q * (1.0 - q);
                    
                    f = myRules[tile_type[j]] * envelope;
                }
                
                force += d * (invDist * f * baseForceFactor);
            }
        }
        workgroupBarrier();
    }

    if (index < particleCount) {
        var p = particles[index]; 
        let speed = length(p.vel);

        // --- Metabolic Decay (Apoptosis) ---
        // If enabled, complex life regulates itself
        if (growthEnabled && myType > 0u) {
            // Die if too crowded (Pressure death) OR too lonely (Isolation death)
            // But only with a small chance to avoid flickering
            let overCrowded = nearNeighbors > 15.0; 
            let lonely = nearNeighbors < 1.0;
            
            if ((overCrowded || lonely) && randomVal < 0.02) {
                newColor = 0.0; // Return to Food
            }
        }

        // --- Physics Dynamics ---

        // Drag/Friction
        // Organisms act more like they are in a viscous fluid
        let crowding = smoothstep(5.0, 50.0, localDensity);
        
        var frict = params.friction;
        // Increase friction in dense areas to stabilize structures (Gel-like)
        frict = mix(frict, frict * 0.5, crowding);

        // --- Temperature/Brownian Motion ---
        if (params.temperature > 0.001) {
             let seed = myPos + vec2f(params.time * 60.0, f32(index) * 0.7123);
             let randDir = rand2(seed); 
             force += randDir * params.temperature;
        }

        // Hard Speed Limit (safety)
        let fLen = length(force);
        let maxForce = 50.0; 
        if (fLen > maxForce) {
            force = (force / fLen) * maxForce;
        }

        p.vel += force * params.dt;
        p.vel *= frict; 
        p.pos += p.vel * params.dt;
        p.pos = fract(p.pos * 0.5 + 0.5) * 2.0 - 1.0;
        
        p.color = newColor; 
        particles[index] = p;
    }
}
`;

const RENDER_SHADER = `
struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) color: vec4f,
    @location(1) uv: vec2f,
    @location(2) speed: f32,
};

struct Particle {
    pos: vec2f,
    vel: vec2f,
    color: f32,
    pad0: f32,
    pad1: f32,
    pad2: f32,
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

@group(0) @binding(0) var<storage, read> particles: array<Particle>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read> colors: array<Color>;

@vertex
fn vs_main(
    @builtin(vertex_index) vIdx: u32,
    @builtin(instance_index) iIdx: u32
) -> VertexOutput {
    let p = particles[iIdx];
    
    // Frustum Culling (Simple Box)
    // Particles outside [-1.1, 1.1] range are definitely off-screen
    if (abs(p.pos.x) > 1.1 || abs(p.pos.y) > 1.1) {
        var cullOut: VertexOutput;
        cullOut.position = vec4f(-2.0, -2.0, 2.0, 1.0); // Move off-clip
        return cullOut;
    }

    let maxType = i32(params.numTypes) - 1;
    let cType = clamp(i32(round(p.color)), 0, maxType);
    let colorData = colors[cType];
    
    // Dim "Food" (Type 0)
    var alpha = 1.0;
    if (cType == 0) {
        alpha = 0.3; 
    }
    
    // --- Alpha Culling & Radius Optimization ---
    let maxAlpha = params.opacity * colorData.a * alpha;
    
    // 1. Cull Invisible
    if (maxAlpha < 0.01) {
        var cullOut: VertexOutput;
        cullOut.position = vec4f(-2.0, -2.0, 2.0, 1.0);
        return cullOut;
    }

    // 2. Shrink Quad to Visible Radius
    // The fragment shader discards if alpha < 0.01.
    // alpha ~= (1-dist)^2.5 * maxAlpha.
    // We solve for dist where alpha == 0.01.
    // dist = 1.0 - (0.01 / maxAlpha)^(1/2.5)
    let cutoff = pow(0.01 / maxAlpha, 0.4);
    // Safety clamp to ensure we don't invert or make it too small to see
    let visibleRadius = clamp(1.0 - cutoff, 0.1, 1.0);

    let w = max(1.0, params.width);
    let h = max(1.0, params.height);
    let aspect = w / h;

    // Soft Particle Geometry
    let quadSize = params.size * 4.0; 
    let sizeNDC = (quadSize / w) * 2.0;
    
    // Apply Optimization: Scale geometry by visible radius
    let effectiveSizeNDC = sizeNDC * visibleRadius;

    if (effectiveSizeNDC < 0.0001) {
        var cullOut: VertexOutput;
        cullOut.position = vec4f(-2.0, -2.0, 2.0, 1.0);
        return cullOut;
    }
    
    var offsets = array<vec2f, 6>(
        vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(-1.0, 1.0),
        vec2f(-1.0, 1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0)
    );
    
    let rawOffset = offsets[vIdx];
    
    // --- Velocity Deformation (Squash & Stretch) ---
    let speed = length(p.vel);
    var dir = vec2f(1.0, 0.0);
    if (speed > 0.001) {
        dir = p.vel / speed;
    }
    let perp = vec2f(-dir.y, dir.x);
    
    let stretch = clamp(1.0 + speed * 1.5, 1.0, 2.5);
    let squash = 1.0 / sqrt(stretch); 
    
    let rotOffset = (dir * rawOffset.x * stretch) + (perp * rawOffset.y * squash);
    
    var finalOffset = rotOffset * effectiveSizeNDC;
    finalOffset.y *= aspect; 

    var output: VertexOutput;
    // Z = 0.5 (Mid-depth)
    output.position = vec4f(p.pos + finalOffset, 0.5, 1.0);
    // UVs are scaled to match the physical shrink, preserving the gradient logic
    output.uv = rawOffset * visibleRadius;
    output.speed = speed;
    
    output.color = vec4f(colorData.r, colorData.g, colorData.b, alpha);
    
    return output;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4f {
    let d2 = dot(input.uv, input.uv); 
    
    // Discard corners of the circle
    if (d2 > 1.0) {
        discard;
    }

    let dist = sqrt(d2);
    let falloff = 1.0 - dist;
    
    let halo = pow(falloff, 2.5);
    let core = smoothstep(0.7, 1.0, falloff);
    let energyGlow = 1.0 + min(input.speed * 2.0, 1.0);
    
    let alphaShape = halo * params.opacity * input.color.a;
    let finalColor = mix(input.color.rgb, vec3f(1.0), core * 0.6) * energyGlow;

    if (alphaShape < 0.01) {
        discard;
    }

    return vec4f(finalColor * alphaShape, alphaShape);
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
    
    private particleBuffer: GPUBuffer | null = null;
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

        this.createParticleBuffer();
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
    }

    private createParticleBuffer() {
        if (!this.device) return;
        const count = this.particleCount;
        const data = new Float32Array(count * 8); 
        for(let i=0; i<count; i++) {
            data[i*8 + 0] = Math.random() * 2 - 1;
            data[i*8 + 1] = Math.random() * 2 - 1;
            data[i*8 + 2] = 0;
            data[i*8 + 3] = 0;
            data[i*8 + 4] = Math.floor(Math.random() * this.numTypes);
        }
        this.particleBuffer = this.device.createBuffer({
            size: data.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
            mappedAtCreation: true,
        });
        new Float32Array(this.particleBuffer.getMappedRange()).set(data);
        this.particleBuffer.unmap();
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
        if (!this.device || !this.computePipeline || !this.pipelineAdditive || !this.particleBuffer || !this.rulesTexture) return;

        this.computeBindGroup = this.device.createBindGroup({
            layout: this.computePipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.particleBuffer } },
                { binding: 1, resource: this.rulesTexture.createView() },
                { binding: 2, resource: { buffer: this.paramsBuffer } },
            ],
        });

        const renderEntries = [
            { binding: 0, resource: { buffer: this.particleBuffer } },
            { binding: 2, resource: { buffer: this.paramsBuffer } },
            { binding: 3, resource: { buffer: this.colorBuffer } },
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
             this.createParticleBuffer();
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
        if (!this.device || !this.particleBuffer) return;
        const count = this.particleCount;
        const data = new Float32Array(count * 8);
        for(let i=0; i<count; i++) {
            data[i*8 + 0] = Math.random() * 2 - 1;
            data[i*8 + 1] = Math.random() * 2 - 1;
            data[i*8 + 2] = 0;
            data[i*8 + 3] = 0;
            data[i*8 + 4] = Math.floor(Math.random() * this.numTypes);
        }
        this.device.queue.writeBuffer(this.particleBuffer, 0, data);
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