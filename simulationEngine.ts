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
const MAX_TYPES = 32u; 

var<workgroup> tile_pos: array<vec2f, BLOCK_SIZE>;
var<workgroup> tile_type: array<u32, BLOCK_SIZE>;

fn hash(p: vec2f) -> f32 {
    return fract(sin(dot(p, vec2f(12.9898, 78.233))) * 43758.5453);
}

fn rand2(seed: vec2f) -> vec2f {
    return vec2f(
        hash(seed + vec2f(1.0, 0.0)),
        hash(seed + vec2f(0.0, 1.0))
    ) * 2.0 - 1.0;
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
    
    // Improved Physics Constants
    let interactionRange = rMax - safeRMin;
    let interactionRangeInv = 1.0 / interactionRange;

    let baseForceFactor = params.forceFactor;
    let growthEnabled = params.growth > 0.5;
    
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

    // --- Optimization: Rule Caching ---
    var myRules: array<f32, 32>;
    let numTypes = u32(params.numTypes);
    
    // Constant loop, usually unrolled by compiler
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
        // Optimization: Fast wrapping using fract
        let rawD = myPos - mousePos;
        let dm = fract(rawD * 0.5 + 0.5) * 2.0 - 1.0;
        
        let distM = length(dm);
        let mouseRadius = 0.4;
        
        if (distM < mouseRadius) {
            let mForce = (1.0 - distM / mouseRadius) * 5.0; 
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
            tile_pos[local_id.x] = vec2f(-1000.0, -1000.0);
            tile_type[local_id.x] = 0u;
        }

        workgroupBarrier(); 

        if (index < particleCount) {
            let limit = min(BLOCK_SIZE, particleCount - i);
            for (var j = 0u; j < BLOCK_SIZE; j++) {
                if (j >= limit) { break; } 

                let otherPos = tile_pos[j];
                let rawD = otherPos - myPos;
                // Fast Wrapping
                let d = fract(rawD * 0.5 + 0.5) * 2.0 - 1.0;
                let d2 = dot(d, d);

                if (d2 > 0.000001 && d2 < rMaxSq) {
                    let invDist = inverseSqrt(d2);
                    let dist = d2 * invDist; 
                    localDensity += (1.0 - dist * rMaxInv);

                    var f = 0.0;
                    if (dist < safeRMin) {
                        // Repulsion: Stronger, but smoother falloff relative to overlap
                        let repulsionStr = 1.0 - (dist / safeRMin);
                        f = -3.0 * repulsionStr;
                        
                        if (growthEnabled && dist < 0.02) {
                             let t = floor(params.time * 2.0); 
                             let randVal = hash(vec2f(f32(j) + t, f32(index)));
                             if (randVal < 0.01) { 
                                 newColor = f32(tile_type[j]); 
                             }
                        }
                    } else {
                        // Attraction/Repulsion: Parabolic envelope for smoother structures
                        // Normalized distance within interaction range (0.0 to 1.0)
                        let q = (dist - safeRMin) * interactionRangeInv;
                        
                        // Parabolic arc: 4 * x * (1 - x). Peak at 0.5.
                        // This prevents harsh cutoffs at max range and min range.
                        let envelope = 4.0 * q * (1.0 - q);
                        
                        let otherType = tile_type[j]; 
                        let ruleVal = myRules[otherType];
                        
                        f = ruleVal * envelope;
                    }
                    force += d * (invDist * f * baseForceFactor);
                }
            }
        }
        workgroupBarrier();
    }

    if (index < particleCount) {
        var p = particles[index]; 
        let speed = length(p.vel);

        // --- Adaptive Friction ---
        // Calculate a "crowding" factor based on local density
        // Increased threshold range so particles can clump tighter before stalling
        let crowding = smoothstep(5.0, 30.0, localDensity);
        
        // Base friction (velocity retention)
        var currentFriction = params.friction;
        
        // 1. Density-based damping: High density -> increased drag
        currentFriction = mix(currentFriction, currentFriction * 0.5, crowding);
        
        // 2. Speed-based damping: Aerodynamic drag limit
        let drag = smoothstep(0.0, 3.0, speed);
        currentFriction = mix(currentFriction, currentFriction * 0.9, drag);

        // --- Adaptive Temperature (Brownian Motion) ---
        if (params.temperature > 0.001) {
             let seed = myPos + vec2f(params.time * 60.0, f32(index) * 0.1337);
             let randDir = rand2(seed); 

             var noiseMag = params.temperature;
             
             // Stagnation boost: Help slow particles move
             let stagnation = 1.0 - smoothstep(0.0, 0.2, speed);
             noiseMag *= (1.0 + stagnation * 2.0);

             // Density boost: Help crowded particles melt
             noiseMag *= (1.0 + crowding * 1.5);
             
             force += randDir * noiseMag;
        }

        // Force Limiting
        let fLen = length(force);
        let maxForce = 10.0; 
        if (fLen > maxForce) {
            force = (force / fLen) * maxForce;
        }

        // Symplectic Euler Integration
        p.vel += force * params.dt;
        p.vel *= currentFriction; 
        p.pos += p.vel * params.dt;
        
        // Boundary Wrapping
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
    
    let w = max(1.0, params.width);
    let h = max(1.0, params.height);
    let aspect = w / h;

    // Soft Particle Geometry
    // We increase the quad size multiplier to 4.0x the physics "size"
    // This provides padding for a soft radial glow without harsh clipping.
    let quadSize = params.size * 4.0; 
    let sizeNDC = (quadSize / w) * 2.0;
    
    var offsets = array<vec2f, 6>(
        vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(-1.0, 1.0),
        vec2f(-1.0, 1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0)
    );
    
    let rawOffset = offsets[vIdx] * sizeNDC;
    var adjustedOffset = rawOffset;
    adjustedOffset.y = rawOffset.y * aspect;

    var output: VertexOutput;
    output.position = vec4f(p.pos + adjustedOffset, 0.0, 1.0);
    output.uv = offsets[vIdx]; // Coordinates from -1.0 to 1.0

    let maxType = i32(params.numTypes) - 1;
    let cType = clamp(i32(round(p.color)), 0, maxType);
    let colorData = colors[cType];
    
    output.color = vec4f(colorData.r, colorData.g, colorData.b, 1.0);
    
    return output;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4f {
    let d2 = dot(input.uv, input.uv); 
    
    // Circular clipping
    if (d2 > 1.0) {
        discard;
    }

    // --- Soft Particle Rendering ---
    // Instead of a sharp Gaussian, we use a cubic polynomial falloff.
    // This creates a "bloom" effect where the particle seems to glow.
    // 1.0 at center, 0.0 at edge.
    
    let falloff = 1.0 - d2;
    // Cubic smoothing (x^3) gives a nice soft edge
    let alphaShape = falloff * falloff * falloff; 

    // Add a hot white core to the center (energy look)
    // Starts fading in at 60% radius squared
    let core = smoothstep(0.6, 1.0, falloff);
    
    let finalColor = mix(input.color.rgb, vec3f(1.0), core * 0.5);
    let finalAlpha = alphaShape * params.opacity;

    // Discard very faint pixels to save fill rate
    if (finalAlpha < 0.01) {
        discard;
    }

    return vec4f(finalColor * finalAlpha, finalAlpha);
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
    // Changed: rulesBuffer is now rulesTexture
    private rulesTexture: GPUTexture | null = null;
    private colorBuffer: GPUBuffer | null = null;
    
    private renderTarget: GPUTexture | null = null;
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
    
    // Rules Texture Config
    // Optimization: Reduced to 32 to match shader
    private readonly MAX_RULE_TEXTURE_SIZE = 32;

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
            primitive: { topology: 'triangle-list' },
        });
        
        // Setup Fade Pipeline
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
        if (this.rulesTexture) {
            this.rulesTexture.destroy();
            this.rulesTexture = null;
        }
    }

    private createParticleBuffer() {
        if (!this.device) return;
        const count = this.particleCount;
        // 8 floats per particle = 32 bytes (16-byte aligned).
        // Format: pos(2), vel(2), color(1), pad(3)
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

        // Ensure texture exists
        if (!this.rulesTexture) {
            this.rulesTexture = this.device.createTexture({
                size: [this.MAX_RULE_TEXTURE_SIZE, this.MAX_RULE_TEXTURE_SIZE],
                format: 'r8snorm', // Signed normalized 8-bit (-1.0 to 1.0)
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
                // Quantize -1.0..1.0 to -127..127
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

        // Expanded to 20 floats (80 bytes)
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

        if (this.renderTarget) this.renderTarget.destroy();
        this.renderTarget = this.device.createTexture({
            size: [scaledWidth, scaledHeight],
            format: format,
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC
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