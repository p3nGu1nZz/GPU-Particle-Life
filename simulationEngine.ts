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

const COMPUTE_SHADER = `
struct Particle {
    pos: vec2f,
    vel: vec2f,
    color: f32,
    pad0: f32,
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
@group(0) @binding(1) var<storage, read> rules: array<f32>; 
@group(0) @binding(2) var<uniform> params: Params;

const BLOCK_SIZE = 256u;

var<workgroup> tile_pos: array<vec2f, BLOCK_SIZE>;
var<workgroup> tile_color: array<f32, BLOCK_SIZE>;

// Pseudo-random hash for noise
fn hash(p: vec2f) -> f32 {
    return fract(sin(dot(p, vec2f(12.9898, 78.233))) * 43758.5453);
}

// 2D Random for thermal noise
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
    let rMaxSq = rMax * rMax; // Pre-calculate square for optimization
    let rMin = params.minDist;
    // Ensure rMin isn't too close to zero to avoid division issues
    let safeRMin = max(0.001, rMin);
    
    let range = max(0.0001, rMax - safeRMin); 
    let mid = (rMax + safeRMin) * 0.5;
    let rangeFactor = 2.0 / range; 
    let rMinInv = 1.0 / safeRMin;
    let baseForceFactor = params.forceFactor;
    let growthEnabled = params.growth > 0.5;

    var myPos = vec2f(0.0);
    var myColor = 0.0;
    
    if (index < particleCount) {
        let p = particles[index];
        myPos = p.pos;
        myColor = p.color;
    }

    let numTypes = i32(params.numTypes);
    let myType = i32(round(myColor));
    
    // --- CACHE OPTIMIZATION START ---
    // Pre-load the specific rule row for this particle's type into a private array (register cache).
    // This avoids accessing the global storage buffer 'rules' inside the hot N-body loop.
    var typeRules: array<f32, 64>;
    let ruleRowOffset = myType * numTypes;
    
    // Unrolled copy (WGSL loop bounds must be uniform or constant ideally, but this works)
    for (var k = 0; k < 64; k++) {
        if (k < numTypes) {
            typeRules[k] = rules[ruleRowOffset + k];
        } else {
            typeRules[k] = 0.0;
        }
    }
    // --- CACHE OPTIMIZATION END ---

    var force = vec2f(0.0, 0.0);
    var newColor = myColor; 
    
    // Adaptive Physics Accumulator
    var localDensity = 0.0; 

    // --- Mouse Interaction ---
    if (params.mouseType != 0.0) {
        let mousePos = vec2f(params.mouseX, params.mouseY);
        var dm = myPos - mousePos;
        
        // Wrap mouse distance (so mouse affects particles across the edge)
        dm = dm - 2.0 * round(dm / 2.0);
        
        let distM = length(dm);
        let mouseRadius = 0.4;
        
        if (distM < mouseRadius) {
            let mForce = (1.0 - distM / mouseRadius) * 5.0; // Strong force
            force += (dm / (distM + 0.001)) * mForce * params.mouseType;
        }
    }

    // --- Tiled N-Body Calculation ---
    for (var i = 0u; i < particleCount; i += BLOCK_SIZE) {
        let tile_idx = i + local_id.x;
        if (tile_idx < particleCount) {
            let p = particles[tile_idx];
            tile_pos[local_id.x] = p.pos;
            tile_color[local_id.x] = p.color;
        } else {
            tile_pos[local_id.x] = vec2f(-1000.0, -1000.0);
            tile_color[local_id.x] = 0.0;
        }

        workgroupBarrier(); 

        if (index < particleCount) {
            let limit = min(BLOCK_SIZE, particleCount - i);
            for (var j = 0u; j < BLOCK_SIZE; j++) {
                if (j >= limit) { break; } 

                let otherPos = tile_pos[j];
                var d = otherPos - myPos;
                d = d - 2.0 * round(d / 2.0);
                
                // --- OPTIMIZATION: Distance Squared Check ---
                // Avoid sqrt() for particles that are too far away.
                let d2 = dot(d, d);

                if (d2 > 0.000001 && d2 < rMaxSq) {
                    let dist = sqrt(d2);
                    
                    // Accumulate Local Density (used for adaptive physics)
                    localDensity += (1.0 - dist / rMax);

                    var f = 0.0;
                    if (dist < safeRMin) {
                        // Repulsion force
                        let relDist = dist * rMinInv;
                        // Smooth but strong repulsion curve
                        f = -4.0 * (1.0 - relDist); 
                        
                        // Growth logic
                        if (growthEnabled && dist < 0.02) {
                             let t = floor(params.time * 2.0); 
                             let randVal = hash(vec2f(f32(j) + t, f32(index)));
                             if (randVal < 0.01) { newColor = tile_color[j]; }
                        }
                    } else {
                        let otherType = i32(round(tile_color[j]));
                        
                        // Optimized Lookup using private cache
                        let safeType = clamp(otherType, 0, numTypes - 1);
                        let ruleVal = typeRules[safeType]; 

                        let strength = 1.0 - abs(dist - mid) * rangeFactor;
                        f = ruleVal * strength;
                    }
                    force += (d / dist) * f * baseForceFactor;
                }
            }
        }
        workgroupBarrier();
    }

    // --- Adaptive Physics Implementation ---
    if (index < particleCount) {
        // Normalize density roughly based on typical neighborhood
        let densityFactor = smoothstep(5.0, 30.0, localDensity);

        // 1. Adaptive Friction (Viscosity)
        // As density increases, friction increases (velocity dampening becomes stronger)
        let viscosity = mix(params.friction, params.friction * 0.8, densityFactor);

        // 2. Adaptive Temperature
        // High density areas generate "pressure heat" -> more noise to break up clumps
        let adaptiveTemp = params.temperature * (1.0 + densityFactor * 0.5);
        let seed = myPos + vec2f(params.time, f32(index) * 0.01);
        let thermalForce = rand2(seed) * adaptiveTemp;
        
        // 3. Pressure Damping
        // If density is very high, reduce the effective total force to prevent explosions
        force *= (1.0 - densityFactor * 0.3);

        force += thermalForce;

        // Force Clamping to prevent jitter/explosion
        let fLen = length(force);
        let maxForce = 20.0; 
        if (fLen > maxForce) {
            force = (force / fLen) * maxForce;
        }

        var p = particles[index];
        p.vel = (p.vel + force * params.dt) * viscosity;
        p.pos += p.vel * params.dt;
        p.color = newColor; 
        p.pos = p.pos - 2.0 * round(p.pos / 2.0);
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
    
    // Safety check for width/height
    let w = max(1.0, params.width);
    let h = max(1.0, params.height);
    let aspect = w / h;

    let sizeNDC = (params.size / w) * 2.0;
    
    // Standard Quad
    var offsets = array<vec2f, 6>(
        vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(-1.0, 1.0),
        vec2f(-1.0, 1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0)
    );
    
    let rawOffset = offsets[vIdx] * sizeNDC;
    
    var adjustedOffset = rawOffset;
    adjustedOffset.y = rawOffset.y * aspect;

    let pos = p.pos + adjustedOffset;

    var output: VertexOutput;
    output.position = vec4f(pos, 0.0, 1.0);
    output.uv = offsets[vIdx];

    // Safely clamp color index to prevent out-of-bounds crash
    let maxType = i32(params.numTypes) - 1;
    let cType = clamp(i32(round(p.color)), 0, maxType);
    
    let colorData = colors[cType];
    output.color = vec4f(colorData.r, colorData.g, colorData.b, 1.0);
    
    return output;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4f {
    let d2 = dot(input.uv, input.uv); 
    
    // Instead of discard, we can just return transparent
    if (d2 > 1.0) {
        return vec4f(0.0, 0.0, 0.0, 0.0);
    }
    
    // Enhanced Visuals: "Glowing Orb"
    let core = exp(-8.0 * d2);
    let halo = exp(-2.0 * d2);
    let intensity = max(core, halo * 0.5);
    
    // Anti-aliasing
    let edgeFade = 1.0 - smoothstep(0.85, 1.0, d2);
    
    // Final alpha
    let alpha = intensity * edgeFade * params.opacity;

    // Hot Core Coloring
    let whiteness = smoothstep(0.5, 1.0, core);
    let finalRgb = mix(input.color.rgb, vec3f(1.0, 1.0, 1.0), whiteness * 0.4);
    
    // Pre-multiplied Alpha Output
    return vec4f(finalRgb * alpha, alpha);
}
`;

export class SimulationEngine {
    private canvas: HTMLCanvasElement;
    private device: GPUDevice | null = null;
    private context: GPUCanvasContext | null = null;
    
    private pipelineAdditive: GPURenderPipeline | null = null;
    private pipelineNormal: GPURenderPipeline | null = null;
    
    private computePipeline: GPUComputePipeline | null = null;
    
    private particleBuffer: GPUBuffer | null = null;
    private paramsBuffer: GPUBuffer | null = null;
    private rulesBuffer: GPUBuffer | null = null;
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
        this.createRulesBuffer(rules);
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

    private createRulesBuffer(rules: RuleMatrix) {
        if (!this.device) return;
        const flat = new Float32Array(rules.flat());
        this.rulesBuffer = this.device.createBuffer({
            size: flat.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            mappedAtCreation: true,
        });
        new Float32Array(this.rulesBuffer.getMappedRange()).set(flat);
        this.rulesBuffer.unmap();
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
        if (!this.device || !this.computePipeline || !this.pipelineAdditive || !this.particleBuffer) return;

        this.computeBindGroup = this.device.createBindGroup({
            layout: this.computePipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.particleBuffer } },
                { binding: 1, resource: { buffer: this.rulesBuffer } },
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
        if (!this.device) return;
        const flat = new Float32Array(rules.flat());
        if (this.rulesBuffer && this.rulesBuffer.size !== flat.byteLength) {
            this.createRulesBuffer(rules);
            this.updateBindGroups();
        } else if (this.rulesBuffer) {
            this.device.queue.writeBuffer(this.rulesBuffer, 0, flat);
        }
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
        
        if (this.isPaused || !this.device || !this.context || !this.computePipeline || !this.pipelineAdditive || !this.renderTarget) return;

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

        const shouldClear = this.textureNeedsClear || !this.trailsEnabled;
        const renderPass = commandEncoder.beginRenderPass({
            colorAttachments: [{
                view: this.renderTarget.createView(),
                clearValue: { r: 0.0, g: 0.0, b: 0.0, a: 1.0 },
                loadOp: shouldClear ? 'clear' : 'load',
                storeOp: 'store',
            }],
        });
        
        this.textureNeedsClear = false;
        const pipeline = this.blendMode === 'normal' ? this.pipelineNormal! : this.pipelineAdditive!;
        const bindGroup = this.blendMode === 'normal' ? this.renderBindGroupNormal! : this.renderBindGroupAdditive!;
        renderPass.setPipeline(pipeline);
        renderPass.setBindGroup(0, bindGroup);
        renderPass.draw(6, this.particleCount);
        renderPass.end();

        const currentTexture = this.context.getCurrentTexture();
        commandEncoder.copyTextureToTexture(
            { texture: this.renderTarget },
            { texture: currentTexture },
            [this.canvas.width, this.canvas.height]
        );

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
