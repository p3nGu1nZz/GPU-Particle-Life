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
    // Standard blending (SrcAlpha, OneMinusSrcAlpha) will result in:
    // Dest = Src * 0.15 + Dest * 0.85
    // Since Src is black (0), Dest = Dest * 0.85
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
    pad0: f32,
    pad1: f32,
    pad2: f32,
};

@group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(1) var rulesTex: texture_2d<f32>; 
@group(0) @binding(2) var<uniform> params: Params;

const BLOCK_SIZE = 256u;

var<workgroup> tile_pos: array<vec2f, BLOCK_SIZE>;
var<workgroup> tile_color: array<f32, BLOCK_SIZE>;

fn hash(p: vec2f) -> f32 {
    return fract(sin(dot(p, vec2f(12.9898, 78.233))) * 43758.5453);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3u, @builtin(local_invocation_id) local_id: vec3u) {
    let index = global_id.x;
    let particleCount = u32(params.count);

    let rMax = params.rMax;
    let rMaxSq = rMax * rMax; 
    let rMin = params.minDist; 
    let baseForceFactor = params.forceFactor;
    
    // Adaptation enabled?
    let growthEnabled = params.growth > 0.5;
    
    var myPos = vec2f(0.0);
    var myColor = 0.0;
    var myVel = vec2f(0.0);
    
    if (index < particleCount) {
        let p = particles[index];
        myPos = p.pos;
        myColor = p.color;
        myVel = p.vel;
    }

    let myType = u32(round(myColor));
    var force = vec2f(0.0, 0.0);
    var newColor = myColor; 
    
    // Track stats for "Learning"
    var neighborCount = 0.0;
    var stress = 0.0;
    var dominantNeighborType = -1.0;
    var maxNeighborCount = 0.0;
    
    // We use a small histogram to track neighbor types
    // Since we can't do arrays easily in this loop structure without register pressure,
    // we just track the most influential neighbor encountered.
    var influentialNeighborColor = -1.0;
    var bestInteractionScore = -999.0;

    // --- Tiled N-Body ---
    for (var i = 0u; i < particleCount; i += BLOCK_SIZE) {
        let tile_idx = i + local_id.x;
        if (tile_idx < particleCount) {
            let p = particles[tile_idx];
            tile_pos[local_id.x] = p.pos;
            tile_color[local_id.x] = p.color;
        } else {
            tile_pos[local_id.x] = vec2f(-1000.0, -1000.0);
        }
        workgroupBarrier(); 

        if (index < particleCount) {
            let limit = min(BLOCK_SIZE, particleCount - i);
            for (var j = 0u; j < BLOCK_SIZE; j++) {
                
                let otherPos = tile_pos[j];
                var d = otherPos - myPos;
                d = d - 2.0 * round(d / 2.0);
                let d2 = dot(d, d);

                if (d2 > 0.000001 && d2 < rMaxSq) {
                    let dist = sqrt(d2);
                    let invDist = 1.0 / dist;
                    let dir = d * invDist;
                    
                    neighborCount += 1.0;

                    var f = 0.0;
                    
                    // --- Soft Body Physics ---
                    // 1. Core Repulsion (Very Strong, linear)
                    if (dist < rMin) {
                        let push = 1.0 - (dist / rMin);
                        f = -10.0 * push; // Hard core
                        stress += 1.0; // High stress when squeezed
                    } 
                    // 2. Interaction (Lennard-Jones-ish)
                    else {
                        let otherType = u32(round(tile_color[j]));
                        let ruleVal = textureLoad(rulesTex, vec2u(otherType, myType), 0).r;
                        
                        // "Life" curve: Peak attraction at center of range
                        let rNorm = (dist - rMin) / (rMax - rMin); // 0.0 to 1.0
                        let strength = 1.0 - abs(rNorm - 0.5) * 2.0; // Triangle shape 0->1->0
                        
                        f = ruleVal * strength * baseForceFactor;
                        
                        // Learning Logic:
                        // If I am attracted to this neighbor (ruleVal > 0), they are "good".
                        // If I am repelled (ruleVal < 0) but forced close, that is "bad".
                        if (ruleVal > bestInteractionScore) {
                            bestInteractionScore = ruleVal;
                            influentialNeighborColor = tile_color[j];
                        }
                    }

                    force += dir * f;
                }
            }
        }
        workgroupBarrier();
    }

    // --- Mouse Force ---
    if (params.mouseType != 0.0) {
        let mousePos = vec2f(params.mouseX, params.mouseY);
        var dm = myPos - mousePos;
        dm = dm - 2.0 * round(dm / 2.0);
        let d2m = dot(dm, dm);
        let mRad = params.rMax * 2.0;
        if (d2m < mRad * mRad) {
            let dist = sqrt(d2m);
            let push = (1.0 - dist/mRad) * 20.0 * params.mouseType;
            force += (dm/dist) * push;
            stress += 5.0; // Mouse causes panic
        }
    }

    if (index < particleCount) {
        var p = particles[index];
        
        // --- Biological Differentiation (The "Learning") ---
        // If the particle is under high stress (jammed or fighting forces),
        // or if it is lonely (low neighbors), it tries to adapt.
        if (growthEnabled) {
            let adaptationThreshold = 0.02; 
            let randomTick = hash(vec2f(params.time, f32(index)));
            
            // Panic switch: If highly stressed, just change color randomly to break lock
            if (stress > 5.0 && randomTick < 0.01) {
                newColor = f32(u32(randomTick * params.numTypes) % u32(params.numTypes));
            }
            // Social Conformity: If my interactions are bad, become like my best neighbor
            else if (bestInteractionScore > 0.0 && neighborCount > 2.0 && randomTick < 0.005) {
                newColor = influentialNeighborColor;
            }
        }

        // Integration
        p.vel += force * params.dt;
        p.vel *= params.friction;
        p.pos += p.vel * params.dt;
        p.pos = p.pos - 2.0 * round(p.pos / 2.0);
        p.color = newColor;
        p.pad0 = stress; // Store stress for debug/viz if needed
        
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
    opacity: f32, // corresponds to baseColorOpacity
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
fn vs_main(@builtin(vertex_index) vIdx: u32, @builtin(instance_index) iIdx: u32) -> VertexOutput {
    let p = particles[iIdx];
    let w = max(1.0, params.width);
    let h = max(1.0, params.height);
    let aspect = w / h;
    let sizeNDC = (params.size * 3.0 / w) * 2.0; // Larger soft glow
    
    var offsets = array<vec2f, 6>(
        vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(-1.0, 1.0),
        vec2f(-1.0, 1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0)
    );
    let rawOffset = offsets[vIdx] * sizeNDC;
    var adjustedOffset = rawOffset;
    adjustedOffset.y = rawOffset.y * aspect;

    var output: VertexOutput;
    output.position = vec4f(p.pos + adjustedOffset, 0.0, 1.0);
    output.uv = offsets[vIdx];
    
    let maxType = i32(params.numTypes) - 1;
    let cType = clamp(i32(round(p.color)), 0, maxType);
    let c = colors[cType];
    output.color = vec4f(c.r, c.g, c.b, 1.0);
    return output;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4f {
    let d2 = dot(input.uv, input.uv);
    if (d2 > 1.0) { discard; }
    
    // Metaball-like soft falloff
    let falloff = pow(1.0 - d2, 3.0); 
    
    let alpha = falloff * params.opacity;
    return vec4f(input.color.rgb * alpha, alpha);
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
                        // Standard blending: Dst = Src*Alpha + Dst*(1-Alpha)
                        // Src is Black (0,0,0), Alpha is 0.15
                        // Result: Dst = 0 + Dst * 0.85
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
        // Align rows to 256 bytes for writeTexture
        const bytesPerRow = 256; 
        const bufferSize = bytesPerRow * this.MAX_RULE_TEXTURE_SIZE;
        const paddedData = new Int8Array(bufferSize);

        for (let row = 0; row < numTypes; row++) {
            const rowOffset = row * bytesPerRow;
            for (let col = 0; col < numTypes; col++) {
                // Quantize -1.0..1.0 to -127..127
                let val = Math.max(-1, Math.min(1, rules[row][col]));
                paddedData[rowOffset + col] = Math.round(val * 127);
            }
        }

        this.device.queue.writeTexture(
            { texture: this.rulesTexture },
            paddedData,
            { bytesPerRow: bytesPerRow },
            { width: this.MAX_RULE_TEXTURE_SIZE, height: this.MAX_RULE_TEXTURE_SIZE }
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
