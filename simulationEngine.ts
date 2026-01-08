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
    pad2: f32,
};

@group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(1) var<storage, read> rules: array<f32>; 
@group(0) @binding(2) var<uniform> params: Params;

// Performance Optimization: 
// Workgroup size 256 is generally more efficient for occupancy.
// We use Shared Memory (var<workgroup>) to reduce Global Memory traffic.
const BLOCK_SIZE = 256u;

var<workgroup> tile_pos: array<vec2f, BLOCK_SIZE>;
var<workgroup> tile_color: array<f32, BLOCK_SIZE>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3u, @builtin(local_invocation_id) local_id: vec3u) {
    let index = global_id.x;
    let particleCount = u32(params.count);

    var myPos = vec2f(0.0);
    var myColor = 0.0;
    
    // 1. Load Self Data
    if (index < particleCount) {
        let p = particles[index];
        myPos = p.pos;
        myColor = p.color;
    }

    // 2. Pre-load Rules into Registers
    // Optimization: Cache row of rules for this particle type to avoid N lookups in storage buffer
    let numTypes = i32(params.numTypes);
    let myType = i32(round(myColor));
    var myRules: array<f32, 16>; // Supports up to 16 types
    
    if (index < particleCount) {
        for (var t = 0; t < numTypes; t++) {
            // rules flat index: row * width + col
            myRules[t] = rules[myType * numTypes + t];
        }
    }

    var force = vec2f(0.0, 0.0);

    // 3. Tiled N-Body Calculation
    // Iterate over particles in chunks (tiles) of BLOCK_SIZE
    for (var i = 0u; i < particleCount; i += BLOCK_SIZE) {
        
        // --- Phase A: Collaborative Loading into Shared Memory ---
        let tile_idx = i + local_id.x;
        if (tile_idx < particleCount) {
            let p = particles[tile_idx];
            tile_pos[local_id.x] = p.pos;
            tile_color[local_id.x] = p.color;
        } else {
            // Pad with dummy data far away so it doesn't affect force
            tile_pos[local_id.x] = vec2f(-1000.0, -1000.0);
            tile_color[local_id.x] = 0.0;
        }

        workgroupBarrier(); // Wait for all threads to load the tile

        // --- Phase B: Compute Forces using Shared Memory ---
        if (index < particleCount) {
            // How many particles in this tile are valid?
            let limit = min(BLOCK_SIZE, particleCount - i);
            
            for (var j = 0u; j < BLOCK_SIZE; j++) {
                if (j >= limit) { break; } // Stop if we hit end of valid data in last tile

                let otherPos = tile_pos[j];
                var d = otherPos - myPos;

                // Torus Wrap
                if (d.x > 1.0) { d.x -= 2.0; }
                if (d.x < -1.0) { d.x += 2.0; }
                if (d.y > 1.0) { d.y -= 2.0; }
                if (d.y < -1.0) { d.y += 2.0; }

                let dist = length(d);

                if (dist > 0.0 && dist < params.rMax) {
                    let otherType = i32(round(tile_color[j]));
                    let ruleVal = myRules[otherType]; // Register lookup
                    
                    var f = 0.0;
                    if (dist < params.minDist) {
                        f = -1.0 * (1.0 - dist / params.minDist) * 3.0; 
                    } else {
                        let numer = abs(2.0 * dist - params.rMax - params.minDist);
                        let denom = params.rMax - params.minDist;
                        f = ruleVal * (1.0 - numer / denom);
                    }

                    force += (d / dist) * f * params.forceFactor;
                }
            }
        }
        
        workgroupBarrier(); // Wait for computation before overwriting shared memory with next tile
    }

    // 4. Update Physics
    if (index < particleCount) {
        var p = particles[index];
        p.vel = (p.vel + force * params.dt) * params.friction;
        p.pos += p.vel * params.dt;

        // Wrap Position
        if (p.pos.x > 1.0) { p.pos.x -= 2.0; }
        if (p.pos.x < -1.0) { p.pos.x += 2.0; }
        if (p.pos.y > 1.0) { p.pos.y -= 2.0; }
        if (p.pos.y < -1.0) { p.pos.y += 2.0; }

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
    pad2: f32,
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
    
    // Calculate Size
    let sizeNDC = (params.size / params.width) * 2.0;
    
    var offsets = array<vec2f, 6>(
        vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(-1.0, 1.0),
        vec2f(-1.0, 1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0)
    );
    
    let offset = offsets[vIdx] * sizeNDC;
    
    // Correct aspect ratio
    let w = select(params.width, 1.0, params.width == 0.0);
    let h = select(params.height, 1.0, params.height == 0.0);
    let aspect = w / h;
    
    var adjustedOffset = offset;
    adjustedOffset.y = adjustedOffset.y * aspect;

    let pos = p.pos + adjustedOffset;

    var output: VertexOutput;
    output.position = vec4f(pos, 0.0, 1.0);
    output.uv = offsets[vIdx]; 

    // Dynamic Color Fetch
    let cType = i32(round(p.color));
    // Retrieve color from buffer. Normalized 0-1 from CPU
    let colorData = colors[cType];
    output.color = vec4f(colorData.r, colorData.g, colorData.b, 1.0);
    
    return output;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4f {
    let d = length(input.uv);
    if (d > 1.0) {
        discard;
    }
    // Harder core, softer glow
    let alpha = max(0.0, 1.0 - d);
    let glow = pow(alpha, 1.5) * params.opacity; 
    
    // Premultiplied Alpha Output
    return vec4f(input.color.rgb * glow, glow);
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

    constructor(canvas: HTMLCanvasElement, onFpsUpdate?: (fps: number) => void) {
        this.canvas = canvas;
        this.onFpsUpdate = onFpsUpdate;
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

        // 1. Create Buffers
        this.createParticleBuffer();
        this.createRulesBuffer(rules);
        this.createParamsBuffer(params);
        this.createColorBuffer(colors);

        // 2. Create Pipelines
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

        // 3. Create BindGroups
        this.updateBindGroups();

        // 4. Start Loop
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
            data[i*8 + 4] = Math.floor(Math.random() * this.numTypes); // Random type based on current numTypes
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
        // 4 floats per color (r,g,b,a)
        const data = new Float32Array(colors.length * 4);
        for(let i=0; i<colors.length; i++) {
            data[i*4 + 0] = colors[i].r / 255.0;
            data[i*4 + 1] = colors[i].g / 255.0;
            data[i*4 + 2] = colors[i].b / 255.0;
            data[i*4 + 3] = 1.0; 
        }

        // We re-create this buffer if size changes, so typical destroy pattern if needed
        // but for now we assume it's created during updateColors or init
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
            0 // Padding
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
        if (!this.device || !this.computePipeline || !this.pipelineAdditive || !this.pipelineNormal || !this.particleBuffer || !this.rulesBuffer || !this.paramsBuffer || !this.colorBuffer) return;

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
            layout: this.pipelineNormal.getBindGroupLayout(0),
            entries: renderEntries,
        });
    }

    public updateColors(colors: ColorDefinition[]) {
        if(!this.device) return;
        
        // If color count changed, we might need to recreate buffer if it's larger
        // But for simplicity, let's just recreate it every time logic changes
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

        // Check if NumTypes changed (requires particle re-init or just shader update)
        // Note: The caller (App.tsx) handles resetting particles if numTypes changes drastically
        // Here we just update the param
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

        this.device.queue.writeBuffer(this.paramsBuffer, 0, this.paramsData);
        
        if(needsBindGroupUpdate) this.updateBindGroups();
    }

    public updateRules(rules: RuleMatrix) {
        if (!this.device) return;
        const flat = new Float32Array(rules.flat());
        // Recreate buffer if size changed
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

        const commandEncoder = this.device.createCommandEncoder();

        // 1. Compute Pass
        const computePass = commandEncoder.beginComputePass();
        computePass.setPipeline(this.computePipeline);
        computePass.setBindGroup(0, this.computeBindGroup!);
        
        // Updated dispatch size for workgroup_size 256
        computePass.dispatchWorkgroups(Math.ceil(this.particleCount / 256));
        
        computePass.end();

        // 2. Render Pass to Offscreen Texture
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

        // 3. Copy Offscreen Texture to Swap Chain (Canvas)
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