
export type RuleMatrix = number[][];

export type GPUPreference = 'default' | 'high-performance' | 'low-power';

export interface SimulationParams {
    particleCount: number;
    friction: number;
    dt: number;
    rMax: number;
    forceFactor: number;
    minDistance: number; // The distance at which repulsion dominates
    particleSize: number; // Visual size in pixels
    trails: boolean; // Whether to clear the screen between frames
    dpiScale: number; // Resolution scale (0.1 - 2.0)
    gpuPreference: GPUPreference;
    blendMode: 'additive' | 'normal';
    baseColorOpacity: number;
    numTypes: number; // Dynamic number of particle types
    growth: boolean; // Biological growth/infection mechanic
    temperature: number; // Entropy/Thermal Noise
    // Interaction fields (handled internally by engine usually, but good to have in type)
    mouseInteractionRadius: number;
    mouseInteractionForce: number;
}

export interface WorkerMessage {
    type: 'INIT' | 'UPDATE_PARAMS' | 'UPDATE_RULES' | 'RESIZE' | 'RESET' | 'PAUSE' | 'RESUME';
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    data?: any;
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    [key: string]: any;
}

export interface ColorDefinition {
    r: number;
    g: number;
    b: number;
    name: string; // Hex string for UI convenience
}