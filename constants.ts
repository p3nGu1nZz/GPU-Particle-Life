import { ColorDefinition, SimulationParams, RuleMatrix } from './types';

// Initial Colors
export const DEFAULT_COLORS: ColorDefinition[] = [
    { r: 255, g: 50, b: 80, name: "#ff3250" },      // Neon Red
    { r: 255, g: 230, b: 20, name: "#ffe614" },  // Neon Yellow
    { r: 50, g: 255, b: 80, name: "#32ff50" },    // Neon Green
    { r: 20, g: 200, b: 255, name: "#14c8ff" }     // Neon Blue
];

export const DEFAULT_PARAMS: SimulationParams = {
    particleCount: 4000,
    friction: 0.82, 
    dt: 0.02,       
    rMax: 0.15,     
    forceFactor: 1.0,
    minDistance: 0.03, 
    particleSize: 4,
    trails: false,
    dpiScale: 0.75,
    gpuPreference: 'high-performance',
    blendMode: 'additive',
    baseColorOpacity: 1.0,
    numTypes: 4
};

// Initial balanced matrix
export const DEFAULT_RULES: RuleMatrix = [
    [0.5, -0.2, 0.3, -0.4],
    [-0.3, 0.5, -0.2, 0.3],
    [0.3, -0.3, 0.5, -0.2],
    [-0.2, 0.3, -0.3, 0.5]
];