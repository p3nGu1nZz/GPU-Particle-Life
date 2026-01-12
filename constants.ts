import { ColorDefinition, SimulationParams, RuleMatrix } from './types';

// Updated to match provided configuration
export const DEFAULT_COLORS: ColorDefinition[] = [
  { "r": 249, "g": 31, "b": 31, "name": "#f91f1f" },
  { "r": 31, "g": 95, "b": 249, "name": "#1f5ff9" },
  { "r": 158, "g": 249, "b": 31, "name": "#9ef91f" },
  { "r": 249, "g": 31, "b": 222, "name": "#f91fde" },
  { "r": 31, "g": 249, "b": 213, "name": "#1ff9d5" },
  { "r": 249, "g": 149, "b": 31, "name": "#f9951f" },
  { "r": 86, "g": 31, "b": 249, "name": "#561ff9" },
  { "r": 41, "g": 249, "b": 31, "name": "#29f91f" },
  { "r": 249, "g": 31, "b": 104, "name": "#f91f68" },
  { "r": 31, "g": 168, "b": 249, "name": "#1fa8f9" },
  { "r": 231, "g": 249, "b": 31, "name": "#e7f91f" },
  { "r": 204, "g": 31, "b": 249, "name": "#cc1ff9" },
  { "r": 31, "g": 249, "b": 140, "name": "#1ff98c" },
  { "r": 249, "g": 76, "b": 31, "name": "#f94c1f" },
  { "r": 31, "g": 50, "b": 249, "name": "#1f32f9" },
  { "r": 113, "g": 249, "b": 31, "name": "#71f91f" },
  { "r": 249, "g": 31, "b": 177, "name": "#f91fb1" },
  { "r": 31, "g": 241, "b": 249, "name": "#1ff1f9" },
  { "r": 249, "g": 194, "b": 31, "name": "#f9c21f" },
  { "r": 131, "g": 31, "b": 249, "name": "#831ff9" },
  { "r": 31, "g": 249, "b": 67, "name": "#1ff943" },
  { "r": 249, "g": 31, "b": 59, "name": "#f91f3b" },
  { "r": 31, "g": 123, "b": 249, "name": "#1f7bf9" },
  { "r": 186, "g": 249, "b": 31, "name": "#baf91f" },
  { "r": 249, "g": 31, "b": 249, "name": "#f91ff9" },
  { "r": 31, "g": 249, "b": 185, "name": "#1ff9b9" },
  { "r": 249, "g": 121, "b": 31, "name": "#f9791f" },
  { "r": 58, "g": 31, "b": 249, "name": "#3a1ff9" },
  { "r": 68, "g": 249, "b": 31, "name": "#44f91f" },
  { "r": 249, "g": 31, "b": 132, "name": "#f91f84" },
  { "r": 31, "g": 196, "b": 249, "name": "#1fc4f9" },
  { "r": 249, "g": 239, "b": 31, "name": "#f9ef1f" }
];

export const DEFAULT_PARAMS: SimulationParams = {
    particleCount: 15000,
    friction: 0.82,
    dt: 0.02,
    rMax: 1,
    forceFactor: 0.137,
    minDistance: 0.01,
    particleSize: 0.5,
    trails: true,
    dpiScale: 1.5,
    gpuPreference: 'high-performance',
    blendMode: 'normal',
    baseColorOpacity: 1,
    numTypes: 32,
    growth: true,
    temperature: 0.5,
    mouseInteractionRadius: 0.3,
    mouseInteractionForce: 5
};

// Initialize with empty matrix (zeros)
export const DEFAULT_RULES: RuleMatrix = Array(32).fill(0).map(() => Array(32).fill(0));