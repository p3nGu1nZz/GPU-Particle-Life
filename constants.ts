import { ColorDefinition, SimulationParams, RuleMatrix } from './types';

// Helper to convert HSL to RGB
function hslToRgb(h: number, s: number, l: number) {
  s /= 100;
  l /= 100;
  const k = (n: number) => (n + h / 30) % 12;
  const a = s * Math.min(l, 1 - l);
  const f = (n: number) =>
    l - a * Math.max(-1, Math.min(k(n) - 3, Math.min(9 - k(n), 1)));
  return {
    r: Math.round(255 * f(0)),
    g: Math.round(255 * f(8)),
    b: Math.round(255 * f(4)),
  };
}

// Helper to convert RGB to Hex for the UI
function rgbToHex(r: number, g: number, b: number): string {
    return "#" + [r, g, b].map(x => x.toString(16).padStart(2, '0')).join('');
}

const generateColors = (count: number): ColorDefinition[] => {
    const colors: ColorDefinition[] = [];
    for (let i = 0; i < count; i++) {
        // Use a golden ratio offset for better color distribution across 64 types
        const golden_ratio_conjugate = 0.618033988749895;
        const hue = (i * golden_ratio_conjugate * 360) % 360;
        const { r, g, b } = hslToRgb(hue, 95, 55);
        colors.push({ 
            r, g, b, 
            name: rgbToHex(r, g, b)
        });
    }
    return colors;
};

// Increased to 64 for more complex emergent structures
export const DEFAULT_COLORS: ColorDefinition[] = generateColors(64);

export const DEFAULT_PARAMS: SimulationParams = {
    particleCount: 16000,
    friction: 0.86,     // Reduced drag (higher value) to allow particles to orbit instead of collapsing
    dt: 0.02,       
    rMax: 0.15,         // Reduced slightly to limit the number of attracting neighbors
    forceFactor: 0.50,  // Moderate force
    minDistance: 0.07,  // Increased core size to enforce spacing
    particleSize: 2.5,  
    trails: false,
    dpiScale: 1.0,
    gpuPreference: 'high-performance',
    blendMode: 'additive',
    baseColorOpacity: 1.0,
    numTypes: 64, 
    growth: true,
    mouseInteractionRadius: 0.3,
    mouseInteractionForce: 5.0, 
};

// Generate a blank matrix (all zeros)
const generateBlankRules = (size: number): RuleMatrix => {
    return Array(size).fill(0).map(() => Array(size).fill(0));
};

export const DEFAULT_RULES: RuleMatrix = generateBlankRules(64);