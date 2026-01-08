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
        const hue = (i * 360) / count;
        const { r, g, b } = hslToRgb(hue, 100, 50);
        colors.push({ 
            r, g, b, 
            name: rgbToHex(r, g, b)
        });
    }
    return colors;
};

export const DEFAULT_COLORS: ColorDefinition[] = generateColors(32);

export const DEFAULT_PARAMS: SimulationParams = {
    particleCount: 16000,
    friction: 0.82,     // Increased to preserve momentum (0.82 retained velocity per frame)
    dt: 0.02,       
    rMax: 0.12,         // Significantly reduced for tighter local clusters
    forceFactor: 0.40,  // Increased force to balance smaller radius
    minDistance: 0.04,  // Reduced to allow tighter packing
    particleSize: 2.0,
    trails: false,
    dpiScale: 1.0,
    gpuPreference: 'high-performance',
    blendMode: 'additive',
    baseColorOpacity: 1.0,
    numTypes: 32, 
    growth: true, // Enabled by default
};

// Generate a blank matrix (all zeros)
const generateBlankRules = (size: number): RuleMatrix => {
    return Array(size).fill(0).map(() => Array(size).fill(0));
};

export const DEFAULT_RULES: RuleMatrix = generateBlankRules(32);