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

export const DEFAULT_COLORS: ColorDefinition[] = generateColors(64);

export const DEFAULT_PARAMS: SimulationParams = {
    particleCount: 16000,
    friction: 0.82,     
    dt: 0.02,       
    rMax: 0.1,          
    forceFactor: 0.4,   
    minDistance: 0.01,  
    particleSize: 2.0,  
    trails: false,
    dpiScale: 1.0,
    gpuPreference: 'high-performance',
    blendMode: 'additive',
    baseColorOpacity: 0.8,
    numTypes: 64, 
    growth: false,      
    temperature: 1.0,   // Default thermal energy to prevent freezing
    mouseInteractionRadius: 0.3,
    mouseInteractionForce: 5.0, 
};

// Generate random rules so the matrix isn't black on start
const generateRandomRules = (size: number): RuleMatrix => {
    return Array(size).fill(0).map(() => 
        Array(size).fill(0).map(() => (Math.random() * 2 - 1) * 0.5) 
    );
};

export const DEFAULT_RULES: RuleMatrix = generateRandomRules(64);