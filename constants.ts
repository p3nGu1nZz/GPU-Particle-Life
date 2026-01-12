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
        // Use Golden Angle (approx 137.5 degrees) for optimal hue distribution
        const hue = (i * 137.508) % 360;

        // Vary Saturation and Lightness to distinctify adjacent colors
        // Cycle saturation: 70, 85, 100
        const s = 70 + (i % 3) * 15; 
        
        // Cycle lightness: 50, 60, 70
        const l = 50 + (i % 3) * 10;

        const { r, g, b } = hslToRgb(hue, s, l);
        colors.push({ 
            r, g, b, 
            name: rgbToHex(r, g, b)
        });
    }
    return colors;
};

// Optimization: Reduced to 32 types for better GPU register usage
export const DEFAULT_COLORS: ColorDefinition[] = generateColors(32);

export const DEFAULT_PARAMS: SimulationParams = {
    particleCount: 15000, 
    friction: 0.82,     
    dt: 0.02,       
    rMax: 0.12,          
    forceFactor: 0.4,   
    minDistance: 0.01,  
    particleSize: 1.0,  
    trails: true,
    dpiScale: 1.0,
    gpuPreference: 'high-performance',
    blendMode: 'normal', 
    baseColorOpacity: 0.8,
    numTypes: 32, 
    growth: false,      
    temperature: 0.5,   
    mouseInteractionRadius: 0.3,
    mouseInteractionForce: 5.0, 
};

// Initialize with empty matrix (zeros) so simulation starts clean
export const DEFAULT_RULES: RuleMatrix = Array(32).fill(0).map(() => Array(32).fill(0));