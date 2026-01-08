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
    // Biological Palette: mix of cyans, greens, pinks, and oranges
    // Avoid pure primary blues/reds to look more organic
    for (let i = 0; i < count; i++) {
        const hue = (i * 137.5) % 360; // Golden angle for natural distribution
        const sat = 70 + Math.random() * 30;
        const light = 50 + Math.random() * 20;
        const { r, g, b } = hslToRgb(hue, sat, light);
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
    friction: 0.90,       // High friction for thick fluid feel
    dt: 0.015,            
    rMax: 0.12,           
    forceFactor: 0.8,     
    minDistance: 0.03,    
    particleSize: 3.0,    // Slightly smaller for cleaner swarms
    trails: true,         // ON for biological slime trails
    dpiScale: 1.0,
    gpuPreference: 'high-performance',
    blendMode: 'additive',
    baseColorOpacity: 0.1, // LOW opacity to prevent white screen with additive blending
    numTypes: 12,         
    growth: true,         
    temperature: 0.5,     
    mouseInteractionRadius: 0.3,
    mouseInteractionForce: 5.0, 
};

// Generate "Life-Like" Rules
const generateLifeRules = (size: number): RuleMatrix => {
    const rules = Array(size).fill(0).map(() => Array(size).fill(0));
    
    for (let i = 0; i < size; i++) {
        for (let j = 0; j < size; j++) {
            if (i === j) {
                // Self-interaction: slight attraction to form tissues
                rules[i][j] = 0.2 + Math.random() * 0.3; 
            } else {
                // Rock-Paper-Scissors-Lizard-Spock dynamics
                // This creates "food chains" which result in chasing/escaping behavior
                const diff = (j - i + size) % size;
                const dist = Math.abs(diff - size/2);
                
                if (diff > 0 && diff < 3) {
                    // i is chased by j (Repulsion)
                    rules[i][j] = -0.5 - Math.random() * 0.5;
                } else if (diff >= 3 && diff < 5) {
                    // i chases j (Attraction)
                    rules[i][j] = 0.3 + Math.random() * 0.5;
                } else {
                    // Neutral / Random weak interaction
                    rules[i][j] = (Math.random() - 0.5) * 0.2;
                }
            }
        }
    }
    return rules;
};

export const DEFAULT_RULES: RuleMatrix = generateLifeRules(12);