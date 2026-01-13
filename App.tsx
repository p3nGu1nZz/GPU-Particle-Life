import React, { useEffect, useRef, useState, useCallback } from 'react';
import ControlPanel from './components/ControlPanel';
import { DEFAULT_RULES, DEFAULT_PARAMS, DEFAULT_COLORS } from './constants';
import { SimulationParams, RuleMatrix, ColorDefinition, SavedConfiguration } from './types';
import { AlertTriangle } from 'lucide-react';
import { SimulationEngine } from './simulationEngine';

const App: React.FC = () => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const engineRef = useRef<SimulationEngine | null>(null);
  
  const [isSupported, setIsSupported] = useState<boolean>(true);
  const [fps, setFps] = useState<number>(0);
  const [params, setParams] = useState<SimulationParams>(DEFAULT_PARAMS);
  const [rules, setRules] = useState<RuleMatrix>(DEFAULT_RULES);
  const [colors, setColors] = useState<ColorDefinition[]>(DEFAULT_COLORS);
  
  const [isPaused, setIsPaused] = useState(false);
  const [isMutating, setIsMutating] = useState(false); 
  const [mutationRate, setMutationRate] = useState(0.01);
  const [driftRate, setDriftRate] = useState(1.0);
  const [driftStrength, setDriftStrength] = useState(0.02);
  const [activeGpuPreference, setActiveGpuPreference] = useState(DEFAULT_PARAMS.gpuPreference);

  // Initialize WebGPU Engine
  useEffect(() => {
    if (!canvasRef.current) return;
    
    // Check support
    if (!(navigator as any).gpu) {
      setIsSupported(false);
      return;
    }

    // Cleanup previous engine if exists
    if (engineRef.current) {
        engineRef.current.destroy();
    }

    const engine = new SimulationEngine(canvasRef.current, (currentFps) => {
        setFps(currentFps);
    });
    engineRef.current = engine;

    const initEngine = async () => {
        try {
            // Set initial size
            canvasRef.current!.width = window.innerWidth;
            canvasRef.current!.height = window.innerHeight;

            await engine.init(params, rules, colors);
        } catch (err) {
            console.error("WebGPU Init Failed:", err);
            setIsSupported(false);
        }
    };

    initEngine();

    const handleResize = () => {
        if (!engineRef.current) return;
        engineRef.current.resize(window.innerWidth, window.innerHeight);
    };

    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      engineRef.current?.destroy();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeGpuPreference]); 

  // Update logic when Color count changes
  useEffect(() => {
      const numColors = colors.length;
      const currentRuleSize = rules.length;
      
      // Update Params to reflect new type count
      if (params.numTypes !== numColors) {
        setParams(p => ({ ...p, numTypes: numColors }));
      }
      
      // Resize Matrix if needed
      if (numColors !== currentRuleSize) {
          let newRules = [...rules];
          
          if (numColors > currentRuleSize) {
              // Add Rows/Cols
              const diff = numColors - currentRuleSize;
              for (let i = 0; i < diff; i++) {
                  // Add 0 to existing rows
                  newRules.forEach(row => row.push(0));
                  // Add new row of 0s
                  newRules.push(new Array(numColors).fill(0));
              }
          } else {
              // Remove Rows/Cols
               newRules = newRules.slice(0, numColors).map(row => row.slice(0, numColors));
          }
          setRules(newRules);
          
          // Force reset particles when changing types to redistribute 
          setTimeout(() => engineRef.current?.reset(), 0);
      }
      
      engineRef.current?.updateColors(colors);
      // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [colors]);

  // Sync Params
  useEffect(() => {
    if (params.gpuPreference !== activeGpuPreference) {
        setActiveGpuPreference(params.gpuPreference);
    } else {
        engineRef.current?.updateParams(params);
    }
  }, [params, activeGpuPreference]);

  // Sync Rules
  useEffect(() => {
    engineRef.current?.updateRules(rules);
  }, [rules]);

  // Pause/Resume
  useEffect(() => {
    engineRef.current?.setPaused(isPaused);
  }, [isPaused]);

  // --- Advanced Evolutionary Algorithm ---
  useEffect(() => {
      if (isPaused) return;

      const evolutionInterval = setInterval(() => {
          setRules(prevRules => {
              const size = prevRules.length;
              // Deep copy
              const nextRules = prevRules.map(row => [...row]); 

              if (isMutating) {
                  // --- Active Evolution (High Pressure) ---
                  const numMutations = Math.max(1, Math.floor(size * size * mutationRate)); 

                  for (let k = 0; k < numMutations; k++) {
                      // Select random cell
                      const i = Math.floor(Math.random() * size);
                      const j = Math.floor(Math.random() * size);
                      if (i === 0 || j === 0) continue; // Protect food

                      const strategy = Math.random();
                      
                      if (strategy < 0.1) {
                          // Strategy: Gene Transfer (Copy behavior from another type)
                          // Takes the reaction of Type X to Type J and applies it to Type I
                          const sourceRow = Math.floor(Math.random() * size);
                          // Lerp towards the source behavior (Inheritance)
                          nextRules[i][j] = nextRules[i][j] * 0.8 + nextRules[sourceRow][j] * 0.2;
                      } else if (strategy < 0.2) {
                          // Strategy: Enforce Symmetry (Bonding)
                          // A attracts B, so B should attract A
                          nextRules[j][i] = nextRules[i][j];
                      } else if (strategy < 0.3) {
                          // Strategy: Invert (Predator/Prey flipping)
                          nextRules[i][j] *= -1;
                      } else {
                          // Strategy: Standard Mutation
                          nextRules[i][j] += (Math.random() - 0.5) * 0.15;
                      }
                  }
              } else {
                  // --- Genetic Drift (Background Evolution) ---
                  // Applies subtle variations to keep the system "breathing"
                  if (Math.random() < driftRate) {
                      const i = Math.floor(Math.random() * size);
                      const j = Math.floor(Math.random() * size);
                      
                      if (i !== 0 && j !== 0) {
                          // Small nudge
                          const delta = (Math.random() - 0.5) * driftStrength;
                          nextRules[i][j] += delta;
                          
                          // Occasional "Sympathetic Drift" - adjust the reciprocal slightly too
                          if (Math.random() < 0.3) {
                              nextRules[j][i] += delta * 0.5;
                          }
                      }
                  }
              }
              
              // Clamp values to valid physics range [-1.0, 1.0]
              for (let r = 0; r < size; r++) {
                for (let c = 0; c < size; c++) {
                    // Keep "Food" (Type 0) inert
                    if (r === 0 || c === 0) nextRules[r][c] = 0;
                    
                    if (nextRules[r][c] > 1.0) nextRules[r][c] = 1.0;
                    if (nextRules[r][c] < -1.0) nextRules[r][c] = -1.0;
                }
            }

              return nextRules;
          });
      }, 200); 

      return () => clearInterval(evolutionInterval);
  }, [isMutating, isPaused, mutationRate, driftRate, driftStrength]);

  const handleGenerateOrganisms = useCallback(() => {
    const num = colors.length;
    // Advanced Organism Generation
    // Concept: Differentiated tissues with specific mechanical properties.
    // Rules:
    // 1. Nutrients (Type 0): Inert background, eaten by specific "feeder" types.
    // 2. Tissue (Type i): Strongly binds to itself. Higher types are "tougher" (skin).
    // 3. Skeleton (Type i, i+1): Binds to immediate neighbors to form chains.
    // 4. Stiffness (Type i, i+2): Repels second neighbors to prevent chain collapse.
    // 5. Segregation (Type i, i>2): Strong repulsion to keep different tissues distinct.
    
    const newRules = Array(num).fill(0).map((_, row) => 
         Array(num).fill(0).map((_, col) => {
             // --- 1. Nutrient / Background Interactions ---
             if (row === 0) {
                 // Nutrients are slightly repelled by everything to maintain ambient pressure
                 // This prevents them from clumping inside organisms excessively
                 return -0.05; 
             }
             if (col === 0) {
                 // "Mouths": Low index types (1-3) attract nutrients for growth/metabolism
                 // This causes organisms to orient towards food
                 if (row <= 3) return 0.5; 
                 return 0.0; 
             }

             const dist = col - row;
             const absDist = Math.abs(dist);

             // --- 2. Self-Cohesion (Tissue Integrity) ---
             if (absDist === 0) {
                 // Higher index types (Skin/Shell) are tougher (higher cohesion)
                 // Lower index types (Organs) are softer
                 return 0.6 + (row / num) * 0.4; 
             }

             // --- 3. Polymer Chain (Skeleton) ---
             if (absDist === 1) {
                 // Strong bond with immediate neighbors
                 // Asymmetry: Pull harder on the next type to create "Head-Tail" directionality
                 // If col > row (Next), pull strong. If col < row (Prev), pull weak.
                 return (dist > 0) ? 0.7 : 0.3;
             }

             // --- 4. Structural Stiffness (Angle limitation) ---
             if (absDist === 2) {
                 // Weak repulsion prevents the chain from folding back perfectly on itself
                 // This creates semi-rigid "bones" or "membranes"
                 return -0.2;
             }
             
             // --- 5. Complex Distant Interactions ---
             // Varied attraction/repulsion to encourage folding and non-linear shapes
             
             // Secondary Folding: Weak attraction to allow chains to fold back
             if (absDist === 3) return 0.1;

             // Structural Loops: Connect distant parts to form rings/compartments
             if (absDist === 5 || absDist === 8) return 0.15;

             // Distant Repulsion Gradient
             // Very distinct types repel strongly to keep "organs" separate
             if (absDist > num / 2) return -0.6;

             // Varied background repulsion based on parity
             // Prevents perfect crystallization, encourages organic fuzziness
             return (absDist % 2 === 0) ? -0.3 : -0.4;
         })
    );
    
    setRules(newRules);
    
    setParams(p => ({
        ...p,
        friction: 0.82, // High friction for stable structures (viscous environment)
        dt: 0.02,
        forceFactor: 1.8, // Strong bonding forces
        rMax: 0.18, // Extended radius to allow complex multi-neighbor interactions
        minDistance: 0.04, // Distinct particles
        growth: true, // Enable metabolic cycle
        temperature: 0.2 // Some thermal noise to vibrate structures into place
    }));
    
    // Reset to distribute types randomly so they can self-assemble
    setTimeout(() => engineRef.current?.reset(), 100);
  }, [colors.length]);

  const handleRandomizeRules = useCallback(() => {
    const num = colors.length;
    const newRules = Array(num).fill(0).map((_, y) => 
         Array(num).fill(0).map((_, x) => (Math.random() * 2 - 1))
    );
    setRules(newRules);
  }, [colors.length]);

  const handleReset = useCallback(() => {
    engineRef.current?.reset();
  }, []);

  const handleLoadPreset = useCallback((config: SavedConfiguration) => {
    if (config.params && config.rules && config.colors) {
        setParams(config.params);
        setRules(config.rules);
        setColors(config.colors);
        setTimeout(() => engineRef.current?.reset(), 100);
    } else {
        alert("Invalid configuration file");
    }
  }, []);

  const toggleFullscreen = useCallback(() => {
      if (!document.fullscreenElement) {
          document.documentElement.requestFullscreen().catch(e => {
              console.error(`Error attempting to enable fullscreen mode: ${e.message} (${e.name})`);
          });
      } else {
          if (document.exitFullscreen) {
              document.exitFullscreen();
          }
      }
  }, []);

  if (!isSupported) {
    return (
      <div className="flex items-center justify-center h-screen bg-neutral-900 text-white p-8">
        <div className="max-w-md text-center space-y-4">
          <AlertTriangle className="w-16 h-16 text-red-500 mx-auto" />
          <h1 className="text-2xl font-bold">WebGPU Not Supported</h1>
          <p className="text-neutral-400">
            Your browser does not support WebGPU, or something went wrong initializing the simulation.
            Please ensure you are using a compatible browser (Chrome/Edge 113+).
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="relative w-full h-screen bg-black overflow-hidden font-sans selection:bg-emerald-500 selection:text-white">
      {/* Simulation Canvas */}
      <canvas 
        ref={canvasRef} 
        className="absolute inset-0 block w-full h-full outline-none"
      />

      {/* Control Panel */}
      <ControlPanel 
        params={params} 
        setParams={setParams} 
        rules={rules} 
        setRules={setRules}
        colors={colors}
        setColors={setColors}
        isPaused={isPaused}
        setIsPaused={setIsPaused}
        onReset={handleReset}
        onRandomize={handleGenerateOrganisms}
        onLoadPreset={handleLoadPreset}
        fps={fps}
        toggleFullscreen={toggleFullscreen}
        isMutating={isMutating}
        setIsMutating={setIsMutating}
        mutationRate={mutationRate}
        setMutationRate={setMutationRate}
        driftRate={driftRate}
        setDriftRate={setDriftRate}
        driftStrength={driftStrength}
        setDriftStrength={setDriftStrength}
      />
    </div>
  );
};

export default App;