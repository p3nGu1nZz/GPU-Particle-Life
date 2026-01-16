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
  const [hasInitialized, setHasInitialized] = useState(false);

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

  // --- Advanced Organism Generation ---
  // This constructs a "Life Matrix" designed to create chains, membranes, and internal pressures.
  const handleGenerateOrganisms = useCallback(() => {
    const num = colors.length;
    
    const newRules = Array(num).fill(0).map((_, row) => 
         Array(num).fill(0).map((_, col) => {
             const dist = col - row;
             const absDist = Math.abs(dist);

             // --- 1. Food (Type 0) Interactions ---
             if (row === 0) {
                 // Food is slightly repelled by everything (pressure)
                 return -0.05; 
             }
             if (col === 0) {
                 // "Mouths": Low index types (1-4) attract food
                 if (row <= 4) return 0.8; 
                 return 0.0; 
             }

             // --- 2. Self-Clustering (Tissue Integrity) ---
             if (absDist === 0) {
                 // Cohesion. Higher index = tougher skin
                 return 0.4 + (row / num) * 0.6; 
             }

             // --- 3. Chains & Skeletons (Nearest Neighbor) ---
             if (absDist === 1) {
                 // Strong bonds to adjacent types create gradient chains
                 // Asymmetry creates directional flow/movement
                 return (dist > 0) ? 0.6 : 0.2;
             }

             // --- 4. Structural Stiffness (Second Neighbor) ---
             if (absDist === 2) {
                 // Repulsion to prevent chains from collapsing into points
                 return -0.3;
             }
             
             // --- 5. Complex Folding ---
             if (absDist === 3) return 0.1; // Weak attraction for folding
             
             // --- 6. Global Separation ---
             // Distant types repel to keep organs distinct
             if (absDist > 4) return -0.4;

             return -0.1;
         })
    );
    
    setRules(newRules);
    
    // Set optimal parameters for this life configuration
    setParams(p => ({
        ...p,
        friction: 0.94, // Very High friction for stable structures
        dt: 0.02,
        forceFactor: 2.5, // Strong bonding to overcome drift
        rMax: 0.25, // Large interaction radius to encourage complexity
        minDistance: 0.04, 
        growth: true, 
        temperature: 0.05 // Lower temperature to prevent "shaking" death
    }));
    
    // Reset simulation state
    setTimeout(() => engineRef.current?.reset(), 50);
  }, [colors.length]);

  // --- Auto-Start with Complex Organisms ---
  useEffect(() => {
      if (!hasInitialized) {
          handleGenerateOrganisms();
          setHasInitialized(true);
      }
  }, [hasInitialized, handleGenerateOrganisms]);

  // --- Evolutionary Algorithm ---
  useEffect(() => {
      if (isPaused) return;

      const evolutionInterval = setInterval(() => {
          setRules(prevRules => {
              const size = prevRules.length;
              const nextRules = prevRules.map(row => [...row]); 

              if (isMutating) {
                  const numMutations = Math.max(1, Math.floor(size * size * mutationRate)); 

                  for (let k = 0; k < numMutations; k++) {
                      const i = Math.floor(Math.random() * size);
                      const j = Math.floor(Math.random() * size);
                      // Protect food (row 0) and self-identity (diagonal) slightly
                      if (i === 0 || j === 0) continue; 

                      const strategy = Math.random();
                      
                      if (strategy < 0.1) {
                          // Copy neighboring strategy
                          const neighbor = (i + 1) % size;
                          nextRules[i][j] = nextRules[i][j] * 0.9 + nextRules[neighbor][j] * 0.1;
                      } else if (strategy < 0.25) {
                          // Invert interaction
                          nextRules[i][j] *= -0.5;
                      } else {
                          // Standard drift
                          nextRules[i][j] += (Math.random() - 0.5) * 0.2;
                      }
                  }
              } else {
                  // Genetic Drift - Smoother, less destructive
                  if (Math.random() < driftRate) {
                      const i = Math.floor(Math.random() * size);
                      const j = Math.floor(Math.random() * size);
                      
                      if (i !== 0 && j !== 0 && i !== j) {
                          // Bias slightly positive to encourage clumping
                          const bias = 0.01;
                          nextRules[i][j] += (Math.random() - 0.5 + bias) * driftStrength;
                      }
                  }
              }
              
              // Clamp
              for (let r = 0; r < size; r++) {
                for (let c = 0; c < size; c++) {
                    if (r === 0 || c === 0) nextRules[r][c] = 0; // Lock food
                    else if (c === 0 && r <= 4) nextRules[r][c] = 0.8; // Lock mouths
                    
                    if (nextRules[r][c] > 1.0) nextRules[r][c] = 1.0;
                    if (nextRules[r][c] < -1.0) nextRules[r][c] = -1.0;
                }
            }

              return nextRules;
          });
      }, 500); // Slower evolution tick (was 200ms) to allow structures to settle

      return () => clearInterval(evolutionInterval);
  }, [isMutating, isPaused, mutationRate, driftRate, driftStrength]);

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