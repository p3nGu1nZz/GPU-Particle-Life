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
  const [isMutating, setIsMutating] = useState(true); 
  const [mutationRate, setMutationRate] = useState(0.02); 
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

  // --- Improved Evolutionary Algorithm ---
  // Uses interaction archetypes (Attract, Repel, Chase) to evolve the matrix
  // towards interesting particle behaviors rather than random noise.
  useEffect(() => {
      if (!isMutating || isPaused) return;

      const evolutionInterval = setInterval(() => {
          setRules(prevRules => {
              const size = prevRules.length;
              // Deep copy to avoid mutating state directly
              const nextRules = prevRules.map(row => [...row]); 

              // Determine number of mutations based on rate
              // e.g., 0.02 * 1024 cells = ~20 cell updates (10 pairs) per tick
              const numPairs = size * size;
              const updatesCount = Math.ceil(numPairs * mutationRate); 

              for (let k = 0; k < updatesCount; k++) {
                  // Pick a random pair (I, J)
                  const i = Math.floor(Math.random() * size);
                  const j = Math.floor(Math.random() * size);

                  // 1. Interaction Strategy Mutation
                  // We apply forces in coupled pairs to create structured behavior
                  
                  const strategy = Math.random();
                  const strength = 0.1; // Strength of the mutation nudge

                  if (strategy < 0.4) {
                      // SYMMETRY (Binding/Crystalizing)
                      // Tend towards I and J having the same feeling about each other (Cluster)
                      // Pull them towards their average, then nudge up or down
                      const avg = (nextRules[i][j] + nextRules[j][i]) / 2;
                      const nudge = (Math.random() - 0.5) * strength;
                      const target = avg + nudge;
                      
                      // Blend towards target
                      nextRules[i][j] = nextRules[i][j] * 0.9 + target * 0.1;
                      nextRules[j][i] = nextRules[j][i] * 0.9 + target * 0.1;
                  } 
                  else if (strategy < 0.7) {
                      // ASYMMETRY (Flow/Chase)
                      // Tend towards I and J having opposite feelings (Predator/Prey)
                      // I likes J, J dislikes I
                      nextRules[i][j] += strength * 0.5;
                      nextRules[j][i] -= strength * 0.5;
                  } 
                  else if (strategy < 0.9) {
                      // REPULSION (Spacing)
                      // Both dislike each other
                      nextRules[i][j] -= strength * 0.5;
                      nextRules[j][i] -= strength * 0.5;
                  }
                  else {
                      // RANDOM (Drift)
                      nextRules[i][j] += (Math.random() - 0.5) * strength;
                  }
              }

              // 2. Global Diffusion & Decay
              // Smooths out the matrix to create "families" of particle types
              // Uses a temporary buffer to calculate averages
              const tempRules = nextRules.map(row => [...row]);
              
              for (let i = 0; i < size; i++) {
                  for (let j = 0; j < size; j++) {
                      // 4-Neighbor Diffusion in Rule Space
                      const n1 = tempRules[(i + 1) % size][j];
                      const n2 = tempRules[(i - 1 + size) % size][j];
                      const n3 = tempRules[i][(j + 1) % size];
                      const n4 = tempRules[i][(j - 1 + size) % size];
                      
                      const avg = (n1 + n2 + n3 + n4) / 4.0;
                      
                      // Blend self with neighbors (Diffusion) - 1% blend
                      let val = tempRules[i][j] * 0.99 + avg * 0.01;
                      
                      // Slow decay to 0 (Entropy) to prevent saturation at -1/1 limits
                      val *= 0.999; 

                      // Clamp
                      if (val > 1.0) val = 1.0;
                      if (val < -1.0) val = -1.0;

                      nextRules[i][j] = val;
                  }
              }

              return nextRules;
          });
      }, 100); // 10Hz update rate

      return () => clearInterval(evolutionInterval);
  }, [isMutating, isPaused, mutationRate]);

  const handleRandomizeRules = useCallback(() => {
    const num = colors.length;
    // Generate clearer structures with sine waves
    const newRules = Array(num).fill(0).map((_, y) => 
         Array(num).fill(0).map((_, x) => Math.sin(x * 0.5) * Math.cos(y * 0.5))
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
        // Force reset to ensure types match
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
        onRandomize={handleRandomizeRules}
        onLoadPreset={handleLoadPreset}
        fps={fps}
        toggleFullscreen={toggleFullscreen}
        isMutating={isMutating}
        setIsMutating={setIsMutating}
        mutationRate={mutationRate}
        setMutationRate={setMutationRate}
      />
    </div>
  );
};

export default App;