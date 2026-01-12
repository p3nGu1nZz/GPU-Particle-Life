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

  // --- Balanced Evolutionary Algorithm ---
  // Uses target-based interpolation to prevent value drift and ensures 
  // a mix of Attraction, Repulsion, and Flow dynamics.
  useEffect(() => {
      if (!isMutating || isPaused) return;

      const evolutionInterval = setInterval(() => {
          setRules(prevRules => {
              const size = prevRules.length;
              const nextRules = prevRules.map(row => [...row]); 

              // Calculate Global Average to detect drift
              let globalSum = 0;
              for(let r=0; r<size; r++) {
                  for(let c=0; c<size; c++) {
                      globalSum += prevRules[r][c];
                  }
              }
              const globalAvg = globalSum / (size * size);
              // If average is too negative, bias towards attraction.
              // If average is too positive, bias towards repulsion.
              const driftCorrection = -globalAvg * 0.5;

              // Number of mutations to perform
              const numMutations = Math.ceil(size * size * mutationRate); 

              for (let k = 0; k < numMutations; k++) {
                  const i = Math.floor(Math.random() * size);
                  const j = Math.floor(Math.random() * size);
                  
                  // Strategy Selection
                  const strategy = Math.random();
                  
                  // Interpolation strength (how fast we move towards the target interaction)
                  const lerpRate = 0.1; 

                  if (strategy < 0.3) {
                      // BINDING (Attraction) - Target +1.0
                      // Strongly pulls particles together
                      // Apply Drift Correction here to balance system
                      const target = 0.5 + Math.random() * 0.5 + driftCorrection;
                      
                      nextRules[i][j] = nextRules[i][j] * (1 - lerpRate) + target * lerpRate;
                      if (i !== j) {
                          nextRules[j][i] = nextRules[j][i] * (1 - lerpRate) + target * lerpRate;
                      }
                  } 
                  else if (strategy < 0.6) {
                      // SEPARATION (Repulsion) - Target -1.0
                      // Keeps particles apart
                      const target = -0.5 - Math.random() * 0.5 + driftCorrection;
                      
                      nextRules[i][j] = nextRules[i][j] * (1 - lerpRate) + target * lerpRate;
                      if (i !== j) {
                          nextRules[j][i] = nextRules[j][i] * (1 - lerpRate) + target * lerpRate;
                      }
                  } 
                  else if (strategy < 0.9) {
                      // FLOW (Asymmetry) - Targets +1 / -1
                      // Creates motion chains (Predator/Prey)
                      // No drift correction here as average is 0
                      const targetA = 0.5 + Math.random() * 0.5;
                      const targetB = -0.5 - Math.random() * 0.5;

                      nextRules[i][j] = nextRules[i][j] * (1 - lerpRate) + targetA * lerpRate;
                      if (i !== j) {
                          nextRules[j][i] = nextRules[j][i] * (1 - lerpRate) + targetB * lerpRate;
                      }
                  }
                  else {
                      // NOISE (Drift)
                      // Small random nudges
                      const nudge = (Math.random() - 0.5) * 0.2;
                      nextRules[i][j] += nudge;
                  }
              }

              // Clamp values
              for (let r = 0; r < size; r++) {
                  for (let c = 0; c < size; c++) {
                       if (nextRules[r][c] > 1.0) nextRules[r][c] = 1.0;
                       if (nextRules[r][c] < -1.0) nextRules[r][c] = -1.0;
                  }
              }

              return nextRules;
          });
      }, 100); 

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