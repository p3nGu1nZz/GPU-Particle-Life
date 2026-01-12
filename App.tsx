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

  // --- Improved Evolution Algorithm ---
  // Uses a convolution-like approach to "diffuse" rules, creating related groups of particles
  useEffect(() => {
      if (!isMutating || isPaused) return;

      const evolutionInterval = setInterval(() => {
          
          setRules(prevRules => {
              const size = prevRules.length;

              // 1. Calculate Global Average for Homeostasis
              // This prevents the entire matrix from becoming all Green (attract) or all Red (repel)
              let sum = 0;
              for(let r=0; r<size; r++) {
                  for(let c=0; c<size; c++) {
                      sum += prevRules[r][c];
                  }
              }
              const globalAvg = sum / (size * size);
              // Push against the dominant color to maintain diversity
              const homeostasisForce = -globalAvg * 0.15; 

              // Create new matrix
              const nextRules = new Array(size);
              
              for(let i = 0; i < size; i++) {
                  nextRules[i] = new Array(size);
                  for(let j = 0; j < size; j++) {
                      
                      let neighborSum = 0;
                      
                      // 3x3 Convolution for localized structure (Cellular Automata-like)
                      const up = (i - 1 + size) % size;
                      const down = (i + 1) % size;
                      const left = (j - 1 + size) % size;
                      const right = (j + 1) % size;

                      neighborSum += prevRules[up][j];
                      neighborSum += prevRules[down][j];
                      neighborSum += prevRules[i][left];
                      neighborSum += prevRules[i][right];
                      
                      // Diagonal neighbors with lower weight
                      const diagSum = (prevRules[up][left] + prevRules[up][right] + prevRules[down][left] + prevRules[down][right]);
                      
                      const current = prevRules[i][j];
                      
                      // Calculate Local Average (Diffusion Target)
                      const localAvg = (neighborSum + diagSum * 0.7) / 6.8; 
                      
                      // Diffusion: Smooth out variations locally
                      const diffusion = (localAvg - current) * 0.1;

                      // Mutation: Random chance to flip
                      let mutation = 0;
                      if (Math.random() < mutationRate * 0.1) {
                          mutation = (Math.random() - 0.5) * 0.5;
                      }

                      // Apply Forces
                      let target = current + diffusion + mutation + homeostasisForce;

                      // Entropy / Decay:
                      // Slowly pull values towards 0 to prevent getting stuck at -1 or 1 limits indefinitely
                      target *= 0.99;

                      // Clamp
                      if (target > 1.0) target = 1.0;
                      if (target < -1.0) target = -1.0;
                      
                      nextRules[i][j] = target;
                  }
              }
              return nextRules;
          });
          
      }, 200); // 5 updates per second

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