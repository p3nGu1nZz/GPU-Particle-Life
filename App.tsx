import React, { useEffect, useRef, useState, useCallback } from 'react';
import ControlPanel from './components/ControlPanel';
import { DEFAULT_RULES, DEFAULT_PARAMS, DEFAULT_COLORS } from './constants';
import { SimulationParams, RuleMatrix, ColorDefinition } from './types';
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
  const [isMutating, setIsMutating] = useState(true); // Evolve on by default
  const [mutationRate, setMutationRate] = useState(0.05); // Lower default mutation rate for stability
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
  }, [activeGpuPreference]); // Re-run when GPU preference changes

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

  // --- Adaptive Physics & Enhanced Matrix Evolution ---
  useEffect(() => {
      if (!isMutating || isPaused) return;

      const evolutionInterval = setInterval(() => {
          
          let totalMagnitude = 0;
          let totalCells = 0;
          
          setRules(prevRules => {
              const size = prevRules.length;
              const nextRules = prevRules.map(row => [...row]);
              
              for(let i = 0; i < size; i++) {
                  for(let j = 0; j < size; j++) {
                      
                      let neighborSum = 0;
                      let count = 0;
                      
                      // Convolution (3x3)
                      for(let di = -1; di <= 1; di++) {
                          for(let dj = -1; dj <= 1; dj++) {
                              if (di === 0 && dj === 0) continue;
                              
                              const ni = (i + di + size) % size;
                              const nj = (j + dj + size) % size;
                              
                              const weight = (Math.abs(di) + Math.abs(dj) === 1) ? 1.0 : 0.7;
                              neighborSum += prevRules[ni][nj] * weight;
                              count += weight;
                          }
                      }
                      
                      const neighborAvg = neighborSum / count;
                      const current = prevRules[i][j];
                      
                      // --- Smoother Turing-like Evolution ---
                      
                      // 1. Diffusion: Pull towards neighbors
                      const diffusion = (neighborAvg - current) * 0.1;
                      
                      // 2. Reaction: Push away from 0 towards stable states (-1 or 1)
                      const reaction = (current - current * current * current) * 0.05;
                      
                      // 3. Small stochastic drift (Mutation)
                      let mutation = 0;
                      if (Math.random() < mutationRate * 0.01) {
                          mutation = (Math.random() - 0.5) * 0.5;
                      }

                      let target = current + diffusion + reaction + mutation;

                      // 4. Global bias correction (prevent all-green takeover)
                      // If the local area is too positive (too attractive), introduce repulsion
                      if (neighborAvg > 0.2) {
                          target -= 0.05; // Constant pressure towards repulsion to break clumps
                      }

                      // Smooth temporal update
                      const lerpSpeed = 0.2; 
                      nextRules[i][j] = current + (target - current) * lerpSpeed;
                      
                      // Hard Clamp
                      nextRules[i][j] = Math.max(-1, Math.min(1, nextRules[i][j]));
                      
                      // Stats for Physics
                      totalMagnitude += Math.abs(nextRules[i][j]);
                      totalCells++;
                  }
              }
              return nextRules;
          });
          
          // Apply Adaptive Physics
          if (totalCells > 0) {
              const avgMagnitude = totalMagnitude / totalCells;
              
              const baseForce = DEFAULT_PARAMS.forceFactor;
              let adaptationMult = 1.0;
              
              // Stabilize energy
              if (avgMagnitude > 0.6) {
                   adaptationMult = 0.5 / avgMagnitude; 
              } else if (avgMagnitude < 0.2) {
                   adaptationMult = 1.3;
              }
              
              const targetForce = baseForce * adaptationMult;
              
              // Only update if difference is significant to avoid React thrashing
              if (Math.abs(targetForce - params.forceFactor) > 0.02) {
                  setParams(p => ({
                      ...p,
                      forceFactor: p.forceFactor + (targetForce - p.forceFactor) * 0.05
                  }));
              }
          }

      }, 100); 

      return () => clearInterval(evolutionInterval);
  }, [isMutating, isPaused, mutationRate, params.forceFactor]);

  const handleRandomizeRules = useCallback(() => {
    const num = colors.length;
    // Initialize with smooth noise (Perlin-ish) for better start than white noise
    const newRules = Array(num).fill(0).map((_, y) => 
         Array(num).fill(0).map((_, x) => Math.sin(x * 0.1) * Math.cos(y * 0.1) + (Math.random() - 0.5))
    );
    setRules(newRules);
  }, [colors.length]);

  const handleReset = useCallback(() => {
    engineRef.current?.reset();
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