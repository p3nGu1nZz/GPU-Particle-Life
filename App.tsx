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
  const [isMutating, setIsMutating] = useState(true); 
  const [mutationRate, setMutationRate] = useState(0.05); 
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

  // Adaptive Time Stepping Logic
  useEffect(() => {
      // Just monitoring for now
  }, [fps]);

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
  useEffect(() => {
      if (!isMutating || isPaused) return;

      const evolutionInterval = setInterval(() => {
          
          let totalMagnitude = 0;
          let totalCells = 0;
          
          setRules(prevRules => {
              const size = prevRules.length;
              const nextRules = new Array(size);
              
              for(let i = 0; i < size; i++) {
                  nextRules[i] = new Array(size);
                  for(let j = 0; j < size; j++) {
                      
                      let neighborSum = 0;
                      
                      // 3x3 Convolution - Optimized loop
                      // Weights: Center 0, Orthogonal 1, Diagonal 0.7
                      // Using explicit indices for performance vs loop overhead
                      const up = (i - 1 + size) % size;
                      const down = (i + 1) % size;
                      const left = (j - 1 + size) % size;
                      const right = (j + 1) % size;

                      neighborSum += prevRules[up][j];
                      neighborSum += prevRules[down][j];
                      neighborSum += prevRules[i][left];
                      neighborSum += prevRules[i][right];
                      
                      neighborSum += (prevRules[up][left] + prevRules[up][right] + prevRules[down][left] + prevRules[down][right]) * 0.7;

                      const neighborAvg = neighborSum / 6.8; // 4 + 4*0.7 = 6.8
                      const current = prevRules[i][j];
                      
                      // Reaction-Diffusion
                      const diffusion = (neighborAvg - current) * 0.15;
                      // Cubic reaction term stabilizes around -1, 0, 1
                      const reaction = (current - Math.pow(current, 3)) * 0.1;
                      
                      let mutation = 0;
                      
                      // Spontaneous Generation in dead zones
                      const energy = Math.abs(current);
                      const localEnergy = Math.abs(neighborAvg);
                      
                      let effectiveMutationChance = mutationRate * 0.05;
                      
                      if (energy < 0.01 && localEnergy < 0.01) {
                          // Boost restart chance in empty space
                          effectiveMutationChance = Math.max(0.01, mutationRate * 0.2); 
                      }

                      if (Math.random() < effectiveMutationChance) {
                          mutation = (Math.random() * 2 - 1) * 0.5;
                      }

                      let target = current + diffusion + reaction + mutation;

                      // Entropy / Decay to zero to prevent noise buildup
                      target *= 0.995;

                      // Clamp smoothly
                      if (target > 1.0) target = 1.0;
                      if (target < -1.0) target = -1.0;
                      
                      // Snap to zero if very small (Performance & Stability)
                      if (Math.abs(target) < 0.005) target = 0;

                      nextRules[i][j] = target;
                      
                      totalMagnitude += Math.abs(target);
                      totalCells++;
                  }
              }
              return nextRules;
          });
          
          // Auto-balance global force based on matrix saturation
          if (totalCells > 0) {
              const avgMagnitude = totalMagnitude / totalCells;
              
              const baseForce = DEFAULT_PARAMS.forceFactor;
              let targetForce = baseForce;

              // If matrix is saturated (chaotic), reduce force
              if (avgMagnitude > 0.4) targetForce = baseForce * 0.7;
              // If matrix is sparse (boring), increase force
              if (avgMagnitude < 0.05) targetForce = baseForce * 1.8;

              if (Math.abs(targetForce - params.forceFactor) > 0.05) {
                  setParams(p => ({
                      ...p,
                      forceFactor: p.forceFactor + (targetForce - p.forceFactor) * 0.1
                  }));
              }
          }

      }, 200);

      return () => clearInterval(evolutionInterval);
  }, [isMutating, isPaused, mutationRate, params.forceFactor]);

  const handleRandomizeRules = useCallback(() => {
    const num = colors.length;
    const newRules = Array(num).fill(0).map((_, y) => 
         Array(num).fill(0).map((_, x) => Math.sin(x * 0.2) * Math.cos(y * 0.2) * 0.8 + (Math.random() - 0.5) * 0.2)
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