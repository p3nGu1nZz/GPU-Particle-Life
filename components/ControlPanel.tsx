import React, { useState, useRef, useEffect, useCallback } from 'react';
import { SimulationParams, RuleMatrix, ColorDefinition, GPUPreference } from '../types';
import { Settings, Play, Pause, RotateCcw, RefreshCw, X, Rocket, Monitor, Maximize, Blend, Plus, Minus, Palette, Dna, Activity, Sprout, MousePointer2, Thermometer } from 'lucide-react';

interface ControlPanelProps {
    params: SimulationParams;
    setParams: (p: SimulationParams) => void;
    rules: RuleMatrix;
    setRules: (r: RuleMatrix) => void;
    colors: ColorDefinition[];
    setColors: (c: ColorDefinition[]) => void;
    isPaused: boolean;
    setIsPaused: (p: boolean) => void;
    onReset: () => void;
    onRandomize: () => void;
    fps: number;
    toggleFullscreen: () => void;
    isMutating: boolean;
    setIsMutating: (m: boolean) => void;
    mutationRate: number;
    setMutationRate: (r: number) => void;
}

const ControlPanel: React.FC<ControlPanelProps> = ({ 
    params, setParams, rules, setRules, colors, setColors,
    isPaused, setIsPaused, onReset, onRandomize, fps, toggleFullscreen,
    isMutating, setIsMutating, mutationRate, setMutationRate
}) => {
    const [isMinimized, setIsMinimized] = useState(false);
    
    // Gear Icon Visibility Logic
    const [isGearVisible, setIsGearVisible] = useState(false);
    const hideTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

    const startHideTimer = () => {
        if (hideTimerRef.current) clearTimeout(hideTimerRef.current);
        hideTimerRef.current = setTimeout(() => {
            setIsGearVisible(false);
        }, 2000); // 2 seconds delay
    };

    const handleMouseEnter = () => {
        if (hideTimerRef.current) clearTimeout(hideTimerRef.current);
        setIsGearVisible(true);
    };

    const handleMouseLeave = () => {
        if (isMinimized) {
            startHideTimer();
        }
    };

    useEffect(() => {
        if (isMinimized) {
            setIsGearVisible(true);
            startHideTimer();
        } else {
            setIsGearVisible(false);
            if (hideTimerRef.current) clearTimeout(hideTimerRef.current);
        }
    }, [isMinimized]);
    
    // Helper to toggle rules (Legacy, mostly unused now as matrix is auto-evolving)
    const updateRule = (sourceIdx: number, targetIdx: number, value: number) => {
        const newRules = rules.map(row => [...row]);
        newRules[sourceIdx][targetIdx] = value;
        setRules(newRules);
    };

    const updateParam = (key: keyof SimulationParams, value: any) => {
        setParams({ ...params, [key]: value });
    };

    const setMatrixUniformly = (val: number) => {
        const newRules = rules.map(row => row.map(() => val));
        setRules(newRules);
    };

    const handleColorChange = (index: number, newHex: string) => {
        const r = parseInt(newHex.slice(1, 3), 16);
        const g = parseInt(newHex.slice(3, 5), 16);
        const b = parseInt(newHex.slice(5, 7), 16);
        
        const newColors = [...colors];
        newColors[index] = { r, g, b, name: newHex };
        setColors(newColors);
    };

    const addColor = () => {
        if (colors.length >= 32) return; 
        
        const hue = Math.floor(Math.random() * 360);
        const newHex = `hsl(${hue}, 100%, 50%)`;
        
        // Random RGB for init
        const r = Math.floor(Math.random() * 255);
        const g = Math.floor(Math.random() * 255);
        const b = Math.floor(Math.random() * 255);
        
        const hex = "#" + [r, g, b].map(x => x.toString(16).padStart(2, '0')).join('');
        
        const newColors = [...colors, { r, g, b, name: hex }];
        setColors(newColors);
    };

    const removeColor = () => {
        if (colors.length <= 2) return;
        const newColors = colors.slice(0, colors.length - 1);
        setColors(newColors);
    };

    const randomizeColors = () => {
        const newColors = colors.map((_, i) => {
            const r = Math.floor(Math.random() * 255);
            const g = Math.floor(Math.random() * 255);
            const b = Math.floor(Math.random() * 255);
            const hex = "#" + [r, g, b].map(x => x.toString(16).padStart(2, '0')).join('');
            return { r, g, b, name: hex };
        });
        setColors(newColors);
    };

    return (
        <>
            {/* Toggle Button Container (Hover Zone) */}
            <div 
                className={`fixed top-0 left-0 w-24 h-24 z-30 flex items-start justify-start pl-4 pt-4 transition-all duration-300 ${isMinimized ? 'pointer-events-auto' : 'pointer-events-none'}`}
                onMouseEnter={handleMouseEnter}
                onMouseLeave={handleMouseLeave}
            >
                <button 
                    onClick={() => setIsMinimized(false)}
                    className={`p-2 bg-neutral-900/80 backdrop-blur-md rounded-lg border border-white/10 hover:bg-white/10 transition-all duration-500 ${
                        isMinimized 
                            ? (isGearVisible ? 'opacity-100 translate-x-0' : 'opacity-0 translate-x-0') 
                            : 'opacity-0 -translate-x-full'
                    }`}
                    style={{ pointerEvents: isMinimized && isGearVisible ? 'auto' : 'none' }}
                    title="Open Settings"
                >
                    <Settings className="w-6 h-6 text-emerald-400" />
                </button>
            </div>

            {/* Sliding Panel */}
            <div 
                className={`fixed top-0 left-0 h-full w-80 z-20 bg-neutral-900/95 backdrop-blur-xl border-r border-white/10 shadow-2xl transform transition-transform duration-300 ease-in-out flex flex-col ${isMinimized ? '-translate-x-full' : 'translate-x-0'}`}
            >
                {/* Header with Stats and Close */}
                <div className="p-4 border-b border-white/10 flex-shrink-0">
                    <div className="flex justify-between items-start mb-4">
                         <div className="flex items-center space-x-2">
                            <Rocket className="w-5 h-5 text-emerald-400" />
                            <h1 className="font-bold text-white tracking-wider">PARTICLE LIFE <span className="text-xs font-normal text-emerald-400 bg-emerald-950 px-1 rounded ml-1">GPU</span></h1>
                        </div>
                        <button onClick={() => setIsMinimized(true)} className="text-neutral-500 hover:text-white transition-colors">
                            <X className="w-5 h-5" />
                        </button>
                    </div>
                    
                    <div className="flex justify-between text-xs font-mono text-neutral-400 bg-black/40 p-2 rounded border border-white/5">
                        <span>{params.particleCount.toLocaleString()} Particles</span>
                        <span className={fps < 30 ? 'text-red-400' : 'text-emerald-400'}>{fps} FPS</span>
                    </div>
                </div>

                {/* Scrollable Content */}
                <div className="flex-1 overflow-y-auto p-4 custom-scrollbar">
                    
                    {/* Play Controls */}
                    <div className="grid grid-cols-4 gap-2 mb-6">
                        <button
                            onClick={() => setIsPaused(!isPaused)}
                            className={`col-span-2 flex items-center justify-center space-x-2 py-2 rounded-lg font-medium text-sm transition-all ${isPaused ? 'bg-emerald-600 hover:bg-emerald-500 text-white' : 'bg-neutral-800 hover:bg-neutral-700 text-neutral-300'}`}
                        >
                            {isPaused ? <Play className="w-4 h-4 fill-current" /> : <Pause className="w-4 h-4 fill-current" />}
                            <span>{isPaused ? 'Resume' : 'Pause'}</span>
                        </button>
                        
                        <button 
                            onClick={onRandomize}
                            className="bg-neutral-800 hover:bg-neutral-700 text-neutral-300 rounded-lg flex items-center justify-center transition-colors"
                            title="Randomize Rules"
                        >
                            <RefreshCw className="w-4 h-4" />
                        </button>
                        
                        <button 
                            onClick={onReset}
                            className="bg-neutral-800 hover:bg-neutral-700 text-neutral-300 rounded-lg flex items-center justify-center transition-colors"
                            title="Respawn Particles"
                        >
                            <RotateCcw className="w-4 h-4" />
                        </button>
                    </div>

                    {/* Global Settings */}
                    <div className="space-y-6 mb-8">
                        {/* GPU & Display Settings */}
                        <div className="space-y-4 p-3 bg-white/5 rounded-lg border border-white/5">
                            <div className="flex justify-between items-center text-xs">
                                <span className="text-neutral-300 font-medium">Render Device</span>
                                <Monitor className="w-3 h-3 text-neutral-500" />
                            </div>
                            <select 
                                value={params.gpuPreference}
                                onChange={(e) => updateParam('gpuPreference', e.target.value as GPUPreference)}
                                className="w-full bg-neutral-900 text-xs text-white border border-white/10 rounded px-2 py-1.5 outline-none focus:border-emerald-500"
                            >
                                <option value="default">Default</option>
                                <option value="high-performance">High Performance (Discrete)</option>
                                <option value="low-power">Low Power (Integrated)</option>
                            </select>

                            <InputSlider 
                                label="Render Scale"
                                value={params.dpiScale}
                                min={0.1} max={1.5} step={0.05}
                                onChange={(v) => updateParam('dpiScale', v)}
                                formatValue={(v) => `${Math.round(v * 100)}%`}
                            />

                            <button
                                onClick={toggleFullscreen}
                                className="w-full flex items-center justify-center space-x-2 bg-neutral-800 hover:bg-neutral-700 text-xs py-1.5 rounded transition-colors"
                            >
                                <Maximize className="w-3 h-3" />
                                <span>Toggle Fullscreen</span>
                            </button>
                        </div>

                         <div className="h-px bg-white/5 my-2" />

                        {/* Trails & Rendering */}
                        <div className="space-y-3">
                            <div className="flex justify-between items-center">
                                <div className="flex flex-col">
                                    <span className="text-neutral-300 font-medium text-xs">Trails</span>
                                    <span className="text-neutral-500 text-[10px]">Disable screen clearing</span>
                                </div>
                                <button
                                    onClick={() => updateParam('trails', !params.trails)}
                                    className={`relative w-12 h-6 rounded-full transition-colors duration-200 ease-in-out border border-white/10 ${params.trails ? 'bg-emerald-600' : 'bg-neutral-800'}`}
                                >
                                    <span 
                                        className={`absolute top-0.5 left-0.5 w-5 h-5 bg-white rounded-full shadow-md transform transition-transform duration-200 ${params.trails ? 'translate-x-6' : 'translate-x-0'}`} 
                                    />
                                </button>
                            </div>

                            {/* Blend Mode */}
                             <div className="flex justify-between items-center bg-white/5 p-2 rounded">
                                <div className="flex items-center space-x-2">
                                    <Blend className="w-4 h-4 text-neutral-400" />
                                    <span className="text-xs text-neutral-300">Blend Mode</span>
                                </div>
                                <div className="flex bg-black/50 rounded p-0.5">
                                    <button 
                                        onClick={() => updateParam('blendMode', 'additive')}
                                        className={`px-3 py-1 text-[10px] rounded transition-colors ${params.blendMode === 'additive' ? 'bg-emerald-600 text-white shadow-sm' : 'text-neutral-400 hover:text-white'}`}
                                    >
                                        Additive
                                    </button>
                                    <button 
                                        onClick={() => updateParam('blendMode', 'normal')}
                                        className={`px-3 py-1 text-[10px] rounded transition-colors ${params.blendMode === 'normal' ? 'bg-emerald-600 text-white shadow-sm' : 'text-neutral-400 hover:text-white'}`}
                                    >
                                        Normal
                                    </button>
                                </div>
                            </div>
                            
                            {/* Biological Growth Toggle */}
                            <div className="flex justify-between items-center bg-white/5 p-2 rounded border border-emerald-500/20">
                                <div className="flex items-center space-x-2">
                                    <Sprout className="w-4 h-4 text-emerald-400" />
                                    <div className="flex flex-col">
                                        <span className="text-xs text-emerald-100">Biological Growth</span>
                                        <span className="text-[9px] text-emerald-500/70">Particles infect neighbors</span>
                                    </div>
                                </div>
                                <button
                                    onClick={() => updateParam('growth', !params.growth)}
                                    className={`relative w-10 h-5 rounded-full transition-colors duration-200 ease-in-out border border-white/10 ${params.growth ? 'bg-emerald-600' : 'bg-neutral-800'}`}
                                >
                                    <span 
                                        className={`absolute top-0.5 left-0.5 w-4 h-4 bg-white rounded-full shadow-md transform transition-transform duration-200 ${params.growth ? 'translate-x-5' : 'translate-x-0'}`} 
                                    />
                                </button>
                            </div>

                            <InputSlider 
                                label="Color Opacity"
                                value={params.baseColorOpacity}
                                min={0.01} max={1.0} step={0.01}
                                onChange={(v) => updateParam('baseColorOpacity', v)}
                                formatValue={(v) => `${Math.round(v * 100)}%`}
                            />
                        </div>

                        <div className="h-px bg-white/5 my-2" />
                        
                        {/* Color Palette Manager */}
                        <div className="space-y-3">
                            <div className="flex justify-between items-center">
                                <div className="flex items-center space-x-2 text-xs font-medium text-neutral-300">
                                    <Palette className="w-3 h-3" />
                                    <span>Palette ({colors.length})</span>
                                </div>
                                <div className="flex space-x-1">
                                    <button onClick={randomizeColors} className="p-1 hover:bg-white/10 rounded text-neutral-400 hover:text-white" title="Randomize All Colors">
                                        <RefreshCw className="w-3 h-3" />
                                    </button>
                                    <button onClick={removeColor} disabled={colors.length <= 2} className="p-1 hover:bg-white/10 rounded text-neutral-400 hover:text-red-400 disabled:opacity-30">
                                        <Minus className="w-3 h-3" />
                                    </button>
                                    <button onClick={addColor} disabled={colors.length >= 32} className="p-1 hover:bg-white/10 rounded text-neutral-400 hover:text-emerald-400 disabled:opacity-30">
                                        <Plus className="w-3 h-3" />
                                    </button>
                                </div>
                            </div>
                            
                            <div className="grid grid-cols-8 gap-1.5">
                                {colors.map((c, i) => (
                                    <div key={i} className="relative group w-full aspect-square">
                                        <input 
                                            type="color" 
                                            value={c.name} 
                                            onChange={(e) => handleColorChange(i, e.target.value)}
                                            className="w-full h-full block bg-transparent cursor-pointer rounded-sm overflow-hidden opacity-0 absolute inset-0 z-10"
                                        />
                                        <div 
                                            className="w-full h-full rounded-sm border border-white/20 shadow-sm"
                                            style={{ backgroundColor: c.name }} 
                                            title={`Color ${i}`}
                                        />
                                    </div>
                                ))}
                            </div>
                        </div>

                        <div className="h-px bg-white/5 my-2" />

                        <InputSlider 
                            label="Particle Count"
                            value={params.particleCount}
                            min={100} max={100000} step={100}
                            onChange={(v) => updateParam('particleCount', v)}
                            integer
                        />
                         <InputSlider 
                            label="Particle Size (px)"
                            value={params.particleSize}
                            min={1} max={10} step={0.5}
                            onChange={(v) => updateParam('particleSize', v)}
                        />
                        <div className="h-px bg-white/5 my-2" />
                        
                        <div className="space-y-3">
                             <div className="flex items-center space-x-2 text-neutral-400">
                                <Thermometer className="w-4 h-4" />
                                <span className="text-xs font-bold tracking-wider">PHYSICS</span>
                             </div>
                            <InputSlider 
                                label="Temperature (Entropy)"
                                value={params.temperature}
                                min={0.0} max={10.0} step={0.1}
                                onChange={(v) => updateParam('temperature', v)}
                            />
                            <InputSlider 
                                label="Friction"
                                value={params.friction}
                                min={0.5} max={0.99} step={0.01}
                                onChange={(v) => updateParam('friction', v)}
                            />
                            <InputSlider 
                                label="Force Strength"
                                value={params.forceFactor}
                                min={0.1} max={5.0} step={0.1}
                                onChange={(v) => updateParam('forceFactor', v)}
                            />
                            <InputSlider 
                                label="Interaction Radius"
                                value={params.rMax}
                                min={0.05} max={0.5} step={0.01}
                                onChange={(v) => updateParam('rMax', v)}
                            />
                        </div>
                    </div>

                    {/* Matrix Heatmap */}
                    <div className="mb-4">
                        <div className="flex justify-between items-center mb-4">
                            <h3 className="text-xs font-bold uppercase tracking-wider text-neutral-400 flex items-center space-x-2">
                                <span>Matrix</span>
                                {isMutating && <Activity className="w-3 h-3 text-emerald-500 animate-pulse" />}
                            </h3>
                            <div className="flex space-x-2 items-center">
                                <button 
                                    onClick={() => setIsMutating(!isMutating)}
                                    className={`flex items-center space-x-1 px-2 py-1 rounded text-[9px] border transition-colors ${isMutating ? 'bg-emerald-900/50 border-emerald-500 text-emerald-400' : 'bg-neutral-800 border-white/10 text-neutral-400'}`}
                                >
                                    <Dna className="w-3 h-3" />
                                    <span>Evolve</span>
                                </button>
                                <div className="h-4 w-px bg-white/10"></div>
                                <div className="flex space-x-1">
                                    <button 
                                        onClick={() => setMatrixUniformly(0)}
                                        className="w-16 h-5 text-[9px] bg-neutral-800 hover:bg-neutral-700 text-neutral-300 rounded flex items-center justify-center border border-white/10"
                                    >
                                        Clear
                                    </button>
                                </div>
                            </div>
                        </div>
                        
                        <div className="mb-4 space-y-2 p-2 bg-white/5 rounded border border-white/10">
                            <InputSlider 
                                label="Mutation Rate"
                                value={mutationRate}
                                min={0.0} max={1.0} step={0.01}
                                onChange={(v) => setMutationRate(v)}
                                formatValue={(v) => `${Math.round(v * 100)}%`}
                            />
                        </div>

                        <MatrixHeatmap 
                            rules={rules}
                            colors={colors}
                        />
                    </div>
                </div>

                {/* Footer */}
                <div className="p-4 border-t border-white/10 flex-shrink-0 flex items-center justify-center space-x-2 text-neutral-500">
                    <MousePointer2 className="w-3 h-3" />
                    <span className="text-[10px]">L-Click to Repel • R-Click to Attract</span>
                </div>
            </div>
        </>
    );
};

// --- Subcomponents ---

const MatrixHeatmap: React.FC<{
    rules: RuleMatrix,
    colors: ColorDefinition[]
}> = ({ rules, colors }) => {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const containerRef = useRef<HTMLDivElement>(null);
    
    const draw = useCallback(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;
        const ctx = canvas.getContext('2d', { alpha: false }); // Optimize for no alpha
        if (!ctx) return;

        const size = rules.length;
        if (size === 0) return;

        // Set dimensions (1 pixel per rule cell for crisp rendering)
        if (canvas.width !== size || canvas.height !== size) {
            canvas.width = size;
            canvas.height = size;
        }

        const imgData = ctx.createImageData(size, size);
        const data = imgData.data;

        // Color Map Logic: Green (Attract, >0) -> Black (0) -> Red (Repel, <0)
        // NOTE: Physics shader uses Positive = Attract, Negative = Repel (or vice versa depending on impl).
        // Let's assume standard: Positive = Pull (Attract), Negative = Push (Repel).
        // Visuals: Positive = Green, Negative = Red.
        
        for (let y = 0; y < size; y++) {
            for (let x = 0; x < size; x++) {
                const val = rules[y][x]; // -1 to 1
                const index = (y * size + x) * 4;

                let r = 0, g = 0, b = 0;

                // Scale brightness so small values are visible
                // Power curve to pop low values: Math.pow(abs(val), 0.7)
                const intensity = Math.pow(Math.abs(val), 0.6) * 255;

                if (val > 0) {
                    // Positive (Attraction) -> Green/Cyan
                    // Use a nice Emerald Green
                    r = 0;
                    g = Math.min(255, intensity);
                    b = Math.min(255, intensity * 0.5); 
                } else {
                    // Negative (Repulsion) -> Red/Orange
                    r = Math.min(255, intensity);
                    g = Math.min(255, intensity * 0.2);
                    b = 0;
                }

                data[index] = r;
                data[index + 1] = g;
                data[index + 2] = b;
                data[index + 3] = 255; // Alpha
            }
        }

        ctx.putImageData(imgData, 0, 0);

    }, [rules]);

    useEffect(() => {
        let frameId: number;
        // Throttle drawing to avoid UI lag on high refresh rate monitors
        const loop = () => {
             draw();
             // Just draw once per update of 'rules', no need to loop constantly 
             // unless we want to animate the scanline in canvas (we use CSS for that)
        };
        
        // Use requestAnimationFrame just to ensure we draw when browser is ready
        frameId = requestAnimationFrame(loop);
        return () => cancelAnimationFrame(frameId);
    }, [draw]);

    return (
        <div ref={containerRef} className="relative w-full aspect-square bg-black border border-white/20 rounded overflow-hidden group">
             {/* The Heatmap */}
            <canvas 
                ref={canvasRef}
                className="w-full h-full block rendering-pixelated"
                style={{ imageRendering: 'pixelated' }} 
            />
            
            {/* Scanline Overlay (Visual Polish) */}
            <div className="absolute inset-0 pointer-events-none bg-gradient-to-b from-transparent via-emerald-500/10 to-transparent opacity-30 animate-scanline" />
            <div className="absolute inset-0 pointer-events-none shadow-[inset_0_0_20px_rgba(0,0,0,0.8)]" />
            
            <style>{`
                @keyframes scanline {
                    0% { transform: translateY(-100%); }
                    100% { transform: translateY(100%); }
                }
                .animate-scanline {
                    animation: scanline 3s linear infinite;
                }
                .rendering-pixelated {
                    image-rendering: pixelated;
                }
            `}</style>
        </div>
    );
};

const InputSlider: React.FC<{
    label: string;
    value: number;
    min: number;
    max: number;
    step: number;
    onChange: (val: number) => void;
    integer?: boolean;
    formatValue?: (val: number) => string;
}> = ({ label, value, min, max, step, onChange, integer, formatValue }) => {
    const [isEditing, setIsEditing] = useState(false);
    const [tempValue, setTempValue] = useState(value.toString());
    const inputRef = useRef<HTMLInputElement>(null);

    useEffect(() => {
        if (!isEditing) setTempValue(value.toString());
    }, [value, isEditing]);

    const handleCommit = () => {
        let val = parseFloat(tempValue);
        if (isNaN(val)) val = value;
        val = Math.max(min, val); 
        if (integer) val = Math.round(val);
        onChange(val);
        setIsEditing(false);
    };

    const handleKeyDown = (e: React.KeyboardEvent) => {
        if (e.key === 'Enter') handleCommit();
    };

    const displayValue = formatValue ? formatValue(value) : (integer ? value : value.toFixed(2));

    return (
        <div className="space-y-2">
            <div className="flex justify-between text-xs items-center">
                <span className="text-neutral-300 font-medium">{label}</span>
                {isEditing ? (
                    <input
                        ref={inputRef}
                        type="number"
                        className="w-16 bg-neutral-800 text-right text-emerald-400 font-mono rounded px-1 py-0.5 outline-none border border-emerald-500/50"
                        value={tempValue}
                        onChange={(e) => setTempValue(e.target.value)}
                        onBlur={handleCommit}
                        onKeyDown={handleKeyDown}
                        autoFocus
                    />
                ) : (
                    <span 
                        className="text-neutral-400 font-mono hover:text-emerald-400 cursor-pointer transition-colors bg-white/5 px-1.5 py-0.5 rounded"
                        onClick={() => setIsEditing(true)}
                    >
                        {displayValue}
                    </span>
                )}
            </div>
            <input 
                type="range" min={min} max={max} step={step}
                value={value}
                onChange={(e) => onChange(parseFloat(e.target.value))}
                className="w-full h-1 bg-neutral-700 rounded-lg appearance-none cursor-pointer accent-emerald-500 hover:accent-emerald-400"
            />
        </div>
    );
};

export default ControlPanel;