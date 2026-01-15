
import React, { useState, useRef, useCallback, useEffect } from 'react';
import { SamplingMethod, NdArray, DebugData } from './types';
import { getPerfectPixel } from './services/perfectPixelService';
import { createNdArray } from './services/ndarray-lite';

const App: React.FC = () => {
  const [image, setImage] = useState<string | null>(null);
  const [processedImage, setProcessedImage] = useState<string | null>(null);
  const [samplingMethod, setSamplingMethod] = useState<SamplingMethod>('center');
  const [downloadScale, setDownloadScale] = useState<number>(4);
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [refinedSize, setRefinedSize] = useState<{ w: number; h: number } | null>(null);
  const [debugData, setDebugData] = useState<DebugData | null>(null);
  const [showDebug, setShowDebug] = useState(false);

  // Pan states
  const [offset, setOffset] = useState({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState(false);
  const [startPos, setStartPos] = useState({ x: 0, y: 0 });

  const fileInputRef = useRef<HTMLInputElement>(null);
  const outputContainerRef = useRef<HTMLDivElement>(null);
  const magCanvasRef = useRef<HTMLCanvasElement>(null);
  const rowChartRef = useRef<HTMLCanvasElement>(null);
  const colChartRef = useRef<HTMLCanvasElement>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (event) => {
        setImage(event.target?.result as string);
        setProcessedImage(null);
        setRefinedSize(null);
        setError(null);
        setDebugData(null);
        setOffset({ x: 0, y: 0 });
      };
      reader.readAsDataURL(file);
    }
  };

  const processImage = useCallback(async () => {
    if (!image) return;

    setIsProcessing(true);
    setError(null);
    setRefinedSize(null);
    setOffset({ x: 0, y: 0 });

    try {
      const img = new Image();
      img.src = image;
      await new Promise((resolve, reject) => {
        img.onload = resolve;
        img.onerror = () => reject(new Error("Failed to load image"));
      });

      const maxDim = 1024;
      const minThreshold = 128;
      const targetMin = 512;
      
      let w = img.width;
      let h = img.height;
      let minSide = Math.min(w, h);
      
      // 1. Upscale if too small (low res pixel art)
      // If short side < 128, expand by integer multiple k such that k * minSide > 512
      if (minSide < minThreshold) {
        const k = Math.ceil((targetMin + 1) / minSide);
        w *= k;
        h *= k;
      }
      
      // 2. Downscale if too large (long side > 1024)
      if (Math.max(w, h) > maxDim) {
        const s = maxDim / Math.max(w, h);
        w = Math.round(w * s);
        h = Math.round(h * s);
      } else {
        w = Math.round(w);
        h = Math.round(h);
      }

      const targetWidth = w;
      const targetHeight = h;

      const canvas = document.createElement('canvas');
      canvas.width = targetWidth;
      canvas.height = targetHeight;
      const ctx = canvas.getContext('2d');
      if (!ctx) throw new Error('Could not create canvas context');

      // Use nearest neighbor if we are upscaling for cleaner grid detection
      if (targetWidth > img.width) {
          ctx.imageSmoothingEnabled = false;
      }

      ctx.drawImage(img, 0, 0, targetWidth, targetHeight);
      const imageData = ctx.getImageData(0, 0, targetWidth, targetHeight);
      const data = new Float32Array(targetWidth * targetHeight * 4);
      for (let i = 0; i < imageData.data.length; i++) {
        data[i] = imageData.data[i];
      }

      const inputNd = createNdArray(data, [targetHeight, targetWidth, 4]);
      const result = getPerfectPixel(inputNd, { sampleMethod: samplingMethod });

      setDebugData(result.debugData || null);

      if (result.refinedW === null || result.refinedH === null) {
        throw new Error('Failed to refine grid. The image might not have a clear pixel grid or it is too small.');
      }

      setRefinedSize({ w: result.refinedW, h: result.refinedH });

      const [resH, resW, resC] = result.scaled.shape;
      const outCanvas = document.createElement('canvas');
      outCanvas.width = resW;
      outCanvas.height = resH;
      const outCtx = outCanvas.getContext('2d');
      if (!outCtx) throw new Error('Could not create output canvas context');

      const outImageData = outCtx.createImageData(resW, resH);
      for (let y = 0; y < resH; y++) {
        for (let x = 0; x < resW; x++) {
          const outIdx = (y * resW + x) * 4;
          if (resC >= 3) {
            outImageData.data[outIdx]     = result.scaled.get(y, x, 0);
            outImageData.data[outIdx + 1] = result.scaled.get(y, x, 1);
            outImageData.data[outIdx + 2] = result.scaled.get(y, x, 2);
            outImageData.data[outIdx + 3] = (resC === 4) ? result.scaled.get(y, x, 3) : 255;
          } else {
            const val = result.scaled.get(y, x, 0);
            outImageData.data[outIdx]     = val;
            outImageData.data[outIdx + 1] = val;
            outImageData.data[outIdx + 2] = val;
            outImageData.data[outIdx + 3] = 255;
          }
        }
      }
      outCtx.putImageData(outImageData, 0, 0);
      setProcessedImage(outCanvas.toDataURL());
    } catch (err: any) {
      console.error(err);
      setError(err.message || 'An unexpected error occurred during processing.');
    } finally {
      setIsProcessing(false);
    }
  }, [image, samplingMethod]);

  const downloadImage = () => {
    if (!processedImage) return;
    const img = new Image();
    img.src = processedImage;
    img.onload = () => {
      const scale = downloadScale;
      const canvas = document.createElement('canvas');
      canvas.width = img.width * scale;
      canvas.height = img.height * scale;
      const ctx = canvas.getContext('2d');
      if (!ctx) return;
      ctx.imageSmoothingEnabled = false;
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
      const link = document.createElement('a');
      link.download = `perfect-pixel-scaled-x${scale}.png`;
      link.href = canvas.toDataURL('image/png');
      link.click();
    };
  };

  const onMouseDown = (e: React.MouseEvent) => {
    if (!processedImage) return;
    setIsDragging(true);
    setStartPos({ x: e.clientX - offset.x, y: e.clientY - offset.y });
    e.preventDefault();
  };
  const onMouseMove = (e: React.MouseEvent) => {
    if (!isDragging) return;
    setOffset({ x: e.clientX - startPos.x, y: e.clientY - startPos.y });
  };
  const onMouseUp = () => setIsDragging(false);

  useEffect(() => {
    setOffset({ x: 0, y: 0 });
  }, [downloadScale]);

  // Render Debug Canvas
  useEffect(() => {
    if (!showDebug || !debugData) return;

    // Render Mag
    if (magCanvasRef.current && debugData.magData && debugData.magShape) {
      const [h, w] = debugData.magShape;
      const ctx = magCanvasRef.current.getContext('2d');
      if (ctx) {
        magCanvasRef.current.width = w;
        magCanvasRef.current.height = h;
        const imgData = ctx.createImageData(w, h);
        for (let i = 0; i < debugData.magData.length; i++) {
          const val = Math.floor(debugData.magData[i] * 255);
          const idx = i * 4;
          imgData.data[idx] = val;
          imgData.data[idx + 1] = val;
          imgData.data[idx + 2] = val;
          imgData.data[idx + 3] = 255;
        }
        ctx.putImageData(imgData, 0, 0);
      }
    }

    const drawChart = (canvas: HTMLCanvasElement | null, data: Float32Array, period: number | null, peaks: [number, number] | null) => {
      if (!canvas) return;
      const ctx = canvas.getContext('2d');
      if (!ctx) return;
      const w = canvas.width = 400;
      const h = canvas.height = 100;
      ctx.clearRect(0, 0, w, h);
      
      // Draw background
      ctx.fillStyle = '#f8fafc';
      ctx.fillRect(0, 0, w, h);
      
      // Draw Grid
      ctx.strokeStyle = '#e2e8f0';
      ctx.lineWidth = 1;
      ctx.beginPath();
      for(let i=0; i<w; i+=40) { ctx.moveTo(i, 0); ctx.lineTo(i, h); }
      for(let i=0; i<h; i+=20) { ctx.moveTo(0, i); ctx.lineTo(w, i); }
      ctx.stroke();

      // Draw Data
      ctx.strokeStyle = '#6366f1';
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      for (let i = 0; i < data.length; i++) {
        const x = (i / data.length) * w;
        const y = h - (data[i] * h);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();

      // Draw Peaks
      if (peaks) {
        ctx.strokeStyle = '#f43f5e';
        ctx.setLineDash([4, 2]);
        ctx.lineWidth = 1.5;

        peaks.forEach(idx => {
            const px = (idx / data.length) * w;
            ctx.beginPath();
            ctx.moveTo(px, 0);
            ctx.lineTo(px, h);
            ctx.stroke();

            // Draw dot on peak
            const py = h - (data[idx] * h);
            ctx.fillStyle = '#f43f5e';
            ctx.beginPath();
            ctx.arc(px, py, 3, 0, Math.PI * 2);
            ctx.fill();
        });
        
        ctx.setLineDash([]);
        if (period !== null) {
            ctx.fillStyle = '#f43f5e';
            ctx.font = 'bold 10px monospace';
            ctx.fillText(`Period: ${period.toFixed(1)}px`, w / 2 - 40, 15);
        }
      }
    };

    drawChart(rowChartRef.current, debugData.smoothRow, debugData.peakRow, debugData.peaksRow || null);
    drawChart(colChartRef.current, debugData.smoothCol, debugData.peakCol, debugData.peaksCol || null);

  }, [showDebug, debugData]);

  return (
    <div className="max-w-6xl mx-auto px-4 py-12">
      <header className="text-center mb-12">
        <h1 className="text-4xl font-extrabold text-slate-800 mb-2 tracking-tight">PerfectPixel</h1>
        <p className="text-slate-500">Automatically detect pixel grids and restore sharp, pixel-perfect pixel art from distorted pixel-style images.</p>
      </header>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 items-stretch mb-8">
        {/* Left: Input */}
        <div className="bg-white rounded-2xl shadow-xl p-6 border border-slate-100 flex flex-col">
          <h2 className="text-xl font-bold text-slate-700 mb-4 flex items-center">
            <span className="w-2 h-6 bg-indigo-500 rounded-full mr-3"></span>
            Original Image
          </h2>
          
          <div className={`flex-grow border-2 border-dashed rounded-xl relative group transition-all duration-300 min-h-[400px] flex items-center justify-center overflow-hidden ${image ? 'border-transparent' : 'border-slate-300 hover:border-indigo-400 bg-slate-50'}`}>
            {image ? (
              <img src={image} alt="Original" className="max-w-full max-h-full object-contain p-2" />
            ) : (
              <div className="absolute inset-0 flex flex-col items-center justify-center text-slate-400">
                <svg className="w-12 h-12 mb-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                </svg>
                <p>Click to upload or drag image</p>
              </div>
            )}
            <input type="file" ref={fileInputRef} className="absolute inset-0 opacity-0 cursor-pointer" accept="image/*" onChange={handleFileChange} />
          </div>

          <div className="mt-6 space-y-4">
            <div className="flex items-center space-x-4">
              <label className="text-sm font-medium text-slate-600 w-32">Sampling:</label>
              <select value={samplingMethod} onChange={(e) => setSamplingMethod(e.target.value as SamplingMethod)} className="flex-grow bg-slate-100 border-none rounded-lg px-4 py-2 text-sm focus:ring-2 focus:ring-indigo-500">
                <option value="center">Center Sample</option>
                <option value="majority">Majority Cluster</option>
              </select>
            </div>
            <button onClick={processImage} disabled={!image || isProcessing} className={`w-full py-3 rounded-xl font-bold transition-all duration-300 flex items-center justify-center ${!image || isProcessing ? 'bg-slate-200 text-slate-400 cursor-not-allowed' : 'bg-indigo-600 text-white hover:bg-indigo-700 shadow-lg shadow-indigo-200 active:scale-[0.98]'}`}>
              {isProcessing ? 'Analyzing Grid...' : 'Generate Scaled Image'}
            </button>
          </div>
        </div>

        {/* Right: Output */}
        <div className="bg-white rounded-2xl shadow-xl p-6 border border-slate-100 flex flex-col">
          <h2 className="text-xl font-bold text-slate-700 mb-4 flex items-center">
            <span className="w-2 h-6 bg-teal-500 rounded-full mr-3"></span>
            Perfect Pixel Result
          </h2>
          <div 
            ref={outputContainerRef}
            onMouseDown={onMouseDown}
            onMouseMove={onMouseMove}
            onMouseUp={onMouseUp}
            onMouseLeave={onMouseUp}
            className={`flex-grow bg-slate-50 border border-slate-200 rounded-xl relative overflow-hidden min-h-[400px] flex items-center justify-center ${processedImage ? 'cursor-grab active:cursor-grabbing' : ''}`}
          >
            {error ? (
              <div className="p-6 text-center">
                <div className="bg-red-50 text-red-600 p-4 rounded-lg flex flex-col items-center">
                  <svg className="w-10 h-10 mb-2" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" /></svg>
                  <p className="font-semibold">Detection Failed</p>
                  <p className="text-xs mt-1 max-w-[250px]">{error}</p>
                </div>
              </div>
            ) : processedImage && refinedSize ? (
              <div 
                className="absolute transition-transform duration-75 ease-out"
                style={{ 
                  transform: `translate(${offset.x}px, ${offset.y}px)`,
                  width: `${refinedSize.w * downloadScale}px`,
                  height: `${refinedSize.h * downloadScale}px`
                }}
              >
                <img 
                  src={processedImage} 
                  alt="Result" 
                  className="w-full h-full pixelated shadow-xl border border-slate-300" 
                  draggable={false}
                />
              </div>
            ) : (
              <p className="text-slate-400 italic">No output yet</p>
            )}

            {/* Scale Info Overlay */}
            {processedImage && refinedSize && (
              <div className="absolute top-4 right-4 bg-slate-800/80 backdrop-blur-md text-white px-3 py-1.5 rounded-lg text-xs font-mono z-10 pointer-events-none shadow-lg">
                Grid: {refinedSize.w} × {refinedSize.h} | View: {Math.round(refinedSize.w * downloadScale)} × {Math.round(refinedSize.h * downloadScale)}
              </div>
            )}
          </div>

          <div className="mt-6 space-y-6">
            <div className="bg-slate-50 p-4 rounded-xl border border-slate-100">
              <div className="flex justify-between items-center mb-2">
                <label className="text-sm font-medium text-slate-600">Zoom / Export Scale:</label>
                <span className="text-indigo-600 font-bold bg-indigo-50 px-3 py-1 rounded-full text-sm">{downloadScale}x</span>
              </div>
              <input type="range" min="1" max="16" step="1" value={downloadScale} onChange={(e) => setDownloadScale(parseInt(e.target.value))} className="w-full h-2 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-indigo-600" />
            </div>
            <button onClick={downloadImage} disabled={!processedImage} className={`w-full py-3 rounded-xl font-bold transition-all duration-300 flex items-center justify-center ${!processedImage ? 'bg-slate-200 text-slate-400 cursor-not-allowed' : 'bg-teal-600 text-white hover:bg-teal-700 shadow-lg shadow-teal-200 active:scale-[0.98]'}`}>
              Download Scaled Image
            </button>
          </div>
        </div>
      </div>

      {/* Debug Section */}
      <div className="bg-white rounded-2xl shadow-xl border border-slate-100 overflow-hidden">
        <button 
          onClick={() => setShowDebug(!showDebug)} 
          className="w-full px-6 py-4 flex items-center justify-between text-slate-700 hover:bg-slate-50 transition-colors"
        >
          <div className="flex items-center">
            <span className="w-2 h-6 bg-slate-400 rounded-full mr-3"></span>
            <span className="font-bold text-lg">Diagnostics & FFT Data</span>
          </div>
          <span className="text-sm text-slate-400">{showDebug ? 'Hide' : 'Show'} details</span>
        </button>
        
        {showDebug && (
          <div className="p-6 border-t border-slate-100 bg-slate-50/50">
            {debugData ? (
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <div>
                  <h3 className="text-sm font-bold text-slate-500 mb-2 uppercase tracking-wider">FFT Magnitude</h3>
                  <div className="bg-black rounded-lg overflow-hidden border border-slate-200 aspect-square flex items-center justify-center">
                    <canvas ref={magCanvasRef} className="max-w-full max-h-full" />
                  </div>
                </div>
                <div className="md:col-span-2 space-y-6">
                  <div>
                    <h3 className="text-sm font-bold text-slate-500 mb-2 uppercase tracking-wider">Row Projection (Vertical Periodicity)</h3>
                    <div className="bg-white rounded-lg p-2 border border-slate-200 overflow-hidden">
                      <canvas ref={rowChartRef} className="w-full h-[100px]" />
                    </div>
                  </div>
                  <div>
                    <h3 className="text-sm font-bold text-slate-500 mb-2 uppercase tracking-wider">Column Projection (Horizontal Periodicity)</h3>
                    <div className="bg-white rounded-lg p-2 border border-slate-200 overflow-hidden">
                      <canvas ref={colChartRef} className="w-full h-[100px]" />
                    </div>
                  </div>
                </div>
              </div>
            ) : (
              <div className="text-center py-12 text-slate-400 italic">
                Process an image to view diagnostic data
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default App;
