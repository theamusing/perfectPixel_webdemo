
import { NdArray, PerfectPixelOptions, PerfectPixelResult, DebugData } from '../types';
import { createNdArray, ops, fft2d } from './ndarray-lite';

function nextPow2(n: number, max_val: number): number {
    return Math.min(Math.pow(2, Math.ceil(Math.log2(n))), max_val);
}

/**
 * RGB to Gray conversion - handles dynamic channel counts (1, 3, 4)
 */
function rgbToGray(imageRgb: NdArray): NdArray {
    const [h, w, c] = imageRgb.shape;
    const grayData = new Float32Array(h * w);
    const gray = createNdArray(grayData, [h, w]);
    
    for (let i = 0; i < h; i++) {
        for (let j = 0; j < w; j++) {
            if (c >= 3) {
                const r = imageRgb.get(i, j, 0);
                const g = imageRgb.get(i, j, 1);
                const b = imageRgb.get(i, j, 2);
                gray.set(i, j, 0.299 * r + 0.587 * g + 0.114 * b);
            } else {
                gray.set(i, j, imageRgb.get(i, j, 0));
            }
        }
    }
    return gray;
}

function normalizeMinMax(x: NdArray, a = 0.0, b = 1.0): NdArray {
    const size = x.size;
    const data = x.data;
    let mn = Infinity;
    let mx = -Infinity;

    for (let i = 0; i < size; i++) {
        if (data[i] < mn) mn = data[i];
        if (data[i] > mx) mx = data[i];
    }

    const out = createNdArray(new Float32Array(size), x.shape);
    const diff = mx - mn;
    if (diff < 1e-8) {
        ops.assigns(out, a);
        return out;
    }

    for (let i = 0; i < size; i++) {
        out.data[i] = a + (b - a) * (data[i] - mn) / diff;
    }
    return out;
}

function conv2dSame(image: NdArray, kernel: NdArray): NdArray {
    const [ih, iw] = image.shape;
    const [kh, kw] = kernel.shape;
    const ph = Math.floor(kh / 2);
    const pw = Math.floor(kw / 2);
    const out = createNdArray(new Float32Array(ih * iw), [ih, iw]);

    for (let i = 0; i < ih; i++) {
        for (let j = 0; j < iw; j++) {
            let sum = 0;
            for (let ky = 0; ky < kh; ky++) {
                for (let kx = 0; kx < kw; kx++) {
                    let py = i + ky - ph;
                    let px = j + kx - pw;
                    py = Math.min(Math.max(py, 0), ih - 1);
                    px = Math.min(Math.max(px, 0), iw - 1);
                    sum += image.get(py, px) * kernel.get(ky, kx);
                }
            }
            out.set(i, j, sum);
        }
    }
    return out;
}

function sobelXy(gray: NdArray, ksize = 3): { gx: NdArray, gy: NdArray } {
    let kx, ky;
    if (ksize === 3) {
        kx = createNdArray(new Float32Array([-1, 0, 1, -2, 0, 2, -1, 0, 1]), [3, 3]);
        ky = createNdArray(new Float32Array([-1, -2, -1, 0, 0, 0, 1, 2, 1]), [3, 3]);
    } else if (ksize === 5) {
        const kxData = [
            -5, -4,  0,  4,  5,
            -8, -10, 0, 10,  8,
            -10,-20, 0, 20, 10,
            -8, -10, 0, 10,  8,
            -5, -4,  0,  4,  5
        ];
        kx = createNdArray(new Float32Array(kxData), [5, 5]);
        ky = createNdArray(new Float32Array(25), [5, 5]);
        for(let r=0; r<5; r++) for(let c=0; c<5; c++) ky.set(c, r, kx.get(r, c));
    } else {
        throw new Error("ksize must be 3 or 5");
    }

    const gx = conv2dSame(gray, kx);
    const gy = conv2dSame(gray, ky);
    return { gx, gy };
}

function computeFftMagnitude(grayImage: NdArray): NdArray {
    const [h, w] = grayImage.shape;
    const ph = nextPow2(h, 1024);
    const pw = nextPow2(w, 1024);

    const real = createNdArray(new Float32Array(ph * pw), [ph, pw]);
    const imag = createNdArray(new Float32Array(ph * pw), [ph, pw]);
    ops.assigns(real, 0);
    ops.assigns(imag, 0);

    for (let i = 0; i < Math.min(h, ph); i++) {
        for (let j = 0; j < Math.min(w,pw); j++) {
            real.set(i, j, grayImage.get(i, j));
        }
    }

    fft2d(1, real, imag);

    const mag = createNdArray(new Float32Array(ph * pw), [ph, pw]);
    const halfH = Math.floor(ph / 2);
    const halfW = Math.floor(pw / 2);

    for (let i = 0; i < ph; i++) {
        for (let j = 0; j < pw; j++) {
            const ni = (i + halfH) % ph;
            const nj = (j + halfW) % pw;
            const r = real.get(ni, nj);
            const im = imag.get(ni, nj);
            const m = Math.sqrt(r * r + im * im);
            mag.set(i, j, 1.0 - Math.log1p(m));
        }
    }
    return normalizeMinMax(mag, 0.0, 1.0);
}

function smooth1d(v: Float32Array, k = 17): Float32Array {
    if (k < 3) return new Float32Array(v);
    if (k % 2 === 0) k += 1;

    const sigma = k / 6.0;
    const center = Math.floor(k / 2);
    const kernel = new Float32Array(k);
    let kernelSum = 0;

    for (let i = 0; i < k; i++) {
        const x = i - center;
        kernel[i] = Math.exp(-(x * x) / (2 * sigma * sigma));
        kernelSum += kernel[i];
    }

    const normSum = kernelSum + 1e-8;

    const out = new Float32Array(v.length);

    for (let i = 0; i < v.length; i++) {
        let acc = 0;
        for (let j = 0; j < k; j++) {
            const idx = i + j - center;
            
            if (idx >= 0 && idx < v.length) {
                acc += v[idx] * (kernel[j] / normSum);
            }
        }
        out[i] = acc;
    }
    return out;
}

interface PeakDetectionResult {
    period: number;
    left: number;
    right: number;
}

function detectPeak(proj: Float32Array, relThr = 0.35, minDist = 6): PeakDetectionResult | null {
    const center = Math.floor(proj.length / 2);
    let mx = 0;
    for (let i = 0; i < proj.length; i++) if (proj[i] > mx) mx = proj[i];

    if (mx < 1e-6) return null;
    const thr = mx * relThr;
    const peakWidth = 6;
    const candidates: Array<{ index: number, score: number }> = [];

    for (let i = 1; i < proj.length - 1; i++) {
        let isPeak = true;
        for (let j = 1; j < peakWidth; j++) {
            if (i - j < 0 || i + j >= proj.length) continue;
            if (proj[i - j + 1] < proj[i - j] || proj[i + j - 1] < proj[i + j]) {
                isPeak = false;
                break;
            }
        }

        if (isPeak && proj[i] >= thr) {
            let leftClimb = 0;
            for (let k = i; k > 0; k--) {
                if (proj[k] > proj[k - 1]) leftClimb = Math.abs(proj[i] - proj[k - 1]);
                else break;
            }
            let rightFall = 0;
            for (let k = i; k < proj.length - 1; k++) {
                if (proj[k] > proj[k + 1]) rightFall = Math.abs(proj[i] - proj[k + 1]);
                else break;
            }
            candidates.push({
                index: i,
                score: Math.max(leftClimb, rightFall)
            });
        }
    }

    const left = candidates.filter(c => c.index < center - minDist && c.index > center * 0.15)
                           .sort((a, b) => b.score - a.score);
    const right = candidates.filter(c => c.index > center + minDist && c.index < center * 1.85)
                            .sort((a, b) => b.score - a.score);

    if (left.length === 0 || right.length === 0) return null;
    
    const lIdx = left[0].index;
    const rIdx = right[0].index;
    return {
        period: Math.abs(rIdx - lIdx) / 2,
        left: lIdx,
        right: rIdx
    };
}

function findBestGrid(origin: number, rangeMin: number, rangeMax: number, gradMag: Float32Array, thr = 0): number {
    let best = Math.round(origin);
    const peaks: Array<{ val: number, idx: number }> = [];
    let mx = 0;
    for (let i = 0; i < gradMag.length; i++) if (gradMag[i] > mx) mx = gradMag[i];
    
    if (mx < 1e-6) return best;
    const relThr = mx * thr;

    const lo = -Math.round(rangeMin);
    const hi = Math.round(rangeMax);

    for (let i = lo; i <= hi; i++) {
        const candidate = Math.round(origin + i);
        if (candidate <= 0 || candidate >= gradMag.length - 1) continue;
        if (gradMag[candidate] > gradMag[candidate - 1] &&
            gradMag[candidate] > gradMag[candidate + 1] &&
            gradMag[candidate] >= relThr) {
            peaks.push({ val: gradMag[candidate], idx: candidate });
        }
    }

    if (peaks.length === 0) return best;
    peaks.sort((a, b) => b.val - a.val);
    return peaks[0].idx;
}

function sampleCenter(image: NdArray, xCoords: number[], yCoords: number[]): NdArray {
    const nx = xCoords.length - 1;
    const ny = yCoords.length - 1;
    const C = image.shape[2];
    const out = createNdArray(new Float32Array(ny * nx * C), [ny, nx, C]);

    for (let j = 0; j < ny; j++) {
        const cy = Math.floor((yCoords[j] + yCoords[j + 1]) * 0.5);
        for (let i = 0; i < nx; i++) {
            const cx = Math.floor((xCoords[i] + xCoords[i + 1]) * 0.5);
            for (let k = 0; k < C; k++) {
                out.set(j, i, k, image.get(cy, cx, k));
            }
        }
    }
    return out;
}

function sampleMajority(image: NdArray, xCoords: number[], yCoords: number[], maxSamples = 256, iters = 6, seed = 0): NdArray {
    const [H, W, C] = image.shape;
    const nx = xCoords.length - 1;
    const ny = yCoords.length - 1;
    const out = createNdArray(new Float32Array(ny * nx * C), [ny, nx, C]);

    let seedVal = seed;
    const lcg = () => {
        seedVal = (1103515245 * seedVal + 12345) & 0x7fffffff;
        return seedVal / 0x7fffffff;
    };

    for (let j = 0; j < ny; j++) {
        let y0 = Math.max(0, Math.min(H, Math.floor(yCoords[j])));
        let y1 = Math.max(0, Math.min(H, Math.floor(yCoords[j + 1])));
        if (y1 <= y0) y1 = Math.min(y0 + 1, H);

        for (let i = 0; i < nx; i++) {
            let x0 = Math.max(0, Math.min(W, Math.floor(xCoords[i])));
            let x1 = Math.max(0, Math.min(W, Math.floor(xCoords[i + 1])));
            if (x1 <= x0) x1 = Math.min(x0 + 1, W);

            const cellPixels: Float32Array[] = [];
            for (let py = y0; py < y1; py++) {
                for (let px = x0; px < x1; px++) {
                    const p = new Float32Array(C);
                    for (let k = 0; k < C; k++) p[k] = image.get(py, px, k);
                    cellPixels.push(p);
                }
            }

            if (cellPixels.length === 0) {
                for (let k = 0; k < C; k++) out.set(j, i, k, 0);
                continue;
            }

            let samples = cellPixels;
            if (cellPixels.length > maxSamples) {
                samples = [];
                for (let s = 0; s < maxSamples; s++) {
                    samples.push(cellPixels[Math.floor(lcg() * cellPixels.length)]);
                }
            }

            let c0 = new Float32Array(samples[0]);
            let c1 = new Float32Array(samples[0]);
            let maxDist = -1;
            for (let s = 0; s < samples.length; s++) {
                let d = 0;
                for (let k = 0; k < C; k++) d += Math.pow(samples[s][k] - c0[k], 2);
                if (d > maxDist) { maxDist = d; c1 = new Float32Array(samples[s]); }
            }

            let finalC = c0;
            for (let it = 0; it < iters; it++) {
                let sum0 = new Float32Array(C), count0 = 0;
                let sum1 = new Float32Array(C), count1 = 0;
                for (let s = 0; s < samples.length; s++) {
                    let d0 = 0, d1 = 0;
                    for (let k = 0; k < C; k++) {
                        d0 += Math.pow(samples[s][k] - c0[k], 2);
                        d1 += Math.pow(samples[s][k] - c1[k], 2);
                    }
                    if (d1 < d0) {
                        for (let k = 0; k < C; k++) sum1[k] += samples[s][k];
                        count1++;
                    } else {
                        for (let k = 0; k < C; k++) sum0[k] += samples[s][k];
                        count0++;
                    }
                }
                if (count0 > 0) for (let k = 0; k < C; k++) c0[k] = sum0[k] / count0;
                if (count1 > 0) for (let k = 0; k < C; k++) c1[k] = sum1[k] / count1;
                finalC = (count1 >= count0) ? c1 : c0;
            }

            for (let k = 0; k < C; k++) out.set(j, i, k, finalC[k]);
        }
    }
    return out;
}

function refineGrids(image: NdArray, gridX: number, gridY: number): { xCoords: number[], yCoords: number[] } {
    const [H, W] = image.shape;
    const cellW = W / gridX;
    const cellH = H / gridY;

    const gray = rgbToGray(image);
    const { gx, gy } = sobelXy(gray, 5);

    const gradXSum = new Float32Array(W);
    const gradYSum = new Float32Array(H);

    for (let i = 0; i < H; i++) {
        for (let j = 0; j < W; j++) {
            gradXSum[j] += Math.abs(gx.get(i, j));
            gradYSum[i] += Math.abs(gy.get(i, j));
        }
    }

    let xCoords: number[] = [];
    let yCoords: number[] = [];

    let x = findBestGrid(W / 2, cellW, cellW, gradXSum);
    while (x < W + cellW / 2) {
        x = findBestGrid(x, cellW / 4, cellW / 4, gradXSum);
        xCoords.push(x);
        x += cellW;
    }
    x = findBestGrid(W / 2, cellW, cellW, gradXSum) - cellW;
    while (x > -cellW/2 && xCoords.length <= W/cellW) {
        x = findBestGrid(x, cellW / 4, cellW / 4, gradXSum);
        xCoords.push(x);
        x -= cellW;
    }

    let y = findBestGrid(H / 2, cellH, cellH, gradYSum);
    while (y < H + cellH/2) {
        y = findBestGrid(y, cellH / 4, cellH / 4, gradYSum);
        yCoords.push(y);
        y += cellH;
    }
    y = findBestGrid(H / 2, cellH, cellH, gradYSum) - cellH;
    while (y > -cellH/2 && yCoords.length <= H/cellH) {
        y = findBestGrid(y, cellH / 4, cellH / 4, gradYSum);
        yCoords.push(y);
        y -= cellH;
    }
    console.log(xCoords.length, yCoords.length, W/cellW, H/cellH)
    
    // force square
    if(Math.abs(xCoords.length - yCoords.length) < 2) {
        if(xCoords.length % 2 === 0)
        {
            if(xCoords.length > yCoords.length)
            {
                xCoords.pop();
            }
            else if(xCoords.length < yCoords.length)
            {
                xCoords.push(0);
            }
            else
            {
                xCoords.push(0);
                yCoords.push(0);
            }
        }
        else
        {
            if(xCoords.length > yCoords.length)
            {
                yCoords.push(0);
            }
            else if(xCoords.length < yCoords.length)
            {
                yCoords.pop();
            }
        }
    }
    return {
        xCoords: xCoords.sort((a, b) => a - b),
        yCoords: yCoords.sort((a, b) => a - b)
    };
}

function estimateGridFft(image: NdArray): { scaleCol: number | null, scaleRow: number | null, peaksRow: [number, number] | null, peaksCol: [number, number] | null, smoothRow: Float32Array, smoothCol: Float32Array, mag: NdArray } {
    const gray = rgbToGray(image);
    const mag = computeFftMagnitude(gray);
    const [PH, PW] = mag.shape;

    const bandRow = Math.floor(PW / 2);
    const bandCol = Math.floor(PH / 2);

    const rowSum = new Float32Array(PH);
    const colSum = new Float32Array(PW);

    for (let i = 0; i < PH; i++) {
        for (let j = Math.floor(PW / 2 - bandRow); j < Math.floor(PW / 2 + bandRow); j++) {
            if (j >= 0 && j < PW) rowSum[i] += mag.get(i, j);
        }
    }
    for (let j = 0; j < PW; j++) {
        for (let i = Math.floor(PH / 2 - bandCol); i < Math.floor(PH / 2 + bandCol); i++) {
            if (i >= 0 && i < PH) colSum[j] += mag.get(i, j);
        }
    }

    const normRow = normalizeMinMax(createNdArray(rowSum, [PH])).data;
    const normCol = normalizeMinMax(createNdArray(colSum, [PW])).data;

    const smoothRow = smooth1d(normRow as Float32Array, 17);
    const smoothCol = smooth1d(normCol as Float32Array, 17);

    const rowResult = detectPeak(smoothRow);
    const colResult = detectPeak(smoothCol);
    
    const [H, W] = image.shape;
    const scaleRow = rowResult ? (rowResult.period * H / PH) : null;
    const scaleCol = colResult ? (colResult.period * W / PW) : null;

    const peaksRow: [number, number] | null = rowResult ? [rowResult.left, rowResult.right] : null;
    const peaksCol: [number, number] | null = colResult ? [colResult.left, colResult.right] : null;

    return { scaleCol, scaleRow, peaksRow, peaksCol, smoothRow, smoothCol, mag };
}

export function getPerfectPixel(image: NdArray, options: PerfectPixelOptions = {}): PerfectPixelResult {
    const { sampleMethod = "center", gridSize = null, minSize = 3.0 } = options;
    const [H, W] = image.shape;

    let scaleCol: number | null, scaleRow: number | null;
    let debugData: DebugData | undefined;

    const est = estimateGridFft(image);
    scaleCol = est.scaleCol;
    scaleRow = est.scaleRow;
    
    debugData = {
        smoothRow: est.smoothRow,
        smoothCol: est.smoothCol,
        peakRow: est.scaleRow,
        peakCol: est.scaleCol,
        peaksRow: est.peaksRow,
        peaksCol: est.peaksCol,
        magData: est.mag.data,
        magShape: est.mag.shape
    };

    if (gridSize) {
        scaleCol = W / gridSize[0];
        scaleRow = H / gridSize[1];
    } else {
        if (scaleCol === null || scaleRow === null || scaleCol <= 0) {
            return { refinedW: null, refinedH: null, scaled: image, debugData };
        }
    }

    const pixelSizeX = W / (scaleCol as number);
    const pixelSizeY = H / (scaleRow as number);
    console.log(pixelSizeX, pixelSizeY)
    const pixelSize = Math.min(pixelSizeX, pixelSizeY);
    
    if (pixelSize <= minSize) {
        return { refinedW: null, refinedH: null, scaled: image, debugData };
    }

    const sizeX = Math.round(scaleCol as number);
    const sizeY = Math.round(scaleRow as number);
    const { xCoords, yCoords } = refineGrids(image, sizeX, sizeY);

    const refinedW = xCoords.length - 1;
    const refinedH = yCoords.length - 1;

    if (refinedW <= 0 || refinedH <= 0) {
        return { refinedW: null, refinedH: null, scaled: image, debugData };
    }

    const scaled = (sampleMethod === "majority") 
        ? sampleMajority(image, xCoords, yCoords)
        : sampleCenter(image, xCoords, yCoords);

    return { refinedW, refinedH, scaled, debugData };
}
