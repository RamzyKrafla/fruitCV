import * as ort from "onnxruntime-web";
import * as pdfjs from "pdfjs-dist/build/pdf";
import pdfjsWorker from "pdfjs-dist/build/pdf.worker.min?url";

// Configure ORT WASM asset path - use a simpler approach for Vite
ort.env.wasm.wasmPaths =
  "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/";
ort.env.wasm.simd = false; // Disable SIMD for better compatibility
ort.env.wasm.numThreads = 1; // safer default for compatibility

// Configure PDF.js worker
pdfjs.GlobalWorkerOptions.workerSrc = pdfjsWorker;

const IMAGE_SIZE = 224;
const IMAGENET_MEAN = [0.485, 0.456, 0.406];
const IMAGENET_STD = [0.229, 0.224, 0.225];

let session = null;
let classes = null;

async function loadImageToCanvas(file) {
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(file);
    const img = new Image();
    img.onload = () => {
      const canvas = document.createElement("canvas");
      canvas.width = IMAGE_SIZE;
      canvas.height = IMAGE_SIZE;
      const ctx = canvas.getContext("2d");
      // cover-like resize
      const scale = Math.max(IMAGE_SIZE / img.width, IMAGE_SIZE / img.height);
      const w = img.width * scale;
      const h = img.height * scale;
      const dx = (IMAGE_SIZE - w) / 2;
      const dy = (IMAGE_SIZE - h) / 2;
      ctx.drawImage(img, dx, dy, w, h);
      URL.revokeObjectURL(url);
      resolve(canvas);
    };
    img.onerror = reject;
    img.src = url;
  });
}

async function loadPdfFirstPageToCanvas(file) {
  const arrayBuffer = await file.arrayBuffer();
  const pdf = await pdfjs.getDocument({ data: arrayBuffer }).promise;
  const page = await pdf.getPage(1);
  const viewport = page.getViewport({ scale: 1.5 });
  const canvas = document.createElement("canvas");
  const ctx = canvas.getContext("2d");
  canvas.width = viewport.width;
  canvas.height = viewport.height;
  await page.render({ canvasContext: ctx, viewport }).promise;

  // Resize to 224x224 with cover
  const out = document.createElement("canvas");
  out.width = IMAGE_SIZE;
  out.height = IMAGE_SIZE;
  const octx = out.getContext("2d");
  const scale = Math.max(IMAGE_SIZE / canvas.width, IMAGE_SIZE / canvas.height);
  const w = canvas.width * scale;
  const h = canvas.height * scale;
  const dx = (IMAGE_SIZE - w) / 2;
  const dy = (IMAGE_SIZE - h) / 2;
  octx.drawImage(canvas, dx, dy, w, h);
  return out;
}

function canvasToCHWFloat32(canvas) {
  const ctx = canvas.getContext("2d");
  const { width, height } = canvas;
  const { data } = ctx.getImageData(0, 0, width, height);
  const out = new Float32Array(3 * width * height);
  let p = 0;
  // Convert HWC RGBA -> CHW and normalize
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const idx = (y * width + x) * 4;
      const r = data[idx] / 255;
      const g = data[idx + 1] / 255;
      const b = data[idx + 2] / 255;
      out[p + 0 * width * height] = (r - IMAGENET_MEAN[0]) / IMAGENET_STD[0];
      out[p + 1 * width * height] = (g - IMAGENET_MEAN[1]) / IMAGENET_STD[1];
      out[p + 2 * width * height] = (b - IMAGENET_MEAN[2]) / IMAGENET_STD[2];
      p++;
    }
  }
  return out;
}

export async function initModel() {
  if (session) return session;

  try {
    console.log("[ORT] Creating session...");
    console.log("[ORT] Version:", ort.version);

    // Fetch model as ArrayBuffer to avoid URL/path resolution issues
    const modelUrl = "/model/best_resnet18_finetune.onnx";
    console.log("[ORT] Fetching model from:", modelUrl);

    const resp = await fetch(modelUrl);
    if (!resp.ok) {
      throw new Error(
        `Failed to fetch model: ${resp.status} ${resp.statusText}`
      );
    }

    const buffer = await resp.arrayBuffer();
    console.log("[ORT] Model loaded, size:", buffer.byteLength, "bytes");
    const bytes = new Uint8Array(buffer);

    session = await ort.InferenceSession.create(bytes, {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all",
    });
    console.log("[ORT] Session ready");

    // Load classes
    const classesResp = await fetch("/model/classes.json");
    if (!classesResp.ok) {
      throw new Error(
        `Failed to fetch classes.json: ${classesResp.status} ${classesResp.statusText}`
      );
    }
    const classesData = await classesResp.json();
    classes = classesData.classes;
    console.log("[ORT] Classes loaded:", classes);

    return session;
  } catch (error) {
    console.error("[ORT] Model initialization failed:", error);
    throw error;
  }
}

export async function preprocessFileToTensor(file) {
  try {
    let canvas;
    if (file.type === "application/pdf") {
      canvas = await loadPdfFirstPageToCanvas(file);
    } else {
      // Treat all other accepted types as images
      canvas = await loadImageToCanvas(file);
    }
    const chw = canvasToCHWFloat32(canvas);
    // Create tensor [1,3,224,224]
    const tensor = new ort.Tensor("float32", chw, [
      1,
      3,
      IMAGE_SIZE,
      IMAGE_SIZE,
    ]);
    return tensor;
  } catch (error) {
    console.error("[ORT] Preprocessing failed:", error);
    throw error;
  }
}

export async function runInference(session, tensor) {
  try {
    const feeds = { input: tensor };
    const results = await session.run(feeds);
    const logits = results.logits.data;

    // softmax
    const maxLogit = Math.max(...logits);
    const exps = logits.map((v) => Math.exp(v - maxLogit));
    const sum = exps.reduce((a, b) => a + b, 0);
    const probs = exps.map((v) => v / sum);

    // Return array of objects with label and probability
    const predictions = classes.map((label, index) => ({
      label,
      probability: probs[index] * 100,
    }));

    // Sort by probability descending
    predictions.sort((a, b) => b.probability - a.probability);

    return predictions;
  } catch (error) {
    console.error("[ORT] Inference failed:", error);
    throw error;
  }
}
