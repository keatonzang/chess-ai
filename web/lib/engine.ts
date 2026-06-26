// Lazy ONNX engine. We load the onnxruntime-web runtime from CDN via a script
// tag (it exposes a global `ort`), which sidesteps all the webpack/Next bundling
// issues around onnxruntime's wasm worker glue. Only the ~small runtime comes
// from CDN; the model itself is served locally from /public.
import type { Evaluator } from "./mcts";
import { N_PLANES } from "./encoding";

const ORT_VERSION = "1.20.1";
const ORT_CDN = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;

export interface Engine {
  evaluator: Evaluator;
  backend: string;
}

let enginePromise: Promise<Engine> | null = null;

function loadScript(src: string): Promise<void> {
  return new Promise((resolve, reject) => {
    const existing = document.querySelector(`script[src="${src}"]`);
    if (existing) {
      resolve();
      return;
    }
    const s = document.createElement("script");
    s.src = src;
    s.async = true;
    s.onload = () => resolve();
    s.onerror = () => reject(new Error("failed to load " + src));
    document.head.appendChild(s);
  });
}

async function init(modelUrl: string): Promise<Engine> {
  await loadScript(ORT_CDN + "ort.webgpu.min.js");
  const ort: any = (window as any).ort;
  if (!ort) throw new Error("onnxruntime failed to initialise");
  ort.env.wasm.wasmPaths = ORT_CDN;
  ort.env.wasm.numThreads = 1;

  let session: any;
  let backend: string;
  try {
    session = await ort.InferenceSession.create(modelUrl, {
      executionProviders: ["webgpu"],
    });
    backend = "webgpu";
  } catch {
    session = await ort.InferenceSession.create(modelUrl, {
      executionProviders: ["wasm"],
    });
    backend = "wasm";
  }

  const evaluator: Evaluator = async (planes: Float32Array) => {
    const tensor = new ort.Tensor("float32", planes, [1, N_PLANES, 8, 8]);
    const out = await session.run({ board: tensor });
    return {
      logits: out.policy.data as Float32Array,
      value: (out.value.data as Float32Array)[0],
    };
  };

  return { evaluator, backend };
}

export function getEngine(modelUrl: string): Promise<Engine> {
  if (!enginePromise) enginePromise = init(modelUrl);
  return enginePromise;
}
