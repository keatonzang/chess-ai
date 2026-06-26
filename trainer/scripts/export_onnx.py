"""Export a trained checkpoint to ONNX for in-browser inference."""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from chessai.model import build_model
from chessai.encoding import N_PLANES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="../models/base_best.pt")
    ap.add_argument("--out", default="../web/public/model/chessnet.onnx")
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu")
    channels = ckpt.get("channels", 128)
    blocks = ckpt.get("blocks", 10)
    model = build_model(channels, blocks)
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    model.eval()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    dummy = torch.zeros(1, N_PLANES, 8, 8)

    torch.onnx.export(
        model, dummy, args.out,
        input_names=["board"],
        output_names=["policy", "value"],
        dynamic_axes={"board": {0: "batch"},
                      "policy": {0: "batch"},
                      "value": {0: "batch"}},
        opset_version=args.opset,
        do_constant_folding=True,
    )
    print(f"[export] wrote {args.out}  ({channels}ch/{blocks}blk)")

    # parity check
    try:
        import onnxruntime as ort
    except ImportError:
        print("[export] onnxruntime not installed; skipping parity check")
        return
    sess = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
    x = np.random.randn(4, N_PLANES, 8, 8).astype(np.float32)
    with torch.no_grad():
        tp, tv = model(torch.from_numpy(x))
    op, ov = sess.run(None, {"board": x})
    dp = np.abs(tp.numpy() - op).max()
    dv = np.abs(tv.numpy() - ov.reshape(tv.shape)).max()
    print(f"[export] parity max|dpolicy|={dp:.2e} max|dvalue|={dv:.2e}")
    size_mb = os.path.getsize(args.out) / 1e6
    print(f"[export] size: {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
