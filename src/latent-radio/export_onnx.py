"""Export trained encoder + decoder to ONNX for in-browser inference, and verify."""

import os
import numpy as np
import torch

from model import LatentRadio
from data import load_fashion

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "web")
os.makedirs(OUT, exist_ok=True)


def export_model(seed, enc_name, dec_name):
    ckpt = torch.load(os.path.join(HERE, "models", f"model_{seed}.pt"), map_location="cpu")
    m = LatentRadio(k=ckpt["k"])
    m.load_state_dict(ckpt["state_dict"])
    m.eval()

    dummy_img = torch.randn(1, 1, 28, 28)
    torch.onnx.export(
        m.enc, dummy_img, os.path.join(OUT, enc_name),
        input_names=["image"], output_names=["symbols"],
        dynamic_axes={"image": {0: "batch"}, "symbols": {0: "batch"}},
        opset_version=17,
    )
    dummy_sym = torch.randn(1, ckpt["k"])
    torch.onnx.export(
        m.dec, dummy_sym, os.path.join(OUT, dec_name),
        input_names=["symbols"], output_names=["recon"],
        dynamic_axes={"symbols": {0: "batch"}, "recon": {0: "batch"}},
        opset_version=17,
    )
    return m, ckpt["k"]


def main():
    m0, k = export_model(0, "encoder.onnx", "decoder.onnx")
    m1, _ = export_model(1, "encoder1.onnx", "decoder1.onnx")  # for the desync demo
    print("exported encoder/decoder (+ mismatched pair) to web/")

    # ---- verify ONNX matches PyTorch ----
    try:
        import onnxruntime as ort
    except ImportError:
        print("onnxruntime not installed locally; skipping numeric check "
              "(browser uses onnxruntime-web).")
    else:
        x = load_fashion("test", limit=4)
        enc = ort.InferenceSession(os.path.join(OUT, "encoder.onnx"))
        dec = ort.InferenceSession(os.path.join(OUT, "decoder.onnx"))
        sym = enc.run(None, {"image": x.numpy()})[0]
        rec = dec.run(None, {"symbols": sym})[0]
        with torch.no_grad():
            sym_t = m0.enc(x).numpy()
            rec_t = m0.dec(torch.from_numpy(sym)).numpy()
        print("symbols max abs diff:", float(np.abs(sym - sym_t).max()))
        print("recon   max abs diff:", float(np.abs(rec - rec_t).max()))

    # ---- export a gallery of sample images as JSON (uint8 pixel arrays) ----
    import json
    gal = load_fashion("test", limit=12)
    arr = (gal[:, 0].numpy() * 255).astype(np.uint8).reshape(12, -1).tolist()
    json.dump({"k": k, "images": arr}, open(os.path.join(OUT, "gallery.json"), "w"))
    print("wrote web/gallery.json (12 sample images)")

    for f in sorted(os.listdir(OUT)):
        p = os.path.join(OUT, f)
        print(f"  {f:16s} {os.path.getsize(p)/1024:8.1f} KB")


if __name__ == "__main__":
    main()
