"""Render report.html: SNR curves + embedded reconstruction montages. Self-contained."""

import os
import json
import base64

HERE = os.path.dirname(os.path.abspath(__file__))
COL = {"latent_radio": "#6366f1", "digital": "#f59e0b",
       "desync": "#ef4444", "ood_digits": "#10b981"}
LAB = {"latent_radio": "Latent Radio (AI-native)", "digital": "Classical digital (Shannon-optimal)",
       "desync": "Failure: mismatched models", "ood_digits": "Failure: out-of-distribution"}


def b64(fname):
    with open(os.path.join(HERE, fname), "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode()


def curve_chart(R, keys, w=780, h=430, pad=58):
    g = R["snr_grid"]
    xmin, xmax = min(g), max(g)
    vals = [v for k in keys for v in R["curves"][k] if v is not None] + [R["outage_psnr"]]
    ymin, ymax = min(vals) - 1, max(vals) + 1
    X = lambda s: pad + (s - xmin) / (xmax - xmin) * (w - 2 * pad)
    Y = lambda v: h - pad - (v - ymin) / (ymax - ymin) * (h - 2 * pad)
    svg = [f'<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg" class="chart">']
    for i in range(6):
        v = ymin + (ymax - ymin) * i / 5
        y = Y(v)
        svg.append(f'<line x1="{pad}" y1="{y:.0f}" x2="{w-pad}" y2="{y:.0f}" class="grid"/>')
        svg.append(f'<text x="{pad-8}" y="{y+4:.0f}" class="ylab">{v:.0f}</text>')
    for s in g:
        svg.append(f'<text x="{X(s):.0f}" y="{h-pad+20}" class="xlab">{s}</text>')
    yo = Y(R["outage_psnr"])
    svg.append(f'<line x1="{pad}" y1="{yo:.0f}" x2="{w-pad}" y2="{yo:.0f}" class="baseline"/>')
    svg.append(f'<text x="{w-pad}" y="{yo-6:.0f}" class="baselab">blank-image floor</text>')
    for k in keys:
        c = R["curves"][k]
        if c is None:
            continue
        pts = " ".join(f"{X(s):.1f},{Y(v):.1f}" for s, v in zip(g, c))
        dash = ' stroke-dasharray="6 4"' if k in ("desync", "ood_digits") else ""
        svg.append(f'<polyline points="{pts}" fill="none" stroke="{COL[k]}" stroke-width="3"{dash}/>')
        for s, v in zip(g, c):
            svg.append(f'<circle cx="{X(s):.1f}" cy="{Y(v):.1f}" r="3" fill="{COL[k]}"/>')
    svg.append(f'<text x="{w/2:.0f}" y="{h-10}" class="axtitle">channel quality — SNR (dB)</text>')
    svg.append(f'<text x="15" y="{h/2:.0f}" class="axtitle" transform="rotate(-90 15 {h/2:.0f})">reconstruction PSNR (dB)</text>')
    svg.append("</svg>")
    return "\n".join(svg)


def legend(keys):
    return '<div class="legend card">' + "".join(
        f'<span><span class="dot" style="background:{COL[k]}"></span>{LAB[k]}</span>' for k in keys
    ) + "</div>"


def main():
    R = json.load(open(os.path.join(HERE, "results.json")))
    g = R["snr_grid"]
    lr = R["curves"]["latent_radio"]
    dg = R["curves"]["digital"]
    # headline numbers
    lo_i = 0  # worst SNR
    gain_lo = lr[lo_i] - dg[lo_i]
    hi_i = len(g) - 1
    gain_hi = lr[hi_i] - dg[hi_i]

    html = f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Latent Radio — AI-native communication</title>
<style>
 :root{{color-scheme:light dark}}
 body{{font:15px/1.6 -apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;max-width:880px;margin:0 auto;padding:34px 20px;background:#0b0f19;color:#e5e7eb}}
 h1{{font-size:28px;margin:0 0 4px}} h2{{font-size:20px;margin:38px 0 10px;border-bottom:1px solid #1f2937;padding-bottom:6px}}
 h3{{font-size:14px;color:#a5b4fc;margin:22px 0 6px;text-transform:uppercase;letter-spacing:.05em}}
 .sub{{color:#9ca3af;margin:0 0 18px}}
 .card{{background:#111827;border:1px solid #1f2937;border-radius:12px;padding:16px 20px;margin:14px 0}}
 .win{{border-color:#4338ca;background:#12142b}}
 .chart{{width:100%;height:auto;background:#0b0f19;border-radius:8px}}
 .grid{{stroke:#1f2937}} .baseline{{stroke:#64748b;stroke-dasharray:5 4}}
 text{{fill:#9ca3af;font:11px sans-serif}} .baselab{{fill:#64748b;text-anchor:end}}
 .ylab{{text-anchor:end}} .xlab{{text-anchor:middle}} .axtitle{{fill:#6b7280;text-anchor:middle;font-size:12px}}
 .legend span{{display:inline-block;margin:2px 16px 2px 0;font-size:13px;white-space:nowrap}}
 .dot{{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:8px;vertical-align:middle}}
 .key{{color:#818cf8;font-weight:600}} code{{background:#1f2937;padding:1px 6px;border-radius:4px;font-size:13px}}
 .big{{font-size:30px;font-weight:700}} .vs{{color:#6b7280;margin:0 8px}}
 .stat{{display:inline-block;margin:6px 24px 6px 0}} .stat .lab{{display:block;color:#9ca3af;font-size:12px}}
 img.montage{{width:100%;image-rendering:pixelated;border-radius:8px;border:1px solid #1f2937;margin-top:8px}}
 .rowkey{{font-size:12.5px;color:#9ca3af;margin:8px 0 0;line-height:1.9}}
 .rowkey b{{color:#e5e7eb}}
</style></head><body>

<h1>Latent Radio</h1>
<p class="sub">A working demo of <b>AI-native (semantic) communication</b>: two neural models —
a sender's encoder and a receiver's decoder — trained end-to-end to push images through a noisy
radio channel. No pretrained weights, no hand-designed codec. Trained on Fashion-MNIST, MPS, this session.</p>

<div class="card">
<b>The idea.</b> Classical digital comms <span class="key">separate</span> the problem: compress to bits (source
coding), then protect those bits (channel coding). Shannon proved that's optimal — but only for infinitely long
messages. For real images over a real channel, a <span class="key">jointly-learned analog code</span> that both
ends share can do better, and crucially it <span class="key">degrades gracefully</span> instead of falling off the
digital "cliff". The encoder and decoder ARE the shared model; what they transmit is closer to meaning than to bits.</p>
</div>

<h2>Result — graceful vs. the cliff</h2>
<div class="card win">
<div class="stat"><span class="big" style="color:#818cf8">+{gain_lo:.1f} dB</span><span class="lab">Latent Radio advantage at {g[lo_i]} dB (bad channel)</span></div>
<span class="vs">·</span>
<div class="stat"><span class="big" style="color:#818cf8">+{gain_hi:.1f} dB</span><span class="lab">advantage at {g[hi_i]} dB (good channel)</span></div>
<p style="margin:8px 0 0">Latent Radio beats the Shannon-optimal digital baseline at <b>every</b> channel quality, and the gap
widens as the channel worsens — at {g[lo_i]} dB the digital system has collapsed to the blank-image floor
({R['outage_psnr']:.1f} dB) while Latent Radio still delivers a recognizable image.</p>
</div>

{legend(["latent_radio", "digital"])}
{curve_chart(R, ["latent_radio", "digital"])}

<h3>See it — same images, both systems, three channel conditions</h3>
<img class="montage" src="{b64('recon.png')}"/>
<div class="rowkey">
Row 1 — <b>original</b>. Rows 2–4 — <b>Latent Radio</b> at {R['display_snrs'][0]}, {R['display_snrs'][1]}, {R['display_snrs'][2]} dB.
Rows 5–7 — <b>classical digital</b> at the same SNRs. At {R['display_snrs'][0]} dB the digital reconstructions are
unrecognizable smear; Latent Radio's are blurred but clearly the right garment.
</div>

<h2>Honest failure modes</h2>
<p class="sub">AI-native communication buys graceful degradation, but it has a hard requirement and a real fragility.
Both are shown here rather than hidden.</p>

{legend(["latent_radio", "desync", "ood_digits"])}
{curve_chart(R, ["latent_radio", "desync", "ood_digits"])}

<div class="card">
<h3 style="margin-top:0">1 · Both ends must share the exact same model</h3>
Pair a sender's encoder with a receiver's decoder from a <b>separately trained</b> model — each excellent on its
own — and the link produces garbage at <b>every</b> SNR, even a perfect channel (red, flat ~9 dB, below the
blank-image floor). There is no interoperability standard here the way there is for, say, Wi-Fi: the "protocol"
is the shared weights.
<img class="montage" src="{b64('desync.png')}" style="margin-top:10px"/>
<div class="rowkey">Row 1 — original. Row 2 — matched encoder+decoder (same model). Row 3 — encoder from model A,
decoder from model B, on a clean channel. Both models are individually good; together they are meaningless.</div>
</div>

<div class="card">
<h3 style="margin-top:0">2 · It leans on knowing the source distribution</h3>
Train on clothing, then transmit <b>handwritten digits</b> (green, dashed): quality drops well below the
in-distribution curve (to {R['curves']['ood_digits'][-1]:.1f} dB vs {lr[-1]:.1f} dB at the best SNR). The shared
model encodes priors about what it expects to see; off-distribution inputs cost you. A production system would need
a model trained on the true source, or an explicit fallback.
</div>

<h2>What this is and isn't</h2>
<div class="card">
<b>Is:</b> a real, reproducible demonstration that a shared learned model can transmit images over a noisy channel
more robustly than the classical separation architecture, with the signature graceful-degradation behavior — and an
honest accounting of the two failure modes that come with it.<br><br>
<b>Isn't:</b> a new physical layer or new spectrum (physics is fixed). It's a new <i>abstraction layer</i> — the same
kind of move radio and the internet were. This is one bandwidth (k={R['k']} real channel uses), one dataset, small
models. A real system would need larger models, real channel effects (fading, interference), and a shared-model
distribution/versioning story. The point is that the core mechanism works and behaves as theory predicts.
</div>

<p class="sub" style="margin-top:24px">Reproduce: <code>python3 train.py 0 &amp;&amp; python3 train.py 1 &amp;&amp; python3 evaluate.py &amp;&amp; python3 make_report.py</code></p>
</body></html>"""
    open(os.path.join(HERE, "report.html"), "w").write(html)
    print("wrote report.html")


if __name__ == "__main__":
    main()
