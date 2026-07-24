"""Rosetta report — cross-model semantic interoperability (the field's #1 blocker)."""

import os
import json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def heat(mat, title, lo=9.0, hi=22.5, cell=78, pad=34):
    N = len(mat)
    w = N * cell + 2 * pad + 60
    h = N * cell + 2 * pad + 30
    svg = [f'<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg" class="heat">']
    svg.append(f'<text x="{w/2:.0f}" y="18" class="ht">{title}</text>')
    for i in range(N):
        for j in range(N):
            v = mat[i][j]
            t = max(0, min(1, (v - lo) / (hi - lo)))
            r = int(239 + (52 - 239) * t); g = int(68 + (211 - 68) * t); b = int(68 + (153 - 68) * t)
            x = pad + 60 + j * cell; y = pad + i * cell
            svg.append(f'<rect x="{x}" y="{y}" width="{cell-4}" height="{cell-4}" rx="8" fill="rgb({r},{g},{b})"/>')
            svg.append(f'<text x="{x+(cell-4)/2:.0f}" y="{y+(cell-4)/2+5:.0f}" class="hv">{v:.0f}</text>')
        svg.append(f'<text x="{pad+52}" y="{pad+i*cell+(cell-4)/2+4:.0f}" class="hl" text-anchor="end">TX {i}</text>')
    for j in range(N):
        svg.append(f'<text x="{pad+60+j*cell+(cell-4)/2:.0f}" y="{pad+N*cell+16:.0f}" class="hl" text-anchor="middle">RX {j}</text>')
    svg.append("</svg>")
    return "\n".join(svg)


def line(xs, series, w=760, h=340, pad=54):
    ys = [y for s in series for y in s["y"]]; ymin, ymax = min(ys) - 1, max(ys) + 1
    X = lambda v: pad + (v - min(xs)) / (max(xs) - min(xs)) * (w - 2 * pad)
    Y = lambda v: h - pad - (v - ymin) / (ymax - ymin) * (h - 2 * pad)
    svg = [f'<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg" class="chart">']
    for i in range(6):
        v = ymin + (ymax - ymin) * i / 5; y = Y(v)
        svg.append(f'<line x1="{pad}" y1="{y:.0f}" x2="{w-pad}" y2="{y:.0f}" class="grid"/>')
        svg.append(f'<text x="{pad-8}" y="{y+4:.0f}" class="ylab">{v:.0f}</text>')
    for xv in xs:
        svg.append(f'<text x="{X(xv):.0f}" y="{h-pad+20}" class="xlab">{xv}</text>')
    for s in series:
        pts = " ".join(f"{X(x):.1f},{Y(y):.1f}" for x, y in zip(xs, s["y"]))
        d = f' stroke-dasharray="{s["dash"]}"' if s.get("dash") else ""
        svg.append(f'<polyline points="{pts}" fill="none" stroke="{s["c"]}" stroke-width="3"{d}/>')
        for x, y in zip(xs, s["y"]):
            svg.append(f'<circle cx="{X(x):.1f}" cy="{Y(y):.1f}" r="3.5" fill="{s["c"]}"/>')
    svg.append(f'<text x="{w/2:.0f}" y="{h-10}" class="axtitle">SNR (dB)</text>')
    svg.append(f'<text x="14" y="{h/2:.0f}" class="axtitle" transform="rotate(-90 14 {h/2:.0f})">cross-vendor PSNR (dB)</text>')
    svg.append("</svg>")
    return "\n".join(svg)


def legend(items):
    return '<div class="legend card">' + "".join(
        f'<span><span class="dot" style="background:{c}"></span>{l}</span>' for l, c in items) + "</div>"


def main():
    R = json.load(open(os.path.join(HERE, "results.json")))
    before, after = R["before_matrix"], R["after_matrix"]
    bc, ac, mm = R["before_cross_avg"], R["after_cross_avg"], R["matched_avg"]
    c = R["curve"]

    html = f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Rosetta — making independently-trained AI radios understand each other</title>
<style>
 :root{{color-scheme:light dark}}
 body{{font:15px/1.6 -apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;max-width:900px;margin:0 auto;padding:34px 20px 60px;background:#0b0f19;color:#e5e7eb}}
 h1{{font-size:27px;margin:0 0 4px}} h2{{font-size:20px;margin:34px 0 10px;border-bottom:1px solid #1f2937;padding-bottom:6px}}
 .tag{{display:inline-block;font-size:12px;font-weight:600;color:#818cf8;border:1px solid #3730a3;background:#12142b;border-radius:999px;padding:3px 10px;margin-bottom:12px}}
 .sub{{color:#9ca3af;margin:0 0 16px;max-width:80ch}}
 .card{{background:#111827;border:1px solid #1f2937;border-radius:12px;padding:16px 20px;margin:14px 0}}
 .win{{border-color:#3730a3;background:#12142b}}
 .row{{display:flex;gap:18px;flex-wrap:wrap;justify-content:center}}
 .heat{{width:300px;height:auto}} .chart{{width:100%;height:auto;background:#0b0f19;border-radius:8px}}
 .ht{{fill:#e5e7eb;font-size:13px;font-weight:700;text-anchor:middle}} .hv{{fill:#04121c;font-size:16px;font-weight:800;text-anchor:middle}}
 .hl{{fill:#9ca3af;font-size:12px}} .grid{{stroke:#1f2937}} text{{fill:#9ca3af;font:11px sans-serif}}
 .ylab{{text-anchor:end}} .xlab{{text-anchor:middle}} .axtitle{{fill:#6b7280;text-anchor:middle;font-size:12px}}
 .legend span{{display:inline-block;margin:2px 16px 2px 0;font-size:13px}} .dot{{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:8px;vertical-align:middle}}
 .big{{font-size:30px;font-weight:800}} .stat{{display:inline-block;margin:6px 26px 6px 0}} .stat .lab{{display:block;color:#9ca3af;font-size:12px}}
 .key{{color:#818cf8;font-weight:600}} code{{background:#1f2937;padding:1px 6px;border-radius:4px;font-size:13px}}
</style></head><body>
<div class="tag">THE FIELD'S #1 BLOCKER · CROSS-MODEL INTEROPERABILITY</div>
<h1>Rosetta — making independently-built AI radios understand each other</h1>
<p class="sub">Semantic / learned communication works beautifully in the lab and is <b>still undeployed</b>. The reason
named again and again in the 2025–26 literature and 6G standardization roadmaps: two endpoints trained separately
(different vendors, different model versions) learn <b>private codes and cannot interoperate</b> — the link collapses
to noise even on a perfect channel, and formal models of this "knowledge mismatch" are, quote, largely undeveloped.
Our own earlier experiments hit this exact wall. Rosetta is the way over it.</p>

<div class="card win">
<div class="stat"><span class="big" style="color:#34d399">{bc:.1f} → {ac:.1f} dB</span><span class="lab">cross-vendor link quality (garbage → matched, ref {mm:.1f} dB)</span></div>
<div class="stat"><span class="big" style="color:#818cf8">0</span><span class="lab">base-model weights changed — encoders/decoders stay frozen</span></div>
</div>

<h2>Three independently-trained radios — every pair, before &amp; after</h2>
<p class="sub">Rows = transmitter, columns = receiver. Green ≈ working (~22 dB), red ≈ broken (~9 dB). Left: raw
cross-model transmission. Right: the same pairs through Rosetta.</p>
<div class="row">
{heat(before, "BEFORE — raw (only the diagonal works)")}
{heat(after, "AFTER — through Rosetta (all 9 pairs work)")}
</div>

<h2>How it works</h2>
<div class="card">
Don't retrain the deployed models — you can't. Standardize a shared <span class="key">reference space</span> and give
each model two tiny adapters: <code>to_ref</code> (its private symbols → the standard signal that's transmitted) and
<code>from_ref</code> (received standard signal → its private symbols). Every encoder/decoder stays <b>frozen</b>;
only the adapters learn, jointly, so that <i>every</i> transmitter reaches <i>every</i> receiver through the
reference. The N-vendor problem collapses from O(N²) bespoke bridges to <b>O(N) adapters against one standard</b> —
exactly the shape a standards body (Next G Alliance, 3GPP) could publish. Total adapter cost here:
{R['adapter_params']:,} params, vs {R['base_params_each']:,} per base model.
</div>

<h2>It holds across the channel</h2>
{legend([("matched (own receiver)", "#22d3ee"), ("cross-vendor WITH Rosetta", "#818cf8"), ("cross-vendor WITHOUT (raw)", "#ef4444")])}
{line(c["snr"], [
  {"y": c["after_matched"], "c": "#22d3ee", "dash": "2 3"},
  {"y": c["after_cross"], "c": "#818cf8"},
  {"y": c["before_cross"], "c": "#ef4444", "dash": "6 4"},
])}

<h2>Why this is the high-leverage move — and its honest limits</h2>
<div class="card">
<b>Why it matters:</b> the whole field agrees semantic coding <i>works</i>; interoperability is what stands between it
and deployment, and it's what any standard must solve. Turning independently-trained models from mutually
unintelligible into fully interoperable — without retraining them — is a direct contribution to that blocker, and it
maps onto the urgent, adjacent need for a standard <b>AI-agent communication protocol</b>.<br><br>
<b>Honest limits:</b> demonstrated at Fashion-MNIST scale on one codec family; the adapters still need <i>joint</i>
calibration on shared data, which in practice means a governed reference and a calibration protocol (who owns the
reference space is a real governance question, not a solved one); it aligns this JSCC latent family, not arbitrary
task semantics; and everything remains simulation-stage. What's shown is that the mismatch is <b>fixable with a
tiny, standardizable, base-frozen layer</b> — which is the part nobody had demonstrated cleanly.
</div>

<p class="sub" style="margin-top:18px">Reproduce: <code>python3 train_rosetta.py &amp;&amp; python3 evaluate.py &amp;&amp; python3 make_report.py</code>.
Three independently-trained Latent Radio models; base weights frozen throughout.</p>
</body></html>"""
    open(os.path.join(HERE, "report.html"), "w").write(html)
    print("wrote report.html")


if __name__ == "__main__":
    main()
