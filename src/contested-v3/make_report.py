"""Render the v3 report — honest mixed result (adaptivity failed, robustness worked)."""

import os
import json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def line_chart(xs, series, w=780, h=380, pad=58, xlabel="", cross=None):
    ys = [y for s in series for y in s["y"]]
    ymin, ymax = min(ys) - 1, max(ys) + 1
    xmin, xmax = min(xs), max(xs)
    X = lambda v: pad + (v - xmin) / (xmax - xmin) * (w - 2 * pad)
    Y = lambda v: h - pad - (v - ymin) / (ymax - ymin) * (h - 2 * pad)
    svg = [f'<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg" class="chart">']
    for i in range(6):
        v = ymin + (ymax - ymin) * i / 5; y = Y(v)
        svg.append(f'<line x1="{pad}" y1="{y:.0f}" x2="{w-pad}" y2="{y:.0f}" class="grid"/>')
        svg.append(f'<text x="{pad-8}" y="{y+4:.0f}" class="ylab">{v:.0f}</text>')
    for xv in xs:
        svg.append(f'<text x="{X(xv):.0f}" y="{h-pad+20}" class="xlab">{xv}</text>')
    if cross is not None:
        cx = X(cross)
        svg.append(f'<line x1="{cx:.0f}" y1="{pad}" x2="{cx:.0f}" y2="{h-pad}" stroke="#475569" stroke-dasharray="4 4"/>')
        svg.append(f'<text x="{cx+5:.0f}" y="{pad+12}" style="fill:#94a3b8">crossover</text>')
    for s in series:
        pts = " ".join(f"{X(x):.1f},{Y(y):.1f}" for x, y in zip(xs, s["y"]))
        dash = f' stroke-dasharray="{s["dash"]}"' if s.get("dash") else ""
        svg.append(f'<polyline points="{pts}" fill="none" stroke="{s["c"]}" stroke-width="3"{dash}/>')
        for x, y in zip(xs, s["y"]):
            svg.append(f'<circle cx="{X(x):.1f}" cy="{Y(y):.1f}" r="3" fill="{s["c"]}"/>')
    svg.append(f'<text x="{w/2:.0f}" y="{h-10}" class="axtitle">{xlabel}</text>')
    svg.append(f'<text x="15" y="{h/2:.0f}" class="axtitle" transform="rotate(-90 15 {h/2:.0f})">PSNR (dB)</text>')
    svg.append("</svg>")
    return "\n".join(svg)


def grouped_bars(groups, seriesnames, colors, w=680, h=340, pad=54):
    ng, ns = len(groups), len(seriesnames)
    vals = [v for g in groups for v in g["vals"]]
    ymax = max(vals) * 1.12
    gw = (w - 2 * pad) / ng
    bw = gw * 0.7 / ns
    Y = lambda v: h - pad - v / ymax * (h - 2 * pad)
    svg = [f'<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg" class="chart">']
    for i in range(6):
        v = ymax * i / 5; y = Y(v)
        svg.append(f'<line x1="{pad}" y1="{y:.0f}" x2="{w-pad}" y2="{y:.0f}" class="grid"/>')
        svg.append(f'<text x="{pad-8}" y="{y+4:.0f}" class="ylab">{v:.0f}</text>')
    for gi, g in enumerate(groups):
        gx = pad + gi * gw + gw * 0.15
        for si, v in enumerate(g["vals"]):
            x = gx + si * bw
            y = Y(v)
            svg.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw-3:.1f}" height="{h-pad-y:.1f}" rx="3" fill="{colors[si]}"/>')
            svg.append(f'<text x="{x+bw/2-1:.1f}" y="{y-5:.1f}" style="fill:#e5e7eb;text-anchor:middle;font-size:11px">{v:.1f}</text>')
        svg.append(f'<text x="{pad+gi*gw+gw/2:.1f}" y="{h-pad+20}" class="xlab">{g["label"]}</text>')
    svg.append("</svg>")
    return "\n".join(svg)


def legend(items):
    return '<div class="legend card">' + "".join(
        f'<span><span class="dot" style="background:{c}"></span>{l}</span>' for l, c in items) + "</div>"


def main():
    R = json.load(open(os.path.join(HERE, "results.json")))
    sg = R["snr_grid"]; A = R["A"]; B = R["B"]
    AD, NA, DG = "#6366f1", "#22d3ee", "#f59e0b"

    a = np.array(A["adaptive"]); d = np.array(A["digital"])
    cross = float(np.array(sg)[np.argmin(np.abs(a - d))])
    hi_gap = A["digital"][-1] - A["adaptive"][-1]

    ba, br = B["adaptive"], B["robust"]
    foll_damage = ba["none"] - ba["follower"]
    foll_recover = br["follower"] - ba["follower"]
    clean_cost = ba["none"] - br["none"]

    RB, RD = "#818cf8", "#ef4444"

    html = f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Rate-adaptivity &amp; reactive jamming — the honest mixed result</title>
<style>
 :root{{color-scheme:light dark}}
 body{{font:15px/1.6 -apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;max-width:880px;margin:0 auto;padding:34px 20px 60px;background:#0b0f19;color:#e5e7eb}}
 h1{{font-size:27px;margin:0 0 4px}} h2{{font-size:20px;margin:34px 0 10px;border-bottom:1px solid #1f2937;padding-bottom:6px}}
 .tag{{display:inline-block;font-size:12px;font-weight:600;color:#818cf8;border:1px solid #3730a3;background:#12142b;border-radius:999px;padding:3px 10px;margin-bottom:12px}}
 .sub{{color:#9ca3af;margin:0 0 16px;max-width:78ch}}
 .card{{background:#111827;border:1px solid #1f2937;border-radius:12px;padding:16px 20px;margin:14px 0}}
 .fail{{border-color:#7c2d12;background:#1a1206}} .win{{border-color:#3730a3;background:#12142b}}
 .chart{{width:100%;height:auto;background:#0b0f19;border-radius:8px}}
 .grid{{stroke:#1f2937}} text{{fill:#9ca3af;font:11px sans-serif}}
 .ylab{{text-anchor:end}} .xlab{{text-anchor:middle}} .axtitle{{fill:#6b7280;text-anchor:middle;font-size:12px}}
 .legend span{{display:inline-block;margin:2px 16px 2px 0;font-size:13px}}
 .dot{{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:8px;vertical-align:middle}}
 .big{{font-size:28px;font-weight:700}} .stat{{display:inline-block;margin:6px 24px 6px 0}}
 .stat .lab{{display:block;color:#9ca3af;font-size:12px}} .key{{color:#818cf8;font-weight:600}}
 code{{background:#1f2937;padding:1px 6px;border-radius:4px;font-size:13px}}
 .verdict b{{color:#e5e7eb}}
</style></head><body>

<div class="tag">CLOSING THE GAPS · RATE-ADAPTIVITY + REACTIVE JAMMING</div>
<h1>Two fixes attempted. One failed, one worked.</h1>
<p class="sub">The rigorous v2 test left two honest gaps: the learned code lost to digital in good channels, and it
faced only a non-reactive jammer. Here I tried to close both — an SNR-adaptive encoder/decoder, and a reactive
jammer with adversarial training. Reporting both outcomes straight, including the one that didn't pan out.</p>

<h2>Attempt 1 — rate adaptivity <span style="color:#f97316">(did not close the gap)</span></h2>
<p class="sub">Idea: condition the encoder and decoder on the operating SNR so they can encode fine detail when
the channel is good and coarse-but-robust structure when it isn't — the JSCC analogue of adaptive modulation.</p>
{legend([("SNR-adaptive JSCC", AD), ("non-adaptive JSCC (v2)", NA), ("rate-adaptive digital", DG)])}
{line_chart(sg, [{"y": A["adaptive"], "c": AD}, {"y": A["nonadaptive_v2"], "c": NA, "dash": "2 3"}, {"y": A["digital"], "c": DG, "dash": "6 4"}], xlabel="SNR (dB)", cross=cross)}
<div class="card fail">
<b>The honest negative:</b> the SNR-adaptive curve sits almost exactly on the non-adaptive one, and both still
lose to digital above ~{cross:.0f} dB (by {hi_gap:.1f} dB at 18 dB). Conditioning made the model <i>aware</i> of the
SNR but didn't give it more to <i>send</i> — a fixed-bandwidth analog code can't manufacture the extra information
a good channel could carry. The low-SNR-analog / high-SNR-digital <b>crossover is fundamental</b>, not a tuning
artifact. Genuinely closing it needs a different mechanism — variable bandwidth (send more symbols when SNR allows)
or a hybrid digital-analog code — not just conditioning. Worth knowing, and worth not overclaiming.</div>

<h2>Attempt 2 — reactive jamming &amp; adversarial training <span style="color:#34d399">(worked)</span></h2>
<p class="sub">A <b>follower jammer</b> senses which symbols carry the most energy and jams exactly those; a
<b>worst-case adversarial jammer</b> runs gradient ascent to find the most damaging power-constrained perturbation.
We compare the standard adaptive model against one <b>adversarially trained</b> against the follower jammer.</p>
{legend([("standard model", RD), ("adversarially-trained model", RB)])}
{grouped_bars([
  {"label": "no jammer", "vals": [ba["none"], br["none"]]},
  {"label": "follower jammer", "vals": [ba["follower"], br["follower"]]},
  {"label": "worst-case adversarial", "vals": [ba["adversarial"], br["adversarial"]]},
], ["standard", "robust"], [RD, RB])}
<div class="card win">
<div class="stat"><span class="big" style="color:#ef4444">−{foll_damage:.1f} dB</span><span class="lab">the follower jammer's damage to the standard model</span></div>
<div class="stat"><span class="big" style="color:#818cf8">+{foll_recover:.1f} dB</span><span class="lab">recovered by adversarial training</span></div>
<p style="margin:8px 0 0">A sensing jammer is a <b>real threat</b> to a neural link — it cuts the standard model from
{ba['none']:.1f} to {ba['follower']:.1f} dB. Training against it recovers {foll_recover:.1f} dB, and the robustness
<b>transfers</b> to the worst-case adversarial jammer ({br['adversarial']:.1f} vs {ba['adversarial']:.1f} dB). The
cost is small: {clean_cost:.1f} dB on a clean channel. This is the defense-relevant win — and it says a neural comms
system must be <b>adversarially hardened by design</b>, not just trained on nice noise.</p>
</div>

<h2>Honest verdict</h2>
<div class="card verdict">
Two attempts, reported straight: <b>rate-adaptivity via conditioning failed</b> to beat digital in good channels
(the crossover is fundamental to analog JSCC — a real architectural limit, not a bug), while <b>adversarial
training succeeded</b> at defending a reactive jammer that badly hurts an unhardened model. Net picture across
v1→v3: this technology's honest home is the <b>hard, contested, low-SNR regime</b> — where it degrades gracefully,
never outages, and (once hardened) resists smart jamming — and it should <b>hand off to classical coding when the
channel is good</b>. Remaining gaps stay as stated in v2: still a simulation, perfect receiver CSI, information-
theoretic digital baseline, and hallucination risk that argues for verifiable residuals in critical uses.
</div>

<p class="sub" style="margin-top:20px">Reproduce: <code>python3 train.py both &amp;&amp; python3 evaluate.py &amp;&amp; python3 make_report.py</code>.</p>
</body></html>"""
    open(os.path.join(HERE, "report.html"), "w").write(html)
    print("wrote report.html")


if __name__ == "__main__":
    main()
