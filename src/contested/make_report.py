"""Render report.html for the contested-channel stress test."""

import os
import json
import base64

HERE = os.path.dirname(os.path.abspath(__file__))


def b64(fname):
    with open(os.path.join(HERE, fname), "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode()


def chart(xs, series, w=780, h=400, pad=58, xlabel="", ylabel="reconstruction PSNR (dB)",
          xfmt=lambda v: f"{v}", floor=None, floorlab=""):
    ys = [y for s in series for y in s["y"]] + ([floor] if floor else [])
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
        svg.append(f'<text x="{X(xv):.0f}" y="{h-pad+20}" class="xlab">{xfmt(xv)}</text>')
    if floor is not None:
        yf = Y(floor)
        svg.append(f'<line x1="{pad}" y1="{yf:.0f}" x2="{w-pad}" y2="{yf:.0f}" class="baseline"/>')
        svg.append(f'<text x="{w-pad}" y="{yf-6:.0f}" class="baselab">{floorlab}</text>')
    for s in series:
        pts = " ".join(f"{X(x):.1f},{Y(y):.1f}" for x, y in zip(xs, s["y"]))
        dash = ' stroke-dasharray="6 4"' if s.get("dash") else ""
        svg.append(f'<polyline points="{pts}" fill="none" stroke="{s["c"]}" stroke-width="3"{dash}/>')
        for x, y in zip(xs, s["y"]):
            svg.append(f'<circle cx="{X(x):.1f}" cy="{Y(y):.1f}" r="3.5" fill="{s["c"]}"/>')
    svg.append(f'<text x="{w/2:.0f}" y="{h-10}" class="axtitle">{xlabel}</text>')
    svg.append(f'<text x="15" y="{h/2:.0f}" class="axtitle" transform="rotate(-90 15 {h/2:.0f})">{ylabel}</text>')
    svg.append("</svg>")
    return "\n".join(svg)


def legend(items):
    return '<div class="legend card">' + "".join(
        f'<span><span class="dot" style="background:{c}"></span>{l}</span>' for l, c in items) + "</div>"


def main():
    R = json.load(open(os.path.join(HERE, "results.json")))
    jam = R["jam_grid"]; sg = R["snr_grid"]
    CON, AWG, DIG = "#6366f1", "#ef4444", "#f59e0b"
    e1, e2 = R["e1"], R["e2"]

    con_drop = e1["contested"][0] - e1["contested"][-1]
    awg_drop = e1["awgn_only"][0] - e1["awgn_only"][-1]
    con70, awg70, dig70 = e1["contested"][-1], e1["awgn_only"][-1], e1["digital"][-1]

    html = f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Latent Radio under fire — contested-channel stress test</title>
<style>
 :root{{color-scheme:light dark}}
 body{{font:15px/1.6 -apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;max-width:880px;margin:0 auto;padding:34px 20px 60px;background:#0b0f19;color:#e5e7eb}}
 h1{{font-size:28px;margin:0 0 4px}} h2{{font-size:20px;margin:36px 0 10px;border-bottom:1px solid #1f2937;padding-bottom:6px}}
 .tag{{display:inline-block;font-size:12px;font-weight:600;color:#818cf8;border:1px solid #3730a3;background:#12142b;border-radius:999px;padding:3px 10px;margin-bottom:12px}}
 .sub{{color:#9ca3af;margin:0 0 18px;max-width:76ch}}
 .card{{background:#111827;border:1px solid #1f2937;border-radius:12px;padding:16px 20px;margin:14px 0}}
 .win{{border-color:#4338ca;background:#12142b}}
 .chart{{width:100%;height:auto;background:#0b0f19;border-radius:8px}}
 .grid{{stroke:#1f2937}} .baseline{{stroke:#64748b;stroke-dasharray:5 4}}
 text{{fill:#9ca3af;font:11px sans-serif}} .baselab{{fill:#64748b;text-anchor:end}}
 .ylab{{text-anchor:end}} .xlab{{text-anchor:middle}} .axtitle{{fill:#6b7280;text-anchor:middle;font-size:12px}}
 .legend span{{display:inline-block;margin:2px 16px 2px 0;font-size:13px}}
 .dot{{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:8px;vertical-align:middle}}
 .big{{font-size:30px;font-weight:700}} .stat{{display:inline-block;margin:6px 26px 6px 0}}
 .stat .lab{{display:block;color:#9ca3af;font-size:12px}} .key{{color:#818cf8;font-weight:600}}
 code{{background:#1f2937;padding:1px 6px;border-radius:4px;font-size:13px}}
 img.row{{width:100%;image-rendering:pixelated;border-radius:8px;border:1px solid #1f2937;margin-top:8px}}
 .rk{{font-size:12.5px;color:#9ca3af;margin-top:6px;line-height:1.8}} .rk b{{color:#e5e7eb}}
</style></head><body>

<div class="tag">CONTESTED-CHANNEL STRESS TEST · SPACE &amp; DEFENSE</div>
<h1>Latent Radio, under fire</h1>
<p class="sub">The shared-model transceiver from Latent Radio, retrained and stress-tested against what
<b>deep-space and jammed military links actually face</b>: burst jamming (chunks of the signal wiped out),
deep fades, and extreme low SNR. The question — does "meaning survives damage" hold up when the channel is
actively hostile, not just noisy?</p>

<div class="card">
<b>Why this technology fits space &amp; defense.</b> These are <span class="key">closed systems</span> — one
operator controls both the transmitter and the receiver — so the shared-model requirement that sinks consumer
use is a natural fit (and doubles as implicit keying). They are <span class="key">bandwidth- and power-starved</span>,
with <span class="key">no retransmission</span> (a Mars round-trip is tens of minutes; you can't ask again under
jamming). That is exactly the regime where graceful degradation beats a digital cliff.</p>
</div>

<h2>Result — jam-resistance</h2>
<div class="card win">
<div class="stat"><span class="big" style="color:#818cf8">−{con_drop:.1f} dB</span><span class="lab">contested-trained model, 0% → 70% burst jamming</span></div>
<span style="color:#6b7280">vs</span>
<div class="stat"><span class="big" style="color:#ef4444">−{awg_drop:.1f} dB</span><span class="lab">same model trained for clean noise only</span></div>
<p style="margin:8px 0 0">Trained through the threat, the link barely flinches as a jammer wipes out up to 70% of
the signal. At 70% jamming it holds <b>{con70:.1f} dB</b> — versus <b>{awg70:.1f} dB</b> for the clean-noise model
(which collapses <i>below</i> even the classical baseline) and <b>{dig70:.1f} dB</b> for idealized digital.</p>
</div>

{legend([("contested-trained (threat-aware)", CON), ("AWGN-only model (ablation)", AWG), ("classical digital, idealized", DIG)])}
{chart(jam, [{"y": e1["contested"], "c": CON}, {"y": e1["awgn_only"], "c": AWG, "dash": True}, {"y": e1["digital"], "c": DIG, "dash": True}],
       xlabel="burst jamming — fraction of signal wiped out", xfmt=lambda v: f"{int(v*100)}%")}
<div class="card">The honest lesson is the red line: a transceiver tuned for clean noise — stock Latent Radio —
<b>degrades badly under jamming</b>, worse than classical digital. Robustness isn't automatic; it comes from
<b>training through the actual threat model</b>. Do that, and the shared-model system is remarkably jam-resistant.</div>

<h2>See it — images under jamming</h2>
<img class="row" src="{b64('contested_montage.png')}"/>
<div class="rk">Top: originals. Next three rows: <b>contested-trained</b> at 0%, 25%, 45% jamming — still clearly
readable. Then the <b>clean-noise model</b> and <b>classical digital</b> at 45% jamming, both visibly worse.</div>

<h2>Extreme low SNR, with jamming</h2>
<p class="sub">Holding a {int(R['jam_fixed']*100)}% jammer on, sweeping the noise floor. Below −4 dB the digital
link runs out of capacity and drops to the <b>blank-image floor</b>; the shared-model system keeps delivering a
recognizable picture.</p>
{legend([("contested-trained", CON), ("classical digital, idealized", DIG)])}
{chart(sg, [{"y": e2["contested"], "c": CON}, {"y": e2["digital"], "c": DIG, "dash": True}],
       xlabel="channel quality — SNR (dB)", floor=R["outage"], floorlab="blank-image floor")}

<h2>What this is / isn't</h2>
<div class="card">
<b>Is:</b> honest evidence that the shared-model / joint-coding idea holds up under contested conditions — burst
jamming, fades, extreme SNR — and a demonstration that you must <i>train for the threat</i> to get there. The
closed-system nature of space and defense makes the approach a natural fit.<br><br>
<b>Isn't:</b> a deployable system. This is a simulation with idealized channel models (real RF adds Doppler,
timing, hardware limits, adaptive jammers). Two more honest cautions specific to these domains: a neural decoder
can <b>hallucinate</b> plausible-but-wrong detail — dangerous for science or targeting data, so you'd send
verifiable residuals — and a shared model is <b>not cryptography</b> (layer real crypto on top). The digital
baseline here is deliberately generous (ideal interleaving), so the true classical cliff under bursts is harder
than shown; the neural advantage is if anything understated.</p>
</div>

<p class="sub" style="margin-top:22px">Reproduce: <code>python3 train_contested.py &amp;&amp; python3 evaluate_contested.py &amp;&amp; python3 make_report.py</code></p>
</body></html>"""
    open(os.path.join(HERE, "report.html"), "w").write(html)
    print("wrote report.html")


if __name__ == "__main__":
    main()
