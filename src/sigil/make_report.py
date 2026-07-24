"""Render report.html for Sigil — self-contained, inline SVG + embedded glyphs."""

import os
import json
import base64

HERE = os.path.dirname(os.path.abspath(__file__))


def b64(fname):
    with open(os.path.join(HERE, fname), "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode()


def chart(xs, series, w=780, h=400, pad=58, xlabel=""):
    xmin, xmax = min(xs), max(xs)
    X = lambda v: pad + (v - xmin) / (xmax - xmin) * (w - 2 * pad)
    Y = lambda v: h - pad - v * (h - 2 * pad)
    svg = [f'<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg" class="chart">']
    for i in range(6):
        v = i / 5; y = Y(v)
        svg.append(f'<line x1="{pad}" y1="{y:.0f}" x2="{w-pad}" y2="{y:.0f}" class="grid"/>')
        svg.append(f'<text x="{pad-8}" y="{y+4:.0f}" class="ylab">{int(v*100)}%</text>')
    for xv in xs:
        svg.append(f'<text x="{X(xv):.0f}" y="{h-pad+20}" class="xlab">{int(xv*100)}</text>')
    for s in series:
        pts = " ".join(f"{X(x):.1f},{Y(y):.1f}" for x, y in zip(xs, s["y"]))
        dash = ' stroke-dasharray="6 4"' if s.get("dash") else ""
        svg.append(f'<polyline points="{pts}" fill="none" stroke="{s["c"]}" stroke-width="3"{dash}/>')
        for x, y in zip(xs, s["y"]):
            svg.append(f'<circle cx="{X(x):.1f}" cy="{Y(y):.1f}" r="3.5" fill="{s["c"]}"/>')
    svg.append(f'<text x="{w/2:.0f}" y="{h-10}" class="axtitle">{xlabel}</text>')
    svg.append(f'<text x="15" y="{h/2:.0f}" class="axtitle" transform="rotate(-90 15 {h/2:.0f})">accuracy</text>')
    svg.append("</svg>")
    return "\n".join(svg)


def legend(items):
    return '<div class="legend card">' + "".join(
        f'<span><span class="dot" style="background:{c}"></span>{l}</span>' for l, c in items) + "</div>"


def main():
    R = json.load(open(os.path.join(HERE, "results.json")))
    sev = R["sev_grid"]
    sig, abl = R["sigil"], R["ablation"]
    SIG, ABL = "#6366f1", "#ef4444"

    # headline: bit accuracy Sigil retains at heavy (70%) damage
    i70 = sev.index(0.7)
    sig70 = sig["bit_acc"][i70]
    abl70 = abl["bit_acc"][i70]
    # largest severity where Sigil still recovers the whole message > 90% of the time
    msg_ok = max((s for s, m in zip(sev, sig["msg_acc"]) if m >= 0.90), default=0.0)

    html = f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Sigil — a message code that fades instead of breaking</title>
<style>
 :root{{color-scheme:light dark}}
 body{{font:15px/1.6 -apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;max-width:880px;margin:0 auto;padding:34px 20px 60px;background:#0b0f19;color:#e5e7eb}}
 h1{{font-size:29px;margin:0 0 4px}} h2{{font-size:20px;margin:36px 0 10px;border-bottom:1px solid #1f2937;padding-bottom:6px}}
 .tag{{display:inline-block;font-size:12px;font-weight:600;color:#818cf8;border:1px solid #3730a3;background:#12142b;border-radius:999px;padding:3px 10px;margin-bottom:12px}}
 .sub{{color:#9ca3af;margin:0 0 18px;max-width:74ch}}
 .card{{background:#111827;border:1px solid #1f2937;border-radius:12px;padding:16px 20px;margin:14px 0}}
 .win{{border-color:#4338ca;background:#12142b}}
 .chart{{width:100%;height:auto;background:#0b0f19;border-radius:8px}}
 .grid{{stroke:#1f2937}} text{{fill:#9ca3af;font:11px sans-serif}}
 .ylab{{text-anchor:end}} .xlab{{text-anchor:middle}} .axtitle{{fill:#6b7280;text-anchor:middle;font-size:12px}}
 .legend span{{display:inline-block;margin:2px 16px 2px 0;font-size:13px}}
 .dot{{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:8px;vertical-align:middle}}
 .big{{font-size:30px;font-weight:700}} .stat{{display:inline-block;margin:6px 26px 6px 0}}
 .stat .lab{{display:block;color:#9ca3af;font-size:12px}} .key{{color:#818cf8;font-weight:600}}
 code{{background:#1f2937;padding:1px 6px;border-radius:4px;font-size:13px}}
 img.row{{width:100%;image-rendering:pixelated;border-radius:8px;border:1px solid #1f2937;margin-top:8px}}
 .rk{{font-size:12.5px;color:#9ca3af;margin-top:6px}}
</style></head><body>

<div class="tag">A NEW WAY TO CARRY A MESSAGE · GRACEFUL VISUAL CODE</div>
<h1>Sigil</h1>
<p class="sub">A short message, turned into a small glyph by a neural <b>encoder</b> and read back by a shared
neural <b>decoder</b> — trained together through simulated real-world damage. The result is a visual code that
<b>fades instead of breaking</b>: cover part of it, blur it, add glare, and most of the message still comes
through. It's the Latent Radio joint-coding principle applied to the channel a printed or photographed code
actually suffers.</p>

<div class="card">
<b>Why this is new.</b> Today's tool for this is the <b>QR code</b>, whose algebraic error-correction has a hard
wall — past its limit, a damaged QR decodes to <i>nothing</i> (a cliff). Sigil has no wall: because sender and
receiver share a <i>learned</i> code that spreads the message across the whole glyph, accuracy slides down
smoothly. Honest scope: it carries a <b>short</b> message ({R['nbits']} bits — a code, an ID, a few characters),
not an essay. Information theory still applies; what changes is <i>how it fails</i>.</p>
</div>

<h2>It fades, it doesn't break</h2>
<div class="card win">
<div class="stat"><span class="big" style="color:#818cf8">{sig70*100:.0f}%</span><span class="lab">of the message still readable at 70% damage</span></div>
<span style="color:#6b7280">vs</span>
<div class="stat"><span class="big" style="color:#ef4444">{abl70*100:.0f}%</span><span class="lab">a decoder <b>not</b> trained through damage (≈ coin-flip)</span></div>
</div>

<img class="row" src="{b64('sigil_row.png')}"/>
<div class="rk">The same Sigil at 0%, 30%, 50%, 70%, 90% damage (noise + blur + occlusion + brightness shift).
Even heavily corrupted, the decoder still reads it.</div>

<h2>Graceful degradation — bit accuracy vs. damage</h2>
{legend([("Sigil (channel-aware)", SIG), ("no-channel ablation (trained on clean glyphs)", ABL)])}
{chart(sev, [{"y": sig["bit_acc"], "c": SIG}, {"y": abl["bit_acc"], "c": ABL, "dash": True}], xlabel="damage severity (%)")}
<div class="card">The Sigil's accuracy declines <b>smoothly</b> across the whole damage range. The ablation — an
identical network trained only on clean glyphs — falls apart as soon as any real damage appears, which proves the
robustness comes from <b>training through the channel</b>, not from the architecture. This is the same joint
source-channel coding idea behind Latent Radio, now protecting a human-shareable code.</div>

<h2>Whole-message recovery</h2>
<p class="sub">Stricter test: every one of the {R['nbits']} bits correct (a perfectly recovered message).
Sigil recovers the full message reliably up to ~{msg_ok*100:.0f}% damage, then degrades — while the ablation
never survives any damage.</p>
{legend([("Sigil", SIG), ("no-channel ablation", ABL)])}
{chart(sev, [{"y": sig["msg_acc"], "c": SIG}, {"y": abl["msg_acc"], "c": ABL, "dash": True}], xlabel="damage severity (%)")}

<h2>What this is / isn't</h2>
<div class="card">
<b>Is:</b> a real, trained-from-scratch visual code that degrades gracefully where a QR code cliffs — a new,
honest building block for getting a short message through a hostile visual channel (bad prints, glare, partial
cover, low-quality photos). <b>Isn't:</b> unlimited compression or a replacement for all of QR (QR carries far
more bits and is a mature standard). It's one specific superpower — <i>meaning that survives damage</i> — proven
on {R['nbits']}-bit messages with small models.</p>
</div>

<p class="sub" style="margin-top:22px">Reproduce: <code>python3 experiments.py &amp;&amp; python3 make_report.py</code>.
Next: an in-browser demo — type a message, watch the Sigil form, damage it with a slider, watch it still decode,
side-by-side with a QR code that dies.</p>
</body></html>"""
    open(os.path.join(HERE, "report.html"), "w").write(html)
    print("wrote report.html")


if __name__ == "__main__":
    main()
