# Rosetta: Cross-Model Interoperability for Learned Communication via Base-Frozen Reference Adapters

**Priyankar Chakraborty** — P CHAK Consulting — priyankar@pchakconsulting.com
*July 2026 · preprint draft*

> Code: https://github.com/suncal/latent-comms · Live study: https://suncal.github.io/latent-comms/projects/rosetta.html

---

## Abstract

Learned ("semantic") joint source–channel coding (JSCC) consistently outperforms classical separation in simulation, yet remains undeployed. A recurring reason cited in the 2025–26 literature and 6G standardization roadmaps is **cross-model interoperability**: two transceivers trained independently — by different vendors, or as different model versions — learn private latent codes and cannot understand one another, so the link collapses to noise even over a clean channel. We present **Rosetta**, an interoperability layer that aligns independently-trained transceivers *without retraining them*. Rosetta standardizes a shared reference signal space and equips each model with two lightweight adapters that map its private symbols to and from the reference; all base encoders and decoders remain frozen, and only the adapters are trained, jointly, so that every transmitter reaches every receiver through the reference. This reduces the *N*-vendor bridging problem from *O(N²)* bespoke translators to *O(N)* adapters against a single standard. On three independently-trained deep-JSCC image transceivers, raw cross-model transmission achieves only **9.2 dB** PSNR (versus **22.6 dB** matched); adding Rosetta raises **every one of the nine sender–receiver pairs to 22.2 dB** — within 0.4 dB of matched — with **zero changes to any base model**. Rosetta is not a new codec; it is a standardizable alignment layer. We report results at proof-of-concept scale and discuss the calibration, governance, and generality limitations that remain.

## 1. Introduction

Classical communication separates source coding (compression) from channel coding (error protection). Deep JSCC instead trains a neural encoder and decoder end-to-end over a channel model, and degrades gracefully — without the digital "cliff" — particularly at low SNR and short blocklengths [1, 2]. Semantic and task-oriented variants extend this to transmitting meaning rather than bits [3, 4], and the approach is a pillar of native-AI 6G visions [5].

Despite strong simulation results, learned communication is not deployed. Surveys and standardization roadmaps repeatedly identify the same barrier: **semantic misalignment / knowledge mismatch across independently-built endpoints**, for which formal models are "largely undeveloped" [6, 5]. A learned encoder and decoder must share a code; two systems that never trained together do not, and interoperate no better than random.

**Contributions.**
1. We formalize cross-model interoperability for learned communication and cleanly characterize the failure: independently-trained transceivers that each achieve ~22.7 dB with their own receiver achieve only ~9 dB across models, even on a good channel.
2. We propose **Rosetta**, a base-frozen adapter scheme that aligns *N* transceivers to a shared reference space with *O(N)* small adapters, requiring no modification or retraining of the deployed base models.
3. We show Rosetta restores full any-to-any interoperability among three independently-trained transceivers (9.2 → 22.2 dB cross-model, all nine pairs), and analyze the remaining limitations honestly.

## 2. Related Work

**Deep JSCC.** Bourtsoulatze et al. [1] introduced deep-JSCC for wireless image transmission; later work added bandwidth/SNR adaptivity and feedback [2]. These establish the per-link performance we build on but assume a single jointly-trained pair.

**Semantic / task-oriented communication.** DeepSC and successors [3, 4] transmit task-relevant semantics; surveys rank interoperability and knowledge alignment among the most critical open problems [6], and 6G efforts call for unified, interoperable, modular semantic architectures [5].

**Agent communication.** As autonomous AI agents proliferate, a standardized agent communication protocol is argued to be urgently needed [7]; the alignment problem there is structurally identical.

**Position.** Prior work almost universally studies a single trained system. Rosetta is orthogonal: it takes *already-trained, frozen* systems and makes them interoperate. To our knowledge this "standardize a reference, adapt at the edges, retrain nothing" formulation and its clean multi-model demonstration are new to the deep-JSCC literature.

## 3. Problem Formulation

Consider *N* transceivers, each a trained encoder *Eᵢ: X → ℝᵏ* and decoder *Dᵢ: ℝᵏ → X*, from independent training runs. For input *x*, sender *i* emits power-normalized symbols *zᵢ = Eᵢ(x)* over a channel *H* (here AWGN), and receiver *j* reconstructs *x̂ = Dⱼ(H(zᵢ))*. When *i = j* the pair is matched and reconstruction is good; when *i ≠ j* the code spaces are unaligned and *x̂* is meaningless. The **interoperability objective** is to make *Dⱼ* usable on transmissions from *Eᵢ* for all *i, j*, **subject to the constraint that Eᵢ, Dᵢ cannot be modified** (deployed / proprietary / fixed).

## 4. Method

Rosetta introduces a shared reference space ℝᵏ (a published standard) and, for each model *i*, two adapters *Aᵢ* (private → reference) and *Bᵢ* (reference → private). Sender *i* transmits the standardized signal *rᵢ = PN(Aᵢ(Eᵢ(x)))*, where PN enforces unit average symbol power; receiver *j* reconstructs *x̂ᵢⱼ = Dⱼ(Bⱼ(H(rᵢ)))*. Adapters are residual, *Aᵢ(z) = z + fᵢ(z)*, so identity is the init prior. All *Eᵢ, Dᵢ* are frozen. Only the adapters are trained, jointly, minimizing mean distortion over all ordered pairs and a range of SNRs:

```
L = E_{x, SNR}  (1/N²) Σᵢ Σⱼ  || Dⱼ(Bⱼ(H(PN(Aᵢ(Eᵢ(x)))))) − x ||²
```

Because every model aligns to one reference rather than to every other model, the number of learned modules is *O(N)*, not *O(N²)* — the structure a standards body could publish once and have vendors implement independently.

## 5. Experimental Setup

- **Base models:** three deep-JSCC image transceivers ("Latent Radio") trained independently on Fashion-MNIST (28×28) with seeds {0,1,2}; convolutional, *k* = 128 real channel uses, power-normalized symbols, end-to-end over AWGN with per-batch SNR ∈ [−2, 12] dB. Each base model: 9.32 × 10⁵ params.
- **Adapters:** each *Aᵢ, Bᵢ* a 3-layer residual MLP (128→256→256→128, GELU). Full Rosetta layer for *N* = 3: **7.90 × 10⁵ trainable params (less than one base model)**, trained 2,500 steps (batch 128, Adam 8e-4), SNR ∼ U[−2,14] dB. Bases frozen throughout.
- **Metrics:** test-set PSNR (dB) for every pair at 10 dB SNR, and cross-model PSNR averaged across SNR ∈ {0,4,8,12} dB.

## 6. Results

**Sender→Receiver PSNR (dB) at 10 dB SNR** (rows = TX, cols = RX):

| | Before: RX0 | RX1 | RX2 | | After: RX0 | RX1 | RX2 |
|--|--|--|--|--|--|--|--|
| **TX0** | 22.6 | 9.2 | 9.1 | | 22.4 | 22.2 | 22.2 |
| **TX1** | 9.5 | 22.7 | 9.3 | | 22.2 | 22.4 | 22.2 |
| **TX2** | 9.0 | 9.1 | 22.6 | | 22.2 | 22.2 | 22.4 |

Raw cross-model transmission works only on the diagonal (~22.6 dB); all six off-diagonal pairs collapse to ~9 dB. With Rosetta, all nine pairs reach 22.2–22.4 dB — within 0.4 dB of matched — with no base weight changed. Averaged over off-diagonal pairs, cross-model PSNR rises **9.2 → 22.2 dB**.

**Mean cross-model PSNR (dB) vs. SNR:**

| SNR (dB) | 0 | 4 | 8 | 12 |
|--|--|--|--|--|
| Raw (no Rosetta) | 9.1 | 9.2 | 9.2 | 9.2 |
| Rosetta (cross) | 19.4 | 21.1 | 21.9 | 22.4 |
| Matched (via ref) | 19.6 | 21.2 | 22.1 | 22.6 |

Aligned cross-model PSNR tracks the matched-through-reference curve to within a few tenths of a dB from 0 to 12 dB.

## 7. Discussion and Limitations

- **Scale.** Small grayscale imagery, one codec family, *N* = 3. This is a proof that the mismatch is *fixable* with a tiny base-frozen layer, not a validated system.
- **Joint calibration.** Adapters are trained jointly on shared calibration data. In practice this implies a governed reference space and a calibration protocol; *who owns and maintains the reference* is a real, unresolved governance question, not a technical afterthought.
- **Scope of alignment.** Rosetta aligns the latent code family of a JSCC transceiver. It does not by itself reconcile arbitrary task semantics or heterogeneous architectures with different channel-use budgets; extending beyond a common *k* is future work.
- **Simulation.** Results are from an AWGN software channel; real deployment requires hardware-in-the-loop validation (fading, Doppler, timing recovery, front-end nonlinearity), which no simulation establishes.

## 8. Broader Impact

If cross-model interoperability can be provided by a small, standardizable, base-frozen layer, it removes the most-cited obstacle between learned communication and deployment, and offers a concrete shape — one published reference, *O(N)* vendor adapters — for standardization bodies (e.g., Next G Alliance, 3GPP) to consider. The same construction applies to interoperable AI-agent communication [7]. The primary risk is premature reliance on simulation-stage results; we therefore foreground the limitations above.

## 9. Conclusion

Learned communication does not fail because the coding is weak; at the link level it is strong. It fails to *ship* because independently-built endpoints cannot understand each other. Rosetta shows this interoperability can be restored — any-to-any, 9.2 → 22.2 dB — by standardizing a reference and adapting at the edges, with the base models untouched. The construction is simple, *O(N)*, and standard-shaped. We release all code and invite independent evaluation and hardware validation.

*Acknowledgment: the prototype and experiments were implemented with AI-assisted software tooling.*

## References

[1] E. Bourtsoulatze, D. Burth Kurka, D. Gündüz. "Deep Joint Source-Channel Coding for Wireless Image Transmission." *IEEE Trans. Cognitive Communications and Networking*, 2019.
[2] D. Burth Kurka, D. Gündüz. "DeepJSCC-f: Deep Joint Source-Channel Coding of Images with Feedback." *IEEE JSAIT*, 2020.
[3] H. Xie, Z. Qin, G. Y. Li, B.-H. Juang. "Deep Learning Enabled Semantic Communication Systems (DeepSC)." *IEEE Trans. Signal Processing*, 2021.
[4] D. Gündüz et al. "Beyond Transmitting Bits: Context, Semantics, and Task-Oriented Communications." *IEEE JSAC*, 2023.
[5] "Towards Native AI in 6G Standardization: The Roadmap of Semantic Communication." arXiv:2509.12758, 2025.
[6] "Ten Challenges in Semantic Communications." 2025.
[7] "LLM Agent Communication Protocol (LACP) Requires Urgent Standardization." arXiv:2510.13821, 2025.
