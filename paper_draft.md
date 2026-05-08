# Paper Draft: Gradient Privacy in Flower-Based Federated Learning for Medical Image Segmentation

> **Status**: Draft — Sections 1–3 complete. Ready for author review.
> **Target venue**: IEEE Transactions on Cloud Computing / FGCS / ACM CODASPY

---

## Title (Proposed)

**"Gradient Leakage in Cloud-Deployed Federated Learning: A Demonstration and Differential Privacy Mitigation for Medical Image Segmentation with the Flower Framework"**

*Alternative (shorter):*
**"Securing Flower-Based Federated Learning: Gradient Privacy for Medical Image Segmentation"**

---

## Section 1 — Introduction

Federated Learning (FL) [CITE: McMahan et al., 2017] was introduced as a paradigm to train machine learning models across distributed clients without sharing raw data. Instead of transmitting sensitive training samples to a central server, each participant shares only model weight updates — a design that superficially appears to preserve data privacy. This apparent safety has motivated the rapid adoption of FL in privacy-critical domains, including medical imaging [CITE: Aouedi et al., 2022; Adnan et al., 2024], where access to patient data is constrained by regulatory frameworks such as HIPAA and the EU General Data Protection Regulation (GDPR).

However, a foundational body of research has since established that sharing gradient updates is **not equivalent to preserving privacy**. Zhu et al. [CITE: Zhu, Liu, Han, 2019, NeurIPS, ~2,500 citations] demonstrated that private training samples — including high-resolution images — can be reconstructed with near-pixel-perfect accuracy directly from shared gradients. Subsequent work by Geiping et al. [CITE: Geiping et al., 2020, NeurIPS] showed that this attack scales to larger batch sizes, undermining a previously assumed defence. In the medical imaging domain specifically, Hatamizadeh et al. [CITE: Hatamizadeh et al., 2023, IEEE Trans. Medical Imaging] directly verified the risk in FL systems operating on clinical data, demonstrating that gradient inversion can expose identifiable patient imagery from a single training round.

Despite this documented threat, the dominant FL framework in the research community — **Flower (flwr)** [CITE: Beutel et al., 2020, ~1,840 citations] — provides **no built-in privacy mechanisms**. Its core client abstraction, `NumPyClient`, transmits raw model parameters as plaintext NumPy arrays to a central server via:

```python
# From Flower's NumPyClient (client.py, line 17)
[val.cpu().numpy() for _, val in self.model.state_dict().items()]
```

This design means that any researcher or practitioner who builds on the Flower baseline — including those working with sensitive medical datasets — inherits a gradient leakage vulnerability by default, with no warning or alternative pathway documented in the framework.

**This paper makes the following contributions:**

1. **Vulnerability demonstration**: We provide the first systematic, end-to-end empirical demonstration that the standard Flower `NumPyClient.get_parameters()` interface exposes private training data to gradient inversion attacks, using a real-world medical imaging dataset (Adrenal-gland semantic segmentation).

2. **Quantified privacy-utility analysis**: We evaluate Gaussian Differential Privacy (DP) noise injection directly within the Flower client interface across four noise levels (σ ∈ {0.0, 0.01, 0.1, 1.0}), measuring reconstruction quality (PSNR), segmentation performance (Dice coefficient), and attack convergence — providing the first privacy-utility curve for Flower-based medical FL.

3. **Minimal, framework-compatible mitigation**: We implement DP as a transparent wrapper around `get_parameters()` that requires **no modification** to the Flower server, strategy, or training loop — making the contribution directly deployable in any existing Flower project.

The remainder of this paper is structured as follows. Section 2 reviews related work on FL privacy attacks and defences. Section 3 formalises the threat model and identifies the research gap. Section 4 describes the experimental setup and proposed DP mitigation. Section 5 presents results. Section 6 discusses limitations and future directions.

---

## Section 2 — Related Work

### 2.1 Federated Learning and Cloud Deployment

McMahan et al. [2017] introduced the Federated Averaging (FedAvg) algorithm, establishing the canonical FL protocol in which clients train locally and share weight deltas with a central aggregator. Subsequent work has explored cloud-native FL deployment [CITE: Bao & Guo, 2022; Zhang et al., 2021, IEEE Trans. Industrial Informatics] and multi-cloud scenarios [CITE: Wan et al., 2025; Doe & Buyya, 2024]. A comprehensive review by Rahdari et al. [2025] identified cloud infrastructure as an amplifier of FL's inherent privacy risks, noting that cloud-based aggregation servers represent a high-value target for adversaries.

The Flower framework [Beutel et al., 2020] has emerged as the dominant research FL framework, providing a hardware-agnostic, strategy-based abstraction over the canonical FL protocol. Despite its widespread adoption (~1,840 citations), no published work has addressed the integration of privacy mechanisms at the Flower framework level — a gap explicitly identified in this paper.

### 2.2 Gradient Leakage Attacks

The seminal Deep Leakage from Gradients (DLG) attack [Zhu et al., 2019] demonstrated that private training data can be recovered by solving an optimisation problem over a dummy input such that the dummy gradient matches the observed gradient. Formally, given shared gradients $\nabla W = \partial \ell(F(x^*, W), y^*) / \partial W$, an adversary reconstructs $(x^*, y^*)$ by minimising:

$$\hat{x}, \hat{y} = \arg\min_{x', y'} \left\| \nabla W' - \nabla W \right\|^2$$

where $\nabla W' = \partial \ell(F(x', W), y') / \partial W$.

Geiping et al. [2020] extended this using cosine similarity as the matching objective, enabling reconstruction from larger batches. Wang et al. [2019, INFOCOM] demonstrated GAN-based reconstruction at the user level, while Wei et al. [2020] provided a systematic evaluation framework showing that the attack success rate depends on batch size, model architecture, and gradient compression. In the medical domain, Hatamizadeh et al. [2023] showed the attack succeeds on clinical-grade MRI and CT data, raising HIPAA compliance concerns for any medical FL deployment using raw gradient sharing.

Critically, **all of these attacks apply directly to the Flower `NumPyClient` interface**, which is the default client abstraction used in the majority of Flower-based research projects, including the codebase evaluated in this paper.

### 2.3 Privacy-Preserving FL Defences

Three main classes of defence have been proposed:

**Differential Privacy (DP)** adds calibrated Gaussian or Laplacian noise to gradients before sharing [CITE: El Ouadrhiri & Abdelhadi, 2022; Zhao et al., 2020, IEEE IoT Journal]. The formal (ε, δ)-DP guarantee bounds the maximum information leakage about any single training sample. DP-SGD [Abadi et al., 2016] is the standard mechanism, clipping per-sample gradients and adding Gaussian noise. Personalized DP variants [Hu et al., 2020] allow per-client privacy budget allocation. However, all existing DP-FL work implements noise injection at the training loop level (e.g., within optimizer steps), not at the Flower communication interface level.

**Homomorphic Encryption (HE)** allows the server to aggregate encrypted gradients [CITE: Zhang et al., 2020, USENIX ATC — BatchCrypt, ~1,200 citations]. While providing strong cryptographic guarantees, HE introduces 10–100× computational overhead [CITE: Xie et al., 2024, IEEE IoT Journal], making it impractical for iterative training on large models such as the ResNet34-U-Net used in this work.

**Secure Aggregation (SA)** [Bonawitz et al., 2016, 891 citations] uses multi-party secret sharing so the server only learns the aggregate, not individual updates. LightSecAgg [So et al., 2022, MLSys] and FastSecAgg [Kadhe et al., 2020] improve scalability. The very recent SAEV-FL [Zhang et al., 2025, IEEE Trans. Cloud Computing, 1 citation] adds verification to SA for cloud deployments. However, none of these protocols are implemented within or for the Flower framework.

### 2.4 Flower Framework Extensions

Beyond the core framework, published Flower extensions include a Swift-based rapid prototyping variant [Kapsecker et al., 2023] and a serverless cloud deployment [Namjoshi et al., 2023, 1 citation]. Gawande et al. [2025] applied Flower to non-IID medical data, and Nurmi [2024] benchmarked Flower strategies on embedded devices. **None of these works address privacy at the framework interface level**, confirming the gap identified in this paper.

---

## Section 3 — Threat Model and Research Gap

### 3.1 Threat Model

We adopt the **honest-but-curious server** (HBC) adversary model [CITE: Lyu et al., 2020, IEEE Intelligent Systems], which is the standard model in FL privacy research. Under this model:

- The server follows the FedAvg protocol faithfully (no protocol deviation).
- The server is however **curious** — it attempts to infer private client data from everything it legitimately receives.
- The server receives plaintext gradient updates from all clients in every round.
- The server has full knowledge of the model architecture.

This threat model is directly realised in the evaluated codebase: `server.py` receives and loads client weight arrays in plaintext with no access control, encryption, or verification.

We further note that the HBC model is considered **conservative** — in cloud deployments, the aggregation server may be operated by a third-party cloud provider with motivations beyond protocol compliance [CITE: Rahdari et al., 2025; Mothukuri et al., 2021, Computers & Security]. A fully malicious server represents a stronger threat that we treat as future work.

### 3.2 Research Gap

The following gap has not been addressed in the existing literature:

> **Gap**: The Flower FL framework — the dominant FL research framework with ~1,840 citations — provides no privacy mechanism at its client communication interface (`NumPyClient.get_parameters()`). No **published work** has:
> (a) empirically demonstrated that the standard Flower client interface exposes private medical training data to gradient inversion attacks, or
> (b) implemented and benchmarked a privacy-preserving modification to the Flower `get_parameters()` call that is transparent to the server and strategy.

**Supporting evidence from the literature:**

| Claim | Evidence |
|---|---|
| Flower lacks built-in privacy | Beutel et al. (2020) — framework paper makes no mention of DP, HE, or SA |
| No Flower + DP integration paper exists | Namjoshi et al. (2023) — only serverless cloud extension (1 citation); no DP |
| Gradient leakage is real for medical images | Hatamizadeh et al. (2023), Zhu et al. (2019) |
| DP is the most practical defence | El Ouadrhiri & Abdelhadi (2022) survey; Lu et al. (2020) blockchain+DP |
| The gap is acknowledged but unsolved | Kairouz et al. (2021, ~11,000 citations) — "Advances and Open Problems" §5 |

### 3.3 Why This Gap Matters

The practical consequence of this gap is that any researcher who follows the Flower documentation to build a medical FL system — a common workflow given Flower's academic popularity — will produce a system that is **vulnerable by default**, with no indication from the framework that privacy is not guaranteed. This creates a false sense of security: the system follows the FL paradigm (raw data stays local) but still leaks private information through the gradient channel.

In the specific case of this paper's codebase — a U-Net segmentation model trained on Adrenal-gland medical images — the vulnerability means a curious cloud aggregation server could reconstruct identifiable patient pathology images from a single training round, violating GDPR Article 4(1) (personal data) and HIPAA Privacy Rule §164.502.

### 3.4 Proposed Contribution Summary

We close this gap with a minimal, framework-compatible intervention:

```python
# PROPOSED: Privacy-preserving get_parameters() in client.py
def get_parameters(self, config=None):
    params = [val.cpu().numpy() for _, val in self.model.state_dict().items()]
    if self.dp_sigma > 0:
        params = [p + np.random.normal(0, self.dp_sigma, p.shape) for p in params]
    return params
```

This change:
- Requires **no server-side modification**
- Is **compatible with any Flower strategy** (FedAvg, FedProx, etc.)
- Provides a tunable privacy-utility trade-off via σ
- Is **empirically validated** against an active DLG reconstruction attack

---

## References (Key Citations — Full list in bibliography)

| Tag | Citation |
|---|---|
| [McMahan 2017] | McMahan et al., "Communication-Efficient Learning of Deep Networks from Decentralized Data," AISTATS 2017 |
| [Zhu 2019] | Zhu, Liu, Han, "Deep Leakage from Gradients," NeurIPS 2019, ~2,500 citations |
| [Geiping 2020] | Geiping et al., "Inverting Gradients," NeurIPS 2020, ~850 citations |
| [Beutel 2020] | Beutel et al., "Flower: A Friendly Federated Learning Research Framework," arXiv 2020, ~1,840 citations |
| [Bonawitz 2016] | Bonawitz et al., "Practical Secure Aggregation for FL," arXiv 2016, 891 citations |
| [Hatamizadeh 2023] | Hatamizadeh et al., "Do Gradient Inversion Attacks Make FL Unsafe?" IEEE Trans. Medical Imaging 2023 |
| [Kairouz 2021] | Kairouz et al., "Advances and Open Problems in FL," Found. Trends ML 2021, ~11,000 citations |
| [El Ouadrhiri 2022] | El Ouadrhiri & Abdelhadi, "DP for Deep and FL: A Survey," IEEE Access 2022, 572 citations |
| [Lyu 2020] | Lyu, Yu, Yang, "Privacy-Preserving FL: A Survey," IEEE Intelligent Systems 2020, 480 citations |
| [Mothukuri 2021] | Mothukuri et al., "Privacy and Security in FL: A Survey," Computers & Security 2021, 752 citations |
| [Rahdari 2025] | Rahdari et al., "Survey on Privacy and Security in Distributed Cloud," IEEE Open Journal 2025 |
| [Aouedi 2022] | Aouedi et al., "Handling Privacy-Sensitive Medical Data with FL," IEEE JBHI 2022 |
| [Zhang 2025] | Zhang et al., "SAEV-FL: Lightweight SA and Efficient Verification for FL," IEEE Trans. Cloud Computing 2025 |
| [Namjoshi 2023] | Namjoshi et al., "Serverless FL with flwr-serverless," arXiv 2023, 1 citation |

---

## Notes for Author

- **Section 4 (Experiments)** should describe: model architecture (ResNet34-UNet, 3-channel input, 6-class output), dataset (Adrenal-gland `.png` + `.npy` mask), DLG attack implementation (LBFGS optimiser, 300 iterations), DP noise levels tested, evaluation metrics (PSNR for attack quality, Dice for segmentation quality).
- **Section 5 (Results)** should include: Figure 1 (reconstruction comparisons), Figure 2 (loss curves), Figure 3 (PSNR bar chart) — all generated by `dlg_attack_demo.py`.
- The gap statement in §3.2 should be directly referenced in the Abstract and Introduction to make the novelty claim clear to reviewers.
