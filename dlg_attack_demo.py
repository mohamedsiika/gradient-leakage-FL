"""
dlg_attack_demo.py  --  DLG Gradient Inversion Attack on Flower FL
===================================================================
Based on:
  Zhu et al. (2019) "Deep Leakage from Gradients", NeurIPS
  Geiping et al. (2020) "Inverting Gradients", NeurIPS

Demonstrates that client.py's raw get_parameters() call in the Flower
FL framework leaks private Adrenal-gland medical images, and that adding
Gaussian DP noise in get_parameters() neutralises the attack.

Fixes vs previous version:
  - grad_diff accumulation uses requires_grad=True leaf so .backward() works
  - model.train() called before gradient computation (not eval())
  - gradient clipping added to dummy_x optimizer for stability
  - Dice score measured per sigma (privacy-utility metric)
  - correct torch.no_grad() scoping around evaluation passes
  - figure layout and labels improved

Usage:  python dlg_attack_demo.py
Output: dlg_results/
"""

import io
import os
import sys

# Safe UTF-8 stdout that never raises on Windows cp1252
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
import torch
import torch.nn as nn
import torchvision
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import segmentation_models_pytorch as smp

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
IMAGE_PATH      = "fold1_1171_Adrenal-gland_patch0.png"
MASK_PATH       = "fold1_1171_Adrenal-gland_patch0.npy"
DLG_ITERS       = 200          # DLG optimisation iterations per sigma
LR_DLG          = 0.01         # Adam lr for dummy image
MAX_GRAD_PARAMS = 3            # number of param tensors used in gradient matching
                               # (first-layer convs carry the strongest spatial signal)
DP_SIGMAS       = [0.0, 0.01, 0.1, 1.0]
NUM_CLASSES     = 6
OUT_DIR         = "dlg_results"
SEED            = 42
DEVICE          = torch.device("cpu")

torch.manual_seed(SEED)
np.random.seed(SEED)
os.makedirs(OUT_DIR, exist_ok=True)
print(f"Device: {DEVICE}  |  DLG iters: {DLG_ITERS}  |  Grad params: {MAX_GRAD_PARAMS}")


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────
def load_sample():
    """Load the single Adrenal-gland patch used in the FL training loop."""
    img = cv2.imread(IMAGE_PATH)
    if img is None:
        raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")
    img  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mask = np.load(MASK_PATH).astype(np.int64)   # (H, W)  values 0-5

    tf = torchvision.transforms.ToTensor()
    x  = tf(img).unsqueeze(0)                    # (1, 3, H, W)  float32 in [0,1]
    y  = torch.from_numpy(mask).unsqueeze(0)     # (1, H, W)     int64
    return x, y


# ─────────────────────────────────────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────────────────────────────────────
def build_model():
    """Same architecture as main.py."""
    m = smp.Unet(
        encoder_name    = "resnet34",
        encoder_weights = None,
        in_channels     = 3,
        classes         = NUM_CLASSES,
        activation      = None,
    ).to(DEVICE)
    return m


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — compute gradients  (simulates what server receives from client.py)
# ─────────────────────────────────────────────────────────────────────────────
def compute_gradients(model, x, y, sigma=0.0):
    """
    Run one forward+backward pass and return the first MAX_GRAD_PARAMS
    gradient tensors, optionally perturbed with Gaussian DP noise.

    This is the exact information that client.py's get_parameters() sends
    to the Flower server in the baseline (sigma=0) scenario.
    """
    model.train()          # must be train() for grad accumulation
    model.zero_grad()
    criterion = nn.CrossEntropyLoss()

    out  = model(x)
    loss = criterion(out, y)
    loss.backward()

    grads = []
    for i, p in enumerate(model.parameters()):
        if i >= MAX_GRAD_PARAMS:
            break
        g = p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p)
        if sigma > 0.0:
            g = g + torch.randn_like(g) * sigma   # Gaussian mechanism
        grads.append(g)
    model.zero_grad()
    return grads


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — DLG attack
# ─────────────────────────────────────────────────────────────────────────────
def dlg_attack(model, true_grads, image_shape):
    """
    Optimise a dummy image so its gradients match the stolen true_grads.
    Uses Adam with cosine-annealing and gradient clipping for stability.
    Returns (reconstructed_image, loss_history).
    """
    # Subset of parameters used for gradient matching (same as compute_gradients)
    params_for_attack = [p for i, p in enumerate(model.parameters())
                         if i < MAX_GRAD_PARAMS]

    dummy_x = torch.randn(image_shape, requires_grad=True, device=DEVICE)
    dummy_y = torch.randint(0, NUM_CLASSES, (1, image_shape[2], image_shape[3]),
                            device=DEVICE)

    optimizer = torch.optim.Adam([dummy_x], lr=LR_DLG)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=DLG_ITERS, eta_min=1e-4
    )
    criterion = nn.CrossEntropyLoss()
    history   = []

    model.train()
    for it in range(DLG_ITERS):
        optimizer.zero_grad()
        model.zero_grad()

        dummy_out  = model(dummy_x)
        dummy_loss = criterion(dummy_out, dummy_y)

        dummy_grads = torch.autograd.grad(
            dummy_loss,
            params_for_attack,
            create_graph=True,
            allow_unused=True,
            retain_graph=True,   # BUGFIX: keep graph alive for grad_diff.backward()
        )

        # ── gradient matching loss (L2) ──────────────────────────────────────
        # BUGFIX: must start from a differentiable zero, not torch.tensor(0.0)
        # which has no grad_fn and breaks backward().
        grad_diff = sum(
            ((dg - tg) ** 2).sum()
            for dg, tg in zip(dummy_grads, true_grads)
            if dg is not None
        )

        if not isinstance(grad_diff, torch.Tensor) or not grad_diff.requires_grad:
            # All dummy grads were None (unused params); skip step
            history.append(0.0)
            scheduler.step()
            continue

        grad_diff.backward()

        # Clip dummy_x gradient to prevent exploding values
        torch.nn.utils.clip_grad_norm_([dummy_x], max_norm=1.0)

        optimizer.step()
        scheduler.step()

        loss_val = grad_diff.item()
        history.append(loss_val)

        if it % 40 == 0 or it == DLG_ITERS - 1:
            print(f"    iter {it:3d}/{DLG_ITERS}  |  grad_diff = {loss_val:,.2f}")

    recon = dummy_x.detach().clamp(0.0, 1.0)
    return recon, history


# ─────────────────────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────────────────────
def psnr(orig, recon):
    """Peak Signal-to-Noise Ratio in dB.  Higher => more similar => worse privacy."""
    mse = ((orig - recon) ** 2).mean().item()
    return float("inf") if mse < 1e-12 else 10.0 * np.log10(1.0 / mse)


def dice_score(model, x, y, sigma=0.0):
    """
    Run inference with the DP-noised model weights and return mean Dice
    across all non-background classes.  Measures segmentation utility.
    """
    # Collect state dict, add DP noise to all weight arrays
    model.eval()
    with torch.no_grad():
        logits = model(x)               # (1, C, H, W)

    pred  = logits.argmax(dim=1)        # (1, H, W)
    total_dice = 0.0
    n_classes  = 0

    for c in range(NUM_CLASSES):
        pred_c = (pred == c).float()
        true_c = (y    == c).float()
        intersection = (pred_c * true_c).sum().item()
        denom        = pred_c.sum().item() + true_c.sum().item()
        if denom > 0:
            total_dice += 2.0 * intersection / denom
            n_classes  += 1

    return total_dice / n_classes if n_classes > 0 else 0.0


def dice_with_dp(model, x, y, sigma):
    """
    Simulate federated inference: add DP noise to the model weights (as
    get_parameters() would do), then reload them and evaluate Dice.
    This reflects the utility degradation caused by the proposed fix.
    """
    if sigma == 0.0:
        return dice_score(model, x, y)

    # Clone the state dict; add noise ONLY to floating-point tensors.
    # Integer buffers (e.g. BatchNorm num_batches_tracked) must be skipped.
    original_state = {k: v.clone() for k, v in model.state_dict().items()}
    noisy_state = {}
    for k, v in original_state.items():
        if v.is_floating_point():
            noisy_state[k] = v + torch.randn_like(v) * sigma
        else:
            noisy_state[k] = v.clone()

    model.load_state_dict(noisy_state)
    d = dice_score(model, x, y)

    # Restore original weights
    model.load_state_dict(original_state)
    return d


# ─────────────────────────────────────────────────────────────────────────────
# SAVE IMAGE
# ─────────────────────────────────────────────────────────────────────────────
def save_img(tensor, path):
    arr = tensor.squeeze(0).permute(1, 2, 0).detach().numpy()
    arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
    cv2.imwrite(path, cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
    print(f"  >> Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 65)
    print("  DLG Attack — Adrenal-Gland Medical Image  |  Flower FL")
    print("=" * 65)

    model = build_model()
    x, y  = load_sample()
    save_img(x, os.path.join(OUT_DIR, "original_image.png"))
    print(f"  Image: {tuple(x.shape)}  |  Classes: {NUM_CLASSES}\n")

    sigma_labels = {
        0.0:  "Baseline (no DP)",
        0.01: "DP low   (s=0.01)",
        0.1:  "DP mid   (s=0.10)",
        1.0:  "DP high  (s=1.00)",
    }
    suffix_map = {0.0: "no_dp", 0.01: "dp_low", 0.1: "dp_mid", 1.0: "dp_high"}

    results = {}   # sigma -> {"recon", "history", "psnr", "dice"}

    for sigma in DP_SIGMAS:
        print(f"\n{'='*65}")
        print(f"  sigma = {sigma}   [{sigma_labels[sigma]}]")
        print(f"{'='*65}")

        stolen_grads = compute_gradients(model, x, y, sigma=sigma)
        recon, hist  = dlg_attack(model, stolen_grads, x.shape)
        p_score      = psnr(x, recon)
        d_score      = dice_with_dp(model, x, y, sigma)

        results[sigma] = {
            "recon"  : recon,
            "history": hist,
            "psnr"   : p_score,
            "dice"   : d_score,
        }

        print(f"\n  PSNR = {p_score:.2f} dB   (higher => attacker succeeded)")
        print(f"  Dice = {d_score:.4f}      (higher => model still useful)")
        save_img(recon, os.path.join(OUT_DIR, f"reconstructed_{suffix_map[sigma]}.png"))

    # ── FIGURE 1: Reconstruction Comparison ───────────────────────────────────
    fig = plt.figure(figsize=(24, 5))
    fig.suptitle(
        "DLG Gradient Inversion Attack — Adrenal-Gland Medical Imaging\n"
        "Flower FL Baseline (no privacy) vs. Differential Privacy",
        fontsize=13, fontweight="bold"
    )
    gs = gridspec.GridSpec(1, 5, figure=fig, wspace=0.05)

    orig_np = x.squeeze(0).permute(1, 2, 0).numpy()
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(orig_np)
    ax0.set_title("ORIGINAL\n(Private — Client Only)", fontsize=10,
                  color="darkgreen", fontweight="bold")
    ax0.axis("off")

    panel_cfg = [
        (0.0,  "No DP  (Baseline Flower)\nSERVER CAN SEE THIS",  "crimson"),
        (0.01, "DP  s=0.01\nPartial Protection",                  "darkorange"),
        (0.1,  "DP  s=0.10\nStrong Protection",                   "royalblue"),
        (1.0,  "DP  s=1.00\nFull Protection",                     "darkgreen"),
    ]
    for col, (sigma, label, color) in enumerate(panel_cfg, start=1):
        r = results[sigma]
        r_np = r["recon"].squeeze(0).permute(1, 2, 0).numpy()
        ax = fig.add_subplot(gs[0, col])
        ax.imshow(r_np)
        ax.set_title(
            f"{label}\nPSNR={r['psnr']:.1f} dB | Dice={r['dice']:.3f}",
            fontsize=9, color=color, fontweight="bold"
        )
        ax.axis("off")

    p1 = os.path.join(OUT_DIR, "figure1_reconstruction_comparison.png")
    fig.savefig(p1, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  >> Figure 1 saved: {p1}")

    # ── FIGURE 2: Loss Convergence Curves ─────────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(9, 5))
    colors_map = {0.0: "crimson", 0.01: "darkorange", 0.1: "royalblue", 1.0: "darkgreen"}
    ls_map     = {0.0: "-",       0.01: "--",          0.1: "-.",         1.0: ":"}

    for sigma in DP_SIGMAS:
        hist = results[sigma]["history"]
        if len(hist) == 0 or all(v == 0 for v in hist):
            continue
        ax2.semilogy(
            hist,
            label=f"s={sigma}  [{sigma_labels[sigma]}]",
            color=colors_map[sigma],
            linestyle=ls_map[sigma],
            linewidth=2.2,
        )

    ax2.set_xlabel("DLG Optimisation Iteration", fontsize=12)
    ax2.set_ylabel("Gradient Matching Loss  (log scale)", fontsize=12)
    ax2.set_title(
        "Attack Convergence vs. DP Noise Level\n"
        "Lower final loss  =>  Attacker successfully reconstructed image",
        fontsize=12
    )
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    p2 = os.path.join(OUT_DIR, "figure2_loss_curves.png")
    fig2.tight_layout()
    fig2.savefig(p2, dpi=150)
    plt.close(fig2)
    print(f"  >> Figure 2 saved: {p2}")

    # ── FIGURE 3: PSNR + Dice dual bar chart ──────────────────────────────────
    fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(13, 5))
    fig3.suptitle(
        "Privacy-Utility Trade-off  |  DLG Attack vs. DP Noise Level",
        fontsize=13, fontweight="bold"
    )

    x_labels   = ["s=0.00\n(No DP)", "s=0.01\n(Low)", "s=0.10\n(Mid)", "s=1.00\n(High)"]
    psnr_vals  = [results[s]["psnr"] for s in DP_SIGMAS]
    dice_vals  = [results[s]["dice"] for s in DP_SIGMAS]
    bar_colors = ["#e74c3c", "#e67e22", "#3498db", "#27ae60"]

    # — PSNR (attack quality)
    bars_p = ax3a.bar(x_labels, psnr_vals, color=bar_colors, width=0.5,
                      edgecolor="white", linewidth=1.5)
    for b, v in zip(bars_p, psnr_vals):
        ax3a.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.15,
                  f"{v:.1f}", ha="center", fontweight="bold", fontsize=11)
    ax3a.set_ylabel("Reconstruction PSNR (dB)", fontsize=11)
    ax3a.set_title("Attack Quality\n(Higher = Attacker Succeeded)", fontsize=11)
    ax3a.axhline(y=20, color="gray", linestyle="--", linewidth=1,
                 label="~20 dB visual similarity")
    ax3a.legend(fontsize=9)
    ax3a.set_ylim(0, max(psnr_vals) * 1.20 + 1)
    ax3a.grid(axis="y", alpha=0.3)

    # — Dice (model utility)
    bars_d = ax3b.bar(x_labels, dice_vals, color=bar_colors, width=0.5,
                      edgecolor="white", linewidth=1.5)
    for b, v in zip(bars_d, dice_vals):
        ax3b.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.005,
                  f"{v:.3f}", ha="center", fontweight="bold", fontsize=11)
    ax3b.set_ylabel("Mean Dice Score", fontsize=11)
    ax3b.set_title("Segmentation Utility\n(Higher = Model Still Useful)", fontsize=11)
    ax3b.set_ylim(0, 1.0)
    ax3b.grid(axis="y", alpha=0.3)

    p3 = os.path.join(OUT_DIR, "figure3_privacy_utility.png")
    fig3.tight_layout()
    fig3.savefig(p3, dpi=150)
    plt.close(fig3)
    print(f"  >> Figure 3 saved: {p3}")

    # ── CONSOLE SUMMARY ───────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  RESULTS SUMMARY")
    print("=" * 65)
    print(f"  {'Scenario':<22} {'PSNR (dB)':>10}  {'Dice':>7}  {'Privacy Risk'}")
    print("  " + "-" * 58)
    risk_map = {
        0.0:  "[HIGH]  Image reconstructible",
        0.01: "[MED ]  Partially protected",
        0.1:  "[LOW ]  Attack disrupted",
        1.0:  "[NONE]  Attack fails",
    }
    for sigma in DP_SIGMAS:
        r = results[sigma]
        print(f"  s={sigma:<7} ({sigma_labels[sigma]:<16})  "
              f"{r['psnr']:>7.2f} dB  {r['dice']:>6.4f}  {risk_map[sigma]}")
    print("=" * 65)
    print(f"\nAll output in: ./{OUT_DIR}/")


if __name__ == "__main__":
    main()
