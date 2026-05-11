import argparse
import math
import os

parser = argparse.ArgumentParser()
parser.add_argument("--gpu",      default="0")
parser.add_argument("--config",   default="config_mae.yaml")
parser.add_argument("--load_run", default=None, type=int)
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from file_management import get_version_folder, save_config, load_config
from model import MAE


import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../ECCV"))
from CityScapes.datasets import CityscapesDataset as MyDataset

# ---------------------------------------------------------------------------
# Config & versioning
# ---------------------------------------------------------------------------

with open(args.config) as f:
    cfg = yaml.safe_load(f)

version_folder = get_version_folder("runs", cfg["run_name"], load_run=args.load_run)

if args.load_run is not None:
    cfg = load_config(version_folder)
else:
    save_config(cfg, version_folder)

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ---------------------------------------------------------------------------
# Dataset & DataLoader
# ---------------------------------------------------------------------------

dcfg = cfg["data"]
dataset = MyDataset(
    data_dir=dcfg["data_dir"],
    get_labels=False,
)
loader = DataLoader(
    dataset,
    batch_size=dcfg["batch_size"],
    shuffle=True,
    num_workers=dcfg["num_workers"],
    pin_memory=True,
    drop_last=True,
)

# Fixed batch used for progress plots throughout training
plot_batch = next(iter(loader))[0][:4].to(device)

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

mcfg = cfg["model"]
model = MAE(
    img_size=dcfg["img_size"],
    patch_size=mcfg["patch_size"],
    in_chans=dcfg["in_chans"],
    encoder_embed_dim=mcfg["encoder_embed_dim"],
    encoder_depth=mcfg["encoder_depth"],
    encoder_num_heads=mcfg["encoder_num_heads"],
    decoder_embed_dim=mcfg["decoder_embed_dim"],
    decoder_depth=mcfg["decoder_depth"],
    decoder_num_heads=mcfg["decoder_num_heads"],
    mlp_ratio=mcfg["mlp_ratio"],
    mask_ratio=mcfg["mask_ratio"],
    norm_pix_loss=mcfg["norm_pix_loss"],
).to(device)

# ---------------------------------------------------------------------------
# Optimiser
# ---------------------------------------------------------------------------

tcfg = cfg["training"]
optimizer = torch.optim.AdamW(
    model.parameters(), lr=tcfg["lr"], weight_decay=tcfg["weight_decay"]
)

# ---------------------------------------------------------------------------
# Resume
# ---------------------------------------------------------------------------

start_epoch = 0
if args.load_run is not None:
    with open(os.path.join(version_folder, "meta.yaml")) as f:
        meta = yaml.safe_load(f)
    last_epoch = meta["epoch"]
    ckpt_path = os.path.join(version_folder, f"mae_e{last_epoch:03d}.pt")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    start_epoch = last_epoch + 1
    print(f"Resumed from epoch {last_epoch} ({ckpt_path})")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

writer = SummaryWriter(log_dir=os.path.join(version_folder, "tb"))
plots_dir = os.path.join(version_folder, "plots")
os.makedirs(plots_dir, exist_ok=True)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _adjust_lr(optimizer, epoch):
    warmup = tcfg["warmup_epochs"]
    base_lr, min_lr = tcfg["lr"], tcfg.get("min_lr", 0.0)
    epochs = tcfg["epochs"]
    if epoch < warmup:
        lr = base_lr * (epoch + 1) / warmup
    else:
        progress = (epoch - warmup) / max(1, epochs - warmup)
        lr = min_lr + (base_lr - min_lr) * 0.5 * (1.0 + math.cos(math.pi * progress))
    for pg in optimizer.param_groups:
        pg["lr"] = lr
    return lr


def _save_plot(model, imgs, save_path):
    model.eval()
    with torch.no_grad():
        original, masked_img, recon_img = model.reconstruct(imgs)

    C = original.shape[1]
    B = original.shape[0]

    def _to_np(t):
        t = t.cpu().mul(0.5).add(0.5).clamp(0, 1)
        if C == 1:
            return t.squeeze(1).numpy()   # (B, H, W)
        return t.permute(0, 2, 3, 1).numpy()  # (B, H, W, 3)

    orig_np   = _to_np(original)
    masked_np = _to_np(masked_img)
    recon_np  = _to_np(recon_img)

    fig, axes = plt.subplots(B, 3, figsize=(9, 3 * B))
    if B == 1:
        axes = axes[None]

    for i in range(B):
        for j, (arr, title) in enumerate([(orig_np[i], "Original"),
                                           (masked_np[i], "Masked"),
                                           (recon_np[i], "Reconstructed")]):
            cmap = "gray" if C == 1 else None
            axes[i, j].imshow(arr, cmap=cmap)
            axes[i, j].set_title(title, fontsize=9)
            axes[i, j].axis("off")

    fig.tight_layout()
    fig.savefig(save_path, dpi=80, bbox_inches="tight")
    plt.close(fig)
    model.train()


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

global_step = start_epoch * len(loader)
plot_every = tcfg["plot_every"]
save_every = tcfg["save_every"]

for epoch in range(start_epoch, tcfg["epochs"]):
    model.train()
    lr = _adjust_lr(optimizer, epoch)

    totals = {"loss": 0.0}
    n_batches = 0

    for imgs, _ in loader:
        imgs = imgs.to(device)

        loss, _pred, _mask = model(imgs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        totals["loss"] += loss.item()
        n_batches += 1
        global_step += 1

        if plot_every > 0 and global_step % plot_every == 0:
            ep_dir = os.path.join(plots_dir, f"e{epoch:03d}")
            os.makedirs(ep_dir, exist_ok=True)
            _save_plot(model, plot_batch,
                       save_path=os.path.join(ep_dir, f"s{global_step:06d}.png"))

    # -- per-epoch logging
    avg_loss = totals["loss"] / n_batches
    writer.add_scalar("train/loss", avg_loss, epoch)
    writer.add_scalar("train/lr",   lr,       epoch)
    print(f"[epoch {epoch:03d}]  loss={avg_loss:.4f}  lr={lr:.2e}")

    # -- checkpoint
    if (epoch + 1) % save_every == 0:
        ckpt_path = os.path.join(version_folder, f"mae_e{epoch:03d}.pt")
        torch.save(model.state_dict(), ckpt_path)
        with open(os.path.join(version_folder, "meta.yaml"), "w") as f:
            yaml.dump({"epoch": epoch}, f)

writer.close()
print("Training complete.")
