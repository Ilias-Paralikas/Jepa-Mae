import argparse
import math
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from Data.CityScapes.datasets import CityscapesDataset

from file_management import get_version_folder, load_config, save_config
from MAE.patchifier import Patchifier
from MAE.patch_embed import PatchEmbed
from MAE.mae import MAE
from MAE.loss_functions.mse_loss import MSELoss


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def denorm(x, tanh_space=True):
    if tanh_space:
        return (x * 0.5 + 0.5).clamp(0, 1)
    return x.clamp(0, 1)


def build_scheduler(optimizer, cfg, steps_per_epoch):
    tcfg        = cfg['training']
    total_steps  = tcfg['epochs'] * steps_per_epoch
    warmup_steps = tcfg['warmup_epochs'] * steps_per_epoch
    base_lr      = tcfg['lr']
    min_lr       = tcfg['min_lr']

    def lr_lambda(step):
        if warmup_steps > 0 and step < warmup_steps:
            return step / warmup_steps
        if min_lr == base_lr:
            return 1.0
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine   = 0.5 * (1 + math.cos(math.pi * progress))
        return min_lr / base_lr + (1.0 - min_lr / base_lr) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def visualize(images, patchifier, patch_embed, mae, step, epoch, vis_dir, device, tanh_space=True):
    patch_embed.eval()
    mae.eval()

    n = min(4, images.shape[0])
    imgs = images[:n].to(device)

    with torch.no_grad():
        raw_patches = patchifier.patchify(imgs)
        patch_embs  = patch_embed(raw_patches)
        rgb_pred, indices = mae(patch_embs)

        # Visible-only: zero masked positions in raw patch space
        visible_patches = torch.zeros_like(raw_patches)
        visible_patches[:, indices] = raw_patches[:, indices]
        visible_imgs = patchifier.unpatchify(visible_patches)

        recon_imgs = patchifier.unpatchify(rgb_pred)

        composite_patches = rgb_pred.clone()
        composite_patches[:, indices] = raw_patches[:, indices]
        composite_imgs = patchifier.unpatchify(composite_patches)

    fig, axes = plt.subplots(n, 4, figsize=(12, 3 * n))
    if n == 1:
        axes = axes[None]

    axes[0, 0].set_title('Original',    fontsize=10)
    axes[0, 1].set_title('Visible',     fontsize=10)
    axes[0, 2].set_title('Reconstructed', fontsize=10)
    axes[0, 3].set_title('Composite',   fontsize=10)

    for i in range(n):
        for j, t in enumerate([imgs[i], visible_imgs[i], recon_imgs[i], composite_imgs[i]]):
            axes[i, j].imshow(denorm(t, tanh_space).permute(1, 2, 0).cpu().float())
            axes[i, j].axis('off')

    epoch_dir = os.path.join(vis_dir, f'epoch_{epoch:04d}')
    os.makedirs(epoch_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(epoch_dir, f'step_{step:07d}.png'), dpi=100)
    plt.close()

    patch_embed.train()
    mae.train()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu',      default='0')
    parser.add_argument('--config',   default='config_mae.yaml')
    parser.add_argument('--load_run', type=int, default=None,
                        help='Version number to resume from')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    cfg = load_config('.', args.config)

    tcfg = cfg['training']
    dcfg = cfg['data']
    mcfg = cfg['model']

    save_dir = get_version_folder(tcfg['base_dir'], cfg['run_name'], args.load_run)
    vis_dir  = os.path.join(save_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)

    if args.load_run is None:
        save_config(cfg, save_dir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # --- Data ---
    dataset = CityscapesDataset(get_labels=False)
    loader  = DataLoader(
        dataset,
        batch_size=dcfg['batch_size'],
        shuffle=True,
        num_workers=dcfg['num_workers'],
        pin_memory=True,
        drop_last=True,
    )

    # --- Models ---
    img_size    = dcfg['img_size']
    patch_size  = mcfg['patch_size']
    in_chans    = dcfg['in_chans']
    patch_dim   = patch_size * patch_size * in_chans
    num_patches = (img_size // patch_size) ** 2

    patchifier  = Patchifier(patch_size, img_size, in_chans)

    encfg = mcfg['encoder']
    deccfg = mcfg['decoder']

    patch_embed = PatchEmbed(
        patch_dim=patch_dim,
        embed_dim=encfg['encoder_embed_dim'],
        hidden_dims=mcfg['patch_embed']['hidden_dims'],
    ).to(device)

    mae = MAE(
        num_patches=num_patches,
        patch_dim=patch_dim,
        encoder_embed_dim=encfg['encoder_embed_dim'],
        encoder_depth=encfg['encoder_depth'],
        encoder_num_heads=encfg['encoder_num_heads'],
        decoder_embed_dim=deccfg['decoder_embed_dim'],
        decoder_depth=deccfg['decoder_depth'],
        decoder_num_heads=deccfg['decoder_num_heads'],
        mask_ratio=mcfg['mask_ratio'],
        mlp_ratio=mcfg['mlp_ratio'],
    ).to(device)

    criterion = MSELoss()

    # --- Optimizer & scheduler ---
    params    = list(patch_embed.parameters()) + list(mae.parameters())
    optimizer = torch.optim.AdamW(params, lr=tcfg['lr'], weight_decay=tcfg['weight_decay'])
    scheduler = build_scheduler(optimizer, cfg, steps_per_epoch=len(loader))

    start_epoch = 0
    global_step = 0

    if args.load_run is not None:
        patch_embed.load_state_dict(torch.load(os.path.join(save_dir, 'patch_embed.pth'), weights_only=True))
        mae.load_state_dict(torch.load(os.path.join(save_dir, 'mae.pth'), weights_only=True))
        optimizer.load_state_dict(torch.load(os.path.join(save_dir, 'optimizer.pth'), weights_only=True))
        meta        = torch.load(os.path.join(save_dir, 'meta.pth'), weights_only=True)
        start_epoch = meta['epoch'] + 1
        global_step = meta['global_step']
        print(f'Resumed from epoch {start_epoch}, step {global_step}')

    recon_weight = cfg['loss']['reconstruction_weight']

    # --- Training loop ---
    for epoch in range(start_epoch, tcfg['epochs']):
        pbar = tqdm(loader, desc=f'Epoch {epoch + 1}/{tcfg["epochs"]}', leave=True)

        for images, _ in pbar:
            images = images.to(device)

            raw_patches = patchifier.patchify(images)
            patch_embs  = patch_embed(raw_patches)
            rgb_pred, indices = mae(patch_embs)

          
            target = raw_patches

            loss = criterion(rgb_pred, target, indices) * recon_weight

            optimizer.zero_grad()
            loss.backward()
            if tcfg['grad_clip'] > 0:
                nn.utils.clip_grad_norm_(params, tcfg['grad_clip'])
            optimizer.step()
            scheduler.step()

            global_step += 1
            pbar.set_postfix({'loss': f'{loss.item():.4f}',
                              'lr':   f'{scheduler.get_last_lr()[0]:.2e}'})

            if global_step % tcfg['plot_every'] == 0:
                visualize(images, patchifier, patch_embed, mae,
                          global_step, epoch, vis_dir, device,
                          tanh_space=dcfg['tanh_space'])

        if (epoch + 1) % tcfg['save_every'] == 0:
            torch.save(patch_embed.state_dict(), os.path.join(save_dir, 'patch_embed.pth'))
            torch.save(mae.state_dict(),         os.path.join(save_dir, 'mae.pth'))
            torch.save(optimizer.state_dict(),   os.path.join(save_dir, 'optimizer.pth'))
            torch.save({'epoch': epoch, 'global_step': global_step},
                       os.path.join(save_dir, 'meta.pth'))


if __name__ == '__main__':
    main()
