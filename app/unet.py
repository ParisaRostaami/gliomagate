"""2D U-Net for Cheng T1c tumor masks."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .segment import postprocess


class DoubleConv(nn.Module):
    def __init__(self, inn: int, out: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(inn, out, 3, padding=1, bias=False),
            nn.BatchNorm2d(out),
            nn.ReLU(inplace=True),
            nn.Conv2d(out, out, 3, padding=1, bias=False),
            nn.BatchNorm2d(out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNet(nn.Module):
    def __init__(self, base: int = 32) -> None:
        super().__init__()
        chs = (base, base * 2, base * 4, base * 8)
        self.down1 = DoubleConv(1, chs[0])
        self.down2 = DoubleConv(chs[0], chs[1])
        self.down3 = DoubleConv(chs[1], chs[2])
        self.bot = DoubleConv(chs[2], chs[3])
        self.pool = nn.MaxPool2d(2)
        self.up3 = nn.ConvTranspose2d(chs[3], chs[2], 2, stride=2)
        self.dec3 = DoubleConv(chs[3], chs[2])
        self.up2 = nn.ConvTranspose2d(chs[2], chs[1], 2, stride=2)
        self.dec2 = DoubleConv(chs[2], chs[1])
        self.up1 = nn.ConvTranspose2d(chs[1], chs[0], 2, stride=2)
        self.dec1 = DoubleConv(chs[1], chs[0])
        self.head = nn.Conv2d(chs[0], 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d1 = self.down1(x)
        d2 = self.down2(self.pool(d1))
        d3 = self.down3(self.pool(d2))
        b = self.bot(self.pool(d3))
        u3 = self.dec3(torch.cat([self.up3(b), d3], dim=1))
        u2 = self.dec2(torch.cat([self.up2(u3), d2], dim=1))
        u1 = self.dec1(torch.cat([self.up1(u2), d1], dim=1))
        return self.head(u1)


def _dice_bce(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, target)
    p = torch.sigmoid(logits)
    inter = (p * target).sum(dim=(1, 2, 3))
    den = p.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = 1.0 - (2.0 * inter + 1.0) / (den + 1.0)
    return bce + dice.mean()


class UNetSegmenter:
    def __init__(self, net: UNet, device: torch.device) -> None:
        self.net = net
        self.device = device

    def predict(self, image: np.ndarray) -> np.ndarray:
        self.net.eval()
        x = torch.from_numpy(image.astype(np.float32)[None, None]).to(self.device)
        with torch.no_grad():
            logit = self.net(x)
            prob = torch.sigmoid(logit)[0, 0].cpu().numpy()
        raw = (prob >= 0.5).astype(np.uint8)
        return postprocess(raw)


def n_params(net: nn.Module) -> int:
    return sum(p.numel() for p in net.parameters() if p.requires_grad)


def _augment(img: np.ndarray, mask: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    if rng.random() < 0.5:
        img = np.fliplr(img).copy()
        mask = np.fliplr(mask).copy()
    if rng.random() < 0.5:
        img = np.flipud(img).copy()
        mask = np.flipud(mask).copy()
    if rng.random() < 0.5:
        k = int(rng.integers(1, 4))
        img = np.rot90(img, k).copy()
        mask = np.rot90(mask, k).copy()
    scale = float(rng.uniform(0.85, 1.15))
    shift = float(rng.uniform(-0.05, 0.05))
    img = np.clip(img * scale + shift, 0.0, 1.0)
    return img, mask


def train_unet(
    images: list[np.ndarray],
    masks: list[np.ndarray],
    seed: int = 42,
    epochs: int = 25,
    batch_size: int = 8,
    base: int = 32,
    lr: float = 1e-3,
) -> tuple[UNetSegmenter, list[float]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    net = UNet(base=base).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    print(f"  U-Net base={base} params={n_params(net):,} device={device} n={len(images)} epochs={epochs}")
    losses: list[float] = []
    n = len(images)
    for epoch in range(1, epochs + 1):
        order = rng.permutation(n)
        epoch_loss = 0.0
        steps = 0
        net.train()
        for start in range(0, n, batch_size):
            idx = order[start : start + batch_size]
            xs, ys = [], []
            for i in idx:
                img, lab = _augment(images[i], (masks[i] > 0).astype(np.float32), rng)
                xs.append(img)
                ys.append(lab)
            x = torch.from_numpy(np.stack(xs)[:, None].astype(np.float32)).to(device)
            y = torch.from_numpy(np.stack(ys)[:, None].astype(np.float32)).to(device)
            opt.zero_grad()
            loss = _dice_bce(net(x), y)
            loss.backward()
            opt.step()
            epoch_loss += float(loss.item())
            steps += 1
        mean_loss = epoch_loss / max(steps, 1)
        losses.append(mean_loss)
        print(f"  epoch {epoch:02d}/{epochs}  loss={mean_loss:.4f}")
    return UNetSegmenter(net, device), losses


def save_unet(seg: UNetSegmenter, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": seg.net.state_dict(), "base": seg.net.down1.net[0].out_channels}, path)


def load_unet(path: Path) -> UNetSegmenter:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    blob = torch.load(path, map_location=device, weights_only=False)
    base = int(blob.get("base", 32))
    net = UNet(base=base).to(device)
    net.load_state_dict(blob["state_dict"])
    net.eval()
    return UNetSegmenter(net, device)
