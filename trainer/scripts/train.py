"""Supervised training: distill Stockfish targets into the policy/value net."""

from __future__ import annotations

import argparse
import os
import time

import torch
from torch.utils.data import DataLoader, random_split

from chessai.model import build_model, count_params
from chessai.dataset import ShardDataset, soft_policy_loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", default="../data/shards/*.jsonl")
    ap.add_argument("--out", default="../models/base.pt")
    ap.add_argument("--channels", type=int, default=128)
    ap.add_argument("--blocks", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--val-frac", type=float, default=0.02)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--value-weight", type=float, default=1.0)
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"[train] device={device}")

    ds = ShardDataset(args.shards, limit=args.limit)
    n_val = max(1, int(len(ds) * args.val_frac))
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(
        ds, [n_train, n_val], generator=torch.Generator().manual_seed(0))
    print(f"[train] dataset: {len(ds):,}  train={n_train:,}  val={n_val:,}")

    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                          num_workers=args.workers, pin_memory=True,
                          persistent_workers=args.workers > 0, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                        num_workers=4, pin_memory=True)

    model = build_model(args.channels, args.blocks).to(device)
    if args.resume and os.path.exists(args.resume):
        sd = torch.load(args.resume, map_location=device)
        model.load_state_dict(sd["model"] if "model" in sd else sd)
        print(f"[train] resumed from {args.resume}")
    print(f"[train] params: {count_params(model):,}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    steps_per_epoch = len(train_dl)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.epochs * steps_per_epoch,
        pct_start=0.1)
    scaler = torch.amp.GradScaler(device.split(":")[0])
    vloss_fn = torch.nn.MSELoss()

    best_val = float("inf")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        run_p = run_v = 0.0
        for it, (planes, idxs, probs, value) in enumerate(train_dl):
            planes = planes.to(device, non_blocking=True)
            idxs = idxs.to(device, non_blocking=True)
            probs = probs.to(device, non_blocking=True)
            value = value.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device.split(":")[0]):
                logits, v = model(planes)
                p_loss = soft_policy_loss(logits, idxs, probs)
                v_loss = vloss_fn(v, value)
                loss = p_loss + args.value_weight * v_loss
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            run_p += p_loss.item()
            run_v += v_loss.item()
            if it % 100 == 0:
                print(f"  e{epoch} it{it}/{steps_per_epoch} "
                      f"p={run_p/(it+1):.3f} v={run_v/(it+1):.3f} "
                      f"lr={sched.get_last_lr()[0]:.2e}", flush=True)

        # validation
        model.eval()
        vp = vv = 0.0
        top1 = total = 0
        with torch.no_grad():
            for planes, idxs, probs, value in val_dl:
                planes = planes.to(device); idxs = idxs.to(device)
                probs = probs.to(device); value = value.to(device)
                with torch.amp.autocast(device.split(":")[0]):
                    logits, v = model(planes)
                    vp += soft_policy_loss(logits, idxs, probs).item()
                    vv += vloss_fn(v, value).item()
                pred = logits.argmax(1)
                best_tgt = idxs[:, 0]
                top1 += (pred == best_tgt).sum().item()
                total += planes.size(0)
        vp /= len(val_dl); vv /= len(val_dl)
        acc = top1 / total
        dt = time.time() - t0
        print(f"[train] epoch {epoch}: train p={run_p/steps_per_epoch:.3f} "
              f"v={run_v/steps_per_epoch:.3f} | val p={vp:.3f} v={vv:.3f} "
              f"top1={acc:.3f} | {dt:.0f}s", flush=True)

        val_total = vp + vv
        ckpt = {"model": model.state_dict(), "channels": args.channels,
                "blocks": args.blocks, "epoch": epoch, "val": val_total}
        torch.save(ckpt, args.out)
        if val_total < best_val:
            best_val = val_total
            torch.save(ckpt, args.out.replace(".pt", "_best.pt"))
    print(f"[train] done. best val={best_val:.3f}")


if __name__ == "__main__":
    main()
