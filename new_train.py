#!/usr/bin/env python3
"""
Clean tiny multimodal pretraining.

This is a cleaner replacement for the first training run:
- TinyStories for coherent small-model language.
- Cleaned Cosmopedia stories for simple factual/explanatory prose.
- COCO + Flickr30k captions for high-quality visual grounding.
- No CC3M by default.

It is standalone: send only this file to RunPod.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from tqdm.auto import tqdm


def import_datasets():
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("Install dependencies first: pip install torch datasets pillow tqdm tokenizers") from exc
    return load_dataset


def import_tokenizers():
    try:
        from tokenizers import ByteLevelBPETokenizer
    except ImportError as exc:
        raise SystemExit("Install dependencies first: pip install tokenizers") from exc
    return ByteLevelBPETokenizer


SPECIAL_TOKENS = [
    "<|pad|>",
    "<|bos|>",
    "<|eos|>",
    "<|image|>",
    "<|/image|>",
    "<|user|>",
    "<|assistant|>",
]


@dataclass
class ModelConfig:
    vocab_size: int = 8192
    d_model: int = 304
    n_layers: int = 8
    n_heads: int = 8
    ffn_mult: int = 4
    max_seq_len: int = 512
    image_size: int = 128
    patch_size: int = 16
    dropout: float = 0.0


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.dropout = cfg.dropout

    def forward(self, x: torch.Tensor, pad_mask: Optional[torch.Tensor]) -> torch.Tensor:
        bsz, seqlen, dim = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(bsz, seqlen, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seqlen, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seqlen, self.n_heads, self.head_dim).transpose(1, 2)

        attn_mask = None
        if pad_mask is not None:
            causal = torch.ones(seqlen, seqlen, device=x.device, dtype=torch.bool).tril()
            attn_mask = causal[None, None, :, :] & pad_mask[:, None, None, :]

        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=attn_mask is None,
        )
        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, dim)
        return self.proj(y)


class SwiGLU(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        hidden = cfg.ffn_mult * cfg.d_model
        self.w1 = nn.Linear(cfg.d_model, hidden, bias=False)
        self.w2 = nn.Linear(hidden, cfg.d_model, bias=False)
        self.w3 = nn.Linear(cfg.d_model, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.norm2 = RMSNorm(cfg.d_model)
        self.ffn = SwiGLU(cfg)

    def forward(self, x: torch.Tensor, pad_mask: Optional[torch.Tensor]) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), pad_mask)
        x = x + self.ffn(self.norm2(x))
        return x


class TinyNativeMultimodalGPT(nn.Module):
    def __init__(self, cfg: ModelConfig, pad_id: int):
        super().__init__()
        self.cfg = cfg
        self.pad_id = pad_id
        patch_dim = cfg.patch_size * cfg.patch_size * 3
        n_patches = (cfg.image_size // cfg.patch_size) ** 2

        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.patch_proj = nn.Linear(patch_dim, cfg.d_model, bias=False)
        self.image_pos = nn.Parameter(torch.zeros(1, n_patches, cfg.d_model))
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def image_to_patches(self, images: torch.Tensor) -> torch.Tensor:
        p = self.cfg.patch_size
        patches = F.unfold(images, kernel_size=p, stride=p).transpose(1, 2)
        return self.patch_proj(patches) + self.image_pos

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        images: Optional[torch.Tensor],
        image_positions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz, seqlen = input_ids.shape
        x = self.tok_emb(input_ids)
        pad_mask = input_ids.ne(self.pad_id)

        if images is not None and images.numel() > 0:
            patch_tokens = self.image_to_patches(images)
            for i in range(bsz):
                pos = int(image_positions[i].item())
                if pos >= 0:
                    n = patch_tokens.shape[1]
                    x[i, pos : pos + n] = patch_tokens[i]
                    pad_mask[i, pos : pos + n] = True

        positions = torch.arange(seqlen, device=input_ids.device)
        x = x + self.pos_emb(positions)[None, :, :]

        for block in self.blocks:
            x = block(x, pad_mask)
        logits = self.lm_head(self.norm(x))

        loss = F.cross_entropy(
            logits[:, :-1].contiguous().view(-1, logits.size(-1)),
            labels[:, 1:].contiguous().view(-1),
            ignore_index=-100,
        )
        return logits, loss


def text_stream_for_tokenizer(args: argparse.Namespace) -> Iterator[str]:
    load_dataset = import_datasets()
    count = 0

    if getattr(args, "use_tinystories", True):
        ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
        for row in ds:
            yield row.get("text", "")
            count += 1
            if count >= args.tokenizer_samples:
                return

    if getattr(args, "use_cosmopedia", True):
        ds = load_dataset("HuggingFaceTB/cosmopedia", "stories", split="train", streaming=True)
        for row in ds:
            yield row.get("text", "")
            count += 1
            if count >= args.tokenizer_samples:
                return


def train_tokenizer_if_needed(args: argparse.Namespace) -> Any:
    ByteLevelBPETokenizer = import_tokenizers()
    tok_dir = Path(args.tokenizer_dir)
    vocab_file = tok_dir / "vocab.json"
    merges_file = tok_dir / "merges.txt"
    tok_dir.mkdir(parents=True, exist_ok=True)

    if args.train_tokenizer or not (vocab_file.exists() and merges_file.exists()):
        print(f"Training tokenizer into {tok_dir} ...")
        tokenizer = ByteLevelBPETokenizer()
        tokenizer.train_from_iterator(
            text_stream_for_tokenizer(args),
            vocab_size=args.vocab_size,
            min_frequency=2,
            special_tokens=SPECIAL_TOKENS,
            length=args.tokenizer_samples,
        )
        tokenizer.save_model(str(tok_dir))

    tokenizer = ByteLevelBPETokenizer(str(vocab_file), str(merges_file))
    tokenizer.add_special_tokens(SPECIAL_TOKENS)
    return tokenizer


def image_transform(img: Image.Image, image_size: int, train: bool = True) -> torch.Tensor:
    img = img.convert("RGB")
    if train:
        scale = random.uniform(1.0, 1.12)
        big = int(image_size * scale)
        img = img.resize((big, big), Image.BICUBIC)
        left = random.randint(0, big - image_size)
        top = random.randint(0, big - image_size)
        img = img.crop((left, top, left + image_size, top + image_size))
    else:
        img = img.resize((image_size, image_size), Image.BICUBIC)
    data = torch.frombuffer(img.tobytes(), dtype=torch.uint8)
    data = data.view(image_size, image_size, 3).permute(2, 0, 1).float() / 255.0
    return (data - 0.5) / 0.5


def encode_text(tokenizer: Any, text: str, add_bos: bool = True, add_eos: bool = True) -> List[int]:
    ids = tokenizer.encode(text).ids
    if add_bos:
        ids = [tokenizer.token_to_id("<|bos|>")] + ids
    if add_eos:
        ids = ids + [tokenizer.token_to_id("<|eos|>")]
    return ids


def encode_sample(
    tokenizer: Any,
    text: str,
    image: Optional[Image.Image],
    cfg: ModelConfig,
    train: bool = True,
) -> Dict[str, Any]:
    pad_id = tokenizer.token_to_id("<|pad|>")
    bos_id = tokenizer.token_to_id("<|bos|>")
    eos_id = tokenizer.token_to_id("<|eos|>")
    image_id = tokenizer.token_to_id("<|image|>")
    image_end_id = tokenizer.token_to_id("<|/image|>")
    n_patches = (cfg.image_size // cfg.patch_size) ** 2

    if image is None:
        ids = encode_text(tokenizer, text, add_bos=True, add_eos=train)
        ids = ids[: cfg.max_seq_len]
        labels = ids.copy()
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "image": None,
            "image_position": -1,
        }

    text_ids = tokenizer.encode(text).ids
    if train:
        text_ids = text_ids + [eos_id]

    prefix = [bos_id, image_id]
    suffix = [image_end_id]
    available = cfg.max_seq_len - len(prefix) - n_patches - len(suffix)
    text_ids = text_ids[: max(0, available)]

    ids = prefix + [image_id] * n_patches + suffix + text_ids
    labels = [-100] * (len(prefix) + n_patches + len(suffix)) + text_ids

    return {
        "input_ids": torch.tensor(ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "image": image_transform(image, cfg.image_size, train=train),
        "image_position": len(prefix),
    }


def collate_samples(samples: List[Dict[str, Any]], pad_id: int, image_size: int) -> Dict[str, torch.Tensor]:
    max_len = max(s["input_ids"].numel() for s in samples)
    bsz = len(samples)
    input_ids = torch.full((bsz, max_len), pad_id, dtype=torch.long)
    labels = torch.full((bsz, max_len), -100, dtype=torch.long)
    image_positions = torch.full((bsz,), -1, dtype=torch.long)
    images = []

    for i, sample in enumerate(samples):
        n = sample["input_ids"].numel()
        input_ids[i, :n] = sample["input_ids"]
        labels[i, :n] = sample["labels"]
        image_positions[i] = int(sample["image_position"])
        image = sample["image"]
        if image is None:
            images.append(torch.zeros(3, image_size, image_size))
        else:
            images.append(image)

    return {
        "input_ids": input_ids,
        "labels": labels,
        "images": torch.stack(images, dim=0),
        "image_positions": image_positions,
    }


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def mostly_ascii(text: str, threshold: float = 0.94) -> bool:
    if not text:
        return False
    printable = [ch for ch in text if not ch.isspace()]
    if not printable:
        return False
    ascii_count = sum(ord(ch) < 128 for ch in printable)
    return ascii_count / len(printable) >= threshold


def clean_text(text: str, max_words: int = 650) -> Optional[str]:
    text = re.sub(r"\s+", " ", str(text).strip())
    if not text:
        return None
    low = text.lower()
    if any(x in low for x in ["http://", "https://", "www.", "<html", "{|", "```"]):
        return None
    if not mostly_ascii(text):
        return None
    # Reject symbol-heavy rows.
    symbol_count = sum(not (ch.isalnum() or ch.isspace() or ch in ".,!?;:'\"()-") for ch in text)
    if symbol_count / max(1, len(text)) > 0.03:
        return None
    words = text.split()
    if len(words) < 30:
        return None
    if len(words) > max_words:
        text = " ".join(words[:max_words])
        if text[-1] not in ".!?":
            text += "."
    return text


def clean_caption(caption: str) -> Optional[str]:
    caption = re.sub(r"\s+", " ", str(caption).strip())
    if not caption:
        return None
    low = caption.lower()
    banned = [
        "available for rent",
        "for sale",
        "stock photo",
        "watermark",
        "copyright",
        "shutterstock",
        "getty",
        "alamy",
        "instagram",
        "facebook",
        "twitter",
        "pinterest",
        "http://",
        "https://",
        "www.",
        "$",
        "€",
        "£",
    ]
    if any(x in low for x in banned):
        return None
    if not mostly_ascii(caption, threshold=0.97):
        return None
    sentences = re.split(r"(?<=[.!?])\s+", caption)
    visual_sentences = []
    sentence_bans = [
        "located in",
        "taken in",
        "available",
        "uploaded",
        "website",
        "photo by",
        "image by",
        "courtesy",
        "caption",
        "titled",
    ]
    for sent in sentences:
        s = sent.strip()
        if not s:
            continue
        sl = s.lower()
        if any(b in sl for b in sentence_bans):
            continue
        visual_sentences.append(s)
        if len(" ".join(visual_sentences).split()) >= 35:
            break
    caption = " ".join(visual_sentences).strip()
    words = caption.split()
    if len(words) < 5 or len(words) > 45:
        return None
    # Avoid proper-noun-heavy captions.
    capitalized = sum(1 for w in words[1:] if w[:1].isupper())
    if capitalized > max(3, len(words) // 4):
        return None
    if caption[-1] not in ".!?":
        caption += "."
    return caption


def pick_image(row: Dict[str, Any]) -> Optional[Image.Image]:
    for key in ["image", "jpg", "img", "picture"]:
        value = row.get(key)
        if isinstance(value, Image.Image):
            return value
    return None


def flatten_captions(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        out = []
        for item in value:
            if isinstance(item, str):
                out.append(item)
            elif isinstance(item, dict):
                out.extend(flatten_captions(item.get("caption") or item.get("text") or item.get("raw")))
        return out
    if isinstance(value, dict):
        out = []
        for key in ["caption", "captions", "sentences", "text", "raw"]:
            out.extend(flatten_captions(value.get(key)))
        return out
    return []


def pick_captions(row: Dict[str, Any]) -> List[str]:
    captions = []
    for key in [
        "caption",
        "captions",
        "sentences",
        "alt_text",
        "original_alt_text",
        "text",
        "texts",
        "raw",
        "caption_0",
        "caption_1",
        "caption_2",
        "caption_3",
        "caption_4",
    ]:
        captions.extend(flatten_captions(row.get(key)))
    # Some Flickr mirrors store one selected caption plus all captions.
    seen = set()
    unique = []
    for cap in captions:
        k = cap.strip().lower()
        if k and k not in seen:
            seen.add(k)
            unique.append(cap)
    return unique


class CleanDataset(IterableDataset):
    def __init__(self, args: argparse.Namespace, tokenizer: Any, cfg: ModelConfig):
        self.args = args
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.load_dataset = import_datasets()

    def _shard_shuffle(self, ds: Any, buffer_size: int, seed: int) -> Any:
        worker = get_worker_info()
        if worker is not None:
            ds = ds.shard(num_shards=worker.num_workers, index=worker.id)
            seed += worker.id
        return ds.shuffle(buffer_size=buffer_size, seed=seed)

    def tiny_stories(self, seed_offset: int = 0) -> Iterator[Dict[str, Any]]:
        ds = self.load_dataset("roneneldan/TinyStories", split="train", streaming=True)
        ds = self._shard_shuffle(ds, self.args.text_shuffle_buffer, self.args.seed + seed_offset)
        for row in ds:
            text = clean_text(row.get("text", ""), max_words=450)
            if text:
                yield encode_sample(self.tokenizer, text, None, self.cfg)

    def cosmopedia(self, seed_offset: int = 0) -> Iterator[Dict[str, Any]]:
        ds = self.load_dataset("HuggingFaceTB/cosmopedia", "stories", split="train", streaming=True)
        ds = self._shard_shuffle(ds, self.args.text_shuffle_buffer, self.args.seed + 1 + seed_offset)
        for row in ds:
            text = clean_text(row.get("text", ""), max_words=550)
            if text:
                yield encode_sample(self.tokenizer, text, None, self.cfg)

    def caption_dataset(self, dataset_name: str, split: str, seed: int) -> Iterator[Dict[str, Any]]:
        ds = self.load_dataset(dataset_name, split=split, streaming=True)
        ds = self._shard_shuffle(ds, self.args.image_shuffle_buffer, seed)
        rng = random.Random(seed + 99)
        for row in ds:
            image = pick_image(row)
            if image is None:
                continue
            caps = [c for c in (clean_caption(x) for x in pick_captions(row)) if c]
            if not caps:
                continue
            caption = rng.choice(caps)
            yield encode_sample(self.tokenizer, caption, image, self.cfg)

    def coco(self, seed_offset: int = 0) -> Iterator[Dict[str, Any]]:
        yield from self.caption_dataset(self.args.coco_dataset, self.args.coco_split, self.args.seed + 2 + seed_offset)

    def flickr(self, seed_offset: int = 0) -> Iterator[Dict[str, Any]]:
        yield from self.caption_dataset(self.args.flickr_dataset, self.args.flickr_split, self.args.seed + 3 + seed_offset)

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        stream_specs: List[Tuple[float, Any]] = []
        if self.args.mix_tinystories > 0:
            stream_specs.append((self.args.mix_tinystories, self.tiny_stories))
        if self.args.mix_cosmopedia > 0:
            stream_specs.append((self.args.mix_cosmopedia, self.cosmopedia))
        if self.args.mix_coco > 0:
            stream_specs.append((self.args.mix_coco, self.coco))
        if self.args.mix_flickr > 0:
            stream_specs.append((self.args.mix_flickr, self.flickr))
        if not stream_specs:
            raise RuntimeError("No datasets enabled.")
        worker = get_worker_info()
        rng = random.Random(self.args.seed + 10_000 + (worker.id if worker else 0))
        weights = [w for w, _ in stream_specs]
        cycles = [0 for _ in stream_specs]
        iters = [factory(0) for _, factory in stream_specs]
        while True:
            idx = rng.choices(range(len(iters)), weights=weights, k=1)[0]
            try:
                yield next(iters[idx])
            except StopIteration:
                cycles[idx] += 1
                iters[idx] = stream_specs[idx][1](cycles[idx] * 100_000)
                continue
            except Exception:
                continue


def get_lr(step: int, args: argparse.Namespace) -> float:
    if step < args.warmup_steps:
        return args.lr * (step + 1) / max(1, args.warmup_steps)
    progress = (step - args.warmup_steps) / max(1, args.max_steps - args.warmup_steps)
    return args.min_lr + 0.5 * (args.lr - args.min_lr) * (1.0 + math.cos(math.pi * progress))


def save_checkpoint(out_dir: Path, model, optimizer, scaler, cfg: ModelConfig, args: argparse.Namespace, step: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "model_config": asdict(cfg),
        "args": vars(args),
        "step": step,
        "stage": "clean_pretrain",
    }
    torch.save(ckpt, out_dir / f"ckpt_step_{step}.pt")
    torch.save(ckpt, out_dir / "ckpt_last.pt")


def preview(args: argparse.Namespace, tokenizer: Any, cfg: ModelConfig) -> None:
    ds = CleanDataset(args, tokenizer, cfg)
    checks = [
        ("TinyStories", ds.tiny_stories()),
        ("Cosmopedia", ds.cosmopedia()),
        ("COCO", ds.coco()),
        ("Flickr", ds.flickr()),
    ]
    for name, iterator in checks:
        print(f"\n## {name}")
        shown = 0
        for sample in iterator:
            labels = sample["labels"].tolist()
            target = [x for x in labels if x != -100]
            print(tokenizer.decode(target[:160], skip_special_tokens=False)[:700])
            print("---")
            shown += 1
            if shown >= args.preview:
                break


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default="/workspace/runs/tiny-mm-clean")
    p.add_argument("--tokenizer_dir", default="/workspace/tokenizer_8k_clean")
    p.add_argument("--resume_from", default=None)
    p.add_argument("--train_tokenizer", action="store_true")
    p.add_argument("--tokenizer_samples", type=int, default=50_000)
    p.add_argument("--preview", type=int, default=0)

    p.add_argument("--vocab_size", type=int, default=8192)
    p.add_argument("--d_model", type=int, default=304)
    p.add_argument("--n_layers", type=int, default=8)
    p.add_argument("--n_heads", type=int, default=8)
    p.add_argument("--max_seq_len", type=int, default=512)
    p.add_argument("--image_size", type=int, default=128)
    p.add_argument("--patch_size", type=int, default=16)
    p.add_argument("--dropout", type=float, default=0.0)

    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--grad_accum", type=int, default=2)
    p.add_argument("--max_steps", type=int, default=100_000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--min_lr", type=float, default=3e-5)
    p.add_argument("--warmup_steps", type=int, default=1000)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--num_workers", type=int, default=1)
    p.add_argument("--save_every", type=int, default=5000)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--seed", type=int, default=4242)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="bfloat16")

    p.add_argument("--mix_tinystories", type=float, default=0.50)
    p.add_argument("--mix_cosmopedia", type=float, default=0.15)
    p.add_argument("--mix_coco", type=float, default=0.25)
    p.add_argument("--mix_flickr", type=float, default=0.10)
    p.add_argument("--coco_dataset", default="jxie/coco_captions")
    p.add_argument("--coco_split", default="train")
    p.add_argument("--flickr_dataset", default="Mozilla/flickr30k-transformed-captions")
    p.add_argument("--flickr_split", default="test")
    p.add_argument("--text_shuffle_buffer", type=int, default=10_000)
    p.add_argument("--image_shuffle_buffer", type=int, default=2_000)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    # The tokenizer stream uses booleans so it can skip disabled text sources.
    args.use_tinystories = args.mix_tinystories > 0
    args.use_cosmopedia = args.mix_cosmopedia > 0
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "args.json").open("w") as f:
        json.dump(vars(args), f, indent=2)

    tokenizer = train_tokenizer_if_needed(args)
    pad_id = tokenizer.token_to_id("<|pad|>")
    cfg = ModelConfig(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        max_seq_len=args.max_seq_len,
        image_size=args.image_size,
        patch_size=args.patch_size,
        dropout=args.dropout,
    )
    if args.preview:
        preview(args, tokenizer, cfg)
        sys.stdout.flush()
        sys.stderr.flush()
        # HF streaming image datasets can leave downloader threads alive; on
        # some RunPod images Python aborts during interpreter finalization.
        os._exit(0)

    device = torch.device(args.device)
    model = TinyNativeMultimodalGPT(cfg, pad_id=pad_id).to(device)
    print(f"Parameters: {count_parameters(model) / 1e6:.2f}M")

    dataset = CleanDataset(args, tokenizer, cfg)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=lambda batch: collate_samples(batch, pad_id, cfg.image_size),
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
        fused=device.type == "cuda",
    )
    use_amp = device.type == "cuda" and args.dtype != "float32"
    amp_dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and args.dtype == "float16")
    start_step = 0
    if args.resume_from:
        ckpt = torch.load(args.resume_from, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if ckpt.get("scaler") is not None and scaler is not None:
            scaler.load_state_dict(ckpt["scaler"])
        start_step = int(ckpt.get("step", 0))
        print(f"Resumed at step {start_step}. Training until {args.max_steps}.")

    model.train()
    data_iter = iter(loader)
    running_loss = 0.0
    target_tokens_seen = 0
    image_samples_seen = 0
    t0 = time.time()
    pbar = tqdm(range(start_step, args.max_steps), desc="clean-pretrain", dynamic_ncols=True)
    for step in pbar:
        lr = get_lr(step, args)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0
        for _ in range(args.grad_accum):
            batch = next(data_iter)
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            images = batch["images"].to(device, non_blocking=True)
            image_positions = batch["image_positions"].to(device, non_blocking=True)
            target_tokens_seen += int(labels[:, 1:].ne(-100).sum().item())
            image_samples_seen += int(image_positions.ge(0).sum().item())
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                _, loss = model(input_ids, labels, images, image_positions)
                loss = loss / args.grad_accum
            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()
            step_loss += float(loss.detach().cpu()) * args.grad_accum

        if scaler.is_enabled():
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        if scaler.is_enabled():
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        running_loss = 0.95 * running_loss + 0.05 * step_loss if step > start_step else step_loss
        if step % args.log_every == 0:
            elapsed = max(time.time() - t0, 1e-6)
            pbar.set_postfix(
                loss=f"{running_loss:.3f}",
                lr=f"{lr:.2e}",
                it_s=f"{(step - start_step + 1) / elapsed:.2f}",
                tgt_tok=f"{target_tokens_seen/1e6:.1f}M",
                img=f"{image_samples_seen/1000:.0f}k",
            )
        if (step + 1) % args.save_every == 0:
            save_checkpoint(out_dir, model, optimizer, scaler, cfg, args, step + 1)

    save_checkpoint(out_dir, model, optimizer, scaler, cfg, args, args.max_steps)
    print(f"Done. Last checkpoint saved to {out_dir / 'ckpt_last.pt'}")
    print(f"Target tokens seen: {target_tokens_seen:,}")
    print(f"Image samples seen: {image_samples_seen:,}")


if __name__ == "__main__":
    main()
