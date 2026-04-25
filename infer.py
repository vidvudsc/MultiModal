#!/usr/bin/env python3
"""
Minimal browser inference server for the tiny native multimodal model.

Run:
  python infer.py --checkpoint /workspace/runs/tiny-mm/ckpt_last.pt --host 0.0.0.0 --port 7860

Then open the RunPod exposed HTTP port.
"""

from __future__ import annotations

import argparse
import cgi
import json
import mimetypes
import socket
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from PIL import Image
from tokenizers import ByteLevelBPETokenizer

from new_train import ModelConfig, TinyNativeMultimodalGPT, encode_sample


MODEL = None
TOKENIZER = None
DEVICE = None
CFG = None
ARGS = None


def load_tokenizer(tokenizer_dir: str) -> ByteLevelBPETokenizer:
    tok_dir = Path(tokenizer_dir)
    tokenizer = ByteLevelBPETokenizer(str(tok_dir / "vocab.json"), str(tok_dir / "merges.txt"))
    tokenizer.add_special_tokens(
        [
            "<|pad|>",
            "<|bos|>",
            "<|eos|>",
            "<|image|>",
            "<|/image|>",
            "<|user|>",
            "<|assistant|>",
        ]
    )
    return tokenizer


def load_model(checkpoint: str, tokenizer_dir: str, device: str):
    tokenizer = load_tokenizer(tokenizer_dir)
    ckpt = torch.load(checkpoint, map_location=device)
    cfg = ModelConfig(**ckpt["model_config"])
    model = TinyNativeMultimodalGPT(cfg, pad_id=tokenizer.token_to_id("<|pad|>"))
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    return model, tokenizer, cfg


def chunk_write(handler: BaseHTTPRequestHandler, data: str) -> None:
    raw = data.encode("utf-8")
    handler.wfile.write(f"{len(raw):x}\r\n".encode("ascii"))
    handler.wfile.write(raw)
    handler.wfile.write(b"\r\n")
    handler.wfile.flush()


def finish_chunks(handler: BaseHTTPRequestHandler) -> None:
    handler.wfile.write(b"0\r\n\r\n")
    handler.wfile.flush()


@torch.no_grad()
def stream_generate(prompt: str, image: Optional[Image.Image]):
    assert MODEL is not None and TOKENIZER is not None and CFG is not None and DEVICE is not None and ARGS is not None

    if ARGS.prompt_format == "chat":
        formatted = f"<|user|> {prompt.strip()}\n<|assistant|> "
    else:
        formatted = prompt.strip()
    sample = encode_sample(TOKENIZER, formatted, image, CFG, train=False)
    input_ids = sample["input_ids"].unsqueeze(0).to(DEVICE)
    images = sample["image"]
    if images is not None:
        images = images.unsqueeze(0).to(DEVICE)
    image_positions = torch.tensor([sample["image_position"]], device=DEVICE)
    eos_id = TOKENIZER.token_to_id("<|eos|>")

    emitted = ""
    for _ in range(ARGS.max_new_tokens):
        if input_ids.shape[1] >= CFG.max_seq_len:
            break

        labels = torch.full_like(input_ids, -100)
        with torch.autocast(
            device_type=DEVICE.type,
            dtype=torch.bfloat16 if ARGS.dtype == "bfloat16" else torch.float16,
            enabled=DEVICE.type in {"cuda", "mps"} and ARGS.dtype != "float32",
        ):
            logits, _ = MODEL(input_ids, labels, images, image_positions)

        next_logits = logits[:, -1, :] / max(ARGS.temperature, 1e-6)
        if ARGS.top_k > 0:
            vals, idx = torch.topk(next_logits, min(ARGS.top_k, next_logits.size(-1)))
            filtered = torch.full_like(next_logits, float("-inf"))
            filtered.scatter_(1, idx, vals)
            next_logits = filtered

        probs = F.softmax(next_logits, dim=-1)
        top_vals, top_idx = torch.topk(probs, min(5, probs.size(-1)), dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        token_id = int(next_id.item())
        token_prob = float(probs[0, token_id].detach().cpu())
        input_ids = torch.cat([input_ids, next_id], dim=1)

        if token_id == eos_id:
            break

        decoded = TOKENIZER.decode([token_id], skip_special_tokens=True)
        if decoded:
            emitted += decoded
            top = []
            for prob, idx in zip(top_vals[0].detach().cpu().tolist(), top_idx[0].detach().cpu().tolist()):
                top.append(
                    {
                        "token": TOKENIZER.decode([int(idx)], skip_special_tokens=True),
                        "prob": float(prob),
                    }
                )
            yield {
                "type": "token",
                "text": decoded,
                "prob": token_prob,
                "token_id": token_id,
                "top": top,
            }

        if any(stop in emitted for stop in ["<|eos|>", "<|user|>", "<|image|>"]):
            break


class Handler(BaseHTTPRequestHandler):
    server_version = "TinyMM/0.1"
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args) -> None:
        print(f"{self.address_string()} - {fmt % args}")

    def do_GET(self) -> None:
        if self.path in {"/", "/index.html"}:
            self.serve_file(Path(ARGS.index))
            return
        self.send_error(404)

    def serve_file(self, path: Path) -> None:
        if not path.exists():
            self.send_error(404, f"Missing {path}")
            return
        body = path.read_bytes()
        mime = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        if self.path != "/api/generate":
            self.send_error(404)
            return

        form = cgi.FieldStorage(
            fp=self.rfile,
            headers=self.headers,
            environ={
                "REQUEST_METHOD": "POST",
                "CONTENT_TYPE": self.headers.get("Content-Type", ""),
                "CONTENT_LENGTH": self.headers.get("Content-Length", "0"),
            },
        )
        prompt = form.getfirst("prompt", "Describe this image.")
        image = None
        if "image" in form and getattr(form["image"], "filename", ""):
            image = Image.open(form["image"].file).convert("RGB")

        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
        self.send_header("Transfer-Encoding", "chunked")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

        try:
            meta = {
                "type": "meta",
                "model": "TinyNativeMultimodalGPT",
                "checkpoint": str(ARGS.checkpoint),
                "device": str(DEVICE),
                "max_new_tokens": ARGS.max_new_tokens,
                "temperature": ARGS.temperature,
                "top_k": ARGS.top_k,
                "prompt_format": ARGS.prompt_format,
                "image": image is not None,
                "prompt_chars": len(prompt),
            }
            chunk_write(self, json.dumps(meta) + "\n")
            for event in stream_generate(prompt, image):
                chunk_write(self, json.dumps(event, ensure_ascii=False) + "\n")
            chunk_write(self, json.dumps({"type": "done"}) + "\n")
        except (BrokenPipeError, ConnectionResetError, socket.timeout):
            print("client disconnected during generation")
        except Exception as exc:
            try:
                chunk_write(self, json.dumps({"type": "error", "message": str(exc)}) + "\n")
            except (BrokenPipeError, ConnectionResetError, socket.timeout):
                print("client disconnected while sending error")
        finally:
            try:
                finish_chunks(self)
            except (BrokenPipeError, ConnectionResetError, socket.timeout):
                pass


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default="/workspace/runs/tiny-mm/ckpt_last.pt")
    p.add_argument("--tokenizer_dir", type=str, default="/workspace/tokenizer_8k")
    p.add_argument("--index", type=str, default="index.html")
    p.add_argument("--host", type=str, default="0.0.0.0")
    p.add_argument("--port", type=int, default=7860)
    default_device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    p.add_argument("--device", type=str, default=default_device)
    p.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="bfloat16")
    p.add_argument("--max_new_tokens", type=int, default=80)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--prompt_format", choices=["plain", "chat"], default="plain")
    return p.parse_args()


def main() -> None:
    global MODEL, TOKENIZER, DEVICE, CFG, ARGS
    ARGS = parse_args()
    DEVICE = torch.device(ARGS.device)
    MODEL, TOKENIZER, CFG = load_model(ARGS.checkpoint, ARGS.tokenizer_dir, ARGS.device)
    print(f"Loaded {ARGS.checkpoint} on {DEVICE}")
    print(f"Serving http://{ARGS.host}:{ARGS.port}")
    ThreadingHTTPServer((ARGS.host, ARGS.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
