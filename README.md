# Tiny Multimodal

Tiny Multimodal is a small from-scratch native multimodal Transformer experiment. The goal is not to compete with large VLMs, but to understand the whole stack: tokenizer, text pretraining, image patch conditioning, supervised fine tuning, and browser inference.

The current model is a roughly 14.7M parameter decoder-only Transformer. It accepts text tokens and optional image patches in the same causal sequence, then predicts the next text token.

## Current Status

The latest published checkpoint is the clean pretrained `~68k` step run. It is a pretrained base model, not an instruction-tuned assistant yet.

Expected behavior:

- Text prompts usually continue as short stories or simple explanations.
- Image prompts can produce short caption-like outputs.
- Colors and broad scene cues are starting to work.
- Fine-grained visual recognition is still weak.
- Chat behavior needs SFT.

The checkpoint is too large for normal Git history, so it is published as a GitHub Release asset.

## Repository Layout

```text
new_train.py                 # clean pretraining script
finetune.py                  # supervised fine-tuning script
get2.py                      # Gemini-assisted SFT data generator
infer.py                     # local browser inference server
index.html                   # simple web UI
requirements.txt

models/pretrained_68k/
  args.json                  # pretraining run config
  tokenizer_8k_clean/        # tokenizer used by the 68k checkpoint

data/
  sft2_final.jsonl           # current SFT dataset manifest
  sft2_final.jsonl.report.json
  sft2_final_images/         # images referenced by the SFT JSONL
```

Local old exports, failed runs, and transfer archives are intentionally ignored under `legacy/`.

## Architecture

The model is a compact decoder-only GPT-style network:

- 8 Transformer blocks
- 304 hidden size
- 8 attention heads
- RMSNorm
- SwiGLU MLP
- tied token embedding / output head
- 8192-token ByteLevel BPE tokenizer
- 128x128 images
- 16x16 image patches, giving 64 image patch tokens

For text-only input, the model behaves like a normal causal language model.

For image input, an image is resized, split into 16x16 patches, projected into the model hidden size, and inserted into the token sequence:

```text
<|bos|> <|image|> [64 image patch vectors] <|/image|> text...
```

The Transformer then attends over the previous tokens and image patches while predicting caption or answer tokens.

## Important Fix From Earlier Attempts

Earlier experiments inserted image patches into positions filled with `<|pad|>` token IDs. The patch embeddings replaced the pad embeddings, but the attention mask still treated those positions as padding. That meant text could fail to attend to image patches.

The clean training code fixes this by forcing image patch positions to be visible in the attention mask after patch insertion. It also uses `<|image|>` placeholder IDs for image patch slots, which makes debugging clearer.

This is the main reason the clean 40k/64k/68k models behave much better on images than the first 50k run.

## Data Strategy

The clean pretraining mix avoids noisy CC3M-style web captions and uses simpler data:

- TinyStories for compact story structure
- filtered Cosmopedia-style story/explanation text
- COCO captions for visual grounding
- Flickr captions for additional image-caption variety

The current SFT dataset was generated with `get2.py`. It mixes:

- short factual answers
- simple explanations
- story continuations
- small math examples
- image captions
- image details
- image VQA
- image yes/no matching
- "not clear from image" uncertainty examples

The SFT data is meant to turn the pretrained base into a small assistant-like model without destroying the base story/caption behavior.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run The Pretrained 68k Model

Download the checkpoint release asset and place it here:

```text
models/pretrained_68k/ckpt_last.pt
```

Then run:

```bash
python3 infer.py \
  --checkpoint models/pretrained_68k/ckpt_last.pt \
  --tokenizer_dir models/pretrained_68k/tokenizer_8k_clean \
  --index index.html \
  --host 127.0.0.1 \
  --port 7860 \
  --device mps \
  --dtype float32 \
  --temperature 0.5 \
  --top_k 30 \
  --prompt_format plain
```

Open:

```text
http://127.0.0.1:7860
```

Use `--prompt_format plain` for the pretrained base model. Use `--prompt_format chat` after SFT.

## Fine Tune

On a MacBook, use small batches:

```bash
python3 finetune.py \
  --checkpoint models/pretrained_68k/ckpt_last.pt \
  --tokenizer_dir models/pretrained_68k/tokenizer_8k_clean \
  --out_dir runs/tiny-mm-clean-sft-mac \
  --jsonl data/sft2_final.jsonl \
  --image_base_dir data \
  --max_steps 1500 \
  --save_every 250 \
  --batch_size 4 \
  --grad_accum 8 \
  --num_workers 0 \
  --lr 1e-5 \
  --min_lr 3e-6 \
  --mix_jsonl 0.85 \
  --mix_tinystories 0.05 \
  --mix_coco 0.07 \
  --mix_flickr 0.03 \
  --device mps \
  --dtype float32
```

On a CUDA machine, use larger batches:

```bash
python finetune.py \
  --checkpoint /workspace/runs/tiny-mm-clean/ckpt_last.pt \
  --tokenizer_dir /workspace/tokenizer_8k_clean \
  --out_dir /workspace/runs/tiny-mm-clean-sft \
  --jsonl /workspace/data/sft2_final.jsonl \
  --image_base_dir /workspace/data \
  --max_steps 3000 \
  --save_every 500 \
  --batch_size 32 \
  --grad_accum 2 \
  --num_workers 1 \
  --lr 2e-5 \
  --min_lr 5e-6 \
  --mix_jsonl 0.85 \
  --mix_tinystories 0.05 \
  --mix_coco 0.07 \
  --mix_flickr 0.03 \
  --device cuda
```

## Pretrain From Scratch

```bash
python new_train.py \
  --out_dir runs/tiny-mm-clean \
  --tokenizer_dir tokenizer_8k_clean \
  --max_steps 100000 \
  --save_every 2000 \
  --batch_size 64 \
  --grad_accum 1 \
  --num_workers 1 \
  --device cuda
```

For Mac experimentation, reduce `batch_size`, set `num_workers 0`, and use `--device mps --dtype float32`.

## Previous Attempts

The first working model used a noisier data mix and reached 50k pretraining steps, then a small SFT run. It could generate text but often fell into TinyStories-style continuation and had weak image grounding.

The main issues were:

- noisy image-caption data was too broad for a tiny model
- SFT data was repetitive and caused forgetting
- image patch positions were likely masked as padding
- chat-formatted prompts were used against a base pretrained model during inference

The current clean line fixes those issues by using simpler caption data, cleaner SFT generation, plain base-model prompting, and correct image attention masking.

## Limitations

This is a tiny educational model. It should be treated as a toy research artifact:

- not robust enough for production
- not reliably factual
- weak on detailed visual reasoning
- likely to hallucinate
- sensitive to prompt format and decoding settings

The useful part is that the entire multimodal path is small enough to inspect, train, modify, and understand end to end.

