---
license: apache-2.0
library_name: pytorch
tags:
  - multimodal
  - vision-language
  - tiny-model
  - pytorch
  - experimental
---

# Tiny Multimodal Pretrained 68k

This repository contains a clean pretrained checkpoint for Tiny Multimodal, a small from-scratch native multimodal Transformer experiment.

This is a **pretrained base model**, not an instruction-tuned chat assistant.

## Files

```text
ckpt_last.pt
args.json
tokenizer_8k_clean/
  vocab.json
  merges.txt
```

## Model

- ~14.7M parameters
- decoder-only Transformer
- 8 layers
- hidden size 304
- 8 attention heads
- 8192-token ByteLevel BPE tokenizer
- 128x128 image input
- 16x16 image patches, 64 image patch tokens

Images are converted into patch vectors and inserted into the causal token sequence:

```text
<|bos|> <|image|> [64 image patch vectors] <|/image|> text...
```

## Training

This checkpoint is from the clean pretraining line, stopped around the ~68k step range.

The data mix was:

- TinyStories for compact story structure
- filtered Cosmopedia-style story/explanation text
- COCO captions for visual grounding
- Flickr captions for additional image-caption variety

## Current Behavior

The model is starting to learn some visual grounding. In simple manual tests, color words and broad scene cues started to affect generations, for example red images increasing red-related text.

The vision side is still pretty bad. It is weak at object identity, counting, spatial details, and robust visual question answering.

Use this as a toy research checkpoint, not a reliable assistant.

## Usage

Clone the code repository:

```bash
git clone https://github.com/vidvudsc/MultiModal
cd MultiModal
```

Download this model folder into:

```text
models/pretrained_68k/
```

Run local inference:

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

Use `--prompt_format plain` for this pretrained base checkpoint. Use chat format only after supervised fine tuning.

