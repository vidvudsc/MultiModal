#!/usr/bin/env python3
"""
Generate higher-quality SFT JSONL for the tiny native multimodal model.

Design goals:
- no image-to-Gemini by default; image examples are made from clean captions
- lots of topic variety so SFT does not collapse into repeated "what is X?" rows
- exact local arithmetic rows, not model-generated math
- short answers for a 15M student model

Output rows:
  {"user": "...", "assistant": "...", "kind": "..."}
  {"user": "...", "assistant": "...", "image": "sft_images/...", "kind": "..."}

Examples:
  export GEMINI_API_KEY=...

  python get2.py --mode text --out data/sft2_text.jsonl --count 2000
  python get2.py --mode math --out data/sft2_math.jsonl --count 500
  python get2.py --mode vision --out data/sft2_vision.jsonl --image_dir data/sft2_images --count 3000
  python get2.py --mode mixed --out data/sft2_mix.jsonl --image_dir data/sft2_images --count 5000
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
import random
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

from PIL import Image
from tqdm.auto import tqdm


SYSTEM_STYLE = """You are creating supervised fine-tuning data for a very tiny 15M parameter multimodal assistant.
The student is small, so every example must be short, concrete, direct, and easy to imitate.
Use natural English. Prefer one sentence. Use two or three sentences only for tiny stories.
No markdown tables. No long lists. No obscure facts. No hidden reasoning."""


FACT_SUBJECTS = [
    "the Sun", "the Moon", "Earth", "stars", "clouds", "rain", "snow", "wind", "thunder",
    "light", "shadows", "sound", "echoes", "magnets", "gravity", "electricity", "fire",
    "water", "ice", "steam", "plants", "seeds", "flowers", "trees", "leaves", "bees",
    "butterflies", "ants", "birds", "fish", "cats", "dogs", "horses", "frogs", "turtles",
    "the ocean", "rivers", "mountains", "volcanoes", "deserts", "forests", "maps",
    "clocks", "calendars", "bicycles", "cars", "trains", "airplanes", "boats", "bridges",
    "books", "music", "paint", "cameras", "computers", "phones", "robots", "recycling",
    "healthy food", "sleep", "exercise", "washing hands", "brushing teeth", "friendship",
    "sharing", "patience", "kindness", "telling the truth", "asking for help",
]


STORY_SUBJECTS = [
    "a lost red ball", "a sleepy cat", "a brave little dog", "a kite stuck in a tree",
    "a tiny robot", "a rainy window", "a magic pencil", "a blue backpack", "a missing sock",
    "a garden snail", "a paper boat", "a birthday cake", "a lonely star", "a noisy clock",
    "a small dragon", "a shy monster", "a fox in the snow", "a child learning to share",
    "two friends building a fort", "a girl finding a shell", "a boy fixing a toy car",
    "a bird with a bent wing", "a rabbit in a vegetable garden", "a train ride",
    "a lantern in the dark", "a treasure map", "a snowman on a warm day",
]


EVERYDAY_TASKS = [
    "packing a school bag", "cleaning a room", "making a sandwich", "watering a plant",
    "getting ready for bed", "crossing the street safely", "feeding a pet", "drawing a house",
    "sharing toys", "waiting in line", "saying sorry", "trying again after a mistake",
    "asking a teacher for help", "choosing a book", "putting on a jacket",
]


USER_STYLES = [
    "one-word prompt", "casual chat", "childlike question", "direct command", "why question",
    "how question", "what question", "yes/no question", "rewrite request", "summary request",
    "tiny story request", "simple explanation request", "friendly greeting", "uncertainty case",
]


BOOTSTRAP_ROWS = [
    {"user": "hello", "assistant": "Hello! How can I help?", "kind": "chat"},
    {"user": "hi", "assistant": "Hi! What would you like to do?", "kind": "chat"},
    {"user": "how are you?", "assistant": "I'm doing well. How can I help?", "kind": "chat"},
    {"user": "what can you do?", "assistant": "I can answer simple questions, write short stories, and describe images.", "kind": "chat"},
    {"user": "keep it short", "assistant": "Sure. I will keep it short.", "kind": "helpful"},
    {"user": "please answer in one sentence", "assistant": "Sure, I can answer in one sentence.", "kind": "helpful"},
    {"user": "if you don't know, what should you say?", "assistant": "I should say that I am not sure instead of guessing.", "kind": "uncertainty"},
    {"user": "what is 1+1?", "assistant": "2", "kind": "math"},
    {"user": "what is 2+2?", "assistant": "4", "kind": "math"},
    {"user": "tell me a tiny story", "assistant": "A little bird found a shiny seed. It planted it in soft dirt, and soon a small flower began to grow.", "kind": "story"},
    {"user": "continue: Once upon a time", "assistant": "there was a small cat who loved warm windows. One morning, it followed a sunbeam across the room and found a lost toy.", "kind": "continuation"},
]


def load_gemini(api_key: Optional[str]) -> Any:
    try:
        from google import genai
    except ImportError as exc:
        raise SystemExit("Install Gemini SDK first: pip install google-genai") from exc
    api_key = api_key or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("Set GEMINI_API_KEY or pass --api_key.")
    return genai.Client(api_key=api_key)


def gemini_text(client: Any, model: str, prompt: str, retries: int = 5, sleep: float = 2.0) -> str:
    last = None
    for attempt in range(retries):
        try:
            resp = client.models.generate_content(model=model, contents=prompt)
            return (resp.text or "").strip()
        except Exception as exc:
            last = exc
            time.sleep(sleep * (attempt + 1))
    raise RuntimeError(f"Gemini failed after {retries} retries: {last}")


def extract_json_array(text: str) -> List[Dict[str, Any]]:
    text = text.strip()
    text = re.sub(r"^```(?:json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    match = re.search(r"\[[\s\S]*\]", text)
    if match:
        text = match.group(0)
    data = json.loads(text)
    if not isinstance(data, list):
        raise ValueError("response is not a JSON array")
    rows = []
    for item in data:
        if isinstance(item, dict):
            rows.append(item)
    return rows


def norm_key(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", "", text.lower()).strip()


def valid_englishish(text: str) -> bool:
    if not text:
        return False
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return False
    return sum(ord(c) < 128 for c in chars) / len(chars) > 0.96


def clean_row(row: Dict[str, Any], default_kind: str = "text") -> Optional[Dict[str, Any]]:
    bad = [
        "as an ai", "language model", "training data", "json array", "cannot browse",
        "<|user|>", "<|assistant|>", "<|image|>", "according to the caption",
    ]
    user = " ".join(str(row.get("user", "")).strip().split())
    assistant = " ".join(str(row.get("assistant", "")).strip().split())
    if not user or not assistant:
        return None
    lower = f"{user} {assistant}".lower()
    if any(x in lower for x in bad):
        return None
    if not valid_englishish(user) or not valid_englishish(assistant):
        return None
    if len(user) > 220 or len(assistant) > 520:
        return None
    if len(assistant.split()) > 85:
        return None
    out = {
        "user": user,
        "assistant": assistant,
        "kind": str(row.get("kind") or default_kind),
    }
    if row.get("image"):
        out["image"] = str(row["image"])
    return out


def load_seen(path: Path) -> set[Tuple[str, str, str]]:
    seen = set()
    if not path.exists():
        return seen
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
            except Exception:
                continue
            seen.add((norm_key(str(row.get("user", ""))), norm_key(str(row.get("assistant", ""))), str(row.get("image", ""))))
    return seen


def append_jsonl(path: Path, rows: Iterable[Dict[str, Any]], seen: set[Tuple[str, str, str]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            key = (norm_key(row["user"]), norm_key(row["assistant"]), str(row.get("image", "")))
            if key in seen:
                continue
            seen.add(key)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def prompt_text_batch(batch_size: int, rng: random.Random) -> str:
    fact_subjects = rng.sample(FACT_SUBJECTS, k=min(12, len(FACT_SUBJECTS)))
    story_subjects = rng.sample(STORY_SUBJECTS, k=min(8, len(STORY_SUBJECTS)))
    tasks = rng.sample(EVERYDAY_TASKS, k=min(6, len(EVERYDAY_TASKS)))
    styles = rng.sample(USER_STYLES, k=min(8, len(USER_STYLES)))
    return f"""{SYSTEM_STYLE}

Create {batch_size} diverse TEXT-ONLY examples.

Use these topic pools, but vary the exact wording:
- simple facts/explanations: {", ".join(fact_subjects)}
- tiny stories/story continuations: {", ".join(story_subjects)}
- everyday help: {", ".join(tasks)}
- user styles: {", ".join(styles)}

Required balance:
- about 35% simple facts or explanations
- about 25% tiny stories or story continuations
- about 15% everyday helpful responses
- about 10% rewrites/summaries/definitions
- about 10% uncertainty or "I need more info" behavior
- about 5% greetings/short chat

Rules:
- Return ONLY a valid JSON array.
- Each item must have exactly "user", "assistant", and "kind".
- "kind" must be one of: fact, explanation, story, continuation, helpful, rewrite, summary, uncertainty, chat.
- User prompts must be varied. Avoid repeating "Can you tell me" or "What is".
- Include some one-word prompts like "moon", "story", "rain", "hello".
- Include some messy-but-normal prompts like "make this shorter: ...".
- Facts must be common and stable. No politics, exact current events, medical/legal/financial advice.
- Stories should be 2-4 short sentences.

Example:
[
  {{"user":"moon","assistant":"The Moon is a rocky ball that goes around Earth. It shines because it reflects sunlight.","kind":"fact"}},
  {{"user":"continue: The tiny robot found a key","assistant":"The tiny robot picked up the key and beeped with joy. It opened a small door and found a room full of glowing stars.","kind":"continuation"}},
  {{"user":"make this shorter: The dog was running very quickly across the green field.","assistant":"The dog ran quickly across the field.","kind":"rewrite"}}
]
"""


def local_math_rows(count: int, rng: random.Random) -> List[Dict[str, Any]]:
    templates = [
        ("What is {a} + {b}?", "{ans}"),
        ("{a} plus {b}", "{ans}"),
        ("What is {a} - {b}?", "{ans}"),
        ("If I have {a} apples and get {b} more, how many do I have?", "You have {ans} apples."),
        ("If there are {a} birds and {b} fly away, how many are left?", "{ans} birds are left."),
        ("Count by twos from {a} to {b}.", "{seq}"),
        ("Which is bigger, {a} or {b}?", "{ans} is bigger."),
    ]
    rows = []
    for _ in range(count):
        kind = rng.randrange(len(templates))
        if kind in [0, 1, 3]:
            a, b = rng.randint(0, 20), rng.randint(0, 20)
            ans = a + b
            user_t, ans_t = templates[kind]
            rows.append({"user": user_t.format(a=a, b=b), "assistant": ans_t.format(ans=ans), "kind": "math"})
        elif kind in [2, 4]:
            a, b = rng.randint(0, 20), rng.randint(0, 20)
            if b > a:
                a, b = b, a
            ans = a - b
            user_t, ans_t = templates[kind]
            rows.append({"user": user_t.format(a=a, b=b), "assistant": ans_t.format(ans=ans), "kind": "math"})
        elif kind == 5:
            start = rng.choice([0, 2, 4, 6, 8, 10])
            end = start + rng.choice([6, 8, 10, 12])
            seq = ", ".join(str(x) for x in range(start, end + 1, 2))
            rows.append({"user": templates[kind][0].format(a=start, b=end), "assistant": seq, "kind": "math"})
        else:
            a, b = rng.randint(0, 30), rng.randint(0, 30)
            ans = max(a, b)
            rows.append({"user": templates[kind][0].format(a=a, b=b), "assistant": templates[kind][1].format(ans=ans), "kind": "math"})
    return rows


def local_bootstrap_rows(count: int, rng: random.Random) -> List[Dict[str, Any]]:
    rows = []
    while len(rows) < count:
        rows.extend(rng.sample(BOOTSTRAP_ROWS, k=len(BOOTSTRAP_ROWS)))
        rows.extend(local_math_rows(max(3, len(BOOTSTRAP_ROWS) // 2), rng))
    return rows[:count]


def load_dataset_fn():
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("Install datasets first: pip install datasets") from exc
    return load_dataset


def flatten(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        out = []
        for item in value:
            out.extend(flatten(item))
        return out
    if isinstance(value, dict):
        out = []
        for key in ["caption", "captions", "alt_text", "original_alt_text", "sentences", "text", "raw"]:
            out.extend(flatten(value.get(key)))
        return out
    return []


def row_image_and_captions(row: Dict[str, Any]) -> Tuple[Optional[Image.Image], List[str]]:
    image = None
    for key in ["image", "jpg", "img", "picture"]:
        if isinstance(row.get(key), Image.Image):
            image = row[key]
            break
    captions = []
    for key in ["caption", "captions", "alt_text", "original_alt_text", "sentences", "text"]:
        captions.extend(flatten(row.get(key)))
    unique = []
    seen = set()
    for cap in captions:
        cap = " ".join(str(cap).strip().split())
        k = cap.lower()
        if cap and k not in seen:
            seen.add(k)
            unique.append(cap)
    return image, unique


def clean_caption(caption: str) -> Optional[str]:
    caption = " ".join(str(caption).strip().split())
    if not caption:
        return None
    bad = ["stock photo", "shutterstock", "alamy", "getty", "copyright", "for sale", "available for rent", "watermark"]
    lower = caption.lower()
    if any(x in lower for x in bad):
        return None
    if not valid_englishish(caption):
        return None
    words = caption.split()
    if len(words) < 5 or len(words) > 45:
        return None
    if caption[-1] not in ".!?":
        caption += "."
    return caption


def prompt_vision_from_caption(caption: str, batch_size: int, rng: random.Random) -> str:
    focus = rng.sample(
        ["objects", "actions", "colors", "location", "people", "animals", "simple story", "one-sentence caption"],
        k=5,
    )
    return f"""{SYSTEM_STYLE}

Create exactly {batch_size} IMAGE examples for a tiny multimodal assistant.
You are given a clean caption that describes the image. Do not mention the word "caption".

Image caption:
{caption}

Required balance:
- at least 1 direct image description
- at least 1 simple visual question-answer
- at least 1 two-sentence image description
- at least 1 tiny story inspired by the image
- at least 1 yes/no image matching question
- sometimes include a "not clear from the image" question

Focus areas: {", ".join(focus)}

Rules:
- Return ONLY a valid JSON array.
- Each item must have exactly "user", "assistant", and "kind".
- "kind" must be one of: image_caption, image_detail, image_vqa, image_story, image_match, image_unclear.
- The answer must only use details supported by the caption.
- For image_match, ask a yes/no question about whether something is visible. Include both Yes and No examples when possible.
- For image_unclear, ask about something not knowable from the caption, such as a person's name, exact age, what happened before, or what will happen next. Answer with "It is not clear from the image."
- Use varied user wording: "Describe this image", "What do you see?", "Tell a tiny story about this picture", "Is there ...?"
- Prefer "Yes." or "No." for image_match, optionally with one short reason.
- Do not write long answers. Image stories should be 2-4 short sentences.

Example:
[
  {{"user":"Describe this image.","assistant":"A dog is pulling a woman on a skateboard.","kind":"image_caption"}},
  {{"user":"Describe this image in two sentences.","assistant":"A dog is pulling a woman on a skateboard. They appear to be moving together outdoors.","kind":"image_detail"}},
  {{"user":"What is the dog doing?","assistant":"The dog is pulling a woman on a skateboard.","kind":"image_vqa"}},
  {{"user":"Does this image show a dog?","assistant":"Yes.","kind":"image_match"}},
  {{"user":"What is the woman's name?","assistant":"It is not clear from the image.","kind":"image_unclear"}},
  {{"user":"Write a tiny story inspired by this picture.","assistant":"The dog tugged the skateboard down the path. The woman laughed and held on tight as they rolled together.","kind":"image_story"}}
]
"""


def iter_caption_rows(dataset_choice: str, seed: int) -> Iterator[Tuple[str, Image.Image, str]]:
    load_dataset = load_dataset_fn()
    datasets: List[Tuple[str, str, str]] = []
    if dataset_choice in ["coco", "both"]:
        datasets.append(("coco", "jxie/coco_captions", "train"))
    if dataset_choice in ["flickr", "both"]:
        datasets.append(("flickr", "Mozilla/flickr30k-transformed-captions", "test"))

    rng = random.Random(seed)
    for source, name, split in datasets:
        ds = load_dataset(name, split=split, streaming=True).shuffle(buffer_size=2000, seed=seed + len(source))
        for row in ds:
            image, captions = row_image_and_captions(row)
            if image is None:
                continue
            captions = [c for c in (clean_caption(x) for x in captions) if c]
            if not captions:
                continue
            yield source, image, rng.choice(captions)


def save_image(image: Image.Image, image_dir: Path, name: str) -> Path:
    image_dir.mkdir(parents=True, exist_ok=True)
    path = image_dir / name
    image.convert("RGB").save(path, format="JPEG", quality=90)
    return path


def generate_text(args: argparse.Namespace, count: int, seen: set[Tuple[str, str, str]]) -> int:
    client = load_gemini(args.api_key)
    rng = random.Random(args.seed)
    made = 0
    pbar = tqdm(total=count, desc="text")
    while made < count:
        want = min(args.batch_examples, count - made)
        try:
            rows = extract_json_array(gemini_text(client, args.model, prompt_text_batch(want, rng), sleep=args.sleep))
            cleaned = [r for r in (clean_row(x, "text") for x in rows) if r]
        except Exception as exc:
            print(f"bad text batch: {exc}")
            continue
        wrote = append_jsonl(Path(args.out), cleaned[:want], seen)
        made += wrote
        pbar.update(wrote)
        time.sleep(args.sleep)
    pbar.close()
    return made


def generate_math(args: argparse.Namespace, count: int, seen: set[Tuple[str, str, str]]) -> int:
    rng = random.Random(args.seed + 1000)
    rows = [r for r in (clean_row(x, "math") for x in local_math_rows(count * 2, rng)) if r]
    return append_jsonl(Path(args.out), rows[:count], seen)


def generate_bootstrap(args: argparse.Namespace, count: int, seen: set[Tuple[str, str, str]]) -> int:
    rng = random.Random(args.seed + 1500)
    rows = [r for r in (clean_row(x, "bootstrap") for x in local_bootstrap_rows(count * 2, rng)) if r]
    return append_jsonl(Path(args.out), rows[:count], seen)


def generate_vision(args: argparse.Namespace, count: int, seen: set[Tuple[str, str, str]]) -> int:
    client = load_gemini(args.api_key)
    rng = random.Random(args.seed + 2000)
    made = 0
    pbar = tqdm(total=count, desc="vision rows")
    image_idx = 0
    for source, image, caption in iter_caption_rows(args.dataset, args.seed):
        if made >= count:
            break
        image_path = save_image(image, Path(args.image_dir), f"{source}_{image_idx:07d}.jpg")
        image_idx += 1
        rel_image = os.path.relpath(image_path, Path(args.out).parent)
        want = min(args.vision_examples_per_image, count - made)
        try:
            rows = extract_json_array(gemini_text(client, args.model, prompt_vision_from_caption(caption, want, rng), sleep=args.sleep))
        except Exception as exc:
            print(f"bad vision batch: {exc}")
            continue
        cleaned = []
        for row in rows:
            row["image"] = rel_image
            cleaned_row = clean_row(row, "image")
            if cleaned_row:
                cleaned.append(cleaned_row)
        wrote = append_jsonl(Path(args.out), cleaned[:want], seen)
        made += wrote
        pbar.update(wrote)
        time.sleep(args.sleep)
    pbar.close()
    return made


def preview_rows(args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)
    print("## Local math preview")
    for row in local_math_rows(5, rng):
        print(json.dumps(row, ensure_ascii=False))
    print("\n## Text prompt preview")
    print(prompt_text_batch(8, rng)[:1600])
    print("\n## Bootstrap preview")
    for row in local_bootstrap_rows(5, rng):
        print(json.dumps(row, ensure_ascii=False))
    print("\n## Vision prompt preview")
    print(prompt_vision_from_caption("A dog is pulling a woman on a skateboard.", 6, rng)[:1800])


def write_report(path: Path) -> None:
    counts = Counter()
    image_rows = 0
    total = 0
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
            except Exception:
                continue
            total += 1
            counts[str(row.get("kind", "unknown"))] += 1
            if row.get("image"):
                image_rows += 1
    report = {
        "rows": total,
        "image_rows": image_rows,
        "text_rows": total - image_rows,
        "kinds": dict(counts.most_common()),
    }
    report_path = path.with_suffix(path.suffix + ".report.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"report saved to {report_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["text", "math", "bootstrap", "vision", "mixed", "preview"], default="preview")
    p.add_argument("--out", default="data/sft2_mix.jsonl")
    p.add_argument("--image_dir", default="data/sft2_images")
    p.add_argument("--count", type=int, default=1000)
    p.add_argument("--model", default="gemini-2.5-flash-lite")
    p.add_argument("--api_key", default=None)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--sleep", type=float, default=0.5)
    p.add_argument("--batch_examples", type=int, default=12)
    p.add_argument("--vision_examples_per_image", type=int, default=6)
    p.add_argument("--dataset", choices=["coco", "flickr", "both"], default="both")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--mix_text", type=float, default=0.40)
    p.add_argument("--mix_math", type=float, default=0.05)
    p.add_argument("--mix_vision", type=float, default=0.55)
    p.add_argument("--bootstrap_rows", type=int, default=100)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.out)
    if args.mode == "preview":
        preview_rows(args)
        return
    if args.overwrite and out.exists():
        out.unlink()
    seen = load_seen(out)

    if args.mode == "text":
        n = generate_text(args, args.count, seen)
    elif args.mode == "math":
        n = generate_math(args, args.count, seen)
    elif args.mode == "bootstrap":
        n = generate_bootstrap(args, args.count, seen)
    elif args.mode == "vision":
        n = generate_vision(args, args.count, seen)
    else:
        total = args.mix_text + args.mix_math + args.mix_vision
        text_n = int(args.count * args.mix_text / total)
        math_n = int(args.count * args.mix_math / total)
        bootstrap_n = min(args.bootstrap_rows, max(0, text_n // 10))
        text_n = max(0, text_n - bootstrap_n)
        vision_n = args.count - text_n - math_n - bootstrap_n
        n = 0
        n += generate_bootstrap(args, bootstrap_n, seen)
        n += generate_math(args, math_n, seen)
        n += generate_text(args, text_n, seen)
        n += generate_vision(args, vision_n, seen)
    print(f"wrote {n} rows to {out}")
    write_report(out)


if __name__ == "__main__":
    main()
