from __future__ import annotations
import base64
import json
import re
import time
from pathlib import Path
import cv2
import numpy as np
from config import setting
from src.data import corrupted_path, list_corrupted, load_image, load_corrupted, path_for_each_id
import pandas as pd
import httpx

OLLAMA_URL = setting.ollama_url
VISION_MODEL = setting.vision_model
TEXT_MODEL = setting.text_model

PROMPT_DIR = setting.prompt_dir
OUT_DIR = setting.out_dir
OUT_DIR.mkdir(parents=True, exist_ok=True)

def load_prompt(prompt_name: str) -> str:
    prompt_path = Path(PROMPT_DIR) / f"{prompt_name}.txt"
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file {prompt_path} does not exist.")
    with open(prompt_path, "r") as f:
        return f.read().strip()

def encode_image(img) -> str:
    if isinstance(img, (str, Path)):
        return base64.b64encode(Path(img).read_bytes()).decode()

    arr = np.asarray(img)
    if arr.dtype != np.uint8:
        if arr.max() <= 1.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).round().astype(np.uint8)   # round, do not truncate

    bgr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR if arr.ndim == 2 else cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".png", bgr)
    if not ok:
        raise ValueError("could not encode the image")
    return base64.b64encode(buf.tobytes()).decode()

def ask(prompt: str, image=None, model: str | None = None, temperature: float = 0.0,
        seed: int | None = None, timeout: int = 300,
        num_predict: int = setting.num_predict, json_mode: bool = False) -> str:
    model = model or (VISION_MODEL if image is not None else TEXT_MODEL)
    payload = {"model": model, "prompt": prompt, "stream": False,
               "options": {"temperature": temperature, "num_predict": num_predict}}
    if json_mode:
        payload["format"] = "json"      # llama3 otherwise stops before the closing brace
    if seed is not None:
        payload["options"]["seed"] = seed
    if image is not None:
        payload["images"] = [encode_image(image)]

    reply = httpx.post(OLLAMA_URL, json=payload, timeout=timeout)
    if reply.status_code != 200:
        raise RuntimeError(f"ollama returned {reply.status_code} for model {model!r}: "
                           f"{reply.text[:300]}")
    return reply.json()["response"]

def extract_json(text: str) -> dict:
    text = re.sub(r"```(?:json)?", "", text)
    start = text.find("{")
    if start == -1:
        raise ValueError("no json object in the reply")

    depth = 0
    for i, ch in enumerate(text[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start:i + 1])
    raise ValueError("json object is not closed")

def ask_json(prompt: str, image=None, retries: int = 2, **kwargs) -> tuple[dict, str]:
    """ask, then parse. If the reply is not valid json, hand it back and ask for a fix, with 2 retry"""
    text = ask(prompt, image, json_mode=True, **kwargs)
    for attempt in range(retries):
        try:
            return extract_json(text), text
        except ValueError:
            repair = (f"{prompt}\n\nYour previous reply was not valid JSON:\n{text}\n\n"
                      f"Reply again with the JSON object only, no explanation, no code fences.")
            text = ask(repair, image, json_mode=True, **kwargs)
    return extract_json(text), text

def check_keys(record: dict, expected: list) -> list:
    """Check which fields the model forgot"""
    return [k for k in expected if k not in record]

def extra_keys(record: dict, expected: list) -> list:
    """fields the model invented on its own, a schema violation that check_keys cannot see"""
    return [k for k in record if k not in expected]

def describe_image(img_id: str, split: str = "test", prompt_name: str = "vlm_structured",
                   **kwargs) -> dict:
    """hand the raw image directly to the model. (for task 1)"""
    img = load_corrupted(img_id) if "_blur" in img_id or "_lowcontrast" in img_id  else load_image(img_id, split)
    start = time.time()
    record , text = ask_json(load_prompt(prompt_name), img, **kwargs)
    return {"image_id": img_id, "prompt": prompt_name, "seconds": round(time.time() - start, 1),
            "missing_keys": check_keys(record, setting.vision_keys), "extra_keys": extra_keys(record, setting.vision_keys), "record": record, "raw": text}

def summarise_features(summary_text: str, prompt_name: str = "numbers_first", **kwargs) -> dict:
    """the model never sees the image, only the numbers measured from the mask (for task 2)"""
    prompt = load_prompt(prompt_name).replace("{{FEATURES}}", summary_text)

    started = time.time()
    record, raw = ask_json(prompt, None, **kwargs)
    return {"prompt": prompt_name, "seconds": round(time.time() - started, 1),
            "missing_keys": check_keys(record, setting.feature_keys), "extra_keys": extra_keys(record,setting.feature_keys), "record": record, "raw": raw}

def repeat_run(img_id: str, split: str = "test", n: int = 3, temperature: float = 0.7,
               prompt_name: str = "vlm_structured") -> list:
    """same image, same prompt, n times, to show how much the answer wanders"""
    return [describe_image(img_id, split, prompt_name, temperature=temperature) for _ in range(n)]

def save_json(obj, name: str) -> Path:
    path = OUT_DIR / f"{name}.json"
    path.write_text(json.dumps(obj, indent=2))
    return path

def server_is_up() -> bool:
    try:
        response = httpx.get(setting.ollama_healcheck_url, timeout=5.0)
        return response.is_success 
    except httpx.RequestError:
        return False

def compare_prompts(img_id: str = "test_000", split: str = "test") -> pd.DataFrame:
    """the naive prompt against the structured one, on the same image"""
    rows = []
    for name in ("vlm_naive", "vlm_structured"):
        path = corrupted_path(img_id) if "_blur" in img_id or "_lowcontrast" in img_id \
            else path_for_each_id(img_id, split)["images"]
        text = ask(load_prompt(name), path)
        try:
            record = extract_json(text)
            parsed, keys = True, len(record)
        except ValueError:
            parsed, keys = False, 0
        rows.append({"prompt": name, "chars": len(text), "valid_json": parsed,
                     "keys": keys, "reply": text[:150].replace("\n", " ")})
    return pd.DataFrame(rows)

def describe_corrupted() -> pd.DataFrame:
    """the vision model on the blurred and contrast crushed images, next to the clean ones"""
    rows = []
    for stem, base_id, corruption in list_corrupted():
        for image_id, tag in [(base_id, "clean"), (stem, corruption)]:
            out = describe_image(image_id, "test")
            rows.append({"base_id": base_id, "variant": tag,
                         "modality": out["record"].get("modality"),
                         "image_quality": out["record"].get("image_quality"),
                         "features": "; ".join(out["record"].get("notable_features", [])[:3])})
    return pd.DataFrame(rows)

if __name__ == "__main__":
    if not server_is_up():
        raise SystemExit("ollama is not answering on port 11434 - start it first")

    out = describe_image("test_000", "test")
    print(f"{out['image_id']} answered in {out['seconds']}s, "
          f"missing keys: {out['missing_keys'] or 'none'}\n")
    print(json.dumps(out["record"], indent=2))
    save_json(out, "task1_smoke_test")


