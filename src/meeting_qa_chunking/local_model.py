"""Small cached wrapper around a local Hugging Face chat model."""

import hashlib
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class LocalChatModel:
    def __init__(
        self,
        model_name: str,
        revision: str | None = None,
        max_new_tokens: int = 256,
        seed: int = 42,
        temperature: float = 0.0,
        cache_dir: Path = Path(".cache/responses"),
        prequantized: bool = False,
    ) -> None:
        self.model_name = model_name
        self.revision = revision
        self.max_new_tokens = max_new_tokens
        self.seed = seed
        self.temperature = temperature
        self.cache_dir = cache_dir
        self.prequantized = prequantized
        self.model_calls = 0
        self.cache_hits = 0
        self.last_cache_hit = False

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if prequantized and self.device.type != "cuda":
            raise RuntimeError("The prequantized model requires a CUDA device")
        dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        load_options = {"revision": revision, "dtype": dtype}
        if prequantized:
            # The bitsandbytes checkpoint already contains its quantization map.
            load_options.update(dtype="auto", device_map={"": 0})

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                local_files_only=True,
                revision=revision,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                local_files_only=True,
                **load_options,
            )
        except OSError:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                revision=revision,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                **load_options,
            )

        if not prequantized:
            self.model.to(self.device)
        self.model.eval()

    def __call__(self, prompt: str) -> str:
        record = {
            "model": self.model_name,
            "revision": self.revision,
            "max_new_tokens": self.max_new_tokens,
            "seed": self.seed,
            "temperature": self.temperature,
            "prompt": prompt,
        }
        if self.prequantized:
            record["prequantized"] = True
        # Any generation-relevant change produces a different cached response.
        cache_key = hashlib.sha256(
            json.dumps(record, sort_keys=True).encode("utf-8")
        ).hexdigest()
        cache_path = self.cache_dir / f"{cache_key}.json"

        if cache_path.exists():
            self.cache_hits += 1
            self.last_cache_hit = True
            return json.loads(cache_path.read_text(encoding="utf-8"))["response"]

        inputs = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.device)
        generation_options = {
            "do_sample": self.temperature > 0,
            "max_new_tokens": self.max_new_tokens,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if self.temperature > 0:
            generation_options["temperature"] = self.temperature

        torch.manual_seed(self.seed)
        with torch.inference_mode():
            output = self.model.generate(**inputs, **generation_options)

        generated = output[0][inputs["input_ids"].shape[-1] :]
        response = self.tokenizer.decode(generated, skip_special_tokens=True).strip()

        self.model_calls += 1
        self.last_cache_hit = False
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        record["response"] = response
        temporary_path = cache_path.with_suffix(".tmp")
        temporary_path.write_text(
            json.dumps(record, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        temporary_path.replace(cache_path)
        return response
