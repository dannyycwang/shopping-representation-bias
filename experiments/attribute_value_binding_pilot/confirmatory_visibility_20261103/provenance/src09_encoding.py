from __future__ import annotations

import gc
import hashlib
import json
import time
from pathlib import Path

import numpy as np


class DenseEncoder:
    def __init__(self, model_spec: dict, config: dict, cache_dir: Path):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.spec = model_spec
        self.config = config
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        torch.manual_seed(config["seed"])
        if self.device == "cuda":
            torch.cuda.manual_seed_all(config["seed"])
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_spec["name"], revision=model_spec["revision"]
        )
        kwargs = {}
        self.attention_implementation = "model_default"
        if model_spec["key"] == "gte_modernbert":
            self.attention_implementation = "sdpa"
            kwargs["attn_implementation"] = self.attention_implementation
        self.model = AutoModel.from_pretrained(
            model_spec["name"], revision=model_spec["revision"], **kwargs
        ).to(self.device).eval()
        if self.device == "cuda":
            self.model.half()
        self.hidden_size = int(self.model.config.hidden_size)
        self.base_batch_size = int(config["encoding"][f"batch_size_{model_spec['key']}"])
        print(
            f"loaded {model_spec['name']}@{model_spec['revision']} on {self.device}; "
            f"hidden={self.hidden_size}",
            flush=True,
        )

    def _pool(self, hidden, attention_mask, pooling: str):
        if pooling == "model_native":
            pooling = "mean" if "mean" in self.spec["native_pooling"].lower() else "cls"
        if pooling == "cls":
            return hidden[:, 0].float()
        mask = attention_mask.unsqueeze(-1)
        return (hidden.float() * mask).sum(1) / mask.sum(1).clamp(min=1)

    def _forward_native(self, texts: list[str], pooling: str, batch_size: int) -> np.ndarray:
        torch = self.torch
        try:
            batch = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.spec["max_tokens"],
                return_tensors="pt",
            )
            batch = {key: value.to(self.device) for key, value in batch.items()}
            with torch.inference_mode():
                output = self.model(**batch).last_hidden_state
                pooled = self._pool(output, batch["attention_mask"], pooling)
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            return pooled.cpu().numpy()
        except torch.cuda.OutOfMemoryError:
            if batch_size <= 1:
                raise
            torch.cuda.empty_cache()
            midpoint = len(texts) // 2
            left = self._forward_native(texts[:midpoint], pooling, max(1, batch_size // 2))
            right = self._forward_native(texts[midpoint:], pooling, max(1, batch_size // 2))
            return np.concatenate([left, right])

    def _encode_native(self, texts: list[str], pooling: str) -> tuple[np.ndarray, dict]:
        lengths = []
        started = time.time()
        for offset in range(0, len(texts), 256):
            block = texts[offset : offset + 256]
            lengths.extend(
                len(ids)
                for ids in self.tokenizer(
                    block, add_special_tokens=True, truncation=False, verbose=False
                )["input_ids"]
            )
        # Long-context padding can dominate runtime and memory. Sorting batches
        # by the (capped) token length changes only which independent examples
        # share padding; outputs are restored to catalog order before saving.
        processing_order = np.argsort(np.minimum(lengths, self.spec["max_tokens"]), kind="stable")
        array = np.empty((len(texts), self.hidden_size), dtype=np.float32)
        completed = 0
        capped_lengths = np.minimum(lengths, self.spec["max_tokens"])
        offset = 0
        while offset < len(texts):
            current_length = int(capped_lengths[processing_order[offset]])
            batch_size = self.base_batch_size
            if self.spec["key"] == "gte_modernbert":
                if current_length > 1500:
                    batch_size = 1
                elif current_length > 1024:
                    batch_size = min(batch_size, 2)
                elif current_length > 768:
                    batch_size = min(batch_size, 4)
                elif current_length > 500:
                    batch_size = min(batch_size, 8)
            indices = processing_order[offset : offset + batch_size]
            block = [texts[int(index)] for index in indices]
            array[indices] = self._forward_native(block, pooling, len(block))
            completed += len(block)
            offset += len(block)
            if completed == len(block) or completed % max(1024, self.base_batch_size * 32) < len(block):
                print(f"native {completed}/{len(texts)} {time.time()-started:.0f}s", flush=True)
        return array, {
            "mean_wordpieces_with_special": float(np.mean(lengths)),
            "p95_wordpieces_with_special": float(np.quantile(lengths, 0.95)),
            "fraction_truncated": float(np.mean(np.asarray(lengths) > self.spec["max_tokens"])),
        }

    def _encode_chunks(self, texts: list[str], profile: dict) -> tuple[np.ndarray, dict]:
        torch = self.torch
        chunk_tokens = int(profile["chunk_tokens"])
        overlap = int(profile["overlap"])
        step = chunk_tokens - overlap
        if step <= 0:
            raise ValueError("overlap must be smaller than chunk_tokens")
        result = np.zeros((len(texts), self.hidden_size), dtype=np.float32)
        denominator = np.zeros(len(texts), dtype=np.float32)
        lengths = []
        chunk_count = 0
        started = time.time()
        source_block = 128
        for source_offset in range(0, len(texts), source_block):
            block = texts[source_offset : source_offset + source_block]
            ids_list = self.tokenizer(
                block, add_special_tokens=False, truncation=False, verbose=False
            )["input_ids"]
            chunks, owners, weights = [], [], []
            for local_owner, ids in enumerate(ids_list):
                lengths.append(len(ids))
                starts = list(range(0, max(1, len(ids)), step))
                for start in starts:
                    part = ids[start : start + chunk_tokens]
                    if not part and chunks:
                        continue
                    chunks.append(self.tokenizer.build_inputs_with_special_tokens(part))
                    owners.append(source_offset + local_owner)
                    weights.append(max(1, len(part)))
                    if start + chunk_tokens >= len(ids):
                        break
            chunk_count += len(chunks)
            chunk_batch = self.base_batch_size
            for chunk_offset in range(0, len(chunks), chunk_batch):
                end = chunk_offset + chunk_batch
                batch_ids = chunks[chunk_offset:end]
                batch = self.tokenizer.pad({"input_ids": batch_ids}, padding=True, return_tensors="pt")
                batch = {key: value.to(self.device) for key, value in batch.items()}
                try:
                    with torch.inference_mode():
                        hidden = self.model(**batch).last_hidden_state
                        pooled = self._pool(hidden, batch["attention_mask"], profile["pooling"])
                    values = pooled.cpu().numpy()
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    values = self._forward_padded_recursive(batch_ids, profile["pooling"])
                owner_slice = np.asarray(owners[chunk_offset:end])
                weight_slice = np.asarray(weights[chunk_offset:end], dtype=np.float32)
                np.add.at(result, owner_slice, values * weight_slice[:, None])
                np.add.at(denominator, owner_slice, weight_slice)
            if source_offset % 2048 == 0:
                print(f"chunks {min(source_offset+len(block),len(texts))}/{len(texts)} {time.time()-started:.0f}s", flush=True)
        result /= np.maximum(denominator[:, None], 1)
        result /= np.maximum(np.linalg.norm(result, axis=1, keepdims=True), 1e-12)
        return result, {
            "mean_wordpieces_without_special": float(np.mean(lengths)),
            "p95_wordpieces_without_special": float(np.quantile(lengths, 0.95)),
            "mean_chunks": float(chunk_count / len(texts)),
            "fraction_multi_chunk": float(np.mean(np.asarray(lengths) > chunk_tokens)),
        }

    def _forward_padded_recursive(self, ids: list[list[int]], pooling: str) -> np.ndarray:
        torch = self.torch
        try:
            batch = self.tokenizer.pad({"input_ids": ids}, padding=True, return_tensors="pt")
            batch = {key: value.to(self.device) for key, value in batch.items()}
            with torch.inference_mode():
                hidden = self.model(**batch).last_hidden_state
                return self._pool(hidden, batch["attention_mask"], pooling).cpu().numpy()
        except torch.cuda.OutOfMemoryError:
            if len(ids) <= 1:
                raise
            torch.cuda.empty_cache()
            middle = len(ids) // 2
            return np.concatenate([
                self._forward_padded_recursive(ids[:middle], pooling),
                self._forward_padded_recursive(ids[middle:], pooling),
            ])

    def encode(self, texts: list[str], cache_name: str, profile: dict,
               source_sha256: str) -> np.ndarray:
        payload = {
            "model": self.spec,
            "profile": profile,
            "source_sha256": source_sha256,
            "seed": self.config["seed"],
            "implementation": "phase2-encoder-v3-length-bucket-sdpa",
            "attention_implementation": self.attention_implementation,
            "dynamic_batching": "GTE: base through 500 tokens, then 8/4/2/1 above 500/768/1024/1500; stable length buckets restored to source order",
            "precision": "fp16_model_fp32_pooling" if self.device == "cuda" else "fp32",
        }
        fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:20]
        path = self.cache_dir / f"{cache_name}_{fingerprint}.npy"
        meta_path = path.with_suffix(".json")
        if path.exists() and meta_path.exists():
            meta = json.loads(meta_path.read_text())
            if meta["fingerprint"] == fingerprint:
                print(f"cache hit {path.name}", flush=True)
                return np.load(path, mmap_mode="r")
        started = time.time()
        if profile["chunk_tokens"] is None:
            result, length_meta = self._encode_native(texts, profile["pooling"])
        else:
            result, length_meta = self._encode_chunks(texts, profile)
        if not np.isfinite(result).all():
            raise AssertionError("non-finite embedding")
        norms = np.linalg.norm(result, axis=1)
        if not np.allclose(norms, 1, atol=2e-3):
            raise AssertionError((float(norms.min()), float(norms.max())))
        np.save(path, result)
        meta_path.write_text(json.dumps({
            **payload,
            **length_meta,
            "fingerprint": fingerprint,
            "shape": list(result.shape),
            "seconds": time.time() - started,
            "tokenizer_class": type(self.tokenizer).__name__,
            "model_class": type(self.model).__name__,
            "device": self.device,
        }, indent=2))
        return np.load(path, mmap_mode="r")

    def close(self):
        del self.model
        gc.collect()
        if self.device == "cuda":
            self.torch.cuda.empty_cache()
