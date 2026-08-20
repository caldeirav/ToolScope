"""
Model wrappers for tool-calling inference.

Supports two backends:
  - HFModel: local HuggingFace causal LM (transformers)
  - OpenAIModel: remote OpenAI-compatible API (e.g. vLLM, TGI)

Both try native tool calling first; fall back to a JSON-in-system-prompt
approach for models whose templates lack tool support.
"""

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


_FALLBACK_SYSTEM = (
    "You are a function-calling assistant. The following tools are available:\n\n"
    "{tools_json}\n\n"
    "When you need to call a tool, respond ONLY with a JSON object inside "
    "<tool_call></tool_call> tags, like this:\n"
    "<tool_call>\n"
    '{{\"name\": \"tool_name\", \"arguments\": {{\"param\": \"value\"}}}}\n'
    "</tool_call>"
)


@dataclass
class ParsedToolCall:
    name: str
    arguments: Dict[str, Any] = field(default_factory=dict)


def _try_json(s: str) -> Optional[Dict]:
    s = s.strip()
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    try:
        obj = ast_eval(s)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    return None


def _extract_tool_call(raw: str) -> Optional[ParsedToolCall]:
    """
    Multi-strategy parser for model tool-call output.

    Strategies in order:
      1. <tool_call>...</tool_call> tags  (Qwen2.5, Hermes, etc.)
      2. ```json ... ``` code block
      3. Bracket-balanced JSON scan for an object with a "name" key
    """
    if not raw:
        return None

    # Strategy 1: <tool_call> tags
    m = re.search(r"<tool_call>\s*(.*?)\s*</tool_call>", raw, re.DOTALL)
    if m:
        obj = _try_json(m.group(1))
        if obj:
            return _dict_to_call(obj)

    # Strategy 2: JSON code block
    m = re.search(r"```(?:json)?\s*(.*?)\s*```", raw, re.DOTALL)
    if m:
        obj = _try_json(m.group(1))
        if obj and ("name" in obj or "tool" in obj):
            return _dict_to_call(obj)

    # Strategy 3: scan for balanced JSON object containing "name"
    for start in range(len(raw)):
        if raw[start] != "{":
            continue
        depth, end = 0, start
        for i in range(start, len(raw)):
            if raw[i] == "{":
                depth += 1
            elif raw[i] == "}":
                depth -= 1
                if depth == 0:
                    end = i
                    break
        candidate = raw[start : end + 1]
        obj = _try_json(candidate)
        if obj and ("name" in obj or "tool" in obj):
            return _dict_to_call(obj)

    return None


def _dict_to_call(obj: Dict) -> Optional[ParsedToolCall]:
    """Normalise varying key names used by different models."""
    name = obj.get("name") or obj.get("tool") or obj.get("function")
    args = obj.get("arguments") or obj.get("parameters") or obj.get("args") or {}
    if not isinstance(name, str) or not name:
        return None
    if not isinstance(args, dict):
        args = {}
    return ParsedToolCall(name=name, arguments=args)


def ast_eval(s: str) -> Any:
    import ast
    return ast.literal_eval(s)


# ── Models ────────────────────────────────────────────────────────────────────

class DummyModel:
    """
    Fake model for --dry-run.  Always picks the first tool in the list.
    No GPU, no network, deterministic.
    """

    def load(self) -> None:
        pass

    def predict(self, messages: List[Dict], tools: List[Dict]) -> str:
        if not tools:
            return ""
        name = tools[0]["function"]["name"]
        return f'<tool_call>{{"name": "{name}", "arguments": {{}}}}</tool_call>'

    def parse_tool_call(self, raw: str) -> Optional[ParsedToolCall]:
        return _extract_tool_call(raw)


class HFModel:
    """
    Wraps a HuggingFace causal LM for tool-calling evaluation.

    Lazy-loads the model on first predict() call.
    """

    _DTYPE_MAP = {
        "auto": "auto",
        "float16": None,   # resolved at load time
        "bfloat16": None,
        "float32": None,
    }

    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        dtype: str = "auto",
        max_new_tokens: int = 256,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.dtype = dtype
        self.max_new_tokens = max_new_tokens
        self._tokenizer = None
        self._model = None
        self._native_tools: Optional[bool] = None  # detected on first call

    def _load(self) -> None:
        if self._model is not None:
            return

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"Loading tokenizer: {self.model_name}")
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True
        )
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token_id = self._tokenizer.eos_token_id

        dtype_map = {
            "auto": "auto",
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(self.dtype, "auto")

        load_kwargs: Dict[str, Any] = {
            "torch_dtype": torch_dtype,
            "trust_remote_code": True,
            # Use PyTorch's Scaled Dot-Product Attention instead of the eager
            # (explicit) implementation.  On MPS (Apple Silicon) with PyTorch
            # 2.3+, SDPA dispatches to the Metal Flash Attention kernel, which
            # computes attention in O(n) memory rather than O(n²).  Without
            # this, a 12,500-token prompt with 14 heads requires a ~4.4 GiB
            # attention-score matrix per layer, causing MPS OOM at ~pool_size=100.
            "attn_implementation": "sdpa",
        }
        if self.device == "auto":
            load_kwargs["device_map"] = "auto"
        elif self.device in ("cuda", "mps", "cpu"):
            load_kwargs["device_map"] = self.device
        else:
            load_kwargs["device_map"] = self.device

        print(f"Loading model: {self.model_name}  "
              f"(device={self.device}, dtype={self.dtype}, attn=sdpa)")
        try:
            self._model = AutoModelForCausalLM.from_pretrained(self.model_name, **load_kwargs)
        except ValueError as exc:
            if "scaled_dot_product_attention" in str(exc) or "sdpa" in str(exc).lower():
                print(f"  Note: sdpa not supported by this model, retrying with eager attention.")
                load_kwargs["attn_implementation"] = "eager"
                self._model = AutoModelForCausalLM.from_pretrained(self.model_name, **load_kwargs)
            else:
                raise
        self._model.eval()

    def _detect_native_tools(self, tools: List[Dict]) -> bool:
        """Check whether the tokenizer's chat template accepts a tools kwarg."""
        try:
            self._tokenizer.apply_chat_template(
                [{"role": "user", "content": "test"}],
                tools=tools[:1],
                tokenize=False,
                add_generation_prompt=True,
            )
            return True
        except Exception:
            return False

    def _build_prompt_native(self, messages: List[Dict], tools: List[Dict]) -> Any:
        return self._tokenizer.apply_chat_template(
            messages,
            tools=tools,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )

    def _build_prompt_fallback(self, messages: List[Dict], tools: List[Dict]) -> Any:
        # Inject tools as a system message
        fn_defs = [t["function"] for t in tools]
        system_content = _FALLBACK_SYSTEM.format(tools_json=json.dumps(fn_defs, indent=2))
        full_messages = [{"role": "system", "content": system_content}] + list(messages)
        return self._tokenizer.apply_chat_template(
            full_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )

    def load(self) -> None:
        """Eagerly load model weights. Called automatically on first predict()."""
        self._load()

    @staticmethod
    def _unpack_tokenizer_output(raw: Any, device) -> tuple:
        """
        Extract (input_ids_tensor, attention_mask_or_None) from whatever
        apply_chat_template returns.  The return type varies across transformers
        versions:
          - old  : plain torch.Tensor
          - 4.40+: BatchEncoding({"input_ids": tensor, "attention_mask": tensor})
          - 4.43+: BatchEncoding({"input_ids": BatchEncoding(…), …})

        Walks up to 4 levels of nesting to find the actual tensor.
        """
        import torch

        def _dig(obj, key: str) -> Optional[Any]:
            for _ in range(4):
                if isinstance(obj, torch.Tensor):
                    return obj
                if not hasattr(obj, "__getitem__") or isinstance(obj, (str, bytes)):
                    return None
                try:
                    obj = obj[key]
                except (KeyError, IndexError, TypeError):
                    return None
            return obj if isinstance(obj, torch.Tensor) else None

        if isinstance(raw, torch.Tensor):
            ids = raw
        else:
            ids = _dig(raw, "input_ids")

        if ids is None:
            raise RuntimeError(
                f"Cannot extract input_ids from tokenizer output of type "
                f"{type(raw).__name__}"
            )

        ids = ids.to(device)
        if ids.dim() == 1:
            ids = ids.unsqueeze(0)

        mask = None
        if not isinstance(raw, torch.Tensor):
            m = _dig(raw, "attention_mask")
            if m is not None:
                mask = m.to(device)

        return ids, mask

    def predict(self, messages: List[Dict], tools: List[Dict]) -> str:
        import torch

        self._load()

        if self._native_tools is None:
            self._native_tools = self._detect_native_tools(tools)

        if self._native_tools:
            try:
                raw = self._build_prompt_native(messages, tools)
            except Exception:
                self._native_tools = False
                raw = self._build_prompt_fallback(messages, tools)
        else:
            raw = self._build_prompt_fallback(messages, tools)

        device = next(self._model.parameters()).device
        ids, mask = self._unpack_tokenizer_output(raw, device)

        generate_inputs: Dict[str, Any] = {"input_ids": ids}
        if mask is not None:
            generate_inputs["attention_mask"] = mask

        prompt_len = ids.shape[1]
        with torch.no_grad():
            output = self._model.generate(
                **generate_inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self._tokenizer.pad_token_id,
            )

        # Clone only the new tokens before deleting the full output tensor.
        # This releases the large MPS buffer back to the allocator immediately
        # rather than keeping it alive until the caller drops the reference.
        new_tokens = output[0][prompt_len:].clone()
        del output, ids, generate_inputs
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True)

    def parse_tool_call(self, raw: str) -> Optional[ParsedToolCall]:
        return _extract_tool_call(raw)


class OpenAIModel:
    """
    Wraps an OpenAI-compatible API (e.g. vLLM, TGI) for tool-calling evaluation.

    Tries native tool calling first (passing ``tools`` to the API).  If the
    endpoint returns an error, falls back to injecting tool definitions into
    a system prompt — the same strategy HFModel uses for models without
    template-level tool support.
    """

    def __init__(
        self,
        model_name: str,
        base_url: str = "http://localhost:8000/v1",
        api_key: str = "EMPTY",
        max_new_tokens: int = 256,
    ) -> None:
        self.model_name = model_name
        self.base_url = base_url
        self.api_key = api_key
        self.max_new_tokens = max_new_tokens
        self._client = None
        self._native_tools: Optional[bool] = None

    def load(self) -> None:
        from openai import OpenAI

        self._client = OpenAI(base_url=self.base_url, api_key=self.api_key)

        print(f"Connecting to OpenAI-compatible endpoint: {self.base_url}")
        print(f"  Model: {self.model_name}")
        try:
            self._client.models.list()
            print("  Connection verified.")
        except Exception as exc:
            print(f"  Warning: could not list models ({exc}). "
                  f"Inference may still work.")

    def _predict_native(self, messages: List[Dict], tools: List[Dict]) -> str:
        response = self._client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            tools=tools,
            max_tokens=self.max_new_tokens,
            temperature=0.0,
        )
        msg = response.choices[0].message
        if msg.tool_calls:
            tc = msg.tool_calls[0]
            args = tc.function.arguments
            return (
                f'<tool_call>{{"name": "{tc.function.name}", '
                f'"arguments": {args}}}</tool_call>'
            )
        return msg.content or ""

    def _predict_fallback(self, messages: List[Dict], tools: List[Dict]) -> str:
        fn_defs = [t["function"] for t in tools]
        system_content = _FALLBACK_SYSTEM.format(
            tools_json=json.dumps(fn_defs, indent=2)
        )
        full_messages = [{"role": "system", "content": system_content}] + list(messages)
        response = self._client.chat.completions.create(
            model=self.model_name,
            messages=full_messages,
            max_tokens=self.max_new_tokens,
            temperature=0.0,
        )
        return response.choices[0].message.content or ""

    def predict(self, messages: List[Dict], tools: List[Dict]) -> str:
        if self._client is None:
            self.load()

        if self._native_tools is None:
            try:
                result = self._predict_native(messages, tools)
                self._native_tools = True
                return result
            except Exception:
                self._native_tools = False
                return self._predict_fallback(messages, tools)

        if self._native_tools:
            try:
                return self._predict_native(messages, tools)
            except Exception:
                self._native_tools = False
                return self._predict_fallback(messages, tools)

        return self._predict_fallback(messages, tools)

    def parse_tool_call(self, raw: str) -> Optional[ParsedToolCall]:
        return _extract_tool_call(raw)
