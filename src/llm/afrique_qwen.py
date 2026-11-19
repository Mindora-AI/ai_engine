import os
from pathlib import Path
from typing import List, Dict, Any, Optional

_DEFAULT_MAX_TOKENS = int(os.environ.get("MWIZERWA_LLM_MAX_TOKENS", "768"))

from src.runtime.model_paths import get_project_root, resolve_llm_gguf_path

PROJECT_ROOT = get_project_root()
DEFAULT_MODEL_PATH = resolve_llm_gguf_path()
_DEFAULT_N_GPU_LAYERS = int(os.environ.get("MWIZERWA_LLM_N_GPU_LAYERS", "0"))


# Structured Response JSON Schema for GBNF grammar-constrained generation.
# Enforces exact presence of response_text, interaction_intent, and requires_clarification.
STRUCTURED_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "response_text": {"type": "string"},
        "interaction_intent": {
            "type": "string",
            "enum": ["LISTENING", "REFLECTION", "PRACTICAL_ADVICE", "UNCLEAR"],
        },
        "requires_clarification": {"type": "boolean"},
    },
    "required": ["response_text", "interaction_intent", "requires_clarification"],
}


class AfriqueQwenLLM:
    """
    Wrapper for local AfriqueQwen3.5 GGUF model using llama-cpp-python.
    Ensures model is loaded once and reused across inference requests.
    Provides accurate token counting and chat prompt formatting.
    """

    _instance = None
    n_ctx: int = 4096
    max_tokens: int = _DEFAULT_MAX_TOKENS

    def __init__(
        self,
        model_path: Path = DEFAULT_MODEL_PATH,
        n_ctx: int = 4096,
        n_threads: int = 4,
        n_gpu_layers: Optional[int] = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = _DEFAULT_MAX_TOKENS,
    ):
        self.model_path = Path(model_path)
        self.n_ctx = n_ctx
        self.n_threads = n_threads
        self.n_gpu_layers = (
            _DEFAULT_N_GPU_LAYERS if n_gpu_layers is None else n_gpu_layers
        )
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self._llm = None
        self._formatter = None
        self.last_finish_reason: Optional[str] = None
        self.last_effective_max_tokens: Optional[int] = None

    def _resolve_chat_formatter(self):
        """
        Resolves the model's actual chat formatter (e.g. Jinja2ChatFormatter from GGUF metadata).
        This matches the exact chat template formatting applied by create_chat_completion.
        """
        if self._llm is None:
            return None

        handler = (
            self._llm.chat_handler
            or self._llm._chat_handlers.get(self._llm.chat_format)
        )
        if handler is None:
            try:
                from llama_cpp import llama_chat_format

                handler = llama_chat_format.get_chat_completion_handler(
                    self._llm.chat_format
                )
            except Exception:
                handler = None

        if handler is not None:
            try:
                import inspect

                cv = inspect.getclosurevars(handler)
                formatter = cv.nonlocals.get("chat_formatter")
                if formatter is not None:
                    return formatter
            except Exception:
                pass

        # Fallback to Jinja2ChatFormatter directly from GGUF metadata if available
        if (
            hasattr(self._llm, "metadata")
            and "tokenizer.chat_template" in self._llm.metadata
        ):
            try:
                from llama_cpp import llama_chat_format

                bos_token_id = self._llm.token_bos()
                eos_token_id = self._llm.token_eos()
                bos_token = (
                    self._llm._model.token_get_text(bos_token_id)
                    if bos_token_id != -1
                    else ""
                )
                eos_token = (
                    self._llm._model.token_get_text(eos_token_id)
                    if eos_token_id != -1
                    else ""
                )
                return llama_chat_format.Jinja2ChatFormatter(
                    template=self._llm.metadata["tokenizer.chat_template"],
                    eos_token=eos_token,
                    bos_token=bos_token,
                    stop_token_ids=[eos_token_id] if eos_token_id != -1 else None,
                )
            except Exception:
                pass

        return None

    def load_model(self):
        """Loads the GGUF model if not already loaded."""
        if self._llm is not None:
            return self._llm

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"AfriqueQwen model GGUF file not found at: {self.model_path.resolve()}"
            )

        try:
            from llama_cpp import Llama
        except ImportError as e:
            raise ImportError(
                "llama-cpp-python is required for AfriqueQwen inference. "
                "Please install llama-cpp-python."
            ) from e

        print(f"Loading AfriqueQwen model from: {self.model_path} ...")
        self._llm = Llama(
            model_path=str(self.model_path.resolve()),
            n_ctx=self.n_ctx,
            n_threads=self.n_threads,
            n_gpu_layers=self.n_gpu_layers,
            verbose=False,
        )
        self._formatter = self._resolve_chat_formatter()
        print("AfriqueQwen model loaded successfully!")
        return self._llm

    def format_prompt(self, messages: List[Dict[str, str]]) -> str:
        """
        Formats a list of chat messages into a model-compatible prompt string
        using the model's official chat template.
        """
        self.load_model()
        if self._formatter is None:
            self._formatter = self._resolve_chat_formatter()

        if self._formatter is not None:
            res = self._formatter(messages=messages)
            return res.prompt

        # Fallback to standard ChatML format if no template handler available
        prompt = ""
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            prompt += f"<|im_start|>{role}\n{content}<|im_end|>\n"
        prompt += "<|im_start|>assistant\n<think>\n"
        return prompt

    def count_tokens(self, messages: List[Dict[str, str]]) -> int:
        """
        Accurately counts the tokens of the complete formatted chat prompt,
        including chat template markers, special tokens, and generation prompt.
        Matches the exact tokenization performed by create_chat_completion.
        """
        llm = self.load_model()
        if self._formatter is None:
            self._formatter = self._resolve_chat_formatter()

        if self._formatter is not None:
            res = self._formatter(messages=messages)
            prompt_bytes = res.prompt.encode("utf-8")
            tokens = llm.tokenize(
                prompt_bytes,
                add_bos=not res.added_special,
                special=True,
            )
            return len(tokens)

        # Fallback using prompt string
        prompt = self.format_prompt(messages)
        tokens = llm.tokenize(prompt.encode("utf-8"), add_bos=False, special=True)
        return len(tokens)

    def count_text_tokens(self, text: str) -> int:
        """Counts raw tokens for an arbitrary text string without chat templating."""
        llm = self.load_model()
        return len(llm.tokenize(text.encode("utf-8"), add_bos=False, special=True))

    def _effective_max_tokens(self, messages: List[Dict[str, str]]) -> int:
        """Cap completion length so prompt + generation fits within n_ctx."""
        prompt_tokens = self.count_tokens(messages)
        slack = 8
        available = self.n_ctx - prompt_tokens - slack
        if available < 32:
            raise ValueError(
                f"Prompt uses {prompt_tokens} tokens; only {available} remain in n_ctx={self.n_ctx}. "
                "Shorten the conversation or reduce input length."
            )
        return min(self.max_tokens, available)

    def generate_completion(self, messages: List[Dict[str, str]]) -> str:
        """
        Generates a completion from a list of chat messages using the model's chat template,
        strictly constrained by the structured response contract schema via GBNF grammar.
        """
        llm = self.load_model()
        effective_max = self._effective_max_tokens(messages)
        self.last_effective_max_tokens = effective_max
        if effective_max < self.max_tokens:
            print(
                f"[LLM Info] Completion budget capped to {effective_max} tokens "
                f"(prompt uses {self.count_tokens(messages)} of n_ctx={self.n_ctx})."
            )

        response_fmt = {
            "type": "json_object",
            "schema": STRUCTURED_RESPONSE_SCHEMA,
        }
        completion_kwargs = dict(
            messages=messages,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=effective_max,
        )
        try:
            response = llm.create_chat_completion(
                **completion_kwargs,
                response_format=response_fmt,
            )
        except Exception as e:
            # Fallback to generic json_object if schema compilation is unsupported
            print(f"[LLM Warning] Schema grammar completion failed ({e}), falling back to generic json_object.")
            response = llm.create_chat_completion(
                **completion_kwargs,
                response_format={"type": "json_object"},
            )

        choices = response.get("choices", [])
        if not choices:
            self.last_finish_reason = None
            return ""

        choice = choices[0]
        self.last_finish_reason = choice.get("finish_reason")
        if self.last_finish_reason == "length":
            print(
                f"[LLM Warning] Generation hit the token limit ({effective_max}). "
                "The JSON contract may be truncated; partial recovery will be attempted."
            )

        content = choice.get("message", {}).get("content", "")
        return content or ""
