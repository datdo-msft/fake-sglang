"""Reasoning-parser registry: the token metadata, not the text parsing.

Dynamo reads ``ReasoningParser(model_type=...).detector`` for two things:
``think_excluded_tokens`` (``thinking_budget.py`` refuses a thinking budget
when the parser filters tokens) and ``think_end_token`` (``register.py``, only
when the engine owns a tokenizer -- never true here, the fake runs with
``--skip-tokenizer-init``). The fake engine emits noise tokens, so there is no
reasoning text to split; what has to match SGLang is the metadata, so the same
request is accepted or refused the same way it would be on a real worker.

Only the families whose tokens are mirrored below are known. Any other name
raises ``ValueError("Unsupported model type")``, exactly as SGLang does for a
name it lacks, which Dynamo turns into an ``InvalidArgument`` for the request.
"""

from __future__ import annotations

from typing import Dict, List, Optional


class BaseReasoningFormatDetector:
    def __init__(
        self,
        think_start_token: str,
        think_end_token: str,
        think_excluded_tokens: Optional[List[str]] = None,
        force_reasoning: bool = False,
        stream_reasoning: bool = True,
        tool_start_token: Optional[str] = None,
    ) -> None:
        self.think_start_token = think_start_token
        self.think_end_token = think_end_token
        self.think_excluded_tokens = think_excluded_tokens
        self.force_reasoning = force_reasoning
        self.stream_reasoning = stream_reasoning
        self.tool_start_token = tool_start_token


class Qwen3Detector(BaseReasoningFormatDetector):
    # Values from SGLang's Qwen3Detector.
    def __init__(self, stream_reasoning: bool = True, force_reasoning: bool = False) -> None:
        super().__init__(
            "<think>",
            "</think>",
            think_excluded_tokens=["<tool_call>", "</tool_call>", "<|im_end|>", "<|endoftext|>"],
            force_reasoning=force_reasoning,
            stream_reasoning=stream_reasoning,
            tool_start_token="<tool_call>",
        )


class DeepSeekR1Detector(BaseReasoningFormatDetector):
    # Values from SGLang's DeepSeekR1Detector: always reasoning, nothing excluded.
    def __init__(self, stream_reasoning: bool = True, force_reasoning: bool = True) -> None:
        super().__init__(
            "<think>",
            "</think>",
            force_reasoning=True,
            stream_reasoning=stream_reasoning,
        )


class GptOssDetector(BaseReasoningFormatDetector):
    # Values from SGLang's GptOssDetector: harmony channels, nothing excluded.
    def __init__(self, stream_reasoning: bool = True, force_reasoning: bool = True) -> None:
        super().__init__(
            "<|channel|>analysis<|message|>",
            "<|end|>",
            force_reasoning=force_reasoning,
            stream_reasoning=stream_reasoning,
        )


class KimiK3Detector(BaseReasoningFormatDetector):
    # Values from SGLang's KimiK3Detector (tokens from function_call/kimik3_format.py).
    def __init__(self, stream_reasoning: bool = True, force_reasoning: bool = True) -> None:
        super().__init__(
            "<|open|>think<|sep|>",
            "<|close|>think<|sep|>",
            think_excluded_tokens=["response", "message", "<|end_of_msg|>", "[EOS]", "[EOT]"],
            force_reasoning=force_reasoning,
            stream_reasoning=stream_reasoning,
            tool_start_token="<|open|>tools<|sep|>",
        )


class ReasoningParser:
    # The subset of SGLang's DetectorMap whose detectors are mirrored above.
    DetectorMap: Dict[str, type] = {
        "qwen3": Qwen3Detector,
        "qwen3-thinking": Qwen3Detector,
        "dots": Qwen3Detector,
        "minimax": Qwen3Detector,
        "nanbeige": Qwen3Detector,
        "interns1": Qwen3Detector,
        "deepseek-r1": DeepSeekR1Detector,
        "step3": DeepSeekR1Detector,
        "step3p5": DeepSeekR1Detector,
        "gigachat35": DeepSeekR1Detector,
        "gpt-oss": GptOssDetector,
        "kimi_k3": KimiK3Detector,
    }

    def __init__(
        self,
        model_type: Optional[str] = None,
        stream_reasoning: bool = True,
        force_reasoning: Optional[bool] = None,
        request=None,
        tokenizer=None,
        tool_call_parser_active: bool = False,
    ) -> None:
        if not model_type:
            raise ValueError("Model type must be specified")
        detector_class = self.DetectorMap.get(model_type.lower())
        if not detector_class:
            raise ValueError(f"Unsupported model type: {model_type}")
        if model_type.lower() in {"qwen3-thinking", "gpt-oss", "minimax"}:
            force_reasoning = True
        kwargs = {"stream_reasoning": stream_reasoning}
        if force_reasoning is not None:
            # None keeps each detector's own default, as in SGLang (GPT-OSS
            # and Kimi-K3 default to reasoning; Qwen3 does not).
            kwargs["force_reasoning"] = bool(force_reasoning)
        self.detector = detector_class(**kwargs)

    def parse_non_stream(self, full_text: str):
        raise NotImplementedError(
            "fake-sglang does not parse reasoning text: its output is noise tokens"
        )

    parse_stream_chunk = parse_non_stream
