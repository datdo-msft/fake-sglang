"""Scheduler batch types read by Dynamo."""

from enum import Enum


class Modality(Enum):
    IMAGE = "image"
    MULTI_IMAGES = "multi-images"
    VIDEO = "video"
    AUDIO = "audio"


class MultimodalProcessorOutput:
    """Dynamo pads multimodal input ids with this, but only when the tokenizer
    manager has ``_tokenize_one_request`` (``_compat.py``). The fake's does
    not, so the path is unreachable; multimodal is not simulated."""

    @staticmethod
    def build_padded_input_ids(input_ids, mm_items):
        raise NotImplementedError("fake-sglang does not simulate multimodal inputs")
