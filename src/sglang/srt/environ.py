"""SGLang's env-var registry. Dynamo reads ``envs`` for the snapshot path,
warmup and the thinking budget."""

import os
from typing import Any, Callable, Dict, Tuple


class _EnvField:
    def __init__(self, name: str, default: Any = None, parse: Callable[[str], Any] = str) -> None:
        self.name = name
        self.default = default
        self.parse = parse

    def get(self) -> Any:
        raw = os.environ.get(self.name)
        return self.default if raw is None else self.parse(raw)

    def set(self, value: Any) -> None:
        os.environ[self.name] = str(value)

    def is_set(self) -> bool:
        return self.name in os.environ


# Fields Dynamo compares or does arithmetic on, with SGLang's own type and
# default. An untyped field would hand back the raw string, and
# ``envs.SGLANG_MAX_THINK_TOKENS.get() >= 0`` raises TypeError on a str.
_TYPED: Dict[str, Tuple[Callable[[str], Any], Any]] = {
    "SGLANG_MAX_THINK_TOKENS": (int, -1),
    "SGLANG_WARMUP_TIMEOUT": (float, -1.0),
}


class _Envs:
    """Attribute access mints a field on demand, matching SGLang's registry
    behavior closely enough that Dynamo's ``getattr(envs, name, None)`` probes
    resolve. Fields in ``_TYPED`` parse like SGLang's EnvInt/EnvFloat; the rest
    return strings."""

    def __getattr__(self, name: str) -> _EnvField:
        env_name = name if name.startswith("SGLANG_") else f"SGLANG_{name.upper()}"
        parse, default = _TYPED.get(env_name, (str, None))
        field = _EnvField(env_name, default, parse)
        setattr(self, name, field)
        return field


envs = _Envs()
