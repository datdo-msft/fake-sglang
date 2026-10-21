"""Keep PyTorch out of the worker process.

The fake engine never touches a tensor, but Dynamo's SGLang component imports
``torch`` at module scope (multimodal handlers, the embedding cache, ...), and
``transformers`` -- seeing torch installed -- drags in its modeling stack on
``from transformers import AutoTokenizer``. Together that is ~170 MB of the
worker's ~280 MB footprint, which caps how many workers fit on a laptop.

So, before either is imported:

1. ``transformers`` is told torch is absent, which it supports natively
   (tokenizers, configs and hub utilities only). Its answer is cached.
2. ``torch`` and ``torch.*`` resolve to a stub. Attribute access yields inert
   placeholders, enough for annotations like ``tensor: torch.Tensor``; *calling*
   or instantiating anything raises, so a code path that really needs torch
   fails loudly instead of computing with junk.

Only worker processes (``python -m dynamo.sglang``) are affected; anything
else importing ``sglang`` -- pytest, Dynamo's own unit tests, which exercise
real torch code -- gets real torch. Set ``FAKE_SGLANG_REAL_TORCH=1`` to opt a
worker out too.
"""

import importlib.abc
import importlib.machinery
import os
import sys
import types

_HINT = "torch is stubbed out by fake-sglang; set FAKE_SGLANG_REAL_TORCH=1 to load the real one"


class _PlaceholderMeta(type):
    def __getattr__(cls, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return _placeholder(f"{cls.__qualname__}.{name}")

    def __call__(cls, *args, **kwargs):
        raise RuntimeError(f"{cls.__qualname__}(): {_HINT}")

    def __getitem__(cls, item):           # torch.Tensor[...] in annotations
        return cls

    def __or__(cls, other):                # torch.Tensor | None
        return cls

    __ror__ = __or__


def _placeholder(qualname):
    return _PlaceholderMeta(qualname.rsplit(".", 1)[-1], (), {"__qualname__": qualname})


class _StubModule(types.ModuleType):
    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        value = _placeholder(f"{self.__name__}.{name}")
        setattr(self, name, value)
        return value


class _StubLoader(importlib.abc.Loader):
    def create_module(self, spec):
        return _StubModule(spec.name)

    def exec_module(self, module):
        module.__path__ = []
        module.__is_fake_torch__ = True


class _StubFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name == "torch" or name.startswith("torch."):
            return importlib.machinery.ModuleSpec(name, _StubLoader(), is_package=True)
        return None


def _is_worker():
    spec = getattr(sys.modules.get("__main__"), "__spec__", None)
    return spec is not None and spec.name == "dynamo.sglang.__main__"


def install():
    if os.environ.get("FAKE_SGLANG_REAL_TORCH", "") not in ("", "0"):
        return
    if not _is_worker():
        return
    if "torch" in sys.modules or "transformers" in sys.modules:
        return   # too late to swap cleanly; leave the process as it is

    sys.modules["torch"] = None            # find_spec("torch") -> None
    try:
        from transformers.utils import import_utils
        import_utils.is_torch_available()  # lru_cached: False from here on
    except ImportError:
        pass
    finally:
        del sys.modules["torch"]
    sys.meta_path.insert(0, _StubFinder())
