"""Workers run on a torch stub; every other importer of sglang gets real torch.

Each case runs in a fresh interpreter: the stub has to be installed before
torch or transformers is first imported, which this pytest process is long past.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import pytest

AS_WORKER = """
import importlib.machinery, sys
sys.modules["__main__"].__spec__ = importlib.machinery.ModuleSpec("dynamo.sglang.__main__", None)
"""


def _run(code, as_worker, **env):
    prelude = AS_WORKER if as_worker else ""
    proc = subprocess.run(
        [sys.executable, "-c", prelude + textwrap.dedent(code)],
        capture_output=True, text=True, env={**os.environ, **env}, timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout.strip()


def test_worker_gets_stub_and_torchless_transformers():
    out = _run("""
        import sglang, torch
        from transformers.utils.import_utils import is_torch_available
        from transformers import AutoTokenizer            # tokenizers still work
        assert torch.__is_fake_torch__ and not is_torch_available()
        x: torch.Tensor | None = None                     # annotations resolve
        try:
            torch.zeros(1)
        except RuntimeError as e:
            print(e)
        print(sorted(m for m in sys.modules if m.startswith("torch")))
    """, as_worker=True)
    assert "FAKE_SGLANG_REAL_TORCH=1" in out
    assert out.splitlines()[-1] == "['torch']"


@pytest.mark.parametrize("as_worker,env", [
    (False, {}),
    (True, {"FAKE_SGLANG_REAL_TORCH": "1"}),
])
def test_real_torch_otherwise(as_worker, env):
    out = _run("""
        import sglang, torch
        print(hasattr(torch, "__is_fake_torch__"), torch.zeros(2).shape[0])
    """, as_worker=as_worker, **env)
    assert out == "False 2"
