"""Multi-tokenizer worker handoff, used by Dynamo's gateway mode.

With ``--gateway-workers N`` Dynamo's leader writes the engine's port and
scheduler info to shared memory and spawns N children, each attaching its own
``TokenizerWorker`` to the leader's scheduler processes over IPC
(``dynamo/sglang/gateway.py``). The fake engine has no scheduler processes: the
whole simulator lives inside the one ``Engine`` object, so there is nothing for
a second process to attach to. These raise instead of pretending; run more
workers with the launch scripts to get more serving processes.
"""

from __future__ import annotations

from typing import Any

_UNSUPPORTED = (
    "fake-sglang does not support Dynamo's gateway mode (--gateway-workers > 1): "
    "the simulated engine runs in-process, with no scheduler processes for a "
    "TokenizerWorker to attach to. Launch more workers instead."
)


def write_data_for_multi_tokenizer(port_args: Any, server_args: Any, scheduler_info: Any):
    raise NotImplementedError(_UNSUPPORTED)


def read_from_shared_memory(name: str):
    raise NotImplementedError(_UNSUPPORTED)


def get_tokenizer_worker_class(server_args: Any):
    raise NotImplementedError(_UNSUPPORTED)
