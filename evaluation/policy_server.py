# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Serve policies from a model's Python environment using --factory module:function.

The factory returns a policy with reset(episode) and act(observation) methods.
Only numpy and websockets are required by this transport module.
"""

import argparse
import sys
from importlib import import_module
from pathlib import Path
from threading import Lock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.policies.base import Policy, validate_action  # noqa: E402
from evaluation.policies.hold import HoldPolicy  # noqa: E402
from evaluation.policies.random import RandomPolicy  # noqa: E402
from evaluation.protocol import decode, encode  # noqa: E402


def serve(policy: Policy | None = None, host="127.0.0.1", port=8000, *, policy_factory=None):
    """Serve separate policy sessions, or allow one connection to a supplied instance.

    Factories create independent episode state per connection. Callers may share
    immutable model weights between those instances. A supplied instance admits
    only one connection at a time to prevent concurrent resets corrupting history.
    """
    from websockets.sync.server import serve as websocket_serve

    if (policy is None) == (policy_factory is None):
        raise ValueError("Provide exactly one policy instance or policy_factory")
    shared_session = Lock()

    def handle(connection):
        if policy_factory is None and not shared_session.acquire(blocking=False):
            connection.close(code=1013, reason="This policy instance already serves another evaluator")
            return
        try:
            current = policy if policy_factory is None else policy_factory()
            action_dim = None
            for message in connection:
                try:
                    request = decode(message)
                    if request["type"] == "reset":
                        action_dim = None
                        episode = request["episode"]
                        if episode["action_dim"] not in (9, 10):
                            raise ValueError("Expected action_dim 9 or 10")
                        current.reset(episode)
                        action_dim = episode["action_dim"]
                        reply = {"ok": True}
                    elif request["type"] == "act":
                        if action_dim is None:
                            raise ValueError("Reset the policy before requesting actions")
                        reply = {"action": validate_action(current.act(request["observation"]), action_dim)}
                    else:
                        raise ValueError(f"Unknown request type: {request['type']!r}")
                except (ValueError, KeyError, RuntimeError, TypeError) as error:
                    reply = {"error": f"{type(error).__name__}: {error}"}
                connection.send(encode(reply))
        finally:
            if policy_factory is None:
                shared_session.release()

    with websocket_serve(handle, host, port, max_size=None) as server:
        print(f"Policy server listening on ws://{host}:{port}", flush=True)
        server.serve_forever()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    choice = parser.add_mutually_exclusive_group()
    choice.add_argument("--policy", choices=("hold", "random"))
    choice.add_argument("--factory", help="Importable module:function returning a configured policy")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    if args.factory:
        module, separator, name = args.factory.partition(":")
        if not separator or not module or not name:
            parser.error("--factory must be module:function")
        factory = getattr(import_module(module), name)
    else:
        factory = RandomPolicy if args.policy == "random" else HoldPolicy
    serve(host=args.host, port=args.port, policy_factory=factory)


if __name__ == "__main__":
    main()
