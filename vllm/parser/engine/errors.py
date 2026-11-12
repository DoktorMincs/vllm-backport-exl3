# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exceptions shared by parser-engine configurations and the engine itself."""

from __future__ import annotations


class MalformedToolArgs(ValueError):
    """A tool-call argument body that the configured converter rejected.

    Raised only when the converter is asked for the *final* value (never
    while streaming a partial one) and only when the caller opted into
    strict header checking.  ``reason`` is a short label suitable for a
    metric dimension.
    """

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason
