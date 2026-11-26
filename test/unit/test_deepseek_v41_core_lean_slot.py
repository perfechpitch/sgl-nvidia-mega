# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Slot attn_core_lean: on by default, but only where it can run -- on SGLang's v41_new KV
layout (the one it reads) and with the attn, swapab_h8 and wo_a_g1 slots that feed it.
Missing any of those turns the core off instead of rejecting the request."""

import pytest

from sgl_nvidia_mega.deepseek_v41 import runtime

BASE = frozenset({"attn", "swapab_h8", "wo_a_g1", "attn_core_lean"})


def test_the_slot_is_on_by_default():
    assert "attn_core_lean" in runtime.DEFAULT_ON_SLOTS
    assert "attn_core_lean" in runtime.SHIPPED_SLOTS
    assert "attn_core_lean" not in runtime.OPT_IN_SLOTS
    assert "attn_core_lean" in runtime.NUMERICS_CHANGE_SLOTS
    assert "attn_core_staged" not in runtime.OPT_IN_SLOTS


@pytest.mark.parametrize("mode,on", [("v41_new", True), ("v4", False), ("v41", False)])
def test_the_slot_only_runs_on_the_v41_new_layout(monkeypatch, mode, on):
    """It reads the v41_new bytes directly: on any other layout it must turn itself off
    rather than read them as v41_new."""
    monkeypatch.setenv("SGLANG_DSV4_KV_LAYOUT", mode)
    assert runtime.kv_layout_v41_new() is on
    assert runtime._core_lean_on(BASE) is on


def test_the_slot_is_off_when_not_requested(monkeypatch):
    monkeypatch.setattr(runtime, "kv_layout_v41_new", lambda: True)
    assert runtime._core_lean_on(BASE - {"attn_core_lean"}) is False


def test_the_slot_needs_the_tp8_attention_slots(monkeypatch):
    monkeypatch.setattr(runtime, "kv_layout_v41_new", lambda: True)
    assert runtime._core_lean_on(BASE) is True
    for needs in ("attn", "swapab_h8", "wo_a_g1"):
        assert runtime._core_lean_on(BASE - {needs}) is False, needs
    # a missing prerequisite is not a rejected request
    monkeypatch.setattr(runtime, "SLOTS", BASE - {"swapab_h8"})
    assert runtime.slot_reasons() == []
