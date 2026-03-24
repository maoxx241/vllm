# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

from vllm.utils.mem_utils import MemorySnapshot, memory_profiling


def test_memory_snapshot_measure_uses_torch_npu(monkeypatch):
    fake_device = SimpleNamespace(type="npu", index=0)

    monkeypatch.setattr("vllm.utils.mem_utils.torch.device", lambda device: device)
    monkeypatch.setattr(
        "vllm.utils.mem_utils.current_platform.mem_get_info",
        lambda device: (600, 1000),
    )

    peak_calls = []
    reserved_calls = []
    accelerator_calls = []

    monkeypatch.setattr(
        "vllm.utils.mem_utils.torch.npu.max_memory_allocated",
        lambda device: peak_calls.append(device) or 321,
    )
    monkeypatch.setattr(
        "vllm.utils.mem_utils.torch.npu.memory_reserved",
        lambda device=None: reserved_calls.append(device) or 123,
    )
    monkeypatch.setattr(
        "vllm.utils.mem_utils.torch.accelerator.memory_stats",
        lambda device: accelerator_calls.append(("stats", device)) or {},
    )
    monkeypatch.setattr(
        "vllm.utils.mem_utils.torch.accelerator.memory_reserved",
        lambda device: accelerator_calls.append(("reserved", device)) or 0,
    )

    snapshot = MemorySnapshot(device=fake_device, auto_measure=False)
    snapshot.measure()

    assert snapshot.torch_peak == 321
    assert snapshot.torch_memory == 123
    assert snapshot.free_memory == 600
    assert snapshot.total_memory == 1000
    assert snapshot.cuda_memory == 400
    assert snapshot.non_torch_memory == 277
    assert peak_calls == [fake_device]
    assert reserved_calls == [fake_device]
    assert accelerator_calls == []


def test_memory_profiling_uses_torch_npu_reset_and_empty(monkeypatch):
    fake_device = SimpleNamespace(type="npu", index=0)

    monkeypatch.setattr("vllm.utils.mem_utils.torch.device", lambda device: device)
    monkeypatch.setattr(
        "vllm.utils.mem_utils.current_platform.mem_get_info",
        lambda device: (700, 1000),
    )

    npu_calls = []
    accelerator_calls = []
    peak_values = iter([100, 180, 260])
    reserved_values = iter([50, 80])

    monkeypatch.setattr(
        "vllm.utils.mem_utils.torch.npu.reset_peak_memory_stats",
        lambda device: npu_calls.append(("reset_peak", device)),
    )
    monkeypatch.setattr(
        "vllm.utils.mem_utils.torch.npu.empty_cache",
        lambda: npu_calls.append(("empty_cache", None)),
    )
    monkeypatch.setattr(
        "vllm.utils.mem_utils.torch.npu.max_memory_allocated",
        lambda device: next(peak_values),
    )
    monkeypatch.setattr(
        "vllm.utils.mem_utils.torch.npu.memory_reserved",
        lambda device=None: next(reserved_values),
    )
    monkeypatch.setattr(
        "vllm.utils.mem_utils.torch.accelerator.empty_cache",
        lambda: accelerator_calls.append("empty_cache"),
    )
    monkeypatch.setattr(
        "vllm.utils.mem_utils.torch.accelerator.reset_peak_memory_stats",
        lambda device: accelerator_calls.append(("reset_peak", device)),
    )

    baseline_snapshot = MemorySnapshot(device=fake_device, auto_measure=False)
    baseline_snapshot.measure()

    with memory_profiling(
        baseline_snapshot=baseline_snapshot,
        weights_memory=20,
    ) as result:
        assert result.before_create is baseline_snapshot

    assert result.torch_peak_increase == 80
    assert result.non_torch_increase == 70
    assert result.non_kv_cache_memory == 170
    assert npu_calls == [
        ("reset_peak", fake_device),
        ("empty_cache", None),
        ("reset_peak", fake_device),
        ("empty_cache", None),
    ]
    assert accelerator_calls == []
