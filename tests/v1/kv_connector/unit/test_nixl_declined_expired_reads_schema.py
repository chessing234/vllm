# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Rust NIXL stats must keep the Python declined-expired-read counter."""

from pathlib import Path

from vllm.distributed.kv_transfer.kv_connector.v1.nixl.stats import (
    NixlKVConnectorStats,
)

_REPO = Path(__file__).resolve().parents[4]


def test_rust_nixl_stats_record_declined_expired_reads():
    stats = NixlKVConnectorStats()
    assert "num_declined_expired_reads" in stats.data

    protocol = (
        _REPO / "rust/src/engine-core-client/src/protocol/stats.rs"
    ).read_text()
    metrics = (_REPO / "rust/src/engine-core-client/src/metrics.rs").read_text()
    scheduler = (_REPO / "rust/src/metrics/src/scheduler.rs").read_text()

    assert "pub num_declined_expired_reads:" in protocol
    assert "nixl_num_declined_expired_reads" in metrics
    assert "vllm:nixl_num_declined_expired_reads" in scheduler
    assert "stats.num_declined_expired_reads" in metrics
