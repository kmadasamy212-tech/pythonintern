"""
Unit tests for Paged KV Cache and block memory management.
"""

import pytest
import torch
from engine.config import KVCacheConfig, ModelConfig
from engine.model.kv_cache import KVCacheManager, BlockTable, PhysicalKVCachePool


def test_block_table_slots():
    bt = BlockTable("seq_1", block_size=16)
    bt.append_block(4)
    bt.append_block(9)

    b_id, off = bt.get_physical_slot(5)
    assert b_id == 4 and off == 5

    b_id, off = bt.get_physical_slot(16)
    assert b_id == 9 and off == 0

    b_id, off = bt.get_physical_slot(20)
    assert b_id == 9 and off == 4


def test_physical_kv_pool_roundtrip():
    num_blocks = 8
    block_size = 4
    num_kv_heads = 2
    head_dim = 16
    num_layers = 2

    pool = PhysicalKVCachePool(
        num_blocks=num_blocks,
        block_size=block_size,
        num_local_kv_heads=num_kv_heads,
        head_dim=head_dim,
        num_layers=num_layers,
    )

    # Store 3 tokens for seq into block 2
    k1 = torch.randn(num_kv_heads, head_dim)
    v1 = torch.randn(num_kv_heads, head_dim)
    k2 = torch.randn(num_kv_heads, head_dim)
    v2 = torch.randn(num_kv_heads, head_dim)

    pool.store_kv(layer_idx=0, physical_block_id=2, offset=0, k=k1, v=v1)
    pool.store_kv(layer_idx=0, physical_block_id=2, offset=1, k=k2, v=v2)

    # Read back
    k_seq, v_seq = pool.read_kv_sequence(layer_idx=0, block_ids=[2], seq_len=2)
    assert k_seq.shape == (2, num_kv_heads, head_dim)
    assert torch.equal(k_seq[0], k1)
    assert torch.equal(k_seq[1], k2)
    assert torch.equal(v_seq[0], v1)
    assert torch.equal(v_seq[1], v2)


def test_kv_cache_manager_lifecycle():
    kv_config = KVCacheConfig(block_size=16, num_blocks=10)
    model_config = ModelConfig(hidden_dim=64, num_heads=4, num_kv_heads=2, num_layers=2)

    mgr = KVCacheManager(kv_config, model_config, world_size=1)
    initial_free = mgr.free_block_count
    assert initial_free == 10

    # Allocate for seq with 30 tokens -> needs ceil(30/16) = 2 blocks
    table = mgr.allocate_blocks_for_sequence("req_1", num_tokens=30)
    assert len(table.physical_block_ids) == 2
    assert mgr.free_block_count == 8

    # Append tokens until crossing into 3rd block
    mgr.append_token("req_1") # 31
    assert len(table.physical_block_ids) == 2
    mgr.append_token("req_1") # 32 -> full
    mgr.append_token("req_1") # 33 -> triggers 3rd block
    assert len(table.physical_block_ids) == 3
    assert mgr.free_block_count == 7

    # Free sequence
    freed = mgr.free_sequence("req_1")
    assert freed == 3
    assert mgr.free_block_count == 10
