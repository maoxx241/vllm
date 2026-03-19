# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Backend for GatedDeltaNet attention."""

from dataclasses import dataclass

import torch

from vllm.config import VllmConfig
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
)
from vllm.v1.attention.backends.utils import (
    PAD_SLOT_ID,
    compute_causal_conv1d_metadata,
    mamba_get_block_table_tensor,
    split_decodes_and_prefills,
)
from vllm.v1.kv_cache_interface import AttentionSpec, MambaSpec


class GDNAttentionBackend(AttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "GDN_ATTN"

    @staticmethod
    def get_builder_cls() -> type["GDNAttentionMetadataBuilder"]:
        return GDNAttentionMetadataBuilder


@dataclass
class GDNChunkedPrefillMetadata:
    chunk_indices_64: torch.Tensor
    chunk_offsets_64: torch.Tensor
    update_chunk_offsets_64: torch.Tensor
    final_chunk_indices_64: torch.Tensor
    chunk_indices_large_block: torch.Tensor
    block_indices_cumsum: torch.Tensor
    _buffer_slot: object | None = None


@dataclass
class _GDNChunkedPrefillBufferSlot:
    chunk_indices_64_cpu: torch.Tensor
    chunk_indices_64_device: torch.Tensor
    chunk_offsets_64_cpu: torch.Tensor
    chunk_offsets_64_device: torch.Tensor
    update_chunk_offsets_64_cpu: torch.Tensor
    update_chunk_offsets_64_device: torch.Tensor
    final_chunk_indices_64_cpu: torch.Tensor
    final_chunk_indices_64_device: torch.Tensor
    chunk_indices_large_block_cpu: torch.Tensor
    chunk_indices_large_block_device: torch.Tensor
    block_indices_cumsum_cpu: torch.Tensor
    block_indices_cumsum_device: torch.Tensor


def _next_power_of_2(value: int) -> int:
    if value <= 1:
        return 1
    return 1 << (value - 1).bit_length()


def _prepare_chunk_counts_cpu(
    cu_seqlens_cpu: torch.Tensor, chunk_size: int
) -> torch.Tensor:
    lens = cu_seqlens_cpu[1:] - cu_seqlens_cpu[:-1]
    return torch.div(lens + chunk_size - 1, chunk_size, rounding_mode="floor")


def _fill_chunk_indices_cpu(
    out: torch.Tensor, chunk_counts: torch.Tensor
) -> int:
    cursor = 0
    for seq_idx, num_chunks in enumerate(chunk_counts.tolist()):
        if num_chunks <= 0:
            continue
        out[cursor : cursor + num_chunks, 0].fill_(seq_idx)
        out[cursor : cursor + num_chunks, 1] = torch.arange(
            num_chunks,
            dtype=out.dtype,
        )
        cursor += num_chunks
    return cursor


def _fill_chunk_offsets_cpu(out: torch.Tensor, chunk_counts: torch.Tensor) -> int:
    out[0] = 0
    if chunk_counts.numel() > 0:
        torch.cumsum(chunk_counts, dim=0, out=out[1 : chunk_counts.numel() + 1])
    return chunk_counts.numel() + 1


def _fill_update_chunk_offsets_cpu(
    out: torch.Tensor, chunk_counts: torch.Tensor
) -> int:
    out[0] = 0
    if chunk_counts.numel() > 0:
        torch.cumsum(
            chunk_counts + 1,
            dim=0,
            out=out[1 : chunk_counts.numel() + 1],
        )
    return chunk_counts.numel() + 1


def _fill_final_chunk_indices_cpu(out: torch.Tensor, chunk_counts: torch.Tensor) -> int:
    if chunk_counts.numel() > 0:
        torch.cumsum(chunk_counts + 1, dim=0, out=out[: chunk_counts.numel()])
        out[: chunk_counts.numel()].sub_(1)
    return chunk_counts.numel()


@dataclass
class GDNAttentionMetadata:
    num_prefills: int
    num_prefill_tokens: int
    num_decodes: int
    num_decode_tokens: int
    num_spec_decodes: int
    num_spec_decode_tokens: int
    num_actual_tokens: int

    has_initial_state: torch.Tensor | None = None

    spec_query_start_loc: torch.Tensor | None = None  # shape: [num_spec_decodes + 1,]
    non_spec_query_start_loc: torch.Tensor | None = (
        None  # shape: [batch - num_spec_decodes + 1,]
    )

    spec_state_indices_tensor: torch.Tensor | None = None  # shape: [batch, num_spec]
    non_spec_state_indices_tensor: torch.Tensor | None = (
        None  # shape: [batch - num_spec_decodes,]
    )
    spec_sequence_masks: torch.Tensor | None = None  # shape: [batch,]
    spec_token_indx: torch.Tensor | None = None
    non_spec_token_indx: torch.Tensor | None = None

    num_accepted_tokens: torch.Tensor | None = None  # shape: [batch,]

    # The following attributes are for triton implementation of causal_conv1d
    nums_dict: dict | None = None
    batch_ptr: torch.Tensor | None = None
    token_chunk_offset_ptr: torch.Tensor | None = None
    non_spec_chunked_prefill_meta: GDNChunkedPrefillMetadata | None = None


class GDNAttentionMetadataBuilder(AttentionMetadataBuilder[GDNAttentionMetadata]):
    _cudagraph_support = AttentionCGSupport.UNIFORM_BATCH

    reorder_batch_threshold: int = 1

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        assert isinstance(kv_cache_spec, MambaSpec)
        self.vllm_config = vllm_config
        self.compilation_config = vllm_config.compilation_config
        self.speculative_config = vllm_config.speculative_config
        self.kv_cache_spec = kv_cache_spec
        self.device = device
        self.max_num_batched_tokens = (
            self.vllm_config.scheduler_config.max_num_batched_tokens
        )
        self.max_num_seqs = self.vllm_config.scheduler_config.max_num_seqs
        self.chunk_size = 64
        self.large_block_size = 608 * 2
        hf_text_config = getattr(self.vllm_config.model_config, "hf_text_config", None)
        if hf_text_config is not None and hasattr(hf_text_config, "linear_num_value_heads"):
            self.gdn_num_heads = (
                hf_text_config.linear_num_value_heads
                // self.vllm_config.parallel_config.tensor_parallel_size
            )
        else:
            self.gdn_num_heads = self.vllm_config.model_config.get_num_attention_heads(
                self.vllm_config.parallel_config
            )
        cumsum_chunks = max(1, (2**18) // (self.gdn_num_heads * self.chunk_size))
        self.cumsum_block_size = _next_power_of_2(cumsum_chunks)
        self._chunked_prefill_pool: list[_GDNChunkedPrefillBufferSlot] = []
        self._chunked_prefill_pool_idx = -1

        if self.speculative_config:
            assert self.speculative_config.num_speculative_tokens is not None
            self.num_spec: int = self.speculative_config.num_speculative_tokens
        else:
            self.num_spec = 0
        self.use_spec_decode = self.num_spec > 0
        self._init_reorder_batch_threshold(1, self.use_spec_decode)

        self.use_full_cuda_graph = (
            self.compilation_config.cudagraph_mode.has_full_cudagraphs()
        )

        self.decode_cudagraph_max_bs = (
            self.vllm_config.scheduler_config.max_num_seqs * (self.num_spec + 1)
        )
        if self.compilation_config.max_cudagraph_capture_size is not None:
            self.decode_cudagraph_max_bs = min(
                self.decode_cudagraph_max_bs,
                self.compilation_config.max_cudagraph_capture_size,
            )

        self.spec_state_indices_tensor = torch.empty(
            (self.decode_cudagraph_max_bs, self.num_spec + 1),
            dtype=torch.int32,
            device=device,
        )
        self.non_spec_state_indices_tensor = torch.empty(
            (self.decode_cudagraph_max_bs,),
            dtype=torch.int32,
            device=device,
        )
        self.spec_sequence_masks = torch.empty(
            (self.decode_cudagraph_max_bs,),
            dtype=torch.bool,
            device=device,
        )
        self.spec_token_indx = torch.empty(
            (self.decode_cudagraph_max_bs * (self.num_spec + 1),),
            dtype=torch.int32,
            device=device,
        )
        self.non_spec_token_indx = torch.empty(
            (self.decode_cudagraph_max_bs * (self.num_spec + 1),),
            dtype=torch.int32,
            device=device,
        )
        self.spec_query_start_loc = torch.empty(
            (self.decode_cudagraph_max_bs + 1,),
            dtype=torch.int32,
            device=device,
        )
        self.non_spec_query_start_loc = torch.empty(
            (self.decode_cudagraph_max_bs + 1,),
            dtype=torch.int32,
            device=device,
        )
        self.num_accepted_tokens = torch.empty(
            (self.decode_cudagraph_max_bs,),
            dtype=torch.int32,
            device=device,
        )

        if device.type != "cpu":
            # Two slots are enough for async scheduling overlap: one batch can
            # execute on device while the next batch is prepared on CPU/H2D.
            self._chunked_prefill_pool = [
                self._allocate_chunked_prefill_slot(device),
                self._allocate_chunked_prefill_slot(device),
            ]

    def _allocate_chunked_prefill_slot(
        self, device: torch.device
    ) -> _GDNChunkedPrefillBufferSlot:
        cpu_kwargs = {
            "dtype": torch.int32,
            "device": "cpu",
            "pin_memory": True,
        }
        device_kwargs = {
            "dtype": torch.int32,
            "device": device,
        }
        return _GDNChunkedPrefillBufferSlot(
            chunk_indices_64_cpu=torch.empty(
                (self.max_num_batched_tokens, 2), **cpu_kwargs
            ),
            chunk_indices_64_device=torch.empty(
                (self.max_num_batched_tokens, 2), **device_kwargs
            ),
            chunk_offsets_64_cpu=torch.empty((self.max_num_seqs + 1,), **cpu_kwargs),
            chunk_offsets_64_device=torch.empty(
                (self.max_num_seqs + 1,), **device_kwargs
            ),
            update_chunk_offsets_64_cpu=torch.empty(
                (self.max_num_seqs + 1,), **cpu_kwargs
            ),
            update_chunk_offsets_64_device=torch.empty(
                (self.max_num_seqs + 1,), **device_kwargs
            ),
            final_chunk_indices_64_cpu=torch.empty((self.max_num_seqs,), **cpu_kwargs),
            final_chunk_indices_64_device=torch.empty(
                (self.max_num_seqs,), **device_kwargs
            ),
            chunk_indices_large_block_cpu=torch.empty(
                (self.max_num_batched_tokens, 2), **cpu_kwargs
            ),
            chunk_indices_large_block_device=torch.empty(
                (self.max_num_batched_tokens, 2), **device_kwargs
            ),
            block_indices_cumsum_cpu=torch.empty(
                (self.max_num_batched_tokens, 2), **cpu_kwargs
            ),
            block_indices_cumsum_device=torch.empty(
                (self.max_num_batched_tokens, 2), **device_kwargs
            ),
        )

    def _build_non_spec_chunked_prefill_meta_cpu(
        self, cu_seqlens_cpu: torch.Tensor
    ) -> GDNChunkedPrefillMetadata:
        chunk_counts_64 = _prepare_chunk_counts_cpu(cu_seqlens_cpu, self.chunk_size)
        chunk_counts_large = _prepare_chunk_counts_cpu(
            cu_seqlens_cpu, self.large_block_size
        )
        chunk_counts_cumsum = _prepare_chunk_counts_cpu(
            cu_seqlens_cpu, self.cumsum_block_size
        )
        num_seqs = chunk_counts_64.numel()
        chunk_indices_64 = torch.empty(
            (int(chunk_counts_64.sum().item()), 2), dtype=torch.int32
        )
        chunk_offsets_64 = torch.empty((num_seqs + 1,), dtype=torch.int32)
        update_chunk_offsets_64 = torch.empty((num_seqs + 1,), dtype=torch.int32)
        final_chunk_indices_64 = torch.empty((num_seqs,), dtype=torch.int32)
        chunk_indices_large_block = torch.empty(
            (int(chunk_counts_large.sum().item()), 2), dtype=torch.int32
        )
        block_indices_cumsum = torch.empty(
            (int(chunk_counts_cumsum.sum().item()), 2), dtype=torch.int32
        )

        _fill_chunk_indices_cpu(chunk_indices_64, chunk_counts_64)
        _fill_chunk_offsets_cpu(chunk_offsets_64, chunk_counts_64)
        _fill_update_chunk_offsets_cpu(update_chunk_offsets_64, chunk_counts_64)
        _fill_final_chunk_indices_cpu(final_chunk_indices_64, chunk_counts_64)
        _fill_chunk_indices_cpu(chunk_indices_large_block, chunk_counts_large)
        _fill_chunk_indices_cpu(block_indices_cumsum, chunk_counts_cumsum)

        return GDNChunkedPrefillMetadata(
            chunk_indices_64=chunk_indices_64.to(self.device),
            chunk_offsets_64=chunk_offsets_64.to(self.device),
            update_chunk_offsets_64=update_chunk_offsets_64.to(self.device),
            final_chunk_indices_64=final_chunk_indices_64.to(self.device),
            chunk_indices_large_block=chunk_indices_large_block.to(self.device),
            block_indices_cumsum=block_indices_cumsum.to(self.device),
        )

    def _build_non_spec_chunked_prefill_meta(
        self, cu_seqlens_cpu: torch.Tensor
    ) -> GDNChunkedPrefillMetadata:
        if self.device.type == "cpu":
            return self._build_non_spec_chunked_prefill_meta_cpu(cu_seqlens_cpu)

        self._chunked_prefill_pool_idx = (
            self._chunked_prefill_pool_idx + 1
        ) % len(self._chunked_prefill_pool)
        slot = self._chunked_prefill_pool[self._chunked_prefill_pool_idx]
        chunk_counts_64 = _prepare_chunk_counts_cpu(cu_seqlens_cpu, self.chunk_size)
        chunk_counts_large = _prepare_chunk_counts_cpu(
            cu_seqlens_cpu, self.large_block_size
        )
        chunk_counts_cumsum = _prepare_chunk_counts_cpu(
            cu_seqlens_cpu, self.cumsum_block_size
        )
        num_chunk_indices_64 = _fill_chunk_indices_cpu(
            slot.chunk_indices_64_cpu, chunk_counts_64
        )
        num_chunk_offsets_64 = _fill_chunk_offsets_cpu(
            slot.chunk_offsets_64_cpu, chunk_counts_64
        )
        num_update_chunk_offsets_64 = _fill_update_chunk_offsets_cpu(
            slot.update_chunk_offsets_64_cpu, chunk_counts_64
        )
        num_final_chunk_indices_64 = _fill_final_chunk_indices_cpu(
            slot.final_chunk_indices_64_cpu, chunk_counts_64
        )
        num_chunk_indices_large = _fill_chunk_indices_cpu(
            slot.chunk_indices_large_block_cpu, chunk_counts_large
        )
        num_block_indices_cumsum = _fill_chunk_indices_cpu(
            slot.block_indices_cumsum_cpu, chunk_counts_cumsum
        )

        chunk_indices_64 = slot.chunk_indices_64_device[:num_chunk_indices_64]
        chunk_indices_64.copy_(
            slot.chunk_indices_64_cpu[:num_chunk_indices_64],
            non_blocking=True,
        )
        chunk_offsets_64 = slot.chunk_offsets_64_device[:num_chunk_offsets_64]
        chunk_offsets_64.copy_(
            slot.chunk_offsets_64_cpu[:num_chunk_offsets_64],
            non_blocking=True,
        )
        update_chunk_offsets_64 = slot.update_chunk_offsets_64_device[
            :num_update_chunk_offsets_64
        ]
        update_chunk_offsets_64.copy_(
            slot.update_chunk_offsets_64_cpu[:num_update_chunk_offsets_64],
            non_blocking=True,
        )
        final_chunk_indices_64 = slot.final_chunk_indices_64_device[
            :num_final_chunk_indices_64
        ]
        final_chunk_indices_64.copy_(
            slot.final_chunk_indices_64_cpu[:num_final_chunk_indices_64],
            non_blocking=True,
        )
        chunk_indices_large_block = slot.chunk_indices_large_block_device[
            :num_chunk_indices_large
        ]
        chunk_indices_large_block.copy_(
            slot.chunk_indices_large_block_cpu[:num_chunk_indices_large],
            non_blocking=True,
        )
        block_indices_cumsum = slot.block_indices_cumsum_device[
            :num_block_indices_cumsum
        ]
        block_indices_cumsum.copy_(
            slot.block_indices_cumsum_cpu[:num_block_indices_cumsum],
            non_blocking=True,
        )
        return GDNChunkedPrefillMetadata(
            chunk_indices_64=chunk_indices_64,
            chunk_offsets_64=chunk_offsets_64,
            update_chunk_offsets_64=update_chunk_offsets_64,
            final_chunk_indices_64=final_chunk_indices_64,
            chunk_indices_large_block=chunk_indices_large_block,
            block_indices_cumsum=block_indices_cumsum,
            _buffer_slot=slot,
        )

    def build(  # type: ignore[override]
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        num_accepted_tokens: torch.Tensor | None = None,
        num_decode_draft_tokens_cpu: torch.Tensor | None = None,
        fast_build: bool = False,
    ) -> GDNAttentionMetadata:
        m = common_attn_metadata

        query_start_loc = m.query_start_loc
        query_start_loc_cpu = m.query_start_loc_cpu
        context_lens_tensor = m.compute_num_computed_tokens()
        nums_dict, batch_ptr, token_chunk_offset_ptr = None, None, None
        block_table_tensor = mamba_get_block_table_tensor(
            m.block_table_tensor,
            m.seq_lens,
            self.kv_cache_spec,
            self.vllm_config.cache_config.mamba_cache_mode,
        )

        spec_sequence_masks_cpu: torch.Tensor | None = None
        if (
            not self.use_spec_decode
            or num_decode_draft_tokens_cpu is None
            or num_decode_draft_tokens_cpu[num_decode_draft_tokens_cpu >= 0]
            .sum()
            .item()
            == 0
        ):
            spec_sequence_masks = None
            num_spec_decodes = 0
        else:
            spec_sequence_masks_cpu = num_decode_draft_tokens_cpu >= 0
            num_spec_decodes = spec_sequence_masks_cpu.sum().item()
            if num_spec_decodes == 0:
                spec_sequence_masks = None
                spec_sequence_masks_cpu = None
            else:
                spec_sequence_masks = spec_sequence_masks_cpu.to(
                    query_start_loc.device, non_blocking=True
                )

        if spec_sequence_masks is None:
            num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
                split_decodes_and_prefills(m, decode_threshold=1)
            )
            num_spec_decode_tokens = 0
            spec_token_indx = None
            non_spec_token_indx = None
            spec_state_indices_tensor = None
            non_spec_state_indices_tensor = block_table_tensor[:, 0]
            spec_query_start_loc = None
            non_spec_query_start_loc = query_start_loc
            non_spec_query_start_loc_cpu = query_start_loc_cpu
            num_accepted_tokens = None
        else:
            query_lens = query_start_loc[1:] - query_start_loc[:-1]
            assert spec_sequence_masks_cpu is not None
            query_lens_cpu = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]

            # Use CPU tensors to avoid CPU-GPU sync
            non_spec_query_lens_cpu = query_lens_cpu[~spec_sequence_masks_cpu]
            num_decodes = (non_spec_query_lens_cpu == 1).sum().item()
            # Exclude zero-length padded sequences from prefill count.
            num_zero_len = (non_spec_query_lens_cpu == 0).sum().item()
            num_prefills = non_spec_query_lens_cpu.size(0) - num_decodes - num_zero_len
            num_decode_tokens = num_decodes
            num_prefill_tokens = (
                non_spec_query_lens_cpu.sum().item() - num_decode_tokens
            )
            num_spec_decode_tokens = (
                query_lens_cpu.sum().item() - num_prefill_tokens - num_decode_tokens
            )

            if num_prefills == 0 and num_decodes == 0:
                spec_token_size = min(
                    num_spec_decodes * (self.num_spec + 1),
                    query_start_loc_cpu[-1].item(),
                )
                spec_token_indx = torch.arange(
                    spec_token_size,
                    dtype=torch.int32,
                    device=query_start_loc.device,
                )
                non_spec_token_indx = torch.empty(
                    0, dtype=torch.int32, device=query_start_loc.device
                )
                # Filter by spec_sequence_masks to exclude padded sequences
                spec_state_indices_tensor = block_table_tensor[
                    spec_sequence_masks, : self.num_spec + 1
                ]
                non_spec_state_indices_tensor = None
                # Padded sequences are always at the back, so the first
                # num_spec_decodes + 1 entries of query_start_loc already
                # contain the correct cumulative token counts.
                spec_query_start_loc = query_start_loc[: num_spec_decodes + 1]
                non_spec_query_start_loc = None
                non_spec_query_start_loc_cpu = None
            else:
                spec_token_masks = torch.repeat_interleave(
                    spec_sequence_masks, query_lens
                )
                index = torch.argsort(spec_token_masks, stable=True)
                num_non_spec_tokens = num_prefill_tokens + num_decode_tokens
                non_spec_token_indx = index[:num_non_spec_tokens]
                spec_token_indx = index[num_non_spec_tokens:]

                spec_state_indices_tensor = block_table_tensor[
                    spec_sequence_masks, : self.num_spec + 1
                ]
                non_spec_state_indices_tensor = block_table_tensor[
                    ~spec_sequence_masks, 0
                ]

                spec_query_start_loc = torch.zeros(
                    num_spec_decodes + 1,
                    dtype=torch.int32,
                    device=query_start_loc.device,
                )
                torch.cumsum(
                    query_lens[spec_sequence_masks], dim=0, out=spec_query_start_loc[1:]
                )
                non_spec_query_start_loc = torch.zeros(
                    query_lens.size(0) - num_spec_decodes + 1,
                    dtype=torch.int32,
                    device=query_start_loc.device,
                )
                torch.cumsum(
                    query_lens[~spec_sequence_masks],
                    dim=0,
                    out=non_spec_query_start_loc[1:],
                )
                non_spec_query_start_loc_cpu = torch.zeros(
                    query_lens_cpu.size(0) - num_spec_decodes + 1,
                    dtype=torch.int32,
                )
                torch.cumsum(
                    query_lens_cpu[~spec_sequence_masks_cpu],
                    dim=0,
                    out=non_spec_query_start_loc_cpu[1:],
                )

            assert num_accepted_tokens is not None
            num_accepted_tokens = num_accepted_tokens[spec_sequence_masks]

        if num_prefills > 0:
            has_initial_state = context_lens_tensor > 0
            if spec_sequence_masks is not None:
                has_initial_state = has_initial_state[~spec_sequence_masks]
                assert non_spec_query_start_loc_cpu is not None
            nums_dict, batch_ptr, token_chunk_offset_ptr = (
                compute_causal_conv1d_metadata(
                    non_spec_query_start_loc_cpu,
                    device=query_start_loc.device,
                )
            )
        else:
            has_initial_state = None

        non_spec_chunked_prefill_meta = None
        if num_prefills > 0:
            assert non_spec_query_start_loc_cpu is not None
            non_spec_chunked_prefill_meta = (
                self._build_non_spec_chunked_prefill_meta(non_spec_query_start_loc_cpu)
            )

        # Function code counted on either presency non-spec decode or spec decode,
        # but not both.
        assert not (num_decodes > 0 and num_spec_decodes > 0), (
            f"num_decodes: {num_decodes}, num_spec_decodes: {num_spec_decodes}"
        )

        # Prepare tensors for cudagraph
        # Note: m.num_actual_tokens is already padded by the model runner for CUDAGraph
        batch_size = m.num_actual_tokens

        if (
            self.use_full_cuda_graph
            and num_prefills == 0
            and num_decodes == 0
            and num_spec_decodes <= self.decode_cudagraph_max_bs
            and num_spec_decode_tokens <= self.decode_cudagraph_max_bs
        ):
            self.spec_state_indices_tensor[:num_spec_decodes].copy_(
                spec_state_indices_tensor, non_blocking=True
            )
            spec_state_indices_tensor = self.spec_state_indices_tensor[:batch_size]
            spec_state_indices_tensor[num_spec_decodes:].fill_(PAD_SLOT_ID)

            self.spec_sequence_masks[:num_spec_decodes].copy_(
                spec_sequence_masks[:num_spec_decodes], non_blocking=True
            )
            spec_sequence_masks = self.spec_sequence_masks[:batch_size]
            spec_sequence_masks[num_spec_decodes:].fill_(False)

            assert non_spec_token_indx is not None and spec_token_indx is not None
            self.non_spec_token_indx[: non_spec_token_indx.size(0)].copy_(
                non_spec_token_indx, non_blocking=True
            )
            non_spec_token_indx = self.non_spec_token_indx[
                : non_spec_token_indx.size(0)
            ]

            self.spec_token_indx[: spec_token_indx.size(0)].copy_(
                spec_token_indx, non_blocking=True
            )
            spec_token_indx = self.spec_token_indx[: spec_token_indx.size(0)]

            self.spec_query_start_loc[: num_spec_decodes + 1].copy_(
                spec_query_start_loc, non_blocking=True
            )
            spec_num_query_tokens = spec_query_start_loc[-1]  # type: ignore[index]
            spec_query_start_loc = self.spec_query_start_loc[: batch_size + 1]
            spec_query_start_loc[num_spec_decodes + 1 :].fill_(spec_num_query_tokens)

            self.num_accepted_tokens[:num_spec_decodes].copy_(
                num_accepted_tokens, non_blocking=True
            )
            num_accepted_tokens = self.num_accepted_tokens[:batch_size]
            num_accepted_tokens[num_spec_decodes:].fill_(1)

        if (
            self.use_full_cuda_graph
            and num_prefills == 0
            and num_spec_decodes == 0
            and num_decodes <= self.decode_cudagraph_max_bs
        ):
            self.non_spec_state_indices_tensor[:num_decodes].copy_(
                non_spec_state_indices_tensor, non_blocking=True
            )
            non_spec_state_indices_tensor = self.non_spec_state_indices_tensor[
                :batch_size
            ]
            non_spec_state_indices_tensor[num_decodes:].fill_(PAD_SLOT_ID)

            self.non_spec_query_start_loc[: num_decodes + 1].copy_(
                non_spec_query_start_loc, non_blocking=True
            )
            non_spec_num_query_tokens = non_spec_query_start_loc[-1]  # type: ignore[index]
            non_spec_query_start_loc = self.non_spec_query_start_loc[: batch_size + 1]
            non_spec_query_start_loc[num_decodes + 1 :].fill_(non_spec_num_query_tokens)

        attn_metadata = GDNAttentionMetadata(
            num_prefills=num_prefills,
            num_prefill_tokens=num_prefill_tokens,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            num_spec_decodes=num_spec_decodes,
            num_spec_decode_tokens=num_spec_decode_tokens,
            num_actual_tokens=m.num_actual_tokens,
            has_initial_state=has_initial_state,
            spec_query_start_loc=spec_query_start_loc,
            non_spec_query_start_loc=non_spec_query_start_loc,
            spec_state_indices_tensor=spec_state_indices_tensor,
            non_spec_state_indices_tensor=non_spec_state_indices_tensor,
            spec_sequence_masks=spec_sequence_masks,
            spec_token_indx=spec_token_indx,
            non_spec_token_indx=non_spec_token_indx,
            num_accepted_tokens=num_accepted_tokens,
            nums_dict=nums_dict,
            batch_ptr=batch_ptr,
            token_chunk_offset_ptr=token_chunk_offset_ptr,
            non_spec_chunked_prefill_meta=non_spec_chunked_prefill_meta,
        )
        return attn_metadata

    def build_for_cudagraph_capture(
        self, common_attn_metadata: CommonAttentionMetadata
    ):
        """
        This method builds the metadata for full cudagraph capture.
        Currently, only decode is supported for full cudagraphs with Mamba.
        """
        m = common_attn_metadata

        assert (
            m.num_reqs <= self.decode_cudagraph_max_bs
            and m.num_actual_tokens <= self.decode_cudagraph_max_bs
        ), (
            f"GDN only supports decode-only full CUDAGraph capture. "
            f"Make sure batch size ({m.num_reqs}) <= "
            f"cudagraph capture sizes ({self.decode_cudagraph_max_bs}), "
            f"and number of tokens ({m.num_actual_tokens}) <= "
            f"cudagraph capture sizes ({self.decode_cudagraph_max_bs})."
        )

        num_accepted_tokens = torch.diff(m.query_start_loc)
        num_decode_draft_tokens_cpu = (num_accepted_tokens - 1).cpu()

        return self.build(0, m, num_accepted_tokens, num_decode_draft_tokens_cpu)
