"""INT4 KV-cache decode attention for the QServe experimental chain."""

from __future__ import annotations

from typing import Any

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - runtime dependent
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _qserve_int4_decode_kernel(
        query, key_code, key_scale, key_min, value_code, value_scale, value_min,
        key_residual, value_residual, output,
        stride_qb: tl.constexpr, stride_qh: tl.constexpr, stride_qd: tl.constexpr,
        stride_cb: tl.constexpr, stride_ch: tl.constexpr, stride_cn: tl.constexpr, stride_cd: tl.constexpr,
        stride_sb: tl.constexpr, stride_sh: tl.constexpr, stride_sn: tl.constexpr, stride_sg: tl.constexpr,
        stride_rb: tl.constexpr, stride_rh: tl.constexpr, stride_rn: tl.constexpr, stride_rd: tl.constexpr,
        stride_ob: tl.constexpr, stride_oh: tl.constexpr, stride_od: tl.constexpr,
        num_heads: tl.constexpr, num_kv_heads: tl.constexpr,
        quant_tokens, residual_tokens, scaling,
        HEAD_DIM: tl.constexpr, GROUP_SIZE: tl.constexpr, BLOCK_N: tl.constexpr,
    ):
        batch_idx = tl.program_id(0)
        head_idx = tl.program_id(1)
        kv_head_idx = head_idx // (num_heads // num_kv_heads)
        offs_d = tl.arange(0, HEAD_DIM)
        q = tl.load(query + batch_idx * stride_qb + head_idx * stride_qh + offs_d * stride_qd).to(tl.float32)
        running_max = -float("inf")
        running_sum = 0.0
        accumulator = tl.zeros((HEAD_DIM,), tl.float32)

        for start_n in range(0, quant_tokens, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            valid_n = offs_n < quant_tokens
            packed_d = offs_d // 2
            packed = tl.load(
                key_code + batch_idx * stride_cb + kv_head_idx * stride_ch
                + offs_n[:, None] * stride_cn + packed_d[None, :] * stride_cd,
                mask=valid_n[:, None], other=0,
            ).to(tl.int32)
            codes = tl.where((offs_d[None, :] & 1) == 0, packed & 15, (packed >> 4) & 15).to(tl.float32)
            group_idx = offs_d // GROUP_SIZE
            scales = tl.load(
                key_scale + batch_idx * stride_sb + kv_head_idx * stride_sh
                + offs_n[:, None] * stride_sn + group_idx[None, :] * stride_sg,
                mask=valid_n[:, None], other=0.0,
            ).to(tl.float32)
            minima = tl.load(
                key_min + batch_idx * stride_sb + kv_head_idx * stride_sh
                + offs_n[:, None] * stride_sn + group_idx[None, :] * stride_sg,
                mask=valid_n[:, None], other=0.0,
            ).to(tl.float32)
            scores = tl.sum((minima + codes * scales) * q[None, :], axis=1) * scaling
            scores = tl.where(valid_n, scores, -float("inf"))
            block_max = tl.max(scores, axis=0)
            new_max = tl.maximum(running_max, block_max)
            correction = tl.exp(running_max - new_max)
            probabilities = tl.exp(scores - new_max)

            packed_v = tl.load(
                value_code + batch_idx * stride_cb + kv_head_idx * stride_ch
                + offs_n[:, None] * stride_cn + packed_d[None, :] * stride_cd,
                mask=valid_n[:, None], other=0,
            ).to(tl.int32)
            codes_v = tl.where((offs_d[None, :] & 1) == 0, packed_v & 15, (packed_v >> 4) & 15).to(tl.float32)
            scales_v = tl.load(
                value_scale + batch_idx * stride_sb + kv_head_idx * stride_sh
                + offs_n[:, None] * stride_sn + group_idx[None, :] * stride_sg,
                mask=valid_n[:, None], other=0.0,
            ).to(tl.float32)
            minima_v = tl.load(
                value_min + batch_idx * stride_sb + kv_head_idx * stride_sh
                + offs_n[:, None] * stride_sn + group_idx[None, :] * stride_sg,
                mask=valid_n[:, None], other=0.0,
            ).to(tl.float32)
            values = minima_v + codes_v * scales_v
            accumulator = accumulator * correction + tl.sum(probabilities[:, None] * values, axis=0)
            running_sum = running_sum * correction + tl.sum(probabilities, axis=0)
            running_max = new_max

        for start_n in range(0, residual_tokens, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            valid_n = offs_n < residual_tokens
            keys = tl.load(
                key_residual + batch_idx * stride_rb + kv_head_idx * stride_rh
                + offs_n[:, None] * stride_rn + offs_d[None, :] * stride_rd,
                mask=valid_n[:, None], other=0.0,
            ).to(tl.float32)
            scores = tl.sum(keys * q[None, :], axis=1) * scaling
            scores = tl.where(valid_n, scores, -float("inf"))
            block_max = tl.max(scores, axis=0)
            new_max = tl.maximum(running_max, block_max)
            correction = tl.exp(running_max - new_max)
            probabilities = tl.exp(scores - new_max)
            values = tl.load(
                value_residual + batch_idx * stride_rb + kv_head_idx * stride_rh
                + offs_n[:, None] * stride_rn + offs_d[None, :] * stride_rd,
                mask=valid_n[:, None], other=0.0,
            ).to(tl.float32)
            accumulator = accumulator * correction + tl.sum(probabilities[:, None] * values, axis=0)
            running_sum = running_sum * correction + tl.sum(probabilities, axis=0)
            running_max = new_max

        tl.store(output + batch_idx * stride_ob + head_idx * stride_oh + offs_d * stride_od, accumulator / running_sum)


    @triton.jit
    def _qserve_int4_qk_kernel(
        query, key_code, key_scale, key_min, key_residual, scores,
        stride_qb: tl.constexpr, stride_qh: tl.constexpr, stride_qd: tl.constexpr,
        stride_cb: tl.constexpr, stride_ch: tl.constexpr, stride_cn: tl.constexpr, stride_cd: tl.constexpr,
        stride_sb: tl.constexpr, stride_sh: tl.constexpr, stride_sn: tl.constexpr, stride_sg: tl.constexpr,
        stride_rb: tl.constexpr, stride_rh: tl.constexpr, stride_rn: tl.constexpr, stride_rd: tl.constexpr,
        stride_xb: tl.constexpr, stride_xh: tl.constexpr, stride_xn: tl.constexpr,
        num_heads: tl.constexpr, num_kv_heads: tl.constexpr,
        quant_tokens, total_tokens, scaling,
        HEAD_DIM: tl.constexpr, GROUP_SIZE: tl.constexpr, BLOCK_N: tl.constexpr,
    ):
        batch_idx = tl.program_id(0)
        head_idx = tl.program_id(1)
        block_idx = tl.program_id(2)
        kv_head_idx = head_idx // (num_heads // num_kv_heads)
        offs_n = block_idx * BLOCK_N + tl.arange(0, BLOCK_N)
        valid_n = offs_n < total_tokens
        is_quantized = offs_n < quant_tokens
        residual_n = offs_n - quant_tokens
        offs_d = tl.arange(0, HEAD_DIM)
        packed_d = offs_d // 2
        group_idx = offs_d // GROUP_SIZE
        q = tl.load(query + batch_idx * stride_qb + head_idx * stride_qh + offs_d * stride_qd).to(tl.float32)

        packed = tl.load(
            key_code + batch_idx * stride_cb + kv_head_idx * stride_ch
            + offs_n[:, None] * stride_cn + packed_d[None, :] * stride_cd,
            mask=valid_n[:, None] & is_quantized[:, None], other=0,
        ).to(tl.int32)
        codes = tl.where((offs_d[None, :] & 1) == 0, packed & 15, (packed >> 4) & 15).to(tl.float32)
        scales = tl.load(
            key_scale + batch_idx * stride_sb + kv_head_idx * stride_sh
            + offs_n[:, None] * stride_sn + group_idx[None, :] * stride_sg,
            mask=valid_n[:, None] & is_quantized[:, None], other=0.0,
        ).to(tl.float32)
        minima = tl.load(
            key_min + batch_idx * stride_sb + kv_head_idx * stride_sh
            + offs_n[:, None] * stride_sn + group_idx[None, :] * stride_sg,
            mask=valid_n[:, None] & is_quantized[:, None], other=0.0,
        ).to(tl.float32)
        quantized_keys = minima + codes * scales
        dense_keys = tl.load(
            key_residual + batch_idx * stride_rb + kv_head_idx * stride_rh
            + residual_n[:, None] * stride_rn + offs_d[None, :] * stride_rd,
            mask=valid_n[:, None] & ~is_quantized[:, None], other=0.0,
        ).to(tl.float32)
        keys = tl.where(is_quantized[:, None], quantized_keys, dense_keys)
        score = tl.sum(keys * q[None, :], axis=1) * scaling
        tl.store(
            scores + batch_idx * stride_xb + head_idx * stride_xh + offs_n * stride_xn,
            score,
            mask=valid_n,
        )


    @triton.jit
    def _qserve_int4_pv_kernel(
        probabilities, value_code, value_scale, value_min, value_residual, output,
        stride_pb: tl.constexpr, stride_ph: tl.constexpr, stride_pn: tl.constexpr,
        stride_cb: tl.constexpr, stride_ch: tl.constexpr, stride_cn: tl.constexpr, stride_cd: tl.constexpr,
        stride_sb: tl.constexpr, stride_sh: tl.constexpr, stride_sn: tl.constexpr, stride_sg: tl.constexpr,
        stride_rb: tl.constexpr, stride_rh: tl.constexpr, stride_rn: tl.constexpr, stride_rd: tl.constexpr,
        stride_ob: tl.constexpr, stride_oh: tl.constexpr, stride_od: tl.constexpr,
        num_heads: tl.constexpr, num_kv_heads: tl.constexpr,
        quant_tokens, total_tokens,
        HEAD_DIM: tl.constexpr, GROUP_SIZE: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
    ):
        batch_idx = tl.program_id(0)
        head_idx = tl.program_id(1)
        d_block_idx = tl.program_id(2)
        kv_head_idx = head_idx // (num_heads // num_kv_heads)
        offs_d = d_block_idx * BLOCK_D + tl.arange(0, BLOCK_D)
        valid_d = offs_d < HEAD_DIM
        packed_d = offs_d // 2
        group_idx = offs_d // GROUP_SIZE
        accumulator = tl.zeros((BLOCK_D,), tl.float32)

        for start_n in range(0, total_tokens, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            valid_n = offs_n < total_tokens
            is_quantized = offs_n < quant_tokens
            residual_n = offs_n - quant_tokens
            probabilities_block = tl.load(
                probabilities + batch_idx * stride_pb + head_idx * stride_ph + offs_n * stride_pn,
                mask=valid_n, other=0.0,
            ).to(tl.float32)
            packed = tl.load(
                value_code + batch_idx * stride_cb + kv_head_idx * stride_ch
                + offs_n[:, None] * stride_cn + packed_d[None, :] * stride_cd,
                mask=valid_n[:, None] & is_quantized[:, None] & valid_d[None, :], other=0,
            ).to(tl.int32)
            codes = tl.where((offs_d[None, :] & 1) == 0, packed & 15, (packed >> 4) & 15).to(tl.float32)
            scales = tl.load(
                value_scale + batch_idx * stride_sb + kv_head_idx * stride_sh
                + offs_n[:, None] * stride_sn + group_idx[None, :] * stride_sg,
                mask=valid_n[:, None] & is_quantized[:, None] & valid_d[None, :], other=0.0,
            ).to(tl.float32)
            minima = tl.load(
                value_min + batch_idx * stride_sb + kv_head_idx * stride_sh
                + offs_n[:, None] * stride_sn + group_idx[None, :] * stride_sg,
                mask=valid_n[:, None] & is_quantized[:, None] & valid_d[None, :], other=0.0,
            ).to(tl.float32)
            quantized_values = minima + codes * scales
            dense_values = tl.load(
                value_residual + batch_idx * stride_rb + kv_head_idx * stride_rh
                + residual_n[:, None] * stride_rn + offs_d[None, :] * stride_rd,
                mask=valid_n[:, None] & ~is_quantized[:, None] & valid_d[None, :], other=0.0,
            ).to(tl.float32)
            values = tl.where(is_quantized[:, None], quantized_values, dense_values)
            accumulator += tl.sum(probabilities_block[:, None] * values, axis=0)

        tl.store(
            output + batch_idx * stride_ob + head_idx * stride_oh + offs_d * stride_od,
            accumulator,
            mask=valid_d,
        )


    @triton.jit
    def _qserve_int4_score_pv_kernel(
        scores, value_code, value_scale, value_min, value_residual, output,
        stride_xb: tl.constexpr, stride_xh: tl.constexpr, stride_xn: tl.constexpr,
        stride_cb: tl.constexpr, stride_ch: tl.constexpr, stride_cn: tl.constexpr, stride_cd: tl.constexpr,
        stride_sb: tl.constexpr, stride_sh: tl.constexpr, stride_sn: tl.constexpr, stride_sg: tl.constexpr,
        stride_rb: tl.constexpr, stride_rh: tl.constexpr, stride_rn: tl.constexpr, stride_rd: tl.constexpr,
        stride_ob: tl.constexpr, stride_oh: tl.constexpr, stride_od: tl.constexpr,
        num_heads: tl.constexpr, num_kv_heads: tl.constexpr,
        quant_tokens, total_tokens,
        HEAD_DIM: tl.constexpr, GROUP_SIZE: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
    ):
        batch_idx = tl.program_id(0)
        head_idx = tl.program_id(1)
        d_block_idx = tl.program_id(2)
        kv_head_idx = head_idx // (num_heads // num_kv_heads)
        offs_d = d_block_idx * BLOCK_D + tl.arange(0, BLOCK_D)
        valid_d = offs_d < HEAD_DIM
        packed_d = offs_d // 2
        group_idx = offs_d // GROUP_SIZE
        running_max = -float("inf")
        running_sum = 0.0
        accumulator = tl.zeros((BLOCK_D,), tl.float32)

        for start_n in range(0, total_tokens, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            valid_n = offs_n < total_tokens
            is_quantized = offs_n < quant_tokens
            residual_n = offs_n - quant_tokens
            score_block = tl.load(
                scores + batch_idx * stride_xb + head_idx * stride_xh + offs_n * stride_xn,
                mask=valid_n, other=-float("inf"),
            ).to(tl.float32)
            block_max = tl.max(score_block, axis=0)
            new_max = tl.maximum(running_max, block_max)
            correction = tl.exp(running_max - new_max)
            probabilities = tl.exp(score_block - new_max)
            packed = tl.load(
                value_code + batch_idx * stride_cb + kv_head_idx * stride_ch
                + offs_n[:, None] * stride_cn + packed_d[None, :] * stride_cd,
                mask=valid_n[:, None] & is_quantized[:, None] & valid_d[None, :], other=0,
            ).to(tl.int32)
            codes = tl.where((offs_d[None, :] & 1) == 0, packed & 15, (packed >> 4) & 15).to(tl.float32)
            scales = tl.load(
                value_scale + batch_idx * stride_sb + kv_head_idx * stride_sh
                + offs_n[:, None] * stride_sn + group_idx[None, :] * stride_sg,
                mask=valid_n[:, None] & is_quantized[:, None] & valid_d[None, :], other=0.0,
            ).to(tl.float32)
            minima = tl.load(
                value_min + batch_idx * stride_sb + kv_head_idx * stride_sh
                + offs_n[:, None] * stride_sn + group_idx[None, :] * stride_sg,
                mask=valid_n[:, None] & is_quantized[:, None] & valid_d[None, :], other=0.0,
            ).to(tl.float32)
            quantized_values = minima + codes * scales
            dense_values = tl.load(
                value_residual + batch_idx * stride_rb + kv_head_idx * stride_rh
                + residual_n[:, None] * stride_rn + offs_d[None, :] * stride_rd,
                mask=valid_n[:, None] & ~is_quantized[:, None] & valid_d[None, :], other=0.0,
            ).to(tl.float32)
            values = tl.where(is_quantized[:, None], quantized_values, dense_values)
            accumulator = accumulator * correction + tl.sum(probabilities[:, None] * values, axis=0)
            running_sum = running_sum * correction + tl.sum(probabilities, axis=0)
            running_max = new_max

        tl.store(
            output + batch_idx * stride_ob + head_idx * stride_oh + offs_d * stride_od,
            accumulator / running_sum,
            mask=valid_d,
        )


    @triton.jit
    def _qserve_int4_gqa_qk_kernel(
        query, key_code, key_scale, key_min, key_residual, scores,
        stride_qb: tl.constexpr, stride_qh: tl.constexpr, stride_qd: tl.constexpr,
        stride_cb: tl.constexpr, stride_ch: tl.constexpr, stride_cn: tl.constexpr, stride_cd: tl.constexpr,
        stride_sb: tl.constexpr, stride_sh: tl.constexpr, stride_sn: tl.constexpr, stride_sg: tl.constexpr,
        stride_rb: tl.constexpr, stride_rh: tl.constexpr, stride_rn: tl.constexpr, stride_rd: tl.constexpr,
        stride_xb: tl.constexpr, stride_xh: tl.constexpr, stride_xn: tl.constexpr,
        heads_per_kv: tl.constexpr, quant_tokens, total_tokens, scaling,
        HEAD_DIM: tl.constexpr, GROUP_SIZE: tl.constexpr, BLOCK_N: tl.constexpr,
    ):
        batch_idx = tl.program_id(0)
        kv_head_idx = tl.program_id(1)
        block_idx = tl.program_id(2)
        offs_h = kv_head_idx * heads_per_kv + tl.arange(0, heads_per_kv)
        offs_n = block_idx * BLOCK_N + tl.arange(0, BLOCK_N)
        valid_n = offs_n < total_tokens
        is_quantized = offs_n < quant_tokens
        residual_n = offs_n - quant_tokens
        offs_d = tl.arange(0, HEAD_DIM)
        packed_d = offs_d // 2
        group_idx = offs_d // GROUP_SIZE
        queries = tl.load(
            query + batch_idx * stride_qb + offs_h[:, None] * stride_qh + offs_d[None, :] * stride_qd
        ).to(tl.float16)
        packed = tl.load(
            key_code + batch_idx * stride_cb + kv_head_idx * stride_ch
            + offs_n[:, None] * stride_cn + packed_d[None, :] * stride_cd,
            mask=valid_n[:, None] & is_quantized[:, None], other=0,
        ).to(tl.int32)
        codes = tl.where((offs_d[None, :] & 1) == 0, packed & 15, (packed >> 4) & 15).to(tl.float32)
        scales = tl.load(
            key_scale + batch_idx * stride_sb + kv_head_idx * stride_sh
            + offs_n[:, None] * stride_sn + group_idx[None, :] * stride_sg,
            mask=valid_n[:, None] & is_quantized[:, None], other=0.0,
        ).to(tl.float32)
        minima = tl.load(
            key_min + batch_idx * stride_sb + kv_head_idx * stride_sh
            + offs_n[:, None] * stride_sn + group_idx[None, :] * stride_sg,
            mask=valid_n[:, None] & is_quantized[:, None], other=0.0,
        ).to(tl.float32)
        quantized_keys = minima + codes * scales
        dense_keys = tl.load(
            key_residual + batch_idx * stride_rb + kv_head_idx * stride_rh
            + residual_n[:, None] * stride_rn + offs_d[None, :] * stride_rd,
            mask=valid_n[:, None] & ~is_quantized[:, None], other=0.0,
        ).to(tl.float32)
        keys = tl.where(is_quantized[:, None], quantized_keys, dense_keys).to(tl.float16)
        score = tl.dot(queries, tl.trans(keys), out_dtype=tl.float32) * scaling
        tl.store(
            scores + batch_idx * stride_xb + offs_h[:, None] * stride_xh + offs_n[None, :] * stride_xn,
            score,
            mask=valid_n[None, :],
        )


    @triton.jit
    def _qserve_int4_gqa_pv_kernel(
        probabilities, value_code, value_scale, value_min, value_residual, output,
        stride_pb: tl.constexpr, stride_ph: tl.constexpr, stride_pn: tl.constexpr,
        stride_cb: tl.constexpr, stride_ch: tl.constexpr, stride_cn: tl.constexpr, stride_cd: tl.constexpr,
        stride_sb: tl.constexpr, stride_sh: tl.constexpr, stride_sn: tl.constexpr, stride_sg: tl.constexpr,
        stride_rb: tl.constexpr, stride_rh: tl.constexpr, stride_rn: tl.constexpr, stride_rd: tl.constexpr,
        stride_ob: tl.constexpr, stride_oh: tl.constexpr, stride_od: tl.constexpr,
        heads_per_kv: tl.constexpr, quant_tokens, total_tokens,
        HEAD_DIM: tl.constexpr, GROUP_SIZE: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
    ):
        batch_idx = tl.program_id(0)
        kv_head_idx = tl.program_id(1)
        d_block_idx = tl.program_id(2)
        offs_h = kv_head_idx * heads_per_kv + tl.arange(0, heads_per_kv)
        offs_d = d_block_idx * BLOCK_D + tl.arange(0, BLOCK_D)
        valid_d = offs_d < HEAD_DIM
        packed_d = offs_d // 2
        group_idx = offs_d // GROUP_SIZE
        accumulator = tl.zeros((heads_per_kv, BLOCK_D), tl.float32)

        for start_n in range(0, total_tokens, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            valid_n = offs_n < total_tokens
            is_quantized = offs_n < quant_tokens
            residual_n = offs_n - quant_tokens
            probability_block = tl.load(
                probabilities + batch_idx * stride_pb + offs_h[:, None] * stride_ph + offs_n[None, :] * stride_pn,
                mask=valid_n[None, :], other=0.0,
            ).to(tl.float16)
            packed = tl.load(
                value_code + batch_idx * stride_cb + kv_head_idx * stride_ch
                + offs_n[:, None] * stride_cn + packed_d[None, :] * stride_cd,
                mask=valid_n[:, None] & is_quantized[:, None] & valid_d[None, :], other=0,
            ).to(tl.int32)
            codes = tl.where((offs_d[None, :] & 1) == 0, packed & 15, (packed >> 4) & 15).to(tl.float32)
            scales = tl.load(
                value_scale + batch_idx * stride_sb + kv_head_idx * stride_sh
                + offs_n[:, None] * stride_sn + group_idx[None, :] * stride_sg,
                mask=valid_n[:, None] & is_quantized[:, None] & valid_d[None, :], other=0.0,
            ).to(tl.float32)
            minima = tl.load(
                value_min + batch_idx * stride_sb + kv_head_idx * stride_sh
                + offs_n[:, None] * stride_sn + group_idx[None, :] * stride_sg,
                mask=valid_n[:, None] & is_quantized[:, None] & valid_d[None, :], other=0.0,
            ).to(tl.float32)
            quantized_values = minima + codes * scales
            dense_values = tl.load(
                value_residual + batch_idx * stride_rb + kv_head_idx * stride_rh
                + residual_n[:, None] * stride_rn + offs_d[None, :] * stride_rd,
                mask=valid_n[:, None] & ~is_quantized[:, None] & valid_d[None, :], other=0.0,
            ).to(tl.float32)
            values = tl.where(is_quantized[:, None], quantized_values, dense_values).to(tl.float16)
            accumulator += tl.dot(probability_block, values, out_dtype=tl.float32)

        tl.store(
            output + batch_idx * stride_ob + offs_h[:, None] * stride_oh + offs_d[None, :] * stride_od,
            accumulator,
            mask=valid_d[None, :],
        )


def triton_int4_decode_available(query: torch.Tensor) -> bool:
    return bool(triton is not None and query.is_cuda and query.shape[-2] == 1)


def qserve_int4_decode_attention(query: torch.Tensor, layer: Any, *, scaling: float) -> torch.Tensor:
    """Attend to packed historical KV plus the dense residual window."""

    if not triton_int4_decode_available(query):
        raise RuntimeError("Triton INT4 decode attention requires a CUDA query with sequence length 1.")
    if layer._key_code is None or layer._key_residual is None or layer._value_residual is None:
        raise RuntimeError("Fused qserve attention requires packed and residual KV tensors.")
    batch, num_heads, _, head_dim = query.shape
    num_kv_heads = int(layer._key_residual.shape[1])
    quant_tokens = int(layer._key_code.shape[-2])
    residual_tokens = int(layer._key_residual.shape[-2])
    output = torch.empty((batch, num_heads, head_dim), device=query.device, dtype=query.dtype)
    _qserve_int4_decode_kernel[(batch, num_heads)](
        query, layer._key_code, layer._key_scale, layer._key_min,
        layer._value_code, layer._value_scale, layer._value_min,
        layer._key_residual, layer._value_residual, output,
        query.stride(0), query.stride(1), query.stride(3),
        layer._key_code.stride(0), layer._key_code.stride(1), layer._key_code.stride(2), layer._key_code.stride(3),
        layer._key_scale.stride(0), layer._key_scale.stride(1), layer._key_scale.stride(2), layer._key_scale.stride(3),
        layer._key_residual.stride(0), layer._key_residual.stride(1), layer._key_residual.stride(2), layer._key_residual.stride(3),
        output.stride(0), output.stride(1), output.stride(2),
        num_heads=num_heads, num_kv_heads=num_kv_heads,
        quant_tokens=quant_tokens, residual_tokens=residual_tokens, scaling=scaling,
        HEAD_DIM=head_dim, GROUP_SIZE=layer.config.group_size, BLOCK_N=32, num_warps=4,
    )
    return output.unsqueeze(1)


def qserve_int4_split_decode_attention(query: torch.Tensor, layer: Any, *, scaling: float) -> torch.Tensor:
    """Two-stage packed-KV decode with parallel score and value reductions.

    The original fused kernel keeps a full 256-wide value accumulator per head.
    This variant first computes INT4 key scores, then distributes INT4 value
    aggregation across 64-wide output tiles.  It is deliberately a separate
    experimental backend because extra launches only pay off when the reduced
    register pressure beats the launch and softmax overhead on a target device.
    """

    if not triton_int4_decode_available(query):
        raise RuntimeError("Triton INT4 split decode requires a CUDA query with sequence length 1.")
    if layer._key_code is None or layer._key_residual is None or layer._value_residual is None:
        raise RuntimeError("Split qserve attention requires packed and residual KV tensors.")

    batch, num_heads, _, head_dim = query.shape
    num_kv_heads = int(layer._key_residual.shape[1])
    quant_tokens = int(layer._key_code.shape[-2])
    residual_tokens = int(layer._key_residual.shape[-2])
    total_tokens = quant_tokens + residual_tokens
    scores = torch.empty((batch, num_heads, total_tokens), device=query.device, dtype=torch.float32)
    _qserve_int4_qk_kernel[(batch, num_heads, triton.cdiv(total_tokens, 32))](
        query, layer._key_code, layer._key_scale, layer._key_min, layer._key_residual, scores,
        query.stride(0), query.stride(1), query.stride(3),
        layer._key_code.stride(0), layer._key_code.stride(1), layer._key_code.stride(2), layer._key_code.stride(3),
        layer._key_scale.stride(0), layer._key_scale.stride(1), layer._key_scale.stride(2), layer._key_scale.stride(3),
        layer._key_residual.stride(0), layer._key_residual.stride(1), layer._key_residual.stride(2), layer._key_residual.stride(3),
        scores.stride(0), scores.stride(1), scores.stride(2),
        num_heads=num_heads, num_kv_heads=num_kv_heads,
        quant_tokens=quant_tokens, total_tokens=total_tokens, scaling=scaling,
        HEAD_DIM=head_dim, GROUP_SIZE=layer.config.group_size, BLOCK_N=32, num_warps=4,
    )
    output = torch.empty((batch, num_heads, head_dim), device=query.device, dtype=query.dtype)
    _qserve_int4_score_pv_kernel[(batch, num_heads, triton.cdiv(head_dim, 64))](
        scores, layer._value_code, layer._value_scale, layer._value_min, layer._value_residual, output,
        scores.stride(0), scores.stride(1), scores.stride(2),
        layer._value_code.stride(0), layer._value_code.stride(1), layer._value_code.stride(2), layer._value_code.stride(3),
        layer._value_scale.stride(0), layer._value_scale.stride(1), layer._value_scale.stride(2), layer._value_scale.stride(3),
        layer._value_residual.stride(0), layer._value_residual.stride(1), layer._value_residual.stride(2), layer._value_residual.stride(3),
        output.stride(0), output.stride(1), output.stride(2),
        num_heads=num_heads, num_kv_heads=num_kv_heads,
        quant_tokens=quant_tokens, total_tokens=total_tokens,
        HEAD_DIM=head_dim, GROUP_SIZE=layer.config.group_size, BLOCK_N=32, BLOCK_D=64, num_warps=4,
    )
    return output.unsqueeze(1)


def qserve_int4_gqa_decode_attention(query: torch.Tensor, layer: Any, *, scaling: float) -> torch.Tensor:
    """Tensor-core split decode that processes all GQA heads for one KV head together."""

    if not triton_int4_decode_available(query):
        raise RuntimeError("Triton INT4 GQA decode requires a CUDA query with sequence length 1.")
    if layer._key_code is None or layer._key_residual is None or layer._value_residual is None:
        raise RuntimeError("GQA qserve attention requires packed and residual KV tensors.")
    batch, num_heads, _, head_dim = query.shape
    num_kv_heads = int(layer._key_residual.shape[1])
    if num_heads % num_kv_heads:
        raise ValueError("Qwen GQA decode requires an integer query-to-KV head ratio.")
    heads_per_kv = num_heads // num_kv_heads
    quant_tokens = int(layer._key_code.shape[-2])
    total_tokens = quant_tokens + int(layer._key_residual.shape[-2])
    scores = torch.empty((batch, num_heads, total_tokens), device=query.device, dtype=torch.float32)
    _qserve_int4_gqa_qk_kernel[(batch, num_kv_heads, triton.cdiv(total_tokens, 32))](
        query, layer._key_code, layer._key_scale, layer._key_min, layer._key_residual, scores,
        query.stride(0), query.stride(1), query.stride(3),
        layer._key_code.stride(0), layer._key_code.stride(1), layer._key_code.stride(2), layer._key_code.stride(3),
        layer._key_scale.stride(0), layer._key_scale.stride(1), layer._key_scale.stride(2), layer._key_scale.stride(3),
        layer._key_residual.stride(0), layer._key_residual.stride(1), layer._key_residual.stride(2), layer._key_residual.stride(3),
        scores.stride(0), scores.stride(1), scores.stride(2),
        heads_per_kv=heads_per_kv, quant_tokens=quant_tokens, total_tokens=total_tokens, scaling=scaling,
        HEAD_DIM=head_dim, GROUP_SIZE=layer.config.group_size, BLOCK_N=32, num_warps=4,
    )
    probabilities = torch.softmax(scores, dim=-1)
    output = torch.empty((batch, num_heads, head_dim), device=query.device, dtype=query.dtype)
    _qserve_int4_gqa_pv_kernel[(batch, num_kv_heads, triton.cdiv(head_dim, 64))](
        probabilities, layer._value_code, layer._value_scale, layer._value_min, layer._value_residual, output,
        probabilities.stride(0), probabilities.stride(1), probabilities.stride(2),
        layer._value_code.stride(0), layer._value_code.stride(1), layer._value_code.stride(2), layer._value_code.stride(3),
        layer._value_scale.stride(0), layer._value_scale.stride(1), layer._value_scale.stride(2), layer._value_scale.stride(3),
        layer._value_residual.stride(0), layer._value_residual.stride(1), layer._value_residual.stride(2), layer._value_residual.stride(3),
        output.stride(0), output.stride(1), output.stride(2),
        heads_per_kv=heads_per_kv, quant_tokens=quant_tokens, total_tokens=total_tokens,
        HEAD_DIM=head_dim, GROUP_SIZE=layer.config.group_size, BLOCK_N=32, BLOCK_D=64, num_warps=4,
    )
    return output.unsqueeze(1)
