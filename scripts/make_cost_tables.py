#!/usr/bin/env python
"""Build the two cost tables of the paper from the recorded measurements.

Table A -- per-stage cost: method, stage, timing scope, median/P95, sample count.
Table B -- accuracy / spectrum / time: task accuracy, communication cost in
           unified units (symbols, equivalent bits, air-time), method-specific
           compute and the end-to-end task latency.

Sources (all recorded, nothing recomputed by hand):
  outputs/stage_costs/stage_costs.csv        per-sample stage timings
  outputs/stage_costs/task_timeline.csv      per-sample task totals
  outputs/channel_main_4b8b_fair/*.csv       accuracy and symbol cost at 15 dB
"""
from __future__ import annotations

import math
import statistics
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
STAGE_CSV = ROOT / "outputs/stage_costs/stage_costs.csv"
TIMELINE_CSV = ROOT / "outputs/stage_costs/task_timeline.csv"
MAIN_CSV = ROOT / "outputs/channel_main_4b8b_fair/channel_results.csv"
OUT_DIR = ROOT / "outputs/stage_costs"

FADE = ["rayleigh", "rician", "frequency_selective"]
SNR = 15.0
BANDWIDTH_HZ = 1_000_000.0
SLOT_S = 0.001
BITS_PER_TOKEN = 16.0

# (key in the CSV, Chinese label, timing scope)
SHARED_STAGES = [
    ("preprocess_ms", "图像预处理", "像素/解码+chat 模板构造（至 CPU 张量）"),
    ("vision_ms", "视觉塔前向", "仅 visual 模块，no_grad，不含投影"),
    ("scoring_report_ms", "报告生成", "流水线实际支付：score() 内部生成，max_new_tokens=320，KV 解码含视觉编码"),
    ("scoring_net_ms", "语用评分（净）", "评分专有部分：组装+前向+聚合+反向+耦合"),
    ("scoring_assemble_ms", "  ├ 嵌入+RoPE 组装", "get_rope_index + embedding，允许梯度"),
    ("scoring_forward_ms", "  ├ 教师强制前向", "output_attentions=True，含视觉 token"),
    ("scoring_attention_aggregate_ms", "  ├ 注意力/语义聚合", "prompt 切片注意力评分 + 语义方向"),
    ("scoring_backward_ms", "  ├ 梯度反向", "对 prompt 嵌入求导，含掩码前向"),
    ("scoring_coupling_ms", "  └ 耦合矩阵构建", "注意力耦合打分 + 凹区间判定"),
    ("coord_k4_ms", "多终端协调 K=4", "语义价格 + 带宽分配（CPU）"),
    ("coord_k10_ms", "多终端协调 K=10", "同上"),
    ("coord_k20_ms", "多终端协调 K=20", "同上"),
    ("rx_prep_ms", "接收端输入处理", "chat 模板 + 分词 + 搬运到设备"),
    ("rx_forward_ms", "接收端前向", "单次前向，取末位 logits"),
    ("rx_readout_ms", "接收端读出", "字母 token 分组取 argmax"),
]
METHOD_STAGES = [
    ("schedule_proposed_ms", "本文方法", "基础 LP + 认证舍入 + 局部改进（CPU/张量）"),
    ("schedule_lp_ms", "本文方法", "  ├ 基础 LP（对偶搜索）"),
    ("schedule_rounding_ms", "本文方法", "  ├ 认证舍入"),
    ("schedule_refine_ms", "本文方法", "  └ 局部改进"),
    ("schedule_adaptive_sc_ms", "Adaptive-SC", "自适应语义压缩选级（CPU）"),
    ("schedule_tokcom_uep_ms", "TokCom-UEP", "TokCom 重要性 + 本文调度器（同码路径）"),
    ("schedule_full_ms", "未压缩参考", "整份提示置于最高保护级（无需优化）"),
]
METHOD_NAMES = {"proposed": "本文方法", "adaptive_sc": "Adaptive-SC",
                "tokcom_uep": "TokCom-UEP", "full": "未压缩参考"}


def stats(values: list[float]) -> tuple[float, float, int]:
    xs = sorted(v for v in values if v == v)
    if not xs:
        return float("nan"), float("nan"), 0
    rank = min(len(xs) - 1, max(0, math.ceil(0.95 * len(xs)) - 1))
    return statistics.median(xs), xs[rank], len(xs)


def table_a(stage: pd.DataFrame) -> str:
    rows = []
    for key, label, scope in SHARED_STAGES:
        if key not in stage:
            continue
        p50, p95, n = stats(stage[key].tolist())
        rows.append(("共享（所有方法）", label, scope, p50, p95, n))
    for key, method, scope in METHOD_STAGES:
        if key not in stage:
            continue
        p50, p95, n = stats(stage[key].tolist())
        rows.append((method, label_of(key), scope, p50, p95, n))
    tex = ["% 表 A：阶段耗时（中位数 / P95），n 为观测数",
           "\\begin{tabular}{llrrr}", "\\hline",
           "方法 & 阶段 & 中位数 (ms) & P95 (ms) & $n$\\\\", "\\hline"]
    for method, label, _scope, p50, p95, n in rows:
        tex.append(f"{method} & {label} & {p50:.2f} & {p95:.2f} & {n}\\\\")
    tex += ["\\hline", "\\end{tabular}"]
    return "\n".join(tex)


def label_of(key: str) -> str:
    for k, label, _ in METHOD_STAGES:
        if k == key:
            return label
    return key


def main() -> None:
    stage = pd.read_csv(STAGE_CSV)
    timeline = pd.read_csv(TIMELINE_CSV) if TIMELINE_CSV.exists() else pd.DataFrame()
    main_frame = pd.read_csv(MAIN_CSV)
    all_frame = main_frame[main_frame.channel.isin(FADE)]          # 0 and 15 dB
    main_frame = all_frame[all_frame.snr_db == SNR]                # 15 dB slice

    print("=" * 92)
    print("表 A  阶段耗时（30 样本，端4B→边8B，bf16，batch=1，预热 2，"
          "GPU 阶段均 torch.cuda.synchronize）")
    print("=" * 92)
    print("%-12s %-22s %10s %10s %5s  %s" % ("方法", "阶段", "中位数/ms", "P95/ms", "n", "计时范围"))
    header_tex = table_a(stage)
    for key, label, scope in SHARED_STAGES:
        if key not in stage:
            continue
        p50, p95, n = stats(stage[key].tolist())
        print("%-12s %-22s %10.2f %10.2f %5d  %s" % ("共享", label, p50, p95, n, scope))
    for key, method, scope in METHOD_STAGES:
        if key not in stage:
            continue
        p50, p95, n = stats(stage[key].tolist())
        print("%-12s %-22s %10.2f %10.2f %5d  %s" % (method, label_of(key), p50, p95, n, scope))

    print()
    print("=" * 108)
    print(f"表 B  精度—资源—时间（精度/符号：0 与 15 dB × 3 衰落信道全条件均值；时间列：15 dB 实测，"
          f"空口时间按 {BANDWIDTH_HZ/1e6:.0f} MHz、{SLOT_S*1e3:.0f} ms 时隙折算）")
    print("=" * 108)
    rows = []
    for method, name in METHOD_NAMES.items():
        sub = main_frame[main_frame.method == method]          # 15 dB slice
        full_slice = all_frame[all_frame.method == method]     # 0 + 15 dB
        if sub.empty or full_slice.empty:
            continue
        acc_all = full_slice.correct.mean() * 100.0
        acc_15 = sub.correct.mean() * 100.0
        symbols = full_slice.transmitted_symbols.mean()
        tokens = full_slice.selected_tokens.mean() if "selected_tokens" in full_slice \
            else float("nan")
        bits = tokens * BITS_PER_TOKEN
        # 1 ms slot over 1 MHz carries 1000 channel uses, so air-time = symbols / 1000 ms.
        airtime_ms = symbols / (BANDWIDTH_HZ * SLOT_S)
        compute_key = {"proposed": "schedule_proposed_ms",
                       "adaptive_sc": "schedule_adaptive_sc_ms",
                       "tokcom_uep": "schedule_tokcom_uep_ms",
                       "full": "schedule_full_ms"}[method]
        compute = stats(stage[compute_key].tolist())[0] if compute_key in stage else float("nan")
        # Task latency: shared measured stages + this method's own stage + simulated radio.
        if not timeline.empty:
            base = (timeline.terminal_ms_measured
                    - stats(stage["schedule_proposed_ms"].tolist())[0])
            totals = base + compute + timeline.edge_ms_measured + timeline.trigger_ms_sim \
                + timeline.uplink_ms_sim + timeline.downlink_ms_sim
            total_p50, total_p95, _ = stats(totals.tolist())
        else:
            total_p50 = total_p95 = float("nan")
        rows.append((name, acc_all, acc_15, symbols, bits, airtime_ms, compute,
                     total_p50, total_p95))
    print("%-12s %12s %10s %10s %11s %11s %13s %11s %11s" % (
        "方法", "精度0/15dB/%", "精度15dB/%", "符号数", "等效比特", "空口/ms",
        "专属计算/ms", "时延P50/ms", "时延P95/ms"))
    for name, a_all, a15, sym, bits, air, comp, t50, t95 in rows:
        print("%-12s %12.1f %10.1f %10.0f %11.0f %11.2f %13.2f %11.0f %11.0f" % (
            name, a_all, a15, sym, bits, air, comp, t50, t95))

    tex = ["% 表 B：精度--资源--时间（精度与符号为 0/15 dB 全条件均值，时间列在 15 dB 实测）",
           "\\begin{tabular}{lrrrrrr}", "\\hline",
           "方法 & 精度 (\\%) & 精度@15dB (\\%) & 符号数 & 等效比特 & 专属计算 (ms) & 任务时延 P50/P95 (ms)\\\\",
           "\\hline"]
    for name, a_all, a15, sym, bits, air, comp, t50, t95 in rows:
        tex.append(f"{name} & {a_all:.1f} & {a15:.1f} & {sym:.0f} & {bits:.0f} & "
                   f"{comp:.2f} & {t50:.0f} / {t95:.0f}\\\\")
    tex += ["\\hline", "\\end{tabular}"]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "table_a_stage_costs.tex").write_text(header_tex + "\n", encoding="utf-8")
    (OUT_DIR / "table_b_cost_time.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
    print()
    print("LaTeX 已写入 outputs/stage_costs/table_a_stage_costs.tex 与 table_b_cost_time.tex")


if __name__ == "__main__":
    main()
