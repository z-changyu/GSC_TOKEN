#!/usr/bin/env python
"""The three cost tables, from the same-condition records.

Two channel instances are reported side by side, because a single fixed reali-
sation is not enough: at 15 dB the reliable budget rarely binds, so every method
fits the whole report at the strongest protection level and the resource
comparison collapses; at 0 dB the budget binds, which is where the schedulers
actually differ. Run the driver once per instance:

    python scripts/measure_task_records.py --snr-db 15 --instance-tag loose
    python scripts/measure_task_records.py --snr-db 0  --instance-tag binding \\
        --output-dir outputs/task_records_binding

Table 1  per-stage cost (method, stage, P50/P95, n, boundary, measured/simulated tag)
Table 2  accuracy / resource in explicit units / necessary compute (with and
         without the shared report generation) / task latency, per instance
Table 3  8B readout comparison (discriminative single forward vs autoregressive
         letter output on identical inputs)

Repeats give per-sample medians; correctness is the mean over all records (never
a median of 0/1); across-sample P50/P95 over 30 inputs are preliminary.
"""
from __future__ import annotations

import math
import statistics
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INSTANCES = [("loose (15 dB)", ROOT / "outputs/task_records"),
             ("binding (0 dB)", ROOT / "outputs/task_records_binding")]
METHOD_CN = {"proposed": "本文方法", "adaptive_sc": "Adaptive-SC",
             "tokcom_uep": "TokCom-UEP", "full": "未压缩参考"}
SCORING_USERS = {"proposed", "adaptive_sc"}
STAGE_ROWS = [
    ("report_ms", "报告生成总计", "含图像处理与文本解码（视觉编码为诊断测量，不计入总时延）", "实测"),
    ("scoring_report_prepare_ms", "├ 报告输入组装", "含图像的 chat template + 搬运到设备", "实测"),
    ("scoring_report_ms", "├ 报告文本生成", "自回归生成（子项，不重复计入总计）", "实测"),
    ("scoring_report_decode_ms", "└ 报告解码与释放", "文本解码、删除张量与 empty_cache", "实测"),
    ("scoring_ms", "方法专属：语用评分总计", "仅消费语用效用的方法（本文方法/Adaptive-SC）", "实测"),
    ("scoring_tokenize_ms", "├ 提示分词与截断", "semantic prompt 分词 + max_prompt_tokens 截断", "实测"),
    ("scoring_prepare_ms", "├ 评分输入准备", "两次 processor（含图像）+ span 定位", "实测"),
    ("scoring_payload_transfer_ms", "├ 提示搬运与标签构造", "input_ids/attention_mask 搬运、答案标签拼接", "实测"),
    ("scoring_assemble_ms", "├ 嵌入与 RoPE 组装", "保留梯度", "实测"),
    ("scoring_forward_ms", "├ 教师强制前向", "output_attentions=True，含视觉 token", "实测"),
    ("scoring_attention_aggregate_ms", "├ 注意力与语义聚合", "prompt 切片 + 语义方向", "实测"),
    ("scoring_backward_ms", "├ 梯度反向", "含掩码前向/损失构造", "实测"),
    ("scoring_coupling_ms", "├ 耦合构建", "耦合打分 + 凹区间判定", "实测"),
    ("scoring_nll_sync_ms", "├ 答案 NLL 同步", "output.loss 取回 CPU", "实测"),
    ("scoring_serialize_ms", "├ 结果序列化（批量搬运）", "一次性 .cpu().tolist()，正式路径", "实测"),
    ("scoring_serialize_legacy_ms", "├（对照）逐元素序列化", "N^2 次标量搬运，诊断项，不计入总计", "实测"),
    ("scoring_cache_release_ms", "└ 缓存释放（empty_cache）", "删除中间张量 + 显存分配器回收", "实测"),
    ("scoring_residual_ms", "（父-子残差）", "T_parent − ΣT_child", "校验"),
    ("selection_ms", "方法专属：Token 选择与保护调度", "本文方法为 LP+舍入+局部改进", "实测"),
    ("selection_lp_ms", "├ 基础 LP（对偶搜索）", "CPU/张量", "实测"),
    ("selection_round_ms", "├ 认证舍入", "CPU", "实测"),
    ("selection_refine_ms", "└ 局部改进", "至 t_max=8", "实测"),
    ("allocation_ms", "RSU 协调：语义价格与带宽分配", "10 终端（CPU）", "实测"),
    ("receiver_prep_ms", "接收端：模板/分词/搬运", "chat 模板 + 分词 + 到设备", "实测"),
    ("receiver_forward_ms", "接收端：前向", "单次前向取末位 logits", "实测"),
    ("receiver_readout_ms", "接收端：答案读出（批量搬运）", "字母行一次性搬到 CPU 后 argmax", "实测"),
    ("receiver_readout_perletter_ms", "（对照）逐字母同步读出", "每字母一次 GPU→CPU 标量搬运", "实测"),
    ("external_signal_ms", "外部信号（TokCom）", "未测量，不默认零", "未测"),
    ("control_wait_ms", "控制等待", "未测量，不默认零", "未测"),
    ("codec_ms", "编解码", "未测量，不默认零", "未测"),
    ("uplink_ms_sim", "上行", "单时隙，设定", "模拟"),
    ("downlink_ms_sim", "回传", "设定", "模拟"),
    ("trigger_ms_sim", "车辆原生感知触发", "假设 10–30 ms", "模拟"),
]


def desc(values: list[float]) -> tuple[float, float, int]:
    xs = sorted(v for v in values if v == v)
    if not xs:
        return float("nan"), float("nan"), 0
    rank = min(len(xs) - 1, max(0, math.ceil(0.95 * len(xs)) - 1))
    return statistics.median(xs), xs[rank], len(xs)


def load(directory: Path) -> pd.DataFrame | None:
    path = directory / "task_records.csv"
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    return frame[~frame.warmup.astype(bool)]


def main() -> None:
    tex1 = ["% 表 1 阶段耗时（同条件，30 样本，每实例一固定信道；重复取每样本中位数）",
            "\\begin{tabular}{llrrrl}", "\\hline",
            "实例 & 方法 & 阶段 & P50 (ms) & P95 (ms) & 来源\\\\", "\\hline"]
    tex2 = ["% 表 2 精度--资源--时间（同条件；必要计算成本按三个缓存场景分列）",
            "% 资源成本＝编码比特；信道使用＝符号数（本设置下 1 bit/符号，故两者相等）",
            "% 窗口数＝逐样本 ceil(编码比特/单窗口物理容量) 的中位(均值)，非由中位开销倒推",
            "\\begin{tabular}{llrrrrrrrrrrr}", "\\hline",
            "实例 & 方法 & 精度 (\\%) & 信息比特 & 资源成本 & 信道使用 & 窗口数 中位(均值) & "
            "预算等价窗口 & 超预算 (\\%) & 场景1 含报告 & 场景2 缓存报告 & 场景3 全缓存 & "
            "主指标 P50/P95 (ms)\\\\", "\\hline"]
    tex3 = []
    for tag, directory in INSTANCES:
        frame = load(directory)
        if frame is None:
            print(f"[skip] {tag}: {directory} 无记录")
            continue
        per_sample = frame.groupby(["method", "sample_id"]).median(numeric_only=True)
        methods = [m for m in METHOD_CN if m in set(frame.method)]
        print("=" * 116)
        print(f"表 1  阶段耗时 —— 实例 {tag}（30 样本 / 3 次重复 / 预热 5 次 / GPU 同步计时）")
        print("=" * 116)
        print("%-12s %-30s %10s %10s %5s %-6s %s" % (
            "方法", "阶段", "P50/ms", "P95/ms", "n", "来源", "计时边界"))
        for method in methods:
            block = per_sample.xs(method, level="method")
            for column, label, boundary, source in STAGE_ROWS:
                if column not in block:
                    continue
                p50, p95, n = desc(block[column].tolist())
                if n == 0 or p50 != p50:
                    continue
                print("%-12s %-30s %10.2f %10.2f %5d %-6s %s" % (
                    METHOD_CN[method], label, p50, p95, n, source, boundary))
                tex1.append(f"{tag} & {METHOD_CN[method]} & {label} & {p50:.2f} & "
                            f"{p95:.2f} & {source} & {boundary}\\\\")
            print("-" * 116)

        print()
        print("表 2  同条件综合 —— 实例 %s（精度＝全部记录均值；资源成本为编码比特，不做空口时间折算）" % tag)
        print("%-12s %7s %9s %9s %14s %10s %7s %11s %11s %11s %12s" % (
            "方法", "精度/%", "信息比特", "资源成本", "窗口中位(均值)", "预算窗口", "超预算%",
            "场景1含报告", "场景2缓存报告", "场景3全缓存", "主指标P50/P95"))
        for method in methods:
            block = per_sample.xs(method, level="method")
            rows = frame[frame.method == method]
            accuracy = rows.correct.mean() * 100.0
            info = block.information_bits.median()
            coded = block.resource_cost_coded_bits.median()
            uses = block.channel_uses.median()
            window_med = block.window_slots_model.median()
            window_mean = block.window_slots_model.mean()
            window_cap = block.physical_window_cap_bits.median()
            budget = block.budget_coded_bits.median()
            budget_windows = budget / window_cap if window_cap else float("nan")
            over = float((block.resource_cost_coded_bits
                          > block.budget_coded_bits).mean()) * 100.0
            scoring = block.scoring_ms.median() if method in SCORING_USERS else 0.0
            # Necessary compute under the three cache scenarios: the shared report
            # generation and the pragmatic scoring are charged only when they are
            # actually recomputed, never mixed within one number.
            tail = (block.selection_ms.median() + block.allocation_ms.median()
                    + block.receiver_ms.median())
            s1 = block.report_ms.median() + scoring + tail       # report not cached
            s2 = scoring + tail                                  # report cached
            s3 = tail                                            # report + scoring cached
            p50, p95, _ = desc(block.total_ms.tolist())
            print("%-12s %7.1f %10.0f %10.0f %9.1f(%.2f) %11.3f %8.1f %12.1f %12.1f %12.1f %6.0f/%-4.0f" % (
                METHOD_CN[method], accuracy, info, coded, window_med, window_mean,
                budget_windows, over, s1, s2, s3, p50, p95))
            tex2.append(f"{tag} & {METHOD_CN[method]} & {accuracy:.1f} & {info:.0f} & "
                        f"{coded:.0f} & {uses:.0f} & {window_med:.1f} ({window_mean:.2f}) & "
                        f"{budget_windows:.2f} & {over:.1f} & {s1:.1f} & {s2:.1f} & {s3:.1f} & "
                        f"{p50:.0f} / {p95:.0f}\\\\")
        print()

        read_path = directory / "readout_comparison.csv"
        if read_path.exists():
            read = pd.read_csv(read_path)
            if not tex3:
                tex3 = ["% 表 3 8B 边侧读出对照（同一交付文本；每个方法每档 30 样本 x 3 重复 = 90 条）",
                        "\\begin{tabular}{llrrrrrrr}", "\\hline",
                        "实例 & 方法 & 判别式 P50 & 判别式 P95 & 自回归 P50 & 自回归 P95 & "
                        "输出长度 & 精度 (\\%) & 一致率 (\\%)\\\\", "\\hline"]
            print("表 3  8B 边侧读出对照 —— 实例 %s（共 %d 条观测）" % (tag, len(read)))
            print("%-12s %6s %10s %10s %10s %10s %8s %8s" % (
                "方法", "n", "判别P50", "判别P95", "自回P50", "自回P95", "输出长度", "一致率%"))
            for method in [m for m in METHOD_CN if m in set(read.method)]:
                sub = read[read.method == method]
                # Same aggregation rule as the stage and summary tables: the repeats
                # of one input are first reduced to a per-sample median, and the
                # reported P50/P95 are taken across the 30 inputs. The direct
                # 90-observation statistics are printed alongside for the record.
                per_input = sub.groupby("sample_id")
                d50, d95, n = desc(per_input.discriminative_ms.median().tolist())
                a50, a95, _ = desc(per_input.autoregressive_ms.median().tolist())
                rawa, _, _ = desc(sub.discriminative_ms.tolist())
                tok50, _, _ = desc(per_input.autoregressive_tokens.median().tolist())
                agree = sub.agree.mean() * 100.0
                acc = sub.discriminative_correct.mean() * 100.0
                print("%-12s %6d %10.1f %10.1f %10.1f %10.1f %8.0f %8.1f   [90条直接: P50 %.2f]"
                      % (METHOD_CN[method], n, d50, d95, a50, a95, tok50, agree, rawa))
                tex3.append(f"{tag} & {METHOD_CN[method]} & {d50:.1f} & {d95:.1f} & "
                            f"{a50:.1f} & {a95:.1f} & {tok50:.0f} & {acc:.1f} & {agree:.1f}\\\\")
            stops = read.stop_reason_proxy.value_counts().to_dict() if "stop_reason_proxy" in read else {}
            print("   停止原因分布: %s（上限 24）| 全体一致率 %.1f%%" % (
                stops, read.agree.mean() * 100))
            tex3.append("\\hline")

    tex1 += ["\\hline", "\\end{tabular}"]
    tex2 += ["\\hline", "\\end{tabular}"]
    out = ROOT / "outputs/task_records"
    out.mkdir(parents=True, exist_ok=True)
    (out / "table1_stage_costs.tex").write_text("\n".join(tex1) + "\n", encoding="utf-8")
    (out / "table2_summary.tex").write_text("\n".join(tex2) + "\n", encoding="utf-8")
    if tex3:
        (out / "table3_readout.tex").write_text("\n".join(tex3) + "\n", encoding="utf-8")
    print("LaTeX -> outputs/task_records/table1_stage_costs.tex, table2_summary.tex, "
          "table3_readout.tex")


if __name__ == "__main__":
    main()
