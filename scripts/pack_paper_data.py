"""Build a zip package with the manuscript and the data behind its tables/figures."""
import hashlib
import pathlib
import zipfile

ROOT = pathlib.Path("/mnt/4t/code_space/zcy/GSC_Token")
OUT = ROOT / "GSC_Token_paper_data_20260923.zip"

# run directory -> what it backs in the paper
RUNS = {
    "channel_main_2b4b_fair": "主表主线（端2B->边4B，v6-norestore，14方法）；现稿 tab:main / tab:modelaxis 第1行",
    "channel_main_2b4b_v0": "协议三臂 v0 臂（tab:protocol 第2列）",
    "channel_main_2b4b_v6": "协议三臂 含补全 v6 臂（tab:protocol 第4列）",
    "channel_main_4b8b_fair": "候选新主线（端4B->边8B，14方法）；tab:modelaxis 第4行",
    "channel_main_4b4b_fair": "模型轴 端4B->边4B（tab:modelaxis 第3行）",
    "channel_main_2b4b_distill_fair": "端侧蒸馏（tab:modelaxis 第2行）",
    "channel_8b_fair": "端8B->边4B 上界参照（含150题；论文按 manifest 过滤到60题）",
    "channel_main_2b4b_snr": "图5 信噪比五点扫描",
    "channel_main_2b4b_b025": "图6(a) 带宽 0.25 MHz",
    "channel_main_2b4b_b050": "图6(a) 带宽 0.5 MHz",
    "channel_main_2b4b_b200": "图6(a) 带宽 2 MHz",
    "channel_main_2b4b_csise005": "图6(b) CSI 误差 0.05",
    "channel_main_2b4b_csise010": "图6(b) CSI 误差 0.10",
    "pareto_main_2b4b": "图4 等成本考核（预算扫描 + 保护级构成）",
    "walkthrough": "图3 调度行为示例",
    "latency_v6_letter": "表 tab:latency 判别式读出（单次前向 argmax）",
    "latency_v6_norestore": "表 tab:latency 自回归 24-Token 对照",
    "latency_letter_check": "判定结果一致性核验（240条 0:0）",
    "warning_latency": "图7 预警时延预算与安全范围",
    "complexity": "计算复杂度与最优性核验",
    "cv_nll_calibration": "可靠容量与中断校准",
    "multiterminal_main": "多终端比例公平前沿（tab:multiterminal）",
    "co_fusion_full": "跨车报告融合（tab:fusion）",
    "tiered_pipeline_full": "三级流水线（云端精炼跳）",
}

RUN_FILES = ["channel_results.csv", "summary.json"]      # partial 副本不含在内
WHOLE_DIRS = ["walkthrough", "warning_latency", "complexity", "latency_v6_letter",
              "latency_v6_norestore", "latency_letter_check", "pareto_main_2b4b",
              "multiterminal_main", "co_fusion_full", "tiered_pipeline_full",
              "cv_nll_calibration"]

CACHES = ["v2xqa_scores_2b.jsonl", "v2xqa_scores_2b_surprisal.jsonl",
          "v2xqa_scores_2b_answer_aware.jsonl", "v2xqa_scores_4b.jsonl",
          "v2xqa_scores_4b_surprisal.jsonl", "v2xqa_scores_4b8b_answer_aware.jsonl",
          "v2xqa_scores_2b_distill.jsonl", "v2xqa_scores_2b_distill_surprisal.jsonl",
          "v2xqa_scores_2b_distill_answer_aware.jsonl",
          "v2xqa_scores_8b.jsonl", "v2xqa_scores_8b_surprisal.jsonl",
          "v2xqa_scores_8b_answer_aware.jsonl", "v2xqa_set_loo.jsonl"]

SCRIPTS = ["scripts/run_channel_experiment.py", "scripts/run_pareto.py",
           "scripts/build_score_cache.py", "scripts/build_answer_aware_scores.py",
           "scripts/build_benchmark_scores.py", "scripts/build_set_loo.py",
           "scripts/measure_warning_latency.py", "scripts/analyze_warning_latency.py",
           "scripts/plot_pareto_figure.py", "scripts/plot_scheduler_walkthrough.py",
           "scripts/plot_experiment_figures.py", "scripts/plot_system_metrics.py"]

README = """# GSC_Token 论文数据包（2026-09-23）

面向 V2X 风险判定的 Token 级语义传输：时变信道下的语用感知选择与调度
（IEEE TVT 投稿稿，中文版）

## 目录结构

    paper/      论文源码与成品
      *.tex / *.pdf / *.bbl        中文稿与英文稿、参考文献
      images/*.pdf                 全部插图（矢量）
    data/       论文表格与插图对应的实验数据
      <run>/channel_results.csv    逐观测结果（每行一次端到端评测）
      <run>/summary.json           该运行的汇总统计
      manifests/                   60 题样本清单（报告充分性筛选后）
    code/       复现脚本与配置
      scripts/ run_channel_experiment.py、run_pareto.py、plot_*.py
      configs/ experiments/*.yaml、channels/*.yaml
    cache/      语用打分缓存（Token 级效用与敏感性，供前端复算）
    DATA_INDEX.csv   数据文件清单（路径、大小、MD5、对应论文位置）

## 关键口径（阅读数据前必看）

1. **主线配置**：论文现稿主线为 端侧 2B -> 边侧 4B；候选新主线为 端侧 4B -> 边侧 8B
   （`data/channel_main_4b8b_fair/`）。
2. **接收端协议**：主口径为 `v6-norestore`（保留任务意图与字母输出约束、关闭受损补全）；
   `v0` 与含补全的 `v6` 见 `channel_main_2b4b_v0` / `_v6`。
3. **样本集合**：主实验为报告充分性筛选后的 60 题；
   `channel_8b_fair` 未筛选（150 题），分析时需按 `data/manifests/` 过滤到 60 题。
4. **观测规模**：每个方法 60 样本 x 3 信道 x 2 SNR x 3 信道种子 = 1080 次评测，
   14 方法合计 15120 行。
5. **protected 字段**：`transmitted_symbols`（等效比特）、`budget`（式(6) 可靠预算）、
   `TRR`（Token 保留率）、`correct`（判定正确）、`schedule_seconds` / `receive_seconds`
   （逐时隙调度与接收端生成时延）、`gpu_peak_memory_delta_gib` / `energy_j`。

## 复现命令（示例）

    # 主实验（端 2B -> 边 4B，公平协议）
    python scripts/run_channel_experiment.py \\
      --config configs/experiments/v2x_edge_edge_main_2b4b.yaml \\
      --receiver-protocol v6_norestore --weights 0.25 0.30 0.45 \\
      --output-dir outputs/channel_main_2b4b_fair

    # 等成本预算扫描（图 4）
    python scripts/run_pareto.py --score-cache outputs/cache/v2xqa_scores_2b.jsonl \\
      --baseline-csv outputs/channel_main_2b4b_fair/channel_results.csv \\
      --budgets 400 600 800 1200 1600 2000 2600 3200 4000 \\
      --output-dir outputs/pareto_main_2b4b

## 说明

- 不含 `channel_results.partial.csv`（与正式文件逐字节相同）。
- 不含 `v2xqa_scores_8b_coupling.jsonl`（40 MB 稠密耦合矩阵；本文耦合项默认关闭）。
- 正在重跑的 端4B->边8B 协议臂（v0/v6）未纳入，完成后可另行补齐。
"""


def add_file(zf, path: pathlib.Path, arcname: str, index: list):
    data = path.read_bytes()
    zf.writestr(arcname, data)
    index.append((arcname, len(data), hashlib.md5(data).hexdigest()[:12]))


def main() -> int:
    index: list = []
    missing: list = []
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        zf.writestr("README.md", README)

        # ---- manuscript
        for pat in ("*.tex", "*.pdf", "*.bbl"):
            for f in sorted((ROOT / "paper").glob(pat)):
                add_file(zf, f, "paper/" + f.name, index)
        for f in sorted((ROOT / "paper" / "images").glob("*.pdf")):
            add_file(zf, f, "paper/images/" + f.name, index)

        # ---- experimental runs
        for run, note in RUNS.items():
            d = ROOT / "outputs" / run
            if not d.is_dir():
                missing.append(run)
                continue
            if run in WHOLE_DIRS:
                for f in sorted(d.iterdir()):
                    if f.is_file():
                        add_file(zf, f, "data/%s/%s" % (run, f.name), index)
            else:
                for name in RUN_FILES:
                    f = d / name
                    if f.is_file():
                        add_file(zf, f, "data/%s/%s" % (run, name), index)
            zf.writestr("data/%s/README.txt" % run, note + "\n")

        # ---- manifests
        for f in sorted((ROOT / "data" / "manifests").glob("*")):
            if f.is_file():
                add_file(zf, f, "data/manifests/" + f.name, index)

        # ---- caches
        for name in CACHES:
            f = ROOT / "outputs" / "cache" / name
            if f.is_file():
                add_file(zf, f, "cache/" + name, index)

        # ---- code
        for rel in SCRIPTS:
            f = ROOT / rel
            if f.is_file():
                add_file(zf, f, "code/" + rel, index)
        for sub in ("experiments", "channels"):
            for f in sorted((ROOT / "configs" / sub).glob("*.yaml")):
                add_file(zf, f, "code/configs/%s/%s" % (sub, f.name), index)

        rows = ["path,bytes,md5_12"]
        rows += ["%s,%d,%s" % r for r in index]
        zf.writestr("DATA_INDEX.csv", "\n".join(rows) + "\n")

    size = OUT.stat().st_size
    print("打包完成: %s" % OUT.name)
    print("  条目 %d, 压缩后 %.1f MB" % (len(index) + 2, size / 1e6))
    if missing:
        print("  缺失目录: %s" % missing)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
