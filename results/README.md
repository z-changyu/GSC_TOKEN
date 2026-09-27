# 实验结果数据

本目录存放论文各表/图所依赖的**原始实验结果**（逐样本记录与汇总），可由 `scripts/` 中的运行器在相同配置下重新生成。除本目录外，`outputs/` 下的缓存、归档运行与蒸馏产物未纳入版本库（见文末"未纳入内容"）。

## 0. 统一口径（阅读本目录前请先确认）

| 项 | 取值 |
|---|---|
| 模型配置 | 主线：端侧 4B → 边侧 8B（`configs/experiments/v2x_edge_edge_main_4b8b*.yaml`）；部分对照为端侧 8B |
| 接收端协议 | 公平主口径 `v6_norestore`；对照 `v0`（无协议）与 `v6`（含受损触发补全） |
| 信息比特/Token | 16 bit |
| 编码开销/Token | 保护级 H/M/L = 48/32/16 个等效比特（码率 1/3、1/2、1） |
| 每符号编码比特 | 1（`MODULATION_ORDER`），故信道使用数 = 编码比特数 |
| 单窗口物理容量 | 1000 个编码比特（`T_s` = 1 ms，`B` = 1 MHz），窗口数 = ⌈编码比特/1000⌉ |
| 可靠预算 $C_\tau$ | $T_s B\,c_\varepsilon$：15 dB 约 4036、0 dB 约 574（等效比特） |
| 聚合规则 | **先把同一输入的重复观测归并为该输入的中位数，再对输入统计 P50/P95 与均值**；正确率取全部记录均值（不取 0/1 的中位）。P95 秩取 ⌈0.95n⌉ |
| 任务时延主指标 | 从"车辆获得图像与问题"到"RSU 输出答案"：端侧计算 + 上行 + 接收端计算；**不含**车载原生触发（模拟 20 ms）与答案回传（模拟 0.5 ms，另存 `total_ms_extended`） |
| 计时环境 | 2×RTX 3090（24 GB，驱动 535.261.03）；Intel i9-12900K；125 GB RAM；Ubuntu Linux 5.15.0-139；Python 3.10.20；PyTorch 2.5.1+cu121（CUDA 12.1、cuDNN 9.1.0）；Transformers 4.57.6 |

## 1. 主线与扫描（论文表 I、图 SNR/带宽-CSI）

| 目录 | 内容 | 对应论文位置 |
|---|---|---|
| `channel_main_4b8b_fair/` | **主运行**：3 衰落信道 × 60 样本 × 3 种子，SNR {0, 15} dB，协议 v6_norestore | 表 I 主结果；图 SNR 的 0/15 dB 点；等成本节锚点 |
| `channel_main_4b8b_snr/` | SNR 五点扫描的 5/10/20 dB 补点 | 图 SNR 曲线 |
| `channel_main_4b8b_b025/ b050/ b200/` | 带宽 0.25 / 0.5 / 2 MHz（1 MHz 取 `_fair`） | 图 带宽-CSI (a) |
| `channel_main_4b8b_csise005/ csise010/` | CSI 误差 $\sigma_e^2$ = 0.05 / 0.10（0.01 取 `_fair`） | 图 带宽-CSI (b) |
| `channel_main_4b8b_pool150/` | 未筛选 150 题复现（筛选前充分性检验） | 鲁棒性小节 |

> 上表所有目录的 `channel_results.csv` 均为逐样本记录，列含义见记录字段（精度、符号数、保留率、超预算、窗口数、预算等）。

## 2. 接收端协议对照（论文表 II）

| 目录 | 协议 |
|---|---|
| `channel_main_4b8b_v0/` | v0（无协议，两臂对照的左列） |
| `channel_main_4b8b_fair/` | v6_norestore（公平主口径，中列） |
| `channel_main_4b8b_v6/` | v6 含受损触发补全（右列） |
| `channel_8b_final14/`、`channel_8b_final14_v6final/` | 端侧 8B 配置下的 v0 与 v6 含补全对照（支撑"补全增益随交付率下降而放大"） |

## 3. 等成本考核（论文图 pareto）

| 目录 | 内容 |
|---|---|
| `pareto_main_4b8b/results.csv` | 人工预算 400–4000 的逐（样本, 信道, SNR, 种子, 策略）记录 |
| `pareto_main_4b8b/summary.json` | 各策略精度/符号数/预算利用/超预算、等成本配对检验与**运行溯源**（边侧模型、协议、`restore_threshold`、打分缓存、信道） |
| `pareto_main_4b8b/pareto.log` | 运行日志 |

## 4. 同条件阶段计时与读出对照（论文表 III–VI）

| 目录 | 内容 |
|---|---|
| `task_records/` | **15 dB 固定信道实例**：30 样本 × 3 重复的阶段计时（报告/评分逐段、选择、协调、接收端）与 8B 读出对照 |
| `task_records_binding/` | **0 dB 固定信道实例**：同上 |
| `task_records/table1_stage_costs.tex` | 表 III（阶段耗时，含计时边界与实测/模拟/校验标识）LaTeX 片段 |
| `task_records/table2_summary.tex` | 表 VI（同条件精度-资源-必要计算三场景）LaTeX 片段 |
| `task_records/table3_readout.tex` | 表 V（逐方法读出对照）LaTeX 片段 |
| `*/readout_comparison.csv` | 逐（方法, 样本, 重复）的判别式/自回归耗时与答案一致性 |
| `*/task_records_summary.json` | 汇总与运行溯源 |

## 5. 预警时延与读出方式（论文时延小节）

`latency_letter_4b8b/`（主线判别式读出）、`latency_v6_letter/`、`latency_v6_norestore/`、`latency_letter_check/`、`warning_latency/warning_latency.json`（阶段 p50/p95 汇总）。

## 6. 消融（论文消融小节）

| 目录 | 内容 |
|---|---|
| `gamma_screen_4b8b/` | 期望效用的凹变换 $\gamma \in \{0, 0.25, 0.5, 1\}$ |
| `span_screen_4b8b/` | 选择单元（Token vs 整子句 span） |
| `multiterminal_4b8b/` | 多终端比例公平（$K$、$\vartheta$ 前沿与 Jain 指数） |

## 7. 运行日志

`_logs/` 保存各次运行与队列的日志（含命令、耗时、退出码），用于溯源与复现核对。

## 8. 未纳入内容

以下目录**不随仓库分发**（体积大且可由代码与输入重新生成）：

- `outputs/cache/`（打分缓存，203 MB，含 2 个 >50 MB 文件）
- `outputs/runs/`（历史运行归档，392 MB）
- `outputs/distill/`（端侧蒸馏中间产物，52 MB）
- 各目录下的 `channel_results.partial.csv`（中断续写文件）
- 早期 2B→4B 与部分 8B 端侧档案（论文主线已改用 4B→8B）

如需其中任一内容，请在仓库 issue 中说明运行名称，或按 `configs/` 与 `scripts/` 重新生成。
