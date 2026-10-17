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

## 1b. 端 8B → 边 8B 主实验（2026-10，修订实现）

> 口径与 §1 的 `channel_main_4b8b_*` 逐项一致（60 题 × 3 信道 × SNR{0,15} × 3 种子，
> 协议 `v6_norestore`），唯一变量是端侧模型 4B → 8B。

| 目录 | 内容 | 记录数 | 方法数 |
|---|---|---|---|
| `main_8b8b_full14/` | **主表**：14 方法全量 | 15 120 | 14 |
| `main_8b8b_full7/` | 七方法子集（含 `random` / `fixed_H` / `truncate`） | 7 560 | 7 |

两条运行均使用**修订后的实现**（与 `channel_main_4b8b_*` 的旧实现不同，两套口径不可混用）：

| 项 | 修订值 | 记录字段 |
|---|---|---|
| 评分的 teacher-forcing 目标 | 端侧**自预测选项**（不读测试真值） | `score_target=predicted`（缓存侧） |
| 成对接收概率 | 共同信道下的**联合**成功概率 `Q_ij` | `coupling_model=joint` |
| 耦合定义 | **冗余惩罚** `M_ij = -cos(e_i,e_j)` | `coupling_variant=B`，`beta=0.13` |
| 耦合作用位置 | 基 LP/舍入用式 (19) 的上下文效用 | `lp_utility=context` |
| 保护级 | 差异化（H/M/L） | `protection_mode=differentiated` |

> 旧口径（`channel_main_4b8b_*`）为 `score_target=ground_truth`、`coupling` 关闭、
> `coupling_variant=raw`、`lp_utility=marginal`。两者在主表上不可直接相减，需按同一口径重跑。

## 1c. 8B→8B 的扫描与消融（2026-10，同一修订实现）

| 目录 | 内容 | 记录数 |
|---|---|---|
| `main_8b8b_snr5/` | SNR 五点扫描（0/5/10/15/20 dB） | 37 800 |
| `ablation2/` | 第 2 组配对消融，5 臂 | 5 × 1 080 |
| `pareto_8b8b/` | 等成本 Pareto（15 dB，7 预算点 + 5 对照策略） | 6 728 |

**SNR 曲线**（3 信道平均）：proposed 在 ≥10 dB 与 `full` 差距 ≤1.1pp，20 dB(rayleigh)、
10/15 dB(rician) 完全追平；差距集中在 0–5 dB（可靠预算 726–1506 符号，装不下 123 token 的报告）。

**消融矩阵**（配对 n=1080，与完整方法 A3 对比）：独立评分 −3.2pp(p=0.005)、固定保护
−3.8pp(p=0.0003)、**打乱耦合 −5.0pp(p<0.0001)**、TokCom+普通UEP −10.2pp(p<0.0001)。
打乱耦合这一臂说明收益来自耦合的**结构**而非幅度。

**等成本**：达到 Adaptive-SC 的 85.2% 只需 1 821 符号（对照 3 133，**省 42%**）；在对照的
3 133 符号成本下 proposed 只用到 1 821 且精度 +1.3pp。注意 `uniform_rL`（整份报告以最低
保护发送，2 101 符号）在 15 dB 处达 87.1%——**预算超过报告长度后方法不再有增量**，这与 SNR
曲线的饱和位置一致，方法的价值区间是低预算（0–5 dB）。

## 1d. VI-B-3 语义代理验证（2026-10 重跑，含负结果）

| 目录 | 内容 |
|---|---|
| `proxy_validation/proxy_8b_pred/` | 目标 = 端侧自预测答案（与缓存 `score_target=predicted` 一致） |
| `proxy_validation/proxy_8b_gold/` | 目标 = 金标准答案（2026-08 的旧协议，作对照） |
| `proxy_validation/proxy_instrument/` | 改用**接收端精度**作仪器 |
| `proxy_validation/report_span_selection/` | 重复区对预算的浪费能否转成精度 |

**结论 1（负）：用答案 NLL 验证代理不成立，但原因是仪器无信号，不是代理错。**
删除 30% 的 prompt 只让答案 NLL 变化 +0.06~+0.5 nats，且符号在样本间翻转；代理量程是 2–17。
用这种近零信号的量做秩相关，得到的是噪声：集合级逐样本 Spearman = **−0.389**（predicted）
/ **−0.331**（gold），保守比例 0.41/0.47。把目标改对只换来 0.06 的差别，且
`recomputed_vs_cached_full_nll_gap = 0.000`（predicted 臂），说明协议本身无问题。
旧值 −0.178 与此同源；此前把它当作"代理与真实删除损失秩相关只有 0.22"，其实是
**混池 + 无信号仪器**两个叠加。

**结论 2（正）：换成接收端精度，代理是有效的删除重要性排序。**
同一批 10 样本、同一下游 8B、协议 `v6_norestore`、理想信道：

| 删除比例 | 按代理排序删除 | 随机删除等量 | Δ |
|---|---|---|---|
| 10% | 90.0 | 90.0 | 0 |
| 20% | 90.0 | 70.0 | **+20.0** |
| 30% | 90.0 | 60.0 | **+30.0** |

未删除基线 100%。代理排序的删除显著更"安全"。

**结论 3（机制，可行动）：效用质量过度集中在"重复区"。**
打分载荷是"问题+选项+报告"，但接收端由 `build_question_block` 自带问题与选项，
前段是**冗余**。实测问题+选项区占载荷 **30%**，却拿走 top-10% 效用的 **44%**（1.46×）。
把这笔预算挪回报告正文，在同一 token 载荷下：保留 35% 时 全prompt 60.0 vs **仅正文 70.0**
（+10pp，n=10 仅 1 例差异，方向一致、统计力不足）；保留 25% 时两者持平（40.0）。

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
