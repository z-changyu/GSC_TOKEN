# GSC Token — 实验执行梳理与计划

> 生成日期:2026-08-23。来源:论文 `GSC_TVT_theory_reinforced_EN.pdf`(11 页)、项目 docs/、src/ 源码与 configs/ 现状。

## 一、现状盘点

| 层面 | 状态 |
|---|---|
| 论文 | 理论完整(含附录 A/B/C 证明),**第六节实验评估零结果**(原文明确:数值结果待实现通过归约/可行性检查后再插入) |
| 代码 | 仅有 pilot 脚手架:数据适配、指标、`pragmatic/` 打分模块、`selection/`(random/utility)、`channel/`(仅 ideal) |
| 核心算法 | **未实现**:Lagrangian / LP 舍入 / KKT / Rayleigh / Rician / 频率选择性 / DeepMIMO / 多终端带宽分配 / 全部基线 |
| GPU | 可用(驱动 535.261.03 / CUDA 12.2);**仅使用 0 号 RTX 3090**(`CUDA_VISIBLE_DEVICES=0`),禁用双卡 |
| 环境 | `GSC_Token` conda 环境存在;曾有 `huggingface-hub` 版本冲突(见阶段 0) |

**结论**:需要先实现论文 Algorithm 1 与信道/基线模块,再跑实验;当前先打通 pilot。

## 二、论文要求的实验(第六节)

**数据/模型(论文原文)**:ScienceQA 分层抽样 5000 个含图多模态样本、五折交叉验证;BLIP-2 发射端、Flan-T5-Large 接收端、CLIP ViT-L/14;`N≤64`;rH/rM/rL=48/32/16;权重 (0.25,0.30,0.45)、β=0.25、Tmax=8。

**信道**:Rayleigh / Rician / 频率选择性 / DeepMIMO I1 × SNR 0/15 dB;σ_e²∈{0.01,0.05},ε_out∈{0.01,0.05,0.1},丢包 0.05/0.1。

### 6 组实验

| # | 实验 | 验证理论 | 产出 |
|---|---|---|---|
| 1 | 端到端信道对比 | 整体方法 | 4 信道×2 SNR:VQA 精度/CLIPScore/实际中断率/保留 token 比/Jain 公平/延迟/峰值显存/能耗 |
| 2 | 可靠容量校准 | 式(5)(6)(8) | 各 (σ_e²,ε_out) 下实测中断率达标 |
| 3 | 语义代理验证 | 式(17)(24)(27) | 保留比 25/50/75/100% 下 F_T(T) 与实测损失增量对比、余项在界内 |
| 4 | 模型失配 | 式(29)(30)(32) | Flan-T5-Base vs Large + 受控 prompt 偏移,留出校准集测失配 |
| 5 | 多终端权衡 | 式(43)–(50) | K∈{4,10,20},扫预算/保留比/β 得精度–Jain 前沿 |
| 6 | 复杂度与最优性 | 式(64)(68) | N_k、\|R\|、Tmax 变化;N_k≤20 穷举作最优值检验 1/(2(1+κ)) |

### 基线
- 外部:Prompt-SC[5]、Adaptive-SC[8]、IB token 选择[6]、LLMLingua[16]
- 内部:① 即插式容量 `T_s B log₂(1+|ĥ|²γ̄)` 替式(6);② 原始 `p_k|h_k|` 替式(46)

### 消融(逐一删,4 信道重复)
Aᵢ、Sᵢ、Gᵢ、耦合项、差异化保护、可靠容量、局部精修、在线价格更新、KKT 带宽分配。

### 指标口径(metrics_spec.md)
精确匹配归一化、TRR=选中/原始 token、符号数=Σxᵢrᵢ(与 token 数两轴)、AUBC(预算-精度曲线梯形积分)、目标精度节省 `1-R_proposed/R_baseline`、均值±95% CI、保存全部信道种子、每样本完整记录。

## 三、先决理论决定(实验前必须拍板)

1. **Gᵢ 的损失函数 L 精确定义**(最关键):式(3) 仅说"发射端 VLM 自回归 NLL",未指明是完整生成序列 / 语义 prompt / QA 目标 / token 局部损失。四者理论上不等价,需作者决定。
2. **Qwen3-VL 因果注意力对齐**:式(10)(16) 要求 teacher-forcing 下 masked 自回归 decoder 真实注意力权重。BLIP-2 Q-Former 双向、不满足;需改用 Qwen3-VL 或将 encoder 注意力仅作消融。
3. **局部交换 removal loss 显式公式**:论文只给插入增益(56),removal loss 未给、接受交换后重算未指定;pilot 先禁用局部交换。

> 注:(0.25,0.30,0.45) 与 β=0.25 为 legacy 参数,换 Qwen3 后需重新标定。

## 四、执行顺序

- **阶段 0 修环境**:降 `huggingface-hub<1.0`;确认 torch==2.5.1+cu121 且 `CUDA_VISIBLE_DEVICES=0` 下 `cuda.is_available()==True`;通过 `verify_environment.py`。
- **阶段 1 pilot**:下载 ScienceQA → 建 manifest(seed 2026,固定 150 val 样本)→ `test_model_loading.py`(eager attention)逐项检查 → `run_pilot.py`(ideal, full/random/proposed, TRR 0.3/0.5/0.7/1.0)。
- **阶段 2 定理论+选模型**:拍板 Gᵢ 定义、removal loss、BLIP-2 vs Qwen3(建议 Qwen3,同步改论文第六节)。
- **阶段 3 实现核心算法**:信道(4 种+可靠容量)→ 评分管线(A/S/G→u→耦合)→ 选择调度(Lagrangian→LP→舍入→局部精修→在线价格)→ 多终端 KKT → 6 个基线。
- **阶段 4 铺开实验**:按 6 组 + 消融,严格指标口径。

## 五、关键文件

- 论文:`GSC_TVT_theory_reinforced_EN.pdf`
- 协议:`docs/experiment_protocol.md`、`docs/metrics_spec.md`、`docs/open_questions.md`、`docs/paper_notes.md`
- 配置:`configs/models/qwen3_primary.yaml`(发射端 Qwen3-VL-2B、接收端 Qwen3-1.7B)、`configs/experiments/pilot_scienceqa.yaml`
- 脚本:`scripts/{verify_environment,download_datasets,build_manifests,test_model_loading,run_pilot}.py/.sh`
