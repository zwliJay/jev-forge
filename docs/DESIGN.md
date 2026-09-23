# JevForge 设计文档(原理 · 架构 · 训练方法)

## 1. 问题与原理

Jev(TypeSafe "System One" 模型,公开文档:docs.typesafe.ai)把 LLM 当成一个
**决策函数**:输入 `state + questions`(choice / score / noul 三种原子问题),
输出**完整的校准概率分布**,不做自回归解码。它的价值在延迟(一次前向)、
成本(无输出 token)与不可能幻觉。

JevForge 的目标:用 0.6B 级开源 backbone 复刻这条技术路线,并且把
**训练数据格式 == 推理请求格式 == Jev 公开契约**,使线上日志可以直接回灌训练。

核心原理:每个 (question, candidate) 组成一条"候选路径",共享 state+question
前缀、只差候选段;取路径末 token 的隐状态,过一个共享标量头得到 logit;
同一问题的 K 个 logit 做 softmax 即得完整分布。训练因此是**概率学习**
(proper scoring),不是生成式 SFT。

## 2. 架构选型(以及为什么)

| 决策 | 选择 | 理由 |
|---|---|---|
| backbone | Qwen3-0.6B(base) | 延迟/成本预算内能力最强的开源小模型;0.6B 单卡即可全参训练 |
| 读出 | 共享前缀 + 每候选标量头(2 层 MLP) | 与 Jev 公开语义一致:候选独立、并行、K 可到 255;问题间天然隔离(无交叉注意力) |
| 备选读出(未采用) | 全候选拼进一个上下文(listwise) | 能建模候选竞争,但上下文随 K 膨胀、受顺序影响,且违反 Score "每级独立评估"语义;留作 v2 消融 |
| 损失 | CE + 0.5·Brier(逐题完整候选集) | CE 对软目标等价 forward KL;Brier 直接优化校准;两者都是 proper scoring rule |
| 校准 | calibration split 上网格拟合标量温度 | 原型常见缺口:默认 T=1 并不代表校准;我们显式拟合并写进 checkpoint |
| 显存 | 梯度检查点(use_reentrant=False)+ SDPA(flash 后端) | 见下节实测;换 `flash_attention_2` 只需改 attn_implementation |

## 2.5 显存与注意力后端(实测)

0.6B、batch 64×600 token、梯度检查点 + bf16 autocast 的训练微批,
`jevforge.bench_attention` 在 RTX 4080 SUPER 32G 上实测:

| attn_implementation | 峰值显存 | 每步耗时 |
|---|---:|---:|
| sdpa(内建 flash kernel,默认) | 8.53 GB | 3553 ms |
| eager(数学参考实现) | 12.45 GB | 5444 ms |
| flash_attention_2(flash-attn 2.8.3 实测) | 8.53 GB | 3554 ms |

结论:OOM 的第一杠杆是梯度检查点(31GB→~7GB),第二杠杆是 SDPA 的 flash
后端(比 eager 省 31% 显存、快 53%——torch≥2.2 自带,无需额外安装)。
**实测 flash-attn 2.8.3(FA2)与内建 SDPA-flash 完全打平**(8.53GB/3554ms),
0.6B+检查点规模下无需额外安装;放大模型/序列时再切 `attn_implementation=
"flash_attention_2"`,训练与推理共用同一开关。

## 3. 数据策略:标签优先,数据合成补缺

数据源:`LangAGI-Lab/Mind2Web-axtree-cleaned-lite`(Mind2Web 的 axtree
清洗版,公开数据集)。每一步标注自带 **pos/neg 候选元素**——这是现成监督,
直接用:

- `action`(choice):gold = 正例候选上的均匀分布(`uniform_over_positive_elements`);
  负例候选随机采样补到上限(默认 12 个,K≥2)。
- `is_target`(noul):随机抽一个候选构造自包含命题,真值 = 是否正例。
- **切分按 website 哈希离散,网站绝不跨 split**(防泄漏;构建器有断言)。
- state 为**规范 JSON 字符串**(sorted keys),修复原型常见的非规范 dict 序列化问题。

数据增强只补两处源数据没有的监督:

1. `difficulty`(score,3 档 rubric):生成并归一化难度分布
   (`generated_score_distribution`)。
2. train 集 `action` 增强:`(1-λ)·gold + λ·generated`(λ=0.25),
   为稀疏标签补充不确定性;dev/test 保持原始标签以做无偏评估。

生成结果进 append-only 缓存(可审计、可续跑、可复现)。

## 4. 训练流程

```
build_web.py  → records(train/dev/calibration/test/ood, 网站不交叉)
synthesize.py → +difficulty score 目标;train action 数据增强(生成缓存)
train.py      → head-only 预热(12 步)→ 全参(600 步,bf16,grad-clip 1.0)
                AdamW: backbone 2e-5 / head 2e-4;每 50 步 dev CE 选优
                结束后在 calibration 上拟合温度
predict.py    → Jev 契约答案(choice/noul/score + confidence)
serve.py      → POST /v1/systemone;请求日志 JSONL = 未来训练数据
evaluate.py   → top1-in-positives / Brier / ECE / MAE / 每网站分解
```

## 5. 与 NanoJev(TianyuCodings)的关系

JevForge 是**独立实现**:未复制其任何代码或 git 历史。参考的公开材料:
TypeSafe 公开 API 文档(接口契约)、Mind2Web 公开数据集、以及社区项目
(NanoJev / TheoLeeCj-jevforge)的**公开结论**——例如"训练域与真实负载域
不匹配"、"原型未做温度校准"、"候选路径重复编码前缀"——这些是设计输入,
不是代码来源。架构差异见上表;主要实证差异:训练域从自制游戏换成真实网页。

## 6. 验收标准

1. `python -m jevforge.train` 全程跑通,dev CE 收敛,checkpoint 可复加载。
2. test/ood 上:choice top1-in-positives 显著高于均匀基线;noul Brier/ECE 报告;
   difficulty MAE 报告;按网站分解无异常泄漏。
3. `python -m jevforge.serve` 通过 /v1/systemone 返回契约合规 JSON,日志落盘。
4. 抽样人工核验:答案分布与任务语义一致。


## 7. v2 改进路线(2026-09-19 评审后)

**已实施(v2 训练中)**:fused AdamW + 长度分桶(实测 ~2x 吞吐)、余弦 LR+
warmup、确定性目标标签平滑(ε=0.05)、分题型校准温度、wandb 全指标
(loss_ce/loss_brier/grad_norm/lr/sec/peak_mem + dev 分题型 CE,本地 JSONL 双写)。

**对比 Jev 的维度**(不止准确率):延迟、成本
(Jev $0.042/MTok 输入 vs 本地推理边际成本≈电费)、校准(Brier/ECE)、契约
互通(同一 /v1/systemone)。

**模型 v2 候选**:① 候选竞争头——在逐候选标量头外加轻量 set-attention
(建模 IIA 限制),消融验证;② 推理前缀 KV 复用(state+question 编码一次,
候选扇形展开,预计延迟再降数倍);③ backbone 扩到 Qwen3-1.7B(32G 卡可训)。

**数据 v2 候选**:K 12→20 + 同标签硬负例;state 加入任务轨迹(前步动作);
每条记录双 noul(一正一负);对齐 Mind2Web 官方 cross-website 切分以外部可比。

**训练 v2 候选**:RLCD 风格对照臂(采样完整分布 + proper-scoring
reward)、按题型损失加权、早停 patience、ECE 直接优化项。TypeSafe 只公开了
RLCD 的名称和校准目标，没有公开奖励函数、sampler 或优化算法；因此该对照臂
是独立研究实现，不宣称复现 Jev。

**通用任务 v2 候选**:意图路由、工具/API 选择、安全/升级分诊、文档决策、
有序风险评分。接入门槛是有限类型化答案、可审计标签或可验证结果，并且能固定
划分 train/dev/test/OOD，不把开放文本生成混入决策基准。

**未调优基线(同底座对照)**:使用原始权重和同一候选打分协议进行对照。
