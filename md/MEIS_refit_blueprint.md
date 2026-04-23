# MEIS 改装蓝图 (Refit Blueprint)

> 本文档的三个职责：
> 1. 固化五个候选 repo 的**真实样貌**（纸面 vs 代码）
> 2. 给出 MEIS 每一层/模块的**来源决定**：直接用 / 借模块 / 重写 / 原创
> 3. 显式划出 **MEIS 的原创贡献边界**，防止后期写论文时贡献点模糊
>
> 证据来源：`/home/erzhu419/mine_code/MEIS/reference/` 下论文方法节 + `/home/erzhu419/mine_code/MEIS/{boxing-gym, AutoToM, language_and_experience, msa-cogsci-2025-data, world-models}` 下源码直接阅读。每条 claim 附文件路径 + 行号 / 论文节号。

---

## 1. 候选 repo 深度画像（关键细节）

### 1.1 `boxing-gym` — **主基座**（评分 5/5）

**架构抽象**（论文 Fig.2, Sec.3；代码 `src/boxing_gym/{envs, agents}/`）

三类：
- `WorldEnv` — 环境，实现 `_build_model`、`reset`、`step`、`get_df`、`run_experiment` 等
- `Goal` — 持有 env 引用，定义 `get_system_message / get_goal_eval_question / evaluate_predictions / expected_information_gain`（见 [envs/goal.py](../boxing-gym/src/boxing_gym/envs/goal.py) 抽象基类）
- `Agent` (Scientist Agent) — 通过 `LMExperimenter` 调 OpenAI/Anthropic，在 `run_experiment.py` 里驱动 `set_goal → act → predict → explain`

两角色：**Scientist**（看得到 env，提假设）+ **Novice**（只看 Scientist 的自然语言解释，做预测）。这个 Scientist/Novice 对偶是 BoxingGym 的核心评测设计，不是一般的 agent 框架。

**概率后端**：PyMC。每个 env 自己写一个 `pymc.Model()`（例如 [envs/dugongs.py:79](../boxing-gym/src/boxing_gym/envs/dugongs.py#L79)）。**没有公共 BN 抽象**——每个 env 各写各的 ground-truth 模型。

**EIG 实现**（论文 Sec.3.2.1 Eq.(2)；代码 [envs/dugongs.py:74-128](../boxing-gym/src/boxing_gym/envs/dugongs.py#L74-L128)）

论文公式：
```
EIG(d) = E_{p(y|d)}[H[p(θ)] - H[p(θ|y,d)]]
```

Nested MC 估计（`num_outer_samples=1000`, `num_inner_samples=10`）：
```
̂μ_NMC(d) = 1/N Σ log[ p(y_n|θ_{n,0},d) / (1/M Σ p(y_n|θ_{n,m},d)) ]
```

代码确认：每个 env 在 Goal 子类里**重复实现**一遍 `expected_information_gain`。非通用，**跨 env 不复用**。`pm.sample(num_outer*num_inner+num_outer, tune=1000)` 做 outer+inner 共享采样池。

**Box's Loop 实现**（论文 Fig.1, Sec.3；代码 [agents/box_loop_experiment.py](../boxing-gym/src/boxing_gym/agents/box_loop_experiment.py) + [agents/model_search.py](../boxing-gym/src/boxing_gym/agents/model_search.py)）

循环结构：
1. 外层 `run_experiment.py` 驱动 Scientist agent 交互
2. 可选 `augment_scientist_with_ppl`（[run_experiment.py:32](../boxing-gym/run_experiment.py#L32)）注入 PPL 模型字符串作为额外 prompt
3. 内层 `run_box_loop` 做 propose(LLM) → `pymc_evaluate` → critic(LLM) → refit 的迭代

**评测指标**：**EI Regret**（与 100 个随机设计的 max EIG 差距）+ **standardized prediction error**（$\epsilon/\sigma_0$）+ **communication error**（Novice 基于 Scientist 解释的预测误差）。

**关键论文发现**（Sec.4.1–4.2, p.7–8）：
> "Prior information does not improve performance... LLMs cannot always optimally leverage statistical models... Box's Apprentice tends to favor overly simple functional forms due to limited data, such as using linear approximations for inherently nonlinear phenomena."

这是 **MEIS 对 BoxingGym 的胜场空间**——BoxingGym 作者自己承认 LLM + PP 不能稳定利用先验。MEIS 的跨域先验库 + 最小扰动评分如果能解决这个，就是 Publishable delta。

**10 个 env**（`src/boxing_gym/envs/`）：
| 文件 | 领域 | 可用性 |
|---|---|---|
| dugongs.py | 海牛生长曲线（年龄→长度） | ✅ Phase 1 首选 |
| peregrines.py | 游隼种群动态 | ✅ |
| death_process.py | 疾病传染率 | ✅ |
| lotka_volterra.py | 捕食-被捕食 | ✅ |
| irt.py | 项目反应理论（学生×题目） | ✅ |
| survival_analysis.py | 乳腺癌术后存活 | ✅ |
| location_finding.py | 信号源定位 | ✅ |
| hyperbolic_temporal_discount.py | 行为经济学延迟折扣 | ✅ |
| emotion.py | 赌博游戏→情绪（LLM 翻译概率程序）| ⚠️ 非纯 PP |
| moral_machines.py | 自动驾驶道德困境（LLM 翻译）| ⚠️ 非纯 PP |

**关键坑**（实测，已修）：
- `requirements.txt` 内部矛盾：同时 pin `numpy==2.0.1 + pytensor==2.20.0`，但 pytensor 2.20 调用 numpy 2 已删的 `np.obj2sctype`。已改为 `pymc>=5.13, pytensor>=2.20` 让环境真实可跑
- `box_loop_prompts.py` 的 prompt 硬编码，不是可配置模板——想换模型/语言要直接改代码
- `EIG` 缓存键 `tuple(existing_data)` 假设 observed_data 是 hashable；MEIS 要扩展时注意

---

### 1.2 `AutoToM` — **借模块参考**（评分 3/5）

**概率后端：无真贝叶斯**（论文 Sec.3.3–3.4；代码 [model/probs.py:50-80](../AutoToM/model/probs.py#L50-L80)）

关键引文（论文 Sec.3.3, p.5）：
> "we integrate LLMs as the computational backend to implement every aspect of the Bayesian inverse planning... we estimate each local conditional in $P(V^{t_s:t}, X^{t_s:t})$ using an LLM."

代码实锤：[probs.py:50](../AutoToM/model/probs.py#L50) 的 `get_likelihood_general` 直接给 LLM 发 "A) Likely / B) Unlikely" 选项 prompt，把选项概率当 likelihood 返回。**没有 PyMC/Gen/NumPyro**。

⚠️ **这意味着 AutoToM 的"贝叶斯推断"不等于 MEIS 的 L1 真概率推断**。AutoToM 论文 Limitations 自己承认 "LLM backend may produce inaccurate likelihood estimation"。

**模型发现循环**（论文 Algorithm 1, p.6）

```
Extract query → Information Extraction (observables X^{1:t})
→ Initial Model Proposal (最小模型, t_s = t)
→ BIP (Hypothesis Sampling → Reduction → Bayesian Inference)
→ 计算 utility U(M,q) = R(M,q) - C(M),
  其中 R = -H(posterior), C = α|M|
→ Variable Adjustment (贪心加变量)
→ 若仍 U < U_min: Timestep Adjustment (t_s ← t_s-1), 循环
→ A = argmax P(q | X^{t_s:t})
```

**`model_adjustment.py` 内部**（[model_adjustment.py:10-20](../AutoToM/model/model_adjustment.py#L10-L20)）：
```python
model_space = [
    ['State', 'Observation', 'Belief', 'Action', 'Goal'],   # POMDP
    ['State', 'Observation', 'Belief', 'Action'],
    ['State', 'Observation', 'Belief'],                     # Markov model
    ... (9 个变体，全是 ToM 语义)
]
all_variables = ["State", "Observation", "Belief", "Action", "Goal", "Utterance", "Belief of Goal"]
```

硬编码 MDP/POMDP/I-POMDP 9 种变体 + ToM 专用变量集。**不是通用模型发现**，是 ToM 专用。

**ElementExtractor** ([model/ElementExtractor.py:7-24](../AutoToM/model/ElementExtractor.py#L7-L24)) 里的 `Variable` dataclass 是**通用的**：
```python
class Variable:
    name, in_model, is_observed, possible_values, prior_probs
```

这个抽象在 MEIS L4 里能直接复用（把 NL 解析成变量表）。**其余的 ElementExtractor prompts 都是 ToM 专用**（`guided_belief_of_state.txt` 等）。

**领域**：ToM only——benchmark 是 ToMi、BigToM、MMToM-QA、MuMA-ToM、Hi-ToM（Sec.4.1）。

**可借 vs 不可借**：
- ✅ 借：`Variable` dataclass；`model_adjustment.py` 的**贪心搜索骨架**（即"加一个变量 → 比较 utility → 接受/回滚"）
- ❌ 不借：`probs.py` 的 LLM-likelihood 方案（与 MEIS L1 目标冲突）；`model_space` 的 MDP/POMDP 硬编码（MEIS 需要一般 Bayesian net 变体）
- ❌ 不借：几十个 ToM 专用 prompt（`prompts/` 目录）

---

### 1.3 `language_and_experience` — **排除**（评分 1/5）

- **域**：VGDL 游戏（aliens, preconditions, missileCommand, jaws, avoidGeorge...）
- **概率后端**：无 PP。`src/agent/thinking/` 是 LLM+自定义推理
- **依赖重度**：`vllm`（本地 LLM 需 GPU 显存）+ `mpi4py`（需 openmpi）+ `pygame` + `gym==0.26.2`
- 与 MEIS 的"身高-体重-脚印/能剧-祭司"类信念网络推理**完全错位**——它做的是开放游戏规则在线习得，不是 belief network inference

**只有一种情况值得看**：如果 MEIS Phase 4 结构同构识别要扩展到"通过多个游戏的规则学一般规则"，L&E 的 `src/agent/datastore.py` 的 rule-store 抽象可以参考。Phase 4 之前不看。

---

### 1.4 `msa-cogsci-2025-data` — **档案附录**（评分 0/5）

MSA (Wong 的 Model Synthesis Architecture) 只公开了**数据和 prompts**，**没代码**：
- `model-olympics-human-experiment/` — JSON 人类实验结果
- `msa-frame-prompts/` — prompt 文件
- `lm-only-baseline-prompts/`
- `example-scenarios/`

**价值**：Phase 2 写 MEIS 的 L4 prompt 时参考他们的 frame prompt 写法。其余不可用。

---

### 1.5 `world-models` (Wong 原版) — **蓝图**（评分 0/5）

4 个 Church `.scm` + 1 个 Jupyter notebook，作者自述 archival。已在 [phase0_smoke_test/smoke_test_tug_of_war.py](../phase0_smoke_test/smoke_test_tug_of_war.py) 手动 port d1 到 numpy 拒绝采样，PASS。

**价值**：读 `.scm` 作为每个域的语义 ground truth。其余不可用。

---

## 2. MEIS 模块 → 源码映射表

| MEIS 层 | 子模块 | 动作 | 来源 | 关键文件 / 行号 | 备注 |
|---|---|---|---|---|---|
| **L1 表示** | 单域贝叶斯程序（域内推断） | **直接用** | boxing-gym | `src/boxing_gym/envs/*.py` | 每个 env 的 `_build_model` 是 PyMC 生成模型 |
| **L1 表示** | `Env / Goal` 抽象 | **直接用** | boxing-gym | [envs/goal.py](../boxing-gym/src/boxing_gym/envs/goal.py) | Phase 1 写 Alice-Charlie env 的模板 |
| **L1 表示** | 持久跨域信念网络 | **原创** | — | 新模块 `phase2_prior_library/belief_network.py` | 聚合多个 env 的 BN + 支持图上 belief propagation |
| **L2 度量** | EIG (观测值排序) | **直接用** | boxing-gym | `envs/<env>.expected_information_gain()` ([dugongs.py:74-128](../boxing-gym/src/boxing_gym/envs/dugongs.py#L74-L128)) | NMC 估计公式 + `num_outer=1000, num_inner=10` |
| **L2 度量** | Fisher Information Matrix | **原创** | — | 新模块 `phase2_metric/fisher.py` | jax.hessian on negative log-lik，用 NumPyro trace |
| **L2 度量** | KL 扰动度 $D_{KL}(P(\mathcal{B})\|P(\mathcal{B}\|h))$ | **原创** | — | 新模块 `phase3_embedding/kl_drift.py` | Phase 3 理论核心 |
| **L2 度量** | BIC 分数差（结构惩罚） | **原创** | — | 新模块 `phase3_embedding/bic_score.py` | MEIS_plan.md 3.3 节公式 |
| **L3 结构** | 跨域先验库（骨架 + 存储） | **原创** | — | 新模块 `phase2_prior_library/{human_body, mechanics}.json + retrieval.py` | 最硬骨头。混合策略：核心域 curated JSON + 长尾 LLM 即时 |
| **L3 结构** | 最小扰动评分循环 | **骨架借 AutoToM，语义重写** | AutoToM | 骨架：[model_adjustment.py:60-80](../AutoToM/model/model_adjustment.py#L60-L80) 的贪心循环；重写：`model_space` 换成 BN 结构变体 | AutoToM 的 "add var → compare utility → accept/revert" 循环可复用；变量语义完全替换 |
| **L3 结构** | Markov 范畴态射 + 结构同构识别 | **原创** | — | Phase 4 `phase4_categorical/` | 可推迟到博士论文 paper 3 |
| **L4 调度** | LLM client 抽象 (OpenAI + Anthropic) | **直接用** | boxing-gym | [agents/agent.py:LMExperimenter](../boxing-gym/src/boxing_gym/agents/agent.py#L5) + [agents/base_agent.py](../boxing-gym/src/boxing_gym/agents/base_agent.py) | 已支持异步 + cost 计算 |
| **L4 调度** | Box's Loop 作为 Phase 1 baseline | **直接用** | boxing-gym | [agents/box_loop_experiment.py](../boxing-gym/src/boxing_gym/agents/box_loop_experiment.py) + [agents/model_search.py](../boxing-gym/src/boxing_gym/agents/model_search.py) | 跑通后作为对照组 |
| **L4 调度** | NL → 变量 schema 解析 | **借数据类 + 重写 prompt** | AutoToM | [ElementExtractor.py:7-24](../AutoToM/model/ElementExtractor.py#L7-L24) 的 `Variable` dataclass | Prompt 换成 MEIS 的"给常识先验 + 标单位 + 标依赖"结构化 JSON |
| **L4 调度** | 跨域先验检索（RAG over 先验库） | **原创** | — | `phase2_prior_library/retrieval.py` | 简单 embedding 检索即可 |
| **L4 调度** | 假设提出 + 反思循环 | **借 model_search 骨架** | boxing-gym | [agents/model_search.py](../boxing-gym/src/boxing_gym/agents/model_search.py) | 扩展为"每轮 critic 时检查融贯度" |
| **L5 评测** | BoxingGym 10 envs | **直接用** | boxing-gym | `envs/*.py` | Phase 1-2 默认评测平台 |
| **L5 评测** | bnlearn BN 数据集 | **原创数据适配器** | — | `evaluation/bnlearn_adapter.py` | `.bif` → MEIS 内部四元组 |
| **L5 评测** | HypoBench / DiscoveryBench / POPPER 适配器 | **原创数据适配器** | — | `evaluation/{hypobench, popper}_adapter.py` | Phase 3-4 |
| **L5 评测** | **定律动物园**（结构同构 benchmark） | **原创** | — | `evaluation/law_zoo/` | MEIS_plan.md Phase 5 的核心贡献，本身一篇 benchmark 论文的种子 |

---

## 3. 原创贡献边界（MEIS 从零构建）

以下 **7 个模块**在任何候选 repo 里都不存在，是 MEIS 的原创贡献点，写博士论文时的核心 delta：

1. **跨域先验库结构 + 填充机制**（Phase 2）
   - 核心域 curated（human body、classical mechanics 各 10–20 条）+ 长尾 LLM 即时生成
   - schema: `{domain, from_var, to_var, relation_type, parametric_form, noise, source}`
   - BoxingGym 明确承认 "prior information does not improve performance"，这是 MEIS 与它的 direct delta

2. **最小扰动嵌入评分函数**（Phase 3, 理论核心）
   - $D(h, \mathcal{B}) = D_{KL}(P(\mathcal{B})\|P(\mathcal{B}\|h)) + \lambda \cdot |\Delta_{\text{structure}}|$
   - `|Δstructure|` 用 BIC 分数差操作化
   - 对接 Perrone "Categorical Information Geometry" (2024)

3. **Fisher information 驱动的下一步观测选择**（Phase 2）
   - 不同于 BoxingGym 的 MC-approximated EIG；用 NumPyro trace + jax.hessian 得闭式 Fisher
   - 在 MEIS 持久 BN 上跨域计算 "哪个观测对 target var 最值钱"

4. **持久多域信念网络**（Phase 2+）
   - BoxingGym 每个 env 是一次性的；MEIS 要跨任务累积
   - 图数据结构 + 节点/边的来源追踪（先验库 vs 证据）

5. **Markov 范畴结构同构识别**（Phase 4）
   - 把 "身高→体重→压强→脚印" 和 "电压→电阻→电流→功率→发热" 识别为同一抽象模式
   - 对接 Fritz et al. (2020) Markov 范畴 BSS 定理

6. **定律动物园** IsomorphBench（Phase 5）
   - 4 个等价类，每类 ≥ 3 个同构系统，每系统 50-100 条时间序列
   - 本身可以独立成一篇 benchmark paper

7. **多 benchmark 统一适配器**（Phase 1 起就要写）
   - bnlearn `.bif` + HypoBench JSON + POPPER API + DiscoveryBench schema → 统一四元组
   - 看似琐碎实则节省 6 个月反复适配工作

---

## 4. Phase 1 第一周的具体施工动作

### 第 1-2 天：走通 BoxingGym baseline
```bash
export ANTHROPIC_API_KEY=sk-...
cd /home/erzhu419/mine_code/MEIS/boxing-gym
python run_experiment.py envs=dugongs llms=anthropic seed=0
```
预期：跑通 10 轮实验，拿到 EI Regret 和 prediction error 数字作为对照基线。

### 第 3-4 天：新建 MEIS Phase 1 骨架
```
MEIS/
├── phase1_mvp/                         # 新建，放入 git
│   ├── __init__.py
│   ├── envs/
│   │   └── alice_charlie.py            # 模仿 envs/dugongs.py 写 Alice-Charlie 体重-脚印 env
│   ├── agents/
│   │   └── prior_injecting_agent.py    # 继承 boxing-gym LMExperimenter + 注入先验库
│   ├── variable_schema.py              # 抄 AutoToM/model/ElementExtractor.py:7-24 的 Variable dataclass
│   └── run_phase1.py                   # 仿 run_experiment.py
```

### 第 5 天：跨域先验库 v0
```
phase2_prior_library/                   # 新建
├── human_body.json                     # 10 条：身高~N(170,10), density~N(1010,30), weight=density×k×h³ ...
├── retrieval.py                        # 简单关键词匹配，后面升级 embedding
```

### 第 6-7 天：对照评测
- 在 Alice-Charlie env 上跑三组：
  1. 原版 BoxingGym LMExperimenter（无先验）
  2. BoxingGym + LLM 即时生成先验（no-library baseline）
  3. MEIS prior-injecting agent（curated 先验库 + LLM 补长尾）
- 比 prediction error + EI Regret + 后验熵下降速度

这 7 天的 deliverable = **MEIS 的第一个可被 BoxingGym 基座评测的改装 MVP**。

---

## 5. 决策记录（后续如遇分歧时翻旧账用）

| 决策 | 时间 | 理由 |
|---|---|---|
| 选 BoxingGym 为主基座 | 2026-04-23 | 唯一同时满足：真 PP 后端 / 可跑 CLI / EIG 现成 / Scientist-Novice 评测框架 |
| AutoToM 借 Variable + model_adjustment 骨架，不借 probs.py | 2026-04-23 | probs.py 用 LLM 估 likelihood，违背 MEIS L1 "真贝叶斯" 目标 |
| 排除 language_and_experience | 2026-04-23 | vllm+mpi4py+pygame+VGDL 与 MEIS 信念网络推理错位，依赖太重 |
| 排除 MSA 代码 | 2026-04-23 | 只有 data + prompts，无可执行代码 |
| 保留 world-models 仅作语义参考 | 2026-04-23 | Church archival，已手动 port 到 numpy（见 phase0_smoke_test） |
| 用 NumPyro 做 MEIS 原生 L1，PyMC 只跟 BoxingGym 跑 | 2026-04-23 | NumPyro+jax 支持自动微分，Phase 2 Fisher info 必需 |
| 暂缓 Phase 4 Markov 范畴到博士论文 paper 3 | 2026-04-23 | Phase 1-3 已够一篇博士论文，范畴论层是可选延伸 |
