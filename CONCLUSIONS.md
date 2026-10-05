# CONCLUSIONS — Qwen-Image 2.1 文本编码器 OpenVINO(CPU) vs ROCm(GPU) 对齐验证

**一句话结论：OpenVINO 实现本身没有缺陷（FP32 下与 PyTorch 逐层对齐到 1e-5 量级）；客户观察到的“25 步相似、40 步明显不同”由 INT8 权重压缩引入的有界量化噪声（~7%），叠加 Qwen-Image 2.1 对微小条件扰动的固有轨迹敏感性所致——与 OpenVINO 无关的等模长随机扰动产生相同甚至更大的图像差异。未检测到任何质量或文字渲染回归。生产可用。**

- 验证对象：Qwen/Qwen-Image-2.1（快照 `d26bb612…`）文本编码器 Qwen3-VL-8B（36 层）
- GPU 参考：AMD Radeon PRO W7900D（gfx1100）+ ROCm 7.14.0 userspace + PyTorch 2.13.0+rocm7.14.0
- CPU 侧：AMD EPYC 9334（Zen4）+ OpenVINO 2026.4.1
- 完整数据/图表/原始证据：`artifacts/`（数值证据与图表已入 git；大体积张量与图像不入 git，可用脚本一键再生）

---

## 1. 五个委托问题的答案（附证据定位）

| # | 问题 | 答案 | 关键证据 |
|---|---|---|---|
| 1 | OpenVINO 是否数值对齐？ | **是**（FP32：rel L2 7.7e-6，cos≈1.0，36 层均匀；BF16 运行时 1.3%） | `artifacts/metrics/layer_alignment_R1_vs_O0_*`、`embedding_summary_stats_*.json` |
| 2 | 分歧从哪里开始？ | INT8 压缩的 **embedding 表**（第 0 层即 3.0%），主干 1.0–1.4%，末层放大 2.8× 至 ~7%；**无单点坏层/坏算子** | `layer_alignment_O0_vs_O3*_summary.json` |
| 3 | 分歧成因归类？ | **量化（Case D）+ 通用轨迹敏感（Case E 机制）**；排除预处理（逐位一致）、转换/运行时（1e-5）、BF16 精度（1.3%）、非确定性（GPU 侧 bitwise 可重复） | 决策树各分支证据见 `reports/TECHNICAL_REPORT.md` §3–§10 |
| 4 | 40 步差异=质量回归吗？ | **不是**。等模长随机扰动效果相同或更强；OCR 文字配对统计无回归（1024：0 坏/6 好；2K Wilcoxon p=0.42；合并符号检验 p=1.0） | `perturbation_*.json`、`image_metrics_*` CSV |
| 5 | 能否生产使用？ | **能**。要求 40 步贴近 GPU 参考时用 OV FP32/BF16；追求吞吐时 INT8 可用（“另一个有效样本”语义） | `reports/CUSTOMER_REPORT.md` §16 |

## 2. 核心数字（全部来自仓库内原始工件，可复算）

**误差链（条件张量 prenorm，median）**

| 比较 | rel L2 | cosine |
|---|---|---|
| GPU 自重复（噪声底） | **0.000000**（bitwise） | 1.000000 |
| **OV FP32 vs GPU FP32** | **7.7e-06** | ≈1.000000 |
| FP16 存储 | 5.7e-05 | ≈1.0 |
| OV 运行时 BF16 | 1.3e-02 | 0.99993 |
| GPU BF16 vs GPU FP32（参考自身精度包络） | 5.3e-02 | 0.9987 |
| **OV INT8 vs OV FP32** | **7.0e-02**（n=100：6.96%±0.1） | 0.9976 |
| **GPU BF16 vs OV INT8（生产总差）** | **9.7e-02**（n=100：9.67%，min cos 0.9954） | 0.9958 |

**机制实验（相同冻结 latent、相同调度、bitwise 确定性后端，唯一变量=embedding）**

| 实验 | 结果 |
|---|---|
| 移植 25 步 | 终态 latent rel L2 **7.6%** |
| 移植 40 步 | 终态 latent rel L2 **18.5%** |
| 交叉（40 步调度，k=25）：25 步后换条件 | 终图仅变 **0.3–0.4%** |
| 交叉：25 步后换累积 latent | 终图变 **18.5–19.5%** |
| 扰动：OV-INT8 delta（40 步） | 18.5% / PSNR 18.5 dB（S02）；18.4% / 21.1 dB（C09） |
| 扰动：等模长随机 delta ×3 种子 | 16.7–23.5% / 18.0–19.6 dB（S02）；**32–47%** / 14.3–16.9 dB（C09） |
| 扰动：0.25×delta | 1.8% / PSNR 37 dB（视觉不可分辨） |

**大规模图像质量（R0 vs OV-INT8 复现 O3r）**

| 套件 | PSNR median | SSIM | LPIPS | 文字 CER 配对 |
|---|---|---|---|---|
| 100 prompts × 25 步 | 28.8 dB | 0.967 | 0.036 | — |
| 100 prompts × 40 步 | 29.3 dB | 0.968 | 0.030 | 0 坏 / 6 好 / 14 平 |
| 2K 生产 30×40 步 @2048² | 27.9 dB | 0.964 | 0.032 | p=0.42（不显著） |

**CPU 性能（EPYC 9334；EXECUTION_MODE=PERFORMANCE；精度结论与之隔离）**

| 档 | 加载 | warm@25tok | warm@171tok |
|---|---|---|---|
| OV FP32 | 34.7 s | 0.83 s | 2.49 s |
| OV BF16 | 12.6 s | 0.39 s | 1.33 s |
| OV INT8 | 1.4 s | 0.22 s | 1.08 s |

## 3. 25 vs 40 步机理（客户核心疑问的直接答案）

1. 25 步调度 ≠ 40 步调度的前 25 步（sigma 网格从第 0 步就不同，已验证记录于
   `artifacts/latents/schedule_overlap_check.json`）。
2. 交叉实验证明：分歧在**早中期（结构形成段）被写入轨迹并锁定**——25 步后更换
   条件几乎不影响终图（0.3%），而更换累积 latent 完全决定终图（18.5%）。
3. 因此同一 embedding 偏差下：更长更细的 40 步调度累积并细化更多差异
   （7.6%→18.5%），越过视觉可分辨阈值；25 步则未越过。

## 4. 附带的重要工程发现

- **客户 INT8 工件 IR 在本 Zen4 主机 >40 token 即失败**（`language_model/aten::add/Add_3`
  eltwise 形状推断错误；此前 Zen5 主机可跑 185 token）。已用同配方本地重 quant
  （O3r，NNCF INT8_ASYM ratio=1.0 group=-1 + 行级 int8 embedding）验证误差特征一致
  （7.0%/9.7% vs 工件 7.0%/9.8%）并用于全覆盖套件。**建议在生产 CPU 上用真实
  prompt 长度复测该工件。**

## 5. 复现（一条命令 / 分阶段）

```bash
bash scripts/run_all_validation.sh        # 全流程（GPU 数小时级）
# 分阶段命令与干净环境准备见 reports/REPRODUCTION.md
python scripts/verify_artifacts.py        # 工件完整性检查（当前 PASS，0 失败）
```

关键单点复现：
```bash
# 语义正确性（bit 级对齐官方 pipeline）
python -m tests.test_conditioning_semantics
# 误差链
python scripts/compare_embeddings.py --pairs R0:R0R R0:R1 R1:O0 O0:O1 O0:O3r R0:O3r --suite canary
# 机制实验
python scripts/crossover_25_40.py --suite canary --pid C09 --steps 40 --k 25 --arm-a R0 --arm-b O3r
python scripts/perturbation_sweep.py --suite smoke --pid S02 --arm-ref R0 --arm-ov O3 --resolutions-steps 1024:25,1024:40
```

## 6. 报告与证据索引

- `reports/EXECUTIVE_SUMMARY.md` — 1 页决策摘要
- `reports/CUSTOMER_REPORT.md` — 面向客户主报告（16 节，全部数字可溯源）
- `reports/TECHNICAL_REPORT.md` — 方法论/公式/命令/原始工件映射
- `reports/REPRODUCTION.md` — 清洁环境逐步复现
- `reports/LIMITATIONS.md` — 偏差与限制（1-seed 套件、单机、人工盲评 PENDING 等）
- `artifacts/plots/*.png` — 13 类关键图表
- `artifacts/metrics/*.csv|json` — 全部数值证据
- `artifacts/reviews/*.md` — 3 次独立子代理验核记录（PASS WITH WARNINGS，警告已整改）
- `reports/gallery_evidence/`（本仓库内含精选图对；`reports/gallery/INDEX.md` 为索引）— 最好/中位/最差/中英文文字/交叉/扰动/2K

## 7. 后续推荐实验的执行结果（已全部完成并经独立验核）

**Step 1 — 精度感知重量化（O3q/O3qL）：负结果但结论明确**
- O3q（仅 embedding 表退出 INT8，第 0 层误差实测降为 0.0000）：总误差 6.97% —— **无改善**
- O3qL（再排除第 36 层）：n=100 总误差 6.10%（vs O3r 6.96%）、对 GPU 参考 9.24%（vs 9.67%）—— 仅边际改善
- 逐层证据：主干 36 层每层 0.7–1.2% 的分布性权重噪声累积才是主体；单点排除无效
- **NNCF 工具链约束（实测复现）**：INT8 模式强制 ratio=1.0、group_size=-1，无混合精度旋钮
- CPU 代价：O3qL warm@25tok 0.239s（O3r 0.221s）
- **⇒ 工程结论：INT8 误差在本工具链内不可调；精度路径 = BF16**

**Step 2 — 图像级确认（canary 40 步三方套件 + 移植轨迹）**

| 臂 | embedding delta (vs R0) | S02@40 终态 rel L2 | canary LPIPS 中位 | **LPIPS 最差** |
|---|---|---|---|---|
| O3r（INT8） | 9.67% | 18.5% | 0.0246 | 0.220 |
| O3qL（最优重化） | 9.24% | 10.3% | 0.0246 | 0.309 |
| **O1（BF16）** | ≈5.4%* | **6.9%** | 0.0231 | **0.098** |

*由 5.28%(R0↔O0) ⊕ 1.28%(O0↔O1) 合成。中位数相近，但 **BF16 把最差情况收紧 2–3 倍**（轨迹敏感的尾部才是生产体验）。注意映射非线性（9.2%→10.3% vs 9.7%→18.5%）：个体轨迹存在折叠，须按分布而非单点判断。

**Step 3 — 2K BF16 抽检（4 个代表性 prompt @2048/40 步，同子集对照）**：R0↔O1 LPIPS 中位 0.031 / 最差 0.132；同 4-prompt 子集的 R0↔O3r 为 0.089 / 0.220（O3r 全 30-prompt 套件中位 0.032 / 最差 0.220）——同子集下 BF16 中位与最差均更优，方向与 1024 一致。

**Step 4 — 客户工件 IR bug：根因 + 运行时修复（O3p）+ 逐位等价证明**
- 根因：导出图烘焙了 visual/deepstack 路径（`NonZero(visual_pos_masks)` 数据依赖 + `Constant 249` 等痕迹期常量）；纯文本输入触发空张量路径，本主机 OpenVINO 2026.4.x 的 eltwise 形状推断拒绝（n>40 必现，部分配置下不定）
- 修复：运行时图手术（`scripts/patch_ov_artifact_textonly.py`）——把 3 个顶层 `Add` 的 visual 分支输入替换为空常量 `[0,4096]`；**客户序列化文件零修改**
- 等价证明：与工件历史冻结输出（首次成功运行所存）在 S01–S03 上 **逐位相同（max_abs=0.0）**
- 全覆盖：真实工件经 O3p 跑通 n=100 套件：O0↔O3p = 6.89%、R0↔O3p = 9.65%；O3p↔O3r 仅差 0.54%（cos 0.99999）——**证实 O3r 复现是忠实代理**
- 附带发现：未打补丁工件在本主机已不可复现首次成功（环境状态漂移），进一步支持“仅作运行时补丁使用”

**最终生产矩阵（实测）**

| 档 | embedding 误差 | 加载 | warm@25tok | 40 步最差 LPIPS | 定位 |
|---|---|---|---|---|---|
| OV FP32 | 0.0008% | 89s* | 0.73s | — | 基准/取证 |
| **OV BF16** | 1.3% (⊕参考包络≈5.4%) | 9.3s | 0.34s | **0.098** | **精度生产档** |
| OV INT8 | 7.0% | 1.4s | 0.22s | 0.22 | 吞吐档（"另一有效样本"） |

*页缓存冷时；首测热缓存 34.7s。

## 8. 原始推荐清单的最终状态

1. ~~精度感知重量化~~ → 已执行：不可行（上表）；改推 BF16
2. 人工盲评 → 包件就绪，仍 PENDING（无人工结果，不造假）
3. 工件 IR bug → 已根因+运行时修复+逐位验证；建议上游以新版本 optimum/NNCF 重导出
4. BF16 过渡方案 → 已在 1024+2K 验证（尾部收紧 2–3 倍）
