# 工业检测 Top10 论文精读（2026-05-20）

> 来源：`/home/kennys/workspace/start-my-day/industry-top10` 下 10 篇 PDF  
> 分析角色：深度学习算法研究员视角，侧重「怎么做、为什么有效」

---

## 目录

| # | 论文 | 任务 |
|---|------|------|
| 01 | AVA-DINO | 零样本异常检测 |
| 02 | MMVIAD / VISTA | 视频多任务异常理解 |
| 03 | RIDE | 隐蔽目标/缺陷分割 |
| 04 | AnomalyClaw | 跨域 VAD Agent |
| 05 | Architecture-Aware Explanation Auditing | 可解释性审计 |
| 06 | SynSur | 合成缺陷 + 检测 |
| 07 | Network Knowledge Prior | 数据高效缺陷分类 |
| 08 | MixCount | 混合物体计数数据集 |
| 09 | CM3D-AD | 3D 点云异常检测 |
| 10 | AOI-SSL | 半导体线键分割（自监督） |

---

## 01. AVA-DINO — Anomaly-Aware Vision-Language Adapters for Zero-Shot Anomaly Detection

**arXiv**: [2605.12069](http://arxiv.org/abs/2605.12069v1) · **代码**: https://github.com/aqeeelmirza/AVA-DINO

### 任务定位

**零样本异常检测（ZSAD）**：在辅助数据集上训练，在未见类别/域上测试；输出图像级分数 + 像素级异常图。工业 + 医学跨域泛化。

### 核心

| 维度 | 内容 |
|------|------|
| **动机** | 正常样本分布紧凑、异常多样且不对称；现有方法用**单路 adapter** 对所有样本做相同变换，无法同时兼顾「保持结构」与「放大缺陷」。 |
| **创新** | 双分支 anomaly-aware adapter + 文本引导动态路由 + 训练期路由正则。 |
| **框架** | 冻结 DINOv3 提特征 → 正常/异常两路轻量 adapter → CLIP 文本（perfect/damaged + 类名）算路由权重 → 加权融合特征 → 与文本算相似度得异常图。 |
| **关键模块** | ① Normal adapter \(\mathcal{A}_n\)：保结构；② Anomaly adapter \(\mathcal{A}_a\)：强调偏差与边界；③ 文本引导路由：CLS 与投影后文本 embedding 余弦相似度 + temperature softmax；④ 残差：\(f_{adapted} = w_n f_n + w_a f_a + f\)。 |
| **损失** | \(\mathcal{L} = \lambda_1 \mathcal{L}_{seg}\)（Focal + Dice 对像素 mask）+ \(\lambda_2 \mathcal{L}_{global}\)（图像级 CE）+ \(\lambda_3 \mathcal{L}_{routing}\)（MSE 约束 \(w_n \approx 1-y,\, w_a \approx y\)）；\(\lambda_1{=}0.5,\lambda_2{=}0.25,\lambda_3{=}0.1,\,\tau{=}0.1\)。 |

**为什么有效**：训练时用标签把两路「分工」学出来；测试时只靠图像 + 固定文本 prompt 就能偏向合适分支，避免 50/50 塌缩，正常走保结构路、异常走放大缺陷路。

### 实验

- **数据**：ViSA 训练，在 MVTec-AD、BTAD、KSDD2、MPDD、MVTec-AD2 及 Kvasir/CVC 等 9 个 benchmark 零样本测。
- **指标**：I-AUC、P-AUC、Pixel-F1。
- **主结果**：MVTec-AD **93.5% / 92.1% / 50.6%**（I/P/F1），全面优于 WinCLIP、AnomalyCLIP、AdaCLIP、Bayes-PFL；MVTec-AD2 上 I-AUC +5.3pp。
- **消融（MVTec-AD）**：单 adapter 91.2 → 双路+可学习路由 92.0 → +文本路由 92.6 → +路由损失 **93.5**；多层特征比单层 +1.7% P-AUC；测试时正常 \(w_n{=}0.83\)、异常 \(w_a{=}0.79\)。

### 总结

| | |
|---|---|
| **问题** | 统一 adapter 无法建模正常/异常不对称性。 |
| **方法** | DINOv3 + 双路 V-L adapter + 文本路由 + 路由正则。 |
| **结果** | 多工业/医学数据集 SOTA 级零样本表现。 |
| **优点** | 可解释路由、28M 可训练参数、跨域强。 |
| **缺点** | 依赖 CLIP 文本模板与辅助集 mask；复杂纹理仍难（MPDD 等）。 |
| **复现** | DINOv3-ViT-L/16 + CLIP-ViT-L/14-336；512²；AdamW 1e-4；batch 64；20 epoch；RTX 4090。 |

---

## 02. MMVIAD / VISTA — Multi-view Multi-task Video Understanding for Industrial Anomaly Detection

**arXiv**: [2605.10833](http://arxiv.org/abs/2605.10833v1) · **代码**: https://github.com/Georgekeepmoving/MMVIAD

### 任务定位

**工业视频异常理解**：连续多视角（约 120°）2 秒巡检片段上的 **检测 / 缺陷分类 / 物类分类 / 异常可见时段定位** 四任务 QA。偏 **VLM + 数据集 + 后训练**，非传统像素级 AD。

### 核心

| 维度 | 内容 |
|------|------|
| **动机** | 真实产线相机绕件运动，结构缺陷常「某视角、某时刻才可见」；静态图或稀疏多视角 benchmark 无法评估时序可见性。 |
| **创新** | 首个连续多视角工业异常视频集 + 可见时段标注；VISTA 两阶段后训练（PS-SFT → VISTA-GRPO）。 |
| **数据构建** | Anomaly-ShapeNet 3D → Blender 渲染（无标记/红标双视频对齐）→ 帧对比得候选可见区间 → 人工校验 → 4 类 QA，共 **4014 clips / 16092 QA**。 |
| **VISTA 训练** | **PS-SFT**：Gemini 3.1 Pro 按 GT 生成结构化推理 trace，标准 CLM 损失；**VISTA-GRPO**：格式奖励 + 答案正确性 + **语义门控缺陷分类奖励** \(R_{sg}\) + **可见性时序奖励** \(R_{vis}\)（IoU + 区间数惩罚），组内相对优势 + GRPO 更新。 |
| **基座** | Qwen3-VL-8B；训练 1008×560、16 帧采样。 |

**为什么有效**：把「何时能看见缺陷」显式标成监督/奖励，逼模型把感知、语义、时间绑在一起；RL 奖励针对工业七类缺陷与空区间幻觉分别设计。

### 实验

- **协议**：MMVIAD-Standard（48 类同分布）；MMVIAD-Unseen（36 训 / 12 未见类）。
- **指标**：各任务准确率 + 定位 mIoU；**Avg** = 四任务算术平均。
- **Standard**：专家 Avg **86.3**；最强商用 Gemini 3.1 Pro **61.8**；开源 Qwen3-VL-8B 仅 **41.6**；细粒度缺陷分类与可见时段定位是瓶颈。
- **Unseen**：VISTA **57.5**（基座 45.0），超过 GPT-5.4（53.5）；检测 60.7%、定位 49.6%、物类 81.9%。
- **消融**：直接 RL 几乎无效；PS-SFT 6k→12k 提升缺陷类；PS-SFT + VISTA-GRPO 最佳；去掉 \(R_{sg}\) 或 \(R_{vis}\) 分别伤分类/定位。
- **迁移**：VISTA 在 MVTec-AD 图像 QA 上 43.8→**67.1** Avg。

### 总结

| | |
|---|---|
| **问题** | 视频 MLLM 不会做「视角依赖的结构缺陷 + 何时可见」。 |
| **方法** | 合成多视角视频数据 + 结构化 SFT + 工业定制 GRPO 奖励。 |
| **结果** | 大幅缩小与专家差距，Unseen 上超 GPT-5.4。 |
| **优点** | 任务耦合设计合理；可复现渲染管线。 |
| **缺点** | 合成域与真实产线仍有 gap；依赖大模型 API 造 SFT 数据。 |
| **复现** | 4×H100、约 6h；16/32 帧评测需与论文一致。 |

---

## 03. RIDE — Retinex-Informed Decoupling for Exposing Concealed Objects

**arXiv**: [2605.15450](http://arxiv.org/abs/2605.15450v1)

### 任务定位

**密集预测 / 分割**：伪装目标检测（COD）、息肉、透明物体、**工业缺陷（CDD）** 等「视觉纠缠」场景；也可作 +R 插件用于阴影、红外等小目标等。

### 核心

| 维度 | 内容 |
|------|------|
| **动机** | 纠缠发生在合成图 \(I{=}L\odot R\) 上，但 \(L\) 与 \(R\) 不必同时匹配；**可辨性间隙定理**：Retinex 分解在多数物理情形下保持或提高 FG-BG 可辨性。 |
| **创新** | 同域（homogeneous）Retinex 分解 + 任务驱动学习，相对傅里叶/小波等异域分解保留像素对齐。 |
| **框架** | **TRD**：U-Net 学 \(L,R\)；**三视图编码**（\(I,L,R\) 共享 ResNet + 每视图 adapter）；**DGA**：估计局部可辨性增益，无增益时抑制分量特征；**渐进解码** + 深监督。 |
| **损失** | Retinex 重建 + 光照平滑 + **互斥边缘** \(\mathcal{L}_{ME}\)（边只属于 \(L\) 或 \(R\)）+ 反射率边界头 BCE + 分割 BCE/IoU + **CBCL**（在 \(R\) 特征空间 InfoNCE，两增强视图）。总损失 \(\mathcal{L} = \mathcal{L}_{seg} + \lambda_{ME}\mathcal{L}_{ME} + \lambda_B\mathcal{L}_B + \lambda_C\mathcal{L}_{CBCL}\)。 |

**为什么有效**：工业/伪装场景里光照与材质常反相关；在 \(R\) 空间对比学习避免 \(I\) 空间正负特征塌缩；DGA 只在「分解确实带来增益」的局部用分量特征。

### 实验

- **COD**：COD10K 上 \(S_\alpha\) **0.834**（ResNet50），NC4K **0.858**；DINOv2-L 达 **0.921** \(S_\alpha\)。
- **息肉 / 玻璃 / 缺陷**：Tables 3–5 均优于 RUN 等 SOTA；CDD（工业）\(S_\alpha\) **0.609** vs RUN 0.590。
- **效率**：352²、RTX4090 上 **27.2 FPS**（ResNet50），参数量约 28.5M，显著快于 FocusDiff 等。
- **消融**：换 FFT/小波/DCT 固定 Retinex 均掉点；\(I\) 空间对比学习反而变差；DGA 优于 concat/SE/Cross-Attn。

### 总结

| | |
|---|---|
| **问题** | 在「已纠缠」的 RGB 上硬找边界。 |
| **方法** | 任务驱动 Retinex + 可辨性注意力 + \(R\) 空间对比。 |
| **结果** | 四 COS 任务 SOTA，+R 插件可迁移六类其他分割。 |
| **优点** | 理论清晰、工业缺陷含在统一框架内。 |
| **缺点** | 需端到端训练 TRD；极端光照需调 \(k,\tau\)。 |
| **复现** | 352²、bs=36、lr=1e-4、每 80 epoch ÷10；TRD 通道 {32,64,128,64,32}。 |

---

## 04. AnomalyClaw — Universal Visual Anomaly Detection Agent via Tool-Grounded Refutation

**arXiv**: [2605.10397](http://arxiv.org/abs/2605.10397v1)

### 任务定位

**跨域图像级异常检测 Agent**（非训练检测头）：工业、基建、遥感、医学、道路等 **12 域**；输入测试图 + 正常参考图，输出异常分数。属于 **VLM + 工具调用 + 反驳式推理**。

### 核心

| 维度 | 内容 |
|------|------|
| **动机** | 单次 VLM 推理依赖先验，少做与正常样本的细粒度对照；已有 Agent 多「累积支持证据」，确认偏误严重。 |
| **创新** | **无训练**、骨干无关；每样本 **K=5 轮反驳**：提出可疑特征 → 工具验证 → 对照参考图**证伪**；并行 Direct 分支融合。 |
| **流程** | ① Suspect list（≤3 特征）；② 从 13 工具 L1–L7 选一项（image_diff、rotate_align、expert_score、domain_knowledge 等，按域硬门控）；③ verdict：found_in_ref / not_found / inconclusive → 更新分数；④ \(s_{final} = 0.5 s_D + 0.5 s_R\)。 |
| **可选 OSR** | Direct 与反驳分支差 \(|s_D-s_R|>0.3\) 的样本入队，满 10 条由 reflector VLM 写域规则，RAG 检索后用于后续样本（零 GT 自进化）。 |

**为什么有效**：把 AD 拆成「定位候选 → 与正常库比对 → 域知识 → 打分」；工具强制**证伪**而非只找支持；难域（裂缝、CT、逻辑异常）提升最大。

### 实验

- **数据**：CrossDomainVAD-12，1418 test items，每域 4-shot 正常参考。
- **指标**：逐域 AUROC、Macro AUROC；bootstrap 显著性。
- **主结果**（Macro AUROC）：GPT-5.5 **0.751→0.814**（+6.23pp）；Seed2.0 **0.688→0.767**（+7.93pp）；Qwen3.5-VL-27B **0.713→0.748**（+3.52pp），均 95% 显著。
- **对比**：优于 SubspaceAD/AnomalyDINO 等训练-free 方法（在强骨干上）；工业微调 AD-Copilot 跨域反而掉点。
- **MMAD**：GPT-5.5 上 76.20→**79.15%** 平均 MC 准确率。
- **成本**：约 3.07–3.42 次 VLM 调用/样本（并行墙钟 ≈ max(Direct, Agent)）。

### 总结

| | |
|---|---|
| **问题** | 通用 VLM 跨域 AD 不可靠且不可验证。 |
| **方法** | 工具接地反驳 Agent + Direct 融合。 |
| **结果** | 三骨干一致显著提升；难域增益集中。 |
| **优点** | 无需训练；行为可诊断（工具/轮次/verdict）。 |
| **缺点** | Token 成本高；部分域（D5/D7）仍可能回退。 |
| **复现** | 固定 α=0.5、K=5、温度 0；注意各域工具可用性表。 |

---

## 05. Architecture-Aware Explanation Auditing for Industrial Visual Inspection

**arXiv**: [2605.14255](http://arxiv.org/abs/2605.14255v2)

### 任务定位

**可解释性 / 模型审计**（非检测本身）：半导体 **WM-811K 晶圆图** 9 类分类上，检验「热力图是否真驱动预测」。方法学论文，MVTec AD 为边界探索。

### 核心

| 维度 | 内容 |
|------|------|
| **动机** | 工业现场常「先训最强分类器，再贴 Grad-CAM」；默认解释器与架构无关，热力图可能好看但不忠实。 |
| **假说（Native-readout）** | 解释方法的忠实度受其与模型**原生决策机制**的结构距离约束：读 attention 矩阵（ViT Rollout）比读梯度激活图（CNN Grad-CAM）更贴 forward path。 |
| **审计协议** | 四族：ResNet18+CBAM、DenseNet121、Swin-Tiny、ViT-Tiny；各配「最自然」解释器 + 统一 **Deletion/Insertion AUC**（zero-fill 扰动）+ Stability；RISE 作黑盒对照。 |
| **指标含义** | **Deletion AUC↓** = 按热力图删像素后置信度掉得快 → 更忠实；Insertion AUC↑ 同理。 |

**为什么有效（反直觉点）**：ViT 分类准确率低于 CNN，但 Attention Rollout 的 Deletion AUC **0.211**，CNN Grad-CAM **0.495–0.525**——说明「准」≠「解释对」；Swin 虽为 Transformer 但因层次特征图仍适合 Grad-CAM（Del 0.432），验证关键在 **readout 结构** 而非 CNN/Transformer 标签。

### 实验

- **数据**：WM-811K 172k 有标签图，64×64；3 seed × 4 族。
- **分类**：CNN Balanced Acc ~0.90；ViT **0.799**（更差）。
- **忠实度（Table 4）**：ViT Del **0.211** / Ins **0.693**；ResNet **0.495** / **0.534**；DenseNet **0.525** / **0.511**；Swin **0.432** / **0.480**。Cohen’s d > 1.1。
- **RISE 对照**：四族 Del 均 ~0.09–0.13，原生方法差距被压缩 → 差距主要在解释通路而非表示本身。
- **MVTec AD**：RGB、预训练、单 seed；结论依赖数据集，不能外推 WM-811K 排序。

### 总结

| | |
|---|---|
| **问题** | 工业 XAI 缺定量「解释是否忠实」审计。 |
| **方法** | 架构–解释器配对 + 扰动忠实度 + 统计检验。 |
| **结果** | ViT+Rollout 忠实度显著优于 CNN+Grad-CAM。 |
| **优点** | 可操作的选型指南（Industry 5.0 人机协同）。 |
| **缺点** | 扰动协议（zero-fill vs blur）影响排序；非因果解释。 |
| **复现** | 200 平衡测试样本/seed；20 步 deletion；配置写 config.yaml。 |

---

## 06. SynSur — Synthetic Industrial Surface Defect Generation and Detection

**arXiv**: [2604.26633](http://arxiv.org/abs/2604.26633v1)

### 任务定位

**数据合成 + 监督缺陷检测**：缓解标注稀缺。管线级工作（VLM prompt → LoRA 扩散 → inpainting → 过滤 → 自动标注 → 训练检测器）。

### 核心

| 维度 | 内容 |
|------|------|
| **动机** | 真实缺陷少、标贵；3D 渲染控型但 sim2real 难；通用扩散不懂工业表面纹理。 |
| **管线** | Qwen2-VL 从缺陷 patch 抽 tag 组 prompt → **Flux.1-dev + LoRA**（随机 100 patch 或按尺寸分桶三 LoRA）→ 掩码引导 inpainting → **DreamSim + CLIPScore** 过滤 → SAM3 分割 + 高斯融合 → **COCO 格式**自动框。 |
| **检测训练** | YOLOX / YOLOv26 / LW-DETR；配置：仅真实、仅合成、混合比例、union（真实+合成总量）。 |

**为什么有效**：LoRA 把生成对齐 BSData 点蚀形态；过滤去掉「像 prompt 但不像真缺陷」的样本；**合成最适合作增广**而非单独训练。

### 实验

- **BSData**：1035 图（点蚀）；主测 YOLOv26 AP。
- **主结果**：Real-only AP **0.652**；最佳 **100%R+100%S union、150 epoch：0.667**；75/25 混合 0.655；Synthetic-only **0.393**。
- **LW-DETR**：Real-only **0.666** 已最佳，合成帮助不明显（架构依赖）。
- **MSD 迁移**：Real AP 0.953，合成混合略降或持平；无 LoRA 的 Flux 生成明显域外。
- **LoRA 选择**：大尺寸分桶 LoRA DreamSim 最优，但操作选 **random LoRA**（CLIP 更高、流程简单）。

### 总结

| | |
|---|---|
| **问题** | 工业缺陷训练数据不足。 |
| **方法** | 端到端 VLM+LoRA 扩散合成与自动标注。 |
| **结果** | 真实数据稀缺时混合/union 可小幅提升 YOLO 系；纯合成不够。 |
| **优点** | 模块可换、指标（DreamSim/CLIP）可指导质控。 |
| **缺点** | 检测器敏感；跨数据集需改标注策略（MSD 不用 SAM3）。 |
| **复现** | 1024² patch；Flux+LoRA 2000 step；1000 候选→过滤池。 |

---

## 07. Network Knowledge Prior Guided Learning for Data-Efficient Surface Defect Detection

**arXiv**: [2605.17780](http://arxiv.org/abs/2605.17780v1)

### 任务定位

**图像级表面缺陷分类**（KolektorSDD / KSDD2）；弱监督/小样本：用**模型自己的显著图**当空间先验，无额外人工 mask。

### 核心

| 维度 | 内容 |
|------|------|
| **动机** | 缺陷少、类不平衡、CNN 易盯背景纹理；Grad-CAM 等多为事后解释，未参与训练。 |
| **创新** | 两阶段：① 训分类网络 → FullGrad/LayerCAM 生成显著图存库；② 同结构网络多任务：分类 + 用显著图伪标签做**分割辅助**（Otsu 二值化）。 |
| **机制** | 显著图拼到特征 \(P(x)=[F(x),H(x)]\) → 分割子网 → GMP+GAP 注入分类头；\(\mathcal{L}_{total} = \lambda \mathcal{L}_{seg} + (1-\lambda)\mathcal{L}_{cls}\)，\(\lambda\) 随 epoch 从 1 衰减到 0。 |
| **骨干** | VGG16 / ResNet 等，与具体架构无关；**推理零额外开销**。 |

**为什么有效**：把「模型第一轮认为重要的区域」固化成空间正则，逼第二轮关注缺陷区；分割权重递减避免错误伪标签过拟合。

### 实验

- **KSDD（Na=0 无 mask）**：Ours **AP 100%** vs MaMiNet 98.5%、MixSup 93.4%。
- **KSDD2（Na=0）**：Ours **93.94%** vs MaMiNet 80.0%、MixSup 73.3%。
- **消融**：无先验 KSDD 84.76%；FullGrad vs LayerCAM 接近；\(\lambda\) 衰减策略关键。

### 总结

| | |
|---|---|
| **问题** | 少样本分类易过拟合背景。 |
| **方法** | 显著图先验 + 多任务渐进损失。 |
| **结果** | 无 mask 设定下大幅超半监督 SOTA。 |
| **优点** | 实现简单、推理不增耗。 |
| **缺点** | 两阶段训练；伪标签质量依赖第一阶段。 |
| **复现** | KSDD 512×1408，KSDD2 224×640；SGD lr 5e-4；分割 batch 4。 |

---

## 08. MixCount — Bridging the Data Gap for Open-Vocabulary Object Counting

**arXiv**: [2605.18063](http://arxiv.org/abs/2605.18063v1)

### 任务定位

**开放词汇物体计数**（非传统缺陷检测，但与产线分拣、质检计数强相关）：**混合多类同图**场景下现有计数模型失效。

> 注：本地 markdown 仅含摘要（PDF 拉取曾失败）；以下基于摘要与 arXiv 信息。

### 核心

| 维度 | 内容 |
|------|------|
| **动机** | 真实计数标注贵、有噪声；合成数据多样性不足；**mixed-object** 是工业常见失败模式。 |
| **创新** | **MixCount** 数据集 + 自动生成管线：图像 + 细粒度文本 + 像素级计数标注，消除计数歧义。 |
| **方法要点** | 规模化合成可无限扩标注数据；用于暴露 SOTA 在 mixed 设定下的退化。 |

### 实验（摘要）

- 在 MixCount 上评测 SOTA → mixed 设定性能**严重下降**。
- 用合成数据训练 → **FSC-147 MAE −20.14%**，**PairTally −18.3%**。
- 结论：数据瓶颈是计数模型长期短板；管线可缓解。

### 总结

| | |
|---|---|
| **问题** | 混合物体计数缺高质量训练/评测数据。 |
| **方法** | 自动合成 MixCount + benchmark。 |
| **结果** | 合成训练显著改善真实 benchmark。 |
| **优点** | 面向工业失败模式；标注清晰。 |
| **缺点** | 本地需补读全文以复现管线细节。 |
| **复现** | 运行 `2026-05-19-industry-top10/download_papers.sh` 获取 PDF。 |

---

## 09. CM3D-AD — Two Steps Are All You Need: Efficient 3D Point Cloud Anomaly Detection

**arXiv**: [2605.05372](http://arxiv.org/abs/2605.05372v1)

### 任务定位

**3D 点云异常检测**（重建残差类）：工业质检、无人机/边缘相机；强调 **低延迟、无 GPU 迭代扩散**。

### 核心

| 维度 | 内容 |
|------|------|
| **动机** | R3D-AD 等扩散方法精度好但推理要数百步，RPi/Jetson 上不可用。 |
| **创新** | **Consistency Model** 将异常检测表述为「单步/两步流形投影」：从任意噪声级直接映射到正常点云。 |
| **框架** | Patch-Gen 合成训练异常 → PointNet 编码形状条件 \(c\) → PointwiseNet（6×ConcatSquashLinear）在线/EMA 双网络 → 测试重建 → 点级残差 + FAISS-NN 得 image-level 分数。 |
| **损失** | \(\mathcal{L}_{Hybrid} = \mathcal{L}_{CT} + \lambda(\mathcal{L}_{Online}+\mathcal{L}_{Target})\)：一致性项 + 对**干净点云** \(x_{raw}\) 的 MSE 重建（保证去异常）。 |

**为什么有效**：Hybrid 损失同时约束跨时间步一致性与重建到正常流形；两步采样在 3D 上足够，再多步收益边际。

### 实验

- **数据**：Anomaly-ShapeNet（40 类）、Real3D-AD（12 类真实工业）。
- **指标**：I-AUROC（per category + Average）。
- **主结果**：Anomaly-ShapeNet Avg **0.762** vs R3D-AD **0.749**；Real3D-AD **0.728** vs **0.734**（略低 ~0.6pp）。
- **效率**：RPi4 推理 **6.34s vs R3D-AD 424.5s**（~80×）；Jetson **1.9s vs 101.3s**；FLOPs 1.72G vs 135G。
- **消融**：去掉 Online 或 Target 重建项 I-AUROC 降至 0.725/0.689；完整 Hybrid **0.762**。

### 总结

| | |
|---|---|
| **问题** | 3D AD 扩散模型无法在边缘实时跑。 |
| **方法** | 条件 Consistency Model + Patch-Gen + Hybrid 损失。 |
| **结果** | 精度持平或略优/略逊，速度数量级提升。 |
| **优点** | 真正面向产线/无人机部署。 |
| **缺点** | Real3D-AD 略逊 R3D-AD；每类需长训练（800k iter）。 |
| **复现** | 2×A100；σ_data=0.5；batch 128；Karras 噪声 schedule ρ=7。 |

---

## 10. AOI-SSL — Self-Supervised Segmentation of Wire-bonded Semiconductors

**arXiv**: [2605.12430](http://arxiv.org/abs/2605.12430v1)

### 任务定位

**半导体 AOI 语义分割**（多标签：wire / ball / wedge / epoxy）；**双通道单色**大图；小数据 + **测试时检索适配**，减少每款芯片重训。

### 核心

| 维度 | 内容 |
|------|------|
| **动机** | 每换器件或成像条件就要重新标数据、训分割网；ImageNet 预训练与 AOI 域差距大。 |
| **创新** | ① 域内 **MAE** 预训练（小数据下优于 DINO/iBOT）；② **Patch 级检索**：Faiss 近邻聚合标签，几乎无需再训；③ FasterViT 层次结构利于细丝状 wire。 |
| **预训练** | ViT-Tiny / FasterViT-0；mask ratio 0.7；MSE 重建；3000 epoch；领域增广（翻转、模糊、噪声等）。 |
| **微调** | UPerNet 头；**cosine + 层衰减 LR**（mIoU 53.7 vs 无 LLRD 33.7）；固定 8h RTX2080 预算；类-wise BCE（多标签重叠）。 |
| **检索** | 记忆库存 patch embedding；相似度加权投票；per-class 阈值与 k 网格搜索。 |

**为什么有效**：MAE 像素重建适合低语义多样性、高几何精度的 AOI；检索对 **wedge** 等少样本细结构比 U-Net 解码器更稳（单设备 71.5% mIoU vs ResNet18+UNet++ 47.5%）。

### 实验

- **数据**：预训练 1.5k 无标；微调 625 有标（4 类前景）；严格无图泄漏。
- **主结果（Table 2）**：FasterViT-0 + UPerNet **60.3% mIoU**（+40.9% vs 监督基线）；ViT-Tiny + UPerNet 53.5%（+50.7%）；Patch 检索 **48.1% mIoU**、266.7 crops/s。
- **类级**：Wire 66.7%、Wedge 仍难（19.3% 微调 vs 检索 78.7% 单设备）。
- **SSL 对比（检索）**：MAE patch 45.7% > DINO 39.3% > iBOT 33.2%；**相似度聚合 ≥ 注意力聚合**（45.7 vs 44.7）。
- **DINOv2 冻结检索**：47.8% mIoU，但需 142M 图预训练且更慢。

### 总结

| | |
|---|---|
| **问题** | 线键 AOI 分割换型成本高。 |
| **方法** | 小域 MAE + 限时微调 + patch 检索。 |
| **结果** | 8h 内 mIoU 大幅提升；检索可快速适配单设备。 |
| **优点** | 贴合产线算力与双通道成像。 |
| **缺点** | ViT patch 边界锯齿；纯检索对 epoxy 等仍弱于微调。 |
| **复现** | MAE 0.7 mask；Faiss GPU；阈值 0.5（微调）/ CV 搜（检索）。 |

---

## 横向对照（落地选型）

| 场景 | 可优先关注 |
|------|------------|
| 新品类零样本瑕疵 | 01 AVA-DINO |
| 产线环绕视频巡检 | 02 MMVIAD / VISTA |
| 低对比度表面/伪装缺陷分割 | 03 RIDE |
| 多行业统一 AD、可审计推理 | 04 AnomalyClaw |
| 上线前解释性合规 | 05 原生 readout 审计 |
| 缺陷样本极少、需扩数据训检测器 | 06 SynSur（混合训练） |
| 只有图像级标签的分类 | 07 显著图先验 |
| 分拣计数、多类混放 | 08 MixCount |
| 3D 扫描件、边缘算力 | 09 CM3D-AD |
| 半导体线键、换型频繁 | 10 AOI-SSL |

---

*生成日期：2026-05-20*
