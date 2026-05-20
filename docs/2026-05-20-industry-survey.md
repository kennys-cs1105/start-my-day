# 工业缺陷检测领域数据集与SOTA模型综述

> **作者角色**：资深深度学习算法研究员（工业检测）  
> **整理时间**：2025年

---

## 一、常用公开数据集

| 数据集 | 样本数量 | 主要特性 | 链接 |
|--------|----------|----------|------|
| **MVTec AD** | 5354张（含正常+缺陷） | 15类工业品，含纹理+物体类，标准异常检测基准，提供像素级标注 | [mvtec.com](https://www.mvtec.com/company/research/datasets/mvtec-ad) |
| **DAGM 2007** | 约6000张 | 10类人工合成纹理缺陷，弱监督标注，适合纹理缺陷分割研究 | [Kaggle](https://www.kaggle.com/datasets/mhskjelvareid/dagm-2007-competition-dataset-optical-inspection) |
| **NEU Surface Defect** | 1800张（6类×300） | 热轧钢材6类表面缺陷，类间相似度高，适合分类+检测任务 | [faculty.neu.edu.cn](http://faculty.neu.edu.cn/yunhyan/NEU_surface_defect_database.html) |
| **AITEX Fabric** | 245张（含140张缺陷） | 纺织品7类缺陷+正常样本，高分辨率(4096×256)，缺陷稀疏 | [aitex.es](https://www.aitex.es/afid/) |
| **BTAD (BeanTech)** | 2830张（3类工业品） | 真实工业场景，含whole/part/patch三种粒度，近年异常检测新基准 | [Kaggle](https://www.kaggle.com/datasets/thtuan/btad-beantech-anomaly-detection) |

---

## 二、主流 SOTA 模型

### 1. PatchCore（CVPR 2022）

| 属性 | 内容 |
|------|------|
| **研究范式** | 无监督异常检测（Memory Bank） |
| **模型架构** | ImageNet预训练Wide-ResNet50作Encoder，贪心核心集采样构建Memory Bank，KNN检索 |
| **损失函数** | 无显式训练损失（推理阶段计算patch特征与Memory Bank最近邻距离作为异常分数） |
| **核心改进点** | 贪心子采样策略（Greedy Coreset Subsampling）大幅压缩Memory Bank体积，同时保持高覆盖率 |
| **解决的问题** | 解决Memory Bank存储爆炸问题；无需缺陷样本即可达到SOTA，MVTec AUROC 99.6% |
| **论文链接** | [arXiv 2106.08265](https://arxiv.org/abs/2106.08265) |
| **代码链接** | [amazon-science/patchcore-inspection](https://github.com/amazon-science/patchcore-inspection) |

---

### 2. RD4AD（CVPR 2022）

| 属性 | 内容 |
|------|------|
| **研究范式** | 无监督异常检测（知识蒸馏） |
| **模型架构** | Teacher为预训练Wide-ResNet50 Encoder，Student为反向蒸馏Decoder，中间接One-Class Bottleneck（OCBE） |
| **损失函数** | 多尺度余弦相似度损失（Cosine Similarity Loss） |
| **核心改进点** | 逆向蒸馏思路：数据流从Encoder→Bottleneck→Decoder反向传递，正常样本可重建，异常区域重建失败产生差异图 |
| **解决的问题** | 传统正向蒸馏中Student会意外泛化至异常区域的问题；多尺度异常感知弱的问题 |
| **论文链接** | [arXiv 2201.10703](https://arxiv.org/abs/2201.10703) |
| **代码链接** | [hq-deng/RD4AD](https://github.com/hq-deng/RD4AD) |

---

### 3. RD++（CVPR 2023）

| 属性 | 内容 |
|------|------|
| **研究范式** | 无监督异常检测（增强知识蒸馏） |
| **模型架构** | 在RD4AD基础上增加自监督最优传输模块（SOT）和伪异常抑制模块（Simplex Noise增强） |
| **损失函数** | 余弦相似度损失 + 最优传输对齐损失 + 伪异常重建抑制损失（多任务学习） |
| **核心改进点** | 引入特征紧密性约束（SOT）和Simplex Noise合成伪异常用于信号抑制；推理速度比PatchCore快6倍 |
| **解决的问题** | RD4AD特征紧密性不足、瓶颈模块无法显式过滤异常信息的问题；精度与效率之间的权衡 |
| **论文链接** | [CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Tien_Revisiting_Reverse_Distillation_for_Anomaly_Detection_CVPR_2023_paper.html) |
| **代码链接** | [tientrandinh/Revisiting-Reverse-Distillation](https://github.com/tientrandinh/Revisiting-Reverse-Distillation) |

---

### 4. SimpleNet（CVPR 2023）

| 属性 | 内容 |
|------|------|
| **研究范式** | 无监督异常检测（特征适配+判别器） |
| **模型架构** | 预训练Wide-ResNet50提取特征 → 轻量MLP Feature Adapter → 高斯噪声合成伪异常 → 二分类Discriminator |
| **损失函数** | 二分类焦点损失（Focal Loss） |
| **核心改进点** | 在特征空间而非图像空间合成伪异常（加高斯噪声扰动正常特征），训练轻量判别器；全流程端到端，推理77 FPS |
| **解决的问题** | 去除复杂Memory Bank和重建网络；解决图像空间伪异常合成质量低导致判别器泛化差的问题 |
| **论文链接** | [arXiv 2303.15140](https://arxiv.org/abs/2303.15140) |
| **代码链接** | [DonaldRR/SimpleNet](https://github.com/DonaldRR/SimpleNet) |

---

### 5. WinCLIP（CVPR 2023）

| 属性 | 内容 |
|------|------|
| **研究范式** | 零/少样本异常检测（视觉-语言对齐） |
| **模型架构** | CLIP ViT作为双塔Encoder，滑窗多尺度Patch特征提取与聚合；WinCLIP+扩展支持少量正常样本输入 |
| **损失函数** | 无需训练（零样本推理），以CLIP对比预训练对齐损失为基础 |
| **核心改进点** | 组合提示词集成（Compositional Prompt Ensemble）+ 多尺度窗口特征聚合（window/patch/image三级），零样本驱动异常定位 |
| **解决的问题** | 工业场景无标注数据时的零样本检测需求；原始CLIP在异常分类与分割任务上性能差的问题 |
| **论文链接** | [arXiv 2303.14814](https://arxiv.org/abs/2303.14814) |
| **代码链接** | [caoyunkang/WinClip](https://github.com/caoyunkang/WinClip)（非官方，官方未开源） |

---

### 6. UniAD（NeurIPS 2022 Spotlight）

| 属性 | 内容 |
|------|------|
| **研究范式** | 统一多类别异常检测（One-for-all重建） |
| **模型架构** | Transformer架构，Layer-wise Query Decoder + Neighbor Masked Attention + Feature Jittering策略，单模型覆盖15类 |
| **损失函数** | MSE重建损失 + 多层特征对齐损失 |
| **核心改进点** | 邻域掩码注意力（防止模型直接复制输入特征走"捷径"）；特征扰动策略（Feature Jittering）提升重建鲁棒性 |
| **解决的问题** | 传统方法每类独立建模部署成本高；Transformer全局注意力导致异常特征被"修复"而漏检的问题 |
| **论文链接** | [arXiv 2206.03687](https://arxiv.org/abs/2206.03687) |
| **代码链接** | [zhiyuanyou/UniAD](https://github.com/zhiyuanyou/UniAD) |

---

## 三、补充说明

- 当前工业缺陷检测主流范式已从**有监督分割**转向**无监督/零样本异常检测**，核心挑战在于缺陷样本稀缺。
- **MVTec AD** 是事实标准基准，AUROC 和 PRO（Per-Region Overlap）是主流评估指标。
- WinCLIP 官方作者（Amazon Research团队）未公开原始代码仓库，表中链接为社区高质量复现版本（已在CVPR 2023 VAND Workshop广泛使用）。
- 未来趋势：多模态大模型（如CLIP/SAM）与工业先验知识融合、实时部署优化（TensorRT量化）。
