# LangSplat：开放词汇三维场景理解可视化 Demo

本仓库用于展示基于 LangSplat 框架的开放词汇三维场景理解实践结果。项目围绕自采桌面多视角场景，完成从 COLMAP 几何初始化、Vanilla 3DGS 预训练，到 SAM / CLIP 语义特征提取、三维语言特征场训练和文本查询可视化的基本流程。

> This repository provides a visualization demo for open-vocabulary 3D scene understanding based on LangSplat.

---

## 1. 项目简介

本项目使用 Canon R50 采集桌面多视角图像数据，并基于 COLMAP 完成相机位姿估计与稀疏重建。在 Vanilla 3DGS 重建结果基础上，进一步引入 SAM / CLIP 提取二维视觉语言特征，并训练三维语言特征场，实现基于文本 prompt 的三维目标检索与高亮可视化。

该项目主要用于个人科研准备与工程实践总结，重点关注：

- 多视角图像到三维场景表示的完整流程；
- COLMAP 与 3DGS 在场景重建中的作用；
- SAM / CLIP 视觉语言特征向三维场景中的迁移；
- 开放词汇文本驱动的三维目标查询与高亮可视化。

---

## 2. 数据与场景

- 数据集名称：`my_desk`
- 采集设备：Canon R50
- 图像数量：248 张
- 典型物体：
  - book
  - power bank
  - blue plush toy
  - hand gripper
  - camera lens

---

## 3. 展示内容

网页 Demo 主要包含以下部分：

### 3.1 原始数据采集

展示自采桌面场景的原始视频，用于说明数据来源和场景内容。

### 3.2 3DGS 三维场景可视化

通过浏览器端 3DGS viewer 展示三维场景表示，支持旋转、缩放和平移观察。

### 3.3 GT 与 Vanilla 3DGS 对比

对比真实图像视角下的 Ground Truth 与 Vanilla 3DGS 渲染结果，用于展示基础外观重建效果。

### 3.4 开放词汇语义查询

展示不同文本 prompt 下的三维目标高亮结果，例如：

- `a book`
- `a blue plush toy`
- `a power bank`
- `a black cylindrical camera lens`

---

## 4. 技术流程

整体流程如下：

```text
自采多视角图像
        ↓
COLMAP 位姿估计与稀疏重建
        ↓
Vanilla 3DGS 预训练
        ↓
SAM / CLIP 视觉语言特征提取
        ↓
场景级 Autoencoder 特征压缩
        ↓
3D Language Gaussians 训练
        ↓
开放词汇文本查询与高亮可视化
