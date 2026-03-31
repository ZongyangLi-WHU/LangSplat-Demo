# 🚀 LangSplat-Demo: 3D Language Field Reconstruction

[![WebGL Supported](https://img.shields.io/badge/WebGL-Supported-brightgreen)](https://zongyangli-whu.github.io/LangSplat-Demo/)
![3DGS](https://img.shields.io/badge/Algorithm-3D%20Gaussian%20Splatting-blue)
[![Technical Blog](https://img.shields.io/badge/Blog-Zhihu%20Article-orange)](https://zhuanlan.zhihu.com/p/2020999456775045188)

> **基于 3D Gaussian Splatting 的跨模态交互式语言场重建系统**
> 
> 本项目为计算机视觉课程期末大作业。实现了从真实物理世界视频采集，到三维场景重建，再到“开放词汇(Open-Vocabulary)”自然语言交互的端到端全栈落地，并成功突破限制，部署于纯静态 Web 端。

## 🌟 核心链接 (Core Links)

- **[🖥️ 交互式 3D 网页 Demo (Live Web Demo)](https://zongyangli-whu.github.io/LangSplat-Demo/)** —— *无需安装任何环境，直接在浏览器中体验 3D 漫游与语义高亮！*
- **[📝 全栈开发技术博客 (Technical Blog)](https://zhuanlan.zhihu.com/p/2020999456775045188)** —— *从数据采集、COLMAP 匹配、模型训练到踩坑 WebGL 部署的万字深度复盘。*
- **[📺 YouTube 项目展示视频 (Demo Video)](https://youtu.be/K092dP0SfVg)** —— *（待更新）完整功能演示与效果对比。*

## 💡 项目亮点 (Highlights)

1. **真实场景重建**：跨越室外庞大背景干扰，利用 3D Bounding Box 切割算法精准提取桌面主体模型。
2. **跨模态语义交互**：结合 CLIP 提取语言特征，支持通过文本 Prompt 在 3D 空间内精准高亮特定物理对象（如："a blue plush toy"）。
3. **顶会级 Web 部署**：纯手工编写 Python 压缩脚本，将数百兆的 `.ply` 点云转换为极轻量级、强制小端序的 32-byte `.splat` 二进制流，完美兼容 WebGL 引擎渲染。

## 📂 仓库结构 (Repository Structure)

```text
├── my_desk.splat               # 经过空间裁剪与智能抽样的超轻量级 3DGS 模型
├── index.html                  # 结合 Three.js 与 GSViewer 的前端展示页面
├── *_web.mp4                   # 经过多维度高压处理的 H.264 网页端展示视频
└── (其他相关代码或脚本视情况上传)
