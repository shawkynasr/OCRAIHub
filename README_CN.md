# OCRAIHub (多引擎 AI OCR 文档解析工具)

![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-green.svg)
![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey)

**OCRAIHub** 是一款功能强大的多引擎桌面端 GUI 应用程序。它利用世界上最先进的视觉语言模型 (VLM)，能够从图像和 PDF 中高精度地提取文本、Markdown 表格、数学公式和图表，实现完美的文档数字化。

目前支持 **Google Gemini (2.5/3.1)** 和 **百度 PaddleOCR-VL (1.5)**。

---

## ✨ 核心功能

*   **🤖 多引擎支持：** 在 Google Gemini API 和百度 AI Studio PaddleOCR 接口之间无缝切换。
*   **📊 智能 Markdown 格式化：** 完美重构文档排版，包括标题、段落、复杂表格和 LaTeX 数学公式。
*   **📄 原生 PDF 与页面切分：** 
    *   *本地模式：* 使用 PyMuPDF 在本地将 PDF 转换为图像，支持高级图像预处理（Otsu 二值化 & 形态学降噪）。
    *   *原生模式（百度）：* 将原始 PDF 文件直接发送至云端引擎进行原生解析。
*   **🖼️ 自动下载插图：** 自动检测、裁剪并下载文档中的图表、图片，保存至本地文件夹。
*   **✂️ 智能分栏切割：** 智能分割多栏文档（如词典或学术论文），防止文本混版。
*   **💾 自动保存与批量处理：** 自动处理包含上百张图片的文件夹或超大 PDF 书籍，内置 API 频率限制处理机制，并自动保存为 `.md` 或 `.csv` 格式。

---
## Screenshots
<img width="3360" height="1898" alt="Gemini OCR" src="https://github.com/user-attachments/assets/a23e11f7-9404-4306-a695-c040ee04e366" />
<img width="3360" height="1896" alt="Baidu VL 1 15" src="https://github.com/user-attachments/assets/e7efe9cb-d31a-447d-b370-c421121b92c9" />

---

## 🚀 安装指南

### 1. 安装依赖
请确保您的电脑已安装 Python 3.8+。在终端运行以下命令：
```bash
pip install PySide6 pymupdf pandas opencv-python pillow requests google-genai
```

### 2. 运行程序
```bash
python OCRAIHub.py
```

---

## ⚙️ API 配置方法

### Google Gemini
1. 前往 [Google AI Studio](https://aistudio.google.com/app/apikey) 免费获取 API Key。
2. 将 Key 粘贴至软件，选择所需模型（如 `gemini-flash-latest`），并输入您的提示词 (Prompt)。

### 百度 PaddleOCR-VL
1. 前往 [Baidu AI Studio](https://aistudio.baidu.com/) 并部署 `PaddleOCR-VL` 线上服务。
2. 复制您的 **Token** 和 **自定义 API URL**（例如：`https://aistudio.baidu.com/serving/online/xxxx?xxxx`）。
3. 选择所需的百度任务模式（OCR, Table, Formula, 或 Chart）。
---    
## 🙏 致谢

本工具的诞生离不开开源社区与 AI 领域的杰出贡献：
*   特别感谢 the-cataloger 开源的 [Gemini-OCR-Arabic](https://github.com/the-cataloger/Gemini-OCR-Arabic) 项目。该项目是本工具早期版本的基础分支与灵感来源。
*   [Google Gemini](https://deepmind.google/technologies/gemini/) 提供了前沿的多模态 VLM API。
*   [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) 百度开源的业界领先的版面分析与文档解析模型。
*   [PyMuPDF (Fitz)](https://github.com/pymupdf/PyMuPDF) 提供了极速的本地 PDF 渲染功能。
*[PySide6](https://doc.qt.io/qtforpython-6/) 提供了稳定强大的 GUI 框架。
---  
## 📝 引用 (Citation)

如果您在研究或项目中使用了本工具，欢迎引用：

```bibtex
@software{Nasr_OCRAIHub_2026,
  author = {Nasr, Shawky},
  title = {OCRAIHub: Multi-Engine AI Document Parsing Tool},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/shawkynasr/OCRAIHub}
}
```
## 📜 开源协议
本项目采用 **GNU General Public License v3.0** 开源协议 - 详情请参阅 [LICENSE](LICENSE) 文件。
