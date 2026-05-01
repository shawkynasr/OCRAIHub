# OCRAIHub (Multi-Engine AI OCR)

![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-green.svg)
![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey)

[中文文档](RREADME_CN.md)  &  [تعريف باللغة العربية](README_AR.md)

**OCRAIHub** is a powerful, multi-engine Desktop GUI application designed to extract highly accurate text, markdown tables, math formulas, and charts from images and PDFs. It leverages the world's most advanced Vision-Language Models (VLMs) to provide perfect document digitization.

Currently supports **Google Gemini (2.5/ 3.1)** and **Baidu PaddleOCR-VL (1.5)**.

---

## ✨ Key Features

*   **🤖 Multi-Engine Support:** Seamlessly switch between Google Gemini API and Baidu AI Studio PaddleOCR endpoints.
*   **📊 Smart Markdown Formatting:** Perfectly reconstructs document layouts, including headers, paragraphs, complex tables, and LaTeX math formulas.
*   **📄 Native PDF & Page Slicing:** 
    *   *Local Mode:* Rasterize PDFs locally using PyMuPDF, with advanced image pre-processing (Otsu Binarization & Morphological Denoising).
    *   *Native Mode (Baidu):* Send raw PDFs directly to the engine for native rendering.
*   **🖼️ Auto-Download Figures:** Automatically detects, crops, and downloads charts, figures, and inline images from your documents into a local folder.
*   **✂️ Auto-Column Slicing:** Intelligently splits multi-column documents (like dictionaries or academic papers) to prevent text-mixing.
*   **💾 Auto-Save & Batch Processing:** Process folders of 100+ images or massive PDF books automatically with rate-limit handling and auto-saving to `.md` or `.csv`.

---

## Screenshots
<img width="3360" height="1898" alt="Gemini OCR" src="https://github.com/user-attachments/assets/a23e11f7-9404-4306-a695-c040ee04e366" />
<img width="3360" height="1896" alt="Baidu VL 1 15" src="https://github.com/user-attachments/assets/e7efe9cb-d31a-447d-b370-c421121b92c9" />

---
## 🚀 Installation

### 1. Install Dependencies
Make sure you have Python 3.8+ installed. Run the following command:
```bash
pip install PySide6 pymupdf pandas opencv-python pillow requests google-genai
```

### 2. Run the Application
```bash
python OCRAIHub.py
```

---

## ⚙️ API Configuration

### Google Gemini
1. Get your free API key from [Google AI Studio](https://aistudio.google.com/app/apikey).
2. Paste the Key into the app, select your preferred model (e.g., `gemini-flash-latest`), and type your extraction prompt.

### Baidu PaddleOCR-VL
1. Go to [Baidu AI Studio](https://aistudio.baidu.com/) and deploy the `PaddleOCR-VL` model.
2. Copy your **Token** and your **Custom API URL** (e.g., `https://aistudio.baidu.com/serving/online/xxxx?xxxx`).
3. Select your Baidu Task (OCR, Table, Formula, or Chart).

---

## 🙏 Acknowledgments

This tool would not be possible without the incredible work of the open-source and AI communities:
*   Special thanks to [Gemini-OCR-Arabic](https://github.com/the-cataloger/Gemini-OCR-Arabic) by the-cataloger, which served as the foundational fork and inspiration for the early versions of this tool.
*[Google Gemini](https://deepmind.google/technologies/gemini/) for their cutting-edge multimodal VLM APIs.
*   [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) by Baidu for their industry-leading layout detection and document parsing models.
*   [PyMuPDF (Fitz)](https://github.com/pymupdf/PyMuPDF) for lightning-fast local PDF rasterization.
*[PySide6](https://doc.qt.io/qtforpython-6/) for the robust GUI framework.

---

## 📝 Citation

If you use this tool in your research or project, please consider citing it:

```bibtex
@software{Nasr_OCRAIHub_2026,
  author = {Nasr, Shawky},
  title = {OCRAIHub: Multi-Engine AI Document Parsing Tool},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/shawkynasr/OCRAIHub}
}
```
## 📜 License
This project is licensed under the **GNU General Public License v3.0** - see the [LICENSE](LICENSE) file for details.

---
