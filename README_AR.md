<div dir="rtl">
  
# OCRAIHub (أداة التعرف الضوئي على الحروف متعددة المحركات بالذكاء الاصطناعي)

![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-green.svg)
![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey)

**OCRAIHub** هو تطبيق سطح مكتب قوي بواجهة رسومية (GUI) يدعم محركات ذكاء اصطناعي متعددة، مُصمم لاستخراج النصوص، جداول Markdown، المعادلات الرياضية، والرسوم البيانية من الصور وملفات PDF بدقة فائقة. يعتمد البرنامج على أحدث نماذج الرؤية واللغة (VLMs) في العالم لتوفير رقمنة مثالية للمستندات.

يدعم التطبيق حالياً **Google Gemini (2.5/3.1)** و **Baidu PaddleOCR-VL (1.5)**.

---

## ✨ الميزات الرئيسية

*   **🤖 دعم محركات متعددة:** التبديل بسلاسة بين واجهة برمجة تطبيقات Google Gemini ونقاط اتصال Baidu AI Studio PaddleOCR.
*   **📊 تنسيق Markdown الذكي:** إعادة بناء تخطيط المستندات بشكل مثالي، بما في ذلك العناوين، الفقرات، الجداول المعقدة، والمعادلات الرياضية (LaTeX).
*   **📄 التعامل مع ملفات PDF وتقسيم الصفحات:** 
    *   *الوضع المحلي (Local Mode):* تحويل ملفات PDF إلى صور محلياً باستخدام PyMuPDF، مع معالجة متقدمة للصور (Otsu Binarization & Morphological Denoising).
    *   *الوضع الأصلي (Native Mode - لـ Baidu):* إرسال ملفات PDF الخام مباشرة إلى المحرك لمعالجتها بشكل كامل.
*   **🖼️ التنزيل التلقائي للرسومات:** اكتشاف واقتطاع وتنزيل الرسوم البيانية والأشكال والصور المضمنة من مستنداتك تلقائياً وحفظها في مجلد محلي.
*   **✂️ تقسيم الأعمدة التلقائي:** تقسيم المستندات متعددة الأعمدة بذكاء (مثل القواميس أو الأوراق الأكاديمية) لمنع تداخل النصوص.
*   **💾 الحفظ التلقائي والمعالجة المجمعة:** معالجة مجلدات تحتوي على مئات الصور أو كتب PDF ضخمة تلقائياً، مع معالجة حدود معدل الطلبات (Rate limits) والحفظ التلقائي بصيغة `.md` أو `.csv`.

---
## صور من واجهة البرنامج
<img width="3360" height="1898" alt="Gemini OCR" src="https://github.com/user-attachments/assets/a23e11f7-9404-4306-a695-c040ee04e366" />
<img width="3360" height="1896" alt="Baidu VL 1 15" src="https://github.com/user-attachments/assets/e7efe9cb-d31a-447d-b370-c421121b92c9" />

---

## 🚀 التثبيت

### 1. تثبيت المتطلبات (Dependencies)
تأكد من تثبيت Python 3.8+ على جهازك. قم بتشغيل الأمر التالي في موجّه الأوامر (Terminal/CMD):
```bash
pip install PySide6 pymupdf pandas opencv-python pillow requests google-genai
```

### 2. تشغيل التطبيق
```bash
python OCRAIHub.py
```

---

## ⚙️ إعدادات واجهة برمجة التطبيقات (API)

### إعداد Google Gemini
1. احصل على مفتاح API مجاني من [Google AI Studio](https://aistudio.google.com/app/apikey).
2. انسخ المفتاح والصقه في البرنامج، اختر النموذج المفضل لديك (مثل `gemini-2.5-flash`)، واكتب أمر الاستخراج (Prompt).

### إعداد Baidu PaddleOCR-VL
1. انتقل إلى [Baidu AI Studio](https://aistudio.baidu.com/) وقم بنشر نموذج `PaddleOCR-VL`.
2. انسخ الـ **Token** الخاص بك ورابط **Custom API URL** (مثال: `https://aistudio.baidu.com/serving/online/xxxx?xxxx`).
3. اختر مهمة بايدو (Baidu Task) المطلوبة للاستخراج: (OCR, Table, Formula, أو Chart).

---

## 🙏 شكر وتقدير

لم يكن هذا العمل ليرى النور لولا الجهود المذهلة لمجتمع المصادر المفتوحة ومجتمع الذكاء الاصطناعي:
*   شكر خاص لمشروع[Gemini-OCR-Arabic](https://github.com/the-cataloger/Gemini-OCR-Arabic) بواسطة the-cataloger، والذي كان بمثابة التفرع (fork) الأساسي والمصدر المُلهم للنسخ الأولى من هذه الأداة.
*   [Google Gemini](https://deepmind.google/technologies/gemini/) لتقديمهم واجهات برمجة تطبيقات VLMs المتقدمة.
*   [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) من Baidu لتقديمهم نماذج رائدة في تحليل التخطيط (Layout detection) وفهم المستندات.
*   [PyMuPDF (Fitz)](https://github.com/pymupdf/PyMuPDF) لسرعتهم الفائقة في معالجة ملفات PDF محلياً.
*   [PySide6](https://doc.qt.io/qtforpython-6/) لإطار عمل واجهة المستخدم الرسومية (GUI) القوي.

---

## 📝 الاقتباس (Citation)

إذا كنت تستخدم هذه الأداة في بحثك أو مشروعك، يُرجى النظر في الاستشهاد بها:

```bibtex
@software{Nasr_OCRAIHub_2026,
  author = {Nasr, Shawky},
  title = {OCRAIHub: Multi-Engine AI Document Parsing Tool},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/shawkynasr/OCRAIHub}
}
```

## 📜 الترخيص (License)
هذا المشروع مرخص بموجب رخصة **GNU General Public License v3.0** - راجع ملف [LICENSE](LICENSE) لمزيد من التفاصيل.