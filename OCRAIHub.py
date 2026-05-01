import sys
import os
import requests
import json
import PIL.Image
import PIL.ImageOps
import pandas as pd
import io
import fitz  # (pymupdf)
import time
import textwrap
import cv2
import numpy as np
import base64

# --- GOOGLE GENAI SDK IMPORTS ---
from google import genai
from google.genai import types

from PySide6.QtCore import (
    Qt, QObject, QThread, Signal, Slot, QDateTime
)
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QTextEdit, QPushButton, QFileDialog,
    QStatusBar, QFrame, QSizePolicy, QComboBox,
    QDialog, QCheckBox, QSpinBox, QDoubleSpinBox, QMessageBox
)
from PySide6.QtGui import QPixmap

# --- (1. Image Preview Dialog) ---
class ImagePreviewDialog(QDialog):
    def __init__(self, image_paths, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Image Preview")
        self.setMinimumSize(600, 500)
        self.setModal(True)
        self.setLayoutDirection(Qt.LeftToRight)
        self.image_paths = image_paths
        self.current_index = 0
        layout = QVBoxLayout(self)
        self.index_label = QLabel()
        self.index_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.index_label)
        self.image_display = QLabel("Loading image...")
        self.image_display.setAlignment(Qt.AlignCenter)
        self.image_display.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.image_display)
        button_layout = QHBoxLayout()
        self.prev_button = QPushButton("Previous")
        self.next_button = QPushButton("Next")
        button_layout.addWidget(self.prev_button)
        button_layout.addWidget(self.next_button)
        self.close_button = QPushButton("Close")
        button_layout.addWidget(self.close_button)
        layout.addLayout(button_layout)
        self.prev_button.clicked.connect(self.show_previous_image)
        self.next_button.clicked.connect(self.show_next_image)
        self.close_button.clicked.connect(self.accept)
        self.show_image()

    def show_image(self):
        if not self.image_paths:
            self.image_display.setText("No images found.")
            return
        path = self.image_paths[self.current_index]
        pixmap = QPixmap(path)
        scaled_pixmap = pixmap.scaled(self.image_display.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_display.setPixmap(scaled_pixmap)
        self.index_label.setText(f"Image {self.current_index + 1} of {len(self.image_paths)}")
        self.prev_button.setEnabled(self.current_index > 0)
        self.next_button.setEnabled(self.current_index < len(self.image_paths) - 1)

    def show_previous_image(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.show_image()

    def show_next_image(self):
        if self.current_index < len(self.image_paths) - 1:
            self.current_index += 1
            self.show_image()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.show_image()

# --- (2. RequestWorker - Handles Both Gemini and Baidu PaddleOCR) ---
class RequestWorker(QObject):
    finished = Signal(str)
    error = Signal(str)
    status_update = Signal(str)
    completed_all = Signal()
    issues_report = Signal(list)

    def __init__(self, api_key, model_name, prompt_text, image_paths, file_paths,
                 pdf_dpi=300, image_batch_size=10, delay_seconds=60, 
                 process_images_binarization=False, process_images_denoising=False,
                 process_images_resize_factor=1.0, 
                 all_pages=True, start_page=1, end_page=1,
                 num_columns=1, engine="Google Gemini", custom_url="", baidu_prompt="ocr",
                 native_pdf=False, download_figures=False): 
        super().__init__()
        self.api_key = api_key
        self.model_name = model_name
        self.prompt_text = prompt_text
        self.image_paths = image_paths if image_paths else[]
        self.file_paths = file_paths if file_paths else[]
        self.PDF_DPI = pdf_dpi
        self.IMAGE_BATCH_SIZE = image_batch_size
        self.FREE_TIER_DELAY = delay_seconds
        self.PROCESS_BINARIZATION = process_images_binarization
        self.PROCESS_DENOISING = process_images_denoising
        self.RESIZE_FACTOR = process_images_resize_factor
        
        self.ALL_PAGES = all_pages
        self.START_PAGE = start_page
        self.END_PAGE = end_page
        self.NUM_COLUMNS = num_columns 
        
        self.engine = engine
        self.custom_url = custom_url
        self.baidu_prompt = baidu_prompt
        self.NATIVE_PDF = native_pdf
        self.DOWNLOAD_FIGURES = download_figures

    def split_image_into_columns(self, pil_img):
        if self.NUM_COLUMNS <= 1:
            return [pil_img]
            
        self.status_update.emit(f"... ✂️ Auto-slicing image into {self.NUM_COLUMNS} columns...")
        img_np = np.array(pil_img)
        
        if img_np.ndim == 3:
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_np
            
        _, thresh_deskew = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        coords = np.column_stack(np.where(thresh_deskew > 0))
        angle = 0.0
        if len(coords) > 0:
            angle = cv2.minAreaRect(coords)[-1]
            if angle < -45:
                angle = -(90 + angle)
            else:
                angle = -angle
                
        if -5.0 < angle < 5.0 and abs(angle) > 0.1:
            self.status_update.emit(f"... 📐 Auto-deskewing: fixing {angle:.2f} degree tilt...")
            (h, w) = gray.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            img_np = cv2.warpAffine(img_np, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            gray = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

        (h, w) = gray.shape[:2]
        start_y = int(h * 0.15)
        end_y = int(h * 0.85)
        middle_section = gray[start_y:end_y, :]
        
        _, thresh = cv2.threshold(middle_section, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        
        kernel = np.ones((1, 5), np.uint8)
        dilated = cv2.dilate(thresh, kernel, iterations=2)
        
        proj = np.sum(dilated, axis=0)
        proj_smoothed = np.convolve(proj, np.ones(30)/30, mode='same') 
        
        expected_width = w // self.NUM_COLUMNS
        split_x_coords =[]
        
        for i in range(1, self.NUM_COLUMNS):
            expected_split = i * expected_width
            window = int(w * 0.12)
            start_x = max(0, expected_split - window)
            end_x = min(w, expected_split + window)
            
            if start_x < end_x:
                local_min_idx = np.argmin(proj_smoothed[start_x:end_x])
                best_x = start_x + local_min_idx
                split_x_coords.append(best_x)
            else:
                split_x_coords.append(expected_split)
                
        if img_np.ndim == 2:
            rotated_pil = PIL.Image.fromarray(img_np, mode='L') 
        else:
            rotated_pil = PIL.Image.fromarray(img_np)
            
        slices =[]
        last_x = 0
        split_x_coords.append(w)
        
        for x in split_x_coords:
            box = (last_x, 0, x, h)
            sliced_img = rotated_pil.crop(box)
            slices.append(sliced_img)
            last_x = x
            
        return slices

    def process_single_image(self, img):
        if img.mode not in ('RGB', 'L'):
            img = img.convert('RGB')
            
        img_processed = img.copy()

        if self.RESIZE_FACTOR < 1.0:
            new_size = (int(img_processed.width * self.RESIZE_FACTOR), 
                        int(img_processed.height * self.RESIZE_FACTOR))
            img_processed = img_processed.resize(new_size, PIL.Image.Resampling.LANCZOS)
            self.status_update.emit(f"... Image resized by factor: {self.RESIZE_FACTOR * 100:.0f}%")

        img_np = np.array(img_processed)
        
        if img_np.ndim == 3:
             img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        else:
             img_gray = img_np

        if self.PROCESS_BINARIZATION:
            _, img_thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            img_np = img_thresh
            self.status_update.emit("... Applying advanced binary thresholding (Otsu) ...")
        else:
            img_np = img_gray 

        if self.PROCESS_DENOISING and img_np.ndim == 2:
            kernel = np.ones((2, 2), np.uint8) 
            img_np = cv2.morphologyEx(img_np, cv2.MORPH_OPEN, kernel, iterations=1)
            img_np = cv2.morphologyEx(img_np, cv2.MORPH_CLOSE, kernel, iterations=1)
            self.status_update.emit("... Applying morphological noise reduction ...")

        if img_np.ndim == 2:
            return PIL.Image.fromarray(img_np, mode='L') 
        else:
            return PIL.Image.fromarray(img_np)
        
    def convert_pdf_to_images(self, pdf_path): 
        self.status_update.emit(f"... ⏳ Converting PDF to images at {self.PDF_DPI} DPI...")
        images =[]
        doc = fitz.open(pdf_path)
        
        num_pages = len(doc)
        start_index = 0
        end_index = num_pages - 1

        if not self.ALL_PAGES:
            start_index = max(0, self.START_PAGE - 1) 
            end_index = min(num_pages - 1, self.END_PAGE - 1) 
            
            if start_index > end_index:
                 self.status_update.emit(f"Warning: Invalid page range. Ignoring.")
                 return[] 
                 
            self.status_update.emit(f"... Processing PDF page range: from {start_index + 1} to {end_index + 1}...")
        else:
             self.status_update.emit(f"... Processing all PDF pages: {num_pages} pages.")
        
        for i in range(start_index, end_index + 1): 
            page = doc[i] 
            pix = page.get_pixmap(dpi=self.PDF_DPI)
            img = PIL.Image.frombytes("RGB",[pix.width, pix.height], pix.samples) 
            
            img_processed = self.process_single_image(img) 
            slices = self.split_image_into_columns(img_processed)
            images.extend(slices)
            
        doc.close()
        self.status_update.emit(f"... PDF processed. Resulting in {len(images)} image slices.")
        return images

    def download_extracted_images(self, image_list, save_base_name):
        folder_path = f"{save_base_name}_images"
        os.makedirs(folder_path, exist_ok=True)
        
        for img_name, img_url in image_list.items():
            if img_url:
                try:
                    self.status_update.emit(f"⬇️ Downloading figure: {img_name}...")
                    img_data = requests.get(img_url, timeout=15).content
                    
                    # Create subdirectories if the image name implies a path (e.g. imgs/img_01.jpg)
                    full_img_path = os.path.join(folder_path, img_name)
                    os.makedirs(os.path.dirname(full_img_path), exist_ok=True)
                    
                    with open(full_img_path, "wb") as f:
                        f.write(img_data)
                except Exception as e:
                    self.status_update.emit(f"⚠️ Failed to download {img_name}: {e}")

    def parse_paddle_response(self, data, save_base_name=None):
        """Extracts text and auto-downloads images based on the PaddleOCR-VL-1.5 JSON spec"""
        if "result" in data and "layoutParsingResults" in data["result"]:
            results = data["result"]["layoutParsingResults"]
            extracted =[]
            for res in results:
                if "markdown" in res:
                    if "text" in res["markdown"]:
                        extracted.append(res["markdown"]["text"])
                    
                    # Download inline markdown images if enabled
                    if self.DOWNLOAD_FIGURES and "images" in res["markdown"] and save_base_name:
                        image_urls = res["markdown"]["images"] # dictionary format in Baidu API
                        if image_urls:
                            self.download_extracted_images(image_urls, save_base_name)
                
                # Also download "outputImages" (cropped figures) if they exist
                if self.DOWNLOAD_FIGURES and "outputImages" in res and save_base_name:
                    out_images = res["outputImages"]
                    if out_images:
                         self.download_extracted_images(out_images, save_base_name)
                            
            if extracted:
                return "\n\n".join(extracted)

        # Legacy PaddleOCR format fallback
        extracted_lines =[]
        def extract_text(obj):
            if isinstance(obj, dict):
                if 'words_result' in obj:
                    for item in obj['words_result']:
                        if isinstance(item, dict) and 'words' in item:
                            extracted_lines.append(str(item['words']))
                    return
                if 'text' in obj: extracted_lines.append(str(obj['text']))
                if 'words' in obj: extracted_lines.append(str(obj['words']))
                for k, v in obj.items(): extract_text(v)
            elif isinstance(obj, list):
                for item in obj:
                    if isinstance(item, list) and len(item) == 2 and isinstance(item[1], list) and len(item[1]) > 0:
                        if isinstance(item[1][0], str):
                            extracted_lines.append(item[1][0])
                            continue
                    extract_text(item)

        extract_text(data)
        if extracted_lines:
            return "\n".join(extracted_lines)
            
        return json.dumps(data, ensure_ascii=False, indent=2)

    def send_baidu_request(self, file_data, file_type=1):
        """Sends payload to Baidu mirroring the exact official optionalPayload structure"""
        file_b64 = base64.b64encode(file_data).decode('ascii')
        headers = {
            "Authorization": f"token {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # This matches the official Baidu AI Studio script exactly for rich Markdown output
        payload = {
            "file": file_b64,
            "fileType": file_type, 
            "markdownIgnoreLabels":[
                "header", "header_image", "footer", "footer_image", 
                "number", "footnote", "aside_text"
            ],
            "useDocOrientationClassify": False,
            "useDocUnwarping": False,
            "useLayoutDetection": True,        # Required for tables & layouts
            "useChartRecognition": False,
            "useSealRecognition": True,
            "useOcrForImageBlock": False,
            "mergeTables": True,               # Required to output actual markdown tables
            "relevelTitles": True,             # Required for correct heading (###) extraction
            "layoutShapeMode": "auto",
            "promptLabel": self.baidu_prompt,
            "repetitionPenalty": 1,
            "temperature": 0,
            "topP": 1,
            "minPixels": 147384,
            "maxPixels": 2822400,
            "layoutNms": True,
            "restructurePages": True           # Required to properly assemble reading order
        }
        
        response = requests.post(self.custom_url, json=payload, headers=headers)
        return response

    @Slot()
    def run(self):
        try:
            client = None
            clean_model_name = self.model_name
            if self.engine == "Google Gemini":
                client = genai.Client(api_key=self.api_key)
                clean_model_name = self.model_name.replace("models/", "")
                
            prompt = self.prompt_text
            
            safe_settings =[
                types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE"),
                types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"),
                types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"),
                types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
            ]
            custom_generation_config = types.GenerateContentConfig(
                max_output_tokens=8192, 
                temperature=0.1,
                safety_settings=safe_settings
            )

            MAX_RETRIES = 2
            RETRY_DELAY = 20
            issues_found =[]
            
            actual_batch_size = self.IMAGE_BATCH_SIZE if self.engine == "Google Gemini" else 1

            # ==========================================
            # 1. PROCESS STANDALONE IMAGES
            # ==========================================
            if self.image_paths: 
                self.status_update.emit("=== 🖼️ Processing Standalone Images ===")
                images_to_process =[]
                for p in self.image_paths:
                    img = PIL.Image.open(p)
                    if img.mode not in ('RGB', 'L'):
                        img = img.convert('RGB')
                    img_processed = self.process_single_image(img)
                    slices = self.split_image_into_columns(img_processed)
                    images_to_process.extend(slices)
                
                if images_to_process:
                    jobs =[]
                    image_batches = [images_to_process[i:i+actual_batch_size] 
                                     for i in range(0, len(images_to_process), actual_batch_size)]
                    for batch in image_batches: 
                        jobs.append( {"type": "image_batch", "content": batch} )

                    total_jobs = len(jobs)

                    for i, job in enumerate(jobs):
                        job_content = job["content"]
                        self.status_update.emit(f"... 🚀 Sending image batch/page {i+1} of {total_jobs}...")
                        
                        text_response = ""
                        for attempt in range(MAX_RETRIES + 1):
                            try:
                                if self.engine == "Google Gemini":
                                    payload = [prompt] 
                                    payload.extend(job_content) 
                                    response = client.models.generate_content(
                                        model=clean_model_name,
                                        contents=payload,
                                        config=custom_generation_config
                                    )
                                    if not response.candidates: raise Exception("Blocked by API (Empty response)")
                                    
                                    finish_reason = str(response.candidates[0].finish_reason)
                                    if "SAFETY" in finish_reason: raise Exception(f"Blocked by API Safety filter. Reason: {finish_reason}")
                                    if "PROHIBITED" in finish_reason or "BLOCK" in finish_reason: raise Exception(f"Blocked by API Prohibited Content. Reason: {finish_reason}")
                                    if "MAX_TOKENS" in finish_reason: raise Exception(f"Stopped by MAX_TOKENS limit. Reason: {finish_reason}")
                                        
                                    text_response = response.text
                                    break 
                                else:
                                    img = job_content[0]
                                    buffer = io.BytesIO()
                                    img.save(buffer, format="JPEG")
                                    image_data = buffer.getvalue()
                                    
                                    response = self.send_baidu_request(image_data, file_type=1)
                                    if response.status_code != 200:
                                        error_msg = response.text[:250]
                                        if response.status_code == 500:
                                            raise Exception(f"HTTP 500 - Baidu Internal Error. Details: {error_msg}")
                                        raise Exception(f"HTTP {response.status_code} - {error_msg}")
                                        
                                    save_base_name = os.path.splitext(self.image_paths[0])[0] if self.image_paths else "image_extracted"
                                    text_response = self.parse_paddle_response(response.json(), save_base_name=save_base_name)
                                    break

                            except Exception as api_error:
                                if attempt < MAX_RETRIES:
                                    self.status_update.emit(f"⚠️ Error on Item {i+1}. Retrying... ({attempt + 1}/{MAX_RETRIES})")
                                    time.sleep(RETRY_DELAY)
                                else:
                                    text_response = f"\n[⚠️ WARNING: Batch {i+1} Exception - {str(api_error)}]\n"
                                    self.status_update.emit(f"❌ API Exception on Item {i+1}: {str(api_error)}")
                                    issues_found.append(f"Images Batch {i+1}: API Exception ({str(api_error)})")

                        self.finished.emit(f"--- OCR Result (Images Batch {i+1}/{total_jobs}) ---\n{text_response}")
                        if i < total_jobs - 1:
                            time.sleep(self.FREE_TIER_DELAY)

            # ==========================================
            # 2. PROCESS PDFs
            # ==========================================
            if self.file_paths:
                total_pdfs = len(self.file_paths)
                for pdf_idx, pdf_path in enumerate(self.file_paths):
                    pdf_name = os.path.basename(pdf_path)
                    save_base_name = os.path.splitext(pdf_path)[0]
                    self.status_update.emit(f"\n{'='*40}\n=== 📄 Starting PDF {pdf_idx+1}/{total_pdfs}: {pdf_name} ===\n{'='*40}")
                    
                    if self.engine == "Baidu PaddleOCR-VL (Custom URL)" and self.NATIVE_PDF:
                        self.status_update.emit(f"... 🚀 Sending Native PDF file to Baidu (bypassing local render)...")
                        with open(pdf_path, "rb") as f:
                            pdf_data = f.read()
                        
                        text_response = ""
                        for attempt in range(MAX_RETRIES + 1):
                            try:
                                response = self.send_baidu_request(pdf_data, file_type=0)
                                if response.status_code != 200:
                                    error_msg = response.text[:250]
                                    if response.status_code == 500:
                                        raise Exception(f"HTTP 500 - Baidu Internal Error. File might be too large. Details: {error_msg}")
                                    raise Exception(f"HTTP {response.status_code} - {error_msg}")
                                    
                                text_response = self.parse_paddle_response(response.json(), save_base_name=save_base_name)
                                break
                            except Exception as api_error:
                                if attempt < MAX_RETRIES:
                                    self.status_update.emit(f"⚠️ Error on Native PDF {pdf_name}. Retrying... ({attempt + 1}/{MAX_RETRIES})")
                                    time.sleep(RETRY_DELAY)
                                else:
                                    text_response = f"\n[⚠️ WARNING: {pdf_name} Exception - {str(api_error)}]\n"
                                    self.status_update.emit(f"❌ API Exception on Native PDF: {str(api_error)}")
                                    issues_found.append(f"{pdf_name} (Native): API Exception ({str(api_error)})")

                        pdf_full_text = text_response + "\n\n"
                        self.finished.emit(f"--- OCR Result ({pdf_name} - Native PDF) ---\n{text_response}")

                    else:
                        pdf_images = self.convert_pdf_to_images(pdf_path)
                        if not pdf_images:
                            self.status_update.emit(f"⚠️ Warning: File {pdf_name} yielded no images. Skipping.")
                            continue
                        
                        jobs =[]
                        image_batches =[pdf_images[i:i+actual_batch_size] 
                                         for i in range(0, len(pdf_images), actual_batch_size)]
                        for batch in image_batches: 
                            jobs.append( {"type": "image_batch", "content": batch} )

                        total_jobs = len(jobs)
                        pdf_full_text = "" 
                        
                        for i, job in enumerate(jobs):
                            job_content = job["content"]
                            self.status_update.emit(f"... 🚀 Sending {pdf_name} batch/page {i+1} of {total_jobs}...")
                            
                            text_response = ""
                            for attempt in range(MAX_RETRIES + 1):
                                try:
                                    if self.engine == "Google Gemini":
                                        payload =[prompt] 
                                        payload.extend(job_content) 
                                        response = client.models.generate_content(
                                            model=clean_model_name,
                                            contents=payload,
                                            config=custom_generation_config
                                        )
                                        if not response.candidates: raise Exception("Blocked by API (Empty response)")
                                        
                                        finish_reason = str(response.candidates[0].finish_reason)
                                        if "SAFETY" in finish_reason: raise Exception(f"Blocked by API Safety filter. Reason: {finish_reason}")
                                        if "PROHIBITED" in finish_reason or "BLOCK" in finish_reason: raise Exception(f"Blocked by API Prohibited Content. Reason: {finish_reason}")
                                        if "MAX_TOKENS" in finish_reason: raise Exception(f"Stopped by MAX_TOKENS limit. Reason: {finish_reason}")
                                            
                                        text_response = response.text
                                        break 
                                    else:
                                        img = job_content[0]
                                        buffer = io.BytesIO()
                                        img.save(buffer, format="JPEG")
                                        image_data = buffer.getvalue()
                                        
                                        response = self.send_baidu_request(image_data, file_type=1)
                                        if response.status_code != 200:
                                            error_msg = response.text[:250]
                                            if response.status_code == 500:
                                                raise Exception(f"HTTP 500 - Baidu Internal Error. Details: {error_msg}")
                                            raise Exception(f"HTTP {response.status_code} - {error_msg}")
                                            
                                        text_response = self.parse_paddle_response(response.json(), save_base_name=save_base_name)
                                        break

                                except Exception as api_error:
                                    if attempt < MAX_RETRIES:
                                        self.status_update.emit(f"⚠️ Error on Batch {i+1}. Retrying... ({attempt + 1}/{MAX_RETRIES})")
                                        time.sleep(RETRY_DELAY)
                                    else:
                                        text_response = f"\n[⚠️ WARNING: Batch {i+1} Exception - {str(api_error)}]\n"
                                        self.status_update.emit(f"❌ API Exception on Batch {i+1}: {str(api_error)}")
                                        issues_found.append(f"{pdf_name} Batch {i+1}: API Exception ({str(api_error)})")

                            pdf_full_text += text_response + "\n\n"
                            self.finished.emit(f"--- OCR Result ({pdf_name} - Batch {i+1}/{total_jobs}) ---\n{text_response}")
                            
                            if i < total_jobs - 1 or pdf_idx < total_pdfs - 1:
                                time.sleep(self.FREE_TIER_DELAY)

                    out_md_path = os.path.splitext(pdf_path)[0] + "_OCR.md"
                    try:
                        with open(out_md_path, "w", encoding="utf-8") as f:
                            f.write(pdf_full_text)
                        self.status_update.emit(f"✅ 💾 Auto-saved Markdown to: {out_md_path}")
                    except Exception as e:
                        self.status_update.emit(f"❌ Failed to auto-save {pdf_name}: {str(e)}")
                        issues_found.append(f"{pdf_name}: Auto-save Failed ({str(e)})")

            if not self.image_paths and not self.file_paths:
                raise Exception("No images or PDF found for OCR processing.")
                
            self.status_update.emit(f"✅ All OCR tasks completed.")
            self.issues_report.emit(issues_found)  
            
        except Exception as critical_error: 
            self.error.emit(f"Critical System Error: {critical_error}")
        finally: 
            self.completed_all.emit()

# --- (3. Main Window) ---
class GeminiApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Multi-Engine AI OCR V2 By (Shawky Nasr) shawkynasr@126.com")
        self.setGeometry(100, 100, 1200, 800)
        self.setLayoutDirection(Qt.LeftToRight) 

        self.VLM_MODELS_LIST =[
            "gemini-2.5-pro", 
            "gemini-2.5-flash",
            "gemini-pro-latest",
            "gemini-flash-latest",
        ]
        
        self.DEFAULT_OCR_PROMPT = """You are an advanced Optical Character Recognition (OCR) engine. Your task is to extract the *full and accurate* text content from the attached images (book pages/documents).

**Key Instructions:**
1.  **Accuracy:** Extract the text exactly as it appears in the image.
2.  **Formatting:** Preserve paragraphs, lines, and indentation as much as possible.
3.  **Language Detection:** Detect the language automatically and extract text accordingly.
4.  **Structured Data:** If you find tables, convert them to **Markdown** table format.
5.  **Sequence:** Process multiple images (batches) sequentially without mixing content.
6.  **Clean Output:** Return **ONLY the extracted text** without any explanations, introductions, or markdown code blocks tags."""
        
        self.current_image_paths =[] 
        self.current_file_paths =[] 

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        
        left_column_widget = QWidget()
        left_column_layout = QVBoxLayout(left_column_widget)
        
        # Add consistent spacing to the left column to make it look clean on macOS
        left_column_layout.setSpacing(12) 
        
        right_column_widget = QWidget()
        right_column_layout = QVBoxLayout(right_column_widget)
        
        main_layout.addWidget(left_column_widget, 1) 
        main_layout.addWidget(right_column_widget, 2)

        # --- OCR Engine Selection ---
        engine_layout = QHBoxLayout()
        engine_label = QLabel("⚙️ OCR Engine:")
        self.engine_combo = QComboBox()
        self.engine_combo.addItems(["Google Gemini", "Baidu PaddleOCR-VL (Custom URL)"])
        engine_layout.addWidget(engine_label)
        engine_layout.addWidget(self.engine_combo)
        left_column_layout.addLayout(engine_layout)

        # --- Token / API Key (Always Visible) ---
        api_layout = QHBoxLayout()
        api_label = QLabel("🔑 Token / API Key:")
        self.api_key_input = QLineEdit()
        self.api_key_input.setPlaceholderText("Enter your Token or API Key...")
        self.api_key_input.setEchoMode(QLineEdit.Password) 
        
        self.test_conn_button = QPushButton("⚡ Test Connection")
        self.test_conn_button.clicked.connect(self.test_connection)
        
        api_layout.addWidget(api_label)
        api_layout.addWidget(self.api_key_input)
        api_layout.addWidget(self.test_conn_button)
        left_column_layout.addLayout(api_layout)

        # --- SMART SWAPPING ENGINE SETTINGS ---
        self.engine_settings_container = QWidget()
        engine_settings_layout = QVBoxLayout(self.engine_settings_container)
        engine_settings_layout.setContentsMargins(0, 0, 0, 0)
        
        # 1. Gemini Widget
        self.gemini_widget = QWidget()
        gemini_layout = QVBoxLayout(self.gemini_widget)
        gemini_layout.setContentsMargins(0, 0, 0, 0)
        
        px_lay = QHBoxLayout()
        px_lay.addWidget(QLabel("🌐 Local Proxy (e.g., 127.0.0.1:8888):"))
        self.proxy_input = QLineEdit()
        self.proxy_input.setPlaceholderText("Leave empty if not in China")
        px_lay.addWidget(self.proxy_input)
        
        md_lay = QHBoxLayout()
        md_lay.addWidget(QLabel("🤖 Gemini Model:"))
        self.model_combo = QComboBox()
        self.model_combo.addItems(self.VLM_MODELS_LIST)
        md_lay.addWidget(self.model_combo)
        
        gemini_layout.addLayout(px_lay)
        gemini_layout.addLayout(md_lay)
        
        # 2. Baidu Widget
        self.baidu_widget = QWidget()
        baidu_layout = QVBoxLayout(self.baidu_widget)
        baidu_layout.setContentsMargins(0, 0, 0, 0)
        
        url_lay = QHBoxLayout()
        url_lay.addWidget(QLabel("🔗 Custom API URL:"))
        self.custom_url_input = QLineEdit()
        self.custom_url_input.setPlaceholderText("https://aistudio.baidu.com/serving/online/xxxx?xxxx")
        url_lay.addWidget(self.custom_url_input)
        
        tsk_lay = QHBoxLayout()
        tsk_lay.addWidget(QLabel("🤖 Baidu Task:"))
        self.baidu_prompt_combo = QComboBox()
        self.baidu_prompt_combo.addItems(["ocr", "table", "formula", "chart"])
        tsk_lay.addWidget(self.baidu_prompt_combo)
        
        baidu_layout.addLayout(url_lay)
        baidu_layout.addLayout(tsk_lay)

        engine_settings_layout.addWidget(self.gemini_widget)
        engine_settings_layout.addWidget(self.baidu_widget)
        left_column_layout.addWidget(self.engine_settings_container)

        # --- COMPACT SINGLE-LINE FILE INPUTS ---
        img_lay = QHBoxLayout()
        self.select_image_button = QPushButton("🖼️ 1. Load Images / Folder")
        self.image_path_display = QLineEdit()
        self.image_path_display.setReadOnly(True)
        self.image_path_display.setPlaceholderText("No images loaded.")
        self.view_images_button = QPushButton("👁️ View")
        self.view_images_button.setEnabled(False)
        img_lay.addWidget(self.select_image_button)
        img_lay.addWidget(self.image_path_display)
        img_lay.addWidget(self.view_images_button)
        left_column_layout.addLayout(img_lay)

        pdf_lay = QHBoxLayout()
        self.select_file_button = QPushButton("📄 2. Select PDF File(s) For Batch")
        self.pdf_path_display = QLineEdit()
        self.pdf_path_display.setReadOnly(True)
        self.pdf_path_display.setPlaceholderText("No file selected.")
        pdf_lay.addWidget(self.select_file_button)
        pdf_lay.addWidget(self.pdf_path_display)
        left_column_layout.addLayout(pdf_lay)

        # Baidu PDF Extras (Right below the PDF selection)
        self.baidu_extras_widget = QWidget()
        bx_lay = QVBoxLayout(self.baidu_extras_widget)
        bx_lay.setContentsMargins(0, 0, 0, 0)
        self.native_pdf_checkbox = QCheckBox("📄 Native PDF Mode (Bypass PyMuPDF - Send directly to Baidu)")
        self.download_figures_checkbox = QCheckBox("🖼️ Auto-Download Charts & Figures to Folder")
        bx_lay.addWidget(self.native_pdf_checkbox)
        bx_lay.addWidget(self.download_figures_checkbox)
        left_column_layout.addWidget(self.baidu_extras_widget)

        # ==========================================
        # ADVANCED SETTINGS TOGGLE AREA
        # ==========================================
        self.toggle_settings_btn = QPushButton("⚙️ Show Advanced Settings")
        self.toggle_settings_btn.setCheckable(True)
        self.toggle_settings_btn.setChecked(False) 
        self.toggle_settings_btn.clicked.connect(self.toggle_settings_visibility)
        left_column_layout.addWidget(self.toggle_settings_btn)

        self.settings_container = QWidget()
        settings_layout = QVBoxLayout(self.settings_container)
        settings_layout.setContentsMargins(0, 0, 0, 0)

        # --- 1. PDF Page Range Options ---
        pdf_page_range_frame = QFrame()
        pdf_page_range_frame.setFrameShape(QFrame.StyledPanel)
        pdf_page_range_layout = QVBoxLayout(pdf_page_range_frame)
        pdf_page_range_label = QLabel("PDF Page Range Selection:")
        pdf_page_range_layout.addWidget(pdf_page_range_label)
        
        self.all_pages_checkbox = QCheckBox("✅ Process **All Pages** (Ignore Start/Stop)")
        self.all_pages_checkbox.setChecked(True)
        pdf_page_range_layout.addWidget(self.all_pages_checkbox)
        
        range_layout = QHBoxLayout()
        start_page_label = QLabel("From Page:")
        range_layout.addWidget(start_page_label)
        self.start_page_spin = QSpinBox()
        self.start_page_spin.setRange(1, 9999)
        self.start_page_spin.setValue(1)
        self.start_page_spin.setEnabled(False) 
        range_layout.addWidget(self.start_page_spin)
        
        end_page_label = QLabel("To Page:")
        range_layout.addWidget(end_page_label)
        self.end_page_spin = QSpinBox()
        self.end_page_spin.setRange(1, 9999)
        self.end_page_spin.setValue(10)
        self.end_page_spin.setEnabled(False) 
        range_layout.addWidget(self.end_page_spin)
        
        pdf_page_range_layout.addLayout(range_layout)
        settings_layout.addWidget(pdf_page_range_frame) 

        # --- 2. OCR Specific Options & Pre-processing ---
        options_frame = QFrame()
        options_frame.setFrameShape(QFrame.StyledPanel)
        options_layout = QVBoxLayout(options_frame)
        options_label = QLabel("Image Pre-processing Options:")
        options_layout.addWidget(options_label)
        
        self.binarization_checkbox = QCheckBox("✨ Advanced Binarization (Otsu) - Recommended for books")
        self.binarization_checkbox.setChecked(True)
        options_layout.addWidget(self.binarization_checkbox)
        
        self.denoising_checkbox = QCheckBox("🧽 Morphological Noise Reduction (Salt & Pepper)")
        self.denoising_checkbox.setChecked(True)
        options_layout.addWidget(self.denoising_checkbox)
        
        # DPI
        dpi_layout = QHBoxLayout()
        dpi_label = QLabel("PDF Conversion DPI:")
        dpi_layout.addWidget(dpi_label)
        self.dpi_spin = QSpinBox()
        self.dpi_spin.setRange(100, 600)
        self.dpi_spin.setSingleStep(50)
        self.dpi_spin.setValue(150) 
        dpi_layout.addWidget(self.dpi_spin)
        options_layout.addLayout(dpi_layout)
        
        # Resize Factor
        resize_layout = QHBoxLayout()
        resize_label = QLabel("Resize Factor (Save Memory):")
        self.resize_factor_spin = QDoubleSpinBox()
        self.resize_factor_spin.setRange(0.25, 1.0)
        self.resize_factor_spin.setSingleStep(0.05)
        self.resize_factor_spin.setValue(1.0)
        resize_layout.addWidget(resize_label)
        resize_layout.addWidget(self.resize_factor_spin)
        options_layout.addLayout(resize_layout)
        
        settings_layout.addWidget(options_frame)
        
        # --- 3. Batch Settings ---
        batch_frame = QFrame()
        batch_frame.setFrameShape(QFrame.StyledPanel)
        batch_layout = QVBoxLayout(batch_frame)
        batch_label = QLabel("Batch & Slicing Settings:")
        batch_layout.addWidget(batch_label)
        
        columns_layout = QHBoxLayout()
        columns_label = QLabel("✂️ Auto Columns (分栏数):")
        self.columns_spin = QSpinBox()
        self.columns_spin.setRange(1, 5)
        self.columns_spin.setValue(1)
        self.columns_spin.setToolTip("Set to 2 or 3 for multi-column dictionaries.")
        columns_layout.addWidget(columns_label)
        columns_layout.addWidget(self.columns_spin)
        batch_layout.addLayout(columns_layout)
        
        image_batch_layout = QHBoxLayout()
        image_batch_label = QLabel("Gemini Batch Size (Pages):")
        self.image_batch_spin = QSpinBox()
        self.image_batch_spin.setRange(1, 50)
        self.image_batch_spin.setValue(10)
        image_batch_layout.addWidget(image_batch_label)
        image_batch_layout.addWidget(self.image_batch_spin)
        batch_layout.addLayout(image_batch_layout)
        
        delay_layout = QHBoxLayout()
        delay_label = QLabel("Wait Delay (Seconds):")
        self.delay_spin = QSpinBox()
        self.delay_spin.setRange(5, 600)
        self.delay_spin.setValue(60)
        delay_layout.addWidget(delay_label)
        delay_layout.addWidget(self.delay_spin)
        batch_layout.addLayout(delay_layout)
        
        settings_layout.addWidget(batch_frame)
        
        left_column_layout.addWidget(self.settings_container)
        self.settings_container.setVisible(False)
        # ==========================================


        # ==========================================
        # 4. PROMPT TOGGLE AREA
        # ==========================================
        self.toggle_prompt_btn = QPushButton("💬 Show OCR Prompt Settings")
        self.toggle_prompt_btn.setCheckable(True)
        self.toggle_prompt_btn.setChecked(False)
        self.toggle_prompt_btn.clicked.connect(self.toggle_prompt_visibility)
        left_column_layout.addWidget(self.toggle_prompt_btn)

        self.prompt_container = QWidget()
        prompt_layout = QVBoxLayout(self.prompt_container)
        prompt_layout.setContentsMargins(0, 0, 0, 0)
        
        prompt_label = QLabel("💬 3. OCR Text Extraction Prompt:")
        prompt_layout.addWidget(prompt_label)
        
        self.prompt_input = QTextEdit()
        self.prompt_input.setText(self.DEFAULT_OCR_PROMPT)
        self.prompt_input.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        prompt_layout.addWidget(self.prompt_input)
        left_column_layout.addWidget(self.prompt_container)
        
        self.prompt_container.setVisible(False)
        # ==========================================

        self.send_button = QPushButton("🚀 Send OCR Request")
        self.send_button.setEnabled(False) 
        self.send_button.setStyleSheet("font-size: 16px; padding: 10px; background-color: #4CAF50; color: white;")
        left_column_layout.addWidget(self.send_button)

        settings_io_layout = QHBoxLayout()
        self.import_settings_button = QPushButton("📥 Import Settings")
        self.export_settings_button = QPushButton("📤 Export Settings")
        settings_io_layout.addWidget(self.import_settings_button)
        settings_io_layout.addWidget(self.export_settings_button)
        left_column_layout.addLayout(settings_io_layout)

        # 💡 THE GRAVITY ANCHOR: Pushes everything perfectly to the top, so NO GAPS when hidden!
        left_column_layout.addStretch(1)

        # --- Fill Right Column (Outputs) ---
        response_label = QLabel("🤖 Model Response (Partial results appear here):")
        right_column_layout.addWidget(response_label)
        self.response_output = QTextEdit()
        self.response_output.setReadOnly(True) 
        self.response_output.setObjectName("responseOutputBox")
        right_column_layout.addWidget(self.response_output, 5)
        
        tools_layout = QHBoxLayout()
        self.save_button = QPushButton("💾 Manual Save Result...")
        self.save_button.setEnabled(False)
        tools_layout.addWidget(self.save_button)
        self.clear_button = QPushButton("🧹 Clear All")
        tools_layout.addWidget(self.clear_button)
        right_column_layout.addLayout(tools_layout)
        
        log_label = QLabel("🖥️ Backend Event Log:")
        right_column_layout.addWidget(log_label)
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setStyleSheet("""
            QTextEdit {
                background-color: #000033; color: #E0E0E0;
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 11px; border-radius: 4px;
            }
        """)
        right_column_layout.addWidget(self.log_output, 1)

        # --- 4. Status Bar and Connections ---
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        self.engine_combo.currentIndexChanged.connect(self.toggle_engine_ui)
        self.select_image_button.clicked.connect(self.open_image_dialog)
        self.view_images_button.clicked.connect(self.show_image_previewer) 
        self.select_file_button.clicked.connect(self.open_file_dialog)
        self.api_key_input.textChanged.connect(self.check_inputs)
        self.prompt_input.textChanged.connect(self.check_inputs)
        self.response_output.textChanged.connect(self.check_save_button_status)
        self.send_button.clicked.connect(self.start_request)
        self.save_button.clicked.connect(self.save_results_to_file)
        self.clear_button.clicked.connect(self.clear_all_inputs)
        
        self.import_settings_button.clicked.connect(self.import_settings_from_file)
        self.export_settings_button.clicked.connect(self.export_settings_to_file)
        self.all_pages_checkbox.stateChanged.connect(self.toggle_page_spins) 
        
        self.toggle_engine_ui()
        self.check_inputs()
        self.check_save_button_status()
        self.append_to_log("OCR App started. Ready for processing or settings import.")

    @Slot(bool)
    def toggle_settings_visibility(self, checked):
        self.settings_container.setVisible(checked)
        if checked:
            self.toggle_settings_btn.setText("⚙️ Hide Advanced Settings")
        else:
            self.toggle_settings_btn.setText("⚙️ Show Advanced Settings")

    @Slot(bool)
    def toggle_prompt_visibility(self, checked):
        self.prompt_container.setVisible(checked)
        if checked:
            self.toggle_prompt_btn.setText("💬 Hide OCR Prompt Settings")
        else:
            self.toggle_prompt_btn.setText("💬 Show OCR Prompt Settings")

    @Slot()
    def toggle_engine_ui(self):
        is_gemini = self.engine_combo.currentIndex() == 0
        
        # Smartly swap engine widgets (Invisible elements don't take up space in Qt Layouts)
        self.gemini_widget.setVisible(is_gemini)
        self.baidu_widget.setVisible(not is_gemini)
        self.baidu_extras_widget.setVisible(not is_gemini)

        # Disable prompt and its toggle button when Baidu is active (DO NOT HIDE to prevent jumping)
        self.prompt_input.setEnabled(is_gemini)
        self.toggle_prompt_btn.setEnabled(is_gemini)
        
        if not is_gemini:
            self.image_batch_spin.setToolTip("PaddleOCR processes 1 image at a time natively.")
            self.prompt_input.setPlaceholderText("Baidu uses predefined tasks above. Free-text prompt disabled.")
        else:
            self.image_batch_spin.setToolTip("")
            self.prompt_input.setPlaceholderText("")
            
        self.check_inputs()

    @Slot()
    def export_settings_to_file(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "Export Settings As...", "", "JSON Files (*.json)")
        if not file_path: return
        settings = {
            "engine_index": self.engine_combo.currentIndex(),
            "baidu_prompt": self.baidu_prompt_combo.currentIndex(),
            "api_key": self.api_key_input.text().strip(),
            "custom_url": self.custom_url_input.text().strip(),
            "proxy": self.proxy_input.text().strip(),
            "model_index": self.model_combo.currentIndex(),
            "prompt": self.prompt_input.toPlainText(),
            "binarization": self.binarization_checkbox.isChecked(),
            "denoising": self.denoising_checkbox.isChecked(),
            "dpi": self.dpi_spin.value(),
            "resize_factor": self.resize_factor_spin.value(),
            "columns": self.columns_spin.value(),
            "batch_size": self.image_batch_spin.value(),
            "delay": self.delay_spin.value(),
            "all_pages": self.all_pages_checkbox.isChecked(),
            "start_page": self.start_page_spin.value(),
            "end_page": self.end_page_spin.value(),
            "native_pdf": self.native_pdf_checkbox.isChecked(),
            "download_figures": self.download_figures_checkbox.isChecked()
        }
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(settings, f, indent=4, ensure_ascii=False)
            self.append_to_log(f"✅ Settings successfully exported to: {file_path}")
            self.status_bar.showMessage("✅ Settings Exported", 5000)
        except Exception as e:
            self.append_to_log(f"❌ Failed to export settings: {e}")

    @Slot()
    def import_settings_from_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Import Settings File", "", "JSON Files (*.json)")
        if not file_path: return
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            if "engine_index" in data: self.engine_combo.setCurrentIndex(data["engine_index"])
            if "baidu_prompt" in data: self.baidu_prompt_combo.setCurrentIndex(data["baidu_prompt"])
            if "api_key" in data: self.api_key_input.setText(data["api_key"])
            if "custom_url" in data: self.custom_url_input.setText(data["custom_url"])
            if "proxy" in data: self.proxy_input.setText(data["proxy"])
            if "model_index" in data: self.model_combo.setCurrentIndex(data["model_index"])
            if "prompt" in data: self.prompt_input.setText(data["prompt"])
            if "binarization" in data: self.binarization_checkbox.setChecked(data["binarization"])
            if "denoising" in data: self.denoising_checkbox.setChecked(data["denoising"])
            if "dpi" in data: self.dpi_spin.setValue(data["dpi"])
            if "resize_factor" in data: self.resize_factor_spin.setValue(data["resize_factor"])
            if "columns" in data: self.columns_spin.setValue(data["columns"])
            if "batch_size" in data: self.image_batch_spin.setValue(data["batch_size"])
            if "delay" in data: self.delay_spin.setValue(data["delay"])
            if "all_pages" in data: 
                self.all_pages_checkbox.setChecked(data["all_pages"])
                self.toggle_page_spins(Qt.Checked if data["all_pages"] else Qt.Unchecked)
            if "start_page" in data: self.start_page_spin.setValue(data["start_page"])
            if "end_page" in data: self.end_page_spin.setValue(data["end_page"])
            if "native_pdf" in data: self.native_pdf_checkbox.setChecked(data["native_pdf"])
            if "download_figures" in data: self.download_figures_checkbox.setChecked(data["download_figures"])
            
            self.append_to_log(f"✅ Settings successfully imported from: {file_path}")
            self.status_bar.showMessage("✅ Settings Imported", 5000)
        except Exception as e:
            self.append_to_log(f"❌ Failed to import settings: {e}")

    @Slot(int)
    def toggle_page_spins(self, state):
        enabled = state != Qt.Checked 
        self.start_page_spin.setEnabled(enabled)
        self.end_page_spin.setEnabled(enabled)

    @Slot(str)
    def append_to_log(self, message):
        now = QDateTime.currentDateTime().toString("yyyy-MM-dd hh:mm:ss")
        self.log_output.append(f"[{now}] {message}")
        self.log_output.verticalScrollBar().setValue(self.log_output.verticalScrollBar().maximum())

    @Slot()
    def save_results_to_file(self):
        text_to_save = self.response_output.toPlainText()
        if not text_to_save: return
        file_path, selected_filter = QFileDialog.getSaveFileName(self, "Save Result As...", "", "CSV (*.csv);;Text Files (*.txt);;Markdown (*.md);;All (*)")
        if not file_path: return
        try:
            if file_path.endswith('.csv'): self.save_as_csv(text_to_save, file_path)
            else: self.save_as_text(text_to_save, file_path)
            self.append_to_log(f"Result saved to: {file_path}")
        except Exception as e: self.append_to_log(f"ERROR: Save failed - {e}")

    def save_as_text(self, text, path):
        with open(path, 'w', encoding='utf-8') as f: f.write(text)
        self.status_bar.showMessage(f"✅ Saved as text in: {path}", 5000)

    def save_as_csv(self, markdown_text, path):
        try:
            lines = markdown_text.strip().split('\n'); table_lines =[line.strip() for line in lines if line.strip().startswith('|') and line.strip().endswith('|')]
            if not table_lines: raise Exception("No Markdown table found")
            data = [[c.strip() for c in line.strip('|').split('|')] for line in table_lines]
            if len(data) > 1 and all(c.replace('-', '').strip() == '' for c in data[1]): header, rows = data[0], data[2:]
            else: header, rows = data[0], data[1:]
            df = pd.DataFrame(rows, columns=header)
            df.to_csv(path, index=False, encoding='utf-8-sig'); self.status_bar.showMessage(f"✅ Saved as Table (CSV) in: {path}", 5000)
        except Exception as e:
            error_msg = f"❌ Table conversion failed: {e}. Saving as plain text."; self.status_bar.showMessage(error_msg, 7000)
            text_path = os.path.splitext(path)[0] + ".txt"; self.save_as_text(markdown_text, text_path)

    @Slot()
    def clear_all_inputs(self):
        self.api_key_input.clear()
        self.custom_url_input.clear()
        self.prompt_input.setText(self.DEFAULT_OCR_PROMPT)
        self.response_output.clear()
        self.current_image_paths =[]
        self.current_file_paths =[]
        self.image_path_display.clear()
        self.pdf_path_display.clear()
        self.view_images_button.setEnabled(False)
        self.engine_combo.setCurrentIndex(0)
        self.baidu_prompt_combo.setCurrentIndex(0)
        self.model_combo.setCurrentIndex(0)
        self.binarization_checkbox.setChecked(True)
        self.denoising_checkbox.setChecked(True)
        self.dpi_spin.setValue(150)
        self.resize_factor_spin.setValue(1.0)
        self.columns_spin.setValue(1)
        self.image_batch_spin.setValue(10)
        self.delay_spin.setValue(60)
        self.proxy_input.clear() 
        self.native_pdf_checkbox.setChecked(False)
        self.download_figures_checkbox.setChecked(False)
        
        self.all_pages_checkbox.setChecked(True)
        self.start_page_spin.setValue(1)
        self.end_page_spin.setValue(10)
        self.toggle_page_spins(Qt.Checked) 
        
        self.status_bar.showMessage("🧹 All inputs and results cleared.")
        self.append_to_log("Interface cleared.")
        self.log_output.clear()
        self.check_inputs()

    @Slot()
    def check_inputs(self):
        api_key_ok = bool(self.api_key_input.text().strip()); 
        image_ok = bool(self.current_image_paths); file_ok = bool(self.current_file_paths)
        self.send_button.setEnabled(api_key_ok and (image_ok or file_ok))
        self.view_images_button.setEnabled(image_ok)

    @Slot()
    def check_save_button_status(self):
        response_ok = bool(self.response_output.toPlainText().strip()); self.save_button.setEnabled(response_ok)

    @Slot()
    def open_image_dialog(self):
        dialog = QFileDialog(self)
        dialog.setFileMode(QFileDialog.ExistingFiles) 
        dialog.setOption(QFileDialog.DontUseNativeDialog, True)
        dialog.setNameFilter("Image Files (*.png *.jpg *.jpeg *.webp)")
        
        open_folder_button = QPushButton("Select Folder")
        try:
             dialog.layout().addWidget(open_folder_button)
        except Exception:
             pass 

        folder_selected =[]
        @Slot()
        def on_folder_button_clicked():
            folder_path = QFileDialog.getExistingDirectory(self, "Select Image Folder")
            if folder_path:
                for filename in os.listdir(folder_path):
                    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                        folder_selected.append(os.path.join(folder_path, filename))
            dialog.done(QDialog.Accepted)
            
        open_folder_button.clicked.connect(on_folder_button_clicked)
        
        if dialog.exec() == QDialog.Accepted:
            file_paths = list(dialog.selectedFiles())
            if folder_selected: 
                file_paths = folder_selected
            
            if file_paths:
                self.current_image_paths = file_paths
                self.image_path_display.setText(f"Loaded {len(file_paths)} image(s)")
                self.append_to_log(f"Loaded {len(file_paths)} images for OCR processing.")
            else:
                self.current_image_paths =[]
                self.image_path_display.clear()
        
        self.check_inputs()

    @Slot()
    def show_image_previewer(self):
        if not self.current_image_paths: return
        dialog = ImagePreviewDialog(self.current_image_paths, self); dialog.exec()

    @Slot()
    def open_file_dialog(self):
        file_paths, _ = QFileDialog.getOpenFileNames(self, "Select Scanned PDF File(s)", "", "PDF Files (*.pdf)")
        
        if file_paths:
            self.current_file_paths = file_paths
            if len(file_paths) == 1:
                self.pdf_path_display.setText(f"{os.path.basename(file_paths[0])}")
            else:
                self.pdf_path_display.setText(f"Selected {len(file_paths)} PDF Files.")
            self.append_to_log(f"Loaded {len(file_paths)} PDF File(s).")
        else:
            self.current_file_paths =[]
            self.pdf_path_display.clear()
            
        if file_paths:
            if len(file_paths) == 1:
                try:
                    doc = fitz.open(file_paths[0])
                    num_pages = len(doc)
                    self.start_page_spin.setRange(1, num_pages)
                    self.end_page_spin.setRange(1, num_pages)
                    self.end_page_spin.setValue(num_pages) 
                    self.append_to_log(f"Detected {num_pages} pages in PDF.")
                    doc.close()
                except Exception as e:
                    self.append_to_log(f"Warning: Failed reading PDF pages - {e}")
                    self.start_page_spin.setRange(1, 9999)
                    self.end_page_spin.setRange(1, 9999)
            else:
                self.start_page_spin.setRange(1, 9999)
                self.end_page_spin.setRange(1, 9999)

        self.check_inputs()

    @Slot()
    def test_connection(self):
        api_key = self.api_key_input.text().strip()
        if not api_key:
            self.append_to_log("❌ Error: Enter a Token / API Key first to test the connection.")
            self.status_bar.showMessage("❌ Missing API Key", 3000)
            return

        self.append_to_log("🔍 Testing connection...")
        self.test_conn_button.setEnabled(False)
        self.status_bar.showMessage("Testing connection...", 0)
        
        proxies = {}
        user_proxy = self.proxy_input.text().strip()
        if user_proxy:
            if not user_proxy.startswith("http"):
                user_proxy = f"http://{user_proxy}"
            proxies = {"http": user_proxy, "https": user_proxy}
            self.append_to_log(f"... Using local proxy tunnel: {user_proxy}")

        try:
            if self.engine_combo.currentIndex() == 0:  # Gemini
                test_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash?key={api_key}"
                response = requests.get(test_url, proxies=proxies, timeout=10)
                if response.status_code == 200:
                    self.append_to_log("✅ Connection Successful! Google server reached.")
                    self.status_bar.showMessage("✅ Connection Success", 5000)
                else:
                    self.append_to_log(f"⚠️ Server reached but returned error {response.status_code}: {response.text}")
                    self.status_bar.showMessage(f"⚠️ API Error: {response.status_code}", 5000)
            else:  # Baidu PaddleOCR
                test_url = self.custom_url_input.text().strip()
                if not test_url:
                    self.append_to_log("❌ Error: Enter a Custom API URL for Baidu PaddleOCR.")
                    self.status_bar.showMessage("❌ Missing Custom API URL", 5000)
                    return
                
                headers = {
                    "Authorization": f"token {api_key}",
                    "Content-Type": "application/json"
                }
                
                dummy_np = np.ones((150, 400, 3), dtype=np.uint8) * 255
                cv2.putText(dummy_np, "API Test Connection OK", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2)
                dummy_img = PIL.Image.fromarray(dummy_np)
                
                buffer = io.BytesIO()
                dummy_img.save(buffer, format="JPEG")
                file_b64 = base64.b64encode(buffer.getvalue()).decode('ascii')
                
                payload = {
                    "file": file_b64,
                    "fileType": 1,
                    "useLayoutDetection": False,
                    "promptLabel": "ocr"
                }
                
                response = requests.post(test_url, headers=headers, json=payload, proxies=proxies, timeout=15)
                
                if response.status_code == 200:
                    self.append_to_log(f"✅ Endpoint reached successfully! Status returned: HTTP 200")
                    self.status_bar.showMessage("✅ API Endpoint Reached", 5000)
                else:
                    self.append_to_log(f"⚠️ Endpoint reached but returned HTTP {response.status_code}: {response.text[:200]}")
                    self.status_bar.showMessage(f"⚠️ API returned {response.status_code}", 5000)
                
        except requests.exceptions.Timeout:
            self.append_to_log("❌ Connection Timeout: Server blocked/down. Check your VPN/Proxy.")
            self.status_bar.showMessage("❌ Connection Timeout", 5000)
        except Exception as e:
            self.append_to_log(f"❌ Connection Failed: {str(e)}")
            self.status_bar.showMessage("❌ Connection Failed", 5000)
        finally:
            self.test_conn_button.setEnabled(True)

    @Slot()
    def start_request(self):
        engine = self.engine_combo.currentText()
        custom_url = self.custom_url_input.text().strip()
        
        if engine == "Baidu PaddleOCR-VL (Custom URL)" and not custom_url:
            self.handle_error("Please enter a Custom API URL for Baidu PaddleOCR.")
            return

        self.append_to_log(f"Starting OCR request process via {engine}...")
        self.send_button.setEnabled(False)
        self.response_output.clear() 
        
        user_proxy = self.proxy_input.text().strip()
        if user_proxy:
            if not user_proxy.startswith("http"):
                user_proxy = f"http://{user_proxy}"
            os.environ["http_proxy"] = user_proxy
            os.environ["https_proxy"] = user_proxy
            self.append_to_log(f"Routing traffic through proxy: {user_proxy}")
        else:
            os.environ.pop("http_proxy", None)
            os.environ.pop("https_proxy", None)
        
        api_key = self.api_key_input.text()
        model = self.model_combo.currentText()
        prompt = self.prompt_input.toPlainText()
        image_paths = self.current_image_paths
        file_paths = self.current_file_paths 
        
        process_images_binarization = self.binarization_checkbox.isChecked()
        process_images_denoising = self.denoising_checkbox.isChecked()
        process_images_resize_factor = self.resize_factor_spin.value()
        
        pdf_dpi = self.dpi_spin.value()
        image_batch_size = self.image_batch_spin.value()
        delay_seconds = self.delay_spin.value() 
        baidu_prompt_val = self.baidu_prompt_combo.currentText()
        
        native_pdf_val = self.native_pdf_checkbox.isChecked()
        download_figures_val = self.download_figures_checkbox.isChecked()
        
        num_columns = self.columns_spin.value()
        
        all_pages = self.all_pages_checkbox.isChecked()
        start_page = self.start_page_spin.value()
        end_page = self.end_page_spin.value()
        
        if file_paths and not all_pages and start_page > end_page:
            self.handle_error("Logic Error: Start page must be less than or equal to End page.")
            self.send_button.setEnabled(True)
            return

        self.thread = QThread()
        self.worker = RequestWorker(api_key, model, prompt, image_paths, file_paths,
                                    pdf_dpi, image_batch_size, delay_seconds,
                                    process_images_binarization, 
                                    process_images_denoising, 
                                    process_images_resize_factor,
                                    all_pages, start_page, end_page, 
                                    num_columns,
                                    engine, custom_url, baidu_prompt_val,
                                    native_pdf_val, download_figures_val) 
        self.worker.moveToThread(self.thread)
        self.worker.status_update.connect(self.append_to_log)
        
        self.worker.issues_report.connect(self.show_issues_report)
        
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.handle_partial_response)
        self.worker.error.connect(self.handle_error)
        self.worker.completed_all.connect(self.handle_all_completed)
        self.worker.completed_all.connect(self.thread.quit)
        self.worker.error.connect(self.thread.quit)
        self.worker.completed_all.connect(self.worker.deleteLater)
        self.worker.error.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.start()

    @Slot(list)
    def show_issues_report(self, issues):
        if not issues:
            self.status_bar.showMessage("✅ Task finished perfectly with no errors.", 5000)
        else:
            issue_text = "\n".join([f"• {issue}" for issue in issues])
            
            summary_block = f"\n\n{'='*50}\n⚠️ TASK COMPLETED WITH WARNINGS ⚠️\nThe following batches require manual review:\n\n{issue_text}\n{'='*50}\n"
            self.response_output.append(summary_block)
            
            QMessageBox.warning(self, "Task Completed with Issues", 
                                f"The OCR batch task has finished, but {len(issues)} issue(s) were found.\n\n"
                                f"{issue_text}\n\n"
                                f"A summary has been added to the bottom of your output text.")

    @Slot(str)
    def handle_partial_response(self, response_text):
        self.response_output.append(response_text + "\n" + ("-"*40) + "\n")
        self.status_bar.showMessage("✅ Partial result received...", 3000) 
        self.append_to_log("SUCCESS: Partial result received.")

    @Slot()
    def handle_all_completed(self):
        self.status_bar.showMessage("✅ All tasks completed.", 5000) 
        self.append_to_log("Finished processing background thread.")
        self.check_inputs()

    @Slot(str)
    def handle_error(self, error_message):
        self.response_output.append(f"❌ Error:\n\n{error_message}\n" + ("-"*40) + "\n")
        self.status_bar.showMessage(f"❌ Error: {error_message}", 10000) 
        self.append_to_log(f"ERROR: {error_message}")
        self.check_inputs()

# --- 4. Run Application ---
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = GeminiApp()
    window.show()
    sys.exit(app.exec())