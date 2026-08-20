"""
File Parser Utility for Salesforce AI Agent.
Handles parsing and text extraction for CSV, Excel, PDF, JSON, TXT, and images.
Prepares base64 payloads for Salesforce ContentVersion record attachments.
Guarantees 100% JSON-serializable output for FastAPI responses.
"""

import base64
import csv
import io
import json
import logging
import os
import time
from typing import Any

logger = logging.getLogger(__name__)

# ── Lazy-cached OCR engine (loaded once, reused across uploads) ──
_ocr_engine = None

def _get_ocr_engine():
    """Return a cached RapidOCR engine instance (avoids slow re-init per upload)."""
    global _ocr_engine
    if _ocr_engine is None:
        try:
            from rapidocr_onnxruntime import RapidOCR
            _ocr_engine = RapidOCR()
            logger.info("✅ OCR engine initialized (cached for future uploads).")
        except ImportError:
            logger.warning("rapidocr_onnxruntime not installed — OCR disabled.")
            _ocr_engine = False  # Sentinel: tried but unavailable
    return _ocr_engine if _ocr_engine is not False else None


# ── Max file size for inline base64 (5 MB) — larger files skip base64 for speed ──
MAX_INLINE_BASE64_BYTES = 5 * 1024 * 1024


def parse_uploaded_file(file_path: str, filename: str) -> dict[str, Any]:
    """
    Parse an uploaded file and extract structured text, tabular preview, and base64.
    
    Returns a dict with:
        - filename: str
        - file_type: str (csv, excel, pdf, text, json, image, binary)
        - file_size: int (bytes)
        - summary: str (human-readable summary)
        - content_preview: str (extracted text / markdown table)
        - base64_data: str (base64 encoded content for Salesforce upload)
    """
    t0 = time.time()
    ext = os.path.splitext(filename)[1].lower()
    file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0

    with open(file_path, "rb") as f:
        file_bytes = f.read()

    # Only compute base64 for files under 5 MB (large files skip this for speed)
    if file_size <= MAX_INLINE_BASE64_BYTES:
        base64_data = base64.b64encode(file_bytes).decode("utf-8")
    else:
        base64_data = ""  # Will be computed on-demand when actually uploading to Salesforce
        logger.info(f"⏩ Skipping inline base64 for large file ({file_size / 1024 / 1024:.1f} MB) — will encode on-demand.")

    result: dict[str, Any] = {
        "filename": filename,
        "file_type": "binary",
        "file_size": file_size,
        "summary": f"File: {filename} ({file_size} bytes)",
        "content_preview": "",
        "base64_data": base64_data,
        "json_records": [],
    }

    # 1. CSV Files
    if ext == ".csv":
        result["file_type"] = "csv"
        try:
            import pandas as pd
            df = pd.read_csv(io.BytesIO(file_bytes))
            rows, cols = df.shape
            cols_list = [str(c) for c in df.columns]
            preview_df = df.head(10).fillna("").astype(str)
            markdown_table = preview_df.to_markdown(index=False)
            
            result["summary"] = f"CSV File with {rows} rows and {cols} columns: {', '.join(cols_list)}"
            result["content_preview"] = f"### CSV Preview ({rows} total records, showing first 10):\n\n{markdown_table}"
            if rows > 10:
                result["content_preview"] += f"\n\n*(Showing 10 of {rows} rows)*"
            
            # Clean JSON serialization using pandas built-in serializer
            result["json_records"] = json.loads(df.fillna("").to_json(orient="records", date_format="iso"))
        except Exception as e:
            logger.warning(f"Error parsing CSV with pandas: {e}, falling back to standard text/csv")
            try:
                text = file_bytes.decode("utf-8", errors="replace")
                result["content_preview"] = f"```\n{text[:3000]}\n```"
                result["summary"] = f"CSV File ({filename})"
            except Exception:
                pass

    # 2. Excel Files (.xlsx, .xls)
    elif ext in (".xlsx", ".xls"):
        result["file_type"] = "excel"
        try:
            import pandas as pd
            # Read first sheet
            excel_file = pd.ExcelFile(io.BytesIO(file_bytes))
            sheet_names = excel_file.sheet_names
            first_sheet = sheet_names[0] if sheet_names else "Sheet1"
            
            df = pd.read_excel(excel_file, sheet_name=first_sheet)
            rows, cols = df.shape
            cols_list = [str(c) for c in df.columns]
            preview_df = df.head(10).fillna("").astype(str)
            markdown_table = preview_df.to_markdown(index=False)
            
            sheet_info = f"Sheet: '{first_sheet}'" + (f" (Total Sheets: {len(sheet_names)})" if len(sheet_names) > 1 else "")
            result["summary"] = f"Excel Spreadsheet [{sheet_info}] with {rows} rows and {cols} columns: {', '.join(cols_list)}"
            result["content_preview"] = f"### Excel Data Preview ({sheet_info}, {rows} total records, showing first 10):\n\n{markdown_table}"
            if rows > 10:
                result["content_preview"] += f"\n\n*(Showing 10 of {rows} rows)*"
            
            # Clean JSON serialization
            result["json_records"] = json.loads(df.fillna("").to_json(orient="records", date_format="iso"))
        except Exception as e:
            logger.error(f"Error parsing Excel file: {e}", exc_info=True)
            result["summary"] = f"Excel File: {filename} ({file_size} bytes)"
            result["content_preview"] = f"*(Excel document attached: {filename})*"

    # 3. PDF Files
    elif ext == ".pdf":
        result["file_type"] = "pdf"
        try:
            from pypdf import PdfReader
            reader = PdfReader(io.BytesIO(file_bytes))
            num_pages = len(reader.pages)
            extracted_text = []
            for i, page in enumerate(reader.pages[:10]):  # Max 10 pages preview
                page_text = page.extract_text() or ""
                if page_text.strip():
                    extracted_text.append(f"--- Page {i+1} ---\n{page_text.strip()}")
            
            full_text = "\n\n".join(extracted_text)

            # If native text extraction yielded no or very little text, use OCR fallback (scanned PDF)
            if len(full_text.strip()) < 50:
                logger.info(f"📄 PDF appears to be scanned or image-based ({filename}). Running OCR fallback...")
                try:
                    import numpy as np
                    import pypdfium2 as pdfium

                    ocr = _get_ocr_engine()
                    if ocr:
                        pdf_doc = pdfium.PdfDocument(io.BytesIO(file_bytes))
                        # OCR first 3 pages only for speed (was 6)
                        total_ocr_pages = min(3, len(pdf_doc))
                        ocr_extracted = []

                        for p_idx in range(total_ocr_pages):
                            page = pdf_doc[p_idx]
                            # Lower render scale for speed (1.0 instead of 1.2)
                            image = page.render(scale=1.0).to_pil()
                            ocr_res, _ = ocr(np.array(image))
                            if ocr_res:
                                page_lines = [line[1] for line in ocr_res if line and len(line) > 1]
                                if page_lines:
                                    ocr_extracted.append(f"--- Page {p_idx+1} (OCR) ---\n" + "\n".join(page_lines))

                        if ocr_extracted:
                            full_text = "\n\n".join(ocr_extracted)
                            logger.info(f"✅ OCR extracted {len(full_text)} characters from {len(ocr_extracted)} pages of {filename}")
                    else:
                        logger.warning("OCR engine not available — skipping OCR for scanned PDF.")
                except Exception as ocr_err:
                    logger.warning(f"OCR fallback failed for {filename}: {ocr_err}")

            if len(full_text) > 5000:
                full_text = full_text[:5000] + "\n... [truncated for prompt length]"

            result["summary"] = f"PDF Document with {num_pages} pages"
            result["content_preview"] = f"### Extracted PDF Content ({num_pages} pages):\n\n{full_text}" if full_text.strip() else f"*(PDF document attached: {filename})*"
        except Exception as e:
            logger.error(f"Error parsing PDF file: {e}", exc_info=True)
            result["summary"] = f"PDF Document: {filename} ({file_size} bytes)"
            result["content_preview"] = f"*(PDF document attached: {filename})*"

    # 4. Plain Text, JSON, Markdown, Code Files
    elif ext in (".txt", ".json", ".md", ".log", ".xml", ".html", ".py", ".js", ".sql"):
        result["file_type"] = "text"
        try:
            text = file_bytes.decode("utf-8", errors="replace")
            if ext == ".json":
                try:
                    parsed_json = json.loads(text)
                    formatted_json = json.dumps(parsed_json, indent=2)
                    text = formatted_json
                    result["file_type"] = "json"
                except Exception:
                    pass
            
            preview = text[:4000]
            if len(text) > 4000:
                preview += "\n... [truncated]"

            result["summary"] = f"Text Document ({ext.upper()[1:]}): {filename}"
            result["content_preview"] = f"```\n{preview}\n```"
        except Exception as e:
            logger.error(f"Error reading text file: {e}")

    # 5. Image Files
    elif ext in (".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"):
        result["file_type"] = "image"
        img_text = ""
        try:
            import numpy as np
            from PIL import Image

            ocr = _get_ocr_engine()
            if ocr:
                img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
                ocr_res, _ = ocr(np.array(img))
                if ocr_res:
                    lines = [line[1] for line in ocr_res if line and len(line) > 1]
                    img_text = "\n".join(lines)
        except Exception as img_err:
            logger.warning(f"Image OCR error for {filename}: {img_err}")

        result["summary"] = f"Image Attachment: {filename} ({file_size} bytes)"
        if img_text.strip():
            result["content_preview"] = f"### Extracted Image Text (OCR):\n\n{img_text[:3000]}"
        else:
            result["content_preview"] = f"*(Image attachment ready for Salesforce upload / processing: {filename})*"

    elapsed = time.time() - t0
    logger.info(f"📁 File parsed in {elapsed:.2f}s: {filename} ({result['file_type']}, {file_size} bytes)")
    return result

