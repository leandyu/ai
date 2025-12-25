#!/usr/bin/env python3
import os
import argparse
import json
from pathlib import Path
from PyPDF2 import PdfReader
import docx

def extract_text_from_pdf(pdf_path):
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text.strip()
    except Exception as e:
        print(f"[ERROR] Cannot read PDF {pdf_path}: {e}")
        return ""

def extract_text_from_docx(docx_path):
    try:
        doc = docx.Document(docx_path)
        text = "\n".join([p.text for p in doc.paragraphs])
        return text.strip()
    except Exception as e:
        print(f"[ERROR] Cannot read DOCX {docx_path}: {e}")
        return ""

def write_markdown(text, md_path):
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(text)

def generate_jsonl_entry(content, mode):
    entries = []
    if mode in ["continue", "all"]:
        entries.append({
            "instruction": "根据以下剧本继续写后续剧情，保持风格一致：",
            "input": content,
            "output": ""
        })
    if mode in ["rewrite", "all"]:
        entries.append({
            "instruction": "改写以下剧本，使其更精彩，但保持人物和主要情节：",
            "input": content,
            "output": ""
        })
    if mode in ["copy", "all"]:
        entries.append({
            "instruction": "根据以下剧本，重新创建一份剧本，保持风格、人物关系、矛盾冲突、高潮等场景类似：",
            "input": content,
            "output": ""
        })
    return entries

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", required=True, help="原始PDF或DOCX文件目录")
    parser.add_argument("--clean_dir", required=True, help="输出Markdown文件目录")
    parser.add_argument("--out_jsonl", required=True, help="输出JSONL文件目录")
    parser.add_argument("--mode", choices=["continue", "rewrite", "copy", "all"], default="all")
    args = parser.parse_args()

    os.makedirs(args.clean_dir, exist_ok=True)
    os.makedirs(args.out_jsonl, exist_ok=True)

    jsonl_path = Path(args.out_jsonl) / "train.jsonl"

    all_entries = []
    file_count = 0
    for file in os.listdir(args.raw_dir):
        if not (file.lower().endswith(".pdf") or file.lower().endswith(".docx")):
            continue

        file_path = Path(args.raw_dir) / file
        print(f"[INFO] Processing {file_path}...")

        if file.lower().endswith(".pdf"):
            text = extract_text_from_pdf(file_path)
        else:
            text = extract_text_from_docx(file_path)

        if not text.strip():
            print(f"[WARN] No text extracted from {file_path}")
            continue

        # 写入Markdown
        md_name = file.replace(".pdf", ".md").replace(".docx", ".md")
        md_path = Path(args.clean_dir) / md_name
        write_markdown(text, md_path)
        print(f"[INFO] Wrote Markdown: {md_path} ({len(text)} chars)")

        # 生成JSONL数据
        entries = generate_jsonl_entry(text, args.mode)
        all_entries.extend(entries)
        file_count += 1

    # 写入JSONL文件
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for e in all_entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    print(f"[INFO] Processed {file_count} files. JSONL saved at {jsonl_path}, total {len(all_entries)} entries.")

if __name__ == "__main__":
    main()
