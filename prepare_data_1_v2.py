#!/usr/bin/env python3
import os
import argparse
import json
import re
from pathlib import Path
from typing import List, Dict, Any
from PyPDF2 import PdfReader
import docx


def extract_text_from_pdf(pdf_path):
    """从PDF文件提取文本"""
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
    """从DOCX文件提取文本"""
    try:
        doc = docx.Document(docx_path)
        text = "\n".join([p.text for p in doc.paragraphs])
        return text.strip()
    except Exception as e:
        print(f"[ERROR] Cannot read DOCX {docx_path}: {e}")
        return ""


def clean_scene_content(text):
    """
    清洗场景内容，去除无关字符和格式标记
    但保留剧本的基本结构和标记
    """
    # 移除多余的空格和换行
    text = re.sub(r'\s+', ' ', text)

    # 保留基本标点符号
    text = re.sub(r'[^\u4e00-\u9fff\u3000-\u303f\uff00-\uffef\w\.\,\!\\?\;\\:\'\"\-\s]', '', text)

    # 标准化标点符号
    text = text.replace('。', '.').replace('，', ',').replace('！', '!').replace('？', '?')

    return text.strip()


def parse_script_structure(script_content: str) -> List[Dict[str, Any]]:
    """
    解析剧本结构，按集和场景分割
    使用原始文本进行解析，保留关键标记
    """
    # 先按集分割
    episodes = re.split(r'(第[零一二三四五六七八九十百千万\d]+集)', script_content)

    # 移除第一个空元素（如果有）
    if not episodes[0].strip():
        episodes = episodes[1:]

    scenes = []
    current_episode = None

    for i in range(0, len(episodes), 2):
        if i + 1 >= len(episodes):
            break

        episode_title = episodes[i]
        episode_content = episodes[i + 1]

        # 提取集数 - 支持中文和阿拉伯数字
        episode_match = re.search(r'第([零一二三四五六七八九十百千万\d]+)集', episode_title)
        if episode_match:
            # 将中文数字转换为阿拉伯数字
            current_episode = chinese_to_arabic(episode_match.group(1))

        # 按场景分割
        scene_parts = re.split(r'(\d+-\d+.*?地点.*?\n)', episode_content)

        # 移除第一个空元素（如果有）
        if not scene_parts[0].strip():
            scene_parts = scene_parts[1:]

        for j in range(0, len(scene_parts), 2):
            if j + 1 >= len(scene_parts):
                break

            scene_header = scene_parts[j]
            scene_content = scene_parts[j + 1]

            # 解析场景头部信息
            scene_info = parse_scene_header(scene_header)

            # 提取特殊指示（如【】内的内容）
            special_instructions = re.findall(r'【(.*?)】', scene_content)

            # 只清洗场景内容，保留剧本结构标记
            clean_content = clean_scene_content(scene_content)

            # 添加到场景列表
            scenes.append({
                "episode": current_episode,
                "scene_header": scene_header.strip(),
                "scene_content": clean_content.strip(),
                "special_instructions": special_instructions,
                **scene_info
            })

    return scenes


def chinese_to_arabic(chinese_num: str) -> str:
    """
    将中文数字转换为阿拉伯数字
    支持: 零一二三四五六七八九十百千万
    """
    # 如果已经是阿拉伯数字，直接返回
    if chinese_num.isdigit():
        return chinese_num

    # 中文数字映射
    chinese_digits = {
        '零': 0, '一': 1, '二': 2, '三': 3, '四': 4,
        '五': 5, '六': 6, '七': 7, '八': 8, '九': 9,
        '十': 10, '百': 100, '千': 1000, '万': 10000
    }

    # 特殊处理"十"开头的数字
    if chinese_num.startswith('十'):
        chinese_num = '一' + chinese_num

    result = 0
    temp = 0
    prev_digit = 0

    for char in chinese_num:
        if char in chinese_digits:
            digit = chinese_digits[char]

            if digit >= 10:  # 十、百、千、万
                if temp == 0:
                    temp = 1
                result += temp * digit
                temp = 0
            else:  # 零到九
                temp = temp * 10 + digit
        else:
            # 遇到非数字字符，停止转换
            break

    result += temp

    return str(result)


def parse_scene_header(header: str) -> Dict[str, Any]:
    """解析场景头部信息"""
    # 提取场景编号
    scene_match = re.search(r'(\d+-\d+)', header)
    scene_number = scene_match.group(1) if scene_match else "未知"

    # 提取地点
    location_match = re.search(r'地点[，,:：]\s*([^，,]+)', header)
    location = location_match.group(1) if location_match else "未知"

    # 提取时间
    time_match = re.search(r'[，,]\s*([日早晚夜]+)\s*[，,]', header)
    time_of_day = time_match.group(1) if time_match else "未知"

    # 提取内外景
    interior_match = re.search(r'[，,]\s*([内外]+)\s*[，,]', header)
    interior = interior_match.group(1) if interior_match else "未知"

    # 提取人物
    characters_match = re.search(r'人物[：:]\s*(.+)', header)
    characters_text = characters_match.group(1) if characters_match else "未知"

    # 分割人物
    characters = []
    if characters_text != "未知":
        # 多种可能的分隔符
        for sep in ["，", ",", "、", "和", "及"]:
            if sep in characters_text:
                characters = [c.strip() for c in characters_text.split(sep)]
                break
        else:
            characters = [characters_text.strip()]

    return {
        "scene_number": scene_number,
        "location": location,
        "time_of_day": time_of_day,
        "interior_exterior": interior,
        "characters": characters
    }


def generate_markdown_from_scenes(scenes: List[Dict[str, Any]], output_path: str):
    """从场景数据生成Markdown内容"""
    md_content = "# 剧本分析\n\n"

    current_episode = None
    for scene in scenes:
        # 添加集标题
        if scene["episode"] != current_episode:
            md_content += f"## 第{scene['episode']}集\n\n"
            current_episode = scene["episode"]

        # 添加场景标题和信息
        md_content += f"### 场景 {scene['scene_number']}\n\n"
        md_content += f"- **地点**: {scene['location']}\n"
        md_content += f"- **时间**: {scene['time_of_day']}\n"
        md_content += f"- **内外景**: {scene['interior_exterior']}\n"
        md_content += f"- **人物**: {', '.join(scene['characters'])}\n\n"

        # 添加场景头部原文
        md_content += "#### 场景头部\n\n"
        md_content += f"{scene['scene_header']}\n\n"

        # 添加特殊指示（如果有）
        if scene["special_instructions"]:
            md_content += "#### 特殊指示\n\n"
            for instruction in scene["special_instructions"]:
                md_content += f"- {instruction}\n"
            md_content += "\n"

        # 添加场景内容
        md_content += "#### 内容\n\n"
        md_content += f"{scene['scene_content']}\n\n"

        md_content += "---\n\n"

    # 写入Markdown文件
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(md_content)


def generate_fine_tuning_data(scenes: List[Dict[str, Any]], output_dir: str):
    """生成微调数据"""
    os.makedirs(output_dir, exist_ok=True)

    # 三种微调目标的数据
    continuation_data = []
    recreation_data = []
    character_story_data = []

    for scene in scenes:
        content = scene["scene_content"]
        characters = scene["characters"]

        # 1. 续写数据
        continuation_data.append({
            "instruction": "根据以下剧本场景继续写后续剧情，保持风格一致：",
            "input": content,
            "output": ""
        })

        # 2. 二次创作数据
        recreation_data.append({
            "instruction": "对以下剧本场景进行二次创作，保持原文风格但加入新的情节：",
            "input": content,
            "output": ""
        })

        # 3. 角色故事数据（为每个角色生成一条数据）
        for character in characters:
            # 前传
            character_story_data.append({
                "instruction": f"根据以下剧本场景，创作角色'{character}'的前传故事，保持角色性格一致：",
                "input": content,
                "output": ""
            })

            # 后传
            character_story_data.append({
                "instruction": f"根据以下剧本场景，创作角色'{character}'的后传故事，保持角色性格一致：",
                "input": content,
                "output": ""
            })

    # 写入JSONL文件
    with open(os.path.join(output_dir, "continuation.jsonl"), "w", encoding="utf-8") as f:
        for item in continuation_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    with open(os.path.join(output_dir, "recreation.jsonl"), "w", encoding="utf-8") as f:
        for item in recreation_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    with open(os.path.join(output_dir, "character_stories.jsonl"), "w", encoding="utf-8") as f:
        for item in character_story_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"已生成微调数据：")
    print(f"  - 续写数据: {len(continuation_data)} 条")
    print(f"  - 二次创作数据: {len(recreation_data)} 条")
    print(f"  - 角色故事数据: {len(character_story_data)} 条")


def main():
    parser = argparse.ArgumentParser(description="处理剧本文件并生成微调数据")
    parser.add_argument("--raw_dir", required=True, help="原始PDF或DOCX文件目录")
    parser.add_argument("--cleaned_dir", required=True, help="输出Markdown文件目录")
    parser.add_argument("--out_jsonl", required=True, help="输出微调数据目录")
    args = parser.parse_args()

    # 创建输出目录
    if not os.path.exists(args.cleaned_dir):
        os.makedirs(args.cleaned_dir)
    if not os.path.exists(args.out_jsonl):
        os.makedirs(args.out_jsonl)

    all_scenes = []

    # 处理所有原始文件
    for file_name in os.listdir(args.raw_dir):
        if not (file_name.lower().endswith(".pdf") or file_name.lower().endswith(".docx")):
            continue

        file_path = os.path.join(args.raw_dir, file_name)
        print(f"[INFO] 处理文件: {file_name}")

        # 提取文本
        if file_name.lower().endswith(".pdf"):
            text = extract_text_from_pdf(file_path)
        else:
            text = extract_text_from_docx(file_path)

        if not text.strip():
            print(f"[WARN] 未从 {file_name} 提取到文本")
            continue
        print(f"[INFO] 已生成Text: {text}")

        # 解析剧本结构（使用原始文本）
        scenes = parse_script_structure(text)
        print(f"[INFO] 已生成scenes: {scenes}")
        all_scenes.extend(scenes)

        # 生成Markdown文件
        base_name = os.path.splitext(file_name)[0]
        md_path = os.path.join(args.cleaned_dir, f"{base_name}.md")
        generate_markdown_from_scenes(scenes, md_path)
        print(f"[INFO] 已生成Markdown: {md_path}")

    # 生成微调数据
    if all_scenes:
        generate_fine_tuning_data(all_scenes, args.train_jsonl)
        print(f"[INFO] 总共处理 {len(all_scenes)} 个场景")
    else:
        print("[WARN] 未找到任何场景数据")


if __name__ == "__main__":
    main()