import os
import re
import json
import PyPDF2
from pathlib import Path
from typing import Dict, List, Any


class ScriptProcessor:
    def __init__(self):
        # 设置目录路径
        self.raw_dir = Path("data/raw")
        self.cleaned_dir = Path("data/cleaned")
        self.train_dir = Path("data/train_jsonl")

        # 创建必要的目录
        # self.raw_dir.mkdir(parents=True, exist_ok=True)
        # self.cleaned_dir.mkdir(parents=True, exist_ok=True)
        # self.train_dir.mkdir(parents=True, exist_ok=True)

    def process_all_pdfs(self):
        """处理所有PDF文件"""
        print("开始处理PDF文件...")
        pdf_files = list(self.raw_dir.glob("*.pdf"))

        if not pdf_files:
            print(f"在 {self.raw_dir} 目录中未找到PDF文件")
            return

        all_scenes = []

        for pdf_file in pdf_files:
            print(f"\n处理文件: {pdf_file.name}")

            # 提取文本
            text = self.extract_text_from_pdf(pdf_file)

            # 清洗文本
            cleaned_text = self.clean_script_text(text)

            # 保存清洗后的文本
            md_file = self.cleaned_dir / f"{pdf_file.stem}.md"
            with open(md_file, "w", encoding="utf-8") as f:
                f.write(cleaned_text)
            print(f"已保存清洗后的文本: {md_file}")

            # 解析剧本结构
            parsed_script = self.parse_script_structure(cleaned_text)

            # 生成微调数据
            training_data = self.generate_training_data(parsed_script)

            # 保存微调数据
            train_file = self.train_dir / f"{pdf_file.stem}_training.jsonl"
            self.save_training_data(training_data, train_file)
            print(f"已保存微调数据: {train_file}")

            all_scenes.extend(parsed_script["scenes"])

        print(f"\n处理完成! 共处理 {len(pdf_files)} 个PDF文件，提取 {len(all_scenes)} 个场景")

    def extract_text_from_pdf(self, pdf_path: Path) -> str:
        """从PDF文件中提取文本"""
        print(f"正在从PDF提取文本: {pdf_path}")
        text = ""
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page in reader.pages:
                    text += page.extract_text() + "\n"
            print(f"成功提取文本，共{len(text)}字符")
        except Exception as e:
            print(f"提取PDF文本时出错: {e}")
        return text

    def clean_script_text(self, text: str) -> str:
        """清洗剧本文本"""
        print("正在清洗剧本文本...")

        # 分割行
        lines = text.split('\n')
        cleaned_lines = []

        # 保留剧名（第一行）
        if lines:
            cleaned_lines.append(lines[0].strip())

        # 处理剩余行
        for line in lines[1:]:
            line = line.strip()
            if not line:
                continue

            # 保留集标记
            if line.startswith('第') and '集' in line:
                cleaned_lines.append(line)
                continue

            # 保留场景标记
            if re.match(r'\d+-\d+', line):
                print("original line: %s", {line})
                # 去掉"地点,"
                if line.__contains__('地点，') :
                    line = line.replace('地点，', '', 1)  # 只替换第一个匹配的“地点，”
                    print("remove location line: %s", {line})
                # 去掉"地点,"
                if line.__contains__('地点') :
                    line = line.replace('地点', '', 1)  # 只替换第一个匹配的“地点，”
                cleaned_lines.append(line)
                continue

            # 保留人物标记
            if line.startswith('人物：'):
                cleaned_lines.append(line)
                continue

            # 保留动作标记
            if line.startswith('△'):
                cleaned_lines.append(line)
                continue

            # 保留特殊指示
            if '【' in line and '】' in line:
                cleaned_lines.append(line)
                continue

            # 保留对话
            if '：' in line and len(line) < 50:  # 假设对话行不会太长
                cleaned_lines.append(line)
                continue

            # 保留结束标记
            if line.startswith('——'):
                cleaned_lines.append(line)
                continue

        # 重新组合为文本
        cleaned_text = '\n'.join(cleaned_lines)
        print(f"文本清洗完成，共{len(cleaned_text)}字符")
        return cleaned_text

    def parse_script_structure(self, text: str) -> Dict[str, Any]:
        """解析剧本结构"""
        print("正在解析剧本结构...")

        lines = text.strip().split('\n')
        if not lines:
            return {"series_name": "未知", "scenes": []}

        # 提取剧名
        series_name = lines[0].strip()

        scenes = []
        current_episode = None
        current_scene = None
        scene_content = []
        current_characters = []

        for i, line in enumerate(lines[1:]):
            line = line.strip()
            if not line:
                continue

            # 检查是否是集标记
            episode_match = re.search(r'第\s*([零一二三四五六七八九十百千万\d]+)\s*集', line)
            if episode_match:
                current_episode = self.chinese_to_arabic(episode_match.group(1))
                continue

            # 检查是否是场景标记
            scene_match = re.match(r'(\d+-\d+)\s*(.*)', line)
            if scene_match:
                # 保存上一个场景
                if current_scene is not None and scene_content:
                    scenes.append({
                        "episode": current_episode,
                        "scene_number": current_scene["number"],
                        "location": current_scene["location"],
                        "time_of_day": current_scene["time_of_day"],
                        "interior_exterior": current_scene["interior_exterior"],
                        "characters": current_characters,
                        "content": "\n".join(scene_content)
                    })

                # 解析新场景头部
                scene_header = scene_match.group(2)
                scene_info = self.parse_scene_header(scene_header)

                # 开始新场景
                current_scene = {
                    "number": scene_match.group(1),
                    "location": scene_info["location"],
                    "time_of_day": scene_info["time_of_day"],
                    "interior_exterior": scene_info["interior_exterior"]
                }
                current_characters = []
                scene_content = []
                continue

            # 检查是否是人物行
            if line.startswith('人物：'):
                characters_text = line[3:].strip()
                current_characters = self.parse_characters(characters_text)
                continue

            # 添加到当前场景内容
            if current_scene is not None:
                scene_content.append(line)

        # 添加最后一个场景
        if current_scene is not None and scene_content:
            scenes.append({
                "episode": current_episode,
                "scene_number": current_scene["number"],
                "location": current_scene["location"],
                "time_of_day": current_scene["time_of_day"],
                "interior_exterior": current_scene["interior_exterior"],
                "characters": current_characters,
                "content": "\n".join(scene_content)
            })

        result = {
            "series_name": series_name,
            "scenes": scenes
        }

        print(f"解析完成，共{len(scenes)}个场景")
        return result

    def parse_scene_header(self, header: str) -> Dict[str, Any]:
        """解析场景头部信息"""
        # 默认值
        result = {
            "location": "未知",
            "time_of_day": "未知",
            "interior_exterior": "未知"
        }

        # 尝试解析地点、时间、内外景
        parts = [p.strip() for p in header.split('，') if p.strip()]
        if len(parts) >= 3:
            result["location"] = parts[0]
            result["time_of_day"] = parts[1]
            result["interior_exterior"] = parts[2]

        return result

    def parse_characters(self, characters_text: str) -> List[str]:
        """解析人物列表"""
        characters = []
        # 使用逗号分割
        parts = [p.strip() for p in characters_text.split('，') if p.strip()]
        for part in parts:
            # 移除描述部分（如果有）
            if '【' in part and '】' in part:
                char_match = re.match(r'([^【]+)【', part)
                if char_match:
                    characters.append(char_match.group(1).strip())
            else:
                characters.append(part)

        return characters

    def chinese_to_arabic(self, chinese_num: str) -> str:
        """将中文数字转换为阿拉伯数字"""
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

    def generate_training_data(self, parsed_script: Dict[str, Any]) -> List[Dict[str, Any]]:
        """生成微调训练数据"""
        print("正在生成微调训练数据...")

        training_examples = []
        series_name = parsed_script["series_name"]

        for scene in parsed_script["scenes"]:
            # 创建输入-输出对
            # 输入: 场景信息
            input_text = f"剧名: {series_name}\n"
            input_text += f"集数: {scene['episode']}\n"
            input_text += f"场景: {scene['scene_number']}\n"
            input_text += f"地点: {scene['location']}\n"
            input_text += f"时间: {scene['time_of_day']}\n"
            input_text += f"内外景: {scene['interior_exterior']}\n"
            input_text += f"人物: {', '.join(scene['characters'])}\n"

            # 输出: 场景内容
            output_text = scene["content"]

            training_examples.append({
                "input": input_text,
                "output": output_text,
                "type": "scene_generation"
            })

        print(f"生成了 {len(training_examples)} 条训练示例")
        return training_examples

    def save_training_data(self, training_data: List[Dict[str, Any]], output_file: Path):
        """保存训练数据为JSONL格式"""
        with open(output_file, "w", encoding="utf-8") as f:
            for example in training_data:
                f.write(json.dumps(example, ensure_ascii=False) + "\n")


# 主程序
if __name__ == "__main__":
    processor = ScriptProcessor()
    processor.process_all_pdfs()