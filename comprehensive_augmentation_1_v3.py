
import requests
import json
from pathlib import Path

def augment_training_data(original_file, output_file, augmentation_factor=3):
    """增强训练数据"""
    augmented_samples = []

    # 确保输入文件存在
    if not Path(original_file).exists():
        print(f"错误：文件 {original_file} 不存在")
        return

    with open(original_file, 'r', encoding='utf-8') as f:
        original_samples = [json.loads(line) for line in f]

    print(f"原始数据有 {len(original_samples)} 条样本")

    for sample in original_samples:
        # 保留原始样本
        augmented_samples.append(sample)

        # 创建变体样本
        for i in range(augmentation_factor - 1):
            augmented_sample = sample.copy()

            # 简单的数据增强策略
            if "input" in augmented_sample:
                # 替换地点
                locations = ["宴会厅", "会议室", "卧室", "客厅", "办公室", "餐厅", "厨房", "阳台", "花园", "街道"]
                for loc in locations:
                    if loc in augmented_sample["input"]:
                        new_loc = locations[(locations.index(loc) + i + 1) % len(locations)]
                        augmented_sample["input"] = augmented_sample["input"].replace(loc, new_loc)
                        break

                # 替换时间
                times = ["日", "夜", "早晨", "傍晚", "午后", "午夜", "黎明", "黄昏"]
                for t in times:
                    if t in augmented_sample["input"]:
                        new_time = times[(times.index(t) + i + 1) % len(times)]
                        augmented_sample["input"] = augmented_sample["input"].replace(t, new_time)
                        break

                # 替换人物（可选）
                characters = ["楚音韵", "秦云上", "秦母", "江雅", "楚江东", "秦四海","秦东林",  "沈老二", "恐怖头目", "西装保镖", "群演"]
                for char in characters:
                    if char in augmented_sample["input"]:
                        new_char = characters[(characters.index(char) + i + 1) % len(characters)]
                        augmented_sample["input"] = augmented_sample["input"].replace(char, new_char)
                        break

            augmented_samples.append(augmented_sample)

    # 保存增强后的数据
    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in augmented_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')

    print(f"数据增强完成：{len(original_samples)} → {len(augmented_samples)} 条样本")
    print(f"增强后的数据已保存到: {output_file}")


def parse_script_input(input_text):
    """解析剧本输入文本为字典"""
    fields = {}
    lines = input_text.split('\n')

    for line in lines:
        if ':' in line:
            key, value = line.split(':', 1)
            fields[key.strip()] = value.strip()

    return fields


def build_script_input(fields):
    """从字典构建剧本输入文本"""
    lines = []
    for key in ["剧名", "集数", "场景", "地点", "时间", "内外景"]:
        if key in fields:
            lines.append(f"{key}: {fields[key]}")

    return "\n".join(lines)


def augment_training_data_structured(original_file, output_file, augmentation_factor=3):
    """使用结构化方法增强训练数据"""
    augmented_samples = []

    # 确保输入文件存在
    if not Path(original_file).exists():
        print(f"错误：文件 {original_file} 不存在")
        return

    with open(original_file, 'r', encoding='utf-8') as f:
        original_samples = [json.loads(line) for line in f]

    print(f"原始数据有 {len(original_samples)} 条样本")

    # 定义替换选项
    location_options = ["宴会厅", "会议室", "卧室", "客厅", "办公室", "餐厅", "厨房", "阳台", "花园", "街道"]
    time_options = ["日", "夜", "早晨", "傍晚", "午后", "午夜", "黎明", "黄昏"]
    interior_options = ["内", "外"]

    for sample_idx, sample in enumerate(original_samples):
        # 保留原始样本
        augmented_samples.append(sample)

        # 解析输入文本
        if "input" in sample:
            fields = parse_script_input(sample["input"])

            # 创建变体样本
            for i in range(augmentation_factor - 1):
                augmented_sample = sample.copy()
                new_fields = fields.copy()  # 创建字段的副本

                # 替换地点
                if "地点" in new_fields and new_fields["地点"] in location_options:
                    current_idx = location_options.index(new_fields["地点"])
                    new_fields["地点"] = location_options[(current_idx + i + 1) % len(location_options)]

                # 替换时间
                if "时间" in new_fields and new_fields["时间"] in time_options:
                    current_idx = time_options.index(new_fields["时间"])
                    new_fields["时间"] = time_options[(current_idx + i + 1) % len(time_options)]

                # 替换内外景
                if "内外景" in new_fields and new_fields["内外景"] in interior_options:
                    current_idx = interior_options.index(new_fields["内外景"])
                    new_fields["内外景"] = interior_options[(current_idx + i + 1) % len(interior_options)]

                # 重建输入文本
                augmented_sample["input"] = build_script_input(new_fields)
                augmented_samples.append(augmented_sample)

    # 保存增强后的数据
    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in augmented_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')

    print(f"数据增强完成：{len(original_samples)} → {len(augmented_samples)} 条样本")
    print(f"增强后的数据已保存到: {output_file}")

def semantic_augmentation(text, api_url="http://localhost:11434/api/generate"):
    """使用语言模型进行语义增强"""
    try:
        prompt = f"""
        请对以下剧本描述进行改写，保持相同的意思但使用不同的表达方式：

        原文本: {text}

        请只返回改写后的文本，不要添加其他内容。
        """

        response = requests.post(
            api_url,
            json={
                "model": "deepseek-coder",
                "prompt": prompt,
                "max_tokens": 200,
                "temperature": 0.8,
                "stream": False
            },
            timeout=30
        )

        if response.status_code == 200:
            return response.json().get("response", "").strip()
        else:
            return text  # 如果失败，返回原文本
    except:
        return text  # 如果失败，返回原文本

# 在增强函数中添加语义增强
def augment_with_semantic(original_file, output_file):
    """使用语义增强"""
    augmented_samples = []

    with open(original_file, 'r', encoding='utf-8') as f:
        original_samples = [json.loads(line) for line in f]

    for sample in original_samples:
        # 保留原始样本
        augmented_samples.append(sample)

        # 创建语义增强样本
        augmented_sample = sample.copy()
        if "input" in augmented_sample:
            augmented_sample["input"] = semantic_augmentation(augmented_sample["input"])

        augmented_samples.append(augmented_sample)

    # 保存增强后的数据
    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in augmented_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')

    print(f"语义增强完成：{len(original_samples)} → {len(augmented_samples)} 条样本")


def comprehensive_augmentation(original_file, combined_file):
    """综合增强策略"""
    # 先进行基础增强 --
    base_augmented_file = "../data/train_jsonl/base_augmented.jsonl"
    augment_training_data(original_file, base_augmented_file, augmentation_factor=2)

    # 再进行语义增强
    augment_with_semantic(base_augmented_file, combined_file)

    # 清理临时文件
    # Path(base_augmented_file).unlink()

    print(f"综合增强完成，结果保存到: {combined_file}")


# def merge_training_files(file_paths, output_file):
#     """合并多个训练文件"""
#     all_samples = []
#
#     for file_path in file_paths:
#         if Path(file_path).exists():
#             with open(file_path, 'r', encoding='utf-8') as f:
#                 samples = [json.loads(line) for line in f]
#                 all_samples.extend(samples)
#                 print(f"从 {file_path} 添加了 {len(samples)} 条样本")
#         else:
#             print(f"警告：文件 {file_path} 不存在")
#
#     # 保存合并后的数据
#     with open(output_file, 'w', encoding='utf-8') as f:
#         for sample in all_samples:
#             f.write(json.dumps(sample, ensure_ascii=False) + '\n')
#
#     print(f"合并完成：总共 {len(all_samples)} 条样本")
#     print(f"合并后的数据已保存到: {output_file}")

# 使用示例
if __name__ == "__main__":
    # 合并多个文件
    comprehensive_augmentation(
        "../data/train_jsonl/godfather_training.jsonl",
        # 可以添加其他剧本文件
        "../data/train_jsonl/combined_training.jsonl")