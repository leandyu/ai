import re
from typing import List, Dict, Any


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
    location_match = re.search(r'地点[，,:：]\s*([^，,\s]+)', header)
    location = location_match.group(1) if location_match else "未知"

    # 提取时间
    time_match = re.search(r'[，,]\s*([日早晚夜]+)\s*[，,]', header)
    time_of_day = time_match.group(1) if time_match else "未知"

    # 提取内外景
    interior_match = re.search(r'[，,]\s*([内外]+)\s*$', header)
    if not interior_match:
        interior_match = re.search(r'[，,]\s*([内外]+)\s*[，,]', header)
    interior = interior_match.group(1) if interior_match else "未知"

    # 提取人物 - 处理可能的多行情况
    characters_text = ""
    lines = header.split('\n')
    for line in lines:
        if '人物' in line:
            characters_match = re.search(r'人物[：:]\s*(.+)', line)
            if characters_match:
                characters_text = characters_match.group(1)
                # 检查是否有换行继续
                next_line_idx = lines.index(line) + 1
                if next_line_idx < len(lines) and not re.search(r'\d+-\d+', lines[next_line_idx]):
                    characters_text += " " + lines[next_line_idx].strip()
            break

    # 分割人物
    characters = []
    if characters_text and characters_text != "未知":
        # 先按逗号分割
        parts = re.split(r'[，,]', characters_text)

        for part in parts:
            part = part.strip()
            # 处理"和"、"及"等连接词
            if '和' in part or '及' in part or '以及' in part:
                sub_parts = re.split(r'和|及|以及', part)
                characters.extend([p.strip() for p in sub_parts if p.strip()])
            else:
                characters.append(part)

    return {
        "scene_number": scene_number,
        "location": location,
        "time_of_day": time_of_day,
        "interior_exterior": interior,
        "characters": characters
    }


def clean_scene_content(text: str) -> str:
    """
    清洗场景内容，去除无关字符和格式标记
    但保留剧本的基本结构和标记
    """
    # 移除多余的空格和换行，但保留段落结构
    text = re.sub(r' +', ' ', text)
    text = re.sub(r'\n\s*\n', '\n\n', text)

    # 保留基本标点符号和中文标点
    text = re.sub(r'[^\u4e00-\u9fff\u3000-\u303f\uff00-\uffef\w\.\,\!\\?\;\\:\'\"\-\s△【】]', '', text)

    # 标准化标点符号但不强制转换中文标点
    text = text.replace('。。', '。').replace('，，', '，')

    return text.strip()


def parse_script_structure(script_content: str) -> Dict[str, Any]:
    """
    解析剧本结构，返回包含剧名和所有场景的完整结构
    """
    # 提取剧名（通常是第一行）
    lines = script_content.strip().split('\n')
    series_name = lines[0].strip()

    # 移除剧名行，继续解析其余内容
    remaining_content = '\n'.join(lines[1:])

    # 按集分割
    episodes = re.split(r'(第[零一二三四五六七八九十百千万\d]+集)', remaining_content)

    # 移除第一个空元素（如果有）
    if episodes and not episodes[0].strip():
        episodes = episodes[1:]

    scenes = []
    current_episode = None

    for i in range(0, len(episodes), 2):
        if i + 1 >= len(episodes):
            break

        episode_title = episodes[i]
        episode_content = episodes[i + 1]

        # 提取集数
        episode_match = re.search(r'第([零一二三四五六七八九十百千万\d]+)集', episode_title)
        if episode_match:
            current_episode = chinese_to_arabic(episode_match.group(1))

        # 按场景分割
        scene_parts = re.split(r'(\d+-\d+.*?(?:\n.*?)*?(?=\d+-\d+|\Z))', episode_content)

        # 移除第一个空元素（如果有）
        if scene_parts and not scene_parts[0].strip():
            scene_parts = scene_parts[1:]

        for j in range(0, len(scene_parts), 2):
            if j + 1 >= len(scene_parts):
                break

            scene_header = scene_parts[j]
            scene_content = scene_parts[j + 1]

            # 解析场景头部信息
            scene_info = parse_scene_header(scene_header)

            # 提取特殊指示
            special_instructions = re.findall(r'【(.*?)】', scene_content)

            # 清洗场景内容
            clean_content = clean_scene_content(scene_content)

            # 添加到场景列表
            scenes.append({
                "episode": current_episode,
                "scene_header": scene_header.strip(),
                "scene_content": clean_content.strip(),
                "special_instructions": special_instructions,
                **scene_info
            })

    # 返回完整结构，剧名在最外层
    return {
        "series_name": series_name,
        "scenes": scenes
    }


def build_vector_documents(parsed_script: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    将解析后的剧本结构转换为向量数据库文档格式
    每个场景作为一个独立文档，但包含剧名作为元数据
    """
    documents = []
    series_name = parsed_script["series_name"]

    for scene in parsed_script["scenes"]:
        # 创建文档内容 - 结合所有相关信息
        content_parts = [
            f"剧名: {series_name}",
            f"集数: {scene['episode']}",
            f"场景: {scene['scene_number']}",
            f"地点: {scene['location']}",
            f"时间: {scene['time_of_day']}",
            f"内外景: {scene['interior_exterior']}",
            f"人物: {', '.join(scene['characters'])}",
            f"内容: {scene['scene_content']}"
        ]

        # 添加特殊指示（如果有）
        if scene['special_instructions']:
            content_parts.append(f"特殊指示: {', '.join(scene['special_instructions'])}")

        # 合并所有部分
        content = "\n".join(content_parts)

        # 创建文档元数据
        metadata = {
            'series_name': series_name,
            'episode': scene['episode'],
            'scene_number': scene['scene_number'],
            'location': scene['location'],
            'time_of_day': scene['time_of_day'],
            'interior_exterior': scene['interior_exterior'],
            'characters': scene['characters'],
            'special_instructions': scene['special_instructions']
        }

        documents.append({
            "content": content,
            "metadata": metadata
        })

    return documents


# 使用示例
if __name__ == "__main__":
    script_text = """教父
第1集
1-1地点，宴会厅，日，内
人物：沈老二，至少二十名男女老少群演，恐怖头目，以及至少六名以上恐怖分子，四名西
装保镖
△一个比较大的宴会厅。
好几桌人，很热闹。
门口站着好几名西装笔挺的保镖。
沈老二起身大喊：来，为我们沈家资产排名世界五百强举杯，大哥不在，我替他敬各位了！
众人一起起身，声音洪亮：干！
画外音大吼：什么人！
众人循声看去。
只见得头上带着黑头罩的恐怖分子迅猛麻利的冲到门口，直接将西装保镖割喉击杀。【手段
非常狠辣】
三两下就把保镖解决，随即退开。
后面多名黑头罩男子怀抱冲锋枪冲入瞄准控制现场。
全场震惊愕然。
沈老二厉声：什么人，竟敢闯我北都沈家！
一名个子高大的黑头罩男子空手进入。【气场做出来】
目光环视：东欧恶龙组织。
众人一下子震惊颤抖，沈老二立刻颤抖：啊，恶，恶龙组织。
恐怖头目：哪位是沈家负责人！
都看向沈老二。
沈老二颤抖：我，我大哥，没在。
恐怖头目缓缓走进他：没在是吧？
沈老二哆嗦：是，是。
恐怖头目手往身上一抹，匕首在手，抓住沈老二，狠狠的就往他身上插了好多下，然后推倒
在地。【手段，鲜血，必须视觉冲击】
回头对其余恐怖分子下令：把这里的人全杀了，写成血书，送去沈家！
立即就是密集的子弹扫射，现场惨不忍睹的死亡和倒下，枪声，惨叫。【这个场景一定要拍
得噼噼啪啪激烈刺激一点】
1-2 会议室，日，内
沈出山，沈京雪，西装墨镜保镖 9 名，周吴郑王四家主，另外加两个群演
△沈出山坐正中位置。
沈京雪坐下首位，周吴郑王和两位看着富有的群演坐两边。
西装墨镜保镖肃杀戒备四周。【整个紧张肃杀的氛围感必须出来，众人很焦急的语气要拍出
来】
沈出山面容严肃：想必各位家主都已经收到了恶龙组织的死亡血书了吧！
周家主急：这下怎么办，恶龙组织是想把我们北都五大家族一网打尽啊！
吴家主：是的，我吴家总资产都不过三万亿，他们竟然开价五万亿，就是要亡我吴家啊！
郑家主：可是恶龙组织盘踞东欧，在全球数十个国家发动恐怖袭击，从未失手过，他们就是
人间最恐怖的恶魔，世界超级大国都深感忌惮，我们哪敢说个不字！
王家主绝望：难道，我们几辈人努力的庞大家业，就要毁于一旦了吗！
周家主看过去：山爷，北都豪门你为龙头，你倒是说句话啊！
沈出山：我想了很久，只有一个人能帮我们了。
下面的第一个人问出：谁！
后面接连的人跟着问出：谁！谁！谁！【这个急切的情绪和节奏一定要做好】
目光都聚焦在沈出山脸上。
沈出山缓缓说：教父！
全场人都一起震惊：教父！
周家主【表情要夸张】：山爷说的可是那位在金融，军事，科技，医学，娱乐，武道，甚至
地下世界数十个领域都登峰造极的神秘王者！统领造神殿的教父！
第 2 集
2-1 会议室，日，内
沈出山，沈京雪，西装墨镜保镖 9 名，周吴郑王四家主，另外加两个群演
△周家主【表情要夸张】：山爷说的可是那位在金融，军事，科技，医学，娱乐，武道，甚
至地下世界数十个领域都登峰造极的神秘王者！统领造神殿的教父！
沈出山：除了他，还能有谁！
周家主：如果能得他帮忙，区区恶龙组织自然不在话下，只可惜他如天上皓月，神龙见首不
见尾，世上只有他的传说，却无人见过，我们怎么找他。
吴家主：还有，传说他在震惊世界的巅峰时隐退，不再救世造神，就算我们能找到他，他也
不可能平白无故的帮我们！
沈出山：事到如今，这唯一的一线生机，只有死马当作活马医了。而且，他也不是全无软肋。
吴家主：他有什么软肋？【众人的视线聚焦过去】
沈出山：他就算创造了这世间再多的神话，也终究只是个男人，需要娶妻生子，我听过一些
传闻，说他喜欢女人，喜欢胸大腿长高贵而且纯洁的女人！
目光看向沈京雪：京雪，就看你的了！
沈京雪一愣：我？
沈出山：虽然，教父也未必看得上你，但你已是当世第一美女，有着大夏首富千金的身份，
万一呢！
吴家主：对对对，就算有一线生机，我们都不要放过，就拜托沈侄女了。
沈京雪：实话，我也倾慕教父已久，能嫁给那样神话般的人物也是我梦寐以求，然而，我去
哪里找他呢？
沈出山的目光一下放远：我知道。
沈京雪：在哪？
沈出山：京海！
下一个镜头迅速切换。"""

    # 解析剧本
    parsed_script = parse_script_structure(script_text)
    print(f"剧名: {parsed_script['series_name']}")
    print(f"场景数量: {len(parsed_script['scenes'])}")

    # 构建向量数据库文档
    documents = build_vector_documents(parsed_script)
    print(f"\n第一个文档的内容预览:")
    print(documents[0]['content'][:200] + "...")
    print(f"\n第一个文档的元数据:")
    print(documents[0]['metadata'])
    print(f"\n第二个文档的内容预览:")
    print(documents[1]['content'][:200] + "...")
    print(f"\n第二个文档的元数据:")
    print(documents[1]['metadata'])
    print(f"\n第三个文档的内容预览:")
    print(documents[2]['content'][:200] + "...")
    print(f"\n第三个文档的元数据:")
    print(documents[2]['metadata'])