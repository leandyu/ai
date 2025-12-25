import os
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
import torch
import signal
import sys

# model_name = "deepseek-ai/deepseek-llm-7b-base"  # HuggingFace 模型
model_name = "DeepSeek-R1-Distill-Qwen-7B"
train_data_path = "data/train_jsonl/train.jsonl"  # 数据文件

print("✅ Step 1: 加载分词器...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
print("✅ 分词器加载成功")

print("✅ Step 2: 加载模型（可能比较慢）...")
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"✅ 使用设备: {device}")

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if device.type=="mps" else torch.float32,
    device_map={"": device}
)


print("✅ 模型加载成功")

print("✅ Step 3: 应用 LoRA 配置...")
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],  # 典型Transformer结构
    lora_dropout=0.05,
    task_type=TaskType.CAUSAL_LM
)
model = get_peft_model(model, lora_config)
print("✅ LoRA 配置应用完成")

print("✅ Step 4: 加载训练数据...")
if not os.path.exists(train_data_path):
    raise FileNotFoundError(f"❌ 训练数据文件不存在: {train_data_path}")

dataset = load_dataset("json", data_files={"train": train_data_path})
print(f"✅ 数据加载成功，共 {len(dataset['train'])} 条")


def preprocess(example):
    instruction = example.get("instruction", "")
    input_text = example.get("input", "")
    output_text = example.get("output", "")

    # 拼成模型输入
    text = f"### 指令:\n{instruction}\n\n### 输入:\n{input_text}\n\n### 回答:\n{output_text}"

    # tokenizer encode
    tokenized = tokenizer(text, truncation=True, padding="max_length", max_length=512)

    # labels = input_ids
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized


dataset = dataset.map(preprocess)
print("✅ 数据预处理完成")

print("✅ Step 5: 设置训练参数...")
training_args = TrainingArguments(
    output_dir="lora-output",
    per_device_train_batch_size=2, #通俗解释： 这个超级大脑一次能吃下多少条训练数据。这里设定为一次看2条。
    gradient_accumulation_steps=16, #通俗解释： “存钱罐”策略。因为一次只处理2条数据（上面设置的），可能太少了，学习方向不稳定。那就让它先偷偷地内部模拟学习16次（每次2条），把这16次学习的经验攒起来，然后再一次性真正地更新大脑
    warmup_steps=50, #训练前的“热身运动”。在最开始的50步训练里，学习率会从0慢慢地、线性地增加到你设定的初始值（2e-4）
    max_steps=300, # 默认1000， 整个训练计划总共要训练多少步。训练完1000步就自动结束
    learning_rate=2e-4, #学习的速度，或者说每次调整的“步伐”大小。2e-4 就是 0.0002。
    logging_dir="logs",
    logging_steps=10, #每训练多少步，就记录一次日志。这里设定为每10步记录一次。
    save_total_limit=2, #最多保留几个模型的“存档点”。这里最多保留2个
    bf16=False,   # Mac MPS 不支持 bf16
    fp16=False    # 禁用 Accelerate 的 fp16 mixed precision
)
print("✅ 训练参数设置完成")

# 定义一个信号处理函数
""" 
不要粗暴地关闭终端。
去运行程序的终端窗口，按一次 Ctrl + C。
观察输出日志，你会看到 Trainer 正在保存检查点（类似 Saving model checkpoint to lora-output/checkpoint-XXX 的提示）。
等待程序自己退出。
完成后，你的 lora-output 目录里就是中断时保存的最终模型。你可以用它来进行推理，或者将来通过 trainer.train(resume_from_checkpoint=True) 从这个点恢复训练
"""
def signal_handler(sig, frame):
    print(f'\n⚠️  收到中断信号，正在保存模型并退出...')
    # 保存模型
    model.save_pretrained("lora-output")
    print('✅ 权重已保存到 "lora-output" 目录')
    sys.exit(0)

# 注册信号处理器，监听 Ctrl+C (SIGINT) 信号
signal.signal(signal.SIGINT, signal_handler)



print("✅ Step 6: 启动训练...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"]
)
try:
    trainer.train()
    print("✅ 训练完成")
except Exception as e:
    print(f"❌ 训练出错: {e}")
finally:
    print("✅ Step 7: 保存LoRA权重...")
    model.save_pretrained("lora-output")
    print("✅ 权重保存到 lora-output 目录")
