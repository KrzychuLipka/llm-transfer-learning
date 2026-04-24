# ============================================
# GEO LLM – Fine-tuning (QLoRA, Mistral 7B)
# ============================================

# ============================================
# HUGGING FACE CACHE & MODEL DOWNLOAD
# ============================================

# 1. (Opcjonalnie) przeniesienie cache modeli na inny dysk
# Windows CMD:
# setx HF_HOME D:\hf_cache
#
# PowerShell:
# $env:HF_HOME="D:\hf_cache"

# 2. (Rekomendowane) ręczne pobranie modelu przed treningiem
# Zapewnia:
# - brak niespodziewanych downloadów w trakcie treningu
# - pełną kontrolę nad wersją modelu (reproducibility)
# - możliwość pracy offline
#
# Instalacja CLI:
# pip install -U "huggingface_hub[cli]"
#
# Pobranie modelu:
# hf download mistralai/Mistral-7B-Instruct-v0.3 --local-dir D:\hf_cache\mistral-7b

# 3. (Opcjonalnie) tryb offline – używać tylko jeśli model jest już w cache
# import os
# os.environ["TRANSFORMERS_OFFLINE"] = "1"

# 4. (Opcjonalnie) wymuszenie cache_dir w kodzie:
# model = AutoModelForCausalLM.from_pretrained(..., cache_dir="D:/hf_cache")

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    BitsAndBytesConfig,
)
from peft import LoraConfig
from trl import SFTTrainer

# ============================================
# DEBUG / SYSTEM INFO
# ============================================

print("=" * 50)
print("SYSTEM INFO")
print("=" * 50)

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(
        f"VRAM total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
    )

print("=" * 50)


# ============================================
# 1. DATASET
# ============================================

print("Step 1/8: Loading dataset...")

dataset = load_dataset("json", data_files="dataset_formatted.jsonl")["train"]

# Split: ważne dla badań → kontrola generalizacji
dataset = dataset.train_test_split(test_size=0.1, seed=42)

train_dataset = dataset["train"]
eval_dataset = dataset["test"]

print(f"Train size: {len(train_dataset)}")
print(f"Eval size: {len(eval_dataset)}")

print("Sample raw example:")
print(train_dataset[0])


# ============================================
# 2. MODEL + TOKENIZER
# ============================================

print("\nStep 2/8: Loading tokenizer & model...")

model_name = "mistralai/Mistral-7B-Instruct-v0.3"

tokenizer = AutoTokenizer.from_pretrained(model_name)

# ważne dla paddingu w modelach instrukcyjnych
tokenizer.pad_token = tokenizer.eos_token


# ============================================
# 3. FORMAT DATASET
# ============================================

print("\nStep 3/8: Formatting dataset (chat → text)...")


def format_chat(example):
    text = tokenizer.apply_chat_template(
        example["messages"], tokenize=False, add_generation_prompt=False
    )
    return {"text": text}


train_dataset = train_dataset.map(format_chat)
eval_dataset = eval_dataset.map(format_chat)

print("Formatted example:")
print(train_dataset[0]["text"][:500])


# ============================================
# 4. LOAD MODEL (QLoRA)
# ============================================

print("\nStep 4/8: Loading model in 4-bit (QLoRA)...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,  # bezpieczniejsze niż bf16 na 3060
)

model = AutoModelForCausalLM.from_pretrained(
    model_name, quantization_config=bnb_config, device_map="auto"
)

model.config.use_cache = False  # wymagane przy gradient checkpointing

print("Model loaded.")

if torch.cuda.is_available():
    print(f"VRAM used after loading: {torch.cuda.memory_allocated()/1e9:.2f} GB")


# ============================================
# 5. LoRA CONFIG
# ============================================

print("\nStep 5/8: Configuring LoRA...")

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

print("LoRA config:", lora_config)


# ============================================
# 6. TRAINING ARGUMENTS
# ============================================

print("\nStep 6/8: Preparing training arguments...")

training_args = TrainingArguments(
    output_dir="./lora-geo-model",
    # --- batch ---
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,
    # --- learning ---
    learning_rate=2e-4,
    warmup_steps=50,
    # --- training length ---
    num_train_epochs=2,
    # --- logging ---
    logging_steps=10,
    save_strategy="epoch",
    evaluation_strategy="epoch",
    # --- memory ---
    gradient_checkpointing=True,
    fp16=True,
    # --- speed ---
    dataloader_num_workers=2,
    # --- misc ---
    report_to="none",
)

print(training_args)


# ============================================
# 7. TRAINER
# ============================================

print("\nStep 7/8: Initializing trainer...")

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=lora_config,
    args=training_args,
    max_seq_length=512,  # KLUCZOWE dla VRAM
    packing=True,  # zwiększa efektywność
)

print("Trainer ready.")


# ============================================
# 8. TRAINING
# ============================================

print("\nStep 8/8: Starting training...")
print("=" * 50)

train_result = trainer.train()

print("=" * 50)
print("TRAINING FINISHED")
print("=" * 50)

# ============================================
# DIAGNOSTYKA
# ============================================

print("\nFinal training metrics:")
print(train_result.metrics)

if torch.cuda.is_available():
    print(f"Final VRAM usage: {torch.cuda.memory_allocated()/1e9:.2f} GB")

# ============================================
# SAVE MODEL
# ============================================

print("\nSaving LoRA adapter...")

trainer.model.save_pretrained("geo-lora")
tokenizer.save_pretrained("geo-lora")

print("Saved in ./geo-lora")


# ============================================
# TEST GENERATION (ważne!)
# ============================================

print("\nRunning quick sanity inference...")

test_prompt = [
    {
        "role": "user",
        "content": "floor: garage level; userPositionInfo: middle of sterilization room;",
    }
]

input_text = tokenizer.apply_chat_template(
    test_prompt, tokenize=False, add_generation_prompt=True
)

inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

with torch.no_grad():
    output = model.generate(**inputs, max_new_tokens=50, temperature=0.2)

print("\nMODEL OUTPUT:")
print(tokenizer.decode(output[0], skip_special_tokens=True))
