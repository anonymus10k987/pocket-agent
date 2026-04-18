"""
LoRA Fine-Tuning Script for Qwen2.5-0.5B-Instruct
====================================================
Designed for Google Colab T4 (16GB VRAM).

Usage (on Colab):
    !pip install transformers peft trl datasets accelerate bitsandbytes torch
    !python train/train_lora.py

Or via the Colab notebook (recommended).
"""
import os
import json
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig


# =====================================================================
# Configuration
# =====================================================================
BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "lora_adapter")
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "training_data.jsonl")

MAX_SEQ_LENGTH = 1024
NUM_EPOCHS = 3
BATCH_SIZE = 4
GRADIENT_ACCUMULATION = 4
LEARNING_RATE = 2e-4
WARMUP_STEPS = 10
LOGGING_STEPS = 10
SAVE_STEPS = 50


def load_training_data(path):
    """Load JSONL training data and format for SFTTrainer."""
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line.strip())
            examples.append(data)
    
    print(f"Loaded {len(examples)} training examples")
    return examples


def format_messages_to_text(example, tokenizer):
    """Convert messages format to formatted text using the tokenizer's chat template."""
    messages = example["messages"]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}


def main():
    print("=" * 60)
    print(f"Fine-tuning {BASE_MODEL} with LoRA")
    print("=" * 60)
    
    # ─── Check GPU ───
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
    else:
        print("⚠ No GPU detected! Training will be very slow.")
    
    # ─── Load tokenizer ───
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # ─── Load model with QLoRA (4-bit) ───
    print("Loading model with 4-bit quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )
    model.config.use_cache = False
    
    # ─── LoRA Config ───
    print("Applying LoRA...")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )
    
    model = get_peft_model(model, lora_config)
    trainable, total = model.get_nb_trainable_parameters()
    print(f"Trainable parameters: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")
    
    # ─── Load and format data ───
    print("\nLoading training data...")
    raw_examples = load_training_data(DATA_PATH)
    
    # Convert to dataset
    dataset = Dataset.from_list(raw_examples)
    dataset = dataset.map(
        lambda x: format_messages_to_text(x, tokenizer),
        remove_columns=dataset.column_names,
    )
    
    print(f"Dataset size: {len(dataset)} examples")
    print(f"Sample (first 200 chars): {dataset[0]['text'][:200]}...")
    
    # ─── Training Arguments ───
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        save_total_limit=2,
        fp16=True,
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        max_seq_length=MAX_SEQ_LENGTH,
        dataset_text_field="text",
        report_to="none",
        seed=42,
    )
    
    # ─── Train ───
    print("\n🚀 Starting training...")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    
    trainer.train()
    
    # ─── Save ───
    print(f"\n💾 Saving LoRA adapter to {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    print("\n✅ Training complete!")
    print(f"   Adapter saved to: {OUTPUT_DIR}")
    print(f"   Next step: python train/merge_and_quantize.py")


if __name__ == "__main__":
    main()
