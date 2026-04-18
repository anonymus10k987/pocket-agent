# ============================================================
# CELL 2: Generate Data + Train + Quantize + Download
# Run this AFTER Cell 1 has restarted the runtime.
# ============================================================

import json
import random
import hashlib
import os
import re
import subprocess
import sys

random.seed(42)

# ============================================================
# STEP 1: Generate Training Data
# ============================================================
print("=" * 60)
print("STEP 1: Generating training data...")
print("=" * 60)

SYSTEM_PROMPT = (
    "You are a helpful mobile assistant. You have access to the following tools: "
    "weather, calendar, convert, currency, sql. "
    "When the user's request matches a tool, respond with a tool call in "
    "<tool_call>{...}</tool_call> tags. Otherwise, respond in plain text."
)

CITIES = [
    "Paris", "Berlin", "Madrid", "Rome", "Mumbai", "Islamabad", "Karachi",
    "Lahore", "Shanghai", "Beijing", "Seoul", "Bangkok", "Jakarta", "Cairo",
    "Nairobi", "Toronto", "Mexico City", "Riyadh", "Doha", "Ankara",
    "Vienna", "Prague", "Warsaw", "Oslo", "Stockholm", "Dublin", "Barcelona",
    "Amsterdam", "Copenhagen", "Singapore", "Manila", "Taipei", "Osaka",
    "Melbourne", "Sydney", "Auckland", "Johannesburg", "Casablanca",
    "Dhaka", "Colombo", "Kathmandu", "Hanoi", "London", "New York",
    "Tokyo", "Dubai", "San Francisco", "Chicago", "Los Angeles", "Moscow",
]

WEATHER_TEMPLATES_C = [
    "What's the weather in {city}?",
    "Tell me the weather in {city}",
    "How's the weather in {city}?",
    "Weather in {city} please",
    "What is the temperature in {city}?",
    "Is it hot in {city} today?",
    "Give me the forecast for {city}",
    "Check the weather for {city}",
    "How warm is it in {city}?",
    "What's it like outside in {city}?",
]

WEATHER_TEMPLATES_F = [
    "What's the weather in {city} in Fahrenheit?",
    "Show me {city} temperature in Fahrenheit",
    "{city} weather in F",
    "Tell me the temperature in {city}, Fahrenheit please",
]

CALENDAR_LIST_TEMPLATES = [
    "What events do I have on {date}?",
    "Show my calendar for {date}",
    "What's on my schedule for {date}?",
    "Any events on {date}?",
    "What's planned for {date}?",
    "List my events for {date}",
    "Do I have anything on {date}?",
]

EVENT_TITLES = [
    "Team Standup", "Sprint Review", "Dentist Appointment", "Lunch with Ali",
    "Doctor Visit", "Gym Session", "Project Kickoff", "Client Call",
    "Birthday Party", "Coffee with Sara", "Code Review", "1:1 with Manager",
    "Yoga Class", "Workshop", "Team Outing", "Budget Meeting",
    "Piano Lesson", "Haircut", "Presentation Prep", "Study Group",
]

CALENDAR_CREATE_TEMPLATES = [
    "Create a meeting called '{title}' on {date}",
    "Schedule '{title}' on {date}",
    "Add '{title}' to my calendar on {date}",
    "Put '{title}' on {date}",
    "Set up '{title}' for {date}",
    "Book '{title}' on {date}",
]

CONVERT_UNITS = [
    ("miles", "kilometers", 1, 100), ("kilometers", "miles", 1, 200),
    ("pounds", "kilograms", 1, 500), ("kilograms", "pounds", 1, 200),
    ("fahrenheit", "celsius", 32, 212), ("celsius", "fahrenheit", -40, 50),
    ("gallons", "liters", 1, 50), ("liters", "gallons", 1, 100),
    ("inches", "centimeters", 1, 100), ("centimeters", "inches", 1, 300),
    ("feet", "meters", 1, 100), ("meters", "feet", 1, 200),
    ("ounces", "grams", 1, 100), ("grams", "ounces", 1, 500),
    ("hours", "minutes", 0.5, 24), ("cups", "ml", 1, 10),
]

CONVERT_TEMPLATES = [
    "Convert {value} {from_unit} to {to_unit}",
    "How many {to_unit} is {value} {from_unit}?",
    "What's {value} {from_unit} in {to_unit}?",
    "{value} {from_unit} to {to_unit}",
    "How much is {value} {from_unit} in {to_unit}?",
]

CURRENCIES = [
    "USD", "EUR", "GBP", "JPY", "CAD", "AUD", "CHF", "CNY",
    "INR", "PKR", "BRL", "MXN", "KRW", "SGD", "HKD", "THB",
    "AED", "SAR", "TRY", "ZAR", "EGP", "NGN", "BDT", "MYR",
]

CURRENCY_TEMPLATES = [
    "Convert {amount} {from_cur} to {to_cur}",
    "How much is {amount} {from_cur} in {to_cur}?",
    "Exchange {amount} {from_cur} to {to_cur}",
    "{amount} {from_cur} in {to_cur}",
    "What's {amount} {from_cur} worth in {to_cur}?",
]

SQL_TABLES = ["users", "orders", "products", "customers", "employees", "inventory", "sales"]
SQL_TEMPLATES = [
    ("Show me all {table}", "SELECT * FROM {table}"),
    ("Get all records from {table}", "SELECT * FROM {table}"),
    ("Count the rows in {table}", "SELECT COUNT(*) FROM {table}"),
    ("Get the top 5 from {table}", "SELECT * FROM {table} LIMIT 5"),
]

REFUSAL_PROMPTS = [
    "Tell me a joke", "What's the meaning of life?", "Send an email to my boss",
    "Book me a flight to Paris", "Order pizza for me", "Play some music",
    "Take a selfie", "Set an alarm for 7 AM", "Call my mom",
    "Navigate to the nearest gas station", "Translate this to French",
    "Write a poem about love", "Who won the World Cup in 2022?",
    "What's the capital of Australia?", "Help me write an essay",
    "Search for nearby restaurants", "Open the camera", "Turn off the lights",
    "What time is it?", "How are you today?", "What's your name?",
    "Can you read my messages?", "Who is the president?",
    "Open YouTube", "Who won the cricket match yesterday?",
    "Find me a good recipe", "Track my package", "Show me my photos",
    "Read my notifications", "Turn on Bluetooth",
]

REFUSAL_RESPONSES = [
    "I'm sorry, I can't help with that. I can assist you with weather, calendar, unit conversions, currency exchange, or database queries.",
    "That's outside my capabilities. I'm able to help with weather, calendar, unit conversion, currency exchange, and SQL queries.",
    "I don't have a tool for that. Let me know if you need help with weather, calendar, conversions, currency, or SQL.",
    "Sorry, that's not something I can do. My tools include weather, calendar, convert, currency, and sql.",
]

ADVERSARIAL_WEATHER = [
    ("Lahore mein aaj ka mausam kya hai?", "Lahore"),
    ("Mujhe Karachi ka weather batao", "Karachi"),
    ("Islamabad mein garmi hai ya sardi?", "Islamabad"),
    ("Peshawar ka mausam kaisa hai?", "Peshawar"),
    ("Delhi ka temperature kya hai?", "Delhi"),
    ("Mumbai mein barish ho rahi hai kya?", "Mumbai"),
    ("Londn weather plz", "London"),
    ("Pars ka mausam", "Paris"),
    ("Berln weathr", "Berlin"),
    ("Wats the weather in Tokyp?", "Tokyo"),
    ("Dubia weather", "Dubai"),
    ("Hows the wether in Singpore", "Singapore"),
    ("Istanbul hava durumu nasil?", "Istanbul"),
    ("cual es el clima en Madrid?", "Madrid"),
    ("como esta o tempo em Lisboa?", "Lisbon"),
    ("Seoul nalssi eottae?", "Seoul"),
    ("Bagdad ka mosam", "Baghdad"),
    ("Shnghai ka mausam bta", "Shanghai"),
]

ADVERSARIAL_CURRENCY = [
    ("Kitne rupees hain 100 dollars mein?", 100, "USD", "PKR"),
    ("50 dollar ko euro mein convert karo", 50, "USD", "EUR"),
    ("Convierte 75 dolares a pesos mexicanos", 75, "USD", "MXN"),
    ("Cuantos euros son 200 libras?", 200, "GBP", "EUR"),
    ("100 dolar kac lira?", 100, "USD", "TRY"),
    ("1000 yen ko dollar mein kro", 1000, "JPY", "USD"),
    ("200 pound kitne rupees?", 200, "GBP", "PKR"),
    ("50 yuro ko pkr mein convert kro", 50, "EUR", "PKR"),
]

ADVERSARIAL_CONVERT = [
    ("10 meel ko kilometer mein convert karo", 10, "miles", "kilometers"),
    ("5 gallon kitne liter hote hain?", 5, "gallons", "liters"),
    ("72 farenheit celsius mein kya hoga?", 72, "fahrenheit", "celsius"),
    ("Convierte 100 kilometros a millas", 100, "kilometers", "miles"),
    ("Convert 1 mile to meters plsss", 1, "mile", "meters"),
    ("How mny liters in 3 gallns?", 3, "gallons", "liters"),
    ("Cnvert 200 grams to ouncs", 200, "grams", "ounces"),
]

MULTI_TURN_SCENARIOS = [
    {
        "turns": [
            {"role": "user", "content": "What's the weather in London?"},
            {"role": "assistant", "content": '<tool_call>{"tool": "weather", "args": {"location": "London", "unit": "C"}}</tool_call>'},
            {"role": "user", "content": "What about Paris?"},
        ],
        "response": '<tool_call>{"tool": "weather", "args": {"location": "Paris", "unit": "C"}}</tool_call>',
    },
    {
        "turns": [
            {"role": "user", "content": "Convert 100 USD to EUR"},
            {"role": "assistant", "content": '<tool_call>{"tool": "currency", "args": {"amount": 100, "from": "USD", "to": "EUR"}}</tool_call>'},
            {"role": "user", "content": "Now convert 100 USD to GBP"},
        ],
        "response": '<tool_call>{"tool": "currency", "args": {"amount": 100, "from": "USD", "to": "GBP"}}</tool_call>',
    },
    {
        "turns": [
            {"role": "user", "content": "How much is 500 EUR in JPY?"},
            {"role": "assistant", "content": '<tool_call>{"tool": "currency", "args": {"amount": 500, "from": "EUR", "to": "JPY"}}</tool_call>'},
            {"role": "user", "content": "And in GBP?"},
        ],
        "response": '<tool_call>{"tool": "currency", "args": {"amount": 500, "from": "EUR", "to": "GBP"}}</tool_call>',
    },
    {
        "turns": [
            {"role": "user", "content": "Convert 10 miles to kilometers"},
            {"role": "assistant", "content": '<tool_call>{"tool": "convert", "args": {"value": 10, "from_unit": "miles", "to_unit": "kilometers"}}</tool_call>'},
            {"role": "user", "content": "Now convert 10 miles to meters"},
        ],
        "response": '<tool_call>{"tool": "convert", "args": {"value": 10, "from_unit": "miles", "to_unit": "meters"}}</tool_call>',
    },
    {
        "turns": [
            {"role": "user", "content": "What's on my calendar for 2025-04-01?"},
            {"role": "assistant", "content": '<tool_call>{"tool": "calendar", "args": {"action": "list", "date": "2025-04-01"}}</tool_call>'},
            {"role": "user", "content": "Create a meeting called Standup on that day"},
        ],
        "response": '<tool_call>{"tool": "calendar", "args": {"action": "create", "date": "2025-04-01", "title": "Standup"}}</tool_call>',
    },
    {
        "turns": [
            {"role": "user", "content": "Show me all users"},
            {"role": "assistant", "content": '<tool_call>{"tool": "sql", "args": {"query": "SELECT * FROM users"}}</tool_call>'},
            {"role": "user", "content": "Now filter by active status"},
        ],
        "response": "<tool_call>{\"tool\": \"sql\", \"args\": {\"query\": \"SELECT * FROM users WHERE status = 'active'\"}}</tool_call>",
    },
    {
        "turns": [
            {"role": "user", "content": "Convert 50 USD to EUR"},
            {"role": "assistant", "content": '<tool_call>{"tool": "currency", "args": {"amount": 50, "from": "USD", "to": "EUR"}}</tool_call>'},
            {"role": "user", "content": "Thanks! Can you also book me a hotel in Paris?"},
        ],
        "response": "I'm sorry, I don't have a hotel booking tool. I can help with weather, calendar, unit conversions, currency exchange, or database queries.",
    },
    {
        "turns": [
            {"role": "user", "content": "Convert that to euros"},
        ],
        "response": "I'm not sure what you'd like me to convert. Could you please specify the amount and the source currency?",
    },
    {
        "turns": [
            {"role": "user", "content": "Weather in Dubai"},
            {"role": "assistant", "content": '<tool_call>{"tool": "weather", "args": {"location": "Dubai", "unit": "C"}}</tool_call>'},
            {"role": "user", "content": "Convert 100 AED to USD"},
        ],
        "response": '<tool_call>{"tool": "currency", "args": {"amount": 100, "from": "AED", "to": "USD"}}</tool_call>',
    },
    {
        "turns": [
            {"role": "user", "content": "Show me Tokyo weather"},
            {"role": "assistant", "content": '<tool_call>{"tool": "weather", "args": {"location": "Tokyo", "unit": "C"}}</tool_call>'},
            {"role": "user", "content": "Now check Seoul"},
        ],
        "response": '<tool_call>{"tool": "weather", "args": {"location": "Seoul", "unit": "C"}}</tool_call>',
    },
    {
        "turns": [
            {"role": "user", "content": "How many grams is 5 ounces?"},
            {"role": "assistant", "content": '<tool_call>{"tool": "convert", "args": {"value": 5, "from_unit": "ounces", "to_unit": "grams"}}</tool_call>'},
            {"role": "user", "content": "What about kilograms?"},
        ],
        "response": '<tool_call>{"tool": "convert", "args": {"value": 5, "from_unit": "ounces", "to_unit": "kilograms"}}</tool_call>',
    },
    {
        "turns": [
            {"role": "user", "content": "Schedule Gym on 2025-05-10"},
            {"role": "assistant", "content": '<tool_call>{"tool": "calendar", "args": {"action": "create", "date": "2025-05-10", "title": "Gym"}}</tool_call>'},
            {"role": "user", "content": "Also add Yoga on the same date"},
        ],
        "response": '<tool_call>{"tool": "calendar", "args": {"action": "create", "date": "2025-05-10", "title": "Yoga"}}</tool_call>',
    },
]


def random_date():
    m = random.randint(1, 12)
    d = random.randint(1, 28)
    return f"2025-{m:02d}-{d:02d}"


def make_ex(msgs):
    return {"messages": [{"role": "system", "content": SYSTEM_PROMPT}] + msgs}


all_examples = []

# Weather (40)
for _ in range(40):
    city = random.choice(CITIES)
    if random.random() < 0.25:
        t = random.choice(WEATHER_TEMPLATES_F); u = "F"
    else:
        t = random.choice(WEATHER_TEMPLATES_C); u = "C"
    all_examples.append(make_ex([
        {"role": "user", "content": t.format(city=city)},
        {"role": "assistant", "content": f'<tool_call>{{"tool": "weather", "args": {{"location": "{city}", "unit": "{u}"}}}}</tool_call>'},
    ]))

# Calendar (35)
for _ in range(17):
    date = random_date()
    t = random.choice(CALENDAR_LIST_TEMPLATES)
    all_examples.append(make_ex([
        {"role": "user", "content": t.format(date=date)},
        {"role": "assistant", "content": f'<tool_call>{{"tool": "calendar", "args": {{"action": "list", "date": "{date}"}}}}</tool_call>'},
    ]))
for _ in range(18):
    date = random_date()
    title = random.choice(EVENT_TITLES)
    t = random.choice(CALENDAR_CREATE_TEMPLATES)
    all_examples.append(make_ex([
        {"role": "user", "content": t.format(title=title, date=date)},
        {"role": "assistant", "content": f'<tool_call>{{"tool": "calendar", "args": {{"action": "create", "date": "{date}", "title": "{title}"}}}}</tool_call>'},
    ]))

# Convert (35)
for _ in range(35):
    fu, tu, lo, hi = random.choice(CONVERT_UNITS)
    v = round(random.uniform(lo, hi), 1)
    if v == int(v): v = int(v)
    t = random.choice(CONVERT_TEMPLATES)
    all_examples.append(make_ex([
        {"role": "user", "content": t.format(value=v, from_unit=fu, to_unit=tu)},
        {"role": "assistant", "content": f'<tool_call>{{"tool": "convert", "args": {{"value": {v}, "from_unit": "{fu}", "to_unit": "{tu}"}}}}</tool_call>'},
    ]))

# Currency (35)
for _ in range(35):
    fc, tc = random.sample(CURRENCIES, 2)
    amt = random.choice([10, 25, 50, 100, 200, 500, 1000, 2500, 5000])
    t = random.choice(CURRENCY_TEMPLATES)
    all_examples.append(make_ex([
        {"role": "user", "content": t.format(amount=amt, from_cur=fc, to_cur=tc)},
        {"role": "assistant", "content": f'<tool_call>{{"tool": "currency", "args": {{"amount": {amt}, "from": "{fc}", "to": "{tc}"}}}}</tool_call>'},
    ]))

# SQL (25)
for _ in range(25):
    table = random.choice(SQL_TABLES)
    tp, tq = random.choice(SQL_TEMPLATES)
    all_examples.append(make_ex([
        {"role": "user", "content": tp.format(table=table)},
        {"role": "assistant", "content": f'<tool_call>{{"tool": "sql", "args": {{"query": "{tq.format(table=table)}"}}}}</tool_call>'},
    ]))

# Refusals (30)
for p in REFUSAL_PROMPTS:
    r = random.choice(REFUSAL_RESPONSES)
    all_examples.append(make_ex([
        {"role": "user", "content": p},
        {"role": "assistant", "content": r},
    ]))

# Adversarial weather (18)
for p, city in ADVERSARIAL_WEATHER:
    all_examples.append(make_ex([
        {"role": "user", "content": p},
        {"role": "assistant", "content": f'<tool_call>{{"tool": "weather", "args": {{"location": "{city}", "unit": "C"}}}}</tool_call>'},
    ]))

# Adversarial currency (8)
for p, amt, fc, tc in ADVERSARIAL_CURRENCY:
    all_examples.append(make_ex([
        {"role": "user", "content": p},
        {"role": "assistant", "content": f'<tool_call>{{"tool": "currency", "args": {{"amount": {amt}, "from": "{fc}", "to": "{tc}"}}}}</tool_call>'},
    ]))

# Adversarial convert (7)
for p, v, fu, tu in ADVERSARIAL_CONVERT:
    all_examples.append(make_ex([
        {"role": "user", "content": p},
        {"role": "assistant", "content": f'<tool_call>{{"tool": "convert", "args": {{"value": {v}, "from_unit": "{fu}", "to_unit": "{tu}"}}}}</tool_call>'},
    ]))

# Multi-turn (12)
for scenario in MULTI_TURN_SCENARIOS:
    msgs = list(scenario["turns"])
    msgs.append({"role": "assistant", "content": scenario["response"]})
    all_examples.append(make_ex(msgs))

random.shuffle(all_examples)

os.makedirs("data", exist_ok=True)
with open("data/training_data.jsonl", "w", encoding="utf-8") as f:
    for ex in all_examples:
        f.write(json.dumps(ex, ensure_ascii=False) + "\n")

print(f"Generated {len(all_examples)} training examples")

# ============================================================
# STEP 2: Fine-Tune with LoRA
# ============================================================
print("\n" + "=" * 60)
print("STEP 2: Fine-tuning with LoRA...")
print("=" * 60)

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, TaskType

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

print("Loading model with QLoRA...")
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
)
model.config.use_cache = False
model.gradient_checkpointing_enable()

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)

model = get_peft_model(model, lora_config)
trainable, total = model.get_nb_trainable_parameters()
print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

# Load data
examples = []
with open("data/training_data.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        examples.append(json.loads(line.strip()))

dataset = Dataset.from_list(examples)
dataset = dataset.map(
    lambda x: {"text": tokenizer.apply_chat_template(x["messages"], tokenize=False, add_generation_prompt=False)},
    remove_columns=dataset.column_names,
)
print(f"Dataset: {len(dataset)} examples")

# Tokenize
def tokenize_fn(example):
    tokens = tokenizer(example["text"], truncation=True, max_length=512, padding="max_length")
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

tokenized_dataset = dataset.map(tokenize_fn, remove_columns=["text"])
print(f"Tokenized: {len(tokenized_dataset)} examples")

os.makedirs("models/lora_adapter", exist_ok=True)

training_args = TrainingArguments(
    output_dir="models/lora_adapter",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    warmup_steps=10,
    logging_steps=10,
    save_steps=50,
    save_total_limit=2,
    fp16=True,
    optim="paged_adamw_8bit",
    lr_scheduler_type="cosine",
    report_to="none",
    seed=42,
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

print("Starting training...")
trainer.train()

trainer.save_model("models/lora_adapter")
tokenizer.save_pretrained("models/lora_adapter")
print("Training complete! Adapter saved.")

# ============================================================
# STEP 3: Merge LoRA + Convert to GGUF
# ============================================================
print("\n" + "=" * 60)
print("STEP 3: Merging LoRA and converting to GGUF...")
print("=" * 60)

del model, trainer
torch.cuda.empty_cache()

from peft import PeftModel

print("Loading base model for merge...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    trust_remote_code=True,
    device_map="cpu",
)

print("Merging LoRA adapter...")
model = PeftModel.from_pretrained(base_model, "models/lora_adapter")
model = model.merge_and_unload()

os.makedirs("models/merged_model", exist_ok=True)
model.save_pretrained("models/merged_model")
tokenizer.save_pretrained("models/merged_model")
print("Merged model saved!")

# Clone llama.cpp
print("Setting up llama.cpp...")
if not os.path.exists("llama.cpp"):
    subprocess.run(["git", "clone", "--depth=1", "https://github.com/ggerganov/llama.cpp"], check=True)

req_file = "llama.cpp/requirements.txt"
if os.path.exists(req_file):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-r", req_file])

# Convert to GGUF
print("Converting to GGUF...")
convert_script = "llama.cpp/convert_hf_to_gguf.py"

subprocess.run([
    sys.executable, convert_script, "models/merged_model",
    "--outfile", "models/pocket-agent-f16.gguf",
    "--outtype", "f16",
], check=True)

print("f16 GGUF created!")

# Try quantization
try:
    os.makedirs("llama.cpp/build", exist_ok=True)
    subprocess.run(["cmake", ".."], cwd="llama.cpp/build", check=True, capture_output=True)
    subprocess.run(["cmake", "--build", ".", "--target", "llama-quantize", "-j4"],
                   cwd="llama.cpp/build", check=True, capture_output=True)

    quantize_bin = "llama.cpp/build/bin/llama-quantize"
    subprocess.run([quantize_bin, "models/pocket-agent-f16.gguf",
                    "models/pocket-agent.gguf", "Q4_K_M"], check=True)
    os.remove("models/pocket-agent-f16.gguf")
    print("Q4_K_M quantization complete!")
except Exception as e:
    print(f"Native quantization failed ({e}), using f16 GGUF as fallback...")
    os.rename("models/pocket-agent-f16.gguf", "models/pocket-agent.gguf")

size_mb = os.path.getsize("models/pocket-agent.gguf") / (1024 * 1024)
print(f"\nFinal model size: {size_mb:.1f} MB")
if size_mb <= 500:
    print("PASSES the 500 MB gate!")
else:
    print("WARNING: Exceeds 500 MB gate. May need stronger quantization.")
if size_mb <= 250:
    print("BONUS: Qualifies for 250 MB bonus!")

# ============================================================
# STEP 4: Download
# ============================================================
print("\n" + "=" * 60)
print("STEP 4: Download your model!")
print("=" * 60)

try:
    from google.colab import files
    files.download("models/pocket-agent.gguf")
    print("Download started! Check your browser.")
except ImportError:
    print("Model saved at: models/pocket-agent.gguf")

print("\n" + "=" * 60)
print("ALL DONE!")
print("=" * 60)
print(f"Model: models/pocket-agent.gguf ({size_mb:.1f} MB)")
