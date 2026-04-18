# ============================================================
# CELL 2 (SMALL MODEL): SmolLM2-360M — Improved Data + Train
# Run AFTER Cell 1 restarts.
# ============================================================

import json, random, os, subprocess, sys
random.seed(42)

# ============================================================
# STEP 1: Generate Training Data (expanded templates)
# ============================================================
print("=" * 60)
print("STEP 1: Generating training data...")
print("=" * 60)

SP = (
    "You are a helpful mobile assistant. You have access to the following tools: "
    "weather, calendar, convert, currency, sql. "
    "When the user's request matches a tool, respond with a tool call in "
    "<tool_call>{...}</tool_call> tags. Otherwise, respond in plain text."
)

def mk(msgs):
    return {"messages": [{"role": "system", "content": SP}] + msgs}

def rd():
    return f"2025-{random.randint(1,12):02d}-{random.randint(1,28):02d}"

all_ex = []

# === WEATHER (45 examples) ===
CITIES = ["Paris","Berlin","Madrid","Rome","Mumbai","Islamabad","Karachi","Lahore",
    "Shanghai","Beijing","Seoul","Bangkok","Jakarta","Cairo","Nairobi","Toronto",
    "Mexico City","Riyadh","Doha","Ankara","Vienna","Prague","Warsaw","Oslo",
    "Stockholm","Dublin","Barcelona","Amsterdam","Singapore","Manila","Taipei",
    "Melbourne","Sydney","Auckland","Johannesburg","London","New York","Tokyo",
    "Dubai","San Francisco","Chicago","Los Angeles","Moscow","Copenhagen","Osaka"]

WT = ["What's the weather in {c}?","Tell me the weather in {c}","How's the weather in {c}?",
    "Weather in {c} please","What is the temperature in {c}?","Check the weather for {c}",
    "Give me the forecast for {c}","How warm is it in {c}?","What's it like outside in {c}?",
    "Is it hot in {c} today?"]
WF = ["What's the weather in {c} in Fahrenheit?","{c} weather in F",
    "Show me {c} temperature in Fahrenheit"]

for _ in range(35):
    c = random.choice(CITIES)
    t = random.choice(WT); u = "C"
    all_ex.append(mk([{"role":"user","content":t.format(c=c)},
        {"role":"assistant","content":f'<tool_call>{{"tool":"weather","args":{{"location":"{c}","unit":"{u}"}}}}</tool_call>'}]))
for _ in range(10):
    c = random.choice(CITIES)
    t = random.choice(WF); u = "F"
    all_ex.append(mk([{"role":"user","content":t.format(c=c)},
        {"role":"assistant","content":f'<tool_call>{{"tool":"weather","args":{{"location":"{c}","unit":"{u}"}}}}</tool_call>'}]))

# === CALENDAR (30 examples) ===
CLT = ["What events do I have on {d}?","Show my calendar for {d}","Any events on {d}?",
    "What's planned for {d}?","List my events for {d}","What's on my schedule for {d}?"]
CCT = ["Create a meeting called '{t}' on {d}","Schedule '{t}' on {d}",
    "Add '{t}' to my calendar on {d}","Put '{t}' on {d}","Book '{t}' on {d}"]
TITLES = ["Team Standup","Sprint Review","Dentist Appointment","Lunch with Ali",
    "Doctor Visit","Gym Session","Project Kickoff","Client Call","Birthday Party",
    "Coffee with Sara","Code Review","1:1 with Manager","Yoga Class","Workshop"]

for _ in range(15):
    d = rd(); t = random.choice(CLT)
    all_ex.append(mk([{"role":"user","content":t.format(d=d)},
        {"role":"assistant","content":f'<tool_call>{{"tool":"calendar","args":{{"action":"list","date":"{d}"}}}}</tool_call>'}]))
for _ in range(15):
    d = rd(); ti = random.choice(TITLES); t = random.choice(CCT)
    all_ex.append(mk([{"role":"user","content":t.format(t=ti,d=d)},
        {"role":"assistant","content":f'<tool_call>{{"tool":"calendar","args":{{"action":"create","date":"{d}","title":"{ti}"}}}}</tool_call>'}]))

# === CONVERT (35 examples) ===
UNITS = [("miles","kilometers",1,100),("kilometers","miles",1,200),("pounds","kilograms",1,500),
    ("kilograms","pounds",1,200),("fahrenheit","celsius",32,212),("celsius","fahrenheit",-40,50),
    ("gallons","liters",1,50),("liters","gallons",1,100),("inches","centimeters",1,100),
    ("feet","meters",1,100),("ounces","grams",1,100),("grams","ounces",1,500)]
CT = ["Convert {v} {f} to {t}","How many {t} is {v} {f}?","What's {v} {f} in {t}?",
    "{v} {f} to {t}","How much is {v} {f} in {t}?"]

for _ in range(35):
    fu,tu,lo,hi = random.choice(UNITS); v = round(random.uniform(lo,hi),1)
    if v == int(v): v = int(v)
    t = random.choice(CT)
    all_ex.append(mk([{"role":"user","content":t.format(v=v,f=fu,t=tu)},
        {"role":"assistant","content":f'<tool_call>{{"tool":"convert","args":{{"value":{v},"from_unit":"{fu}","to_unit":"{tu}"}}}}</tool_call>'}]))

# === CURRENCY (35 examples) ===
CURS = ["USD","EUR","GBP","JPY","CAD","AUD","CHF","CNY","INR","PKR",
    "BRL","MXN","KRW","SGD","HKD","THB","AED","SAR","TRY","ZAR"]
CUT = ["Convert {a} {f} to {t}","How much is {a} {f} in {t}?","Exchange {a} {f} to {t}",
    "{a} {f} in {t}","What's {a} {f} worth in {t}?"]

for _ in range(35):
    fc,tc = random.sample(CURS,2); a = random.choice([10,25,50,100,200,500,1000,2500])
    t = random.choice(CUT)
    all_ex.append(mk([{"role":"user","content":t.format(a=a,f=fc,t=tc)},
        {"role":"assistant","content":f'<tool_call>{{"tool":"currency","args":{{"amount":{a},"from":"{fc}","to":"{tc}"}}}}</tool_call>'}]))

# === SQL (25 examples) ===
TBLS = ["users","orders","products","customers","employees","inventory","sales"]
SQT = [("Show me all {t}","SELECT * FROM {t}"),("Get all records from {t}","SELECT * FROM {t}"),
    ("Count the rows in {t}","SELECT COUNT(*) FROM {t}"),("Get the top 5 from {t}","SELECT * FROM {t} LIMIT 5"),
    ("List everything from {t}","SELECT * FROM {t}"),("How many {t} are there?","SELECT COUNT(*) FROM {t}"),
    ("Get {t} sorted by id","SELECT * FROM {t} ORDER BY id"),
    ("Show the first 10 {t}","SELECT * FROM {t} LIMIT 10")]

for _ in range(25):
    tb = random.choice(TBLS); tp,tq = random.choice(SQT)
    all_ex.append(mk([{"role":"user","content":tp.format(t=tb)},
        {"role":"assistant","content":f'<tool_call>{{"tool":"sql","args":{{"query":"{tq.format(t=tb)}"}}}}</tool_call>'}]))

# === REFUSALS (45 examples - CRITICAL for avoiding -0.5 penalties) ===
# Standard refusals
REFUSALS = [
    "Tell me a joke","What's the meaning of life?","Send an email to my boss",
    "Order pizza for me","Play some music","Take a selfie","Set an alarm for 7 AM",
    "Call my mom","Navigate to the nearest gas station","Translate this to French",
    "Write a poem about love","Who won the World Cup in 2022?","What's the capital of Australia?",
    "Help me write an essay","Search for nearby restaurants","Open the camera",
    "Turn off the lights","What time is it?","How are you today?","What's your name?",
    "Can you read my messages?","Who is the president?","Open YouTube",
    "Who won the cricket match yesterday?","Find me a good recipe",
    "Track my package","Show me my photos","Read my notifications","Turn on Bluetooth",
]
# TRICKY refusals that mention cities/numbers but should NOT trigger tools
TRICKY_REFUSALS = [
    "Book me a flight to Paris",
    "Book a flight to London",
    "Find me a hotel in Dubai",
    "Find a hotel in Tokyo",
    "Navigate to Tokyo",
    "Get me an Uber to Berlin",
    "Order food delivery in Mumbai",
    "Find restaurants near Rome",
    "Book a taxi to the airport in Istanbul",
    "Send 500 dollars to my mom",
    "Pay my rent of 1000 dollars",
    "Buy me a ticket to Seoul",
    "Reserve a table in London for tonight",
    "Find me a gym in New York",
    "Get directions to Chicago",
    "Call a cab to LAX",
]

REFUSAL_R = [
    "I'm sorry, I can't help with that. I can assist you with weather, calendar, unit conversions, currency exchange, or database queries.",
    "That's outside my capabilities. I'm able to help with weather, calendar, unit conversion, currency exchange, and SQL queries.",
    "I don't have a tool for that. Let me know if you need help with weather, calendar, conversions, currency, or SQL.",
    "Sorry, that's not something I can do. My tools include weather, calendar, convert, currency, and sql.",
]

for p in REFUSALS:
    all_ex.append(mk([{"role":"user","content":p},
        {"role":"assistant","content":random.choice(REFUSAL_R)}]))
for p in TRICKY_REFUSALS:
    all_ex.append(mk([{"role":"user","content":p},
        {"role":"assistant","content":random.choice(REFUSAL_R)}]))

# === ADVERSARIAL (30 examples - code-switched, typos) ===
ADV_W = [
    ("Lahore mein aaj ka mausam kya hai?","Lahore"),("Mujhe Karachi ka weather batao","Karachi"),
    ("Islamabad mein garmi hai ya sardi?","Islamabad"),("Peshawar ka mausam kaisa hai?","Peshawar"),
    ("Delhi ka temperature kya hai?","Delhi"),("Mumbai mein barish ho rahi hai kya?","Mumbai"),
    ("Londn weather plz","London"),("Pars ka mausam","Paris"),("Berln weathr","Berlin"),
    ("Wats the weather in Tokyp?","Tokyo"),("Dubia weather","Dubai"),
    ("Hows the wether in Singpore","Singapore"),("Istanbul hava durumu nasil?","Istanbul"),
    ("cual es el clima en Madrid?","Madrid"),("como esta o tempo em Lisboa?","Lisbon"),
    ("Seoul nalssi eottae?","Seoul"),("Bagdad ka mosam","Baghdad"),("Shnghai ka mausam bta","Shanghai"),
]
ADV_CUR = [
    ("Kitne rupees hain 100 dollars mein?",100,"USD","PKR"),
    ("50 dollar ko euro mein convert karo",50,"USD","EUR"),
    ("Convierte 75 dolares a pesos mexicanos",75,"USD","MXN"),
    ("Cuantos euros son 200 libras?",200,"GBP","EUR"),
    ("100 dolar kac lira?",100,"USD","TRY"),
    ("1000 yen ko dollar mein kro",1000,"JPY","USD"),
]
ADV_CON = [
    ("10 meel ko kilometer mein convert karo",10,"miles","kilometers"),
    ("5 gallon kitne liter hote hain?",5,"gallons","liters"),
    ("72 farenheit celsius mein kya hoga?",72,"fahrenheit","celsius"),
    ("Convierte 100 kilometros a millas",100,"kilometers","miles"),
]

for p,c in ADV_W:
    all_ex.append(mk([{"role":"user","content":p},
        {"role":"assistant","content":f'<tool_call>{{"tool":"weather","args":{{"location":"{c}","unit":"C"}}}}</tool_call>'}]))
for p,a,fc,tc in ADV_CUR:
    all_ex.append(mk([{"role":"user","content":p},
        {"role":"assistant","content":f'<tool_call>{{"tool":"currency","args":{{"amount":{a},"from":"{fc}","to":"{tc}"}}}}</tool_call>'}]))
for p,v,fu,tu in ADV_CON:
    all_ex.append(mk([{"role":"user","content":p},
        {"role":"assistant","content":f'<tool_call>{{"tool":"convert","args":{{"value":{v},"from_unit":"{fu}","to_unit":"{tu}"}}}}</tool_call>'}]))

# === MULTI-TURN (12 examples) ===
MT = [
    ([{"role":"user","content":"What's the weather in London?"},
      {"role":"assistant","content":'<tool_call>{"tool":"weather","args":{"location":"London","unit":"C"}}</tool_call>'},
      {"role":"user","content":"What about Paris?"}],
     '<tool_call>{"tool":"weather","args":{"location":"Paris","unit":"C"}}</tool_call>'),
    ([{"role":"user","content":"Convert 100 USD to EUR"},
      {"role":"assistant","content":'<tool_call>{"tool":"currency","args":{"amount":100,"from":"USD","to":"EUR"}}</tool_call>'},
      {"role":"user","content":"Now to GBP"}],
     '<tool_call>{"tool":"currency","args":{"amount":100,"from":"USD","to":"GBP"}}</tool_call>'),
    ([{"role":"user","content":"Convert 10 miles to km"},
      {"role":"assistant","content":'<tool_call>{"tool":"convert","args":{"value":10,"from_unit":"miles","to_unit":"kilometers"}}</tool_call>'},
      {"role":"user","content":"Now to meters"}],
     '<tool_call>{"tool":"convert","args":{"value":10,"from_unit":"miles","to_unit":"meters"}}</tool_call>'),
    ([{"role":"user","content":"Show my calendar for 2025-04-01"},
      {"role":"assistant","content":'<tool_call>{"tool":"calendar","args":{"action":"list","date":"2025-04-01"}}</tool_call>'},
      {"role":"user","content":"Add a standup on that day"}],
     '<tool_call>{"tool":"calendar","args":{"action":"create","date":"2025-04-01","title":"Standup"}}</tool_call>'),
    ([{"role":"user","content":"Weather in Dubai"},
      {"role":"assistant","content":'<tool_call>{"tool":"weather","args":{"location":"Dubai","unit":"C"}}</tool_call>'},
      {"role":"user","content":"Thanks! Book me a hotel there"}],
     "I'm sorry, I don't have a hotel booking tool. I can help with weather, calendar, unit conversions, currency exchange, or database queries."),
    ([{"role":"user","content":"Convert 50 USD to EUR"},
      {"role":"assistant","content":'<tool_call>{"tool":"currency","args":{"amount":50,"from":"USD","to":"EUR"}}</tool_call>'},
      {"role":"user","content":"Can you also order me dinner?"}],
     "Sorry, I can't help with ordering food. I can assist with weather, calendar, conversions, currency, or SQL."),
    ([{"role":"user","content":"Show me Tokyo weather"},
      {"role":"assistant","content":'<tool_call>{"tool":"weather","args":{"location":"Tokyo","unit":"C"}}</tool_call>'},
      {"role":"user","content":"Now check Seoul"}],
     '<tool_call>{"tool":"weather","args":{"location":"Seoul","unit":"C"}}</tool_call>'),
    ([{"role":"user","content":"Show me all users"},
      {"role":"assistant","content":'<tool_call>{"tool":"sql","args":{"query":"SELECT * FROM users"}}</tool_call>'},
      {"role":"user","content":"Filter by active ones"}],
     "<tool_call>{\"tool\":\"sql\",\"args\":{\"query\":\"SELECT * FROM users WHERE status = 'active'\"}}</tool_call>"),
    ([{"role":"user","content":"5 ounces to grams"},
      {"role":"assistant","content":'<tool_call>{"tool":"convert","args":{"value":5,"from_unit":"ounces","to_unit":"grams"}}</tool_call>'},
      {"role":"user","content":"What about kilograms?"}],
     '<tool_call>{"tool":"convert","args":{"value":5,"from_unit":"ounces","to_unit":"kilograms"}}</tool_call>'),
    ([{"role":"user","content":"Schedule Gym on 2025-05-10"},
      {"role":"assistant","content":'<tool_call>{"tool":"calendar","args":{"action":"create","date":"2025-05-10","title":"Gym"}}</tool_call>'},
      {"role":"user","content":"Also add Yoga same day"}],
     '<tool_call>{"tool":"calendar","args":{"action":"create","date":"2025-05-10","title":"Yoga"}}</tool_call>'),
    ([{"role":"user","content":"Convert that to euros"}],
     "I'm not sure what you'd like me to convert. Could you specify the amount and source currency?"),
    ([{"role":"user","content":"How much is 500 EUR in JPY?"},
      {"role":"assistant","content":'<tool_call>{"tool":"currency","args":{"amount":500,"from":"EUR","to":"JPY"}}</tool_call>'},
      {"role":"user","content":"And in GBP?"}],
     '<tool_call>{"tool":"currency","args":{"amount":500,"from":"EUR","to":"GBP"}}</tool_call>'),
]

for turns,resp in MT:
    msgs = list(turns) + [{"role":"assistant","content":resp}]
    all_ex.append(mk(msgs))

random.shuffle(all_ex)
os.makedirs("data", exist_ok=True)
with open("data/training_data.jsonl","w",encoding="utf-8") as f:
    for ex in all_ex:
        f.write(json.dumps(ex,ensure_ascii=False)+"\n")

print(f"Generated {len(all_ex)} training examples")
cats = {}
for ex in all_ex:
    a = [m["content"] for m in ex["messages"] if m["role"]=="assistant"][0]
    if "<tool_call>" in a:
        import re; m = re.search(r'"tool"\s*:\s*"(\w+)"', a)
        cats[m.group(1) if m else "?"]=cats.get(m.group(1) if m else "?",0)+1
    else:
        cats["refusal"]=cats.get("refusal",0)+1
for k,v in sorted(cats.items()): print(f"  {k}: {v}")

# ============================================================
# STEP 2: Fine-Tune SmolLM2-360M with LoRA
# ============================================================
print("\n" + "=" * 60)
print("STEP 2: Fine-tuning SmolLM2-360M-Instruct with LoRA...")
print("=" * 60)

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, TaskType

BASE_MODEL = "HuggingFaceTB/SmolLM2-360M-Instruct"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

print("Loading model in fp16 (small enough, no QLoRA needed)...")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, torch_dtype=torch.float16,
    device_map="auto", trust_remote_code=True,
)
model.config.use_cache = False
model.gradient_checkpointing_enable()

lora_config = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
)

model = get_peft_model(model, lora_config)
trainable, total = model.get_nb_trainable_parameters()
print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

examples = []
with open("data/training_data.jsonl","r",encoding="utf-8") as f:
    for line in f: examples.append(json.loads(line.strip()))

dataset = Dataset.from_list(examples)
dataset = dataset.map(
    lambda x: {"text": tokenizer.apply_chat_template(x["messages"], tokenize=False, add_generation_prompt=False)},
    remove_columns=dataset.column_names,
)
print(f"Dataset: {len(dataset)} examples")

def tokenize_fn(ex):
    t = tokenizer(ex["text"], truncation=True, max_length=512, padding="max_length")
    t["labels"] = t["input_ids"].copy()
    return t

tokenized = dataset.map(tokenize_fn, remove_columns=["text"])

os.makedirs("models/lora_adapter_small", exist_ok=True)
args = TrainingArguments(
    output_dir="models/lora_adapter_small", num_train_epochs=5,
    per_device_train_batch_size=4, gradient_accumulation_steps=4,
    learning_rate=3e-4, warmup_steps=10, logging_steps=10,
    save_steps=50, save_total_limit=2, fp16=True,
    optim="adamw_torch", lr_scheduler_type="cosine",
    report_to="none", seed=42,
)

trainer = Trainer(
    model=model, args=args, train_dataset=tokenized,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
)

print("Starting training...")
trainer.train()
trainer.save_model("models/lora_adapter_small")
tokenizer.save_pretrained("models/lora_adapter_small")
print("Training complete!")

# ============================================================
# STEP 3: Merge LoRA + Convert to GGUF
# ============================================================
print("\n" + "=" * 60)
print("STEP 3: Merging LoRA and converting to GGUF...")
print("=" * 60)

del model, trainer
torch.cuda.empty_cache()

from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, torch_dtype=torch.float16, trust_remote_code=True, device_map="cpu",
)
model = PeftModel.from_pretrained(base_model, "models/lora_adapter_small")
model = model.merge_and_unload()

os.makedirs("models/merged_model_small", exist_ok=True)
model.save_pretrained("models/merged_model_small")
tokenizer.save_pretrained("models/merged_model_small")
print("Merged model saved!")

if not os.path.exists("llama.cpp"):
    subprocess.run(["git","clone","--depth=1","https://github.com/ggerganov/llama.cpp"], check=True)

req_file = "llama.cpp/requirements.txt"
if os.path.exists(req_file):
    subprocess.check_call([sys.executable,"-m","pip","install","-q","-r",req_file])

print("Converting to GGUF...")
res = subprocess.run([sys.executable,"llama.cpp/convert_hf_to_gguf.py","models/merged_model_small",
    "--outfile","models/pocket-agent-small-f16.gguf","--outtype","f16"], capture_output=True, text=True)

if res.returncode != 0:
    print("🚨 CONVERSION FAILED! 🚨")
    print("STDOUT:\n", res.stdout)
    print("STDERR:\n", res.stderr)
    raise RuntimeError("GGUF Conversion failed")


try:
    os.makedirs("llama.cpp/build", exist_ok=True)
    subprocess.run(["cmake",".."], cwd="llama.cpp/build", check=True, capture_output=True)
    subprocess.run(["cmake","--build",".","--target","llama-quantize","-j4"],
                   cwd="llama.cpp/build", check=True, capture_output=True)
    subprocess.run(["llama.cpp/build/bin/llama-quantize","models/pocket-agent-small-f16.gguf",
                    "models/pocket-agent-small.gguf","Q4_K_M"], check=True)
    os.remove("models/pocket-agent-small-f16.gguf")
    print("Q4_K_M quantization complete!")
except Exception as e:
    print(f"Quantization failed ({e}), using f16...")
    os.rename("models/pocket-agent-small-f16.gguf","models/pocket-agent-small.gguf")

size_mb = os.path.getsize("models/pocket-agent-small.gguf") / (1024*1024)
print(f"\nModel size: {size_mb:.1f} MB")
print(f"{'PASSES' if size_mb<=500 else 'FAILS'} 500MB gate")
if size_mb<=250: print("BONUS: <=250MB achieved!")

try:
    from google.colab import files
    files.download("models/pocket-agent-small.gguf")
except: print("Model at: models/pocket-agent-small.gguf")
print("DONE!")
