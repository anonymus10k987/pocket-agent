"""
Template-Based Synthetic Data Generator (No API Key Required)
=============================================================
Generates ~300 training examples using random templates.
This is the fallback generator if you don't have a Gemini/OpenAI API key.

Usage:
    python data/generate_templates.py
    
Output:
    data/training_data.jsonl
"""
import json
import random
import hashlib
import os

random.seed(42)

SYSTEM_PROMPT = (
    "You are a helpful mobile assistant. You have access to the following tools: "
    "weather, calendar, convert, currency, sql. "
    "When the user's request matches a tool, respond with a tool call in "
    "<tool_call>{...}</tool_call> tags. Otherwise, respond in plain text."
)

# ---------------------------------------------------------------------------
# Data pools
# ---------------------------------------------------------------------------
CITIES = [
    "Paris", "Berlin", "Madrid", "Rome", "Mumbai", "Islamabad", "Karachi",
    "Lahore", "Shanghai", "Beijing", "Seoul", "Bangkok", "Jakarta", "Cairo",
    "Nairobi", "Lagos", "Toronto", "Vancouver", "Mexico City", "São Paulo",
    "Buenos Aires", "Lima", "Bogotá", "Riyadh", "Doha", "Muscat", "Amman",
    "Beirut", "Ankara", "Athens", "Vienna", "Prague", "Warsaw", "Oslo",
    "Stockholm", "Helsinki", "Dublin", "Edinburgh", "Manchester", "Barcelona",
    "Lisbon", "Zurich", "Amsterdam", "Brussels", "Copenhagen", "Kuala Lumpur",
    "Singapore", "Ho Chi Minh City", "Manila", "Taipei", "Osaka", "Kyoto",
    "Melbourne", "Sydney", "Auckland", "Johannesburg", "Casablanca", "Tunis",
    "Addis Ababa", "Dhaka", "Colombo", "Kathmandu", "Hanoi", "Phnom Penh"
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
    "What's it like outside in {city}?",
    "How warm is it in {city}?",
]

WEATHER_TEMPLATES_F = [
    "What's the weather in {city} in Fahrenheit?",
    "Show me {city} temperature in Fahrenheit",
    "Tell me how hot it is in {city}, use Fahrenheit",
    "Weather in {city}, Fahrenheit please",
    "{city} weather in F",
]

CALENDAR_LIST_TEMPLATES = [
    "What events do I have on {date}?",
    "Show my calendar for {date}",
    "What's on my schedule for {date}?",
    "Any events on {date}?",
    "What's planned for {date}?",
    "List my events for {date}",
    "Do I have anything on {date}?",
    "Check my calendar on {date}",
]

EVENT_TITLES = [
    "Team Standup", "Sprint Review", "Dentist Appointment", "Lunch with Ali",
    "Doctor Visit", "Gym Session", "Project Kickoff", "Client Call",
    "Birthday Party", "Coffee with Sara", "Code Review", "1:1 with Manager",
    "Interview Prep", "Grocery Shopping", "Car Service", "Yoga Class",
    "Flight to Dubai", "Workshop", "Team Outing", "Budget Meeting",
    "Presentation Prep", "Study Group", "Piano Lesson", "Haircut",
]

CALENDAR_CREATE_TEMPLATES = [
    "Create a meeting called '{title}' on {date}",
    "Schedule '{title}' on {date}",
    "Add '{title}' to my calendar on {date}",
    "Put '{title}' on {date}",
    "Set up '{title}' for {date}",
    "Book '{title}' on {date}",
    "I need to add '{title}' on {date}",
    "Remind me about '{title}' on {date}",
]

CONVERT_UNITS = [
    ("miles", "kilometers", 1, 100),
    ("kilometers", "miles", 1, 200),
    ("pounds", "kilograms", 1, 500),
    ("kilograms", "pounds", 1, 200),
    ("fahrenheit", "celsius", 32, 212),
    ("celsius", "fahrenheit", -40, 50),
    ("gallons", "liters", 1, 50),
    ("liters", "gallons", 1, 100),
    ("inches", "centimeters", 1, 100),
    ("centimeters", "inches", 1, 300),
    ("feet", "meters", 1, 100),
    ("meters", "feet", 1, 200),
    ("ounces", "grams", 1, 100),
    ("grams", "ounces", 1, 500),
    ("hours", "minutes", 0.5, 24),
    ("minutes", "seconds", 1, 120),
    ("cups", "ml", 1, 10),
    ("ml", "cups", 100, 1000),
    ("yards", "meters", 1, 100),
    ("meters", "yards", 1, 200),
    ("tablespoons", "ml", 1, 20),
    ("teaspoons", "ml", 1, 30),
]

CONVERT_TEMPLATES = [
    "Convert {value} {from_unit} to {to_unit}",
    "How many {to_unit} is {value} {from_unit}?",
    "What's {value} {from_unit} in {to_unit}?",
    "{value} {from_unit} to {to_unit}",
    "How much is {value} {from_unit} in {to_unit}?",
    "I need to convert {value} {from_unit} to {to_unit}",
    "What does {value} {from_unit} equal in {to_unit}?",
]

CURRENCIES = [
    "USD", "EUR", "GBP", "JPY", "CAD", "AUD", "CHF", "CNY",
    "INR", "PKR", "BRL", "MXN", "KRW", "SGD", "HKD", "THB",
    "AED", "SAR", "TRY", "ZAR", "EGP", "NGN", "KES", "MAD",
    "QAR", "OMR", "BDT", "LKR", "NPR", "VND", "PHP", "MYR",
    "IDR", "TWD", "NZD", "SEK", "NOK", "DKK", "PLN", "CZK",
]

CURRENCY_TEMPLATES = [
    "Convert {amount} {from_cur} to {to_cur}",
    "How much is {amount} {from_cur} in {to_cur}?",
    "Exchange {amount} {from_cur} to {to_cur}",
    "{amount} {from_cur} in {to_cur}",
    "What's {amount} {from_cur} worth in {to_cur}?",
    "I want to convert {amount} {from_cur} to {to_cur}",
    "How many {to_cur} for {amount} {from_cur}?",
]

SQL_TABLES = ["users", "orders", "products", "customers", "employees", "inventory", "sales", "staff", "transactions", "payments"]
SQL_COLUMNS = {
    "users": ["id", "name", "email", "created_at", "status"],
    "orders": ["id", "customer_id", "total_amount", "order_date", "status"],
    "products": ["id", "name", "price", "category", "stock"],
    "customers": ["id", "name", "email", "phone", "city"],
    "employees": ["id", "name", "department", "salary", "hire_date"],
    "inventory": ["id", "product_id", "quantity", "warehouse", "last_updated"],
    "sales": ["id", "product_id", "amount", "sale_date", "region"],
    "staff": ["id", "name", "role", "department", "salary"],
    "transactions": ["id", "account_id", "amount", "type", "date"],
    "payments": ["id", "order_id", "amount", "method", "date"],
}

SQL_TEMPLATES = [
    ("Show me all {table}", "SELECT * FROM {table}"),
    ("Get all records from {table}", "SELECT * FROM {table}"),
    ("Count the rows in {table}", "SELECT COUNT(*) FROM {table}"),
    ("Show the top 5 {table} by {col}", "SELECT * FROM {table} ORDER BY {col} DESC LIMIT 5"),
    ("What's the average {col} in {table}?", "SELECT AVG({col}) FROM {table}"),
    ("Find {table} where {col} > 100", "SELECT * FROM {table} WHERE {col} > 100"),
    ("Get all {table} ordered by {col}", "SELECT * FROM {table} ORDER BY {col}"),
    ("Show distinct {col} from {table}", "SELECT DISTINCT {col} FROM {table}"),
    ("How many {table} have {col} > 50?", "SELECT COUNT(*) FROM {table} WHERE {col} > 50"),
    ("Delete all records from {table}", "DELETE FROM {table}"),
    ("List {col} from {table}", "SELECT {col} FROM {table}"),
]

REFUSAL_PROMPTS = [
    "Tell me a joke",
    "What's the meaning of life?",
    "Send an email to my boss",
    "Book me a flight to Paris",
    "Order pizza for me",
    "Play some music",
    "Take a selfie",
    "Set an alarm for 7 AM",
    "Call my mom",
    "Navigate to the nearest gas station",
    "Translate this to French",
    "Write a poem about love",
    "Summarize this article",
    "Who won the World Cup in 2022?",
    "What's the capital of Australia?",
    "Help me write an essay",
    "Search for nearby restaurants",
    "Open the camera",
    "Turn off the lights",
    "What time is it?",
    "How are you today?",
    "What's your name?",
    "Can you read my messages?",
    "Take a screenshot",
    "Who is the president?",
    "Remind me tomorrow",
    "Open YouTube",
    "Who won the cricket match yesterday?",
    "Find me a good recipe",
    "What song is this?",
    "Scan this QR code",
    "Track my package",
    "Show me my photos",
    "Read my notifications",
    "Turn on Bluetooth",
]

REFUSAL_RESPONSES = [
    "I'm sorry, I can't help with that. I can assist you with weather information, calendar management, unit conversions, currency exchange, or database queries.",
    "That's outside my capabilities. I'm able to help with weather, calendar, unit conversion, currency exchange, and SQL queries.",
    "I don't have a tool for that. I can help you with weather checks, calendar events, unit conversions, currency exchanges, or database queries.",
    "I'm unable to do that, but I'd be happy to help with weather info, calendar management, unit conversions, currency exchange, or SQL queries!",
    "Sorry, that's not something I can do. My tools include weather, calendar, convert, currency, and sql.",
]

# Adversarial / Code-Switched templates
ADVERSARIAL_WEATHER = [
    ("Lahore mein aaj ka mausam kya hai?", "Lahore"),
    ("Mujhe Karachi ka weather batao", "Karachi"),
    ("Islamabad mein garmi hai ya sardi?", "Islamabad"),
    ("Peshawar ka mausam kaisa hai?", "Peshawar"),
    ("Delhi ka temperature kya hai?", "Delhi"),
    ("Mumbai mein barish ho rahi hai kya?", "Mumbai"),
    ("दिल्ली का तापमान बताओ", "Delhi"),
    ("कोलकाता में मौसम कैसा है?", "Kolkata"),
    ("चेन्नई का वेदर क्या है?", "Chennai"),
    ("بیجنگ کا موسم کیسا ہے؟", "Beijing"),
    ("ٹوکیو میں آج کتنی گرمی ہے؟", "Tokyo"),
    ("Londn weather plz", "London"),
    ("Pars ka mausam", "Paris"),
    ("Berln weathr", "Berlin"),
    ("Wats the weather in Tokyp?", "Tokyo"),
    ("Dubia weather", "Dubai"),
    ("Hows the wether in Singpore", "Singapore"),
    ("quel temps fait-il à Montréal?", "Montreal"),
    ("como está o tempo em Lisboa?", "Lisbon"),
    ("Istanbul hava durumu nasıl?", "Istanbul"),
    ("Cuál es el clima en Madrid?", "Madrid"),
    ("Bagdad ka mosam", "Baghdad"),
    ("Shnghai ka mausam bta", "Shanghai"),
    ("Wie ist das Wetter in München?", "Munich"),
    ("서울 날씨 어때?", "Seoul"),
    ("Qual è il meteo a Roma?", "Rome"),
]

ADVERSARIAL_CURRENCY = [
    ("Kitne rupees hain 100 dollars mein?", 100, "USD", "PKR"),
    ("50 dollar ko euro mein convert karo", 50, "USD", "EUR"),
    ("100 रुपये कितने डॉलर हैं?", 100, "INR", "USD"),
    ("Convierte 75 dólares a pesos mexicanos", 75, "USD", "MXN"),
    ("٥٠٠ ريال كم دولار؟", 500, "SAR", "USD"),
    ("كم يساوي 1000 ريال سعودي بالدولار؟", 1000, "SAR", "USD"),
    ("Cuántos euros son 200 libras?", 200, "GBP", "EUR"),
    ("100 dolar kaç lira?", 100, "USD", "TRY"),
    ("1000 yen ko dollar mein kro", 1000, "JPY", "USD"),
    ("200 pound kitne rupees?", 200, "GBP", "PKR"),
    ("500 دلار چند روپے ہیں؟", 500, "USD", "PKR"),
    ("50 yuro ko pkr mein convert kro", 50, "EUR", "PKR"),
]

ADVERSARIAL_CONVERT = [
    ("10 meel ko kilometer mein convert karo", 10, "miles", "kilometers"),
    ("5 gallon kitne liter hote hain?", 5, "gallons", "liters"),
    ("72 farenheit celsius mein kya hoga?", 72, "fahrenheit", "celsius"),
    ("Convierte 100 kilómetros a millas", 100, "kilometers", "miles"),
    ("कितने किलोग्राम होते हैं 50 पाउंड में?", 50, "pounds", "kilograms"),
    ("30 سینٹی میٹر کتنے انچ ہیں؟", 30, "centimeters", "inches"),
    ("Convert 1 mile to meters plsss", 1, "mile", "meters"),
    ("How mny liters in 3 gallns?", 3, "gallons", "liters"),
    ("Cnvert 200 grams to ouncs", 200, "grams", "ounces"),
]

MULTI_TURN_SCENARIOS = [
    # Weather follow-up
    {
        "turns": [
            {"role": "user", "content": "What's the weather in London?"},
            {"role": "assistant", "content": "<tool_call>{\"tool\": \"weather\", \"args\": {\"location\": \"London\", \"unit\": \"C\"}}</tool_call>"},
            {"role": "user", "content": "What about Paris?"},
        ],
        "expected_response": "<tool_call>{\"tool\": \"weather\", \"args\": {\"location\": \"Paris\", \"unit\": \"C\"}}</tool_call>",
    },
    {
        "turns": [
            {"role": "user", "content": "Show me the weather in Tokyo"},
            {"role": "assistant", "content": "<tool_call>{\"tool\": \"weather\", \"args\": {\"location\": \"Tokyo\", \"unit\": \"C\"}}</tool_call>"},
            {"role": "user", "content": "Now check Seoul"},
        ],
        "expected_response": "<tool_call>{\"tool\": \"weather\", \"args\": {\"location\": \"Seoul\", \"unit\": \"C\"}}</tool_call>",
    },
    # Currency follow-up
    {
        "turns": [
            {"role": "user", "content": "Convert 100 USD to EUR"},
            {"role": "assistant", "content": "<tool_call>{\"tool\": \"currency\", \"args\": {\"amount\": 100, \"from\": \"USD\", \"to\": \"EUR\"}}</tool_call>"},
            {"role": "user", "content": "Now convert 100 USD to GBP"},
        ],
        "expected_response": "<tool_call>{\"tool\": \"currency\", \"args\": {\"amount\": 100, \"from\": \"USD\", \"to\": \"GBP\"}}</tool_call>",
    },
    {
        "turns": [
            {"role": "user", "content": "How much is 500 EUR in JPY?"},
            {"role": "assistant", "content": "<tool_call>{\"tool\": \"currency\", \"args\": {\"amount\": 500, \"from\": \"EUR\", \"to\": \"JPY\"}}</tool_call>"},
            {"role": "user", "content": "And in GBP?"},
        ],
        "expected_response": "<tool_call>{\"tool\": \"currency\", \"args\": {\"amount\": 500, \"from\": \"EUR\", \"to\": \"GBP\"}}</tool_call>",
    },
    # Convert follow-up
    {
        "turns": [
            {"role": "user", "content": "Convert 10 miles to kilometers"},
            {"role": "assistant", "content": "<tool_call>{\"tool\": \"convert\", \"args\": {\"value\": 10, \"from_unit\": \"miles\", \"to_unit\": \"kilometers\"}}</tool_call>"},
            {"role": "user", "content": "Now convert 10 miles to meters"},
        ],
        "expected_response": "<tool_call>{\"tool\": \"convert\", \"args\": {\"value\": 10, \"from_unit\": \"miles\", \"to_unit\": \"meters\"}}</tool_call>",
    },
    {
        "turns": [
            {"role": "user", "content": "How many grams is 5 ounces?"},
            {"role": "assistant", "content": "<tool_call>{\"tool\": \"convert\", \"args\": {\"value\": 5, \"from_unit\": \"ounces\", \"to_unit\": \"grams\"}}</tool_call>"},
            {"role": "user", "content": "What about kilograms?"},
        ],
        "expected_response": "<tool_call>{\"tool\": \"convert\", \"args\": {\"value\": 5, \"from_unit\": \"ounces\", \"to_unit\": \"kilograms\"}}</tool_call>",
    },
    # Calendar follow-up
    {
        "turns": [
            {"role": "user", "content": "What's on my calendar for 2025-04-01?"},
            {"role": "assistant", "content": "<tool_call>{\"tool\": \"calendar\", \"args\": {\"action\": \"list\", \"date\": \"2025-04-01\"}}</tool_call>"},
            {"role": "user", "content": "Create a meeting called 'Standup' on that day"},
        ],
        "expected_response": "<tool_call>{\"tool\": \"calendar\", \"args\": {\"action\": \"create\", \"date\": \"2025-04-01\", \"title\": \"Standup\"}}</tool_call>",
    },
    {
        "turns": [
            {"role": "user", "content": "Schedule 'Gym' on 2025-05-10"},
            {"role": "assistant", "content": "<tool_call>{\"tool\": \"calendar\", \"args\": {\"action\": \"create\", \"date\": \"2025-05-10\", \"title\": \"Gym\"}}</tool_call>"},
            {"role": "user", "content": "Also add 'Yoga' on the same date"},
        ],
        "expected_response": "<tool_call>{\"tool\": \"calendar\", \"args\": {\"action\": \"create\", \"date\": \"2025-05-10\", \"title\": \"Yoga\"}}</tool_call>",
    },
    # Cross-tool follow-up
    {
        "turns": [
            {"role": "user", "content": "What's the weather in Dubai?"},
            {"role": "assistant", "content": "<tool_call>{\"tool\": \"weather\", \"args\": {\"location\": \"Dubai\", \"unit\": \"C\"}}</tool_call>"},
            {"role": "user", "content": "Convert 100 AED to USD"},
        ],
        "expected_response": "<tool_call>{\"tool\": \"currency\", \"args\": {\"amount\": 100, \"from\": \"AED\", \"to\": \"USD\"}}</tool_call>",
    },
    # Refusal in multi-turn
    {
        "turns": [
            {"role": "user", "content": "Convert 50 USD to EUR"},
            {"role": "assistant", "content": "<tool_call>{\"tool\": \"currency\", \"args\": {\"amount\": 50, \"from\": \"USD\", \"to\": \"EUR\"}}</tool_call>"},
            {"role": "user", "content": "Thanks! Can you also book me a hotel in Paris?"},
        ],
        "expected_response": "I'm sorry, I don't have a hotel booking tool. I can help with weather, calendar, unit conversions, currency exchange, or database queries.",
    },
    # Ambiguous reference with no history → refusal
    {
        "turns": [
            {"role": "user", "content": "Convert that to euros"},
        ],
        "expected_response": "I'm not sure what you'd like me to convert. Could you please specify the amount and the source currency?",
    },
    # SQL follow-up
    {
        "turns": [
            {"role": "user", "content": "Show me all users"},
            {"role": "assistant", "content": "<tool_call>{\"tool\": \"sql\", \"args\": {\"query\": \"SELECT * FROM users\"}}</tool_call>"},
            {"role": "user", "content": "Now filter by active status"},
        ],
        "expected_response": "<tool_call>{\"tool\": \"sql\", \"args\": {\"query\": \"SELECT * FROM users WHERE status = 'active'\"}}</tool_call>",
    },
]


def random_date():
    """Generate a random date in 2025."""
    month = random.randint(1, 12)
    day = random.randint(1, 28)
    return f"2025-{month:02d}-{day:02d}"


def make_example(messages):
    """Create a training example."""
    return {"messages": [{"role": "system", "content": SYSTEM_PROMPT}] + messages}


def generate_weather_examples(n=40):
    examples = []
    for _ in range(n):
        city = random.choice(CITIES)
        if random.random() < 0.25:
            template = random.choice(WEATHER_TEMPLATES_F)
            unit = "F"
        else:
            template = random.choice(WEATHER_TEMPLATES_C)
            unit = "C"
        prompt = template.format(city=city)
        response = f'<tool_call>{{"tool": "weather", "args": {{"location": "{city}", "unit": "{unit}"}}}}</tool_call>'
        examples.append(make_example([
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]))
    return examples


def generate_calendar_examples(n=35):
    examples = []
    for _ in range(n // 2):
        date = random_date()
        template = random.choice(CALENDAR_LIST_TEMPLATES)
        prompt = template.format(date=date)
        response = f'<tool_call>{{"tool": "calendar", "args": {{"action": "list", "date": "{date}"}}}}</tool_call>'
        examples.append(make_example([
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]))
    for _ in range(n - n // 2):
        date = random_date()
        title = random.choice(EVENT_TITLES)
        template = random.choice(CALENDAR_CREATE_TEMPLATES)
        prompt = template.format(title=title, date=date)
        response = f'<tool_call>{{"tool": "calendar", "args": {{"action": "create", "date": "{date}", "title": "{title}"}}}}</tool_call>'
        examples.append(make_example([
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]))
    return examples


def generate_convert_examples(n=35):
    examples = []
    for _ in range(n):
        from_unit, to_unit, lo, hi = random.choice(CONVERT_UNITS)
        value = round(random.uniform(lo, hi), 1)
        if value == int(value):
            value = int(value)
        template = random.choice(CONVERT_TEMPLATES)
        prompt = template.format(value=value, from_unit=from_unit, to_unit=to_unit)
        response = f'<tool_call>{{"tool": "convert", "args": {{"value": {value}, "from_unit": "{from_unit}", "to_unit": "{to_unit}"}}}}</tool_call>'
        examples.append(make_example([
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]))
    return examples


def generate_currency_examples(n=35):
    examples = []
    for _ in range(n):
        from_cur, to_cur = random.sample(CURRENCIES, 2)
        amount = random.choice([10, 25, 50, 100, 150, 200, 250, 500, 750, 1000, 1500, 2000, 2500, 5000, 10000])
        template = random.choice(CURRENCY_TEMPLATES)
        prompt = template.format(amount=amount, from_cur=from_cur, to_cur=to_cur)
        response = f'<tool_call>{{"tool": "currency", "args": {{"amount": {amount}, "from": "{from_cur}", "to": "{to_cur}"}}}}</tool_call>'
        examples.append(make_example([
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]))
    return examples


def generate_sql_examples(n=30):
    examples = []
    for _ in range(n):
        table = random.choice(SQL_TABLES)
        cols = SQL_COLUMNS[table]
        template_prompt, template_query = random.choice(SQL_TEMPLATES)
        col = random.choice(cols)
        prompt = template_prompt.format(table=table, col=col)
        query = template_query.format(table=table, col=col)
        response = f'<tool_call>{{"tool": "sql", "args": {{"query": "{query}"}}}}</tool_call>'
        examples.append(make_example([
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]))
    return examples


def generate_refusal_examples(n=45):
    examples = []
    prompts_used = set()
    for prompt in REFUSAL_PROMPTS:
        if len(examples) >= n:
            break
        if prompt in prompts_used:
            continue
        prompts_used.add(prompt)
        resp = random.choice(REFUSAL_RESPONSES)
        examples.append(make_example([
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": resp},
        ]))
    return examples


def generate_adversarial_weather(n=26):
    examples = []
    for prompt, city in ADVERSARIAL_WEATHER[:n]:
        response = f'<tool_call>{{"tool": "weather", "args": {{"location": "{city}", "unit": "C"}}}}</tool_call>'
        examples.append(make_example([
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]))
    return examples


def generate_adversarial_currency(n=12):
    examples = []
    for prompt, amount, from_cur, to_cur in ADVERSARIAL_CURRENCY[:n]:
        response = f'<tool_call>{{"tool": "currency", "args": {{"amount": {amount}, "from": "{from_cur}", "to": "{to_cur}"}}}}</tool_call>'
        examples.append(make_example([
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]))
    return examples


def generate_adversarial_convert(n=9):
    examples = []
    for prompt, value, from_unit, to_unit in ADVERSARIAL_CONVERT[:n]:
        response = f'<tool_call>{{"tool": "convert", "args": {{"value": {value}, "from_unit": "{from_unit}", "to_unit": "{to_unit}"}}}}</tool_call>'
        examples.append(make_example([
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]))
    return examples


def generate_multi_turn_examples():
    examples = []
    for scenario in MULTI_TURN_SCENARIOS:
        messages = []
        for turn in scenario["turns"]:
            messages.append(turn)
        messages.append({"role": "assistant", "content": scenario["expected_response"]})
        examples.append(make_example(messages))
    return examples


def dedup_against_test(examples, test_path):
    """Remove any examples whose user prompt hash matches the test set."""
    if not os.path.exists(test_path):
        return examples
    
    test_hashes = set()
    with open(test_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            prompt = data.get("prompt", "")
            h = hashlib.sha256(prompt.strip().lower().encode()).hexdigest()
            test_hashes.add(h)
    
    filtered = []
    for ex in examples:
        user_msgs = [m["content"] for m in ex["messages"] if m["role"] == "user"]
        is_dup = False
        for msg in user_msgs:
            h = hashlib.sha256(msg.strip().lower().encode()).hexdigest()
            if h in test_hashes:
                is_dup = True
                break
        if not is_dup:
            filtered.append(ex)
    
    removed = len(examples) - len(filtered)
    if removed > 0:
        print(f"  Removed {removed} examples that matched the test set")
    return filtered


def main():
    print("Generating training data...")
    
    all_examples = []
    
    # Standard tool calls
    print("  Weather examples...")
    all_examples.extend(generate_weather_examples(40))
    print("  Calendar examples...")
    all_examples.extend(generate_calendar_examples(35))
    print("  Convert examples...")
    all_examples.extend(generate_convert_examples(35))
    print("  Currency examples...")
    all_examples.extend(generate_currency_examples(35))
    print("  SQL examples...")
    all_examples.extend(generate_sql_examples(30))
    
    # Refusals
    print("  Refusal examples...")
    all_examples.extend(generate_refusal_examples(45))
    
    # Adversarial
    print("  Adversarial weather...")
    all_examples.extend(generate_adversarial_weather())
    print("  Adversarial currency...")
    all_examples.extend(generate_adversarial_currency())
    print("  Adversarial convert...")
    all_examples.extend(generate_adversarial_convert())
    
    # Multi-turn
    print("  Multi-turn examples...")
    all_examples.extend(generate_multi_turn_examples())
    
    # Dedup
    test_path = os.path.join(os.path.dirname(__file__), "..", "starter", "public_test.jsonl")
    all_examples = dedup_against_test(all_examples, test_path)
    
    # Shuffle
    random.shuffle(all_examples)
    
    # Save
    output_dir = os.path.dirname(__file__)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "training_data.jsonl")
    
    with open(output_path, "w", encoding="utf-8") as f:
        for ex in all_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    
    print(f"\nGenerated {len(all_examples)} training examples → {output_path}")
    
    # Stats
    tool_counts = {}
    refusal_count = 0
    multi_turn_count = 0
    for ex in all_examples:
        msgs = ex["messages"]
        user_msgs = [m for m in msgs if m["role"] == "user"]
        if len(user_msgs) > 1:
            multi_turn_count += 1
            continue
        
        assistant_msgs = [m for m in msgs if m["role"] == "assistant"]
        if assistant_msgs:
            last = assistant_msgs[-1]["content"]
            if "<tool_call>" in last:
                try:
                    import re
                    match = re.search(r'"tool":\s*"(\w+)"', last)
                    if match:
                        tool = match.group(1)
                        tool_counts[tool] = tool_counts.get(tool, 0) + 1
                except:
                    pass
            else:
                refusal_count += 1
    
    print(f"\nBreakdown:")
    for tool, count in sorted(tool_counts.items()):
        print(f"  {tool}: {count}")
    print(f"  refusals: {refusal_count}")
    print(f"  multi-turn: {multi_turn_count}")


if __name__ == "__main__":
    main()
