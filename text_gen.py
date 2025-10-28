import os
import time
import math
import requests
import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --------------------------
# Config
# --------------------------
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL = "anthropic/claude-sonnet-4"
INPUT_PATH = "bls_occupations_input.csv"  # Input data with SOC titles and BLS descriptions
OUTPUT_PATH = "bls_with_descriptions.csv"
OUTPUT_PARTIAL = "bls_with_descriptions.partial.csv"

INPUT_SOC_COL = "SOC Title"
INPUT_DESC_COL = "From Occupation Data"
OUTPUT_COL = "Stacker Description"  # new column to add
BATCH_SIZE = 5  # Process multiple jobs per API call to save costs
MAX_RETRIES = 3

# --------------------------
# Load data
# --------------------------
# Try to resume from partial file first
try:
    print("Loading data from partial file...")
    df = pd.read_csv(OUTPUT_PARTIAL)
    print(f"Resumed from partial file with {len(df)} rows")
except FileNotFoundError:
    print(f"Loading data from {INPUT_PATH}...")
    df = pd.read_csv(INPUT_PATH)
    print(f"Loaded {len(df)} rows from input file")

# Create output column if missing
if OUTPUT_COL not in df.columns:
    df[OUTPUT_COL] = ""

# Keep rows that still need generation
mask_needs = df[OUTPUT_COL].isna() | (df[OUTPUT_COL].astype(str).str.strip() == "")
work_idx = df.index[mask_needs].tolist()
if not work_idx:
    print("No rows to process.")
    df.to_csv(OUTPUT_PATH, index=False)
    raise SystemExit

# --------------------------
# Prompts
# --------------------------
SYSTEM_PROMPT = (
    "You're writing engaging job descriptions for Stacker's occupation listicle series. I'm giving you job titles and BLS descriptions, and your role is to write the \"Slide copy\" - engaging, AP-style descriptions that explain what these jobs actually involve. Use the BLS data as a guide but don't copy it. Make it informative and compelling for general readers. Keep it 50-80 words, one paragraph. Don't include salary numbers or statistics.\n\n"
    "Here are examples of the transformation:\n\n"
    "<EXAMPLE_1>\n"
    "Job title: Orthopedic surgeons, except pediatric\n"
    "BLS description: Diagnose and perform surgery to treat and prevent rheumatic and other diseases in the musculoskeletal system.\n"
    "Slide copy: Orthopedic surgeons treat conditions involving the musculoskeletal system to improve their patients' quality of life. Some orthopedic oncologists remove life-threatening tumors. However, most care for patients with pain and mobility issues. To be recognized by the American Board of Orthopaedic Surgery, these surgeons must practice for 17 months and pass exams following medical school and a five-year residency.\n"
    "</EXAMPLE_1>\n\n"
    "<EXAMPLE_2>\n"
    "Job title: Cardiologists\n"
    "BLS description: Diagnose, treat, manage, and prevent diseases or conditions of the cardiovascular system. May further subspecialize in interventional procedures (e.g., balloon angioplasty and stent placement), echocardiography, or electrophysiology.\n"
    "Slide copy: Cardiologists help prevent and treat heart and blood vessel diseases. Since the vocation plays a crucial role in patients' health (and mortality), this specialty requires at least 10 years of training plus passing an American Board of Internal Medicine exam and career-long continuing education.\n"
    "</EXAMPLE_2>\n\n"
    "<EXAMPLE_3>\n"
    "Job title: Pediatric surgeons\n"
    "BLS description: Diagnose and perform surgery to treat fetal abnormalities, diseases, injuries, and malformations in fetuses, premature and newborn infants, children, and adolescents.\n"
    "Slide copy: Of the 50 highest-paying jobs in America, pediatric surgeons comprise one of only two careers that employ fewer than 1,200 people. They have one of the most arduous education and training paths of any physician and are responsible for detecting and treating abnormalities, diseases, and injuries in fetuses through adolescence.\n"
    "</EXAMPLE_3>\n\n"
    "<EXAMPLE_4>\n"
    "Job title: Human Resources Managers\n"
    "BLS description: Plan, direct, or coordinate human resources activities and staff of an organization.\n"
    "Slide copy: Human resources managers oversee the essential acts of recruiting, interviewing, and hiring staff. They also mediate disputes and discipline workers. Aside from a bachelor's degree, many human resources managers attain certifications from the Society for Human Resource Management and other organizations.\n"
    "</EXAMPLE_4>"
)

def build_user_prompt(row_indices):
    """Build prompt for multiple jobs"""
    jobs_text = []
    for i, row_idx in enumerate(row_indices, 1):
        soc = "" if pd.isna(df.at[row_idx, INPUT_SOC_COL]) else str(df.at[row_idx, INPUT_SOC_COL])
        bls = "" if pd.isna(df.at[row_idx, INPUT_DESC_COL]) else str(df.at[row_idx, INPUT_DESC_COL])

        jobs_text.append(f"""{i}. Job title: {soc}
BLS description: {bls}
Slide copy:""")

    jobs_section = "\n\n".join(jobs_text)

    return f"""Now transform these {len(row_indices)} jobs:

{jobs_section}

For each job, write the slide copy following the examples in the system prompt. Number your responses 1, 2, 3, etc. to match the job numbers above.

Output only the numbered slide copy descriptions, nothing else."""

# --------------------------
# OpenRouter client
# --------------------------
def call_openrouter(messages, retries=MAX_RETRIES):
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        # Optional provenance headers
        "Referer": "https://example.org",
        "X-Title": "BLS Description Generator",
    }
    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": 0.7,  # Balanced creativity vs consistency for professional writing
        "max_tokens": 8000,  # Increased for batch processing
    }
    for attempt in range(1, retries + 1):
        resp = requests.post(url, headers=headers, json=payload, timeout=300)
        if resp.status_code == 200:
            try:
                data = resp.json()
                return data["choices"][0]["message"]["content"]
            except Exception as e:
                if attempt == retries:
                    raise
        else:
            if attempt == retries:
                raise RuntimeError(f"OpenRouter error {resp.status_code}: {resp.text}")
        time.sleep(1.25 * attempt)  # simple backoff

def parse_numbered_descriptions(text, expected_count):
    """Parse numbered descriptions from model response"""
    lines = text.strip().split('\n')
    descriptions = []

    current_desc = []
    current_number = None

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Check if line starts with a number
        if line[0].isdigit() and '.' in line[:3]:
            # Save previous description if exists
            if current_desc and current_number is not None:
                descriptions.append((current_number, ' '.join(current_desc)))

            # Start new description
            try:
                current_number = int(line.split('.')[0])
                current_desc = [line.split('.', 1)[1].strip()]
            except (ValueError, IndexError):
                current_desc.append(line)
        else:
            # Continue current description
            current_desc.append(line)

    # Don't forget the last description
    if current_desc and current_number is not None:
        descriptions.append((current_number, ' '.join(current_desc)))

    # Validate we got expected number of descriptions
    if len(descriptions) != expected_count:
        print(f"Warning: Expected {expected_count} descriptions, got {len(descriptions)}")

    return descriptions

# --------------------------
# Process in batches
# --------------------------
total = len(work_idx)
num_batches = math.ceil(total / BATCH_SIZE)
print(f"Processing {total} jobs in {num_batches} batches of {BATCH_SIZE}...")

for b in range(num_batches):
    batch_rows = work_idx[b*BATCH_SIZE:(b+1)*BATCH_SIZE]
    if not batch_rows:
        break

    batch_num = b + 1
    print(f"Processing batch {batch_num}/{num_batches} (jobs {b*BATCH_SIZE+1}-{min((b+1)*BATCH_SIZE, total)})")

    user_prompt = build_user_prompt(batch_rows)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    text = call_openrouter(messages)
    descriptions = parse_numbered_descriptions(text, len(batch_rows))

    # Apply results
    for number, description in descriptions:
        if 1 <= number <= len(batch_rows):
            row_idx = batch_rows[number - 1]  # Convert to 0-based index
            if description.strip():
                df.at[row_idx, OUTPUT_COL] = description.strip()

    # Persist after each batch
    df.to_csv(OUTPUT_PARTIAL, index=False)

# Final save
df.to_csv(OUTPUT_PATH, index=False)
print(f"Done. Saved to {OUTPUT_PATH}")
