import os
import requests
import pandas as pd
import time
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Config
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
INPUT_CSV = "bls_occupations_input.csv"

TEST_MODELS = [
    "deepseek/deepseek-chat-v3-0324",
    "google/gemini-2.5-pro"
]

TEMPERATURES = [0.7, 0.8, 0.9]  # Test different temps

PROMPT = """You're writing engaging job descriptions for Stacker's occupation listicle series. I'm giving you job titles and BLS descriptions, and your role is to write the "Slide copy" - engaging, AP-style descriptions that explain what these jobs actually involve. Use the BLS data as a guide but don't copy it. Make it informative and compelling for general readers. Keep it 70-110 words, one paragraph. Don't include salary numbers or statistics.

Here are examples of the transformation:

<EXAMPLE_1>
Job title: Orthopedic surgeons, except pediatric
BLS description: Diagnose and perform surgery to treat and prevent rheumatic and other diseases in the musculoskeletal system.
Slide copy: Orthopedic surgeons treat conditions involving the musculoskeletal system to improve their patients' quality of life. Some orthopedic oncologists remove life-threatening tumors. However, most care for patients with pain and mobility issues. To be recognized by the American Board of Orthopaedic Surgery, these surgeons must practice for 17 months and pass exams following medical school and a five-year residency.
</EXAMPLE_1>

<EXAMPLE_2>
Job title: Cardiologists
BLS description: Diagnose, treat, manage, and prevent diseases or conditions of the cardiovascular system. May further subspecialize in interventional procedures (e.g., balloon angioplasty and stent placement), echocardiography, or electrophysiology.
Slide copy: Cardiologists help prevent and treat heart and blood vessel diseases. Since the vocation plays a crucial role in patients' health (and mortality), this specialty requires at least 10 years of training plus passing an American Board of Internal Medicine exam and career-long continuing education.
</EXAMPLE_2>

<EXAMPLE_3>
Job title: Pediatric surgeons
BLS description: Diagnose and perform surgery to treat fetal abnormalities, diseases, injuries, and malformations in fetuses, premature and newborn infants, children, and adolescents.
Slide copy: Of the 50 highest-paying jobs in America, pediatric surgeons comprise one of only two careers that employ fewer than 1,200 people. They have one of the most arduous education and training paths of any physician and are responsible for detecting and treating abnormalities, diseases, and injuries in fetuses through adolescence.
</EXAMPLE_3>

<EXAMPLE_4>
Job title: Human Resources Managers
BLS description: Plan, direct, or coordinate human resources activities and staff of an organization.
Slide copy: Human resources managers oversee the essential acts of recruiting, interviewing, and hiring staff. They also mediate disputes and discipline workers. Aside from a bachelor's degree, many human resources managers attain certifications from the Society for Human Resource Management and other organizations.
</EXAMPLE_4>

Now transform this job:

Job title: {title}
BLS description: {description}
Slide copy:"""

# Load data
df = pd.read_csv(INPUT_CSV)
test_jobs = df.head(5)

results = {}

for model in TEST_MODELS:
    for temp in TEMPERATURES:
        key = f"{model}_temp{temp}"
        print(f"\nTesting {key}...")
        results[key] = []
        
        for i, row in test_jobs.iterrows():
            title = row['SOC Title']
            desc = row['From Occupation Data']
            
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": PROMPT.format(title=title, description=desc)}],
                "temperature": temp,
                "max_tokens": 200,
            }
            
            resp = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}"},
                json=payload
            )
            
            if resp.status_code == 200:
                output = resp.json()["choices"][0]["message"]["content"].strip()
                results[key].append({"job": title, "output": output})
                print(f"  ✓ {title[:25]}...")
            else:
                print(f"  ✗ Failed: {resp.status_code}")
                
            time.sleep(1)

# Save raw results
with open("two_models_test.json", "w") as f:
    json.dump(results, f, indent=2)

# Create comparison DataFrame
comparison_data = []

for i, row in test_jobs.iterrows():
    job_title = row['SOC Title']
    bls_desc = row['From Occupation Data']
    
    for key in results:
        if results[key] and len(results[key]) > i:
            output = results[key][i]['output']
            model_name = key.split('_temp')[0]
            temperature = float(key.split('_temp')[1])
            word_count = len(output.split())
            
            comparison_data.append({
                'job_number': i + 1,
                'job_title': job_title,
                'bls_description': bls_desc,
                'model': model_name,
                'temperature': temperature,
                'output': output,
                'word_count': word_count
            })

# Load existing CSV if it exists
try:
    existing_df = pd.read_csv("model_comparison.csv")
    print(f"Found existing CSV with {len(existing_df)} rows")
except FileNotFoundError:
    existing_df = pd.DataFrame()
    print("No existing CSV found, creating new one")

# Create new results DataFrame
new_comparison_df = pd.DataFrame(comparison_data)

# Append new results to existing data
if not existing_df.empty:
    combined_df = pd.concat([existing_df, new_comparison_df], ignore_index=True)
else:
    combined_df = new_comparison_df

# Save combined results
combined_df.to_csv("model_comparison.csv", index=False)

print(f"\nResults saved to:")
print(f"- Raw data: two_models_test.json")
print(f"- Comparison CSV: model_comparison.csv")
print(f"CSV now has {len(combined_df)} total rows (added {len(new_comparison_df)} new rows)")
print(f"Columns: {list(combined_df.columns)}")
