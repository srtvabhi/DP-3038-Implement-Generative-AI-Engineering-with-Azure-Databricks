# Fine-Tuning Large Language Models using Azure Databricks and Microsoft Foundry (FULL CODE + EXPLANATION)

---

## 1️⃣ Create Volume in Databricks

```sql
%sql 
CREATE VOLUME databrick_ft.default.fine_tuning;
```

### Explanation
- Creates a Unity Catalog volume
- Storage path:
```
/Volumes/databrick_ft/default/fine_tuning/
```
- Used to store datasets

---

## 2️⃣ Download Training and Validation Data

```bash
%sh
wget -O /Volumes/databrick_ft/default/fine_tuning/training_set.jsonl https://github.com/MicrosoftLearning/mslearn-databricks/raw/main/data/training_set.jsonl
wget -O /Volumes/databrick_ft/default/fine_tuning/validation_set.jsonl https://github.com/MicrosoftLearning/mslearn-databricks/raw/main/data/validation_set.jsonl
```

### Explanation
- Downloads JSONL dataset files
- JSONL = one JSON object per line
- Each record contains conversation messages

Example:
```json
{
  "messages": [
    {"role": "user", "content": "What is Azure?"},
    {"role": "assistant", "content": "Azure is Microsoft cloud."}
  ]
}
```

---

## 3️⃣ Set Environment Variables

```python
import os

os.environ["AZURE_OPENAI_ENDPOINT"] = "https://xxxxxxxxxxxxxxxx.openai.azure.com/"
os.environ["COGNITIVE_SERVICES_TOKEN"] = "eyxxxxxxxxxxxxx"
os.environ["MANAGEMENT_TOKEN"] = "eyxxxxxxxxxxxx"
```

### Explanation
- AZURE_OPENAI_ENDPOINT → Azure OpenAI API endpoint
- COGNITIVE_SERVICES_TOKEN → used for model APIs
- MANAGEMENT_TOKEN → used for deployment APIs

---

## 4️⃣ Token Analysis Code

```python
import json
import tiktoken
import numpy as np
from collections import defaultdict

encoding = tiktoken.get_encoding("cl100k_base")

def num_tokens_from_messages(messages, tokens_per_message=3, tokens_per_name=1):
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3
    return num_tokens


def num_assistant_tokens_from_messages(messages):
    num_tokens = 0
    for message in messages:
        if message["role"] == "assistant":
            num_tokens += len(encoding.encode(message["content"]))
    return num_tokens


def print_distribution(values, name):
    print(f"
##### Distribution of {name}:")
    print(f"min / max: {min(values)}, {max(values)}")
    print(f"mean / median: {np.mean(values)}, {np.median(values)}")

files = [
    '/Volumes/databrick_ft/default/fine_tuning/training_set.jsonl',
    '/Volumes/databrick_ft/default/fine_tuning/validation_set.jsonl'
]

for file in files:
    print(f"File: {file}")
    with open(file, 'r', encoding='utf-8') as f:
        dataset = [json.loads(line) for line in f]

    total_tokens = []
    assistant_tokens = []

    for ex in dataset:
        messages = ex.get("messages", {})
        total_tokens.append(num_tokens_from_messages(messages))
        assistant_tokens.append(num_assistant_tokens_from_messages(messages))

    print_distribution(total_tokens, "total tokens")
    print_distribution(assistant_tokens, "assistant tokens")
    print('*' * 75)
```

### Explanation
- Counts tokens in each training example
- Helps estimate:
  - cost
  - token limits
- Outputs distribution stats

---

## 5️⃣ Upload Files to Azure OpenAI

```python
import os
from openai import AzureOpenAI

client = AzureOpenAI(
   azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"),
   azure_ad_token = os.getenv("COGNITIVE_SERVICES_TOKEN"),
   api_version = "2025-04-01-preview"
)

training_file_name = '/Volumes/databrick_ft/default/fine_tuning/training_set.jsonl'
validation_file_name = '/Volumes/databrick_ft/default/fine_tuning/validation_set.jsonl'

training_response = client.files.create(
     file = open(training_file_name, "rb"), purpose="fine-tune"
)
training_file_id = training_response.id

validation_response = client.files.create(
     file = open(validation_file_name, "rb"), purpose="fine-tune"
)
validation_file_id = validation_response.id

print("Training file ID:", training_file_id)
print("Validation file ID:", validation_file_id)
```

### Explanation
- Uploads dataset to Azure OpenAI
- Returns file IDs used for training

---

## 6️⃣ Create Fine-Tuning Job

```python
response = client.fine_tuning.jobs.create(
    training_file = training_file_id,
    validation_file = validation_file_id,
    model = "gpt-4.1-2025-04-14",
    seed = 105
)

job_id = response.id
```

### Explanation
- Starts fine-tuning process
- Uses GPT-4.1 base model
- seed ensures reproducibility

---

## 7️⃣ Check Job Status

```python
print("Job ID:", response.id)
print("Status:", response.status)
```

---

## 8️⃣ Retrieve Job Details

```python
response = client.fine_tuning.jobs.retrieve(job_id)

print(response.model_dump_json(indent=2))
fine_tuned_model = response.fine_tuned_model
```

### Explanation
- Gets job progress and results
- Extracts fine-tuned model ID

---

## 9️⃣ Deploy Model in Azure

```python
import json
import requests

token = os.getenv("MANAGEMENT_TOKEN")
subscription = "905702fe-7792-4e77-a763-7d77dd0322c9"
resource_group = "rg-ft-databrick"
resource_name = "lab-finetune-777"
model_deployment_name = "gpt-4.1-ft"

deploy_params = {'api-version': "2023-05-01"}
deploy_headers = {
    'Authorization': f'Bearer {token}',
    'Content-Type': 'application/json'
}

deploy_data = {
    "sku": {"name": "standard", "capacity": 1},
    "properties": {
        "model": {
            "format": "OpenAI",
            "name": fine_tuned_model
        }
    }
}

deploy_data = json.dumps(deploy_data)

request_url = f'https://management.azure.com/subscriptions/{subscription}/resourceGroups/{resource_group}/providers/Microsoft.CognitiveServices/accounts/{resource_name}/deployments/{model_deployment_name}'

print('Creating deployment...')

r = requests.put(request_url, params=deploy_params, headers=deploy_headers, data=deploy_data)

print(r)
print(r.reason)
print(r.json())
```

### Explanation
- Deploys fine-tuned model
- Makes it usable via endpoint

---

## 🔟 Use Fine-Tuned Model

```python
import os
from openai import AzureOpenAI

client = AzureOpenAI(
  azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"),
  azure_ad_token = os.getenv("COGNITIVE_SERVICES_TOKEN"),
  api_version = "2024-02-01"
)

response = client.chat.completions.create(
    model = "gpt-4.1-ft",
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Does Microsoft Foundry support customer managed keys?"},
        {"role": "assistant", "content": "Yes, customer managed keys are supported by Microsoft Foundry."},
        {"role": "user", "content": "Do other Azure AI services support this too?"}
    ]
)

print(response.choices[0].message.content)
```

### Explanation
- Calls deployed model
- Uses trained behavior
- Returns custom response

---

## ✅ FINAL FLOW
```
Create Volume
 → Download Data
 → Analyze Tokens
 → Upload Data
 → Fine-Tune Model
 → Retrieve Model
 → Deploy Model
 → Use Model
```

---

## ✅ KEY TAKEAWAYS
- Databricks → data handling
- Azure OpenAI → training
- Azure Foundry → deployment
- Token analysis → cost control
- Output → customized GPT model
