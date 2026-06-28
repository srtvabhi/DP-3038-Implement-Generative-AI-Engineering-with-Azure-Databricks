# Evaluate Large Language Models using Azure Databricks and Azure OpenAI 

This notebook demonstrates how to generate responses using **Azure OpenAI**, evaluate them using **MLflow GenAI**, and automatically track prompts, outputs, and evaluation metrics.

---

# 1. Environment Setup

## ✅ Code

```python
%pip install -U mlflow[databricks]>=3.1.0 openai databricks-connect==16.1

dbutils.library.restartPython()
```

## ✅ Explanation

The following libraries are installed inside the Databricks notebook:

- `%pip install` – Installs required Python packages.
- `mlflow[databricks]` – Enables MLflow GenAI evaluation capabilities.
- `openai` – SDK for calling Azure OpenAI models.
- `databricks-connect` – Connects your notebook with the Databricks runtime.

### Why `restartPython()` is important?

- Applies newly installed dependencies.
- Prevents version conflicts.
- Ensures the notebook uses the latest installed packages.

### 🎯 DP-3028 Concept

**Environment setup for Generative AI workloads in Databricks.**

---

# 2. Import Libraries

## ✅ Code

```python
import os
import json
import mlflow

from openai import AzureOpenAI
from mlflow.genai import evaluate
from mlflow.genai.scorers import Guidelines, Safety
```

## ✅ Explanation

These libraries provide the core functionality for the notebook:

| Library | Purpose |
|----------|---------|
| `AzureOpenAI` | Connects to Azure OpenAI models |
| `mlflow.genai.evaluate` | Executes LLM evaluation |
| `Guidelines` | Defines custom quality rules |
| `Safety` | Performs Responsible AI safety evaluation |

### 🎯 DP-3028 Concept

**Using the MLflow GenAI SDK for evaluation pipelines.**

---

# 3. Configure Azure OpenAI

## ✅ Code

```python
os.environ["AZURE_OPENAI_API_KEY"] = "<your-key>"
os.environ["AZURE_OPENAI_ENDPOINT"] = "<your-endpoint>"
os.environ["AZURE_OPENAI_API_VERSION"] = "2024-12-01-preview"
```

## ✅ Explanation

Configure Azure OpenAI using environment variables.

| Variable | Purpose |
|----------|---------|
| `AZURE_OPENAI_API_KEY` | Authentication |
| `AZURE_OPENAI_ENDPOINT` | Azure OpenAI endpoint URL |
| `AZURE_OPENAI_API_VERSION` | Specifies the API version |

These settings are required before making requests to Azure-hosted GPT models.

### 🎯 DP-3028 Concept

**Secure configuration of Azure AI services.**

---

# 4. Create Azure OpenAI Client

## ✅ Code

```python
client = AzureOpenAI(
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"]
)
```

## ✅ Explanation

Creates an Azure OpenAI client object.

All LLM requests are sent through this client.

**Think of it as:**

```
Databricks Notebook
        │
        ▼
 AzureOpenAI Client
        │
        ▼
 Azure OpenAI GPT Model
```

---

# 5. Enable MLflow Tracking

## ✅ Code

```python
mlflow.autolog()
```

## ✅ Explanation

MLflow automatically records:

- Prompts
- Responses
- Evaluation scores
- Metadata

No manual logging is required.

### 🎯 DP-3028 Concept

**Observability for LLM applications.**

---

# 6. Define the Prompt Template

## ✅ Code

```python
SYSTEM_PROMPT = """
You are a creative assistant generating funny, absurd outputs for children (age 1–6).
Avoid any normal/realistic responses.
"""
```

## ✅ Explanation

The system prompt controls how the model behaves.

It instructs the model to:

- Generate creative responses.
- Produce child-safe outputs.
- Avoid realistic answers.

The prompt acts as the **core logic** of the GenAI application.

### 🎯 DP-3028 Concept

**Prompt Engineering.**

---

# 7. Generate Responses

## ✅ Code

```python
def generate_response(user_prompt):
    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.9
    )

    return response.choices[0].message.content
```

## ✅ Explanation

This function sends a request to Azure OpenAI.

### Inputs

- System Prompt
- User Prompt

### Model Configuration

```text
Model: gpt-4.1
Temperature: 0.9
```

A higher temperature increases creativity and randomness.

### Output

Returns generated text from the LLM.

### 🎯 DP-3028 Concept

**LLM inference using Azure OpenAI APIs.**

---

# 8. Create the Evaluation Dataset

## ✅ Code

```python
inputs = [
    "rainbow spaghetti",
    "giggling ice cream",
    "bubble wrap popper",
    "walking backwards on the moon"
]
```

## ✅ Explanation

These prompts act as evaluation test cases.

They help evaluate:

- Creativity
- Unusual scenarios
- Edge cases

### 🎯 DP-3028 Concept

**Evaluation dataset design for LLMs.**

---

# 9. Generate Model Outputs

## ✅ Code

```python
outputs = []

for inp in inputs:
    result = generate_response(inp)

    outputs.append({
        "input": inp,
        "output": result
    })
```

## ✅ Explanation

The notebook:

1. Iterates through each prompt.
2. Generates an LLM response.
3. Stores the result.

Example output:

```json
{
  "input": "rainbow spaghetti",
  "output": "Twinkly noodles dancing in the sky!"
}
```

---

# 10. Define Evaluation Criteria

## ✅ Custom Guidelines

```python
guidelines = Guidelines(
    name="CreativityCheck",
    guidelines="""
    - Response should be creative and funny
    - Should not be realistic
    - Must follow child-friendly tone
    """
)
```

## ✅ Safety Scorer

```python
safety = Safety()
```

## ✅ Explanation

### Guidelines Scorer

Checks whether responses:

- Are creative
- Are humorous
- Follow the required style
- Remain child-friendly

### Safety Scorer

Ensures responses contain:

- No harmful content
- No unsafe language
- Child-safe outputs

### 🎯 DP-3028 Concept

**Custom and built-in evaluation metrics for LLMs.**

---

# 11. Run MLflow Evaluation

## ✅ Code

```python
evaluation_results = evaluate(
    data=outputs,
    scorers=[guidelines, safety]
)
```

## ✅ Explanation

This is the main evaluation step.

### Inputs

- Generated outputs
- Evaluation scorers

### MLflow Produces

- Creativity score
- Safety score
- Guideline compliance

### 🎯 DP-3028 Concept

**Automated LLM evaluation framework.**

---

# 12. View Evaluation Results

## ✅ Code

```python
print(json.dumps(evaluation_results, indent=2))
```

## ✅ Explanation

Displays evaluation results.

Example:

```json
{
  "input": "bubble wrap popper",
  "output": "...",
  "CreativityCheck": 0.9,
  "Safety": 1.0
}
```

These results help you:

- Compare model responses.
- Identify weak outputs.
- Improve prompt quality.

---

# 13. MLflow Tracking (Behind the Scenes)

## ✅ What MLflow Logs Automatically

- Prompt
- Output
- Evaluation scores
- Model information
- Metadata

## ✅ Why It Matters

MLflow enables you to:

- Compare multiple experiment runs.
- Monitor model quality.
- Debug prompt issues.
- Track improvements over time.

### 🎯 DP-3028 Concept

**Experiment tracking for Generative AI systems.**

---

# Complete Workflow Summary

```text
Install Required Libraries
            │
            ▼
Configure Azure OpenAI
            │
            ▼
Create System Prompt
            │
            ▼
Generate LLM Responses
            │
            ▼
Define Evaluation Rules
            │
            ▼
Evaluate Responses with MLflow
            │
            ▼
Track Results Automatically
```

---

# Key DP-3028 Concepts Covered

| Step | DP-3028 Concept |
|------|------------------|
| Environment Setup | Databricks GenAI Environment |
| Import Libraries | MLflow GenAI SDK |
| Azure Configuration | Secure Azure AI Configuration |
| OpenAI Client | Azure OpenAI Integration |
| MLflow Autolog | LLM Observability |
| Prompt Template | Prompt Engineering |
| Generate Responses | Azure OpenAI Inference |
| Test Dataset | LLM Evaluation Dataset Design |
| Evaluation Rules | Custom & Built-in Metrics |
| Evaluate | Automated LLM Evaluation |
| Results | Performance Analysis |
| MLflow Tracking | Experiment Tracking |
