
# Evaluate Large Language Models using Azure Databricks and Azure OpenAI

> Training Notes

## Table of Contents
1. Lab Objective
2. Cell 1 – Install Required Libraries
3. Cell 2 – Configure Azure OpenAI Credentials
4. Cell 3 – Build the LLM Application
5. Cell 4 – Create the Evaluation Dataset
6. Cell 5 – Define Evaluation Scorers
7. Cell 6 – Improve the System Prompt
8. Cell 7 – Evaluate the Updated Prompt
9. Overall Workflow
10. Learning Outcomes

---

# Lab Objective

In this lab, you will learn how to:

- Connect Azure Databricks with Azure OpenAI
- Build a simple LLM application
- Automatically trace LLM calls using MLflow
- Create an evaluation dataset
- Define evaluation criteria
- Evaluate GPT-4.1 responses
- Improve prompts and compare results

---

# Cell 1 – Install Required Libraries

```python
%pip install --upgrade "mlflow[databricks]>=3.1.0" openai "databricks-connect>=16.1"
dbutils.library.restartPython()
```

## What it does
- Installs mlflow, openai and databricks-connect.
- Restarts the Python runtime.

## Why it is needed
Prepares the environment for LLM development and evaluation.

## Output
The environment is ready.

👉 **Key Takeaway:** Prepares the notebook environment.

---

# Cell 2 – Configure Azure OpenAI Credentials

```python
import os

os.environ["AZURE_OPENAI_API_KEY"] = "..."
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://<your-resource>.openai.azure.com/"
os.environ["AZURE_OPENAI_API_VERSION"] = "2024-12-01-preview"
```

## What it does
Stores API Key, Endpoint and API Version.

## Why it is needed
Azure OpenAI requires these values for every request.

## Output
No visible output.

👉 **Key Takeaway:** Configures Azure OpenAI.

---

# Cell 3 – Build the LLM Application

```python
import json
import os
import mlflow
from openai import AzureOpenAI

mlflow.openai.autolog()
```

## What it does
- Enables MLflow tracing.
- Creates Azure OpenAI client.
- Defines the system prompt.
- Builds the `generate_game()` function.

## Why it is needed
Creates the application that will be evaluated.

## Output
Generates a funny sentence.

👉 **Key Takeaway:** Builds the LLM application.

---

# Cell 4 – Create the Evaluation Dataset

```python
eval_data = [...]
```

## What it does
Creates five evaluation prompts.

## Why it is needed
Provides standard test inputs.

## Output
Python list.

👉 **Key Takeaway:** Creates benchmark prompts.

---

# Cell 5 – Define Evaluation Scorers

```python
from mlflow.genai.scorers import Guidelines, Safety
```

## What it does
Defines:
- Same Language
- Funny
- Child Safe
- Template Match
- Safety

## Why it is needed
Measures response quality.

## Output
List of scorers.

👉 **Key Takeaway:** Defines evaluation rules.

---

# Cell 6 – Improve the System Prompt

```python
SYSTEM_PROMPT = """..."""
```

## What it does
Uses prompt engineering and few-shot examples.

## Why it is needed
Improves creativity and consistency.

## Output
Updated prompt.

👉 **Key Takeaway:** Better prompt = better responses.

---

# Cell 7 – Evaluate the Updated Prompt

```python
results = mlflow.genai.evaluate(
    data=eval_data,
    predict_fn=generate_game,
    scorers=scorers
)
```

## What it does
Runs MLflow GenAI evaluation.

## Why it is needed
Automatically evaluates GPT responses.

## Output
Evaluation report.

👉 **Key Takeaway:** Measures LLM quality.

---

# Overall Workflow

```text
Install Libraries
      ↓
Configure Azure OpenAI
      ↓
Build LLM
      ↓
Create Evaluation Dataset
      ↓
Define Scorers
      ↓
Improve Prompt
      ↓
Run MLflow Evaluation
```

# Learning Outcomes

- Connect Azure Databricks to Azure OpenAI.
- Build GPT-4.1 applications.
- Enable MLflow tracing.
- Create evaluation datasets.
- Define evaluation guidelines.
- Improve prompts.
- Evaluate responses using MLflow.
