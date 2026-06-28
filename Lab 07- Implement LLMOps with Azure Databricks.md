# Implement LLMOps with Azure Databricks — Cell-by-Cell Explanation

## ✅ Cell 1: Install Required Libraries
```python
%pip install openai mlflow
dbutils.library.restartPython()
```
### Explanation
- Installs required libraries: `openai`, `mlflow`
- Restarts Python kernel so changes take effect

### Purpose
Prepare environment for LLM + MLflow tracking

---

## ✅ Cell 2: Set Azure OpenAI Credentials
```python
import os

os.environ["AZURE_OPENAI_API_KEY"] = "..."
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://....openai.azure.com/"
os.environ["AZURE_OPENAI_API_VERSION"] = "2024-12-01-preview"
```
### Explanation
- Stores credentials securely using environment variables

### Purpose
Enable secure connection to Azure OpenAI

---

## ✅ Cell 3: Initialize Azure OpenAI Client
```python
from openai import AzureOpenAI

client = AzureOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
)
```
### Explanation
- Creates client object for communicating with Azure OpenAI

### Purpose
Used to send prompts to LLM

---

## ✅ Cell 4: Setup MLflow + System Prompt
```python
import mlflow

system_prompt = "Assistant is a large language model trained by OpenAI."

mlflow.openai.autolog()
```
### Explanation
- Defines system behavior of model
- Enables automatic logging using MLflow

### Purpose
Track prompts, responses, tokens automatically

---

## ✅ Cell 5: Run LLM + Track with MLflow
```python
with mlflow.start_run():
    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "Tell me a joke about animals."}
        ]
    )

    print(response.choices[0].message.content)

    mlflow.log_param("completion_tokens", response.usage.completion_tokens)

mlflow.end_run()
```

### Explanation
1. Start MLflow run
2. Send prompt to LLM
3. Receive response
4. Print output
5. Log token usage
6. End run

### Purpose
Run LLM query and track everything using MLflow

---

## Overall Workflow
1. Install dependencies
2. Set credentials
3. Initialize client
4. Enable MLflow tracking
5. Run LLM request
6. Track outputs and metrics

---

## Key Concepts

### LLMOps
- Managing LLM lifecycle (tracking, monitoring, evaluation)

### MLflow
- Tracks experiments, parameters, outputs

### Azure OpenAI
- Enterprise API for OpenAI models

---

## Final Summary
This notebook demonstrates:
- Azure OpenAI integration
- MLflow tracking
- LLM execution workflow

Send prompt → get response → track everything
