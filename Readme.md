
## Create a New Cluster (Compute)

On the **New Compute (Cluster)** page, create a new cluster with the following settings:

- **Cluster name:** User Name’s cluster (default cluster name)
- **Policy:** Unrestricted
- **Machine learning:** Enabled
- **Databricks runtime:** 17.3 LTS
- **Use Photon Acceleration:** Unselected
- **Worker type:** Standard_D4pds_v6
- **Single node:** Checked
- **Terminate after:** 30 minutes of inactivity

## Python Configuration

```python
import os

os.environ["AZURE_OPENAI_API_KEY"] = "xxxxxxxxxxx"
os.environ["AZURE_OPENAI_ENDPOINT"] = "xxxxxxxxxxxx"
os.environ["AZURE_OPENAI_API_VERSION"] = "2024-12-01-preview"

## Model Details

- **Model name:** `gpt-4.1-mini`
> **Note:** Key and endpoint will be provided during the training.
