## Rugby Betting Intelligence Bot

A cloud-native machine learning system for analysing rugby match data and generating match predictions.  
This project is designed as a long-term software engineering exercise, following real-world patterns, clean architecture, and Azure serverless best practices.

---

## Project Overview

The Rugby Intelligence Bot:

- Ingests historical and live rugby data (fixtures, results, and contextual information such as venue and team form).
- Builds predictive models for match outcomes and point margins.
- Sends regular emails summarising upcoming fixtures and the model’s predictions.
- (Future extension) Can be enhanced to compare predictions to bookmaker lines to detect potential value opportunities.
- Runs entirely on Azure Functions using serverless compute.

The architecture is designed to be maintainable, scalable, and production-ready.

---

# Infrastructure Overview

All resources are deployed into the following resource group:

```
rg-rugbybot-dev-zn
```

## Core Azure Services

| Resource Type | Resource Name | Purpose |
|---------------|---------------|---------|
| Function App | `func-rugbybot-dev` | Executes ingestion, API, ML, and scheduled tasks |
| App Service Plan (Consumption) | `ASP-rgrugbybotdevzn-a2fd` | Serverless compute for the Function App |
| Storage Account | `stgrugbybotdev` | Required runtime storage and application data |
| SQL Server | `sql-rugbybot-dev` | Server for Azure SQL database |
| SQL Database | `sqldb-rugbybot-core-dev` | Stores match data, features, and results |
| Key Vault | `kv-rugbybot-dev` | Secure secrets store |
| Application Insights | `func-rugbybot-dev` | Application monitoring and telemetry |

---

# Authentication and Security

## Managed Identity

A system-assigned Managed Identity is enabled for the Function App and is used for:

- Authenticating to Azure SQL (passwordless)
- Retrieving secrets from Azure Key Vault
- Future integration with other Azure resources

## Azure SQL Access

Azure SQL authentication is configured using Entra ID:

1. SQL Server has an Entra ID admin assigned.
2. A contained database user is created for the Function App using its Managed Identity.

```sql
CREATE USER [func-rugbybot-dev] FROM EXTERNAL PROVIDER;
ALTER ROLE db_datareader ADD MEMBER [func-rugbybot-dev];
ALTER ROLE db_datawriter ADD MEMBER [func-rugbybot-dev];
```

### Event and Ingestion Metadata Tables

To coordinate work between ingestion, preprocessing, and other functions, the system uses a small hierarchy of SQL tables:

- `dbo.system_events`: **high-level function lifecycle tracking**
  - One row per function execution (e.g. `IngestRugby365ResultsFunction`, `BuildFeatureTablesFunction`).
  - Key columns:
    - `id` (`UNIQUEIDENTIFIER`, PK) – event ID.
    - `function_name` – name of the Azure Function.
    - `trigger_type` – e.g. `http`, `timer`, `queue`.
    - `event_type` – semantic label, e.g. `ingestion`, `preprocessing`.
    - `status` – e.g. `started`, `succeeded`, `failed`.
    - `started_at`, `completed_at`, `created_at` – UTC timestamps.
- `dbo.ingestion_events`: **detailed ingestion file / batch tracking**
  - Used both for logging and as orchestration metadata between ingestion and preprocessing.
  - Key columns:
    - `id` (`UNIQUEIDENTIFIER`, PK).
    - `batch_id` – groups related ingestion events in a single run.
    - `system_event_id` – optional FK → `system_events.id` to tie back to the parent function invocation.
    - `integration_type` – e.g. `historical_results`, `rugby365_results`, `rugby365_fixtures`.
    - `integration_provider` – e.g. `kaggle`, `rugby365`.
    - `container_name`, `blob_path` – where the raw file lives in Blob Storage.
    - `status` – lifecycle of the file/batch, e.g. `ingested`, `preprocessing`, `preprocessed`, `failed`.
    - `error_message` – optional error details.
    - `created_at`, `updated_at` – UTC timestamps.

Ingestion functions insert rows into `ingestion_events` (and optionally `system_events`) when raw files are written to Blob.  
Preprocessing functions query `ingestion_events` by `status` / `integration_type` / `integration_provider` to discover which raw files need to be processed next, providing a clean, decoupled handoff between stages.

## Key Vault Access

The Function App identity has:

Role: `Key Vault Secrets User`  
Principal: The Function App system-assigned identity

Secrets retrieved in Python using:

```python
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient
```

---

# Local Development Environment

## Virtual Environment Setup

```bash
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
```

## Azure Functions Core Tools

```bash
npm install -g azure-functions-core-tools@4 --unsafe-perm true
```

## Initialize Azure Functions Project

```bash
func init . --python
func new
func start
```

---

# CI/CD Pipeline (GitHub Actions → Azure Functions)

GitHub Actions automates deployment on each push to `main`.

Pipeline tasks:

- Setup Python 3.12
- Build the function app
- Create a deployment artifact
- Deploy using Azure publish profile

Workflow file location:

```
.github/workflows/
```

Logs visible in GitHub under Actions, and in Azure under Function App → Deployment Center.

---

## Deployment and Runtime

After deployment:

- Functions appear under Azure Function App → Functions
- Each HTTP function exposes an executable endpoint
- Logs and telemetry collected by Application Insights

---

## Monitoring and Logging

Application Insights provides:

- Structured logs
- Exception traces
- Performance metrics
- Request traces
- SQL dependency analysis

---

## Project Structure

Current structure:

```
rugby-bot/
│
├── .github/workflows/              # CI/CD pipelines
├── .venv/                          # Local venv (ignored in Git)
├── host.json                       # Function runtime config
├── local.settings.json             # Local config (not committed)
├── requirements.txt                # Dependencies
├── function_app.py                 # Azure Functions entrypoint / wiring
│
└── functions/                      # Application modules (reusable logic)
    ├── config/
    │   └── settings.py             # AppSettings + get_settings()
    ├── data_ingestion/
    │   └── services.py             # Kaggle + Rugby365 ingestion orchestration
    ├── data_preprocessing/
    │   └── services.py             # Bronze → silver feature building
    ├── logging/
    │   └── logger.py               # get_logger() and logging helpers
    ├── ml_models/
    │   └── services.py             # Training and scoring logic
    ├── notifications/
    │   └── services.py             # Prediction summary + email sending
    └── utils/
        └── utils.py                # Shared utilities (placeholder)
```
