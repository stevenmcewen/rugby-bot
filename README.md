# Rugby Betting Intelligence Bot

A cloud-native machine learning system for analysing rugby match data, generating predictions, and identifying potential bookmaker mispricing.  
This project is designed as a long-term software engineering exercise, following real-world patterns, clean architecture, and Azure serverless best practices.

---

## Project Overview

The Rugby Betting Intelligence Bot:

- Ingests historical and live rugby data (fixtures, results, odds, weather, venue, team form).
- Builds predictive models for match outcomes and point margins.
- Compares predictions to bookmaker lines to detect value opportunities.
- Sends alerts through email or API.
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

# Deployment and Runtime

After deployment:

- Functions appear under Azure Function App → Functions
- Each HTTP function exposes an executable endpoint
- Logs and telemetry collected by Application Insights

---

# Monitoring and Logging

Application Insights provides:

- Structured logs
- Exception traces
- Performance metrics
- Request traces
- SQL dependency analysis

---

# Project Structure

Recommended structure as project evolves:

```
rugby-bot/
│
├── .github/workflows/          # CI/CD pipelines
├── .venv/                      # Local venv (ignored in Git)
├── host.json                   # Function runtime config
├── local.settings.json         # Local config (not committed)
├── requirements.txt            # Dependencies
│
├── HttpTriggerSample/          # Example Azure Function
│   ├── __init__.py
│   └── function.json
│
└── rugbybot/                   # Application modules
    ├── ingestion/
    ├── preprocessing/
    ├── features/
    ├── models/
    ├── bookmakers/
    ├── notifications/
    ├── storage/
    └── utils/
```
