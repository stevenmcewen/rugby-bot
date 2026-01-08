## Rugby Betting Intelligence Bot

A cloud-native machine learning system for analysing rugby match data and generating match predictions.  
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

# Design Pattern
This application was designed from the perspective of making any future adjustments as seemless as possible. 
This required the approach of maximising the seperation of each component, independantly generating each step and automatically orchestrating
the independantly generated steps all with one uniform metadata tracking system to record the outcomes of eachstep and pass the required parameters
into the system steps where needed.
Thus, the system at each level was designed with an orchestration, pipeline and factory design philosophy.

At the highest level (function_app.py) each of the core systems required by the application are seperated into their own functions i.e.
1. Ingest (one function per source) automated to earliest in the morning.
2. Preprocess. Preprocess all the ingested sources that completed in the morning.
3. Model training (once per week after preprocessing), train on all existing preprocessed data.
4. Model Scoring (every morning after preprocessing), scores the preprocessed data on the latest trained model.
5. Notifications (last process every morning), inform us of the preditions from model scoring.

At each step, the output of a function is stored in allocated sql tables ready to be ingested by the next function, finishing that functions action for the day.
This fundamentally means that no one function is directly speaking to another function, the interface between each function is the SQL database.
This allows as many changes as possible to each function completely independantly of another so long as the sql database interfacing both functions remains consistent.

This design pattern principality leaks into each stage of the design
1. An orchestrator that orchestrates steps.
2. Independant steps that communicate with an orchestration level context. Steps which are made from repeatable step specific helpers or generalized utility helpers.
3. A Factory method that builds out the orchestration based on the steps assigned to the blueprint being called.

This allows us to implement variations of blueprints with new steps and variations of existing steps without having to interact with the overall orchestration flow or any existing working factory blueprints.

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

### Data Access and Event Tracking Philosophy

At runtime, the Functions code never talks directly to Azure SQL with ad‑hoc connection strings.  
Instead, it uses a small data access layer (`SqlClient` in `functions/sql/sql_client.py`) that:

- **Centralises how we talk to the database** so every function (ingestion, preprocessing, modelling, notifications) uses the same, tested path to SQL.
- **Uses managed identity under the hood** so credentials are never checked into source control or stored in configuration – they live only in Entra ID.
- **Records high-level “system events” and detailed pipeline events** in SQL whenever functions run or batches are processed, giving us an audit trail of:
  - which function ran,
  - which data batches were ingested / preprocessed,
  - and whether they succeeded, are in progress, or failed.
- **Supports orchestration-by-metadata**: downstream jobs (e.g. preprocessing) discover what to work on by querying these event tables, rather than by being tightly coupled to the ingestion code.

The goal is to treat the database not just as storage, but also as the “control plane” for long‑running ingestion pipelines: transparent, queryable, and easy to debug.

---

### Event and Ingestion Metadata Tables

To coordinate work between ingestion, preprocessing, and other functions, the system uses a small hierarchy of SQL tables:

- `dbo.system_events`: **high-level function lifecycle tracking**

- `dbo.ingestion_events`: **detailed ingestion file / batch tracking**

- `dbo.preprocessing_events`: **planned preprocessing work tracking**

Ingestion functions insert rows into `ingestion_events` (and `system_events`) when raw files are written to Blob.  
Preprocessing functions then:

1. Create corresponding rows in `preprocessing_events` for each planned preprocessing job.
2. Use the status fields on both `ingestion_events` and `preprocessing_events` to coordinate which batches are ready, in progress, or failed.

---

### Source / Target Mapping and Schema Metadata Tables

Beyond tracking *events*, the system also stores **schema metadata** so that pipelines can be driven by configuration rather than hard-coded schemas:

- `dbo.preprocessing_source_target_mappings`: **logical mapping from sources to preprocessing pipelines and targets**
  - Maps a `(source_provider, source_type)` pair to a target table and pipeline implementation.
  - Used by preprocessing orchestration to answer: “Given this ingested source (e.g. `kaggle` + `historical_results`), which preprocessing pipeline should run, and where should the data land?”

- `dbo.schema_tables`: **catalogue of logical tables**
  - One row per logical table in the system (source or target, or sometimes both).
  - Designed to be flexible:
    - Some rows may only have `integration_type` / `integration_provider` filled (for unnamed source “shapes”).
    - Others may only have `table_name` filled (for concrete target tables).
    - Some may use both when the same logical table acts as both source and target in different stages.

- `dbo.schema_columns`: **column-level schema definition**
  - One row per column belonging to a row in `schema_tables`.
  - Used to describe the expected structure of a table or file, and to drive validation / transformation logic.

Together, these metadata tables allow the platform to:

- Discover which preprocessing pipeline to run for a given source (`preprocessing_source_target_mappings`).
- Understand what the input and output schemas should look like (`schema_tables` and `schema_columns`).
- Track the lifecycle of data as it moves from raw ingestion through preprocessing to final model-ready tables (`system_events`, `ingestion_events`, `preprocessing_events`).

## Key Vault Access

The Function App identity has:

Role: `Key Vault Secrets User`  
Principal: The Function App system-assigned identity

Secrets are retrieved at runtime using the Azure SDK for Python (`DefaultAzureCredential` + `SecretClient`).

---

# Local Development Environment

## Virtual Environment Setup

```bash
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install -r requirements.txt
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

## CI/CD Pipeline (GitHub Actions → Azure Functions + Tests)

GitHub Actions automates deployment on each push to `main`, and can also run tests on pull requests.

Pipeline tasks:

- Setup Python 3.12
- Build the function app
- Create a deployment artifact
- Deploy using Azure publish profile

### Test Execution (pytest)

The same GitHub Actions workflow also runs Python unit tests using `pytest` as part of the `build` job:

- Creates a virtual environment and installs dependencies from `requirements.txt`.
- Installs `pytest` as a dev/CI-only dependency.
- Runs `pytest` from the repository root, using standard discovery rules:
  - Files named `test_*.py` or `*_test.py`
  - Located anywhere under the repo (e.g. a `tests/` folder or module-adjacent tests).
- If tests fail (non‑zero pytest exit code other than “no tests collected”), the `build` job fails and deployment does not proceed.
- If no tests are present yet, the workflow logs a message and continues, so will gradually add tests.

### Enforcing “tests must pass before merge” (GitHub branch protection)

This was something that I ideally wanted to do, I set up the branch policy but i needed a github teams account to enforce it.

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
├── .github/workflows/                  # CI/CD pipelines (build, tests, deploy)
├── host.json                           # Function runtime config
├── local.settings.json                 # Local config (not committed)
├── requirements.txt                    # Runtime dependencies
├── function_app.py                     # Azure Functions entrypoint / wiring
├── local testing notebooks/            # Local experimentation / diagnostics
│   └── local_ingest_historical_results_test.ipynb
│
└── functions/                          # Application modules (reusable logic)
    ├── config/
    │   ├── __init__.py
    │   └── settings.py                 # AppSettings + get_settings()
    ├── data_ingestion/
    │   ├── __init__.py
    │   ├── integration_helpers.py      # Helper functions for ingestion workflows
    │   └── integration_services.py     # ingestion orchestration
    ├── data_preprocessing/
    │   ├── __init__.py
    │   ├── preprocessing_helpers.py    # Helper functions for fine grained preprocessing actions
    │   ├── preprocessing_pipelines.py  # preprocessing function selection and orchestration  
    │   └── preprocessing_services.py   # preprocessor orchestration
    ├── logging/
    │   ├── __init__.py
    │   └── logger.py                   # get_logger() and logging helpers
    ├── ml_models/
    │   ├── __init__.py
    │   ├── ml_orchestrator.py
    │   ├── ml_pipelines.py
    │   └── helpers/
    │        ├── ml_scoring_helpers.py
    │        ├── ml_training_helpers.py
    │        ├── ml_utils.py
    │        ├── model_factory.py
    │        └── trainer_factory.py
    ├── notifications/
    │   ├── __init__.py
    │   └── services.py                 # Prediction summary + email sending
    ├── sql/
    │   ├── __init__.py
    │   └── sql_client.py               # Centralised SQL access layer (Managed Identity)
    └── utils/
        ├── __init__.py
        └── utils.py                    # Shared utilities and helpers
```
