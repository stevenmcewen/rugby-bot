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

---

# Extending the System

The architecture is designed so that new data sources, preprocessing pipelines, and models can be added with **minimal changes to existing code**. This section documents what stays the same and what needs to be added.

## What Never Changes

These components form the backbone and **require no modification** when adding new functionality:

1. **The Event Tracking Layer** (`dbo.system_events`, `dbo.ingestion_events`, `dbo.preprocessing_events`)
   - All new sources automatically use the same event tables.
   - Status tracking and audit trails work identically for any new source.

2. **The Orchestration Engines**
   - `SelectIngestedSourcesStep`, `BuildPreprocessingPlansStep`, `PersistPreprocessingEventsStep`, `RunPreprocessingPipelinesStep` — these are generic and query metadata to discover work.
   - They don't know or care whether you're ingesting Rugby365 or URC; they read the same `ingestion_events` table.

3. **The Factory Pattern**
   - Pipeline factories (`_default_preprocessing_factory`, etc.) remain unchanged.
   - New pipelines are *registered* in existing registries, not added to orchestration logic.

4. **The SQL Data Access Layer** (`functions/sql/sql_client.py`)
   - Provides all CRUD operations needed by any stage.
   - New sources use the same `SqlClient` methods; no SQL layer changes needed.

## Adding a New Ingestion Source

To ingest data from a completely new provider (e.g., football/soccer data from ESPN instead of rugby data from Rugby365):

### 1. Create an ingestion function in `function_app.py`

```python
@app.function_name("IngestESPNFootballFixtures")
@app.schedule(schedule="0 0 * * *")  # Daily at midnight
def ingest_espn_football_fixtures(mytimer: functions.TimerRequest) -> None:
    """Ingest football (soccer) fixtures from ESPN API."""
    orchestrate_ingestion(
        sql_client=sql_client,
        system_event_id=uuid4(),
        source_provider="espn",
        source_type="fixtures",
        pipeline_name="default_ingestion",
    )
```

### 2. Implement the concrete integration in `functions/data_ingestion/integration_helpers.py`

Create helper functions to fetch, validate, and write your data:

```python
def fetch_espn_football_fixtures_data() -> pd.DataFrame:
    """Fetch football fixtures from ESPN API."""
    # ... ESPN API call logic ...
    return fixtures_df

def validate_espn_football_fixtures_data(df: pd.DataFrame) -> None:
    """Validate structure and required columns."""
    # ... validation logic ...
```

### 3. Register the integration in `functions/data_ingestion/integration_services.py`

Add your helper functions to the integration registry:

```python
INTEGRATION_REGISTRY = {
    ("rugby365", "fixtures"): fetch_rugby365_fixtures_data,
    ("rugby365", "results"): fetch_rugby365_results_data,
    ("espn", "fixtures"): fetch_espn_football_fixtures_data,        # NEW
    ("espn", "results"): fetch_espn_football_results_data,           # NEW
}
```

### 4. Add schema metadata to SQL

Insert rows into `dbo.schema_tables` and `dbo.schema_columns` describing your source structure:

```sql
INSERT INTO dbo.schema_tables (table_name, integration_provider, integration_type, is_source)
VALUES ('ESPN Football Fixtures Raw', 'espn', 'fixtures', 1);

INSERT INTO dbo.schema_columns (table_id, column_name, column_type, nullable, ...)
VALUES (...);  -- One row per column
```

**That's it for ingestion.** The orchestration automatically:
- Picks up your new `ingest_espn_football_fixtures` function at runtime.
- Writes ingestion events to the same `ingestion_events` table.
- Downstream preprocessing discovers work by querying that table.

---

## Adding a New Preprocessing Pipeline

To transform data from a new sport/provider (or a new transformation of an existing source):

### 1. Create the pipeline function in `functions/data_preprocessing/preprocessing_pipelines.py`

```python
def espn_football_fixtures_preprocessing_pipeline(preprocessing_event: "PreprocessingEvent", sql_client: SqlClient) -> None:
    """Preprocessing pipeline for ESPN football fixtures data."""
    try:
        source_data = get_source_data(preprocessing_event, sql_client)
        if source_data.empty:
            logger.warning("No source rows for event=%s; skipping.", preprocessing_event.id)
            return
        
        source_schema = get_source_schema(preprocessing_event, sql_client)
        target_schema = get_target_schema(preprocessing_event, sql_client)
        validate_source_data(source_data, source_schema)
        
        # Transform using your new helper
        transformed_data = transform_espn_football_fixtures_to_international_fixtures(source_data, preprocessing_event, sql_client)
        
        if transformed_data.empty:
            logger.warning("No transformed rows for event=%s; skipping write.", preprocessing_event.id)
            return
        
        validate_transformed_data(transformed_data, target_schema)
        truncate_target_table(preprocessing_event, sql_client)
        write_data_to_target_table(transformed_data, preprocessing_event, sql_client)
    except Exception as e:
        logger.error("Error running ESPN football fixtures preprocessing for event %s: %s", preprocessing_event.id, e)
        raise
```

### 2. Implement transformation logic in `functions/data_preprocessing/preprocessing_helpers.py`

```python
def transform_espn_football_fixtures_to_international_fixtures(
    source_data: pd.DataFrame,
    preprocessing_event: "PreprocessingEvent",
    sql_client: SqlClient,
) -> pd.DataFrame:
    """Convert ESPN football fixtures schema to a standardized fixtures schema."""
    # ... transformation logic ...
    return transformed_df
```

### 3. Register the pipeline in `PREPROCESSING_HANDLER_REGISTRY`

```python
PREPROCESSING_HANDLER_REGISTRY: dict[str, PreprocessingHandler] = {
    # ... existing entries ...
    "espn_football_fixtures_preprocessing": espn_football_fixtures_preprocessing_pipeline,  # NEW
}
```

### 4. Add a source-to-target mapping in SQL

Insert into `dbo.preprocessing_source_target_mappings`:

```sql
INSERT INTO dbo.preprocessing_source_target_mappings
    (source_provider, source_type, target_table, pipeline_name, source_table, execution_order)
VALUES
    ('espn', 'fixtures', 'InternationalMatchFixtures', 'espn_football_fixtures_preprocessing', NULL, 1);
```

**That's it for preprocessing.** The orchestration automatically:
- Queries `preprocessing_source_target_mappings` to discover which pipeline to run.
- Creates `preprocessing_events` for each source matching your mapping.
- Executes your pipeline via the registered handler.
- No changes to `preprocessing_services.py` or the orchestration steps are needed.

---

## Adding a New Model

To train a new model for a different sport or add a new scoring variant:

### 1. Create model and trainer helpers in `functions/ml_models/helpers/`

```python
# model_factory.py
def build_football_prediction_model() -> BaseEstimator:
    """Factory to build a football-specific model."""
    return XGBClassifier(...)

# ml_training_helpers.py
def train_football_model(X_train: pd.DataFrame, y_train: pd.Series) -> BaseEstimator:
    """Train the football prediction model."""
    model = build_football_prediction_model()
    model.fit(X_train, y_train)
    return model
```

### 2. Register in the model factory registry

```python
MODEL_REGISTRY = {
    "international_match_home_win": build_international_match_home_win_model,
    "football_match_prediction": build_football_prediction_model,  # NEW
}
```

### 3. Create or update an `ml_pipeline` function in `functions/ml_models/ml_pipelines.py`

```python
def train_football_model_pipeline(ml_event: "MLEvent", sql_client: SqlClient) -> None:
    """Training pipeline for football prediction model."""
    # ... training logic ...
```

### 4. Register in SQL

Insert into any model registry table (if you have one) or add configuration to `dbo.model_configuration`:

```sql
INSERT INTO dbo.model_configuration (model_name, status, ...)
VALUES ('football_match_prediction', 'active', ...);
```

**Again, the orchestration engine** (`ml_orchestrator.py`) queries metadata tables to discover which models to train and score — no changes to the core orchestration needed.

---

## Design Pattern Payoffs

This architecture's robustness comes from:

1. **Metadata-Driven Discovery**
   - Orchestration steps query config tables, not hard-coded lists.
   - Add a row to `preprocessing_source_target_mappings`, and the system discovers your new pipeline automatically.

2. **Pluggable Registries**
   - Each layer has a registry (`PREPROCESSING_HANDLER_REGISTRY`, `INTEGRATION_REGISTRY`, `MODEL_REGISTRY`).
   - Register your new handler/helper without modifying the orchestration that calls it.

3. **Step-Based Orchestration**
   - Each responsibility (discover sources → build plans → run pipelines) is isolated in a step.
   - Adding a new source type doesn't require touching `RunPreprocessingPipelinesStep`.

4. **Uniform Event Tracking**
   - All sources, pipelines, and models use the same `system_events`, `ingestion_events`, `preprocessing_events` tables.
   - No new tracking infrastructure needed for new functionality.

5. **Centralised Data Access**
   - `SqlClient` provides all CRUD operations.
   - New integrations use the same SQL layer; no custom DB access code.

---

## Checklist: Adding a New Component

When adding a **new data source, preprocessing pipeline, or model**, verify:

- ✅ Implemented concrete helper functions (fetch, validate, transform, score, etc.)
- ✅ Registered in the appropriate registry (`INTEGRATION_REGISTRY`, `PREPROCESSING_HANDLER_REGISTRY`, etc.)
- ✅ Added metadata rows to SQL (`schema_tables`, `preprocessing_source_target_mappings`, etc.)
- ✅ Added corresponding test file in `tests/` (following naming convention `test_<module>.py`)
- ✅ No modifications to orchestration engines or factory methods
- ✅ No new SQL tables (reuse event and metadata tables)

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
| Communication Service | `acs-rugbybot-email` | Communication service for sending emails |
| Email Communication Service | `ecs-rugbybot-email` | Email communication service for sending emails |


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
├── tests/                              # Unit tests (one unit test file per module) 
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
    │   └── notification_services_helpers.py    # Helper functions for Prediction summary + email sending
    ├── sql/
    │   ├── __init__.py
    │   └── sql_client.py               # Centralised SQL access layer (Managed Identity)
    └── utils/
        ├── __init__.py
        └── utils.py                    # Shared utilities and helpers
```
