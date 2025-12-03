## Core Systems

### 1. Ingestion (Bronze)

- **Goal**: Collect raw rugby match data (results + fixtures) into append-only bronze tables.
- **Data flow**: External sources → **Azure Blob Storage** (Storage Account linked to the Function App) as a historical file system.
- **Functions**:
  - `IngestHistoricalResults` (HTTP-triggered; one-off/bootstrap or on-demand)
  - `IngestRugby365ResultsFunction` (TimerTrigger; daily results ingestion)
  - `IngestRugby365FixturesFunction` (TimerTrigger; daily fixtures ingestion)
- **Key modules**:
  - `functions.data_ingestion.services` (orchestration for all rugby data ingestion)
- **Patterns**: Adapter, Repository, Strategy
- **Storage conventions**:
  - Historical rugby data written to Blob paths like: `raw/historical/...`
  - Rugby365 results written to: `raw/rugby365/results/YYYY/MM/DD/...`
  - Rugby365 fixtures written to: `raw/rugby365/fixtures/YYYY/MM/DD/...`
  - Ingestion functions **do not write to SQL directly**; they are append-only and auditable.


## Data Sources

- **Historical results (up to 2024-08-17)**  
  - Primary source: Kaggle dataset `lylebegbie/international-rugby-union-results-from-18712022` (tier one men’s internationals).  
  - Ingestion approach:
    - One-off **bootstrap job** (local script or `BootstrapHistoricalDataFunction`) that:
      - Downloads the dataset (e.g. via `kagglehub`).
      - Uploads the raw CSV/JSON files into Blob Storage under `raw/historical/...`.
    - From that point on, the historical data is treated like any other bronze source (read by preprocessing into SQL silver).

- **Ongoing data (after 2024-08-17)**  
  - Candidate source: match results and fixtures from Rugby365 (via HTML scraping if APIs are unavailable).  
  - Ingestion approach:
    - A dedicated **results** function `IngestRugby365ResultsFunction` (TimerTrigger) that:
      - Scrapes completed match results from Rugby365.
      - Normalises to your internal raw results schema.
      - Writes snapshots to Blob under `raw/rugby365/results/YYYY/MM/DD/...`.
    - A dedicated **fixtures** function `IngestRugby365FixturesFunction` (TimerTrigger) that:
      - Scrapes upcoming match fixtures (teams, venue, kickoff time, competition).
      - Normalises to your internal raw fixtures schema.
      - Writes snapshots to Blob under `raw/rugby365/fixtures/YYYY/MM/DD/...`.
  - Considerations:
    - Be polite: modest frequency, caching where possible, and minimal load on their site.

- **(Future extension) Bookmaker odds data**  
  - Not required for the initial version of the project.  
  - Can be added later to compare model probabilities to market prices, re-using the same bronze → silver pattern.


### 2. Preprocessing (Silver)

- **Goal**: Transform bronze into clean, model-ready silver tables (features).
- **Data flow**: Blob Storage (bronze) → **Azure SQL Database** (silver).
- **Functions**:
  - `BuildFeatureTablesFunction` (TimerTrigger)
- **Key modules**:
  - `functions.data_preprocessing.services` (feature pipeline orchestration)
- **Patterns**: Pipeline, Template Method, Repository
- **SQL access**:
  - Functions use the Function App **managed identity** to connect to Azure SQL.
  - Database roles configured as per `README.md` (Entra ID user with `db_datareader` / `db_datawriter`).
  - Preprocessing functions are responsible for:
    - Reading new raw blobs (by schedule or via queue messages).
    - Validating/transforming into standard tabular schemas (matches, teams, odds, features).
    - Writing cleaned/standardised data to SQL as the **silver layer**.

### 3. ML Models

- **Goal**: Train and serve prediction models for match outcome probabilities.
- **Functions**:
  - `TrainModelsFunction` (TimerTrigger)
  - `ScoreUpcomingMatchesFunction` (TimerTrigger/QueueTrigger)
- **Key modules**:
  - `functions.ml_models.services` (training and scoring logic)
- **Patterns**: Strategy, Factory, Service Layer
- **Training strategy**:
  - **Phase 1 (baseline)**: periodic **full retrain** on a rolling window of historical data (e.g. last N seasons) from the silver tables in SQL.
  - **Phase 2 (future enhancement)**: optional **incremental / online learning** using algorithms that support `partial_fit` or warm-starting; training interface designed as `ModelTrainer.train(training_data, previous_model=None)` so implementations can switch between full and incremental without changing the surrounding Functions.

### 4. Notifications

- **Goal**: Email a list of upcoming fixtures (e.g. weekend matches) with the model’s predicted probabilities / scores.
- **Functions**:
  - `GenerateWeekendPredictionsFunction` (TimerTrigger) — selects upcoming fixtures, calls the prediction service, and prepares a summary payload.
  - `SendPredictionEmailFunction` (QueueTrigger) — takes the prepared payload and sends an email with the fixtures + predictions.
- **Key modules**:
  - `functions.notifications.services` (preparing payloads and sending emails)
- **Patterns**: Observer (via queues), Strategy (different email formats or recipient profiles)


## Cross-cutting Concerns

### Configuration

- **Goals**:
  - Centralise all configuration (connection strings, schedules, feature flags, email settings, etc.).
  - Make Functions easy to test locally by overriding config via environment variables / `local.settings.json`.
  - Keep secrets outside code, retrieved securely (Key Vault + managed identity).
- **Approach**:
  - Module: `functions.config.settings` containing:
    - An `AppSettings` object (e.g. `@dataclass` or Pydantic model) with typed fields for:
      - Azure SQL settings (server, database, etc. if needed beyond managed identity).
      - Storage containers and blob path prefixes (bronze/historical/results/fixtures).
      - Email settings (from address, recipients, subject prefixes).
      - Model training/scoring parameters (e.g. training window, default model name).
    - A `get_settings()` function that:
      - Reads environment variables and `local.settings.json` (via standard env in the Functions runtime).
      - Optionally calls Key Vault for secrets that cannot live in env/local settings.
  - **Reuse pattern**:
    - Each Function module calls a small helper (e.g. `get_settings()`) that caches the `AppSettings` instance at module level so it is created once per worker process, not per request.

### Logging

- **Goals**:
  - Consistent, structured logging across all Functions.
  - Logs automatically shipped to Application Insights with useful context (function name, correlation IDs).
  - Reusable logging helpers to avoid copy-paste.
- **Approach**:
  - Module: `functions.logging.logger` containing:
    - A `get_logger(name: str)` helper that returns a configured logger with:
      - Standard format (timestamp, level, function name, message).
      - Optional extra fields (e.g. correlation IDs, execution IDs) when available.
    - Convenience functions for common patterns:
      - `log_function_start(logger, function_name, **context)`
      - `log_function_error(logger, function_name, exc)`
  - **Reuse pattern**:
    - Each Function obtains a module-level logger once, e.g.:
      - `logger = get_logger(__name__)`
      - Then uses `logger.info(...)`, `logger.error(...)` throughout the function.
  - **Integration**:
    - Ensure Python logging is wired to Azure Functions host, so all logs go to Application Insights by default.