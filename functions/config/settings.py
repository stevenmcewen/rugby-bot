from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient

from functions.logging.logger import get_logger
logger = get_logger(__name__)

@dataclass
class AppSettings:
    """
    Strongly-typed application configuration.

    Values are primarily sourced from environment variables, which in Azure
    Functions come from Application Settings or `local.settings.json` in
    local development.
    """

    environment: str | None = None
    sql_server: str | None = None
    sql_database: str | None = None

    storage_connection: str | None = None
    raw_container_name: str | None = None
    artifact_container_name: str | None = None
    kaggle_dataset: str | None = None

    enable_scheduled_functions: bool = False

### key vault helpers ###
@lru_cache()
def get_key_vault_client() -> SecretClient:
    """ gets a secret client for the key vault, returns None if the client cannot be created """
    try:
        credential = DefaultAzureCredential()
        client = SecretClient(vault_url=os.getenv("KEY_VAULT_URL"), credential=credential)
        return client
    except Exception as e:
        logger.error(f"Error getting key vault client: {e}")
        return None

def get_secret(secret_name: str, default_env_var: str|None = None) -> str|None:
    """ gets a secret from the key vault, or returns the default value if not found,
        which should be found in the system environment variables 
        
        accepts:
            secret_name: the name of the secret to get
            default_env_var: the environment variable that contains the value to return if not stored in the key vault
        returns:
            the secret value or the default value if not stored in the key vault
        """
    if default_env_var:
        if os.getenv(default_env_var):
            secret_value = os.getenv(default_env_var)
            return secret_value

    client = get_key_vault_client()
    if client:
        try:
            secret = client.get_secret(secret_name)
            secret_value = secret.value
            return secret_value
        except Exception as e:
            logger.error(f"Error getting secret {secret_name}: {e}")
            return None
    else:
        logger.error(f"Key vault client not found")
        return None

### cached settings helper ###
@lru_cache()
def get_settings() -> AppSettings:
    """
    Load and cache application settings.

    This is called once per worker process and reused across function
    invocations, so configuration access is cheap and consistent.
    """

    environment = get_secret(secret_name="ENVIRONMENT")
    sql_server = get_secret(secret_name="SQL-SERVER")
    sql_database = get_secret(secret_name="SQL-DATABASE")
    storage_connection = get_secret(
        secret_name="STORAGE-CONNECTION",
        default_env_var="AzureWebJobsStorage",
    )
    raw_container_name = get_secret(secret_name="RAW-INGESTION-CONTAINER")
    artifact_container_name = get_secret(secret_name="ARTIFACT-CONTAINER")

    # Treat a missing or unset flag as "false" so local dev / tests
    # without Key Vault configured don't crash on `.lower()`.
    enable_flag_raw = get_secret(secret_name="ENABLE-SCHEDULED-FUNCTIONS")
    enable_scheduled_functions = (enable_flag_raw or "").lower() == "true"

    kaggle_dataset = get_secret(secret_name="KAGGLE-DATASET")


    return AppSettings(
        environment=environment,
        sql_server=sql_server,
        sql_database=sql_database,
        storage_connection=storage_connection,
        raw_container_name=raw_container_name,
        artifact_container_name=artifact_container_name,
        enable_scheduled_functions=enable_scheduled_functions,
        kaggle_dataset=kaggle_dataset,
    )


