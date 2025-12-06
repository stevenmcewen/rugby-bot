import os
import types

import pytest

from functions.config import settings as app_settings


@pytest.fixture(autouse=True)
def clear_settings_caches():
    """
    Clear caches between tests so each test can control behaviour deterministically
    and isn't affected by earlier imports of settings in other modules.
    """
    app_settings.get_key_vault_client.cache_clear()
    app_settings.get_settings.cache_clear()

# Does get_key_vault_client return the correct client?
# must return the correct client and set the correct vault url and credential
def test_get_key_vault_client_success(monkeypatch):
    created = {}

    class FakeCredential:
        pass

    class FakeSecretClient:
        def __init__(self, vault_url: str, credential):
            created["vault_url"] = vault_url
            created["credential"] = credential

    monkeypatch.setattr(app_settings, "DefaultAzureCredential", FakeCredential)
    monkeypatch.setattr(app_settings, "SecretClient", FakeSecretClient)
    monkeypatch.setenv("KEY_VAULT_URL", "https://example-vault.vault.azure.net/")

    client = app_settings.get_key_vault_client()

    assert isinstance(client, FakeSecretClient)
    assert created["vault_url"] == "https://example-vault.vault.azure.net/"
    assert isinstance(created["credential"], FakeCredential)

    # Ensure cached instance is reused
    same_client = app_settings.get_key_vault_client()
    assert same_client is client

# Does get_key_vault_client return None if the client cannot be created?
# must return None if the client cannot be created and set the correct error message
def test_get_key_vault_client_failure(monkeypatch):
    def failing_credential():
        raise RuntimeError("no identity")

    monkeypatch.setattr(app_settings, "DefaultAzureCredential", failing_credential)

    client = app_settings.get_key_vault_client()

    assert client is None


# Does get_secret prefer the environment variable if it is present?
# must return the correct secret from the environment variable and not use the key vault client
def test_get_secret_prefers_env_var(monkeypatch):
    # Ensure that if the env var is present, key vault client is never used.
    monkeypatch.setenv("MY_ENV", "from-env")

    # If this is called, the test should fail.
    monkeypatch.setattr(
        app_settings,
        "get_key_vault_client",
        lambda: (_ for _ in ()).throw(RuntimeError("should not be called")),
    )

    value = app_settings.get_secret("SOME-SECRET", default_env_var="MY_ENV")
    assert value == "from-env"

# Does get_secret return the correct secret from the key vault?
# must return the correct secret from the key vault and set the correct requested secret name
def test_get_secret_from_key_vault(monkeypatch):
    class FakeSecret:
        def __init__(self, value: str):
            self.value = value

    class FakeClient:
        def __init__(self):
            self.requested = []

        def get_secret(self, name: str):
            self.requested.append(name)
            return FakeSecret("from-kv")

    fake_client = FakeClient()

    monkeypatch.setattr(app_settings, "get_key_vault_client", lambda: fake_client)

    value = app_settings.get_secret("MY-SECRET")

    assert value == "from-kv"
    assert fake_client.requested == ["MY-SECRET"]

# Does get_secret return None if the key vault client raises an error?
# must return None if the key vault client raises an error and set the correct error message
def test_get_secret_key_vault_error_returns_none(monkeypatch):
    class FakeClient:
        def get_secret(self, name: str):
            raise RuntimeError("kv error")

    monkeypatch.setattr(app_settings, "get_key_vault_client", lambda: FakeClient())

    value = app_settings.get_secret("MY-SECRET")
    assert value is None


# Does get_settings use the correct secrets and parse booleans correctly?
# must use the correct secrets and parse booleans correctly and set the correct settings
def test_get_settings_uses_secrets_and_parses_booleans(monkeypatch):
    values = {
        "ENVIRONMENT": "dev",
        "SQL-SERVER": "server.database.windows.net",
        "SQL-DATABASE": "db",
        "STORAGE-CONNECTION": "UseDevelopmentStorage=true",
        "RAW-INGESTION-CONTAINER": "raw",
        "ENABLE-SCHEDULED-FUNCTIONS": "TrUe",
        "KAGGLE-DATASET": "owner/dataset",
    }

    def fake_get_secret(secret_name: str, default_env_var: str | None = None) -> str | None:
        # Simulate the current implementation: if default env var is provided and exists, use it first
        if default_env_var and os.getenv(default_env_var):
            return os.getenv(default_env_var)
        return values.get(secret_name)

    monkeypatch.setenv("AzureWebJobsStorage", "UseDevelopmentStorage=true")
    monkeypatch.setattr(app_settings, "get_secret", fake_get_secret)

    settings = app_settings.get_settings()

    assert settings.environment == "dev"
    assert settings.sql_server == "server.database.windows.net"
    assert settings.sql_database == "db"
    assert settings.storage_connection == "UseDevelopmentStorage=true"
    assert settings.raw_container_name == "raw"
    assert settings.enable_scheduled_functions is True
    assert settings.kaggle_dataset == "owner/dataset"

    # Cached behaviour
    same_settings = app_settings.get_settings()
    assert same_settings is settings


