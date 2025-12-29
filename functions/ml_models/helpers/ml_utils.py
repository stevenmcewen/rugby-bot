from __future__ import annotations

from functions.logging.logger import get_logger
from functions.config.settings import get_settings
import pickle
from typing import Any
from azure.storage.blob import BlobClient

logger = get_logger(__name__)
settings = get_settings()

def serialize_model_artifact(model_object: Any) -> bytes:
    """
    Serialize a trained model object to a byte stream.

    Accepts:
        model_object: The trained model instance to serialize.

    Returns:
        A bytes object representing the serialized model.
    """
    bytes_object = pickle.dumps(model_object)
    return bytes_object

def deserialize_model_artifact(model_bytes: bytes) -> Any:
    """
    Deserialize a byte stream back into a trained model object.

    Accepts:
        model_bytes: A bytes object representing the serialized model.

    Returns:
        The deserialized trained model instance.
    """
    model_object = pickle.loads(model_bytes)
    return model_object

def persist_model_artifact(sql_client, system_event_id, model_key, trainer_key, prediction_type, target_column, schema_hash, metrics, artifact_bytes) -> tuple[str, int]:
    """
    Persist a trained model artifact by uploading it to blob storage and saving metadata to SQL.
    Accepts:
        sql_client: An instance of the SQL client to interact with the database.
        system_event_id: The ID of the system event triggering this persistence.
        model_key: The key identifying the model type.
        trainer_key: The key identifying the trainer type.
        prediction_type: The type of prediction (e.g., classification, regression).
        target_column: The name of the target column in the dataset.
        schema_hash: A hash representing the data schema used for training.
        metrics: A dictionary of evaluation metrics for the trained model.
        artifact_bytes: The serialized model artifact as a bytes object.
    Returns:
        An instance of PersistedArtifact containing metadata about the stored model artifact.
    """

    # 1) Build blob path
    artifact_version = sql_client.get_next_artifact_version(model_key=model_key)
    blob_path = build_blob_path(
        model_key=model_key,
        artifact_version=artifact_version
    )

    # 2) Upload to blob storage
    blob_container = settings.artifact_container_name
    logger.info(f"Uploading model artifact to blob storage at {blob_container}/{blob_path}")
    try:
        upload_bytes_to_blob(
            container_name=blob_container,
            blob_path=blob_path,
            data=artifact_bytes,
        )
    except Exception as e:
        logger.exception("Error uploading model artifact to blob storage")
        raise ValueError("Model artifact upload failed") from e

    # 3) Persist metadata to SQL
    logger.info("Persisting model artifact metadata to SQL")
    artefact_id, artefact_version = sql_client.persist_artifact_metadata(
        system_event_id=system_event_id,
        model_key=model_key,
        trainer_key=trainer_key,
        prediction_type=prediction_type,
        target_column=target_column,
        schema_hash=schema_hash,
        artifact_version=artifact_version,
        blob_container=blob_container,
        blob_path=blob_path,
        metrics=metrics,
    )

    return artefact_id, artefact_version

def build_blob_path(*, model_key: str, artifact_version: int) -> str:
    """
    Build a blob storage path for the model artifact based on its metadata.
    Accepts:
        model_key: The key identifying the model type.
        artifact_version: The version of the artifact.
    Returns:
        A string representing the blob storage path.
    """
    blob_path = f"models/{model_key}/v{artifact_version}/model_artifact.pkl"
    return blob_path

def upload_bytes_to_blob(*, container_name: str, blob_path: str, data: bytes) -> None:
    """
    Upload a bytes object to blob storage at the specified container and path.
    Accepts:
        container_name: The name of the container to upload the data to.
        blob_path: The path to the blob in the container.
        data: The data to upload as a bytes object.
    Returns:
        None
    """
    # Check if the container name and storage connection are configured before attempting to upload
    if not container_name:
        raise ValueError("container_name must be provided (artifact container is not configured)")
    if not settings.storage_connection:
        raise ValueError("storage_connection is not configured")

    blob_client = BlobClient.from_connection_string(
                conn_str=settings.storage_connection,
                container_name=container_name,
                blob_name=blob_path,
            )

    blob_client.upload_blob(data, overwrite=True)

