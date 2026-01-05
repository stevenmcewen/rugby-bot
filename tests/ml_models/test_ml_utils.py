from __future__ import annotations

from types import SimpleNamespace

import pytest

from functions.ml_models.helpers import ml_utils


def test_serialize_and_deserialize_roundtrip():
    obj = {"a": 1, "b": [1, 2, 3]}
    b = ml_utils.serialize_model_artifact(obj)
    out = ml_utils.deserialize_model_artifact(b)
    assert out == obj


def test_build_blob_path_uses_model_key_and_version():
    assert ml_utils.build_blob_path(model_key="m1", artifact_version=7) == "models/m1/v7/model_artifact.pkl"


def test_upload_bytes_to_blob_validates_config(monkeypatch):
    monkeypatch.setattr(ml_utils, "settings", SimpleNamespace(storage_connection=None))
    with pytest.raises(ValueError, match="storage_connection is not configured"):
        ml_utils.upload_bytes_to_blob(container_name="c", blob_path="p", data=b"x")

    monkeypatch.setattr(ml_utils, "settings", SimpleNamespace(storage_connection="UseDev"))
    with pytest.raises(ValueError, match="container_name must be provided"):
        ml_utils.upload_bytes_to_blob(container_name="", blob_path="p", data=b"x")


def test_upload_bytes_to_blob_calls_blob_client(monkeypatch):
    calls = {}

    class FakeBlobClient:
        def upload_blob(self, data, overwrite: bool):
            calls["data"] = data
            calls["overwrite"] = overwrite

    def fake_from_connection_string(*, conn_str, container_name, blob_name):
        calls["conn_str"] = conn_str
        calls["container_name"] = container_name
        calls["blob_name"] = blob_name
        return FakeBlobClient()

    monkeypatch.setattr(ml_utils, "settings", SimpleNamespace(storage_connection="UseDev"))
    monkeypatch.setattr(ml_utils, "BlobClient", SimpleNamespace(from_connection_string=fake_from_connection_string))

    ml_utils.upload_bytes_to_blob(container_name="cont", blob_path="x/y.pkl", data=b"abc")
    assert calls["container_name"] == "cont"
    assert calls["blob_name"] == "x/y.pkl"
    assert calls["data"] == b"abc"
    assert calls["overwrite"] is True


def test_persist_model_artifact_uploads_and_persists_metadata(monkeypatch):
    calls = {}

    class FakeSqlClient:
        def get_next_artifact_version(self, *, model_key: str) -> int:
            assert model_key == "m1"
            return 2

        def persist_artifact_metadata(
            self,
            *,
            system_event_id,
            model_key,
            trainer_key,
            prediction_type,
            target_column,
            schema_hash,
            artifact_version,
            blob_container,
            blob_path,
            metrics,
        ):
            calls["meta"] = {
                "system_event_id": system_event_id,
                "model_key": model_key,
                "trainer_key": trainer_key,
                "prediction_type": prediction_type,
                "target_column": target_column,
                "schema_hash": schema_hash,
                "artifact_version": artifact_version,
                "blob_container": blob_container,
                "blob_path": blob_path,
                "metrics": metrics,
            }
            return ("artifact-id", artifact_version)

    monkeypatch.setattr(ml_utils, "settings", SimpleNamespace(artifact_container_name="artifacts", storage_connection="UseDev"))
    monkeypatch.setattr(ml_utils, "upload_bytes_to_blob", lambda **kwargs: calls.setdefault("upload", kwargs))

    artefact_id, artefact_version = ml_utils.persist_model_artifact(
        FakeSqlClient(),
        system_event_id="sys",
        model_key="m1",
        trainer_key="t1",
        prediction_type="classification",
        target_column="HomeWin",
        schema_hash="hash",
        metrics={"acc": 0.9},
        artifact_bytes=b"blob",
    )

    assert artefact_id == "artifact-id"
    assert artefact_version == 2
    assert calls["upload"]["container_name"] == "artifacts"
    assert calls["upload"]["blob_path"] == "models/m1/v2/model_artifact.pkl"
    assert calls["upload"]["data"] == b"blob"
    assert calls["meta"]["artifact_version"] == 2
    assert calls["meta"]["blob_container"] == "artifacts"
    assert calls["meta"]["blob_path"] == "models/m1/v2/model_artifact.pkl"


def test_persist_model_artifact_raises_value_error_on_upload_failure(monkeypatch):
    class FakeSqlClient:
        def get_next_artifact_version(self, *, model_key: str) -> int:
            return 1

        def persist_artifact_metadata(self, **_kwargs):
            raise AssertionError("Should not be called if upload fails")

    monkeypatch.setattr(ml_utils, "settings", SimpleNamespace(artifact_container_name="artifacts", storage_connection="UseDev"))

    def boom(**_kwargs):
        raise RuntimeError("blob down")

    monkeypatch.setattr(ml_utils, "upload_bytes_to_blob", boom)

    with pytest.raises(ValueError, match="Model artifact upload failed"):
        ml_utils.persist_model_artifact(
            FakeSqlClient(),
            system_event_id="sys",
            model_key="m1",
            trainer_key="t1",
            prediction_type="classification",
            target_column="Y",
            schema_hash="h",
            metrics={"acc": 1.0},
            artifact_bytes=b"x",
        )


def test_load_model_artifact_accepts_blob_container_alias(monkeypatch):
    called = {}

    monkeypatch.setattr(ml_utils, "download_bytes_from_blob", lambda *, container_name, blob_path: (called.setdefault("args", (container_name, blob_path)) or b"bytes"))
    monkeypatch.setattr(ml_utils, "deserialize_model_artifact", lambda b: {"ok": True, "bytes": b})

    out = ml_utils.load_model_artifact({"blob_container": "cont", "blob_path": "p.pkl"})
    assert called["args"] == ("cont", "p.pkl")
    assert out["ok"] is True


def test_download_bytes_from_blob_validates_config(monkeypatch):
    monkeypatch.setattr(ml_utils, "settings", SimpleNamespace(storage_connection=None))
    with pytest.raises(ValueError, match="storage_connection is not configured"):
        ml_utils.download_bytes_from_blob(container_name="c", blob_path="p")

    monkeypatch.setattr(ml_utils, "settings", SimpleNamespace(storage_connection="UseDev"))
    with pytest.raises(ValueError, match="container_name must be provided"):
        ml_utils.download_bytes_from_blob(container_name="", blob_path="p")

