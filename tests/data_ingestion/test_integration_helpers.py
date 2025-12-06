import types

import pytest

from functions.data_ingestion import integration_helpers as helpers

## positive test case for download_historical_results_from_kaggle###
# must download the historical results from Kaggle and return the correct local file path and integration dataset
def test_download_historical_results_from_kaggle_success(monkeypatch, tmp_path):
    # Arrange
    fake_dir = tmp_path / "kaggle-dataset"
    fake_dir.mkdir()

    def fake_dataset_download(dataset: str) -> str:
        # Ensure we received the dataset from settings
        assert dataset == "owner/dataset"
        return str(fake_dir)

    # Patch kagglehub
    monkeypatch.setattr(helpers, "kagglehub", types.SimpleNamespace(dataset_download=fake_dataset_download))

    # Patch settings used inside the helpers module
    helpers.settings = types.SimpleNamespace(kaggle_dataset="owner/dataset")

    # Act
    local_file_path, integration_dataset = helpers.download_historical_results_from_kaggle()

    # Assert
    assert integration_dataset == "owner/dataset"
    assert local_file_path.endswith("results.csv")
    assert str(fake_dir) in local_file_path

## negative test case for download_historical_results_from_kaggle###
# must raise a ValueError with the correct error message
def test_download_historical_results_from_kaggle_error(monkeypatch):
    # Arrange
    def fake_dataset_download(dataset: str) -> str:
        raise RuntimeError("kaggle down")

    monkeypatch.setattr(
        helpers,
        "kagglehub",
        types.SimpleNamespace(dataset_download=fake_dataset_download),
    )
    helpers.settings = types.SimpleNamespace(kaggle_dataset="owner/dataset")

    # Act / Assert
    with pytest.raises(ValueError) as exc:
        helpers.download_historical_results_from_kaggle()

    assert "Error downloading historical results from Kaggle" in str(exc.value)

## positive test case for download_historical_results###
# must download the historical results from the correct provider and return the correct local file path and integration dataset
def test_download_historical_results_valid_provider(monkeypatch):
    # Arrange
    def fake_download_from_kaggle() -> tuple[str, str]:
        return "/tmp/results.csv", "owner/dataset"

    monkeypatch.setattr(
        helpers,
        "download_historical_results_from_kaggle",
        fake_download_from_kaggle,
    )

    # Act
    path, dataset = helpers.download_historical_results("kaggle")

    # Assert
    assert path == "/tmp/results.csv"
    assert dataset == "owner/dataset"

## negative test case for download_historical_results###
# must raise a ValueError with the correct error message
def test_download_historical_results_invalid_provider():
    with pytest.raises(ValueError) as exc:
        helpers.download_historical_results("other-provider")

    assert "Unsupported integration provider" in str(exc.value)

## positive test case for scrape_results (placeholder)###
def test_scrape_results_not_implemented():
    with pytest.raises(NotImplementedError) as exc:
        helpers.scrape_results("rugby365")

    assert "scrape_results() not implemented" in str(exc.value)

## positive test case for scrape_fixtures (placeholder)###
def test_scrape_fixtures_not_implemented():
    with pytest.raises(NotImplementedError) as exc:
        helpers.scrape_fixtures("rugby365")

    assert "scrape_fixtures() not implemented" in str(exc.value)


