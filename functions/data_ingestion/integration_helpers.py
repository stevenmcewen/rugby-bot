import os
import kagglehub

### provider_helpers ###
def download_historical_results_from_kaggle() -> tuple[str, str]:
    """
    Download historical results from Kaggle.
    
    Returns:
        local_file_path: The path to the downloaded file.
        integration_dataset: The dataset that was downloaded.
    """
    try:
        integration_dataset = "lylebegbie/international-rugby-union-results-from-18712022"
        local_directory = kagglehub.dataset_download(integration_dataset)
        local_file_path = os.path.join(local_directory, "results.csv")
        return local_file_path, integration_dataset
    except Exception as e:
        raise ValueError(f"Error downloading historical results from Kaggle: {e!r}")

## main_functions ###
def download_historical_results(integration_provider: str) -> tuple[str, str]:
    """
    Download historical results from the given integration provider.

    Accepts:
        integration_provider: The provider of the integration (e.g. "kaggle", "rugby365").

    Returns:
        local_file_path: The path to the downloaded file.
        integration_dataset: The dataset that was downloaded.
    """
    if integration_provider != "kaggle":
        raise ValueError(f"Unsupported integration provider: {integration_provider!r}")
    
    local_file_path, integration_dataset = download_historical_results_from_kaggle()
    return local_file_path, integration_dataset

def scrape_results(integration_provider: str) -> tuple[str, str]:
    """
    Placeholder function for scraping results.

    For now this just raises a NotImplementedError so that you remember to
    implement the concrete scraping logic per provider.
    """
    raise NotImplementedError(f"scrape_results() not implemented for provider={integration_provider!r}")


def scrape_fixtures(integration_provider: str) -> tuple[str, str]:
    """
    Placeholder function for scraping fixtures.

    For now this just raises a NotImplementedError so that you remember to
    implement the concrete scraping logic per provider.
    """
    raise NotImplementedError(f"scrape_fixtures() not implemented for provider={integration_provider!r}")