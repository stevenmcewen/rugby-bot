
from __future__ import annotations

"""
Helper functions for the preprocessing (bronze → silver) layer.

This module is intentionally kept for *fine-grained* helpers and
transformation utilities that are used by the higher‑level orchestration
defined in `preprocessing_services.py`.

At the moment there are no concrete helpers yet – they will live here as
you start implementing specific preprocessing pipelines, schema
validation utilities, small dataframe transforms, etc.
"""

def run_preprocessing_pipeline(preprocessing_event: PreprocessingEvent) -> None:
    """
    Run the preprocessing pipeline for a given preprocessing event.
    It reads the data from the blob, validates the schema, runs the preprocessing pipeline and updates the preprocessing event status.
    """
    preprocessing_event.status = "running"
    sql_client.update_preprocessing_event(preprocessing_event_id=preprocessing_event.id, status="running")
    # read the data from the blob
    pipeline = build_preprocessing_pipeline(preprocessing_event.pipeline_name)
    try:
        pipeline.run(preprocessing_event)
        preprocessing_event.status = "succeeded"
        preprocessing_event.error_message = None
        return 
    except Exception as e:
        preprocessing_event.status = "failed"
        preprocessing_event.error_message = str(e)
    return 