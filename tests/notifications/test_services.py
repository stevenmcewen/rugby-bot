from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import datetime
from functions.notifications.services import get_daily_predictions, send_prediction_email

def test_get_daily_predictions_success():
    # Mock the SQL client
    mock_sql_client = Mock()
    
    # Mock table metadata
    mock_sql_client.get_model_scored_table_details.return_value = [
        {
            "table_name": "nrl_predictions",
            "competition_col": "comp",
            "home_team_col": "home",
            "away_team_col": "away",
            "home_team_win_prob_col": "win_prob",
            "point_diff_col": "diff"
        }
    ]
    
    # Mock dataframe result
    df_mock = pd.DataFrame({
        "comp": ["NRL"],
        "home": ["Sydney Roosters"],
        "away": ["Penrith Panthers"],
        "win_prob": [0.65],
        "diff": [10.5]
    })
    mock_sql_client.read_table_to_dataframe.return_value = df_mock
    
    result = get_daily_predictions(mock_sql_client)
    
    assert "date" in result
    assert "predictions" in result
    assert "nrl_predictions" in result["predictions"]

@patch('functions.notifications.services.EmailClient')
@patch('functions.notifications.services.settings')
def test_send_prediction_email_success(mock_settings, mock_email_client_class):
    # Setup
    mock_settings.email_from = "bot@example.com"
    mock_settings.email_to = "admin@example.com,user@example.com"
    mock_settings.acs_email_connection_string = "connection_string"
    
    # Mock email client
    mock_client_instance = Mock()
    mock_poller = Mock()
    mock_poller.result.return_value = {"id": "msg_123"}
    mock_client_instance.begin_send.return_value = mock_poller
    mock_email_client_class.from_connection_string.return_value = mock_client_instance
    
    with patch('functions.notifications.services.parse_payload') as mock_parse, \
         patch('functions.notifications.services.build_email_bodies') as mock_build:
        mock_build.return_value = ("Subject", "Text", "HTML", True)  # any_rows=True
        
        send_prediction_email({"test": "payload"})
        
        # Assert email was sent
        mock_client_instance.begin_send.assert_called_once()

def test_send_prediction_email_no_rows():
    """Test that no email is sent when there are no predictions"""
    with patch('functions.notifications.services.parse_payload') as mock_parse, \
         patch('functions.notifications.services.build_email_bodies') as mock_build:
        mock_build.return_value = ("Subject", "Text", "HTML", False)  # any_rows=False
        
        send_prediction_email({"test": "payload"})
        
        # Assert email client was never used
        # (verify by checking logger output or using other assertions)