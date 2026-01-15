import json
import pytest
import pandas as pd
from functions.notifications.notification_services_helpers import (
    parse_payload,
    parse_recipients,
    build_email_bodies,
)


class TestParsePayload:
    """Test cases for parse_payload function"""

    def test_parse_payload_with_dict(self):
        """Test that a dictionary payload is returned unchanged"""
        payload_dict = {"date": "2026-01-15", "predictions": {}}
        result = parse_payload(payload_dict)
        assert result == payload_dict
        assert result is payload_dict  # Should be the same object

    def test_parse_payload_with_json_string(self):
        """Test that a JSON string is correctly parsed to dict"""
        payload_dict = {"date": "2026-01-15", "predictions": {"table1": "data"}}
        payload_str = json.dumps(payload_dict)
        result = parse_payload(payload_str)
        assert result == payload_dict
        assert isinstance(result, dict)

    def test_parse_payload_with_empty_dict(self):
        """Test that an empty dictionary is handled correctly"""
        payload_dict = {}
        result = parse_payload(payload_dict)
        assert result == {}

    def test_parse_payload_with_invalid_json_string(self):
        """Test that invalid JSON string raises error"""
        invalid_json = '{"incomplete": json'
        with pytest.raises(json.JSONDecodeError):
            parse_payload(invalid_json)

    def test_parse_payload_with_invalid_type(self):
        """Test that unsupported types raise TypeError"""
        with pytest.raises(TypeError, match="Unsupported payload type"):
            parse_payload(123)

        with pytest.raises(TypeError, match="Unsupported payload type"):
            parse_payload([1, 2, 3])

        with pytest.raises(TypeError, match="Unsupported payload type"):
            parse_payload(None)


class TestParseRecipients:
    """Test cases for parse_recipients function"""

    def test_parse_recipients_single_email(self):
        """Test parsing a single email address"""
        result = parse_recipients("admin@example.com")
        assert result == ["admin@example.com"]

    def test_parse_recipients_comma_separated(self):
        """Test parsing multiple emails separated by commas"""
        result = parse_recipients("admin@example.com,user@example.com,test@example.com")
        assert result == ["admin@example.com", "user@example.com", "test@example.com"]

    def test_parse_recipients_semicolon_separated(self):
        """Test parsing multiple emails separated by semicolons"""
        result = parse_recipients("admin@example.com;user@example.com;test@example.com")
        assert result == ["admin@example.com", "user@example.com", "test@example.com"]

    def test_parse_recipients_mixed_separators(self):
        """Test parsing emails with mixed comma and semicolon separators"""
        result = parse_recipients("admin@example.com,user@example.com;test@example.com")
        assert result == ["admin@example.com", "user@example.com", "test@example.com"]

    def test_parse_recipients_with_whitespace(self):
        """Test that leading/trailing whitespace is stripped"""
        result = parse_recipients("  admin@example.com  ,  user@example.com  ;  test@example.com  ")
        assert result == ["admin@example.com", "user@example.com", "test@example.com"]

    def test_parse_recipients_empty_string(self):
        """Test that empty string returns empty list"""
        result = parse_recipients("")
        assert result == []

    def test_parse_recipients_none(self):
        """Test that None returns empty list"""
        result = parse_recipients(None)
        assert result == []

    def test_parse_recipients_only_separators(self):
        """Test that string with only separators returns empty list"""
        result = parse_recipients(";;;,,,")
        assert result == []

    def test_parse_recipients_with_empty_elements(self):
        """Test that empty elements between separators are filtered out"""
        result = parse_recipients("admin@example.com,,user@example.com;;test@example.com")
        assert result == ["admin@example.com", "user@example.com", "test@example.com"]


class TestBuildEmailBodies:
    """Test cases for build_email_bodies function"""

    def test_build_email_bodies_with_predictions(self):
        """Test building email with valid predictions data"""
        df = pd.DataFrame({
            "competition": ["NRL"],
            "home_team": ["Sydney Roosters"],
            "away_team": ["Penrith Panthers"],
            "predicted_winner": ["Sydney Roosters"],
            "home_team_win_prob_pct": ["65.0%"],
            "predicted_margin": ["10.5 pts"],
        })
        
        payload = {
            "date": "2026-01-15",
            "predictions": {"nrl_predictions": df}
        }
        
        subject, text_body, html_body, any_rows = build_email_bodies(payload)
        
        assert "Stevens Rugby Predictions" in subject
        assert "2026-01-15" in subject
        assert any_rows is True
        assert "Sydney Roosters" in text_body
        assert "Penrith Panthers" in text_body
        assert "nrl_predictions" in text_body
        assert "<html>" in html_body
        assert "</html>" in html_body
        assert "Sydney Roosters" in html_body

    def test_build_email_bodies_no_date(self):
        """Test building email when date is missing"""
        df = pd.DataFrame({
            "competition": ["NRL"],
            "home_team": ["Sydney Roosters"],
            "away_team": ["Penrith Panthers"],
            "predicted_winner": ["Sydney Roosters"],
            "home_team_win_prob_pct": ["65.0%"],
            "predicted_margin": ["10.5 pts"],
        })
        
        payload = {
            "date": "",
            "predictions": {"nrl_predictions": df}
        }
        
        subject, text_body, html_body, any_rows = build_email_bodies(payload)
        
        assert "Stevens Rugby Predictions: daily predictions" in subject
        assert "for " not in subject  # Should not have date
        assert any_rows is True

    def test_build_email_bodies_empty_predictions(self):
        """Test building email with no prediction data"""
        payload = {
            "date": "2026-01-15",
            "predictions": {}
        }
        
        subject, text_body, html_body, any_rows = build_email_bodies(payload)
        
        assert "Stevens Rugby Predictions" in subject
        assert any_rows is False
        assert "No predictions were found for today" in text_body
        assert "No predictions were found for today" in html_body

    def test_build_email_bodies_with_none_dataframe(self):
        """Test building email when dataframe is None"""
        payload = {
            "date": "2026-01-15",
            "predictions": {"nrl_predictions": None}
        }
        
        subject, text_body, html_body, any_rows = build_email_bodies(payload)
        
        assert any_rows is False
        assert "No predictions were found for today" in text_body

    def test_build_email_bodies_with_empty_dataframe(self):
        """Test building email when dataframe is empty"""
        df = pd.DataFrame(columns=["competition", "home_team", "away_team"])
        
        payload = {
            "date": "2026-01-15",
            "predictions": {"nrl_predictions": df}
        }
        
        subject, text_body, html_body, any_rows = build_email_bodies(payload)
        
        assert any_rows is False
        assert "No predictions were found for today" in text_body

    def test_build_email_bodies_multiple_tables(self):
        """Test building email with multiple prediction tables"""
        df1 = pd.DataFrame({
            "competition": ["NRL"],
            "home_team": ["Sydney Roosters"],
            "away_team": ["Penrith Panthers"],
            "predicted_winner": ["Sydney Roosters"],
            "home_team_win_prob_pct": ["65.0%"],
            "predicted_margin": ["10.5 pts"],
        })
        
        df2 = pd.DataFrame({
            "competition": ["Super Rugby"],
            "home_team": ["Crusaders"],
            "away_team": ["Blues"],
            "predicted_winner": ["Crusaders"],
            "home_team_win_prob_pct": ["72.0%"],
            "predicted_margin": ["15.2 pts"],
        })
        
        payload = {
            "date": "2026-01-15",
            "predictions": {
                "nrl_predictions": df1,
                "super_rugby_predictions": df2
            }
        }
        
        subject, text_body, html_body, any_rows = build_email_bodies(payload)
        
        assert any_rows is True
        assert "nrl_predictions" in text_body
        assert "super_rugby_predictions" in text_body
        assert "Sydney Roosters" in text_body
        assert "Crusaders" in text_body

    def test_build_email_bodies_with_extra_columns(self):
        """Test that only preferred columns are displayed"""
        df = pd.DataFrame({
            "competition": ["NRL"],
            "home_team": ["Sydney Roosters"],
            "away_team": ["Penrith Panthers"],
            "predicted_winner": ["Sydney Roosters"],
            "home_team_win_prob_pct": ["65.0%"],
            "predicted_margin": ["10.5 pts"],
            "extra_col": ["should_not_appear"],
            "another_col": ["also_hidden"],
        })
        
        payload = {
            "date": "2026-01-15",
            "predictions": {"nrl_predictions": df}
        }
        
        subject, text_body, html_body, any_rows = build_email_bodies(payload)
        
        assert "extra_col" not in text_body
        assert "another_col" not in text_body
        assert "should_not_appear" not in text_body

    def test_build_email_bodies_missing_preferred_columns(self):
        """Test that function works with dataframe missing some preferred columns"""
        df = pd.DataFrame({
            "competition": ["NRL"],
            "home_team": ["Sydney Roosters"],
            "away_team": ["Penrith Panthers"],
        })
        
        payload = {
            "date": "2026-01-15",
            "predictions": {"nrl_predictions": df}
        }
        
        subject, text_body, html_body, any_rows = build_email_bodies(payload)
        
        assert any_rows is True
        assert "Sydney Roosters" in text_body
        # Should still generate valid output with available columns
        assert "nrl_predictions" in text_body

    def test_build_email_bodies_html_structure(self):
        """Test that HTML output has proper structure"""
        df = pd.DataFrame({
            "competition": ["NRL"],
            "home_team": ["Sydney Roosters"],
            "away_team": ["Penrith Panthers"],
            "predicted_winner": ["Sydney Roosters"],
            "home_team_win_prob_pct": ["65.0%"],
            "predicted_margin": ["10.5 pts"],
        })
        
        payload = {
            "date": "2026-01-15",
            "predictions": {"nrl_predictions": df}
        }
        
        subject, text_body, html_body, any_rows = build_email_bodies(payload)
        
        assert html_body.startswith("<html>")
        assert html_body.endswith("</html>")
        assert "<h2>Daily predictions for 2026-01-15</h2>" in html_body
        assert "<h3>nrl_predictions</h3>" in html_body
        assert "<body>" in html_body
        assert "</body>" in html_body

    def test_build_email_bodies_text_structure(self):
        """Test that text output has proper structure"""
        df = pd.DataFrame({
            "competition": ["NRL"],
            "home_team": ["Sydney Roosters"],
            "away_team": ["Penrith Panthers"],
            "predicted_winner": ["Sydney Roosters"],
            "home_team_win_prob_pct": ["65.0%"],
            "predicted_margin": ["10.5 pts"],
        })
        
        payload = {
            "date": "2026-01-15",
            "predictions": {"nrl_predictions": df}
        }
        
        subject, text_body, html_body, any_rows = build_email_bodies(payload)
        
        assert "Daily predictions for 2026-01-15" in text_body
        assert "== nrl_predictions ==" in text_body
        assert "Sydney Roosters" in text_body

    def test_build_email_bodies_all_none_predictions(self):
        """Test building email when all predictions are None or empty"""
        payload = {
            "date": "2026-01-15",
            "predictions": {
                "table1": None,
                "table2": pd.DataFrame(columns=["col1", "col2"]),
                "table3": None,
            }
        }
        
        subject, text_body, html_body, any_rows = build_email_bodies(payload)
        
        assert any_rows is False
        assert "No predictions were found for today" in text_body

    def test_build_email_bodies_missing_predictions_key(self):
        """Test building email when predictions key is missing from payload"""
        payload = {
            "date": "2026-01-15"
        }
        
        subject, text_body, html_body, any_rows = build_email_bodies(payload)
        
        assert any_rows is False
        assert "No predictions were found for today" in text_body

    def test_build_email_bodies_missing_date_key(self):
        """Test building email when date key is missing from payload"""
        df = pd.DataFrame({
            "competition": ["NRL"],
            "home_team": ["Sydney Roosters"],
            "away_team": ["Penrith Panthers"],
            "predicted_winner": ["Sydney Roosters"],
            "home_team_win_prob_pct": ["65.0%"],
            "predicted_margin": ["10.5 pts"],
        })
        
        payload = {
            "predictions": {"nrl_predictions": df}
        }
        
        subject, text_body, html_body, any_rows = build_email_bodies(payload)
        
        assert any_rows is True
        assert "daily predictions" in subject
        assert "<h2>Daily predictions</h2>" in html_body
