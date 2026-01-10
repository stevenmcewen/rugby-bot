import json
import re

from functions.config.settings import get_settings
settings = get_settings()


def parse_payload(payload: dict | str) -> dict:
    """
    Parse a payload into a dictionary regardless of if the provided information is a string or a dictionary
    """
    if isinstance(payload, dict):
        return payload
    if isinstance(payload, str):
        return json.loads(payload)
    raise TypeError(f"Unsupported payload type: {type(payload)}")


def parse_recipients(value: str | None) -> list[str]:
    "create a list of all recipient addresses from a string value, seperating on , and ;"
    if not value:
        recipient_list = []
    parts = re.split(r"[;,]", value)
    # make sure that the strings dont have any leading or trailing whitespaces in the list 
    recipient_list =  [p.strip() for p in parts if p.strip()]
    return recipient_list


def build_email_bodies(payload_dict: dict) -> tuple[str, str, str, bool]:
    """
    Logic for building out email content
    Accepts:
            payload_dict: DIctionary with all the scores of predictions that occured today 
    Returns:
            tuple(
                subject(str): Subject of email,
                email_txt(str): All text of the email,
                html_parts(str): All email html that WIll allow the email to be written out in a readible form
                any_rows(bool): A check to see if there is any content in the email
                )
    """
    # Get two main inputs from the result payload, default if empty
    date = payload_dict.get("date") or ""
    predictions = payload_dict.get("predictions") or {}

    # Subject line can be altered in key vault
    subject_prefix = (settings.email_subject_prefix or "Rugby Bot").strip()
    subject = f"{subject_prefix}: daily predictions ({date})" if date else f"{subject_prefix}: daily predictions"

    # Build HTML
    html_parts: list[str] = [
        "<html><body>",
        f"<h2>Daily predictions{f' for {date}' if date else ''}</h2>",
    ]
    text_parts: list[str] = [
        f"Daily predictions{f' for {date}' if date else ''}",
        "",
    ]

    # initialize any_rows checker
    any_rows = False
    # Loop through all prediction tables and their results dataframes, if empty then continue to next one
    for table_name, df in predictions.items():
        if df is None or getattr(df, "empty", True):
            continue
        # If non-empty one sent any_rows to True
        any_rows = True

        html_parts.append(f"<h3>{table_name}</h3>")
        text_parts.append(f"== {table_name} ==")

        preferred_cols = [
            "competition",
            "home_team",
            "away_team",
            "predicted_winner",
            "home_team_win_prob_pct",
            "predicted_margin",
        ]
        cols = [c for c in preferred_cols if c in df.columns]
        df_view = df[cols] if cols else df

        # pandas DataFrame -> HTML table
        html_parts.append(
            df_view.to_html(index=False, border=0, justify="left")
        )
        text_parts.append(
            df_view.to_string(index=False) 
        )
        text_parts.append("")

    if not any_rows:
        html_parts.append("<p>No predictions were found for today.</p>")
        text_parts.append("No predictions were found for today.")

    html_parts.append("</body></html>")
    email_text = "\n".join(text_parts).strip()
    html_parts = "".join(html_parts)


    return subject, email_text, html_parts, any_rows