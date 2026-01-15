from __future__ import annotations

import datetime

from azure.communication.email import EmailClient
from functions.config.settings import get_settings
from functions.logging.logger import get_logger
from functions.sql.sql_client import SqlClient
from functions.notifications.notification_services_helpers import parse_payload, parse_recipients, build_email_bodies

logger = get_logger(__name__)
settings = get_settings()

def get_daily_predictions(sql_client: SqlClient) -> dict:
    """
    Prepare a summary of daily model prediction scores.
    """
    logger.info("Generating daily predictions summary for notifications.")

    # 1) get list of fixture tables to query
    table_details = sql_client.get_model_scored_table_details()
    # 2) query each table for today's predictions
    results = {}
    today_utc = datetime.datetime.now(datetime.timezone.utc).date()
    prediction_date = today_utc.isoformat()

    for detail in table_details:
        # get table and column details from metadata
        table_name = detail["table_name"]
        try:
            competition_col = detail["competition_col"]
            home_team_col = detail["home_team_col"]
            away_team_col = detail["away_team_col"]
            home_team_win_prob_col = detail["home_team_win_prob_col"]
            point_diff_col = detail["point_diff_col"]

            start = f"{today_utc} 00:00:00"
            end = f"{today_utc + datetime.timedelta(days=1)} 00:00:00"

            # filter for all games scored today
            date_filter = f"""
            ScoredAtUtc >= '{start}'
            AND ScoredAtUtc < '{end}'
            """

            # get the data from the table
            df = sql_client.read_table_to_dataframe(
                table_name=table_name,
                columns=[
                    competition_col,
                    home_team_col,
                    away_team_col,
                    home_team_win_prob_col,
                    point_diff_col
                ],
                where_sql=date_filter,
            )

            # rename columns to standard names
            df_renamed = df.rename(columns={
                competition_col: "competition",
                home_team_col: "home_team",
                away_team_col: "away_team",
                home_team_win_prob_col: "home_team_win_prob",
                point_diff_col: "point_diff"
            })

            # add readibility columns to dataframe
            # predicted winner
            df_renamed["predicted_winner"] = df_renamed.apply(
                lambda row: row["home_team"] if row["home_team_win_prob"] >= 0.5 else row["away_team"],
                axis=1
            )
            # format win probability as percentage
            df_renamed["home_team_win_prob_pct"] = df_renamed["home_team_win_prob"].apply(lambda x: f"{x:.1%}")

            # winning margin
            df_renamed["predicted_margin"] = df_renamed["point_diff"].abs().apply(lambda x: f"{x:.1f} pts")
            results[table_name] = df_renamed
        except Exception as e:
            logger.error("Error getting predictions for %s: %s", table_name, e)
            raise

    # 3) compile results into a summary structure
    payload = {
        "date": prediction_date,
        "predictions": results
    }

    return payload


def send_prediction_email(payload: dict | str) -> None:
    """
    Send an email containing fixtures and predictions.
    If there were no scoring predictions for the day, dont send any email.
    """
    # 1) parse the payload and extract all the email content
    logger.info("Parsing Email information from dataframe scoring results search payload")
    payload_dict = parse_payload(payload)
    subject, text_body, html_body, any_rows = build_email_bodies(payload_dict)

    # 2) Only do email processing if we have data to send via email
    if not any_rows:
        logger.info("There were no predictions today so no email will be sent")
        return

    # 3) Get email routing information
    from_address = settings.email_from
    to_addresses = parse_recipients(settings.email_to)
    conn_str = settings.acs_email_connection_string

    if not from_address or not to_addresses:
        logger.warning("Email not sent: missing EMAIL_FROM and/or EMAIL_TO configuration.")
        return
    if not conn_str:
        logger.warning("Email not sent: missing ACS_EMAIL_CONNECTION_STRING configuration.")
        return

    logger.info(
        "Sending prediction email from=%r to=%r",
        from_address,
        to_addresses,
    )

    # 4) Build ACS message payload
    message = {
        "senderAddress": from_address,
        "recipients": {"to": [{"address": addr} for addr in to_addresses]},
        "content": {
            "subject": subject,
            "plainText": text_body,
            "html": html_body,
        },
    }

    #  5) Send via ACS SDK
    try:
        client = EmailClient.from_connection_string(conn_str)
        poller = client.begin_send(message)
        result = poller.result()

        # result is a dictionary with 'id' key containing the message_id
        message_id = result.get("id") if isinstance(result, dict) else getattr(result, "id", None)
        logger.info("Prediction email sent via ACS. message_id=%s", message_id)

    except Exception as exc:
        logger.exception("Failed to send prediction email via ACS: %s", exc)
        raise



