"""
Email backup helper using SendGrid API (works on Render - doesn't use SMTP).
Sends CSV + JSON attachments immediately after participant completion.
"""

import base64
import json
import os
from datetime import datetime
from io import StringIO

from dotenv import load_dotenv
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import (
    Attachment,
    Disposition,
    FileContent,
    FileName,
    FileType,
    Mail,
)

load_dotenv()


def format_experiment_data_to_csv(participant_id, data_dict):
    """
    Convert experiment data dictionary to CSV format.

    Args:
        participant_id: Participant ID
        data_dict: Dictionary containing all experiment data

    Returns:
        str: CSV content as string
    """
    output = StringIO()

    # Write participant info header
    output.write("=== PARTICIPANT INFO ===\n")
    output.write(f"Participant ID,{participant_id}\n")

    session_data = data_dict.get("sessionData", {})
    output.write(f"Start Time,{session_data.get('startTime', '')}\n")
    output.write(f"End Time,{data_dict.get('endTime', '')}\n")
    output.write(f"Total Duration (seconds),{data_dict.get('totalDuration', '')}\n")

    # Device info
    device_info = session_data.get("deviceInfo", {})
    output.write(f"Device Type,{device_info.get('platform', '')}\n")
    output.write(
        f"Screen Size,{device_info.get('screenWidth')}x{device_info.get('screenHeight')}\n"
    )
    output.write("\n")

    # Questionnaire responses
    output.write("=== QUESTIONNAIRE RESPONSES ===\n")
    questionnaire = data_dict.get("questionnaire", {})

    genres_list = questionnaire.get("genresPreferred", [])
    genres_preferred = ", ".join(genres_list)

    comments_raw = questionnaire.get("comments", "")
    comments_escaped = comments_raw.replace('"', '""')

    output.write(f"Age,{questionnaire.get('age', '')}\n")
    output.write(f"Gender,{questionnaire.get('gender', '')}\n")
    output.write(f"Movie Frequency,{questionnaire.get('movieFrequency', '')}\n")
    output.write(f'Genres Preferred,"{genres_preferred}"\n')
    output.write(f"Classic vs Recent,{questionnaire.get('classicVsRecent', '')}\n")
    output.write(f"Overall Relevance,{questionnaire.get('overallRelevance', '')}\n")
    output.write(f"Overall Diversity,{questionnaire.get('overallDiversity', '')}\n")
    output.write(f"Overall Novelty,{questionnaire.get('overallNovelty', '')}\n")
    output.write(f"Decision Difficulty,{questionnaire.get('decisionDifficulty', '')}\n")
    output.write(f"Familiarity Balance,{questionnaire.get('familiarityBalance', '')}\n")
    output.write(
        f"Prompt Based Search Interest,{questionnaire.get('promptBasedSearch', '')}\n"
    )
    output.write(f'Comments,"{comments_escaped}"\n')
    output.write("\n")

    # Prompt responses
    output.write("=== PROMPT RESPONSES ===\n")
    responses = data_dict.get("responses", [])

    if responses:
        # CSV header
        output.write("Sequence,Prompt ID,Prompt Text,Category,")
        output.write("Model A,Model B,Model C,")
        output.write("Rank A,Rank B,Rank C,")
        output.write("1st Place Model,2nd Place Model,3rd Place Model,")
        output.write("Time on Prompt (s),Time on Slates (s)\n")

        # Write each response
        for resp in responses:
            model_mapping = resp.get("modelMapping", {})
            rankings = resp.get("rankings", {})
            ranked_models = resp.get("rankedModels", {})

            prompt_text = resp.get("promptText", "")
            prompt_text_escaped = prompt_text.replace('"', '""')

            output.write(f"{resp.get('sequencePosition', '')},")
            output.write(f"{resp.get('promptId', '')},")
            output.write(f"\"{prompt_text_escaped}\",")
            output.write(f"{resp.get('promptCategory', '')},")
            output.write(f"{model_mapping.get('A', '')},")
            output.write(f"{model_mapping.get('B', '')},")
            output.write(f"{model_mapping.get('C', '')},")
            output.write(f"{rankings.get('A', '')},")
            output.write(f"{rankings.get('B', '')},")
            output.write(f"{rankings.get('C', '')},")
            output.write(f"{ranked_models.get('first', '')},")
            output.write(f"{ranked_models.get('second', '')},")
            output.write(f"{ranked_models.get('third', '')},")
            output.write(f"{resp.get('timeOnPrompt', '')},")
            output.write(f"{resp.get('timeOnSlates', '')}\n")

    csv_content = output.getvalue()
    output.close()
    return csv_content


def send_experiment_email(participant_id, data_dict):
    """
    Send experiment data via SendGrid API with CSV and JSON attachments.

    Args:
        participant_id: Participant ID
        data_dict: Dictionary containing all experiment data

    Returns:
        bool: True if successful, False otherwise
    """
    sendgrid_api_key = os.getenv("SENDGRID_API_KEY")
    sender_email = os.getenv("SENDER_EMAIL", "noreply@movie-experiment.com")
    recipient_email = os.getenv("BACKUP_EMAIL_RECIPIENT", "guodala@gmail.com")

    if not sendgrid_api_key:
        print("Warning: SENDGRID_API_KEY not configured")
        return False

    try:
        end_time = data_dict.get("endTime", "Unknown")
        total_duration = data_dict.get("totalDuration", 0)
        responses_count = len(data_dict.get("responses", []))

        email_body = f"""
New participant completed the movie recommendation experiment!

Participant ID: {participant_id}
Completion Time: {end_time}
Duration: {total_duration:.1f} seconds ({total_duration / 60:.1f} minutes)
Prompts Completed: {responses_count}

The complete data is attached as CSV and JSON files.

---
Automated email from Movie Experiment Backend
"""

        message = Mail(
            from_email=sender_email,
            to_emails=recipient_email,
            subject=f"Movie Experiment Data - Participant {participant_id}",
            plain_text_content=email_body,
        )

        # CSV attachment
        csv_content = format_experiment_data_to_csv(participant_id, data_dict)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = f"{participant_id}_{timestamp}.csv"

        csv_encoded = base64.b64encode(csv_content.encode("utf-8")).decode("utf-8")
        csv_attachment = Attachment(
            FileContent(csv_encoded),
            FileName(csv_filename),
            FileType("text/csv"),
            Disposition("attachment"),
        )
        message.attachment = csv_attachment

        # JSON attachment
        json_content = json.dumps(data_dict, indent=2)
        json_filename = f"{participant_id}_{timestamp}.json"

        json_encoded = base64.b64encode(json_content.encode("utf-8")).decode("utf-8")
        json_attachment = Attachment(
            FileContent(json_encoded),
            FileName(json_filename),
            FileType("application/json"),
            Disposition("attachment"),
        )
        message.add_attachment(json_attachment)

        sg = SendGridAPIClient(sendgrid_api_key)
        response = sg.send(message)

        print(
            f"✓ Successfully emailed experiment data for {participant_id} "
            f"to {recipient_email}"
        )
        print(f"  SendGrid response: {response.status_code}")
        return True

    except Exception as e:
        print(f"Error sending email via SendGrid: {e}")
        return False
