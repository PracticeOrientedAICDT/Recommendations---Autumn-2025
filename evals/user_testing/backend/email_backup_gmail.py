"""
BULLETPROOF Email backup using Gmail SMTP.
This is the most reliable solution - uses your own Gmail account.
Works everywhere including Render.
"""

import os
import json
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from io import StringIO
from datetime import datetime
from dotenv import load_dotenv

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

    session_data = data_dict.get('sessionData', {})
    output.write(f"Start Time,{session_data.get('startTime', '')}\n")
    output.write(f"End Time,{data_dict.get('endTime', '')}\n")
    output.write(f"Total Duration (seconds),{data_dict.get('totalDuration', '')}\n")

    # Device info
    device_info = session_data.get('deviceInfo', {})
    output.write(f"Device Type,{device_info.get('platform', '')}\n")
    output.write(f"Screen Size,{device_info.get('screenWidth')}x{device_info.get('screenHeight')}\n")
    output.write("\n")

    # Questionnaire responses
    output.write("=== QUESTIONNAIRE RESPONSES ===\n")
    questionnaire = data_dict.get('questionnaire', {})
    output.write(f"Age,{questionnaire.get('age', '')}\n")
    output.write(f"Gender,{questionnaire.get('gender', '')}\n")
    output.write(f"Movie Frequency,{questionnaire.get('movieFrequency', '')}\n")
    output.write(f"Genres Preferred,\"{', '.join(questionnaire.get('genresPreferred', []))}\"\n")
    output.write(f"Classic vs Recent,{questionnaire.get('classicVsRecent', '')}\n")
    output.write(f"Overall Relevance,{questionnaire.get('overallRelevance', '')}\n")
    output.write(f"Overall Diversity,{questionnaire.get('overallDiversity', '')}\n")
    output.write(f"Overall Novelty,{questionnaire.get('overallNovelty', '')}\n")
    output.write(f"Decision Difficulty,{questionnaire.get('decisionDifficulty', '')}\n")
    output.write(f"Familiarity Balance,{questionnaire.get('familiarityBalance', '')}\n")
    output.write(f"Prompt Based Search Interest,{questionnaire.get('promptBasedSearch', '')}\n")
    comments = questionnaire.get('comments', '').replace('"', '""')
    output.write(f'Comments,"{comments}"\n')
    output.write("\n")

    # Prompt responses
    output.write("=== PROMPT RESPONSES ===\n")
    responses = data_dict.get('responses', [])

    if responses:
        # CSV header
        output.write("Sequence,Prompt ID,Prompt Text,Category,")
        output.write("Model A,Model B,Model C,")
        output.write("Rank A,Rank B,Rank C,")
        output.write("1st Place Model,2nd Place Model,3rd Place Model,")
        output.write("Time on Prompt (s),Time on Slates (s)\n")

        # Write each response
        for resp in responses:
            model_mapping = resp.get('modelMapping', {})
            rankings = resp.get('rankings', {})
            ranked_models = resp.get('rankedModels', {})

            prompt_text = resp.get('promptText', '').replace('"', '""')
            output.write(f"{resp.get('sequencePosition', '')},")
            output.write(f"{resp.get('promptId', '')},")
            output.write(f'"{prompt_text}",')
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


def send_experiment_email_gmail(participant_id, data_dict):
    """
    Send experiment data via Gmail SMTP with CSV and JSON attachments.
    This is the most reliable method - uses your own Gmail account.

    Args:
        participant_id: Participant ID
        data_dict: Dictionary containing all experiment data

    Returns:
        bool: True if successful, False otherwise
    """
    # Get Gmail credentials from environment
    gmail_email = os.getenv('GMAIL_EMAIL', 'guodala@gmail.com')
    gmail_app_password = os.getenv('GMAIL_APP_PASSWORD')  # Gmail App Password (NOT regular password)
    recipient_email = os.getenv('BACKUP_EMAIL_RECIPIENT', 'guodala@gmail.com')

    if not gmail_app_password:
        print("❌ ERROR: GMAIL_APP_PASSWORD not configured!")
        print("   You need to generate a Gmail App Password at:")
        print("   https://myaccount.google.com/apppasswords")
        return False

    try:
        # Create email body
        end_time = data_dict.get('endTime', 'Unknown')
        total_duration = data_dict.get('totalDuration', 0)
        responses_count = len(data_dict.get('responses', []))

        email_body = f"""New participant completed the movie recommendation experiment!

Participant ID: {participant_id}
Completion Time: {end_time}
Duration: {total_duration:.1f} seconds ({total_duration/60:.1f} minutes)
Prompts Completed: {responses_count}

The complete data is attached as CSV and JSON files.

---
Automated email from Movie Experiment Backend
"""

        # Create multipart message
        msg = MIMEMultipart()
        msg['From'] = gmail_email
        msg['To'] = recipient_email
        msg['Subject'] = f"Movie Experiment Data - Participant {participant_id}"

        # Attach body
        msg.attach(MIMEText(email_body, 'plain'))

        # Create and attach CSV file
        csv_content = format_experiment_data_to_csv(participant_id, data_dict)
        csv_filename = f"{participant_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

        csv_attachment = MIMEBase('text', 'csv')
        csv_attachment.set_payload(csv_content.encode('utf-8'))
        encoders.encode_base64(csv_attachment)
        csv_attachment.add_header('Content-Disposition', f'attachment; filename={csv_filename}')
        msg.attach(csv_attachment)

        # Create and attach JSON file
        json_content = json.dumps(data_dict, indent=2)
        json_filename = f"{participant_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        json_attachment = MIMEBase('application', 'json')
        json_attachment.set_payload(json_content.encode('utf-8'))
        encoders.encode_base64(json_attachment)
        json_attachment.add_header('Content-Disposition', f'attachment; filename={json_filename}')
        msg.attach(json_attachment)

        # Connect to Gmail SMTP server and send
        print(f"📧 Sending email to {recipient_email}...")

        # Try port 465 with SSL (some cloud providers block port 587)
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(gmail_email, gmail_app_password)
            server.send_message(msg)

        print(f"✅ Successfully emailed experiment data for {participant_id} to {recipient_email}")
        print(f"   CSV file: {csv_filename}")
        print(f"   JSON file: {json_filename}")
        return True

    except smtplib.SMTPAuthenticationError as e:
        print(f"❌ Gmail authentication failed: {e}")
        print("   Make sure you're using a Gmail App Password, not your regular password.")
        print("   Generate one at: https://myaccount.google.com/apppasswords")
        return False

    except Exception as e:
        print(f"❌ Error sending email: {e}")
        return False
