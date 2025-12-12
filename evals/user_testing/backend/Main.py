import os
import json
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from email_backup_resend import send_experiment_email_resend

# Load environment variables
load_dotenv()

app = FastAPI(title="Movie Recommender API")

# CORS middleware for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # Local development
        "http://localhost:3000",
    ],
    allow_origin_regex=r"https://.*\.vercel\.app",  # Vercel deployments
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "ok", "message": "Movie Recommender API - Email-only mode"}


@app.post("/api/experiment-session")
async def create_experiment_session(request: dict):
    """Acknowledge session creation - no database needed."""
    session_data = request.get('sessionData')

    if not session_data:
        raise HTTPException(status_code=400, detail="Missing session data")

    participant_id = session_data.get('participantId')

    # Just acknowledge - we'll save everything at the end via email
    return {
        'status': 'created',
        'participant_id': participant_id,
        'message': 'Session acknowledged - data will be saved on completion'
    }


@app.post("/api/track-interaction")
async def track_interaction(request: dict):
    """Acknowledge interaction tracking - no database needed."""
    # Just return success to not break frontend
    # All data will be saved when experiment completes
    return {'status': 'success', 'message': 'Interaction logged locally'}


@app.post("/api/save-prompt-response")
async def save_prompt_response(request: dict):
    """Acknowledge prompt response - no database needed."""
    # Just return success to not break frontend
    # All data will be saved when experiment completes
    return {'status': 'success', 'message': 'Response logged locally'}


@app.post("/api/experiment-results")
async def save_experiment_results(request: dict):
    """Save experiment results via email and JSON file."""
    session_data = request.get('sessionData')
    responses = request.get('responses', [])
    questionnaire = request.get('questionnaire', {})
    end_time = request.get('endTime')
    total_duration = request.get('totalDuration')

    if not session_data or not responses:
        raise HTTPException(status_code=400, detail="Missing session data or responses")

    participant_id = session_data.get('participantId')

    try:
        # Save as JSON backup file
        results_dir = 'data/experiment-results'
        os.makedirs(results_dir, exist_ok=True)

        result_data = {
            'sessionData': session_data,
            'responses': responses,
            'questionnaire': questionnaire,
            'endTime': end_time,
            'totalDuration': total_duration
        }

        filename = f"{results_dir}/{participant_id}_{end_time.replace(':', '-').replace('.', '-')}.json"
        with open(filename, 'w') as f:
            json.dump(result_data, f, indent=2)

        print(f"✅ Saved JSON backup: {filename}")

        # Send email backup via Resend API
        print(f"📧 Sending email for {participant_id}...")
        try:
            result = send_experiment_email_resend(participant_id, result_data)
            if result:
                print(f"✅ Email sent successfully for {participant_id}")
            else:
                print(f"⚠️ Email sending returned False for {participant_id}")
        except Exception as e:
            print(f"❌ Email sending failed for {participant_id}: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            # Don't fail the request if email fails - we have JSON backup

        return {
            'status': 'success',
            'message': f'Saved experiment results for participant {participant_id}',
            'responses_count': len(responses),
            'backup_file': filename
        }

    except Exception as e:
        print(f"Error saving results: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
