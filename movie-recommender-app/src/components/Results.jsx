import { useState, useEffect } from 'react'
import './Results.css'
import { API_URL } from '../config'

function Results() {
  const [isAuthenticated, setIsAuthenticated] = useState(false)
  const [password, setPassword] = useState('')
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)

  const [summary, setSummary] = useState(null)
  const [participants, setParticipants] = useState([])
  const [selectedParticipant, setSelectedParticipant] = useState(null)
  const [activeTab, setActiveTab] = useState('summary') // 'summary' or 'participants'

  const handleLogin = async (e) => {
    e.preventDefault()
    setError('')
    setLoading(true)

    try {
      const response = await fetch(`${API_URL}/api/admin/verify-password`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ password })
      })

      if (response.ok) {
        setIsAuthenticated(true)
        // Keep password in state for this session only (not persisted)
        loadData(password)
      } else {
        const errorData = await response.json()
        setError(errorData.detail || 'Invalid password')
      }
    } catch (err) {
      setError('Failed to verify password')
    } finally {
      setLoading(false)
    }
  }

  const loadData = async (pwd) => {
    if (!pwd) return

    try {
      // Load summary with Authorization header
      const summaryResponse = await fetch(`${API_URL}/api/admin/results-summary`, {
        headers: {
          'Authorization': `Bearer ${pwd}`
        }
      })
      if (summaryResponse.ok) {
        const summaryData = await summaryResponse.json()
        setSummary(summaryData)
      } else if (summaryResponse.status === 429) {
        const error = await summaryResponse.json()
        alert(error.detail)
      }

      // Load participants with Authorization header
      const participantsResponse = await fetch(`${API_URL}/api/admin/participants`, {
        headers: {
          'Authorization': `Bearer ${pwd}`
        }
      })
      if (participantsResponse.ok) {
        const participantsData = await participantsResponse.json()
        setParticipants(participantsData.participants)
      }
    } catch (err) {
      console.error('Failed to load data:', err)
    }
  }

  const handleDeleteParticipant = async (participantId) => {
    if (!confirm(`Are you sure you want to delete participant ${participantId}? This cannot be undone.`)) {
      return
    }

    if (!password) return

    try {
      const response = await fetch(`${API_URL}/api/admin/delete-participant?participantId=${participantId}`, {
        method: 'DELETE',
        headers: {
          'Authorization': `Bearer ${password}`
        }
      })

      if (response.ok) {
        // Refresh data
        await loadData(password)
        setSelectedParticipant(null)
        alert('Participant deleted successfully')
      } else if (response.status === 429) {
        const error = await response.json()
        alert(error.detail)
      } else {
        alert('Failed to delete participant')
      }
    } catch (err) {
      console.error('Error deleting participant:', err)
      alert('Error deleting participant')
    }
  }

  const exportToCSV = () => {
    if (participants.length === 0) return

    // Create CSV header
    const headers = [
      'Participant ID',
      'Start Time',
      'End Time',
      'Total Duration (s)',
      'Completed',
      'Device Type',
      'IP Address',
      'Age',
      'Gender',
      'Movie Frequency',
      'Genres Preferred',
      'Classic vs Recent',
      'Overall Relevance',
      'Overall Diversity',
      'Overall Novelty',
      'Decision Difficulty',
      'Familiarity Balance',
      'Prompt-Based Search Interest',
      'Comments',
      'Prompt Sequence',
      'Prompt Text',
      'Model A',
      'Model B',
      'Model C',
      'Ranking A',
      'Ranking B',
      'Ranking C',
      'Winner',
      'Time on Prompt',
      'Time on Slates'
    ]

    // Create CSV rows
    const rows = []
    participants.forEach(p => {
      if (p.responses && p.responses.length > 0) {
        p.responses.forEach(r => {
          rows.push([
            p.participantId,
            p.startTime,
            p.endTime,
            p.totalDuration,
            p.completed,
            p.deviceInfo?.deviceType || '',
            p.deviceInfo?.ipAddress || '',
            p.questionnaire?.age || '',
            p.questionnaire?.gender || '',
            p.questionnaire?.movieFrequency || '',
            (p.questionnaire?.genresPreferred || []).join(';'),
            p.questionnaire?.classicVsRecent || '',
            p.questionnaire?.overallRelevance || '',
            p.questionnaire?.overallDiversity || '',
            p.questionnaire?.overallNovelty || '',
            p.questionnaire?.decisionDifficulty || '',
            p.questionnaire?.familiarityBalance || '',
            p.questionnaire?.promptBasedSearch || '',
            (p.questionnaire?.comments || '').replace(/"/g, '""'),
            r.sequencePosition,
            (r.promptText || '').replace(/"/g, '""'),
            r.modelMapping?.A || '',
            r.modelMapping?.B || '',
            r.modelMapping?.C || '',
            r.rankings?.A || '',
            r.rankings?.B || '',
            r.rankings?.C || '',
            r.rankedModels?.first || '',
            r.timeOnPrompt,
            r.timeOnSlates
          ])
        })
      }
    })

    // Create CSV content
    const csvContent = [
      headers.map(h => `"${h}"`).join(','),
      ...rows.map(row => row.map(cell => `"${cell}"`).join(','))
    ].join('\n')

    // Download
    const blob = new Blob([csvContent], { type: 'text/csv' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `experiment-results-${new Date().toISOString().split('T')[0]}.csv`
    a.click()
    URL.revokeObjectURL(url)
  }

  // Removed auto-login - require password every time

  // Login screen
  if (!isAuthenticated) {
    return (
      <div className="results-login">
        <div className="login-box">
          <h1>Results Dashboard</h1>
          <p>Enter admin password to view experiment results</p>

          <form onSubmit={handleLogin}>
            <input
              type="password"
              placeholder="Admin password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              className="password-input"
              autoFocus
            />
            {error && <p className="error-message">{error}</p>}
            <button type="submit" disabled={loading} className="login-button">
              {loading ? 'Verifying...' : 'Access Results'}
            </button>
          </form>
        </div>
      </div>
    )
  }

  const goToHome = () => {
    window.location.href = '/'
  }

  // Main dashboard
  return (
    <div className="results-dashboard">
      <div className="dashboard-header">
        <h1>Experiment Results Dashboard</h1>
        <div className="header-buttons">
          <button onClick={goToHome} className="home-button">
            ← Back to Home
          </button>
          <button onClick={exportToCSV} className="export-button">
            Export to CSV
          </button>
        </div>
      </div>

      <div className="dashboard-tabs">
        <button
          className={`tab ${activeTab === 'summary' ? 'active' : ''}`}
          onClick={() => setActiveTab('summary')}
        >
          Summary
        </button>
        <button
          className={`tab ${activeTab === 'participants' ? 'active' : ''}`}
          onClick={() => setActiveTab('participants')}
        >
          Participants
        </button>
      </div>

      {activeTab === 'summary' && (
        <div className="summary-section">
          {!summary || (summary.totalParticipants === 0 && summary.totalResponses === 0) ? (
            <div className="no-data-message">
              <h2>No data collected yet</h2>
              <p>Total Participants: {summary?.totalParticipants || 0}</p>
              <p>Run the experiment and collect some data to see full statistics.</p>
            </div>
          ) : (
            <>
          <div className="stats-grid">
            <div className="stat-card">
              <h3>Total Participants</h3>
              <p className="stat-number">{summary.totalParticipants}</p>
            </div>
            <div className="stat-card">
              <h3>Completed</h3>
              <p className="stat-number">{summary.completedParticipants}</p>
            </div>
            <div className="stat-card">
              <h3>Total Responses</h3>
              <p className="stat-number">{summary.totalResponses}</p>
            </div>
            <div className="stat-card">
              <h3>Total Interactions</h3>
              <p className="stat-number">{summary.totalInteractions}</p>
            </div>
          </div>

          <div className="model-performance">
            <h2>Model Performance</h2>
            <div className="model-stats">
              <div className="model-card">
                <h4>GPT-4o-mini</h4>
                <p className="wins">🥇 {summary.modelWins.gpt} wins</p>
                <p className="total">Total appearances: {summary.modelTotal.gpt}</p>
                <p className="win-rate">
                  Win rate: {summary.modelTotal.gpt > 0
                    ? ((summary.modelWins.gpt / summary.modelTotal.gpt) * 100).toFixed(1)
                    : 0}%
                </p>
              </div>
              <div className="model-card">
                <h4>Claude 3 Haiku</h4>
                <p className="wins">🥇 {summary.modelWins.claude} wins</p>
                <p className="total">Total appearances: {summary.modelTotal.claude}</p>
                <p className="win-rate">
                  Win rate: {summary.modelTotal.claude > 0
                    ? ((summary.modelWins.claude / summary.modelTotal.claude) * 100).toFixed(1)
                    : 0}%
                </p>
              </div>
              <div className="model-card">
                <h4>Diffusion Baseline</h4>
                <p className="wins">🥇 {summary.modelWins.diffusion} wins</p>
                <p className="total">Total appearances: {summary.modelTotal.diffusion}</p>
                <p className="win-rate">
                  Win rate: {summary.modelTotal.diffusion > 0
                    ? ((summary.modelWins.diffusion / summary.modelTotal.diffusion) * 100).toFixed(1)
                    : 0}%
                </p>
              </div>
            </div>
          </div>

          <div className="timing-stats">
            <h2>Average Timing</h2>
            <div className="timing-grid">
              <div className="timing-card">
                <h4>Time on Prompt</h4>
                <p className="time-value">{summary.avgTimeOnPrompt}s</p>
              </div>
              <div className="timing-card">
                <h4>Time on Slates</h4>
                <p className="time-value">{summary.avgTimeOnSlates}s</p>
              </div>
            </div>
          </div>
            </>
          )}
        </div>
      )}

      {activeTab === 'participants' && (
        <div className="participants-section">
          <div className="participants-list">
            <h2>Participants ({participants.length})</h2>
            {participants.map((p) => (
              <div
                key={p.participantId}
                className={`participant-item ${selectedParticipant?.participantId === p.participantId ? 'selected' : ''}`}
                onClick={() => setSelectedParticipant(p)}
              >
                <div className="participant-id">{p.participantId}</div>
                <div className="participant-meta">
                  {p.completed ? '✅ Completed' : '⏳ In Progress'}
                  {p.totalDuration && ` • ${Math.round(p.totalDuration / 60)} min`}
                </div>
              </div>
            ))}
          </div>

          {selectedParticipant && (
            <div className="participant-details">
              <h2>Participant Details</h2>

              <div className="details-section">
                <h3>Session Info</h3>
                <p><strong>ID:</strong> {selectedParticipant.participantId}</p>
                <p><strong>Start:</strong> {new Date(selectedParticipant.startTime).toLocaleString()}</p>
                {selectedParticipant.endTime && (
                  <p><strong>End:</strong> {new Date(selectedParticipant.endTime).toLocaleString()}</p>
                )}
                {selectedParticipant.totalDuration && (
                  <p><strong>Duration:</strong> {Math.round(selectedParticipant.totalDuration / 60)} minutes</p>
                )}
                {selectedParticipant.deviceInfo?.deviceType && (
                  <p><strong>Device Type:</strong> {selectedParticipant.deviceInfo.deviceType}</p>
                )}
                {selectedParticipant.deviceInfo?.ipAddress && (
                  <p><strong>IP Address:</strong> {selectedParticipant.deviceInfo.ipAddress}</p>
                )}
                <button
                  className="delete-participant-button"
                  onClick={() => handleDeleteParticipant(selectedParticipant.participantId)}
                >
                  Delete This Participant
                </button>
              </div>

              <div className="details-section">
                <h3>Questionnaire Responses</h3>
                <p><strong>Age:</strong> {selectedParticipant.questionnaire?.age}</p>
                <p><strong>Gender:</strong> {selectedParticipant.questionnaire?.gender}</p>
                <p><strong>Movie Frequency:</strong> {selectedParticipant.questionnaire?.movieFrequency}</p>
                <p><strong>Preferred Genres:</strong> {(selectedParticipant.questionnaire?.genresPreferred || []).join(', ')}</p>
                <p><strong>Classic vs Recent:</strong> {selectedParticipant.questionnaire?.classicVsRecent}/5</p>
                <p><strong>Overall Relevance:</strong> {selectedParticipant.questionnaire?.overallRelevance}/5</p>
                <p><strong>Overall Diversity:</strong> {selectedParticipant.questionnaire?.overallDiversity}/5</p>
                <p><strong>Overall Novelty:</strong> {selectedParticipant.questionnaire?.overallNovelty}/5</p>
                <p><strong>Decision Difficulty:</strong> {selectedParticipant.questionnaire?.decisionDifficulty}/5</p>
                <p><strong>Familiarity Balance:</strong> {selectedParticipant.questionnaire?.familiarityBalance}/5</p>
                {selectedParticipant.questionnaire?.comments && (
                  <div className="comments-box">
                    <strong>Comments:</strong>
                    <p>{selectedParticipant.questionnaire.comments}</p>
                  </div>
                )}
              </div>

              <div className="details-section">
                <h3>Prompt Responses ({selectedParticipant.responses?.length || 0})</h3>
                {selectedParticipant.responses?.map((resp, idx) => {
                  // Find which slate got which star rating
                  const slate3Stars = Object.keys(resp.rankings || {}).find(k => resp.rankings[k] === 3)
                  const slate2Stars = Object.keys(resp.rankings || {}).find(k => resp.rankings[k] === 2)
                  const slate1Star = Object.keys(resp.rankings || {}).find(k => resp.rankings[k] === 1)

                  return (
                    <div key={idx} className="response-card">
                      <p><strong>#{resp.sequencePosition}:</strong> {resp.promptText}</p>
                      <p><strong>Model Assignments:</strong> A={resp.modelMapping?.A}, B={resp.modelMapping?.B}, C={resp.modelMapping?.C}</p>
                      <p><strong>Rankings (best to worst):</strong></p>
                      <p className="ranking-line">
                        🥇 {resp.rankedModels?.first} (Slate {slate3Stars}, 3★) →
                        🥈 {resp.rankedModels?.second} (Slate {slate2Stars}, 2★) →
                        🥉 {resp.rankedModels?.third} (Slate {slate1Star}, 1★)
                      </p>
                      <p><strong>Time:</strong> {resp.timeOnPrompt?.toFixed(1)}s on prompt, {resp.timeOnSlates?.toFixed(1)}s on slates</p>
                    </div>
                  )
                })}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

export default Results
