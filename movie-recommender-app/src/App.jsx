import { useState, useEffect } from 'react'
import './App.css'
import SearchBar from './components/SearchBar'
import MovieGrid from './components/MovieGrid'
import Logo from './components/Logo'
import LandingPage from './components/LandingPage'
import Experiment from './components/Experiment'
import Results from './components/Results'

function App() {
  const [currentView, setCurrentView] = useState('landing') // 'landing', 'main', 'experiment', 'results'

  // Check for /results route
  useEffect(() => {
    const path = window.location.pathname
    if (path === '/results') {
      setCurrentView('results')
    }
  }, [])
  const [loading, setLoading] = useState(false)
  const [gptRecommendations, setGptRecommendations] = useState([])
  const [claudeRecommendations, setClaudeRecommendations] = useState([])
  const [error, setError] = useState(null)

  const handleSearch = async (prompt) => {
    setLoading(true)
    setError(null)
    setGptRecommendations([])
    setClaudeRecommendations([])

    try {
      const response = await fetch('http://localhost:8000/api/recommendations', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ prompt }),
      })

      if (!response.ok) {
        throw new Error('Failed to fetch recommendations')
      }

      const data = await response.json()
      setGptRecommendations(data.gpt_recommendations)
      setClaudeRecommendations(data.claude_recommendations)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  // Handle different views
  if (currentView === 'landing') {
    return (
      <LandingPage
        onEnter={() => setCurrentView('main')}
        onExperiment={() => setCurrentView('experiment')}
        onResults={() => setCurrentView('results')}
      />
    )
  }

  if (currentView === 'experiment') {
    return <Experiment />
  }

  if (currentView === 'results') {
    return <Results />
  }

  // Main recommendation view
  return (
    <div className="app">
      <Logo />
      <SearchBar onSearch={handleSearch} loading={loading} />

      {error && (
        <div className="error-message">
          <p>Error: {error}</p>
          <p className="error-hint">Make sure the backend is running on port 8000</p>
        </div>
      )}

      {loading && (
        <div className="loading">
          <div className="spinner"></div>
          <p>Finding the perfect movies for you...</p>
        </div>
      )}

      {!loading && (gptRecommendations.length > 0 || claudeRecommendations.length > 0) && (
        <div className="recommendations-container">
          <MovieGrid
            title="GPT Recommendations"
            movies={gptRecommendations}
            color="#e50914"
          />
          <MovieGrid
            title="Claude Recommendations"
            movies={claudeRecommendations}
            color="#0071eb"
          />
        </div>
      )}

      {!loading && gptRecommendations.length === 0 && claudeRecommendations.length === 0 && !error && (
        <div className="empty-state">
          <h2>Discover Your Next Favorite Movie</h2>
          <p>Enter a prompt above to get personalized recommendations from GPT and Claude</p>
          <div className="example-prompts">
            <p>Try examples like:</p>
            <ul>
              <li>"Mind-bending sci-fi thrillers"</li>
              <li>"Feel-good romantic comedies"</li>
              <li>"Dark psychological thrillers"</li>
            </ul>
          </div>
        </div>
      )}
    </div>
  )
}

export default App
