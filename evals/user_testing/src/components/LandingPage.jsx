import './LandingPage.css'

function LandingPage({ onEnter, onExperiment, onResults }) {
  return (
    <div className="landing-page">
      <div className="landing-content">
        <h1 className="landing-title">Movie Recommender</h1>
        <div className="landing-buttons">
          <button className="landing-button primary" onClick={onExperiment}>
            Recommendation Study
          </button>
          <button className="landing-button secondary disabled" disabled>
            Generate Slates
          </button>
          <button className="landing-button admin" onClick={onResults}>
            View Results 📊
          </button>
        </div>
      </div>
    </div>
  )
}

export default LandingPage
