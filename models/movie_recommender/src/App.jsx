import { useState } from 'react';
import { Film, Sparkles, Loader2 } from 'lucide-react';
import './index.css';

export default function MovieRecommender() {
  const [apiKey, setApiKey] = useState('');
  const [description, setDescription] = useState('');
  const [movies, setMovies] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [hoveredButton, setHoveredButton] = useState(false);
  const [hoveredCards, setHoveredCards] = useState({});

  const getRecommendations = async () => {
    if (!apiKey.trim()) {
      setError('Please enter your OpenAI API key');
      return;
    }
    if (!description.trim()) {
      setError('Please describe yourself or your preferences');
      return;
    }

    setLoading(true);
    setError('');
    setMovies([]);

    try {
      const response = await fetch('https://api.openai.com/v1/chat/completions', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${apiKey}`
        },
        body: JSON.stringify({
          model: 'gpt-4',
          messages: [
            {
              role: 'system',
              content: 'You are a movie recommendation expert. Based on the user description, recommend exactly 10 movies. You must respond with ONLY a valid JSON object in this exact format: {"movies": [{"title": "Movie Name", "year": 2020, "reason": "Why this movie suits them"}]}. Do not include any other text before or after the JSON.'
            },
            {
              role: 'user',
              content: `Based on this description, recommend 10 movies: ${description}`
            }
          ],
          temperature: 0.8
        })
      });

      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.error?.message || 'API request failed');
      }

      const data = await response.json();
      const content = data.choices[0].message.content;
      const parsed = JSON.parse(content);
      
      const movieList = parsed.movies || parsed.recommendations || Object.values(parsed)[0];
      
      if (Array.isArray(movieList) && movieList.length > 0) {
        setMovies(movieList.slice(0, 10));
      } else {
        throw new Error('Unexpected response format');
      }
    } catch (err) {
      setError(err.message || 'Failed to get recommendations. Please check your API key and try again.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-container">
      <div className="content-wrapper">
        <div className="header">
          <div className="icon-wrapper">
            <Film className="icon" />
          </div>
          <h1 className="title">MovieMatch</h1>
          <p className="subtitle">Discover your next favorite film</p>
        </div>

        <div className="form-container">
          <div className="input-group">
            <label className="label">API Key</label>
            <input
              type="password"
              value={apiKey}
              onChange={(e) => setApiKey(e.target.value)}
              placeholder="sk-..."
              className="input"
            />
            <p className="help-text">Your key is stored locally and never saved</p>
          </div>

          <div className="input-group">
            <label className="label">About You</label>
            <textarea
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              placeholder="Tell us about your taste in movies, favorite genres, or what you're in the mood for..."
              rows="6"
              className="textarea"
            />
          </div>

          {error && (
            <div className="error-box">
              {error}
            </div>
          )}

          <button
            onClick={getRecommendations}
            disabled={loading}
            onMouseEnter={() => setHoveredButton(true)}
            onMouseLeave={() => setHoveredButton(false)}
            className={`button ${hoveredButton && !loading ? 'button-hover' : ''} ${loading ? 'button-disabled' : ''}`}
          >
            {loading ? (
              <>
                <Loader2 className="button-icon spinner" />
                Analyzing
              </>
            ) : (
              <>
                <Sparkles className="button-icon" />
                Get Recommendations
              </>
            )}
          </button>
        </div>

        {movies.length > 0 && (
          <div className="results-container">
            <h2 className="results-title">Recommended for You</h2>
            <div className="movie-list">
              {movies.map((movie, idx) => (
                <div
                  key={idx}
                  onMouseEnter={() => setHoveredCards({ ...hoveredCards, [idx]: true })}
                  onMouseLeave={() => setHoveredCards({ ...hoveredCards, [idx]: false })}
                  className={`movie-card ${hoveredCards[idx] ? 'movie-card-hover' : ''}`}
                >
                  <div className="movie-content">
                    <div className="number-badge">
                      {idx + 1}
                    </div>
                    <div className="movie-info">
                      <div className="movie-title-row">
                        <h3 className="movie-title">{movie.title}</h3>
                        {movie.year && <span className="movie-year">({movie.year})</span>}
                      </div>
                      <p className="movie-reason">{movie.reason}</p>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}