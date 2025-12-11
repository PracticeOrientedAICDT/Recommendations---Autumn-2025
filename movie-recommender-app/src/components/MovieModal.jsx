import { useEffect, useState } from 'react'
import './MovieModal.css'

function MovieModal({ movie, onClose }) {
  const [imageError, setImageError] = useState(false)
  const [description, setDescription] = useState('')
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    // Prevent body scroll when modal is open
    document.body.style.overflow = 'hidden'

    // Handle escape key
    const handleEscape = (e) => {
      if (e.key === 'Escape') {
        onClose()
      }
    }

    document.addEventListener('keydown', handleEscape)

    // Fetch movie description from TMDB
    const fetchDescription = async () => {
      try {
        // Extract year from title (e.g., "Toy Story (1995)" -> 1995)
        const yearMatch = movie.title.match(/\((\d{4})\)/)
        const year = yearMatch ? yearMatch[1] : null
        const cleanTitle = movie.title.replace(/\s*\(\d{4}\)/, '').trim()

        const TMDB_API_KEY = import.meta.env.VITE_TMDB_API_KEY || '15d2ea6d0dc1d476efbca3eba2b9bbfb'
        const response = await fetch(
          `https://api.themoviedb.org/3/search/movie?api_key=${TMDB_API_KEY}&query=${encodeURIComponent(cleanTitle)}${year ? `&year=${year}` : ''}`
        )
        const data = await response.json()

        if (data.results && data.results.length > 0) {
          setDescription(data.results[0].overview || 'No description available.')
        } else {
          setDescription('No description available.')
        }
      } catch (error) {
        console.error('Error fetching movie description:', error)
        setDescription('Unable to load description.')
      } finally {
        setLoading(false)
      }
    }

    fetchDescription()

    return () => {
      document.body.style.overflow = 'unset'
      document.removeEventListener('keydown', handleEscape)
    }
  }, [onClose, movie.title])

  const handleBackdropClick = (e) => {
    if (e.target.className === 'modal-backdrop') {
      onClose()
    }
  }

  return (
    <div className="modal-backdrop" onClick={handleBackdropClick}>
      <div className="modal-content">
        <button className="modal-close" onClick={onClose}>
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <line x1="18" y1="6" x2="6" y2="18"></line>
            <line x1="6" y1="6" x2="18" y2="18"></line>
          </svg>
        </button>

        <div className="modal-body">
          <div className="modal-poster">
            {!imageError ? (
              <img
                src={movie.poster_url}
                alt={movie.title}
                onError={() => setImageError(true)}
              />
            ) : (
              <div className="modal-poster-placeholder">
                <svg
                  width="80"
                  height="80"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                >
                  <rect x="2" y="2" width="20" height="20" rx="2.18" ry="2.18"></rect>
                  <line x1="7" y1="2" x2="7" y2="22"></line>
                  <line x1="17" y1="2" x2="17" y2="22"></line>
                  <line x1="2" y1="12" x2="22" y2="12"></line>
                  <line x1="2" y1="7" x2="7" y2="7"></line>
                  <line x1="2" y1="17" x2="7" y2="17"></line>
                  <line x1="17" y1="17" x2="22" y2="17"></line>
                  <line x1="17" y1="7" x2="22" y2="7"></line>
                </svg>
              </div>
            )}
          </div>

          <div className="modal-info">
            <h2 className="modal-title">{movie.title}</h2>

            <div className="modal-genres">
              {movie.genres.split('|').map((genre, index) => (
                <span key={index} className="genre-tag">{genre}</span>
              ))}
            </div>

            <div className="modal-section">
              <h3>Description</h3>
              {loading ? (
                <p className="description-loading">Loading description...</p>
              ) : (
                <p className="movie-description">{description}</p>
              )}
            </div>

            <div className="modal-meta">
              <div className="meta-item">
                <span className="meta-label">Movie ID:</span>
                <span className="meta-value">{movie.id}</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default MovieModal
