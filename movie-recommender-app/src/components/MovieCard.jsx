import { useState } from 'react'
import './MovieCard.css'
import MovieModal from './MovieModal'

function MovieCard({ movie, index, onTrackInteraction, slate }) {
  const [imageError, setImageError] = useState(false)
  const [showHover, setShowHover] = useState(false)
  const [showModal, setShowModal] = useState(false)
  const [hoverStartTime, setHoverStartTime] = useState(null)
  const [modalOpenTime, setModalOpenTime] = useState(null)
  const [description, setDescription] = useState('')
  const [loadingDescription, setLoadingDescription] = useState(false)

  const handleMouseEnter = () => {
    const enterTime = Date.now()
    setHoverStartTime(enterTime)
    setShowHover(true)
    setLoadingDescription(true)

    // Track movie hover
    if (onTrackInteraction) {
      onTrackInteraction('movie_hover', {
        movieId: movie.id,
        movieTitle: movie.title,
        slate: slate,
        position: index
      })
    }

    // Fetch movie description from TMDB
    const fetchDescription = async () => {
      try {
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
        setLoadingDescription(false)
      }
    }

    fetchDescription()
  }

  const handleMouseLeave = () => {
    setShowHover(false)

    // Track hover duration
    if (onTrackInteraction && hoverStartTime) {
      const duration = (Date.now() - hoverStartTime) / 1000
      onTrackInteraction('movie_hover_end', {
        movieId: movie.id,
        movieTitle: movie.title,
        slate: slate,
        hoverDuration: duration
      })
    }
  }

  const handleClick = () => {
    const openTime = Date.now()
    setModalOpenTime(openTime)
    setShowModal(true)

    // Track movie card click
    if (onTrackInteraction) {
      onTrackInteraction('movie_click', {
        movieId: movie.id,
        movieTitle: movie.title,
        slate: slate,
        position: index
      })
    }
  }

  const handleCloseModal = () => {
    setShowModal(false)

    // Track modal close with duration
    if (onTrackInteraction && modalOpenTime) {
      const duration = (Date.now() - modalOpenTime) / 1000
      onTrackInteraction('modal_close', {
        movieId: movie.id,
        movieTitle: movie.title,
        slate: slate,
        durationOpen: duration
      })
    }
  }

  return (
    <>
      <div
        className="movie-card"
        style={{ animationDelay: `${index * 0.1}s` }}
        onMouseEnter={handleMouseEnter}
        onMouseLeave={handleMouseLeave}
        onClick={handleClick}
      >
      <div className="poster-container">
        {!imageError ? (
          <img
            src={movie.poster_url}
            alt={movie.title}
            className="poster"
            onError={() => setImageError(true)}
          />
        ) : (
          <div className="poster-placeholder">
            <svg
              width="60"
              height="60"
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

        {showHover && (
          <div className="hover-overlay">
            <div className="hover-content">
              <h3 className="hover-title">{movie.title}</h3>
              <div className="hover-genres">
                {movie.genres.split('|').slice(0, 3).map((genre, i) => (
                  <span key={i} className="hover-genre-tag">{genre}</span>
                ))}
              </div>
              <div className="hover-description">
                {loadingDescription ? (
                  <p className="loading-text">Loading...</p>
                ) : (
                  <p>{description}</p>
                )}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>

    {showModal && (
      <MovieModal movie={movie} onClose={handleCloseModal} />
    )}
  </>
  )
}

export default MovieCard
