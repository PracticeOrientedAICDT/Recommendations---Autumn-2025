import { useState } from 'react'
import MovieCard from './MovieCard'
import './MovieGrid.css'

function MovieGrid({ title, movies, color }) {
  const [validationStatus, setValidationStatus] = useState(null) // null, 'validating', 'valid', 'invalid'
  const [validationResult, setValidationResult] = useState(null)

  const handleValidate = async () => {
    setValidationStatus('validating')

    try {
      const response = await fetch('http://localhost:8000/api/validate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          movie_ids: movies.map(m => m.id)
        }),
      })

      const data = await response.json()
      setValidationResult(data)

      if (data.all_valid) {
        setValidationStatus('valid')
      } else {
        setValidationStatus('invalid')
      }
    } catch (err) {
      setValidationStatus('error')
    }
  }

  const getButtonClass = () => {
    if (validationStatus === 'valid') return 'validate-button valid'
    if (validationStatus === 'invalid') return 'validate-button invalid'
    if (validationStatus === 'error') return 'validate-button error'
    return 'validate-button'
  }

  const getButtonText = () => {
    if (validationStatus === 'validating') return 'Validating...'
    if (validationStatus === 'valid') return `✓ All ${movies.length} Movies Valid`
    if (validationStatus === 'invalid') {
      return `✗ ${validationResult.num_invalid} Invalid Movies`
    }
    if (validationStatus === 'error') return '✗ Validation Error'
    return 'Validate Dataset'
  }

  return (
    <div className="movie-grid-section">
      <div className="grid-header">
        <h2 className="grid-title" style={{ borderLeftColor: color }}>
          {title}
        </h2>
        <button
          className={getButtonClass()}
          onClick={handleValidate}
          disabled={validationStatus === 'validating'}
        >
          {getButtonText()}
        </button>
      </div>
      <div className="movie-grid">
        {movies.map((movie, index) => (
          <MovieCard key={movie.id} movie={movie} index={index} />
        ))}
      </div>
    </div>
  )
}

export default MovieGrid
