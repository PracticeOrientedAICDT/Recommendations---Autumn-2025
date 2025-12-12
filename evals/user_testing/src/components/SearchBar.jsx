import { useState } from 'react'
import './SearchBar.css'

function SearchBar({ onSearch, loading }) {
  const [prompt, setPrompt] = useState('')

  const handleSubmit = (e) => {
    e.preventDefault()
    if (prompt.trim() && !loading) {
      onSearch(prompt)
    }
  }

  return (
    <div className="search-container">
      <form onSubmit={handleSubmit} className="search-form">
        <input
          type="text"
          className="search-input"
          placeholder="What kind of movies are you looking for?"
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          disabled={loading}
        />
        <button
          type="submit"
          className="search-button"
          disabled={loading || !prompt.trim()}
        >
          {loading ? 'Searching...' : 'Get Recommendations'}
        </button>
      </form>
    </div>
  )
}

export default SearchBar
