// API Configuration
export const API_URL = import.meta.env.VITE_API_URL ||
  (import.meta.env.PROD
    ? 'https://movie-experiment-backend.onrender.com'
    : 'http://localhost:8000')
