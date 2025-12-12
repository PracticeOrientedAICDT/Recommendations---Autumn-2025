import { useState, useEffect } from 'react'
import './Experiment.css'
import MovieCard from './MovieCard'
import { experimentPrompts, experimentSlates } from '../data/experimentPrompts'
import { API_URL } from '../config'

function Experiment() {
  // Main state
  const [currentScreen, setCurrentScreen] = useState('welcome') // 'welcome', 'promptDisplay', 'slateSelection', 'questionnaire', 'completion'
  const [participantId, setParticipantId] = useState('')
  const [sessionData, setSessionData] = useState(null)
  const [currentPromptIndex, setCurrentPromptIndex] = useState(0)
  const [responses, setResponses] = useState([])
  const [rankings, setRankings] = useState({}) // { A: 1, B: 2, C: 3 } or similar
  const [timestamps, setTimestamps] = useState({})
  const [questionnaireAnswers, setQuestionnaireAnswers] = useState({
    movieFrequency: '',
    genresPreferred: [],
    classicVsRecent: '',
    overallRelevance: '',
    overallDiversity: '',
    overallNovelty: '',
    decisionDifficulty: '',
    familiarityBalance: '',
    promptBasedSearch: '',
    comments: ''
  })

  // experimentSlates imported from experimentPrompts.js

  // Generate randomization on study start
  const startStudy = async () => {
    const id = generateParticipantId()
    setParticipantId(id)

    // Randomize prompt order
    const promptOrder = shuffleArray([...Array(10).keys()])

    // Randomize model positions for each prompt
    const modelMappings = {}
    const models = ['gpt', 'claude', 'diffusion']

    promptOrder.forEach((promptIdx) => {
      const shuffledModels = shuffleArray([...models])
      modelMappings[promptIdx] = {
        A: shuffledModels[0],
        B: shuffledModels[1],
        C: shuffledModels[2]
      }
    })

    const session = {
      participantId: id,
      startTime: new Date().toISOString(),
      promptOrder,
      modelMappings,
      deviceInfo: getDeviceInfo()
    }

    setSessionData(session)
    setTimestamps({ studyStart: Date.now() })

    // Create session in database
    try {
      await fetch(`${API_URL}/api/experiment-session`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ sessionData: session }),
      })
    } catch (err) {
      console.error('Failed to create session:', err)
    }

    setCurrentScreen('promptDisplay')
  }

  // Helper functions
  const generateParticipantId = () => {
    return 'P' + Date.now() + Math.random().toString(36).substr(2, 9)
  }

  const shuffleArray = (array) => {
    const shuffled = [...array]
    for (let i = shuffled.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]]
    }
    return shuffled
  }

  const getDeviceInfo = () => {
    return {
      userAgent: navigator.userAgent,
      platform: navigator.platform,
      language: navigator.language,
      screenWidth: window.screen.width,
      screenHeight: window.screen.height
    }
  }

  // Track interaction function (non-blocking)
  const trackInteraction = (type, data = {}) => {
    if (!participantId) return

    // Fire and forget - don't wait for response
    fetch(`${API_URL}/api/track-interaction`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        participantId,
        type,
        data: {
          ...data,
          currentPromptIndex,
          currentScreen
        },
        timestamp: Date.now()
      }),
    }).catch(err => {
      console.error('Failed to track interaction:', err)
    })
  }

  // Navigate to slate selection
  const viewRecommendations = () => {
    setTimestamps({
      ...timestamps,
      [`prompt${currentPromptIndex}_click`]: Date.now()
    })
    setCurrentScreen('slateSelection')
  }

  // Handle ranking selection
  const handleRanking = (option, rank) => {
    const oldRanking = rankings[option] || null
    const newRankings = { ...rankings }

    // If this rank is already assigned to another option, remove it from that option
    const optionWithThisRank = Object.keys(newRankings).find(key => newRankings[key] === rank && key !== option)
    if (optionWithThisRank) {
      delete newRankings[optionWithThisRank]
    }

    // If clicking the same star again, remove the rating (toggle off)
    if (newRankings[option] === rank) {
      delete newRankings[option]
    } else {
      newRankings[option] = rank
    }

    setRankings(newRankings)

    // Track the star click (non-blocking)
    setTimeout(() => {
      trackInteraction('star_click', {
        slate: option,
        oldRating: oldRanking,
        newRating: newRankings[option] || null,
        promptIndex: currentPromptIndex
      })
    }, 0)
  }

  // Navigate to next prompt or completion
  const handleNext = async () => {
    // Validate that all three rankings are complete and unique
    const rankingsComplete = Object.keys(rankings).length === 3 &&
      Object.values(rankings).includes(1) &&
      Object.values(rankings).includes(2) &&
      Object.values(rankings).includes(3)

    if (!rankingsComplete) return

    const actualPromptIdx = sessionData.promptOrder[currentPromptIndex]
    const prompt = experimentPrompts[actualPromptIdx]
    const mapping = sessionData.modelMappings[actualPromptIdx]

    const response = {
      participantId: sessionData.participantId,
      sequencePosition: currentPromptIndex + 1,
      promptId: prompt.id,
      promptText: prompt.text,
      promptCategory: prompt.category,
      modelMapping: mapping,
      rankings, // { A: 1, B: 3, C: 2 } for example (1★=worst, 3★=best)
      rankedModels: {
        first: mapping[Object.keys(rankings).find(key => rankings[key] === 3)],  // 3 stars = 1st place
        second: mapping[Object.keys(rankings).find(key => rankings[key] === 2)], // 2 stars = 2nd place
        third: mapping[Object.keys(rankings).find(key => rankings[key] === 1)]   // 1 star = 3rd place
      },
      promptDisplayTime: timestamps[`prompt${currentPromptIndex}_display`],
      slateViewTime: timestamps[`prompt${currentPromptIndex}_click`],
      selectionTime: Date.now(),
      timeOnPrompt: (timestamps[`prompt${currentPromptIndex}_click`] - timestamps[`prompt${currentPromptIndex}_display`]) / 1000,
      timeOnSlates: (Date.now() - timestamps[`prompt${currentPromptIndex}_click`]) / 1000
    }

    const newResponses = [...responses, response]
    setResponses(newResponses)

    // Save this prompt response to database immediately (fire and forget)
    fetch(`${API_URL}/api/save-prompt-response`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        participantId: sessionData.participantId,
        response
      }),
    }).catch(err => {
      console.error('Failed to save prompt response:', err)
    })

    // Update UI immediately
    if (currentPromptIndex + 1 >= 10) {
      setCurrentScreen('questionnaire')
    } else {
      setCurrentPromptIndex(currentPromptIndex + 1)
      setRankings({})
      setCurrentScreen('promptDisplay')
    }
  }

  // Handle questionnaire submission
  const handleQuestionnaireSubmit = async () => {
    await saveResults(responses)
    setCurrentScreen('completion')
  }

  // Update questionnaire answer
  const handleQuestionnaireChange = (field, value) => {
    setQuestionnaireAnswers({
      ...questionnaireAnswers,
      [field]: value
    })
  }

  // Save results to backend
  const saveResults = async (allResponses) => {
    try {
      await fetch(`${API_URL}/api/experiment-results`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          sessionData,
          responses: allResponses,
          questionnaire: questionnaireAnswers,
          endTime: new Date().toISOString(),
          totalDuration: (Date.now() - timestamps.studyStart) / 1000
        }),
      })
    } catch (err) {
      console.error('Failed to save results:', err)
    }
  }

  // Record timestamp when prompt is displayed
  useEffect(() => {
    if (currentScreen === 'promptDisplay') {
      setTimestamps(prev => ({
        ...prev,
        [`prompt${currentPromptIndex}_display`]: Date.now()
      }))
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [currentScreen, currentPromptIndex])

  // SCREEN 1: Welcome & Instructions
  if (currentScreen === 'welcome') {
    return (
      <div className="experiment-welcome">
        <div className="welcome-content">
          <h1>Movie Recommendation Study</h1>
          <p className="study-description">Help us understand movie recommendation preferences</p>

          <div className="instructions">
            <h2>Instructions</h2>
            <ul>
              <li>You will see 10 different prompts (movie requests like "recommend me something for kids" or "a dark psychological film") about movie recommendations</li>
              <li>For each prompt, you'll be shown three different sets of movie recommendations (labeled Option A, B, and C)</li>
              <li>You can hover over any movie poster to see a quick preview, or click on it to view the full description</li>
              <li><strong>Rate each set with stars: you must give exactly one set 3★ (best fit), one set 2★ (medium fit), and one set 1★ (worst fit). Each rating must be unique!</strong></li>
              <li>You must rate all three options before proceeding to the next prompt</li>
              <li>There are no right or wrong answers - we want your honest preference</li>
              <li>The study takes approximately 5 minutes</li>
              <li className="dataset-note"><strong>Note:</strong> The movies in this study are from the MovieLens dataset, which contains films released up to the early 2000s. You won't see recently released movies.</li>
            </ul>
          </div>

          <div className="study-details">
            <p><strong>Estimated duration:</strong> 5 minutes</p>
            <p><strong>Number of prompts:</strong> 10</p>
            <p><strong>Your responses are anonymous</strong></p>
          </div>

          <button className="start-button" onClick={startStudy}>
            Start Study
          </button>

          <button className="back-button" onClick={() => window.location.href = '/'}>
            Back to Home
          </button>
        </div>
      </div>
    )
  }

  // Get current prompt info
  const actualPromptIdx = sessionData.promptOrder[currentPromptIndex]
  const currentPrompt = experimentPrompts[actualPromptIdx]
  const progressPercent = (currentPromptIndex / 10) * 100

  // SCREEN 2 - STEP 1: Prompt Display
  if (currentScreen === 'promptDisplay') {
    return (
      <div className="experiment-prompt-display">
        <div className="progress-indicator">
          <p className="progress-text">Prompt {currentPromptIndex + 1} of 10</p>
          <div className="progress-bar">
            <div className="progress-fill" style={{ width: `${progressPercent}%` }}></div>
          </div>
        </div>

        <div className="prompt-content">
          <h2 className="prompt-text">"{currentPrompt.text}"</h2>
          <button className="view-recommendations-btn" onClick={viewRecommendations}>
            View Recommendations →
          </button>
        </div>
      </div>
    )
  }

  // SCREEN 2 - STEP 2: Slate Selection
  if (currentScreen === 'slateSelection') {
    const mapping = sessionData.modelMappings[actualPromptIdx]

    // Get slates for each option
    const slateA = experimentSlates[actualPromptIdx]?.[mapping.A] || []
    const slateB = experimentSlates[actualPromptIdx]?.[mapping.B] || []
    const slateC = experimentSlates[actualPromptIdx]?.[mapping.C] || []

    return (
      <div className="experiment-slate-selection">
        <div className="progress-indicator">
          <p className="progress-text">Prompt {currentPromptIndex + 1} of 10</p>
          <div className="progress-bar">
            <div className="progress-fill" style={{ width: `${progressPercent}%` }}></div>
          </div>
        </div>

        <div className="prompt-reminder">
          <p className="prompt-text">"{currentPrompt.text}"</p>
          <p className="instruction">Rate each option with stars (3★ = best fit, 1★ = worst fit)</p>
        </div>

        <div className="slates-container">
          <div className={`slate-option ${rankings['A'] ? 'ranked rank-' + rankings['A'] : ''}`}>
            <div className="slate-header">
              <h3>Option A</h3>
              <div className="star-rating">
                <span className="rating-label">Rate:</span>
                <div className="stars-wrapper">
                  {[1, 2, 3].map((star) => (
                    <button
                      key={star}
                      className={`star-btn ${rankings['A'] >= star ? 'filled' : ''}`}
                      onClick={() => handleRanking('A', star)}
                      aria-label={`${star} star${star > 1 ? 's' : ''}`}
                    >
                      ★
                    </button>
                  ))}
                </div>
              </div>
            </div>
            <div className="slate-movies">
              {slateA.map((movie, index) => (
                <MovieCard
                  key={`${movie.id}-${index}`}
                  movie={movie}
                  index={index}
                  onTrackInteraction={trackInteraction}
                  slate="A"
                />
              ))}
            </div>
          </div>

          <div className={`slate-option ${rankings['B'] ? 'ranked rank-' + rankings['B'] : ''}`}>
            <div className="slate-header">
              <h3>Option B</h3>
              <div className="star-rating">
                <span className="rating-label">Rate:</span>
                <div className="stars-wrapper">
                  {[1, 2, 3].map((star) => (
                    <button
                      key={star}
                      className={`star-btn ${rankings['B'] >= star ? 'filled' : ''}`}
                      onClick={() => handleRanking('B', star)}
                      aria-label={`${star} star${star > 1 ? 's' : ''}`}
                    >
                      ★
                    </button>
                  ))}
                </div>
              </div>
            </div>
            <div className="slate-movies">
              {slateB.map((movie, index) => (
                <MovieCard
                  key={`${movie.id}-${index}`}
                  movie={movie}
                  index={index}
                  onTrackInteraction={trackInteraction}
                  slate="B"
                />
              ))}
            </div>
          </div>

          <div className={`slate-option ${rankings['C'] ? 'ranked rank-' + rankings['C'] : ''}`}>
            <div className="slate-header">
              <h3>Option C</h3>
              <div className="star-rating">
                <span className="rating-label">Rate:</span>
                <div className="stars-wrapper">
                  {[1, 2, 3].map((star) => (
                    <button
                      key={star}
                      className={`star-btn ${rankings['C'] >= star ? 'filled' : ''}`}
                      onClick={() => handleRanking('C', star)}
                      aria-label={`${star} star${star > 1 ? 's' : ''}`}
                    >
                      ★
                    </button>
                  ))}
                </div>
              </div>
            </div>
            <div className="slate-movies">
              {slateC.map((movie, index) => (
                <MovieCard
                  key={`${movie.id}-${index}`}
                  movie={movie}
                  index={index}
                  onTrackInteraction={trackInteraction}
                  slate="C"
                />
              ))}
            </div>
          </div>
        </div>

        <button
          className="next-button"
          onClick={handleNext}
          disabled={
            !(Object.keys(rankings).length === 3 &&
              Object.values(rankings).includes(1) &&
              Object.values(rankings).includes(2) &&
              Object.values(rankings).includes(3))
          }
        >
          Next →
        </button>
      </div>
    )
  }

  // SCREEN 3: Questionnaire
  if (currentScreen === 'questionnaire') {
    const isComplete = questionnaireAnswers.movieFrequency &&
                      questionnaireAnswers.genresPreferred.length > 0 &&
                      questionnaireAnswers.classicVsRecent &&
                      questionnaireAnswers.overallRelevance &&
                      questionnaireAnswers.overallDiversity &&
                      questionnaireAnswers.overallNovelty &&
                      questionnaireAnswers.decisionDifficulty &&
                      questionnaireAnswers.familiarityBalance &&
                      questionnaireAnswers.promptBasedSearch

    return (
      <div className="experiment-questionnaire">
        <div className="questionnaire-content">
          <h1>Post-Study Questionnaire</h1>
          <p className="questionnaire-intro">Please answer a few questions about yourself and your experience</p>

          <div className="questionnaire-form">
            <div className="form-group">
              <label htmlFor="movieFrequency">How often do you watch movies? *</label>
              <select
                id="movieFrequency"
                value={questionnaireAnswers.movieFrequency}
                onChange={(e) => handleQuestionnaireChange('movieFrequency', e.target.value)}
                className="form-select"
              >
                <option value="">Select frequency</option>
                <option value="daily">Daily</option>
                <option value="few-times-week">A few times a week</option>
                <option value="once-week">Once a week</option>
                <option value="few-times-month">A few times a month</option>
                <option value="rarely">Rarely</option>
              </select>
            </div>

            <div className="form-group">
              <label>Which genres do you typically enjoy? * (Select all that apply)</label>
              <div className="checkbox-group">
                {['Action', 'Adventure', 'Animation', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Horror', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'Western', 'Other'].map(genre => (
                  <label key={genre} className="checkbox-label">
                    <input
                      type="checkbox"
                      checked={questionnaireAnswers.genresPreferred.includes(genre)}
                      onChange={(e) => {
                        const newGenres = e.target.checked
                          ? [...questionnaireAnswers.genresPreferred, genre]
                          : questionnaireAnswers.genresPreferred.filter(g => g !== genre)
                        handleQuestionnaireChange('genresPreferred', newGenres)
                      }}
                    />
                    {genre}
                  </label>
                ))}
              </div>
            </div>

            <div className="form-group">
              <label htmlFor="classicVsRecent">Do you prefer recent releases or classic films? *</label>
              <div className="scale-container">
                <span className="scale-label-left">Strongly prefer classics</span>
                <div className="radio-scale">
                  {[1, 2, 3, 4, 5].map(value => (
                    <label key={value} className="radio-label">
                      <input
                        type="radio"
                        name="classicVsRecent"
                        value={value}
                        checked={questionnaireAnswers.classicVsRecent === String(value)}
                        onChange={(e) => handleQuestionnaireChange('classicVsRecent', e.target.value)}
                      />
                      <span>{value}</span>
                    </label>
                  ))}
                </div>
                <span className="scale-label-right">Strongly prefer recent</span>
              </div>
            </div>

            <h3 className="section-heading">Overall Experience</h3>

            <div className="form-group">
              <label htmlFor="overallRelevance">Overall, how relevant were the movie recommendations to your preferences? *</label>
              <div className="scale-container">
                <span className="scale-label-left">Not at all relevant</span>
                <div className="radio-scale">
                  {[1, 2, 3, 4, 5].map(value => (
                    <label key={value} className="radio-label">
                      <input
                        type="radio"
                        name="overallRelevance"
                        value={value}
                        checked={questionnaireAnswers.overallRelevance === String(value)}
                        onChange={(e) => handleQuestionnaireChange('overallRelevance', e.target.value)}
                      />
                      <span>{value}</span>
                    </label>
                  ))}
                </div>
                <span className="scale-label-right">Extremely relevant</span>
              </div>
            </div>

            <div className="form-group">
              <label htmlFor="overallDiversity">Overall, how diverse were the movie recommendations? *</label>
              <div className="scale-container">
                <span className="scale-label-left">Not at all diverse</span>
                <div className="radio-scale">
                  {[1, 2, 3, 4, 5].map(value => (
                    <label key={value} className="radio-label">
                      <input
                        type="radio"
                        name="overallDiversity"
                        value={value}
                        checked={questionnaireAnswers.overallDiversity === String(value)}
                        onChange={(e) => handleQuestionnaireChange('overallDiversity', e.target.value)}
                      />
                      <span>{value}</span>
                    </label>
                  ))}
                </div>
                <span className="scale-label-right">Extremely diverse</span>
              </div>
            </div>

            <div className="form-group">
              <label htmlFor="overallNovelty">Overall, how novel/surprising were the movie recommendations? *</label>
              <div className="scale-container">
                <span className="scale-label-left">Not at all novel</span>
                <div className="radio-scale">
                  {[1, 2, 3, 4, 5].map(value => (
                    <label key={value} className="radio-label">
                      <input
                        type="radio"
                        name="overallNovelty"
                        value={value}
                        checked={questionnaireAnswers.overallNovelty === String(value)}
                        onChange={(e) => handleQuestionnaireChange('overallNovelty', e.target.value)}
                      />
                      <span>{value}</span>
                    </label>
                  ))}
                </div>
                <span className="scale-label-right">Extremely novel</span>
              </div>
            </div>

            <h3 className="section-heading">Decision-Making Experience</h3>

            <div className="form-group">
              <label htmlFor="decisionDifficulty">How difficult was it to choose between the movie sets presented to you? *</label>
              <div className="scale-container">
                <span className="scale-label-left">Very easy</span>
                <div className="radio-scale">
                  {[1, 2, 3, 4, 5].map(value => (
                    <label key={value} className="radio-label">
                      <input
                        type="radio"
                        name="decisionDifficulty"
                        value={value}
                        checked={questionnaireAnswers.decisionDifficulty === String(value)}
                        onChange={(e) => handleQuestionnaireChange('decisionDifficulty', e.target.value)}
                      />
                      <span>{value}</span>
                    </label>
                  ))}
                </div>
                <span className="scale-label-right">Very difficult</span>
              </div>
            </div>

            <h3 className="section-heading">Familiarity vs. Discovery Balance</h3>

            <div className="form-group">
              <label htmlFor="familiarityBalance">Did the recommendations strike the right balance between familiar movies and new discoveries? *</label>
              <div className="scale-container">
                <span className="scale-label-left">Too familiar</span>
                <div className="radio-scale">
                  {[1, 2, 3, 4, 5].map(value => (
                    <label key={value} className="radio-label">
                      <input
                        type="radio"
                        name="familiarityBalance"
                        value={value}
                        checked={questionnaireAnswers.familiarityBalance === String(value)}
                        onChange={(e) => handleQuestionnaireChange('familiarityBalance', e.target.value)}
                      />
                      <span>{value === 3 ? 'Perfect' : value}</span>
                    </label>
                  ))}
                </div>
                <span className="scale-label-right">Too unfamiliar</span>
              </div>
            </div>

            <h3 className="section-heading">Future Applications</h3>

            <div className="form-group">
              <label htmlFor="promptBasedSearch">Would you like to see this functionality (prompt-based movie search) in your favorite streaming service? *</label>
              <div className="scale-container">
                <span className="scale-label-left">Not at all interested</span>
                <div className="radio-scale">
                  {[1, 2, 3, 4, 5].map(value => (
                    <label key={value} className="radio-label">
                      <input
                        type="radio"
                        name="promptBasedSearch"
                        value={value}
                        checked={questionnaireAnswers.promptBasedSearch === String(value)}
                        onChange={(e) => handleQuestionnaireChange('promptBasedSearch', e.target.value)}
                      />
                      <span>{value}</span>
                    </label>
                  ))}
                </div>
                <span className="scale-label-right">Extremely interested</span>
              </div>
            </div>

            <h3 className="section-heading">Additional Feedback</h3>

            <div className="form-group">
              <label htmlFor="comments">Additional comments or feedback</label>
              <textarea
                id="comments"
                value={questionnaireAnswers.comments}
                onChange={(e) => handleQuestionnaireChange('comments', e.target.value)}
                className="form-textarea"
                placeholder="Please describe any patterns or reasoning you used when choosing one slate over another. For example, did you prefer certain types of movies, diversity in genres, familiarity with titles, etc.? Any other thoughts about the study or suggestions for improvement..."
                rows="5"
              />
            </div>

            <p className="required-note">* Required fields</p>

            <button
              className="submit-button"
              onClick={handleQuestionnaireSubmit}
              disabled={!isComplete}
            >
              Submit
            </button>
          </div>
        </div>
      </div>
    )
  }

  // SCREEN 4: Completion
  if (currentScreen === 'completion') {
    const totalMinutes = Math.floor((Date.now() - timestamps.studyStart) / 60000)

    return (
      <div className="experiment-completion">
        <div className="completion-content">
          <h1>Thank you for completing the study!</h1>
          <p>Your responses have been recorded and will help improve movie recommendation systems</p>

          <div className="completion-details">
            <p><strong>Participant ID:</strong> {participantId}</p>
            <p><strong>Duration:</strong> {totalMinutes} minutes</p>
            <p><strong>Prompts completed:</strong> 10/10</p>
          </div>

          <button className="close-button" onClick={() => window.location.href = '/'}>
            Close
          </button>
        </div>
      </div>
    )
  }

  return null
}

export default Experiment
