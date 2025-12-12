// The 10 experiment prompts with metadata
export const experimentPrompts = [
  {
    id: 0,
    text: "Animated movies for kids",
    category: "Categorical",
    specificity: "High",
    groundTruth: "Objective",
    complexity: "Simple"
  },
  {
    id: 1,
    text: "Dark psychological films",
    category: "Mood",
    specificity: "Open",
    groundTruth: "Subjective",
    complexity: "Medium"
  },
  {
    id: 2,
    text: "Movies for a boys' night in",
    category: "Situational",
    specificity: "Medium",
    groundTruth: "Subjective",
    complexity: "Medium"
  },
  {
    id: 3,
    text: "Movies that engineers and PhD students would enjoy",
    category: "Identity",
    specificity: "Medium",
    groundTruth: "Subjective",
    complexity: "Complex"
  },
  {
    id: 4,
    text: "Movies with Tom Hanks",
    category: "Categorical",
    specificity: "High",
    groundTruth: "Objective",
    complexity: "Simple"
  },
  {
    id: 5,
    text: "I liked Goodfellas and The Departed",
    category: "Similarity",
    specificity: "Medium",
    groundTruth: "Subjective",
    complexity: "Complex"
  },
  {
    id: 6,
    text: "Romantic comedies with female leads",
    category: "Categorical",
    specificity: "High",
    groundTruth: "Semi-objective",
    complexity: "Medium"
  },
  {
    id: 7,
    text: "Feel-good movies after a long day",
    category: "Mood",
    specificity: "Open",
    groundTruth: "Subjective",
    complexity: "Simple"
  },
  {
    id: 8,
    text: "Scary movies that aren't too gory",
    category: "Mood",
    specificity: "Medium",
    groundTruth: "Subjective",
    complexity: "Medium"
  },
  {
    id: 9,
    text: "Movies with unexpected plot twists",
    category: "Categorical",
    specificity: "Medium",
    groundTruth: "Semi-objective",
    complexity: "Medium"
  }
]

// Import pre-generated slates from separate file
export { experimentSlates } from './experimentSlates.js'
