# Movie Recommender System

A comprehensive movie recommendation system that implements multiple recommendation algorithms and provides a web interface for users to get personalized movie recommendations.

## Features

### 1. Multiple Recommendation Algorithms

#### Content-Based Filtering
- Analyzes movie content features (genres, descriptions) to find similar movies
- Uses TF-IDF vectorization and cosine similarity to measure content similarity
- Ideal for recommending movies similar to ones you've enjoyed
- Example: If you liked "The Dark Knight", it might recommend other action-packed superhero movies

#### Collaborative Filtering
- Recommends movies based on user preferences and similar users' tastes
- Uses a user-item matrix to find patterns in user ratings
- Implements k-nearest neighbors algorithm to find similar users
- Great for discovering movies that users with similar tastes have enjoyed
- Example: If users who liked "Inception" also enjoyed "Interstellar", it might recommend the latter

#### Hybrid Approach
- Combines content-based and collaborative filtering for more accurate recommendations
- Uses weighted scoring (60% content-based, 40% collaborative)
- Provides more balanced and diverse recommendations
- Example: Combines your movie preferences with what similar users liked

#### Genre-Based Recommendations
- Recommends movies based on genre preferences
- Creates a binary matrix of genres for each movie
- Uses cosine similarity to find movies with similar genre combinations
- Perfect for when you're in the mood for a specific genre
- Example: If you select "Action", it recommends high-rated action movies

#### Few-Shot Learning with LLM
- Uses OpenAI's GPT model to generate recommendations
- Implements few-shot learning with example recommendations
- Can understand natural language queries about movie preferences
- Provides personalized recommendations with explanations
- Example: Can understand queries like "I want a movie like The Matrix but with more romance"

### 2. Web Interface Features
- Clean and intuitive user interface
- Multiple recommendation method selection
- Support for various input types (movie ID, user ID, genre)
- Detailed movie information display
- Real-time recommendation generation

### 3. Data Processing
- Efficient data loading and preprocessing
- Genre parsing and normalization
- Missing value handling
- Text feature extraction
- Rating matrix generation

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd movie_recommender_app
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
Create a `.env` file in the root directory with your OpenAI API key:
```
OPENAI_API_KEY=your_api_key_here
```

## Usage

1. Start the Flask server:
```bash
python main_app.py
```

2. Open your web browser and navigate to `http://localhost:5000`

3. Use the web interface to:
   - Select a recommendation method
   - Input your preferences (movie ID, user ID, or genre)
   - Get personalized movie recommendations

## Project Structure

```
movie_recommender_app/
│
├── main_app.py              # Main Flask application
├── utils/
│   └── data_loader.py      # Data loading and preprocessing
├── llm/
│   └── llm_helper.py       # LLM integration
├── recommenders/
│   ├── content_based.py    # Content-based filtering
│   ├── collaborative.py    # Collaborative filtering
│   ├── hybrid.py          # Hybrid approach
│   ├── few_shot.py        # Few-shot learning
│   └── genre_based.py     # Genre-based recommendations
└── movies_metadata.csv     # Movie dataset
```

## Technical Details

### Data Processing
- Uses pandas for efficient data manipulation
- Implements TF-IDF vectorization for text features
- Handles missing values and data normalization
- Creates genre-based binary matrices

### Machine Learning
- Implements cosine similarity for content matching
- Uses k-nearest neighbors for collaborative filtering
- Combines multiple recommendation strategies
- Implements few-shot learning with LLM

### API Integration
- OpenAI API integration for LLM-based recommendations
- RESTful API endpoints for recommendation requests
- JSON response format for easy integration

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 