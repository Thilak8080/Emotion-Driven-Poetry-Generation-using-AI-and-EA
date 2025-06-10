# Emotion-Driven-Poetry-Generation-using-AI-and-EA

This project is a backend web application built using Flask that generates short poems based on user-provided emotions and situations. It uses Google’s Generative AI (Gemini API) for initial poem generation and applies evolutionary algorithms to refine poems based on rhyme, syllable count, and emotional accuracy. Each poem is evaluated, stored, and further optimized using user feedback.

🛠 Tools & Technologies Used
1. Programming Language: Python 3.10.13
2. Framework: Flask (for building the backend API)
3. AI Model: Google Generative AI (Gemini)
4. Optimization: DEAP (Distributed Evolutionary Algorithms in Python)
5. Database: SQLite3 (for storing poems and evaluations)
6. Others: Flask-CORS, NumPy, scikit-learn, dotenv

Project Structure Overview
1. app.py – Main Flask app; handles poem generation, refinement, feedback loop
2. database.py – Functions for storing and retrieving poems in SQLite
3. query_poems.py – Script for querying stored poems from the database
4. poems.db and poems_new.db – Databases that hold original and evolved poems
5. ui/ – Frontend folder (React or other UI framework)
6. .env – Environment variable file storing the Google API key
7. .python-version – File specifying the Python version used
8. __pycache__/ – Auto-generated cache by Python (not manually edited)

How to Set Up

1. Make sure you have Python 3.10.13 installed.

2. Install required libraries using:
pip install flask flask-cors openai google-generativeai deap numpy scikit-learn

3. Create a .env file in the project root and add:
GOOGLE_API_KEY=your_google_generative_ai_key


4. Run the backend app:
python app.py
