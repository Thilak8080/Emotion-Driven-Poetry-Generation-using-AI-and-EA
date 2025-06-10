
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import google.generativeai as genai
import os
from dotenv import load_dotenv
import random
import numpy as np
from deap import base, creator, tools, algorithms
from textblob import TextBlob
import nltk 
import re
from nltk.corpus import cmudict
import logging
from database import db

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Download required NLTK data
nltk.download('cmudict')
nltk.download('punkt')

load_dotenv()

# Configure the Generative AI API
genai.configure(api_key='Google API Key')
model = genai.GenerativeModel(model_name="gemini-1.5-flash")

# Setup evolutionary algorithm components
creator.create("FitnessMax", base.Fitness, weights=(1.0, 1.0, 1.0))  # Multiple objectives
creator.create("Individual", str, fitness=creator.FitnessMax)

class PoemEvolver:
    def __init__(self, emotion, population_size=10):
        self.emotion = emotion
        self.population_size = population_size
        self.toolbox = base.Toolbox()
        self.setup_evolutionary_operators()
        self.d = cmudict.dict()
        self.human_feedback_history = []
        self.rhyme_patterns = ['ABA']  # Common rhyme schemes
        self.generation = 1

    def incorporate_feedback(self, feedback_score):
        """
        Incorporate human feedback into the evolution process.
        """
        self.human_feedback_history.append(feedback_score)
        logger.info(f"Feedback incorporated: {feedback_score}")

    def count_syllables(self, word):
        """Count syllables in a word using CMU dictionary"""
        try:
            return max([len([y for y in x if y[-1].isdigit()]) for x in self.d[word.lower()]])
        except KeyError:
            # Fallback syllable counting for unknown words
            count = 0
            vowels = 'aeiouy'
            word = word.lower()
            if word[0] in vowels:
                count += 1
            for index in range(1, len(word)):
                if word[index] in vowels and word[index - 1] not in vowels:
                    count += 1
            if word.endswith('e'):
                count -= 1
            if count == 0:
                count += 1
            return count

    def get_rhyme_sound(self, word):
        """Get the rhyming sound of a word"""
        try:
            pronunciations = self.d[word.lower()]
            return pronunciations[0][-2:]  # Last two phonemes
        except (KeyError, IndexError):
            return None

    def evaluate_rhyme_scheme(self, poem):
        """Evaluate how well the poem follows a rhyme scheme"""
        lines = poem.strip().split('\n')
        if len(lines) < 2:
            return 0.0

        # Get last word of each line
        last_words = [line.strip().split()[-1].lower() for line in lines]
        rhyme_sounds = [self.get_rhyme_sound(word) for word in last_words]

        # Check against common rhyme patterns
        best_score = 0.0
        for pattern in self.rhyme_patterns:
            score = 0
            expected_rhymes = {}
            for i, char in enumerate(pattern[:len(lines)]):
                if char not in expected_rhymes:
                    expected_rhymes[char] = rhyme_sounds[i]
                elif rhyme_sounds[i] == expected_rhymes[char]:
                    score += 1
            best_score = max(best_score, score / (len(lines) - 1))

        return best_score

    def evaluate_syllable_pattern(self, poem):
        """Evaluate syllable pattern consistency"""
        lines = poem.strip().split('\n')
        syllables_per_line = [sum(self.count_syllables(word) for word in line.split()) for line in lines]

        # Calculate variance in syllable counts (lower is better)
        mean_syllables = np.mean(syllables_per_line)
        variance = np.var(syllables_per_line)

        # Convert variance to a score between 0 and 1 (inverse relationship)
        return 1 / (1 + variance)

    def evaluate_emotion_alignment(self, poem):
        """Evaluate emotional alignment using sentiment analysis and past feedback"""
        # TextBlob sentiment analysis
        analysis = TextBlob(poem)
        sentiment_score = analysis.sentiment.polarity

        # Adjust score based on target emotion
        if self.emotion.lower() in ['sadness', 'anger'] and sentiment_score < 0:
            emotion_score = abs(sentiment_score)
        elif self.emotion.lower() == 'joy' and sentiment_score > 0:
            emotion_score = sentiment_score
        else:
            emotion_score = 0.1

        # Incorporate human feedback history
        if self.human_feedback_history:
            feedback_influence = np.mean(self.human_feedback_history) * 0.3  # 30% weight to human feedback
            emotion_score = (emotion_score * 0.7) + feedback_influence

        return emotion_score

    def evaluate(self, individual, is_mutated=False):
        """
        Evaluate a poem and store the scores in the database only if needed.
        """
        poem = str(individual)

        # Calculate scores
        emotion_score = self.evaluate_emotion_alignment(poem)
        rhyme_score = self.evaluate_rhyme_scheme(poem)
        syllable_score = self.evaluate_syllable_pattern(poem)

        # Store scores in the database only when explicitly required
        db.insert_poem(
            poem_text=poem,
            emotion_score=emotion_score,
            rhyme_score=rhyme_score,
            syllable_score=syllable_score,
            is_mutated=is_mutated,  # Track if it is mutated
            generation=self.generation
        )

        return (emotion_score, rhyme_score, syllable_score)

    def generate_initial_poem(self, situation):
        """Generate initial poem using Gemini"""
        try:
            prompt = f"""
            Generate a short 3-line haiku poem about this situation: {situation}
            The poem should express {self.emotion}.

            Rules:
            1. MUST be exactly 3 lines
            2. Keep it simple and direct
            3. Express {self.emotion} clearly
            4. Use basic rhyming (ABA)
            5. The syllable count in this poem must be 5 syllables in the first line, 7 in the second line, and 5 again in the last

            Format the response as just the poem text, nothing else.
            """

            response = model.generate_content(
                prompt,
                generation_config={
                    "temperature": 0.7,
                    "top_p": 0.8,
                    "top_k": 40,
                    "max_output_tokens": 200,
                }
            )

            # Clean up the response
            poem = response.text.strip()
            lines = [line.strip() for line in poem.split('\n') if line.strip()]

            # Ensure exactly 3 lines
            if len(lines) != 3:
                logger.warning(f"Generated poem had {len(lines)} lines, expected 3")
                # Take first 3 lines or pad with simple lines if less
                while len(lines) < 3:
                    lines.append(f"And life goes on, day by day")
                lines = lines[:3]

            return '\n'.join(lines)

        except Exception as e:
            logger.error(f"Error in generate_initial_poem: {str(e)}")
            # Fallback poem generation
            if self.emotion.lower() == 'joy':
                return "This is a sample poem as you cant refine the same poem two times using the same prompt\nSunshine fills the air,\nHappy birds all sing with flair,\nJoyful day is here."
            elif self.emotion.lower() == 'sadness':
                return "This is a sample poem as you cant refine the same poem two times using the same prompt\nWorld feels dark and gray,\nHeavy heart, a tear does stray,\nSadness fills the day."
            else:  # anger
                return "This is a sample poem as you cant refine the same poem two times using the same prompt\nDay went so wrong, see?\nFrustration burns, a bitter spree,\nAnger boils in me."

    def mutate_poem(self, individual):
        """
        Mutate a poem and store the scores of the mutated poem in the database.
        """
        prompt = f"""
        Slightly modify this poem while:
        1. Maintaining the emotion of {self.emotion}
        2. Keeping the same rhyme scheme
        3. Preserving syllable count per line
        4. Enhancing emotional impact

        Original poem:
        {individual}
        """
        response = model.generate_content(prompt)
        mutated_poem = response.text.strip()

        # Evaluate the mutated poem
        emotion_score, rhyme_score, syllable_score = self.evaluate(mutated_poem, is_mutated=True)

        return mutated_poem,

    def setup_evolutionary_operators(self):
        """Setup genetic operators"""
        self.toolbox.register("individual", self.generate_initial_poem)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("evaluate", self.evaluate)
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("mutate", self.mutate_poem)
        self.toolbox.register("select", tools.selNSGA2)

    def evolve_poem(self, situation, generations=3):
        """
        Evolve poems over multiple generations using NSGA-II for multi-objective optimization.
        First generation: Generate 5 poems and store their evaluations.
        Next 2 generations: Mutate and evaluate only the mutated poems.
        Final selection: Evaluate the best poem after mutation.
        """
        try:
            # Initialize population
            logger.info("Initializing population...")
            pop = []
            for _ in range(5):  # Generate 5 initial poems
                try:
                    poem = self.generate_initial_poem(situation)
                    if poem:
                        pop.append(poem)
                        # Evaluate and store scores for the initial poem
                        self.evaluate(poem)
                except Exception as e:
                    logger.error(f"Error generating initial poem: {str(e)}")
                    continue

            if not pop:
                logger.error("Failed to generate initial population")
                return self.generate_initial_poem(situation)  # Fallback to single poem

            # Evolution loop for mutations (next 2 generations)
            logger.info("Starting evolution...")
            for gen in range(1, generations):
                self.generation = gen + 1  # Track the current generation
                logger.info(f"Generation {self.generation}/{generations}")

                new_population = []
                for poem in pop:
                    try:
                        mutated = self.mutate_poem(poem)[0]  # Mutate and store in the database
                        if mutated:
                            new_population.append(mutated)
                    except Exception as e:
                        logger.error(f"Error mutating poem: {str(e)}")

                pop = new_population  # Only keep mutated poems

            # Final selection of the best poem
            logger.info("Selecting best poem...")
            best_poem = max(pop, key=lambda x: sum(self.evaluate(x)))  # Evaluate the best poem

            # Store the final best poem explicitly
            self.evaluate(best_poem, is_mutated=True)
            return best_poem

        except Exception as e:
            logger.error(f"Error in evolve_poem: {str(e)}")
            return self.generate_initial_poem(situation)  # Fallback to single poem

    def refine_poem(self, poem, emotion):
        """
        Refine a given poem for 2 iterations and store the evaluations of the mutated poems.
        """
        logger.info("Refining poem...")
        refined_poem = poem

        for _ in range(2):  # Two refinement iterations
            try:
                refined_poem = self.mutate_poem(refined_poem)[0]  # Mutate and store in the database
            except Exception as e:
                logger.error(f"Error refining poem: {str(e)}")

        return refined_poem

@app.route('/generate', methods=['POST'])
def generate_poem():
    try:
        situation = request.json['situation']
        emotion = request.json.get('emotion', 'joy')

        logger.info(f"Generating poem for situation: {situation}, emotion: {emotion}")

        evolver = PoemEvolver(emotion=emotion)
        # Reduce generations for faster response
        poem = evolver.evolve_poem(situation, generations=3)

        if not poem:
            logger.error("Failed to generate poem")
            return jsonify({'error': 'Failed to generate poem'}), 500

        logger.info("Successfully generated poem")
        return jsonify({'poem': poem})

    except Exception as e:
        logger.error(f"Error generating poem: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/refine', methods=['POST'])
def refine_poem():
    try:
        situation = request.json['situation']
        current_poem = request.json['current_poem']
        feedback = request.json['feedback']
        emotion = request.json.get('emotion', 'joy')

        logger.info(f"Refining poem with feedback for emotion: {emotion}")

        # Convert feedback text to a numerical score using sentiment analysis
        feedback_analysis = TextBlob(feedback)
        feedback_score = feedback_analysis.sentiment.polarity

        evolver = PoemEvolver(emotion=emotion)
        evolver.incorporate_feedback(feedback_score)  # Incorporate feedback

        # Use the feedback to guide the next generation
        refined_poem = evolver.refine_poem(current_poem, emotion)

        if not refined_poem:
            logger.error("Failed to refine poem")
            return jsonify({'error': 'Failed to refine poem'}), 500

        logger.info("Successfully refined poem")
        return jsonify({'poem': refined_poem})

    except Exception as e:
        logger.error(f"Error refining poem: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
