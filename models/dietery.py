import os
import json
import time
from flask import jsonify, request, Blueprint, current_app # MODIFIED: Added current_app
from groq import Groq, APIConnectionError, RateLimitError, APIStatusError, InternalServerError, APITimeoutError, APIError
from datetime import datetime, date

dietery_bp = Blueprint('mealplan', __name__)

# --- Groq Client Initialization ---
try:
    client = Groq(api_key=os.environ.get("GROQ_API_KEY_2"))
except Exception as e:
    print(f"Error initializing Groq client: {e}")
    print("Please ensure the GROQ_API_KEY_2 environment variable is set correctly.")
    client = None

# --- Retry Configuration ---
MAX_RETRIES = 5
INITIAL_RETRY_DELAY_SECONDS = 2
BACKOFF_FACTOR = 2
MAX_RETRY_DELAY_SECONDS = 60

# --- Groq Meal Plan Generation Logic ---

def construct_groq_prompt(user_profile, target_date_str):
    # This function now directly uses what's in user_profile from the request body
    prompt = f"""
    You are a helpful AI assistant specialized in creating Pakistani/Indian meal plans.
    Your task is to generate a one-day meal plan based on the following user profile and requirements.
    The output MUST be a single JSON object that strictly adheres to the specified schema structure.

    User Profile:
    - Name: {user_profile.get('name', 'N/A')}
    - Goal: {user_profile.get('goal', 'Not specified')}
    - Target Daily Calories: Approximately {user_profile.get('calories_intake', 2000)} kcal
    - Number of Meals: {user_profile.get('num_meals', 3)}
    - Allergies: {', '.join(user_profile.get('allergies', [])) if user_profile.get('allergies') else 'None'}
    - Preferred Foods: {', '.join(user_profile.get('preferred_foods', [])) if user_profile.get('preferred_foods') else 'Any suitable Pakistani/Indian dishes'}

    Date for the meal plan: {target_date_str}

    Output JSON Structure Requirements:
    The JSON output should represent a 'DieterySchema'.
    - 'UserId': This should be the ID of the user for whom the plan is generated (taken from input).
    - 'Date': The target date in "YYYY-MM-DD" format.
    - 'Day': The day of the week for the target date (e.g., "Monday").
    - 'Meals': An array of meal objects. Each meal object should follow the 'MealSchema' structure detailed below, plus an 'isCompleted' field set to false.
        - 'Name': Name of the dish (Pakistani/Indian).
        - 'Calories': Estimated calories (number).
        - 'Protein': Estimated protein in grams (number).
        - 'Carbs': Estimated carbohydrates in grams (number).
        - 'Fats': Estimated fats in grams (number).
        - 'Ingredients': An array of strings listing key ingredients and approximate quantities.
        - 'Instructions': Step-by-step cooking instructions.
        - 'Image': A placeholder string like "placeholder_image_url.png". You cannot generate images.
        - 'Category': Type of meal (e.g., "Breakfast", "Lunch", "Dinner", "Snack" - distribute meals according to 'num_meals').
        - 'UserCreated_ID': Set to "AI_GENERATED".
        - 'isCompleted': boolean, set to false.
    - 'TotalCalories': Sum of calories from all meals for the day. This should be close to the user's target.
    - 'TotalProtein': Sum of protein from all meals for the day (number).
    - 'TotalCarbs': Sum of carbs from all meals for the day (number).
    - 'TotalFats': Sum of fats from all meals for the day (number).

    Important Instructions for the AI:
    1.  Focus on Pakistani and Indian cuisine.
    2.  Distribute the total calories and meals throughout the day according to `num_meals`.
    3.  Ensure the generated meal plan considers the user's allergies and preferred foods.
    4.  Provide realistic ingredients and concise instructions.
    5.  The sum of calories for all meals should be as close as possible to the user's target daily calorie intake.
    6.  Calculate and provide the total protein, carbs, and fats for the entire day.
    7.  The output MUST be ONLY the JSON object, without any introductory text, comments, or explanations before or after the JSON.

    Now, generate the complete one-day meal plan JSON based on the user profile and the date {target_date_str}.
    """
    return prompt

def generate_meal_plan_for_user(user_id, user_profile, target_date=None):
    if not client:
        current_app.logger.error("Groq client not initialized.") # CHANGED from app.logger
        return None
    
    
    
    if target_date is None:
        target_date = date.today()
    target_date_str = target_date.isoformat()
    day_of_week = target_date.strftime("%A")

    prompt = construct_groq_prompt(user_profile, target_date_str)
    current_app.logger.info(f"Preparing to generate meal plan for user {user_id} (profile from request) for {target_date_str} ({day_of_week}).") # CHANGED

    current_retry_delay = INITIAL_RETRY_DELAY_SECONDS
    response_content = ""   

    for attempt in range(MAX_RETRIES):
        try:
            current_app.logger.info(f"Attempt {attempt + 1}/{MAX_RETRIES} to call Groq API for user {user_id}.") # CHANGED
            chat_completion = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama3-70b-8192",
                temperature=0.7,
                response_format={"type": "json_object"}
            )
            response_content = chat_completion.choices[0].message.content
            
            meal_plan_data = json.loads(response_content)
            current_app.logger.info(f"Groq API call and JSON parsing successful on attempt {attempt + 1} for user {user_id}.") # CHANGED

            meal_plan_data['UserId'] = user_id
            meal_plan_data['Date'] = target_date_str
            meal_plan_data['Day'] = day_of_week

            if 'Meals' in meal_plan_data and isinstance(meal_plan_data['Meals'], list):
                for meal in meal_plan_data['Meals']:
                    if isinstance(meal, dict):
                        meal.setdefault('isCompleted', False)
                        meal.setdefault('UserCreated_ID', "AI_GENERATED")
                        meal.setdefault('Image', "placeholder_image_url.png")
                    else:
                        current_app.logger.warning(f"Invalid meal item found in LLM response for user {user_id}: {meal}") # CHANGED
            else:
                current_app.logger.warning(f"'Meals' key is missing, not a list, or invalid in LLM response for user {user_id}. Setting to empty list.") # CHANGED
                meal_plan_data['Meals'] = []
            
            expected_totals = ['TotalCalories', 'TotalProtein', 'TotalCarbs', 'TotalFats']
            recalculate = False
            if not all(k in meal_plan_data for k in expected_totals) or \
               not all(isinstance(meal_plan_data.get(k), (int, float)) for k in expected_totals if k in meal_plan_data):
                current_app.logger.info(f"Successfully processed meal plan for user1 {user_id}.") # CHANGED
                recalculate = True
            
            if recalculate:
                total_calories, total_protein, total_carbs, total_fats = 0, 0, 0, 0
                for meal_item in meal_plan_data.get('Meals', []): # Renamed meal to meal_item to avoid conflict
                    if isinstance(meal_item, dict):
                        total_calories += meal_item.get('Calories', 0)
                        total_protein += meal_item.get('Protein', 0)
                        total_carbs += meal_item.get('Carbs', 0)
                        total_fats += meal_item.get('Fats', 0)
                meal_plan_data['TotalCalories'] = total_calories
                meal_plan_data['TotalProtein'] = total_protein
                meal_plan_data['TotalCarbs'] = total_carbs
                meal_plan_data['TotalFats'] = total_fats
            
            current_app.logger.info(f"Successfully processed meal plan for user {user_id}.") # CHANGED
            return meal_plan_data

        except (APIConnectionError, RateLimitError, InternalServerError, APITimeoutError) as e:
            current_app.logger.warning(f"Attempt {attempt + 1}/{MAX_RETRIES} for user {user_id} failed (Groq API Error: {type(e).__name__} - {e}).")
        except APIStatusError as e:
            if e.status_code >= 500 or e.status_code == 429:
                current_app.logger.warning(f"Attempt {attempt + 1}/{MAX_RETRIES} for user {user_id} failed (API Status Error {e.status_code}: {e}).")
            else:
                current_app.logger.error(f"Groq API Status Error {e.status_code}: {e} for user {user_id}. This error will not be retried.")
                return None
        except json.JSONDecodeError as e:
            current_app.logger.warning(f"Attempt {attempt + 1}/{MAX_RETRIES} for user {user_id} failed to decode JSON: {e}. Raw response (first 200 chars): '{response_content[:200]}...'")
        except APIError as e:
            current_app.logger.warning(f"Attempt {attempt + 1}/{MAX_RETRIES} for user {user_id} failed (Generic Groq APIError: {type(e).__name__} - {e}).")
        except Exception as e:
            current_app.logger.error(f"Attempt {attempt + 1}/{MAX_RETRIES} for user {user_id} failed with an unexpected error: {type(e).__name__} - {e}.")

        if attempt < MAX_RETRIES - 1:
            current_app.logger.info(f"Retrying in {current_retry_delay}s for user {user_id}...")
            time.sleep(current_retry_delay)
            current_retry_delay = min(current_retry_delay * BACKOFF_FACTOR, MAX_RETRY_DELAY_SECONDS)
        else:
            current_app.logger.error(f"All {MAX_RETRIES} retries failed for user {user_id} after final attempt.")
            return None

    current_app.logger.error(f"Exited retry loop without success for user {user_id}. This indicates an issue in the loop logic.")
    return None


# --- Flask Routes ---
@dietery_bp.route('/mealplan', methods=['POST'])
def create_meal_plan_route():
    if not client:
        return jsonify({"error": "Groq client not configured"}), 500

    if not request.is_json:
        return jsonify({"error": "Invalid request: Content-Type must be application/json"}), 400

    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid request: No JSON data received or data is empty"}), 400

    # Extract user ID and profile details from request body
    user_id = data.get('userId')
    if not user_id:
        return jsonify({"error": "Missing required field in request body: userId"}), 400

    # Construct user_profile dictionary from request data
    # Providing defaults for optional fields if not present in the request
    user_profile = {
        'name': data.get('name'), # Optional
        'goal': data.get('goal'),
        'calories_intake': data.get('calories_intake'),
        'num_meals': data.get('num_meals'),
        'allergies': data.get('allergies', []), # Default to empty list if not provided
        'preferred_foods': data.get('preferred_foods', []) # Default to empty list
    }

    # Basic validation for essential profile fields for the prompt
    required_profile_fields = ['goal', 'calories_intake', 'num_meals']
    missing_fields = [field for field in required_profile_fields if user_profile.get(field) is None]
    if missing_fields:
        return jsonify({"error": f"Missing required profile fields in request body: {', '.join(missing_fields)}"}), 400
    
    # Type validation for calories_intake and num_meals
    try:
        user_profile['calories_intake'] = int(user_profile['calories_intake'])
        user_profile['num_meals'] = int(user_profile['num_meals'])
        if user_profile['calories_intake'] <= 0 or user_profile['num_meals'] <= 0:
            raise ValueError("Calories and number of meals must be positive.")
    except (ValueError, TypeError):
        return jsonify({"error": "Invalid type or value for 'calories_intake' or 'num_meals'. They must be positive integers."}), 400


    current_app.logger.info(f"Request received to generate meal plan for UserId: {user_id} with profile: {user_profile}")
    
    plan_date = date.today()
    dietary_plan = generate_meal_plan_for_user(user_id, user_profile, plan_date)

    if dietary_plan:
        return jsonify(dietary_plan), 200
    else:
        return jsonify({"error": "Failed to generate meal plan after multiple attempts or due to a non-retryable error."}), 500

