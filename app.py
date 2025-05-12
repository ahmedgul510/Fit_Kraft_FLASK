from flask import Flask, request, jsonify
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import requests
import joblib  # Add this import for loading the pickle file
from groq import Groq
import os
import json
import math # To check for NaN
from datetime import datetime, timedelta, timezone
from models.dietery import dietery_bp

app = Flask(__name__)

CSV_FILE_PATH = "Gym Exercises Dataset (1).csv" # Make sure this path is correct
# Load both models when the app starts
bmi_model = joblib.load('BMI_model.pkl')
mental_model = joblib.load('MENTAL_SCORE.pkl')




#This is code for workout exercise AI 


# --- Function to get all equipment types from CSV for default ---
def get_all_equipment_types_from_csv(file_path):
    try:
        df = pd.read_csv(file_path)
        # Get unique, non-null, stripped string values from 'Equipment' column
        unique_equipment_series = df['Equipment'].dropna().astype(str).str.strip()
        # Filter out any empty strings that might have resulted after stripping
        all_equipment = list(unique_equipment_series[unique_equipment_series != ''].unique())
        
        # Ensure "Bodyweight" is always included in the default list if not already present
        # (case-insensitive check)
        if not any(eq.lower() == 'bodyweight' for eq in all_equipment):
            all_equipment.append("Bodyweight")
            
        app.logger.info(f"Dynamically loaded default equipment types: {all_equipment}")
        return all_equipment
    except FileNotFoundError:
        app.logger.error(f"CRITICAL: CSV file not found at '{file_path}' for initializing default equipment. Using a hardcoded fallback list.")
        # Provide a comprehensive fallback list if CSV is missing at startup
        return [
            "Bodyweight", "Dumbbells", "Barbell", "Kettlebells", "Resistance Bands",
            "Machine", "Cable", "Pull-up bar", "Bench", "Medicine Ball", "Exercise Ball", "Other"
        ]
    except Exception as e:
        app.logger.error(f"CRITICAL: Could not load equipment from CSV due to {e}. Using a hardcoded fallback list.")
        return [
            "Bodyweight", "Dumbbells", "Barbell", "Kettlebells", "Resistance Bands",
            "Machine", "Cable", "Pull-up bar", "Bench", "Medicine Ball", "Exercise Ball", "Other"
        ]

# --- Default Planning Parameters (Updated) ---
DEFAULT_AVAILABLE_DAYS = 5
DEFAULT_TIME_PER_SESSION_MINUTES = 60
# DEFAULT_AVAILABLE_EQUIPMENT will be populated when the app starts
DEFAULT_AVAILABLE_EQUIPMENT = [] # Initialize and populate below

# Load API key securely from environment variable for Groq
try:
    if "GROQ_API_KEY" not in os.environ:
        raise KeyError("GROQ_API_KEY environment variable not set.")
    groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    print("Groq client configured successfully.")
    # Populate default equipment list at startup after logger is available
    with app.app_context():
        DEFAULT_AVAILABLE_EQUIPMENT = get_all_equipment_types_from_csv(CSV_FILE_PATH)
    print(f"Initialized with DEFAULT_AVAILABLE_EQUIPMENT: {DEFAULT_AVAILABLE_EQUIPMENT}")

except KeyError as e:
    print(f"CRITICAL ERROR: {e}. Please set GROQ_API_KEY.")
    groq_client = None 
except Exception as e:
    print(f"Error configuring Groq client or loading default equipment: {e}")
    groq_client = None
    # Fallback if CSV loading fails during client setup as well
    if not DEFAULT_AVAILABLE_EQUIPMENT:
        DEFAULT_AVAILABLE_EQUIPMENT = [
            "Bodyweight", "Dumbbells", "Barbell", "Kettlebells", "Resistance Bands",
            "Machine", "Cable", "Pull-up bar", "Bench", "Medicine Ball", "Exercise Ball", "Other"
        ]
        print(f"Using fallback DEFAULT_AVAILABLE_EQUIPMENT: {DEFAULT_AVAILABLE_EQUIPMENT}")


# --- Helper Function to Load and Filter Exercises ---
def load_and_filter_exercises(file_path, user_equipment_list):
    try:
        df = pd.read_csv(file_path)
        df['Equipment'] = df['Equipment'].fillna('Bodyweight')
        
        # Ensure user_equipment_list contains strings and is not None
        if user_equipment_list is None:
            user_equipment_list = []
            
        user_equipment_lower = {str(eq).lower() for eq in user_equipment_list if isinstance(eq, (str,int,float))} # handle potential non-strings
        user_equipment_lower.add('bodyweight')

        def check_equipment(required_eq):
            required_lower = str(required_eq).lower()
            if required_lower == 'bodyweight': return True
            for user_eq in user_equipment_lower:
                if user_eq in required_lower: return True
            # Handling for 'Other' if it implies general availability
            if 'other' in required_lower and len(user_equipment_lower) > 1 : # if user has more than just bodyweight
                 return True
            return False

        filtered_df = df[df['Equipment'].apply(check_equipment)].copy()
        exercise_list_for_prompt = []
        for _, row in filtered_df.iterrows():
            exercise_list_for_prompt.append({
                "name": row['Exercise_Name'],
                "muscle_group": row['muscle_gp'],
                "equipment": row['Equipment']
            })
        MAX_EXERCISES_IN_PROMPT = 70 # Reduced slightly due to increased complexity of history/output
        if len(exercise_list_for_prompt) > MAX_EXERCISES_IN_PROMPT:
            app.logger.warning(f"Exercise list is long ({len(exercise_list_for_prompt)}). Truncating to {MAX_EXERCISES_IN_PROMPT}.")
            exercise_list_for_prompt = exercise_list_for_prompt[:MAX_EXERCISES_IN_PROMPT]
        return exercise_list_for_prompt
    except FileNotFoundError:
        app.logger.error(f"Exercise CSV file not found at {file_path}")
        return None
    except Exception as e:
        app.logger.error(f"Error loading/filtering exercises: {e}")
        return None

# --- Helper Function to Preprocess Last Week's Workout History for LLM ---
def preprocess_history_for_llm(last_week_workout_input):
    processed_history = []
    if not isinstance(last_week_workout_input, list) or not last_week_workout_input:
        return processed_history

    try:
        for entry in last_week_workout_input:
            if isinstance(entry.get("date"), str):
                # Attempt to parse, handling potential presence of 'Z'
                date_str = entry["date"].replace("Z", "")
                if '.' in date_str: # Handle milliseconds if present
                     date_str = date_str.split('.')[0]
                entry["parsed_date"] = datetime.fromisoformat(date_str).replace(tzinfo=timezone.utc)

            elif isinstance(entry.get("date"), datetime):
                 entry["parsed_date"] = entry.get("date") # Already a datetime object
            else:
                entry["parsed_date"] = datetime.min.replace(tzinfo=timezone.utc)
        last_week_workout_input.sort(key=lambda x: x["parsed_date"], reverse=True)
    except Exception as e:
        app.logger.warning(f"Could not sort workout history by date due to: {e}. Using original order.")


    for day_workout in last_week_workout_input[:7]: 
        day_exercises_for_llm = []
        if not isinstance(day_workout.get("exercises"), list):
            continue
        for ex_hist in day_workout["exercises"]:
            if not isinstance(ex_hist, dict) or "name" not in ex_hist:
                continue
            
            llm_ex_hist = {
                "name": ex_hist.get("name"),
                "type": ex_hist.get("type", "Unknown"),
                "duration": ex_hist.get("duration", {"minutes": 0, "seconds": 0}),
                "sets": ex_hist.get("sets") if isinstance(ex_hist.get("sets"), int) else None,
                "reps": ex_hist.get("reps") if isinstance(ex_hist.get("reps"), int) else None,
                "weight": ex_hist.get("weight") if isinstance(ex_hist.get("weight"), (int, float)) else None,
                "completed": ex_hist.get("completed", False)
            }
            if llm_ex_hist["reps"] is None and \
               isinstance(llm_ex_hist["duration"], dict) and \
               (llm_ex_hist["duration"].get("minutes",0) > 0 or llm_ex_hist["duration"].get("seconds",0) > 0):
                llm_ex_hist["reps"] = 1 # Convention for timed exercises
            
            day_exercises_for_llm.append(llm_ex_hist)

        if day_exercises_for_llm:
            history_date_str = "Unknown Past Day"
            if isinstance(day_workout.get("parsed_date"), datetime):
                history_date_str = day_workout["parsed_date"].strftime("%Y-%m-%d (%A)")
            processed_history.append({
                "day_identifier": history_date_str,
                "exercises": day_exercises_for_llm
            })
    return processed_history

# --- Helper Function to Infer Experience Level ---
def infer_experience_level(activity_level_str):
    if not activity_level_str or not isinstance(activity_level_str, str):
        return "Beginner" 
    
    activity_level_lower = activity_level_str.lower()
    if "sedentary" in activity_level_lower or "lightly active" in activity_level_lower:
        return "Beginner"
    elif "moderately active" in activity_level_lower:
        return "Intermediate"
    elif "very active" in activity_level_lower or "extremely active" in activity_level_lower:
        return "Advanced"
    return "Beginner"

@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/api/predict/BMI', methods=['POST'])
def predict_bmi():
    try:
        # Get data from request
        data = request.get_json()
        
        # Convert to DataFrame
        input_data = pd.DataFrame({
            'Weight': [float(data['Weight'])],
            'Height': [float(data['Height'])],
            'Age': [float(data['Age'])],
            'Gender': [data['Gender']]
        })
        
        # Convert gender to binary (0 for Female, 1 for Male)
        input_data['Gender_Female'] = (input_data['Gender'].str.lower() == 'female').astype(int)
        input_data['Gender_Male'] = (input_data['Gender'].str.lower() == 'male').astype(int)
        
        # Drop original gender column
        input_data = input_data.drop('Gender', axis=1)
        
        # Make prediction using the loaded model
        prediction = bmi_model.predict(input_data)[0]
        print(prediction)
        
        # Return prediction in a variable named predicted_value
        return jsonify({
            "predicted_value": float(prediction)
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 400

@app.route('/api/predict/mental_score', methods=['POST'])
def predict_mental_score():
    try:
        # Get data from request
        data = request.get_json()
        
        # Convert to DataFrame
        input_data = pd.DataFrame({
            'Age': [float(data['Age'])],
            'Mood_Score': [float(data['Mood_Score'])],
            'Physical_Activity': [float(data['Physical_Activity'])],
            'Stress_Level': [float(data['Stress_Level'])]
        })
        
        # Make prediction using the mental health model
        prediction = mental_model.predict(input_data)[0]
        
        # Return prediction in a variable named predicted_value
        return jsonify({
            "AI-Detected_Emotional_State": str(prediction)
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 400
        

@app.route('/generate-workout', methods=['POST'])
def generate_workout_route():
    if not groq_client:
        return jsonify({"error": "Groq client not configured. Check server logs."}), 500

    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid JSON payload"}), 400

    user_data = data.get('user')
    last_week_workout_history_input = data.get('lastWeekWorkout', [])

    if not user_data or not isinstance(user_data, dict):
        return jsonify({"error": "Missing or invalid 'user' object in request"}), 400
    
    user_id_for_logging = user_data.get('email') or user_data.get('name', 'UnknownUser')

    # --- Determine Planning Parameters ---
    inferred_experience_level = infer_experience_level(user_data.get('activityLevel'))
    
    # Use defaults (now updated) if these are not provided in the request at the top level
    available_days_for_new_plan = data.get('availableDaysForNewPlan', DEFAULT_AVAILABLE_DAYS)
    time_per_session_minutes = data.get('timePerSessionMinutes', DEFAULT_TIME_PER_SESSION_MINUTES)
    # If 'availableEquipment' is sent in the request, use it, otherwise use the new dynamic default
    available_equipment = data.get('availableEquipment', DEFAULT_AVAILABLE_EQUIPMENT)


    user_profile_for_llm = {
        "goal": user_data.get('goal', "General Fitness"),
        "experience_level": inferred_experience_level,
        "available_days": available_days_for_new_plan,
        "time_per_session_minutes": time_per_session_minutes,
        "available_equipment": available_equipment,
        "age": user_data.get('age'),
        "gender": user_data.get('gender'),
        "context_summary": f"User Age: {user_data.get('age', 'N/A')}, Gender: {user_data.get('gender', 'N/A')}. " +
                           f"Weight: {user_data.get('weight', 'N/A')}kg, Height: {user_data.get('height', 'N/A')}cm. " +
                           f"Activity Level: {user_data.get('activityLevel', 'N/A')}. " +
                           (f"BMI: {user_data.get('bmi'):.1f}, " if isinstance(user_data.get('bmi'), (int, float)) else "") +
                           (f"BMR: {user_data.get('bmr')} kcal." if isinstance(user_data.get('bmr'), (int, float)) else "")
    }

    is_new_user_plan = not last_week_workout_history_input
    llm_history_prompt_data = []
    if not is_new_user_plan:
        llm_history_prompt_data = preprocess_history_for_llm(last_week_workout_history_input)

    current_server_date = datetime.now(timezone.utc)
    days_until_monday = (0 - current_server_date.weekday() + 7) % 7
    if days_until_monday == 0 and current_server_date.weekday() !=0 : days_until_monday = 7
    elif days_until_monday == 0 and current_server_date.weekday() == 0: days_until_monday = 0
    first_workout_date = current_server_date + timedelta(days=days_until_monday)
    first_workout_date_str = first_workout_date.strftime("%Y-%m-%d")
    app.logger.info(f"New plan for {user_id_for_logging} starts {first_workout_date_str}. New user plan: {is_new_user_plan}")

    available_exercises_for_llm = load_and_filter_exercises(CSV_FILE_PATH, user_profile_for_llm["available_equipment"])
    if available_exercises_for_llm is None:
        return jsonify({"error": "Could not load or filter exercises from CSV"}), 500
    if not available_exercises_for_llm:
        return jsonify({"error": f"No exercises found in CSV for equipment: {user_profile_for_llm['available_equipment']}"}), 400

    # --- Construct the Prompt (Dynamically adjust for new/existing user) ---
    history_section_for_prompt = ""
    progression_instructions = ""
    if not is_new_user_plan and llm_history_prompt_data:
        history_section_for_prompt = f"""
# **LAST WEEK'S WORKOUT HISTORY (for context and progression - note the structure):**
{json.dumps(llm_history_prompt_data, indent=2)}"""
        progression_instructions = """
# 5.  Progressive Overload & Variation (using "LAST WEEK'S WORKOUT HISTORY"):
#     - For exercises repeated from last week, if performance data (sets/reps/weight) is available, suggest a slight increase in 'weight' or 'reps' if appropriate for the goal.
#     - Introduce some new accessory exercises or vary some from last week.
"""
    else: 
        history_section_for_prompt = "# **No prior workout history provided. Generating a foundational plan.**"
        progression_instructions = """
# 5.  Since no prior history is provided or for a fresh start:
#     - Focus on establishing good form with moderate weights/intensity.
#     - Include a balanced mix of compound and accessory exercises suitable for the user's experience level and goals.
"""

    user_prompt_content = f"""
# GOAL: Create a personalized {user_profile_for_llm['available_days']}-day weekly workout plan.
# Output MUST be a single valid JSON object with a top-level key "workout_schedule".
# Do NOT include "estimated_total_workout_duration", "estimated_calories_burned", or "overall_notes".

# USER PROFILE (for the new plan):
# Goal: {user_profile_for_llm['goal']}
# Experience Level: {user_profile_for_llm['experience_level']}
# Available Days for New Plan: {user_profile_for_llm['available_days']}
# Available Equipment for New Plan: {', '.join(user_profile_for_llm['available_equipment'])}
# User Context: {user_profile_for_llm['context_summary']}
{history_section_for_prompt} # This variable holds the formatted history or "no history" message

# AVAILABLE EXERCISES (for the new plan - use ONLY exercises from this list):
# Each object has "name", "muscle_group", "equipment".
{json.dumps(available_exercises_for_llm, indent=2)}

# INSTRUCTIONS FOR LLM:
# 1.  Create a workout plan for exactly {user_profile_for_llm['available_days']} distinct workout days.
# 2.  The top-level JSON output MUST be an object with a single key "workout_schedule", which is an array of daily workout objects.
# 3.  Each daily workout object MUST follow this structure:
#     - "date": String (in "YYYY-MM-DD" format. The first workout day should be {first_workout_date_str}. Assign subsequent dates logically).
#     - "exercises": Array of exercise objects for that day. This array MUST NOT be empty.
# 4.  Each exercise object within the "exercises" array MUST be a valid JSON object and strictly follow this exact structure with ALL specified keys present:
#     - "name": String (MUST be chosen from the "AVAILABLE EXERCISES" list and match user's equipment).
#     - "type": String (e.g., "Strength/Compound", "Strength/Isolation", "Core/Isometric", "Cardio").
#     - "duration": Object. This object MUST contain two keys: "minutes": Number, and "seconds": Number.
#         - For set/rep based exercises (like most strength training): "duration" MUST be `{{"minutes": 0, "seconds": 0}}`.
#         - For timed exercises (e.g., Plank, Cycling): "duration" MUST reflect the actual time (e.g., for a 60 second plank, `{{"minutes": 1, "seconds": 0}}`).
#     - "sets": Number (The key "sets" must always be present. For exercises where sets are not strictly applicable, like continuous cardio or some timed holds, use `1` or `null` as the value).
#     - "reps": Number (MUST be a single integer. The key "reps" must always be present. For timed exercises where 'duration' holds the time, 'reps' should be `1`. For exercises where reps are not strictly applicable, like continuous cardio, use `1` or `null` as the value).
#     - "weight": Number (in kg. The key "weight" must always be present. Use `0` for bodyweight exercises. For exercises where weight is not applicable (e.g. some cardio, some bodyweight), use `null` as the value).
#     - "completed": Boolean (Always set to `false`).
{progression_instructions} # This variable holds progression rules
# 6.  Do NOT include "overall_notes", "estimated_calories_burned", or "estimated_total_workout_duration".
# 7.  Ensure correct JSON syntax: pay extreme attention to commas between array elements and object properties, and correct use of curly braces {{}} for objects and square brackets [] for arrays.

# EXAMPLE OF THE DESIRED TOP-LEVEL JSON OUTPUT STRUCTURE (showing correct exercise structure):
# {{
#   "workout_schedule": [
#     {{
#       "date": "{first_workout_date_str}",
#       "exercises": [
#         {{
#           "name": "Barbell Bench Press", "type": "Strength/Compound",
#           "duration": {{"minutes": 0, "seconds": 0}}, "sets": 3, "reps": 8, "weight": 75, "completed": false
#         }},
#         {{
#           "name": "Cycling", "type": "Cardio",
#           "duration": {{"minutes": 20, "seconds": 0}}, "sets": 1, "reps": 1, "weight": null, "completed": false
#         }},
#         {{
#           "name": "Plank", "type": "Core/Isometric",
#           "duration": {{"minutes": 1, "seconds": 0}}, "sets": 3, "reps": 1, "weight": 0, "completed": false
#         }}
#       ]
#     }}
#     // ... continue for all {user_profile_for_llm['available_days']} days
#   ]
# }}

# GENERATE THE WORKOUT PLAN:
"""

    system_prompt_behavior = "You are an AI Fitness Coach."
    if not is_new_user_plan and llm_history_prompt_data: # is_new_user_plan logic should be in Python
        system_prompt_behavior += " You MUST use the provided 'LAST WEEK'S WORKOUT HISTORY' for progression and variation."
    else:
         system_prompt_behavior += " You are generating a foundational plan."

    system_prompt = f"{system_prompt_behavior} Generate a {user_profile_for_llm['available_days']}-day workout plan. Output MUST be a single, perfectly valid JSON object with a root key 'workout_schedule'. Each item in 'workout_schedule' is a day and MUST have 'date' (YYYY-MM-DD string, starting {first_workout_date_str}, increment logically) and an 'exercises' array. Each exercise object MUST contain ALL of the following keys: 'name'(String), 'type'(String), 'duration'(Object: {{\"minutes\":Number,\"seconds\":Number}}), 'sets'(Number or null), 'reps'(Number or null), 'weight'(Number or null), 'completed'(Boolean:false). For timed exercises, 'reps' is 1 and 'duration' object contains the time. For set/rep exercises, 'duration' object is {{\"minutes\":0,\"seconds\":0}}. The keys 'sets', 'reps', and 'weight' must ALWAYS be present, using null if not applicable. NO 'overall_notes', 'estimated_calories_burned', 'estimated_total_workout_duration'. Adhere STRICTLY to JSON syntax, especially commas and nesting."

   # --- Call the Groq API with Retry Logic ---
    attempts = 0
    MAX_LLM_RETRIES = 3
    plan_data = None
    json_response_text_for_error = "" # To store the text for error reporting if all retries fail

    while attempts < MAX_LLM_RETRIES:
        attempts += 1
        try:
            app.logger.info(f"Attempt {attempts}/{MAX_LLM_RETRIES}: Sending request to Groq API for userId {user_id_for_logging}...")
            
            # Add a specific instruction for retry if it's not the first attempt
            current_user_prompt = user_prompt_content
            if attempts > 1:
                retry_instruction = "\n# IMPORTANT RETRY NOTE: Your previous attempt to generate JSON was invalid. Please pay EXTREME attention to all JSON formatting rules, especially commas within arrays and the exact structure of each exercise object, including the nested 'duration' object. Ensure all specified keys like 'sets', 'reps', 'weight' are present even if their value is null for certain exercise types.\n"
                current_user_prompt = retry_instruction + user_prompt_content


            chat_completion = groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": current_user_prompt} # Use potentially modified prompt
                ],
                model="llama3-70b-8192",
                response_format={"type": "json_object"},
                temperature=0.3, # Keep low for structure adherence
            )
            
            json_response_text_for_error = chat_completion.choices[0].message.content
            plan_data = json.loads(json_response_text_for_error) # Attempt to parse
            app.logger.info(f"Attempt {attempts}: Successfully parsed JSON from Groq for userId {user_id_for_logging}.")
            
            # Perform basic structural validation immediately after parsing
            is_valid_format = True
            validation_errors_on_attempt = []
            expected_days_val = user_profile_for_llm["available_days"]

            if not isinstance(plan_data, dict) or "workout_schedule" not in plan_data or \
               not isinstance(plan_data["workout_schedule"], list):
                is_valid_format = False; validation_errors_on_attempt.append("Invalid top-level structure.")
            elif len(plan_data.get("workout_schedule", [])) != expected_days_val:
                 is_valid_format = False; validation_errors_on_attempt.append(f"Incorrect number of days. Expected {expected_days_val}, got {len(plan_data.get('workout_schedule', []))}.")
            else: # Basic check on first day's first exercise structure if plan is not empty
                if plan_data["workout_schedule"] and plan_data["workout_schedule"][0].get("exercises"):
                    first_ex = plan_data["workout_schedule"][0]["exercises"][0] if plan_data["workout_schedule"][0]["exercises"] else {}
                    if not all(k in first_ex for k in ["name", "type", "duration", "sets", "reps", "weight", "completed"]) or \
                       not isinstance(first_ex.get("duration"), dict):
                        is_valid_format = False; validation_errors_on_attempt.append("First exercise object has incorrect structure.")
            
            if is_valid_format:
                break # Successful parse and basic structure looks okay, exit retry loop
            else:
                app.logger.warning(f"Attempt {attempts}: JSON parsed but failed structural validation: {validation_errors_on_attempt}. Raw: {json_response_text_for_error}")
                if attempts < MAX_LLM_RETRIES:
                    app.logger.info("Retrying due to structural validation failure...")
                    time.sleep(1) # Wait a bit before retrying
                    continue
                else:
                    # This means max retries reached, and the last attempt also failed structural validation
                    return jsonify({
                        "error": "LLM response failed structural validation after multiple attempts.",
                        "details": validation_errors_on_attempt,
                        "raw_response": json.loads(json_response_text_for_error) # It was parsable but not valid structure
                    }), 422


        except APIStatusError as e: # Groq's specific error for API issues (like 400, 429, 500 from Groq)
            app.logger.warning(f"Attempt {attempts}: Groq APIStatusError: Status {e.status_code} - Body: {e.response.text}")
            error_details = {}
            try:
                error_details = e.response.json().get('error', {})
            except: # If error response itself isn't JSON
                error_details = {"message": e.response.text}

            if e.status_code == 400 and error_details.get('code') == 'json_validate_failed':
                app.logger.warning(f"Attempt {attempts}: json_validate_failed by Groq. Failed generation: {error_details.get('failed_generation')}")
                json_response_text_for_error = error_details.get('failed_generation', e.response.text) # Store the malformed JSON
                if attempts < MAX_LLM_RETRIES:
                    app.logger.info("Retrying due to json_validate_failed...")
                    time.sleep(1) 
                    continue 
                else:
                    app.logger.error("Max retries reached for json_validate_failed by Groq.")
                    return jsonify({"error": "LLM failed to generate valid JSON after multiple attempts (Groq validation).", 
                                    "details": error_details}), 400
            else: # Other Groq API errors
                app.logger.error(f"Attempt {attempts}: Non-retryable or different Groq APIStatusError: {e}")
                return jsonify({"error": f"Groq API error: {e.status_code}", "details": error_details}), e.status_code

        except json.JSONDecodeError as e: # If Groq returns text that isn't even attemptable JSON
            app.logger.warning(f"Attempt {attempts}: Failed to decode Groq response text as JSON. Raw: {json_response_text_for_error}. Error: {e}")
            if attempts < MAX_LLM_RETRIES:
                app.logger.info("Retrying due to JSONDecodeError...")
                time.sleep(1)
                continue
            else:
                app.logger.error(f"Max retries reached for JSONDecodeError. Raw response: {json_response_text_for_error}")
                return jsonify({"error": "LLM response was not valid JSON after multiple attempts.", 
                                "raw_response": json_response_text_for_error}), 500
        
        except Exception as e: # Catch other potential errors during the API call
            app.logger.error(f"Attempt {attempts}: An unexpected error occurred calling Groq API: {e}")
            if attempts < MAX_LLM_RETRIES:
                app.logger.info("Retrying due to unexpected error...")
                time.sleep(1)
                continue
            else: # Max retries for other errors
                return jsonify({"error": f"An unexpected error occurred with the LLM API after multiple attempts: {str(e)}"}), 500

    if plan_data is None: # Should have been caught by error returns inside loop
        app.logger.error(f"Failed to get valid plan_data for userId {user_id_for_logging} after all attempts.")
        return jsonify({"error": "Failed to generate workout plan after multiple attempts.", "raw_response_on_failure": json_response_text_for_error}), 500

    # --- If here, plan_data is successfully parsed and passed basic structural check ---
    # Proceed with your original, more detailed validation of the content if needed,
    # although the prompt is now very strict.
    # For this example, we assume if it passes the initial structural check in the retry loop, it's good to proceed.

    app.logger.info(f"Successfully generated and validated (structurally) workout for userId {user_id_for_logging}")
    return jsonify(plan_data), 200


app.register_blueprint(dietery_bp, url_prefix="/generate")



if __name__ == '__main__':
    if not os.path.exists(CSV_FILE_PATH):
        print(f"ERROR: CSV file not found at '{CSV_FILE_PATH}'. The Flask app might not work correctly.")
    if not groq_client:
        print(f"ERROR: Groq client failed to initialize. Check GROQ_API_KEY and logs.")
    
    # This block ensures default equipment is loaded using app context if script is run directly
    if not DEFAULT_AVAILABLE_EQUIPMENT and os.path.exists(CSV_FILE_PATH):
         with app.app_context():
            DEFAULT_AVAILABLE_EQUIPMENT = get_all_equipment_types_from_csv(CSV_FILE_PATH)
         print(f"Re-initialized DEFAULT_AVAILABLE_EQUIPMENT (main block): {DEFAULT_AVAILABLE_EQUIPMENT}")

    app.run(host='0.0.0.0', port=5000, debug=True)

# # Test code for mental score prediction
# url = 'http://127.0.0.1:5000/api/predict/mental_score'
# test_data = {
#     "Age": 25,
#     "Mood_Score": 7,
#     "Physical_Activity": 5,
#     "Stress_Level": 4
# }

# response = requests.post(url, json=test_data)
# print(response.json())