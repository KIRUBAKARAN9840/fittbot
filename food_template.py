from fastapi import APIRouter, HTTPException, Query, Depends
from fastapi.responses import StreamingResponse
from fastapi_limiter.depends import RateLimiter
from pydantic import BaseModel
import pytz, os, hashlib, orjson, re, json
from datetime import datetime
from sqlalchemy.orm import Session
from app.models.database import get_db
from app.models.deps import get_http, get_oai, get_mem
from app.models.fittbot_models import WeightJourney,Client,ClientTarget
import openai
from openai import OpenAI
import json, re, os, random, uuid, traceback

from app.fittbot_api.v1.client.client_api.chatbot.chatbot_services.llm_helpers import (
    PlainTextStreamFilter, oai_chat_stream, GENERAL_SYSTEM, TOP_K,
    build_messages, heuristic_confidence, gpt_extract_items, first_missing_quantity,OPENAI_MODEL,
    sse_json, sse_escape, gpt_small_route, _scale_macros, is_yes, is_no,is_fit_chat,
    has_action_verb, food_hits,ensure_per_unit_macros, is_fittbot_meta_query,normalize_food, explicit_log_command, STYLE_PLAN, is_plan_request,STYLE_CHAT_FORMAT,pretty_plan
)

router = APIRouter(prefix="/food_template", tags=["food_template"])

APP_ENV = os.getenv("APP_ENV", "prod")
TZNAME = os.getenv("TZ", "Asia/Kolkata")
IST = pytz.timezone(TZNAME)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

@router.get("/healthz")
async def healthz():
    return {"ok": True, "env": APP_ENV, "tz": TZNAME}

def _fetch_profile(db: Session, client_id: int):
    """Fetch complete client profile including weight journey and calorie targets"""
    try:
        # Get latest weight journey
        w = (
            db.query(WeightJourney)
            .where(WeightJourney.client_id == client_id)
            .order_by(WeightJourney.id.desc())
            .first()
        )
        
        current_weight = float(w.actual_weight) if w and w.actual_weight is not None else 70.0
        target_weight = float(w.target_weight) if w and w.target_weight is not None else 65.0
        
        weight_delta_text = None
        goal_type = "maintain"
        
        if current_weight is not None and target_weight is not None:
            diff = round(target_weight - current_weight, 1)
            if diff > 0:
                weight_delta_text = f"Gain {abs(diff)} kg (from {current_weight} → {target_weight})"
                goal_type = "weight_gain"
            elif diff < 0:
                weight_delta_text = f"Lose {abs(diff)} kg (from {current_weight} → {target_weight})"
                goal_type = "weight_loss"
            else:
                weight_delta_text = f"Maintain {current_weight} kg"
                goal_type = "maintain"
        
        # Get client details
        c = db.query(Client).where(Client.client_id == client_id).first()
        client_goal = (getattr(c, "goals", None) or getattr(c, "goal", None) or "muscle gain") if c else "muscle gain"
        lifestyle= c.lifestyle if c else "moderate"
        
        # Get calorie target
        ct = db.query(ClientTarget).where(ClientTarget.client_id == client_id).first()
        target_calories = float(ct.calories) if ct and ct.calories else 2000.0
        
        return {
            "client_id": client_id,
            "current_weight": current_weight,
            "target_weight": target_weight,
            "weight_delta_text": weight_delta_text,
            "client_goal": client_goal,
            "goal_type": goal_type,
            "target_calories": target_calories,
            "lifestyle": lifestyle,
            "days_per_week": 6,  # Mon–Sat
        }
    
    except Exception as e:
        print(f"Error fetching profile for client {client_id}: {e}")
        print(f"Profile fetch traceback: {traceback.format_exc()}")
        # Return default profile for testing
        return {
            "client_id": client_id,
            "current_weight": 70.0,
            "target_weight": 65.0,
            "weight_delta_text": "Lose 5.0 kg (from 70.0 → 65.0)",
            "client_goal": "weight loss",
            "goal_type": "weight_loss",
            "target_calories": 1800.0,
            "lifestyle": "moderate",
            "days_per_week": 6,
        }

def get_meal_template():
    """Get the base meal template structure"""
    return [
        {
            "id": "1",
            "title": "Early Morning Detox",
            "foodList": [],
            "timeRange": "6:00 am - 7:00 am",
            "itemsCount": 0
        },
        {
            "id": "2", 
            "title": "Pre-Breakfast / Pre-Meal Starter",
            "foodList": [],
            "timeRange": "7:30 am - 8:00 am",
            "itemsCount": 0
        },
        {
            "id": "3",
            "title": "Breakfast",
            "foodList": [],
            "timeRange": "8:00 am - 9:30 am",
            "itemsCount": 0
        },
        {
            "id": "4",
            "title": "Mid-Morning Snack", 
            "foodList": [],
            "timeRange": "10:30 am - 11:30 am",
            "itemsCount": 0
        },
        {
            "id": "5",
            "title": "Lunch",
            "foodList": [],
            "timeRange": "12:30 pm - 2:00 pm",
            "itemsCount": 0
        },
        {
            "id": "6",
            "title": "Evening Snack",
            "foodList": [],
            "timeRange": "4:00 pm - 6:00 pm", 
            "itemsCount": 0
        },
        {
            "id": "7",
            "title": "Dinner",
            "foodList": [],
            "timeRange": "7:00 pm - 8:30 pm",
            "itemsCount": 0
        }
    ]

def generate_food_id():
    """Generate a random food ID similar to the example format"""
    return f"{random.randint(100, 999)}-{int(datetime.now().timestamp() * 1000)}-{random.random()}"

def get_food_image_url(food_name):
    """Generate food image URL based on food name"""
    # Clean food name for URL
    clean_name = food_name.replace(' ', '+')
    return f"add_image{clean_name}.png"

def create_food_item(name, calories, protein, carbs, fat, quantity):
    """Create a food item in the required format with proper quantity"""
    return {
        "id": generate_food_id(),
        "fat": fat,
        "pic": get_food_image_url(name),
        "date": datetime.now().strftime("%Y-%m-%d"),
        "name": name,
        "carbs": carbs,
        "protein": protein,
        "calories": calories,
        "quantity": quantity,  # Now accepts full quantity string with units
        "timeAdded": ""  # Empty as requested
    }

def get_standard_quantity_for_food(food_name):
    """Return standard quantity with unit based on food type"""
    food_name_lower = food_name.lower()
    
    # Liquids - use ml
    if any(liquid in food_name_lower for liquid in ['water', 'juice', 'lassi', 'buttermilk', 'coconut water']):
        if 'water' in food_name_lower:
            return "250 ml"
        else:
            return "200 ml"
    
    # Hot beverages
    if any(drink in food_name_lower for drink in ['tea', 'coffee', 'chai']):
        return "150 ml"
    
    # Milk
    if 'milk' in food_name_lower:
        return "200 ml"
    
    # Bread items
    if any(bread in food_name_lower for bread in ['roti', 'chapati']):
        return "2 medium pieces"
    if any(bread in food_name_lower for bread in ['naan', 'kulcha']):
        return "1 medium piece"
    if 'paratha' in food_name_lower:
        return "1 medium piece"
    if any(bread in food_name_lower for bread in ['bread', 'toast']):
        return "2 slices"
    
    # Rice and grains
    if 'rice' in food_name_lower:
        return "150 grams"
    if any(grain in food_name_lower for grain in ['upma', 'poha', 'daliya']):
        return "150 grams"
    if 'oats' in food_name_lower:
        return "50 grams (dry)"
    if 'cornflakes' in food_name_lower:
        return "30 grams"
    
    # South Indian items
    if 'idli' in food_name_lower:
        return "3 pieces"
    if 'dosa' in food_name_lower:
        return "1 medium piece"
    if 'uttapam' in food_name_lower:
        return "1 piece"
    
    # Dal and curries
    if any(dal in food_name_lower for dal in ['dal', 'sambar', 'rasam']):
        return "150 grams"
    if 'curry' in food_name_lower or 'sabzi' in food_name_lower:
        return "100 grams"
    
    # Paneer dishes
    if 'paneer' in food_name_lower:
        return "100 grams"
    
    # Non-veg items
    if any(meat in food_name_lower for meat in ['chicken', 'mutton', 'fish']):
        return "100 grams"
    if 'egg' in food_name_lower:
        if 'boiled' in food_name_lower or 'fried' in food_name_lower:
            return "2 pieces"
        else:
            return "1 serving"
    
    # Vegetables
    if any(veg in food_name_lower for veg in ['vegetables', 'aloo', 'gobi', 'bhindi', 'palak']):
        return "100 grams"
    
    # Dairy
    if 'curd' in food_name_lower or 'raita' in food_name_lower:
        return "100 grams"
    
    # Fruits
    if any(fruit in food_name_lower for fruit in ['banana', 'apple', 'orange']):
        return "1 medium piece"
    if 'fruit' in food_name_lower:
        return "100 grams"
    
    # Nuts and snacks
    if any(nut in food_name_lower for nut in ['nuts', 'almonds', 'mixture', 'namkeen']):
        return "30 grams"
    
    # Chutneys and condiments
    if 'chutney' in food_name_lower:
        return "20 grams"
    if 'pickle' in food_name_lower:
        return "10 grams"
    
    # Papad and similar
    if 'papad' in food_name_lower:
        return "1 piece"
    
    # Salad
    if 'salad' in food_name_lower:
        return "80 grams"
    
    # Default fallback
    return "1 serving"

def calculate_meal_calories_distribution(target_calories):
    """Calculate calorie distribution across meal slots"""
    distributions = {
        "1": 0.02,  # Early Morning Detox - 2%
        "2": 0.05,  # Pre-Breakfast Starter - 5%
        "3": 0.25,  # Breakfast - 25%
        "4": 0.10,  # Mid-Morning Snack - 10%
        "5": 0.35,  # Lunch - 35%
        "6": 0.10,  # Evening Snack - 10%
        "7": 0.25   # Dinner - 25%
    }
    
    # Allow ±10% flexibility per slot
    slot_calories = {}
    for slot_id, percentage in distributions.items():
        base_calories = target_calories * percentage
        slot_calories[slot_id] = {
            "target": round(base_calories),
            "min": round(base_calories * 0.9),
            "max": round(base_calories * 1.1)
        }
    
    return slot_calories

def validate_and_adjust_calories(meal_data, target_calories, max_attempts=3):
    """Validate total calories and adjust if needed"""
    
    def calculate_total_calories(meal_data):
        total = 0
        for meal_slot in meal_data.get("meals", []):
            for food in meal_slot.get("foods", []):
                total += food.get("calories", 0)
        return total
    
    def scale_meal_calories(meal_data, scale_factor):
        """Scale all calories in the meal plan by a factor"""
        for meal_slot in meal_data.get("meals", []):
            for food in meal_slot.get("foods", []):
                food["calories"] = round(food["calories"] * scale_factor)
                food["protein"] = round(food["protein"] * scale_factor, 1)
                food["carbs"] = round(food["carbs"] * scale_factor, 1)
                food["fat"] = round(food["fat"] * scale_factor, 1)
        return meal_data
    
    current_calories = calculate_total_calories(meal_data)
    tolerance = target_calories * 0.13 # Allow 13% tolerance
    
    print(f"DEBUG: Current calories: {current_calories}, Target: {target_calories}")
    
    if abs(current_calories - target_calories) <= tolerance:
        print(f"DEBUG: Calories within tolerance")
        return meal_data
    
    # Calculate adjustment factor
    if current_calories > 0:
        scale_factor = target_calories / current_calories
        print(f"DEBUG: Scaling meals by factor: {scale_factor:.2f}")
        return scale_meal_calories(meal_data, scale_factor)
    
    return meal_data

def calculate_meal_calories_distribution(target_calories):
    """Calculate calorie distribution across meal slots"""
    distributions = {
        "1": 0.02,  # Early Morning Detox - 2%
        "2": 0.05,  # Pre-Breakfast Starter - 5%
        "3": 0.25,  # Breakfast - 25%
        "4": 0.10,  # Mid-Morning Snack - 10%
        "5": 0.35,  # Lunch - 35%
        "6": 0.10,  # Evening Snack - 10%
        "7": 0.25   # Dinner - 25%
    }
    
    # Allow ±10% flexibility per slot
    slot_calories = {}
    for slot_id, percentage in distributions.items():
        base_calories = target_calories * percentage
        slot_calories[slot_id] = {
            "target": round(base_calories),
            "min": round(base_calories * 0.9),
            "max": round(base_calories * 1.1)
        }
    
    return slot_calories

def validate_and_adjust_calories(meal_data, target_calories, max_attempts=3):
    """Validate total calories and adjust if needed"""
    
    def calculate_total_calories(meal_data):
        total = 0
        for meal_slot in meal_data.get("meals", []):
            for food in meal_slot.get("foods", []):
                total += food.get("calories", 0)
        return total
    
    def scale_meal_calories(meal_data, scale_factor):
        """Scale all calories in the meal plan by a factor"""
        for meal_slot in meal_data.get("meals", []):
            for food in meal_slot.get("foods", []):
                food["calories"] = round(food["calories"] * scale_factor)
                food["protein"] = round(food["protein"] * scale_factor, 1)
                food["carbs"] = round(food["carbs"] * scale_factor, 1)
                food["fat"] = round(food["fat"] * scale_factor, 1)
        return meal_data
    
    current_calories = calculate_total_calories(meal_data)
    tolerance = target_calories * 0.15  # Allow 15% tolerance
    
    print(f"DEBUG: Current calories: {current_calories}, Target: {target_calories}")
    
    if abs(current_calories - target_calories) <= tolerance:
        print(f"DEBUG: Calories within tolerance")
        return meal_data
    
    # Calculate adjustment factor
    if current_calories > 0:
        scale_factor = target_calories / current_calories
        print(f"DEBUG: Scaling meals by factor: {scale_factor:.2f}")
        return scale_meal_calories(meal_data, scale_factor)
    
    return meal_data

def generate_meal_plan_with_ai(profile, diet_type, cuisine_type, day_name, previous_meals=None):
    """Generate meal plan for a specific day using AI with accurate calorie control"""
    
    target_calories = profile['target_calories']
    slot_calories = calculate_meal_calories_distribution(target_calories)
    
    # Build detailed calorie requirements for each slot
    calorie_breakdown = "\n".join([
        f"Slot {slot_id} ({get_slot_name(slot_id)}): {info['target']} calories (range: {info['min']}-{info['max']})"
        for slot_id, info in slot_calories.items()
    ])
    
    # Define cuisine-specific context (keeping your existing logic)
    cuisine_context = {
        "north_indian": """
        Focus on North Indian cuisine including:
        - Breads: Roti, Chapati, Naan, Paratha, Kulcha
        - Rice dishes: Jeera rice, Pulao, Biryani
        - Dals: Dal tadka, Dal makhani, Rajma, Chana masala
        - Vegetables: Aloo gobi, Palak paneer, Butter paneer, Mixed vegetables
        - Snacks: Samosa, Pakoras, Chaat items
        - Drinks: Lassi, Buttermilk, Masala chai
        - Non-veg: Butter chicken, Tandoori chicken, Mutton curry, Kebabs
        """,
        
        "south_indian": """
        Focus on South Indian cuisine including:
        - Breakfast: Idli, Dosa, Uttapam, Upma, Poha, Vada
        - Rice dishes: Sambar rice, Rasam rice, Curd rice, Lemon rice, Coconut rice
        - Curries: Sambar, Rasam, Kootu, Avial, Mor kulambu
        - Vegetables: Beans poriyal, Cabbage thoran, Drumstick curry
        - Snacks: Murukku, Mixture, Banana Chips, Seedai, Thenkuzhal, Ribbon Pakoda, Kara Boondi, Thattai, Pakoda, Vada, Samosa, Paniyaram, Bonda, Bajji, Masala Peanuts, Chivda, Sev, Potato Chips
        - Drinks: Filter coffee, Buttermilk, Coconut water
        - Non-veg: Fish curry, Chicken curry, Mutton pepper fry, Prawn curry
        - Chutneys: Coconut chutney, Tomato chutney, Mint chutney
        """,
        
        "commonly_available": """
        Focus on commonly available Indian foods that are widely accessible:
        - Simple Indian breads: Roti, Chapati, Plain paratha
        - Basic rice dishes: Plain rice, Jeera rice, Simple pulao
        - Common dals: Moong dal, Toor dal, Chana dal
        - Everyday vegetables: Aloo sabzi, Mixed vegetables, Seasonal vegetables
        - Common snacks: Fruits, Nuts, Biscuits, Namkeen
        - Basic drinks: Tea, Milk, Buttermilk, Fresh juices
        - Simple non-veg: Basic chicken curry, Egg curry, Simple fish preparations
        - Accessible ingredients that don't require specialty stores
        """
    }
    
    cuisine_instruction = cuisine_context.get(cuisine_type, cuisine_context["commonly_available"])
    
    # Build context about previous meals to avoid repetition
    previous_context = ""
    if previous_meals:
        previous_context = f"""
        
    IMPORTANT - AVOID REPETITION:
    Here are the main meals from previous days that you should NOT repeat:
    
    Previous Breakfasts: {', '.join(previous_meals.get('breakfasts', []))}
    Previous Lunches: {', '.join(previous_meals.get('lunches', []))}
    Previous Dinners: {', '.join(previous_meals.get('dinners', []))}
    
    Please ensure {day_name}'s main meals (Breakfast, Lunch, Dinner) are DIFFERENT from the above.
        """
    
    prompt = f"""
    Create a detailed meal plan for {day_name} based on the following client profile:
    
    Client Details:
    - Goal: {profile['goal_type']} ({profile['weight_delta_text']})
    - Target Calories: {profile['target_calories']} calories per day (STRICT REQUIREMENT)
    - Diet Type: {diet_type}
    - Cuisine Preference: {cuisine_type.replace('_', ' ').title()}
    - Current Weight: {profile['current_weight']} kg
    - Target Weight: {profile['target_weight']} kg
    {previous_context}
    
    CRITICAL CALORIE REQUIREMENTS:
    You MUST distribute calories across meal slots as follows:
    {calorie_breakdown}
    
    Total must equal {target_calories} calories (±5% tolerance allowed).
    
    CUISINE FOCUS:
    {cuisine_instruction}
    
    IMPORTANT - QUANTITY SPECIFICATIONS:
    For each food item, specify realistic quantities with proper units and ACCURATE calories.
    
    CALORIE ACCURACY GUIDELINES:
    - Use standard calorie values for Indian foods
    - Breakfast (slot 3): Include protein + carbs + some fat
    - Lunch (slot 5): Largest meal with balanced macros
    - Dinner (slot 7): Moderate portions, lighter carbs if weight loss
    - Snacks: Keep within calorie limits but nutritious
    - Early morning/pre-breakfast: Very low calorie options
    
    Please create meals for these time slots with EXACT calorie targets:
    1. Early Morning Detox ({slot_calories['1']['target']} cal) - Detox drinks, lemon water
    2. Pre-Breakfast Starter ({slot_calories['2']['target']} cal) - Light starter
    3. Breakfast ({slot_calories['3']['target']} cal) - Main breakfast (MUST BE UNIQUE)
    4. Mid-Morning Snack ({slot_calories['4']['target']} cal) - Healthy snack
    5. Lunch ({slot_calories['5']['target']} cal) - Main lunch (MUST BE UNIQUE) 
    6. Evening Snack ({slot_calories['6']['target']} cal) - Evening refreshment
    7. Dinner ({slot_calories['7']['target']} cal) - Main dinner (MUST BE UNIQUE)
    
    Guidelines:
    - Each meal should fit the {profile['goal_type']} goal
    - Total daily calories MUST be {target_calories} (±5%)
    - Use {diet_type} foods only
    - STRICTLY follow {cuisine_type.replace('_', ' ').title()} cuisine preferences
    - Include variety of nutrients: proteins, carbs, healthy fats, fiber
    - Suggest appropriate portion sizes with units
    - Use authentic ingredients and preparations from the specified cuisine
    - CRITICAL: Main meals (Breakfast, Lunch, Dinner) must be completely different from previous days
    - Be precise with calorie calculations - use standard food calorie values
    
    Return ONLY valid JSON in this exact format:
    {{
        "day_name": "{day_name}",
        "total_target_calories": {target_calories},
        "meals": [
            {{
                "slot_id": "1",
                "target_calories": {slot_calories['1']['target']},
                "foods": [
                    {{
                        "name": "food_name",
                        "calories": number,
                        "protein": number,
                        "carbs": number,
                        "fat": number,
                        "quantity": "amount unit (e.g., 250 ml, 100 grams, 2 pieces)"
                    }}
                ]
            }}
        ]
    }}
    
    VERIFY: Calculate total calories across all slots and ensure it equals {target_calories} (±5%).
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": f"""You are a professional nutritionist and meal planning expert specializing in {cuisine_type.replace('_', ' ').title()} cuisine. 

CRITICAL: You must create meal plans that precisely match the target calorie requirements. Use accurate calorie values for Indian foods:

Common Calorie References:
- 1 medium roti (30g): ~80 calories
- 1 cup cooked rice (150g): ~200 calories  
- 1 cup dal (150g): ~150-200 calories
- 1 cup milk (200ml): ~120 calories
- 1 medium idli: ~40 calories
- 1 medium dosa: ~150 calories
- 100g chicken curry: ~200 calories
- 1 tsp oil: ~40 calories
- 1 medium banana: ~100 calories

Always calculate and verify total calories match the target. Focus on creating variety and avoiding repetitive meals across days. Always specify proper quantities with units for each food item."""
                },
                {"role": "user", "content": prompt}
            ],
            max_tokens=2500,
            temperature=0.3  # Lower temperature for more consistent calorie calculations
        )
        
        result = response.choices[0].message.content.strip()
        result = re.sub(r"^```json\s*", "", result)
        result = re.sub(r"\s*```$", "", result)
        
        meal_data = json.loads(result)
        
        # Validate and adjust calories
        validated_meal_data = validate_and_adjust_calories(meal_data, target_calories)
        
        # Final verification
        final_calories = sum(
            food.get("calories", 0)
            for meal_slot in validated_meal_data.get("meals", [])
            for food in meal_slot.get("foods", [])
        )
        
        print(f"DEBUG: Final meal plan calories: {final_calories} (target: {target_calories})")
        
        return validated_meal_data
        
    except Exception as e:
        print(f"AI meal generation error: {e}")
        print(f"AI generation traceback: {traceback.format_exc()}")
        return create_fallback_meal_plan(day_name, diet_type, cuisine_type, profile, previous_meals)

def get_slot_name(slot_id):
    """Get readable name for meal slot"""
    slot_names = {
        "1": "Early Morning Detox",
        "2": "Pre-Breakfast Starter", 
        "3": "Breakfast",
        "4": "Mid-Morning Snack",
        "5": "Lunch",
        "6": "Evening Snack",
        "7": "Dinner"
    }
    return slot_names.get(slot_id, f"Slot {slot_id}")

def create_fallback_meal_plan(day_name, diet_type, cuisine_type, profile, previous_meals=None):
    """Create a fallback meal plan with accurate calories"""
    target_calories = profile['target_calories']
    slot_calories = calculate_meal_calories_distribution(target_calories)
    
    # Create basic fallback based on cuisine and diet type
    fallback_meals = []
    
    for slot_id in ["1", "2", "3", "4", "5", "6", "7"]:
        slot_target = slot_calories[slot_id]["target"]
        
        if slot_id == "1":  # Early Morning
            foods = [{"name": "Warm lemon water", "calories": slot_target, "protein": 0, "carbs": 2, "fat": 0, "quantity": "250 ml"}]
        elif slot_id == "2":  # Pre-breakfast
            foods = [{"name": "Soaked almonds", "calories": slot_target, "protein": 3, "carbs": 2, "fat": 5, "quantity": "5 pieces"}]
        elif slot_id == "3":  # Breakfast
            if cuisine_type == "south_indian":
                foods = [
                    {"name": "Idli", "calories": int(slot_target * 0.6), "protein": 8, "carbs": 40, "fat": 2, "quantity": "4 pieces"},
                    {"name": "Sambar", "calories": int(slot_target * 0.3), "protein": 5, "carbs": 15, "fat": 3, "quantity": "150 ml"},
                    {"name": "Coconut chutney", "calories": int(slot_target * 0.1), "protein": 1, "carbs": 3, "fat": 2, "quantity": "20 grams"}
                ]
            else:  # North Indian or common
                foods = [
                    {"name": "Roti", "calories": int(slot_target * 0.5), "protein": 6, "carbs": 30, "fat": 2, "quantity": "2 medium pieces"},
                    {"name": "Dal", "calories": int(slot_target * 0.3), "protein": 8, "carbs": 15, "fat": 4, "quantity": "150 grams"},
                    {"name": "Mixed vegetables", "calories": int(slot_target * 0.2), "protein": 3, "carbs": 8, "fat": 2, "quantity": "100 grams"}
                ]
        elif slot_id == "4":  # Mid-morning
            foods = [{"name": "Mixed fruit", "calories": slot_target, "protein": 2, "carbs": 20, "fat": 1, "quantity": "100 grams"}]
        elif slot_id == "5":  # Lunch
            if cuisine_type == "south_indian":
                foods = [
                    {"name": "Rice", "calories": int(slot_target * 0.4), "protein": 8, "carbs": 60, "fat": 2, "quantity": "200 grams"},
                    {"name": "Sambar", "calories": int(slot_target * 0.25), "protein": 8, "carbs": 20, "fat": 5, "quantity": "200 grams"},
                    {"name": "Rasam", "calories": int(slot_target * 0.15), "protein": 3, "carbs": 10, "fat": 3, "quantity": "150 ml"},
                    {"name": "Vegetables poriyal", "calories": int(slot_target * 0.2), "protein": 4, "carbs": 12, "fat": 4, "quantity": "100 grams"}
                ]
            else:
                foods = [
                    {"name": "Rice", "calories": int(slot_target * 0.35), "protein": 6, "carbs": 50, "fat": 2, "quantity": "150 grams"},
                    {"name": "Dal", "calories": int(slot_target * 0.25), "protein": 10, "carbs": 18, "fat": 5, "quantity": "150 grams"},
                    {"name": "Mixed vegetables", "calories": int(slot_target * 0.25), "protein": 5, "carbs": 15, "fat": 4, "quantity": "150 grams"},
                    {"name": "Curd", "calories": int(slot_target * 0.15), "protein": 5, "carbs": 8, "fat": 3, "quantity": "100 grams"}
                ]
        elif slot_id == "6":  # Evening
            foods = [{"name": "Tea with biscuits", "calories": slot_target, "protein": 3, "carbs": 15, "fat": 4, "quantity": "1 cup + 2 biscuits"}]
        elif slot_id == "7":  # Dinner
            if cuisine_type == "south_indian":
                foods = [
                    {"name": "Rice", "calories": int(slot_target * 0.4), "protein": 6, "carbs": 45, "fat": 1, "quantity": "150 grams"},
                    {"name": "Rasam", "calories": int(slot_target * 0.3), "protein": 4, "carbs": 12, "fat": 4, "quantity": "200 ml"},
                    {"name": "Vegetable curry", "calories": int(slot_target * 0.3), "protein": 5, "carbs": 10, "fat": 5, "quantity": "100 grams"}
                ]
            else:
                foods = [
                    {"name": "Roti", "calories": int(slot_target * 0.4), "protein": 8, "carbs": 35, "fat": 3, "quantity": "3 medium pieces"},
                    {"name": "Dal", "calories": int(slot_target * 0.35), "protein": 10, "carbs": 20, "fat": 6, "quantity": "150 grams"},
                    {"name": "Vegetable curry", "calories": int(slot_target * 0.25), "protein": 4, "carbs": 12, "fat": 4, "quantity": "100 grams"}
                ]
        
        fallback_meals.append({
            "slot_id": slot_id,
            "target_calories": slot_target,
            "foods": foods
        })
    
    return {
        "day_name": day_name,
        "total_target_calories": target_calories,
        "meals": fallback_meals
    }
def convert_ai_meal_to_template(meal_data):
    """Convert AI generated meal data to template format"""
    template = get_meal_template()
    
    for meal_slot in meal_data.get("meals", []):
        slot_id = meal_slot.get("slot_id")
        foods = meal_slot.get("foods", [])
        
        if slot_id and foods:
            slot_index = int(slot_id) - 1
            if 0 <= slot_index < len(template):
                food_list = []
                for food in foods:
                    # Get quantity - either from AI response or use standard quantity
                    quantity = food.get("quantity", get_standard_quantity_for_food(food.get("name", "")))
                    
                    food_item = create_food_item(
                        name=food.get("name", "Unknown Food"),
                        calories=food.get("calories", 0),
                        protein=food.get("protein", 0),
                        carbs=food.get("carbs", 0),
                        fat=food.get("fat", 0),
                        quantity=quantity
                    )
                    food_list.append(food_item)
                
                template[slot_index]["foodList"] = food_list
                template[slot_index]["itemsCount"] = len(food_list)
    
    return template

def extract_main_meals(meal_data):
    """Extract main meal names for tracking variety"""
    main_meals = {
        'breakfasts': [],
        'lunches': [],
        'dinners': []
    }
    
    for meal_slot in meal_data.get("meals", []):
        slot_id = meal_slot.get("slot_id")
        foods = meal_slot.get("foods", [])
        
        if slot_id == "3":  # Breakfast
            main_meals['breakfasts'].extend([food.get("name", "") for food in foods])
        elif slot_id == "5":  # Lunch
            main_meals['lunches'].extend([food.get("name", "") for food in foods])
        elif slot_id == "7":  # Dinner
            main_meals['dinners'].extend([food.get("name", "") for food in foods])
    
    return main_meals

def generate_7_day_meal_plan(profile, diet_type, cuisine_type):
    """Generate complete 7-day meal plan with variety tracking and cuisine preference"""
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    meal_plan = {}
    previous_meals = {
        'breakfasts': [],
        'lunches': [],
        'dinners': []
    }
    
    for day in days:
        print(f"DEBUG: Generating {cuisine_type} {diet_type} meal plan for {day}")
        try:
            meal_data = generate_meal_plan_with_ai(profile, diet_type, cuisine_type, day, previous_meals)
            template = convert_ai_meal_to_template(meal_data)
            meal_plan[day.lower()] = template
            
            # Extract and track main meals for next day's variety
            day_meals = extract_main_meals(meal_data)
            previous_meals['breakfasts'].extend(day_meals['breakfasts'])
            previous_meals['lunches'].extend(day_meals['lunches'])
            previous_meals['dinners'].extend(day_meals['dinners'])
            
            # Keep only last 3 days of meals to allow some repetition after gap
            if len(previous_meals['breakfasts']) > 9:  # 3 days * ~3 items
                previous_meals['breakfasts'] = previous_meals['breakfasts'][-9:]
            if len(previous_meals['lunches']) > 9:
                previous_meals['lunches'] = previous_meals['lunches'][-9:]
            if len(previous_meals['dinners']) > 9:
                previous_meals['dinners'] = previous_meals['dinners'][-9:]
                
        except Exception as e:
            print(f"Error generating meal plan for {day}: {e}")
            # Use fallback for this day
            fallback_data = create_fallback_meal_plan(day, diet_type, cuisine_type, profile, previous_meals)
            template = convert_ai_meal_to_template(fallback_data)
            meal_plan[day.lower()] = template
            
            # Track fallback meals too
            day_meals = extract_main_meals(fallback_data)
            previous_meals['breakfasts'].extend(day_meals['breakfasts'])
            previous_meals['lunches'].extend(day_meals['lunches'])
            previous_meals['dinners'].extend(day_meals['dinners'])
    
    return meal_plan

def detect_diet_preference(text):
    """Detect diet preference from user input with better patterns"""
    text = text.lower().strip()
    
    # Non-vegetarian patterns - check these FIRST
    non_veg_patterns = [
        r'\bnon\s*-?\s*veg\b',
        r'\bnon\s*-?\s*vegetarian\b', 
        r'\bmeat\b',
        r'\bchicken\b',
        r'\bfish\b',
        r'\begg\b',
        r'\beggs\b',
        r'\bbeef\b',
        r'\bmutton\b',
        r'\bpork\b',
        r'\bseafood\b',
        r'\bomni\b',
        r'\bomnivore\b',
        r'\bi eat meat\b',
        r'\bi eat chicken\b',
        r'\bi eat fish\b',
        r'\bi eat eggs\b',
        r'\bnon vegetarian\b',
        r'\bnon veg\b',
        r'\bnon-veg\b',
        r'\bnon-vegetarian\b'
    ]
    
    # Vegetarian patterns - check these AFTER non-veg patterns
    veg_patterns = [
        r'(?<!non\s)(?<!non\s-)(?<!non-)(?<!non)\bveg\b',  # veg not preceded by non
        r'(?<!non\s)(?<!non\s-)(?<!non-)(?<!non)\bvegetarian\b',  # vegetarian not preceded by non
        r'\bpure veg\b',
        r'\bonly veg\b',
        r'\bno meat\b',
        r'\bno chicken\b',
        r'\bno fish\b',
        r'\bno eggs\b',
        r'\bplant based\b',
        r'\bplant-based\b',
        r'\bvegan\b',
        r'\bi am veg\b',
        r'\bi am vegetarian\b'
    ]
    
    print(f"DEBUG: Analyzing text: '{text}'")
    
    # Check for non-vegetarian first
    for pattern in non_veg_patterns:
        if re.search(pattern, text):
            print(f"DEBUG: Matched non-veg pattern: {pattern}")
            return "non-vegetarian"
    
    # Check for vegetarian
    for pattern in veg_patterns:
        if re.search(pattern, text):
            print(f"DEBUG: Matched veg pattern: {pattern}")
            return "vegetarian"
    
    print("DEBUG: No pattern matched")
    return None

def detect_cuisine_preference(text):
    """Detect cuisine preference from user input"""
    text = text.lower().strip()
    
    # North Indian patterns
    north_indian_patterns = [
        r'\bnorth\s*indian\b',
        r'\bnorthern\s*indian\b',
        r'\bnorth\b',
        r'\bpunjabi\b',
        r'\bdelhi\b',
        r'\bmughlai\b',
        r'\broti\b',
        r'\bchapati\b',
        r'\bparatha\b',
        r'\bnaan\b',
        r'\brajasthani\b',
        r'\bharyanvi\b',
        r'\bnorthside\b',
        r'\bnorth\s*side\b'
    ]
    
    # South Indian patterns
    south_indian_patterns = [
        r'\bsouth\s*indian\b',
        r'\bsouthern\s*indian\b',
        r'\bsouth\b',
        r'\btamil\b',
        r'\bkerala\b',
        r'\bkarnataka\b',
        r'\bandhra\b',
        r'\btelugu\b',
        r'\bidli\b',
        r'\bdosa\b',
        r'\bsambar\b',
        r'\bcoconut\b',
        r'\brice\b.*\bfocus\b',
        r'\bfilter\s*coffee\b'
    ]
    
    # Commonly available patterns
    common_patterns = [
        r'\bcommon\b',
        r'\bcommonly\b',
        r'\bavailable\b',
        r'\bsimple\b',
        r'\bbasic\b',
        r'\beveryday\b',
        r'\bregular\b',
        r'\bnormal\b',
        r'\bany\b',
        r'\banything\b',
        r'\bmixed\b',
        r'\bgeneral\b'
        r'\bcommonlyavailable\b',
        r'\bcommonavailable\b',
        r'\bcommomly\s*available\b',
        r'\bcommon\s*available\b',
        r'\ball\b',
    ]
    
    print(f"DEBUG: Analyzing cuisine preference: '{text}'")
    
    # Check for North Indian
    for pattern in north_indian_patterns:
        if re.search(pattern, text):
            print(f"DEBUG: Matched North Indian pattern: {pattern}")
            return "north_indian"
    
    # Check for South Indian
    for pattern in south_indian_patterns:
        if re.search(pattern, text):
            print(f"DEBUG: Matched South Indian pattern: {pattern}")
            return "south_indian"
    
    # Check for commonly available
    for pattern in common_patterns:
        if re.search(pattern, text):
            print(f"DEBUG: Matched commonly available pattern: {pattern}")
            return "commonly_available"
    
    print("DEBUG: No cuisine pattern matched")
    return None

@router.get("/chat/stream", dependencies=[Depends(RateLimiter(times=30, seconds=60))])
async def chat_stream(
    user_id: int,
    client_id: int = Query(..., description="Client ID for whom to create meal plan"),
    text: str = Query(None),
    mem = Depends(get_mem),
    oai = Depends(get_oai),
    db: Session = Depends(get_db),
):
    try:
        print(f"DEBUG: Received request - user_id: {user_id}, client_id: {client_id}, text: {text}")
        
        if not text:
            # Fetch client profile and start conversation
            try:
                profile = _fetch_profile(db, client_id)
                print(f"DEBUG: Client profile: {profile}")
                
                async def _welcome():
                    welcome_msg = f"""Hello! I'm your meal template assistant.

I can see your profile:
• Current Weight: {profile['current_weight']} kg
• Target Weight: {profile['target_weight']} kg  
• Goal: {profile['weight_delta_text']}
• Daily Calorie Target: {profile['target_calories']} calories

I'll create a personalized 7-day meal template for you. First, are you vegetarian or non-vegetarian?"""
                    
                    await mem.set_pending(user_id, {
                        "state": "awaiting_diet_preference",
                        "client_id": client_id,
                        "profile": profile
                    })
                    
                    yield f"data: {json.dumps({'message': welcome_msg, 'type': 'welcome'})}\n\n"
                    yield "event: done\ndata: [DONE]\n\n"
                
                return StreamingResponse(_welcome(), media_type="text/event-stream",
                                        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
                                        
            except Exception as e:
                print(f"Error fetching profile: {e}")
                print(f"Profile fetch full traceback: {traceback.format_exc()}")
                async def _profile_error():
                    yield f"data: {json.dumps({'message': 'Error fetching client profile. Please check the client ID and try again.', 'type': 'error'})}\n\n"
                    yield "event: done\ndata: [DONE]\n\n"
                
                return StreamingResponse(_profile_error(), media_type="text/event-stream",
                                        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
        
        text = text.strip()
        print(f"DEBUG: Processing text: '{text}'")
        
        # Get pending state from memory
        try:
            pending_state = await mem.get_pending(user_id)
            print(f"DEBUG: Pending state: {pending_state}")
        except Exception as e:
            print(f"Error getting pending state: {e}")
            pending_state = None
        
        # Handle diet preference selection
        if pending_state and pending_state.get("state") == "awaiting_diet_preference":
            print("DEBUG: In awaiting_diet_preference state")
            
            diet_type = detect_diet_preference(text)
            
            if diet_type:
                print(f"DEBUG: Detected diet type: {diet_type}")
                
                # Move to cuisine preference selection
                await mem.set_pending(user_id, {
                    "state": "awaiting_cuisine_preference",
                    "client_id": pending_state.get("client_id"),
                    "profile": pending_state.get("profile"),
                    "diet_type": diet_type
                })
                
                async def _ask_cuisine():
                    cuisine_msg = f"""Great! You've selected {diet_type}.

Now, please choose your cuisine preference:

• **North Indian** - Roti, chapati, paratha, naan, dal makhani, butter chicken, etc.
• **South Indian** - Idli, dosa, sambar, rasam, coconut-based dishes, etc.  
• **Commonly Available** - Simple, everyday foods that are widely accessible

Please type one of: "North Indian", "South Indian", or "Commonly Available"""
                    
                    yield f"data: {json.dumps({'message': cuisine_msg, 'type': 'cuisine_selection'})}\n\n"
                    yield "event: done\ndata: [DONE]\n\n"
                
                return StreamingResponse(_ask_cuisine(), media_type="text/event-stream",
                                        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
            else:
                async def _ask_diet_again():
                    msg = """I didn't understand your preference. Please be more specific:

• Type "vegetarian" or "veg" if you don't eat meat, fish, chicken, eggs
• Type "non-vegetarian" or "non-veg" if you eat meat, chicken, fish, eggs

Examples:
- "I am vegetarian"
- "non-veg"  
- "I eat chicken and fish"
- "veg only" """
                    yield f"data: {json.dumps({'message': msg, 'type': 'clarification'})}\n\n"
                    yield "event: done\ndata: [DONE]\n\n"
                
                return StreamingResponse(_ask_diet_again(), media_type="text/event-stream",
                                        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
        
        # Handle cuisine preference selection  
        elif pending_state and pending_state.get("state") == "awaiting_cuisine_preference":
            print("DEBUG: In awaiting_cuisine_preference state")
            
            cuisine_type = detect_cuisine_preference(text)
            
            if cuisine_type:
                print(f"DEBUG: Detected cuisine type: {cuisine_type}")
                
                # Generate meal plan with both diet and cuisine preferences
                profile = pending_state.get("profile")
                diet_type = pending_state.get("diet_type")
                
                async def _generate_plan():
                    try:
                        cuisine_display = cuisine_type.replace('_', ' ').title()
                        yield f"data: {json.dumps({'message': f'Perfect! Creating a {diet_type} {cuisine_display} 7-day meal plan for you...', 'type': 'progress'})}\n\n"
                        
                        meal_plan = generate_7_day_meal_plan(profile, diet_type, cuisine_type)
                        
                        # Store the meal plan
                        await mem.set_pending(user_id, {
                            "state": "awaiting_name_change",
                            "client_id": pending_state.get("client_id"),
                            "profile": profile,
                            "diet_type": diet_type,
                            "cuisine_type": cuisine_type,
                            "meal_plan": meal_plan
                        })
                        
                        success_msg = f"""✅ Your 7-day {diet_type} {cuisine_display} meal plan has been created!

The plan includes:
• Monday to Sunday templates
• {profile['target_calories']} calories per day (approximately)
• Meals optimized for: {profile['weight_delta_text']}
• {cuisine_display} cuisine focus
• 7 meal slots per day (Early Morning to Dinner)
• Proper portion sizes with units (grams, ml, pieces)

Would you like to change any day names? The current names are: Monday, Tuesday, Wednesday, Thursday, Friday, Saturday, Sunday.

Say 'yes' to customize names or 'no' to keep them as is."""
                        
                        # Return the meal plan data
                        yield f"data: {json.dumps({'message': success_msg, 'type': 'meal_plan_created'})}\n\n"
                        yield sse_json({
                            "type": "meal_plan",
                            "status": "created",
                            "diet_type": diet_type,
                            "cuisine_type": cuisine_type,
                            "total_calories_per_day": profile['target_calories'],
                            "goal": profile['weight_delta_text'],
                            "meal_plan": meal_plan
                        })
                        yield "event: done\ndata: [DONE]\n\n"
                        
                    except Exception as e:
                        print(f"Error generating meal plan: {e}")
                        print(f"Meal generation full traceback: {traceback.format_exc()}")
                        yield f"data: {json.dumps({'message': 'Sorry, there was an error creating your meal plan. Please try again or contact support.', 'type': 'error'})}\n\n"
                        yield "event: done\ndata: [DONE]\n\n"
                
                return StreamingResponse(_generate_plan(), media_type="text/event-stream",
                                        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
            else:
                async def _ask_cuisine_again():
                    msg = """Please choose from the available cuisine options:

• **North Indian** - Type "north indian" or "north"
• **South Indian** - Type "south indian" or "south"  
• **Commonly Available** - Type "commonly available" or "common"

Examples:
- "North Indian cuisine"
- "south"
- "commonly available foods"
- "simple everyday meals" """
                    yield f"data: {json.dumps({'message': msg, 'type': 'clarification'})}\n\n"
                    yield "event: done\ndata: [DONE]\n\n"
                
                return StreamingResponse(_ask_cuisine_again(), media_type="text/event-stream",
                                        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
        
        # Handle name change confirmation
        elif pending_state and pending_state.get("state") == "awaiting_name_change":
            print("DEBUG: In awaiting_name_change state")
            
            if is_yes(text):
                async def _ask_custom_names():
                    msg = """You can customize the day names. Please provide 7 names separated by commas.

Example: Detox Day, Power Monday, Healthy Tuesday, Wellness Wednesday, Fit Thursday, Strong Friday, Relax Saturday

Or say 'cancel' to keep the default names (Monday to Sunday)."""
                    
                    await mem.set_pending(user_id, {
                        "state": "awaiting_custom_names",
                        "client_id": pending_state.get("client_id"),
                        "profile": pending_state.get("profile"),
                        "diet_type": pending_state.get("diet_type"),
                        "cuisine_type": pending_state.get("cuisine_type"),
                        "meal_plan": pending_state.get("meal_plan")
                    })
                    
                    yield f"data: {json.dumps({'message': msg, 'type': 'custom_names_request'})}\n\n"
                    yield "event: done\ndata: [DONE]\n\n"
                
                return StreamingResponse(_ask_custom_names(), media_type="text/event-stream",
                                        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
            
            elif is_no(text):
                # Finalize with default names
                meal_plan = pending_state.get("meal_plan")
                await mem.clear_pending(user_id)
                
                async def _finalize_default():
                    yield f"data: {json.dumps({'message': '✅ Your 7-day meal plan is ready with default day names (Monday to Sunday)!', 'type': 'finalized'})}\n\n"
                    yield sse_json({
                        "type": "meal_plan_final",
                        "status": "completed",
                        "meal_plan": meal_plan,
                        "day_names": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
                    })
                    yield "event: done\ndata: [DONE]\n\n"
                
                return StreamingResponse(_finalize_default(), media_type="text/event-stream",
                                        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
            
            else:
                async def _ask_name_change_again():
                    yield f"data: {json.dumps({'message': 'Do you want to change the day names? Please say Yes or No.', 'type': 'clarification'})}\n\n"
                    yield "event: done\ndata: [DONE]\n\n"
                
                return StreamingResponse(_ask_name_change_again(), media_type="text/event-stream",
                                        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
        
        # Handle custom name input
        elif pending_state and pending_state.get("state") == "awaiting_custom_names":
            print("DEBUG: In awaiting_custom_names state")
            
            if text.lower() == "cancel":
                meal_plan = pending_state.get("meal_plan")
                await mem.clear_pending(user_id)
                
                async def _cancel_custom():
                    yield f"data: {json.dumps({'message': '✅ Your 7-day meal plan is ready with default day names!', 'type': 'finalized'})}\n\n"
                    yield sse_json({
                        "type": "meal_plan_final", 
                        "status": "completed",
                        "meal_plan": meal_plan,
                        "day_names": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
                    })
                    yield "event: done\ndata: [DONE]\n\n"
                
                return StreamingResponse(_cancel_custom(), media_type="text/event-stream",
                                        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
            
            # Parse custom names
            custom_names = [name.strip() for name in text.split(",") if name.strip()]
            
            if len(custom_names) == 7:
                # Update meal plan with custom names
                meal_plan = pending_state.get("meal_plan")
                default_days = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
                
                updated_meal_plan = {}
                for i, custom_name in enumerate(custom_names):
                    if i < len(default_days):
                        updated_meal_plan[custom_name.lower().replace(" ", "_")] = meal_plan.get(default_days[i], get_meal_template())
                
                await mem.clear_pending(user_id)
                
                async def _finalize_custom():
                    yield f"data: {json.dumps({'message': f'✅ Your 7-day meal plan is ready with custom day names: {', '.join(custom_names)}!', 'type': 'finalized'})}\n\n"
                    yield sse_json({
                        "type": "meal_plan_final",
                        "status": "completed", 
                        "meal_plan": updated_meal_plan,
                        "day_names": custom_names
                    })
                    yield "event: done\ndata: [DONE]\n\n"
                
                return StreamingResponse(_finalize_custom(), media_type="text/event-stream",
                                        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
            else:
                async def _ask_names_again():
                    yield f"data: {json.dumps({'message': f'Please provide exactly 7 day names separated by commas. You provided {len(custom_names)} names. Try again or say \"cancel\" to use default names.', 'type': 'error'})}\n\n"
                    yield "event: done\ndata: [DONE]\n\n"
                
                return StreamingResponse(_ask_names_again(), media_type="text/event-stream",
                                        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
        
        # Handle case where no pending state exists but user is sending text
        else:
            print(f"DEBUG: No valid pending state found. Current state: {pending_state}")
            async def _restart():
                yield f"data: {json.dumps({'message': 'It seems our conversation got reset. Let me start fresh. Please use the endpoint without text parameter to begin creating your meal plan.', 'type': 'restart'})}\n\n"
                yield "event: done\ndata: [DONE]\n\n"
            
            return StreamingResponse(_restart(), media_type="text/event-stream",
                                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
    
    except Exception as e:
        print(f"Critical error in chat_stream: {e}")
        print(f"Full critical traceback: {traceback.format_exc()}")
        
        try:
            await mem.clear_pending(user_id)
        except Exception as cleanup_error:
            print(f"Error clearing pending state: {cleanup_error}")
        
        async def _critical_error():
            yield f"data: {json.dumps({'message': 'Sorry, I encountered a technical error. Please try starting over. If the issue persists, contact support.', 'type': 'critical_error'})}\n\n"
            yield "event: done\ndata: [DONE]\n\n"
        
        return StreamingResponse(_critical_error(), media_type="text/event-stream",
                                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


class UserId(BaseModel):
    user_id: int


@router.get("/debug_pending")
async def debug_pending(
    user_id: int,
    mem = Depends(get_mem),
):
    """Debug endpoint to check pending state"""
    try:
        pending_state = await mem.get_pending(user_id)
        return {"user_id": user_id, "pending_state": pending_state, "status": "success"}
    except Exception as e:
        return {"user_id": user_id, "pending_state": None, "error": str(e), "status": "error"}


@router.post("/clear_pending")
async def clear_pending_state(
    req: UserId,
    mem = Depends(get_mem),
):
    """Clear pending state for a user"""
    try:
        await mem.clear_pending(req.user_id)
        return {"status": "cleared", "user_id": req.user_id}
    except Exception as e:
        return {"status": "error", "user_id": req.user_id, "error": str(e)}


@router.post("/delete_chat")
async def delete_chat(
    req: UserId,
    mem = Depends(get_mem),
):
    """Delete chat history and pending state for a user"""
    try:
        print(f"Deleting template chat history for user {req.user_id}")
        history_key = f"template_chat:{req.user_id}:history"
        pending_key = f"template_chat:{req.user_id}:pending"
        deleted = await mem.r.delete(history_key, pending_key)
        return {"status": "deleted", "user_id": req.user_id, "keys_deleted": deleted}
    except Exception as e:
        print(f"Error deleting chat for user {req.user_id}: {e}")
        return {"status": "error", "user_id": req.user_id, "error": str(e)}


@router.get("/test_diet_detection")
async def test_diet_detection(text: str):
    """Test endpoint to check diet preference detection"""
    result = detect_diet_preference(text)
    return {
        "input": text,
        "detected_diet": result,
        "status": "success" if result else "no_match"
    }

@router.get("/test_cuisine_detection")
async def test_cuisine_detection(text: str):
    """Test endpoint to check cuisine preference detection"""
    result = detect_cuisine_preference(text)
    return {
        "input": text,
        "detected_cuisine": result,
        "status": "success" if result else "no_match"
    }