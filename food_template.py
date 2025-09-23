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
   sse_json, sse_escape, gpt_small_route, _scale_macros, is_yes, is_no, is_fit_chat,
   has_action_verb, food_hits,ensure_per_unit_macros, is_fittbot_meta_query,normalize_food, 
   explicit_log_command, STYLE_PLAN, is_plan_request,STYLE_CHAT_FORMAT,pretty_plan
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
    """Return standard quantity with exact gram/ml measurements plus cup/bowl descriptions"""
    food_name_lower = food_name.lower()
    
    # Liquids - use ml with cup measurements
    if any(liquid in food_name_lower for liquid in ['water', 'juice', 'coconut water']):
        if 'water' in food_name_lower:
            return "250 ml | 1 cup | 1 large glass"
        else:
            return "200 ml | 3/4 cup | 1 medium glass"
    
    if any(liquid in food_name_lower for liquid in ['lassi', 'buttermilk']):
        return "200 ml | 3/4 cup | 1 glass"
    
    # Hot beverages
    if any(drink in food_name_lower for drink in ['tea', 'coffee', 'chai']):
        return "150 ml | 2/3 cup | 1 small glass"
    
    # Milk
    if 'milk' in food_name_lower:
        return "200 ml | 3/4 cup | 1 glass"
    
    # Bread items
    if any(bread in food_name_lower for bread in ['roti', 'chapati']):
        return "60 grams | 2 medium pieces"
    if any(bread in food_name_lower for bread in ['naan', 'kulcha']):
        return "80 grams | 1 medium piece"
    if 'paratha' in food_name_lower:
        return "70 grams | 1 medium piece"
    if any(bread in food_name_lower for bread in ['bread', 'toast']):
        return "50 grams | 2 slices"
    
    # Rice and grains
    if 'rice' in food_name_lower:
        return "150 grams | 3/4 cup cooked | 1 bowl"
    if any(grain in food_name_lower for grain in ['upma', 'poha', 'daliya']):
        return "150 grams | 3/4 cup | 1 bowl"
    if 'oats' in food_name_lower:
        return "50 grams | 1/2 cup dry | 1 bowl cooked"
    if 'cornflakes' in food_name_lower:
        return "30 grams | 1 cup | 1 bowl"
    
    # South Indian items
    if 'idli' in food_name_lower:
        return "120 grams | 3 pieces"
    if 'dosa' in food_name_lower:
        return "150 grams | 1 medium piece"
    if 'uttapam' in food_name_lower:
        return "120 grams | 1 piece"
    
    # Dal and curries
    if any(dal in food_name_lower for dal in ['dal', 'sambar', 'rasam']):
        return "150 grams | 3/4 cup | 1 bowl"
    if 'curry' in food_name_lower or 'sabzi' in food_name_lower:
        return "100 grams | 1/2 cup | 3/4 bowl"
    
    # Paneer dishes
    if 'paneer' in food_name_lower:
        return "100 grams | 1/2 cup cubes | 3/4 bowl"
    
    # Non-veg items
    if any(meat in food_name_lower for meat in ['chicken', 'mutton', 'fish']):
        return "100 grams | 1/2 cup | 3-4 pieces"
    if 'egg' in food_name_lower:
        if 'boiled' in food_name_lower or 'fried' in food_name_lower:
            return "100 grams | 2 pieces"
        else:
            return "50 grams | 1 piece"
    
    # Vegetables
    if any(veg in food_name_lower for veg in ['vegetables', 'aloo', 'gobi', 'bhindi', 'palak']):
        return "100 grams | 1/2 cup | 3/4 bowl"
    
    # Dairy
    if 'curd' in food_name_lower or 'raita' in food_name_lower:
        return "100 grams | 1/2 cup | 3/4 bowl"
    
    # Fruits
    if 'banana' in food_name_lower:
        return "100 grams | 1 medium piece"
    if 'apple' in food_name_lower:
        return "150 grams | 1 medium piece"
    if 'orange' in food_name_lower:
        return "120 grams | 1 medium piece"
    if any(fruit in food_name_lower for fruit in ['mango', 'papaya']):
        return "100 grams | 1/2 cup sliced | 3/4 bowl"
    if 'fruit' in food_name_lower:
        return "100 grams | 1/2 cup mixed | 3/4 bowl"
    
    # Nuts and snacks
    if any(nut in food_name_lower for nut in ['almonds', 'walnuts', 'cashews']):
        return "30 grams | 1/4 cup | 20-25 pieces"
    if 'nuts' in food_name_lower:
        return "30 grams | 1/4 cup | handful"
    if any(snack in food_name_lower for snack in ['mixture', 'namkeen', 'bhujia']):
        return "30 grams | 1/4 cup | small bowl"
    
    # Chutneys and condiments
    if 'chutney' in food_name_lower:
        return "20 grams | 2 tablespoons | 1/4 cup"
    if 'pickle' in food_name_lower:
        return "10 grams | 1 tablespoon"
    
    # Papad and similar
    if 'papad' in food_name_lower:
        return "5 grams | 1 piece"
    
    # Salad
    if 'salad' in food_name_lower:
        return "80 grams | 1/2 cup | 1 small bowl"
    
    # Oil and ghee
    if any(fat in food_name_lower for fat in ['oil', 'ghee', 'butter']):
        return "10 grams | 2 teaspoons"
    
    # Sugar and jaggery
    if any(sweet in food_name_lower for sweet in ['sugar', 'jaggery', 'honey']):
        return "15 grams | 1 tablespoon"
    
    # Soups
    if 'soup' in food_name_lower:
        return "200 ml | 3/4 cup | 1 bowl"
    
    # Dry fruits
    if any(dry_fruit in food_name_lower for dry_fruit in ['dates', 'raisins', 'figs']):
        return "30 grams | 1/4 cup | 8-10 pieces"
    
    # Cereals and pulses (raw)
    if any(pulse in food_name_lower for pulse in ['moong', 'chana', 'rajma', 'lentil']):
        return "30 grams | 1/4 cup | 2 tablespoons dry"
    
    # Sweets and desserts
    if any(sweet in food_name_lower for sweet in ['laddu', 'barfi', 'halwa', 'kheer']):
        return "50 grams | 1/4 cup | 1 small piece"
    
    # Biscuits and cookies
    if any(biscuit in food_name_lower for biscuit in ['biscuit', 'cookie', 'rusk']):
        return "25 grams | 3-4 pieces"
    
    # Additional items
    if any(drink in food_name_lower for drink in ['smoothie', 'shake']):
        return "200 ml | 3/4 cup | 1 glass"
    
    if any(grain in food_name_lower for grain in ['quinoa', 'millet', 'barley']):
        return "150 grams | 3/4 cup cooked | 1 bowl"
    
    # Default fallback
    return "100 grams | 1/2 cup | 1 serving"


#######ALLERGY##############################
# ADD THESE TWO FUNCTIONS HERE (after get_standard_quantity_for_food function)

def detect_food_restrictions(text):
    """Detect food allergies/restrictions from user input"""
    text = text.lower().strip()
    
    # Common allergen patterns
    allergen_patterns = {
        'peanuts': [r'\bpeanut\b', r'\bpeanuts\b', r'\bgroundnut\b', r'\bgroundnuts\b'],
        'tree_nuts': [r'\bnuts?\b', r'\balmond\b', r'\bwalnut\b', r'\bcashew\b', r'\bpistachio\b'],
        'dairy': [r'\bdairy\b', r'\bmilk\b', r'\bcheese\b', r'\bpaneer\b', r'\bbutter\b', r'\bghee\b', r'\byogurt\b', r'\bcurd\b'],
        'gluten': [r'\bgluten\b', r'\bwheat\b', r'\bbread\b', r'\broti\b', r'\bchapati\b'],
        'eggs': [r'\begg\b', r'\beggs\b'],
        'fish': [r'\bfish\b', r'\bseafood\b'],
        'shellfish': [r'\bshrimp\b', r'\bcrab\b', r'\blobster\b', r'\bshellfish\b'],
        'soy': [r'\bsoy\b', r'\bsoya\b', r'\btofu\b'],
        'sesame': [r'\bsesame\b', r'\btil\b'],
        'coconut': [r'\bcoconut\b', r'\bnariyal\b'],
        'onion_garlic': [r'\bonion\b', r'\bgarlic\b', r'\bpyaz\b', r'\blahsun\b']
    }
    
    # Restriction trigger words
    restriction_triggers = [
        r'\ballerg\w*\b',  # allergic, allergy
        r'\bremove\b',
        r'\bavoid\b',
        r'\bcan\'?t\s+eat\b',
        r'\bdont?\s+eat\b',
        r'\bdon\'?t\s+eat\b',
        r'\bnot\s+allowed\b',
        r'\brestrict\w*\b',
        r'\bintoleran\w*\b',
        r'\bsensitiv\w*\b',
        r'\bexclude\b',
        r'\btake\s+out\b',
        r'\bno\s+(eating\s+)?\b'
    ]
    
    # Check if text contains restriction triggers
    has_restriction_trigger = any(re.search(pattern, text) for pattern in restriction_triggers)
    
    if not has_restriction_trigger:
        return None
    
    # Find specific allergens/foods mentioned
    found_restrictions = []
    for allergen, patterns in allergen_patterns.items():
        if any(re.search(pattern, text) for pattern in patterns):
            found_restrictions.append(allergen)
    
    # Also extract any other food names mentioned
    other_foods = []
    words = text.split()
    for i, word in enumerate(words):
        if any(re.search(trigger, word) for trigger in restriction_triggers):
            for j in range(i+1, min(i+5, len(words))):
                next_word = words[j].strip('.,!?')
                if len(next_word) > 2 and next_word not in ['the', 'and', 'or', 'any', 'from', 'plan']:
                    other_foods.append(next_word)
    
    result = {
        'found_allergens': found_restrictions,
        'other_foods': other_foods,
        'raw_text': text
    }
    
    return result if found_restrictions or other_foods else None


def regenerate_meal_plan_without_allergens(meal_plan, restrictions, profile, diet_type, cuisine_type):
    """Regenerate meal plan avoiding specified allergens/foods with daily variety"""
    
    allergen_foods_map = {
        'peanuts': ['peanut', 'groundnut', 'peanut butter', 'groundnut oil'],
        'tree_nuts': ['almond', 'walnut', 'cashew', 'pistachio', 'nuts', 'mixed nuts'],
        'dairy': ['milk', 'cheese', 'paneer', 'butter', 'ghee', 'yogurt', 'curd', 'lassi', 'buttermilk'],
        'gluten': ['wheat', 'bread', 'roti', 'chapati', 'naan', 'paratha', 'biscuit'],
        'eggs': ['egg', 'eggs', 'boiled egg', 'fried egg', 'scrambled egg'],
        'fish': ['fish', 'fish curry', 'fried fish', 'fish fry'],
        'shellfish': ['shrimp', 'crab', 'lobster', 'prawns'],
        'soy': ['soy', 'soya', 'tofu', 'soybean'],
        'sesame': ['sesame', 'til', 'sesame oil'],
        'coconut': ['coconut', 'coconut oil', 'coconut milk', 'coconut chutney'],
        'onion_garlic': ['onion', 'garlic', 'pyaz', 'lahsun']
    }
    
    # Build list of foods to avoid
    avoid_foods = []
    
    # Add allergen-specific foods
    for allergen in restrictions.get('found_allergens', []):
        avoid_foods.extend(allergen_foods_map.get(allergen, []))
    
    # Add other mentioned foods
    avoid_foods.extend(restrictions.get('other_foods', []))
    
    # Convert to lowercase for matching
    avoid_foods = [food.lower() for food in avoid_foods]
    
    print(f"DEBUG: Foods to avoid: {avoid_foods}")
    
    avoid_list = ', '.join(avoid_foods)
    target_calories = profile['target_calories']
    slot_calories = calculate_meal_calories_distribution(target_calories)
    
    calorie_breakdown = "\n".join([
        f"Slot {slot_id} ({get_slot_name(slot_id)}): {info['target']} calories (range: {info['min']}-{info['max']})"
        for slot_id, info in slot_calories.items()
    ])
    
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
        - Snacks: Murukku, Mixture, Banana chips
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
    
    # Generate 7 different daily meal plans
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    new_meal_plan = {}
    previous_meals = {'breakfasts': [], 'lunches': [], 'dinners': []}
    
    for day in days:
        print(f"DEBUG: Regenerating {day} without allergens")
        
        # Create day-specific context to avoid repetition
        previous_context = ""
        if previous_meals['breakfasts'] or previous_meals['lunches'] or previous_meals['dinners']:
            previous_context = f"""
        
IMPORTANT - AVOID REPETITION:
Previous Breakfasts: {', '.join(previous_meals.get('breakfasts', [])[-3:])}  # Only last 3
Previous Lunches: {', '.join(previous_meals.get('lunches', [])[-3:])}
Previous Dinners: {', '.join(previous_meals.get('dinners', [])[-3:])}

Please ensure {day}'s main meals are DIFFERENT from the above.
        """
        
        prompt = f"""
URGENT: Create {day} meal plan avoiding ALL of these foods/ingredients: {avoid_list}

Client Profile:
- Goal: {profile['goal_type']} ({profile['weight_delta_text']})
- Target Calories: {profile['target_calories']} calories per day (STRICT REQUIREMENT)
- Diet Type: {diet_type}
- Cuisine Preference: {cuisine_type.replace('_', ' ').title()}

CRITICAL - FOOD RESTRICTIONS:
You MUST completely avoid these foods/ingredients: {avoid_list}
Check every food item and ensure NONE of the restricted items are included.
Find suitable alternatives that maintain the same calorie and nutritional profile.

{previous_context}

CALORIE REQUIREMENTS (MUST MATCH EXACTLY):
{calorie_breakdown}

CUISINE FOCUS:
{cuisine_instruction}

Alternative suggestions for restricted foods:
- If avoiding dairy: Use plant-based alternatives, coconut milk, or remove dairy items
- If avoiding nuts: Use seeds (pumpkin, sunflower) or other protein sources
- If avoiding gluten: Use rice, quinoa, or gluten-free grains
- If avoiding specific vegetables: Use other vegetables with similar nutrition

Create a complete {day} meal plan avoiding ALL restricted foods while maintaining:
- Same total daily calories: {target_calories}
- Same meal structure (7 time slots)
- Same cuisine preference: {cuisine_type.replace('_', ' ').title()}
- Same diet type: {diet_type}
- UNIQUE meals different from previous days

Return ONLY valid JSON in this format:
{{
    "day_name": "{day}",
    "total_target_calories": {target_calories},
    "restrictions_avoided": [list of avoided foods],
    "meals": [
        {{
            "slot_id": "1",
            "target_calories": number,
            "foods": [
                {{
                    "name": "food_name",
                    "calories": number,
                    "protein": number,
                    "carbs": number,
                    "fat": number
                }}
            ]
        }}
    ]
}}

VERIFY: Ensure NO restricted foods are included and total calories = {target_calories}.
"""
        
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": f"""You are a professional nutritionist specializing in {cuisine_type.replace('_', ' ').title()} cuisine and food allergy management.

CRITICAL: You must avoid ALL specified foods/allergens completely. Double-check every food item against the restriction list. Use precise calorie calculations to maintain the target calories while substituting restricted foods with safe alternatives.

Create UNIQUE daily variations - each day should have different main meals even when avoiding the same allergens."""
                    },
                    {"role": "user", "content": prompt}
                ],
                max_tokens=3000,
                temperature=0.4  # Slightly higher for more variety between days
            )
            
            result = response.choices[0].message.content.strip()
            result = re.sub(r"^```json\s*", "", result)
            result = re.sub(r"\s*```$", "", result)
            
            day_meal_data = json.loads(result)
            
            # Validate and adjust calories for this day
            validated_day_data = validate_and_adjust_calories(day_meal_data, target_calories)
            
            # Convert to template format for this specific day
            day_template = convert_ai_meal_to_template(validated_day_data)
            new_meal_plan[day.lower()] = day_template
            
            # Track meals for variety in next days
            day_meals = extract_main_meals(validated_day_data)
            previous_meals['breakfasts'].extend(day_meals['breakfasts'])
            previous_meals['lunches'].extend(day_meals['lunches'])
            previous_meals['dinners'].extend(day_meals['dinners'])
            
            # Keep only recent meals for variety tracking
            if len(previous_meals['breakfasts']) > 6:
                previous_meals['breakfasts'] = previous_meals['breakfasts'][-6:]
            if len(previous_meals['lunches']) > 6:
                previous_meals['lunches'] = previous_meals['lunches'][-6:]
            if len(previous_meals['dinners']) > 6:
                previous_meals['dinners'] = previous_meals['dinners'][-6:]
                
        except Exception as e:
            print(f"Error regenerating {day}: {e}")
            # Note: create_fallback_meal_plan function has been removed
            # Using get_meal_template() as basic fallback
            day_template = get_meal_template()
            new_meal_plan[day.lower()] = day_template
    
    print(f"DEBUG: Successfully regenerated {len(new_meal_plan)} unique daily plans")
    return new_meal_plan


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
   tolerance = target_calories * 0.13 # Allow 15% tolerance
  
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
    """Generate meal plan for a specific day using AI with accurate calorie control and diet restrictions"""
    
    target_calories = profile['target_calories']
    slot_calories = calculate_meal_calories_distribution(target_calories)
    
    # Get diet-specific restrictions
    diet_restrictions = get_diet_specific_restrictions(diet_type)
    
    # Build detailed calorie requirements for each slot
    calorie_breakdown = "\n".join([
        f"Slot {slot_id} ({get_slot_name(slot_id)}): {info['target']} calories (range: {info['min']}-{info['max']})"
        for slot_id, info in slot_calories.items()
    ])
    
    # Enhanced cuisine context with diet considerations
    cuisine_context = {
        "north_indian": f"""
        Focus on North Indian cuisine with {diet_type} restrictions:
        - Breads: {"Almond/coconut flour rotis" if diet_type == "ketogenic" else "Roti, Chapati, Naan, Paratha, Kulcha"}
        - Rice dishes: {"Cauliflower rice dishes" if diet_type == "ketogenic" else "Jeera rice, Pulao, Biryani"}
        - Proteins: {get_protein_suggestions(diet_type, "north_indian")}
        - Vegetables: {get_vegetable_suggestions(diet_type, "north_indian")}
        - Fats: {get_fat_suggestions(diet_type)}
        - Avoid: {', '.join(diet_restrictions['avoid'])}
        """,
        
        "south_indian": f"""
        Focus on South Indian cuisine with {diet_type} restrictions:
        - Traditional: {"Coconut-based dishes, cauliflower rice" if diet_type == "ketogenic" else "Idli, Dosa, Uttapam, Upma"}
        - Rice dishes: {"Coconut rice with vegetables" if diet_type == "ketogenic" else "Sambar rice, Rasam rice, Curd rice"}
        - Proteins: {get_protein_suggestions(diet_type, "south_indian")}
        - Curries: {get_curry_suggestions(diet_type, "south_indian") if 'get_curry_suggestions' in globals() else 'Traditional curries'}
        - Fats: Coconut oil, ghee (if allowed)
        - Avoid: {', '.join(diet_restrictions['avoid'])}
        """,
        
        "commonly_available": f"""
        Focus on commonly available Indian foods with {diet_type} restrictions:
        - Staples: {get_staple_suggestions(diet_type)}
        - Proteins: {get_protein_suggestions(diet_type, "common")}
        - Vegetables: {get_vegetable_suggestions(diet_type, "common")}
        - Snacks: {get_snack_suggestions(diet_type)}
        - Avoid: {', '.join(diet_restrictions['avoid'])}
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
- Diet Type: {diet_type} - {diet_restrictions['special_notes']}
- Cuisine Preference: {cuisine_type.replace('_', ' ').title()}
- Current Weight: {profile['current_weight']} kg
- Target Weight: {profile['target_weight']} kg
{previous_context}

CRITICAL DIET RESTRICTIONS FOR {diet_type.upper()}:
MUST AVOID: {', '.join(diet_restrictions['avoid'])}
MUST INCLUDE: {', '.join(diet_restrictions['allow'])}
Special Notes: {diet_restrictions['special_notes']}

CRITICAL CALORIE REQUIREMENTS:
You MUST distribute calories across meal slots as follows:
{calorie_breakdown}

Total must equal {target_calories} calories (±5% tolerance allowed).

CUISINE FOCUS WITH DIET RESTRICTIONS:
{cuisine_instruction}

{get_diet_specific_meal_guidelines(diet_type)}

Please create meals for these time slots with EXACT calorie targets:
1. Early Morning Detox ({slot_calories['1']['target']} cal) - {get_early_morning_suggestions(diet_type)}
2. Pre-Breakfast Starter ({slot_calories['2']['target']} cal) - {get_pre_breakfast_suggestions(diet_type)}
3. Breakfast ({slot_calories['3']['target']} cal) - Main breakfast (MUST BE UNIQUE)
4. Mid-Morning Snack ({slot_calories['4']['target']} cal) - {get_snack_suggestions(diet_type)}
5. Lunch ({slot_calories['5']['target']} cal) - Main lunch (MUST BE UNIQUE)
6. Evening Snack ({slot_calories['6']['target']} cal) - Evening refreshment
7. Dinner ({slot_calories['7']['target']} cal) - Main dinner (MUST BE UNIQUE)

Guidelines:
- Each meal should fit the {profile['goal_type']} goal
- Total daily calories MUST be {target_calories} (±5%)
- STRICTLY follow {diet_type} dietary restrictions - NO EXCEPTIONS
- Use {diet_type} appropriate foods only
- STRICTLY follow {cuisine_type.replace('_', ' ').title()} cuisine preferences
- Include variety of nutrients appropriate for {diet_type} diet
- CRITICAL: Main meals must be completely different from previous days
- For Ketogenic: Keep total daily carbs under 20g, focus on healthy fats
- For Paleo: Use only paleolithic-approved foods
- For Jain: Absolutely no root vegetables, onion, or garlic
- For Vegan: Zero animal products including dairy and eggs
- Be precise with calorie calculations using diet-appropriate foods

Return ONLY valid JSON in this exact format:
{{
    "day_name": "{day_name}",
    "diet_type": "{diet_type}",
    "total_target_calories": {target_calories},
    "diet_restrictions_followed": true,
    "meals": [
        {{
            "slot_id": "1",
            "target_calories": {slot_calories['1']['target']},
            "foods": [
                {{
                    "name": "food_name",
                    "calories": 50,
                    "protein": 2.0,
                    "carbs": 8.0,
                    "fat": 1.0,
                    "diet_compliant": true
                }}
            ]
        }}
    ]
}}

VERIFY: 
1. All foods are {diet_type} compliant
2. Total calories equals {target_calories} (±5%)
3. No forbidden foods included
4. Appropriate macronutrient ratios for {diet_type}
"""
    
    try:
        print(f"DEBUG: Calling OpenAI for {day_name} meal plan...")
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": f"""You are a professional nutritionist and meal planning expert specializing in {diet_type} diet and {cuisine_type.replace('_', ' ').title()} cuisine.

CRITICAL DIET COMPLIANCE: You must create meal plans that strictly follow {diet_type} dietary restrictions:
- FORBIDDEN FOODS: {', '.join(diet_restrictions['avoid'])}
- ALLOWED FOODS: {', '.join(diet_restrictions['allow'])}
- SPECIAL RULES: {diet_restrictions['special_notes']}

For Ketogenic: Aim for 70-75% fat, 20-25% protein, 5-10% carbs
For Paleo: Focus on whole foods available to paleolithic humans
For Jain: Absolutely no root vegetables, onion, garlic, or foods that harm multiple lives
For Vegan: Zero animal products - use plant-based alternatives only
For Eggetarian: Vegetarian + eggs allowed

Always verify each food item against diet restrictions before including it.
Return ONLY valid JSON - no markdown formatting or extra text."""
                },
                {"role": "user", "content": prompt}
            ],
            max_tokens=2500,
            temperature=0.3
        )
        
        result = response.choices[0].message.content.strip()
        print(f"DEBUG: Raw AI response for {day_name}: {result[:200]}...")
        
        # Clean JSON response
        result = re.sub(r"^```json\s*", "", result)
        result = re.sub(r"\s*```$", "", result)
        result = result.strip()
        
        print(f"DEBUG: Cleaned response: {result[:200]}...")
        
        try:
            meal_data = json.loads(result)
            print(f"DEBUG: Successfully parsed JSON for {day_name}")
        except json.JSONDecodeError as je:
            print(f"DEBUG: JSON parse error for {day_name}: {je}")
            print(f"DEBUG: Problematic JSON: {result}")
            raise je
        
        # Validate diet compliance
        if not validate_diet_compliance(meal_data, diet_type):
            print(f"WARNING: Generated meal plan may not be fully {diet_type} compliant")
        
        # Validate and adjust calories
        validated_meal_data = validate_and_adjust_calories(meal_data, target_calories)
        print(f"DEBUG: Successfully generated and validated {day_name} meal plan")
        
        return validated_meal_data
        
    except Exception as e:
        print(f"AI meal generation error for {day_name}: {e}")
        print(f"AI generation traceback: {traceback.format_exc()}")
        print(f"DEBUG: Falling back to default meal plan for {day_name}")
        # Using get_meal_template() as basic fallback
        return {
            "day_name": day_name,
            "total_target_calories": target_calories,
            "meals": [
                {"slot_id": str(i+1), "target_calories": 0, "foods": []}
                for i in range(7)
            ]
        }


def get_curry_suggestions(diet_type, cuisine):
    """Get curry suggestions based on diet and cuisine"""
    if diet_type == "jain":
        return "Dal curries without onion/garlic, vegetable curries with hing"
    elif diet_type == "vegan":
        return "Coconut-based curries, tomato-based curries without dairy"
    elif diet_type == "ketogenic":
        return "High-fat curries with coconut milk, minimal vegetables"
    else:
        return "Traditional curries based on cuisine preference"


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
           # Using get_meal_template() as basic fallback
           template = get_meal_template()
           meal_plan[day.lower()] = template
  
   return meal_plan


def detect_diet_preference(text):
    """Detect diet preference from user input with comprehensive diet options"""
    text = text.lower().strip()
    
    print(f"DEBUG: Analyzing diet text: '{text}'")
    
    # Ketogenic patterns - check first (most specific)
    keto_patterns = [
        r'\bketo\b',
        r'\bketogenic\b',
        r'\bketo diet\b',
        r'\bketogenic diet\b',
        r'\blow carb\b',
        r'\bno carb\b',
        r'\bhigh fat low carb\b',
        r'\bhflc\b',
        r'\blchf\b'  # low carb high fat
    ]
    
    # Paleo patterns
    paleo_patterns = [
        r'\bpaleo\b',
        r'\bpalaeolithic\b',
        r'\bpaleo diet\b',
        r'\bpaleolithic diet\b',
        r'\bcaveman diet\b',
        r'\bstone age diet\b',
        r'\bprimal\b',
        r'\bprimal diet\b'
    ]
    
    # Vegan patterns (more restrictive than vegetarian)
    vegan_patterns = [
        r'\bvegan\b',
        r'\bplant based\b',
        r'\bplant-based\b',
        r'\bno dairy\b.*\bno eggs\b',
        r'\bno animal products\b',
        r'\bstrictly plant\b',
        r'\bonly plants\b',
        r'\bvegan diet\b'
    ]
    
    # Jain diet patterns
    jain_patterns = [
        r'\bjain\b',
        r'\bjain diet\b',
        r'\bjainism\b',
        r'\bno root vegetables\b',
        r'\bno onion garlic\b',
        r'\bno underground\b',
        r'\bjain food\b',
        r'\bjain meal\b'
    ]
    
    # Eggetarian patterns (vegetarian + eggs)
    eggetarian_patterns = [
        r'\beggetarian\b',
        r'\begg vegetarian\b',
        r'\bvegetarian with eggs\b',
        r'\bveg with egg\b',
        r'\bveg plus egg\b',
        r'\beggs allowed\b.*\bveg\b',
        r'\bveg.*\beggs ok\b',
        r'\bovo vegetarian\b'
    ]
    
    # Non-vegetarian patterns
    non_veg_patterns = [
        r'\bnon\s*-?\s*veg\b',
        r'\bnon\s*-?\s*vegetarian\b',
        r'\bmeat\b',
        r'\bchicken\b',
        r'\bfish\b',
        r'\bbeef\b',
        r'\bmutton\b',
        r'\bpork\b',
        r'\bseafood\b',
        r'\bomni\b',
        r'\bomnivore\b',
        r'\bi eat meat\b',
        r'\bi eat chicken\b',
        r'\bi eat fish\b',
        r'\bnon vegetarian\b',
        r'\bnon veg\b',
        r'\bnon-veg\b',
        r'\bnon-vegetarian\b'
    ]
    
    # Pure vegetarian patterns (most restrictive vegetarian)
    veg_patterns = [
        r'(?<!non\s)(?<!non\s-)(?<!non-)(?<!non)\bveg\b',
        r'(?<!non\s)(?<!non\s-)(?<!non-)(?<!non)\bvegetarian\b',
        r'\bpure veg\b',
        r'\bonly veg\b',
        r'\bno meat\b',
        r'\bno chicken\b',
        r'\bno fish\b',
        r'\bplant vegetarian\b',
        r'\bi am veg\b',
        r'\bi am vegetarian\b'
    ]
    
    # Check in order of specificity (most specific first)
    
    # 1. Ketogenic (most specific dietary restriction)
    for pattern in keto_patterns:
        if re.search(pattern, text):
            print(f"DEBUG: Matched ketogenic pattern: {pattern}")
            return "ketogenic"
    
    # 2. Paleo
    for pattern in paleo_patterns:
        if re.search(pattern, text):
            print(f"DEBUG: Matched paleo pattern: {pattern}")
            return "paleo"
    
    # 3. Jain diet (very specific restrictions)
    for pattern in jain_patterns:
        if re.search(pattern, text):
            print(f"DEBUG: Matched jain pattern: {pattern}")
            return "jain"
    
    # 4. Vegan (more restrictive than vegetarian)
    for pattern in vegan_patterns:
        if re.search(pattern, text):
            print(f"DEBUG: Matched vegan pattern: {pattern}")
            return "vegan"
    
    # 5. Eggetarian (vegetarian + eggs)
    for pattern in eggetarian_patterns:
        if re.search(pattern, text):
            print(f"DEBUG: Matched eggetarian pattern: {pattern}")
            return "eggetarian"
    
    # 6. Non-vegetarian
    for pattern in non_veg_patterns:
        if re.search(pattern, text):
            print(f"DEBUG: Matched non-veg pattern: {pattern}")
            return "non-vegetarian"
    
    # 7. Vegetarian (least specific)
    for pattern in veg_patterns:
        if re.search(pattern, text):
            print(f"DEBUG: Matched veg pattern: {pattern}")
            return "vegetarian"
    
    print("DEBUG: No diet pattern matched")
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


def get_diet_specific_restrictions(diet_type):
    """Get specific food restrictions and guidelines for each diet type"""
    
    restrictions = {
        "vegetarian": {
            "avoid": ["meat", "chicken", "fish", "beef", "mutton", "pork", "seafood", "eggs"],
            "allow": ["dairy", "milk", "paneer", "ghee", "vegetables", "grains", "legumes", "fruits"],
            "special_notes": "No meat, fish, or eggs. Dairy products allowed."
        },
        
        "non-vegetarian": {
            "avoid": [],
            "allow": ["meat", "chicken", "fish", "eggs", "dairy", "vegetables", "grains", "legumes"],
            "special_notes": "All foods allowed including meat, fish, eggs, and dairy."
        },
        
        "eggetarian": {
            "avoid": ["meat", "chicken", "fish", "beef", "mutton", "pork", "seafood"],
            "allow": ["eggs", "dairy", "milk", "paneer", "ghee", "vegetables", "grains", "legumes"],
            "special_notes": "Vegetarian diet that includes eggs and dairy products."
        },
        
        "vegan": {
            "avoid": ["meat", "chicken", "fish", "eggs", "dairy", "milk", "paneer", "ghee", "butter", "curd", "cheese", "honey"],
            "allow": ["vegetables", "fruits", "grains", "legumes", "nuts", "seeds", "plant-based milk"],
            "special_notes": "Strictly plant-based. No animal products including dairy, eggs, or honey."
        },
        
        "jain": {
            "avoid": ["meat", "fish", "eggs", "onion", "garlic", "potato", "carrot", "radish", "beetroot", 
                     "ginger", "turmeric", "mushrooms", "yeast", "alcohol", "root vegetables", "underground vegetables"],
            "allow": ["dairy", "milk", "paneer", "above-ground vegetables", "fruits", "grains", "legumes"],
            "special_notes": "Vegetarian diet avoiding all root/underground vegetables, onion, garlic. No violence to plants with multiple lives."
        },
        
        "ketogenic": {
            "avoid": ["rice", "wheat", "bread", "roti", "chapati", "potato", "banana", "mango", "grapes", 
                     "sugar", "honey", "jaggery", "high-carb fruits", "legumes", "beans", "lentils"],
            "allow": ["meat", "fish", "eggs", "cheese", "butter", "ghee", "coconut oil", "avocado", 
                     "leafy greens", "cauliflower", "broccoli", "nuts", "seeds"],
            "special_notes": "Very low carb (under 20g/day), high fat, moderate protein. Focus on ketosis."
        },
        
        "paleo": {
            "avoid": ["grains", "rice", "wheat", "legumes", "dairy", "processed foods", "sugar", 
                     "artificial sweeteners", "vegetable oils", "beans", "lentils", "peanuts"],
            "allow": ["meat", "fish", "eggs", "vegetables", "fruits", "nuts", "seeds", "coconut", "olive oil"],
            "special_notes": "Foods available to paleolithic humans. No grains, legumes, or dairy."
        }
    }
    
    return restrictions.get(diet_type, restrictions["vegetarian"])


def get_protein_suggestions(diet_type, cuisine):
    """Get protein suggestions based on diet type and cuisine"""
    if diet_type == "vegetarian":
        return "Paneer, dal, legumes, dairy products"
    elif diet_type == "non-vegetarian":
        return "Chicken, fish, eggs, paneer, dal"
    elif diet_type == "eggetarian":
        return "Eggs, paneer, dal, legumes, dairy"
    elif diet_type == "vegan":
        return "Tofu, tempeh, legumes, nuts, seeds"
    elif diet_type == "jain":
        return "Paneer, dal (without onion/garlic), dairy"
    elif diet_type == "ketogenic":
        return "Eggs, fish, chicken, paneer, nuts, cheese"
    elif diet_type == "paleo":
        return "Grass-fed meat, wild fish, eggs, nuts, seeds"
    return "Mixed proteins"

def get_vegetable_suggestions(diet_type, cuisine):
    """Get vegetable suggestions based on diet restrictions"""
    if diet_type == "jain":
        return "Above-ground vegetables only: spinach, cabbage, cauliflower, broccoli, green beans"
    elif diet_type == "ketogenic":
        return "Leafy greens, cauliflower, broccoli, zucchini, bell peppers, avocado"
    elif diet_type == "paleo":
        return "All vegetables, sweet potato, squash, leafy greens"
    else:
        return "Seasonal vegetables, leafy greens, root vegetables (if allowed)"

def get_fat_suggestions(diet_type):
    """Get healthy fat suggestions based on diet type"""
    if diet_type == "ketogenic":
        return "Coconut oil, ghee, avocado, nuts, seeds, MCT oil"
    elif diet_type == "paleo":
        return "Coconut oil, olive oil, avocado, nuts, seeds"
    elif diet_type == "vegan":
        return "Coconut oil, olive oil, avocado, nuts, seeds, tahini"
    else:
        return "Ghee, coconut oil, nuts, seeds"

def get_early_morning_suggestions(diet_type):
    """Get early morning drink suggestions"""
    if diet_type == "ketogenic":
        return "Bulletproof coffee, green tea, lemon water"
    elif diet_type == "jain":
        return "Warm water, herbal tea (no ginger)"
    else:
        return "Lemon water, green tea, herbal tea"

def get_pre_breakfast_suggestions(diet_type):
    """Get pre-breakfast suggestions"""
    if diet_type == "ketogenic":
        return "Nuts, seeds, coconut"
    elif diet_type == "paleo":
        return "Nuts, seeds, coconut flakes"
    elif diet_type == "vegan":
        return "Soaked almonds, seeds"
    else:
        return "Soaked nuts, dates"

def get_snack_suggestions(diet_type):
    """Get snack suggestions based on diet"""
    if diet_type == "ketogenic":
        return "Nuts, cheese, avocado, coconut"
    elif diet_type == "paleo":
        return "Fruits, nuts, seeds, coconut"
    elif diet_type == "jain":
        return "Fruits, nuts, dairy-based snacks"
    elif diet_type == "vegan":
        return "Fruits, nuts, plant-based yogurt"
    else:
        return "Fruits, nuts, healthy snacks"

def get_staple_suggestions(diet_type):
    """Get staple food suggestions"""
    if diet_type == "ketogenic":
        return "Cauliflower rice, coconut flour items, cheese"
    elif diet_type == "paleo":
        return "Sweet potato, vegetables, fruits"
    elif diet_type == "jain":
        return "Rice, wheat (no yeast bread), above-ground vegetables"
    else:
        return "Rice, wheat, millet, quinoa"

def get_diet_specific_meal_guidelines(diet_type):
    """Get specific meal planning guidelines for each diet"""
    guidelines = {
        "ketogenic": """
KETOGENIC SPECIFIC GUIDELINES:
- Daily carbs must be under 20g total
- Fat should be 70-75% of calories
- Protein should be 20-25% of calories
- Use MCT oil, coconut oil, avocado for healthy fats
- Avoid all grains, legumes, high-carb vegetables
- Focus on leafy greens, cauliflower, broccoli
        """,
        
        "paleo": """
PALEO SPECIFIC GUIDELINES:
- Only foods available to paleolithic humans
- No grains, legumes, dairy, processed foods
- Emphasize grass-fed meats, wild fish, vegetables, fruits
- Use coconut oil, olive oil for cooking
- Include variety of vegetables and seasonal fruits
        """,
        
        "jain": """
JAIN DIET SPECIFIC GUIDELINES:
- Absolutely no root/underground vegetables (potato, onion, garlic, carrot, radish, ginger)
- No foods that harm multiple lives (mushrooms, yeast-based bread)
- Eat only during daylight hours traditionally
- Focus on above-ground vegetables, fruits, dairy, grains
- Use hing (asafoetida) instead of onion/garlic for flavor
        """,
        
        "vegan": """
VEGAN SPECIFIC GUIDELINES:
- Zero animal products including honey
- Use plant-based milk alternatives (almond, soy, coconut)
- Include B12-rich foods or supplementation consideration
- Focus on legumes, nuts, seeds for protein
- Use nutritional yeast for cheesy flavor
        """,
        
        "eggetarian": """
EGGETARIAN SPECIFIC GUIDELINES:
- Vegetarian diet with eggs included
- Eggs can be used in all preparations
- Include dairy products freely
- No meat, fish, or poultry
- Eggs provide complete protein source
        """
    }
    
    return guidelines.get(diet_type, "Standard dietary guidelines apply.")

def validate_diet_compliance(meal_data, diet_type):
    """Validate if meal plan complies with diet restrictions"""
    restrictions = get_diet_specific_restrictions(diet_type)
    avoid_foods = [food.lower() for food in restrictions['avoid']]
    
    for meal_slot in meal_data.get("meals", []):
        for food in meal_slot.get("foods", []):
            food_name = food.get("name", "").lower()
            for avoided_food in avoid_foods:
                if avoided_food in food_name:
                    print(f"WARNING: Found {avoided_food} in {food_name} for {diet_type} diet")
                    return False
    return True


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


I'll create a personalized 7-day meal template for you. First, are you vegetarian or non-vegetarian or eggetarian or vegan or ketogic or paleo or jain?"""
                  
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


Please type one of: "North Indian", "South Indian", or "Commonly Available
Note: For Ketogenic and Paleo diets, I can include specialized ingredients that might be less common in regular Indian stores if needed for proper nutrition."""
                  
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
                    
                    print(f"DEBUG: Starting meal plan generation with diet_type={diet_type}, cuisine_type={cuisine_type}")
                    
                    meal_plan = generate_7_day_meal_plan(profile, diet_type, cuisine_type)
                    
                    print(f"DEBUG: Generated meal plan with {len(meal_plan)} days")
                    
                    # Validate meal plan has content
                    if not meal_plan or len(meal_plan) == 0:
                        print("ERROR: Empty meal plan generated")
                        yield f"data: {json.dumps({'message': 'Sorry, I encountered an issue generating your meal plan. Please try again.', 'type': 'error'})}\n\n"
                        yield "event: done\ndata: [DONE]\n\n"
                        return
                    
                    # FIRST: Send the meal plan data (so user sees it)
                    yield sse_json({
                        "type": "meal_plan",
                        "status": "created",
                        "diet_type": diet_type,
                        "cuisine_type": cuisine_display,
                        "total_calories_per_day": profile['target_calories'],
                        "goal": profile['weight_delta_text'],
                        "meal_plan": meal_plan
                    })
                    
                    # THEN: Ask about name changes (after user sees meal plan)
                    await mem.set_pending(user_id, {
                        "state": "awaiting_name_change",
                        "client_id": pending_state.get("client_id"),
                        "profile": profile,
                        "diet_type": diet_type,
                        "cuisine_type": cuisine_type,
                        "meal_plan": meal_plan
                    })
                    
                    # Show success message with better name change options
                    success_msg = f"""✅ Your 7-day {diet_type} {cuisine_display} meal plan has been created!

            The plan includes:
            - Monday to Sunday templates
            - {profile['target_calories']} calories per day (approximately)  
            - Meals optimized for: {profile['weight_delta_text']}
            - {cuisine_display} cuisine focus
            - 7 meal slots per day (Early Morning to Dinner)
            - Proper portion sizes with units (grams, ml, pieces)

            You can now:
            - Tell me about any food allergies/restrictions (e.g., "I am allergic to peanuts", "remove dairy items")
            - Type "yes" to customize day names
            - Type "no" to finalize with current names (Monday-Sunday)"""
                    
                    yield f"data: {json.dumps({'message': success_msg, 'type': 'meal_plan_created'})}\n\n"
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
       ##################################
       # Handle name change confirmation
            # Handle name change confirmation
       # Handle name change and allergy check
       elif pending_state and pending_state.get("state") == "awaiting_name_change":
            print("DEBUG: In awaiting_name_change state")
            
            # Check for food restrictions/allergies first
            restrictions = detect_food_restrictions(text)
            if restrictions:
                print(f"DEBUG: Detected food restrictions: {restrictions}")
                
                # Regenerate meal plan without allergens
                async def _regenerate_with_restrictions():
                    try:
                        avoid_items = restrictions.get('found_allergens', []) + restrictions.get('other_foods', [])
                        avoid_text = ', '.join(avoid_items)
                        
                        yield f"data: {json.dumps({'message': f'I understand you want to avoid: {avoid_text}. Let me regenerate your complete 7-day meal plan without these items. This may take a moment...', 'type': 'allergy_detected'})}\n\n"
                        
                        # Regenerate complete 7-day meal plan (returns the full meal_plan dict)
                        new_meal_plan = regenerate_meal_plan_without_allergens(
                            pending_state.get("meal_plan"),
                            restrictions,
                            pending_state.get("profile"),
                            pending_state.get("diet_type"),
                            pending_state.get("cuisine_type")
                        )
                        
                        if new_meal_plan and len(new_meal_plan) == 7:  # Should have 7 days
                            
                            # Update pending state with new meal plan
                            await mem.set_pending(user_id, {
                                "state": "awaiting_name_change_after_allergy",
                                "client_id": pending_state.get("client_id"),
                                "profile": pending_state.get("profile"),
                                "diet_type": pending_state.get("diet_type"),
                                "cuisine_type": pending_state.get("cuisine_type"),
                                "meal_plan": new_meal_plan,  # This is now the complete 7-day plan
                                "avoided_foods": avoid_items
                            })
                            
                            # Send updated meal plan
                            yield sse_json({
                                "type": "meal_plan_updated",
                                "status": "regenerated", 
                                "avoided_foods": avoid_items,
                                "meal_plan": new_meal_plan  # Full 7-day plan with variety
                            })
                            
                            # Calculate unique meals across all days for confirmation
                            all_breakfasts = []
                            all_lunches = []
                            all_dinners = []
                            
                            for day_name, day_data in new_meal_plan.items():
                                if isinstance(day_data, list) and len(day_data) >= 7:
                                    # Extract breakfast foods (slot 3)
                                    if len(day_data) > 2:
                                        breakfast_foods = [food.get('name', '') for food in day_data[2].get('foodList', [])]
                                        all_breakfasts.extend(breakfast_foods)
                                    
                                    # Extract lunch foods (slot 5)
                                    if len(day_data) > 4:
                                        lunch_foods = [food.get('name', '') for food in day_data[4].get('foodList', [])]
                                        all_lunches.extend(lunch_foods)
                                    
                                    # Extract dinner foods (slot 7)
                                    if len(day_data) > 6:
                                        dinner_foods = [food.get('name', '') for food in day_data[6].get('foodList', [])]
                                        all_dinners.extend(dinner_foods)
                            
                            unique_breakfasts = len(set(all_breakfasts))
                            unique_lunches = len(set(all_lunches))
                            unique_dinners = len(set(all_dinners))
                            
                            success_msg = f"""✅ Your 7-day meal plan has been completely regenerated to avoid: {avoid_text}

        The new plan includes:
        - {len(new_meal_plan)} unique daily meal plans
        - {unique_breakfasts} different breakfast items across the week
        - {unique_lunches} different lunch items across the week  
        - {unique_dinners} different dinner items across the week
        - Same calorie target: {pending_state.get('profile', {}).get('target_calories', 0)} calories/day
        - Same cuisine preference: {pending_state.get('cuisine_type', '').replace('_', ' ').title()}
        - Safe alternatives for all restricted items

        Now, would you like to customize day names?
        - Type "yes" to customize day names  
        - Type "no" to finalize with current names (Monday-Sunday)"""
                            
                            yield f"data: {json.dumps({'message': success_msg, 'type': 'meal_plan_updated_success'})}\n\n"
                            yield "event: done\ndata: [DONE]\n\n"
                            
                        else:
                            print(f"DEBUG: Regeneration failed - got {len(new_meal_plan) if new_meal_plan else 0} days")
                            yield f"data: {json.dumps({'message': 'Sorry, I had trouble regenerating the complete meal plan. Would you like to continue with the original plan and customize day names instead?', 'type': 'regeneration_error'})}\n\n"
                            yield "event: done\ndata: [DONE]\n\n"
                            
                    except Exception as e:
                        print(f"Error in allergy regeneration: {e}")
                        print(f"Allergy regeneration traceback: {traceback.format_exc()}")
                        yield f"data: {json.dumps({'message': 'Error updating meal plan. Would you like to continue with day name customization?', 'type': 'error'})}\n\n"
                        yield "event: done\ndata: [DONE]\n\n"
                
                return StreamingResponse(_regenerate_with_restrictions(), media_type="text/event-stream",
                                        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
           ########################################################################## 
            # Handle regular yes/no for name change
            elif is_yes(text):
                async def _ask_which_days():
                    msg = """You can customize day names individually. Current names are:

        1. Monday    2. Tuesday    3. Wednesday    4. Thursday
        5. Friday    6. Saturday   7. Sunday

        Which day(s) would you like to rename? You can:
        - Type a number (1-7) to change one day: "3"
        - Type multiple numbers: "1, 3, 5" 
        - Type "all" to change all days
        - Type "cancel" to keep current names

        Example: Type "2" to change only Tuesday"""
                    
                    await mem.set_pending(user_id, {
                        "state": "awaiting_day_selection",
                        "client_id": pending_state.get("client_id"),
                        "profile": pending_state.get("profile"),
                        "diet_type": pending_state.get("diet_type"),
                        "cuisine_type": pending_state.get("cuisine_type"),
                        "meal_plan": pending_state.get("meal_plan"),
                        "custom_names": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
                    })
                    
                    yield f"data: {json.dumps({'message': msg, 'type': 'day_selection_request'})}\n\n"
                    yield "event: done\ndata: [DONE]\n\n"
                
                return StreamingResponse(_ask_which_days(), media_type="text/event-stream",
                                        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
            
            elif is_no(text):
                # Finalize with default names
                meal_plan = pending_state.get("meal_plan")
                await mem.clear_pending(user_id)
                
                async def _finalize_default():
                    yield f"data: {json.dumps({'message': '✅ Your 7-day meal plan is finalized!', 'type': 'finalized'})}\n\n"
                    yield sse_json({
                        "type": "meal_plan_final",
                        "status": "completed",
                        "diet_type": pending_state.get("diet_type"),
                        "meal_plan": meal_plan,
                        "day_names": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
                    })
                    yield "event: done\ndata: [DONE]\n\n"
                
                return StreamingResponse(_finalize_default(), media_type="text/event-stream",
                                        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
            
            else:
                async def _ask_name_change_again():
                    yield f"data: {json.dumps({'message': 'You can either: \\n• Tell me about any food allergies/restrictions (e.g., \"I am allergic to peanuts\")\\n• Say \"yes\" to customize day names\\n• Say \"no\" to finalize the plan', 'type': 'clarification'})}\n\n"
                    yield "event: done\ndata: [DONE]\n\n"
                
                return StreamingResponse(_ask_name_change_again(), media_type="text/event-stream",
                                        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
        #############################################################################################

        # Handle which days to change
       elif pending_state and pending_state.get("state") == "awaiting_day_selection":
            if text.lower() == "cancel":
                meal_plan = pending_state.get("meal_plan")
                await mem.clear_pending(user_id)
                
                async def _cancel_changes():
                    yield f"data: {json.dumps({'message': '✅ Your meal plan is finalized!', 'type': 'finalized'})}\n\n"
                    yield sse_json({
                        "type": "meal_plan_final", 
                        "status": "completed",
                        "diet_type": pending_state.get("diet_type"),
                        "meal_plan": meal_plan,
                        "day_names": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
                    })
                    yield "event: done\ndata: [DONE]\n\n"
                
                return StreamingResponse(_cancel_changes(), media_type="text/event-stream",
                                        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
            
            elif text.lower() == "all":
                async def _ask_all_names():
                    msg = """Please provide new names for all 7 days, separated by commas:

        Example: "Detox Day, Power Monday, Wellness Day, Fit Day, Strong Day, Active Day, Rest Day"

        Or type "cancel" to go back."""
                    
                    await mem.set_pending(user_id, {
                        "state": "awaiting_all_names",
                        "client_id": pending_state.get("client_id"),
                        "profile": pending_state.get("profile"),
                        "diet_type": pending_state.get("diet_type"),
                        "cuisine_type": pending_state.get("cuisine_type"),
                        "meal_plan": pending_state.get("meal_plan"),
                    })
                    
                    yield f"data: {json.dumps({'message': msg, 'type': 'all_names_request'})}\n\n"
                    yield "event: done\ndata: [DONE]\n\n"
                
                return StreamingResponse(_ask_all_names(), media_type="text/event-stream",
                                        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
            
            else:
                # User typed numbers like "1,3,5"
                try:
                    selected_numbers = [int(x.strip()) for x in text.split(",") if x.strip().isdigit()]
                    valid_numbers = [n for n in selected_numbers if 1 <= n <= 7]
                    
                    if valid_numbers:
                        default_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
                        
                        async def _ask_selected_names():
                            if len(valid_numbers) == 1:
                                msg = f"""You selected day {valid_numbers[0]} ({default_names[valid_numbers[0]-1]}).

        Please enter the new name for this day:

        Example: "Wellness Wednesday"

        Or type "cancel" to go back."""
                            else:
                                selected_days = [default_names[i-1] for i in valid_numbers]
                                msg = f"""You selected days: {', '.join(selected_days)}

        Please provide new names for these {len(valid_numbers)} days, separated by commas:

        Example: "Power Day, Wellness Day"

        Or type "cancel" to go back."""
                            
                            await mem.set_pending(user_id, {
                                "state": "awaiting_selected_names",
                                "client_id": pending_state.get("client_id"),
                                "profile": pending_state.get("profile"),
                                "diet_type": pending_state.get("diet_type"),
                                "cuisine_type": pending_state.get("cuisine_type"),
                                "meal_plan": pending_state.get("meal_plan"),
                                "selected_numbers": valid_numbers
                            })
                            
                            yield f"data: {json.dumps({'message': msg, 'type': 'selected_names_request'})}\n\n"
                            yield "event: done\ndata: [DONE]\n\n"
                        
                        return StreamingResponse(_ask_selected_names(), media_type="text/event-stream",
                                                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
                    else:
                        async def _invalid_selection():
                            yield f"data: {json.dumps({'message': 'Please enter valid numbers (1-7). Example: \"2\" or \"1,3,5\"', 'type': 'error'})}\n\n"
                            yield "event: done\ndata: [DONE]\n\n"
                        
                        return StreamingResponse(_invalid_selection(), media_type="text/event-stream",
                                                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
                        
                except ValueError:
                    async def _invalid_format():
                        yield f"data: {json.dumps({'message': 'Invalid format. Example: \"2\" or \"1,3,5\" or \"all\"', 'type': 'error'})}\n\n"
                        yield "event: done\ndata: [DONE]\n\n"
                    
                    return StreamingResponse(_invalid_format(), media_type="text/event-stream",
                                            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

        # Handle all names input
       elif pending_state and pending_state.get("state") == "awaiting_all_names":
            if text.lower() == "cancel":
                await mem.set_pending(user_id, {
                    "state": "awaiting_day_selection",
                    "client_id": pending_state.get("client_id"),
                    "profile": pending_state.get("profile"),
                    "diet_type": pending_state.get("diet_type"),
                    "cuisine_type": pending_state.get("cuisine_type"),
                    "meal_plan": pending_state.get("meal_plan"),
                })
                
                async def _back_to_selection():
                    yield f"data: {json.dumps({'message': 'Which days to rename? Type numbers like \"2\" or \"1,3,5\"', 'type': 'day_selection_request'})}\n\n"
                    yield "event: done\ndata: [DONE]\n\n"
                
                return StreamingResponse(_back_to_selection(), media_type="text/event-stream",
                                        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
            
            custom_names = [name.strip() for name in text.split(",") if name.strip()]
            
            if len(custom_names) == 7:
                meal_plan = pending_state.get("meal_plan")
                default_days = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
                
                updated_meal_plan = {}
                for i, custom_name in enumerate(custom_names):
                    if i < len(default_days):
                        updated_meal_plan[custom_name.lower().replace(" ", "_")] = meal_plan.get(default_days[i], get_meal_template())
                
                await mem.clear_pending(user_id)
                
                async def _finalize_all_custom():
                    yield f"data: {json.dumps({'message': f'✅ Your meal plan is finalized with names: {', '.join(custom_names)}!', 'type': 'finalized'})}\n\n"
                    yield sse_json({
                        "type": "meal_plan_final",
                        "status": "completed", 
                        "diet_type": pending_state.get("diet_type"),
                        "meal_plan": updated_meal_plan,
                        "day_names": custom_names
                    })
                    yield "event: done\ndata: [DONE]\n\n"
                
                return StreamingResponse(_finalize_all_custom(), media_type="text/event-stream",
                                        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
            else:
                async def _invalid_all_names():
                    yield f"data: {json.dumps({'message': f'Need exactly 7 names. You gave {len(custom_names)}. Try again.', 'type': 'error'})}\n\n"
                    yield "event: done\ndata: [DONE]\n\n"
                
                return StreamingResponse(_invalid_all_names(), media_type="text/event-stream",
                                        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

        # Handle selected names input
       elif pending_state and pending_state.get("state") == "awaiting_selected_names":
            if text.lower() == "cancel":
                await mem.set_pending(user_id, {
                    "state": "awaiting_day_selection",
                    "client_id": pending_state.get("client_id"),
                    "profile": pending_state.get("profile"),
                    "diet_type": pending_state.get("diet_type"),
                    "cuisine_type": pending_state.get("cuisine_type"),
                    "meal_plan": pending_state.get("meal_plan"),
                })
                
                async def _back_to_selection2():
                    yield f"data: {json.dumps({'message': 'Which days to rename? Type numbers like \"2\" or \"1,3,5\"', 'type': 'day_selection_request'})}\n\n"
                    yield "event: done\ndata: [DONE]\n\n"
                
                return StreamingResponse(_back_to_selection2(), media_type="text/event-stream",
                                        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
            
            selected_numbers = pending_state.get("selected_numbers", [])
            custom_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            
            new_names = [name.strip() for name in text.split(",") if name.strip()]
            
            if len(new_names) == len(selected_numbers):
                # Update selected days with new names
                for i, day_num in enumerate(selected_numbers):
                    if i < len(new_names) and 1 <= day_num <= 7:
                        custom_names[day_num - 1] = new_names[i]
                
                # Finalize with updated names
                meal_plan = pending_state.get("meal_plan")
                default_days = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
                
                updated_meal_plan = {}
                for i, custom_name in enumerate(custom_names):
                    if i < len(default_days):
                        updated_meal_plan[custom_name.lower().replace(" ", "_")] = meal_plan.get(default_days[i], get_meal_template())
                
                await mem.clear_pending(user_id)
                
                async def _finalize_selected():
                    yield f"data: {json.dumps({'message': f'✅ Your meal plan is finalized with names: {', '.join(custom_names)}!', 'type': 'finalized'})}\n\n"
                    yield sse_json({
                        "type": "meal_plan_final",
                        "status": "completed",
                        "diet_type": pending_state.get("diet_type"),
                        "meal_plan": updated_meal_plan,
                        "day_names": custom_names
                    })
                    yield "event: done\ndata: [DONE]\n\n"
                
                return StreamingResponse(_finalize_selected(), media_type="text/event-stream",
                                        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
            else:
                async def _invalid_selected_names():
                    yield f"data: {json.dumps({'message': f'Need exactly {len(selected_numbers)} names. You gave {len(new_names)}.', 'type': 'error'})}\n\n"
                    yield "event: done\ndata: [DONE]\n\n"
                
                return StreamingResponse(_invalid_selected_names(), media_type="text/event-stream",
                                        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

       # Handle name change after allergy update
       elif pending_state and pending_state.get("state") == "awaiting_name_change_after_allergy":
            print("DEBUG: In awaiting_name_change_after_allergy state")
            
            if is_yes(text):
                async def _ask_which_days_after_allergy():
                    msg = """You can customize day names individually. Current names are:

        1. Monday    2. Tuesday    3. Wednesday    4. Thursday
        5. Friday    6. Saturday   7. Sunday

        Which day(s) would you like to rename? You can:
        - Type a number (1-7) to change one day: "3"
        - Type multiple numbers: "1, 3, 5" 
        - Type "all" to change all days
        - Type "cancel" to keep current names

        Example: Type "2" to change only Tuesday"""
                    
                    await mem.set_pending(user_id, {
                        "state": "awaiting_day_selection",
                        "client_id": pending_state.get("client_id"),
                        "profile": pending_state.get("profile"),
                        "diet_type": pending_state.get("diet_type"),
                        "cuisine_type": pending_state.get("cuisine_type"),
                        "meal_plan": pending_state.get("meal_plan"),
                        "avoided_foods": pending_state.get("avoided_foods"),
                        "custom_names": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
                    })
                    
                    yield f"data: {json.dumps({'message': msg, 'type': 'day_selection_request'})}\n\n"
                    yield "event: done\ndata: [DONE]\n\n"
                
                return StreamingResponse(_ask_which_days_after_allergy(), media_type="text/event-stream",
                                        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
            
            elif is_no(text):
                # Finalize with default names
                meal_plan = pending_state.get("meal_plan")
                await mem.clear_pending(user_id)
                
                async def _finalize_default_after_allergy():
                    yield f"data: {json.dumps({'message': '✅ Your 7-day meal plan is finalized!', 'type': 'finalized'})}\n\n"
                    yield sse_json({
                        "type": "meal_plan_final",
                        "status": "completed",
                        "diet_type": pending_state.get("diet_type"),
                        "meal_plan": meal_plan,
                        "day_names": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
                    })
                    yield "event: done\ndata: [DONE]\n\n"
                
                return StreamingResponse(_finalize_default_after_allergy(), media_type="text/event-stream",
                                        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
            
            else:
                async def _ask_name_change_again_after_allergy():
                    yield f"data: {json.dumps({'message': 'Do you want to customize day names? Please say \"yes\" or \"no\".', 'type': 'clarification'})}\n\n"
                    yield "event: done\ndata: [DONE]\n\n"
                
                return StreamingResponse(_ask_name_change_again_after_allergy(), media_type="text/event-stream",
                                        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
      
       # Handle case where no pending state exists but user is sending text
       else:
           print(f"DEBUG: No valid pending state found. Current state: {pending_state}")
           
           # If no pending state exists, treat this as the first interaction and start fresh
           try:
               profile = _fetch_profile(db, client_id)
               print(f"DEBUG: Starting fresh conversation with profile: {profile}")
              
               async def _start_fresh():
                   welcome_msg = f"""Hello! I'm your meal template assistant.

I can see your profile:
• Current Weight: {profile['current_weight']} kg
• Target Weight: {profile['target_weight']} kg 
• Goal: {profile['weight_delta_text']}
• Daily Calorie Target: {profile['target_calories']} calories

I'll create a personalized 7-day meal template for you. First, are you vegetarian or non-vegetarian or eggetarian or vegan or ketogenic or paleo or jain?

(You said: "{text}")"""
                  
                   await mem.set_pending(user_id, {
                       "state": "awaiting_diet_preference",
                       "client_id": client_id,
                       "profile": profile
                   })
                  
                   yield f"data: {json.dumps({'message': welcome_msg, 'type': 'welcome'})}\n\n"
                   yield "event: done\ndata: [DONE]\n\n"
              
               return StreamingResponse(_start_fresh(), media_type="text/event-stream",
                                       headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
                                      
           except Exception as e:
               print(f"Error starting fresh conversation: {e}")
               print(f"Fresh start traceback: {traceback.format_exc()}")
               async def _error_start():
                   yield f"data: {json.dumps({'message': 'Error starting conversation. Please check the client ID and try again.', 'type': 'error'})}\n\n"
                   yield "event: done\ndata: [DONE]\n\n"
              
               return StreamingResponse(_error_start(), media_type="text/event-stream",
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
