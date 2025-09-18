#food_log
from fastapi import APIRouter, HTTPException, Query, Depends
from fastapi.responses import StreamingResponse
from fastapi_limiter.depends import RateLimiter
from pydantic import BaseModel
import pytz, os, hashlib, orjson, re, json
from datetime import datetime
from sqlalchemy.orm import Session
from app.models.database import get_db
from app.models.deps import get_http, get_oai, get_mem
from openai import OpenAI
import google.generativeai as genai
import json, re, os


from app.fittbot_api.v1.client.client_api.chatbot.chatbot_services.llm_helpers import (
    PlainTextStreamFilter, oai_chat_stream, GENERAL_SYSTEM, TOP_K,
    build_messages, heuristic_confidence, gpt_extract_items, first_missing_quantity,OPENAI_MODEL,
    sse_json, sse_escape, gpt_small_route, _scale_macros, is_yes, is_no,is_fit_chat,
    has_action_verb, food_hits,ensure_per_unit_macros, is_fittbot_meta_query,normalize_food, explicit_log_command, STYLE_PLAN, is_plan_request,STYLE_CHAT_FORMAT,pretty_plan
)



router = APIRouter(prefix="/food_log", tags=["food_log"])

APP_ENV = os.getenv("APP_ENV", "prod")
TZNAME = os.getenv("TZ", "Asia/Kolkata")
IST = pytz.timezone(TZNAME)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Validate API keys
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is not set")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is not set")

# Initialize OpenAI client
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-1.5-flash")

@router.get("/healthz")
async def healthz():
    return {"ok": True, "env": APP_ENV, "tz": TZNAME}

def get_food_config(food_name):
    """Get comprehensive food configuration including nutrition and preferred units"""
    food_name_lower = food_name.lower()
    
    # Comprehensive food database with preferred units and conversions
    food_configs = {
        # Rice and grains - prefer plates/bowls
        'rice': {
            'preferred_unit': 'plates',
            'conversions': {'plates': 150, 'bowls': 100, 'cups': 180, 'grams': 1},
            'nutrition_per_100g': {'calories': 130, 'protein': 2.7, 'carbs': 28, 'fat': 0.3, 'fiber': 0.4, 'sugar': 0.1}
        },
        'basmati rice': {
            'preferred_unit': 'plates',
            'conversions': {'plates': 150, 'bowls': 100, 'cups': 180, 'grams': 1},
            'nutrition_per_100g': {'calories': 130, 'protein': 2.7, 'carbs': 28, 'fat': 0.3, 'fiber': 0.4, 'sugar': 0.1}
        },
        'brown rice': {
            'preferred_unit': 'plates',
            'conversions': {'plates': 150, 'bowls': 100, 'cups': 180, 'grams': 1},
            'nutrition_per_100g': {'calories': 111, 'protein': 2.6, 'carbs': 22, 'fat': 0.9, 'fiber': 1.8, 'sugar': 0.4}
        },
        'wheat': {
            'preferred_unit': 'bowls',
            'conversions': {'bowls': 120, 'plates': 180, 'cups': 150, 'grams': 1},
            'nutrition_per_100g': {'calories': 340, 'protein': 13, 'carbs': 72, 'fat': 2.5, 'fiber': 11, 'sugar': 0.4}
        },
        'quinoa': {
            'preferred_unit': 'bowls',
            'conversions': {'bowls': 120, 'plates': 180, 'cups': 150, 'grams': 1},
            'nutrition_per_100g': {'calories': 120, 'protein': 4.4, 'carbs': 22, 'fat': 1.9, 'fiber': 2.8, 'sugar': 0.9}
        },
        
        # Fruits - prefer pieces
        'apple': {
            'preferred_unit': 'pieces',
            'conversions': {'pieces': 150, 'slices': 25, 'grams': 1},
            'nutrition_per_100g': {'calories': 52, 'protein': 0.3, 'carbs': 14, 'fat': 0.2, 'fiber': 2.4, 'sugar': 10}
        },
        'banana': {
            'preferred_unit': 'pieces',
            'conversions': {'pieces': 120, 'slices': 20, 'grams': 1},
            'nutrition_per_100g': {'calories': 89, 'protein': 1.1, 'carbs': 23, 'fat': 0.3, 'fiber': 2.6, 'sugar': 12}
        },
        'orange': {
            'preferred_unit': 'pieces',
            'conversions': {'pieces': 130, 'slices': 25, 'grams': 1},
            'nutrition_per_100g': {'calories': 47, 'protein': 0.9, 'carbs': 12, 'fat': 0.1, 'fiber': 2.4, 'sugar': 9}
        },
        'avocado': {
            'preferred_unit': 'pieces',
            'conversions': {'pieces': 200, 'slices': 30, 'grams': 1},
            'nutrition_per_100g': {'calories': 160, 'protein': 2, 'carbs': 9, 'fat': 15, 'fiber': 7, 'sugar': 1}
        },
        'watermelon': {
            'preferred_unit': 'slices',
            'conversions': {'slices': 150, 'pieces': 400, 'grams': 1},
            'nutrition_per_100g': {'calories': 30, 'protein': 0.6, 'carbs': 8, 'fat': 0.2, 'fiber': 0.4, 'sugar': 6}
        },
        'jackfruit': {
            'preferred_unit': 'pieces',
            'conversions': {'pieces': 100, 'slices': 50, 'grams': 1},
            'nutrition_per_100g': {'calories': 95, 'protein': 1.7, 'carbs': 23, 'fat': 0.6, 'fiber': 1.5, 'sugar': 19}
        },
        'mango': {
            'preferred_unit': 'pieces',
            'conversions': {'pieces': 150, 'slices': 30, 'grams': 1},
            'nutrition_per_100g': {'calories': 60, 'protein': 0.8, 'carbs': 15, 'fat': 0.4, 'fiber': 1.6, 'sugar': 13.7}
        },
        
        # Juices and beverages - prefer ml
        'avocado juice': {
            'preferred_unit': 'ml',
            'conversions': {'ml': 1, 'cups': 240, 'glasses': 200},
            'nutrition_per_100g': {'calories': 80, 'protein': 1.5, 'carbs': 8, 'fat': 5, 'fiber': 2, 'sugar': 6}
        },
        'orange juice': {
            'preferred_unit': 'ml',
            'conversions': {'ml': 1, 'cups': 240, 'glasses': 200},
            'nutrition_per_100g': {'calories': 45, 'protein': 0.7, 'carbs': 10, 'fat': 0.2, 'fiber': 0.2, 'sugar': 8}
        },
        'apple juice': {
            'preferred_unit': 'ml',
            'conversions': {'ml': 1, 'cups': 240, 'glasses': 200},
            'nutrition_per_100g': {'calories': 46, 'protein': 0.1, 'carbs': 11, 'fat': 0.1, 'fiber': 0.2, 'sugar': 10}
        },
        'sugar cane juice': {
            'preferred_unit': 'ml',
            'conversions': {'ml': 1, 'cups': 240, 'glasses': 200},
            'nutrition_per_100g': {'calories': 45, 'protein': 0.1, 'carbs': 11, 'fat': 0, 'fiber': 0, 'sugar': 11}
        },
        'lime juice': {
            'preferred_unit': 'ml',
            'conversions': {'ml': 1, 'cups': 240, 'glasses': 200},
            'nutrition_per_100g': {'calories': 25, 'protein': 0.4, 'carbs': 8, 'fat': 0.2, 'fiber': 0.4, 'sugar': 1.5}
        },
        'lemon juice': {
            'preferred_unit': 'ml',
            'conversions': {'ml': 1, 'cups': 240, 'glasses': 200},
            'nutrition_per_100g': {'calories': 22, 'protein': 0.4, 'carbs': 6.9, 'fat': 0.2, 'fiber': 0.3, 'sugar': 2.5}
        },
        'mango juice': {
            'preferred_unit': 'ml',
            'conversions': {'ml': 1, 'cups': 240, 'glasses': 200},
            'nutrition_per_100g': {'calories': 60, 'protein': 0.4, 'carbs': 15, 'fat': 0.2, 'fiber': 0.4, 'sugar': 13}
        },
        
        # Vegetables - prefer pieces or grams
        'carrot': {
            'preferred_unit': 'pieces',
            'conversions': {'pieces': 60, 'slices': 10, 'grams': 1},
            'nutrition_per_100g': {'calories': 41, 'protein': 0.9, 'carbs': 10, 'fat': 0.2, 'fiber': 2.8, 'sugar': 4.7}
        },
        'tomato': {
            'preferred_unit': 'pieces',
            'conversions': {'pieces': 100, 'slices': 15, 'grams': 1},
            'nutrition_per_100g': {'calories': 18, 'protein': 0.9, 'carbs': 3.9, 'fat': 0.2, 'fiber': 1.2, 'sugar': 2.6}
        },
        'onion': {
            'preferred_unit': 'pieces',
            'conversions': {'pieces': 110, 'slices': 15, 'grams': 1},
            'nutrition_per_100g': {'calories': 40, 'protein': 1.1, 'carbs': 9.3, 'fat': 0.1, 'fiber': 1.7, 'sugar': 4.2}
        },
        'potato': {
            'preferred_unit': 'pieces',
            'conversions': {'pieces': 150, 'slices': 20, 'grams': 1},
            'nutrition_per_100g': {'calories': 77, 'protein': 2, 'carbs': 17, 'fat': 0.1, 'fiber': 2.2, 'sugar': 0.8}
        },
        
        # Proteins
        'chicken': {
            'preferred_unit': 'grams',
            'conversions': {'grams': 1, 'pieces': 150, 'kg': 1000},
            'nutrition_per_100g': {'calories': 165, 'protein': 31, 'carbs': 0, 'fat': 3.6, 'fiber': 0, 'sugar': 0}
        },
        'fish': {
            'preferred_unit': 'grams',
            'conversions': {'grams': 1, 'pieces': 120, 'kg': 1000},
            'nutrition_per_100g': {'calories': 140, 'protein': 25, 'carbs': 0, 'fat': 4, 'fiber': 0, 'sugar': 0}
        },
        'egg': {
            'preferred_unit': 'pieces',
            'conversions': {'pieces': 50, 'grams': 1},
            'nutrition_per_100g': {'calories': 155, 'protein': 13, 'carbs': 1.1, 'fat': 11, 'fiber': 0, 'sugar': 1.1}
        },
        
        # Dairy
        'milk': {
            'preferred_unit': 'ml',
            'conversions': {'ml': 1, 'cups': 240, 'glasses': 200},
            'nutrition_per_100g': {'calories': 42, 'protein': 3.4, 'carbs': 5, 'fat': 1, 'fiber': 0, 'sugar': 5}
        },
        'yogurt': {
            'preferred_unit': 'grams',
            'conversions': {'grams': 1, 'cups': 240, 'bowls': 150},
            'nutrition_per_100g': {'calories': 59, 'protein': 10, 'carbs': 3.6, 'fat': 0.4, 'fiber': 0, 'sugar': 3.2}
        },
    }
    
    # Try exact match first
    if food_name_lower in food_configs:
        return food_configs[food_name_lower]
    
    # Try partial matches
    for key in food_configs:
        if key in food_name_lower or food_name_lower in key:
            return food_configs[key]
    
    # Default config based on food type detection
    if 'juice' in food_name_lower or 'drink' in food_name_lower:
        return {
            'preferred_unit': 'ml',
            'conversions': {'ml': 1, 'cups': 240, 'glasses': 200},
            'nutrition_per_100g': {'calories': 40, 'protein': 0.5, 'carbs': 10, 'fat': 0.1, 'fiber': 0.1, 'sugar': 8}
        }
    elif 'rice' in food_name_lower or 'grain' in food_name_lower:
        return {
            'preferred_unit': 'plates',
            'conversions': {'plates': 250, 'bowls': 100, 'cups': 180, 'grams': 1},
            'nutrition_per_100g': {'calories': 130, 'protein': 2.7, 'carbs': 28, 'fat': 0.3, 'fiber': 0.4, 'sugar': 0.1}
        }
    else:
        return {
            'preferred_unit': 'pieces',
            'conversions': {'pieces': 100, 'slices': 20, 'grams': 1},
            'nutrition_per_100g': {'calories': 50, 'protein': 1, 'carbs': 12, 'fat': 0.2, 'fiber': 2, 'sugar': 8}
        }

def convert_to_grams(quantity, unit, food_name):
    """Convert any unit to grams for nutrition calculation"""
    config = get_food_config(food_name)
    conversions = config.get('conversions', {})
    
    # Normalize unit names
    unit_normalized = normalize_unit(unit)
    
    if unit_normalized in conversions:
        grams = quantity * conversions[unit_normalized]
        print(f"DEBUG: Converting {quantity} {unit} of {food_name} = {grams}g")
        return grams
    
    # If no conversion found, assume it's already in grams
    print(f"DEBUG: No conversion for {unit}, assuming grams: {quantity}g")
    return quantity

def normalize_unit(unit):
    """Normalize unit names to standard forms"""
    if not unit:
        return 'pieces'
    
    unit_lower = unit.lower().strip()
    
    # Unit mappings
    unit_map = {
        'g': 'grams', 'gram': 'grams', 'gms': 'grams',
        'kg': 'kg', 'kilogram': 'kg', 'kilograms': 'kg',
        'piece': 'pieces', 'pcs': 'pieces', 'pc': 'pieces',
        'slice': 'slices', 'slc': 'slices',
        'plate': 'plates', 'bowl': 'bowls',
        'cup': 'cups', 'glass': 'glasses',
        'ml': 'ml', 'milliliter': 'ml', 'milliliters': 'ml',
        'liter': 'liters', 'litre': 'liters', 'l': 'liters',
        'can': 'cans', 'tin': 'cans'
    }
    
    return unit_map.get(unit_lower, unit_lower)

def calculate_nutrition_for_quantity(food_name, quantity, unit):
    """Calculate nutrition values for a specific quantity"""
    if not quantity or quantity <= 0:
        return {'calories': 0, 'protein': 0, 'carbs': 0, 'fat': 0, 'fiber': 0, 'sugar': 0}
    
    # Convert to grams first
    grams = convert_to_grams(quantity, unit, food_name)
    
    # Get nutrition per 100g
    config = get_food_config(food_name)
    nutrition_per_100g = config.get('nutrition_per_100g', {})
    
    # Calculate nutrition for the actual grams
    ratio = grams / 100.0
    calculated_nutrition = {}
    
    for key, value in nutrition_per_100g.items():
        calculated_nutrition[key] = round(value * ratio, 2)
    
    print(f"DEBUG: Nutrition calculation for {quantity} {unit} of {food_name}:")
    print(f"DEBUG: Converted to {grams}g, ratio: {ratio}")
    print(f"DEBUG: Final nutrition: {calculated_nutrition}")
    
    return calculated_nutrition

def get_unit_hint(unit):
    """Generate unit hint for the specified unit"""
    unit_hints = {
        'pieces': 'How many pieces?',
        'plates': 'How many plates?',
        'bowls': 'How many bowls?',
        'cups': 'How many cups?',
        'glasses': 'How many glasses?',
        'slices': 'How many slices?',
        'ml': 'How much ml?',
        'grams': 'How many grams?',
        'kg': 'How many kg?'
    }
    return unit_hints.get(unit, f'How many {unit}?')

def convert_food_to_item_format(food):
    """Convert food object to items format"""
    return {
        "food": food.get('name', ''),
        "unit": food.get('unit', 'pieces'),
        "quantity": food.get('quantity'),
        "calories": food.get('calories', 0),
        "protein": food.get('protein', 0),
        "carbs": food.get('carbs', 0),
        "fat": food.get('fat', 0),
        "fiber": food.get('fiber', 0),
        "sugar": food.get('sugar', 0),
        "unit_hint": get_unit_hint(food.get('unit', 'pieces')),
        "ask": food.get('quantity') is None,
        "qty_from": "provided" if food.get('quantity') is not None else "ask"
    }

def extract_food_info_using_openai(text: str):
    """Enhanced AI-driven food extraction with better food recognition and unit handling"""
    try:
        reasoning_prompt = f"""
        You are analyzing this text: "{text}"

        FOOD IDENTIFICATION RULES:
        1. Extract ANY item that could reasonably be food, beverage, or drink
        2. Include ALL fruits, vegetables, grains, meats, dairy, beverages, juices, snacks
        3. Be very permissive - if it might be food, include it
        4. Handle common misspellings and normalize names
        5. Recognize compound food names (sugarcanjuice = sugar cane juice, etc.)
        
        QUANTITY AND UNIT RULES:
        1. Only set quantity if user explicitly provided a specific amount
        2. If just food name without quantity, set quantity to null
        3. Use the most natural unit for each food type:
           - Rice/grains: prefer "plates" or "bowls"
           - Juices/liquids: prefer "ml" or "glasses"  
           - Fruits/vegetables: prefer "pieces"
           - Meat/protein: prefer "grams"
        4. If user provides quantity with unit, use their unit
        5. If user provides just number, use the natural unit for that food
        
        EXAMPLES:
        - "rice" → quantity: null, unit: "plates"
        - "2 plates rice" → quantity: 2, unit: "plates"  
        - "500g rice" → quantity: 500, unit: "grams"
        - "apple juice" → quantity: null, unit: "ml"
        - "200ml orange juice" → quantity: 200, unit: "ml"
        - "2 apples" → quantity: 2, unit: "pieces"

        Return valid JSON:
        {{
            "foods": [
                {{
                    "name": "normalized_food_name",
                    "quantity": number_or_null,
                    "unit": "most_appropriate_unit"
                }}
            ]
        }}
        
        Do NOT include nutrition values in the response - they will be calculated separately.
        """

        # Try OpenAI first
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system", 
                    "content": """You are a food recognition expert. Focus only on identifying foods and their quantities/units. 
                    Be very permissive with food identification and use natural units for each food type."""
                },
                {"role": "user", "content": reasoning_prompt}
            ],
            max_tokens=800,
            temperature=0.1
        )

        result = response.choices[0].message.content.strip()
        result = re.sub(r"^```json\s*", "", result)
        result = re.sub(r"\s*```$", "", result)
        
        parsed_result = json.loads(result)
        foods = parsed_result.get("foods", [])
        
        # Process each food to add nutrition info
        processed_foods = []
        for food in foods:
            food_name = food.get('name', '').strip()
            if not food_name:
                continue
                
            # Get preferred unit if none specified
            if not food.get('unit'):
                config = get_food_config(food_name)
                food['unit'] = config.get('preferred_unit', 'pieces')
            
            processed_foods.append(food)
        
        print(f"DEBUG: Processed foods: {[(f['name'], f.get('quantity'), f['unit']) for f in processed_foods]}")
        return {"foods": processed_foods}

    except Exception as e:
        print(f"OpenAI extraction error: {e}, trying Gemini fallback...")
        
        # Try Gemini fallback
        try:
            response = gemini_model.generate_content(reasoning_prompt)
            
            if response.text:
                result = response.text.strip()
                result = re.sub(r"^```json\s*", "", result)
                result = re.sub(r"\s*```$", "", result)
                
                parsed_result = json.loads(result)
                foods = parsed_result.get("foods", [])
                
                # Process each food to add nutrition info
                processed_foods = []
                for food in foods:
                    food_name = food.get('name', '').strip()
                    if not food_name:
                        continue
                        
                    # Get preferred unit if none specified
                    if not food.get('unit'):
                        config = get_food_config(food_name)
                        food['unit'] = config.get('preferred_unit', 'pieces')
                    
                    processed_foods.append(food)
                
                print(f"DEBUG: Gemini fallback processed foods: {[(f['name'], f.get('quantity'), f['unit']) for f in processed_foods]}")
                return {"foods": processed_foods}
                
        except Exception as gemini_error:
            print(f"Gemini extraction error: {gemini_error}")
        
        # Enhanced fallback parsing
        text_lower = text.lower().strip()
        
        # Handle common corrections
        corrections = {
            'avacado': 'avocado', 'avacadojuice': 'avocado juice',
            'sugarcanjuice': 'sugar cane juice', 'orangejuice': 'orange juice'
        }
        
        for wrong, correct in corrections.items():
            text_lower = text_lower.replace(wrong, correct)
        
        # Extract quantity and unit
        quantity_match = re.search(r'(\d+(?:\.\d+)?)\s*(g|grams?|plates?|bowls?|pieces?|ml|glasses?|cups?|kg)?', text_lower)
        
        if quantity_match:
            quantity = float(quantity_match.group(1))
            unit = quantity_match.group(2) if quantity_match.group(2) else None
        else:
            quantity = None
            unit = None
        
        # Get food name (remove quantity part)
        food_name = re.sub(r'\d+(?:\.\d+)?\s*\w*', '', text_lower).strip()
        
        if food_name:
            config = get_food_config(food_name)
            if not unit:
                unit = config.get('preferred_unit', 'pieces')
            else:
                unit = normalize_unit(unit)
            
            return {"foods": [{"name": food_name,"unit": unit,"quantity": quantity,}]}
        
        return {"foods": []}

def is_food_related(text):
    """Enhanced food detection"""
    if not text:
        return False
    
    text_lower = text.lower().strip()
    
    # Short inputs are likely food names
    words = text_lower.split()
    if len(words) <= 4:
        non_food_patterns = {
            'hello', 'hi', 'hey', 'thanks', 'help', 'what', 'how', 'when', 'where',
            'good morning', 'good evening', 'bye', 'goodbye'
        }
        
        if text_lower in non_food_patterns:
            return False
        return True
    
    # Check for food indicators
    food_indicators = [
        'ate', 'eat', 'eating', 'had', 'drink', 'drinking', 'consumed', 'meal',
        'juice', 'fruit', 'rice', 'food', 'breakfast', 'lunch', 'dinner'
    ]
    
    return any(indicator in text_lower for indicator in food_indicators)

def create_food_log_response_with_message(logged_foods):
    """Create food log response with summary message and nutrition totals"""
    items = [convert_food_to_item_format(food) for food in logged_foods]
    
    # Calculate total nutrition
    total_nutrition = {
        'calories': 0,
        'protein': 0,
        'carbs': 0,
        'fat': 0,
        'fiber': 0,
        'sugar': 0
    }
    
    # Create food summary and calculate totals
    food_summaries = []
    for food in logged_foods:
        quantity = food.get('quantity', 0)
        unit = food.get('unit', 'pieces')
        name = food.get('name', '')
        
        # Add to summary
        food_summaries.append(f"{quantity} {unit} of {name}")
        
        # Add to totals
        for nutrient in total_nutrition:
            total_nutrition[nutrient] += food.get(nutrient, 0)
    
    # Round totals to 1 decimal place
    for nutrient in total_nutrition:
        total_nutrition[nutrient] = round(total_nutrition[nutrient], 1)
    
    # Create message
    if len(food_summaries) == 1:
        message = f"✅ Logged {food_summaries[0]}! "
    elif len(food_summaries) == 2:
        message = f"✅ Logged {food_summaries[0]} and {food_summaries[1]}! "
    else:
        message = f"✅ Logged {', '.join(food_summaries[:-1])} and {food_summaries[-1]}! "
    
    # Add nutrition info to message
    message += f"\n📊 Nutrition: {total_nutrition['calories']} calories, {total_nutrition['protein']}g protein, {total_nutrition['carbs']}g carbs, {total_nutrition['fat']}g fat"
    
    return {
        "type": "food_log",
        "status": "logged", 
        "is_log": True,
        "message": message,
        "items": items,

    }

@router.get("/chat/stream_test", dependencies=[Depends(RateLimiter(times=30, seconds=60))])
async def chat_stream(
    user_id: int,
    text: str = Query(None),
    mem = Depends(get_mem),
    oai = Depends(get_oai),
    db: Session = Depends(get_db),
):
    try:
        if not text:
            async def _welcome():
                welcome_msg = "Hello! I'm your food logging assistant. What would you like to log today?"
                yield f"data: {json.dumps({'message': welcome_msg, 'type': 'welcome'})}\n\n"
                yield "event: done\ndata: [DONE]\n\n"
            
            return StreamingResponse(_welcome(), media_type="text/event-stream",
                                headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})
        
        text = text.strip()
        print(f"DEBUG: Processing text: '{text}'")
        
        # Check for pending state and debug
        pending_state = await mem.get_pending(user_id)
        print(f"DEBUG: Current pending state: {pending_state}")
        
        # Handle navigation confirmation
        if pending_state and pending_state.get("state") == "awaiting_nav_confirm":
            print("DEBUG: In awaiting_nav_confirm state")
            
            if is_yes(text):
                print("DEBUG: User said yes to navigation")
                await mem.clear_pending(user_id)
                async def _nav_yes():
                    yield sse_json({"type":"nav","is_navigation": True,
                                    "prompt":"Thanks for your confirmation. Redirecting to today's diet logs"})
                    yield "event: done\ndata: [DONE]\n\n"
                return StreamingResponse(_nav_yes(), media_type="text/event-stream",
                                        headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})
            
            elif is_no(text):
                print("DEBUG: User said no to navigation")
                await mem.clear_pending(user_id)
                async def _nav_no():
                    yield sse_json({"type":"nav","is_navigation": False,
                                    "prompt":"Thanks for your response. You can continue chatting here."})
                    yield "event: done\ndata: [DONE]\n\n"
                return StreamingResponse(_nav_no(), media_type="text/event-stream",
                                        headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})
            
            elif is_food_related(text):
                print("DEBUG: User entered food during nav confirmation, clearing state")
                await mem.clear_pending(user_id)
                # Continue to food processing below (don't return here)
            
            else:
                print("DEBUG: User input not recognized, asking for nav confirmation again")
                async def _nav_clar():
                    yield f"data: {json.dumps({'message': 'Do you want to go to your diet log? Please say Yes or No.', 'type': 'nav_confirm'})}\n\n"
                    yield "event: done\ndata: [DONE]\n\n"
                return StreamingResponse(_nav_clar(), media_type="text/event-stream",
                                     headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})

        # Handle pending food confirmation
        elif pending_state and pending_state.get("state") == "awaiting_pending_confirm":
            print("DEBUG: In awaiting_pending_confirm state")
            pending_foods = pending_state.get("pending_foods", [])
            logged_foods = pending_state.get("logged_foods", [])
            
            if is_yes(text):
                print("DEBUG: User wants to log pending foods")
                first_pending = pending_foods[0] if pending_foods else None
                if first_pending:
                    if first_pending['unit'] == 'ml':
                        ask_message = f"How many ml or cups of {first_pending['name']} did you have?"
                    else:
                        ask_message = f"How many {first_pending['unit']} of {first_pending['name']} did you have?"
                                        
                    await mem.set_pending(user_id, {
                        "state": "awaiting_quantity",
                        "foods": pending_foods,
                        "current_food_index": 0,
                        "logged_foods": logged_foods,
                        "original_input": pending_state.get("original_input", "")
                    })
                    
                    async def _ask_pending():
                        yield f"data: {json.dumps({'message': ask_message, 'type': 'ask_quantity', 'food': first_pending['name']})}\n\n"
                        yield "event: done\ndata: [DONE]\n\n"
                    
                    return StreamingResponse(_ask_pending(), media_type="text/event-stream",
                                            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
            
            elif is_no(text):
                print("DEBUG: User doesn't want pending foods")
                await mem.clear_pending(user_id)
                
                if logged_foods:
                    items = [convert_food_to_item_format(food) for food in logged_foods]
                    
                    async def _final_logged():
                        response_data = create_food_log_response_with_message(logged_foods)
                        yield sse_json(response_data)
                        yield "event: ping\ndata: {}\n\n"
                        yield "event: done\ndata: [DONE]\n\n"
                        await mem.set_pending(user_id, {"state":"awaiting_nav_confirm"})
                    return StreamingResponse(_final_logged(), media_type="text/event-stream",
                                            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
                else:
                    async def _no_logged():
                        yield f"data: {json.dumps({'message': 'Okay, nothing logged. What else would you like to log?', 'type': 'response'})}\n\n"
                        yield "event: done\ndata: [DONE]\n\n"
                    
                    return StreamingResponse(_no_logged(), media_type="text/event-stream",
                                            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
            
            elif is_food_related(text):
                print("DEBUG: User entered food during pending confirmation")
                await mem.clear_pending(user_id)
                # Continue to food processing below
            
            else:
                print("DEBUG: Asking for pending confirmation again")
                pending_names = [food['name'] for food in pending_foods]
                ask_message = f"Do you want to log these foods: {', '.join(pending_names)}? Please say Yes or No."
                
                async def _ask_confirm_again():
                    yield f"data: {json.dumps({'message': ask_message, 'type': 'confirm_pending'})}\n\n"
                    yield "event: done\ndata: [DONE]\n\n"
                
                return StreamingResponse(_ask_confirm_again(), media_type="text/event-stream",
                                        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

        # Handle quantity input
        elif pending_state and pending_state.get("state") == "awaiting_quantity":
            print("DEBUG: In awaiting_quantity state")
            try:
                foods = pending_state.get("foods", [])
                current_index = pending_state.get("current_food_index", 0)
                current_food = foods[current_index] if current_index < len(foods) else None
                logged_foods = pending_state.get("logged_foods", [])
                
                if not current_food:
                    print("DEBUG: No current food found, clearing state")
                    await mem.clear_pending(user_id)
                    raise Exception("No current food found")
                
                # Check if user entered a new food instead of quantity
                if is_food_related(text) and not is_quantity_input(text):
                    print("DEBUG: User entered new food during quantity request")
                    # User entered new food, extract it
                    new_food_info = extract_food_info_using_openai(text)
                    new_foods = new_food_info.get("foods", [])
                    
                    if new_foods:
                        # Check if new foods have quantities
                        foods_with_quantity = [f for f in new_foods if f.get("quantity") is not None]
                        foods_without_quantity = [f for f in new_foods if f.get("quantity") is None]
                        
                        # Calculate nutrition for foods with quantities and add to logged
                        for food in foods_with_quantity:
                            nutrition = calculate_nutrition_for_quantity(
                                food['name'], food['quantity'], food['unit']
                            )
                            food.update(nutrition)
                            logged_foods.append(food)
                        
                        # Combine all pending foods (previous + new without quantity)
                        all_pending_foods = foods + foods_without_quantity
                        
                        if foods_with_quantity:
                            # Log the foods with quantities immediately
                            food_list = []
                            for food in foods_with_quantity:
                                food_list.append(f"{food['quantity']} {food['unit']} of {food['name']}")
                            
                            logged_summary = f"Logged: {', '.join(food_list)}"
                            
                            if all_pending_foods:
                                # Ask about pending foods
                                pending_names = [food['name'] for food in all_pending_foods]
                                ask_message = f"{logged_summary}\n\nDo you also want to log these foods: {', '.join(pending_names)}? (Yes/No)"
                                
                                await mem.set_pending(user_id, {
                                    "state": "awaiting_pending_confirm",
                                    "pending_foods": all_pending_foods,
                                    "logged_foods": logged_foods,
                                    "original_input": text
                                })
                                
                                async def _ask_pending_after_log():
                                    yield f"data: {json.dumps({'message': ask_message, 'type': 'confirm_pending'})}\n\n"
                                    yield "event: done\ndata: [DONE]\n\n"
                                
                                return StreamingResponse(_ask_pending_after_log(), media_type="text/event-stream",
                                                        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
                            else:
                                # No pending foods, just log and finish
                                await mem.clear_pending(user_id)
                                items = [convert_food_to_item_format(food) for food in logged_foods]
                                
                                async def _final_log():
                                    yield sse_json({"type":"food_log","status":"logged","is_log":True,"items":items})
                                    yield "event: ping\ndata: {}\n\n"
                                    yield "event: done\ndata: [DONE]\n\n"
                                    await mem.set_pending(user_id, {"state":"awaiting_nav_confirm"})

                                return StreamingResponse(_final_log(), media_type="text/event-stream",
                                                        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
                        else:
                            # Only foods without quantities, ask about all pending
                            if all_pending_foods:
                                pending_names = [food['name'] for food in all_pending_foods]
                                ask_message = f"Do you want to log these foods: {', '.join(pending_names)}? (Yes/No)"
                                
                                await mem.set_pending(user_id, {
                                    "state": "awaiting_pending_confirm",
                                    "pending_foods": all_pending_foods,
                                    "logged_foods": logged_foods,
                                    "original_input": text
                                })
                                
                                async def _ask_all_pending():
                                    yield f"data: {json.dumps({'message': ask_message, 'type': 'confirm_pending'})}\n\n"
                                    yield "event: done\ndata: [DONE]\n\n"
                                
                                return StreamingResponse(_ask_all_pending(), media_type="text/event-stream",
                                                        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
                
                # Process as quantity input
                print(f"DEBUG: Processing '{text}' as quantity for {current_food['name']}")
                quantity_value, parsed_unit = parse_quantity_and_unit(text, current_food['name'])
                
                if quantity_value is not None:
                    print(f"DEBUG: Parsed quantity: {quantity_value} {parsed_unit}")
                    # Use parsed unit or default to food's preferred unit
                    final_unit = parsed_unit if parsed_unit else current_food.get("unit", "pieces")
                    
                    # Update food
                    foods[current_index]["quantity"] = quantity_value
                    foods[current_index]["unit"] = final_unit
                    
                    # Calculate nutrition
                    nutrition = calculate_nutrition_for_quantity(
                        current_food['name'], quantity_value, final_unit
                    )
                    foods[current_index].update(nutrition)
                    
                    # Move completed food to logged
                    logged_foods.append(foods[current_index])
                    
                    # Check for next food needing quantity
                    next_food_index = -1
                    for i in range(current_index + 1, len(foods)):
                        if foods[i].get("quantity") is None:
                            next_food_index = i
                            break
                    
                    if next_food_index != -1:
                        next_food = foods[next_food_index]
                        if next_food['unit'] == 'ml':
                            ask_message = f"How many ml or cups of {next_food['name']} did you have?"
                        else:
                            ask_message = f"How many {next_food['unit']} of {next_food['name']} did you have?"
                        
                        await mem.set_pending(user_id, {
                            "state": "awaiting_quantity",
                            "foods": foods,
                            "current_food_index": next_food_index,
                            "logged_foods": logged_foods,
                            "original_input": pending_state.get("original_input", text)
                        })
                        
                        async def _ask_next():
                            yield f"data: {json.dumps({'message': ask_message, 'type': 'ask_quantity', 'food': next_food['name']})}\n\n"
                            yield "event: done\ndata: [DONE]\n\n"
                        
                        return StreamingResponse(_ask_next(), media_type="text/event-stream",
                                                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
                    else:
                        # All foods processed, log everything
                        print("DEBUG: All foods processed, logging")
                        await mem.clear_pending(user_id)
                        
                        # Convert to items format
                        items = [convert_food_to_item_format(food) for food in logged_foods]
                        
                        async def _logged_then_nav():
                            response_data = create_food_log_response_with_message(logged_foods)
                            
                            # First yield the message separately to ensure it's displayed
                            if 'message' in response_data:
                                yield f"data: {json.dumps({'message': response_data['message'], 'type': 'response'})}\n\n"
                            
                            # Then yield the food log data
                            yield sse_json(response_data)
                            yield "event: ping\ndata: {}\n\n"
                            yield "event: done\ndata: [DONE]\n\n"
                            await mem.set_pending(user_id, {"state":"awaiting_nav_confirm"})

                        return StreamingResponse(_logged_then_nav(), media_type="text/event-stream",
                                                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
                else:
                    # Ask again with better guidance
                    ask_message = f"Please enter a number for {current_food['name']}. For example: '2', '1.5', or '500g'"
                    
                    async def _ask_again():
                        yield f"data: {json.dumps({'message': ask_message, 'type': 'ask_quantity', 'food': current_food['name']})}\n\n"
                        yield "event: done\ndata: [DONE]\n\n"
                    
                    return StreamingResponse(_ask_again(), media_type="text/event-stream",
                                            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
                                            
            except Exception as e:
                print(f"Error processing quantity: {e}")
                await mem.clear_pending(user_id)

        # Normal food processing (no pending state or cleared above)
        print("DEBUG: Processing as normal food input")
        
        if not is_food_related(text):
            print("DEBUG: Text is not food related")
            async def _not_food():
                response = "I'm here to help you log food. What did you eat or drink?"
                yield f"data: {json.dumps({'message': response, 'type': 'response'})}\n\n"
                yield "event: done\ndata: [DONE]\n\n"
            
            return StreamingResponse(_not_food(), media_type="text/event-stream",
                                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

        # Extract food information
        print(f"DEBUG: Extracting food info from: '{text}'")
        food_info = extract_food_info_using_openai(text)
        foods = food_info.get("foods", [])
        print(f"DEBUG: Extracted foods: {foods}")
        
        if not foods:
            async def _no_food():
                response = "I couldn't identify any food. Could you tell me what you ate? For example: 'rice', '2 apples', or 'orange juice'"
                yield f"data: {json.dumps({'message': response, 'type': 'response'})}\n\n"
                yield "event: done\ndata: [DONE]\n\n"
            
            return StreamingResponse(_no_food(), media_type="text/event-stream",
                                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

        # Separate foods with and without quantities
        foods_with_quantity = [f for f in foods if f.get("quantity") is not None]
        foods_without_quantity = [f for f in foods if f.get("quantity") is None]
        print(f"DEBUG: Foods with quantity: {foods_with_quantity}")
        print(f"DEBUG: Foods without quantity: {foods_without_quantity}")

        # Calculate nutrition for foods with quantities
        logged_foods = []
        for food in foods_with_quantity:
            nutrition = calculate_nutrition_for_quantity(
                food['name'], food['quantity'], food['unit']
            )
            food.update(nutrition)
            logged_foods.append(food)

        if foods_without_quantity:
            if logged_foods:
                # Some foods logged, ask about pending ones
                food_list = []
                for food in logged_foods:
                    food_list.append(f"{food['quantity']} {food['unit']} of {food['name']}")
                
                logged_summary = f"Logged: {', '.join(food_list)}"
                pending_names = [food['name'] for food in foods_without_quantity]
                ask_message = f"{logged_summary}\n\nDo you also want to log these foods: {', '.join(pending_names)}? (Yes/No)"
                
                await mem.set_pending(user_id, {
                    "state": "awaiting_pending_confirm",
                    "pending_foods": foods_without_quantity,
                    "logged_foods": logged_foods,
                    "original_input": text
                })
                
                async def _ask_pending():
                    yield f"data: {json.dumps({'message': ask_message, 'type': 'confirm_pending'})}\n\n"
                    yield "event: done\ndata: [DONE]\n\n"
                
                return StreamingResponse(_ask_pending(), media_type="text/event-stream",
                                        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
            else:
                # No foods with quantities, ask for first one's quantity
                first_food = foods_without_quantity[0]
                if first_food['unit'] == 'ml':
                    ask_message = f"How many ml or cups of {first_food['name']} did you have?"
                else:
                    ask_message = f"How many {first_food['unit']} of {first_food['name']} did you have?"
                
                await mem.set_pending(user_id, {
                    "state": "awaiting_quantity",
                    "foods": foods_without_quantity,
                    "current_food_index": 0,
                    "logged_foods": [],
                    "original_input": text
                })
                
                async def _ask_first_quantity():
                    yield f"data: {json.dumps({'message': ask_message, 'type': 'ask_quantity', 'food': first_food['name']})}\n\n"
                    yield "event: done\ndata: [DONE]\n\n"
                
                return StreamingResponse(_ask_first_quantity(), media_type="text/event-stream",
                                        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
        else:
            # All foods have quantities, log immediately
            print("DEBUG: All foods have quantities, logging immediately")
            items = [convert_food_to_item_format(food) for food in logged_foods]
            
            async def _logged_then_nav():
                response_data = create_food_log_response_with_message(logged_foods)
                
                # First yield the message separately to ensure it's displayed
                if 'message' in response_data:
                    yield f"data: {json.dumps({'message': response_data['message'], 'type': 'response'})}\n\n"
                
                # Then yield the food log data
                yield sse_json(response_data)
                yield "event: ping\ndata: {}\n\n"
                yield "event: done\ndata: [DONE]\n\n"
                await mem.set_pending(user_id, {"state":"awaiting_nav_confirm"})
            return StreamingResponse(_logged_then_nav(), media_type="text/event-stream",
                                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        print(f"Full traceback: {traceback.format_exc()}")
        
        try:
            await mem.clear_pending(user_id)
        except:
            pass
        
        async def _error():
            yield f"data: {json.dumps({'message': 'Sorry, I encountered an error. Please try again.', 'type': 'error'})}\n\n"
            yield "event: done\ndata: [DONE]\n\n"
        
        return StreamingResponse(_error(), media_type="text/event-stream",
                                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


def is_quantity_input(text):
    """Check if the input is a quantity (number with optional unit)"""
    if not text:
        return False
    
    text_lower = text.strip().lower()
    
    # Check for pure numbers or numbers with units
    quantity_patterns = [
        r'^\d+(?:\.\d+)?,  # Just numbers: 2, 1.5, 100'
        r'^\d+(?:\.\d+)?\s*(g|grams?|kg|plates?|bowls?|pieces?|ml|glasses?|cups?|slices?)'  # Numbers with units
    ]
    
    return any(re.match(pattern, text_lower) for pattern in quantity_patterns)
    
def parse_quantity_and_unit(text, food_name):
    """Enhanced quantity and unit parsing"""
    text_lower = text.lower().strip()
    
    # Pattern matching for different input formats
    patterns = [
        r'(\d+(?:\.\d+)?)\s*(g|grams?|kg|kilograms?)',  # Weight units
        r'(\d+(?:\.\d+)?)\s*(plates?|bowls?)',          # Serving units
        r'(\d+(?:\.\d+)?)\s*(pieces?|pcs?|pc)',         # Count units
        r'(\d+(?:\.\d+)?)\s*(ml|milliliters?|glasses?|cups?)', # Volume units
        r'(\d+(?:\.\d+)?)\s*(slices?|slc)',             # Slice units
        r'(\d+(?:\.\d+)?)',                             # Just number
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text_lower)
        if match:
            quantity = float(match.group(1))
            unit = match.group(2) if len(match.groups()) > 1 else None
            
            # Handle unit conversion
            if unit:
                if unit in ['kg', 'kilogram', 'kilograms']:
                    quantity = quantity * 1000
                    unit = 'grams'
                else:
                    unit = normalize_unit(unit)
            
            print(f"DEBUG: Parsed '{text}' as {quantity} {unit}")
            return quantity, unit
        
    print(f"DEBUG: Could not parse quantity from '{text}'")
    return None, None

class Userid(BaseModel):
    user_id: int




@router.get("/debug_pending")
async def debug_pending(
    user_id: int,
    mem = Depends(get_mem),
):
    pending_state = await mem.get_pending(user_id)
    return {"user_id": user_id, "pending_state": pending_state}

@router.post("/delete_chat")
async def chat_close(
    req: Userid,
    mem = Depends(get_mem),
):
    print(f"Deleting chat history for user {req.user_id}")
    history_key = f"chat:{req.user_id}:history"
    pending_key = f"chat:{req.user_id}:pending"
    deleted = await mem.r.delete(history_key, pending_key)
    return {"status": 200}
async def clear_pending_state(
    req: Userid,
    mem = Depends(get_mem),
):
    await mem.clear_pending(req.user_id)
    return {"status": "cleared", "user_id": req.user_id}
