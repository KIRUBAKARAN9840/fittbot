#food_log
from fastapi import APIRouter, HTTPException, Query, Depends, UploadFile, File
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
from app.models.fittbot_models import ActualDiet
from app.fittbot_api.v1.client.client_api.chatbot.chatbot_services.asr import transcribe_audio

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

@router.post("/voice/transcribe")
async def voice_transcribe(
    audio: UploadFile = File(...),
    http = Depends(get_http),
    oai = Depends(get_oai),
):
    """Transcribe audio to text and translate to English"""
    transcript = await transcribe_audio(audio, http=http)
    if not transcript:
        raise HTTPException(400, "empty transcript")

    def _translate_to_english(text: str) -> dict:
        try:
            sys = (
                "You are a translator. Output ONLY JSON like "
                "{\"lang\":\"xx\",\"english\":\"...\"}. "
                "Detect source language code (ISO-639-1 if possible). "
                "Translate to natural English. Do not add extra words. "
                "Keep food names recognizable; use common transliterations if needed."
            )
            resp = oai.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[{"role":"system","content":sys},{"role":"user","content":text}],
                response_format={"type":"json_object"},
                temperature=0
            )
            return orjson.loads(resp.choices[0].message.content)
        except Exception as e:
            print(f"Translation error: {e}")
            return {"lang": "unknown", "english": text}

    result = _translate_to_english(transcript)
    return {
        "transcript": transcript,
        "detected_language": result.get("lang", "unknown"),
        "english_text": result.get("english", transcript)
    }

@router.post("/voice/stream")
async def voice_stream_sse(
    user_id: int,
    audio: UploadFile = File(...),
    mem = Depends(get_mem),
    oai = Depends(get_oai),
    http = Depends(get_http),
    db: Session = Depends(get_db),
):
    """Transcribe audio and process it through the food log stream"""
    transcript = await transcribe_audio(audio, http=http)
    if not transcript:
        raise HTTPException(400, "empty transcript")

    # Use the existing food log stream function with the transcript
    return await chat_stream(
        user_id=user_id,
        text=transcript,
        mem=mem,
        oai=oai,
        db=db
    )

def get_smart_unit_and_question(food_name):
    """
    Universal function to determine appropriate unit and generate 
    context-aware questions for any food item
    """
    food_lower = food_name.lower().strip()
    
    # Food category mapping with multiple unit options
    food_categories = {
        'grains_starches': {
            'foods': ['dosa', 'rice', 'roti', 'chapati', 'naan', 'bread', 'pasta', 'noodles', 'quinoa', 'wheat', 'idli', 'uttapam'],
            'primary_unit': 'pieces',
            'alternatives': ['grams', 'plates', 'bowls'],
            'question_template': "How many {food} did you have? (pieces, plates, or grams)"
        },
        # ADD THIS NEW CATEGORY for rice dishes
        'rice_dishes': {
            'foods': ['biryani', 'biriyani', 'briyani', 'pulao', 'pilaf', 'fried rice', 'jeera rice', 'lemon rice', 'pongal'],
            'primary_unit': 'plates',  # Changed from pieces to plates
            'alternatives': ['bowls', 'grams', 'pieces'],
            'question_template': "How many plates of {food} did you have? (plates, bowls, or grams)"
        },
        'liquids': {
            'foods': ['juice', 'milk', 'water', 'tea', 'coffee', 'drink', 'smoothie', 'shake', 'lassi'],
            'primary_unit': 'ml',
            'alternatives': ['cups', 'glasses'],
            'question_template': "How much {food} did you have? (ml, cups, or glasses)"
        },
        'fruits': {
            'foods': ['apple', 'banana', 'orange', 'mango', 'grapes', 'berries', 'fruit'],
            'primary_unit': 'pieces',
            'alternatives': ['grams', 'cups'],
            'question_template': "How many {food} did you have? (pieces or grams)"
        },
        'vegetables': {
            'foods': ['potato', 'tomato', 'onion', 'carrot', 'vegetable', 'salad', 'greens'],
            'primary_unit': 'pieces',
            'alternatives': ['grams', 'cups', 'plates'],
            'question_template': "How many {food} did you have? (pieces, grams, or cups)"
        },
        'proteins': {
            'foods': ['chicken', 'fish', 'meat', 'egg', 'beef', 'pork', 'mutton', 'turkey', 'prawns', 'paneer', 'tofu'],
            'primary_unit': 'grams',
            'alternatives': ['pieces'],
            'question_template': "How much {food} did you have? (grams or pieces)"
        },
        'curries_gravies': {
            'foods': ['curry', 'dal', 'sambar', 'rasam', 'gravy', 'sabji'],
            'primary_unit': 'bowls',
            'alternatives': ['plates', 'cups', 'grams'],
            'question_template': "How much {food} did you have? (bowls, plates, or grams)"
        },
        'snacks': {
            'foods': ['chips', 'biscuit', 'cookie', 'cake', 'chocolate', 'candy', 'nuts', 'samosa', 'vadai'],
            'primary_unit': 'pieces',
            'alternatives': ['grams', 'packets'],
            'question_template': "How many {food} did you have? (pieces or grams)"
        },
        'dairy': {
            'foods': ['yogurt', 'curd', 'cheese', 'butter', 'ghee', 'cream'],
            'primary_unit': 'grams',
            'alternatives': ['cups', 'tablespoons'],
            'question_template': "How much {food} did you have? (grams, cups, or tablespoons)"
        }
    }
    
    # Find matching category (check rice_dishes first for biryani)
    for category, data in food_categories.items():
        if any(food_item in food_lower for food_item in data['foods']):
            return {
                'primary_unit': data['primary_unit'],
                'alternatives': data['alternatives'],
                'question': data['question_template'].format(food=food_name),
                'category': category
            }
    
    # Default fallback for unknown foods
    return {
        'primary_unit': 'pieces',
        'alternatives': ['grams'],
        'question': f"How much {food_name} did you have? (pieces or grams)",
        'category': 'unknown'
    }

def normalize_unit_with_context(unit, food_name):
    """Simplified unit normalization - trust AI decisions more"""
    if not unit:
        return 'pieces'  # Simple fallback
    
    unit_lower = unit.lower().strip()
    
    # Standard unit mappings - expanded for spoons
    unit_map = {
        'g': 'grams', 'gram': 'grams', 'gms': 'grams',
        'kg': 'kg', 'kilogram': 'kg', 'kilograms': 'kg',
        'piece': 'pieces', 'pcs': 'pieces', 'pc': 'pieces',
        'slice': 'slices', 'slc': 'slices',
        'plate': 'plates', 'bowl': 'bowls',
        'cup': 'cups', 'glass': 'glasses',
        'ml': 'ml', 'milliliter': 'ml', 'milliliters': 'ml',
        'liter': 'liters', 'litre': 'liters', 'l': 'liters',
        'can': 'cans', 'tin': 'cans', 'packet': 'packets',
        'tablespoon': 'tablespoons', 'tbsp': 'tablespoons',
        'teaspoon': 'teaspoons', 'tsp': 'teaspoons',
        'spoon': 'tablespoons', 'spoons': 'tablespoons'  # Added spoon mapping
    }
    
    normalized = unit_map.get(unit_lower, unit_lower)
    
    # Accept any reasonable unit without overriding
    common_food_units = [
        'pieces', 'plates', 'bowls', 'cups', 'glasses', 'grams', 'ml', 'slices', 
        'tablespoons', 'teaspoons', 'kg', 'liters', 'cans', 'packets'
    ]
    
    if normalized in common_food_units:
        print(f"DEBUG: Using unit '{normalized}' for {food_name}")
        return normalized
    else:
        print(f"DEBUG: Unknown unit '{unit}', defaulting to pieces")
        return 'pieces'


def normalize_unit(unit):
    """Legacy function - kept for backward compatibility"""
    if not unit:
        return 'pieces'
    return normalize_unit_with_context(unit, 'unknown')

def get_unit_hint(unit):
    """Generate unit hint for the specified unit"""
    unit_hints = {
        'pieces': 'How many pieces?',
        'plates': 'How many plates or grams?',
        'bowls': 'How many bowls or grams?',
        'cups': 'How many cups or ml?',
        'glasses': 'How many glasses or ml?',
        'slices': 'How many slices?',
        'ml': 'How much ml or cups?',
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

def get_enhanced_ai_prompt(text):
    """Generate comprehensive AI prompt that handles all edge cases"""
    return f"""
    Analyze this text and extract food information: "{text}"

    CRITICAL FOOD IDENTIFICATION RULES:
    1. COMPOUND FOODS: Treat compound words as SINGLE dishes:
       - "curdrice" = "curd rice" (one dish, not separate curd and rice)
          note: if curd,rice are separate with space/comma, treat as two dishes
       - "lemonrice" = "lemon rice" (one dish)
       - "masalatea" = "masala tea" (one dish)
    
    2. FOOD DETECTION: Extract ALL foods/drinks, handle misspellings liberally
    3. CONTEXT AWARENESS: Consider Indian cuisine context for units and dishes

    INTELLIGENT UNIT ASSIGNMENT:
    When quantity is provided, choose the MOST LOGICAL unit based on:
    
    INDIAN RICE DISHES (use plates/bowls):
    - Any rice dish: biryani, pulao, fried rice, lemon rice, curd rice → plates
    - Curries, dal, sambar → bowls
    
    MEASUREMENT CONTEXT:
    - "spoon", "spoons" → tablespoons (NOT grams)
    - Small countable items → pieces  
    - Liquids → ml, cups, glasses
    - Large servings → plates, bowls
    - Precise measurements → grams, kg
    
    QUANTITY INTERPRETATION:
    - If user provides quantity, extract it exactly
    - If no quantity, set to null
    - Choose unit that matches natural serving size

    EXAMPLES OF CORRECT INTERPRETATION:
    - "curdrice" → {{"name": "curd rice", "quantity": null, "unit": "plates"}}
    - "3 spoon curd" → {{"name": "curd", "quantity": 3, "unit": "tablespoons"}}
    - "lemonrice" → {{"name": "lemon rice", "quantity": null, "unit": "plates"}}
    - "2 dosa" → {{"name": "dosa", "quantity": 2, "unit": "pieces"}}
    - "chicken curry" → {{"name": "chicken curry", "quantity": null, "unit": "bowls"}}

    NUTRITION CALCULATION (when quantity provided):
    - Use REALISTIC conversions:
      * 1 plate = 300g for rice dishes
      * 1 tablespoon = 15g for dense foods, 15ml for liquids
      * 1 piece = varies by food (dosa=80g, chicken piece=100g)
    - Calculate accurate nutrition for exact quantity
    - If quantity is null, omit nutrition fields

    Return ONLY valid JSON array:
    [
        {{
            "name": "properly_formatted_food_name",
            "quantity": number_or_null,
            "unit": "contextually_appropriate_unit",
            "calories": number_or_null,
            "protein": number_or_null,
            "carbs": number_or_null,
            "fat": number_or_null,
            "fiber": number_or_null,
            "sugar": number_or_null
        }}
    ]
    
    CRITICAL: Use cultural context for Indian foods. Don't split compound food words.
    """

def extract_food_info_using_ai(text: str):
    """AI-driven food extraction with comprehensive prompt"""
    
    reasoning_prompt = get_enhanced_ai_prompt(text)

    # Try OpenAI first
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system", 
                    "content": """You are a nutrition expert specializing in Indian cuisine. 
                    CRITICAL RULES:
                    1. Compound food words are SINGLE dishes (curdrice = curd rice as one dish)
                    2. Use culturally appropriate units (plates for rice dishes, tablespoons for spoons)
                    3. Never split single dishes into multiple foods
                    4. Respect user's measurement context (3 spoon = 3 tablespoons, not grams)"""
                },
                {"role": "user", "content": reasoning_prompt}
            ],
            max_tokens=1000,
            temperature=0.1
        )

        result = response.choices[0].message.content.strip()
        result = re.sub(r"^```json\s*", "", result)
        result = re.sub(r"\s*```$", "", result)
        
        foods = json.loads(result)
        if not isinstance(foods, list):
            foods = [foods] if isinstance(foods, dict) else []
        
        print(f"DEBUG: AI extracted foods: {[(f.get('name'), f.get('quantity'), f.get('unit')) for f in foods]}")
        return {"foods": foods}

    except Exception as e:
        print(f"OpenAI extraction error: {e}, trying Gemini fallback...")
        
        # Try Gemini fallback
        try:
            response = gemini_model.generate_content(reasoning_prompt)
            
            if response.text:
                result = response.text.strip()
                result = re.sub(r"^```json\s*", "", result)
                result = re.sub(r"\s*```$", "", result)
                
                foods = json.loads(result)
                if not isinstance(foods, list):
                    foods = [foods] if isinstance(foods, dict) else []
                
                print(f"DEBUG: Gemini extracted foods: {[(f.get('name'), f.get('quantity'), f.get('unit')) for f in foods]}")
                return {"foods": foods}
                
        except Exception as gemini_error:
            print(f"Gemini extraction error: {gemini_error}")
        
        # Simple fallback parsing
        print("Both AI models failed, using fallback parsing...")
        return parse_food_with_smart_units(text)


def parse_food_with_smart_units(text):
    """Enhanced fallback parsing with smart unit assignment"""
    text_lower = text.lower().strip()
    
    # Handle common misspellings
    corrections = {
        'avacado': 'avocado', 'avacadojuice': 'avocado juice',
        'sugarcanjuice': 'sugar cane juice', 'orangejuice': 'orange juice'
    }
    
    for wrong, correct in corrections.items():
        text_lower = text_lower.replace(wrong, correct)
    
    # Split multiple foods by comma
    food_items = [item.strip() for item in text_lower.split(',')]
    foods = []
    
    for item in food_items:
        # Extract quantity and unit
        quantity_match = re.search(r'(\d+(?:\.\d+)?)\s*(\w+)?', item)
        
        if quantity_match:
            quantity = float(quantity_match.group(1))
            unit_part = quantity_match.group(2)
            food_name = re.sub(r'\d+(?:\.\d+)?\s*\w*', '', item).strip()
        else:
            quantity = None
            unit_part = None
            food_name = item
        
        if food_name:
            # Use smart unit assignment
            food_info = get_smart_unit_and_question(food_name)
            
            if unit_part:
                unit = normalize_unit_with_context(unit_part, food_name)
            else:
                unit = food_info['primary_unit']  # Use contextually appropriate primary unit
            
            foods.append({
                "name": food_name,
                "quantity": quantity,
                "unit": unit,
                "calories": None,
                "protein": None,
                "carbs": None,
                "fat": None,
                "fiber": None,
                "sugar": None
            })
    
    return {"foods": foods}

def calculate_nutrition_using_ai(food_name, quantity, unit):
    """Enhanced nutrition calculation with better unit handling"""
    try:
        prompt = f"""
        Calculate nutrition for: {quantity} {unit} of {food_name}
        
        Use these REALISTIC conversions:
        - 1 plate (rice dishes) = 300 grams
        - 1 tablespoon = 15 grams (solids) or 15 ml (liquids)
        - 1 teaspoon = 5 grams (solids) or 5 ml (liquids)
        - 1 cup = 200 grams (solids) or 200 ml (liquids)
        - 1 bowl = 200 grams
        - 1 glass = 200 ml
        - 1 piece varies by food type (estimate appropriately)
        
        For spoon measurements, consider the food density:
        - Curd/yogurt: 1 tablespoon ≈ 15g
        - Ghee/oil: 1 tablespoon ≈ 14g
        - Rice: 1 tablespoon ≈ 12g
        
        Examples:
        - "3 tablespoons of curd" = 45g of curd
        - "1 plate of lemon rice" = 300g of lemon rice
        - "2 pieces of dosa" = 160g total (80g each)
        
        Return ONLY valid JSON with realistic values:
        {{
            "calories": number,
            "protein": number,
            "carbs": number,
            "fat": number,
            "fiber": number,
            "sugar": number
        }}
        """
        
        print(f"DEBUG: Calculating nutrition for {quantity} {unit} of {food_name}")
        
        # Try OpenAI first
        try:
            response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system", 
                        "content": "You are a nutrition expert. Always provide realistic nutrition values based on the specified quantity and unit. Use the conversions provided."
                    },
                    {"role": "user", "content": prompt}
                ],
                max_tokens=300,
                temperature=0
            )
            
            result = response.choices[0].message.content.strip()
            print(f"DEBUG: OpenAI nutrition response: {result}")
            
            result = re.sub(r"^```json\s*", "", result)
            result = re.sub(r"\s*```$", "", result)
            
            nutrition = json.loads(result)
            print(f"DEBUG: Parsed nutrition: {nutrition}")
            
            return nutrition
            
        except Exception as openai_error:
            print(f"OpenAI nutrition error: {openai_error}, trying Gemini...")
            
            # Gemini fallback
            response = gemini_model.generate_content(prompt)
            if response.text:
                result = response.text.strip()
                result = re.sub(r"^```json\s*", "", result)
                result = re.sub(r"\s*```$", "", result)
                
                nutrition = json.loads(result)
                return nutrition
    
    except Exception as e:
        print(f"AI nutrition calculation failed: {e}, using fallback values")
        return get_fallback_nutrition(food_name, quantity, unit)

def get_fallback_nutrition(food_name, quantity, unit):
    """Enhanced fallback with better unit conversion"""
    food_lower = food_name.lower()
    
    # Convert to approximate grams for calculation
    if unit == 'plates':
        grams = quantity * 300
    elif unit == 'bowls':
        grams = quantity * 200
    elif unit == 'cups':
        grams = quantity * 200
    elif unit == 'tablespoons':
        if 'oil' in food_lower or 'ghee' in food_lower:
            grams = quantity * 14  # Fat is denser
        elif 'curd' in food_lower or 'yogurt' in food_lower:
            grams = quantity * 15
        else:
            grams = quantity * 12  # Rice, etc.
    elif unit == 'teaspoons':
        grams = quantity * 5
    elif unit == 'pieces':
        if 'dosa' in food_lower:
            grams = quantity * 80
        elif 'chicken' in food_lower:
            grams = quantity * 100
        else:
            grams = quantity * 50
    elif unit == 'grams':
        grams = quantity
    elif unit == 'ml':
        grams = quantity  # For liquids
    else:
        grams = quantity * 50  # Default
    
    # Basic nutrition per 100g with better estimates
    if 'curd' in food_lower and 'rice' in food_lower:
        per_100g = {"calories": 110, "protein": 3, "carbs": 20, "fat": 2, "fiber": 1, "sugar": 3}
    elif 'lemon rice' in food_lower:
        per_100g = {"calories": 150, "protein": 3, "carbs": 28, "fat": 4, "fiber": 1, "sugar": 1}
    elif 'curd' in food_lower:
        per_100g = {"calories": 60, "protein": 3.5, "carbs": 4.5, "fat": 3.5, "fiber": 0, "sugar": 4.5}
    elif 'rice' in food_lower:
        per_100g = {"calories": 130, "protein": 2.7, "carbs": 28, "fat": 0.3, "fiber": 0.4, "sugar": 0.1}
    else:
        per_100g = {"calories": 100, "protein": 3, "carbs": 15, "fat": 2, "fiber": 1, "sugar": 2}
    
    # Calculate for actual portion
    ratio = grams / 100.0
    nutrition = {}
    for key, value in per_100g.items():
        nutrition[key] = round(value * ratio, 1)
    
    print(f"DEBUG: Fallback nutrition for {grams}g of {food_name}: {nutrition}")
    return nutrition

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

def handle_quantity_question(food_name, unit=None):
    """Generate AI-driven quantity question"""
    try:
        prompt = f"""
        Generate a natural quantity question for "{food_name}".
        
        Consider the most common way this food is measured:
        - Rice dishes (biryani, fried rice, lemon rice): plates, bowls
        - Small items (dosa, roti): pieces
        - Liquids: glasses, cups, ml
        - Condiments, curd: spoons, tablespoons
        - Vegetables: pieces, grams
        
        Return a friendly question like:
        "How many plates of biryani did you have?"
        "How many pieces of dosa?"
        "How many tablespoons of curd?"
        
        Return ONLY the question text, nothing else.
        """
        
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Generate natural food quantity questions based on cultural context."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=50,
            temperature=0.3
        )
        
        question = response.choices[0].message.content.strip().strip('"')
        print(f"DEBUG: AI generated question: {question}")
        return question
        
    except Exception as e:
        print(f"AI question generation failed: {e}")
        return f"How much {food_name} did you have?"


@router.get("/chat/stream_test")
async def chat_stream(
    user_id: int,
    text: str = Query(None),
    meal: str = Query(None),
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
        
        # Check for pending state
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
                # Continue to food processing below
            
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
                    # Use context-aware question
                    ask_message = handle_quantity_question(first_pending['name'])
                                        
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
                    async def _final_logged():
                        # Store to database
                        today_date = datetime.now(IST).strftime("%Y-%m-%d")
                        if meal:
                            store_diet_data_to_db(db, user_id, today_date, logged_foods, meal)
                        
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
                    new_food_info = extract_food_info_using_ai(text)
                    new_foods = new_food_info.get("foods", [])
                    
                    if new_foods:
                        # Check if new foods have quantities
                        foods_with_quantity = [f for f in new_foods if f.get("quantity") is not None]
                        foods_without_quantity = [f for f in new_foods if f.get("quantity") is None]
                        
                        # Add foods with quantities to logged
                        logged_foods.extend(foods_with_quantity)
                        
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
                                
                                async def _final_log():
                                    # Store to database
                                    today_date = datetime.now(IST).strftime("%Y-%m-%d")
                                    if meal:
                                        store_diet_data_to_db(db, user_id, today_date, logged_foods, meal)
                                    
                                    response_data = create_food_log_response_with_message(logged_foods)
                                    
                                    # Send only the complete food log data (includes the message)
                                    yield sse_json(response_data)
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
                quantity_value, parsed_unit = parse_quantity_and_unit(text, current_food['name'], current_food.get('unit'))
                
                if quantity_value is not None:
                    print(f"DEBUG: Parsed quantity: {quantity_value} {parsed_unit}")
                    # Use parsed unit or default to food's preferred unit
                    final_unit = parsed_unit
                    
                    # Update food
                    foods[current_index]["quantity"] = quantity_value
                    foods[current_index]["unit"] = final_unit
                    
                    # Calculate nutrition using AI
                    nutrition = calculate_nutrition_using_ai(
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
                        # Use context-aware question
                        ask_message = handle_quantity_question(next_food['name'])
                        
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
                        
                        async def _logged_then_nav():
                            # Store to database
                            today_date = datetime.now(IST).strftime("%Y-%m-%d")
                            if meal:
                                store_diet_data_to_db(db, user_id, today_date, logged_foods, meal)
                            
                            response_data = create_food_log_response_with_message(logged_foods)
                            
                            # Send only the complete food log data (includes the message)
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

        # Extract food information using AI
        print(f"DEBUG: Extracting food info from: '{text}'")
        food_info = extract_food_info_using_ai(text)
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

        # Foods with quantities are already processed by AI (have nutrition)
        logged_foods = foods_with_quantity

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
                # Use context-aware question
                ask_message = handle_quantity_question(first_food['name'])
                
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
            
            async def _logged_then_nav():
                # Store to database
                today_date = datetime.now(IST).strftime("%Y-%m-%d")
                if meal:
                    store_diet_data_to_db(db, user_id, today_date, logged_foods, meal)
                
                response_data = create_food_log_response_with_message(logged_foods)
                
                # Send only the complete food log data (includes the message)
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
        r'^\d+(?:\.\d+)?',  # Just numbers: 2, 1.5, 100
        r'^\d+(?:\.\d+)?\s*(g|grams?|kg|plates?|bowls?|pieces?|ml|glasses?|cups?|slices?)'
                          # Numbers with units
    ]
    
    return any(re.match(pattern, text_lower) for pattern in quantity_patterns)
    
def parse_quantity_and_unit(text, food_name, existing_unit=None):
    """Smart parsing that handles large numbers and auto-converts to appropriate units"""
    text_lower = text.lower().strip()

    # Pattern matching - check if user specified a unit explicitly
    patterns_with_units = [
        r'(\d+(?:\.\d+)?)\s*(spoons?|tablespoons?|tbsp|teaspoons?|tsp)',
        r'(\d+(?:\.\d+)?)\s*(g|grams?|kg|kilograms?)',
        r'(\d+(?:\.\d+)?)\s*(plates?|bowls?)',
        r'(\d+(?:\.\d+)?)\s*(pieces?|pcs?|pc)',
        r'(\d+(?:\.\d+)?)\s*(ml|milliliters?|glasses?|cups?)',
        r'(\d+(?:\.\d+)?)\s*(slices?|slc)',
    ]

    # Check if user explicitly provided a unit
    user_provided_unit = False
    for pattern in patterns_with_units:
        match = re.search(pattern, text_lower)
        if match:
            quantity = float(match.group(1))
            unit = match.group(2)
            user_provided_unit = True

            # Handle unit conversion
            if unit in ['kg', 'kilogram', 'kilograms']:
                quantity = quantity * 1000
                unit = 'grams'
            else:
                unit = normalize_unit_with_context(unit, food_name)

            print(f"DEBUG: User provided unit - parsed '{text}' as {quantity} {unit} for {food_name}")
            return quantity, unit

    # Check for just numbers (no unit provided by user)
    number_match = re.search(r'(\d+(?:\.\d+)?)', text_lower)
    if number_match:
        quantity = float(number_match.group(1))

        if not user_provided_unit:
            # User didn't provide a unit - use the food's existing unit
            final_unit = existing_unit if existing_unit else 'pieces'
            print(f"DEBUG: No unit provided - using existing unit: {quantity} {final_unit} for {food_name}")
            return quantity, final_unit

    print(f"DEBUG: Could not parse quantity from '{text}'")
    return None, None

def is_liquid_food(food_name):
    """Determine if a food is primarily liquid"""
    food_lower = food_name.lower()
    liquid_keywords = [
        'juice', 'milk', 'water', 'tea', 'coffee', 'drink', 'smoothie', 
        'shake', 'lassi', 'soup', 'broth', 'wine', 'beer', 'soda'
    ]
    return any(keyword in food_lower for keyword in liquid_keywords)

def is_liquid_food(food_name):
    """Determine if a food is primarily liquid"""
    food_lower = food_name.lower()
    liquid_keywords = [
        'juice', 'milk', 'water', 'tea', 'coffee', 'drink', 'smoothie', 
        'shake', 'lassi', 'soup', 'broth', 'wine', 'beer', 'soda'
    ]
    return any(keyword in food_lower for keyword in liquid_keywords)

# ADD THESE FUNCTIONS HERE (starting around line 634)
def store_diet_data_to_db(db: Session, client_id: int, date: str, logged_foods: list, meal: str):
    """Store logged food data in actual_diet table"""
    try:
        # Check if entry exists for this client and date
        existing_entry = db.query(ActualDiet).filter(
            ActualDiet.client_id == client_id,
            ActualDiet.date == date
        ).first()
        
        # Create the food item structure for the meal
        food_items = []
        for food in logged_foods:
            food_item = {
                "id": str(int(datetime.now().timestamp() * 1000)),  # Generate unique ID
                "name": food.get('name', ''),
                "quantity": f"{food.get('quantity', 0)} {food.get('unit', 'serving')}",
                "calories": food.get('calories', 0),
                "protein": food.get('protein', 0),
                "carbs": food.get('carbs', 0),
                "fat": food.get('fat', 0),
                "fiber": food.get('fiber', 0),
                "sugar": food.get('sugar', 0),
                "image_url": ""  # Empty as requested
            }
            food_items.append(food_item)
        
        if existing_entry:
            # Parse existing diet_data
            diet_data = existing_entry.diet_data if existing_entry.diet_data else []
            
            # Find the meal category and update it
            meal_found = False
            for meal_category in diet_data:
                print(meal_category.get("title", "").lower(), meal.lower())
                if meal_category.get("title", "").lower() == meal.lower():
                    # Append to existing foodList
                    meal_category["foodList"].extend(food_items)
                    meal_category["itemsCount"] = len(meal_category["foodList"])
                    meal_found = True
                    break
            
            if not meal_found:
                # If meal category doesn't exist, this shouldn't happen with your predefined structure
                # But we can handle it by finding the right meal from the template
                default_structure = get_default_diet_structure()
                for default_meal in default_structure:
                    if default_meal.get("title", "").lower() == meal.lower():
                        default_meal["foodList"] = food_items
                        default_meal["itemsCount"] = len(food_items)
                        diet_data.append(default_meal)
                        break
            from sqlalchemy.orm import attributes
            attributes.flag_modified(existing_entry, "diet_data")
            # Update existing record
            existing_entry.diet_data = diet_data
            db.commit()
            
        else:
            # Create new entry with full structure
            diet_data = get_default_diet_structure()
            
            # Update the specific meal
            for meal_category in diet_data:
                print(str(meal_category.get("title", "")).lower(), meal.lower())
                if meal_category.get("title", "").lower() == meal.lower():
                    meal_category["foodList"] = food_items
                    meal_category["itemsCount"] = len(food_items)
                    break
            
            # Create new record
            new_entry = ActualDiet(
                client_id=client_id,
                date=date,
                diet_data=diet_data
            )
            db.add(new_entry)
            db.commit()
            
        print(f"Successfully stored diet data for client {client_id}, date {date}, meal {meal}")
        return True
        
    except Exception as e:
        print(f"Error storing diet data: {e}")
        db.rollback()
        return False

def get_default_diet_structure():
    """Return the default diet structure as shown in your example"""
    return [
        {
            "id": "1",
            "title": "Pre workout",
            "tagline": "Energy boost",
            "foodList": [],
            "timeRange": "6:30-7:00 AM",
            "itemsCount": 0
        },
        {
            "id": "2",
            "title": "Post workout",
            "tagline": "Recovery fuel",
            "foodList": [],
            "timeRange": "7:30-8:00 AM",
            "itemsCount": 0
        },
        {
            "id": "3",
            "title": "Early morning Detox",
            "tagline": "Early morning nutrition",
            "foodList": [],
            "timeRange": "5:30-6:00 AM",
            "itemsCount": 0
        },
        {
            "id": "4",
            "title": "Pre-Breakfast / Pre-Meal Starter",
            "tagline": "Pre-breakfast fuel",
            "foodList": [],
            "timeRange": "7:00-7:30 AM",
            "itemsCount": 0
        },
        {
            "id": "5",
            "title": "Breakfast",
            "tagline": "Start your day right",
            "foodList": [],
            "timeRange": "8:30-9:30 AM",
            "itemsCount": 0
        },
        {
            "id": "6",
            "title": "Mid-Morning snack",
            "tagline": "Healthy meal",
            "foodList": [],
            "timeRange": "10:00-11:00 AM",
            "itemsCount": 0
        },
        {
            "id": "7",
            "title": "Lunch",
            "tagline": "Nutritious midday meal",
            "foodList": [],
            "timeRange": "1:00-2:00 PM",
            "itemsCount": 0
        },
        {
            "id": "8",
            "title": "Evening snack",
            "tagline": "Healthy meal",
            "foodList": [],
            "timeRange": "4:00-5:00 PM",
            "itemsCount": 0
        },
        {
            "id": "9",
            "title": "Dinner",
            "tagline": "End your day well",
            "foodList": [],
            "timeRange": "7:30-8:30 PM",
            "itemsCount": 0
        },
        {
            "id": "10",
            "title": "Bed time",
            "tagline": "Rest well",
            "foodList": [],
            "timeRange": "9:30-10:00 PM",
            "itemsCount": 0
        }
    ]


def get_food_max_reasonable_serving(food_name, unit):
    """Get maximum reasonable serving size for different foods"""
    food_lower = food_name.lower()
    
    # Define reasonable maximums by food type and unit
    max_servings = {
        # Rice dishes
        'rice': {'plates': 3, 'bowls': 4, 'grams': 400},
        'biryani': {'plates': 2, 'bowls': 3, 'grams': 600},
        'pongal': {'plates': 2, 'bowls': 3, 'grams': 400},
        
        # Liquids
        'juice': {'ml': 500, 'glasses': 3, 'cups': 3},
        'milk': {'ml': 500, 'glasses': 3, 'cups': 3},
        'water': {'ml': 1000, 'glasses': 5, 'cups': 5},
        
        # Default maximums by unit
        'default': {'plates': 3, 'bowls': 4, 'pieces': 10, 'grams': 500, 'ml': 1000}
    }
    
    # Check specific food first
    for food_key in max_servings:
        if food_key in food_lower and food_key != 'default':
            return max_servings[food_key].get(unit, max_servings['default'].get(unit, 10))
    
    # Use default
    return max_servings['default'].get(unit, 10)



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

@router.post("/clear_pending")
async def clear_pending_state(
    req: Userid,
    mem = Depends(get_mem),
):
    await mem.clear_pending(req.user_id)
    return {"status": "cleared", "user_id": req.user_id}
