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
import httpx
from fastapi import APIRouter, HTTPException, UploadFile, File, Query, Depends

from app.models.fittbot_models import ClientDietTemplate
from app.fittbot_api.v1.client.client_api.chatbot.chatbot_services.asr import transcribe_audio
import orjson 

from app.fittbot_api.v1.client.client_api.chatbot.chatbot_services.llm_helpers import (
   PlainTextStreamFilter, oai_chat_stream, GENERAL_SYSTEM, TOP_K,
   build_messages, heuristic_confidence, gpt_extract_items, first_missing_quantity,OPENAI_MODEL,
   sse_json, sse_escape, gpt_small_route, _scale_macros, is_yes, is_no, is_fit_chat,
   has_action_verb, food_hits,ensure_per_unit_macros, is_fittbot_meta_query,normalize_food, 
   explicit_log_command, STYLE_PLAN, is_plan_request,STYLE_CHAT_FORMAT,pretty_plan
)

def sse_data(content: str) -> str:
    """
    Properly format content for SSE transmission with UTF-8 Unicode support.
    SSE requires 'data: ' prefix and double newline. Content is sent as plain UTF-8.
    """
    if isinstance(content, bytes):
        content = content.decode('utf-8', errors='replace')

    lines = content.split('\n')
    if len(lines) == 1:
        return f"data: {content}\n\n"
    else:
        return ''.join(f"data: {line}\n" for line in lines) + "\n"


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

@router.post("/voice/transcribe")
async def voice_transcribe(
    audio: UploadFile = File(...),
    http = Depends(get_http),
    oai = Depends(get_oai),
):
    """Transcribe audio to text and translate to English"""
    try:
        # Use the transcribe function that's already imported from asr.py
        
        transcript = await transcribe_audio(audio, http=http)
        if not transcript:
            raise HTTPException(400, "empty transcript")

        def _translate_to_english(text: str) -> dict:
            try:
                sys = (
                    "You are a translator. Output ONLY JSON like "
                    "{\"lang\":\"xx\",\"english\":\"...\"} "
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
                data = orjson.loads(resp.choices[0].message.content)
                lang = (data.get("lang") or "unknown").strip()
                eng = (data.get("english") or text).strip()
                return {"lang": lang, "english": eng}
            except Exception as e:
                print(f"Translation error: {e}")
                return {"lang":"unknown","english":text}

        tinfo = _translate_to_english(transcript)
        transcript_en = tinfo["english"]
        lang_code = tinfo["lang"]

        return {
            "transcript": transcript_en,
            "lang": lang_code,
            "english": transcript_en,
            "raw_transcript": transcript  # Include original for debugging
        }
        
    except Exception as e:
        print(f"Voice transcribe error: {e}")
        raise HTTPException(500, f"Transcription failed: {str(e)}")

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


_food_id_counter = 0

def generate_food_id():
   """Generate a unique food ID similar to the example format"""
   global _food_id_counter
   import time
   # Create a unique ID using timestamp + counter + random number to ensure uniqueness
   timestamp_ms = int(time.time() * 1000)
   _food_id_counter += 1
   random_suffix = random.randint(100, 999)
   return str(timestamp_ms + _food_id_counter + random_suffix)


def get_food_image_url(food_name):
   """Generate food image URL based on food name"""
   # Clean food name for URL
   clean_name = food_name.replace(' ', '+')
   return f"add_image{clean_name}.png"


def create_food_item(name, calories, protein, carbs, fat, quantity, fiber=0, sugar=0):
   """Create a food item in the required format with proper quantity"""
   return {
       "id": generate_food_id(),
       "fat": fat,
       "name": name,
       "carbs": carbs,
       "fiber": fiber,
       "sugar": sugar,
       "protein": protein,
       "calories": calories,
       "quantity": quantity,
       "image_url": ""
   }


def save_meal_plan_to_database(client_id: int, meal_plan: dict, db: Session, replace_all: bool = False):
    """Save meal plan to database - only replaces templates with matching names"""
    try:
        # Standard day mapping
        standard_day_names = {
            'monday': 'Monday', 'tuesday': 'Tuesday', 'wednesday': 'Wednesday',
            'thursday': 'Thursday', 'friday': 'Friday', 'saturday': 'Saturday', 
            'sunday': 'Sunday'
        }
        
        if replace_all:
            # Full replacement - delete all existing records (only when explicitly requested)
            deleted_count = db.query(ClientDietTemplate).filter(ClientDietTemplate.client_id == client_id).delete()
        else:
            # Selective replacement - only delete records that will be replaced
            names_to_replace = []
            for day_key in meal_plan.keys():
                day_name = standard_day_names.get(day_key.lower(), day_key.replace('_', ' ').title())
                names_to_replace.append(day_name)

            if names_to_replace:
                deleted_count = db.query(ClientDietTemplate).filter(
                    ClientDietTemplate.client_id == client_id,
                    ClientDietTemplate.template_name.in_(names_to_replace)
                ).delete(synchronize_session=False)
            else:
                deleted_count = 0
        
        saved_count = 0
        
        # Save each day
        for day_key, day_data in meal_plan.items():
            try:
                # Handle custom day names
                day_name = standard_day_names.get(day_key.lower(), day_key.replace('_', ' ').title())
                
                # FIX: Store raw Python objects in JSON column, not JSON strings
                if isinstance(day_data, str):
                    # If it's a JSON string, parse it back to Python object
                    try:
                        diet_data_obj = json.loads(day_data)
                    except json.JSONDecodeError:
                        # If parsing fails, treat as plain text (shouldn't happen for meal plans)
                        diet_data_obj = {"error": "Invalid JSON", "raw_data": day_data}
                else:
                    # If it's already a Python object, use directly
                    diet_data_obj = day_data

                new_record = ClientDietTemplate(
                    client_id=client_id,
                    template_name=day_name,
                    diet_data=diet_data_obj
                )
                db.add(new_record)
                saved_count += 1
                
            except Exception as day_error:
                print(f"ERROR: Failed to save day {day_key}: {day_error}")
                continue
        
        # Commit all changes
        db.commit()

        return {
            'success': True, 
            'saved_count': saved_count,
            'deleted_count': deleted_count,
            'replace_all': replace_all,
            'message': f'Saved {saved_count} meal templates (deleted {deleted_count} old ones)'
        }
        
    except Exception as e:
        print(f"ERROR: Database save failed: {e}")
        print(f"ERROR: Full traceback: {traceback.format_exc()}")
        try:
            db.rollback()
        except Exception as rollback_error:
            print(f"ERROR: Rollback failed: {rollback_error}")
        
        return {
            'success': False, 
            'error': str(e),
            'message': f'Database save failed: {str(e)}'
        }


def has_custom_day_names(meal_plan: dict):
    """Check if meal plan contains custom day names (not standard Monday-Sunday)"""
    standard_keys = {'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday'}
    actual_keys = {key.lower().replace('_', ' ').replace(' ', '') for key in meal_plan.keys()}
    
    # Check if any key doesn't match standard day names
    for key in meal_plan.keys():
        normalized_key = key.lower().replace('_', '').replace(' ', '')
        if normalized_key not in standard_keys:
            return True
    return False


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


async def _store_meal_template(mem, db, user_id, meal_plan, name):
    """Store meal template using the async method"""
    try:
        await mem.set(f"meal_template:{user_id}", {
            "template": meal_plan,
            "name": name,
            "client_id": user_id,
            "created_at": datetime.now().isoformat()
        })
        print(f"Stored meal template '{name}' for user {user_id}")
        return True
    except Exception as e:
        print(f"Failed to store meal template: {e}")
        return False



def format_meal_plan_for_user_display(meal_plan, diet_type, cuisine_type, target_calories):
    """Format meal plan data into an attractive user-friendly display"""
    
    formatted_display = {
        "type": "meal_plan_display",
        "diet_type": diet_type,
        "cuisine_type": cuisine_type.replace('_', ' ').title(),
        "target_calories_per_day": target_calories,
        "total_days": len(meal_plan),
        "days": []
    }
    
    # Process each day
    for day_name, day_meals in meal_plan.items():
        day_display = {
            "day_name": day_name.replace('_', ' ').title(),
            "day_key": day_name,
            "total_day_calories": 0,
            "total_day_protein": 0,
            "total_day_carbs": 0,
            "total_day_fat": 0,
            "meal_slots": []
        }
        
        # Process each meal slot for this day
        for meal_slot in day_meals:
            slot_display = {
                "slot_id": meal_slot.get("id"),
                "title": meal_slot.get("title", ""),
                "time_range": meal_slot.get("timeRange", ""),
                "foods": [],
                "slot_calories": 0,
                "slot_protein": 0,
                "slot_carbs": 0,
                "slot_fat": 0,
                "food_count": meal_slot.get("itemsCount", 0)
            }
            
            # Process each food item in this slot
            for food_item in meal_slot.get("foodList", []):
                food_display = {
                    "name": food_item.get("name", ""),
                    "quantity": food_item.get("quantity", ""),
                    "calories": food_item.get("calories", 0),
                    "protein": food_item.get("protein", 0),
                    "carbs": food_item.get("carbs", 0),
                    "fat": food_item.get("fat", 0),
                    "date": food_item.get("date", ""),
                    "editable": True,
                    "food_id": food_item.get("id", "")
                }
                
                slot_display["foods"].append(food_display)
                
                # Add to slot totals
                slot_display["slot_calories"] += food_display["calories"]
                slot_display["slot_protein"] += food_display["protein"]
                slot_display["slot_carbs"] += food_display["carbs"]
                slot_display["slot_fat"] += food_display["fat"]
            
            # Round slot totals
            slot_display["slot_calories"] = round(slot_display["slot_calories"])
            slot_display["slot_protein"] = round(slot_display["slot_protein"], 1)
            slot_display["slot_carbs"] = round(slot_display["slot_carbs"], 1)
            slot_display["slot_fat"] = round(slot_display["slot_fat"], 1)
            
            day_display["meal_slots"].append(slot_display)
            
            # Add to day totals
            day_display["total_day_calories"] += slot_display["slot_calories"]
            day_display["total_day_protein"] += slot_display["slot_protein"]
            day_display["total_day_carbs"] += slot_display["slot_carbs"]
            day_display["total_day_fat"] += slot_display["slot_fat"]
        
        # Round day totals
        day_display["total_day_calories"] = round(day_display["total_day_calories"])
        day_display["total_day_protein"] = round(day_display["total_day_protein"], 1)
        day_display["total_day_carbs"] = round(day_display["total_day_carbs"], 1)
        day_display["total_day_fat"] = round(day_display["total_day_fat"], 1)
        
        formatted_display["days"].append(day_display)
    
    return formatted_display
def get_meal_emoji(meal_title):
    """Get emoji for meal slot - mobile friendly"""
    title_lower = meal_title.lower()
    
    if "early morning" in title_lower:
        return "🌅"
    elif "breakfast" in title_lower:
        return "🍳"
    elif "morning" in title_lower:
        return "🥨"
    elif "lunch" in title_lower:
        return "🍽️"
    elif "evening" in title_lower:
        return "☕"
    elif "dinner" in title_lower:
        return "🌙"
    else:
        return "🍴"

def format_day_loader_json(day_name):
    """Format a loader JSON event for generating a specific day with animation"""
    day_display_name = day_name.replace('_', ' ').title()
    return {
        "type": "loader",
        "is_loader": True,
        "day_name": day_display_name,
        "message": f"Generating {day_display_name} meal plan..."
    }


def format_single_day_for_streaming(day_name, day_meals):
    """Format a single day's meal plan for streaming (appends to existing message)"""

    message_parts = []
    day_display_name = day_name.replace('_', ' ').title()
    message_parts.append(f"📅 {day_display_name.upper()}")
    message_parts.append("─" * 30)

    day_total_calories = 0

    for meal_slot in day_meals:
        title = meal_slot.get("title", "")
        time_range = meal_slot.get("timeRange", "")
        foods = meal_slot.get("foodList", [])

        if foods:
            meal_emoji = get_meal_emoji(title)
            message_parts.append(f"\n{meal_emoji} {title}")
            message_parts.append(f"⏰ {time_range}")

            slot_calories = 0
            for food in foods:
                name = food.get("name", "")
                quantity = food.get("quantity", "")
                calories = food.get("calories", 0)
                protein = food.get("protein", 0)
                carbs = food.get("carbs", 0)
                fat = food.get("fat", 0)

                message_parts.append(f"  • {name}")
                message_parts.append(f"    Qty: {quantity}")
                message_parts.append(f"    {calories}cal | {protein}g protein | {carbs}g carbs | {fat}g fat")

                slot_calories += calories
                day_total_calories += calories

            message_parts.append(f"Total: {slot_calories} cal")

    message_parts.append(f"\n  Day Total: {day_total_calories} calories")
    message_parts.append("=" * 50 + "\n")

    return "\n".join(message_parts)


def create_user_friendly_meal_plan_message(meal_plan, diet_type, cuisine_type, target_calories):
    """Create a formatted text message showing the meal plan structure"""

    message_parts = []
    message_parts.append("🍽️ YOUR MEAL PLAN")
    message_parts.append(f"Diet: {diet_type.title()}")
    message_parts.append(f"Style: {cuisine_type.replace('_', ' ').title()}")
    message_parts.append(f"Daily Goal: {target_calories} cal")
    message_parts.append("")
    
    for day_name, day_meals in meal_plan.items():
        day_display_name = day_name.replace('_', ' ').title()
        message_parts.append(f"📅 {day_display_name.upper()}")
        message_parts.append("─" * 18)
        
        day_total_calories = 0
        
        for meal_slot in day_meals:
            title = meal_slot.get("title", "")
            time_range = meal_slot.get("timeRange", "")
            foods = meal_slot.get("foodList", [])
            
            if foods:
                meal_emoji = get_meal_emoji(title)
                message_parts.append(f"{meal_emoji} {title}")
                message_parts.append(f"⏰ {time_range}")
                
                slot_calories = 0
                for food in foods:
                    name = food.get("name", "")
                    quantity = food.get("quantity", "")
                    calories = food.get("calories", 0)
                    protein = food.get("protein", 0)
                    carbs = food.get("carbs", 0)
                    fat = food.get("fat", 0)
                    
                    message_parts.append(f"  • {name}")
                    message_parts.append(f"    Qty: {quantity}")
                    message_parts.append(f"    {calories}cal | {protein}g protein | {carbs}g carbs | {fat}g fat")
                    
                    slot_calories += calories
                
                message_parts.append(f"Total: {slot_calories} cal")
                message_parts.append("")
        
        # message_parts.append(f"  Day Total: {day_total_calories} calories")
        message_parts.append("=" * 50 + "\n")
    
    message_parts.append("You can edit any food item by telling me what you'd like to change!")
    message_parts.append("Continue with day name customization or finalization when ready.")
    
    return "\n".join(message_parts)

def detect_food_edit_request(text):
    """Detect if user wants to edit a specific food item"""
    text_lower = text.lower().strip()
    
    edit_patterns = [
        r'change\s+(\w+)\s+(\w+)\s+to\s+(.+)',
        r'replace\s+(.+)\s+with\s+(.+)',
        r'edit\s+(\w+)\s+(\w+)',
        r'modify\s+(.+)',
        r'update\s+(.+)',
    ]
    
    for pattern in edit_patterns:
        match = re.search(pattern, text_lower)
        if match:
            return {
                'is_edit_request': True,
                'original_text': text,
                'pattern_match': match.groups()
            }
    
    return {'is_edit_request': False}

def detect_food_restrictions(text):
    """Detect food allergies/restrictions from user input"""
    text = text.lower().strip()
    
    # Skip simple action words that aren't about allergies
    simple_actions = ['remove', 'change', 'edit', 'modify', 'update', 'replace']
    if text in simple_actions:
        return None
    
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
    
    # Restriction trigger words - must have both trigger AND specific food
    restriction_triggers = [
        r'\ballerg\w*\s+to\b',  # allergic to
        r'\bremove.*\b(all|any)\b',  # remove all/any
        r'\bavoid.*\b(all|any)\b',  # avoid all/any  
        r'\bcan\'?t\s+eat\b',
        r'\bdont?\s+eat\b',
        r'\bdon\'?t\s+eat\b',
        r'\bnot\s+allowed\b',
        r'\brestrict\w*\s+from\b',
        r'\bintoleran\w*\s+to\b',
        r'\bsensitiv\w*\s+to\b',
        r'\bexclude.*from\b',
        r'\bi\s+am\s+allergic\b',
        r'\bi\s+have.*allergy\b'
    ]
    
    # Check if text contains meaningful restriction triggers
    has_restriction_trigger = any(re.search(pattern, text) for pattern in restriction_triggers)
    
    # Also check for "no [food]" patterns
    no_food_pattern = r'\bno\s+(\w+)'
    no_food_matches = re.findall(no_food_pattern, text)
    
    if not has_restriction_trigger and not no_food_matches:
        return None
    
    # Find specific allergens/foods mentioned
    found_restrictions = []
    for allergen, patterns in allergen_patterns.items():
        if any(re.search(pattern, text) for pattern in patterns):
            found_restrictions.append(allergen)
    
    # Also extract other food names mentioned with restrictions
    other_foods = []
    if no_food_matches:
        other_foods.extend(no_food_matches)
    
    # Look for foods mentioned after restriction triggers
    words = text.split()
    for i, word in enumerate(words):
        if any(re.search(trigger, ' '.join(words[i:i+3])) for trigger in restriction_triggers):
            for j in range(i+1, min(i+5, len(words))):
                next_word = words[j].strip('.,!?')
                if len(next_word) > 2 and next_word not in ['the', 'and', 'or', 'any', 'from', 'plan', 'all']:
                    other_foods.append(next_word)
    
    result = {
        'found_allergens': found_restrictions,
        'other_foods': list(set(other_foods)),  # Remove duplicates
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
- Same meal structure (10 time slots)
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
            # Use fallback for this day
            fallback_data = create_fallback_meal_plan(day, diet_type, cuisine_type, profile, previous_meals)
            day_template = convert_ai_meal_to_template(fallback_data)
            new_meal_plan[day.lower()] = day_template
    
    return new_meal_plan

def simple_food_removal(meal_plan, foods_to_remove):
    """Simple food removal without AI regeneration - for quick fixes"""
    try:
        if not foods_to_remove:
            return meal_plan
        
        # Convert foods to lowercase for matching
        remove_foods_lower = [food.lower().strip() for food in foods_to_remove]
        
        updated_meal_plan = {}
        removed_items_count = 0
        
        for day_key, day_data in meal_plan.items():
            updated_day_data = []
            
            for meal_slot in day_data:
                updated_food_list = []
                slot_calories_removed = 0
                
                for food_item in meal_slot.get('foodList', []):
                    food_name = food_item.get('name', '').lower()
                    
                    # Check if this food should be removed
                    should_remove = any(remove_food in food_name for remove_food in remove_foods_lower)
                    
                    if should_remove:
                        slot_calories_removed += food_item.get('calories', 0)
                        removed_items_count += 1
                    else:
                        updated_food_list.append(food_item)
                
                # Update the meal slot
                updated_slot = meal_slot.copy()
                updated_slot['foodList'] = updated_food_list
                updated_slot['itemsCount'] = len(updated_food_list)
                
                # If we removed significant calories, add a simple replacement
                if slot_calories_removed > 50 and len(updated_food_list) == 0:
                    # Add a simple replacement food
                    replacement_food = create_food_item(
                        name="Mixed vegetables",
                        calories=slot_calories_removed,
                        protein=3,
                        carbs=8,
                        fat=2,
                        quantity="100 grams | 1/2 cup | 3/4 bowl",
                        fiber=3,
                        sugar=5
                    )
                    updated_slot['foodList'].append(replacement_food)
                    updated_slot['itemsCount'] = 1

                updated_day_data.append(updated_slot)
            
            updated_meal_plan[day_key] = updated_day_data

        return updated_meal_plan

    except Exception as e:
        print(f"ERROR: Simple food removal failed: {e}")
        print(f"ERROR: Simple removal traceback: {traceback.format_exc()}")
        # Return original plan if removal fails
        return meal_plan


def calculate_meal_calories_distribution(target_calories):
   """Calculate calorie distribution across meal slots"""
   distributions = {
       "1": 0.05,  # Pre workout - 5%
       "2": 0.08,  # Post workout - 8%
       "3": 0.02,  # Early morning Detox - 2%
       "4": 0.05,  # Pre-Breakfast Starter - 5%
       "5": 0.25,  # Breakfast - 25%
       "6": 0.10,  # Mid-Morning Snack - 10%
       "7": 0.25,  # Lunch - 25%
       "8": 0.08,  # Evening Snack - 8%
       "9": 0.20,  # Dinner - 20%
       "10": 0.02  # Bed time - 2%
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

   if abs(current_calories - target_calories) <= tolerance:
       return meal_data

   # Calculate adjustment factor
   if current_calories > 0:
       scale_factor = target_calories / current_calories
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

        # Clean JSON response
        result = re.sub(r"^```json\s*", "", result)
        result = re.sub(r"\s*```$", "", result)
        result = result.strip()

        try:
            meal_data = json.loads(result)
        except json.JSONDecodeError as je:
            raise je

        # Validate diet compliance
        validate_diet_compliance(meal_data, diet_type)

        # Validate and adjust calories
        validated_meal_data = validate_and_adjust_calories(meal_data, target_calories)

        return validated_meal_data
        
    except Exception as e:
        print(f"AI meal generation error for {day_name}: {e}")
        print(f"AI generation traceback: {traceback.format_exc()}")
        return create_fallback_meal_plan(day_name, diet_type, cuisine_type, profile, previous_meals)


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
       "1": "Pre workout",
       "2": "Post workout",
       "3": "Early morning Detox",
       "4": "Pre-Breakfast / Pre-Meal Starter",
       "5": "Breakfast",
       "6": "Mid-Morning snack",
       "7": "Lunch",
       "8": "Evening snack",
       "9": "Dinner",
       "10": "Bed time"
   }
   return slot_names.get(slot_id, f"Slot {slot_id}")


def create_fallback_meal_plan(day_name, diet_type, cuisine_type, profile, previous_meals=None):
   """Create a fallback meal plan with accurate calories"""
   target_calories = profile['target_calories']
   slot_calories = calculate_meal_calories_distribution(target_calories)
  
   # Create basic fallback based on cuisine and diet type
   fallback_meals = []
  
   for slot_id in ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]:
       slot_target = slot_calories[slot_id]["target"]

       if slot_id == "1":  # Pre workout
           foods = [{"name": "Banana", "calories": slot_target, "protein": 1, "carbs": 27, "fat": 0, "quantity": "1 medium"}]
       elif slot_id == "2":  # Post workout
           foods = [{"name": "Protein shake", "calories": slot_target, "protein": 20, "carbs": 15, "fat": 2, "quantity": "1 serving"}]
       elif slot_id == "3":  # Early morning Detox
           foods = [{"name": "Warm lemon water", "calories": slot_target, "protein": 0, "carbs": 2, "fat": 0, "quantity": "250 ml"}]
       elif slot_id == "4":  # Pre-breakfast
           foods = [{"name": "Soaked almonds", "calories": slot_target, "protein": 3, "carbs": 2, "fat": 5, "quantity": "5 pieces"}]
       elif slot_id == "5":  # Breakfast
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
       elif slot_id == "6":  # Mid-morning
           foods = [{"name": "Mixed fruit", "calories": slot_target, "protein": 2, "carbs": 20, "fat": 1, "quantity": "100 grams"}]
       elif slot_id == "7":  # Lunch
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
       elif slot_id == "8":  # Evening snack
           foods = [{"name": "Tea with biscuits", "calories": slot_target, "protein": 3, "carbs": 15, "fat": 4, "quantity": "1 cup + 2 biscuits"}]
       elif slot_id == "9":  # Dinner
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
       elif slot_id == "10":  # Bed time
           foods = [{"name": "Warm milk", "calories": slot_target, "protein": 3, "carbs": 5, "fat": 3, "quantity": "150 ml"}]

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

       if slot_id == "5":  # Breakfast
           main_meals['breakfasts'].extend([food.get("name", "") for food in foods])
       elif slot_id == "7":  # Lunch
           main_meals['lunches'].extend([food.get("name", "") for food in foods])
       elif slot_id == "9":  # Dinner
           main_meals['dinners'].extend([food.get("name", "") for food in foods])

   return main_meals


def generate_7_day_meal_plan(profile, diet_type, cuisine_type, progress_callback=None):
   """Generate complete 7-day meal plan with variety tracking and cuisine preference

   Args:
       profile: User profile dict
       diet_type: Diet type string
       cuisine_type: Cuisine preference string
       progress_callback: Optional callable to report progress (day_name, day_number, total_days)
   """
   days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
   meal_plan = {}
   previous_meals = {
       'breakfasts': [],
       'lunches': [],
       'dinners': []
   }

   for day_index, day in enumerate(days):
       # Report progress if callback provided
       if progress_callback:
           progress_callback(day, day_index + 1, len(days))

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
    """Detect diet preference from user input with comprehensive diet options"""
    text = text.lower().strip()

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
            return "ketogenic"

    # 2. Paleo
    for pattern in paleo_patterns:
        if re.search(pattern, text):
            return "paleo"

    # 3. Jain diet (very specific restrictions)
    for pattern in jain_patterns:
        if re.search(pattern, text):
            return "jain"

    # 4. Vegan (more restrictive than vegetarian)
    for pattern in vegan_patterns:
        if re.search(pattern, text):
            return "vegan"

    # 5. Eggetarian (vegetarian + eggs)
    for pattern in eggetarian_patterns:
        if re.search(pattern, text):
            return "eggetarian"

    # 6. Non-vegetarian
    for pattern in non_veg_patterns:
        if re.search(pattern, text):
            return "non-vegetarian"

    # 7. Vegetarian (least specific)
    for pattern in veg_patterns:
        if re.search(pattern, text):
            return "vegetarian"

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
  
   # Check for North Indian
   for pattern in north_indian_patterns:
       if re.search(pattern, text):
           return "north_indian"

   # Check for South Indian
   for pattern in south_indian_patterns:
       if re.search(pattern, text):
           return "south_indian"

   # Check for commonly available
   for pattern in common_patterns:
       if re.search(pattern, text):
           return "commonly_available"

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
                    return False
    return True


def generate_single_day_with_restrictions(day_name, profile, diet_type, cuisine_type, avoid_foods, previous_meals):
    """Generate a single day's meal plan avoiding specific foods (for streaming regeneration)"""

    target_calories = profile['target_calories']
    slot_calories = calculate_meal_calories_distribution(target_calories)

    calorie_breakdown = "\n".join([
        f"Slot {slot_id} ({get_slot_name(slot_id)}): {info['target']} calories (range: {info['min']}-{info['max']})"
        for slot_id, info in slot_calories.items()
    ])

    cuisine_context = {
        "north_indian": "Focus on North Indian cuisine: Roti, dal, curry-based dishes, paneer, rice dishes",
        "south_indian": "Focus on South Indian cuisine: Idli, dosa, sambar, rice-based dishes, coconut",
        "commonly_available": "Focus on commonly available Indian foods: Simple breads, rice, dal, vegetables"
    }

    cuisine_instruction = cuisine_context.get(cuisine_type, cuisine_context["commonly_available"])
    avoid_list = ', '.join(avoid_foods) if avoid_foods else "none"

    # Previous meals context
    previous_context = ""
    if previous_meals and (previous_meals.get('breakfasts') or previous_meals.get('lunches') or previous_meals.get('dinners')):
        previous_context = f"""
AVOID REPETITION:
Previous Breakfasts: {', '.join(previous_meals.get('breakfasts', [])[-3:])}
Previous Lunches: {', '.join(previous_meals.get('lunches', [])[-3:])}
Previous Dinners: {', '.join(previous_meals.get('dinners', [])[-3:])}
Ensure {day_name}'s meals are DIFFERENT.
"""

    prompt = f"""
Create {day_name} meal plan AVOIDING these foods: {avoid_list}

Profile:
- Goal: {profile['goal_type']} ({profile['weight_delta_text']})
- Target: {target_calories} calories/day
- Diet: {diet_type}
- Cuisine: {cuisine_type.replace('_', ' ').title()}

CRITICAL: Completely avoid: {avoid_list}
{previous_context}

CALORIE REQUIREMENTS:
{calorie_breakdown}

{cuisine_instruction}

Return ONLY valid JSON:
{{
    "day_name": "{day_name}",
    "total_target_calories": {target_calories},
    "restrictions_avoided": [{avoid_list}],
    "meals": [
        {{
            "slot_id": "1-10",
            "target_calories": number,
            "foods": [
                {{"name": "food", "calories": num, "protein": num, "carbs": num, "fat": num, "quantity": "amount"}}
            ]
        }}
    ]
}}
"""

    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.7
        )

        result = orjson.loads(response.choices[0].message.content)
        return result
    except Exception as e:
        print(f"Error generating {day_name}: {e}")
        # Return fallback
        return create_fallback_meal_plan(day_name, diet_type, cuisine_type, profile, previous_meals)


def ai_intent_classifier(user_input: str, current_state: str, oai) -> dict:
    """
    AI-driven intent classifier that understands user intent flexibly.
    Handles typos, natural language, and context-aware interpretation.

    Args:
        user_input: The user's message (can contain typos, be informal)
        current_state: Current conversation state
        oai: OpenAI client

    Returns:
        dict with 'intent' and extracted parameters
    """

    system_prompt = f"""You are an intent classifier for a meal planning chatbot.
Current conversation state: {current_state}

Classify the user's intent into ONE of these categories:

1. **diet_preference**: User is specifying their diet type
   - Extract: diet_type (vegetarian, non-vegetarian, vegan, eggetarian, jain, ketogenic, paleo)

2. **cuisine_preference**: User is specifying cuisine preference
   - Extract: cuisine_type (north_indian, south_indian, commonly_available)
   - Note: "simple", "basic", "common", "everyday" → commonly_available

3. **food_allergy**: User mentions food allergies or items to avoid
   - Extract: allergens (list of foods/ingredients to avoid)

4. **food_removal**: User wants to remove specific foods
   - Extract: foods_to_remove (list of food items)

5. **day_rename_yes**: User agrees to rename days (yes, sure, okay, etc.)

6. **day_rename_no**: User declines to rename days or wants to finalize (no, nope, done, finish, save, etc.)

7. **day_selection**: User is selecting which days to rename (numbers, "all", day names)
   - Extract: selected_days (list or "all")

8. **custom_day_name**: User is providing a custom name for a day
   - Extract: day_name (the custom name)

9. **unclear**: User input is unclear or doesn't match any intent

IMPORTANT: Be flexible with typos, informal language, and variations.
Examples:
- "im allergick to nuts" → food_allergy, allergens: ["nuts"]
- "rmove dairy plz" → food_removal, foods_to_remove: ["dairy"]
- "ya sure" → day_rename_yes
- "nah im good" → day_rename_no
- "non vej" → diet_preference, diet_type: "non-vegetarian"
- "south indain cuisne" → cuisine_preference, cuisine_type: "south_indian"

Return ONLY valid JSON in this format:
{{
    "intent": "intent_name",
    "confidence": 0.95,
    "extracted_data": {{
        "key": "value"
    }},
    "normalized_input": "corrected version of user input"
}}"""

    try:
        response = oai.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input}
            ],
            response_format={"type": "json_object"},
            temperature=0.3,
            max_tokens=300
        )

        result = orjson.loads(response.choices[0].message.content)
        return result

    except Exception as e:
        print(f"ERROR: AI intent classification failed: {e}")
        return {
            "intent": "unclear",
            "confidence": 0.0,
            "extracted_data": {},
            "normalized_input": user_input
        }


def extract_diet_from_ai_intent(intent_result: dict) -> str:
    """Extract and normalize diet preference from AI intent result"""
    if intent_result.get("intent") != "diet_preference":
        return None

    extracted = intent_result.get("extracted_data", {})
    diet = extracted.get("diet_type", "").lower()

    # Normalize variations
    diet_map = {
        "veg": "vegetarian",
        "vegetarian": "vegetarian",
        "non-veg": "non-vegetarian",
        "nonveg": "non-vegetarian",
        "non-vegetarian": "non-vegetarian",
        "non vegetarian": "non-vegetarian",
        "eggetarian": "eggetarian",
        "egg": "eggetarian",
        "vegan": "vegan",
        "jain": "jain",
        "keto": "ketogenic",
        "ketogenic": "ketogenic",
        "paleo": "paleo"
    }

    return diet_map.get(diet)


def extract_cuisine_from_ai_intent(intent_result: dict) -> str:
    """Extract and normalize cuisine preference from AI intent result"""
    if intent_result.get("intent") != "cuisine_preference":
        return None

    extracted = intent_result.get("extracted_data", {})
    cuisine = extracted.get("cuisine_type", "").lower().replace(" ", "_")

    # Normalize variations
    cuisine_map = {
        "north_indian": "north_indian",
        "northindian": "north_indian",
        "north": "north_indian",
        "south_indian": "south_indian",
        "southindian": "south_indian",
        "south": "south_indian",
        "common": "commonly_available",
        "commonly_available": "commonly_available",
        "simple": "commonly_available",
        "basic": "commonly_available"
    }

    return cuisine_map.get(cuisine)


def extract_food_restrictions_from_ai(intent_result: dict) -> dict:
    """Extract food restrictions from AI intent result"""
    if intent_result.get("intent") not in ["food_allergy", "food_removal"]:
        return None

    extracted = intent_result.get("extracted_data", {})

    if intent_result.get("intent") == "food_allergy":
        allergens = extracted.get("allergens", [])
        return {
            'found_allergens': [],
            'other_foods': allergens if isinstance(allergens, list) else [allergens],
            'raw_text': intent_result.get("normalized_input", ""),
            'ai_detected': True
        }
    else:  # food_removal
        foods = extracted.get("foods_to_remove", [])
        return {
            'found_allergens': [],
            'other_foods': foods if isinstance(foods, list) else [foods],
            'raw_text': intent_result.get("normalized_input", ""),
            'simple_removal': True,
            'ai_detected': True
        }


@router.get("/chat/stream")
# @router.get("/chat/stream", dependencies=[Depends(RateLimiter(times=30, seconds=60))])
async def chat_stream(
   user_id: int,
   client_id: int = Query(..., description="Client ID for whom to create meal plan"),
   text: str = Query(None),
   audio_transcript: str = Query(None, description="Transcribed audio text"),
   mem = Depends(get_mem),
   oai = Depends(get_oai),
   db: Session = Depends(get_db),
):
   try:
       if not text:
           # Fetch client profile and start conversation
           try:
               profile = _fetch_profile(db, client_id)

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
      
       text = text.strip() if text else ""

       # Handle voice input
       if audio_transcript and not text:
            text = audio_transcript.strip()
      
       # Get pending state from memory
       try:
           pending_state = await mem.get_pending(user_id)
       except Exception as e:
           print(f"Error getting pending state: {e}")
           pending_state = None
      
       # Handle diet preference selection
       if pending_state and pending_state.get("state") == "awaiting_diet_preference":
           # Use AI intent classification
           ai_intent = ai_intent_classifier(text, "awaiting_diet_preference", oai)
           diet_type = extract_diet_from_ai_intent(ai_intent)

           # Fallback to regex patterns if AI fails
           if not diet_type:
               diet_type = detect_diet_preference(text)

           if diet_type:
               # Move to cuisine preference selection
               await mem.set_pending(user_id, {
                   "state": "awaiting_cuisine_preference",
                   "client_id": pending_state.get("client_id"),
                   "profile": pending_state.get("profile"),
                   "diet_type": diet_type
               })

               async def _ask_cuisine():
                   cuisine_msg = f"""Great! You've selected {diet_type} 🎉

Which cuisine do you prefer?

🍛 North Indian
🥥 South Indian
🍽️ Commonly Available

Please choose: North Indian, South Indian, or Commonly Available"""

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
            # Use AI intent classification
            ai_intent = ai_intent_classifier(text, "awaiting_cuisine_preference", oai)
            cuisine_type = extract_cuisine_from_ai_intent(ai_intent)

            # Fallback to regex patterns if AI fails
            if not cuisine_type:
                cuisine_type = detect_cuisine_preference(text)

            if cuisine_type:
                # Generate meal plan with both diet and cuisine preferences
                profile = pending_state.get("profile")
                diet_type = pending_state.get("diet_type")

                async def _generate_plan():
                    try:
                        cuisine_display = cuisine_type.replace('_', ' ').title()

                        # Start with header - stream as plain text
                        header = f"""🍽️ YOUR MEAL PLAN
Diet: {diet_type.title()}
Style: {cuisine_display}
Daily Goal: {profile['target_calories']} cal

"""
                        yield sse_data(header)

                        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
                        meal_plan = {}
                        previous_meals = {'breakfasts': [], 'lunches': [], 'dinners': []}

                        for day_index, day in enumerate(days):
                            try:
                                # Send animated loader event before generating each day
                                loader_event = format_day_loader_json(day)
                                yield sse_json(loader_event)

                                meal_data = generate_meal_plan_with_ai(profile, diet_type, cuisine_type, day, previous_meals)
                                template = convert_ai_meal_to_template(meal_data)
                                meal_plan[day.lower()] = template

                                # Format and stream this day's content immediately as plain text
                                day_content = format_single_day_for_streaming(day, template)
                                yield sse_data(day_content)

                                # Extract and track main meals for next day's variety
                                day_meals = extract_main_meals(meal_data)
                                previous_meals['breakfasts'].extend(day_meals['breakfasts'])
                                previous_meals['lunches'].extend(day_meals['lunches'])
                                previous_meals['dinners'].extend(day_meals['dinners'])

                                # Keep only last 3 days of meals
                                if len(previous_meals['breakfasts']) > 9:
                                    previous_meals['breakfasts'] = previous_meals['breakfasts'][-9:]
                                if len(previous_meals['lunches']) > 9:
                                    previous_meals['lunches'] = previous_meals['lunches'][-9:]
                                if len(previous_meals['dinners']) > 9:
                                    previous_meals['dinners'] = previous_meals['dinners'][-9:]

                            except Exception as e:
                                print(f"Error generating meal plan for {day}: {e}")

                                # Send loader event for fallback too
                                loader_event = format_day_loader_json(day)
                                yield sse_json(loader_event)

                                # Use fallback for this day
                                fallback_data = create_fallback_meal_plan(day, diet_type, cuisine_type, profile, previous_meals)
                                template = convert_ai_meal_to_template(fallback_data)
                                meal_plan[day.lower()] = template

                                # Stream the fallback day's content as plain text
                                day_content = format_single_day_for_streaming(day, template)
                                yield sse_data(day_content)

                                # Track fallback meals too
                                day_meals = extract_main_meals(fallback_data)
                                previous_meals['breakfasts'].extend(day_meals['breakfasts'])
                                previous_meals['lunches'].extend(day_meals['lunches'])
                                previous_meals['dinners'].extend(day_meals['dinners'])

                        # Validate meal plan has content
                        if not meal_plan or len(meal_plan) == 0:
                            print("ERROR: Empty meal plan generated")
                            yield sse_data('Sorry, I encountered an issue generating your meal plan. Please try again.')
                            yield "event: done\ndata: [DONE]\n\n"
                            return

                        # Send consolidated meal plan data (for backend storage)
                        yield sse_json({
                            "type": "meal_plan_complete",
                            "status": "created",
                            "diet_type": diet_type,
                            "cuisine_type": cuisine_display,
                            "total_calories_per_day": profile['target_calories'],
                            "goal": profile['weight_delta_text'],
                            "meal_plan": meal_plan,
                            "message": f"Created {diet_type} {cuisine_display} meal plan successfully!"
                        })

                        # Set pending state and ask about customization
                        await mem.set_pending(user_id, {
                            "state": "awaiting_name_change_or_edit",
                            "client_id": pending_state.get("client_id"),
                            "profile": profile,
                            "diet_type": diet_type,
                            "cuisine_type": cuisine_type,
                            "meal_plan": meal_plan
                        })

                        # Show options message as plain text
                        options_msg = f"""

🎉 Your {diet_type.title()} meal plan is ready!

 What's next?

🔧 Edit foods: "remove dairy"
📝 Rename days: "yes"
✅ Finalize: "no" or "done"

What would you like to do?"""

                        yield sse_data(options_msg)
                        yield "event: done\ndata: [DONE]\n\n"

                    except Exception as e:
                        print(f"Error generating meal plan: {e}")
                        print(f"Meal generation full traceback: {traceback.format_exc()}")
                        yield sse_data('Sorry, there was an error creating your meal plan. Please try again or contact support.')
                        yield "event: done\ndata: [DONE]\n\n"
                
                return StreamingResponse(_generate_plan(), media_type="text/event-stream",
                                        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
            else:
                # Handle unrecognized cuisine input - ask for clarification
                async def _ask_cuisine_again():
                    cuisine_msg = """I didn't recognize that cuisine preference. Please choose from:

• North Indian - Roti, dal, curry-based dishes
• South Indian - Rice, sambar, coconut-based dishes
• Commonly Available - Simple, everyday foods

Please type one of: North Indian, South Indian, or Commonly Available"""

                    yield f"data: {json.dumps({'message': cuisine_msg, 'type': 'cuisine_clarification'})}\n\n"
                    yield "event: done\ndata: [DONE]\n\n"

                return StreamingResponse(_ask_cuisine_again(), media_type="text/event-stream",
                                        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

       # Handle name change and allergy check
       elif pending_state and pending_state.get("state") == "awaiting_name_change_or_edit":
            # Use AI intent classification
            ai_intent = ai_intent_classifier(text, "awaiting_name_change_or_edit", oai)
            intent_type = ai_intent.get("intent")

            # Handle food allergies/removal
            if intent_type in ["food_allergy", "food_removal"]:
                restrictions = extract_food_restrictions_from_ai(ai_intent)

                # Fallback to regex if AI didn't extract anything useful
                if not restrictions or not restrictions.get('other_foods'):
                    restrictions = detect_food_restrictions(text)

            elif intent_type in ["day_rename_yes"]:
                # User wants to rename days
                async def _ask_which_days():
                    msg = """📅 Rename days:
1️⃣ Mon  2️⃣ Tue  3️⃣ Wed  4️⃣ Thu
5️⃣ Fri  6️⃣ Sat  7️⃣ Sun

✏️ Type:
- One: "3"
- Many: "1,3,5"
- All: "all"
- Cancel: "cancel"
💡 Ex: "2" → rename Tue"""

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

            elif intent_type in ["day_rename_no"] or text.lower() in ['finalize', 'finish', 'done', 'save']:
                # Finalize with default names
                meal_plan = pending_state.get("meal_plan")
                client_id = pending_state.get("client_id")
                template_name = "Meal Template (Mon-Sun)"

                # Store template in memory first
                await _store_meal_template(mem, db, client_id, meal_plan, template_name)

                # Save to database directly
                save_result = save_meal_plan_to_database(client_id, meal_plan, db, replace_all=False)

                await mem.clear_pending(user_id)

                async def _finalize_default():
                    if save_result and save_result.get('success'):
                        if save_result.get('merge_mode'):
                            message = f'✅ Your meal plan is updated! Custom day names added as new templates. Total templates: {save_result.get("saved_count", 0)}'
                        else:
                            message = '✅ Your 7-day meal plan is finalized and saved to database!'
                    else:
                        message = '⚠️ Your 7-day meal plan is finalized but database save failed!'

                    yield sse_json({
                        "type": "meal_template",
                        "status": "stored" if (save_result and save_result.get('success')) else "error",
                        "template_name": template_name,
                        "meal_plan": meal_plan,
                        "day_names": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
                        "message": message
                    })
                    yield "event: done\ndata: [DONE]\n\n"

                return StreamingResponse(_finalize_default(), media_type="text/event-stream",
                                        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

            else:
                # Check for restrictions using fallback regex
                restrictions = detect_food_restrictions(text)

            if restrictions:
                # Regenerate meal plan without allergens
                async def _regenerate_with_restrictions():
                    try:
                        avoid_items = restrictions.get('found_allergens', []) + restrictions.get('other_foods', [])
                        avoid_text = ', '.join(avoid_items)

                        diet_type = pending_state.get("diet_type")
                        cuisine_type = pending_state.get("cuisine_type")
                        profile = pending_state.get("profile")

                        # Stream header FIRST with "understanding" message
                        cuisine_display = cuisine_type.replace('_', ' ').title()
                        header = f"""I understand you want to avoid: {avoid_text}.

Regenerating your 7-day meal plan...

🍽️ YOUR UPDATED MEAL PLAN
Diet: {diet_type.title()}
Style: {cuisine_display}
Daily Goal: {profile['target_calories']} cal
Avoided: {avoid_text}

"""
                        yield sse_data(header)

                        # Regenerate meal plan day by day (inline, streaming each as generated)
                        import asyncio
                        from functools import partial

                        # Prepare restrictions
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

                        avoid_foods = []
                        for allergen in restrictions.get('found_allergens', []):
                            avoid_foods.extend(allergen_foods_map.get(allergen, []))
                        avoid_foods.extend(restrictions.get('other_foods', []))
                        avoid_foods = [food.lower() for food in avoid_foods]

                        # Generate each day ONE AT A TIME and stream immediately
                        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
                        new_meal_plan = {}
                        previous_meals = {'breakfasts': [], 'lunches': [], 'dinners': []}
                        loop = asyncio.get_event_loop()

                        for day in days:
                            try:
                                # Send animated loader event before generating each day
                                loader_event = format_day_loader_json(day)
                                yield sse_json(loader_event)

                                # Generate THIS day in executor (non-blocking)
                                day_gen_func = partial(
                                    generate_single_day_with_restrictions,
                                    day, profile, diet_type, cuisine_type, avoid_foods, previous_meals
                                )

                                meal_data = await loop.run_in_executor(None, day_gen_func)
                                template = convert_ai_meal_to_template(meal_data)
                                new_meal_plan[day.lower()] = template

                                # Stream THIS day immediately
                                day_content = format_single_day_for_streaming(day, template)
                                yield sse_data(day_content)

                                # Track meals for variety
                                day_meals = extract_main_meals(meal_data)
                                previous_meals['breakfasts'].extend(day_meals['breakfasts'])
                                previous_meals['lunches'].extend(day_meals['lunches'])
                                previous_meals['dinners'].extend(day_meals['dinners'])

                                # Keep only last 3 days
                                if len(previous_meals['breakfasts']) > 9:
                                    previous_meals['breakfasts'] = previous_meals['breakfasts'][-9:]
                                if len(previous_meals['lunches']) > 9:
                                    previous_meals['lunches'] = previous_meals['lunches'][-9:]
                                if len(previous_meals['dinners']) > 9:
                                    previous_meals['dinners'] = previous_meals['dinners'][-9:]

                            except Exception as e:
                                print(f"Error generating {day}: {e}")

                                # Send loader event for fallback too
                                loader_event = format_day_loader_json(day)
                                yield sse_json(loader_event)

                                # Use fallback
                                fallback_data = create_fallback_meal_plan(day, diet_type, cuisine_type, profile, previous_meals)
                                template = convert_ai_meal_to_template(fallback_data)
                                new_meal_plan[day.lower()] = template

                                day_content = format_single_day_for_streaming(day, template)
                                yield sse_data(day_content)

                        if new_meal_plan and len(new_meal_plan) == 7:  # Should have 7 days

                            # Update pending state with new meal plan
                            await mem.set_pending(user_id, {
                                "state": "awaiting_name_change_after_allergy",
                                "client_id": pending_state.get("client_id"),
                                "profile": profile,
                                "diet_type": diet_type,
                                "cuisine_type": cuisine_type,
                                "meal_plan": new_meal_plan,
                                "avoided_foods": avoid_items
                            })

                            # Send consolidated meal plan data (for backend storage)
                            yield sse_json({
                                "type": "meal_plan_updated_complete",
                                "status": "regenerated",
                                "diet_type": diet_type,
                                "avoided_foods": avoid_items,
                                "meal_plan": new_meal_plan,
                                "message": f"Updated meal plan without {', '.join(avoid_items)} successfully!"
                            })

                            # Show options message
                            options_msg = f"""

🎉 Your meal plan is updated (avoiding {avoid_text})!

What's next?

🔧 Edit foods: "remove dairy"
📝 Rename days: "yes"
✅ Finalize: "no" or "done"

What would you like to do?"""

                            yield sse_data(options_msg)
                            yield "event: done\ndata: [DONE]\n\n"
                            
                        else:
                            yield f"data: {json.dumps({'message': 'Sorry, I had trouble regenerating the complete meal plan. Would you like to continue with the original plan and customize day names instead?', 'type': 'regeneration_error'})}\n\n"
                            yield "event: done\ndata: [DONE]\n\n"
                            
                    except Exception as e:
                        print(f"Error in allergy regeneration: {e}")
                        print(f"Allergy regeneration traceback: {traceback.format_exc()}")
                        yield f"data: {json.dumps({'message': 'Error updating meal plan. Would you like to continue with day name customization?', 'type': 'error'})}\n\n"
                        yield "event: done\ndata: [DONE]\n\n"
                
                return StreamingResponse(_regenerate_with_restrictions(), media_type="text/event-stream",
                                        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

            

            # Handle unclear intents
            if not restrictions:
                # Handle unrecognized input in name change state
                async def _clarify_options():
                    msg = """I didn't understand that. Here are your options:

🍽️ **Food Changes**: Say things like:
• "I'm allergic to peanuts"
• "Remove dairy from plan"
• "I can't eat eggs"

📝 **Day Names**: Say:
• "yes" - to customize day names
• "no" - to keep Monday-Sunday

✅ **Finalize**: Say:
• "done" or "finalize" to save your plan

What would you like to do?"""

                    yield f"data: {json.dumps({'message': msg, 'type': 'clarification'})}\n\n"
                    yield "event: done\ndata: [DONE]\n\n"

                return StreamingResponse(_clarify_options(), media_type="text/event-stream",
                                        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

        # Handle which days to change
       elif pending_state and pending_state.get("state") == "awaiting_day_selection":
            if text.lower() == "cancel":
                meal_plan = pending_state.get("meal_plan")
                await mem.clear_pending(user_id)
                
                async def _cancel_changes():
                    yield sse_json({
                        "type": "meal_plan_final",
                        "status": "completed",
                        "diet_type": pending_state.get("diet_type"),
                        "meal_plan": meal_plan,
                        "day_names": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
                        "message": "✅ Your meal plan is finalized!"
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
                            yield f"data: {json.dumps({'message': 'Please enter valid numbers (1-10). Example: \"2\" or \"1,3,5\"', 'type': 'error'})}\n\n"
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
                
                # Save to database
                client_id = pending_state.get("client_id")
                # Check if we have custom names
                save_result = save_meal_plan_to_database(client_id, updated_meal_plan, db, replace_all=False)


                
                await mem.clear_pending(user_id)
                
                async def _finalize_all_custom():
                    if save_result['success']:
                        message = f"Your meal plan is finalized with custom names and saved: {', '.join(custom_names)}!"
                    else:
                        message = f"Your meal plan is finalized with names: {', '.join(custom_names)}! (Save failed)"

                    yield sse_json({
                        "type": "meal_plan_final",
                        "status": "completed",
                        "diet_type": pending_state.get("diet_type"),
                        "meal_plan": updated_meal_plan,
                        "day_names": custom_names,
                        "message": message
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
                
                # Save to database
                client_id = pending_state.get("client_id")
                # Check if we have custom names
                save_result = save_meal_plan_to_database(client_id, updated_meal_plan, db, replace_all=False)

                await mem.clear_pending(user_id)
                
                async def _finalize_selected():
                    if save_result['success']:
                        message = f"✅ Your meal plan is finalized with names: {', '.join(custom_names)} and saved!"
                    else:
                        message = f"✅ Your meal plan is finalized with names: {', '.join(custom_names)}! (Save failed)"

                    yield sse_json({
                        "type": "meal_plan_final",
                        "status": "completed",
                        "diet_type": pending_state.get("diet_type"),
                        "meal_plan": updated_meal_plan,
                        "day_names": custom_names,
                        "message": message
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
                client_id = pending_state.get("client_id")
                template_name = "Meal Template (Mon-Sun)"

                # Store template in memory first
                await _store_meal_template(mem, db, client_id, meal_plan, template_name)

                # Save to database directly
                save_result = save_meal_plan_to_database(client_id, meal_plan, db, replace_all=False)
                await mem.clear_pending(user_id)

                async def _finalize_default():
                    if save_result['success']:
                        message = "Your 7-day meal plan is finalized and saved!"
                    else:
                        message = "Your 7-day meal plan is finalized! (Save failed)"

                    yield sse_json({
                        "type": "meal_plan_final",
                        "status": "completed",
                        "diet_type": pending_state.get("diet_type"),
                        "meal_plan": meal_plan,
                        "day_names": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
                        "message": message
                    })
                    yield "event: done\ndata: [DONE]\n\n"

                return StreamingResponse(_finalize_default(), media_type="text/event-stream",
                                        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
            
            else:
                async def _ask_name_change_again_after_allergy():
                    yield f"data: {json.dumps({'message': 'Do you want to customize day names? Please say \"yes\" or \"no\".', 'type': 'clarification'})}\n\n"
                    yield "event: done\ndata: [DONE]\n\n"
                
                return StreamingResponse(_ask_name_change_again_after_allergy(), media_type="text/event-stream",
                                        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

       # Handle case where no pending state exists but user is sending text (FIRST MESSAGE HANDLER)
       else:
           # Automatically start new template creation for any message
           # Treat as first message and show welcome
           profile = _fetch_profile(db, client_id)

           async def _welcome_with_greeting():
               # Then show the welcome message with profile info
               welcome_msg = f"""👋 Hello! I can see your profile:

⚖️ Current Weight: {profile['current_weight']} kg
🎯 Target Weight: {profile['target_weight']} kg
🏆 Goal: {profile['weight_delta_text']}
🍽️ Daily Calorie Target: {profile['target_calories']} kcal

🥗 I'll create a personalized 7-day meal plan just for you!  
First, please tell me your diet preference:  
🌱 Vegetarian  
🍗 Non-Vegetarian  
🥚 Eggetarian  
🌿 Vegan  
🥩 Ketogenic  
🥓 Paleo  
🙏 Jain"""

               await mem.set_pending(user_id, {
                   "state": "awaiting_diet_preference",
                   "client_id": client_id,
                   "profile": profile
               })

               yield f"data: {json.dumps({'message': welcome_msg, 'type': 'welcome'})}\n\n"
               yield "event: done\ndata: [DONE]\n\n"

           return StreamingResponse(_welcome_with_greeting(), media_type="text/event-stream",
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


@router.post("/exit_chat")
async def exit_chat(
   req: UserId,
   mem = Depends(get_mem),
):
   """Called when user exits the chatbot - clears conversation state and chat history"""
   try:
       print(f"User {req.user_id} exiting food template chatbot - clearing all chat data")

       # Clear both chat history and pending state
       await mem.clear_chat_on_exit(req.user_id)

       return {
           "status": "success",
           "user_id": req.user_id,
           "message": "Chat history and state cleared successfully"
       }
   except Exception as e:
       print(f"Error clearing chat for user {req.user_id}: {e}")
       return {"status": "error", "user_id": req.user_id, "error": str(e)}




@router.get("/get_saved_template/{client_id}")
async def get_saved_template(client_id: int, db: Session = Depends(get_db)):
    """Get saved meal plan template for a client"""
    try:
        templates = db.query(ClientDietTemplate).filter(
            ClientDietTemplate.client_id == client_id
        ).order_by(ClientDietTemplate.template_name).all()
        
        if not templates:
            return {"status": "error", "message": "No saved templates found"}
        
        meal_plan = {}
        for template in templates:
            day_key = template.template_name.lower()
            # FIX: No need to json.loads() since diet_data is already a Python object from JSON column
            day_data = template.diet_data
            meal_plan[day_key] = day_data
        
        return {"status": "success", "client_id": client_id, "meal_plan": meal_plan, "total_days": len(templates)}
        
    except Exception as e:
        return {"status": "error", "message": str(e)}

@router.delete("/delete_saved_template/{client_id}")
async def delete_saved_template(client_id: int, db: Session = Depends(get_db)):
    """Delete saved meal plan templates for a client"""
    try:
        count = db.query(ClientDietTemplate).filter(ClientDietTemplate.client_id == client_id).delete()
        db.commit()
        
        if count == 0:
            return {"status": "error", "message": "No templates found to delete"}
        
        return {"status": "success", "message": f"Deleted {count} templates", "deleted_count": count}
        
    except Exception as e:
        db.rollback()
        return {"status": "error", "message": str(e)}


@router.post("/structurize_and_save")
async def structurize_and_save_meal_template(
    request: dict,
    db: Session = Depends(get_db)
):
    """Structurize and save meal template to database"""
    try:
        client_id = request.get("client_id")
        meal_plan = request.get("template")
        
        if not client_id or not meal_plan:
            return {"status": "error", "message": "Missing client_id or template"}

        # Delete existing templates
        db.query(ClientDietTemplate).filter(ClientDietTemplate.client_id == client_id).delete()
        
        # Day mapping
        day_names = {
            'monday': 'Monday', 'tuesday': 'Tuesday', 'wednesday': 'Wednesday',
            'thursday': 'Thursday', 'friday': 'Friday', 'saturday': 'Saturday', 'sunday': 'Sunday'
        }
        
        # Save each day
        templates_created = 0
        for day_key, day_data in meal_plan.items():
            day_name = day_names.get(day_key.lower(), day_key.title())
            
            # FIX: Store raw Python objects in JSON column, not JSON strings
            if isinstance(day_data, str):
                # If it's a JSON string, parse it back to Python object
                try:
                    diet_data_obj = json.loads(day_data)
                except json.JSONDecodeError:
                    # If parsing fails, treat as plain text
                    diet_data_obj = {"error": "Invalid JSON", "raw_data": day_data}
            else:
                # If it's already a Python object, use directly
                diet_data_obj = day_data

            new_template = ClientDietTemplate(
                client_id=client_id,
                template_name=day_name,
                diet_data=diet_data_obj
            )
            db.add(new_template)
            templates_created += 1

        db.commit()

        return {
            "status": "success", 
            "message": f"Saved {templates_created} meal templates",
            "templates_created": templates_created
        }
        
    except Exception as e:
        print(f"ERROR: Structurize and save failed: {e}")
        db.rollback()
        return {"status": "error", "message": str(e)}
