#food_log.py - AI-Driven Version
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

# Initialize clients
openai_client = OpenAI(api_key=OPENAI_API_KEY)
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-1.5-flash")

@router.get("/healthz")
async def healthz():
    return {"ok": True, "env": APP_ENV, "tz": TZNAME}

def normalize_unit(unit):
    """Basic unit normalization"""
    if not unit:
        return 'pieces'
    
    unit_lower = unit.lower().strip()
    unit_map = {
        'g': 'grams', 'gram': 'grams', 'gms': 'grams',
        'kg': 'kg', 'kilogram': 'kg', 'kilograms': 'kg',
        'piece': 'pieces', 'pcs': 'pieces', 'pc': 'pieces',
        'slice': 'slices', 'slc': 'slices',
        'plate': 'plates', 'bowl': 'bowls',
        'cup': 'cups', 'glass': 'glasses',
        'ml': 'ml', 'milliliter': 'ml', 'milliliters': 'ml',
        'liter': 'liters', 'litre': 'liters', 'l': 'liters'
    }
    return unit_map.get(unit_lower, unit_lower)

def get_ai_unit_and_question(food_name):
    """Use AI to determine the most appropriate unit and generate question"""
    try:
        prompt = f"""
        For the food item "{food_name}", determine:
        1. The most appropriate primary unit for measuring this food
        2. 1-2 alternative units that users might commonly use
        3. Generate a natural question asking for quantity
        
        Consider:
        - How is this food typically served/consumed?
        - What's the most natural way people measure it?
        - What units would be most user-friendly?
        
        Examples:
        - dosa: typically counted as individual pieces
        - rice: typically served in plates or bowls 
        - orange juice: typically measured in ml or glasses
        - chicken: typically measured by weight (grams) or pieces
        
        Return ONLY valid JSON:
        {{
            "primary_unit": "pieces",
            "alternative_units": ["grams"],
            "question": "How many pieces or grams of dosa did you have?"
        }}
        """
        
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a food measurement expert. Always return practical, user-friendly units based on how food is commonly consumed."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=200,
            temperature=0.1
        )
        
        result = response.choices[0].message.content.strip()
        result = re.sub(r"^```json\s*", "", result)
        result = re.sub(r"\s*```$", "", result)
        
        data = json.loads(result)
        return data.get('primary_unit', 'pieces'), data.get('alternative_units', []), data.get('question', f"How much {food_name} did you have?")
        
    except Exception as e:
        print(f"AI unit determination failed: {e}")
        # Minimal fallback
        return 'pieces', ['grams'], f"How many pieces of {food_name} did you have?"

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
        "unit_hint": f"How much {food.get('name', '')} did you have?",
        "ask": food.get('quantity') is None,
        "qty_from": "provided" if food.get('quantity') is not None else "ask"
    }

def extract_food_info_using_ai(text: str):
    """AI-driven food extraction - fully AI-powered unit selection"""
    
    reasoning_prompt = f"""
    Analyze this text and extract food information: "{text}"

    TASK:
    1. Identify ALL foods, beverages, drinks, snacks mentioned
    2. Handle misspellings and variations (avacado→avocado, sugarcanjuice→sugar cane juice)
    3. For each food, determine the MOST APPROPRIATE unit based on how it's typically consumed
    4. Extract quantities if provided, otherwise set to null
    5. Calculate nutrition if quantity is provided

    UNIT SELECTION PRINCIPLES:
    - Think about how people typically measure and consume each food
    - Consider cultural context and common serving practices
    - Choose units that are practical and user-friendly
    - Examples:
      * dosa, idli, chapati: counted as individual pieces
      * rice, pasta: served in plates or bowls  
      * juice, milk: measured in ml or glasses
      * chicken, meat: measured by weight (grams) or pieces
      * fruits: typically counted as pieces

    QUANTITY HANDLING:
    - If specific quantity provided, use it exactly
    - If no quantity, set to null
    - For nutrition: use conversions (1 plate=300g, 1 cup=200ml/g, 1 bowl=200g, 1 glass=200ml)

    SMART INTERPRETATION:
    - When only number given (no unit): choose most logical unit for that food and quantity
    - Small numbers (1-5): likely portions/pieces for most foods
    - Large numbers (>100): likely grams/ml for measurable quantities

    Return valid JSON array:
    [
        {{
            "name": "food_name",
            "quantity": number_or_null,
            "unit": "most_appropriate_unit",
            "calories": number_or_null,
            "protein": number_or_null,
            "carbs": number_or_null,
            "fat": number_or_null,
            "fiber": number_or_null,
            "sugar": number_or_null
        }}
    ]
    
    Be intelligent about unit selection - think about what makes most sense for each specific food!
    """

    # Try OpenAI first
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a nutrition expert with deep knowledge of how different foods are typically measured and consumed. Make intelligent decisions about appropriate units."},
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
        
        # Normalize units
        for food in foods:
            if food.get('unit'):
                food['unit'] = normalize_unit(food['unit'])
        
        print(f"DEBUG: OpenAI extracted: {[(f.get('name'), f.get('quantity'), f.get('unit')) for f in foods]}")
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
                
                # Normalize units
                for food in foods:
                    if food.get('unit'):
                        food['unit'] = normalize_unit(food['unit'])
                
                print(f"DEBUG: Gemini extracted: {[(f.get('name'), f.get('quantity'), f.get('unit')) for f in foods]}")
                return {"foods": foods}
                
        except Exception as gemini_error:
            print(f"Gemini extraction error: {gemini_error}")
        
        # Simple fallback parsing
        print("Both AI models failed, using simple fallback...")
        return parse_simple_fallback(text)

def parse_simple_fallback(text):
    """Minimal fallback when AI fails - just extract food name and use AI for unit"""
    text_lower = text.lower().strip()
    
    # Basic food name extraction
    food_name = text_lower
    quantity = None
    
    # Try to extract a number if present
    quantity_match = re.search(r'(\d+(?:\.\d+)?)', text_lower)
    if quantity_match:
        quantity = float(quantity_match.group(1))
        food_name = re.sub(r'\d+(?:\.\d+)?', '', text_lower).strip()
    
    if food_name:
        # Use AI to determine unit
        unit, _, _ = get_ai_unit_and_question(food_name)
        
        return {"foods": [{
            "name": food_name,
            "quantity": quantity,
            "unit": unit,
            "calories": None,
            "protein": None,
            "carbs": None,
            "fat": None,
            "fiber": None,
            "sugar": None
        }]}
    
    return {"foods": []}

def calculate_nutrition_using_ai(food_name, quantity, unit):
    """Use AI to calculate nutrition"""
    try:
        prompt = f"""
        Calculate nutrition for: {quantity} {unit} of {food_name}
        
        Use these conversions for portion sizes:
        - 1 plate = 300 grams
        - 1 cup = 200 ml (for liquids) or 200 grams (for solids)  
        - 1 bowl = 200 grams
        - 1 glass = 200 ml
        - 1 piece = estimate typical piece size for the food
        
        Return realistic nutrition values (not zeros).
        
        Return ONLY valid JSON:
        {{
            "calories": 165,
            "protein": 31,
            "carbs": 0,
            "fat": 3.6,
            "fiber": 0,
            "sugar": 0
        }}
        """
        
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a nutrition expert. Always return realistic nutrition values based on standard food databases."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300,
            temperature=0
        )
        
        result = response.choices[0].message.content.strip()
        result = re.sub(r"^```json\s*", "", result)
        result = re.sub(r"\s*```$", "", result)
        
        nutrition = json.loads(result)
        
        # Validate nutrition values
        if all(v == 0 for v in nutrition.values()):
            raise Exception("All nutrition values are zero")
        
        return nutrition
        
    except Exception as e:
        print(f"AI nutrition calculation failed: {e}")
        
        # Simple fallback nutrition
        return {
            "calories": 100,
            "protein": 5,
            "carbs": 15,
            "fat": 3,
            "fiber": 2,
            "sugar": 2
        }

def is_food_related(text):
    """Simple food detection - let AI handle the complexity"""
    if not text:
        return False
    
    text_lower = text.lower().strip()
    
    # Very basic non-food patterns
    obvious_non_food = {'hello', 'hi', 'hey', 'thanks', 'help', 'what', 'how', 'when', 'where', 'bye', 'goodbye', 'yes', 'no'}
    
    if text_lower in obvious_non_food:
        return False
    
    # Short inputs are likely food names, let AI decide
    return len(text_lower.split()) <= 6

def create_food_log_response_with_message(logged_foods):
    """Create food log response with summary"""
    items = [convert_food_to_item_format(food) for food in logged_foods]
    
    # Calculate totals
    total_nutrition = {'calories': 0, 'protein': 0, 'carbs': 0, 'fat': 0, 'fiber': 0, 'sugar': 0}
    
    food_summaries = []
    for food in logged_foods:
        quantity = food.get('quantity', 0)
        unit = food.get('unit', 'pieces')
        name = food.get('name', '')
        
        food_summaries.append(f"{quantity} {unit} of {name}")
        
        for nutrient in total_nutrition:
            total_nutrition[nutrient] += food.get(nutrient, 0)
    
    # Round totals
    for nutrient in total_nutrition:
        total_nutrition[nutrient] = round(total_nutrition[nutrient], 1)
    
    # Create message
    if len(food_summaries) == 1:
        message = f"✅ Logged {food_summaries[0]}! "
    elif len(food_summaries) == 2:
        message = f"✅ Logged {food_summaries[0]} and {food_summaries[1]}! "
    else:
        message = f"✅ Logged {', '.join(food_summaries[:-1])} and {food_summaries[-1]}! "
    
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
        
        # Check for pending state
        pending_state = await mem.get_pending(user_id)
        print(f"DEBUG: Current pending state: {pending_state}")
        
        # Handle navigation confirmation
        if pending_state and pending_state.get("state") == "awaiting_nav_confirm":
            if is_yes(text):
                await mem.clear_pending(user_id)
                async def _nav_yes():
                    yield sse_json({"type":"nav","is_navigation": True,
                                    "prompt":"Thanks for your confirmation. Redirecting to today's diet logs"})
                    yield "event: done\ndata: [DONE]\n\n"
                return StreamingResponse(_nav_yes(), media_type="text/event-stream",
                                        headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})
            
            elif is_no(text):
                await mem.clear_pending(user_id)
                async def _nav_no():
                    yield sse_json({"type":"nav","is_navigation": False,
                                    "prompt":"Thanks for your response. You can continue chatting here."})
                    yield "event: done\ndata: [DONE]\n\n"
                return StreamingResponse(_nav_no(), media_type="text/event-stream",
                                        headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})
            
            elif is_food_related(text):
                await mem.clear_pending(user_id)
                # Continue to food processing
            
            else:
                async def _nav_clar():
                    yield f"data: {json.dumps({'message': 'Do you want to go to your diet log? Please say Yes or No.', 'type': 'nav_confirm'})}\n\n"
                    yield "event: done\ndata: [DONE]\n\n"
                return StreamingResponse(_nav_clar(), media_type="text/event-stream",
                                     headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})

        # Handle pending food confirmation
        elif pending_state and pending_state.get("state") == "awaiting_pending_confirm":
            pending_foods = pending_state.get("pending_foods", [])
            logged_foods = pending_state.get("logged_foods", [])
            
            if is_yes(text):
                first_pending = pending_foods[0] if pending_foods else None
                if first_pending:
                    # Use AI to generate the question
                    _, _, ask_message = get_ai_unit_and_question(first_pending['name'])
                                        
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
                await mem.clear_pending(user_id)
                
                if logged_foods:
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
                await mem.clear_pending(user_id)
                # Continue to food processing
            
            else:
                pending_names = [food['name'] for food in pending_foods]
                ask_message = f"Do you want to log these foods: {', '.join(pending_names)}? Please say Yes or No."
                
                async def _ask_confirm_again():
                    yield f"data: {json.dumps({'message': ask_message, 'type': 'confirm_pending'})}\n\n"
                    yield "event: done\ndata: [DONE]\n\n"
                
                return StreamingResponse(_ask_confirm_again(), media_type="text/event-stream",
                                        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

        # Handle quantity input
        elif pending_state and pending_state.get("state") == "awaiting_quantity":
            try:
                foods = pending_state.get("foods", [])
                current_index = pending_state.get("current_food_index", 0)
                current_food = foods[current_index] if current_index < len(foods) else None
                logged_foods = pending_state.get("logged_foods", [])
                
                if not current_food:
                    await mem.clear_pending(user_id)
                    raise Exception("No current food found")
                
                # Check if user entered new food
                if is_food_related(text) and not is_quantity_input(text):
                    # Process new food input
                    new_food_info = extract_food_info_using_ai(text)
                    new_foods = new_food_info.get("foods", [])
                    
                    if new_foods:
                        foods_with_quantity = [f for f in new_foods if f.get("quantity") is not None]
                        foods_without_quantity = [f for f in new_foods if f.get("quantity") is None]
                        
                        logged_foods.extend(foods_with_quantity)
                        all_pending_foods = foods + foods_without_quantity
                        
                        if foods_with_quantity:
                            food_list = [f"{food['quantity']} {food['unit']} of {food['name']}" for food in foods_with_quantity]
                            logged_summary = f"Logged: {', '.join(food_list)}"
                            
                            if all_pending_foods:
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
                                await mem.clear_pending(user_id)
                                
                                async def _final_log():
                                    response_data = create_food_log_response_with_message(logged_foods)
                                    yield sse_json(response_data)
                                    yield "event: ping\ndata: {}\n\n"
                                    yield "event: done\ndata: [DONE]\n\n"
                                    await mem.set_pending(user_id, {"state":"awaiting_nav_confirm"})

                                return StreamingResponse(_final_log(), media_type="text/event-stream",
                                                        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
                
                # Process as quantity input
                quantity_value, parsed_unit = parse_quantity_and_unit(text, current_food['name'])
                
                if quantity_value is not None:
                    final_unit = parsed_unit if parsed_unit else current_food.get("unit", "pieces")
                    
                    # Update food
                    foods[current_index]["quantity"] = quantity_value
                    foods[current_index]["unit"] = final_unit
                    
                    # Calculate nutrition
                    nutrition = calculate_nutrition_using_ai(
                        current_food['name'], quantity_value, final_unit
                    )
                    foods[current_index].update(nutrition)
                    
                    # Move to logged
                    logged_foods.append(foods[current_index])
                    
                    # Check for next food
                    next_food_index = -1
                    for i in range(current_index + 1, len(foods)):
                        if foods[i].get("quantity") is None:
                            next_food_index = i
                            break
                    
                    if next_food_index != -1:
                        next_food = foods[next_food_index]
                        # Use AI to generate question
                        _, _, ask_message = get_ai_unit_and_question(next_food['name'])
                        
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
                        # All foods processed
                        await mem.clear_pending(user_id)
                        
                        async def _logged_then_nav():
                            response_data = create_food_log_response_with_message(logged_foods)
                            
                            if 'message' in response_data:
                                yield f"data: {json.dumps({'message': response_data['message'], 'type': 'response'})}\n\n"
                            
                            yield sse_json(response_data)
                            yield "event: ping\ndata: {}\n\n"
                            yield "event: done\ndata: [DONE]\n\n"
                            await mem.set_pending(user_id, {"state":"awaiting_nav_confirm"})

                        return StreamingResponse(_logged_then_nav(), media_type="text/event-stream",
                                                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
                else:
                    # Ask again with AI-generated question
                    _, _, ask_message = get_ai_unit_and_question(current_food['name'])
                    
                    async def _ask_again():
                        yield f"data: {json.dumps({'message': f'Please enter a number. {ask_message}', 'type': 'ask_quantity', 'food': current_food['name']})}\n\n"
                        yield "event: done\ndata: [DONE]\n\n"
                    
                    return StreamingResponse(_ask_again(), media_type="text/event-stream",
                                            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
                                            
            except Exception as e:
                print(f"Error processing quantity: {e}")
                await mem.clear_pending(user_id)

        # Normal food processing
        if not is_food_related(text):
            async def _not_food():
                response = "I'm here to help you log food. What did you eat or drink?"
                yield f"data: {json.dumps({'message': response, 'type': 'response'})}\n\n"
                yield "event: done\ndata: [DONE]\n\n"
            
            return StreamingResponse(_not_food(), media_type="text/event-stream",
                                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

        # Extract food information using AI
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
                food_list = [f"{food['quantity']} {food['unit']} of {food['name']}" for food in logged_foods]
                
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
                # Use AI to generate question
                _, _, ask_message = get_ai_unit_and_question(first_food['name'])
                
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
    
    # Check for numbers with or without units
    quantity_patterns = [
        r'^\d+(?:\.\d+)?,'  # Just numbers: 2, 1.5, 100
        r'^\d+(?:\.\d+)?\s*(g|grams?|kg|plates?|bowls?|pieces?|ml|glasses?|cups?|slices?)'  # Numbers with units
    ]
    
    return any(re.match(pattern, text_lower) for pattern in quantity_patterns)
    
def parse_quantity_and_unit(text, food_name):
    """Parse quantity and unit from text using AI assistance"""
    text_lower = text.lower().strip()
    
    # Basic patterns for extraction
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
            else:
                # If no unit provided, use AI to determine appropriate unit
                unit, _, _ = get_ai_unit_and_question(food_name)
            
            print(f"DEBUG: Parsed '{text}' as {quantity} {unit} for {food_name}")
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

@router.post("/clear_pending")
async def clear_pending_state(
    req: Userid,
    mem = Depends(get_mem),
):
    await mem.clear_pending(req.user_id)
    return {"status": "cleared", "user_id": req.user_id}
