from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import json
import os
import random
import unicodedata
from collections import deque
import time

app = Flask(__name__)
CORS(app)

# Lazy loading model with caching
model = None
tokenizer = None
generator = None
model_loaded_at = 0

def load_model():
    global model, tokenizer, generator, model_loaded_at
    current_time = time.time()
    if model is None or (current_time - model_loaded_at > 3600):
        try:
            model_name = "distilbert-base-multilingual-cased"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            generator = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_new_tokens=50)
            model_loaded_at = current_time
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")
            return None
    return generator

# Load and validate training data
try:
    with open("chatbot_training_data.json", "r", encoding="utf-8") as f:
        training_data = json.load(f)
    if not isinstance(training_data.get("intents"), list):
        raise ValueError("JSON file must contain an 'intents' list")
    print("JSON file loaded successfully.")
except Exception as e:
    print(f"Error loading JSON: {e}")
    training_data = {"intents": [], "products": []}

# Lưu trữ ngữ cảnh
context_history = deque(maxlen=3)

# ==================== INTENT DETECTION ====================
def detect_intent(user_input, context=None):
    user_input_normalized = unicodedata.normalize("NFKC", user_input.lower().strip())
    context_text = " ".join(context) if context else ""
    combined_input = f"{context_text} {user_input_normalized}".strip()
    
    product_keywords = ["có", "tìm", "đâu", "có không"]
    clothing_keywords = ["áo", "váy", "quần", "yếm", "áo khoác"]
    if any(pk in combined_input for pk in product_keywords) and any(ck in combined_input for ck in clothing_keywords):
        intent = next((i for i in training_data.get("intents", []) if i["intent"] == "inquire_product"), None)
        if intent:
            print(f"Matched intent: inquire_product (prioritized)")
            price_max, color, category, pet_type, size, material, location = extract_query_info(user_input)
            response = random.choice(intent["responses"])
            response = response.replace("{clothing_type}", category or "quần áo")
            response = response.replace("{pet_type}", pet_type or "thú cưng")
            response = response.replace("{size}", size or "phù hợp")
            response = response.replace("{color}", color or "đẹp")
            return response

    for intent in training_data.get("intents", []):
        if intent["intent"] == "inquire_product":
            continue
        for pattern in intent.get("examples", []):
            pattern_normalized = unicodedata.normalize("NFKC", pattern.lower().strip())
            pattern_keywords = set(pattern_normalized.split())
            if any(keyword in combined_input for keyword in pattern_keywords) and \
               not (any(pk in combined_input for pk in product_keywords) and any(ck in combined_input for ck in clothing_keywords)):
                print(f"Matched intent: {intent['intent']} with pattern: '{pattern}'")
                price_max, color, category, pet_type, size, material, location = extract_query_info(user_input)
                response = random.choice(intent["responses"])
                response = response.replace("{clothing_type}", category or "quần áo")
                response = response.replace("{pet_type}", pet_type or "thú cưng")
                response = response.replace("{size}", size or "phù hợp")
                response = response.replace("{color}", color or "đẹp")
                response = response.replace("{location}", location or "bạn")
                response = response.replace("{age}", "phù hợp")
                response = response.replace("{material}", material or "chất liệu tốt")
                response = response.replace("{price}", str(price_max or 200000))
                response = response.replace("{season}", "phù hợp")
                return response
    print(f"No intent matched for input: '{user_input_normalized}'")
    return None

# ==================== PRODUCT FILTERING ====================
def recommend_products(price_max=None, color=None, category=None, pet_type=None, size=None, material=None):
    products = training_data.get("products", [])
    results = []
    for product in products:
        match = True
        if price_max is not None and product["price"] > price_max:
            match = False
        if color and product["color"].lower() != color.lower():
            match = False
        if category and product["name"].lower().find(category.lower()) == -1:
            match = False
        if pet_type and product["pet_type"].lower() != pet_type.lower():
            match = False
        if size and product["size"].lower() != size.lower():
            match = False
        if material and product["material"].lower() != material.lower():
            match = False
        if match:
            results.append(product)
    return results

# ==================== INFO EXTRACTION ====================
def extract_query_info(user_input):
    price_max = color = category = pet_type = size = material = location = None
    user_input_lower = unicodedata.normalize("NFKC", user_input.lower().strip())

    if "dưới" in user_input_lower:
        try:
            price_max = int(user_input_lower.split("dưới")[-1].split("k")[0].strip()) * 1000
        except:
            pass

    if "màu" in user_input_lower:
        color_words = user_input_lower.split("màu")[-1].strip().split()
        if color_words:
            color = color_words[0]

    if any(key in user_input_lower for key in ["áo", "váy", "quần", "yếm", "áo khoác"]):
        if "áo" in user_input_lower:
            category = "áo"
        elif "váy" in user_input_lower:
            category = "váy"
        elif "quần" in user_input_lower:
            category = "quần"
        elif "yếm" in user_input_lower:
            category = "yếm"
        elif "áo khoác" in user_input_lower:
            category = "áo khoác"

    if "chó" in user_input_lower:
        pet_type = "chó"
    elif "mèo" in user_input_lower:
        pet_type = "mèo"

    if "size s" in user_input_lower or " s " in user_input_lower:
        size = "S"
    elif "size m" in user_input_lower or " m " in user_input_lower:
        size = "M"
    elif "size l" in user_input_lower or " l " in user_input_lower:
        size = "L"
    elif "size xl" in user_input_lower or " xl " in user_input_lower:
        size = "XL"

    if "cotton" in user_input_lower:
        material = "cotton"
    elif "voan" in user_input_lower:
        material = "voan"
    elif "jeans" in user_input_lower:
        material = "jeans"
    elif "len" in user_input_lower:
        material = "len"
    elif "polyester" in user_input_lower:
        material = "polyester"

    if "hà nội" in user_input_lower:
        location = "Hà Nội"
    elif "tp.hcm" in user_input_lower or "sài gòn" in user_input_lower:
        location = "TP.HCM"
    elif "đà nẵng" in user_input_lower:
        location = "Đà Nẵng"
    elif "cần thơ" in user_input_lower:
        location = "Cần Thơ"

    return price_max, color, category, pet_type, size, material, location

# ==================== RESPONSE GENERATION ====================
def generate_response(user_input):
    global generator
    if generator is None:
        generator = load_model()
        if generator is None:
            return "Xin lỗi, hệ thống đang gặp lỗi. Vui lòng thử lại sau! 😔"

    user_input_normalized = unicodedata.normalize("NFKC", user_input.strip())
    user_input_lower = user_input_normalized.lower()

    # Cập nhật ngữ cảnh
    context_history.append(user_input_normalized)
    context = list(context_history)

    # Handle vague or short inputs
    if len(user_input_normalized) <= 3 or user_input_lower in ["có", "ok", "ừ", "vâng"]:
        return "Dạ, bạn muốn tìm sản phẩm nào cho bé nhà mình nhỉ? Mình có áo, váy, quần cho chó và mèo, giá từ 150k-300k! 😊"

    # Prioritize intent matching with context
    intent_response = detect_intent(user_input_normalized, context)
    if intent_response:
        return intent_response

    # Fallback with keyword-based handling and product-based response
    price_max, color, category, pet_type, size, material, location = extract_query_info(user_input_normalized)

    # Handle product inquiry specifically
    if any(key in user_input_lower for key in ["có", "tìm", "đâu", "có không"]) and \
       any(cat in user_input_lower for cat in ["áo", "váy", "quần", "yếm", "áo khoác"]):
        products = recommend_products(price_max, color, category, pet_type, size, material)
        if products:
            product_list = ", ".join([f"{p['name']} (Giá: {p['price']} VNĐ, Màu: {p['color']})" for p in products])
            return f"Dạ, shop có {product_list}. Bạn muốn mình gửi hình chi tiết hay chốt đơn luôn không? 😊"
        else:
            return f"Xin lỗi bạn nha, hiện tại shop chưa có {category or 'sản phẩm'} {pet_type or ''} {color or ''} {size or ''}. Bạn thử tìm mẫu khác không? 😊"

    if any(keyword in user_input_lower for keyword in ["giặt", "bảo quản", "phơi"]):
        intent = next((i for i in training_data.get("intents", []) if i["intent"] == "ask_care_instructions"), None)
        if intent:
            response = random.choice(intent["responses"])
            response = response.replace("{clothing_type}", category or "quần áo")
            return response or "Nên giặt tay với nước mát, tránh chất tẩy mạnh và phơi nơi thoáng mát nhé! 😊"

    if any(keyword in user_input_lower for keyword in ["giao hàng", "bao lâu", "phí ship", "khi nào tới"]):
        intent = next((i for i in training_data.get("intents", []) if i["intent"] == "ask_delivery_time"), None)
        if intent:
            response = random.choice(intent["responses"])
            response = response.replace("{location}", location or "bạn")
            return response or f"Bạn ở {location or 'khu vực của bạn'} thì hàng sẽ tới trong 1-2 ngày, phí ship 30k, miễn phí cho đơn từ 500k nha! 😊 (Hôm nay là 07/06/2025, 04:42 PM)"

    products = recommend_products(price_max, color, category, pet_type, size, material)
    if products:
        product_list = ", ".join([f"{p['name']} (Giá: {p['price']} VNĐ, Màu: {p['color']})" for p in products])
        return f"Dạ, shop có {product_list}. Bạn muốn mình tư vấn thêm về mẫu nào không? 😊"
    else:
        return "Xin lỗi bạn nha, hiện tại shop chưa có sản phẩm phù hợp. Bạn thử tìm màu hoặc size khác xem, mình sẵn sàng tư vấn thêm! 😊"

# ==================== FLASK ROUTES ====================
@app.route("/")
def serve_index():
    try:
        return render_template("index.html")
    except Exception as e:
        print(f"Error rendering index.html: {e}")
        return "Lỗi khi tải trang, vui lòng thử lại sau! 😔", 500

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message", "").strip()
    if not user_input:
        return jsonify({"response": "Vui lòng nhập tin nhắn! 😊"}), 400
    start_time = time.time()
    response = generate_response(user_input)
    print(f"User input: '{user_input}', Response: '{response}', Processing time: {time.time() - start_time:.2f}s")
    return jsonify({"response": response})

# WSGI application for Gunicorn
application = app