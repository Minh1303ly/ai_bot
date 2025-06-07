from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import json
import os
import random
import unicodedata
from collections import deque
import time
import gzip
import torch

app = Flask(__name__)
CORS(app)

# Lazy loading model with caching
model = None
tokenizer = None
classifier = None
model_loaded_at = 0
training_data = None

def load_model():
    global model, tokenizer, classifier, model_loaded_at
    current_time = time.time()
    if model is None or (current_time - model_loaded_at > 3600):
        try:
            model_name = "vinai/phobert-base"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(training_data.get("intents", [])))
            model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
            classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, return_all_scores=True)
            model_loaded_at = current_time
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")
            return None
    return classifier

# Load training data incrementally
def load_training_data():
    global training_data
    try:
        json_file = "chatbot_training_data.json.gz" if os.path.exists("chatbot_training_data.json.gz") else "chatbot_training_data.json"
        training_data = {"intents": [], "products": []}
        if json_file.endswith(".gz"):
            with gzip.open(json_file, "rt", encoding="utf-8") as f:
                data = json.load(f)
                training_data["intents"] = data.get("intents", [])[:100]  # Limit intents
                training_data["products"] = data.get("products", [])[:1000]  # Limit products
        else:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                training_data["intents"] = data.get("intents", [])[:100]
                training_data["products"] = data.get("products", [])[:1000]
        if not isinstance(training_data.get("intents"), list):
            raise ValueError("Intents must be a list")
        print("JSON file loaded successfully.")
    except Exception as e:
        print(f"Error loading JSON: {e}")
        training_data = {"intents": [], "products": []}

# Load training data at startup
load_training_data()

# Context history
context_history = deque(maxlen=3)

# Intent mapping
intent_map = {i: intent["intent"] for i, intent in enumerate(training_data.get("intents", []))}

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
            print(f"Matched intent: inquire_product")
            price_max, color, category, pet_type, size, material, location = extract_query_info(user_input)
            response = random.choice(intent["responses"])
            response = response.replace("{clothing_type}", category or "quần áo")
            response = response.replace("{pet_type}", pet_type or "thú cưng")
            response = response.replace("{size}", size or "phù hợp")
            response = response.replace("{color}", color or "đẹp")
            return response

    global classifier
    if classifier is None:
        classifier = load_model()
        if classifier is None:
            print("Model not loaded, using rule-based detection")
            return None
    
    try:
        results = classifier(combined_input)[0]
        top_intent = max(results, key=lambda x: x["score"])
        intent_idx = int(top_intent["label"].replace("LABEL_", ""))
        intent_name = intent_map.get(intent_idx)
        intent = next((i for i in training_data.get("intents", []) if i["intent"] == intent_name), None)
        if intent and top_intent["score"] > 0.7:
            print(f"Matched intent: {intent['intent']} (score: {top_intent['score']})")
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
    except Exception as e:
        print(f"Error in intent classification: {e}")
    
    for intent in training_data.get("intents", []):
        if intent["intent"] == "inquire_product":
            continue
        for pattern in intent.get("examples", []):
            pattern_normalized = unicodedata.normalize("NFKC", pattern.lower().strip())
            pattern_keywords = set(pattern_normalized.split())
            if any(keyword in combined_input for keyword in pattern_keywords) and \
               not (any(pk in combined_input for pk in product_keywords) and any(ck in combined_input for ck in clothing_keywords)):
                print(f"Matched intent: {intent['intent']} (pattern: '{pattern}')")
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
    for product in products[:10]:  # Limit to 10 products
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
    return results[:2]  # Limit to 2 results

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
    global classifier
    if classifier is None:
        classifier = load_model()
        if classifier is None:
            return "Xin lỗi, hệ thống đang gặp lỗi. Vui lòng thử lại sau! 😔"[:1000]

    user_input_normalized = unicodedata.normalize("NFKC", user_input.strip())
    user_input_lower = user_input_normalized.lower()

    context_history.append(user_input_normalized)
    context = list(context_history)

    if len(user_input_normalized) <= 3 or user_input_lower in ["có", "ok", "ừ", "vâng"]:
        return "Dạ, bạn muốn tìm sản phẩm nào cho bé nhà mình nhỉ? Mình có áo, váy, quần cho chó và mèo, giá từ 150k-300k! 😊"[:1000]

    intent_response = detect_intent(user_input_normalized, context)
    if intent_response:
        return intent_response[:300]

    price_max, color, category, pet_type, size, material, location = extract_query_info(user_input_normalized)

    if any(key in user_input_lower for key in ["có", "tìm", "đâu", "có không"]) and \
       any(cat in user_input_lower for cat in ["áo", "váy", "quần", "yếm", "áo khoác"]):
        products = recommend_products(price_max, color, category, pet_type, size, material)
        if products:
            product_list = ", ".join([f"{p['name']} ({p['price']} VNĐ)" for p in products[:2]])
            return f"Shop có {product_list}. Xem thêm không? 😊"[:300]
        else:
            return f"Xin lỗi bạn nha, hiện tại shop chưa có {category or 'sản phẩm'} {pet_type or ''} {color or ''} {size or ''}. Bạn thử tìm mẫu khác không? 😊"[:1000]

    if any(keyword in user_input_lower for keyword in ["giặt", "bảo quản", "phơi"]):
        intent = next((i for i in training_data.get("intents", []) if i["intent"] == "ask_care_instructions"), None)
        if intent:
            response = random.choice(intent["responses"])
            response = response.replace("{clothing_type}", category or "quần áo")
            return response[:300] or "Giặt tay nước mát, phơi thoáng! 😊"[:300]

    if any(keyword in user_input_lower for keyword in ["giao hàng", "bao lâu", "phí ship", "khi nào tới"]):
        intent = next((i for i in training_data.get("intents", []) if i["intent"] == "ask_delivery_time"), None)
        if intent:
            response = random.choice(intent["responses"])
            response = response.replace("{location}", location or "bạn")
            return response[:300] or f"Giao {location or 'bạn'} 1-2 ngày, ship 30k, miễn phí đơn 500k! 😊 (08/06/2025, 12:58 AM)"[:300]

    products = recommend_products(price_max, color, category, pet_type, size, material)
    if products:
        product_list = ", ".join([f"{p['name']} ({p['price']} VNĐ)" for p in products[:2]])
        return f"Shop có {product_list}. Tư vấn thêm? 😊"[:300]
    else:
        return "Chưa có sản phẩm phù hợp. Thử màu/size khác? 😊"[:300]

# ==================== FLASK ROUTES ====================
@app.route("/")
def serve_index():
    try:
        return render_template("index.html")
    except Exception as e:
        print(f"Error rendering index.html: {e}")
        return "Lỗi tải trang, thử lại sau! 😔", 500

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message", "").strip()
    if not user_input:
        return jsonify({"response": "Nhập tin nhắn nhé! 😊"}), 400
    if len(user_input) > 300:
        return jsonify({"response": "Tin nhắn quá dài, ngắn gọn thôi! 😊"}), 400
    start_time = time.time()
    response = generate_response(user_input)
    print(f"Input: '{user_input}', Response: '{response}', Time: {time.time() - start_time:.2f}s")
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))