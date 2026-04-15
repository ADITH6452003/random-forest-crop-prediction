from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv
from groq import Groq

# Load env variables
load_dotenv(override=True)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if GROQ_API_KEY:
    groq_client = Groq(api_key=GROQ_API_KEY)
    chat_model = "llama-3.3-70b-versatile"
else:
    groq_client = None

MARKET_API_KEY = os.getenv("MARKET_API_KEY", "")

app = Flask(__name__)

# Load Models
try:
    model = joblib.load("crop_model.pkl")
    encoders = joblib.load("encoders.pkl")
    target_encoder = joblib.load("crop_encoder.pkl")
    print("Models loaded successfully.")
except Exception as e:
    print(f"Error loading models: {e}")
    model = None


@app.route("/")
def home():
    return render_template("welcome.html")

@app.route("/dashboard")
def dashboard():
    return render_template("index.html")

@app.route("/scout")
def scout_page():
    return render_template("scout.html")

@app.route("/scout-data", methods=["POST"])
def scout_data():
    try:
        import requests as req
        data = request.json
        lat = float(data.get("lat"))
        lon = float(data.get("lon"))

        # Open-Meteo
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat,
            "longitude": lon,
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,relativehumidity_2m_max,relativehumidity_2m_min",
            "timezone": "auto",
            "forecast_days": 7
        }
        r = req.get(url, params=params, timeout=6)
        d = r.json().get("daily", {})

        temp_max = d.get("temperature_2m_max", [])
        temp_min = d.get("temperature_2m_min", [])
        hum_max = d.get("relativehumidity_2m_max", [])
        hum_min = d.get("relativehumidity_2m_min", [])

        avg_temp = round((sum(temp_max) + sum(temp_min)) / (len(temp_max) + len(temp_min)), 1) if temp_max else 28.0
        avg_humidity = round((sum(hum_max) + sum(hum_min)) / (len(hum_max) + len(hum_min)), 1) if hum_max else 65.0

        from datetime import datetime
        month = datetime.now().month
        if month in [6, 7, 8, 9]:      season = "kharif"
        elif month in [10, 11, 12, 1, 2]: season = "rabi"
        else:                            season = "Zaid"

        # SoilGrids
        soil_ph = 6.5
        nitrogen = 80.0
        phosphorus = 40.0
        potassium = 40.0
        
        try:
            sg_url = "https://rest.isric.org/soilgrids/v2.0/properties/query"
            sg_params = {
                "lon": lon,
                "lat": lat,
                "property": ["nitrogen", "phh2o"],
                "depth": "0-5cm",
                "value": "mean"
            }
            sg_r = req.get(sg_url, params=sg_params, timeout=5)
            if sg_r.status_code == 200:
                sg_data = sg_r.json()
                layers = sg_data.get("properties", {}).get("layers", [])
                for layer in layers:
                    if layer["name"] == "phh2o":
                        val = layer["depths"][0]["values"]["mean"]
                        if val is not None: soil_ph = round(val / 10, 1)
                    if layer["name"] == "nitrogen":
                        val = layer["depths"][0]["values"]["mean"]
                        if val is not None: 
                            n_val = val / 10 
                            if n_val > 140: n_val = 140
                            if n_val < 0: n_val = 0
                            nitrogen = round(n_val, 1)
                            phosphorus = round(nitrogen * 0.5, 1)
                            potassium = round(nitrogen * 0.5, 1)
        except Exception as sg_e:
            print("SoilGrids API error:", sg_e)
            pass

        return jsonify({
            "success": True,
            "temp": avg_temp,
            "humidity": avg_humidity,
            "season": season,
            "soil_ph": soil_ph,
            "nitrogen": nitrogen,
            "phosphorus": phosphorus,
            "potassium": potassium
        })
    except Exception as e:
        print("Scout data error:", e)
        return jsonify({"success": False, "error": str(e)}), 400



@app.route("/location-data", methods=["POST"])
def location_data():
    try:
        import requests as req
        data = request.json
        lat = float(data.get("lat"))
        lon = float(data.get("lon"))

        # Open-Meteo: free, no API key needed
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat,
            "longitude": lon,
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,relativehumidity_2m_max,relativehumidity_2m_min",
            "timezone": "auto",
            "forecast_days": 7
        }
        r = req.get(url, params=params, timeout=6)
        d = r.json().get("daily", {})

        temp_max = d.get("temperature_2m_max", [])
        temp_min = d.get("temperature_2m_min", [])
        hum_max = d.get("relativehumidity_2m_max", [])
        hum_min = d.get("relativehumidity_2m_min", [])
        precip = d.get("precipitation_sum", [])

        avg_temp = round((sum(temp_max) + sum(temp_min)) / (len(temp_max) + len(temp_min)), 1) if temp_max else None
        avg_humidity = round((sum(hum_max) + sum(hum_min)) / (len(hum_max) + len(hum_min)), 1) if hum_max else None
        avg_precip = round(sum(precip) / len(precip), 1) if precip else 0

        # Estimate season from month
        from datetime import datetime
        month = datetime.now().month
        if month in [6, 7, 8, 9]:      season = "kharif"
        elif month in [10, 11, 12, 1, 2]: season = "rabi"
        else:                            season = "Zaid"

        return jsonify({
            "success": True,
            "temp": avg_temp,
            "humidity": avg_humidity,
            "season": season,
            "avg_precip_mm": avg_precip
        })
    except Exception as e:
        print("Location data error:", e)
        return jsonify({"success": False, "error": str(e)}), 400


@app.route("/predict", methods=["POST"])
def predict():
    if not model:
        return jsonify({"error": "Model not loaded properly."}), 500

    try:
        data = request.json
        print("Received data:", data)
        
        # Prepare input payload for features:
        input_data = pd.DataFrame([{
            "SOIL": data.get("soil"),
            "SEASON": data.get("season"),
            "WATER_SOURCE": data.get("water_source"),
            "SOIL_PH": float(data.get("soil_ph", 0)),
            "TEMP": float(data.get("temp", 0)),
            "RELATIVE_HUMIDITY": float(data.get("humidity", 0)),
            "N": float(data.get("nitrogen", 0)),
            "P": float(data.get("phosphorus", 0)),
            "K": float(data.get("potassium", 0))
        }], columns=['SOIL', 'SEASON', 'WATER_SOURCE', 'SOIL_PH', 'TEMP', 'RELATIVE_HUMIDITY', 'N', 'P', 'K'])

        # Encode categorical
        if "SOIL" in encoders:
            input_data["SOIL"] = encoders["SOIL"].transform(input_data["SOIL"])
        if "SEASON" in encoders:
            input_data["SEASON"] = encoders["SEASON"].transform(input_data["SEASON"])
        if "WATER_SOURCE" in encoders:
            input_data["WATER_SOURCE"] = encoders["WATER_SOURCE"].transform(input_data["WATER_SOURCE"])

        # Create predictions (Top 4)
        probabilities = model.predict_proba(input_data)[0]
        # Get indices of the top 4 probabilities in descending order
        top_4_indices = np.argsort(probabilities)[-4:][::-1]
        
        # Map indices back to crop names
        top_4_crops = target_encoder.inverse_transform(top_4_indices)
        top_4_crops_list = top_4_crops.tolist()

        return jsonify({"predictions": top_4_crops_list, "success": True})

    except Exception as e:
        print("Prediction error:", e)
        return jsonify({"error": str(e), "success": False}), 400


def fetch_market_price(crop_name):
    """Fetch real-time mandi price from data.gov.in if API key is available."""
    if not MARKET_API_KEY:
        return None
    try:
        import requests as req
        url = "https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070"
        params = {
            "api-key": MARKET_API_KEY,
            "format": "json",
            "filters[commodity]": crop_name,
            "limit": 5
        }
        r = req.get(url, params=params, timeout=5)
        records = r.json().get("records", [])
        if records:
            prices = [f"{rec.get('market','')}, {rec.get('state','')}: ₹{rec.get('modal_price','N/A')}/quintal" for rec in records]
            return "\n".join(prices)
    except Exception as e:
        print("Market API error:", e)
    return None


@app.route("/chat", methods=["POST"])
def chat():
    if not groq_client:
        return jsonify({"error": "Groq API key is not configured properly.", "success": False}), 500

    try:
        data = request.json
        user_message = data.get("message", "").strip()
        context = data.get("context", "")
        mode = data.get("mode", "general")  # fertilizer | market | general

        predicted_crops = []
        import json
        if context:
            try:
                ctx_dict = json.loads(context)
                predicted_crops = ctx_dict.get("predicted_crops", [])
            except:
                pass

        if mode == "fertilizer":
            system_prompt = (
                "You are an expert agronomist specializing in fertilizer recommendations. "
                "Given the crop name and soil/nutrient parameters, recommend the best fertilizers "
                "(type, quantity per acre, application timing). Be specific and practical. "
                "Format: Fertilizer name → Dose → When to apply. "
                "If the user doesn't specify a crop in their message, provide the recommendations for the 'predicted_crops' listed in the context."
            )
        elif mode == "market":
            market_data = None
            if user_message and len(user_message.split()) <= 3:
                market_data = fetch_market_price(user_message)
            
            # Fallback to the top predicted crop if user message isn't a specific crop that yields results
            if not market_data and predicted_crops:
                top_crop = predicted_crops[0]
                fetched = fetch_market_price(top_crop)
                if fetched:
                    market_data = f"Market prices for {top_crop}:\n{fetched}"

            if market_data:
                system_prompt = (
                    "You are an agricultural market analyst. "
                    f"Here are the latest mandi prices fetched:\n{market_data}\n"
                    "Analyze these prices, give the price range, best market to sell, and a short trend insight. "
                    "If the user didn't specify a crop, assume they are asking about the fetched crop data provided."
                )
            else:
                system_prompt = (
                    "You are an agricultural market analyst for Indian crops. "
                    "Provide estimated current market price range (₹/quintal) for the crop requested. "
                    "If the user doesn't specify a crop, provide market estimates for the 'predicted_crops' given in the context. "
                    "Clearly mention these are estimates based on historical data, not real-time prices."
                )
        else:
            system_prompt = (
                "You are an expert agronomist AI assistant. "
                "Help the user with crop selection, soil management, season planning, "
                "fertilizer choice, and pest control. Keep answers concise and practical. "
                "If the user asks for advice without specifying a crop, primarily focus your advice on the 'predicted_crops' given in the context."
            )

        if context:
            system_prompt += f"\nFarm parameters context: {context}"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
        response = groq_client.chat.completions.create(
            model=chat_model,
            messages=messages
        )
        return jsonify({"reply": response.choices[0].message.content, "success": True})

    except Exception as e:
        print("Chat error:", e)
        return jsonify({"error": str(e), "success": False}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5001)
