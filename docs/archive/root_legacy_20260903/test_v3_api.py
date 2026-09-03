import requests
import base64
from PIL import Image
import io
import json

def test_predict():
    url = "http://localhost:8000/predict"
    headers = {
        "Content-Type": "application/json",
        "X-API-Key": "v3_test_key"
    }

    # Create dummy image
    img = Image.new('RGB', (224, 224), color='red')
    buf = io.BytesIO()
    img.save(buf, format='JPEG')
    img_b64 = base64.b64encode(buf.getvalue()).decode()

    payload = {
        "image": img_b64,
        "instruction": "Navigate to the object",
        "snap_to_grid": True
    }

    print(f"Sending request to {url}...")
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            print("Response JSON:")
            print(json.dumps(response.json(), indent=2))
        else:
            print(f"Error Response: {response.text}")
    except Exception as e:
        print(f"Request failed: {e}")

if __name__ == "__main__":
    test_predict()
