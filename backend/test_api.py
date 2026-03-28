import requests

url = "http://127.0.0.1:5000/predict_text"

data = {"text": "i am walking"}

response = requests.post(url, json=data)

print(response.json())