import requests

url = "http://localhost:5000/chat"
message = "hi"

print("Sending request to:", url)
print("Message:", message)
try:
	response = requests.post(url, json={"message": message}, timeout=5)
	print("Status Code:", response.status_code)
	print("Response:", response.json())
except requests.exceptions.RequestException as e:
	print("Request failed:", e)
