import requests

# Prepare your API endpoint and the file
url = "http://127.0.0.1:8000/generate_response/"

# Path to the image and the text you want to send
image_path = "restaurant.jpg"
text = "What is the Grand Total?"

# Prepare the files to send
files = {
    "image": open(image_path, "rb"),  # The image file
    "text": (None, text)  # The text data
}

# Send the request
response = requests.post(url, files=files)

# Print the response from the API
print(response.json())
