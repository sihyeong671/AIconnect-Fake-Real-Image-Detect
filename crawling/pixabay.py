import requests

api_key = "37919976-d3053b3c5f0b06431cd8f7565"
url = f"https://pixabay.com/api/?key={api_key}&image_type=photo"

r = requests.get(url)
json_data = r.json()
print(len(json_data))

# for img in json_data["hits"]:
#   name = img["id"]
#   img_url = img["largeImageURL"]
#   r = requests.get(img_url, stream=True)
#   with open(str(name) + ".jpg", "wb") as f:
#     f.write(r.content)
    