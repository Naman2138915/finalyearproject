@app.route('/process', methods=['POST'])
def process():
    # Extract text inputs from the form
    text1 = request.form['text1']
    
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    image_path = "../images/dog.jpg"
    base64_image = encode_image(image_path)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    payload = {
        "model": "gpt-4-vision-preview",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": text1
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Give me Captions to post it on social media?"
                    }
                ]
            }
        ],
        "max_tokens": 300
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

    if response.status_code == 200:
        caption1 = response.json()["choices"][0]["message"]["content"]
    else:
        caption1 = "Error: Unable to get caption"

    # Render a new page with the captions
    return render_template('result.html', caption1=caption1)
















@app.route('/process', methods=['POST'])
def process():
    # Extract text inputs from the form
    text1 = request.form['text1']
    
    
    # Make API call with the extracted texts
    response = client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text1},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",
                        },
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Give me Captions to post it on social media?"},
                ],
            }
        ],
        max_tokens=300,
    )

    # Extract and format the responses
    caption1 = response.choices[0].message.content if response.choices else "No caption found"


    # Render a new page with the captions
    return render_template('result.html', caption1=caption1)