from PIL import Image as PILImage
from flask import Flask, jsonify, render_template, request, send_file, redirect, session, url_for
from PIL import Image
from io import BytesIO
import openai
import loader
import modelpipeline
from transformers import CLIPTokenizer
import torch
from flask_sqlalchemy import SQLAlchemy
import bcrypt
import numpy as np
import os
import tempfile     
from openai import OpenAI
import base64
import requests

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
db = SQLAlchemy(app)
app.secret_key = 'secret_key'
api_key = "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"
openai.api_key = "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"
client = OpenAI()
api_key = "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"
openai.api_key = "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"

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), unique=True)
    password = db.Column(db.String(100),)
    def __init__(self, name, email, password):    
        self.name = name
        self.email = email
        self.password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    def check_password(self, password):
        return bcrypt.checkpw(password.encode('utf-8'), self.password.encode('utf-8'))
with app.app_context():
    db.create_all()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ALLOW_CUDA = True
ALLOW_MPS = False
if torch.cuda.is_available() and ALLOW_CUDA:
    DEVICE = "cuda"
elif (torch.backends.mps.is_built() or torch.backends.mps.is_available()) and ALLOW_MPS:
    DEVICE = "mps"
print(f"Using device: {DEVICE}")

tokenizer = CLIPTokenizer("../data/vocab.json", merges_file="../data/merges.txt")
model_file = "../data/test.ckpt"
models = loader.preload_models_from_standard_weights(model_file, DEVICE)

def generate_image(prompt, uncond_prompt, strength, image_file):
    input_image = Image.open(image_file)
    do_cfg = True  
    cfg_scale = 8  
    sampler = "ddpm" 
    num_inference_steps = 50  
    seed = 42  
    output_image_np = modelpipeline.generate(
        prompt=prompt,
        uncond_prompt=uncond_prompt,
        input_image=input_image,
        strength=strength,
        do_cfg=do_cfg,
        cfg_scale=cfg_scale,
        sampler_name=sampler,
        n_inference_steps=num_inference_steps,
        seed=seed,
        models=models,
        device=DEVICE,
        idle_device="cpu",
        tokenizer=tokenizer,
    )
    output_image_pil = Image.fromarray(output_image_np)
    output_buffer = BytesIO()
    output_image_pil.save(output_buffer, format="PNG")
    output_buffer.seek(0)
    return output_buffer

output_folder = "./generatedimages"  
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

output_folder = "./generatedimages"  
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

@app.route("/openaiimage", methods=['GET', 'POST'])
def openaiimage():
    image_url = None
    if request.method == 'POST':
        text = request.form['text']
        response = openai.images.generate(
            prompt=text,
            size="1024x1024",
            n=1
        )
        image_url = response.data[0].url
        image_response = requests.get(image_url)
        if image_response.status_code == 200:
            output_filename = "generated_image.png"  
            output_path = os.path.join(output_folder, output_filename)
            with open(output_path, "wb") as f:
                f.write(image_response.content)
            return render_template('openai.html', image_url=image_url)
    return render_template('openai.html', image_url=image_url)

@app.route("/generatefromimage", methods=["POST"])
def generatefromimage():
    prompt = request.form.get("prompt")
    uncond_prompt = request.form.get("uncond_prompt")
    strength = float(request.form.get("strength"))
    image = request.files["image"]
    output_buffer = generate_image(prompt, uncond_prompt, strength, image)
    output_folder = "./generatedimages" 
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    output_filename = "generated_image.png"  
    output_path = os.path.join(output_folder, output_filename)
    with open(output_path, "wb") as f:
        f.write(output_buffer.getvalue())
    return send_file(output_path, mimetype="image/png")

@app.route('/index')
def index():
    if session['name']:
        user = User.query.filter_by(email=session['email']).first()
        return render_template('index.html', user=user)
    return render_template('/login.html')

@app.route('/imagetoimage')
def imagetoimage():
    if session['name']:
        user = User.query.filter_by(email=session['email']).first()
        return render_template('imagetoimage.html', user=user)
    return render_template('/login.html')

@app.route('/texttoimage')
def texttoimage():
    if session['name']:
        user = User.query.filter_by(email=session['email']).first()
        return render_template('texttoimage.html', user=user)
    return render_template('/login.html')

@app.route('/openai')
def openai_page():
    if session['name']:
        user = User.query.filter_by(email=session['email']).first()
        return render_template('openai.html', user=user)
    return render_template('/login.html')

@app.route('/', methods=['GET','POST'])
@app.route('/login', methods=['GET','POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        user = User.query.filter_by(email=email).first()
        if user and user.check_password(password):
            session['email'] = user.email
            session['name'] = user.name  
            return redirect('/index')
        else:
            return render_template('login.html', error='Invalid email or password')   
    return render_template('login.html')
    
@app.route('/logout')
def logout():
    session.pop('email', None)
    return redirect('/login')

@app.route('/register', methods=['GET','POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']  
        email = request.form['email']
        password = request.form['password'] 
        new_user = User(name=name , email=email, password=password)
        db.session.add(new_user)
        db.session.commit()
        return redirect('login')
    return render_template('register.html')

@app.route("/generate", methods=["POST"])
def generate():
    prompt = request.form.get("prompt")
    uncond_prompt = request.form.get("uncond_prompt")
    if not prompt:
        return "Prompt cannot be empty.", 400
    do_cfg = True 
    cfg_scale = 8  
    sampler = "ddpm"  
    num_inference_steps = 50  
    seed = 42  
    try:
        output_image_np = modelpipeline.generate(
            prompt=prompt,
            uncond_prompt=uncond_prompt,
            input_image=None,  
            do_cfg=do_cfg,
            cfg_scale=cfg_scale,
            sampler_name=sampler,
            n_inference_steps=num_inference_steps,
            seed=seed,
            models=models,
            device=DEVICE,
            idle_device="cpu",
            tokenizer=tokenizer,
        )
        output_image_pil = Image.fromarray(output_image_np)
        output_filename = "generated_image.png"  
        output_path = os.path.join(output_folder, output_filename)
        output_image_pil.save(output_path)
        return send_file(output_path, mimetype="image/png")
    except Exception as e:
        print("Error generating image:", e)
        return "Failed to generate image.", 500

@app.route('/imageeditor')
def image_editor():
    return render_template('imageeditor.html')

@app.route('/caption')
def caption():
    return render_template('caption.html')

@app.route('/process', methods=['POST'])
def process():
    text1 = request.form['text1']
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    image_path = "../sd/generatedimages/generated_image.png"
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
    return render_template('imageeditor.html', caption1=caption1)

@app.route("/showimage")
def show_image():
    image_path = "./generatedimages/generated_image.png"
    return send_file(image_path, mimetype="image/png")

@app.route('/crop')
def crop():
    return render_template('crop.html')

@app.route('/help')
def help():
    return render_template('help.html')

if __name__ == "__main__":
    app.run(debug=True)
