import PIL
from flask import Flask, render_template, request, send_file, redirect, session
from PIL import Image
from io import BytesIO
import loader
import modelpipeline
from transformers import CLIPTokenizer
import torch
from flask_sqlalchemy import SQLAlchemy
import bcrypt

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
db = SQLAlchemy(app)
app.secret_key = 'secret_key'



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
model_file = "../data/v1-5-pruned-emaonly.ckpt"
models = loader.preload_models_from_standard_weights(model_file, DEVICE)

def generate_image(prompt, uncond_prompt, strength, image_file):
    input_image = Image.open(image_file)

    do_cfg = True  # Example value, modify as needed
    cfg_scale = 8  # Example value, modify as needed
    sampler = "ddpm"  # Example value, modify as needed
    num_inference_steps = 50  # Example value, modify as needed
    seed = 42  # Example value, modify as needed

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

    # Convert the NumPy array to PIL Image
    output_image_pil = Image.fromarray(output_image_np)

    output_buffer = BytesIO()
    output_image_pil.save(output_buffer, format="PNG")
    output_buffer.seek(0)

    return output_buffer

@app.route("/generatefromimage", methods=["POST"])
def generatefromimage():
    prompt = request.form.get("prompt")
    uncond_prompt = request.form.get("uncond_prompt")
    strength = float(request.form.get("strength"))
    image = request.files["image"]

    output_buffer = generate_image(prompt, uncond_prompt, strength, image)
    
    return send_file(output_buffer, mimetype="image/png")

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

@app.route('/', methods=['GET','POST'])
@app.route('/login', methods=['GET','POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        user = User.query.filter_by(email=email).first()
        if user and user.check_password(password):
            session['email'] = user.email
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


    # Check if the prompt is empty
    if not prompt:
        return "Prompt cannot be empty.", 400

    # Generate image based on prompt
    do_cfg = True  # Example value, modify as needed
    cfg_scale = 8  # Example value, modify as needed
    sampler = "ddpm"  # Example value, modify as needed
    num_inference_steps = 50  # Example value, modify as needed
    seed = 42  # Example value, modify as needed

    try:
        output_image_np = modelpipeline.generate(
            prompt=prompt,
            uncond_prompt=uncond_prompt,
            input_image=None,  # No input image provided
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

        # Convert the NumPy array to PIL Image
        output_image_pil = Image.fromarray(output_image_np)

        output_buffer = BytesIO()
        output_image_pil.save(output_buffer, format="PNG")
        output_buffer.seek(0)

        return send_file(output_buffer, mimetype="image/png")

    except Exception as e:
        print("Error generating image:", e)
        return "Failed to generate image.", 500

if __name__ == "__main__":
    app.run(debug=True)
