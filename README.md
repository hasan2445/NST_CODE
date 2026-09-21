Absolutely — paste this entire code into your `README.md`:

````markdown
# 🎨 Neural Style Transfer using AdaIN

A deep learning based **Neural Style Transfer (NST)** application that combines the content of one image with the artistic style of another image using **Adaptive Instance Normalization (AdaIN)**.

The project uses **PyTorch, VGG-19, AdaIN, and a trained decoder**, with a **FastAPI backend** and a modern web interface for uploading images and generating stylized results.

---

## ✨ Features

- 🎨 Neural Style Transfer using AdaIN
- 🧠 Pretrained VGG-19 encoder
- ⚡ FastAPI backend
- 🖼️ Upload content and style images
- 🎚️ Adjustable style strength using Alpha
- 💻 Modern web interface
- 📥 Download generated images
- 🔌 REST API endpoint
- 🏥 CPU/GPU support through PyTorch
- 📊 Health-check API
- 📁 Example style/content images

---

# 🧠 How Neural Style Transfer Works

Neural Style Transfer combines:

- **Content Image** → What should be present in the final image
- **Style Image** → Artistic appearance that should be transferred
- **AdaIN** → Aligns the feature statistics of the content and style
- **Decoder** → Converts the transformed feature representation back into an image

### Basic idea

```text
              CONTENT IMAGE
                    │
                    ▼
             ┌─────────────┐
             │   VGG-19    │
             │   Encoder   │
             └──────┬──────┘
                    │
                    │ Content Features
                    ▼
             ┌─────────────┐
             │    AdaIN    │◄──────────────┐
             └──────┬──────┘               │
                    │                      │
                    │ Target Features      │
                    │                      │
                    ▼                      │
             ┌─────────────┐        ┌──────┴──────┐
             │   Decoder   │        │ Style Image │
             └──────┬──────┘        └──────┬──────┘
                    │                      │
                    │                ┌─────▼─────┐
                    │                │  VGG-19   │
                    │                │  Encoder  │
                    │                └─────┬─────┘
                    │                      │
                    │                Style Features
                    │
                    ▼
            🎨 STYLIZED IMAGE
```

---

# 🔬 AdaIN Architecture

The core of this project is **Adaptive Instance Normalization**.

AdaIN transfers the channel-wise mean and variance of the style features to the content features.

### Formula

For content feature `x` and style feature `y`:

```text
AdaIN(x, y) =
σ(y) * (x - μ(x)) / σ(x) + μ(y)
```

Where:

```text
μ(x) = Mean of content features
σ(x) = Standard deviation of content features

μ(y) = Mean of style features
σ(y) = Standard deviation of style features
```

The result preserves the content structure while adopting the statistical characteristics of the style.

---

# 🏗️ Project Architecture

```text
                         ┌─────────────────────┐
                         │     Web Browser     │
                         │    index.html       │
                         └──────────┬──────────┘
                                    │
                              HTTP Request
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │      FastAPI        │
                         │       app.py        │
                         └──────────┬──────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    │                               │
                    ▼                               ▼
             Content Image                    Style Image
                    │                               │
                    └───────────────┬───────────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │   Image Transform   │
                         │    Resize + Tensor  │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │      VGG-19        │
                         │      Encoder       │
                         └──────────┬──────────┘
                                    │
                             Feature Maps
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │       AdaIN         │
                         │ Adaptive Instance   │
                         │    Normalization    │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │      Decoder       │
                         │   decoder_200.pth   │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │   Generated Image   │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Browser / Download  │
                         └─────────────────────┘
```

---

# 📁 Project Structure

```text
NST_CODE/
│
├── app.py
├── train.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── vgg_normalised.pth
├── decoder_200.pth
│
├── content_data/
│
├── style_data/
│
├── examples/
│
├── templates/
│   └── index.html
│
├── static/
│   └── uploads/
│
└── utils/
    ├── models.py
    └── utils.py
```

### Main files

| File | Description |
|------|-------------|
| `app.py` | FastAPI backend and inference API |
| `train.py` | Decoder training script |
| `utils/models.py` | VGG encoder and decoder architecture |
| `utils/utils.py` | AdaIN and utility functions |
| `templates/index.html` | Frontend interface |
| `requirements.txt` | Python dependencies |
| `vgg_normalised.pth` | Pretrained VGG encoder weights |
| `decoder_200.pth` | Trained decoder weights |
| `examples/` | Example images |
| `static/uploads/` | Generated images |

---

# 🛠️ Technologies Used

### Deep Learning

- Python
- PyTorch
- Torchvision
- VGG-19
- Adaptive Instance Normalization (AdaIN)

### Backend

- FastAPI
- Uvicorn
- Python Multipart
- Jinja2

### Image Processing

- Pillow
- Torchvision Transforms

### Frontend

- HTML5
- CSS3
- JavaScript
- Bootstrap
- Font Awesome

---

# ⚙️ Installation

## 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
cd NST_CODE
```

---

## 2. Create a Virtual Environment

### Windows

```powershell
python -m venv venv
```

Activate it:

```powershell
venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 📦 Model Weights

The model checkpoint files are not included in the repository because they are large.

You need:

```text
vgg_normalised.pth
decoder_200.pth
```

Place them in the project root:

```text
NST_CODE/
│
├── vgg_normalised.pth
├── decoder_200.pth
└── ...
```

> Add the actual model-download links here before publishing the repository.

Example:

```markdown
## Model Weights

Download the pretrained model weights from:

- VGG-19: YOUR_VGG_DOWNLOAD_LINK
- Decoder: YOUR_DECODER_DOWNLOAD_LINK

Place both `.pth` files in the project root directory.
```

---

# 🚀 Running the Application

Start the FastAPI server:

```bash
python app.py
```

The server will run at:

```text
http://127.0.0.1:8000
```

Open the URL in your browser.

---

# 🎨 Using the Application

### Step 1 — Upload Content Image

Select an image whose structure you want to preserve.

Example:

```text
Content Image
     ↓
   🏙️ City
```

### Step 2 — Upload Style Image

Select an artistic image whose visual style you want to transfer.

Example:

```text
Style Image
     ↓
   🎨 Painting
```

### Step 3 — Adjust Alpha

The application provides an Alpha parameter:

```text
0 ─────────────── 1
```

Conceptually:

```text
Alpha = 0
    ↓
Mostly content representation

Alpha = 0.5
    ↓
Balanced combination

Alpha = 1
    ↓
Full AdaIN style transformation
```

### Step 4 — Transfer Style

Click:

```text
TRANSFER STYLE
```

The backend performs:

```text
Content Image
      ↓
VGG Encoder
      ↓
Content Features
      ↓
     AdaIN
      ↑
Style Features
      ↑
Style Image
      ↓
Target Features
      ↓
Decoder
      ↓
Generated Image
```

### Step 5 — Download

The generated image can be downloaded directly from the web interface.

---

# 🔌 API

The application exposes a REST API using FastAPI.

## Health Check

```http
GET /health
```

Example response:

```json
{
    "status": "healthy",
    "device": "cpu",
    "cuda_available": false
}
```

---

## Style Transfer

```http
POST /style-transfer
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `content` | File | Content image |
| `style` | File | Style image |
| `alpha` | Float | Style strength from `0` to `1` |

### Example using Python

```python
import requests

files = {
    "content": open("content.jpg", "rb"),
    "style": open("style.jpg", "rb")
}

data = {
    "alpha": 1.0
}

response = requests.post(
    "http://127.0.0.1:8000/style-transfer",
    files=files,
    data=data
)

with open("result.png", "wb") as f:
    f.write(response.content)
```

---

# 📖 API Documentation

FastAPI automatically generates interactive API documentation.

After starting the server, open:

```text
http://127.0.0.1:8000/docs
```

Alternative documentation:

```text
http://127.0.0.1:8000/redoc
```

---

# 🧠 Model Pipeline

The complete inference pipeline is:

```text
                Content Image
                      │
                      ▼
              ┌───────────────┐
              │   Resize      │
              │   512 × 512   │
              └───────┬───────┘
                      │
                      ▼
              ┌───────────────┐
              │   VGG-19      │
              │   Encoder     │
              └───────┬───────┘
                      │
                      ▼
              Content Feature
                      │
                      │
                      ├───────────────┐
                      │               │
                      ▼               ▼
                 Normalize        Style Stats
                      │               ▲
                      │               │
                      ▼               │
                    AdaIN ◄───────────┘
                      ▲
                      │
               Style Feature
                      ▲
                      │
                Style Image
                      │
                      ▼
              ┌───────────────┐
              │    Decoder    │
              └───────┬───────┘
                      │
                      ▼
               Stylized Image
```

---

# 🎯 Why AdaIN?

Traditional optimization-based Neural Style Transfer optimizes an image directly for every content/style pair.

AdaIN instead performs style transfer through feature statistics.

### Traditional NST

```text
Content + Style
      ↓
Optimization
      ↓
Many Iterations
      ↓
Output
```

### AdaIN

```text
Content
   ↓
VGG Encoder
   ↓
AdaIN ← Style
   ↓
Decoder
   ↓
Output
```

This makes AdaIN suitable for fast feed-forward style transfer.

---

# 🧪 Training Pipeline

The decoder can be trained using content images.

```text
                 Content Image
                       │
                       ▼
                ┌─────────────┐
                │   VGG-19    │
                │   Encoder   │
                └──────┬──────┘
                       │
                       ▼
                Content Feature
                       │
                       ▼
                ┌─────────────┐
                │   Decoder   │
                │   Training  │
                └──────┬──────┘
                       │
                       ▼
              Reconstructed Image
                       │
                       ▼
                   Loss
                       │
                       ▼
                 Backpropagation
                       │
                       ▼
                Updated Decoder
```

The trained decoder is saved as:

```text
decoder_200.pth
```

---

# 🖥️ CPU / GPU

The application automatically detects the available device:

```python
device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)
```

### CPU

```text
Device: cpu
```

### NVIDIA GPU

```text
Device: cuda
```

GPU inference is recommended for faster image generation.

---

# 🔒 Git & Large Files

Large files and generated data are excluded using `.gitignore`.

Ignored files include:

```text
*.pth
*.pt
*.ckpt
content_data/
style_data/
static/uploads/
```

This keeps the Git repository lightweight.

---

# 🐛 Troubleshooting

## `ModuleNotFoundError`

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## `python-multipart` error

Install:

```bash
pip install python-multipart
```

---

## Model file not found

Make sure:

```text
vgg_normalised.pth
decoder_200.pth
```

are located in the project root.

---

## Port already in use

Run the application on another port:

```python
uvicorn.run(
    app,
    host="0.0.0.0",
    port=8001
)
```

Then open:

```text
http://127.0.0.1:8001
```

---

# 📸 Results

Add screenshots of your application here.

Example:

```markdown
## Screenshots

### User Interface

![Neural Style Transfer UI](screenshots/home.png)

### Generated Result

![Generated Result](screenshots/result.png)
```

---

# 🔮 Future Improvements

- [ ] GPU optimization
- [ ] Multiple style images
- [ ] Batch style transfer
- [ ] Higher-resolution generation
- [ ] Drag-and-drop image upload
- [ ] User authentication
- [ ] Cloud deployment
- [ ] Image history
- [ ] Multiple pretrained style-transfer models
- [ ] Video style transfer

---

# 📚 Concepts Used

This project demonstrates concepts from:

- Convolutional Neural Networks
- VGG-19
- Feature Extraction
- Neural Style Transfer
- Adaptive Instance Normalization
- Instance Normalization
- Mean and Standard Deviation
- Encoder-Decoder Architecture
- PyTorch
- REST APIs
- FastAPI
- Image Processing

---

# 👨‍💻 Author

**Mohammad Hasan**

B.Tech Computer Engineering  
Jamia Millia Islamia, New Delhi

---

# ⭐ If You Like This Project

If you found this project useful:

⭐ Star the repository

🍴 Fork the repository

🐛 Open an issue

💡 Suggest improvements

---

## 📜 License

This project is intended for educational and research purposes.
````
