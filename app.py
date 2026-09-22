import os
import io
import uuid

import torch

from fastapi import (
    FastAPI,
    File,
    UploadFile,
    Form,
    HTTPException,
    Request
)

from fastapi.responses import (
    FileResponse,
    HTMLResponse
)

from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

from PIL import Image
from torchvision import transforms

from utils.models import VGGEncoder, Decoder
from utils.utils import adaptive_instance_normalization


# ============================================================
# 1. FASTAPI APP
# ============================================================

app = FastAPI(
    title="Neural Style Transfer API",
    description="AdaIN Neural Style Transfer using PyTorch",
    version="1.0.0"
)


# ============================================================
# 2. PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

VGG_PATH = os.path.join(
    BASE_DIR,
    "vgg_normalised.pth"
)

DECODER_PATH = os.path.join(
    BASE_DIR,
    "decoder_200.pth"
)

STATIC_DIR = os.path.join(
    BASE_DIR,
    "static"
)

UPLOAD_FOLDER = os.path.join(
    STATIC_DIR,
    "uploads"
)

TEMPLATES_DIR = os.path.join(
    BASE_DIR,
    "templates"
)

EXAMPLES_DIR = os.path.join(
    BASE_DIR,
    "examples"
)


# ============================================================
# 3. CREATE REQUIRED DIRECTORIES
# ============================================================

os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)

os.makedirs(
    TEMPLATES_DIR,
    exist_ok=True
)

os.makedirs(
    STATIC_DIR,
    exist_ok=True
)


# ============================================================
# 4. JINJA2 TEMPLATES
# ============================================================

templates = Jinja2Templates(
    directory=TEMPLATES_DIR
)


# ============================================================
# 5. STATIC FILES
# ============================================================

app.mount(
    "/static",
    StaticFiles(directory=STATIC_DIR),
    name="static"
)


# ============================================================
# 6. EXAMPLE IMAGES
# ============================================================

if os.path.isdir(EXAMPLES_DIR):

    app.mount(
        "/examples",
        StaticFiles(directory=EXAMPLES_DIR),
        name="examples"
    )


# ============================================================
# 7. DEVICE
# ============================================================

# Render Free does not provide a GPU.
# Therefore this will normally be CPU.

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("=" * 60)
print("NEURAL STYLE TRANSFER API")
print("=" * 60)

print("Device:", device)

if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

print("VGG path:", VGG_PATH)
print("Decoder path:", DECODER_PATH)

print("=" * 60)


# ============================================================
# 8. CHECK MODEL FILES
# ============================================================

if not os.path.exists(VGG_PATH):

    raise FileNotFoundError(
        f"VGG file not found:\n{VGG_PATH}"
    )


if not os.path.exists(DECODER_PATH):

    raise FileNotFoundError(
        f"Decoder file not found:\n{DECODER_PATH}"
    )


# ============================================================
# 9. LAZY MODEL LOADING
# ============================================================

# IMPORTANT:
# Do NOT load VGG and Decoder during application startup.
#
# Render Free has only 512 MB RAM.
#
# We load the models only when the first
# style-transfer request is received.

encoder = None
decoder = None


def load_models():

    global encoder
    global decoder

    # --------------------------------------------------------
    # Models already loaded
    # --------------------------------------------------------

    if (
        encoder is not None
        and
        decoder is not None
    ):

        return


    print("=" * 60)
    print("LOADING NEURAL STYLE TRANSFER MODELS")
    print("=" * 60)


    # --------------------------------------------------------
    # Load VGG Encoder
    # --------------------------------------------------------

    print("Loading VGG encoder...")

    encoder = VGGEncoder(
        VGG_PATH
    ).to(device)

    encoder.eval()

    print("VGG encoder loaded.")


    # --------------------------------------------------------
    # Load Decoder
    # --------------------------------------------------------

    print("Loading trained decoder...")

    decoder = Decoder().to(device)

    decoder.load_state_dict(
        torch.load(
            DECODER_PATH,
            map_location=device,
            weights_only=True
        )
    )

    decoder.eval()

    print("Decoder loaded successfully.")

    print("=" * 60)


# ============================================================
# 10. IMAGE TRANSFORM
# ============================================================

# 256 x 256 is used instead of 512 x 512
# to reduce RAM usage on Render Free.

image_transform = transforms.Compose([

    transforms.Resize(
        (256, 256)
    ),

    transforms.ToTensor()

])


# ============================================================
# 11. STYLE TRANSFER FUNCTION
# ============================================================

def style_transfer(
    content_image,
    style_image,
    alpha=1.0
):

    # --------------------------------------------------------
    # Load models only when needed
    # --------------------------------------------------------

    load_models()


    # --------------------------------------------------------
    # Convert PIL images to tensors
    # --------------------------------------------------------

    content_tensor = image_transform(
        content_image
    ).unsqueeze(0).to(device)


    style_tensor = image_transform(
        style_image
    ).unsqueeze(0).to(device)


    print(
        "Content tensor:",
        content_tensor.shape
    )

    print(
        "Style tensor:",
        style_tensor.shape
    )


    # --------------------------------------------------------
    # Neural Style Transfer
    # --------------------------------------------------------

    with torch.no_grad():


        # ----------------------------------------------------
        # Extract VGG features
        # ----------------------------------------------------

        content_features = encoder(
            content_tensor
        )

        style_features = encoder(
            style_tensor
        )


        # ----------------------------------------------------
        # Get deepest feature
        # ----------------------------------------------------

        content_feature = content_features[-1]

        style_feature = style_features[-1]


        print(
            "Content feature:",
            content_feature.shape
        )

        print(
            "Style feature:",
            style_feature.shape
        )


        # ----------------------------------------------------
        # Adaptive Instance Normalization
        # ----------------------------------------------------

        target_feature = adaptive_instance_normalization(
            content_feature,
            style_feature
        )


        # ----------------------------------------------------
        # Alpha blending
        # ----------------------------------------------------

        target_feature = (

            alpha * target_feature

            +

            (1.0 - alpha) * content_feature

        )


        # ----------------------------------------------------
        # Decoder
        # ----------------------------------------------------

        generated_image = decoder(
            target_feature
        )


    print(
        "Generated image:",
        generated_image.shape
    )


    return generated_image


# ============================================================
# 12. SAVE GENERATED IMAGE
# ============================================================

def save_generated_image(
    image_tensor,
    output_path
):

    # --------------------------------------------------------
    # Remove gradients
    # --------------------------------------------------------

    image_tensor = (
        image_tensor
        .detach()
        .cpu()
    )


    # --------------------------------------------------------
    # Remove batch dimension
    # --------------------------------------------------------

    image_tensor = image_tensor.squeeze(0)


    # --------------------------------------------------------
    # Keep pixel values between 0 and 1
    # --------------------------------------------------------

    image_tensor = image_tensor.clamp(
        0,
        1
    )


    # --------------------------------------------------------
    # Convert tensor -> PIL image
    # --------------------------------------------------------

    image = transforms.ToPILImage()(
        image_tensor
    )


    # --------------------------------------------------------
    # Save image
    # --------------------------------------------------------

    image.save(
        output_path
    )


# ============================================================
# 13. ROOT PAGE
# ============================================================

@app.get(
    "/",
    response_class=HTMLResponse
)
async def root(
    request: Request
):

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={}
    )


# ============================================================
# 14. HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    return {

        "status": "healthy",

        "device": str(device),

        "cuda_available":
            torch.cuda.is_available()

    }


# ============================================================
# 15. STYLE TRANSFER API
# ============================================================

@app.post("/style-transfer")
async def style_transfer_api(

    content: UploadFile = File(...),

    style: UploadFile = File(...),

    alpha: float = Form(1.0)

):

    print()
    print("=" * 60)
    print("STYLE TRANSFER REQUEST")
    print("=" * 60)


    # ========================================================
    # CHECK CONTENT FILE
    # ========================================================

    if content is None:

        raise HTTPException(
            status_code=400,
            detail="Content image was not provided."
        )


    # ========================================================
    # CHECK STYLE FILE
    # ========================================================

    if style is None:

        raise HTTPException(
            status_code=400,
            detail="Style image was not provided."
        )


    # ========================================================
    # REQUEST INFORMATION
    # ========================================================

    print(
        "Content filename:",
        content.filename
    )

    print(
        "Style filename:",
        style.filename
    )

    print(
        "Alpha:",
        alpha
    )


    # ========================================================
    # VALIDATE ALPHA
    # ========================================================

    if alpha < 0.0 or alpha > 1.0:

        raise HTTPException(
            status_code=400,
            detail="Alpha must be between 0 and 1."
        )


    # ========================================================
    # VALIDATE FILENAMES
    # ========================================================

    if not content.filename:

        raise HTTPException(
            status_code=400,
            detail="Content filename is empty."
        )


    if not style.filename:

        raise HTTPException(
            status_code=400,
            detail="Style filename is empty."
        )


    # ========================================================
    # VALIDATE EXTENSIONS
    # ========================================================

    allowed_extensions = {

        ".jpg",
        ".jpeg",
        ".png"

    }


    content_extension = os.path.splitext(
        content.filename
    )[1].lower()


    style_extension = os.path.splitext(
        style.filename
    )[1].lower()


    if content_extension not in allowed_extensions:

        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid content image format. "
                "Use JPG, JPEG or PNG."
            )
        )


    if style_extension not in allowed_extensions:

        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid style image format. "
                "Use JPG, JPEG or PNG."
            )
        )


    # ========================================================
    # READ UPLOADED IMAGES
    # ========================================================

    try:

        print("Reading uploaded images...")


        content_data = await content.read()

        style_data = await style.read()


        print(
            "Content bytes:",
            len(content_data)
        )

        print(
            "Style bytes:",
            len(style_data)
        )


        # ----------------------------------------------------
        # Check empty files
        # ----------------------------------------------------

        if len(content_data) == 0:

            raise ValueError(
                "Content image is empty."
            )


        if len(style_data) == 0:

            raise ValueError(
                "Style image is empty."
            )


        # ----------------------------------------------------
        # Convert bytes -> PIL images
        # ----------------------------------------------------

        content_image = Image.open(
            io.BytesIO(content_data)
        ).convert("RGB")


        style_image = Image.open(
            io.BytesIO(style_data)
        ).convert("RGB")


        print(
            "Content image size:",
            content_image.size
        )

        print(
            "Style image size:",
            style_image.size
        )


    except Exception as e:

        print(
            "IMAGE READING ERROR:",
            repr(e)
        )

        raise HTTPException(
            status_code=400,
            detail=(
                "Could not read uploaded images: "
                + str(e)
            )
        )


    # ========================================================
    # RUN STYLE TRANSFER
    # ========================================================

    try:

        print(
            "Running AdaIN style transfer..."
        )


        generated_image = style_transfer(

            content_image,

            style_image,

            alpha

        )


        print(
            "Style transfer completed."
        )


    except Exception as e:

        print(
            "MODEL ERROR:",
            repr(e)
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Style transfer failed: "
                + str(e)
            )
        )


    # ========================================================
    # CREATE OUTPUT FILE
    # ========================================================

    try:

        filename = (

            "stylized_"

            + uuid.uuid4().hex

            + ".png"

        )


        output_path = os.path.join(

            UPLOAD_FOLDER,

            filename

        )


        save_generated_image(

            generated_image,

            output_path

        )


        print(
            "Output saved:",
            output_path
        )


    except Exception as e:

        print(
            "SAVE ERROR:",
            repr(e)
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Could not save generated image: "
                + str(e)
            )
        )


    # ========================================================
    # RETURN GENERATED IMAGE
    # ========================================================

    print(
        "Returning generated image..."
    )

    print("=" * 60)


    return FileResponse(

        path=output_path,

        media_type="image/png",

        filename=filename

    )


# ============================================================
# 16. START SERVER
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(

        app,

        host="0.0.0.0",

        port=8000,

        reload=False

    )