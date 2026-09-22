import os
import io
import uuid
import gc
import threading

import torch

# Limit CPU thread usage on small/free machines
torch.set_num_threads(2)
torch.set_num_interop_threads(1)

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
        StaticFiles(
            directory=EXAMPLES_DIR
        ),
        name="examples"
    )


# ============================================================
# 7. DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
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
# 9. MODEL VARIABLES
# ============================================================

# Models are NOT loaded during application startup.
#
# This is important for Render Free because it has
# limited RAM.

encoder = None
decoder = None


# Only one inference request at a time.
#
# This prevents two simultaneous requests from consuming
# the available RAM.

inference_lock = threading.Lock()


# ============================================================
# 10. LOAD MODELS
# ============================================================

def load_models():

    global encoder
    global decoder

    # --------------------------------------------------------
    # Already loaded
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
# 11. IMAGE TRANSFORM
# ============================================================

# Smaller resolution reduces memory consumption.
#
# If quality is too low, you can later change this to
# (256, 256).

image_transform = transforms.Compose([

    transforms.Resize(
        (192, 192)
    ),

    transforms.ToTensor()

])


# ============================================================
# 12. STYLE TRANSFER
# ============================================================

def style_transfer(
    content_image,
    style_image,
    alpha=1.0
):

    # --------------------------------------------------------
    # Load models if this is the first request
    # --------------------------------------------------------

    load_models()


    # --------------------------------------------------------
    # Create tensors
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
    # Inference
    # --------------------------------------------------------

    with torch.inference_mode():

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
        # Deepest feature
        # ----------------------------------------------------

        content_feature = (
            content_features[-1]
        )

        style_feature = (
            style_features[-1]
        )


        print(
            "Content feature:",
            content_feature.shape
        )

        print(
            "Style feature:",
            style_feature.shape
        )


        # ----------------------------------------------------
        # AdaIN
        # ----------------------------------------------------

        target_feature = (
            adaptive_instance_normalization(
                content_feature,
                style_feature
            )
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


        # ----------------------------------------------------
        # Move result to CPU
        # ----------------------------------------------------

        result = (
            generated_image
            .detach()
            .cpu()
        )


    # ========================================================
    # FREE TEMPORARY MEMORY
    # ========================================================

    del content_tensor
    del style_tensor

    del content_features
    del style_features

    del content_feature
    del style_feature

    del target_feature
    del generated_image

    gc.collect()


    # CUDA cleanup if a GPU is available
    if torch.cuda.is_available():

        torch.cuda.empty_cache()


    return result


# ============================================================
# 13. SAVE GENERATED IMAGE
# ============================================================

def save_generated_image(
    image_tensor,
    output_path
):

    # --------------------------------------------------------
    # Remove batch dimension
    # --------------------------------------------------------

    image_tensor = (
        image_tensor
        .squeeze(0)
    )


    # --------------------------------------------------------
    # Clamp values
    # --------------------------------------------------------

    image_tensor = (
        image_tensor
        .clamp(0, 1)
    )


    # --------------------------------------------------------
    # Convert tensor to PIL
    # --------------------------------------------------------

    image = transforms.ToPILImage()(
        image_tensor
    )


    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    image.save(
        output_path
    )


# ============================================================
# 14. ROOT PAGE
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
# 15. HEALTH CHECK
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
# 16. STYLE TRANSFER API
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
    # CHECK FILES
    # ========================================================

    if content is None:

        raise HTTPException(
            status_code=400,
            detail="Content image was not provided."
        )


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


    content_extension = (
        os.path.splitext(
            content.filename
        )[1]
        .lower()
    )


    style_extension = (
        os.path.splitext(
            style.filename
        )[1]
        .lower()
    )


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
    # READ IMAGES
    # ========================================================

    try:

        print(
            "Reading uploaded images..."
        )


        content_data = (
            await content.read()
        )

        style_data = (
            await style.read()
        )


        print(
            "Content bytes:",
            len(content_data)
        )

        print(
            "Style bytes:",
            len(style_data)
        )


        # ----------------------------------------------------
        # Empty file check
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
        # Bytes -> PIL
        # ----------------------------------------------------

        content_image = (
            Image.open(
                io.BytesIO(
                    content_data
                )
            )
            .convert("RGB")
        )


        style_image = (
            Image.open(
                io.BytesIO(
                    style_data
                )
            )
            .convert("RGB")
        )


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
            "Waiting for inference lock..."
        )


        # Only one inference at a time.
        with inference_lock:

            print(
                "Running AdaIN style transfer..."
            )

            generated_image = (
                style_transfer(
                    content_image,
                    style_image,
                    alpha
                )
            )

            print(
                "Style transfer completed."
            )


    except Exception as e:

        print(
            "MODEL ERROR:",
            repr(e)
        )

        # Cleanup
        gc.collect()

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


        # ----------------------------------------------------
        # Free output tensor memory
        # ----------------------------------------------------

        del generated_image

        gc.collect()


    except Exception as e:

        print(
            "SAVE ERROR:",
            repr(e)
        )

        gc.collect()

        raise HTTPException(
            status_code=500,
            detail=(
                "Could not save generated image: "
                + str(e)
            )
        )


    # ========================================================
    # RETURN IMAGE
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
# 17. START SERVER
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(

        app,

        host="0.0.0.0",

        port=8000,

        reload=False

    )