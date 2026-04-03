import streamlit as st
import torch
import torch.nn as nn
import torchvision.utils as vutils
import numpy as np
from PIL import Image
import io
import os

# ---- Generator Architecture (same as training notebook) ----
class Generator(nn.Module):
    def __init__(self, latent_dim, channels):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)


LATENT_DIM = 100
CHANNELS = 3

# ---- Load models (cached so they only load once) ----
@st.cache_resource
def load_generator(model_path, device):
    gen = Generator(LATENT_DIM, CHANNELS)
    state = torch.load(model_path, map_location=device, weights_only=True)
    gen.load_state_dict(state)
    gen.to(device)
    gen.eval()
    return gen


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def generate_faces(model, num_images, device, seed=None, truncation=1.0):
    """
    Generate anime faces from a trained generator.
    
    truncation: scales the noise vector. Values < 1.0 reduce variety but
    improve average quality. Values > 1.0 increase variety but may reduce quality.
    This is known as the 'truncation trick' in GAN literature.
    """
    if seed is not None:
        torch.manual_seed(seed)

    noise = torch.randn(num_images, LATENT_DIM, 1, 1, device=device)
    noise = noise * truncation  # truncation trick

    with torch.no_grad():
        fake_images = model(noise).cpu()

    # denormalize from [-1,1] to [0,1]
    fake_images = fake_images * 0.5 + 0.5
    fake_images = fake_images.clamp(0, 1)
    return fake_images


def tensor_to_pil(tensor_img):
    """Convert a single [C, H, W] tensor to PIL image."""
    img_np = tensor_img.permute(1, 2, 0).numpy()
    img_np = (img_np * 255).astype(np.uint8)
    return Image.fromarray(img_np)


def make_grid_image(tensors, nrow=4):
    """Convert batch of tensors to a single PIL grid image."""
    grid = vutils.make_grid(tensors, nrow=nrow, padding=2, pad_value=1.0)
    img_np = grid.permute(1, 2, 0).numpy()
    img_np = (img_np * 255).astype(np.uint8)
    return Image.fromarray(img_np)


# ---- Page config ----
st.set_page_config(
    page_title="Anime Face Generator — DCGAN vs WGAN-GP",
    page_icon="🎨",
    layout="wide"
)

# ---- Custom CSS ----
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
    }
    .main-header h1 {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 800;
    }
    .model-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        padding: 1.2rem;
        border-radius: 12px;
        border: 1px solid #334155;
        margin-bottom: 1rem;
    }
    .metric-box {
        background: #0f172a;
        padding: 0.8rem;
        border-radius: 8px;
        text-align: center;
        border: 1px solid #1e293b;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 1.1rem;
        font-weight: 600;
    }
    div[data-testid="stImage"] {
        border-radius: 10px;
        overflow: hidden;
    }
</style>
""", unsafe_allow_html=True)

# ---- Header ----
st.markdown('<div class="main-header"><h1>🎨 Anime Face Generator</h1></div>', unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center; color:#94a3b8; font-size:1.1rem;'>"
    "Comparing DCGAN (baseline) vs WGAN-GP (improved) for anime face generation"
    "</p>",
    unsafe_allow_html=True
)
st.markdown("---")

# ---- Check model files ----
DCGAN_PATH = "dcgan_generator_final.pth"
WGAN_PATH = "wgan_generator_final.pth"

dcgan_available = os.path.exists(DCGAN_PATH)
wgan_available = os.path.exists(WGAN_PATH)

if not dcgan_available and not wgan_available:
    st.error(
        "No model files found. Place `dcgan_generator_final.pth` and "
        "`wgan_generator_final.pth` in the same directory as this script."
    )
    st.stop()

# load whatever is available
device = get_device()
st.sidebar.markdown(f"**Device:** `{device}`")

dcgan_model = load_generator(DCGAN_PATH, device) if dcgan_available else None
wgan_model = load_generator(WGAN_PATH, device) if wgan_available else None

# ---- Sidebar Controls ----
st.sidebar.markdown("## Generation Settings")

num_images = st.sidebar.slider("Number of faces", min_value=1, max_value=16, value=8, step=1)

truncation = st.sidebar.slider(
    "Truncation",
    min_value=0.3, max_value=2.0, value=1.0, step=0.1,
    help="Controls variety vs quality tradeoff. "
         "Lower values = less variety but better average quality. "
         "Higher values = more variety but some faces may look odd."
)

use_seed = st.sidebar.checkbox("Set random seed (for reproducible results)")
seed_value = None
if use_seed:
    seed_value = st.sidebar.number_input("Seed", min_value=0, max_value=99999, value=42, step=1)

grid_cols = st.sidebar.select_slider(
    "Grid columns",
    options=[2, 4, 8],
    value=4,
    help="How many images per row in the output grid"
)

st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.markdown(
    "This app demonstrates two GAN architectures trained on the "
    "[Anime Faces 64x64](https://www.kaggle.com/datasets/soumikrakshit/anime-faces) dataset.\n\n"
    "**DCGAN** uses BCE loss and is prone to mode collapse.\n\n"
    "**WGAN-GP** uses Wasserstein loss with gradient penalty for stable training."
)

# ---- Tabs ----
tab_compare, tab_dcgan, tab_wgan, tab_explore = st.tabs([
    "🔀 Compare", "📘 DCGAN", "📗 WGAN-GP", "🔍 Explore Latent Space"
])

# ---- Tab 1: Side by side comparison ----
with tab_compare:
    st.subheader("Side-by-Side Comparison")
    st.caption("Both models receive the same noise input for a fair comparison.")

    if st.button("Generate Comparison", key="btn_compare", type="primary"):
        if not dcgan_available or not wgan_available:
            st.warning("Both models need to be loaded for comparison.")
        else:
            # same seed for both so they get identical noise
            comp_seed = seed_value if seed_value is not None else np.random.randint(0, 99999)

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### DCGAN (Baseline)")
                dcgan_imgs = generate_faces(dcgan_model, num_images, device, seed=comp_seed, truncation=truncation)
                grid_img = make_grid_image(dcgan_imgs, nrow=grid_cols)
                st.image(grid_img, use_container_width=True)

            with col2:
                st.markdown("#### WGAN-GP (Improved)")
                wgan_imgs = generate_faces(wgan_model, num_images, device, seed=comp_seed, truncation=truncation)
                grid_img = make_grid_image(wgan_imgs, nrow=grid_cols)
                st.image(grid_img, use_container_width=True)

            st.info(f"Seed used: **{comp_seed}** — set the same seed to reproduce these results.")

# ---- Tab 2: DCGAN only ----
with tab_dcgan:
    st.subheader("DCGAN Generator")
    if not dcgan_available:
        st.warning("DCGAN model file not found.")
    else:
        st.markdown(
            "DCGAN uses **Binary Cross-Entropy loss** and a standard discriminator with sigmoid output. "
            "It can suffer from training instability and mode collapse — you might notice "
            "some generated faces look repetitive or low quality."
        )
        if st.button("Generate with DCGAN", key="btn_dcgan", type="primary"):
            images = generate_faces(dcgan_model, num_images, device, seed=seed_value, truncation=truncation)
            grid_img = make_grid_image(images, nrow=grid_cols)
            st.image(grid_img, use_container_width=True)

            # also show individual faces
            st.markdown("**Individual Faces:**")
            cols = st.columns(min(num_images, 4))
            for idx in range(min(num_images, 4)):
                with cols[idx]:
                    pil_img = tensor_to_pil(images[idx])
                    st.image(pil_img, caption=f"Face {idx+1}", use_container_width=True)

# ---- Tab 3: WGAN-GP only ----
with tab_wgan:
    st.subheader("WGAN-GP Generator")
    if not wgan_available:
        st.warning("WGAN-GP model file not found.")
    else:
        st.markdown(
            "WGAN-GP uses **Wasserstein loss with gradient penalty** and a critic (no sigmoid). "
            "This provides smoother gradients during training, leading to more stable convergence "
            "and better diversity in generated samples."
        )
        if st.button("Generate with WGAN-GP", key="btn_wgan", type="primary"):
            images = generate_faces(wgan_model, num_images, device, seed=seed_value, truncation=truncation)
            grid_img = make_grid_image(images, nrow=grid_cols)
            st.image(grid_img, use_container_width=True)

            st.markdown("**Individual Faces:**")
            cols = st.columns(min(num_images, 4))
            for idx in range(min(num_images, 4)):
                with cols[idx]:
                    pil_img = tensor_to_pil(images[idx])
                    st.image(pil_img, caption=f"Face {idx+1}", use_container_width=True)

# ---- Tab 4: Latent Space Exploration ----
with tab_explore:
    st.subheader("Latent Space Interpolation")
    st.caption(
        "Generate two random faces and smoothly interpolate between them. "
        "This shows how the generator maps the noise space to face space."
    )

    model_choice = st.radio(
        "Model to use",
        ["DCGAN", "WGAN-GP"],
        horizontal=True,
        key="interp_model"
    )

    interp_steps = st.slider("Interpolation steps", min_value=3, max_value=12, value=8, step=1)

    if st.button("Interpolate", key="btn_interp", type="primary"):
        chosen_model = dcgan_model if model_choice == "DCGAN" else wgan_model
        if chosen_model is None:
            st.warning(f"{model_choice} model not loaded.")
        else:
            # two random endpoints in latent space
            if seed_value is not None:
                torch.manual_seed(seed_value)
            z1 = torch.randn(1, LATENT_DIM, 1, 1, device=device) * truncation
            z2 = torch.randn(1, LATENT_DIM, 1, 1, device=device) * truncation

            # linear interpolation between z1 and z2
            alphas = np.linspace(0, 1, interp_steps)
            interp_images = []
            for a in alphas:
                z = z1 * (1 - a) + z2 * a
                with torch.no_grad():
                    img = chosen_model(z).cpu()
                img = img * 0.5 + 0.5
                img = img.clamp(0, 1)
                interp_images.append(img)

            all_imgs = torch.cat(interp_images, dim=0)
            grid_img = make_grid_image(all_imgs, nrow=interp_steps)
            st.image(grid_img, use_container_width=True, caption="Smooth interpolation from Face A → Face B")

            st.markdown(
                "Notice how the transition is smooth — this means the generator learned a "
                "**continuous mapping** from noise space to face space, not just memorized "
                "a few examples."
            )
