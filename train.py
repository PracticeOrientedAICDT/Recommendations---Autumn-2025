import os
import math
import argparse
import time
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import GradScaler
from torch.nn.utils import clip_grad_norm_

import wandb
from tqdm import tqdm


# === Time Embedding ===
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


# === Custom ResBlock with Adaptive Layer Norm (AdaLN) for time conditioning ===
class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # Use LayerNorm instead of GroupNorm for better stability
        self.norm1 = nn.LayerNorm(dim)
        self.act1 = nn.SiLU()  # SiLU is Swish
        self.lin1 = nn.Linear(dim, dim)

        self.norm2 = nn.LayerNorm(dim)
        self.act2 = nn.SiLU()
        self.lin2 = nn.Linear(dim, dim)

        # Adaptive scaling from time embedding
        self.time_scale1 = nn.Linear(dim, dim)
        self.time_scale2 = nn.Linear(dim, dim)
        self.time_shift1 = nn.Linear(dim, dim)
        self.time_shift2 = nn.Linear(dim, dim)

    def forward(self, x, t_emb=None):
        h = x
        h = self.norm1(h)

        # Apply adaptive time scaling if provided
        if t_emb is not None:
            scale = self.time_scale1(t_emb).unsqueeze(1)
            shift = self.time_shift1(t_emb).unsqueeze(1)
            h = h * (1 + scale) + shift

        h = self.act1(h)
        h = self.lin1(h)

        h = self.norm2(h)

        # Apply adaptive time scaling if provided
        if t_emb is not None:
            scale = self.time_scale2(t_emb).unsqueeze(1)
            shift = self.time_shift2(t_emb).unsqueeze(1)
            h = h * (1 + scale) + shift

        h = self.act2(h)
        h = self.lin2(h)

        return h + x  # Skip connection


# === DMSG Paper-Style Model ===
class DiffusionSlateModel(nn.Module):
    def __init__(
        self,
        dim=1024,
        hidden=320,
        layers=3,
        heads=10,
        dropout=0.2,
        slate_len=10,
        ctx_layers=3,
        ctx_heads=8,
    ):
        super().__init__()

        self.in_proj = nn.Linear(dim, hidden)
        self.ctx_proj_in = nn.Linear(dim, hidden)

        ctx_encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=ctx_heads,
            dim_feedforward=hidden * 4,
            batch_first=True,
            activation="gelu",
            dropout=dropout,
        )
        self.ctx_encoder = nn.TransformerEncoder(
            ctx_encoder_layer, num_layers=ctx_layers
        )

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )

        self.pos_emb = nn.Parameter(torch.randn(1, slate_len, hidden))

        self.model_layers = nn.ModuleList()
        for _ in range(layers):
            self.model_layers.append(ResidualBlock(hidden))
            self.model_layers.append(
                nn.TransformerDecoderLayer(
                    d_model=hidden,
                    nhead=heads,
                    dim_feedforward=hidden * 4,
                    batch_first=True,
                    activation=F.gelu,
                    dropout=dropout,
                )
            )

        self.out_proj = nn.Linear(hidden, dim)

    def forward(self, x_t, t, ctx):
        """
        x_t: (B, K, D) - noisy slate where D is item_dim (384 for MiniLM)
        t: (B,) - time steps
        ctx: (B, D) - prompt embedding where D is item_dim (384 for MiniLM)
        """
        # 1. Project noisy slate to hidden dim
        h = self.in_proj(x_t)

        # 2. Process context
        c_emb_in = self.ctx_proj_in(ctx).unsqueeze(1)
        c_emb = self.ctx_encoder(c_emb_in)

        # 3. Get time embedding
        t_emb = self.time_mlp(t).unsqueeze(1)  # (B, 1, 320)

        # 4. Add position and time conditioning ONCE
        # h = (B, K, 320), pos_emb = (1, K, 320), t_emb = (B, 1, 320)
        h = h + self.pos_emb + t_emb

        # 5. Run through the stack of layers
        # Pass time embedding to residual blocks for adaptive conditioning
        for layer in self.model_layers:
            if isinstance(layer, ResidualBlock):
                # Pass time embedding for adaptive layer norm
                h = layer(h, t_emb=self.time_mlp(t))
            elif isinstance(layer, nn.TransformerDecoderLayer):
                # Pass context to cross-attention
                h = layer(tgt=h, memory=c_emb)

        return self.out_proj(h)


# === Dataset (with CFG) ===
class SlateDataset(Dataset):
    def __init__(self, path, slate_len, item_dim, cfg_drop_prob=0.1, data=None):
        if data is None:
            print("Loading data from path...")
            data = torch.load(path, map_location="cpu")

        self.slates = data["slates"]
        self.prompts = data["prompts"].cpu()
        self.slate_len = slate_len
        self.item_dim = item_dim
        self.cfg_drop_prob = cfg_drop_prob
        assert len(self.slates) == self.prompts.size(0)

    def __len__(self):
        return len(self.slates)

    def __getitem__(self, i):
        x0 = self.slates[i].float()
        c = self.prompts[i].float()

        # Ensure embeddings are normalized (safety check)
        x0 = F.normalize(x0, dim=-1) if x0.size(0) > 0 else x0
        c = F.normalize(c.unsqueeze(0), dim=-1).squeeze(0) if c.numel() > 0 else c

        if x0.size(0) < self.slate_len:
            pad = torch.zeros(self.slate_len - x0.size(0), self.item_dim)
            x0 = torch.cat([x0, pad], 0)
        elif x0.size(0) > self.slate_len:
            x0 = x0[: self.slate_len]

        if torch.rand(1) < self.cfg_drop_prob:
            c = torch.zeros_like(c)

        return x0, c


# === Diffusion schedule ===
def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return betas.clamp(0.0001, 0.9999)


def get_diffusion_vars(timesteps, device):
    betas = cosine_beta_schedule(timesteps).to(device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

    return {
        "betas": betas,
        "alphas_cumprod": alphas_cumprod,
        "sqrt_alphas_cumprod": sqrt_alphas_cumprod,
        "sqrt_one_minus_alphas_cumprod": sqrt_one_minus_alphas_cumprod,
        "alphas_cumprod_prev": alphas_cumprod_prev,
    }


# === Generation Function (DDIM Sampler + NN Decoder) ===
@torch.no_grad()
def generate(
    model,
    prompt_emb,
    item_catalog_embs,
    diff_vars,
    args,
    device,
    num_inference_steps=50,
):
    """
    Generates a slate of item *indices* using DDIM sampling and Nearest Neighbor decoding.

    item_catalog_embs: (N, D) tensor of all item embeddings in the catalog.
    Returns: A list of lists, shape (B, K), containing item indices.
    """
    model.eval()
    B = prompt_emb.size(0)
    K, D = args.slate_len, args.item_dim
    T = args.t_steps

    # Create a strided schedule from T-1 down to 0
    timesteps = torch.linspace(
        T - 1, 0, num_inference_steps, dtype=torch.long, device=device
    )

    # All diffusion variables, indexed by full T steps
    a_t_all = diff_vars["alphas_cumprod"]
    sqrt_a_t_all = diff_vars["sqrt_alphas_cumprod"]
    sqrt_1m_a_t_all = diff_vars["sqrt_one_minus_alphas_cumprod"]

    # Start with noise
    x_t = torch.randn((B, K, D), device=device)
    uncond_prompt_emb = torch.zeros_like(prompt_emb).to(device)

    # Normalize catalog embeddings once
    catalog_norm = F.normalize(item_catalog_embs, dim=-1)  # (N, D)

    loop = tqdm(
        range(num_inference_steps),
        desc="DDIM Sampling",
        total=num_inference_steps,
        leave=False,
    )
    for i in loop:
        # Get current and previous timesteps
        t = timesteps[i].expand(B)
        # t_prev is the next step in the schedule, or -1 for the last step
        t_prev = (
            timesteps[i + 1]
            if i < num_inference_steps - 1
            else torch.tensor(-1, device=device)
        )

        # Get diffusion variables for current timestep t
        _sqrt_a_t = sqrt_a_t_all[t].view(B, 1, 1)
        _sqrt_1m_a_t = sqrt_1m_a_t_all[t].view(B, 1, 1)

        # Get diffusion variables for previous timestep t_prev
        # if t_prev is -1 (end), a_prev is 1.0 (original data)
        if isinstance(t_prev, torch.Tensor):
            t_prev_val = t_prev.item() if t_prev.numel() == 1 else t_prev
        else:
            t_prev_val = t_prev

        if t_prev_val >= 0:
            _a_prev = (
                a_t_all[t_prev].view(B, 1, 1)
                if isinstance(t_prev, torch.Tensor)
                else a_t_all[t_prev]
            )
        else:
            _a_prev = torch.tensor(1.0, device=device)

        if _a_prev.dim() == 0:
            _a_prev = _a_prev.view(B, 1, 1)
        elif _a_prev.dim() == 1:
            _a_prev = _a_prev.view(B, 1, 1)

        _sqrt_a_prev = torch.sqrt(_a_prev)
        _sqrt_1m_a_prev = torch.sqrt(1.0 - _a_prev)

        # Run model
        with torch.amp.autocast(
            device_type=device.type, enabled=(device.type == "cuda")
        ):
            v_cond = model(x_t, t, prompt_emb)
            v_uncond = model(x_t, t, uncond_prompt_emb)

        # Classifier-Free Guidance
        v_pred = v_uncond + args.cfg_scale * (v_cond - v_uncond)

        # Predict x0 from v_pred
        pred_x0 = _sqrt_a_t * x_t - _sqrt_1m_a_t * v_pred
        # Predict noise (epsilon) from v_pred
        pred_noise = _sqrt_a_t * v_pred + _sqrt_1m_a_t * x_t

        # DDIM update rule (with eta=0 for deterministic sampling)
        x_t = _sqrt_a_prev * pred_x0 + _sqrt_1m_a_prev * pred_noise

    # Final x_t is the predicted x_0
    x_0_pred = x_t

    # Nearest Neighbor Search - map continuous latents back to discrete items
    # Clip to prevent extreme values before normalization
    x_0_pred = torch.clamp(x_0_pred, min=-10.0, max=10.0)

    generated_latents = F.normalize(x_0_pred, dim=-1)  # (B, K, D)

    # Compute cosine similarity: (B, K, D) @ (D, N) -> (B, K, N)
    # More efficient: use matmul instead of bmm
    sims = torch.matmul(generated_latents, catalog_norm.T)  # (B, K, N)

    # Apply temperature scaling for more diverse sampling (optional, can be disabled)
    temperature = getattr(args, "nn_temperature", 1.0)
    if temperature != 1.0:
        sims = sims / temperature

    # Get top 5 neighbors for anti-duplicate logic (as mentioned in paper)
    # We get 5 just in case the top 1s are duplicates
    top_k_neighbors = 5
    _, top_k_indices = torch.topk(sims, k=top_k_neighbors, dim=-1)  # (B, K, 5)

    final_slates_indices = []
    for b in range(B):
        batch_slate_indices = []
        used_indices = set()
        for k_slot in range(args.slate_len):
            found_item = False
            # Iterate through neighbors for this slot
            for k_neighbor_idx in range(top_k_neighbors):
                item_idx = top_k_indices[b, k_slot, k_neighbor_idx].item()
                if item_idx not in used_indices:
                    # Found a new item, add it
                    batch_slate_indices.append(item_idx)
                    used_indices.add(item_idx)
                    found_item = True
                    break
            if not found_item:
                # If all top 5 are duplicates (unlikely), just add the best one
                batch_slate_indices.append(top_k_indices[b, k_slot, 0].item())

    final_slates_indices.append(batch_slate_indices)

    return final_slates_indices  # List of lists, (B, K)


# === Validation Function with Reconstruction Metrics ===
@torch.no_grad()
def validate(model, val_loader, diff_vars, args, device, item_catalog_embs=None):
    model.eval()
    running_loss_mse = 0.0
    running_loss_cos = 0.0
    running_loss_l1 = 0.0
    running_x0_cosine = 0.0  # Cosine similarity between predicted and true x0

    sqrt_a_t = diff_vars["sqrt_alphas_cumprod"].to(device)
    sqrt_1m_a_t = diff_vars["sqrt_one_minus_alphas_cumprod"].to(device)
    T = args.t_steps

    loop = tqdm(val_loader, total=len(val_loader), desc="Validating", leave=False)
    for i, (x0, c) in enumerate(loop):
        x0, c = x0.to(device, non_blocking=True), c.to(device, non_blocking=True)
        B = x0.size(0)
        t = torch.randint(0, T, (B,), device=device)

        _sqrt_a_t = sqrt_a_t[t].view(B, 1, 1)
        _sqrt_1m_a_t = sqrt_1m_a_t[t].view(B, 1, 1)

        noise = torch.randn_like(x0)
        x_t = _sqrt_a_t * x0 + _sqrt_1m_a_t * noise
        v_target = _sqrt_a_t * noise - _sqrt_1m_a_t * x0

        with torch.amp.autocast(
            device_type=device.type, enabled=(device.type == "cuda")
        ):
            v_pred = model(x_t, t, c)

            # Normalize for cosine similarity
            v_pred_norm = F.normalize(v_pred, dim=-1)
            v_target_norm = F.normalize(v_target, dim=-1)

            loss_mse = F.mse_loss(v_pred, v_target)
            loss_cos = (
                1.0 - F.cosine_similarity(v_pred_norm, v_target_norm, dim=-1)
            ).mean()
            loss_l1 = F.l1_loss(v_pred, v_target)

            # Reconstruct x0 from v_pred to check if we can recover the original slate
            pred_x0 = _sqrt_a_t * x_t - _sqrt_1m_a_t * v_pred
            pred_x0_norm = F.normalize(pred_x0, dim=-1)
            x0_norm = F.normalize(x0, dim=-1)

            # Cosine similarity between predicted and true x0 (per item, then average)
            x0_cosine = F.cosine_similarity(
                pred_x0_norm, x0_norm, dim=-1
            ).mean()  # (B, K) -> (B,) -> scalar
            running_x0_cosine += x0_cosine.item()

            # If we have catalog, check if we can retrieve the correct items
            if (
                item_catalog_embs is not None and i == 0
            ):  # Only check first batch to save time
                catalog_norm = F.normalize(item_catalog_embs, dim=-1)
                # For each item in the slate, check if true item is in top-10 nearest neighbors
                sims = torch.matmul(pred_x0_norm, catalog_norm.T)  # (B, K, N)
                _, top_10_indices = torch.topk(
                    sims, k=min(10, item_catalog_embs.size(0)), dim=-1
                )
                # Note: This is a simplified check - we'd need to know the true indices to compute accuracy
                # For now, just track the cosine similarity

        running_loss_mse += loss_mse.item()
        running_loss_cos += loss_cos.item()
        running_loss_l1 += loss_l1.item()

    avg_loss_mse = running_loss_mse / len(val_loader)
    avg_loss_cos = running_loss_cos / len(val_loader)
    avg_loss_l1 = running_loss_l1 / len(val_loader)
    avg_x0_cosine = running_x0_cosine / len(val_loader)
    avg_total_loss = (
        avg_loss_mse + (avg_loss_cos * args.loss_cosine_weight) + 0.1 * avg_loss_l1
    )

    return avg_total_loss, avg_loss_mse, avg_loss_cos, avg_loss_l1, avg_x0_cosine


# === LR Scheduler Function ===
def get_scheduler(optimizer, args, total_steps):
    def lr_lambda(current_step):
        if current_step < args.lr_warmup_steps:
            return float(current_step) / float(max(1, args.lr_warmup_steps))
        progress = float(current_step - args.lr_warmup_steps) / float(
            max(1, total_steps - args.lr_warmup_steps)
        )
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# === Training Function ===
def train(args):
    device = torch.device(args.device)

    run = wandb.init(project=args.project_name, config=args)
    ckpt_name = f"{run.name}.pt"
    ckpt_path = os.path.join(args.ckpt_dir, ckpt_name)
    print(f"Weights & Biases run name: {run.name}")
    print(f"Checkpoint will be saved to: {ckpt_path}")

    print(f"Loading data from {args.data_path}...")
    full_data_dict = torch.load(args.data_path, map_location="cpu")

    # Validate data structure
    assert "slates" in full_data_dict, "Data file must contain 'slates' key"
    assert "prompts" in full_data_dict, "Data file must contain 'prompts' key"
    assert len(full_data_dict["slates"]) == full_data_dict["prompts"].size(0), (
        f"Mismatch: {len(full_data_dict['slates'])} slates but {full_data_dict['prompts'].size(0)} prompts"
    )

    # Check data normalization
    sample_prompt = full_data_dict["prompts"][0]
    prompt_norm = torch.norm(sample_prompt)
    print(
        f"Sample prompt norm: {prompt_norm:.4f} (expected ~{math.sqrt(args.item_dim):.4f} for normalized)"
    )

    sample_slate = full_data_dict["slates"][0]
    slate_norm = torch.norm(sample_slate)
    print(
        f"Sample slate norm per item: {slate_norm / sample_slate.size(0):.4f} (expected ~{math.sqrt(args.item_dim):.4f} for normalized)"
    )

    # Load Item Catalog for NN search
    if "item_catalog" in full_data_dict:
        item_catalog_embs = full_data_dict["item_catalog"].float().to(device)
        print(f"Loaded item catalog with {item_catalog_embs.size(0)} items.")
    else:
        print("Warning: 'item_catalog' tensor not found in data file.")
        print("Using a dummy catalog of 1000 items.")
        item_catalog_embs = torch.randn(1000, args.item_dim).to(device)

    # Load catalog index to film_id mapping
    catalog_index_to_film_id = full_data_dict.get("catalog_index_to_film_id", None)
    if catalog_index_to_film_id is None:
        print("Warning: 'catalog_index_to_film_id' mapping not found.")
        catalog_index_to_film_id = None

    # Load movie titles mapping - try movielens first, then fallback to big_list
    movie_titles = {}
    try:
        print("Loading movie titles mapping from movielens_movies_with_details.json...")
        movielens_movies = json.load(open("movielens_movies_with_details.json"))
        if isinstance(movielens_movies, list):
            for movie in movielens_movies:
                film_id = str(movie.get("film_id", ""))
                if film_id:
                    title = movie.get("title", "") or movie.get("movielens_title", "")
                    if title:
                        movie_titles[film_id] = title
        print(f"Loaded {len(movie_titles)} movie titles from movielens")
    except Exception as e:
        print(f"Warning: Could not load from movielens: {e}")
        try:
            print("Trying big_list_with_movies.json as fallback...")
            lists = json.load(open("big_list_with_movies.json"))
            for lst in lists:
                for movie in lst.get("movies", []):
                    film_id = movie.get("film_id")
                    if film_id and film_id not in movie_titles:
                        movie_titles[str(film_id)] = movie.get(
                            "title", f"Unknown (ID: {film_id})"
                        )
            print(f"Loaded {len(movie_titles)} movie titles from big_list")
        except Exception as e2:
            print(f"Warning: Could not load movie titles: {e2}")
            movie_titles = {}

    # Load prompt texts for display
    prompt_texts = full_data_dict.get("prompt_texts", None)
    if prompt_texts is None:
        print("Warning: 'prompt_texts' not found in data file.")
        prompt_texts = None

    full_dataset = SlateDataset(
        args.data_path,
        args.slate_len,
        args.item_dim,
        args.cfg_drop_prob,
        data=full_data_dict,  # Pass pre-loaded data
    )

    total_size = len(full_dataset)
    val_size = int(total_size * args.val_split)
    train_size = total_size - val_size

    print(f"Dataset size: {total_size} (Train: {train_size}, Val: {val_size})")

    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_dl = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=True,
    )
    val_dl = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
        pin_memory=True,
    )

    model = DiffusionSlateModel(
        dim=args.item_dim,
        hidden=args.hidden_dim,
        layers=args.layers,
        heads=args.heads,
        dropout=args.dropout,
        slate_len=args.slate_len,
        ctx_layers=args.ctx_layers,
        ctx_heads=args.ctx_heads,
    ).to(device)

    print(f"Model created. Input dim: {args.item_dim}, Hidden dim: {args.hidden_dim}")
    print(f"Total params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    wandb.watch(model, log="all", log_freq=100)

    opt = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    total_steps = len(train_dl) * args.epochs
    scheduler = get_scheduler(opt, args, total_steps)
    scaler = GradScaler(device.type)

    diff_vars = get_diffusion_vars(args.t_steps, device)
    sqrt_a_t = diff_vars["sqrt_alphas_cumprod"]
    sqrt_1m_a_t = diff_vars["sqrt_one_minus_alphas_cumprod"]
    T = args.t_steps

    best_val_loss = float("inf")
    os.makedirs(args.ckpt_dir, exist_ok=True)

    if args.resume_ckpt:
        print(f"Resuming from checkpoint: {args.resume_ckpt}")
        model.load_state_dict(torch.load(args.resume_ckpt, map_location=device))

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0

        loop = tqdm(train_dl, total=len(train_dl), desc=f"Epoch {epoch}/{args.epochs}")
        for i, (x0, c) in enumerate(loop):
            x0, c = x0.to(device, non_blocking=True), c.to(device, non_blocking=True)
            B = x0.size(0)
            t = torch.randint(0, T, (B,), device=device)

            _sqrt_a_t = sqrt_a_t[t].view(B, 1, 1)
            _sqrt_1m_a_t = sqrt_1m_a_t[t].view(B, 1, 1)

            noise = torch.randn_like(x0)
            x_t = _sqrt_a_t * x0 + _sqrt_1m_a_t * noise
            v_target = _sqrt_a_t * noise - _sqrt_1m_a_t * x0

            with torch.amp.autocast(
                device_type=device.type, enabled=(device.type == "cuda")
            ):
                v_pred = model(x_t, t, c)

                # Normalize for cosine similarity (more stable)
                v_pred_norm = F.normalize(v_pred, dim=-1)
                v_target_norm = F.normalize(v_target, dim=-1)

                loss_mse = F.mse_loss(v_pred, v_target)
                loss_cos = (
                    1.0 - F.cosine_similarity(v_pred_norm, v_target_norm, dim=-1)
                ).mean()
                loss_l1 = F.l1_loss(v_pred, v_target)

                # Reconstruct x0 from v_pred
                pred_x0 = _sqrt_a_t * x_t - _sqrt_1m_a_t * v_pred
                pred_x0_norm = F.normalize(pred_x0, dim=-1)
                x0_norm = F.normalize(x0, dim=-1)

                # Reconstruction loss: how well can we recover the original slate?
                loss_recon_cos = (
                    1.0 - F.cosine_similarity(pred_x0_norm, x0_norm, dim=-1)
                ).mean()
                loss_recon_mse = F.mse_loss(pred_x0, x0)

                # Catalog Retrieval Loss
                loss_catalog_retrieval = 0.0
                if item_catalog_embs is not None and args.catalog_loss_weight > 0:
                    # The ground truth items (x0) ARE in the catalog, so pred_x0 should be close to them
                    # This is already partially covered by loss_recon_cos, but we can make it stronger
                    # by ensuring we're actually close enough to retrieve them correctly

                    # For each predicted item, check if it's close enough to the ground truth item
                    # to actually retrieve it from the catalog (accounting for catalog size)
                    item_similarity = F.cosine_similarity(
                        pred_x0_norm, x0_norm, dim=-1
                    )  # (B, K)
                    # We want this to be very high (close to 1.0) to ensure correct retrieval
                    # With 363k items, even a small error can map to wrong items
                    loss_catalog_retrieval = (1.0 - item_similarity).mean()

                    # Additionally: Ensure pred_x0 items are similar to the PROMPT (semantic alignment)
                    # The prompt should be similar to items in the slate
                    # This helps bridge the gap between list descriptions and movie titles
                    c_expanded = c.unsqueeze(1).expand(
                        -1, args.slate_len, -1
                    )  # (B, K, D)
                    prompt_item_similarity = F.cosine_similarity(
                        pred_x0_norm, c_expanded, dim=-1
                    )  # (B, K)
                    # We want items to be somewhat similar to the prompt (not too strong, as items != prompt)
                    # But this helps ensure semantic coherence
                    loss_catalog_retrieval = (
                        loss_catalog_retrieval
                        + 0.3 * (1.0 - prompt_item_similarity).mean()
                    )

                # Combined loss with reconstruction and catalog components
                loss = (
                    loss_mse
                    + args.loss_cosine_weight * loss_cos
                    + 0.1 * loss_l1
                    + 0.5 * loss_recon_cos
                    + 0.3 * loss_recon_mse
                    + args.catalog_loss_weight * loss_catalog_retrieval
                )

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()

            scheduler.step()
            global_step += 1

            running_loss += loss.item()
            loop.set_postfix(loss=loss.item())

            if global_step % args.log_interval == 0:
                log_dict = {
                    "batch_loss_total": loss.item(),
                    "batch_loss_mse": loss_mse.item(),
                    "batch_loss_cos": loss_cos.item(),
                    "global_step": global_step,
                    "lr": scheduler.get_last_lr()[0],
                }
                if item_catalog_embs is not None and args.catalog_loss_weight > 0:
                    log_dict["batch_loss_catalog"] = loss_catalog_retrieval.item()
                wandb.log(log_dict)

        avg_train_loss = running_loss / len(train_dl)

        avg_val_loss, avg_val_mse, avg_val_cos, avg_val_l1, avg_x0_cosine = validate(
            model, val_dl, diff_vars, args, device, item_catalog_embs=item_catalog_embs
        )

        wandb.log(
            {
                "epoch": epoch,
                "train_loss": avg_train_loss,
                "val_loss_total": avg_val_loss,
                "val_loss_mse": avg_val_mse,
                "val_loss_cos": avg_val_cos,
                "val_loss_l1": avg_val_l1,
                "val_x0_cosine": avg_x0_cosine,  # How well can we reconstruct the original slate?
                "gpu_mem_MB": torch.cuda.memory_allocated() / 1e6
                if torch.cuda.is_available()
                else 0,
            },
            step=epoch,
        )

        print(
            f"Epoch {epoch}/{args.epochs}  TrainLoss: {avg_train_loss:.6f}  ValLoss: {avg_val_loss:.6f}"
        )

        # === Generate and display sample slate after validation ===
        if epoch % 5 == 0 or epoch == 1:  # Show every 5 epochs or on first epoch
            model.eval()
            with torch.no_grad():
                # Get a sample prompt from validation set
                val_idx = epoch % len(val_dataset)
                val_sample_x0, val_sample_c = val_dataset[val_idx]
                val_sample_c = val_sample_c.unsqueeze(0).to(device)

                # Get original prompt text
                prompt_text = "Unknown prompt"
                if prompt_texts:
                    # Get the actual index in the full dataset
                    actual_idx = (
                        val_dataset.indices[val_idx]
                        if hasattr(val_dataset, "indices")
                        else val_idx
                    )
                    if actual_idx < len(prompt_texts):
                        prompt_text = prompt_texts[actual_idx]

                # Generate slate
                generated_slate_indices = generate(
                    model,
                    val_sample_c,
                    item_catalog_embs,
                    diff_vars,
                    args,
                    device,
                    num_inference_steps=50,
                )

                # Map indices to film_ids and titles
                if catalog_index_to_film_id and generated_slate_indices:
                    print(f"\n{'=' * 70}")
                    print(f"Sample Generation (Epoch {epoch}):")
                    print(f"{'=' * 70}")
                    print(f'Prompt: "{prompt_text}"')
                    print("\nGenerated Slate:")
                    movie_list = []
                    for i, idx in enumerate(generated_slate_indices[0], 1):
                        if idx < len(catalog_index_to_film_id):
                            film_id = catalog_index_to_film_id[idx]
                            # Ensure film_id is string for consistent lookup
                            film_id_str = str(film_id)
                            title = movie_titles.get(
                                film_id_str,
                                movie_titles.get(film_id, f"Unknown (ID: {film_id})"),
                            )
                            movie_list.append(title)
                            print(f"  {i}. {title}")
                        else:
                            movie_list.append(f"[Invalid index: {idx}]")
                            print(f"  {i}. [Invalid index: {idx}]")
                    # Also print as comma-separated list
                    print(f"\nSlate: {', '.join(movie_list)}")
                    print(f"{'=' * 70}\n")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), ckpt_path)
            wandb.save(ckpt_path)
            print(
                f"💾 New best model saved (val_loss {best_val_loss:.6f}) to {ckpt_path}"
            )

    wandb.finish()
    print("Training finished.")

    print("\n--- Running Generation Example ---")
    print("Loading best model for final generation...")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))

    # Get a sample prompt from the validation set
    val_sample_x0, val_sample_c = val_dataset[0]  # Get first item
    val_sample_c = val_sample_c.unsqueeze(0).to(device)
    print(f"Using validation sample prompt embedding (shape: {val_sample_c.shape})")

    start_gen_time = time.time()
    generated_slate_indices = generate(
        model,
        val_sample_c,
        item_catalog_embs,  # Pass the loaded catalog
        diff_vars,
        args,
        device,
        num_inference_steps=50,
    )
    end_gen_time = time.time()

    print(
        f"\n✅ Successfully generated slate indices in {end_gen_time - start_gen_time:.4f}s."
    )
    print(f"Generated slate (item indices): {generated_slate_indices[0]}")

    # Save indices to a text file
    output_txt_path = "generated_slate_example.txt"
    with open(output_txt_path, "w") as f:
        for indices in generated_slate_indices:
            f.write(",".join(map(str, indices)) + "\n")
    print(f"Saved generated indices to {output_txt_path}")


# === ARGS updated to match DMSG paper ===
def get_args():
    parser = argparse.ArgumentParser(description="Train DMSG-Style Diffusion Model")

    # Data
    parser.add_argument("--data_path", type=str, default="train_data_split.pt")
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--num_workers", type=int, default=0)

    # Model
    parser.add_argument(
        "--item_dim",
        type=int,
        default=384,
        help="Original item dim (all-MiniLM-L6-v2 384)",
    )
    parser.add_argument("--slate_len", type=int, default=10)
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=640,
        help="Paper's hidden dim (10 heads * 32 dim)",
    )
    parser.add_argument("--layers", type=int, default=6, help="Paper's layer count")
    parser.add_argument("--heads", type=int, default=10, help="Paper's head count")
    parser.add_argument(
        "--ctx_layers", type=int, default=3, help="Paper's context encoder layer count"
    )
    parser.add_argument(
        "--ctx_heads", type=int, default=8, help="Paper's context encoder head count"
    )

    parser.add_argument("--dropout", type=float, default=0.2)

    # Training
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--resume_ckpt", type=str, default=None)

    # Diffusion
    parser.add_argument("--t_steps", type=int, default=1000)
    parser.add_argument("--cfg_drop_prob", type=float, default=0.1)
    parser.add_argument(
        "--cfg_scale",
        type=float,
        default=3.0,
        help="CFG scale (reduced from 4.0 for better stability)",
    )
    parser.add_argument(
        "--nn_temperature",
        type=float,
        default=1.0,
        help="Temperature for nearest neighbor search (1.0 = no temperature)",
    )

    # Loss
    parser.add_argument(
        "--loss_cosine_weight",
        type=float,
        default=0.5,
        help="Weight for cosine loss (reduced from 1.0)",
    )
    parser.add_argument(
        "--catalog_loss_weight",
        type=float,
        default=1.0,
        help="Weight for catalog retrieval loss (ensures generated embeddings are close to catalog items)",
    )

    # Logging
    parser.add_argument(
        "--project_name", type=str, default="movie-diffusion-v11-arch-fix"
    )
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints")

    return parser.parse_args()


if __name__ == "__main__":
    import sys

    # Handle notebook environments
    if "ipykernel" in sys.modules:
        print("Running in notebook-like env, using default args.")
        sys.argv = [sys.argv[0]]

    args = get_args()

    print("Using args:")
    print(args)

    train(args)
