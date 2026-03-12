# utils/clip_metrics.py
import torch
import numpy as np
from PIL import Image
import clip

_DEVICE = "cpu"  # keep CPU by default for your GTX 1060; change to "cuda" if you install a torch build that supports sm_61

def to_pil_uint8(img_f01: np.ndarray) -> Image.Image:
    arr = (np.clip(img_f01, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    return Image.fromarray(arr)

class CLIPHandle:
    def __init__(self, model_name="ViT-B/32", device=None):
        self.device = device or _DEVICE
        self.model, self.preprocess = clip.load(model_name, device=self.device, jit=False)
        self.model.eval()

    @torch.no_grad()
    def batch_image_features(self, imgs_f01, batch_size=32):
        """
        imgs_f01: list[np.ndarray] each [H,W,C] float in [0,1]
        returns (N, D) L2-normalized features on CPU
        """
        # Preprocess to a tensor batch
        tensors = []
        for im in imgs_f01:
            pil = to_pil_uint8(im)
            tensors.append(self.preprocess(pil))
        batch = torch.stack(tensors, dim=0)

        feats = []
        for i in range(0, len(imgs_f01), batch_size):
            chunk = batch[i:i+batch_size].to(self.device, non_blocking=True)
            f = self.model.encode_image(chunk)
            f = f / f.norm(dim=-1, keepdim=True)
            feats.append(f.cpu())
        return torch.cat(feats, dim=0).numpy()  # (N, D)

def clip_cosine_similarities(imgs_A_f01, imgs_B_f01, handle: CLIPHandle, batch_size=32):
    """Compute cosine sims between pairs (A[i], B[i]). Returns np.array of shape (N,)."""
    assert len(imgs_A_f01) == len(imgs_B_f01)
    # Compute features in two batches (faster & simpler than interleaving)
    feats_A = handle.batch_image_features(imgs_A_f01, batch_size=batch_size)  # (N, D)
    feats_B = handle.batch_image_features(imgs_B_f01, batch_size=batch_size)  # (N, D)
    # Both are L2-normalized already -> cosine sim is rowwise dot
    sims = np.sum(feats_A * feats_B, axis=1)
    return sims  # values in [-1, 1], 1 is best


def to_pil_uint8(img_float01: np.ndarray) -> Image.Image:
    # img_float01 is (H,W,C) in [0,1]; convert safely to uint8
    arr = (np.clip(img_float01, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    return Image.fromarray(arr)

def clip_image_similarity(orig_img_f01: np.ndarray, rec_img_f01: np.ndarray, handle: CLIPHandle) -> float:
    """
    Returns cosine similarity in [-1, 1], where 1 is best.
    """
    pil_o = to_pil_uint8(orig_img_f01)
    pil_r = to_pil_uint8(rec_img_f01)
    f_o = handle.image_features(pil_o)
    f_r = handle.image_features(pil_r)
    # cosine similarity (already normalized)
    return float((f_o * f_r).sum())

# OPTIONAL: zero-shot label agreement using CLIP text prompts
@torch.no_grad()
def clip_zero_shot_label(orig_img_f01, rec_img_f01, class_names, handle: CLIPHandle):
    """
    Returns (orig_top1, rec_top1, agree_bool, orig_conf, rec_conf)
    class_names: list[str] like ["airplane","automobile",...]
    """
    pil_o = to_pil_uint8(orig_img_f01)
    pil_r = to_pil_uint8(rec_img_f01)

    # Build prompts once outside if you call many times
    # Simple templates; you can expand this list for better accuracy
    templates = [ "a photo of a {}.", "a blurry photo of a {}.", "a close-up photo of a {}." ]
    texts = []
    for cls in class_names:
        for t in templates:
            texts.append(t.format(cls))
    device = handle.device
    text_tokens = clip.tokenize(texts).to(device)

    img_o = handle.preprocess(pil_o).unsqueeze(0).to(device)
    img_r = handle.preprocess(pil_r).unsqueeze(0).to(device)

    model = handle.model
    text_feats = model.encode_text(text_tokens)
    text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)

    img_feats_o = model.encode_image(img_o)
    img_feats_o = img_feats_o / img_feats_o.norm(dim=-1, keepdim=True)

    img_feats_r = model.encode_image(img_r)
    img_feats_r = img_feats_r / img_feats_r.norm(dim=-1, keepdim=True)

    # average the template logits per class
    with torch.no_grad():
        logits_o = img_feats_o @ text_feats.T  # (1, num_templates*classes)
        logits_r = img_feats_r @ text_feats.T

    num_cls = len(class_names)
    logits_o = logits_o.view(1, num_cls, -1).mean(dim=-1)  # (1, num_cls)
    logits_r = logits_r.view(1, num_cls, -1).mean(dim=-1)

    o_top = int(torch.argmax(logits_o, dim=-1).item())
    r_top = int(torch.argmax(logits_r, dim=-1).item())
    o_conf = float(torch.softmax(logits_o, dim=-1).max().item())
    r_conf = float(torch.softmax(logits_r, dim=-1).max().item())
    return class_names[o_top], class_names[r_top], (o_top == r_top), o_conf, r_conf
