import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Tuple

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torchvision import transforms
from transformers import RobertaTokenizerFast

from models.CSCL import CSCL


def text_input_adjust(text_input, fake_word_pos, device):
    input_ids_remove_sep = [x[:-1] for x in text_input.input_ids]
    maxlen = max([len(x) for x in text_input.input_ids]) - 1
    input_ids_remove_sep_pad = [x + [0] * (maxlen - len(x)) for x in input_ids_remove_sep]
    text_input.input_ids = torch.LongTensor(input_ids_remove_sep_pad).to(device)

    attention_mask_remove_sep = [x[:-1] for x in text_input.attention_mask]
    attention_mask_remove_sep_pad = [x + [0] * (maxlen - len(x)) for x in attention_mask_remove_sep]
    text_input.attention_mask = torch.LongTensor(attention_mask_remove_sep_pad).to(device)

    fake_token_pos_batch = []
    for i in range(len(fake_word_pos)):
        fake_token_pos = []
        fake_word_pos_decimal = torch.where(fake_word_pos[i] == 1)[0].tolist()
        subword_idx = text_input.word_ids(i)
        subword_idx_rm_clssep = subword_idx[1:-1]
        subword_idx_rm_clssep_array = torch.tensor(
            [-1 if x is None else x for x in subword_idx_rm_clssep], dtype=torch.long
        ).numpy()
        for word_pos in fake_word_pos_decimal:
            token_pos = (subword_idx_rm_clssep_array == word_pos).nonzero()[0].tolist()
            fake_token_pos.extend(token_pos)
        fake_token_pos_batch.append(fake_token_pos)

    return text_input, fake_token_pos_batch


def build_transform(image_res):
    normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    return transforms.Compose(
        [
            transforms.Resize((image_res, image_res), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            normalize,
        ]
    )


def resolve_device(device: str, allow_cuda_fallback: bool = False) -> Tuple[torch.device, str]:
    normalized = (device or "auto").lower()

    if normalized == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda"), "Auto selected CUDA device."
        return torch.device("cpu"), "CUDA is unavailable; fell back to CPU."

    if normalized == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda"), "Using CUDA device."
        if allow_cuda_fallback:
            return torch.device("cpu"), "CUDA is unavailable; fell back to CPU."
        raise RuntimeError("CUDA is unavailable. Please choose 'auto' or 'cpu'.")

    if normalized == "cpu":
        return torch.device("cpu"), "Using CPU device."

    raise ValueError(f"Unsupported device '{device}'. Expected one of: auto/cuda/cpu.")


def resolve_checkpoint_path(checkpoint: str) -> Path:
    checkpoint_path = Path(checkpoint)
    if checkpoint_path.exists():
        return checkpoint_path

    candidate = Path.cwd() / checkpoint_path.name
    if candidate.exists():
        return candidate

    raise FileNotFoundError(
        f"Checkpoint not found: {checkpoint}\n"
        f"Try: {candidate}"
    )


def load_inference_runtime(
    checkpoint: str,
    config_path: str = "./configs/test.yaml",
    text_encoder: str = "./roberta-base",
    device: str = "cuda",
    allow_cuda_fallback: bool = False,
) -> Dict[str, Any]:
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    text_encoder_path = Path(text_encoder)
    if not text_encoder_path.exists():
        raise FileNotFoundError(f"Text encoder path not found: {text_encoder}")

    config = yaml.load(open(config_file, "r"), Loader=yaml.Loader)
    runtime_device, device_message = resolve_device(device, allow_cuda_fallback=allow_cuda_fallback)

    tokenizer = RobertaTokenizerFast.from_pretrained(text_encoder)
    model = CSCL(args=None, config=config).to(runtime_device)

    checkpoint_path = resolve_checkpoint_path(checkpoint)
    checkpoint_data = torch.load(str(checkpoint_path), map_location="cpu")
    state_dict = checkpoint_data["model"] if "model" in checkpoint_data else checkpoint_data
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    return {
        "model": model,
        "tokenizer": tokenizer,
        "transform": build_transform(config["image_res"]),
        "config": config,
        "device": runtime_device,
        "device_message": device_message,
        "checkpoint": str(checkpoint_path),
        "config_path": str(config_file),
        "text_encoder": text_encoder,
    }


@torch.no_grad()
def run_single_inference(
    runtime: Dict[str, Any],
    image: Image.Image,
    text: str = "",
    threshold: float = 0.5,
    image_name: str = "",
) -> Dict[str, Any]:
    model = runtime["model"]
    tokenizer = runtime["tokenizer"]
    transform = runtime["transform"]
    config = runtime["config"]
    device = runtime["device"]

    image_rgb = image.convert("RGB")
    image_tensor = transform(image_rgb).unsqueeze(0).to(device)

    user_text = text if text is not None else ""
    text_for_model = user_text if user_text.strip() else " "
    text_input = tokenizer(
        [text_for_model],
        max_length=128,
        truncation=True,
        add_special_tokens=True,
        return_attention_mask=True,
        return_token_type_ids=False,
    )

    fake_word_pos = torch.zeros((1, config["max_words"]))
    text_input, fake_token_pos = text_input_adjust(text_input, fake_word_pos, device)

    fake_image_box = torch.zeros((1, 4), dtype=torch.float32, device=device)
    labels = ["orig"]

    logits_real_fake, logits_multicls, output_coord, _ = model(
        image_tensor, labels, text_input, fake_image_box, fake_token_pos, is_train=False
    )

    prob = F.softmax(logits_real_fake, dim=1)[0]
    fake_prob = float(prob[1].item())
    real_prob = float(prob[0].item())
    pred = "fake" if fake_prob >= threshold else "real"

    multicls = (logits_multicls[0] >= 0).int().tolist()
    bbox = output_coord[0].detach().cpu().tolist()

    return {
        "image": image_name,
        "text": user_text,
        "prediction": pred,
        "fake_probability": round(fake_prob, 6),
        "real_probability": round(real_prob, 6),
        "threshold": threshold,
        "multiclass_flags": {
            "face_swap": int(multicls[0]),
            "face_attribute": int(multicls[1]),
            "text_swap": int(multicls[2]),
            "text_attribute": int(multicls[3]),
        },
        "pred_box_cxcywh_norm": [round(float(x), 6) for x in bbox],
    }


@torch.no_grad()
def infer(args):
    runtime = load_inference_runtime(
        checkpoint=args.checkpoint,
        config_path=args.config,
        text_encoder=args.text_encoder,
        device=args.device,
        allow_cuda_fallback=True,
    )

    image = Image.open(args.image).convert("RGB")
    result = run_single_inference(
        runtime=runtime,
        image=image,
        text=args.text,
        threshold=args.threshold,
        image_name=args.image,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, type=str)
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--text", default="", type=str)
    parser.add_argument("--config", default="./configs/test.yaml", type=str)
    parser.add_argument("--text_encoder", default="./roberta-base", type=str)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--threshold", default=0.5, type=float)
    return parser.parse_args()


if __name__ == "__main__":
    infer(parse_args())
