import argparse
import json
import os
from pathlib import Path

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


@torch.no_grad()
def infer(args):
    config = yaml.load(open(args.config, "r"), Loader=yaml.Loader)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    tokenizer = RobertaTokenizerFast.from_pretrained(args.text_encoder)
    model = CSCL(args=None, config=config).to(device)
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        candidate = Path.cwd() / checkpoint_path.name
        if candidate.exists():
            checkpoint_path = candidate
        else:
            raise FileNotFoundError(
                f"Checkpoint not found: {args.checkpoint}\n"
                f"Try: {candidate}"
            )
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    image = Image.open(args.image).convert("RGB")
    image_tensor = build_transform(config["image_res"])(image).unsqueeze(0).to(device)

    text = args.text if args.text is not None else ""
    text_input = tokenizer(
        [text],
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
    pred = "fake" if fake_prob >= args.threshold else "real"

    multicls = (logits_multicls[0] >= 0).int().tolist()
    bbox = output_coord[0].detach().cpu().tolist()

    result = {
        "image": args.image,
        "text": text,
        "prediction": pred,
        "fake_probability": round(fake_prob, 6),
        "real_probability": round(real_prob, 6),
        "threshold": args.threshold,
        "multiclass_flags": {
            "face_swap": int(multicls[0]),
            "face_attribute": int(multicls[1]),
            "text_swap": int(multicls[2]),
            "text_attribute": int(multicls[3]),
        },
        "pred_box_cxcywh_norm": [round(float(x), 6) for x in bbox],
    }
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
