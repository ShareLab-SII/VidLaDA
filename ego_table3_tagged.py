import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from decord import VideoReader, cpu
from huggingface_hub import hf_hub_download, snapshot_download
from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import tokenizer_image_token
from llava.model.builder import load_pretrained_model
from lmms_eval.tasks.egoschema.utils import egoschema_process_results_generation

parser = argparse.ArgumentParser()
parser.add_argument("--limit", type=int, default=3)
parser.add_argument("--out", type=Path, default=Path("runs/ego-table3-tagged96"))
args = parser.parse_args()
if not 1 <= args.limit <= 500:
    parser.error("--limit must be between 1 and 500")

data = Path(os.environ["HF_HOME"]) / "lmms-lab" / "egoschema"
parquet = hf_hub_download(
    "lmms-eval/egoschema", "Subset/test-00000-of-00001.parquet",
    repo_type="dataset", local_dir=data,
)
rows = pq.read_table(parquet).to_pylist()
answers = json.loads((data / "subset_answers.json").read_text())
assert len(rows) == len(answers) == 500
assert {row["video_idx"] for row in rows} == set(answers)
assert all(int(row["answer"]) == int(answers[row["video_idx"]]) for row in rows)
rows = rows[:args.limit]

videos = [f"videos/{row['video_idx']}.mp4" for row in rows]
missing = [name for name in videos if not (data / name).is_file()]
if missing:
    snapshot_download(
        "yanlaiy/EgoSchema", repo_type="dataset",
        revision="f0cd655ce94132b7accb9e40b3c192b8a62e1629",
        allow_patterns=missing, local_dir=data, max_workers=4,
    )
assert all((data / name).is_file() for name in videos)
args.out.mkdir(parents=True, exist_ok=True)

tokenizer, model, processor, _ = load_pretrained_model(
    "ziHoHe/VidLaDA-8B", None, "llava_vidlada",
    device_map="cuda:0", torch_dtype="float16", attn_implementation="sdpa",
    overwrite_config={
        "mm_spatial_pool_stride": 2, "mm_spatial_pool_mode": "bilinear",
        "mm_newline_position": "grid", "use_cache": False,
    },
)
model.eval()
model.config.use_cache = False
torch.manual_seed(0)

decoder_lengths, head_lengths = [], []
model.get_model().register_forward_pre_hook(
    lambda _, args, kwargs: decoder_lengths.append(kwargs["inputs_embeds"].shape[1]),
    with_kwargs=True,
)
model.lm_head.register_forward_pre_hook(
    lambda _, args: head_lengths.append(args[0].shape[-2]),
)
cfg = model.config
d, m = cfg.hidden_size, cfg.intermediate_size
d_kv = d * cfg.num_key_value_heads // cfg.num_attention_heads

for row, video in zip(rows, videos):
    uid = row["video_idx"]
    path = args.out / f"{uid}.json"
    if path.is_file():
        continue

    reader = VideoReader(str(data / video), ctx=cpu(0), num_threads=1)
    indices = np.arange(0, len(reader), max(1, round(reader.get_avg_fps())))
    if len(indices) > 32:
        indices = indices[np.linspace(0, len(indices) - 1, 32, dtype=int)]
    frames = reader.get_batch(indices.tolist()).asnumpy()
    pixels = processor.preprocess(frames, return_tensors="pt")["pixel_values"]
    pixels = pixels.to(device="cuda:0", dtype=torch.float16)

    question = "\n".join([row["question"], *row["option"]])
    instruction = "Reason briefly inside <think>...</think>. Then give only the option letter.\n/think"
    conv = conv_templates["llava_llada"].copy()
    conv.tokenizer = tokenizer
    conv.append_message(conv.roles[0], f"{DEFAULT_IMAGE_TOKEN}\n{question}\n{instruction}")
    conv.append_message(conv.roles[1], None)
    ids = tokenizer_image_token(
        conv.get_prompt(), tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt",
    )[None].to("cuda:0")

    answer_start = 96
    opening = tokenizer.encode("<think>\n", add_special_tokens=False)
    closing = tokenizer.encode("\n</think>\n", add_special_tokens=False)
    
    context_ids = torch.full((1, 128), 126336, device=ids.device, dtype=torch.long)
    context_ids[0, :len(opening)] = torch.tensor(opening, device=ids.device)
    context_ids[0, answer_start - len(closing):answer_start] = torch.tensor(
        closing, device=ids.device
    )
    expected_forwards = 128 - len(opening) - len(closing)

    decoder_lengths.clear()
    head_lengths.clear()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    with torch.inference_mode():
        generated = model.generate(
            ids, images=[pixels], modalities=["video"],
            gen_length=128, steps=128, block_length=32,
            temperature=0.0, cfg_scale=0.0, remasking="low_confidence",
            generation_context_ids=context_ids,
        )
    torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    peak_gib = torch.cuda.max_memory_allocated() / 2**30
    assert len(decoder_lengths) == len(head_lengths) == expected_forwards

    raw = tokenizer.decode(generated[0], skip_special_tokens=False)
    tail = tokenizer.decode(generated[0, answer_start:], skip_special_tokens=False)
    answer = tail.split(conv.sep, 1)[0].strip()
    random.seed(uid)
    pred = egoschema_process_results_generation(row, [answer])["score"]["pred"] if answer else -1

    flops = cfg.num_hidden_layers * sum(
        4 * n * d * (d + d_kv) + 4 * n * n * d + 6 * n * d * m
        for n in decoder_lengths
    )
    flops += 2 * d * model.lm_head.out_features * sum(head_lengths)
    record = {
        "uid": uid, "frames": len(indices), "frame_indices": indices.tolist(),
        "forward_count": len(decoder_lengths),
        "raw": raw, "answer_span": answer, "pred": pred, "gold": int(row["answer"]),
        "seconds": seconds, "peak_gib": peak_gib, "flops": flops,
    }
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)
    print(uid, f"frames={len(indices)}", f"forwards={len(decoder_lengths)}",
          f"{seconds:.1f}s", f"{peak_gib:.2f} GiB", repr(answer[:80]), flush=True)

records = [
    json.loads((args.out / f"{row['video_idx']}.json").read_text(encoding="utf-8"))
    for row in rows
]
correct = sum(r["pred"] == r["gold"] for r in records)
print(f"Score: {100 * correct / len(records):.2f}% ({correct}/{len(records)})")
print(f"Empty answer spans: {sum(not r['answer_span'] for r in records)}")
print(f"Baseline FLOPs: {sum(r['flops'] for r in records):,}")
print(f"Mean generation time: {sum(r['seconds'] for r in records) / len(records):.1f} s")
