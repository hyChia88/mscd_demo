"""
MSCD VLM LoRA Training — Modal GPU Training Script

Fine-tunes Qwen2.5-VL-7B-Instruct with LoRA for multimodal constraints
extraction (site photos + floorplans + chat → JSON constraints with spatial_relations).

LoRA_3 version: trains on synth_v0.5 dataset (1,111 train / 19 test) with
spatial triplet extraction (FILLS, ADJACENT_TO, CONTINUOUS predicates).

Based on Unsloth's official Qwen2.5-VL vision fine-tuning notebook:
  Qwen2_5_VL_(7B)_Vision.ipynb

Usage:
    modal run training/train.py
    modal run training/train.py --epochs 5 --lr 1e-4 --lora-r 32

    # Download trained adapter after training:
    modal volume get mscd-checkpoints /mscd-lora-v3/final ./models/adapters/v3_lora_qwen

Requires:
    pip install modal
    modal setup         # One-time auth
    modal secret create wandb-secret WANDB_API_KEY=<your-key>
"""

import dataclasses
import json
import os
from pathlib import Path

import modal

# ── Local data paths ─────────────────────────────────────────────────────────

DATA_ROOT    = Path(__file__).parent.parent.parent / "data_curation"
DATASETS_DIR = DATA_ROOT / "datasets"

# synth_v0.5 — LoRA_3 training data (1,111 train / 19 test)
V05_DIR     = DATASETS_DIR / "synth_v0.5" / "train"
TRAIN_JSONL = V05_DIR / "lora3_train.jsonl"
TEST_JSONL  = V05_DIR / "lora3_test.jsonl"

# v0.5 image directories (site photos + floorplans only — global renders are pipeline artifacts)
V05_SITE_IMGS_DIR  = DATASETS_DIR / "synth_v0.5" / "imgs"
V05_FLOORPLANS_DIR = DATASETS_DIR / "synth_v0.5" / "floorplans"

# v0.4 image directories (site photos + floorplans for enriched records)
AP_IMGS_DIR   = DATASETS_DIR / "synth_v0.4_ap"  / "cases" / "imgs"
AP_PLANS_DIR  = DATASETS_DIR / "synth_v0.4_ap"  / "cases" / "plans"
BH_IMGS_DIR   = DATASETS_DIR / "synth_v0.4_bh"  / "cases" / "imgs"
BH_PLANS_DIR  = DATASETS_DIR / "synth_v0.4_bh"  / "cases" / "plans"
DXA_IMGS_DIR  = DATASETS_DIR / "synth_v0.4_dxa" / "cases" / "imgs"
DXA_PLANS_DIR = DATASETS_DIR / "synth_v0.4_dxa" / "cases" / "plans"

# ── Modal infrastructure ─────────────────────────────────────────────────────

app = modal.App("mscd-vlm-lora3-train")

# Container image — matches Unsloth's official Qwen2.5-VL notebook install order:
#   1. Install unsloth ecosystem
#   2. Override transformers & trl to known-good vision versions
train_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    # 1. Install unsloth from PyPI to pull ALL transitive deps
    #    (torch, torchvision, peft, bitsandbytes, accelerate, xformers, triton, etc.)
    .pip_install(
        "unsloth",
        "qwen-vl-utils",
        "wandb",
        "datasets==4.3.0",
        "hf-transfer",
    )
    # 2. Override unsloth with latest git (has VARIANT_KWARG_KEYS fix for peft>=0.18)
    .run_commands("pip install --no-deps --force-reinstall 'unsloth @ git+https://github.com/unslothai/unsloth.git'")
    # 3. Pin transformers & trl to known-good versions for vision finetuning
    .pip_install("transformers==4.56.2")
    .run_commands("pip install --no-deps trl==0.22.2")
    .env({"HF_HOME": "/model_cache"})
    # Bake v0.5 training JSONL files
    .add_local_file(str(TRAIN_JSONL), remote_path="/data/train/lora3_train.jsonl")
    .add_local_file(str(TEST_JSONL),  remote_path="/data/train/lora3_test.jsonl")
    # Bake v0.5 image directories (site photos + floorplans only)
    .add_local_dir(str(V05_SITE_IMGS_DIR),  remote_path="/data/images/v05/imgs")
    .add_local_dir(str(V05_FLOORPLANS_DIR), remote_path="/data/images/v05/floorplans")
    # Bake v0.4 image directories (for enriched records that still reference v0.4 paths)
    .add_local_dir(str(AP_IMGS_DIR),   remote_path="/data/images/ap/imgs")
    .add_local_dir(str(AP_PLANS_DIR),  remote_path="/data/images/ap/plans")
    .add_local_dir(str(BH_IMGS_DIR),   remote_path="/data/images/bh/imgs")
    .add_local_dir(str(BH_PLANS_DIR),  remote_path="/data/images/bh/plans")
    .add_local_dir(str(DXA_IMGS_DIR),  remote_path="/data/images/dxa/imgs")
    .add_local_dir(str(DXA_PLANS_DIR), remote_path="/data/images/dxa/plans")
)

# Persistent volumes
model_cache = modal.Volume.from_name("mscd-model-cache", create_if_missing=True)
checkpoint_vol = modal.Volume.from_name("mscd-checkpoints", create_if_missing=True)


# ── Training config ──────────────────────────────────────────────────────────

@dataclasses.dataclass
class TrainConfig:
    # Model
    model_name: str = "unsloth/Qwen2.5-VL-7B-Instruct-bnb-4bit"

    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0

    # Training
    epochs: int = 5
    batch_size: int = 2
    grad_accum: int = 8        # effective batch = 16
    lr: float = 2e-4
    warmup_steps: int = 10
    max_seq_length: int = 4096  # up from 2048 — multiple images per sample
    weight_decay: float = 0.01

    # Evaluation & checkpoints
    eval_steps: int = 10
    save_total_limit: int = 5   # keep best + last few

    # Wandb
    wandb_project: str = "mscd-vlm-lora"
    wandb_run: str = "qwen25vl-7b-r16-lora3-synth_v05"

    # Paths (inside Modal container)
    train_file: str = "/data/train/lora3_train.jsonl"
    test_file: str = "/data/train/lora3_test.jsonl"
    output_dir: str = "/checkpoints/mscd-lora-v3"

    seed: int = 42


# ── Path remapping ───────────────────────────────────────────────────────────

# Maps local dataset prefix → container image prefix under /data/images/<tag>/
_LOCAL_ROOT  = "file:///root/cmu/master_thesis/data_curation/datasets/"
_REMOTE_ROOT = "file:///data/images/"
_DATASET_MAP = {
    # v0.5 image paths (site photos + floorplans)
    "synth_v0.5/imgs/":           "v05/imgs/",
    "synth_v0.5/floorplans/":     "v05/floorplans/",
    # v0.4 image paths (for enriched records)
    "synth_v0.4_ap/cases/":       "ap/",
    "synth_v0.4_bh/cases/":       "bh/",
    "synth_v0.4_dxa/cases/":      "dxa/",
}

def remap_image_paths(sample: dict, config: TrainConfig) -> dict:
    """Remap local absolute paths to Modal container paths.

    e.g. file:///root/.../synth_v0.4_ap/cases/plans/plan_XXX.png
      →  file:///data/images/ap/plans/plan_XXX.png
    """
    for msg in sample.get("messages", []):
        content = msg.get("content")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "image":
                    path = part["image"]
                    if path.startswith(_LOCAL_ROOT):
                        rel = path[len(_LOCAL_ROOT):]  # e.g. "synth_v0.4_ap/cases/plans/plan_XXX.png"
                        for ds_prefix, remote_prefix in _DATASET_MAP.items():
                            if rel.startswith(ds_prefix):
                                rel = remote_prefix + rel[len(ds_prefix):]  # "ap/plans/plan_XXX.png"
                                break
                        part["image"] = _REMOTE_ROOT + rel  # "file:///data/images/ap/plans/plan_XXX.png"
    return sample


def load_and_remap(jsonl_path: str, config: TrainConfig) -> list:
    """Load JSONL and remap image paths. Returns raw Python list (not HF Dataset).

    Unsloth's UnslothVisionDataCollator expects a plain list of dicts,
    NOT a HuggingFace Dataset object. See notebook cell-26.
    """
    samples = []
    with open(jsonl_path) as f:
        for line in f:
            sample = json.loads(line)
            sample = remap_image_paths(sample, config)
            samples.append(sample)
    return samples


# ── Training function ────────────────────────────────────────────────────────

@app.function(
    image=train_image,
    gpu="A100",
    volumes={
        "/model_cache": model_cache,
        "/checkpoints": checkpoint_vol,
    },
    secrets=[modal.Secret.from_name("wandb-secret")],
    timeout=4 * 60 * 60,  # 4 hours
)
def train(
    epochs: int = 3,
    lr: float = 2e-4,
    lora_r: int = 16,
    lora_alpha: int = 32,
    batch_size: int = 2,
    grad_accum: int = 8,
    wandb_run: str = "",
):
    import torch
    import wandb
    from unsloth import FastVisionModel
    from unsloth.trainer import UnslothVisionDataCollator

    from trl import SFTTrainer, SFTConfig

    # Unsloth patches SFTConfig with eos_token='<EOS_TOKEN>' which doesn't
    # exist in Qwen2.5-VL's vocab. Patch the CLASS default so any internally
    # recreated SFTConfig also gets None (skips the validation).
    if hasattr(SFTConfig, '__dataclass_fields__') and 'eos_token' in SFTConfig.__dataclass_fields__:
        SFTConfig.__dataclass_fields__['eos_token'].default = None
    elif hasattr(SFTConfig, 'eos_token'):
        SFTConfig.eos_token = None

    # Build config
    config = TrainConfig(
        epochs=epochs,
        lr=lr,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        batch_size=batch_size,
        grad_accum=grad_accum,
    )
    if wandb_run:
        config.wandb_run = wandb_run
    # Define run_id variable outside
    run_id = None
    try:
        run = wandb.init(
            project=config.wandb_project,
            name=config.wandb_run,
            config=dataclasses.asdict(config),
            reinit=True
        )
        run_id = run.id  # ✅ 保存 ID，稍后重连用
    except Exception as e:
        print(f"WandB init failed (ignoring): {e}")

    # ── 1. Verify data ───────────────────────────────────────────────────
    print("=" * 60)
    print("MSCD VLM LoRA Training")
    print("=" * 60)

    assert os.path.exists(config.train_file), f"Missing: {config.train_file}"
    assert os.path.exists(config.test_file), f"Missing: {config.test_file}"

    # v0.5 images (site photos + floorplans)
    _v05_dirs = ["/data/images/v05/imgs", "/data/images/v05/floorplans"]
    n_v05 = {d.split("/")[-1]: len(os.listdir(d)) for d in _v05_dirs if os.path.isdir(d)}
    # v0.4 images (for enriched records)
    _v04_img_dirs  = ["/data/images/ap/imgs",  "/data/images/bh/imgs",  "/data/images/dxa/imgs"]
    _v04_plan_dirs = ["/data/images/ap/plans", "/data/images/bh/plans", "/data/images/dxa/plans"]
    n_v04_imgs  = sum(len(os.listdir(d)) for d in _v04_img_dirs  if os.path.isdir(d))
    n_v04_plans = sum(len(os.listdir(d)) for d in _v04_plan_dirs if os.path.isdir(d))
    print(f"  v0.5 images: {n_v05}")
    print(f"  v0.4 images: {n_v04_imgs} site photos, {n_v04_plans} floorplans")

    # ── 2. Load and remap data ───────────────────────────────────────────
    # Returns plain Python list — NOT HF Dataset (matches notebook pattern)
    print("\nLoading training data...")
    train_samples = load_and_remap(config.train_file, config)
    test_samples = load_and_remap(config.test_file, config)
    print(f"  Train: {len(train_samples)} samples")
    print(f"  Test:  {len(test_samples)} samples")

    # Verify first sample image path resolves
    first_msg = train_samples[0]["messages"][1]  # user message
    for part in first_msg["content"]:
        if isinstance(part, dict) and part.get("type") == "image":
            img_path = part["image"].replace("file://", "")
            exists = os.path.exists(img_path)
            print(f"  Image check: {'OK' if exists else 'MISSING'} {part['image']}")

    # ── 3. Load model ────────────────────────────────────────────────────
    print(f"\nLoading model: {config.model_name}")
    model, tokenizer = FastVisionModel.from_pretrained(
        config.model_name,
        load_in_4bit=True,
        use_gradient_checkpointing="unsloth",
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params:,}")

    # ── 4. Apply LoRA ────────────────────────────────────────────────────
    print(f"\nApplying LoRA (r={config.lora_r}, alpha={config.lora_alpha})")
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=True,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        random_state=config.seed,
        use_rslora=False,
        loftq_config=None,
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable: {trainable:,} / {total_params:,} "
          f"({100 * trainable / total_params:.2f}%)")

    # ── 5. Train ─────────────────────────────────────────────────────────
    # Enable training mode (required by Unsloth before creating trainer)
    FastVisionModel.for_training(model)

    os.makedirs(config.output_dir, exist_ok=True)

    # Progress callback — prints loss/lr/step to stdout (streamed by Modal)
    from transformers import TrainerCallback

    steps_per_epoch = max(1, len(train_samples) // (config.batch_size * config.grad_accum))
    total_steps = steps_per_epoch * config.epochs

    class ProgressCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs is None:
                return
            step = state.global_step
            pct = 100 * step / total_steps if total_steps else 0
            loss = logs.get("loss", logs.get("eval_loss", ""))
            lr_val = logs.get("learning_rate", "")
            parts = [f"Step {step}/{total_steps} ({pct:.0f}%)"]
            if isinstance(loss, float):
                parts.append(f"loss={loss:.4f}")
            if isinstance(lr_val, float):
                parts.append(f"lr={lr_val:.2e}")
            if "eval_loss" in logs:
                parts.append(f"eval_loss={logs['eval_loss']:.4f}")
            print(f"  [{' | '.join(parts)}]", flush=True)

        def on_evaluate(self, args, state, control, metrics=None, **kwargs):
            if metrics:
                print(f"  >> Eval @ step {state.global_step}: "
                      f"loss={metrics.get('eval_loss', 'N/A'):.4f}", flush=True)

    # SFTTrainer setup — follows Unsloth notebook exactly:
    #   - UnslothVisionDataCollator handles multimodal batching
    #   - train_dataset is a plain Python list (NOT HF Dataset)
    #   - dataset_kwargs={"skip_prepare_dataset": True} required for vision
    #   - remove_unused_columns=False required for vision
    #   - dataset_text_field="" required for vision
    sft_args = SFTConfig(
        output_dir=config.output_dir,
        num_train_epochs=config.epochs,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.grad_accum,
        learning_rate=config.lr,
        weight_decay=config.weight_decay,
        warmup_steps=config.warmup_steps,
        lr_scheduler_type="cosine",
        optim="adamw_8bit",
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        eval_strategy="epoch",
        per_device_eval_batch_size=1,
        save_strategy="epoch",
        save_total_limit=config.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=5,
        report_to="wandb",
        run_name=config.wandb_run,
        seed=config.seed,

        # === 关键：视觉微调必须保留的参数 (参考官方文件) ===
        remove_unused_columns=False,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True}, # 这行让 Trainer 不去检查 EOS
        max_length=config.max_seq_length,
        # ==================================================
        
        # ❌ 删除这一行：
        # eos_token=None 或 "<|im_end|>" 都不要写！让它保持默认。
    )
    # Unsloth 2025.11.x patches SFTConfig with eos_token defaulting to
    # '<EOS_TOKEN>' which doesn't exist in Qwen2.5-VL's vocab. Set to None
    # to skip the validation entirely — the data collator handles EOS.
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        data_collator=UnslothVisionDataCollator(model, tokenizer),
        train_dataset=train_samples,
        eval_dataset=test_samples,
        args=sft_args,
        callbacks=[ProgressCallback()],
    )

    print(f"\nStarting training...")
    print(f"  Effective batch size: {config.batch_size * config.grad_accum}")
    print(f"  Steps/epoch: {steps_per_epoch} | Total steps: {total_steps}")
    print(f"  Epochs: {config.epochs}")
    print(f"  Logging every 5 steps | Eval & save every epoch")
    print(flush=True)

    result = trainer.train()

    print(f"\nTraining complete!")
    print(f"  Final train loss: {result.training_loss:.4f}")
    print(f"  Total steps: {result.global_step}")

    # ── CRITICAL FIX: Resurrect WandB if Trainer closed it ────────────────
    # 如果 Trainer 关闭了 run，我们用之前的 ID 重新连回去，
    # 这样 eval_loss 和 test_json_rate 就能记录在同一个图表里。
    if wandb.run is None and run_id is not None:
        print("  [Info] WandB run was closed by Trainer. Re-connecting...")
        try:
            wandb.init(
                project=config.wandb_project,
                id=run_id, 
                resume="must"
            )
        except Exception as e:
            print(f"  [Warn] Failed to reconnect WandB: {e}")

    # ── 7. Evaluate ──────────────────────────────────────────────────────
    try:
        eval_results = trainer.evaluate()
        print(f"  Eval loss: {eval_results['eval_loss']:.4f}")
        # 此时 wandb.run 应该是活着的，可以正常 log
        if wandb.run is not None:
            trainer.log({"final_eval_loss": eval_results["eval_loss"]})
    except Exception as e:
        print(f"  [Warn] Evaluation logging failed: {e}")
        eval_results = {"eval_loss": 0.0}

    # ── 8. Manual inference check ────────────────────────────────────────
    print("\nInference check on test samples...")
    FastVisionModel.for_inference(model)
    _run_inference_check(model, tokenizer, test_samples, step=result.global_step)

    # ── 9. Save adapter ─────────────────────────────────────────────────
    final_path = os.path.join(config.output_dir, "final")
    os.makedirs(final_path, exist_ok=True)

    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    checkpoint_vol.commit()

    print(f"\nAdapter saved to volume: {final_path}")
    print("Download with:")
    print(f"  modal volume get mscd-checkpoints /mscd-lora-v3/final "
          f"./models/adapters/v3_lora_qwen")

    try:
        wandb.finish()
    except wandb.Error:
        pass

    return {
        "train_loss": result.training_loss,
        "eval_loss": eval_results["eval_loss"],
        "steps": result.global_step,
        "adapter_path": final_path,
    }


def _run_inference_check(model, tokenizer, test_samples, step=0):
    """Quick inference on test samples to verify JSON output quality.

    Checks: JSON validity, ifc_class, storey_name, spatial_relations predicate.
    """
    import torch
    import wandb
    from PIL import Image

    n_valid_json = 0
    n_class_match = 0
    n_storey_match = 0
    n_spatial_match = 0  # predicate match for topology test samples
    n_spatial_total = 0  # test samples that have spatial_relations
    n_false_positive = 0  # model outputs spatial_relations when GT has []
    n_attr_only = 0       # GT has spatial_relations: []
    n_total = min(len(test_samples), 20)  # check more samples for LoRA_3

    for i in range(n_total):
        sample = test_samples[i]
        messages = sample["messages"]

        # Get ground truth from assistant message
        gt_text = None
        for m in messages:
            if m["role"] == "assistant":
                content = m["content"]
                if isinstance(content, str):
                    gt_text = content
                elif isinstance(content, list):
                    gt_text = content[0].get("text", "") if content else ""
                break
        if not gt_text:
            continue

        try:
            gt = json.loads(gt_text)
        except json.JSONDecodeError:
            continue

        # Build inference input (system + user only, no assistant)
        inference_msgs = [m for m in messages if m["role"] != "assistant"]

        input_text = tokenizer.apply_chat_template(
            inference_msgs, tokenize=False, add_generation_prompt=True
        )

        # Use tokenizer directly (Unsloth notebook pattern for Qwen2.5-VL)
        # Extract ALL images from user messages
        images = []
        for m in inference_msgs:
            if m["role"] == "user" and isinstance(m["content"], list):
                for part in m["content"]:
                    if isinstance(part, dict) and part.get("type") == "image":
                        img_ref = part.get("image", "")
                        if isinstance(img_ref, str) and img_ref.startswith("file://"):
                            # 加载每一张图片并添加到列表
                            images.append(Image.open(img_ref.replace("file://", "")))
        
        if images:
            inputs = tokenizer(
                text=[input_text], # 显式指定 text 参数更安全
                images=images,     # 传入的是列表 [Image, Image, ...]
                add_special_tokens=False,
                return_tensors="pt",
            ).to(model.device)
        else:
            inputs = tokenizer(
                text=[input_text],
                add_special_tokens=False,
                return_tensors="pt",
            ).to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs, max_new_tokens=256, do_sample=False, use_cache=True
            )

        trimmed = output_ids[0][len(inputs.input_ids[0]):]
        output_text = tokenizer.decode(trimmed, skip_special_tokens=True).strip()

        # Check JSON validity
        try:
            parsed = json.loads(output_text)
            n_valid_json += 1

            if parsed.get("ifc_class") == gt.get("ifc_class"):
                n_class_match += 1
            if parsed.get("storey_name") == gt.get("storey_name"):
                n_storey_match += 1

            # Check spatial_relations
            gt_rels = gt.get("spatial_relations", [])
            pred_rels = parsed.get("spatial_relations", [])
            if gt_rels:
                n_spatial_total += 1
                gt_pred = gt_rels[0].get("predicate", "")
                pred_pred = pred_rels[0].get("predicate", "") if pred_rels else ""
                if gt_pred == pred_pred:
                    n_spatial_match += 1
            else:
                n_attr_only += 1
                if pred_rels:
                    n_false_positive += 1

            status = "OK"
        except json.JSONDecodeError:
            parsed = None
            status = "FAIL"

        class_ok = parsed and parsed.get("ifc_class") == gt.get("ifc_class")
        storey_ok = parsed and parsed.get("storey_name") == gt.get("storey_name")
        spatial_tag = ""
        if parsed:
            gt_rels = gt.get("spatial_relations", [])
            pred_rels = parsed.get("spatial_relations", [])
            if gt_rels:
                gt_p = gt_rels[0].get("predicate", "")
                pred_p = pred_rels[0].get("predicate", "") if pred_rels else "NONE"
                spatial_tag = f" | spatial:{gt_p}={'Y' if gt_p == pred_p else 'N('+pred_p+')'}"
            elif pred_rels:
                spatial_tag = f" | spatial:FP({pred_rels[0].get('predicate','')})"

        print(f"  [{i+1}] JSON:{status} | "
              f"class:{'Y' if class_ok else 'N'} | "
              f"storey:{'Y' if storey_ok else 'N'}{spatial_tag}")
        if status == "FAIL":
            print(f"       Raw: {output_text[:120]}")

    spatial_acc = n_spatial_match / n_spatial_total if n_spatial_total else 0
    fp_rate = n_false_positive / n_attr_only if n_attr_only else 0

    if wandb.run is not None:
        try:
            wandb.log({
                "test_json_rate": n_valid_json / n_total if n_total else 0,
                "test_class_accuracy": n_class_match / n_total if n_total else 0,
                "test_storey_accuracy": n_storey_match / n_total if n_total else 0,
                "test_spatial_predicate_acc": spatial_acc,
                "test_spatial_false_positive_rate": fp_rate,
                "train/global_step": step,
            })
        except Exception:
            pass
    else:
        print("  [Warn] WandB run closed, skipping inference metrics logging.")

    print(f"\n  JSON parse rate:      {n_valid_json}/{n_total}")
    print(f"  Class accuracy:       {n_class_match}/{n_total}")
    print(f"  Storey accuracy:      {n_storey_match}/{n_total}")
    print(f"  Spatial predicate:    {n_spatial_match}/{n_spatial_total} ({spatial_acc:.0%})")
    print(f"  False positive rate:  {n_false_positive}/{n_attr_only} ({fp_rate:.0%})")


# ── CLI entry point ──────────────────────────────────────────────────────────

@app.local_entrypoint()
def main(
    epochs: int = 3,
    lr: float = 2e-4,
    lora_r: int = 16,
    lora_alpha: int = 32,
    batch_size: int = 2,
    grad_accum: int = 8,
    wandb_run: str = "",
):
    """Launch training on Modal GPU."""
    n_train = sum(1 for _ in open(TRAIN_JSONL))
    n_test  = sum(1 for _ in open(TEST_JSONL))
    n_v05_imgs = len(list(V05_SITE_IMGS_DIR.glob("*"))) if V05_SITE_IMGS_DIR.exists() else 0
    n_v05_fp   = len(list(V05_FLOORPLANS_DIR.glob("*"))) if V05_FLOORPLANS_DIR.exists() else 0
    n_v04_imgs  = sum(len(list(d.glob("*"))) for d in [AP_IMGS_DIR, BH_IMGS_DIR, DXA_IMGS_DIR] if d.exists())
    n_v04_plans = sum(len(list(d.glob("*"))) for d in [AP_PLANS_DIR, BH_PLANS_DIR, DXA_PLANS_DIR] if d.exists())

    print("Launching MSCD VLM LoRA_3 training on Modal...")
    print(f"  Config: epochs={epochs}, lr={lr}, r={lora_r}, alpha={lora_alpha}")
    print(f"  Data:   {V05_DIR} ({n_train} train / {n_test} test)")
    print(f"  v0.5:   {n_v05_imgs} site photos, {n_v05_fp} floorplans")
    print(f"  v0.4:   {n_v04_imgs} site photos, {n_v04_plans} floorplans (AP+BH+DXA)")

    # .spawn() submits the job and returns immediately — no blocking, no gRPC timeout.
    # .remote() blocks until the result is ready (~25 min) and hits Deadline exceeded.
    handle = train.spawn(
        epochs=epochs,
        lr=lr,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        batch_size=batch_size,
        grad_accum=grad_accum,
        wandb_run=wandb_run,
    )

    print("\n" + "=" * 60)
    print("TRAINING JOB SUBMITTED")
    print("=" * 60)
    print(f"  Job ID: {handle.object_id}")
    print(f"\nMonitor:")
    print(f"  modal app logs mscd-vlm-lora3-train")
    print(f"  wandb: project=mscd-vlm-lora  run={wandb_run or 'qwen25vl-7b-r16-lora3-synth_v05'}")
    print(f"\nWhen complete, download adapter:")
    print(f"  ./training/train.sh --download-only")
    print(f"\nThen evaluate:")
    print(f"  ./training/eval.sh --step paired-ablation --adapter final --skip-v2-prompt")
