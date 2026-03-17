"""
MSCD VLM LoRA_5 Training — Modal GPU Training Script (Floorplan Pivot)

Fine-tunes Qwen2.5-VL-7B-Instruct with LoRA for multimodal constraints
extraction (floorplans + site photos + chat -> JSON constraints with spatial_relations).

LoRA_5 changes from LoRA_5:
  - 5 predicates: FILLS, ADJACENT_TO, CONTINUOUS, NEXT_TO, CONNECTS_TO (was 4)
  - Floorplan is PRIMARY modality (always included), site photo SECONDARY (~30%)
  - 3 IFC models: AP + BH + DXA (was AP-only in practice)
  - Dataset: 616 train / 57 test (from 181 KEEP skins + 903 v0.4)
  - Modality tag per record: fp_only / fp_site / site_only (for eval ablation)

Based on Unsloth's official Qwen2.5-VL vision fine-tuning notebook:
  Qwen2_5_VL_(7B)_Vision.ipynb

Usage:
    modal run training/train_lora5.py
    modal run training/train_lora5.py --epochs 5 --lr 2e-4 --lora-r 16

    # Download trained adapter after training:
    modal volume get mscd-checkpoints /mscd-lora-v5/final ./models/adapters/v5_lora_qwen

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

# synth_v0.5 — LoRA_5 training data (616 train / 57 test)
V05_DIR     = DATASETS_DIR / "synth_v0.5" / "train"
TRAIN_JSONL = V05_DIR / "lora5_train.jsonl"
TEST_JSONL  = V05_DIR / "lora5_test.jsonl"

# v0.5 image directories per model (site photos + floorplans only)
V05_AP_IMGS_DIR  = DATASETS_DIR / "synth_v0.5"     / "imgs"
V05_AP_FP_DIR    = DATASETS_DIR / "synth_v0.5"     / "floorplans"
V05_BH_IMGS_DIR  = DATASETS_DIR / "synth_v0.5_bh"  / "imgs"
V05_BH_FP_DIR    = DATASETS_DIR / "synth_v0.5_bh"  / "floorplans"
V05_DXA_IMGS_DIR = DATASETS_DIR / "synth_v0.5_dxa" / "imgs"
V05_DXA_FP_DIR   = DATASETS_DIR / "synth_v0.5_dxa" / "floorplans"

# v0.4 image directories (site photos + floorplans for enriched records)
AP_IMGS_DIR   = DATASETS_DIR / "synth_v0.4_ap"  / "cases" / "imgs"
AP_PLANS_DIR  = DATASETS_DIR / "synth_v0.4_ap"  / "cases" / "plans"
BH_IMGS_DIR   = DATASETS_DIR / "synth_v0.4_bh"  / "cases" / "imgs"
BH_PLANS_DIR  = DATASETS_DIR / "synth_v0.4_bh"  / "cases" / "plans"
DXA_IMGS_DIR  = DATASETS_DIR / "synth_v0.4_dxa" / "cases" / "imgs"
DXA_PLANS_DIR = DATASETS_DIR / "synth_v0.4_dxa" / "cases" / "plans"

# ── Modal infrastructure ─────────────────────────────────────────────────────

app = modal.App("mscd-vlm-lora5-train")

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
    # Bake LoRA_5 training JSONL files
    .add_local_file(str(TRAIN_JSONL), remote_path="/data/train/lora5_train.jsonl")
    .add_local_file(str(TEST_JSONL),  remote_path="/data/train/lora5_test.jsonl")
    # Bake v0.5 image directories per model (site photos + floorplans)
    .add_local_dir(str(V05_AP_IMGS_DIR),  remote_path="/data/images/v05_ap/imgs")
    .add_local_dir(str(V05_AP_FP_DIR),    remote_path="/data/images/v05_ap/floorplans")
    .add_local_dir(str(V05_BH_IMGS_DIR),  remote_path="/data/images/v05_bh/imgs")
    .add_local_dir(str(V05_BH_FP_DIR),    remote_path="/data/images/v05_bh/floorplans")
    .add_local_dir(str(V05_DXA_IMGS_DIR), remote_path="/data/images/v05_dxa/imgs")
    .add_local_dir(str(V05_DXA_FP_DIR),   remote_path="/data/images/v05_dxa/floorplans")
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
    lora_dropout: float = 0.1

    # Training — 5 epochs default (616 samples, similar to LoRA_4's 553)
    epochs: int = 5
    batch_size: int = 2
    grad_accum: int = 8        # effective batch = 16
    lr: float = 2e-4
    warmup_steps: int = 10
    max_seq_length: int = 4096  # multiple images per sample
    weight_decay: float = 0.01

    # Evaluation & checkpoints
    eval_steps: int = 10
    save_total_limit: int = 5   # keep best + last few

    # Wandb
    wandb_project: str = "mscd-vlm-lora"
    wandb_run: str = "qwen25vl-7b-r16-lora5-fp-pivot"

    # Paths (inside Modal container)
    train_file: str = "/data/train/lora5_train.jsonl"
    test_file: str = "/data/train/lora5_test.jsonl"
    output_dir: str = "/checkpoints/mscd-lora-v5"

    seed: int = 42


# ── Path remapping ───────────────────────────────────────────────────────────

# Maps local dataset prefix -> container image prefix under /data/images/<tag>/
_LOCAL_ROOT  = "file:///root/cmu/master_thesis/data_curation/datasets/"
_REMOTE_ROOT = "file:///data/images/"
_DATASET_MAP = {
    # v0.5 image paths per model (site photos + floorplans)
    "synth_v0.5/imgs/":            "v05_ap/imgs/",
    "synth_v0.5/floorplans/":      "v05_ap/floorplans/",
    "synth_v0.5_bh/imgs/":         "v05_bh/imgs/",
    "synth_v0.5_bh/floorplans/":   "v05_bh/floorplans/",
    "synth_v0.5_dxa/imgs/":        "v05_dxa/imgs/",
    "synth_v0.5_dxa/floorplans/":  "v05_dxa/floorplans/",
    # v0.4 image paths (for enriched records)
    "synth_v0.4_ap/cases/":        "ap/",
    "synth_v0.4_bh/cases/":        "bh/",
    "synth_v0.4_dxa/cases/":       "dxa/",
}

def remap_image_paths(sample: dict, config: TrainConfig) -> dict:
    """Remap local absolute paths to Modal container paths.

    e.g. file:///root/.../synth_v0.4_ap/cases/plans/plan_XXX.png
      ->  file:///data/images/ap/plans/plan_XXX.png
    """
    for msg in sample.get("messages", []):
        content = msg.get("content")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "image":
                    path = part["image"]
                    if path.startswith(_LOCAL_ROOT):
                        rel = path[len(_LOCAL_ROOT):]
                        for ds_prefix, remote_prefix in _DATASET_MAP.items():
                            if rel.startswith(ds_prefix):
                                rel = remote_prefix + rel[len(ds_prefix):]
                                break
                        part["image"] = _REMOTE_ROOT + rel
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
    epochs: int = 5,
    lr: float = 2e-4,
    lora_r: int = 16,
    lora_alpha: int = 32,
    batch_size: int = 2,
    grad_accum: int = 8,
    wandb_run: str = "",
    resume_from_checkpoint: bool = False,
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
    run_id = None
    try:
        run = wandb.init(
            project=config.wandb_project,
            name=config.wandb_run,
            config=dataclasses.asdict(config),
            reinit=True
        )
        run_id = run.id
    except Exception as e:
        print(f"WandB init failed (ignoring): {e}")

    # ── 1. Verify data ───────────────────────────────────────────────────
    print("=" * 60)
    print("MSCD VLM LoRA_5 Training (Floorplan Pivot — 5 predicates)")
    print("=" * 60)

    assert os.path.exists(config.train_file), f"Missing: {config.train_file}"
    assert os.path.exists(config.test_file), f"Missing: {config.test_file}"

    # v0.5 images per model (site photos + floorplans)
    _v05_dirs = [
        "/data/images/v05_ap/imgs", "/data/images/v05_ap/floorplans",
        "/data/images/v05_bh/imgs", "/data/images/v05_bh/floorplans",
        "/data/images/v05_dxa/imgs", "/data/images/v05_dxa/floorplans",
    ]
    n_v05 = {d.split("/")[-2] + "/" + d.split("/")[-1]: len(os.listdir(d))
             for d in _v05_dirs if os.path.isdir(d)}
    # v0.4 images (for enriched records)
    _v04_img_dirs  = ["/data/images/ap/imgs",  "/data/images/bh/imgs",  "/data/images/dxa/imgs"]
    _v04_plan_dirs = ["/data/images/ap/plans", "/data/images/bh/plans", "/data/images/dxa/plans"]
    n_v04_imgs  = sum(len(os.listdir(d)) for d in _v04_img_dirs  if os.path.isdir(d))
    n_v04_plans = sum(len(os.listdir(d)) for d in _v04_plan_dirs if os.path.isdir(d))
    print(f"  v0.5 images: {n_v05}")
    print(f"  v0.4 images: {n_v04_imgs} site photos, {n_v04_plans} floorplans")

    # ── 2. Load and remap data ───────────────────────────────────────────
    print("\nLoading training data...")
    train_samples = load_and_remap(config.train_file, config)
    test_samples = load_and_remap(config.test_file, config)
    print(f"  Train: {len(train_samples)} samples")
    print(f"  Test:  {len(test_samples)} samples")

    # Dataset composition audit
    n_sr_counts = {0: 0, 1: 0, 2: 0}
    for s in train_samples:
        asst = next(m for m in s["messages"] if m["role"] == "assistant")
        label = json.loads(asst["content"])
        n_sr = len(label.get("spatial_relations", []))
        n_sr_counts[min(n_sr, 2)] = n_sr_counts.get(min(n_sr, 2), 0) + 1
    print(f"  SR distribution: {n_sr_counts}")

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
    FastVisionModel.for_training(model)

    os.makedirs(config.output_dir, exist_ok=True)

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

    class EpochInferenceCallback(TrainerCallback):
        """Run full inference eval on test set after each epoch."""
        def __init__(self, model_ref, tokenizer_ref, test_data):
            self._model = model_ref
            self._tokenizer = tokenizer_ref
            self._test_data = test_data
            self._last_epoch = -1

        def on_epoch_end(self, args, state, control, **kwargs):
            epoch = int(state.epoch)
            if epoch == self._last_epoch:
                return
            self._last_epoch = epoch
            print(f"\n{'='*50}")
            print(f"  Inference eval — Epoch {epoch}")
            print(f"{'='*50}", flush=True)
            # Switch to inference mode
            FastVisionModel.for_inference(self._model)
            _run_inference_check(
                self._model, self._tokenizer, self._test_data,
                step=state.global_step, epoch=epoch,
            )
            # Switch back to training mode
            FastVisionModel.for_training(self._model)
            print(f"{'='*50}\n", flush=True)

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
        # Vision fine-tuning required params
        remove_unused_columns=False,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        max_length=config.max_seq_length,
    )
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        data_collator=UnslothVisionDataCollator(model, tokenizer),
        train_dataset=train_samples,
        eval_dataset=test_samples,
        args=sft_args,
        callbacks=[ProgressCallback(), EpochInferenceCallback(model, tokenizer, test_samples)],
    )

    print(f"\nStarting training...")
    print(f"  Effective batch size: {config.batch_size * config.grad_accum}")
    print(f"  Steps/epoch: {steps_per_epoch} | Total steps: {total_steps}")
    print(f"  Epochs: {config.epochs}")
    print(f"  Logging every 5 steps | Eval & save every epoch")
    print(flush=True)

    result = trainer.train(resume_from_checkpoint=resume_from_checkpoint if resume_from_checkpoint else None)

    print(f"\nTraining complete!")
    print(f"  Final train loss: {result.training_loss:.4f}")
    print(f"  Total steps: {result.global_step}")

    # Resurrect WandB if Trainer closed it
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
    print(f"  modal volume get mscd-checkpoints /mscd-lora-v5/final "
          f"./models/adapters/v5_lora_qwen")

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


def _run_inference_check(model, tokenizer, test_samples, step=0, epoch=None):
    """Inference on test samples to measure JSON output quality.

    LoRA_5 checks: JSON validity, ifc_class, storey_name, spatial_relations
    predicate (hop-1 AND hop-2 for multi-triplet records).
    Runs after each epoch (via EpochInferenceCallback) and at end of training.
    """
    import torch
    import wandb
    from PIL import Image

    from collections import Counter as _Counter
    n_valid_json = 0
    n_class_match = 0
    n_storey_match = 0
    n_spatial_match = 0    # hop-1 predicate match
    n_spatial_total = 0    # test samples that have spatial_relations
    n_hop2_match = 0       # hop-2 predicate match (2-triplet records)
    n_hop2_total = 0       # test samples with 2+ spatial_relations
    n_false_positive = 0   # model outputs spatial_relations when GT has []
    n_attr_only = 0        # GT has spatial_relations: []
    n_total = len(test_samples)
    per_pred_total = _Counter()    # per-predicate: count of GT occurrences
    per_pred_correct = _Counter()  # per-predicate: count of correct predictions

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

        # Extract ALL images from user messages
        images = []
        for m in inference_msgs:
            if m["role"] == "user" and isinstance(m["content"], list):
                for part in m["content"]:
                    if isinstance(part, dict) and part.get("type") == "image":
                        img_ref = part.get("image", "")
                        if isinstance(img_ref, str) and img_ref.startswith("file://"):
                            images.append(Image.open(img_ref.replace("file://", "")))

        if images:
            inputs = tokenizer(
                text=[input_text],
                images=images,
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
                **inputs, max_new_tokens=512, do_sample=False, use_cache=True
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

            # Check spatial_relations — hop-1 and hop-2
            gt_rels = gt.get("spatial_relations", [])
            pred_rels = parsed.get("spatial_relations", [])
            if gt_rels:
                n_spatial_total += 1
                # Hop-1: first predicate
                gt_pred = gt_rels[0].get("predicate", "")
                pred_pred = pred_rels[0].get("predicate", "") if pred_rels else ""
                per_pred_total[gt_pred] += 1
                if gt_pred == pred_pred:
                    n_spatial_match += 1
                    per_pred_correct[gt_pred] += 1

                # Hop-2: second predicate (multi-triplet records)
                if len(gt_rels) >= 2:
                    n_hop2_total += 1
                    gt_pred2 = gt_rels[1].get("predicate", "")
                    pred_pred2 = pred_rels[1].get("predicate", "") if len(pred_rels) >= 2 else ""
                    per_pred_total[gt_pred2 + "_hop2"] += 1
                    if gt_pred2 == pred_pred2:
                        n_hop2_match += 1
                        per_pred_correct[gt_pred2 + "_hop2"] += 1
            else:
                n_attr_only += 1
                if pred_rels:
                    n_false_positive += 1

            status = "OK"
        except json.JSONDecodeError:
            parsed = None
            status = "FAIL"

        # Print per-sample status
        class_ok = parsed and parsed.get("ifc_class") == gt.get("ifc_class")
        storey_ok = parsed and parsed.get("storey_name") == gt.get("storey_name")
        spatial_tag = ""
        if parsed:
            gt_rels = gt.get("spatial_relations", [])
            pred_rels = parsed.get("spatial_relations", [])
            if gt_rels:
                gt_p = gt_rels[0].get("predicate", "")
                pred_p = pred_rels[0].get("predicate", "") if pred_rels else "NONE"
                spatial_tag = f" | SR1:{gt_p}={'Y' if gt_p == pred_p else 'N('+pred_p+')'}"
                # Show hop-2 status for multi-triplet
                if len(gt_rels) >= 2:
                    gt_p2 = gt_rels[1].get("predicate", "")
                    pred_p2 = pred_rels[1].get("predicate", "") if len(pred_rels) >= 2 else "NONE"
                    spatial_tag += f" SR2:{gt_p2}={'Y' if gt_p2 == pred_p2 else 'N('+pred_p2+')'}"
            elif pred_rels:
                spatial_tag = f" | SR:FP({pred_rels[0].get('predicate','')})"

        print(f"  [{i+1}] JSON:{status} | "
              f"class:{'Y' if class_ok else 'N'} | "
              f"storey:{'Y' if storey_ok else 'N'}{spatial_tag}")
        if status == "FAIL":
            print(f"       Raw: {output_text[:120]}")

    spatial_acc = n_spatial_match / n_spatial_total if n_spatial_total else 0
    hop2_acc = n_hop2_match / n_hop2_total if n_hop2_total else 0
    fp_rate = n_false_positive / n_attr_only if n_attr_only else 0

    # Per-predicate accuracy
    pred_correct = {}
    pred_total_ct = {}
    for gt_pred_name in per_pred_total:
        pred_total_ct[gt_pred_name] = per_pred_total[gt_pred_name]
        pred_correct[gt_pred_name] = per_pred_correct.get(gt_pred_name, 0)

    epoch_tag = f"epoch{epoch}" if epoch is not None else "final"
    metrics = {
        f"infer/{epoch_tag}/json_rate": n_valid_json / n_total if n_total else 0,
        f"infer/{epoch_tag}/class_acc": n_class_match / n_total if n_total else 0,
        f"infer/{epoch_tag}/storey_acc": n_storey_match / n_total if n_total else 0,
        f"infer/{epoch_tag}/hop1_acc": spatial_acc,
        f"infer/{epoch_tag}/hop2_acc": hop2_acc,
        f"infer/{epoch_tag}/fp_rate": fp_rate,
        "train/global_step": step,
    }
    for pred_name in pred_total_ct:
        acc = pred_correct[pred_name] / pred_total_ct[pred_name] if pred_total_ct[pred_name] else 0
        metrics[f"infer/{epoch_tag}/pred_{pred_name}"] = acc

    if wandb.run is not None:
        try:
            wandb.log(metrics)
        except Exception:
            pass
    else:
        print("  [Warn] WandB run closed, skipping inference metrics logging.")

    print(f"\n  JSON parse rate:      {n_valid_json}/{n_total}")
    print(f"  Class accuracy:       {n_class_match}/{n_total}")
    print(f"  Storey accuracy:      {n_storey_match}/{n_total}")
    print(f"  Spatial hop-1 acc:    {n_spatial_match}/{n_spatial_total} ({spatial_acc:.0%})")
    print(f"  Spatial hop-2 acc:    {n_hop2_match}/{n_hop2_total} ({hop2_acc:.0%})")
    print(f"  False positive rate:  {n_false_positive}/{n_attr_only} ({fp_rate:.0%})")
    if pred_total_ct:
        print(f"  Per-predicate hop-1:")
        for pred_name in sorted(pred_total_ct):
            c = pred_correct[pred_name]
            t = pred_total_ct[pred_name]
            print(f"    {pred_name:18s} {c}/{t} ({c/t:.0%})" if t else f"    {pred_name:18s} 0/0")


# ── CLI entry point ──────────────────────────────────────────────────────────

@app.local_entrypoint()
def main(
    epochs: int = 5,
    lr: float = 2e-4,
    lora_r: int = 16,
    lora_alpha: int = 32,
    batch_size: int = 2,
    grad_accum: int = 8,
    wandb_run: str = "",
    resume: bool = False,
):
    """Launch LoRA_5 training on Modal GPU."""
    n_train = sum(1 for _ in open(TRAIN_JSONL))
    n_test  = sum(1 for _ in open(TEST_JSONL))
    _v05_img_dirs = [V05_AP_IMGS_DIR, V05_BH_IMGS_DIR, V05_DXA_IMGS_DIR]
    _v05_fp_dirs  = [V05_AP_FP_DIR, V05_BH_FP_DIR, V05_DXA_FP_DIR]
    n_v05_imgs = sum(len(list(d.glob("*"))) for d in _v05_img_dirs if d.exists())
    n_v05_fp   = sum(len(list(d.glob("*"))) for d in _v05_fp_dirs if d.exists())
    n_v04_imgs  = sum(len(list(d.glob("*"))) for d in [AP_IMGS_DIR, BH_IMGS_DIR, DXA_IMGS_DIR] if d.exists())
    n_v04_plans = sum(len(list(d.glob("*"))) for d in [AP_PLANS_DIR, BH_PLANS_DIR, DXA_PLANS_DIR] if d.exists())

    print("Launching MSCD VLM LoRA_5 training on Modal...")
    print(f"  Config: epochs={epochs}, lr={lr}, r={lora_r}, alpha={lora_alpha}")
    print(f"  Data:   {V05_DIR} ({n_train} train / {n_test} test)")
    print(f"  v0.5:   {n_v05_imgs} site photos, {n_v05_fp} floorplans (AP+BH+DXA)")
    print(f"  v0.4:   {n_v04_imgs} site photos, {n_v04_plans} floorplans (AP+BH+DXA)")

    if resume:
        print(f"\n  RESUMING from latest checkpoint in /checkpoints/mscd-lora-v5/")
    print(f"\nMonitor on WandB: project=mscd-vlm-lora  run={wandb_run or 'qwen25vl-7b-r16-lora5-fp-pivot'}")
    print("Training in progress (blocking until complete)...\n")

    result = train.remote(
        epochs=epochs,
        lr=lr,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        batch_size=batch_size,
        grad_accum=grad_accum,
        wandb_run=wandb_run,
        resume_from_checkpoint=resume,
    )

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Train loss: {result['train_loss']:.4f}")
    print(f"  Eval loss:  {result['eval_loss']:.4f}")
    print(f"  Steps:      {result['steps']}")
    print(f"\nDownload adapter:")
    print(f"  modal volume get mscd-checkpoints /mscd-lora-v5/final ./models/adapters/v5_lora_qwen")
