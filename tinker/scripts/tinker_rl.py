#!/usr/bin/env python3
"""CholecT50 GRPO/RLVR on Thinking Machines Tinker, on top of the SFT checkpoint.

Trains Qwen3.5-4B (LoRA) to emit surgical action triplets ("instrument-verb-target")
for an endoscopic frame, using RLVR with a dense triplet-F1 reward. This reuses
tinker_cookbook's RL framework (train.main) with a small vision-language Env:

  * initial_observation  -> image + prompt, rendered to a ModelInput
  * step                 -> decode the completion, reward it with triplet F1
  * a "group" is G rollouts of the same frame; advantages are centered within
    the group (GRPO), and the LoRA adapter is updated with the
    importance-sampling policy-gradient loss.

The LoRA adapter is initialized from the SFT checkpoint (read from the file
written by tinker_sft.py), so this is the RL-after-SFT stage.

Smoke test (2 batches, 2 rollouts each):
    TINKER_API_KEY=... python tinker_rl.py --max_samples 4 --batch_size 2 --group_size 2
"""
from __future__ import annotations

import argparse
import asyncio
import math
import os
import sys
from collections.abc import Sequence

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import chz
import tinker
from PIL import Image

from tinker_cookbook import checkpoint_utils, renderers
from tinker_cookbook.image_processing_utils import get_image_processor
from tinker_cookbook.rl import train
from tinker_cookbook.rl.types import (
    Action,
    ActionExtra,
    Env,
    EnvGroupBuilder,
    RLDataset,
    RLDatasetBuilder,
    StepResult,
)
from tinker_cookbook.tokenizer_utils import get_tokenizer

from cholec_data import download_frames, load_manifest
from reward import PROMPT, parse_triplets, reward_single

MODEL = "Qwen/Qwen3.5-4B"
RENDERER_NAME = "qwen3_5_disable_thinking"  # no <think> blocks -> plain triplet output


class TripletEnv(Env):
    """Single-turn VLM episode: one frame -> reward for the emitted triplets."""

    def __init__(self, image: Image.Image, gt: dict, renderer: renderers.Renderer):
        self.image = image
        self.gt = gt
        self.renderer = renderer

    async def initial_observation(self):
        convo = [{"role": "user", "content": [
            {"type": "image", "image": self.image},
            {"type": "text", "text": PROMPT},
        ]}]
        return self.renderer.build_generation_prompt(convo), self.renderer.get_stop_sequences()

    async def step(self, action: Action, *, extra: ActionExtra | None = None) -> StepResult:
        message, _termination = self.renderer.parse_response(action)
        text = renderers.get_text_content(message)
        reward = reward_single(parse_triplets(text), self.gt)
        return StepResult(
            reward=reward,
            episode_done=True,
            next_observation=tinker.ModelInput.empty(),
            next_stop_condition=self.renderer.get_stop_sequences(),
            metrics={"triplet_reward": reward},
        )


class TripletGroupBuilder(EnvGroupBuilder):
    """G independent rollouts of the same frame -> one GRPO group."""

    def __init__(self, image: Image.Image, gt: dict, renderer: renderers.Renderer, num_envs: int):
        self.image = image
        self.gt = gt
        self.renderer = renderer
        self.num_envs = num_envs

    async def make_envs(self) -> Sequence[Env]:
        return [TripletEnv(self.image, self.gt, self.renderer) for _ in range(self.num_envs)]

    def logging_tags(self) -> list[str]:
        return ["cholec50", "triplet"]


class TripletRLDataset(RLDataset):
    def __init__(self, rows, cache_dir, batch_size, group_size, renderer):
        self.rows = rows
        self.cache_dir = cache_dir
        self.batch_size = batch_size
        self.group_size = group_size
        self.renderer = renderer

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        batch_rows = self.rows[index * self.batch_size: (index + 1) * self.batch_size]
        builders = []
        for r in batch_rows:
            p = os.path.join(self.cache_dir, r["member"])
            img = Image.open(p).convert("RGB")
            gt = {"triplets": r["triplets"], "instruments": r["instruments"],
                  "verbs": r["verbs"], "targets": r["targets"]}
            builders.append(TripletGroupBuilder(img, gt, self.renderer, self.group_size))
        return builders

    def __len__(self) -> int:
        return math.ceil(len(self.rows) / self.batch_size)


@chz.chz
class TripletRLDatasetBuilder(RLDatasetBuilder):
    cache_dir: str = "/tmp/cholec_frames"
    max_samples: int = 256
    batch_size: int = 16
    group_size: int = 8
    renderer_name: str = RENDERER_NAME
    model_name: str = MODEL

    async def __call__(self) -> tuple[RLDataset, RLDataset | None]:
        tokenizer = get_tokenizer(self.model_name)
        imgproc = get_image_processor(self.model_name)
        renderer = renderers.get_renderer(
            self.renderer_name, tokenizer, image_processor=imgproc, model_name=self.model_name)
        rows = load_manifest("train")[: self.max_samples]
        download_frames([r["member"] for r in rows], self.cache_dir)
        return TripletRLDataset(rows, self.cache_dir, self.batch_size, self.group_size, renderer), None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft_ckpt_file", type=str, default="/tmp/cholec_tinker_sft_ckpt.txt")
    ap.add_argument("--max_samples", type=int, default=256)
    ap.add_argument("--group_size", type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--max_steps", type=int, default=0, help="0 = full dataset (one epoch)")
    ap.add_argument("--max_tokens", type=int, default=256)
    ap.add_argument("--log_path", type=str, default="/tmp/cholec_tinker_rl")
    ap.add_argument("--ckpt_out", type=str, default="/tmp/cholec_tinker_rl_ckpt.txt")
    ap.add_argument("--cache_dir", type=str, default="/tmp/cholec_frames")
    args = ap.parse_args()

    with open(args.sft_ckpt_file) as f:
        sft_path = f.read().strip()
    print(f"RL starting from SFT checkpoint: {sft_path}", flush=True)

    builder = TripletRLDatasetBuilder(
        cache_dir=args.cache_dir,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        group_size=args.group_size,
    )

    config = train.Config(
        learning_rate=args.lr,
        dataset_builder=builder,
        model_name=MODEL,
        recipe_name="cholec_rl",
        renderer_name=RENDERER_NAME,
        max_tokens=args.max_tokens,
        log_path=args.log_path,
        load_checkpoint_path=sft_path,
        loss_fn="importance_sampling",
        lora_rank=16,
        eval_every=0,
        save_every=0,
        remove_constant_reward_groups=True,
        num_groups_to_log=2,
        rollout_json_export=True,
        max_steps=(None if args.max_steps == 0 else args.max_steps),
    )

    asyncio.run(train.main(config))

    rec = checkpoint_utils.get_last_checkpoint(args.log_path, required_key="sampler_path")
    assert rec is not None and rec.sampler_path is not None, "no final sampler checkpoint found"
    print(f"RL final sampler checkpoint: {rec.sampler_path}", flush=True)
    with open(args.ckpt_out, "w") as f:
        f.write(rec.sampler_path + "\n")
    print(f"wrote checkpoint path to {args.ckpt_out}", flush=True)


if __name__ == "__main__":
    main()
