"""goemotions-v1: 28-label multi-label emotion classification (single-turn).

The model is asked to reply with a comma-separated list of GoEmotions label
names; the reward is the multi-label Jaccard similarity between the parsed
prediction and the ground-truth label set. A pure-trace reward (no runtime
needed) — the GoEmotions GRPO environment for Exp 4.

Parity (Fireworks §7): reward = Jaccard on the predicted vs gold label sets;
micro-F1 is tracked separately at eval, not used as the RL reward.
"""

import json
import re

import verifiers.v1 as vf

SYSTEM = (
    "Classify the text into one or more of the 28 GoEmotions labels. "
    "Reply with a comma-separated list of label names only."
)


def parse_labels(text: str) -> set[str]:
    out: set[str] = set()
    for tok in re.split(r"[,\n;|]+", text or ""):
        tok = tok.strip().lower().strip(".,;:!?\"'` ")
        if tok:
            out.add(tok)
    return out


class GoEmotionsData(vf.TaskData):
    labels: list[str]
    """Ground-truth GoEmotions label names for this example."""


class GoEmotionsTask(vf.Task[GoEmotionsData]):
    @vf.stop
    async def single_turn(self, trace: vf.Trace) -> bool:
        # Single-turn classification: refuse a second turn so the model answers once.
        return trace.num_turns >= 1

    @vf.reward(weight=1.0)
    async def jaccard(self, trace: vf.Trace) -> float:
        reply = trace.last_reply or ""
        if "</think>" in reply:
            reply = reply.split("</think>")[-1]
        pred = parse_labels(reply)
        gold = set(self.data.labels)
        union = pred | gold
        if not union:
            return 1.0
        return len(pred & gold) / len(union)


class GoEmotionsConfig(vf.TasksetConfig):
    data_path: str = "/workspace/data/goemotions/rl/train.jsonl"
    """Local JSONL path. Each row: {messages: [system, user], ground_truth: {labels: [...]}}."""


class GoEmotionsTaskset(vf.Taskset[GoEmotionsTask, GoEmotionsConfig]):
    def load(self) -> list[GoEmotionsTask]:
        rows = [json.loads(line) for line in open(self.config.data_path)]
        tasks = []
        for i, r in enumerate(rows):
            sys_msg = next(
                (m["content"] for m in r["messages"] if m["role"] == "system"), SYSTEM
            )
            user_msg = next(
                (m["content"] for m in r["messages"] if m["role"] == "user"), ""
            )
            labels = list(r["ground_truth"]["labels"])
            tasks.append(
                GoEmotionsTask(
                    GoEmotionsData(
                        idx=i,
                        prompt=user_msg,
                        system_prompt=sys_msg,
                        labels=labels,
                    ),
                    self.config.task,
                )
            )
        return tasks
