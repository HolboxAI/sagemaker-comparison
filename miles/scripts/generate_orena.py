"""ORena VLM rollout generate function for Miles (TEMPLATE).

Miles builds multimodal rollouts through a custom generate function, wired via
`--custom-generate-function-path <module>.<fn>`, plus
`--multimodal-keys '{"image": "images"}'`. The canonical example is
`examples/geo3k_vlm/rollout/generate.py` in the Miles repo.

Our ORena data stores each frame as a base64 data URL inside the user message
(see prepare_data.py). SGLang accepts image data URLs, so this generate function
passes the image through unchanged.

WARNING: the exact GenerateFn signature differs between Miles releases. Before
running the ORena RL leg, diff this file against the installed
`examples/geo3k_vlm/rollout/generate.py` and adjust the return shape. The
GoEmotions leg needs no generate function (text only).
"""
from __future__ import annotations


def generate(args, sample, **kwargs):
    """Build the SGLang request for one ORena sample.

    Expected return (mirrors the geo3k example): a dict SGLang understands,
    with the chat `messages` and the image under the `image` multimodal key.
    """
    messages = list(getattr(sample, "prompt", []) or [])
    images = getattr(sample, "multimodal_inputs", None) or {}
    image = images.get("image")

    if image is not None:
        # Inject the image into the last user message, OpenAI vision style.
        messages = [dict(m) for m in messages]
        if messages and messages[-1]["role"] == "user":
            content = messages[-1]["content"]
            if isinstance(content, str):
                content = [{"type": "image", "image": image},
                           {"type": "text", "text": content}]
            messages[-1]["content"] = content

    return {"messages": messages}
