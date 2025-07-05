"""Utilities for generating DALL·E tween frames"""

import base64
import logging
from typing import List

from openai import OpenAI

logger = logging.getLogger(__name__)


def generate_dalle_prompts(start_image: str, end_image: str, frame_count: int, api_key: str) -> List[str]:
    """Generate DALL·E prompts for in-between frames.

    Args:
        start_image: Path to the starting keyframe image file.
        end_image: Path to the ending keyframe image file.
        frame_count: Number of intermediate frames to describe.
        api_key: OpenAI API key.

    Returns:
        A list of text prompts for each tween frame.

    Raises:
        Exception: Propagates any errors from the OpenAI API call.
    """
    logger.info(
        "Generating %s DALL·E prompts between %s and %s",
        frame_count,
        start_image,
        end_image,
    )

    with open(start_image, "rb") as f:
        start_b64 = base64.b64encode(f.read()).decode()

    with open(end_image, "rb") as f:
        end_b64 = base64.b64encode(f.read()).decode()

    client = OpenAI(api_key=api_key)

    system_message = (
        "You generate concise DALL·E prompts describing a smooth visual "
        "transition between two images. Number the prompts starting at 1 "
        "and do not include additional commentary."
    )
    user_content = [
        {
            "type": "text",
            "text": (
                f"Create {frame_count} intermediate image descriptions for DALL·E 3 "
                "that morph the first image into the second. Reply with one "
                "numbered prompt per line."
            ),
        },
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{start_b64}"}},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{end_b64}"}},
    ]

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_content},
            ],
            temperature=0.7,
            max_tokens=frame_count * 50,
        )
    except Exception as exc:
        logger.error("OpenAI request failed: %s", exc)
        raise

    text = response.choices[0].message.content.strip()
    prompts = [line.split(".", 1)[-1].strip() for line in text.splitlines() if line.strip()]
    prompts = [p for p in prompts if p]

    if len(prompts) != frame_count:
        logger.warning(
            "Expected %s prompts but received %s", frame_count, len(prompts)
        )

    return prompts
