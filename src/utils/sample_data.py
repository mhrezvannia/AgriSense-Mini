from __future__ import annotations

import random
from pathlib import Path

import pandas as pd
from PIL import Image, ImageChops, ImageDraw, ImageFilter

from src.utils.io import ensure_dir


CLASS_NAMES = ["healthy", "leaf_spot", "rust"]
REGIONS = ["north_farm", "central_valley", "highland_plot"]
SEASONS = ["spring", "summer", "autumn"]
HUMIDITY = ["low", "medium", "high"]
TEMPERATURE = ["cool", "mild", "warm"]
SOILS = ["loam", "sandy", "clay"]
IMAGING = ["clear", "cloudy", "overcast"]
SENSOR_VIEWS = ["topdown", "angled"]


def _weighted_choice(rng: random.Random, values: list[str], weights: list[float]) -> str:
    return rng.choices(values, weights=weights, k=1)[0]


def _class_conditionals(label: str, rng: random.Random) -> dict[str, str]:
    if label == "healthy":
        return {
            "region": _weighted_choice(rng, REGIONS, [0.45, 0.35, 0.20]),
            "humidity_band": _weighted_choice(rng, HUMIDITY, [0.50, 0.40, 0.10]),
            "temperature_band": _weighted_choice(rng, TEMPERATURE, [0.45, 0.45, 0.10]),
            "season": _weighted_choice(rng, SEASONS, [0.50, 0.30, 0.20]),
            "soil_type": _weighted_choice(rng, SOILS, [0.45, 0.35, 0.20]),
        }
    if label == "leaf_spot":
        return {
            "region": _weighted_choice(rng, REGIONS, [0.20, 0.50, 0.30]),
            "humidity_band": _weighted_choice(rng, HUMIDITY, [0.10, 0.45, 0.45]),
            "temperature_band": _weighted_choice(rng, TEMPERATURE, [0.15, 0.40, 0.45]),
            "season": _weighted_choice(rng, SEASONS, [0.35, 0.45, 0.20]),
            "soil_type": _weighted_choice(rng, SOILS, [0.30, 0.25, 0.45]),
        }
    return {
        "region": _weighted_choice(rng, REGIONS, [0.15, 0.30, 0.55]),
        "humidity_band": _weighted_choice(rng, HUMIDITY, [0.10, 0.35, 0.55]),
        "temperature_band": _weighted_choice(rng, TEMPERATURE, [0.05, 0.25, 0.70]),
        "season": _weighted_choice(rng, SEASONS, [0.10, 0.30, 0.60]),
        "soil_type": _weighted_choice(rng, SOILS, [0.40, 0.15, 0.45]),
    }


def _background_canvas(image_size: int, rng: random.Random, shift_variant: str) -> Image.Image:
    if shift_variant == "overcast":
        base = (236, 239, 238)
    elif shift_variant == "cloudy":
        base = (241, 245, 242)
    else:
        base = (248, 250, 246)
    image = Image.new("RGB", (image_size, image_size), base)
    draw = ImageDraw.Draw(image)
    for _ in range(22):
        x0 = rng.randint(0, image_size - 20)
        y0 = rng.randint(0, image_size - 20)
        size = rng.randint(12, 36)
        shade = rng.randint(-7, 7)
        color = tuple(max(220, min(255, channel + shade)) for channel in base)
        draw.ellipse((x0, y0, x0 + size, y0 + size), fill=color)
    if shift_variant == "cloudy":
        image = image.filter(ImageFilter.GaussianBlur(radius=0.6))
    return image


def _leaf_mask(image_size: int, rng: random.Random) -> Image.Image:
    mask = Image.new("L", (image_size, image_size), 0)
    draw = ImageDraw.Draw(mask)
    center_x = image_size // 2 + rng.randint(-6, 6)
    center_y = image_size // 2 + rng.randint(-3, 3)
    leaf_width = rng.randint(int(image_size * 0.58), int(image_size * 0.72))
    leaf_height = rng.randint(int(image_size * 0.78), int(image_size * 0.88))
    bbox = (
        center_x - leaf_width // 2,
        center_y - leaf_height // 2,
        center_x + leaf_width // 2,
        center_y + leaf_height // 2,
    )
    draw.ellipse(bbox, fill=255)
    tip = Image.new("L", (image_size, image_size), 0)
    tip_draw = ImageDraw.Draw(tip)
    tip_height = rng.randint(16, 26)
    tip_draw.polygon(
        [
            (center_x - leaf_width // 8, bbox[1] + 8),
            (center_x + leaf_width // 8, bbox[1] + 8),
            (center_x, bbox[1] - tip_height),
        ],
        fill=255,
    )
    stem = Image.new("L", (image_size, image_size), 0)
    stem_draw = ImageDraw.Draw(stem)
    stem_draw.line(
        [
            (center_x, bbox[3] - 5),
            (center_x + rng.randint(-10, 10), bbox[3] + rng.randint(18, 28)),
        ],
        fill=255,
        width=5,
    )
    combined = ImageChops.lighter(ImageChops.lighter(mask, tip), stem)
    angle = rng.uniform(-28, 28)
    return combined.rotate(angle, resample=Image.Resampling.BICUBIC)


def _apply_leaf_texture(image: Image.Image, mask: Image.Image, label: str, rng: random.Random) -> Image.Image:
    leaf_layer = Image.new("RGB", image.size, (0, 0, 0))
    draw = ImageDraw.Draw(leaf_layer)
    base_palette = {
        "healthy": (116, 176, 74),
        "leaf_spot": (126, 159, 68),
        "rust": (138, 142, 66),
    }[label]

    for row in range(0, image.size[1], 5):
        shift = rng.randint(-10, 10)
        stripe = tuple(max(45, min(215, channel + shift)) for channel in base_palette)
        draw.rectangle((0, row, image.size[0], row + 4), fill=stripe)

    leaf_layer.putalpha(mask)
    composite = Image.alpha_composite(image.convert("RGBA"), leaf_layer)
    return composite.convert("RGB")


def _draw_veins(image: Image.Image, mask: Image.Image, rng: random.Random) -> Image.Image:
    vein_layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(vein_layer)
    bbox = mask.getbbox()
    if bbox is None:
        return image
    center_x = (bbox[0] + bbox[2]) // 2
    top = bbox[1] + 10
    bottom = bbox[3] - 8
    draw.line((center_x, top, center_x, bottom), fill=(238, 246, 200, 220), width=3)
    for ratio in [0.18, 0.30, 0.42, 0.54, 0.66]:
        y = int(top + (bottom - top) * ratio)
        spread = int((bbox[2] - bbox[0]) * (0.22 + ratio * 0.25))
        curvature = rng.randint(6, 16)
        draw.line((center_x, y, center_x - spread, y - curvature), fill=(224, 235, 190, 170), width=2)
        draw.line((center_x, y, center_x + spread, y - curvature), fill=(224, 235, 190, 170), width=2)
    return Image.alpha_composite(image.convert("RGBA"), vein_layer).convert("RGB")


def _draw_lesions(image: Image.Image, mask: Image.Image, label: str, rng: random.Random) -> Image.Image:
    lesion_layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(lesion_layer)
    bbox = mask.getbbox()
    if bbox is None:
        return image

    if label == "healthy":
        lesion_count = rng.randint(2, 5)
        colors = [(98, 192, 84, 90), (130, 200, 92, 80)]
        radius_range = (2, 5)
    elif label == "leaf_spot":
        lesion_count = rng.randint(11, 18)
        colors = [(82, 58, 35, 210), (112, 74, 46, 170)]
        radius_range = (4, 10)
    else:
        lesion_count = rng.randint(14, 22)
        colors = [(173, 95, 22, 210), (201, 122, 40, 160)]
        radius_range = (3, 8)

    for _ in range(lesion_count):
        x = rng.randint(bbox[0] + 12, bbox[2] - 12)
        y = rng.randint(bbox[1] + 12, bbox[3] - 12)
        radius = rng.randint(*radius_range)
        color = rng.choice(colors)
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)
        if label in {"leaf_spot", "rust"} and rng.random() < 0.35:
            halo_radius = radius + rng.randint(2, 4)
            draw.ellipse(
                (x - halo_radius, y - halo_radius, x + halo_radius, y + halo_radius),
                outline=(232, 201, 120, 110),
                width=1,
            )
    lesion_layer.putalpha(mask.point(lambda value: min(value, 200)))
    return Image.alpha_composite(image.convert("RGBA"), lesion_layer).convert("RGB")


def _render_leaf_image(image_size: int, label: str, rng: random.Random, shift_variant: str) -> Image.Image:
    background = _background_canvas(image_size=image_size, rng=rng, shift_variant=shift_variant)
    mask = _leaf_mask(image_size=image_size, rng=rng)
    image = _apply_leaf_texture(background, mask, label=label, rng=rng)
    image = _draw_veins(image, mask=mask, rng=rng)
    image = _draw_lesions(image, mask=mask, label=label, rng=rng)

    if shift_variant == "cloudy":
        image = image.filter(ImageFilter.GaussianBlur(radius=0.45))
    if shift_variant == "overcast":
        overlay = Image.new("RGB", image.size, (234, 238, 240))
        image = Image.blend(image, overlay, alpha=0.16)
    return image


def generate_sample_dataset(
    image_root: str | Path,
    metadata_path: str | Path,
    image_size: int = 160,
    samples_per_class: int = 60,
    seed: int = 42,
) -> pd.DataFrame:
    rng = random.Random(seed)
    image_root = Path(image_root)
    metadata_path = Path(metadata_path)
    ensure_dir(image_root)
    ensure_dir(metadata_path.parent)

    rows: list[dict[str, str]] = []
    shift_samples = max(6, samples_per_class // 8)
    for label in CLASS_NAMES:
        label_dir = image_root / label
        ensure_dir(label_dir)
        for sample_index in range(samples_per_class):
            sample_id = f"{label[:2]}_{sample_index:03d}"
            imaging_condition = "overcast" if sample_index >= samples_per_class - shift_samples else _weighted_choice(rng, IMAGING[:-1], [0.6, 0.4])
            image_filename = f"{label}/{sample_id}.png"
            image = _render_leaf_image(
                image_size=image_size,
                label=label,
                rng=rng,
                shift_variant=imaging_condition,
            )
            image.save(image_root / image_filename)

            base_conditions = _class_conditionals(label, rng)
            rows.append(
                {
                    "sample_id": sample_id,
                    "image_filename": image_filename.replace("\\", "/"),
                    "label": label,
                    "region": base_conditions["region"],
                    "humidity_band": base_conditions["humidity_band"],
                    "temperature_band": base_conditions["temperature_band"],
                    "season": base_conditions["season"],
                    "soil_type": base_conditions["soil_type"],
                    "imaging_condition": imaging_condition,
                    "sensor_view": _weighted_choice(rng, SENSOR_VIEWS, [0.65, 0.35]),
                }
            )

    frame = pd.DataFrame(rows)
    for row_index in frame.sample(frac=0.06, random_state=seed).index:
        frame.loc[row_index, "soil_type"] = None
    frame.to_csv(metadata_path, index=False)
    return frame
