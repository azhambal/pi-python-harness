"""Deterministic input images; outputs are computed by Pi, not by Pillow."""

import base64
import io
import json
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
image = Image.new("RGBA", (19, 11))
image.putdata(
    [
        ((x * 13) % 256, (y * 23) % 256, ((x + y) * 17) % 256, 80 if x % 3 else 255)
        for y in range(11)
        for x in range(19)
    ]
)
cases = []
for format, mime in (("PNG", "png"), ("JPEG", "jpeg"), ("BMP", "bmp"), ("GIF", "gif"), ("WEBP", "webp")):
    source = image.convert("RGB") if format in ("JPEG", "BMP") else image
    out = io.BytesIO()
    source.save(out, format=format)
    data = base64.b64encode(out.getvalue()).decode()
    for label, options in (
        ("unchanged", {}),
        ("no resize", {"autoResizeImages": False}),
        ("model profile", {"resizeOptions": {"maxWidth": 9, "maxHeight": 6}}),
        ("byte budget", {"resizeOptions": {"maxBytes": 240, "jpegQuality": 30}}),
        ("impossible budget", {"resizeOptions": {"maxBytes": 10}}),
    ):
        cases.append(
            {"name": f"{format} {label}", "data": data, "mimeType": f"image/{mime}", "options": options}
        )
for format in ("JPEG", "WEBP", "PNG"):
    for orientation in range(1, 9):
        exif = Image.Exif()
        exif[0x0112] = orientation
        out = io.BytesIO()
        image.convert("RGB").save(out, format=format, exif=exif)
        cases.append(
            {
                "name": f"{format} EXIF {orientation}",
                "data": base64.b64encode(out.getvalue()).decode(),
                "mimeType": f"image/{format.lower()}",
                "options": {"resizeOptions": {"maxWidth": 8, "maxHeight": 7}},
            }
        )
for mime in ("image/png", "image/bmp"):
    cases.append({"name": f"broken {mime}", "data": "bm90IGFuIGltYWdl", "mimeType": mime})
(ROOT / "tests/parity/image_cases.json").write_text(json.dumps(cases, indent=1) + "\n")
