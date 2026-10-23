"""Image detection, conversion and resizing translated from Pi.

Mirrors ``utils/mime.ts``, ``utils/image-process.ts`` and
``utils/image-resize-core.ts``. Pi and this port execute the same Photon WASM bytes, including its image
codecs and Lanczos3 filter. Wasmtime hosts the module without Node.js.
"""

from __future__ import annotations

import base64
import math
import struct
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from pi_agent._js import to_fixed
from pi_agent.types import JsonValue

if TYPE_CHECKING:
    from ._photon import Photon

IMAGE_TYPE_SNIFF_BYTES = 4100
_PNG_SIGNATURE = bytes([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A])

#: 4.5MB of base64 payload, below Anthropic's 5MB limit.
DEFAULT_MAX_BYTES = int(4.5 * 1024 * 1024)
DEFAULT_MAX_WIDTH = 2000
DEFAULT_MAX_HEIGHT = 2000
DEFAULT_JPEG_QUALITY = 80


def _u32be(buffer: bytes, offset: int) -> int:
    chunk = buffer[offset : offset + 4].ljust(4, b"\0")
    return struct.unpack(">I", chunk)[0]


def _u32le(buffer: bytes, offset: int) -> int:
    chunk = buffer[offset : offset + 4].ljust(4, b"\0")
    return struct.unpack("<I", chunk)[0]


def _u16le(buffer: bytes, offset: int) -> int:
    chunk = buffer[offset : offset + 2].ljust(2, b"\0")
    return struct.unpack("<H", chunk)[0]


def _ascii_at(buffer: bytes, offset: int, text: str) -> bool:
    return buffer[offset : offset + len(text)] == text.encode("ascii")


def _is_png(buffer: bytes) -> bool:
    return len(buffer) >= 16 and _u32be(buffer, 8) == 13 and _ascii_at(buffer, 12, "IHDR")


def _is_animated_png(buffer: bytes) -> bool:
    offset = len(_PNG_SIGNATURE)
    while offset + 8 <= len(buffer):
        length = _u32be(buffer, offset)
        if _ascii_at(buffer, offset + 4, "acTL"):
            return True
        if _ascii_at(buffer, offset + 4, "IDAT"):
            return False
        next_offset = offset + 8 + length + 4
        if next_offset <= offset or next_offset > len(buffer):
            return False
        offset = next_offset
    return False


def _is_bmp(buffer: bytes) -> bool:
    if len(buffer) < 26:
        return False
    declared_size = _u32le(buffer, 2)
    pixel_offset = _u32le(buffer, 10)
    dib_size = _u32le(buffer, 14)
    if declared_size != 0 and declared_size < 26:
        return False
    if pixel_offset < 14 + dib_size:
        return False
    if declared_size != 0 and pixel_offset >= declared_size:
        return False
    if dib_size == 12:
        planes, bits = _u16le(buffer, 22), _u16le(buffer, 24)
    elif 40 <= dib_size <= 124:
        if len(buffer) < 30:
            return False
        planes, bits = _u16le(buffer, 26), _u16le(buffer, 28)
    else:
        return False
    return planes == 1 and bits in (1, 4, 8, 16, 24, 32)


def detect_supported_image_mime_type(buffer: bytes) -> str | None:
    """Pi's ``detectSupportedImageMimeType``: sniff jpg/png/gif/webp/bmp."""
    if buffer.startswith(b"\xff\xd8\xff"):
        return None if len(buffer) > 3 and buffer[3] == 0xF7 else "image/jpeg"
    if buffer.startswith(_PNG_SIGNATURE):
        return "image/png" if _is_png(buffer) and not _is_animated_png(buffer) else None
    if _ascii_at(buffer, 0, "GIF87a") or _ascii_at(buffer, 0, "GIF89a"):
        return "image/gif"
    if _ascii_at(buffer, 0, "RIFF") and _ascii_at(buffer, 8, "WEBP"):
        return "image/webp"
    if _ascii_at(buffer, 0, "BM") and _is_bmp(buffer):
        return "image/bmp"
    return None


def detect_supported_image_mime_type_from_file(path: str) -> str | None:
    with open(path, "rb") as file:
        return detect_supported_image_mime_type(file.read(IMAGE_TYPE_SNIFF_BYTES))


@dataclass
class ProcessedImage:
    ok: bool
    data: str = ""
    mime_type: str = ""
    hints: list[str] = field(default_factory=list)
    message: str = ""


def _base_mime_type(mime_type: str) -> str:
    return mime_type.split(";")[0].strip().lower()


def _normalize_supported_mime_type(mime_type: str) -> str | None:
    base = _base_mime_type(mime_type)
    if base == "image/png":
        return "image/png"
    if base in ("image/jpeg", "image/jpg"):
        return "image/jpeg"
    if base == "image/gif":
        return "image/gif"
    if base == "image/webp":
        return "image/webp"
    return None


def _exif_orientation(data: bytes) -> int:
    """Pi's getExifOrientation: only JPEG APP1 and WebP EXIF (not PNG/TIFF metadata)."""
    tiff = -1
    if data.startswith(b"\xff\xd8"):
        offset = 2
        while offset < len(data) - 1:
            if data[offset] != 0xFF:
                break
            marker = data[offset + 1]
            if marker == 0xFF:
                offset += 1
                continue
            if marker == 0xE1 and data[offset + 4 : offset + 10] == b"Exif\0\0":
                tiff = offset + 10
                break
            if offset + 4 > len(data):
                break
            offset += 2 + int.from_bytes(data[offset + 2 : offset + 4], "big")
    elif data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        offset = 12
        while offset + 8 <= len(data):
            size = int.from_bytes(data[offset + 4 : offset + 8], "little")
            start = offset + 8
            if data[offset : offset + 4] == b"EXIF":
                if start + size <= len(data):
                    tiff = start + (6 if size >= 6 and data[start : start + 6] == b"Exif\0\0" else 0)
                break
            offset = start + size + size % 2
    if tiff < 0 or tiff + 8 > len(data):
        return 1
    endian: Literal["little", "big"] = "little" if data[tiff : tiff + 2] == b"II" else "big"
    directory = tiff + int.from_bytes(data[tiff + 4 : tiff + 8], endian)
    if directory + 2 > len(data):
        return 1
    count = int.from_bytes(data[directory : directory + 2], endian)
    for index in range(count):
        entry = directory + 2 + index * 12
        if entry + 12 > len(data):
            return 1
        if int.from_bytes(data[entry : entry + 2], endian) == 0x0112:
            value = int.from_bytes(data[entry + 8 : entry + 10], endian)
            return value if 1 <= value <= 8 else 1
    return 1


def _oriented_image(photon: Photon, data: bytes) -> int:
    image = photon.load(data)
    orientation = _exif_orientation(data)
    if orientation in (2, 3):
        photon.call("fliph", image)
    if orientation in (3, 4):
        photon.call("flipv", image)
    if orientation not in (5, 6, 7, 8):
        return image
    width, height = photon.dimensions(image)
    source = photon.take_bytes("photonimage_get_raw_pixels", image)
    target = bytearray(len(source))
    for y in range(height):
        for x in range(width):
            origin = (y * width + x) * 4
            index = (x * height + height - 1 - y) if orientation in (5, 6) else ((width - 1 - x) * height + y)
            target[index * 4 : index * 4 + 4] = source[origin : origin + 4]
    rotated = photon.from_pixels(bytes(target), height, width)
    photon.free(image)
    if orientation in (5, 7):
        photon.call("fliph", rotated)
    return rotated


def _convert_to_png(data: bytes) -> bytes | None:
    from ._photon import Photon

    try:
        with Photon() as photon:
            image = _oriented_image(photon, data)
            try:
                return photon.encode(image)
            finally:
                photon.free(image)
    except Exception:
        return None


@dataclass
class ResizedImage:
    data: str
    mime_type: str
    original_width: int
    original_height: int
    width: int
    height: int
    was_resized: bool


def resize_image(
    data: bytes,
    mime_type: str,
    *,
    max_width: int = DEFAULT_MAX_WIDTH,
    max_height: int = DEFAULT_MAX_HEIGHT,
    max_bytes: int = DEFAULT_MAX_BYTES,
    jpeg_quality: int = DEFAULT_JPEG_QUALITY,
) -> ResizedImage | None:
    """Pi's ``resizeImageInProcess``; None when the image cannot fit."""
    from ._photon import Photon

    try:
        with Photon() as photon:
            return _resize_image(photon, data, mime_type, max_width, max_height, max_bytes, jpeg_quality)
    except Exception:
        return None


def _resize_image(
    photon: Photon,
    data: bytes,
    mime_type: str,
    max_width: int,
    max_height: int,
    max_bytes: int,
    jpeg_quality: int,
) -> ResizedImage | None:
    image = _oriented_image(photon, data)
    input_base64_size = math.ceil(len(data) / 3) * 4
    original_width, original_height = photon.dimensions(image)
    if original_width <= max_width and original_height <= max_height and input_base64_size < max_bytes:
        photon.free(image)
        return ResizedImage(
            base64.b64encode(data).decode("ascii"),
            mime_type or "image/png",
            original_width,
            original_height,
            original_width,
            original_height,
            False,
        )

    target_width, target_height = original_width, original_height
    if target_width > max_width:
        target_height = round_half_up(target_height * max_width / target_width)
        target_width = max_width
    if target_height > max_height:
        target_width = round_half_up(target_width * max_height / target_height)
        target_height = max_height

    qualities = list(dict.fromkeys([jpeg_quality, 85, 70, 55, 40]))
    width, height = target_width, target_height
    try:
        while True:
            resized = photon.call("resize", image, width, height, 5)  # Photon SamplingFilter.Lanczos3
            try:
                candidates = [(photon.encode(resized), "image/png")]
                candidates.extend((photon.encode(resized, quality), "image/jpeg") for quality in qualities)
            finally:
                photon.free(resized)
            for candidate, candidate_mime in candidates:
                encoded = base64.b64encode(candidate).decode("ascii")
                size = len(encoded)
                if size < max_bytes:
                    return ResizedImage(
                        encoded, candidate_mime, original_width, original_height, width, height, True
                    )
            if width == 1 and height == 1:
                return None
            next_width = 1 if width == 1 else max(1, math.floor(width * 0.75))
            next_height = 1 if height == 1 else max(1, math.floor(height * 0.75))
            if (next_width, next_height) == (width, height):
                return None
            width, height = next_width, next_height
    except Exception:
        return None
    finally:
        photon.free(image)


def round_half_up(value: float) -> int:
    """JavaScript ``Math.round``."""
    return math.floor(value + 0.5)


def format_dimension_note(result: ResizedImage) -> str | None:
    if not result.was_resized:
        return None
    scale = result.original_width / result.width
    return (
        f"[Image: original {result.original_width}x{result.original_height}, displayed at "
        f"{result.width}x{result.height}. Multiply coordinates by {to_fixed(scale, 2)} to map to original image.]"
    )


def _option(options: Mapping[str, JsonValue], name: str, default: int) -> int:
    value = options.get(name, default)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"Image resize {name} must be a number")
    return int(value)


def process_image(
    data: bytes,
    mime_type: str,
    *,
    auto_resize: bool = True,
    resize_options: Mapping[str, JsonValue] | None = None,
) -> ProcessedImage:
    """Pi's ``processImage``."""
    normalized_mime = _normalize_supported_mime_type(mime_type)
    converted_from: str | None = None
    if normalized_mime is None:
        png = _convert_to_png(data)
        if png is None:
            return ProcessedImage(
                ok=False,
                message="[Image omitted: could not be converted to a supported inline image format.]",
            )
        data, normalized_mime, converted_from = png, "image/png", _base_mime_type(mime_type)

    hints: list[str] = []
    if auto_resize:
        options = resize_options or {}
        resized = resize_image(
            data,
            normalized_mime,
            max_width=_option(options, "maxWidth", DEFAULT_MAX_WIDTH),
            max_height=_option(options, "maxHeight", DEFAULT_MAX_HEIGHT),
            max_bytes=_option(options, "maxBytes", DEFAULT_MAX_BYTES),
            jpeg_quality=_option(options, "jpegQuality", DEFAULT_JPEG_QUALITY),
        )
        if resized is None:
            return ProcessedImage(
                ok=False, message="[Image omitted: could not be resized below the inline image size limit.]"
            )
        if converted_from and converted_from != resized.mime_type:
            hints.append(f"[Image converted from {converted_from} to {resized.mime_type}.]")
        note = format_dimension_note(resized)
        if note:
            hints.append(note)
        return ProcessedImage(ok=True, data=resized.data, mime_type=resized.mime_type, hints=hints)

    if converted_from and converted_from != normalized_mime:
        hints.append(f"[Image converted from {converted_from} to {normalized_mime}.]")
    return ProcessedImage(
        ok=True, data=base64.b64encode(data).decode("ascii"), mime_type=normalized_mime, hints=hints
    )
