import re
import struct
import socket
import os


def wii_type(cat, filename):
    return extract_id(filename)


def extract_id(filename: str) -> int:
    """Extracts the integer ID from strings like 'hair_14', 'eye_03.png'."""
    nums = re.findall(r"\d+", filename)
    return int(nums[-1]) if nums else 0


def build_libmii_state(matched_data: dict) -> dict:
    """
    OPTION 1: Unpacked libmii dictionary variables.
    Used for custom rendering, logic, or JSON transmission.
    """

    def get_var(cat, key="asset", default=0):
        val = matched_data.get(cat, {}).get(key, default)
        if key == "asset" and isinstance(val, str):
            return wii_type(cat, val)
        return val

    glasses_val = (get_var("glasses") + 1) if "glasses" in matched_data else 0
    mustache_val = (get_var("mustaches") + 1) if "mustaches" in matched_data else 0
    beard_val = (get_var("beards") + 1) if "beards" in matched_data else 0
    hair_col = get_var("hair", "color_id", 0)

    return {
        "faceShape": get_var("heads"),
        "skinColor": get_var("heads", "color_id", 0),
        "facialFeature": 0,
        "hairType": get_var("hair"),
        "hairColor": hair_col,
        "hairPart": 0,
        "eyebrowType": get_var("eyebrows"),
        "eyebrowRotation": 6,
        "eyebrowColor": get_var("eyebrows", "color_id", hair_col),
        "eyebrowSize": 4,
        "eyebrowVertPos": 10,
        "eyebrowHorizSpacing": 4,
        "eyeType": get_var("eyes"),
        "eyeRotation": 4,
        "eyeColor": get_var("eyes", "color_id", 0),
        "eyeSize": 4,
        "eyeVertPos": 10,
        "eyeHorizSpacing": 4,
        "noseType": get_var("noses"),
        "noseSize": 4,
        "noseVertPos": 9,
        "lipType": get_var("mouths"),
        "lipColor": get_var("mouths", "color_id", 0),
        "lipSize": 4,
        "lipVertPos": 9,
        "beardType": beard_val,
        "mustacheType": mustache_val,
        "mustacheSize": 4,
        "mustacheVertPos": 8,
        "facialHairColor": hair_col,
        "glassesType": glasses_val,
        "glassesColor": get_var("glasses", "color_id", 0),
        "glassesSize": 4,
        "glassesVertPos": 10,
        "mole": 0,
        "moleSize": 0,
        "moleVertPos": 0,
        "moleHorizPos": 0,
    }


def _set(data: bytearray, off: int, nbytes: int, hi: int, lo: int, value: int):
    """Set bits [hi:lo] (big-endian, bit 0 = LSB) of the nbytes word at off."""
    word = int.from_bytes(data[off : off + nbytes], "big")
    mask = ((1 << (hi - lo + 1)) - 1) << lo
    word = (word & ~mask) | ((value << lo) & mask)
    data[off : off + nbytes] = word.to_bytes(nbytes, "big")


def build_official_74byte_mii(matched_data, template_path="template.mii") -> bytes:
    if os.path.exists(template_path):
        with open(template_path, "rb") as f:
            data = bytearray(f.read(74))
        if len(data) != 74:
            raise ValueError(f"template.mii is {len(data)} bytes, expected 74")
    else:
        data = bytearray(74)

    print(f"decode(template): {decode_mii(bytes(data))}")

    def get_var(cat, key="asset", default=0):
        val = matched_data.get(cat, {}).get(key, default)
        if key == "asset" and isinstance(val, str):
            return wii_type(cat, val)
        return val

    hair_color = get_var("hair", "color_id", 0)

    data[0] &= 0x7F  # invalid = 0

    _set(data, 0x20, 2, 15, 13, get_var("heads") & 0x7)  # face shape
    _set(data, 0x20, 2, 12, 10, get_var("heads", "color_id") & 0x7)  # skin
    _set(data, 0x22, 2, 15, 9, get_var("hair") & 0x7F)  # hair type
    _set(data, 0x22, 2, 8, 6, hair_color & 0x7)  # hair colour
    _set(data, 0x24, 4, 31, 27, get_var("eyebrows") & 0x1F)  # brow type
    _set(data, 0x24, 4, 15, 13, get_var("eyebrows", "color_id", hair_color) & 0x7)
    _set(data, 0x28, 4, 31, 26, get_var("eyes") & 0x3F)  # eye type
    _set(data, 0x28, 4, 15, 13, get_var("eyes", "color_id") & 0x7)
    _set(data, 0x2C, 2, 15, 12, get_var("noses") & 0xF)  # nose type
    _set(data, 0x2E, 2, 15, 11, get_var("mouths") & 0x1F)  # lip type
    _set(data, 0x2E, 2, 10, 9, get_var("mouths", "color_id") & 0x3)

    # Optional extras (0 = none on the Wii, so +1 only if detected)
    gl = (get_var("glasses") + 1) if "glasses" in matched_data else 0
    mu = (get_var("mustaches") + 1) if "mustaches" in matched_data else 0
    be = (get_var("beards") + 1) if "beards" in matched_data else 0
    _set(data, 0x30, 2, 15, 12, gl & 0xF)
    _set(data, 0x30, 2, 11, 9, get_var("glasses", "color_id") & 0x7)
    _set(data, 0x32, 2, 15, 14, mu & 0x3)
    _set(data, 0x32, 2, 13, 12, be & 0x3)
    _set(data, 0x32, 2, 11, 9, hair_color & 0x7)

    _set(data, 0x20, 2, 9, 6, 0)  # facialFeature = none

    print(f"decode(data): {decode_mii(bytes(data))}")
    return bytes(data)


def send_mii_packet(
    matched_data: dict, wii_ip: str, port: int = 8080, format_type: str = "official"
):
    """
    Sends either the 74-byte official blob ('official') or
    the unpacked libmii flat struct bytes ('libmii').
    """
    if format_type == "official":
        packet = build_official_74byte_mii(matched_data)
    else:
        # 38-byte flat array for custom apps
        state = build_libmii_state(matched_data)
        field_order = [
            "faceShape",
            "skinColor",
            "facialFeature",
            "hairType",
            "hairColor",
            "hairPart",
            "eyebrowType",
            "eyebrowRotation",
            "eyebrowColor",
            "eyebrowSize",
            "eyebrowVertPos",
            "eyebrowHorizSpacing",
            "eyeType",
            "eyeRotation",
            "eyeColor",
            "eyeSize",
            "eyeVertPos",
            "eyeHorizSpacing",
            "noseType",
            "noseSize",
            "noseVertPos",
            "lipType",
            "lipColor",
            "lipSize",
            "lipVertPos",
            "beardType",
            "mustacheType",
            "mustacheSize",
            "mustacheVertPos",
            "facialHairColor",
            "glassesType",
            "glassesColor",
            "glassesSize",
            "glassesVertPos",
            "mole",
            "moleSize",
            "moleVertPos",
            "moleHorizPos",
        ]
        packet = struct.pack(
            f"{len(field_order)}B", *[int(state[k]) for k in field_order]
        )

    if wii_ip:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.sendto(packet, (wii_ip, port))
        print(
            f"Sent {len(packet)}-byte '{format_type}' packet to Wii ({wii_ip}:{port})"
        )

    return packet


def decode_mii(b: bytes) -> dict:
    g = lambda off, n, hi, lo: (
        (int.from_bytes(b[off : off + n], "big") >> lo) & ((1 << (hi - lo + 1)) - 1)
    )
    return {
        "len": len(b),
        "invalid": g(0, 2, 15, 15),
        "face": g(0x20, 2, 15, 13),
        "skin": g(0x20, 2, 12, 10),
        "hair": g(0x22, 2, 15, 9),
        "hairCol": g(0x22, 2, 8, 6),
        "brow": g(0x24, 4, 31, 27),
        "browCol": g(0x24, 4, 15, 13),
        "browVert": g(0x24, 4, 8, 4),
        "eye": g(0x28, 4, 31, 26),
        "eyeCol": g(0x28, 4, 15, 13),
        "eyeVert": g(0x28, 4, 20, 16),
        "nose": g(0x2C, 2, 15, 12),
        "noseVert": g(0x2C, 2, 7, 3),
        "lip": g(0x2E, 2, 15, 11),
        "lipCol": g(0x2E, 2, 10, 9),
        "glasses": g(0x30, 2, 15, 12),
        "mustache": g(0x32, 2, 15, 14),
        "beard": g(0x32, 2, 13, 12),
    }
