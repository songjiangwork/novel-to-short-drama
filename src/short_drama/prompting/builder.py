from __future__ import annotations

from ..io import load_json, load_yaml
from ..paths import TEMPLATES_DIR

_LANGUAGE_LABELS = {"zh": "Chinese", "en": "English", "fr": "French"}
_SPEECH_LANGUAGE_LABELS = {"zh": "Mandarin", "en": "English", "fr": "French"}


def _picture_list(indices: list[int]) -> str:
    labels = [f"<Picture {i}>" for i in indices]
    if len(labels) == 1:
        return labels[0]
    if len(labels) == 2:
        return f"{labels[0]} and {labels[1]}"
    return ", ".join(labels[:-1]) + f", and {labels[-1]}"


def _character_label(character_id: str) -> str:
    return character_id.replace("_", " ").title()


def _language_label(language: str) -> str:
    return _LANGUAGE_LABELS.get(language.split("-", 1)[0].lower(), language)


def _speech_language_label(language: str) -> str:
    return _SPEECH_LANGUAGE_LABELS.get(language.split("-", 1)[0].lower(), language)


def _listener_sentence(listeners: list[str]) -> str:
    if not listeners:
        return ""
    if len(listeners) == 1:
        return f" {listeners[0]} listens silently with a closed/resting mouth."
    return f" {' and '.join(listeners)} listen silently with closed/resting mouths."


def build_prompt(shot_path: str, refs_path: str) -> str:
    shot = load_yaml(shot_path)
    refs = load_json(refs_path)
    template = (TEMPLATES_DIR / "h3_prompt_v1.txt").read_text(encoding="utf-8")

    groups: dict[str, list[int]] = {}
    for ref in refs["images"]:
        if ref["owner_type"] == "character":
            groups.setdefault(ref["owner_id"], []).append(ref["index"])

    audio_map = {ref["owner_id"]: ref["index"] for ref in refs["audios"]}
    scene_ref = next(ref for ref in refs["images"] if ref["owner_type"] == "location")
    location_id = shot["location"]["location_id"]

    mapping_lines = []
    for character in shot["characters"]:
        cid = character["character_id"]
        label = _character_label(cid)
        mapping_lines.append(
            f"- {label} is the same person shown in {_picture_list(groups[cid])}. "
            f"Use those references only for {label}'s face, hairstyle, persistent attributes, "
            "wardrobe, and body proportions."
        )

    mapping_lines.append(
        f"- <Picture {scene_ref['index']}> is the canonical {location_id} environment "
        "and must control the scene/background."
    )

    for speaker, audio_index in audio_map.items():
        mapping_lines.append(
            f"- <Audio {audio_index}> is {_character_label(speaker)}'s VOICE IDENTITY ONLY."
        )

    placement = []
    for character in shot["characters"]:
        label = _character_label(character["character_id"])
        position = character["screen_position"]
        placement.append(
            f"{label} stands on screen-{position}." if position in {"left", "right"}
            else f"{label} is {position}."
        )

    shot_kind = "two-shot" if len(shot["characters"]) == 2 else "shot"
    shot_description = (
        f"A natural {shot['camera']['framing']} {shot_kind} in the {location_id} "
        f"from <Picture {scene_ref['index']}>. "
        + " ".join(placement)
        + " Preserve each person's identity, face, hairstyle, persistent attributes, wardrobe, and body proportions. "
          "Do not merge faces, swap identities, swap clothes, or transfer attributes between them."
    )

    presence_lines = []
    for character in shot["characters"]:
        requirements = []
        if character["presence"]["first_frame"]:
            requirements.append("visible FROM THE FIRST FRAME")
        if character["presence"]["entire_shot"]:
            requirements.append("remain visible throughout the entire clip")
        if requirements:
            presence_lines.append(
                f"{_character_label(character['character_id'])} must be "
                + " and ".join(requirements) + "."
            )
    if presence_lines:
        shot_description += " " + " ".join(presence_lines)

    camera = (
        "Locked static camera. No pan, no dolly, no orbit, no zoom, no reframing, "
        f"no cut, no camera-angle transition. Keep the {location_id} geometry and visual identity stable."
        if shot["camera"]["movement"] == "locked"
        else "Use only the simple camera movement explicitly described by the shot."
    )

    turns = sorted(shot.get("dialogue", []), key=lambda turn: turn["order"])
    dialogue_lines = []
    step = 1
    for index, turn in enumerate(turns):
        speaker = turn["speaker"]
        speaker_label = _character_label(speaker)
        ordinal = "FIRST" if index == 0 else "SECOND" if index == 1 else f"TURN {index + 1}"
        listeners = [
            _character_label(character["character_id"])
            for character in shot["characters"]
            if character["character_id"] != speaker
        ]
        dialogue_lines.append(
            f"{step}. {speaker_label} speaks {ordinal}, "
            f"using the voice identity from <Audio {audio_map[speaker]}>:\n"
            f"<d>[{_language_label(turn['language'])}]{turn['text']}</d>\n"
            f"While {speaker_label} speaks, ONLY {speaker_label}'s mouth should move for speech."
            + _listener_sentence(listeners)
        )
        step += 1
        if index < len(turns) - 1:
            dialogue_lines.append(
                f"{step}. {speaker_label} must then STOP COMPLETELY. No overlap."
            )
            step += 1

    replacements = {
        "{{REFERENCE_MAPPING}}": "\n".join(mapping_lines),
        "{{SHOT_DESCRIPTION}}": shot_description,
        "{{CAMERA_DESCRIPTION}}": camera,
        "{{DIALOGUE_BLOCK}}": "\n\n".join(dialogue_lines) if dialogue_lines else "No spoken dialogue. Do not generate speech.",
        "{{SPEECH_LANGUAGE}}": _speech_language_label(turns[0]["language"]) if turns else "target-language",
        "{{LOCATION_ID}}": location_id,
    }
    for key, value in replacements.items():
        template = template.replace(key, value)

    return template.strip() + "\n"
