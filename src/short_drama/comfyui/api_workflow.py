from __future__ import annotations
from typing import Any

from ..io import load_yaml
from ..paths import PROFILES_DIR
from .dimensions import h3_length_for_seconds, resolution_from_profile

REQUIRED_NODE_CLASSES = {
    "UNETLoader", "CLIPLoader", "VAELoader", "RandomNoise", "KSamplerSelect",
    "BasicScheduler", "BasicGuider", "SamplerCustomAdvanced",
    "MiniMaxH3ReferenceToVideo", "VAEDecode", "VAEDecodeAudio",
    "CreateVideo", "SaveVideo", "LoadImage", "LoadAudio",
}


def _profile(request: dict[str, Any]) -> dict[str, Any]:
    return load_yaml(PROFILES_DIR / f"{request['profile']}.yaml")


def _runtime_name(ref: dict[str, Any]) -> str:
    name = ref.get("runtime_filename")
    if not name:
        raise ValueError(f"Reference is not staged: {ref.get('asset_id')}")
    return str(name).replace("\\", "/")


def build_api_workflow(request: dict[str, Any]) -> dict[str, Any]:
    """Build flat ComfyUI API-format H3 R2V workflow."""
    profile = _profile(request)
    if profile.get("lora", {}).get("turbo_lightning") is not False:
        raise ValueError(
            "h3_v1 API workflow supports only the frozen LoRA-disabled profile "
            "(lora.turbo_lightning: false)"
        )

    width, height = resolution_from_profile(
        profile["video"]["megapixels"], profile["video"]["aspect_ratio"], 32
    )
    length = h3_length_for_seconds(
        request["duration_seconds"], profile["video"]["fps"]
    )
    model = profile["model"]

    workflow: dict[str, Any] = {
        "119": {"class_type":"VAELoader","inputs":{"vae_name":model["video_vae"]},"_meta":{"title":"Video VAE"}},
        "120": {"class_type":"VAELoader","inputs":{"vae_name":model["audio_vae"]},"_meta":{"title":"Audio VAE"}},
        "127": {"class_type":"UNETLoader","inputs":{"unet_name":model["diffusion"],"weight_dtype":"default"},"_meta":{"title":"H3 UNET"}},
        "128": {"class_type":"CLIPLoader","inputs":{"clip_name":model["clip"],"type":"minimax","device":"default"},"_meta":{"title":"H3 Text Encoder"}},
        "129": {"class_type":"RandomNoise","inputs":{"noise_seed":int(request["seed"])},"_meta":{"title":"Fixed Seed"}},
        "123": {"class_type":"KSamplerSelect","inputs":{"sampler_name":profile["sampling"]["sampler"]},"_meta":{"title":"Sampler"}},
        "124": {"class_type":"BasicScheduler","inputs":{"model":["127",0],"scheduler":profile["sampling"]["scheduler"],"steps":profile["sampling"]["steps"],"denoise":1.0},"_meta":{"title":"Scheduler"}},
        "136": {"class_type":"MiniMaxH3ReferenceToVideo","inputs":{
            "clip":["128",0],"vae":["119",0],"audio_vae":["120",0],
            "prompt":request["prompt"],"width":width,"height":height,"length":length,
            "ref_image_size":profile["reference"]["ref_image_size"],
        },"_meta":{"title":"MiniMax H3 Reference to Video"}},
        "126": {"class_type":"BasicGuider","inputs":{"model":["127",0],"conditioning":["136",0]},"_meta":{"title":"Guider"}},
        "125": {"class_type":"SamplerCustomAdvanced","inputs":{"noise":["129",0],"guider":["126",0],"sampler":["123",0],"sigmas":["124",0],"latent_image":["136",1]},"_meta":{"title":"Sampler Custom Advanced"}},
        "122": {"class_type":"VAEDecode","inputs":{"samples":["125",0],"vae":["119",0]},"_meta":{"title":"Video Decode"}},
        "121": {"class_type":"VAEDecodeAudio","inputs":{"samples":["125",0],"vae":["120",0]},"_meta":{"title":"Audio Decode"}},
        "130": {"class_type":"CreateVideo","inputs":{"images":["122",0],"audio":["121",0],"fps":profile["video"]["fps"],"bit_depth":8,"color_space":"sRGB"},"_meta":{"title":"Create Video"}},
        "92": {"class_type":"SaveVideo","inputs":{"video":["130",0],"filename_prefix":f"video/short_drama/{request['project_id']}/{request['output_prefix']}","format":"auto","codec":"auto"},"_meta":{"title":"Save Video"}},
    }

    h3_inputs = workflow["136"]["inputs"]
    for ref in request["references"].get("images", []):
        node_id = str(200 + int(ref["index"]))
        workflow[node_id] = {"class_type":"LoadImage","inputs":{"image":_runtime_name(ref)},"_meta":{"title":ref["asset_id"]}}
        h3_inputs[f"ref_images.ref_image_{int(ref['index'])-1}"] = [node_id, 0]

    for ref in request["references"].get("audios", []):
        node_id = str(300 + int(ref["index"]))
        workflow[node_id] = {"class_type":"LoadAudio","inputs":{"audio":_runtime_name(ref)},"_meta":{"title":ref["asset_id"]}}
        h3_inputs[f"ref_audios.ref_audio_{int(ref['index'])-1}"] = [node_id, 0]

    if any(k in h3_inputs for k in ("ref_images", "ref_audios")):
        raise AssertionError("H3 dynamic reference inputs must remain flat dotted keys")
    return workflow
