import json
import logging

import torch

import comfy.sd
import comfy.utils
import folder_paths


def _convert_quanto_state_dict(sd, metadata):
    layers = {}
    scaled_acts = []
    out = {}
    for k, v in sd.items():
        if k.endswith(".weight._data"):
            layer = k[:-len(".weight._data")]
            out["{}.weight".format(layer)] = v
            layers[layer] = {"format": "int8_tensorwise"}
        elif k.endswith(".weight._scale"):
            layer = k[:-len(".weight._scale")]
            out["{}.weight_scale".format(layer)] = v.to(torch.float32)
        elif k.endswith(("input_scale", "output_scale")):
            if float(v) != 1.0:
                scaled_acts.append(k)
            continue
        else:
            out[k] = v
    metadata = dict(metadata)
    metadata["_quantization_metadata"] = json.dumps({"layers": layers})
    if scaled_acts:
        logging.warning("{} activation scales are not 1.0 but int8_tensorwise has no slot for them, output will differ from Wan2GP".format(len(scaled_acts)))
    logging.info("Converted {} quanto layers to int8_tensorwise".format(len(layers)))
    return out, metadata


def load_quanto_diffusion_model(unet_path, model_options={}, disable_dynamic=False):
    sd, metadata = comfy.utils.load_torch_file(unet_path, return_metadata=True)
    if metadata is not None and metadata.get("quantization_format") == "quanto":
        sd, metadata = _convert_quanto_state_dict(sd, metadata)
    model = comfy.sd.load_diffusion_model_state_dict(sd, model_options=model_options, metadata=metadata, disable_dynamic=disable_dynamic)
    if model is None:
        raise RuntimeError("ERROR: Could not detect model type of: {}\n{}".format(unet_path, comfy.sd.model_detection_error_hint(unet_path, sd)))
    model.cached_patcher_init = (load_quanto_diffusion_model, (unet_path, model_options))
    return model


class DV_QuantoUNETLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"unet_name": (folder_paths.get_filename_list("diffusion_models"), )}}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_unet"

    CATEGORY = "model/loaders"

    def load_unet(self, unet_name):
        unet_path = folder_paths.get_full_path_or_raise("diffusion_models", unet_name)
        return (load_quanto_diffusion_model(unet_path),)


NODE_CLASS_MAPPINGS = {
    "DV_QuantoUNETLoader": DV_QuantoUNETLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DV_QuantoUNETLoader": "DV Load Quanto UNET (Wan2GP int8)",
}
