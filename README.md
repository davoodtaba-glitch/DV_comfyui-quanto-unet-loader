# ComfyUI Quanto UNET Loader for Wan2GP

Custom node for loading checkpoints saved in Wan2GP's **quanto** int8 format.

ComfyUI does not understand the quanto key layout (`<layer>.weight._data` /
`<layer>.weight._scale`), so stock model detection fails with:

```
KeyError: 'blocks.0.attn.wq.weight'
```

This node rewrites those keys to ComfyUI's native `int8_tensorwise` layout in memory
before loading, then hands the state dict to the normal ComfyUI loader.

## Node

| | |
|---|---|
| Node id | `DV_QuantoUNETLoader` |
| Display name | `DV Load Quanto UNET (Wan2GP int8)` |
| Category | `model/loaders` |
| Output | `MODEL` |

Single input `unet_name`, backed by the same `diffusion_models` list the stock
*Load Diffusion Model* node uses (`models/diffusion_models`, `models/unet`).

## Install

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/davoodtaba-glitch/DV_comfyui-quanto-unet-loader.git
```

Restart ComfyUI.

## Conversion

For every tensor in a file whose safetensors metadata declares
`quantization_format: "quanto"`:

| quanto key | result |
|---|---|
| `<layer>.weight._data` (int8) | `<layer>.weight` |
| `<layer>.weight._scale` (per-row) | `<layer>.weight_scale` (fp32) |
| `<layer>.input_scale`, `<layer>.output_scale` | dropped |

`_quantization_metadata` is then emitted so ComfyUI marks every layer
`int8_tensorwise`. `convrot` is deliberately not set — quanto stores plain
row-major weights, not Hadamard-rotated ones.

Detection, dtype selection and the mixed-precision int8 kernels are all stock
ComfyUI; the node only changes how the file is read.

## Notes

- Weight scales are per-output-channel, which matches quanto's row-wise
  quantization and what ComfyUI's `int8_tensorwise` expects.
- Activation scales are always 1.0 in quanto files (activations are not
  quantized). `int8_tensorwise` has no slot for them, so if a file ever ships
  a non-identity value it is dropped and a warning is logged.
- Files that are not quanto — such as already-converted `int8_tensorwise`
  checkpoints — pass straight through the normal load path.
- The node registers a `cached_patcher_init` factory, so model reloads run the
  conversion again instead of re-reading raw quanto keys.
