# Forge Krea2Edit Reference

Standalone txt2img reference conditioning for **Forge Neo Krea 2**. It supplies both Krea2Edit paths: Qwen3-VL image grounding and clean VAE reference tokens. It is designed for the Forge Neo `Krea2` engine API and should be tested against the exact Forge revision deployed by the user.

## Use

1. Load a Krea 2 Raw or Turbo model (including its Qwen3-VL encoder) in Forge.
2. In **txt2img**, open **Krea 2 Edit Reference (ImageStitch)** and enable it.
3. Add one source/subject image, or two images ordered **scene first, subject second**.
4. Load the Krea 2 Identity Edit LoRA through Forge's usual LoRA UI or prompt syntax. This extension never discovers, loads, validates, or adjusts a LoRA.
5. Use `fit` normally; use `crop (legacy)` for legacy geometry comparisons.

Keep outputs at roughly 2 MP or below. For difficult removals, Krea 2 Raw with real CFG is generally preferable. Fidelity values other than 1.0 create a dense attention mask; this can use more memory and slow FlashAttention. Grounding resolution only affects Qwen3-VL; original pixels are retained for VAE reference encoding.

Hires. fix, img2img, inpainting, non-Krea engines, and more than two references are intentionally rejected in this release. Reference image files are not stored in infotext, so reproduction requires the same ordered files.

## Installation

Place this repository in Forge's `extensions/` directory, then restart Forge. `requirements.txt` intentionally adds no packages.

## Attribution and licenses

The implementation follows the public Krea2Edit recipe and is informed by [ComfyUI-Krea2Edit](https://github.com/lbouaraba/comfyui-krea2edit) (Apache-2.0). Forge Neo is AGPL-3.0; ensure your distribution complies with applicable licenses.
