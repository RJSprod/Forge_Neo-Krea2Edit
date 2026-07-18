# Manual integration checklist

Run each matrix item from the implementation specification on a pinned Forge Neo revision before release: Krea 2 Raw/Turbo, CFG 1/3, one/two references, normal LoRA loading, 512/1024/non-square outputs, and supported attention backends. Confirm enabled jobs ground both positive and negative conditioning, and perform a forced-error then disabled-job cleanup check.
