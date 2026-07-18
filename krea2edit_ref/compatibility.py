REQUIRED = ("patch", "channels", "tdim", "txtdim", "txtlayers", "first", "blocks", "tmlp", "txtfusion", "txtmlp", "last", "tproj", "pe_embedder", "_unpack_context")

def require_krea_engine(engine):
    if engine.__class__.__name__ != "Krea2" or not hasattr(engine, "text_processing_engine_qwen"):
        raise RuntimeError("Krea 2 Edit Reference requires a Krea 2 Raw or Krea 2 Turbo model.")
    try: dm = engine.forge_objects.unet.model.diffusion_model
    except Exception as exc: raise RuntimeError("Krea 2 Edit Reference requires a Krea 2 Raw or Krea 2 Turbo model.") from exc
    missing = [name for name in REQUIRED if not hasattr(dm, name)]
    if missing: raise RuntimeError("Unsupported Forge Krea 2 implementation. The extension must be updated for this Forge Neo version. Missing: " + ", ".join(missing))
    if not callable(getattr(engine, "text_processing_engine_qwen", None)): raise RuntimeError("Krea 2 Qwen3-VL image support is unavailable in this Forge build.")
    return dm
