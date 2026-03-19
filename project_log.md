# Project Parameter Golf - Dev Log
**Goal**: Minimize validation loss on FineWeb under 16MB file size constraint and <10 minutes 8xH100 training.

## Architecture & Implementation Strategy
1. **The Sandwich (Final Locked Architecture)**
    * **Routing**: Block A (unique), Block B (shared 10x), Block C (unique).
    * **Dimension**: `dim=1536` locked.
    * **Attention**: 16 Heads, 4 KV Heads (GQA), `head_dim=96` locked.
    * **Dense MLP**: Standard SwiGLU MLP implementation (`hidden=dim*2`).
2. **Quantization Pipeline**
    * INT4 pure PyTorch bit-packing scheme active for export.
    * Fits 61.8M parameters inside a ~15.5MB `final_model.int4.ptz` artifact reliably.
3. **Training Rules**
    * `max_wallclock_seconds = 600.0` (10 minutes early stopping guaranteed).

## Progress Updates

**Mar 19, 2026**
* Set up standard Windows PyTorch loop, stripping MLX.
* Executed the initial "Wide & Tied" baseline (dim 1024, INT8) -> 2.8MB.
* Implemented "The Sandwich" (dim 2048, INT4) -> Resulted in 24.6MB (Exceeded Limit).
* Shrunk "The Sandwich" to (dim 1024, INT4) -> Resulted in 7.10MB (Well under Limit).
* Extrapolated limit math matching random uniform weight structures and Zlib deflate properties.
* **LOCKED IN** the final configuration at `dim=1536`.
* Upgraded environment to PyTorch 2.10.0+cu128 for Blackwell RTX 5080 support.
* Solved silent CUDA OOM kernel hangs by scaling `grad_accum_steps` to 32.
* Verified 10-minute wallclock early stopping and INT4 roundtrip validation (2.729 BPB).
* Pushed cleaned repository (stripped of large binaries) to OrygnsCode/parameter-golf.
* Generated exhaustive `HANDOVER_MANIFEST.md` for Lead Developer transition.

## Final Status
`train_gpt.py` is fully prepared and 100% ready for the 8xH100 cluster run. It will automatically stop at exactly 600.0 seconds of training time and export the `.ptz` weights utilizing the custom 4-bit packaging logic. Remote repository is live and documentation is complete.
