# Parameter Golf: 1-10-1 Sandwich Architecture

Entry for the [OpenAI Parameter Golf](https://github.com/openai/parameter-golf) competition. The objective is to minimize validation BPB (bits per byte) on the FineWeb dataset under a strict **16,000,000-byte** artifact limit and a **10-minute** training wallclock on 8xH100 SXM GPUs.

## Architecture

The 16MB cap makes standard depth scaling impossible. A 9-layer transformer at dim=512 fills the budget with ~15M parameters. To push beyond this, we use **Sandwich Parameter Tying** to create 12 effective layers from only 3 unique parameter blocks.

### 1-10-1 Routing

| Block | Role | Count |
|-------|------|-------|
| **A** (Entry) | Projects vocabulary embeddings into the continuous residual stream | 1 unique |
| **B** (Body) | Shared transformer block unrolled 10 times in the forward pass | 1 shared x10 |
| **C** (Exit) | Decouples the shared representation before the output head | 1 unique |

A persistent residual stream (`x0`) from the embedding output is injected at every block to prevent gradient vanishing across the 12-layer effective depth.

### Attention

Grouped Query Attention (GQA) with 16 query heads and 4 key-value heads. Head dimension is 84. QK-normalization with learned per-head gain scaling. Rotary position embeddings (RoPE) with base 10000.

### SwiGLU MLP

Standard GELU expansions are replaced with SwiGLU gated linear units. The gating mechanism provides stronger non-linear capacity per parameter, which is critical when the same block is reused 10 times in sequence. The MLP uses a 2x expansion factor with the gate and value projections fused into a single linear layer.

## Parameters

| Component | Count |
|-----------|------:|
| Per block (attention + MLP + control) | 15,354,176 |
| 3 unique blocks | 46,062,528 |
| Tied embedding (1024 x 1344) | 1,376,256 |
| Skip weights | 15,216 |
| **Total** | **47,454,000** |

Model width is set to `dim=1344`. This is the result of a deliberate trade: we reduced from dim=1536 to make room for FP16 embedding protection in the compressed artifact (see Quantization below).

## Quantization: Hybrid FP16/INT4 Precision

Training runs natively in `torch.bfloat16` with `autocast`. At the end of training, we export using a custom hybrid precision pipeline:

### INT4 (4-bit packed) for transformer weights
All 2D weight matrices in the attention and MLP layers are quantized to 4-bit integers with per-row FP16 scales. Two INT4 values are packed into a single `uint8` byte using bit shifts: `(val_even & 0x0F) | (val_odd << 4)`. Dequantization reconstructs the BF16 tensors using arithmetic shifts.

### FP16 passthrough for the tied embedding
The tied embedding (`tok_emb.weight`) serves as both the input token encoder and the output logits projection. INT4 quantization noise on this single tensor propagates through the entire forward pass twice, making it the dominant source of post-quantization BPB degradation. By keeping it in FP16 passthrough, we eliminate this bottleneck at a cost of ~2.4MB of artifact space.

### FP32 passthrough for control tensors
Small control parameters (attention scales, MLP scales, residual mixing weights, QK gains) are kept in full precision. Their combined footprint is negligible.

### Compression
The mixed-precision payload is serialized via `torch.save` and compressed with `zlib` at level 9. The zlib deflate algorithm exploits the structured sparsity patterns in quantized weight distributions, achieving ~1.95x compression on the raw payload.

## Evaluation: Sliding Window BPB

Standard evaluation splits the validation set into non-overlapping chunks of 1024 tokens. Every chunk boundary resets context to zero, artificially inflating loss for early positions. The average effective context is only ~512 tokens.

Our sliding window evaluation fixes this by using overlapping windows with a stride of 64 tokens. Each scored token gets up to 960 tokens of preceding context. Only the last 64 positions of each window are counted (except the first window, which scores all positions). This ensures every token is scored exactly once with near-maximum context.

The competition allows separate 10-minute budgets for training and evaluation, so the 16x compute increase from overlapping windows is fully accommodated.

Typical improvement: ~0.03 BPB with zero training changes.

## Local Benchmarks (RTX 5080, Blackwell sm_120)

Tested on a single 16GB NVIDIA RTX 5080 running PyTorch 2.10.0+cu128 (nightly, required for sm_120 kernel support).

| Metric | Value |
|--------|-------|
| Compressed artifact (.ptz) | 12.68 MB |
| Code size | 55.4 KB |
| **Total submission** | **12.73 MB** |
| Headroom to 16MB cap | 2.59 MB |
| Starting BPB (step 0) | 4.365 |
| Quantization roundtrip | Verified (all tensors restored) |
| Blackwell sm_120 | Compatible |

## Cluster Configuration (8xH100 SXM)

For the competition run, the following settings are tuned for maximum throughput:

- `GRAD_ACCUM_TOTAL=8`: 1 accumulation step per GPU, minimizing micro-step overhead
- `WARMDOWN_ITERS=500`: Tuned for the ~1200-step regime at our per-step throughput
- `WARMUP_STEPS=20`: Re-enable for `torch.compile` kernel warmup (not counted against wallclock)
- `EVAL_SLIDING_STRIDE=64`: Sliding window for the final post-quantization BPB score

## Running

```bash
# Local single-GPU run (auto-selects grad_accum_steps=32 for VRAM safety)
python train_gpt.py

# Override for smaller local batches
TRAIN_BATCH_TOKENS=131072 python train_gpt.py

# 8xH100 distributed run
torchrun --nproc_per_node=8 train_gpt.py
```
