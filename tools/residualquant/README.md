# ResidualQuant experiments for Inkling-2-Mini (fork-only, not for upstream)

`rq_kv_study.py` fake-quantizes the global-attention KV cache of Inkling-2-Mini in HF transformers.

- `--phase stats`: cross-pass K/V similarity and reconstruction NMSE per scheme (layers 11 and 17, 3 passes).
- `--phase ppl`: prefill a long context, quantize the past KV in place, then score a continuation in chunks. Reports PPL, KL to BF16 and top-1 agreement.

Schemes: `bf16`, `fp8`, `int{2,4}_g{16,32,128}[_had]`, `cb{2,3,4}_had` (rotated Gaussian codebook, per-vector RMS), `rq_<res><anchor>_g32[_had][_nols]` (last-pass anchor + LS-scaled residuals), `rqm_<c1><c2><anchor>_g32_had` (per-pass residual bits or `p` = pointer to anchor).

## Result (2026-11-22, h200-3, ctx 8192, 16 wikitext docs, 8064 scored tokens; only layers 11 and 17 quantized)

| scheme | bits | PPL | KL vs BF16 | top-1 agree |
|---|---|---|---|---|
| bf16 | 16 | 5.5370 | 0 | 1 |
| fp8 | 8 | 5.5456 | 0.00126 | 0.9839 |
| int4_g32_had | 5.0 | 5.5377 | 0.00193 | 0.9814 |
| cb4_had | 4.125 | 5.5372 | 0.00254 | 0.9797 |
| rq_24_g32_had | 3.75 | 5.5504 | 0.00398 | 0.9737 |
| cb3_had | 3.125 | 5.5597 | 0.00602 | 0.9669 |
| rq_22_g32_had | 3.083 | 5.5938 | 0.01219 | 0.9513 |
| int2_g32_had | 3.0 | 5.5959 | 0.01763 | 0.9422 |
| rqm_pp4_g32_had | 1.75 | 5.6325 | 0.03416 | 0.9231 |

Cross-pass cosine to pass 3: K 0.78-0.93, V 0.62-0.78. Residual energy after LS scaling 0.13-0.57.
Conclusion: residual coding is no better than a rotated codebook at equal bits, and plain 4-bit is already close to BF16. ResidualQuant is not worth a kernel for this model.
