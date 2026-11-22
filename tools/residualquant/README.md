# ResidualQuant experiments for Inkling-2-Mini (fork-only, not for upstream)

`rq_kv_study.py` fake-quantizes the global-attention KV cache of Inkling-2-Mini in HF transformers.

- `--phase stats`: cross-pass K/V similarity and reconstruction NMSE per scheme (layers 11 and 17, 3 passes).
- `--phase ppl`: prefill a long context, quantize the past KV in place, then score a continuation in chunks. Reports PPL, KL to BF16 and top-1 agreement.

Schemes: `bf16`, `fp8`, `int{2,4}_g{16,32,128}[_had]`, `cb{2,3,4}_had` (rotated Gaussian codebook, per-vector RMS), `rq_<res><anchor>_g32[_had][_nols]` (last-pass anchor + LS-scaled residuals), `rqm_<c1><c2><anchor>_g32_had` (per-pass residual bits or `p` = pointer to anchor).
