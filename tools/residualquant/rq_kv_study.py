#!/usr/bin/env python
"""
ResidualQuant feasibility study for Inkling-2-Mini (fake-quant, HF transformers).

Phase A  (--phase stats): capture the global-layer KV of all 3 loop passes on real text and measure
         cross-pass similarity + reconstruction NMSE for each quantization scheme.
Phase B  (--phase ppl):   end-to-end quality. Prefill a long context, fake-quantize the global-layer KV
         cache in place, then teacher-force a continuation in small chunks (re-quantizing each new
         chunk after it is written, i.e. "current chunk BF16, past quantized", like the paper's decode).
         Reports NLL / PPL and KL(bf16 || quant) + top-1 agreement on the continuation.

Cache layout (transformers inkling2_loop): global layers are 5, 11, 17, 23. Core = layers 6..17, 3 passes.
cache index of core layer L in pass p (1-based): L if p == 1 else 24 + (p-2)*12 + (L-6).

Usage:
  python rq_kv_study.py --model <path> --phase stats --ctx 8192 --n-docs 4 --out stats.json
  python rq_kv_study.py --model <path> --phase ppl --ctx 8192 --cont 512 --chunk 64 --n-docs 16 --out ppl.json
"""
import argparse, json, math, time, os, sys
import torch
import torch.nn.functional as F

N_PHYS, CORE_S, CORE_E, N_PASS = 24, 6, 18, 3
CORE_GLOBALS = [11, 17]
OUTER_GLOBALS = [5, 23]


def cache_idx(layer, p):
    return layer if p == 1 else N_PHYS + (p - 2) * (CORE_E - CORE_S) + (layer - CORE_S)


# ---------------------------------------------------------------- quantizers
def hadamard(n, device):
    H = torch.ones(1, 1, device=device, dtype=torch.float32)
    while H.shape[0] < n:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    g = torch.Generator(device="cpu").manual_seed(0)
    signs = (torch.randint(0, 2, (n,), generator=g) * 2 - 1).float().to(device)
    return (H * signs[None, :]) / math.sqrt(n)  # randomized Hadamard, orthogonal


def affine_q(x, bits, g):
    """group-wise asymmetric quant-dequant along last dim. x float32 [..., D]"""
    if bits >= 16:
        return x
    shp = x.shape
    xg = x.reshape(*shp[:-1], shp[-1] // g, g)
    mn, mx = xg.amin(-1, keepdim=True), xg.amax(-1, keepdim=True)
    qmax = 2 ** bits - 1
    s = ((mx - mn) / qmax).clamp(min=1e-8)
    s, mn = s.half().float(), mn.half().float()  # metadata stored as fp16
    q = ((xg - mn) / s).round().clamp(0, qmax)
    return (q * s + mn).reshape(shp)


# Lloyd-Max levels for N(0,1) (symmetric, positive half)
_LM = {4: [0.1284, 0.3881, 0.6568, 0.9424, 1.2562, 1.6181, 2.0690, 2.7326], 2: [0.4528, 1.5104], 3: [0.2451, 0.7560, 1.3439, 2.1520]}


def normcodebook_q(x, bits, R):
    """scale-free-ish: rotate, divide by per-vector RMS (fp16, 16 bits/vector), snap to a Gaussian Lloyd-Max codebook.
    For Inkling V the RMS is exactly 1 so the stored norm could even be dropped."""
    y = x @ R.T
    rms = y.pow(2).mean(-1, keepdim=True).sqrt().clamp(min=1e-8).half().float()
    u = y / rms
    pos = torch.tensor(_LM[bits], device=x.device)
    lv = torch.cat([-pos.flip(0), pos])
    idx = (u[..., None] - lv).abs().argmin(-1)
    return (lv[idx] * rms) @ R


def fp8_q(x):
    return x.to(torch.float8_e4m3fn).float()


def bits_per_elem(bits, g):
    if bits >= 16:
        return 16.0
    return bits + 32.0 / g  # fp16 scale + fp16 offset per group


# A scheme maps the list of per-pass tensors [X1, X2, X3] (float32, [B,H,T,D]) to reconstructions,
# and reports its storage cost in bits/element averaged over passes.
def make_scheme(name):
    # returns fn(xs, rot) -> (recons, bits)
    if name == "bf16":
        return lambda xs, R: (xs, 16.0)
    if name == "fp8":
        return lambda xs, R: ([fp8_q(x) for x in xs], 8.0)
    if name.startswith("int"):  # e.g. int4_g32, int2_g16, int4_g32_had
        parts = name.split("_")
        b, g, had = int(parts[0][3:]), int(parts[1][1:]), "had" in parts

        def fn(xs, R):
            out = []
            for x in xs:
                if had:
                    out.append(affine_q(x @ R.T, b, g) @ R)
                else:
                    out.append(affine_q(x, b, g))
            return out, bits_per_elem(b, g)
        return fn
    if name.startswith("cb"):  # cb4_had: rotated Gaussian codebook with per-vector rms, ~b + 16/D bits
        b = int(name[2])
        return lambda xs, R: ([normcodebook_q(x, b, R) for x in xs], b + 16.0 / xs[0].shape[-1])
    if name.startswith("rqm_"):  # per-pass codes: rqm_<c1><c2><anchorbits>_g32[_had]; c in {p (pointer: alpha*anchor), 2, 4}
        parts = name.split("_")
        codes, ab, g = parts[1][:-1], int(parts[1][-1]), int(parts[2][1:])
        had = "had" in parts

        def fn(xs, R):
            rot = (lambda t, b: affine_q(t @ R.T, b, g) @ R) if had else (lambda t, b: affine_q(t, b, g))
            a_hat = rot(xs[-1], ab)
            out, bits = [], bits_per_elem(ab, g)
            for c, x in zip(codes, xs[:-1]):
                alpha = ((x * a_hat).sum(-1, keepdim=True) / (a_hat * a_hat).sum(-1, keepdim=True).clamp(min=1e-8)).half().float()
                if c == "p":
                    out.append(alpha * a_hat); bits += 16.0 / x.shape[-1]
                else:
                    out.append(alpha * a_hat + rot(x - alpha * a_hat, int(c))); bits += bits_per_elem(int(c), g) + 16.0 / x.shape[-1]
            out.append(a_hat)
            return out, bits / len(xs)
        return fn
    if name.startswith("rq"):  # rq_{resbits}{anchorbits}_g{g}[_had][_nols]  e.g. rq_24_g32_had, rq_22_g32
        parts = name.split("_")
        rb, ab = int(parts[1][0]), int(parts[1][1])
        g = int(parts[2][1:])
        had, nols = "had" in parts, "nols" in parts

        def fn(xs, R):
            anchor = xs[-1]
            if had:
                a_hat = affine_q(anchor @ R.T, ab, g) @ R
            else:
                a_hat = affine_q(anchor, ab, g)
            out = []
            for x in xs[:-1]:
                if nols:
                    alpha = torch.ones_like(x[..., :1])
                else:
                    alpha = (x * a_hat).sum(-1, keepdim=True) / (a_hat * a_hat).sum(-1, keepdim=True).clamp(min=1e-8)
                    alpha = alpha.half().float()  # per token/head fp16 LS coef
                res = x - alpha * a_hat
                if had:
                    rq = affine_q(res @ R.T, rb, g) @ R
                else:
                    rq = affine_q(res, rb, g)
                out.append(alpha * a_hat + rq)
            out.append(a_hat)
            n = len(xs)
            ls_bits = 0 if nols else 16.0 / xs[0].shape[-1]
            bits = (bits_per_elem(ab, g) + (n - 1) * (bits_per_elem(rb, g) + ls_bits)) / n
            return out, bits
        return fn
    raise ValueError(name)


SCHEMES = [
    "bf16", "fp8",
    "int4_g32", "int4_g32_had", "int4_g128_had",
    "int2_g16", "int2_g32_had", "int2_g16_had",
    "rq_22_g32", "rq_22_g32_had", "rq_22_g16_had",
    "rq_24_g32", "rq_24_g32_had", "rq_24_g32_nols",
    "rq_44_g32_had",
    # LoopKV-style pointer sharing (arXiv 2611.04137): pass reuses LS-scaled anchor, no residual
    "cb4_had", "cb3_had", "cb2_had",
    "rqm_pp4_g32_had", "rqm_2p4_g32_had", "rqm_p24_g32_had",
]
# what to do with the 2 non-loop globals (5, 23) and SWA layers: kept as given by --outer
OUTER_FOR = {"bf16": "bf16", "fp8": "fp8"}


# ---------------------------------------------------------------- model / data
def load(model_path, attn):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_path)
    kw = dict(dtype=torch.bfloat16, device_map="cuda")
    if attn:
        kw["attn_implementation"] = attn
    try:
        model = AutoModelForCausalLM.from_pretrained(model_path, **kw)
    except Exception as e:  # multimodal checkpoint
        print("AutoModelForCausalLM failed:", repr(e)[:300], "-> trying AutoModelForImageTextToText", flush=True)
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(model_path, **kw)
    model.eval()
    return tok, model


def get_docs(tok, n_docs, min_tokens, source):
    from datasets import load_dataset
    docs = []
    if source == "wikitext":
        ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="test")
        text = "".join(ds["text"])
        ids = tok(text, return_tensors="pt").input_ids[0]
        for i in range(n_docs):
            s = i * min_tokens
            if s + min_tokens > len(ids):
                break
            docs.append(ids[s:s + min_tokens])
    else:  # pg19-like long books
        ds = load_dataset("emozilla/pg19-test", split="test")
        for row in ds:
            ids = tok(row["text"][:min_tokens * 8], return_tensors="pt").input_ids[0]
            if len(ids) >= min_tokens:
                docs.append(ids[:min_tokens])
            if len(docs) >= n_docs:
                break
    return docs


def layer_kv(cache, idx):
    lay = cache.layers[idx]
    return lay.keys, lay.values


def global_groups():
    """list of (name, [cache indices]) groups that are quantized jointly."""
    groups = [(f"L{L}", [cache_idx(L, p) for p in range(1, N_PASS + 1)]) for L in CORE_GLOBALS]
    outer = [(f"L{L}", [L]) for L in OUTER_GLOBALS]
    return groups, outer


# ---------------------------------------------------------------- phase A
@torch.no_grad()
def phase_stats(args, tok, model):
    docs = get_docs(tok, args.n_docs, args.ctx, args.data)
    from transformers import DynamicCache
    R = hadamard(128, "cuda")
    core, _ = global_groups()
    agg = {}
    for di, ids in enumerate(docs):
        cache = DynamicCache(config=model.config.get_text_config() if hasattr(model.config, "get_text_config") else model.config)
        model(input_ids=ids[None].cuda(), past_key_values=cache, use_cache=True)
        for gname, idxs in core:
            for kv_i, kvname in enumerate(["K", "V"]):
                xs = [layer_kv(cache, i)[kv_i].float() for i in idxs]
                # similarity to last pass
                for p in range(N_PASS - 1):
                    cos = F.cosine_similarity(xs[p], xs[-1], dim=-1)  # [B,H,T]
                    a = (xs[p] * xs[-1]).sum(-1, keepdim=True) / (xs[-1] ** 2).sum(-1, keepdim=True)
                    res_e = ((xs[p] - a * xs[-1]) ** 2).sum(-1) / (xs[p] ** 2).sum(-1).clamp(min=1e-8)
                    for k, v in {
                        f"{gname}/{kvname}/cos_p{p+1}_p3_mean": cos.mean().item(),
                        f"{gname}/{kvname}/cos_p{p+1}_p3_p5": cos.flatten().float().quantile(0.05).item() if cos.numel() < 16_000_000 else float("nan"),
                        f"{gname}/{kvname}/resid_energy_p{p+1}": res_e.mean().item(),
                        f"{gname}/{kvname}/cos_p{p+1}_p3_perhead": cos.mean(dim=(0, 2)).tolist(),
                    }.items():
                        agg.setdefault(k, []).append(v)
                cos12 = F.cosine_similarity(xs[0], xs[1], dim=-1).mean().item()
                agg.setdefault(f"{gname}/{kvname}/cos_p1_p2_mean", []).append(cos12)
                for p in range(N_PASS):
                    agg.setdefault(f"{gname}/{kvname}/norm_p{p+1}", []).append(xs[p].norm(dim=-1).mean().item())
                # scheme NMSE
                for sname in SCHEMES:
                    recon, bits = make_scheme(sname)(xs, R)
                    nmse = [(((r - x) ** 2).sum() / (x ** 2).sum()).item() for r, x in zip(recon, xs)]
                    agg.setdefault(f"{gname}/{kvname}/nmse/{sname}", []).append(nmse)
                    agg[f"bits/{sname}"] = [bits]
        del cache
        torch.cuda.empty_cache()
        print(f"doc {di} done", flush=True)

    def mean(v):
        if isinstance(v[0], list):
            return [sum(c) / len(c) for c in zip(*v)]
        return sum(v) / len(v)
    out = {k: mean(v) for k, v in agg.items()}
    return out


# ---------------------------------------------------------------- phase B
def quantize_cache_slice(cache, s, e, scheme, outer_scheme, R):
    """fake-quantize positions [s, e) of all global layers in place."""
    core, outer = global_groups()
    fn = make_scheme(scheme)
    for _, idxs in core:
        for kv_i in range(2):
            ts = [layer_kv(cache, i)[kv_i] for i in idxs]
            xs = [t[:, :, s:e].float() for t in ts]
            recon, _ = fn(xs, R)
            for t, r in zip(ts, recon):
                t[:, :, s:e] = r.to(t.dtype)
    ofn = make_scheme(outer_scheme)
    for _, idxs in outer:
        for kv_i in range(2):
            t = layer_kv(cache, idxs[0])[kv_i]
            r, _ = ofn([t[:, :, s:e].float()], R)
            t[:, :, s:e] = r[0].to(t.dtype)


@torch.no_grad()
def phase_ppl(args, tok, model):
    from transformers import DynamicCache
    docs = get_docs(tok, args.n_docs, args.ctx + args.cont, args.data)
    R = hadamard(128, "cuda")
    schemes = args.schemes.split(",") if args.schemes else SCHEMES
    res = {s: {"nll": 0.0, "n": 0, "kl": 0.0, "top1": 0} for s in schemes}
    cfg = model.config.get_text_config() if hasattr(model.config, "get_text_config") else model.config
    for di, ids in enumerate(docs):
        ids = ids[None].cuda()
        ctx, cont = ids[:, :args.ctx], ids[:, args.ctx:]
        ref_logp = None
        for s in schemes:
            t0 = time.time()
            cache = DynamicCache(config=cfg)
            model(input_ids=ctx, past_key_values=cache, use_cache=True)
            outer = args.outer if args.outer != "same" else (s if not s.startswith("rq") else "int4_g32_had")
            if s != "bf16":
                quantize_cache_slice(cache, 0, args.ctx, s, outer, R)
            logps = []
            pos = args.ctx
            # first continuation token is predicted by last ctx token: re-run it? we score cont[1:] given chunks
            for c0 in range(0, args.cont, args.chunk):
                chunk = cont[:, c0:c0 + args.chunk]
                out = model(input_ids=chunk, past_key_values=cache, use_cache=True)
                logps.append(F.log_softmax(out.logits.float()[:, :-1], -1))
                if s != "bf16":
                    quantize_cache_slice(cache, pos, pos + chunk.shape[1], s, outer, R)
                pos += chunk.shape[1]
            # targets: next token inside each chunk
            tgt = torch.cat([cont[:, c0 + 1:c0 + args.chunk] for c0 in range(0, args.cont, args.chunk)], 1)
            lp = torch.cat(logps, 1)
            nll = -lp.gather(-1, tgt[..., None]).squeeze(-1)
            res[s]["nll"] += nll.sum().item()
            res[s]["n"] += nll.numel()
            if s == "bf16":
                ref_logp = lp
            elif ref_logp is not None:
                kl = (ref_logp.exp() * (ref_logp - lp)).sum(-1)
                res[s]["kl"] += kl.sum().item()
                res[s]["top1"] += (ref_logp.argmax(-1) == lp.argmax(-1)).sum().item()
            del cache, out, logps, lp
            torch.cuda.empty_cache()
            print(f"doc {di} {s} nll={nll.mean().item():.4f} ({time.time()-t0:.1f}s)", flush=True)
    for s in schemes:
        n = max(res[s]["n"], 1)
        res[s] = {"ppl": math.exp(res[s]["nll"] / n), "nll": res[s]["nll"] / n,
                  "kl": res[s]["kl"] / n, "top1_agree": res[s]["top1"] / n, "tokens": n}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--phase", choices=["stats", "ppl"], required=True)
    ap.add_argument("--ctx", type=int, default=8192)
    ap.add_argument("--cont", type=int, default=512)
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--n-docs", type=int, default=4)
    ap.add_argument("--data", default="wikitext", choices=["wikitext", "pg19"])
    ap.add_argument("--schemes", default="")
    ap.add_argument("--outer", default="bf16", help="scheme for layers 5 and 23 ('same' = int4_g32_had for rq, else same as core)")
    ap.add_argument("--attn", default="", help="attn_implementation override, e.g. sdpa or eager")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    tok, model = load(args.model, args.attn)
    print("loaded", type(model).__name__, flush=True)
    t0 = time.time()
    res = phase_stats(args, tok, model) if args.phase == "stats" else phase_ppl(args, tok, model)
    res["_args"] = vars(args)
    res["_seconds"] = time.time() - t0
    json.dump(res, open(args.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if not isinstance(v, list)}, indent=1)[:20000])


if __name__ == "__main__":
    main()
