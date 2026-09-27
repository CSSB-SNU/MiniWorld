"""Numerical model only: share exp(bias) across pair rows, approximate exp(QK).

The original BF16 Q/K/V are retained. No timing or actual CUDA qualification
is claimed. Finite inputs outside a fitted interval use exact exp in this
model, and masked probabilities are zeroed explicitly.
"""
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
source = (HERE / 'bench.py').read_text().split('\nwith torch.no_grad():')[0]
exec(compile(source, str(HERE / 'bench.py'), 'exec'))
rows = [0, 17, 257, 767]
fits = {}
for bound, degree in [(1., 3), (2., 3), (2., 4)]:
    grid = np.linspace(-bound, bound, 200001)
    target = np.exp(grid)
    coeff = np.linalg.lstsq(np.polynomial.polynomial.polyvander(grid, degree) / target[:, None], np.ones_like(grid), rcond=None)[0]
    relative = np.polynomial.polynomial.polyval(grid, coeff) / target - 1
    fits[f'b{int(bound)}d{degree}'] = dict(bound=bound, degree=degree, coefficients=list(coeff), real_max_relative_error=float(np.max(np.abs(relative))))
results = []
with torch.no_grad():
    q, k, v, g, b = pf.prologue(x, W, impl='fpf', ending=ending)
    shape = (1, a.length, 4, a.length, 32)
    for strength in [1., 2., 4.]:
        qs = (q.float() * strength).bfloat16()
        ks = (k.float() * strength).bfloat16()
        actual = pf.core_attention(qs, ks, v, b, m5, core='tier:triattn_native').reshape(shape)[:, rows].double()
        qq = qs.reshape(shape)[:, rows].double()
        kk = ks.reshape(shape)[:, rows].double()
        vv = v.reshape(shape)[:, rows].double()
        bias = b.reshape(1, 1, 4, a.length, a.length).double()
        logits_qk = qq @ kk.transpose(-1, -2) / 32**.5
        mask = m5.expand(1, a.length, 1, 1, a.length)[:, rows]
        logits = (logits_qk + bias).masked_fill(~mask, -torch.inf)
        reference = logits.softmax(-1) @ vv
        base_error = rms(actual, reference)
        # Per-query bias normalization is shared by all pair rows, and
        # cancels in P/sum(P). The 2^-64 factor keeps BF16 P well scaled.
        bias_factor = torch.exp(bias - bias.max(-1, keepdim=True).values) * 2.**-64
        for mode in ['native_factor'] + list(fits):
            if mode == 'native_factor':
                y = torch.exp(logits_qk)
                fraction = 1.
            else:
                fit = fits[mode]
                y = torch.full_like(logits_qk, fit['coefficients'][-1])
                for coefficient in reversed(fit['coefficients'][:-1]):
                    y = y * logits_qk + coefficient
                inside = logits_qk.abs() <= fit['bound']
                fraction = float(inside.double().mean())
                y = torch.where(inside, y, torch.exp(logits_qk))
            p = (y * bias_factor).masked_fill(~mask, 0.).bfloat16().double()
            answer = ((p @ vv) / p.sum(-1, keepdim=True)).bfloat16().double()
            error = rms(answer, reference)
            item = dict(strength=strength, mode=mode, rms=error, baseline_rms=base_error, ratio=error/base_error, fraction_inside=fraction, finite=bool(torch.isfinite(answer).all()))
            results.append(item)
            print('MODEL', json.dumps(item), flush=True)
Path(a.output).write_text(json.dumps(dict(model_only=True, ending=ending, rows=rows, fits=fits, results=results), indent=2) + '\n')
