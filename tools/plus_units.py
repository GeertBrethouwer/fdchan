#!/usr/bin/env python3
"""Statistics in wall units: python3 tools/plus_units.py data/stats.dat RE [log.dat t_start]

Prints u_tau (from the mean profile in stats.dat, both walls averaged), Re_tau, the centreline U+ and the
peaks of the rms profiles. With log.dat and t_start it also gives the time-averaged Re_tau from log.dat.
"""
import math
import sys

f, re = sys.argv[1], float(sys.argv[2])
rows = [list(map(float, l.split())) for l in open(f) if not l.startswith('#')]
y = [r[0] for r in rows]
U = [r[1] for r in rows]
n = len(y)
# wall gradients from a 4-point one-sided fit (Lagrange derivative at the wall)
def dwall(ys, fs):
    x0 = ys[0]
    d = 0.0
    for i in range(len(ys)):
        w = 0.0
        for m in range(len(ys)):
            if m == i:
                continue
            p = 1.0 / (ys[i] - ys[m])
            for l in range(len(ys)):
                if l != i and l != m:
                    p *= (x0 - ys[l]) / (ys[i] - ys[l])
            w += p
        d += w * fs[i]
    return d
gb = dwall(y[:5], U[:5])
gt = -dwall(y[::-1][:5], U[::-1][:5])
ut = math.sqrt(0.5 * (gb + gt) / re)
print('u_tau = %.5f   Re_tau = %.1f   (walls: %.1f, %.1f)' % (ut, ut * re, math.sqrt(gb / re) * re, math.sqrt(gt / re) * re))
c = n // 2
print('centreline U+ = %.2f' % (U[c] / ut))
for name, col in (('u_rms+', 4), ('v_rms+', 5), ('w_rms+', 6)):
    v = [math.sqrt(max(r[col], 0)) / ut for r in rows]
    j = max(range(n // 2), key=lambda k: v[k])
    print('%s peak %.3f at y+ = %.1f' % (name, v[j], (y[j] + 1) * ut * re))
uv = [-r[7] / ut**2 for r in rows]
j = max(range(n // 2), key=lambda k: uv[k])
print("-u'v'+ peak %.3f at y+ = %.1f" % (uv[j], (y[j] + 1) * ut * re))
if len(sys.argv) > 4:
    t0 = float(sys.argv[4])
    d = [list(map(float, l.split())) for l in open(sys.argv[3]) if not l.startswith('#')]
    d = [r for r in d if r[1] >= t0]
    print('Re_tau from log.dat, t >= %g: %.1f (%d samples)' % (t0, sum(r[6] for r in d) / len(d), len(d)))
