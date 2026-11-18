#!/usr/bin/env python3
# decay rate of the TS wave from log.dat: python3 fit.py data/log.dat [t_start] [omega_i reference]
import sys, math
t=[];e=[]
for l in open(sys.argv[1]):
    if l.startswith('#'): continue
    s=l.split(); t.append(float(s[1])); e.append(float(s[10]))
t0=float(sys.argv[2]) if len(sys.argv)>2 else 200
pts=[(a,math.log(b)) for a,b in zip(t,e) if a>=t0]
n=len(pts); sx=sum(p[0] for p in pts); sy=sum(p[1] for p in pts); sxx=sum(p[0]**2 for p in pts); sxy=sum(p[0]*p[1] for p in pts)
slope=(n*sxy-sx*sy)/(n*sxx-sx*sx)
# also from the last two thirds separately (convergence check)
wi=-2.7829009e-3
print("omega_i = %.8e   abs. error = %.2e   rel. error = %.2e"%(slope/2,abs(slope/2-wi),abs(slope/2-wi)/abs(wi)))
