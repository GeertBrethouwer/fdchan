# FDCHAN design: a GPU channel-flow code with Fourier x, compact differences in y and explicit differences in z

This note explains the choices behind FDCHAN, compares it with HiCHAN (Fourier x and z, compact finite
differences in y, influence-matrix pressure step), and records what has been checked. Sections marked
*estimate* are reasoning and arithmetic, not measurements; nothing in this repository has been run on a GPU yet.

## 1. Why this discretisation can be faster than HiCHAN

HiCHAN spends most of a time step in transforms: every RK stage needs 9 two-dimensional (x, z) FFTs on the
3/2 × 3/2 dealiased grid, and each of them passes through two pencil transposes (x ↔ y ↔ z). On several GPUs
those transposes are all-to-all communication and dominate the time per step (PERF.md of HiCHAN).

FDCHAN keeps the Fourier series only in x:

* **x:** pseudo-spectral, 3/2-rule dealiasing in x only. All transforms are batched 1-D FFTs along contiguous
  x-lines (cuFFT/hipFFT). No transposes are needed for them as long as x-lines are not split between GPUs.
* **z:** explicit central differences (order 2–8). Split over GPUs (z-slabs), they need only nearest-neighbour
  halo exchanges of p/2 planes. A compact scheme in z would need a distributed tridiagonal solve per
  derivative (section 1.3).
* **y:** compact differences (`y_scheme = compact`, default; HiCHAN's schemes of order 4 and 6, plus an
  8th-order variant) or explicit ones (`y_scheme = explicit`). With z-slabs every GPU holds complete y-columns,
  so the tridiagonal solves of a compact scheme are local and cost no communication.
* **Pressure:** one Poisson problem per RK stage, the only global operation (section 3).

### 1.1 Memory traffic per RK stage on one GPU (*estimate*)

Units: bytes per grid point N = nx·ny·nz (double precision; a complex spectral array is 8 bytes per point,
a padded array in x is 12, in x and z 18). Rotational form, no scalars.

| part | HiCHAN | FDCHAN, compact y |
|---|---|---|
| inverse transforms | 6 × ≈150 (2 transposes, 2 pad copies, z-FFT, x-FFT) | 6 × 24 (x-FFT in place) + ≈110 (one kernel writes all 6 padded inputs) |
| forward transforms | 3 × ≈140 | 3 × 24 |
| vorticity, products | ≈160 + compact y-derivatives | ≈65 (2 compact y-derivatives) + 108 (pointwise u × ω) |
| RK update, viscous terms | ≈100 | ≈240 (3 compact D2 sweeps + update) |
| pressure step | ≈200 (banded compact solves, influence matrix) | ≈310 (right-hand side, 2 transposes, 2 z-FFTs, banded LU, D1 φ, correction) |
| **total** | **≈1800** | **≈1050** |

At the same grid one FDCHAN stage moves about 0.6× the data of a HiCHAN stage. But z needs more points with
differences than with Fourier modes (section 5, about 1.3–1.5×), and the advantage on one GPU then becomes
small (*estimate*, Re_τ = 180 box of HiCHAN's `examples/retau180`):

| | grid | traffic per stage | relative |
|---|---|---|---|
| HiCHAN | 128 × 129 × 128 modes | 1800 × 2.1 M | 1 |
| FDCHAN, compact 6th in y, z as HiCHAN's modes | 128 × 129 × 128 | 1050 × 2.1 M | 0.58 |
| FDCHAN, compact 6th in y, 1.5× points in z | 128 × 129 × 192 | 1050 × 3.2 M | 0.87 |
| FDCHAN, compact 8th in y (fewer y points), 1.5× in z | 128 × 97 × 192 | 1050 × 2.4 M | 0.65 |

The pressure step is the largest single item; its scratch traffic (every U row of the banded LU written and
read back once) can probably be halved by keeping the rows in shared memory. On one GPU FDCHAN is therefore
expected to be about as fast as HiCHAN, at best somewhat faster. The real gain is on several GPUs (section 1.2).
`examples/arrhenius_bench.sh` measures this on a GH200.

### 1.2 Communication per RK stage on several GPUs (*estimate*)

Decomposition: z-slabs (each GPU holds complete x-lines and y-columns for nz/P values of z).

| | HiCHAN (2-D pencils) | FDCHAN (z-slabs) |
|---|---|---|
| all-to-all volume | 9 transforms × 2 transposes ≈ 9 × (8 + 12) = 180 bytes/point | 2 transposes of one complex field (pressure source and solution) = 16 bytes/point |
| halo exchanges | none | ≈ 8 fields × 2·(p/2) planes; for nz/P = 48 and p = 6 that is ≈ 0.12 × 8 fields × 8–12 bytes ≈ 10 bytes/point, nearest neighbours only |

That is roughly 10× less all-to-all data, and the rest is neighbour communication that can overlap with the
interior work. Slabs limit the number of GPUs to about nz/(p/2+1), several hundred for nz = 1536.

**Status:** single GPU. The storage layout (complete x-lines, (kx, z, y) order, pressure data in (z, kx, y)
order for the transpose) was chosen so that the z-slab version only adds halo exchanges and the two
all-to-alls; it has not been written yet.

### 1.3 Compact differences in z?

On one GPU a compact z-derivative costs a stencil pass plus a periodic tridiagonal sweep (about 3 passes instead
of 1), about +10–15 % per stage for the ≈ 7 z-derivatives of a stage, in exchange for fewer z points (≈ 1.1–1.2×
the Fourier modes instead of 1.5×). On several GPUs every z-derivative becomes a distributed cyclic tridiagonal
solve (a local sweep, a small reduced system exchanged among the z-ranks, a back-substitution; P-TDMA of Diez
Sanhueza et al. 2025). The data volume is tiny, but these are ≈ 20 synchronising collectives per time step.
Doing it with z-FFTs instead would bring back HiCHAN's transposes, and then Fourier in z is better. Not
implemented; worth it only if measurements show that z dominates.

## 2. Spatial discretisation

* Collocated grid (u, v, w, p at the same points), as in HiCHAN.
* y, compact (default): A f′ = B f with tridiagonal A, coefficients computed on the stretched grid by
  polynomial exactness (as HiCHAN's `compact.f90`).
  * `fd_order = 4, 6` with `y_closure = hichan` (default): exactly HiCHAN's (3-4-3) and (3-5-6-5-3) schemes.
  * `fd_order = 8`: interior B on 7 points (8th order on a uniform grid), explicit one-sided wall rows, rows 1
    and 2 compact with 4 B points.
  * Each derivative is one thread per (kx, z) line: a sweep down (B f and forward elimination) and a sweep up.
* y, explicit: Fornberg weights, central p+1 points, one-sided p+1 (D1) or p+2 (D2) points at the walls.
* z: central stencils on the uniform periodic grid; their Fourier symbols kz′ (modified wavenumbers) are used
  in the pressure solver.
* Nonlinear terms: rotational form u × ω (default, 9 x-transforms per stage, pointwise energy neutral) or
  advective form (15 x-transforms, as HiCHAN's option).
* Viscous terms explicit with the RK3 weights, D2 in y.

## 3. Pressure solver

The discrete problem is that of HiCHAN's explicit scheme (Lenaers et al. 2014, App. B.1):

```
u = r_x − i kx φ,   w = r_z − D1z φ,   v = r_y − D1y φ        (interior points)
u = v = w = 0 at the walls,     i kx u + D1z w + D1y v = 0     at all points, walls included
```

The z-operators are circulant, so a z-FFT diagonalises them exactly (D1z → i kz′). For every pair (kx, kz)
this leaves one problem in y with k² = kx² + kz′².

**Chosen: banded solve for (φ, ψ = D1 φ).** With a compact D1 = A1⁻¹B1 the pressure operator D1 E D1 − k²P is
dense. Keeping ψ = D1 φ as a second unknown makes the system banded again:

```
A1 ψ − B1 φ = 0
B1 E ψ − k² A1 P φ = B1 E r_y − A1 c,     c = −(i kx r_x + D1z r_z) inside, 0 at the walls
```

(E zeroes the wall values, P is the identity on the interior rows). The right-hand side needs only banded
products, no compact solve. With the unknowns interleaved (φ_0, ψ_0, φ_1, …) the matrix has 4 sub- and
super-diagonals for the compact 4th/6th-order schemes, 8 for compact 8th order and 8–16 for explicit
differences. It depends on kz only through kz′², so one thread solves the columns +kz and −kz together (four
real right-hand sides), with a banded LU with partial pivoting computed on the fly: a window of the active rows
in local memory, every finished U row written once to scratch and read back in the upward sweep. The matrix
is built from two shared band matrices, G0 + k² G1, so nothing is stored per wavenumber. Pivoting is
needed: the wall rows have zero diagonal entries in this ordering.

Tested against dense LU with pivoting (ny = 33–385, orders 4–8, k = 0.1–1e4): φ agrees to 1e-12 or better
and ψ = D1 φ holds to round-off. The two singular columns (kx = 0 with kz = 0 or kz = Nyquist, where kz′ = 0)
are handled exactly: there continuity and v = 0 at the walls give v = 0, which the code imposes directly.
The divergence of the velocity is at round-off (1e-16 laminar, 1e-14 turbulent) at every point, walls included,
as in HiCHAN.

**Other options considered.**

* *HiCHAN's factorisation with influence matrix:* D1² − k² = (D1 + k)(D1 − k), two banded first-order solves,
  and a precomputed 6×6 influence matrix per k for the conditions the banded solves cannot impose. It gives
  the same discrete solution. The (φ, ψ) system reaches it with one banded solve and no superposition, and it
  works unchanged for compact and explicit differences; HiCHAN's route would need fewer scratch bytes
  (narrower bands) and is the natural alternative if the pressure step turns out to be expensive on the GPU.
* *Eigen-decomposition in y (CaNS-EIGEN, GEMM):* O(ny²) per column instead of O(ny), and the collocated
  operator is strongly non-normal. Rejected.
* *Approximate projection with the narrow Laplacian:* divergence only zero to truncation error. Rejected.
* *Staggered (MAC) grid:* narrower symmetric pressure operator and no odd–even decoupling, but high-order
  staggered closures on a stretched grid (summation by parts, O'Reilly et al. 2017) and interpolations in every
  nonlinear term. A larger change; not pursued.

## 4. Velocity–vorticity formulation and the influence matrix

The velocity–vorticity (Kim–Moin–Moser) formulation evolves φ = ∇²v and ω_y and recovers u and w from
`u = (i kx D1 v − i kz ω_y)/k²`. It suits HiCHAN because every horizontal operator is diagonal in its spectral
(kx, kz) space. With differences in z:

* the state must be kept in (kx, kz, y) space for every substep (Helmholtz problems for φ and ω_y, the Poisson
  problem for v and the recovery of u, w are all per-(kx, kz) solves), so the velocity and the nonlinear terms
  need z-transforms and all-to-all transposes every stage: HiCHAN's communication pattern returns;
* per stage it needs three or four wall-normal solves (plus the 2×2 influence-matrix correction for
  dv/dy = 0), against one in the projection method;
* in physical z instead, the horizontal Laplacian (−kx² + D1z²) would have to be inverted to recover u and w,
  again a global operation in z.

Its one advantage, implicit viscous terms at no extra cost, can be had in the projection method too by
treating only the wall-normal viscous term implicitly. **Conclusion:** the primitive-variable projection
method is the cheaper choice for Fourier-x / finite-difference-z: one global solve per stage, 2 transposes of
one field.

**Implicit wall-normal viscous term (planned, not implemented).** With z-slabs the Helmholtz problems
(1 − c D2y) u = r are local. Two variants: (1) fractional step with a pressure increment (Kim & Moin): one extra
local solve per component, no extra communication, continuity still exact, O(Δt²) splitting error in the
near-wall tangential velocity; (2) exact coupling as in HiCHAN's B.2 or with a Kleiser–Schumann influence
matrix: all three components in (kx, kz, y) space, three times the pressure transposes. Variant 1 fits this code.

## 5. Accuracy

**Tollmien–Schlichting wave**, Re = 5000 (centreline), α = 1.12, the test of Lenaers et al. Sec. 5.1 and
HiCHAN's `examples/ts_wave` (tanh grid α_s = 4, Δt = 0.002). Absolute error of the decay rate ω_i against the
Orr–Sommerfeld value −2.7829009e-3 (the relative error is 360× larger), from the energy of v over t = 250–350.

| y scheme | form | ny = 65 | ny = 97 | ny = 129 |
|---|---|---|---|---|
| HiCHAN, compact 4th (README) | advective | 2.6e-5 | | |
| HiCHAN, compact 6th (README) | advective | 3e-7 | | |
| FDCHAN, compact 4th (HiCHAN stencils) | advective | 2.65e-5 | 5.2e-6 | |
| FDCHAN, compact 6th (HiCHAN stencils) | advective | 3.4e-7 | 8.6e-8 | 2.8e-8 |
| FDCHAN, compact 6th (HiCHAN stencils) | rotational | 1.4e-6 | 1.7e-7 | |
| FDCHAN, compact 8th | advective | 1.5e-7 | 2.1e-9 | |
| FDCHAN, compact 8th | rotational | 2.2e-8 | | |
| FDCHAN, explicit 6th | advective | 3.5e-6 | 3.3e-7 | 6.0e-8 |
| FDCHAN, explicit 8th | advective | 1.2e-6 | | |

* With HiCHAN's stencils and the advective form, FDCHAN reproduces HiCHAN's published errors (2.65e-5 vs
  2.6e-5, 3.4e-7 vs 3e-7). Earlier versions of this note compared relative FDCHAN errors with these absolute
  numbers and concluded wrongly that explicit differences were 100–1000× less accurate; the factor is about 10.
* Explicit 6th order needs about 1.5× the points of compact 6th order for the same accuracy (97 vs 65).
* The compact 8th-order scheme is the most accurate per point: at ny = 97 it is 40× more accurate than compact
  6th at 129. In the advective form at ny = 65 it beats HiCHAN's scheme by 2×; in the rotational form by 60×.
* The nonlinear form changes the error by a truncation-size amount (the discrete product rule does not hold), 
  differently for each scheme.

**Oblique wave, z-differences alone.** By Squire's theorem the wave (α, β) = 1.12·(cos 60°, sin 60°) at
Re = 10000 has the same v(y) as the 2-D TS wave and ω_i(3D) = ω_i(2D)·cos 60°. Starting from the converged
2-D eigenfunction on the same y-grid (explicit 6th order, ny = 129), the difference to ω_i(2D)/2 on that grid is
the error of the z-discretisation alone (6th order, one wavelength in the box):

| points per spanwise wavelength | 8 | 12 | 16 | 24 |
|---|---|---|---|---|
| relative error of ω_i | 1.8e-2 | 1.6e-3 | 2.0e-4 | ≈1e-5 (still converging at t = 60) |

A Fourier method is exact for this mode. For the smallest energetic spanwise scales of wall turbulence this
suggests about 1.3–1.5× more points in z than Fourier modes with 6th-order differences.

## 6. Validation (CPU build)

| test | result |
|---|---|
| operators (`make test`) | explicit D1, D2 converge at orders 2–8; compact schemes at the order of their wall rows (3rd for HiCHAN's closures, 4th for the 8th-order variant) in the max norm |
| pressure solver | banded LU with pivoting vs dense LU: 1e-12 or better, ψ = D1 φ to round-off |
| laminar Poiseuille, CFR | −dp/dx = 3/Re to all printed digits, divergence 3e-16 |
| random disturbances, Re = 200 | E′ decays, −dp/dx → 3/Re |
| TS wave, oblique wave | section 5 |
| turbulent channel, Re_b = 2800, 32 × 65 × 48, box 2π × π (`runs/turb`) | transition from noise (rms 0.15) and sustained turbulence to t = 200 with all three y schemes below; divergence ≤ 3e-13 throughout. Very coarse grid (Δx⁺ ≈ 35, Δz⁺ ≈ 12) and a short average: a stability check, not a converged DNS |

Averages over t = 120–200 of the turbulent runs (rotational form, explicit 6th order in z):

| y scheme | Re_τ | U_c⁺ | u_rms⁺ peak | v_rms⁺ peak | w_rms⁺ peak | −u′v′⁺ peak |
|---|---|---|---|---|---|---|
| explicit 6th (`data/`) | 179 | 18.3 | 2.64 (y⁺ = 16) | 0.80 | 1.13 | 0.72 |
| compact 6th, HiCHAN stencils (`cmp6/`) | 178 | 18.6 | 2.58 (y⁺ = 16) | 0.79 | 1.13 | 0.74 |
| compact 8th (`cmp8/`) | 177 | 18.8 | 2.50 (y⁺ = 16) | 0.83 | 1.12 | 0.75 |

The differences are within what a short average on a coarse grid gives; they say that all three schemes are
stable through transition, not which is more accurate.

CPU cost (Intel i5): explicit 6th alone on 4 threads 0.14 s per step; the two compact runs ran at the same
time on 2 threads each, 0.31 s (6th) and 0.32 s (8th) per step, of which the pressure step took 20 % and 27 %.
The CPU FFT is a simple one written for testing; on the GPU, cuFFT replaces it.

Not yet tested: the CUDA and HIP builds (never compiled), multi-GPU, comparison with HiCHAN on the same case at
production resolution.
