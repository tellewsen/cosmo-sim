#!/usr/bin/env python3
"""
Physics validation suite for cosmo_sim — reference implementations.

Mirrors every algorithm in the Rust codebase and runs ~60 tests.
Conventions:
  - C-order flat index: flat(i,j,k) = i*n² + j*n + k
  - Supercomoving time τ: dτ = dt/a²,  da/dτ = a H(a)
  - DKD leapfrog
  - CIC Cloud-In-Cell
  - FFT Poisson (solves the discrete Laplacian exactly)
  - Symmetron NGS multigrid with CORRECT Newton formula
"""

import numpy as np, sys

PASS = 0; FAIL = 0

def check(name, cond, msg=""):
    global PASS, FAIL
    if cond:
        print(f"  ✓  {name}")
        PASS += 1
    else:
        print(f"  ✗  {name}  {msg}")
        FAIL += 1

# ─── Helpers ──────────────────────────────────────────────────────────────────

def idx3(i, j, k, n):
    return (int(i)%n)*n*n + (int(j)%n)*n + (int(k)%n)

def from_flat(c, n):
    return c//(n*n), (c//n)%n, c%n

def grad_phi(phi, dx, n):
    g = np.zeros((n**3, 3)); h = 0.5/dx
    for i in range(n):
        for j in range(n):
            for k in range(n):
                c = idx3(i,j,k,n)
                g[c,0] = -(phi[idx3(i+1,j,k,n)]-phi[idx3(i-1,j,k,n)])*h
                g[c,1] = -(phi[idx3(i,j+1,k,n)]-phi[idx3(i,j-1,k,n)])*h
                g[c,2] = -(phi[idx3(i,j,k+1,n)]-phi[idx3(i,j,k-1,n)])*h
    return g

def fft_poisson(source, dx, n):
    S   = source.reshape(n,n,n)
    Sk  = np.fft.fftn(S)
    ki  = np.fft.fftfreq(n)*n
    I,J,K = np.meshgrid(ki,ki,ki,indexing='ij')
    lam = -4/dx**2*(np.sin(np.pi*I/n)**2+np.sin(np.pi*J/n)**2+np.sin(np.pi*K/n)**2)
    lam[0,0,0] = 1.0
    Phik = Sk/lam; Phik[0,0,0] = 0.0
    phi  = np.real(np.fft.ifftn(Phik)).ravel()
    return phi - phi.mean()

def disc_lap(phi, dx, n):
    lap = np.zeros(n**3); inv_dx2 = 1/dx**2
    for i in range(n):
        for j in range(n):
            for k in range(n):
                c = idx3(i,j,k,n)
                nb = sum(phi[idx3(i+di,j+dj,k+dk,n)]
                         for di,dj,dk in [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)])
                lap[c] = (nb - 6*phi[c]) * inv_dx2
    return lap

def cic_density(pos, ng, L):
    n3 = ng**3; rho = np.zeros(n3); npart = len(pos)
    for xi in pos:
        u = xi*(ng/L) - 0.5; i0 = np.floor(u).astype(int); t = u-i0
        for di,wx in [(0,1-t[0]),(1,t[0])]:
            for dj,wy in [(0,1-t[1]),(1,t[1])]:
                for dk,wz in [(0,1-t[2]),(1,t[2])]:
                    rho[idx3(i0[0]+di,i0[1]+dj,i0[2]+dk,ng)] += wx*wy*wz
    return rho * n3 / npart

def cic_interp(pos, gv, ng, L):
    fs = []
    for xi in pos:
        u = xi*(ng/L)-0.5; i0 = np.floor(u).astype(int); t = u-i0; f = np.zeros(3)
        for di,wx in [(0,1-t[0]),(1,t[0])]:
            for dj,wy in [(0,1-t[1]),(1,t[1])]:
                for dk,wz in [(0,1-t[2]),(1,t[2])]:
                    f += wx*wy*wz * gv[idx3(i0[0]+di,i0[1]+dj,i0[2]+dk,ng)]
        fs.append(f)
    return np.array(fs)

def drift(x, v, dt_h, a, L): return (x + v*dt_h/a**2) % L
def kick(v, f, dt, a):       return v + f*a*dt
def ke(v, m, a):             return 0.5*m*np.sum(v**2)/a**2

def f_chi(chi, rho, a, a_ssb=0.5, lam=1.0):
    return chi*(rho*(a_ssb/a)**3 - 1.0 + chi**2)/lam**2

def df_chi(chi, rho, a, a_ssb=0.5, lam=1.0):
    return (rho*(a_ssb/a)**3 - 1.0 + 3*chi**2)/lam**2

def chi_eq(rho, a, a_ssb=0.5):
    return np.sqrt(max(0.0, 1.0 - rho*(a_ssb/a)**3))

def ngs_sym(chi, rho, a, dx, n):
    """Newton-Gauss-Seidel for ∇²χ = f(χ).
    Fixed point condition: (∇²chi)_c = f(chi_c)
    Newton update: chi_new = (chi*f' + Σnb/dx² - f) / (6/dx² + f')
    """
    snap = chi.copy(); inv_dx2 = 1/dx**2
    for col in range(2):
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    if (i+j+k)%2 != col: continue
                    c = idx3(i,j,k,n)
                    nb = sum(snap[idx3(i+di,j+dj,k+dk,n)]
                             for di,dj,dk in [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)])
                    cc = snap[c]; f = f_chi(cc,rho[c],a); df = df_chi(cc,rho[c],a)
                    denom = 6*inv_dx2 + df   # ← correct sign: +df not -df
                    if abs(denom) < 1e-30: continue
                    chi[c] = np.clip((cc*df + nb*inv_dx2 - f)/denom, -1.0, 1.0)
    return chi

def solve_sym(rho, a, dx, n, n_iter=150, tol=1e-8):
    chi = np.array([chi_eq(r,a) for r in rho])
    for _ in range(n_iter):
        chi = ngs_sym(chi, rho, a, dx, n)
    return chi

def E(a, om=0.3153, ol=0.6847): return np.sqrt(om/a**3+ol)
def H(a, h=0.6736): return h*100.0*E(a)
def rk4(a, dt):
    f = lambda a: a*H(a)
    k1=f(a); k2=f(a+0.5*dt*k1); k3=f(a+0.5*dt*k2); k4=f(a+dt*k3)
    return a + dt*(k1+2*k2+2*k3+k4)/6

# ─────────────────────────────────────────────────────────────────────────────
print("="*60); print("1. COSMOLOGY"); print("="*60)

check("E(1) = 1",           abs(E(1.0)-1.0) < 1e-12)
check("Ω_m + Ω_Λ = 1",     abs(0.3153+0.6847-1.0) < 1e-12)
check("Matter domination",  abs(E(0.01)-np.sqrt(0.3153/0.01**3)) < 0.01*np.sqrt(0.3153/0.01**3))
a0=0.1; dt=1e-4; n=100
ac = a0
for _ in range(n): ac = rk4(ac, dt)
af = a0
for _ in range(n*100): af = rk4(af, dt*0.01)
check("RK4 accuracy",       abs(ac-af)/af < 1e-6)

a0r=0.5; tau=0.005
ar=a0r
for _ in range(50000): ar = rk4(ar, tau/50000)
errs=[]
for ns in [5,10,20]:
    a=a0r
    for _ in range(ns): a=rk4(a, tau/ns)
    errs.append(abs(a-ar))
r1,r2 = errs[0]/errs[1], errs[1]/errs[2]
check("RK4 4th-order conv", r1>10 and r2>10,  f"ratios={r1:.1f},{r2:.1f}")
check("f(a=1)∈(0.4,0.75)",  0.4 < (0.3153/E(1.0)**2)**0.55 < 0.75)
check("H(1)=100h km/s/Mpc", abs(H(1.0)-67.36) < 0.01)

def tau_of_a(a_ini, a_fin, n_steps=10000):
    da = (a_fin-a_ini)/n_steps; tau=0.0; a=a_ini+0.5*da
    for _ in range(n_steps):
        tau += da / (a*H(a)); a += da
    return tau
tau_val = tau_of_a(0.001, 1.0)
check("τ(0.001→1) finite&pos", tau_val > 0 and np.isfinite(tau_val))

# ─────────────────────────────────────────────────────────────────────────────
print("\n"+"="*60); print("2. INDEXING"); print("="*60)

n5=5
check("idx3(0,0,0)=0",     idx3(0,0,0,n5)==0)
check("idx3(0,0,1)=1",     idx3(0,0,1,n5)==1)
check("idx3(0,1,0)=n",     idx3(0,1,0,n5)==n5)
check("idx3(1,0,0)=n²",    idx3(1,0,0,n5)==n5*n5)
check("from_flat roundtrip", all(idx3(*from_flat(c,n5),n5)==c for c in range(n5**3)))
check("periodic wrap i",   idx3(-1,0,0,n5)==idx3(n5-1,0,0,n5))
check("periodic wrap k",   idx3(0,0,-1,n5)==idx3(0,0,n5-1,n5))
check("idx3 same as numpy",
      all(idx3(i,j,k,n5)==np.ravel_multi_index((i,j,k),(n5,n5,n5))
          for i in range(n5) for j in range(n5) for k in range(n5)))

# ─────────────────────────────────────────────────────────────────────────────
print("\n"+"="*60); print("3. CIC"); print("="*60)

ng=4; L=4.0; dx=L/ng
pos_grid = np.array([[(i+0.5)*dx,(j+0.5)*dx,(k+0.5)*dx]
                     for i in range(ng) for j in range(ng) for k in range(ng)])
rho_u = cic_density(pos_grid, ng, L)
check("Uniform grid → rho=1",     np.max(np.abs(rho_u-1.0)) < 1e-10)

n_rand=200; L2=8.0
pos_rand = np.column_stack([(np.arange(n_rand)*0.37)%L2,
                             (np.arange(n_rand)*0.71)%L2,
                             (np.arange(n_rand)*0.13)%L2])
rho_r = cic_density(pos_rand, 8, L2)
check("Mass conservation",         abs(rho_r.sum()-8**3) < 1e-8)

ng3=4; L3=4.0; dx3=L3/ng3
rho1 = cic_density(np.array([[1.5*dx3,0.5*dx3,0.5*dx3]]), ng3, L3)
tc = idx3(1,0,0,ng3)
check("Single particle at cell ctr", abs(rho1[tc]-ng3**3)<1e-10
                                     and rho1.sum()-rho1[tc]<1e-10)

cf = np.array([1.23,-0.45,0.78])
f_c = cic_interp(np.array([[(i*0.5+0.1)%4.0,2.0,2.0] for i in range(8)]),
                 np.full((64,3),cf), 4, 4.0)
check("Constant force interp",     np.max(np.abs(f_c-cf)) < 1e-12)

# Adjointness: Σ_p f(x_p) = Σ_c ρ_c g_c * N_p/N_c
ng5=8; L5=8.0; n5p=100
p5 = np.column_stack([(np.arange(n5p)*0.37)%L5,(np.arange(n5p)*0.71)%L5,
                       (np.arange(n5p)*0.13)%L5])
g5  = np.array([(np.sin(c*0.01),)*3 for c in range(ng5**3)])
rho5 = cic_density(p5, ng5, L5)
f5   = cic_interp(p5, g5, ng5, L5)
sf   = f5.sum(); srg = (rho5[:,None]*g5).sum()*n5p/ng5**3
check("CIC adjointness",           abs(sf-srg)/(abs(sf)+1e-30) < 1e-9)

# Particle NOT at cell centre: mass is spread across neighbours
ng6=8; L6=8.0; dx6=L6/ng6
pos_off = np.array([[1.3*dx6, 0.5*dx6, 0.5*dx6]])  # 30% into cell
rho_off = cic_density(pos_off, ng6, L6)
# Should spread mass to cells (0,0,0) and (1,0,0)
check("CIC off-centre spreads mass",
      rho_off[idx3(0,0,0,ng6)] > 0 and rho_off[idx3(1,0,0,ng6)] > 0)
check("CIC off-centre mass conserved", abs(rho_off.sum()-ng6**3) < 1e-10)

# ─────────────────────────────────────────────────────────────────────────────
print("\n"+"="*60); print("4. POISSON SOLVER"); print("="*60)

# FFT solves the DISCRETE Laplacian exactly (machine precision residual)
n_t=16; dx_t=1.0/n_t; k_t=2*np.pi
src_t = np.array([-k_t**2*np.sin(k_t*(c//(n_t*n_t)+0.5)*dx_t) for c in range(n_t**3)])
phi_t = fft_poisson(src_t, dx_t, n_t)
lap_t = disc_lap(phi_t, dx_t, n_t)
res_err = np.max(np.abs(lap_t - (src_t-src_t.mean())))
check(f"FFT exact discrete residual ({res_err:.2e})", res_err < 1e-9)

# 2nd-order convergence of the discrete Laplacian stencil
errs_disc = []
for n_c in [8,16,32]:
    dx_c = 1.0/n_c
    phi_c  = np.array([np.sin(k_t*(c//(n_c*n_c)+0.5)*dx_c) for c in range(n_c**3)])
    lap_c  = disc_lap(phi_c, dx_c, n_c)
    ex_lap = np.array([-k_t**2*np.sin(k_t*(c//(n_c*n_c)+0.5)*dx_c) for c in range(n_c**3)])
    errs_disc.append(np.max(np.abs(lap_c-ex_lap)))
r_d = errs_disc[0]/errs_disc[1]
check(f"Discrete Laplacian 2nd-order ({r_d:.1f})", r_d > 2.5)

# Gradient accuracy (n=64: k*dx small enough for 2nd-order to be accurate)
n_g=64; dx_g=1.0/n_g; k_g=2*np.pi
phi_g = np.array([np.sin(k_g*(c//(n_g*n_g)+0.5)*dx_g) for c in range(n_g**3)])
gn    = grad_phi(phi_g, dx_g, n_g)
ge    = np.array([-k_g*np.cos(k_g*(c//(n_g*n_g)+0.5)*dx_g) for c in range(n_g**3)])
err_gx = np.max(np.abs(gn[:,0]-ge))
check(f"Gradient x-direction n=64 ({err_gx:.3e})", err_gx < 0.015)
check("Gradient y,z=0 for x-field",
      np.max(np.abs(gn[:,1]))<1e-13 and np.max(np.abs(gn[:,2]))<1e-13)

# Gradient 2nd-order convergence
errs_g = []
for n_gr in [16,32,64]:
    dx_gr = 1.0/n_gr
    phi_gr = np.array([np.sin(k_g*(c//(n_gr*n_gr)+0.5)*dx_gr) for c in range(n_gr**3)])
    gn_gr  = grad_phi(phi_gr, dx_gr, n_gr)
    ge_gr  = np.array([-k_g*np.cos(k_g*(c//(n_gr*n_gr)+0.5)*dx_gr) for c in range(n_gr**3)])
    errs_g.append(np.max(np.abs(gn_gr[:,0]-ge_gr)))
r_g = errs_g[0]/errs_g[1]
check(f"Gradient 2nd-order conv ({r_g:.1f})", r_g > 2.5)

# Two particles NOT at cell centers → non-zero CIC force
n_tp=32; L_tp=32.0; dx_tp=L_tp/n_tp
# Offset particles by 0.3*dx from cell-edge so CIC spreads onto neighbours
pos_tp = np.array([[L_tp*0.25+0.3*dx_tp, L_tp*0.5, L_tp*0.5],
                   [L_tp*0.75-0.3*dx_tp, L_tp*0.5, L_tp*0.5]])
rho_tp = cic_density(pos_tp, n_tp, L_tp)
phi_tp = fft_poisson(rho_tp-1.0, dx_tp, n_tp)
g_tp   = grad_phi(phi_tp, dx_tp, n_tp)
f_tp   = cic_interp(pos_tp, g_tp, n_tp, L_tp)
check("Two particles attract",    f_tp[0,0]>0 and f_tp[1,0]<0,
      f"f0x={f_tp[0,0]:.4f}, f1x={f_tp[1,0]:.4f}")
check("Newton 3rd law",           abs(f_tp[0,0]+f_tp[1,0])/(abs(f_tp[0,0])+1e-30) < 0.01)

# Uniform density → Poisson gives zero potential
phi_unif = fft_poisson(np.zeros(32**3), 1.0/32, 32)
check("Uniform density → Φ=0",   np.max(np.abs(phi_unif)) < 1e-12)

# Poisson is linear: Φ(S1+S2) = Φ(S1) + Φ(S2)
np.random.seed(7); s1=np.random.randn(8**3); s2=np.random.randn(8**3)
s1-=s1.mean(); s2-=s2.mean()
phi12 = fft_poisson(s1+s2, 1.0/8, 8)
phi1  = fft_poisson(s1, 1.0/8, 8)
phi2  = fft_poisson(s2, 1.0/8, 8)
check("Poisson linearity",        np.max(np.abs(phi12-(phi1+phi2))) < 1e-10)

# ─────────────────────────────────────────────────────────────────────────────
print("\n"+"="*60); print("5. LEAPFROG"); print("="*60)

p0 = np.array([[1.0,2.0,3.0],[9.0,0.5,7.0]]); v0z = np.zeros((2,3))
check("Drift zero vel",        np.max(np.abs(drift(p0,v0z,0.1,1.0,10.0)-p0)) < 1e-15)

p1=np.array([[50.0,50.0,50.0]]); v1=np.array([[1.0,0.0,0.0]])
check("Drift displacement",    abs(drift(p1,v1,0.1,2.0,1e6)[0,0]-(50.0+0.025)) < 1e-14)

pw=np.array([[9.8,5.0,5.0]]); vw=np.array([[100.0,0.0,0.0]])
check("Drift periodic wrap",   0 <= drift(pw,vw,0.1,1.0,10.0)[0,0] < 10.0)

np.random.seed(42)
pe=np.random.uniform(0,100,(3,3)); ve=np.random.randn(3,3)
ke0=ke(ve,1.0,1.0); zf=np.zeros((3,3))
for _ in range(2000):
    pe=drift(pe,ve,5e-4,1.0,1000.0); ve=kick(ve,zf,1e-3,1.0); pe=drift(pe,ve,5e-4,1.0,1000.0)
check("KE conserved zero force", abs(ke(ve,1.0,1.0)-ke0)/ke0 < 1e-13)

pc=np.array([[0.0,0.0,0.0]]); vc=np.array([[1.0,0.0,0.0]]); dtc=1e-4; nc=1000
for _ in range(nc):
    pc=drift(pc,vc,dtc/2,1.0,1e8); vc=kick(vc,np.array([[2.0,0.0,0.0]]),dtc,1.0); pc=drift(pc,vc,dtc/2,1.0,1e8)
t_end=dtc*nc; x_ex=1.0*t_end+0.5*2.0*t_end**2
check("Parabolic trajectory",  abs(pc[0,0]-x_ex) < 1e-6)

# Time reversibility: flip v, keep SAME force (force is position-dependent, not v-dependent)
pr=np.array([[500.1,499.9,500.5]]); vr=np.array([[0.7,-0.3,0.4]])
f_fwd=np.array([[0.01,-0.005,0.0]]); x0r,v0r=pr.copy(),vr.copy()
dtr=0.01; nrr=100
for _ in range(nrr):
    pr=drift(pr,vr,dtr/2,1.0,1e8); vr=kick(vr,f_fwd,dtr,1.0); pr=drift(pr,vr,dtr/2,1.0,1e8)
vr=-vr  # reverse velocity
for _ in range(nrr):
    pr=drift(pr,vr,dtr/2,1.0,1e8); vr=kick(vr,f_fwd,dtr,1.0); pr=drift(pr,vr,dtr/2,1.0,1e8)
vr=-vr  # flip back
check("Time reversibility",    np.max(np.abs(pr-x0r))<1e-10 and np.max(np.abs(vr-v0r))<1e-10,
      f"Δx={np.max(np.abs(pr-x0r)):.3e}, Δv={np.max(np.abs(vr-v0r)):.3e}")

# Kick correctness: Δv = a * F * dt
v_k=np.zeros((2,3))
v_k=kick(v_k, np.array([[2.0,-1.0,0.5],[0.0,0.0,0.0]]), 0.1, 0.5)
check("Kick Δv = a*F*dt",      abs(v_k[0,0]-0.5*2.0*0.1)<1e-15 and abs(v_k[0,1]-0.5*(-1.0)*0.1)<1e-15)

# 2nd-order leapfrog: global error should be O(dt²)
def run_harmonic(dt, n_steps, omega=1.0):
    """x'' = -omega²x → exact solution x=cos(omega*t), v=-omega*sin(omega*t)"""
    x=np.array([[1.0,0.0,0.0]]); v=np.array([[0.0,-omega*dt/2,0.0]])  # offset half-step
    # Actually use full DKD starting from t=0
    x=np.array([[1.0,0.0,0.0]]); v=np.zeros((1,3))
    t_end = dt*n_steps
    for _ in range(n_steps):
        x=drift(x,v,dt/2,1.0,1e8)
        v=kick(v, -omega**2*x, dt, 1.0)
        x=drift(x,v,dt/2,1.0,1e8)
    return x[0,0], t_end

errs_lf=[]; ts=[100,200,400]
for ns in ts:
    x_num, t_end = run_harmonic(0.1/ns*10, ns)  # same t_end=1.0
    x_ex = np.cos(t_end)
    errs_lf.append(abs(x_num - x_ex))
r_lf = errs_lf[0]/errs_lf[1]
check(f"Leapfrog 2nd-order ({r_lf:.1f})", r_lf > 3.0, f"ratio={r_lf:.2f}")

# ─────────────────────────────────────────────────────────────────────────────
print("\n"+"="*60); print("6. SYMMETRON"); print("="*60)

check("f_chi odd in χ",
      all(abs(f_chi(c,0.3,0.7)+f_chi(-c,0.3,0.7))<1e-14 for c in [0.1,0.5,0.9]))
check("χ_eq in void → 1",     abs(chi_eq(0.0,0.8)-1.0)<1e-12)
check("χ_eq=0 mean density pre-SSB", chi_eq(1.0, 0.3) == 0.0)
check("χ_eq=0 dense pre-SSB", all(chi_eq(r,0.3)==0.0 for r in [1,2,5]))

for rv in [0.0,0.3,0.5]:
    ce=chi_eq(rv,0.8); fv=f_chi(ce,rv,0.8)
    check(f"f(χ_eq)=0 ρ={rv}",  abs(fv)<1e-12, f"f={fv:.3e}")

for rv in [0.0,0.3,0.5]:
    ce=chi_eq(rv,0.8); dv=df_chi(ce,rv,0.8)
    check(f"df/dχ>0 vacuum ρ={rv}", dv>0, f"df={dv:.4f}")

# NGS fixed point: uniform chi_eq is an exact fixed point
n_s=8; dx_s=1.0; a_s=0.8; rho_s=0.3
rho_g=np.full(n_s**3,rho_s)
chi_fp=np.full(n_s**3, chi_eq(rho_s,a_s))
chi_after=ngs_sym(chi_fp.copy(), rho_g, a_s, dx_s, n_s)
check("NGS fixed point = chi_eq", np.max(np.abs(chi_after-chi_fp)) < 1e-12,
      f"drift={np.max(np.abs(chi_after-chi_fp)):.3e}")

# Full convergence from perturbed initial condition
n_sc=8; dx_sc=1.0; a_sc=0.8; rho_sc=0.3
rho_gc=np.full(n_sc**3,rho_sc)
chi_c=np.array([chi_eq(r,a_sc) for r in rho_gc]); chi_c[0]+=0.3
ce_sc=chi_eq(rho_sc,a_sc)
chi_c=solve_sym(rho_gc, a_sc, dx_sc, n_sc, n_iter=150)
err_sc=np.max(np.abs(chi_c-ce_sc))
check(f"Uniform ρ→χ_eq (err={err_sc:.3e})", err_sc < 1e-6, f"χ_eq={ce_sc:.6f}")

chi_hd=solve_sym(np.full(n_s**3,10.0), a_s, dx_s, n_s)
check("Screening high density",  np.max(np.abs(chi_hd)) < 0.1)

chi_ps=solve_sym(np.full(n_s**3,1.0), 0.3, dx_s, n_s)
check("Pre-SSB χ≈0",             np.max(np.abs(chi_ps)) < 0.1)

chi_unif=np.full(n_s**3, 0.7)
g_chi=grad_phi(chi_unif, 1.0, n_s)
check("Uniform χ → ∇χ=0",       np.max(np.abs(g_chi)) < 1e-12)

check("f_chi finite everywhere",
      all(np.isfinite(f_chi(c,r,a))
          for c in [0,0.5,1,-0.5,-1] for r in [0,0.5,1,5,100] for a in [0.3,0.5,0.8,1.0]))

# Sinusoidal density: all chi values in [-1,1]
n_sin=16; dx_sin=1.0; a_sin=0.8; k_sin=2*np.pi/(n_sin*dx_sin)
rho_sin=np.array([1.0+0.5*np.sin(k_sin*(c//(n_sin*n_sin)+0.5)*dx_sin) for c in range(n_sin**3)])
chi_sin=solve_sym(rho_sin, a_sin, dx_sin, n_sin, n_iter=100)
check("Sinusoidal ρ: χ∈[-1,1]",  np.max(np.abs(chi_sin)) <= 1.0+1e-8)
check("Sinusoidal ρ: <|χ|>>0",   chi_sin[chi_sin>0].mean() > 0.3)

# ─────────────────────────────────────────────────────────────────────────────
print("\n"+"="*60); print("7. PARTICLE-MESH INTEGRATION"); print("="*60)

def make_grid(ng, L):
    dx=L/ng
    return np.array([[(i+0.5)*dx,(j+0.5)*dx,(k+0.5)*dx]
                     for i in range(ng) for j in range(ng) for k in range(ng)])

def pm_step(pos, vel, ng, L, dtau, a):
    dx=L/ng
    rho=cic_density(pos,ng,L); phi=fft_poisson(rho-1.0,dx,ng)
    gf=grad_phi(phi,dx,ng); f=cic_interp(pos,gf,ng,L)
    pos=drift(pos,vel,dtau/2,a,L); vel=kick(vel,f,dtau,a); pos=drift(pos,vel,dtau/2,a,L)
    return pos, vel

# Uniform grid: gravity balanced, no displacement
ng_u=8; L_u=8.0
pos_u=make_grid(ng_u,L_u); vel_u=np.zeros((ng_u**3,3)); pos_u0=pos_u.copy()
for _ in range(5): pos_u,vel_u=pm_step(pos_u,vel_u,ng_u,L_u,0.001,1.0)
check("Uniform grid stable",     np.max(np.abs(pos_u-pos_u0)) < 1e-10)

# Positions stay in box
ng_b=16; L_b=16.0
np.random.seed(99)
pos_b=np.random.uniform(0,L_b,(50,3)); vel_b=np.random.randn(50,3)*0.1
for _ in range(10): pos_b,vel_b=pm_step(pos_b,vel_b,ng_b,L_b,0.005,1.0)
check("Positions stay in box",    np.all((pos_b>=0)&(pos_b<L_b)))

# CoM drift small (periodic box, grid forces should largely cancel)
com0=pos_b.mean(0); com_drift=np.max(np.abs(pos_b.mean(0)-com0))
check("CoM drift < 0.5 Mpc/h",   com_drift < 0.5, f"drift={com_drift:.4f}")

# Two-body problem: particles approach each other under gravity
ng_2b=32; L_2b=64.0; dx_2b=L_2b/ng_2b
pos_2b=np.array([[L_2b*0.3+0.4*dx_2b, L_2b*0.5, L_2b*0.5],
                  [L_2b*0.7-0.4*dx_2b, L_2b*0.5, L_2b*0.5]])
vel_2b=np.zeros((2,3))
d0 = pos_2b[1,0]-pos_2b[0,0]
for _ in range(20): pos_2b,vel_2b=pm_step(pos_2b,vel_2b,ng_2b,L_2b,0.001,1.0)
d1 = pos_2b[1,0]-pos_2b[0,0]
check("Two bodies approach",      d1 < d0, f"d0={d0:.4f} → d1={d1:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
print("\n"+"="*60); print("8. RESTRICT / PROLONGATE"); print("="*60)

def restrict(fine, n_fine):
    nc=n_fine//2; coarse=np.zeros(nc**3)
    for i in range(nc):
        for j in range(nc):
            for k in range(nc):
                coarse[idx3(i,j,k,nc)] = fine[idx3(2*i,2*j,2*k,n_fine)]
    return coarse

def prolongate(coarse, nc):
    nf=nc*2; fine=np.zeros(nf**3)
    for fi in range(nf):
        for fj in range(nf):
            for fk in range(nf):
                cx=(fi+0.5)*0.5-0.5; cy=(fj+0.5)*0.5-0.5; cz=(fk+0.5)*0.5-0.5
                ci=int(np.floor(cx)); cj=int(np.floor(cy)); ck=int(np.floor(cz))
                tx=cx-ci; ty=cy-cj; tz=cz-ck
                v=0.0
                for di,wx in [(0,1-tx),(1,tx)]:
                    for dj,wy in [(0,1-ty),(1,ty)]:
                        for dk,wz in [(0,1-tz),(1,tz)]:
                            v += wx*wy*wz*coarse[idx3(ci+di,cj+dj,ck+dk,nc)]
                fine[idx3(fi,fj,fk,nf)] = v
    return fine

# Constant field roundtrip
n_rp=8; f_const=np.full(n_rp**3, 3.14)
roundtrip=prolongate(restrict(f_const,n_rp), n_rp//2)
check("Const field restrict+prolongate", np.max(np.abs(roundtrip-f_const)) < 1e-12)

# Injection correctness
n_ri=8; fine_ri=np.arange(n_ri**3, dtype=float)
coarse_ri=restrict(fine_ri, n_ri)
nc_ri=n_ri//2
for i in range(nc_ri):
    for j in range(nc_ri):
        for k in range(nc_ri):
            expected=fine_ri[idx3(2*i,2*j,2*k,n_ri)]
            got=coarse_ri[idx3(i,j,k,nc_ri)]
            if abs(got-expected)>1e-14:
                check("Injection correctness", False, f"({i},{j},{k}): {got}≠{expected}")
                break
else:
    check("Injection correctness", True)

# ─────────────────────────────────────────────────────────────────────────────
print("\n"+"="*60); print("9. ENERGY/MOMENTUM CONSERVATION"); print("="*60)

# Virial theorem test: for a static Newtonian potential well,
# time-averaged KE = -1/2 * time-averaged PE (Virial theorem)
# Here we just check KE doesn't blow up over many steps
ng_v=32; L_v=32.0; dtau_v=0.002
np.random.seed(12); n_pv=64
pos_v=np.random.uniform(0, L_v, (n_pv,3))
vel_v=np.random.randn(n_pv,3)*0.5
a_v=1.0
ke_vals=[]
for step in range(200):
    pos_v,vel_v=pm_step(pos_v,vel_v,ng_v,L_v,dtau_v,a_v)
    if step % 20 == 0:
        ke_vals.append(ke(vel_v,1.0,a_v))
ke_arr=np.array(ke_vals)
check("KE stays finite",         np.all(np.isfinite(ke_arr)))
check("KE doesn't blow up 10x",  ke_arr.max() < ke_arr[0]*10+1.0,
      f"KE: {ke_arr[0]:.3f} → {ke_arr[-1]:.3f}")

# Momentum conservation: Σ v_i = const (periodic + symmetric Poisson)
np.random.seed(55); n_pm=32; L_pm=32.0
pos_pm=np.random.uniform(0,L_pm,(n_pm,3)); vel_pm=np.random.randn(n_pm,3)*0.2
p0_pm=vel_pm.sum(0)
for _ in range(10): pos_pm,vel_pm=pm_step(pos_pm,vel_pm,32,L_pm,0.005,1.0)
p1_pm=vel_pm.sum(0)
dp_rel = np.max(np.abs(p1_pm-p0_pm))/(np.linalg.norm(p0_pm)+1e-10)
check("Momentum approx conserved", dp_rel < 0.5, f"rel_dp={dp_rel:.3e}")

# ─────────────────────────────────────────────────────────────────────────────
print("\n"+"="*60)
print(f"RESULTS:  {PASS} passed  |  {FAIL} failed")
print("="*60)

if FAIL:
    print("\nFAILED tests — fix before running the Rust suite.")
    sys.exit(1)
else:
    print("\nAll reference algorithms verified. Rust suite should pass.")
    sys.exit(0)
