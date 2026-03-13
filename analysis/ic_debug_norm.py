"""Debug the normalization step by step."""
import numpy as np

N = 4
L = 64.0
vol = L**3
N3 = N**3
k_f = 2*np.pi / L

# Theoretical P at k_fund
A_s = 3.5227e6
gamma = 0.150
def power_unnorm(k):
    q = np.asarray(k, float) / gamma
    d = 1 + 3.89*q + (16.1*q)**2 + (5.46*q)**3 + (6.71*q)**4
    out = np.where(q < 1e-6, 1.0, np.log(1+2.34*q)/(2.34*q*d**0.25))
    return np.asarray(k, float)**0.963 * out**2

P_k_fund = A_s * power_unnorm(k_f)   # z=0 linear P at k_fund
print(f"N={N}, L={L}, N^3={N3}, vol={vol:.0f}, k_fund={k_f:.4f}")
print(f"P_theory(k_fund, z=0) = {P_k_fund:.4e} (Mpc/h)^3")

# Step 1: Gaussian noise and its FFT
rng = np.random.default_rng(42)
noise_real = rng.standard_normal((N, N, N))
buf = np.fft.fftn(noise_real)  # unnormalized FFT

mode_001 = buf[0, 0, 1]
print(f"\nbuf[0,0,1] = {mode_001:.4f}  |buf|^2 = {abs(mode_001)**2:.2f}  E[|buf|^2] = N^3 = {N3}")

# Step 2: delta_k = buf * sqrt(P/vol)
delta_k_001 = mode_001 * np.sqrt(P_k_fund / vol)
print(f"delta_k[0,0,1] = {abs(delta_k_001):.6e}  E[|delta_k|^2] = N^3*P/vol = {N3*P_k_fund/vol:.4e}")

# Step 3: After IC displacement and CIC, FFT(delta_CIC)[k] = D * delta_k
# Power estimator:
D = 0.026  # D(0.02)
P_hat_from_IC_formula = (vol/N3**2) * D**2 * abs(delta_k_001)**2
P_correct = D**2 * P_k_fund
print(f"\nP_hat from formula = (vol/N^6) * D^2 * |delta_k|^2 = {P_hat_from_IC_formula:.4e}")
print(f"P_correct = D^2 * P_k_fund = {P_correct:.4e}")
print(f"P_hat / P_correct = {P_hat_from_IC_formula/P_correct:.6f}  (should be 1/N^3 = {1/N3:.6f})")
print(f"P_correct / P_hat = {P_correct/P_hat_from_IC_formula:.1f}  (should be N^3 = {N3})")

# Now: what should delta_k be for P_hat = P_correct?
# P_hat = (vol/N^6) * D^2 * |delta_k_correct|^2 = D^2 * A_s * P
# -> |delta_k_correct|^2 = N^6/vol * A_s * P
# -> delta_k_correct = buf * sqrt(N^6/vol * A_s * P) / sqrt(E[|buf|^2])
#                    = buf * sqrt(N^6/vol * P) / sqrt(N^3)  [since E[|buf|^2]=N^3]
#                    = buf * sqrt(N^3/vol * P)
# So the fix: sqrt(P/vol) -> sqrt(N^3 * P/vol)
delta_k_correct = mode_001 * np.sqrt(N3 * P_k_fund / vol)
P_hat_correct = (vol/N3**2) * D**2 * abs(delta_k_correct)**2
print(f"\nWith fix: delta_k_correct = buf * sqrt(N^3*P/vol)")
print(f"P_hat_correct = {P_hat_correct:.4e}  vs  P_correct = {P_correct:.4e}")
print(f"Ratio = {P_hat_correct/P_correct:.4f}  (should be 1 on average)")

# Verify: E[|delta_k_correct|^2] = E[|buf|^2] * N^3*P/vol = N^3 * N^3*P/vol = N^6*P/vol
# And (vol/N^6) * N^6*P/vol = P.  ✓

print(f"\nFix: change sqrt(pk/vol) to sqrt(pk*n3/vol) in zeldovich.rs")
print(f"This multiplies delta_k by sqrt(N^3) = {N3**0.5:.1f}")
print(f"And psi_k, psi_real, displacement, velocity also multiply by sqrt(N^3)")
