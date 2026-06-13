import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm


# -----------------------------
# Black-Scholes Call Price
# -----------------------------
def bs_call_price(S0, K, T, r, sigma):
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


# -----------------------------
# Heston Monte Carlo Call Price
# -----------------------------
def heston_mc_price(
    S0, K, T, r,
    v0, kappa, theta, sigma_v, rho,
    steps=252,
    paths=20000,
    seed=42
):
    np.random.seed(seed)
    dt = T / steps

    S = np.full(paths, S0)
    v = np.full(paths, v0)

    for _ in range(steps):
        z1 = np.random.normal(size=paths)
        z2 = np.random.normal(size=paths)

        w_s = z1
        w_v = rho * z1 + np.sqrt(1 - rho**2) * z2

        v_pos = np.maximum(v, 0)

        v = (
            v
            + kappa * (theta - v_pos) * dt
            + sigma_v * np.sqrt(v_pos * dt) * w_v
        )

        v_pos_new = np.maximum(v, 0)

        S = S * np.exp(
            (r - 0.5 * v_pos_new) * dt
            + np.sqrt(v_pos_new * dt) * w_s
        )

    payoff = np.maximum(S - K, 0)
    return np.exp(-r * T) * np.mean(payoff)


# -----------------------------
# Market Option Data
# Replace this with real market data later
# -----------------------------
S0 = 100
r = 0.05

market_data = [
    # K, T, market implied vol
    (80,  0.5, 0.28),
    (90,  0.5, 0.24),
    (100, 0.5, 0.21),
    (110, 0.5, 0.22),
    (120, 0.5, 0.25),

    (80,  1.0, 0.30),
    (90,  1.0, 0.26),
    (100, 1.0, 0.23),
    (110, 1.0, 0.24),
    (120, 1.0, 0.27),
]

# Convert market IV to market option prices
market_prices = []
for K, T, iv in market_data:
    price = bs_call_price(S0, K, T, r, iv)
    market_prices.append((K, T, price))


# -----------------------------
# Calibration Objective
# -----------------------------
def objective(params):
    v0, kappa, theta, sigma_v, rho = params

    # Penalize invalid parameters
    if v0 <= 0 or kappa <= 0 or theta <= 0 or sigma_v <= 0:
        return 1e10

    if rho < -0.999 or rho > 0.999:
        return 1e10

    error = 0.0

    for K, T, market_price in market_prices:
        model_price = heston_mc_price(
            S0=S0,
            K=K,
            T=T,
            r=r,
            v0=v0,
            kappa=kappa,
            theta=theta,
            sigma_v=sigma_v,
            rho=rho,
            steps=100,
            paths=5000,
            seed=42
        )

        error += (model_price - market_price) ** 2

    return error


# -----------------------------
# Initial Guess
# -----------------------------
initial_guess = [
    0.04,   # v0
    2.0,    # kappa
    0.04,   # theta
    0.40,   # sigma_v
    -0.50   # rho
]

bounds = [
    (0.001, 0.50),   # v0
    (0.01, 10.0),    # kappa
    (0.001, 0.50),   # theta
    (0.01, 2.00),    # sigma_v
    (-0.99, 0.99),   # rho
]

result = minimize(
    objective,
    initial_guess,
    method="Nelder-Mead",
    options={"maxiter": 100}
)

print("Calibration completed")
print("Success:", result.success)
print("Objective value:", result.fun)

v0, kappa, theta, sigma_v, rho = result.x

print("\nCalibrated Heston Parameters")
print(f"v0       = {v0:.6f}")
print(f"kappa    = {kappa:.6f}")
print(f"theta    = {theta:.6f}")
print(f"sigma_v  = {sigma_v:.6f}")
print(f"rho      = {rho:.6f}")