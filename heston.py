import numpy as np
import matplotlib.pyplot as plt


def simulate_heston(
    S0=100,
    v0=0.04,
    r=0.05,
    kappa=2.0,
    theta=0.04,
    sigma_v=0.3,
    rho=-0.7,
    T=1.0,
    steps=252,
    paths=10000,
    seed=42,
):
    np.random.seed(seed)
    dt = T / steps

    S = np.zeros((steps + 1, paths))
    v = np.zeros((steps + 1, paths))

    S[0] = S0
    v[0] = v0

    for t in range(1, steps + 1):
        z1 = np.random.normal(size=paths)
        z2 = np.random.normal(size=paths)

        w_s = z1
        w_v = rho * z1 + np.sqrt(1 - rho**2) * z2

        v_prev = np.maximum(v[t - 1], 0)

        v[t] = (
            v[t - 1]
            + kappa * (theta - v_prev) * dt
            + sigma_v * np.sqrt(v_prev * dt) * w_v
        )

        v_pos = np.maximum(v[t], 0)

        S[t] = S[t - 1] * np.exp(
            (r - 0.5 * v_pos) * dt
            + np.sqrt(v_pos * dt) * w_s
        )

    return S, v


if __name__ == "__main__":
    S, v = simulate_heston()

    K = 100
    r = 0.05
    T = 1.0

    payoff = np.maximum(S[-1] - K, 0)
    call_price = np.exp(-r * T) * np.mean(payoff)

    print("Heston simulation completed")
    print(f"European Call Price: {call_price:.4f}")
    print(f"Final average stock price: {np.mean(S[-1]):.4f}")
    print(f"Final average variance: {np.mean(v[-1]):.4f}")

    plt.figure(figsize=(10, 5))
    plt.plot(S[:, :20])
    plt.title("Heston Stock Price Paths")
    plt.xlabel("Time Steps")
    plt.ylabel("Stock Price")
    plt.savefig("heston_stock_paths.png")
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(v[:, :20])
    plt.title("Heston Variance Paths")
    plt.xlabel("Time Steps")
    plt.ylabel("Variance")
    plt.savefig("heston_variance_paths.png")
    plt.show()