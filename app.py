import numpy as np
import matplotlib.pyplot as plt
import streamlit as st


def simulate_heston(S0, v0, r, kappa, theta, sigma_v, rho, T, steps, paths):
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


st.title("Heston Stochastic Volatility Model")

st.markdown("""
The Heston model allows volatility to be stochastic rather than constant.

Stock price process:

$$dS_t = rS_tdt + \sqrt{v_t}S_tdW_t^S$$

Variance process:

$$dv_t = \\kappa(\\theta - v_t)dt + \\sigma_v\\sqrt{v_t}dW_t^v$$
""")

st.sidebar.header("Model Parameters")

S0 = st.sidebar.number_input("Initial Stock Price S0", value=100.0)
v0 = st.sidebar.number_input("Initial Variance v0", value=0.04)
r = st.sidebar.number_input("Risk-free Rate r", value=0.05)
kappa = st.sidebar.number_input("Mean Reversion kappa", value=2.0)
theta = st.sidebar.number_input("Long-run Variance theta", value=0.04)
sigma_v = st.sidebar.number_input("Vol of Variance sigma_v", value=0.30)
rho = st.sidebar.slider("Correlation rho", -0.99, 0.99, -0.70)
T = st.sidebar.number_input("Maturity T", value=1.0)
steps = st.sidebar.slider("Time Steps", 50, 500, 252)
paths = st.sidebar.slider("Number of Paths", 100, 20000, 5000)

K = st.sidebar.number_input("Strike K", value=100.0)

if st.button("Run Heston Simulation"):
    S, v = simulate_heston(
        S0=S0,
        v0=v0,
        r=r,
        kappa=kappa,
        theta=theta,
        sigma_v=sigma_v,
        rho=rho,
        T=T,
        steps=steps,
        paths=paths,
    )

    payoff = np.maximum(S[-1] - K, 0)
    call_price = np.exp(-r * T) * np.mean(payoff)

    st.subheader("Model Output")

    st.metric("European Call Price", f"{call_price:.4f}")
    st.metric("Final Average Stock Price", f"{np.mean(S[-1]):.4f}")
    st.metric("Final Average Variance", f"{np.mean(v[-1]):.4f}")

    st.subheader("Simulated Stock Price Paths")
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(S[:, :20])
    ax1.set_xlabel("Time Steps")
    ax1.set_ylabel("Stock Price")
    ax1.set_title("Heston Stock Price Paths")
    st.pyplot(fig1)

    st.subheader("Simulated Variance Paths")
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.plot(v[:, :20])
    ax2.set_xlabel("Time Steps")
    ax2.set_ylabel("Variance")
    ax2.set_title("Heston Variance Paths")
    st.pyplot(fig2)