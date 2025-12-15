# Quantitative Portfolio Optimizer & Factor Engine

**A Python-based wealth simulation engine that builds mathematically optimal investment portfolios using Monte Carlo simulations and Fama-French Factor Analysis.**

> **Project Evolution:** This engine is a combanation of my Quantitative Finance projects, combining logic from teh following:
> 1.  **(Asset Correlation Matrix):** Provided the logic to see which assets move together (and which don't) to prevent doubling up on risk.
> 2.  **(Factor Exposure Engine):** Provided the Fama-French regression model to check if returns are coming from skill or just hidden market exposure.
> 3.  **(Sharpe Ratio Comparison):** Established the core metric for finding the best possible return for every unit of risk taken.

## Project Overview
This project was built to explore a practical question:
* **How can data and mathematics be used to make more informed financial decisions under uncertainty?**

Instead of relying on intuition or simple rules, this project uses historical market data to test thousands of possible portfolio combinations. It compares their risk and return to identify allocations that perform efficiently over time.

The project combines data analysis, problem-solving, and real-world decision making, making it relevant across finance, economics, engineering, and business.

## Core Components
#### Portfolio Simulation & Optimization
* Using Monte Carlo simulation, the model generates thousands of random portfolio weight combinations. Each portfolio is evaluated based on expected return, volatility, and risk-adjusted performance.
* An optimization routine then solves for asset weights that maximize the Sharpe Ratio, a widely used measure of return per unit of risk. A high Sharpe ratio means the portfolio is generating higher returns per unit of volatility.

#### Risk Attribution & Factor Analysis
To move beyond surface-level performance, the project incorporates Fama-French 3-Factor Analysis to examine deeper sources of risk:
* Market exposure – sensitivity to overall market movements
* Size exposure (SMB) – bias toward small or large companies
* Value exposure (HML) – tilt toward value or growth characteristics
This helps distinguish true diversification from hidden risk concentration.
#### Backtesting & Benchmarking
The optimized portfolio is backtested against the S&P 500 using aligned historical data. The backtesting engine:
* Adjusts for dividends
* Synchronizes asset start dates
* Handles missing or inconsistent data gracefully
This ensures fair and realistic performance comparisons.
#### Data Visualization
The project generates a visual analytics dashboard that includes:
* Efficient Frontier plots to visualize the risk–return landscape
* Portfolio allocation charts to show asset weight distribution
* Correlation heatmaps to identify relationships between assets
* Performance comparison charts versus the market benchmark


## Tech Stack
* **Python 3.10+**
* **Pandas & NumPy:** For vectorization and handling large financial datasets.
* **SciPy:** Used `SLSQP` minimization to solve the optimization problem.
* **Matplotlib & Seaborn:** For plotting the financial charts and heatmaps.
* **Statsmodels:** For running the OLS regressions in the Factor Analysis.
* **YFinance:** API for fetching real-time market data.



## Visualization

#### 1. The Efficient Frontier
*Visualizes the risk/reward tradeoff of the 5,000 simulated portfolios. The red star marks the "Mathematically Optimal" choice.*

#### 2. Optimized Asset Allocation
*A "Donut Chart" breakdown of the final portfolio, filtering out insignificant positions to show exactly where the capital is allocated.*

#### 3. Wealth Simulation (Backtest)
*A "Spaghetti Chart" that plots the optimized portfolio (Blue) against the S&P 500 (Black) and 50 random scenarios (Grey) to show relative performance.*

#### 4. Asset Correlation Matrix
*A heatmap I built to quickly identify which assets move together, helping to spot "fake diversification."*

---

### Quick Start

1.  **Clone the Repo**
    ```bash
    git clone [https://github.com/ClaysonV/Portfolio_Optimizer.git](https://github.com/ClaysonV/Portfolio_Optimizer.git)
    cd Portfolio_Optimizer
    ```

2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Engine**
    ```bash
    python optimizer.py
    ```

4.  **Follow the Prompts**
    * Enter a list of tickers (e.g., `NVDA AAPL JPM GLD BTC-USD`)
    * Enter your starting capital.
    * *The script handles the rest.*

### Lessons Learned
Building this helped me quantify diversification rather than just guessing at it. It’s not just about owning "different stuff"—it's about owning assets with **low covariance**. 

One of the trickiest parts was handling **data alignment**. If I input 50 stocks and one of them IPO'd last month, it would break the historical analysis for everything else. I wrote logic to sync the start dates of the portfolio and the benchmark so the comparison is always accurate, regardless of the assets chosen.


*Disclaimer: This is a research project, not financial advice.*
