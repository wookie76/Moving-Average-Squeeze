# HMA Squeeze Trading System: Two-Phase Stock Analysis

**A comprehensive Python-based trading system implementing the Hull Moving Average (HMA) squeeze strategy, designed for both rapid screening and in-depth analysis of potential breakout opportunities. This project provides a two-phase workflow: a fast screener to identify promising candidates and a detailed analyzer for visual exploration and backtesting.**

This project is inspired by Kridtapon P.'s Medium article, ["The Moving Average Squeeze Strategy: Spotting Hidden Breakout Signals"](https://medium.com/@kridtapon/the-moving-average-squeeze-strategy-spotting-hidden-breakout-signals-4330c9632bda) (Feb 24, 2025), and builds upon the core concepts of the Hull Moving Average (HMA) developed by Alan Hull.  It offers a complete solution for traders and developers, from initial stock screening to detailed strategy evaluation.

---

## Two-Phase Workflow

This project offers a two-phase workflow for analyzing stocks:

1.  **Phase 1: HMA Squeeze Screener (`GEM2-HULL-HOURLY-SCREENER.py`)**

    *   **Purpose:** Rapidly scan a large list of stocks to identify potential candidates exhibiting HMA squeeze behavior.
    *   **Input:** A text file (`all_tickers.txt` by default) containing a list of ticker symbols, one per line.
    *   **Process:**  Calculates HMA 21 and HMA 89, identifies squeeze signals, and performs a basic backtest.
    *   **Output:** An Excel file (`hma_squeeze_results.xlsx` by default) containing:
        *   Ticker
        *   Final Value (Strategy)
        *   CAGR (Strategy)
        *   Final Value (Buy & Hold)
        *   CAGR (Buy & Hold)
        *   Last Signal (Entry, Exit, or None)
    *   **Use Case:** Quickly filter a large universe of stocks down to a manageable list of potential trading opportunities based on the HMA squeeze.

2.  **Phase 2: HMA Squeeze Analyzer (`GEM2-HULL-HOURLY-SQUEEZE.py`)**

    *   **Purpose:** Perform in-depth analysis, visualization, and backtesting of individual stocks identified by the screener (or any stock of interest).
    *   **Input:** One or more ticker symbols provided as command-line arguments.
    *   **Process:** Calculates HMA 21, HMA 89, and HMA 200, identifies squeeze, entry, and exit signals, and performs a more detailed backtest.
    *   **Output:**
        *   Interactive Plotly charts (saved as HTML files in the `hma_plots/` directory) showing:
            *   Closing Price
            *   HMA 21, HMA 89, HMA 200
            *   HMA Cloud (between HMA 21 and HMA 89)
            *   Squeeze Signals
            *   Entry/Exit Signals
        *   Backtesting results (including CAGR) displayed in the console.
    *   **Use Case:** Thoroughly evaluate the HMA squeeze strategy on specific stocks, visually inspect price action and signals, and assess potential profitability.

---



## Acknowledgements

*   **Kridtapon P.:** This project is directly inspired by Kridtapon P.'s Medium article on the Moving Average Squeeze Strategy.
*   **Alan Hull:**  The foundation of this project is the Hull Moving Average, developed by Alan Hull.
*   **Relevant Financial Resources:** Concepts have been refined using information from resources discussing HMA strategies.

---

## Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/yourusername/hma-squeeze-system.git # REPLACEME
    cd hma-squeeze-system
    ```

2.  **Set up a Miniconda environment (recommended, Windows 11 tested):**

    ```bash
    conda create -n hma-system-env python=3.11
    conda activate hma-system-env
    pip install -r requirements.txt
    ```

3.  **Install dependencies (see `requirements.txt` below).**

---

## Usage

**Phase 1 (Screener):**

1.  Create a text file (e.g., `all_tickers.txt`) with ticker symbols, one per line.
2.  Run the screener:

    ```bash
    python GEM2-HULL-HOURLY-SCREENER.py
    ```

    This uses the default settings. You can customize the settings within the `Config` class.
3.  Review the results in `hma_squeeze_results.xlsx`.

**Phase 2 (Analyzer):**

1.  Identify promising tickers from the screener output (or choose any tickers).
2.  Run the analyzer:
    ```bash
    python GEM2-HULL-HOURLY-SQUEEZE.py AAPL MSFT TSLA
    ```
    Replace `AAPL MSFT TSLA` with your desired tickers.  You can customize period, interval, and squeeze window using command-line arguments:
    ```bash
    python GEM2-HULL-HOURLY-SQUEEZE.py AAPL -p 1y -i 1h -sw 15
    ```

---

## Notes

*   **Environment:** Developed and tested on Windows 11 with Miniconda.
*   **Educational Purpose:** This code is intended for educational and research purposes. It is not financial advice.  Always conduct thorough due diligence and risk management before implementing any trading strategy.

---
