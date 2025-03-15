import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pandera as pa
import plotly.graph_objects as go
import yfinance as yf
from pandera.typing import Series
from plotly.subplots import make_subplots
from pydantic import BaseModel, Field, field_validator
from rich.console import Console
from rich.table import Table
from rich.text import Text
from tqdm import tqdm

# Configure logging with detailed output
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Color scheme for the plot
COLOR_SCHEME = {
    "raw_price": "#3498db",
    "pole": {  # Colors for different HMA lines
        89: {"line": "#e67e22", "marker": "#d35400"},
        21: {"line": "#2ecc71", "marker": "#27ae60"},
        200: {"line": "#e74c3c", "marker": "#c0392b"},
    },
    "cloud": "rgba(128,128,128,0.2)",
}

HMA_21_LABEL = "HMA (21)"
HMA_89_LABEL = "HMA (89)"
HMA_200_LABEL = "HMA (200)"


class HullMovingAverage:
    """Calculates the Hull Moving Average with Jedi-level efficiency."""

    def __init__(self, prices: np.ndarray, n: int):
        self.prices = prices
        self.n = n

    def _wma(self, arr: np.ndarray, periods: int) -> np.ndarray:
        """Vectorized WMA using convolution for speed."""
        if len(arr) < periods:
            return np.full_like(arr, np.nan)
        weights = np.arange(1, periods + 1, dtype=np.float64)
        wma = np.convolve(arr, weights / weights.sum(), mode="valid")
        return np.pad(
            wma, (len(arr) - len(wma), 0), mode="constant", constant_values=np.nan
        )

    def calculate(self) -> np.ndarray:
        """Computes HMA with minimal memory and maximal speed."""
        if len(self.prices) < self.n:
            return np.full_like(self.prices, np.nan)

        n_half, n_sqrt = self.n // 2, int(self.n**0.5)
        wma_half = self._wma(self.prices, n_half)
        wma_full = self._wma(self.prices, self.n)
        raw_hma = 2 * wma_half[-len(wma_full) :] - wma_full
        result = np.full_like(self.prices, np.nan)
        hma = self._wma(raw_hma, n_sqrt)
        result[-len(hma) :] = hma
        return result


class YFinanceData(pa.DataFrameModel):
    """Schema for yfinance data validation."""
    Open: Series[float] = pa.Field(ge=0)
    High: Series[float] = pa.Field(ge=0)
    Low: Series[float] = pa.Field(ge=0)
    Close: Series[float] = pa.Field(ge=0)
    Volume: Series[int] = pa.Field(ge=0)

    class Config:
        strict = True
        coerce = True


class HMAOutputData(pa.DataFrameModel):
    """Schema for HMA calculation output."""
    hma_21: Series[float] = pa.Field(nullable=True)
    hma_89: Series[float] = pa.Field(nullable=True)
    hma_200: Series[float] = pa.Field(nullable=True)
    squeeze: Series[bool] = pa.Field(nullable=True)
    entry: Series[bool] = pa.Field(nullable=True)
    exit: Series[bool] = pa.Field(nullable=True)

    class Config:
        strict = True
        coerce = True


def get_closing_prices(
    ticker: str, period: str, interval: str
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Fetches and validates historical closing prices, handling multi-level columns."""
    try:
        data = yf.download(ticker, period=period, interval=interval, auto_adjust=True)
        if data.empty:
            raise ValueError("No data returned from yfinance")

        # Flatten multi-level columns if present
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [col[0] for col in data.columns]

        # Log data shape for debugging
        logging.debug(f"Data shape for {ticker}: {data.shape}, columns: {data.columns}")

        # Validate with Pandera
        validated_data = YFinanceData.validate(data)
        return validated_data["Close"].to_numpy(copy=False), validated_data.index.to_numpy(
            copy=False
        )
    except Exception as e:
        logging.error(f"Error fetching data for {ticker}: {e}")
        return None, None


@pa.check_output(HMAOutputData, lazy=True)
def calculate_all_hmas(closing_prices: np.ndarray, squeeze_window: int) -> pd.DataFrame:
    """Calculates HMAs and signals with validation."""
    hma_21 = HullMovingAverage(closing_prices, 21).calculate()
    hma_89 = HullMovingAverage(closing_prices, 89).calculate()
    hma_200 = HullMovingAverage(closing_prices, 200).calculate()

    squeeze, entry, exit_signal = calculate_hma_squeeze_signals(
        hma_21, hma_89, squeeze_window
    )
    logging.debug(f"HMA_21 sample for {closing_prices.size} points: {hma_21[-5:]}")
    logging.debug(f"Squeeze sample: {squeeze[-5:]}")

    return pd.DataFrame(
        {
            "hma_21": hma_21,
            "hma_89": hma_89,
            "hma_200": hma_200,
            "squeeze": squeeze,
            "entry": entry,
            "exit": exit_signal,
        }
    )


def calculate_hma_squeeze_signals(
    hma_21: np.ndarray, hma_89: np.ndarray, squeeze_window: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Computes squeeze and entry/exit signals with NumPy magic."""
    hma_diff = np.abs(hma_21 - hma_89)
    squeeze_threshold = HullMovingAverage(hma_diff, squeeze_window).calculate()
    squeeze = hma_diff <= squeeze_threshold

    # Vectorized signals with boundary safety
    prev_hma_21 = np.roll(hma_21, 1)
    prev_hma_89 = np.roll(hma_89, 1)
    entry = (hma_21 > hma_89) & (prev_hma_21 <= prev_hma_89) & squeeze
    exit_signal = (hma_21 < hma_89) & (prev_hma_21 >= prev_hma_89)
    entry[0], exit_signal[0] = False, False  # No signals at t=0

    return squeeze, entry, exit_signal


def plot_hma_with_squeeze(
    prices: np.ndarray, dates: np.ndarray, hma_data: pd.DataFrame, ticker: str
) -> None:
    """Generates and displays an interactive Plotly chart."""
    fig = make_subplots(rows=1, cols=1, shared_xaxes=True, vertical_spacing=0.03)

    # HMA Cloud
    fig.add_trace(
        go.Scatter(
            x=np.concatenate([dates, dates[::-1]]),
            y=np.concatenate([hma_data["hma_21"], hma_data["hma_89"][::-1]]),
            fill="toself",
            fillcolor=COLOR_SCHEME["cloud"],
            line=dict(color="rgba(255,255,255,0)"),
            showlegend=False,
            name="HMA Cloud",
        ),
        row=1,
        col=1,
    )

    # Price and HMAs
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=prices,
            mode="lines",
            name="Closing Price",
            line=dict(color=COLOR_SCHEME["raw_price"]),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=hma_data["hma_21"],
            mode="lines",
            name=HMA_21_LABEL,
            line=dict(color=COLOR_SCHEME["pole"][21]["line"]),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=hma_data["hma_89"],
            mode="lines",
            name=HMA_89_LABEL,
            line=dict(color=COLOR_SCHEME["pole"][89]["line"]),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=hma_data["hma_200"],
            mode="lines",
            name=HMA_200_LABEL,
            line=dict(color=COLOR_SCHEME["pole"][200]["line"]),
        ),
        row=1,
        col=1,
    )

    # Signals
    fig.add_trace(
        go.Scatter(
            x=dates[hma_data["squeeze"]],
            y=prices[hma_data["squeeze"]],
            mode="markers",
            marker=dict(color="yellow", size=10, line=dict(color="white", width=1)),
            showlegend=False,
            name="Squeeze",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=dates[hma_data["entry"]],
            y=prices[hma_data["entry"]],
            mode="markers",
            marker=dict(
                color="blue", size=25, symbol="circle", line=dict(color="white", width=1)
            ),
            showlegend=False,
            name="Entry",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=dates[hma_data["exit"]],
            y=prices[hma_data["exit"]],
            mode="markers",
            marker=dict(
                color="orange",
                size=25,
                symbol="cross",
                line=dict(color="white", width=1),
            ),
            showlegend=False,
            name="Exit",
        ),
        row=1,
        col=1,
    )

    # Layout
    fig.update_layout(
        title=dict(
            text=f"{ticker} Closing Price with HMA (21, 89, 200) and Squeeze",
            x=0.5,
            xanchor="center",
        ),
        xaxis_title="Date",
        yaxis_title="Price",
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="v", yanchor="bottom", y=0, xanchor="right", x=1),
    )

    # Save and show
    output_dir = Path("hma_plots")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{ticker}_hma.html"
    with open(output_file, "w", encoding="utf-8") as f:
        fig.write_html(f)
    fig.show()


def backtest_strategy(
    prices: np.ndarray,
    entry_signals: np.ndarray,
    exit_signals: np.ndarray,
    initial_capital: float = 10000.0,
) -> tuple[float, np.ndarray]:
    """Backtests the strategy with precise position tracking."""
    positions = np.zeros_like(prices, dtype=int)
    pos_change = np.where(entry_signals, 1, 0) - np.where(exit_signals, 1, 0)
    positions = np.cumsum(pos_change)
    positions = np.clip(positions, 0, 1)  # Long or neutral only

    # Calculate daily returns
    returns = np.zeros_like(prices, dtype=float)
    active = positions > 0
    price_changes = np.diff(prices, prepend=prices[0])
    returns[active] = price_changes[active] * positions[active]
    portfolio_value = initial_capital + np.cumsum(returns)

    # Final value accounts for open positions
    final_value = (
        portfolio_value[-1]
        if not active[-1]
        else portfolio_value[-1] + prices[-1] * positions[-1]
    )
    return final_value, portfolio_value


def calculate_cagr(
    begin_value: float, end_value: float, num_periods: float
) -> float | None:
    """Calculates Compound Annual Growth Rate."""
    if begin_value <= 0 or num_periods <= 0:
        return None
    return (end_value / begin_value) ** (1 / num_periods) - 1


class Config(BaseModel):
    """Configuration with strict validation."""
    tickers: list[str] = Field(..., description="List of stock tickers")
    period: str = Field(default="2y", description="Data period (e.g., '2y', '6mo')")
    interval: str = Field(default="1h", description="Data interval (e.g., '1h', '1d')")
    squeeze_window: int = Field(default=20, description="Squeeze window", ge=5)

    @field_validator("period")
    def validate_period(cls, v):
        if not any(char.isdigit() for char in v):
            raise ValueError("Period must contain a number")
        if not v.endswith(("d", "wk", "mo", "y")):
            raise ValueError("Period must end with 'd', 'wk', 'mo', or 'y'")
        return v

    @field_validator("interval")
    def validate_interval(cls, v):
        if not any(char.isdigit() for char in v):
            raise ValueError("Interval must contain a number")
        if not v.endswith(("m", "h", "d", "wk", "mo")):
            raise ValueError("Interval must end with 'm', 'h', 'd', 'wk', or 'mo'")
        return v


def main() -> None:
    """Orchestrates the HMA squeeze analysis."""
    parser = argparse.ArgumentParser(description="Calculate and plot HMA with Squeeze.")
    parser.add_argument(
        "tickers", nargs="+", help="List of stock tickers (e.g., SPY AAPL MSFT)"
    )
    parser.add_argument("-p", "--period", type=str, help="Data period (default: 2y)")
    parser.add_argument("-i", "--interval", type=str, help="Data interval (default: 1h)")
    parser.add_argument(
        "-sw", "--squeeze_window", type=int, help="Squeeze window (default: 20)"
    )
    args = parser.parse_args()

    args_dict = {k: v for k, v in vars(args).items() if v is not None}
    config = Config(**args_dict)

    console = Console()

    for ticker in tqdm(config.tickers, desc="Processing Tickers"):
        closing_prices, dates = get_closing_prices(ticker, config.period, config.interval)
        if closing_prices is None or dates is None:
            console.print(f"[bold red]Error: Could not fetch data for {ticker}[/]")
            continue

        hma_data = calculate_all_hmas(closing_prices, config.squeeze_window)
        final_value_strategy, portfolio_value = backtest_strategy(
            closing_prices,
            hma_data["entry"].to_numpy(),
            hma_data["exit"].to_numpy(),
        )
        initial_price = closing_prices[0]
        final_price_bh = closing_prices[-1]
        final_value_bh = 10000.0 * (final_price_bh / initial_price)

        num_years = (
            float(config.period.replace("y", ""))
            if "y" in config.period
            else float(config.period.replace("mo", "")) / 12
        )
        cagr_strategy = calculate_cagr(10000.0, final_value_strategy, num_years)
        cagr_bh = calculate_cagr(10000.0, final_value_bh, num_years)

        # Rich table output
        table = Table(title=Text(f"Backtesting Results for {ticker}", style="bold blue"))
        table.add_column("Strategy", justify="right", style="cyan", no_wrap=True)
        table.add_column("Final Value", justify="right", style="magenta")
        table.add_column("CAGR", justify="right", style="green")
        table.add_row(
            "HMA Squeeze",
            f"${final_value_strategy:.2f}",
            f"{cagr_strategy:.2%}" if cagr_strategy is not None else "N/A",
        )
        table.add_row(
            "Buy & Hold",
            f"${final_value_bh:.2f}",
            f"{cagr_bh:.2%}" if cagr_bh is not None else "N/A",
        )
        console.print(table)

        plot_hma_with_squeeze(closing_prices, dates, hma_data, ticker)


if __name__ == "__main__":
    main()