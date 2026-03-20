import streamlit as st
import pandas as pd
import requests
import traceback

st.set_page_config(page_title="Systematic Trading Demo", layout="wide")

API_URL = "https://api-469354767887.europe-west1.run.app/backtest"

st.title("Systematic Trading Demo")
st.write("Run a simple backtest with a cutoff date and starting budget.")

show_debug = st.sidebar.checkbox("Show debug logs", value=True)

with st.form("backtest_form"):
    cutoff_date = st.date_input("Cutoff date", value=pd.to_datetime("2025-01-01"))
    initial_capital = st.number_input(
        "Budget / Initial capital",
        min_value=100.0,
        value=1000.0,
        step=100.0,
    )
    submitted = st.form_submit_button("Run backtest")

if submitted:
    try:
        if show_debug:
            st.write("Sending request to API...")
            st.write(
                {
                    "url": API_URL,
                    "cutoff_date": cutoff_date.strftime("%Y-%m-%d"),
                    "initial_capital": float(initial_capital),
                }
            )

        with st.spinner("Running backtest..."):
            response = requests.get(
                API_URL,
                params={
                    "cutoff_date": cutoff_date.strftime("%Y-%m-%d"),
                    "initial_capital": float(initial_capital),
                },
                timeout=30,
            )

        if show_debug:
            st.write(f"Response status: {response.status_code}")
            st.write("Response headers:")
            st.write(dict(response.headers))

        response.raise_for_status()
        summary = response.json()

        if show_debug:
            st.write("Response received successfully.")
            st.write("Top-level keys:")
            st.write(list(summary.keys()))

        st.success("Backtest complete")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Final capital", f"${summary.get('final_capital', 0):,.2f}")
        col2.metric("Total return", f"{summary.get('total_return_pct', 0)}%")
        col3.metric("Win rate", f"{summary.get('win_rate_pct', 0)}%")
        col4.metric("Max drawdown", f"{summary.get('max_drawdown_pct', 0)}%")

        st.subheader("Performance summary")
        summary_table = {
            "Initial capital": summary.get("initial_capital"),
            "Final capital": summary.get("final_capital"),
            "Total return %": summary.get("total_return_pct"),
            "Annualised return %": summary.get("annualised_return_pct"),
            "Total trades": summary.get("total_trades"),
            "Winning trades": summary.get("winning_trades"),
            "Losing trades": summary.get("losing_trades"),
            "Win rate %": summary.get("win_rate_pct"),
            "Loss rate %": summary.get("loss_rate_pct"),
            "Avg win PnL": summary.get("avg_win_pnl"),
            "Avg loss PnL": summary.get("avg_loss_pnl"),
            "Implied costs": summary.get("implied_costs"),
            "Sharpe ratio": summary.get("sharpe_ratio"),
            "Max drawdown %": summary.get("max_drawdown_pct"),
            "Profit factor": summary.get("profit_factor"),
            "Buy & hold return %": summary.get("bnh_return_pct"),
        }
        summary_df = pd.DataFrame(
            {"Metric": list(summary_table.keys()), "Value": list(summary_table.values())}
        )
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

        equity_curve = summary.get("equity_curve", [])
        if equity_curve:
            st.subheader("Equity curve")
            equity_df = pd.DataFrame(equity_curve)
            equity_df["date"] = pd.to_datetime(equity_df["date"])
            equity_df = equity_df.set_index("date")
            st.line_chart(equity_df["equity"])

            if show_debug:
                with st.expander("Raw equity curve data"):
                    st.dataframe(equity_df.reset_index(), use_container_width=True)

        action_breakdown = summary.get("action_breakdown", {})
        if action_breakdown:
            st.subheader("Action breakdown")
            action_df = pd.DataFrame(
                {
                    "Action": list(action_breakdown.keys()),
                    "Count": list(action_breakdown.values()),
                }
            )
            st.dataframe(action_df, use_container_width=True, hide_index=True)

    except requests.exceptions.HTTPError as e:
        st.error(f"HTTP error: {e}")
        st.code(response.text if "response" in locals() else "No response body", language="text")
        print("HTTP ERROR")
        print(traceback.format_exc())

    except requests.exceptions.RequestException as e:
        st.error(f"Request error: {e}")
        st.code(traceback.format_exc(), language="python")
        print("REQUEST ERROR")
        print(traceback.format_exc())

    except Exception as e:
        st.error(f"Unexpected error: {e}")
        st.code(traceback.format_exc(), language="python")
        print("UNEXPECTED ERROR")
        print(traceback.format_exc())
