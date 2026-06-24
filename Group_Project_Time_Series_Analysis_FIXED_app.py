import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import boxcox
from statsmodels.tsa.stattools import adfuller
import plotly.graph_objects as go

st.set_page_config(page_title="Time Series Analysis - Group Project", layout="wide")

st.title("Axis 1 - Data Ingestion & Stationarity")

st.markdown("""
Welcome to the Time Series Analysis Application.

This module allows you to:
- Upload datasets
- Select columns
- Handle missing values
- Visualize time series
- Prepare data for stationarity analysis
""")

# =========================
# Session State Initialization
# =========================
# Streamlit reruns this whole script top-to-bottom on every click.
# Anything we want to "survive" a rerun (cleaned data, transformed
# series, last ADF result, etc.) must live in st.session_state,
# otherwise it resets to None every time a different button is pressed.

if "df" not in st.session_state:
    st.session_state.df = None

if "cleaned" not in st.session_state:
    st.session_state.cleaned = False

if "boxcox_result" not in st.session_state:
    st.session_state.boxcox_result = None  # (transformed_data, lambda_used)

if "diff_result" not in st.session_state:
    st.session_state.diff_result = None  # (diff_series, d, D, s)

# =========================
# Upload File
# =========================

uploaded_file = st.file_uploader(
    "Upload your dataset",
    type=["csv", "xlsx", "json"]
)

# =========================
# Before Upload
# =========================

if uploaded_file is None:

    st.info(
        "Please upload a dataset to begin."
    )

# =========================
# After Upload
# =========================

else:

    file_name = uploaded_file.name

    # Only re-read the file when a NEW file is uploaded, so that
    # cleaning/transform actions done in previous reruns are not
    # wiped out just because Streamlit reran the script.
    if (
        st.session_state.df is None
        or st.session_state.get("last_uploaded_name") != file_name
    ):

        # CSV
        if file_name.endswith(".csv"):
            df_loaded = pd.read_csv(uploaded_file)

        # Excel
        elif file_name.endswith(".xlsx"):
            df_loaded = pd.read_excel(uploaded_file)

        # JSON
        elif file_name.endswith(".json"):
            df_loaded = pd.read_json(uploaded_file)

        else:
            st.error("Unsupported file type.")
            st.stop()

        st.session_state.df = df_loaded
        st.session_state.last_uploaded_name = file_name
        st.session_state.cleaned = False
        st.session_state.boxcox_result = None
        st.session_state.diff_result = None

    df = st.session_state.df

    # =========================
    # Preview
    # =========================

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # =========================
    # Column Selection
    # =========================

    columns = df.columns.tolist()

    time_col = st.selectbox(
        "Select Time Column",
        columns
    )

    value_col = st.selectbox(
        "Select Value Column",
        columns,
        index=min(1, len(columns) - 1)
    )

    st.success(f"Time Column: {time_col}")
    st.success(f"Value Column: {value_col}")

    # Guard: time and value columns must be different
    if time_col == value_col:
        st.warning(
            "Time column and value column are the same. "
            "Please select two different columns."
        )
        st.stop()

    # Guard: value column must be numeric
    if not pd.api.types.is_numeric_dtype(df[value_col]):
        st.error(
            f"Column '{value_col}' is not numeric. "
            "Please choose a numeric column as the value series."
        )
        st.stop()

    # Try to parse the time column as datetime for nicer plotting.
    # If it fails, we just keep it as-is (e.g. integer time index).
    try:
        df[time_col] = pd.to_datetime(df[time_col])
    except (ValueError, TypeError):
        pass

    # =========================
    # Missing Values
    # =========================

    st.subheader("Missing Values Analysis")

    missing_count = int(df[value_col].isnull().sum())

    total_values = len(df[value_col])

    missing_percent = (
        missing_count / total_values
    ) * 100

    col_a, col_b = st.columns(2)
    col_a.metric("Missing Values", missing_count)
    col_b.metric("Missing Percentage", f"{missing_percent:.2f}%")

    # Warning
    if missing_percent > 5:

        st.warning(
            "Warning: More than 5% of data is missing! "
            "Heavy imputation can distort the autocorrelation structure "
            "and bias the stationarity tests below."
        )

    # =========================
    # Cleaning Method
    # =========================

    method = st.selectbox(
        "Choose Missing Value Handling Method",
        [
            "Linear Interpolation",
            "Forward Fill",
            "Mean Imputation"
        ]
    )

    # =========================
    # Apply Cleaning
    # =========================

    if missing_count == 0:
        st.info("No missing values detected — cleaning is optional.")

    if st.button("Apply Cleaning"):

        # Linear interpolation
        if method == "Linear Interpolation":

            df[value_col] = (
                df[value_col]
                .interpolate()
            )

        # Forward fill
        elif method == "Forward Fill":

            df[value_col] = (
                df[value_col]
                .ffill()
            )

        # Mean imputation
        elif method == "Mean Imputation":

            mean_value = (
                df[value_col].mean()
            )

            df[value_col] = (
                df[value_col]
                .fillna(mean_value)
            )

        # Any leftover NaNs (e.g. forward-fill with a leading NaN)
        # are dropped so downstream plots/tests don't break.
        df[value_col] = df[value_col].bfill()

        st.session_state.df = df
        st.session_state.cleaned = True

        st.success(
            "Missing values handled successfully!"
        )

    if st.session_state.cleaned:
        st.caption("Cleaning has been applied to the working dataset.")

    # =========================
    # Raw Time Series Plot
    # =========================

    st.subheader("Raw Time Series Plot")

    fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(
        df[time_col],
        df[value_col]
    )

    ax.set_title("Original Time Series")
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")

    fig.autofmt_xdate(rotation=45)

    st.pyplot(fig)

    # =========================
    # Rolling Statistics
    # =========================

    st.subheader("Rolling Statistics")

    window = st.slider(
        "Select Rolling Window Size",
        min_value=2,
        max_value=50,
        value=10
    )

    # Rolling calculations

    rolling_mean = (
        df[value_col]
        .rolling(window=window)
        .mean()
    )

    rolling_std = (
        df[value_col]
        .rolling(window=window)
        .std()
    )

    # =========================
    # Plot Rolling Statistics
    # =========================

    fig2, ax2 = plt.subplots(figsize=(10, 4))

    ax2.plot(
        df[time_col],
        df[value_col],
        label="Original Series",
        alpha=0.5
    )

    ax2.plot(
        df[time_col],
        rolling_mean,
        label="Rolling Mean"
    )

    ax2.plot(
        df[time_col],
        rolling_std,
        label="Rolling Std"
    )

    ax2.set_title(
        "Rolling Mean & Standard Deviation"
    )

    ax2.set_xlabel("Time")
    ax2.set_ylabel("Value")

    ax2.legend()

    fig2.autofmt_xdate(rotation=45)

    st.pyplot(fig2)

    st.caption(
        "If the rolling mean drifts noticeably (trend) or the rolling "
        "standard deviation visibly widens/narrows over time (changing "
        "variance), the series is likely non-stationary — this is the "
        "informal visual counterpart to the formal ADF test below."
    )

    # =========================
    # ADF Test (on raw / cleaned series)
    # =========================

    st.subheader("Augmented Dickey-Fuller Test")

    with st.expander("What does this test check?"):
        st.markdown(
            "The ADF test fits the regression "
            r"$\Delta X_t = \alpha + \beta t + \gamma X_{t-1} + "
            r"\sum_{i=1}^{p}\delta_i \Delta X_{t-i} + \varepsilon_t$ "
            "and tests $H_0: \\gamma = 0$ (a unit root, i.e. "
            "non-stationary) against $H_1: \\gamma < 0$ (stationary). "
            "A small p-value lets us reject $H_0$."
        )

    if st.button("Run ADF Test", key="adf_raw"):

        # Remove missing values
        series = df[value_col].dropna()

        if len(series) < 10:
            st.error("Not enough data points to run the ADF test.")
        else:
            # Run test
            result = adfuller(series)

            adf_statistic = result[0]
            p_value = result[1]
            critical_values = result[4]

            # =========================
            # Display Results
            # =========================

            st.write(
                f"ADF Statistic: {adf_statistic:.4f}"
            )

            st.write(
                f"P-value: {p_value:.4f}"
            )

            st.write("Critical Values:")

            for key, value in critical_values.items():

                st.write(
                    f"{key}: {value:.4f}"
                )

            # =========================
            # Final Verdict
            # =========================

            if p_value < 0.05:

                st.success(
                    "The series is stationary at the 5% significance level."
                )

            else:

                st.error(
                    "The series is non-stationary at the 5% significance level."
                )

    # =========================
    # Box-Cox Transformation
    # =========================

    st.subheader("Box-Cox Transformation")

    st.markdown("""
    Box-Cox transformation helps stabilize variance
    and improve stationarity.
    """)

    # Check positivity

    if (df[value_col] <= 0).any():

        st.error(
            "Box-Cox requires strictly positive values. "
            "This series contains zero or negative values, so the "
            "Box-Cox transformation cannot be applied."
        )

    else:

        bc_mode = st.radio(
            "Lambda selection mode",
            ["Manual (slider)", "Auto-Optimize (max log-likelihood)"],
            horizontal=True
        )

        if bc_mode == "Manual (slider)":

            lambda_value = st.slider(
                "Select Lambda Value",
                min_value=-2.0,
                max_value=2.0,
                value=0.0,
                step=0.1
            )

        else:
            st.caption(
                "Lambda will be chosen automatically to maximize the "
                "Box-Cox profile log-likelihood (scipy's lmbda=None mode)."
            )
            lambda_value = None

        # Apply transformation

        if st.button("Apply Box-Cox Transformation"):

            clean_series = df[value_col].dropna()

            if bc_mode == "Manual (slider)":
                transformed_data = boxcox(clean_series, lmbda=lambda_value)
                fitted_lambda = lambda_value
            else:
                transformed_data, fitted_lambda = boxcox(clean_series, lmbda=None)

            st.session_state.boxcox_result = (transformed_data, fitted_lambda)

            st.success(
                f"Box-Cox transformation applied with lambda = {fitted_lambda:.4f}"
            )

        # Render results from session_state so they persist across reruns
        # (e.g. while the user then plays with the differencing controls below).
        if st.session_state.boxcox_result is not None:

            transformed_data, fitted_lambda = st.session_state.boxcox_result

            # =========================
            # Plot transformed series
            # =========================

            st.subheader(
                "Transformed Time Series"
            )

            fig3, ax3 = plt.subplots(
                figsize=(10, 4)
            )

            ax3.plot(
                df[time_col].iloc[:len(transformed_data)],
                transformed_data
            )

            ax3.set_title(
                f"Box-Cox Transformed Series (lambda = {fitted_lambda:.4f})"
            )

            ax3.set_xlabel("Time")
            ax3.set_ylabel("Transformed Value")

            fig3.autofmt_xdate(rotation=45)

            st.pyplot(fig3)

            # =========================
            # Rolling Statistics
            # =========================

            st.subheader(
                "Rolling Statistics After Box-Cox"
            )

            rolling_mean_bc = (
                pd.Series(transformed_data)
                .rolling(window=window)
                .mean()
            )

            rolling_std_bc = (
                pd.Series(transformed_data)
                .rolling(window=window)
                .std()
            )

            fig4, ax4 = plt.subplots(
                figsize=(10, 4)
            )

            ax4.plot(
                transformed_data,
                label="Transformed Series",
                alpha=0.5
            )

            ax4.plot(
                rolling_mean_bc,
                label="Rolling Mean"
            )

            ax4.plot(
                rolling_std_bc,
                label="Rolling Std"
            )

            ax4.set_title(
                "Rolling Statistics After Box-Cox"
            )

            ax4.legend()

            st.pyplot(fig4)

            # =========================
            # ADF Test After Box-Cox
            # =========================

            st.subheader(
                "ADF Test After Box-Cox"
            )

            result_bc = adfuller(
                transformed_data
            )

            adf_stat_bc = result_bc[0]
            p_value_bc = result_bc[1]
            critical_values_bc = result_bc[4]

            st.write(
                f"ADF Statistic: {adf_stat_bc:.4f}"
            )

            st.write(
                f"P-value: {p_value_bc:.4f}"
            )

            st.write("Critical Values:")

            for key, value in critical_values_bc.items():

                st.write(
                    f"{key}: {value:.4f}"
                )

            # =========================
            # Final Verdict
            # =========================

            if p_value_bc < 0.05:

                st.success(
                    "The transformed series is stationary."
                )

            else:

                st.error(
                    "The transformed series is still non-stationary."
                )

            if st.button("Clear Box-Cox result"):
                st.session_state.boxcox_result = None
                st.rerun()

    # =========================
    # Differencing Engine
    # =========================

    st.subheader("Differencing Engine")

    st.markdown("""
    Differencing helps remove trends and seasonality
    to improve stationarity.
    """)

    # Regular differencing order

    d = st.number_input(
        "Regular Differencing Order (d)",
        min_value=0,
        max_value=5,
        value=1
    )

    # Seasonal differencing order

    D = st.number_input(
        "Seasonal Differencing Order (D)",
        min_value=0,
        max_value=5,
        value=0
    )

    # Seasonal period

    s = st.number_input(
        "Seasonal Period (s)",
        min_value=1,
        max_value=365,
        value=12
    )

    # Guard against asking for more differences than the data can support
    min_required = d + D * s + 1
    if min_required >= len(df[value_col].dropna()):
        st.warning(
            f"This combination of d={d}, D={D}, s={s} needs at least "
            f"{min_required} observations, but the series only has "
            f"{len(df[value_col].dropna())}. Reduce the orders or the period."
        )

    # =========================
    # Apply Differencing
    # =========================

    if st.button("Apply Differencing"):

        diff_series = df[value_col].copy()

        # Regular differencing

        for i in range(d):

            diff_series = diff_series.diff()

        # Seasonal differencing

        for i in range(D):

            diff_series = diff_series.diff(s)

        # Remove missing values

        diff_series = diff_series.dropna()

        if len(diff_series) < 10:
            st.error(
                "Differencing left too few observations to analyze. "
                "Try a smaller d, D, or s."
            )
        else:
            st.session_state.diff_result = (diff_series, d, D, s)
            st.session_state.stationary_series = diff_series
            st.success(
                "Differencing applied successfully!"
            )

    if st.session_state.diff_result is not None:

        diff_series, used_d, used_D, used_s = st.session_state.diff_result

        # =========================
        # Plot Differenced Series
        # =========================

        st.subheader(
            "Differenced Time Series"
        )

        fig5, ax5 = plt.subplots(
            figsize=(10, 4)
        )

        ax5.plot(
            df[time_col].iloc[-len(diff_series):],
            diff_series
        )

        ax5.set_title(
            f"Differenced Series (d={used_d}, D={used_D}, s={used_s})"
        )

        ax5.set_xlabel("Time")
        ax5.set_ylabel("Differenced Value")

        fig5.autofmt_xdate(rotation=45)

        st.pyplot(fig5)

        # =========================
        # Rolling Statistics
        # =========================

        st.subheader(
            "Rolling Statistics After Differencing"
        )

        rolling_mean_diff = (
            diff_series
            .rolling(window=window)
            .mean()
        )

        rolling_std_diff = (
            diff_series
            .rolling(window=window)
            .std()
        )

        fig6, ax6 = plt.subplots(
            figsize=(10, 4)
        )

        ax6.plot(
            diff_series.values,
            label="Differenced Series",
            alpha=0.5
        )

        ax6.plot(
            rolling_mean_diff.values,
            label="Rolling Mean"
        )

        ax6.plot(
            rolling_std_diff.values,
            label="Rolling Std"
        )

        ax6.legend()

        ax6.set_title(
            "Rolling Statistics After Differencing"
        )

        st.pyplot(fig6)

        # =========================
        # ADF Test
        # =========================

        st.subheader(
            "ADF Test After Differencing"
        )

        result_diff = adfuller(
            diff_series
        )

        adf_stat_diff = result_diff[0]
        p_value_diff = result_diff[1]
        critical_values_diff = result_diff[4]

        st.write(
            f"ADF Statistic: {adf_stat_diff:.4f}"
        )

        st.write(
            f"P-value: {p_value_diff:.4f}"
        )

        st.write("Critical Values:")

        for key, value in critical_values_diff.items():

            st.write(
                f"{key}: {value:.4f}"
            )

        # Final verdict

        if p_value_diff < 0.05:

            st.success(
                "The differenced series is stationary."
            )

        else:

            st.error(
                "The differenced series is still non-stationary."
            )

        # =========================
        # BONUS: Frequency Response of the Differencing Filter
        # =========================
        # Connects Axis 1 to Axis 2: differencing is a linear filter,
        # and its squared gain function shows it acts as a high-pass
        # filter, amplifying high-frequency (short-period) variation
        # and suppressing low-frequency (trend-like) variation.

        with st.expander(
            "Bonus: Frequency Response of the Differencing Filter"
        ):

            st.markdown(
                "Regular differencing of order d has squared gain "
                r"$|1 - e^{-i\lambda}|^{2d}$, and seasonal differencing "
                r"of order D at period s has squared gain "
                r"$|1 - e^{-i\lambda s}|^{2D}$. Multiplying them gives the "
                "combined filter response below. Frequencies where the "
                "gain is near zero are heavily damped; frequencies where "
                "the gain is large are amplified."
            )

            lam = np.linspace(0.001, np.pi, 500)

            gain_regular = np.abs(1 - np.exp(-1j * lam)) ** (2 * used_d) \
                if used_d > 0 else np.ones_like(lam)

            gain_seasonal = np.abs(1 - np.exp(-1j * lam * used_s)) ** (2 * used_D) \
                if used_D > 0 else np.ones_like(lam)

            total_gain = gain_regular * gain_seasonal

            fig7, ax7 = plt.subplots(figsize=(10, 4))
            ax7.plot(lam, total_gain, label="Combined filter gain")
            if used_d > 0:
                ax7.plot(lam, gain_regular, "--", alpha=0.6, label=f"Regular (d={used_d})")
            if used_D > 0:
                ax7.plot(lam, gain_seasonal, "--", alpha=0.6, label=f"Seasonal (D={used_D}, s={used_s})")
            ax7.set_title("Squared Gain Function of the Differencing Filter")
            ax7.set_xlabel(r"Frequency $\lambda$ (radians)")
            ax7.set_ylabel(r"$|H(\lambda)|^2$")
            ax7.legend()
            st.pyplot(fig7)

            st.caption(
                "Note how the gain is near zero at low frequencies "
                "(the trend) and grows toward high frequencies — this is "
                "exactly why differencing removes trends but can inflate "
                "high-frequency noise."
            )

    # =========================
    # Output Contract for Axis 2 / Axis 3
    # =========================
    # Per the project spec, Axis 1 must hand off a stationary series
    # plus metadata describing every transformation applied, so that
    # Axis 5 can back-transform forecasts to the original scale.

    st.divider()
    st.subheader("Output Contract (for Axis 2 / Axis 3)")

    output_metadata = {
        "time_column": time_col,
        "value_column": value_col,
        "missing_value_method": method if st.session_state.cleaned else None,
        "boxcox_lambda": (
            float(st.session_state.boxcox_result[1])
            if st.session_state.boxcox_result is not None
            else None
        ),
        "differencing_d": (
            int(st.session_state.diff_result[1])
            if st.session_state.diff_result is not None
            else 0
        ),
        "differencing_D": (
            int(st.session_state.diff_result[2])
            if st.session_state.diff_result is not None
            else 0
        ),
        "seasonal_period_s": (
            int(st.session_state.diff_result[3])
            if st.session_state.diff_result is not None
            else None
        ),
    }

    st.json(output_metadata)

    st.caption(
        "This metadata (and the resulting stationary series) is what "
        "should be passed downstream so forecasts in Axis 5 can be "
        "back-transformed to the original scale."
    )

    # =========================
    # FIX: Axis 1 must also export the transformation metadata under
    # the exact key Axis 5 looks for ("transformation_metadata"), and
    # must include the original (untransformed) series values so
    # differencing can be reversed by cumulative summation. Previously
    # this dict was only built for display (st.json) and never written
    # to session_state, so Axis 5 always fell back to its default
    # (no Box-Cox, no differencing) metadata regardless of what was
    # actually done in Axis 1.
    # =========================

    st.session_state["transformation_metadata"] = {
        "boxcox_lambda": output_metadata["boxcox_lambda"],
        "differencing_d": output_metadata["differencing_d"],
        "differencing_D": output_metadata["differencing_D"],
        "seasonal_period_s": output_metadata["seasonal_period_s"],
        "original_series": df[value_col].dropna().values.tolist(),
    }





































































##########################################################
# AXIS 2 : SPECTRAL ANALYSIS
# Time Series Project
#
# Author : Your Group
#
# This module receives a stationary series from Axis 1
# and performs Frequency Domain Analysis.
##########################################################

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal import find_peaks
from scipy.stats import chi2

##########################################################
# PAGE CONFIG
##########################################################




st.title("Axis 2 : Spectral Analysis & Frequency Exploration")

st.markdown("""
This module receives a stationary time series from Axis 1
and explores hidden cycles using spectral methods.
""")

##########################################################
# AXIS 1 CONNECTION AREA
##########################################################

st.header("Input From Axis 1")

if "stationary_series" not in st.session_state:
    st.error("Run Axis 1 first.")
    st.stop()

series = st.session_state.stationary_series.dropna().values


##########################################################
# BASIC CHECKS
##########################################################

if len(series) < 20:
    st.error(
        "Series too short. Need at least 20 observations."
    )
    st.stop()

n = len(series)

##########################################################
# REMOVE MEAN
##########################################################

x = series - np.mean(series)

st.success(
    f"Loaded {n} observations."
)

##########################################################
# FREQUENCY / PERIOD TOGGLE
##########################################################

st.header("Display Options")

display_mode = st.radio(
    "X Axis",
    [
        "Frequency",
        "Period"
    ]
)

##########################################################
# TAPERING
##########################################################

st.header("Tapering")

taper_type = st.selectbox(
    "Select Taper",
    [
        "None",
        "Hann",
        "Hamming",
        "Cosine Bell"
    ]
)

##########################################################
# WINDOW CREATION
##########################################################

if taper_type == "Hann":

    taper_window = np.hanning(n)

elif taper_type == "Hamming":

    taper_window = np.hamming(n)

elif taper_type == "Cosine Bell":

    taper_window = np.sin(
        np.pi * np.arange(n) / (n - 1)
    )

else:

    taper_window = np.ones(n)

##########################################################
# APPLY TAPER
##########################################################

x_tapered = x * taper_window

##########################################################
# RAW PERIODOGRAM
##########################################################

st.header("Raw Periodogram")

##########################################################
# FFT
##########################################################

fft_values = np.fft.rfft(x_tapered)

##########################################################
# FREQUENCIES
##########################################################

frequencies = np.fft.rfftfreq(n)
frequencies_lambda = 2*np.pi*frequencies

##########################################################
# PERIODOGRAM
##########################################################

periodogram = (
    1 / (2 * np.pi * n)
) * np.abs(fft_values) ** 2

##########################################################
# PERIOD AXIS
##########################################################

period_axis = np.zeros_like(frequencies)

for i in range(1, len(frequencies)):

    if frequencies[i] > 0:

        period_axis[i] = (
            1 / frequencies[i]
        )

##########################################################
# PLOT
##########################################################

fig, ax = plt.subplots(
    figsize=(10,5)
)

if display_mode == "Frequency":

    ax.plot(
        frequencies_lambda,
        periodogram
    )

    ax.set_xlabel(
        "Frequency λ (radians)"
    )

else:

    valid = period_axis > 0

    ax.plot(
        period_axis[valid],
        periodogram[valid]
    )

    ax.set_xlabel(
        "Period"
    )

ax.set_ylabel(
    "Periodogram"
)

ax.set_title(
    "Raw Periodogram"
)

ax.grid(True)

st.pyplot(fig)

##########################################################
# STATISTICAL WARNING
##########################################################

st.warning(
"""
IMPORTANT:

The raw periodogram is asymptotically unbiased
but NOT a consistent estimator.

Its variance does not decrease with sample size.

Use the smoothed spectral estimator
for reliable inference.
"""
)

##########################################################
# SUMMARY PANEL
##########################################################

with st.expander(
    "What has been done so far?"
):

    st.write(
        """
        1. Stationary series received from Axis 1

        2. Mean removed

        3. Optional taper applied

        4. FFT computed

        5. Raw Periodogram displayed
        """
    )

##########################################################
# PART 2
# SAMPLE AUTOCOVARIANCE
# LAG WINDOWS
# SMOOTHED SPECTRAL ESTIMATOR
##########################################################

st.header("Smoothed Spectral Estimator")

st.markdown("""
This is the core tool of Axis 2.

Students must implement the lag-window
spectral estimator themselves.
""")

##########################################################
# BANDWIDTH
##########################################################

max_M = min(150, n // 2)

M = st.slider(
    "Bandwidth (M)",
    min_value=5,
    max_value=max_M,
    value=min(30, max_M)
)

##########################################################
# WINDOW SELECTION
##########################################################

window_type = st.selectbox(
    "Lag Window",
    [
        "Daniell",
        "Bartlett",
        "Parzen"
    ]
)

##########################################################
# SAMPLE AUTOCOVARIANCE
##########################################################

def sample_autocovariance(x, maxlag):

    n = len(x)

    gamma = []

    for h in range(maxlag + 1):

        value = np.sum(
            x[:n-h] * x[h:]
        ) / n

        gamma.append(value)

    return np.array(gamma)

##########################################################
# COMPUTE GAMMA(H)
##########################################################

gamma_hat = sample_autocovariance(
    x_tapered,
    M
)

##########################################################
# DANIELL WINDOW
##########################################################

def daniell_window(u):

    u = np.abs(u)

    return np.where(
        u <= 1,
        1,
        0
    )

##########################################################
# BARTLETT WINDOW
##########################################################

def bartlett_window(u):

    u = np.abs(u)

    return np.where(
        u <= 1,
        1 - u,
        0
    )

##########################################################
# PARZEN WINDOW
##########################################################

def parzen_window(u):

    u = np.abs(u)

    w = np.zeros_like(u)

    region1 = u <= 0.5

    w[region1] = (
        1
        - 6 * u[region1]**2
        + 6 * u[region1]**3
    )

    region2 = (
        (u > 0.5)
        &
        (u <= 1)
    )

    w[region2] = (
        2 * (1 - u[region2])**3
    )

    return w

##########################################################
# SELECT WINDOW FUNCTION
##########################################################

if window_type == "Daniell":

    selected_window = daniell_window

elif window_type == "Bartlett":

    selected_window = bartlett_window

else:

    selected_window = parzen_window

##########################################################
# DISPLAY WINDOW SHAPE
##########################################################

st.subheader("Selected Window Shape")

u_plot = np.linspace(
    -1,
    1,
    500
)

fig, ax = plt.subplots(
    figsize=(8,4)
)

ax.plot(
    u_plot,
    selected_window(u_plot)
)

ax.set_title(
    f"{window_type} Window"
)

ax.grid(True)

st.pyplot(fig)

##########################################################
# FREQUENCY GRID
##########################################################

lambda_grid = np.linspace(
    0,
    np.pi,
    500
)

##########################################################
# SPECTRAL ESTIMATOR
##########################################################

def lag_window_estimator(
    frequencies,
    gamma_hat,
    M,
    window_function
):

    result = []

    for lam in frequencies:

        total = gamma_hat[0]

        for h in range(1, M + 1):

            weight = window_function(
                np.array([h / M])
            )[0]

            total += (
                2
                * weight
                * gamma_hat[h]
                * np.cos(h * lam)
            )

        total = total / (2 * np.pi)

        result.append(total)

    return np.array(result)

##########################################################
# COMPUTE SPECTRUM
##########################################################

fhat = lag_window_estimator(
    lambda_grid,
    gamma_hat,
    M,
    selected_window
)

##########################################################
# REMOVE NEGATIVE NUMERICAL VALUES
##########################################################

fhat = np.maximum(
    fhat,
    0
)

##########################################################
# PLOT SPECTRUM
##########################################################

st.subheader(
    "Smoothed Spectrum"
)

fig, ax = plt.subplots(
    figsize=(10,5)
)

# Overlay raw periodogram and smoothed spectrum
ax.plot(
    frequencies_lambda[:len(periodogram)],
    periodogram,
    alpha=0.4,
    label="Raw Periodogram"
)

ax.plot(
    lambda_grid,
    fhat,
    linewidth=2,
    label="Smoothed Spectrum"
)
ax.legend()

ax.set_xlabel(
    "Frequency λ"
)

ax.set_ylabel(
    "Estimated Spectral Density"
)

ax.set_title(
    "Smoothed Spectral Density"
)

ax.grid(True)

st.pyplot(fig)

##########################################################
# AUTOCOVARIANCE PANEL
##########################################################

with st.expander(
    "Sample Autocovariances"
):

    auto_df = pd.DataFrame(
        {
            "Lag": np.arange(M + 1),
            "Gamma(h)": gamma_hat
        }
    )

    st.dataframe(
        auto_df,
        use_container_width=True
    )

##########################################################
# PART 3
# CONFIDENCE BANDS
# CYCLE DETECTION
# REPORTING
##########################################################

st.header("Confidence Bands")

##########################################################
# APPROXIMATE DEGREES OF FREEDOM
##########################################################

nu = max(
    2 * M,
    4
)

##########################################################
# CONFIDENCE INTERVALS
##########################################################

lower_band = (
    nu * fhat
) / chi2.ppf(
    0.975,
    nu
)

upper_band = (
    nu * fhat
) / chi2.ppf(
    0.025,
    nu
)

##########################################################
# PLOT SPECTRUM + CI
##########################################################

fig, ax = plt.subplots(
    figsize=(10,5)
)

ax.plot(
    lambda_grid,
    fhat,
    label="Smoothed Spectrum"
)

ax.fill_between(
    lambda_grid,
    lower_band,
    upper_band,
    alpha=0.25,
    label="95% CI"
)

ax.set_xlabel("Frequency λ")
ax.set_ylabel("Spectral Density")

ax.legend()

ax.grid(True)

st.pyplot(fig)

##########################################################
# DETECT CYCLES
##########################################################

st.header("Cycle Detection")

st.markdown("""\nSearch for dominant cycles in the smoothed spectrum.\n""")
run_cycle_detection = st.button("Detect Cycles")

##########################################################
# PEAK DETECTION
##########################################################

threshold = st.slider(
    "Peak Prominence",
    min_value=0.0,
    max_value=float(np.max(fhat)),
    value=float(np.max(fhat) * 0.10)
)

peaks, properties = find_peaks(
    fhat,
    prominence=threshold
) if run_cycle_detection else (np.array([],dtype=int), {})

##########################################################
# PLOT PEAKS
##########################################################

fig, ax = plt.subplots(
    figsize=(10,5)
)

ax.plot(
    lambda_grid,
    fhat
)

if len(peaks) > 0:

    ax.scatter(
        lambda_grid[peaks],
        fhat[peaks]
    )

ax.set_title(
    "Detected Spectral Peaks"
)

ax.grid(True)

st.pyplot(fig)

##########################################################
# BUILD RESULTS TABLE
##########################################################

cycles = []

for peak in peaks:

    lam = lambda_grid[peak]

    frequency = lam / (2*np.pi)

    if frequency <= 0:
        continue

    period = 1 / frequency

    peak_height = fhat[peak]

    significant = bool(peak_height > np.percentile(fhat,90))

    cycles.append(
        [
            lam,
            frequency,
            period,
            peak_height,
            significant
        ]
    )

##########################################################
# RESULTS TABLE
##########################################################

if len(cycles) == 0:

    st.warning(
        "No dominant cycles detected."
    )

else:

    cycles_df = pd.DataFrame(
        cycles,
        columns=[
            "Lambda",
            "Frequency",
            "Period",
            "Peak Height",
            "Significant"
        ]
    )

    cycles_df = cycles_df.sort_values(
        by="Peak Height",
        ascending=False
    )

    st.session_state["frequency_suggestions"] = {
        "dominant_period": float(cycles_df.iloc[0]["Period"]),
        "dominant_frequency": float(cycles_df.iloc[0]["Frequency"]),
        "detected_cycles": int(len(cycles_df))
    }

    st.subheader(
        "Detected Cycles"
    )

    st.dataframe(
        cycles_df,
        use_container_width=True
    )

##########################################################
# TEXT REPORT
##########################################################

st.header("Cycle Report")

if len(cycles) > 0:

    dominant = cycles_df.iloc[0]

    st.success(
        f"""
        Dominant cycle detected.

        Period ≈ {dominant['Period']:.2f}

        Frequency ≈ {dominant['Frequency']:.4f}
        """
    )

    if len(cycles_df) > 1:

        st.info(
            f"""
            Secondary cycle detected.

            Period ≈
            {cycles_df.iloc[1]['Period']:.2f}
            """
        )

##########################################################
# EXPORT RESULTS
##########################################################

csv_cycles = cycles_df.to_csv(
    index=False
) if len(cycles)>0 else ""

st.download_button(
    "Download Cycle Report",
    csv_cycles,
    file_name="cycles.csv",
    mime="text/csv"
)

##########################################################
# SPECTRAL INTERPRETATION PANEL
##########################################################

with st.expander(
    "How to interpret detected cycles?"
):

    st.write(
        """
        Large peaks correspond to strong periodic
        behavior inside the time series.

        Example:

        Period = 12

        => a yearly cycle in monthly data.

        Period = 4

        => a quarterly cycle.

        Period = 7

        => a weekly cycle.
        """
    )

##########################################################
# PART 4
# PARAMETRIC SPECTRUM
# AXIS 3 CONNECTION
# FINAL REPORT
##########################################################



##########################################################
# PARAMETRIC SPECTRUM
##########################################################

def arma_spectrum(
    frequencies,
    phi,
    theta,
    sigma2
):

    spectrum = []

    for lam in frequencies:

        ##################################################
        # MA POLYNOMIAL
        ##################################################

        theta_poly = 1.0 + 0j

        for j, th in enumerate(theta):

            theta_poly += (
                th
                * np.exp(
                    -1j * (j+1) * lam
                )
            )

        ##################################################
        # AR POLYNOMIAL
        ##################################################

        phi_poly = 1.0 + 0j

        for j, ph in enumerate(phi):

            phi_poly -= (
                ph
                * np.exp(
                    -1j * (j+1) * lam
                )
            )

        ##################################################
        # SPECTRAL DENSITY
        ##################################################

        value = (
            sigma2
            /
            (2*np.pi)
        ) * (
            np.abs(theta_poly)**2
            /
            np.abs(phi_poly)**2
        )

        spectrum.append(value)

    return np.array(
        spectrum
    )


##########################################################
# FINAL SUMMARY
##########################################################

st.header("Axis 2 Summary")

summary = {}

summary["Observations"] = n

summary["Taper"] = taper_type

summary["Lag Window"] = window_type

summary["Bandwidth"] = M

summary["Detected Cycles"] = len(cycles)

summary_df = pd.DataFrame(
    summary.items(),
    columns=[
        "Metric",
        "Value"
    ]
)

st.dataframe(
    summary_df,
    use_container_width=True
)

##########################################################
# FINAL REPORT
##########################################################

st.success(
"""
Axis 2 completed successfully.

Included:

✓ Raw Periodogram

✓ Frequency / Period View

✓ Tapering

✓ Daniell Window

✓ Bartlett Window

✓ Parzen Window

✓ Smoothed Spectral Estimator

✓ Confidence Bands

✓ Cycle Detection

✓ Cycle Reporting

✓ Parametric Spectrum

✓ Spectrum Comparison

Ready for integration with Axis 1 and Axis 3.
"""
)

##########################################################
# END OF AXIS 2
##########################################################





































































# ==================================================
# AXIS 3 MERGED
# ==================================================

import streamlit as st
import pandas as pd
import numpy as np

from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX

import plotly.graph_objects as go


# ==================================================
# Page configuration
# ==================================================

# NOTE: st.set_page_config() can only be called ONCE per Streamlit app,
# as the very first Streamlit command. The original script called it
# four times (once per axis page), which crashes immediately with
# StreamlitAPIException the moment the merged script runs. The single
# call at the very top of this file (Axis 1's, now covering the whole
# app) is kept; the other three calls are removed here.


# ==================================================
# Title
# ==================================================

st.title("📊 Axis 3: The Model Architect")

st.markdown("""
## Time Domain Modeling & Identification

This module performs:

- ACF/PACF analysis
- Box-Jenkins model identification
- ARMA/ARIMA/SARIMA selection
- Parameter estimation
- Model validation preparation
""")


# ==================================================
# AXIS 2 FREQUENCY-DOMAIN SUGGESTIONS
# ==================================================

if "frequency_suggestions" in st.session_state:

    freq_info = st.session_state["frequency_suggestions"]

    dominant_period = freq_info["dominant_period"]
    dominant_frequency = freq_info["dominant_frequency"]
    detected_cycles = freq_info["detected_cycles"]

    st.subheader("🔄 Frequency-Domain Suggestions")

    st.write(
        f"Dominant Period: {dominant_period:.2f}"
    )

    st.write(
        f"Dominant Frequency: {dominant_frequency:.4f}"
    )

    st.write(
        f"Detected Cycles: {detected_cycles}"
    )

    # Seasonal suggestion
    if dominant_period >= 4:

        seasonal_period = round(dominant_period)
        st.session_state["suggested_seasonal_period"] = round(dominant_period)

        st.info(
            f"""
            Spectral peak suggests a seasonal component.

            Recommended seasonal period:
            s = {seasonal_period}

            Consider SARIMA(p,d,q)×(P,D,Q){seasonal_period}
            """
        )

    # Non-seasonal oscillation suggestion
    else:

        st.info(
            """
            Spectral peak occurs at a short period.

            Consider adding an AR(2) component
            with complex roots to model oscillatory behavior.
            """
        )


# ==================================================
# ACF / PACF FUNCTIONS
# ==================================================


def compute_acf_pacf(series, max_lags=40):

    n = len(series)

    # ACF
    acf_values = acf(
        series,
        nlags=max_lags,
        fft=True
    )

    # PACF
    pacf_values = pacf(
        series,
        nlags=max_lags,
        method="ywm"
    )

    # 95% confidence bounds
    confidence = 1.96 / np.sqrt(n)

    lags = np.arange(len(acf_values))

    return {
        "lags": lags,
        "acf": acf_values,
        "pacf": pacf_values,
        "confidence": confidence
    }


def create_acf_plot(results):

    lags = results["lags"]
    values = results["acf"]
    bound = results["confidence"]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=lags,
            y=values,
            name="ACF",
            hovertemplate=
            "Lag: %{x}<br>ACF: %{y:.4f}"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=lags,
            y=[bound]*len(lags),
            mode="lines",
            name="+1.96/sqrt(n)"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=lags,
            y=[-bound]*len(lags),
            mode="lines",
            name="-1.96/sqrt(n)"
        )
    )

    fig.update_layout(
        title="Sample Autocorrelation Function (ACF)",
        xaxis_title="Lag",
        yaxis_title="Correlation",
        height=450
    )

    return fig


def create_pacf_plot(results):

    lags = results["lags"]
    values = results["pacf"]
    bound = results["confidence"]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=lags,
            y=values,
            name="PACF",
            hovertemplate=
            "Lag: %{x}<br>PACF: %{y:.4f}"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=lags,
            y=[bound]*len(lags),
            mode="lines",
            name="+1.96/sqrt(n)"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=lags,
            y=[-bound]*len(lags),
            mode="lines",
            name="-1.96/sqrt(n)"
        )
    )

    fig.update_layout(
        title="Partial Autocorrelation Function (PACF)",
        xaxis_title="Lag",
        yaxis_title="Correlation",
        height=450
    )

    return fig

# ==================================================
# AUTOMATIC MODEL SUGGESTION
# ==================================================


def detect_significant_lags(values, bound):

    """
    Detect significant ACF/PACF lags
    outside the 95% confidence bounds
    """

    significant_lags = []

    for lag, value in enumerate(values[1:], start=1):

        if abs(value) > bound:

            significant_lags.append(lag)

    return significant_lags


def find_cutoff(sig_lags):

    if len(sig_lags) == 0:
        return 0

    cutoff = sig_lags[0]

    for i in range(1, len(sig_lags)):

        if sig_lags[i] == sig_lags[i-1] + 1:
            cutoff = sig_lags[i]

        else:
            break

    return cutoff


def suggest_model(results):

    acf_values = results["acf"]
    pacf_values = results["pacf"]
    bound = results["confidence"]

    acf_sig = detect_significant_lags(
        acf_values,
        bound
    )

    pacf_sig = detect_significant_lags(
        pacf_values,
        bound
    )

    st.write("ACF significant lags:", acf_sig)
    st.write("PACF significant lags:", pacf_sig)

    p = find_cutoff(pacf_sig)
    q = find_cutoff(acf_sig)

    # AR
    if p > 0 and q == 0:

        return {
            "model": f"AR({p})",
            "order": (p,0,0),
            "reason":
            "PACF cuts off while ACF tails off."
        }

    # MA
    elif q > 0 and p == 0:

        return {
            "model": f"MA({q})",
            "order": (0,0,q),
            "reason":
            "ACF cuts off while PACF tails off."
        }

    # ARMA
    else:

        return {
            "model": "ARMA(1,1)",
            "order": (1,0,1),
            "reason":
            "Both ACF and PACF decay."
        }


# ==================================================
# GRID SEARCH ARIMA / SARIMA
# ==================================================


def calculate_aicc(result, n):

    """
    Compute corrected Akaike Information Criterion
    """

    k = len(result.params)

    aic = result.aic

    aicc = (
        aic
        +
        (2*k*(k+1))/(n-k-1)
    )

    return aicc


def grid_search_models(
        series,
        max_p=3,
        max_d=1,
        max_q=3
):

    results = []

    n = len(series)

    # ARIMA(p,d,q)

    for p in range(max_p+1):

        for d in range(max_d+1):

            for q in range(max_q+1):

                try:

                    model = SARIMAX(
                        series,
                        order=(p,d,q),
                        enforce_stationarity=False,
                        enforce_invertibility=False
                    )

                    fitted = model.fit()

                    aicc = calculate_aicc(
                        fitted,
                        n
                    )

                    results.append({

                        "Model":
                        f"ARIMA({p},{d},{q})",

                        "AICc":
                        round(aicc,3),

                        "BIC":
                        round(fitted.bic,3),

                        "LogLikelihood":
                        round(fitted.llf,3),

                        "Parameters":
                        len(fitted.params)

                    })

                except Exception:

                    pass

    # Sort by AICc

    table = pd.DataFrame(results)

    table = table.sort_values(
        by="AICc"
    )

    return table

# ==================================================
# SARIMA GRID SEARCH
# ==================================================


def sarima_grid_search(
        series,
        seasonal_period=12,
        max_p=2,
        max_d=1,
        max_q=2,
        max_P=1,
        max_D=1,
        max_Q=1
):

    results = []

    n = len(series)

    for p in range(max_p+1):

        for d in range(max_d+1):

            for q in range(max_q+1):

                for P in range(max_P+1):

                    for D in range(max_D+1):

                        for Q in range(max_Q+1):

                            try:

                                model = SARIMAX(

                                    series,

                                    order=(
                                        p,
                                        d,
                                        q
                                    ),

                                    seasonal_order=(
                                        P,
                                        D,
                                        Q,
                                        seasonal_period
                                    ),

                                    enforce_stationarity=False,

                                    enforce_invertibility=False
                                )

                                fitted = model.fit()

                                aicc = calculate_aicc(
                                    fitted,
                                    n
                                )

                                results.append({

                                    "Model":
                                    f"SARIMA({p},{d},{q})x({P},{D},{Q})_{seasonal_period}",

                                    "AICc":
                                    round(aicc,3),

                                    "BIC":
                                    round(fitted.bic,3),

                                    "LogLikelihood":
                                    round(fitted.llf,3),

                                    "Parameters":
                                    len(fitted.params)

                                })

                            except Exception:

                                pass

    table = pd.DataFrame(results)

    if len(table)>0:

        table = table.sort_values(
            by="AICc"
        )

    return table


# ==================================================
# FINAL MODEL FITTING
# ==================================================


def fit_final_model(series, order, seasonal_order=None):

    """
    Fit the selected ARIMA/SARIMA model
    """

    if seasonal_order is None:

        model = SARIMAX(
            series,
            order=order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )

    else:

        model = SARIMAX(
            series,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )

    fitted_model = model.fit()

    return fitted_model


# ==================================================
# GLOBAL MODEL SELECTION
# ==================================================


def select_best_model():

    """
    Compare best ARIMA and best SARIMA
    using AICc
    """

    candidates = []

    # ARIMA result exists
    if "grid_results" in st.session_state:

        arima_best = st.session_state["grid_results"].iloc[0]

        candidates.append({

            "type": "ARIMA",

            "model": arima_best["Model"],

            "AICc": arima_best["AICc"]

        })

    # SARIMA result exists
    if "sarima_results" in st.session_state:

        sarima_best = st.session_state["sarima_results"].iloc[0]

        candidates.append({

            "type": "SARIMA",

            "model": sarima_best["Model"],

            "AICc": sarima_best["AICc"]

        })

    if len(candidates) == 0:

        return None

    comparison = pd.DataFrame(candidates)

    comparison = comparison.sort_values(
        by="AICc"
    )

    return comparison


# ==================================================
# Sidebar
# ==================================================

st.sidebar.header("Axis 3 Controls")


# ==============================
# SARIMA Settings
# ==============================

st.sidebar.subheader(
    "SARIMA Settings"
)

season_period = st.sidebar.number_input(
    "Seasonal period (s)",
    min_value=2,
    max_value=100,
    value=12
)

max_p = st.sidebar.number_input(
    "Maximum p",
    min_value=0,
    max_value=5,
    value=2
)

max_q = st.sidebar.number_input(
    "Maximum q",
    min_value=0,
    max_value=5,
    value=2
)

max_P = st.sidebar.number_input(
    "Maximum P",
    min_value=0,
    max_value=3,
    value=1
)

max_Q = st.sidebar.number_input(
    "Maximum Q",
    min_value=0,
    max_value=3,
    value=1
)


# ==================================================
# Data loading from Axis 1 (minimal integration)
# ==================================================

if "stationary_series" in st.session_state:

    series = pd.Series(st.session_state["stationary_series"]).dropna()
    st.session_state["series"] = series

    st.subheader("Stationary Series Preview")
    st.dataframe(series.head().to_frame(name="Series"))

    st.success("Stationary series received from Axis 1")

else:

    st.info("Run Axis 1 first to create a stationary series.")

# ==================================================
# Axis 3 Workflow
# ==================================================

st.divider()

st.subheader(
    "Model Identification Workflow"
)


if st.button("1. Compute ACF/PACF"):

    if "series" in st.session_state:

        results = compute_acf_pacf(
            st.session_state["series"]
        )

        st.session_state["acf_results"] = results

        col1, col2 = st.columns(2)

        with col1:

            st.plotly_chart(
                create_acf_plot(results),
                use_container_width=True
            )

        with col2:

            st.plotly_chart(
                create_pacf_plot(results),
                use_container_width=True
            )

    else:

        st.warning(
            "Please upload a series first"
        )


# Future steps

st.divider()

col1, col2 = st.columns(2)

with col1:

    if st.button("2. Suggest Model"):

        if "acf_results" in st.session_state:

            suggestion = suggest_model(
                st.session_state["acf_results"]
            )

            st.session_state["suggestion"] = suggestion

            st.success(
                "Model identification completed"
            )

            st.subheader(
                "Suggested Model"
            )

            st.write(
                "### " + suggestion["model"]
            )

            st.write(
                "**ARIMA order (p,d,q):**",
                suggestion["order"]
            )

            st.write(
                "**Pattern detected:**",
                suggestion["reason"]
            )

        else:

            st.warning(
                "Please compute ACF/PACF first"
            )


with col2:

    if st.button("3. Grid Search"):

        if "series" in st.session_state:

            st.info(
                "Running ARIMA grid search..."
            )

            with st.spinner(
                "Testing models..."
            ):

                table = grid_search_models(
                    st.session_state["series"],
                    max_p=3,
                    max_d=1,
                    max_q=3
                )

            st.session_state["grid_results"] = table

            st.success(
                "Grid search completed"
            )

            st.subheader(
                "Top ARIMA Models"
            )

            top_models = table.head(10).copy()

            top_models.insert(
                0,
                "Rank",
                range(1, len(top_models)+1)
            )

            st.dataframe(
                top_models,
                use_container_width=True
            )

        else:

            st.warning(
                "Please upload a series first"
            )


# ==================================================
# Spectrum Comparison (single, de-duplicated block)
# ==================================================
# NOTE: the original pasted script had this exact block duplicated
# twice in a row (once inside the "3. Grid Search" button handler,
# once again right after it at top level). Both copies used the same
# widget key f"spectrum_{idx}", which Streamlit does not allow --
# it raises a DuplicateWidgetID error the moment "grid_results" exists
# and this section tries to render a second time. Keeping a single
# top-level copy (gated on "grid_results" being present) fixes that
# crash and is functionally identical to what was intended.

if "grid_results" in st.session_state:

    table = st.session_state["grid_results"]

    st.subheader("Spectrum Comparison")

    for idx, row in table.head(10).iterrows():

        if st.button(
            f"Show Spectrum {idx}",
            key=f"spectrum_{idx}"
        ):

            st.session_state["selected_spectrum_model"] = row["Model"]

    if "selected_spectrum_model" in st.session_state:

        st.success(
            "Selected model:"
        )

        st.write(
            st.session_state["selected_spectrum_model"]
        )


# ==================================================
# SARIMA GRID SEARCH
# ==================================================

st.divider()

st.subheader(
    "Seasonal ARIMA Model Search"
)


if st.button("4. SARIMA Grid Search"):

    if "series" in st.session_state:

        st.info(
            "Running SARIMA grid search..."
        )

        with st.spinner(
            "Testing seasonal models..."
        ):

            sarima_table = sarima_grid_search(

                st.session_state["series"],

                max_p=int(max_p),

                max_q=int(max_q),

                max_P=int(max_P),

                max_Q=int(max_Q),

                seasonal_period=int(season_period)

            )

        st.session_state["sarima_results"] = sarima_table

        st.success(
            "SARIMA grid search completed"
        )

        st.subheader(
            "Top SARIMA Models"
        )

        st.dataframe(

            sarima_table.head(10),

            use_container_width=True

        )

    else:

        st.warning(
            "Please upload a series first"
        )


# ==================================================
# FINAL MODEL FITTING
# ==================================================

st.divider()

st.subheader(
    "Final Model Estimation"
)


# ==========================================
# Frequency information from Axis 2
# ==========================================

if "frequency_suggestions" in st.session_state:

    freq_info = st.session_state["frequency_suggestions"]
    st.subheader(
        "Frequency-Based Model Suggestion"
    )

    dominant_period = freq_info["dominant_period"]
    st.write(
        f"Detected dominant period: {dominant_period:.2f}"
    )

    if dominant_period >= 4:

        st.session_state["frequency_model_hint"] = "SARIMA"

        st.info(
            f"""
            Frequency analysis suggests seasonality.

            Recommended seasonal period:
            s = {round(dominant_period)}

            SARIMA models are recommended.
            """
        )

    else:

        st.session_state["frequency_model_hint"] = "ARIMA"

        st.info(
            """
            No dominant seasonal period detected.

            Model selection will still compare ARIMA and SARIMA
            using AICc/BIC criteria.
            """
        )


if st.button("5. Fit Global Best Model"):

    comparison = select_best_model()

    if comparison is not None:

        st.subheader(
            "ARIMA vs SARIMA Comparison"
        )

        st.dataframe(
            comparison,
            use_container_width=True
        )

        # Select best model (lowest AICc)

        best = comparison.iloc[0]

        # Save selected model
        st.session_state["best_model"] = best

        st.success(
            f"Selected model: {best['model']}"
        )

        st.write(
            f"Model type: {best['type']}"
        )

        st.write(
            f"Best AICc: {best['AICc']}"
        )

        # ==========================================
        # Parameter estimation
        # ==========================================

        if best["type"] == "ARIMA":

            model_name = best["model"]

            # Example:
            # ARIMA(1,1,2)

            order_text = model_name.replace(
                "ARIMA",
                ""
            )

            order_text = order_text.replace(
                "(",
                ""
            ).replace(
                ")",
                ""
            )

            p, d, q = map(
                int,
                order_text.split(",")
            )

            fitted = fit_final_model(
                st.session_state["series"],
                (p, d, q)
            )

            # ==================================================
            # FIX (was the root cause of the Axis 5 error):
            #
            # The original code only ran
            #     st.session_state["final_model"] = fitted
            # in this ARIMA branch. It never set "model_residuals",
            # "model_order", "seasonal_order", or "model_type" here,
            # even though the SARIMA branch below DOES set all of
            # them. Axis 4 immediately stops if "model_residuals" is
            # missing, so whenever the AICc grid search picked a
            # plain (non-seasonal) ARIMA model as the global best
            # model, Axis 4's "Run Diagnostics" button could never
            # set "model_validated" -- which is exactly why Axis 5
            # always showed "Model not validated. Run Axis 4 first."
            # even after Axis 4 appeared to run.
            #
            # Saving the same five keys here, symmetrically with the
            # SARIMA branch, fixes the bug.
            # ==================================================

            st.session_state["final_model"] = fitted
            st.session_state["model_order"] = (p, d, q)
            st.session_state["seasonal_order"] = (0, 0, 0, 0)
            st.session_state["model_residuals"] = fitted.resid
            st.session_state["model_type"] = best["type"]



            # ==================================================
            # SEND MODEL TO SPECTRAL COMPARISON
            # ==================================================

            if hasattr(fitted, "arparams"):
                st.session_state["axis3_phi"] = fitted.arparams
            else:
                st.session_state["axis3_phi"] = []

            if hasattr(fitted, "maparams"):
                st.session_state["axis3_theta"] = fitted.maparams
            else:
                st.session_state["axis3_theta"] = []

            st.session_state["axis3_sigma2"] = fitted.scale






            st.subheader(
                "Estimated Parameters"
            )

            st.dataframe(
                fitted.params
            )

            st.subheader(
                "Model Summary"
            )

            st.text(
                fitted.summary()
            )

            # Residual preview (matches the SARIMA branch below,
            # so both paths give the user the same feedback)

            st.subheader(
                "Residual Analysis Preview"
            )

            residuals = fitted.resid

            fig = go.Figure()

            fig.add_trace(
                go.Scatter(
                    y=residuals,
                    mode="lines",
                    name="Residuals"
                )
            )

            fig.update_layout(
                title="Model Residuals",
                xaxis_title="Time",
                yaxis_title="Residual"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        else:

            model_name = best["model"]

            # Example:
            # SARIMA(1,1,1)x(1,0,1)_12

            sarima_text = model_name.replace(
                "SARIMA",
                ""
            )

            sarima_text = sarima_text.replace(
                "x",
                ","
            )

            sarima_text = sarima_text.replace(
                "_",
                ","
            )

            sarima_text = sarima_text.replace(
                "(",
                ""
            ).replace(
                ")",
                ""
            )

            values = list(
                map(
                    int,
                    sarima_text.split(",")
                )
            )

            p, d, q, P, D, Q, s = values

            fitted = SARIMAX(

                st.session_state["series"],

                order=(
                    p,
                    d,
                    q
                ),

                seasonal_order=(
                    P,
                    D,
                    Q,
                    s
                ),

                enforce_stationarity=False,

                enforce_invertibility=False

            ).fit()

            # Save fitted model for Axis 4

            st.session_state["final_model"] = fitted
            st.session_state["model_order"] = (p, d, q)
            st.session_state["seasonal_order"] = (P, D, Q, s)
            st.session_state["model_residuals"] = fitted.resid
            st.session_state["model_type"] = best["type"]

            st.subheader(
                "Estimated Parameters"
            )

            st.dataframe(
                fitted.params
            )

            st.subheader(
                "Model Summary"
            )

            st.text(
                fitted.summary()
            )

            # ==========================================
            # Residual Preview
            # ==========================================

            st.subheader(
                "Residual Analysis Preview"
            )

            residuals = fitted.resid

            fig = go.Figure()

            fig.add_trace(
                go.Scatter(
                    y=residuals,
                    mode="lines",
                    name="Residuals"
                )
            )

            fig.update_layout(
                title="Model Residuals",
                xaxis_title="Time",
                yaxis_title="Residual"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    else:

        st.warning(
            "Run ARIMA or SARIMA Grid Search first"
        )













st.header("Parametric Spectral Analysis")

st.info(
"""
This section receives the fitted ARMA model
from Axis 3.
"""
)




# ==================================================
# PARAMETRIC SPECTRUM COMPARISON
# ==================================================

st.divider()

st.subheader(
    "Parametric Spectrum Comparison"
)

if (
    "axis3_phi" in st.session_state
    and "axis3_theta" in st.session_state
    and "axis3_sigma2" in st.session_state
):

    phi = st.session_state["axis3_phi"]

    theta = st.session_state["axis3_theta"]

    sigma2 = st.session_state["axis3_sigma2"]

    parametric_spectrum = arma_spectrum(
        lambda_grid,
        phi,
        theta,
        sigma2
    )

    fig, ax = plt.subplots(
        figsize=(10,5)
    )

    ax.plot(
        lambda_grid,
        fhat,
        linewidth=2,
        label="Nonparametric Spectrum"
    )

    ax.plot(
        lambda_grid,
        parametric_spectrum,
        linewidth=2,
        label="ARMA Spectrum"
    )

    ax.set_title(
        "Parametric vs Nonparametric Spectrum"
    )

    ax.set_xlabel(
        "Frequency λ"
    )

    ax.set_ylabel(
        "Spectral Density"
    )

    ax.legend()

    ax.grid(True)

    st.pyplot(fig)

else:

    st.info(
        "Fit a model first to compare spectra."
    )



  

   
























































# ==================================================
# AXIS 4: THE MODEL AUDITOR (SIMPLIFIED)
# Residual Diagnostics & Validation
# ==================================================

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2, jarque_bera, norm
from scipy.fft import fft, fftfreq
from statsmodels.tsa.stattools import acf, pacf
from scipy import stats
from scipy.stats import gaussian_kde

# (st.set_page_config call removed -- see note above Axis 3 import block)
st.title("🔍 Axis 4: The Model Auditor — Residual Diagnostics")

# ==================================================
# CHECK PREREQUISITES
# ==================================================

if "final_model" not in st.session_state:
    st.error("❌ No fitted model found. Please run Axis 3 first.")
    st.stop()

if "model_residuals" not in st.session_state:
    st.error("❌ No residuals found. Please fit a model in Axis 3 first.")
    st.stop()

# ==================================================
# EXTRACT DATA
# ==================================================

residuals = st.session_state["model_residuals"].dropna()
n = len(residuals)

# Get number of model parameters (p + q)
model_params = len(st.session_state["final_model"].params) if hasattr(st.session_state["final_model"], 'params') else 1

# Standardize residuals
standardized_residuals = residuals / np.std(residuals) if np.std(residuals) > 0 else residuals

st.success(f"✅ Loaded {n} residuals for validation")

# ==================================================
# SIDEBAR SETTINGS
# ==================================================

H = st.sidebar.slider(
    "Maximum Lags (H) for Ljung-Box",
    min_value=5,
    max_value=min(50, n // 2),
    value=min(20, n // 2)
)

# ==================================================
# HELPER FUNCTIONS
# ==================================================

def ljung_box_test(residuals, H, k_params):
    """Ljung-Box Test: Q_LB = n(n+2) * Σ(ρ̂_k² / (n-k))"""
    n = len(residuals)
    acf_vals = acf(residuals, nlags=H, fft=True)

    p_values = []
    for h in range(1, H + 1):
        rho_sq = sum(acf_vals[1:h+1]**2 / (n - np.arange(1, h+1)))
        q = n * (n + 2) * rho_sq
        df = max(1, h - k_params)
        p_val = 1 - chi2.cdf(q, df)
        p_values.append(p_val)

    return np.array(p_values)

def cumulative_periodogram_test(residuals):
    """Cumulative Periodogram with KS bounds"""
    n = len(residuals)
    x = residuals - np.mean(residuals)

    fft_vals = fft(x)
    power = np.abs(fft_vals[:n//2])**2 / (2 * np.pi * n)
    total_power = np.sum(power)

    cum_power = np.cumsum(power) / total_power if total_power > 0 else np.cumsum(power)
    freqs = fftfreq(n)[:n//2]

    q = len(freqs)
    K_005 = 1.358
    ks_bound = K_005 / np.sqrt(q)
    ref_line = 2 * freqs

    within_bounds = np.all(np.abs(cum_power - ref_line) <= ks_bound)

    return {
        'frequencies': freqs,
        'cum_periodogram': cum_power,
        'ref_line': ref_line,
        'ks_bound': ks_bound,
        'passed': within_bounds,
        'q': q
    }

# ==================================================
# RUN DIAGNOSTICS
# ==================================================

if st.button(" Run Diagnostics", type="primary", use_container_width=True):

    st.markdown("---")
    st.subheader(" Diagnostic Results")

    # ==================================================
    # 1. LJUNG-BOX TEST (TIME DOMAIN)
    # ==================================================

    st.markdown("###  Time Domain White Noise Tests")

    # Compute Ljung-Box
    p_values = ljung_box_test(standardized_residuals, H, model_params)
    lb_passed = np.all(p_values >= 0.05)

    # Plot p-values
    fig, ax = plt.subplots(figsize=(10, 4))
    colors = ['green' if p >= 0.05 else 'red' for p in p_values]
    ax.bar(range(1, len(p_values)+1), p_values, color=colors, alpha=0.7)
    ax.axhline(y=0.05, color='red', linestyle='--', linewidth=2, label='5% Significance Level')
    ax.set_xlabel("Lag (h)")
    ax.set_ylabel("p-value")
    ax.set_title(f"Ljung-Box Test p-values (H = {H})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    st.pyplot(fig)
    plt.close()

    # Display results
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Ljung-Box Test", "✅ PASS" if lb_passed else "❌ FAIL")
        st.metric("Min p-value", f"{np.min(p_values):.4f}")
    with col2:
        if not lb_passed:
            st.warning(f"⚠️ p-values below 0.05 at lags: {np.where(p_values < 0.05)[0] + 1}")
        st.caption(f"Under H0: Q_LB ~ χ²(h - {model_params})")

    # ==================================================
    # 2. JARQUE-BERA TEST (TIME DOMAIN)
    # ==================================================

    st.markdown("#### Jarque-Bera Test for Normality")

    jb_stat, jb_pvalue = jarque_bera(standardized_residuals)
    jb_passed = jb_pvalue >= 0.05

    S = stats.skew(standardized_residuals)
    K = stats.kurtosis(standardized_residuals)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("JB Statistic", f"{jb_stat:.4f}")
    with col2:
        st.metric("p-value", f"{jb_pvalue:.4f}")
    with col3:
        st.metric("Skewness", f"{S:.4f}")
    with col4:
        st.metric("Kurtosis", f"{K:.4f}")

    st.caption("JB = (n/6)[S² + (K-3)²/4] ~ χ²₂")

    # Histogram
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(standardized_residuals, bins=30, density=True, alpha=0.6, color='blue', edgecolor='black')

    kde = gaussian_kde(standardized_residuals)
    x_range = np.linspace(-4, 4, 200)
    ax.plot(x_range, kde(x_range), 'b-', linewidth=2, label='KDE')
    ax.plot(x_range, norm.pdf(x_range), 'r--', linewidth=2, label='N(0,1)')

    ax.set_xlabel("Value")
    ax.set_ylabel("Density")
    ax.set_title("Histogram with KDE and N(0,1)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    plt.close()

    # ==================================================
    # 3. FREQUENCY DOMAIN TESTS
    # ==================================================

    st.markdown("###  Frequency Domain White Noise Tests")

    # Cumulative Periodogram
    cp_result = cumulative_periodogram_test(standardized_residuals)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(cp_result['frequencies'], cp_result['cum_periodogram'], 'b-', linewidth=2, label='C(ω)')
    ax.plot(cp_result['frequencies'], cp_result['ref_line'], 'k--', alpha=0.7, label='Reference Line')
    ax.fill_between(cp_result['frequencies'],
                    cp_result['ref_line'] - cp_result['ks_bound'],
                    cp_result['ref_line'] + cp_result['ks_bound'],
                    alpha=0.2, color='red', label='95% KS Bounds')
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Cumulative Periodogram")
    ax.set_title(f"Cumulative Periodogram Test (q = {cp_result['q']})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 0.5)
    ax.set_ylim(0, 1)
    st.pyplot(fig)
    plt.close()

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Cumulative Periodogram", "✅ PASS" if cp_result['passed'] else "❌ FAIL")
    with col2:
        st.caption(f"KS bound = 1.358/√{cp_result['q']} = {cp_result['ks_bound']:.4f}")

    # ==================================================
    # 4. DIAGNOSTIC PLOT GRID (2 × 3)
    # ==================================================

    st.markdown("### Diagnostic Plot Grid")

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 1. Residuals over time
    axes[0, 0].plot(standardized_residuals, 'b-', linewidth=1)
    axes[0, 0].axhline(y=0, color='red', linestyle='--')
    axes[0, 0].axhline(y=2, color='red', linestyle=':', alpha=0.5)
    axes[0, 0].axhline(y=-2, color='red', linestyle=':', alpha=0.5)
    axes[0, 0].set_title("Standardized Residuals")
    axes[0, 0].grid(True, alpha=0.3)

    # 2. ACF
    max_lag = min(40, n // 4)
    acf_vals = acf(standardized_residuals, nlags=max_lag, fft=True)
    bound = 1.96 / np.sqrt(n)
    axes[0, 1].bar(range(len(acf_vals)), acf_vals, alpha=0.7, color='blue')
    axes[0, 1].axhline(y=bound, color='red', linestyle='--', label='±1.96/√n')
    axes[0, 1].axhline(y=-bound, color='red', linestyle='--')
    axes[0, 1].axhline(y=0, color='black', linewidth=0.5)
    axes[0, 1].set_title("ACF of Residuals")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 3. PACF
    pacf_vals = pacf(standardized_residuals, nlags=max_lag, method="ywm")
    axes[0, 2].bar(range(len(pacf_vals)), pacf_vals, alpha=0.7, color='green')
    axes[0, 2].axhline(y=bound, color='red', linestyle='--', label='±1.96/√n')
    axes[0, 2].axhline(y=-bound, color='red', linestyle='--')
    axes[0, 2].axhline(y=0, color='black', linewidth=0.5)
    axes[0, 2].set_title("PACF of Residuals")
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    # 4. QQ-Plot
    stats.probplot(standardized_residuals, dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title("QQ-Plot")
    axes[1, 0].grid(True, alpha=0.3)

    # 5. Histogram
    axes[1, 1].hist(standardized_residuals, bins=30, density=True, alpha=0.6, color='blue', edgecolor='black')
    axes[1, 1].plot(x_range, norm.pdf(x_range), 'r--', linewidth=2, label='N(0,1)')
    kde = gaussian_kde(standardized_residuals)
    axes[1, 1].plot(x_range, kde(x_range), 'b-', linewidth=2, label='KDE')
    axes[1, 1].set_title("Histogram")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # 6. Cumulative Periodogram
    axes[1, 2].plot(cp_result['frequencies'], cp_result['cum_periodogram'], 'b-', linewidth=2)
    axes[1, 2].plot(cp_result['frequencies'], cp_result['ref_line'], 'k--', alpha=0.5)
    axes[1, 2].fill_between(cp_result['frequencies'],
                            cp_result['ref_line'] - cp_result['ks_bound'],
                            cp_result['ref_line'] + cp_result['ks_bound'],
                            alpha=0.2, color='red')
    axes[1, 2].set_title("Cumulative Periodogram")
    axes[1, 2].set_xlim(0, 0.5)
    axes[1, 2].set_ylim(0, 1)
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # ==================================================
    # 5. FINAL VERDICT
    # ==================================================

    st.markdown("---")
    st.subheader("⚖️ Final Validation Verdict")

    # Check ACF/PACF bounds
    acf_exceeds = np.any(np.abs(acf_vals[1:]) > bound)
    pacf_exceeds = np.any(np.abs(pacf_vals[1:]) > bound)

    tests_passed = {
        "Ljung-Box Test": lb_passed,
        "Jarque-Bera Test": jb_passed,
        "Cum. Periodogram Test": cp_result['passed'],
        "ACF (within bounds)": not acf_exceeds,
        "PACF (within bounds)": not pacf_exceeds
    }

    all_passed = all(tests_passed.values())

    # ==================================================
    # FINAL VERDICT BOX
    # ==================================================

    st.markdown("### ===== MODEL VALIDATION SUMMARY =====")

    for test, passed in tests_passed.items():
        if test == "Ljung-Box Test":
            st.markdown(f"- **{test} (h={H})**: p-value = {np.min(p_values):.4f} [{'PASS' if passed else 'FAIL'}]")
        elif test == "Jarque-Bera Test":
            st.markdown(f"- **{test}**: p-value = {jb_pvalue:.4f} [{'PASS' if passed else 'FAIL'}]")
        else:
            st.markdown(f"- **{test}**: [{'PASS' if passed else 'FAIL'}]")

    st.markdown("---")

    if all_passed:
        st.success("""
        ✅ **VERDICT: MODEL IS ADEQUATE**

        Residuals are consistent with Gaussian White Noise.
        **Proceed to Forecasting (Axis 5).**
        """)
        st.session_state["model_validated"] = True
        st.balloons()
    else:
        failed_tests = [test for test, passed in tests_passed.items() if not passed]
        st.error(f"""
        ❌ **VERDICT: MODEL IS INADEQUATE**

        Failed: {', '.join(failed_tests)}
        **Return to Axis 3 and consider alternative specifications.**
        """)
        st.session_state["model_validated"] = False

else:
    st.info(" Click 'Run Diagnostics' to validate the fitted model.")

# ==================================================
# MODEL INFO
# ==================================================

with st.expander("📋 Model Information"):
    st.write("**Model Type:**", st.session_state.get("model_type", "Unknown"))
    st.write("**Order:**", st.session_state.get("model_order", "Unknown"))
    st.write("**Parameters:**", model_params)
    st.write("**Residuals:**", n)
    st.write("**Validation:**", "✅ Validated" if st.session_state.get("model_validated", False) else "❌ Not Validated")





































































# ==================================================
# AXIS 5: FORECASTING & UNCERTAINTY QUANTIFICATION
# Based on Brockwell & Davis Chapter 5
# ==================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import io

# (st.set_page_config call removed -- see note above Axis 3 import block)
st.title(" Axis 5: The Probabilist — Forecasting")

# ==================================================
# DEBUG MODE (sidebar toggle) -- NEW
# ==================================================
# Lets you bypass the Axis 4 gate while testing Axis 5 in isolation,
# without re-running the whole pipeline (Axis 1 -> 2 -> 3 -> 4) every
# single time you tweak something here. Forecasts produced this way
# have NOT actually passed validation, so treat them as provisional.

debug_mode = st.sidebar.checkbox(
    "🐞 Debug mode (skip Axis 4 validation gate)",
    value=False,
    help="For development only. Bypasses the model-validated check so "
         "you can test the forecasting engine without re-running "
         "Axis 3/4 first. Forecasts are NOT guaranteed valid in this mode."
)

with st.sidebar.expander("🔍 Session State Inspector"):
    st.write("**Keys currently in session_state:**")
    st.write(list(st.session_state.keys()))
    st.write("**final_model present:**", "final_model" in st.session_state)
    st.write("**model_residuals present:**", "model_residuals" in st.session_state)
    st.write("**series present:**", "series" in st.session_state)
    st.write(
        "**model_validated value:**",
        st.session_state.get("model_validated", "<<not set>>")
    )

# ==================================================
# PREREQUISITE CHECKS
# ==================================================

if "final_model" not in st.session_state:
    st.error("❌ No model found. Run Axis 3 first.")
    st.stop()

if "series" not in st.session_state:
    st.error("❌ No series found. Run Axis 1 first.")
    st.stop()

if not debug_mode and not st.session_state.get("model_validated", False):
    st.error(
        "❌ Model not validated. Run Axis 4 first, "
        "or enable debug mode in the sidebar."
    )

    # ---- Helpful diagnostics instead of a dead end -- NEW ----
    if "final_model" in st.session_state and "model_residuals" not in st.session_state:
        st.warning(
            "A fitted model exists, but `model_residuals` was never stored. "
            "This used to happen whenever Axis 3's grid search picked a "
            "plain ARIMA model (the ARIMA branch did not save residuals); "
            "that has been fixed in this version of Axis 3, so if you still "
            "see this, re-run step 5 ('Fit Global Best Model') in Axis 3."
        )
    if "model_residuals" in st.session_state and "model_validated" not in st.session_state:
        st.warning(
            "Residuals exist, but `model_validated` was never written. "
            "Streamlit reruns the whole script on every interaction, and "
            "the flag is only set at the exact moment the **'Run "
            "Diagnostics'** button in Axis 4 is clicked. Go back to Axis 4 "
            "and click that button."
        )
    if st.session_state.get("model_validated") is False:
        st.warning(
            "`model_validated` is explicitly `False` — Axis 4 ran, but at "
            "least one diagnostic test failed. Check the Final Verdict box "
            "in Axis 4 to see which test(s) failed."
        )

    st.stop()

if debug_mode and not st.session_state.get("model_validated", False):
    st.warning(
        "🐞 Debug mode is ON — proceeding without a passed Axis 4 "
        "validation. Treat the forecasts below as provisional."
    )

# ==================================================
# EXTRACT DATA
# ==================================================

series = st.session_state["series"]
final_model = st.session_state["final_model"]
model_order = st.session_state.get("model_order", (0, 0, 0))
seasonal_order = st.session_state.get("seasonal_order", (0, 0, 0, 0))

# Transformation metadata from Axis 1
transform_meta = st.session_state.get("transformation_metadata", {
    "boxcox_lambda": None,
    "differencing_d": 0,
    "differencing_D": 0,
    "seasonal_period_s": None,
    "original_series": series.values.tolist()
})

st.success(
    " Model validated and ready for forecasting!"
    if not debug_mode else
    " Proceeding in debug mode (validation bypassed)."
)

# ==================================================
# HELPER FUNCTIONS
# ==================================================

def get_arma_coefficients(model):
    """Extract AR, MA coefficients and sigma2 from fitted model"""
    phi = list(model.arparams) if hasattr(model, 'arparams') else []
    theta = list(model.maparams) if hasattr(model, 'maparams') else []
    sigma2 = model.sigma2 if hasattr(model, 'sigma2') else 1.0
    return phi, theta, sigma2

def back_transform(forecast, lower_50, upper_50, lower_80, upper_80, lower_95, upper_95, meta):
    """Back-transform forecasts to original scale"""
    f, l50, u50, l80, u80, l95, u95 = forecast, lower_50, upper_50, lower_80, upper_80, lower_95, upper_95

    # Reverse Box-Cox
    lam = meta.get("boxcox_lambda")
    if lam is not None:
        if lam == 0:
            f, l50, u50, l80, u80, l95, u95 = np.exp(f), np.exp(l50), np.exp(u50), np.exp(l80), np.exp(u80), np.exp(l95), np.exp(u95)
        else:
            f = (f * lam + 1) ** (1/lam)
            l50 = (l50 * lam + 1) ** (1/lam)
            u50 = (u50 * lam + 1) ** (1/lam)
            l80 = (l80 * lam + 1) ** (1/lam)
            u80 = (u80 * lam + 1) ** (1/lam)
            l95 = (l95 * lam + 1) ** (1/lam)
            u95 = (u95 * lam + 1) ** (1/lam)

    # Reverse differencing
    d = meta.get("differencing_d", 0)
    if d > 0:
        orig = np.array(meta.get("original_series", series.values))
        last_vals = orig[-d:] if len(orig) >= d else orig
        f = np.concatenate([last_vals, f])
        l50 = np.concatenate([last_vals, l50])
        u50 = np.concatenate([last_vals, u50])
        l80 = np.concatenate([last_vals, l80])
        u80 = np.concatenate([last_vals, u80])
        l95 = np.concatenate([last_vals, l95])
        u95 = np.concatenate([last_vals, u95])
        f, l50, u50, l80, u80, l95, u95 = np.cumsum(f)[d:], np.cumsum(l50)[d:], np.cumsum(u50)[d:], np.cumsum(l80)[d:], np.cumsum(u80)[d:], np.cumsum(l95)[d:], np.cumsum(u95)[d:]

    return {"forecast": f, "lower_50": l50, "upper_50": u50, "lower_80": l80, "upper_80": u80, "lower_95": l95, "upper_95": u95}

# ==================================================
# FORECAST SETTINGS
# ==================================================

col1, col2 = st.columns(2)
with col1:
    h = st.slider("Forecast Horizon (h)", 1, 50, 12)
with col2:
    show_historical = st.slider("Historical Points", 5, min(100, len(series)), min(30, len(series)))

# ==================================================
# GENERATE FORECAST
# ==================================================

if st.button(" Generate Forecast", type="primary", use_container_width=True):

    phi, theta, sigma2 = get_arma_coefficients(final_model)

    with st.spinner("Generating forecasts..."):
        try:
            # Use statsmodels forecasting
            forecast_result = final_model.get_forecast(steps=h)
            mean = forecast_result.predicted_mean.values

            # Prediction intervals
            ci_50 = forecast_result.conf_int(alpha=0.50).values
            ci_80 = forecast_result.conf_int(alpha=0.20).values
            ci_95 = forecast_result.conf_int(alpha=0.05).values

            # Back-transform
            bt = back_transform(
                mean, ci_50[:,0], ci_50[:,1],
                ci_80[:,0], ci_80[:,1],
                ci_95[:,0], ci_95[:,1],
                transform_meta
            )

        except Exception as e:
            st.warning(f"Statsmodels forecast failed ({e}); using fallback method.")
            # Fallback: simple forecast
            last_val = series.iloc[-1]
            mean = np.array([last_val] * h)
            std_resid = np.std(st.session_state.get("model_residuals", [0]))
            std_forecast = std_resid * np.sqrt(np.arange(1, h+1))

            bt = {
                "forecast": mean,
                "lower_50": mean - 0.674 * std_forecast,
                "upper_50": mean + 0.674 * std_forecast,
                "lower_80": mean - 1.282 * std_forecast,
                "upper_80": mean + 1.282 * std_forecast,
                "lower_95": mean - 1.96 * std_forecast,
                "upper_95": mean + 1.96 * std_forecast,
            }

    st.session_state["forecast_result"] = bt

    # ==================================================
    # FAN CHART
    # ==================================================

    st.subheader(" Fan Chart")

    fig, ax = plt.subplots(figsize=(14, 5))

    # Historical
    hist_series = series.iloc[-show_historical:]
    hist_times = range(len(series) - show_historical, len(series))
    ax.plot(hist_times, hist_series, 'b-', lw=2, label='Historical')

    # Forecast
    future_times = range(len(series), len(series) + h)
    ax.plot(future_times, bt["forecast"], 'r-', lw=2.5, label='Forecast')

    # Prediction intervals (cascading)
    ax.fill_between(future_times, bt["lower_95"], bt["upper_95"], color='red', alpha=0.08, label='95% PI')
    ax.fill_between(future_times, bt["lower_80"], bt["upper_80"], color='red', alpha=0.15, label='80% PI')
    ax.fill_between(future_times, bt["lower_50"], bt["upper_50"], color='red', alpha=0.30, label='50% PI')

    ax.axvline(x=len(series)-0.5, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.set_title(f"{h}-Step Ahead Forecast")
    ax.legend()
    ax.grid(True, alpha=0.3)

    st.pyplot(fig)
    plt.close()

    # ==================================================
    # FORECAST TABLE
    # ==================================================

    st.subheader("📋 Forecast Table")

    table = pd.DataFrame({
        "Step": range(1, h+1),
        "Forecast": bt["forecast"],
        "50% PI": [f"[{bt['lower_50'][i]:.2f}, {bt['upper_50'][i]:.2f}]" for i in range(h)],
        "80% PI": [f"[{bt['lower_80'][i]:.2f}, {bt['upper_80'][i]:.2f}]" for i in range(h)],
        "95% PI": [f"[{bt['lower_95'][i]:.2f}, {bt['upper_95'][i]:.2f}]" for i in range(h)],
    })
    st.dataframe(table, use_container_width=True)

    # ==================================================
    # EXPORT
    # ==================================================

    csv = table.to_csv(index=False)
    st.download_button(" Download CSV", csv, f"forecast_h{h}.csv", "text/csv")

    st.success(" Forecast complete!")

else:
    st.info(" Configure settings and click 'Generate Forecast'")

# ==================================================
# MODEL INFO
# ==================================================

with st.expander("📋 Model Information"):
    st.write("**Model:**", st.session_state.get("model_type", "Unknown"))
    st.write("**Order:**", model_order)
    if any(seasonal_order):
        st.write("**Seasonal Order:**", seasonal_order)
    st.write("**Validation:**", "Validated" if st.session_state.get("model_validated") else " Not Validated")
