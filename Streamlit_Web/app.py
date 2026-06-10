import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Ecotoxicity Analysis System",
    layout="wide"
)

# ================= SESSION STORAGE =================
if "data_store" not in st.session_state:
    st.session_state.data_store = {}

if "input_df" not in st.session_state:
    st.session_state.input_df = pd.DataFrame()


# ================= FUNCTIONS =================
def abbott_corrected(treatment_decimal, control_decimal):
    """
    Abbott correction using mortality values in decimal form.
    Example: 20% mortality = 0.20.
    Returns corrected mortality in decimal form from 0 to 1.
    """
    if control_decimal >= 1:
        return 0

    corrected = (treatment_decimal - control_decimal) / (1 - control_decimal)
    return max(0, min(corrected, 1))


def sigmoid_model_percent(x, k, lc50):
    """
    Simple logistic sigmoid concentration-response model.

    y = 100 / [1 + e^(-k(x - LC50))]

    y = corrected mortality (%)
    x = sample concentration (%)
    k = slope of the curve
    LC50 = concentration causing 50% mortality
    """
    return 100 / (1 + np.exp(-k * (x - lc50)))


def toxicity_level(lc50):
    if pd.isna(lc50):
        return "Not available"

    if lc50 < 1:
        return "Extremely toxic"
    elif lc50 < 10:
        return "Highly toxic"
    elif lc50 < 50:
        return "Moderately toxic"
    else:
        return "Low toxicity"


def save_data(sample_name, df):
    sample_name = sample_name.strip()

    if sample_name in st.session_state.data_store:
        old_df = st.session_state.data_store[sample_name]

        combined = pd.concat([old_df, df], ignore_index=True)
        combined = combined.drop_duplicates(
            subset=["Sample", "Concentration", "Time", "Replicate"],
            keep="last"
        )
    else:
        combined = df.copy()

    st.session_state.data_store[sample_name] = combined


def read_data(sample_name):
    sample_name = sample_name.strip()

    if sample_name in st.session_state.data_store:
        return st.session_state.data_store[sample_name].copy()

    return pd.DataFrame()


def get_all_data():
    dfs = []

    for sample_name, df in st.session_state.data_store.items():
        if not df.empty:
            dfs.append(df)

    if dfs:
        return pd.concat(dfs, ignore_index=True)

    return pd.DataFrame()


def get_sample_names(include_control=True):
    sample_names = list(st.session_state.data_store.keys())

    if include_control:
        return sample_names

    return [
        name for name in sample_names
        if name.lower() != "control"
    ]


def get_control_data():
    for sample_name, df in st.session_state.data_store.items():
        if sample_name.lower() == "control":
            return df.copy()

    return pd.DataFrame()


def get_control_mortality(control_df, time, replicate):
    """
    Get control mortality for Abbott correction.

    Priority:
    1. Same time and same replicate
    2. Average control mortality at the same time
    3. If unavailable, control mortality = 0
    """
    if control_df.empty:
        return 0

    control_df = control_df.copy()

    control_df["Time"] = pd.to_numeric(control_df["Time"], errors="coerce")
    control_df["Replicate"] = pd.to_numeric(control_df["Replicate"], errors="coerce")

    if "Mortality Decimal" not in control_df.columns:
        if "Mortality %" in control_df.columns:
            control_df["Mortality Decimal"] = pd.to_numeric(
                control_df["Mortality %"],
                errors="coerce"
            ) / 100

        elif "Dead" in control_df.columns and "Initial" in control_df.columns:
            control_df["Dead"] = pd.to_numeric(control_df["Dead"], errors="coerce")
            control_df["Initial"] = pd.to_numeric(control_df["Initial"], errors="coerce")
            control_df["Mortality Decimal"] = control_df["Dead"] / control_df["Initial"]

        else:
            return 0

    control_df["Mortality Decimal"] = pd.to_numeric(
        control_df["Mortality Decimal"],
        errors="coerce"
    )

    same_rep = control_df[
        (control_df["Time"] == time) &
        (control_df["Replicate"] == replicate)
    ]

    if not same_rep.empty:
        return float(same_rep["Mortality Decimal"].iloc[0])

    same_time = control_df[control_df["Time"] == time]

    if not same_time.empty:
        return float(same_time["Mortality Decimal"].mean())

    return 0


def fit_sigmoid_lc50(df):
    """
    Fit simple logistic sigmoid model using all replicate corrected mortality values.
    """
    plot_df = df.copy()
    plot_df = plot_df[plot_df["Concentration"] > 0]

    plot_df["Concentration"] = pd.to_numeric(
        plot_df["Concentration"],
        errors="coerce"
    )

    plot_df["Corrected Mortality %"] = pd.to_numeric(
        plot_df["Corrected Mortality %"],
        errors="coerce"
    )

    plot_df = plot_df.dropna(
        subset=["Concentration", "Corrected Mortality %"]
    )

    if len(plot_df) < 3:
        raise ValueError("At least three data points are required for sigmoid fitting.")

    x = plot_df["Concentration"].astype(float).values
    y = plot_df["Corrected Mortality %"].astype(float).values

    p0 = [0.15, np.median(x)]

    upper_lc50_bound = max(100.0, float(np.max(x)) * 5)

    params, _ = curve_fit(
        sigmoid_model_percent,
        x,
        y,
        p0=p0,
        bounds=([0.0001, 0.01], [5.0, upper_lc50_bound]),
        maxfev=20000
    )

    k, lc50 = params
    y_pred = sigmoid_model_percent(x, k, lc50)

    ss_res = np.sum((y - y_pred) ** 2)
    ss_total = np.sum((y - np.mean(y)) ** 2)

    r2 = 1 - (ss_res / ss_total) if ss_total != 0 else 0

    return plot_df, k, lc50, r2


def manual_linear_interpolation_lc50(df):
    """
    Manual LC50 validation using linear interpolation.
    Uses the mean corrected mortality at each concentration.
    LC50 is estimated between the two concentration levels surrounding 50% mortality.
    """
    temp = df.copy()
    temp = temp[temp["Concentration"] > 0]

    temp["Concentration"] = pd.to_numeric(
        temp["Concentration"],
        errors="coerce"
    )

    temp["Corrected Mortality %"] = pd.to_numeric(
        temp["Corrected Mortality %"],
        errors="coerce"
    )

    temp = temp.dropna(
        subset=["Concentration", "Corrected Mortality %"]
    )

    mean_df = (
        temp.groupby("Concentration", as_index=False)["Corrected Mortality %"]
        .mean()
        .sort_values("Concentration")
    )

    if len(mean_df) < 2:
        return np.nan, mean_df, "Not enough concentration levels."

    for i in range(len(mean_df) - 1):
        c1 = float(mean_df.iloc[i]["Concentration"])
        y1 = float(mean_df.iloc[i]["Corrected Mortality %"])

        c2 = float(mean_df.iloc[i + 1]["Concentration"])
        y2 = float(mean_df.iloc[i + 1]["Corrected Mortality %"])

        if (y1 <= 50 <= y2) or (y2 <= 50 <= y1):
            if y2 == y1:
                return np.nan, mean_df, "Cannot interpolate because both mortality values are equal."

            lc50 = c1 + ((50 - y1) / (y2 - y1)) * (c2 - c1)
            return lc50, mean_df, "OK"

    return np.nan, mean_df, "50% mortality is outside the observed concentration range."


def create_sigmoid_figure(sample_name, time, plot_df, k, lc50, r2):
    """
    Create a concentration-response sigmoid graph with:
    - all replicate points
    - mean ± SD error bars
    - fitted concentration-response sigmoid curve
    - LC50 line
    - 50% corrected mortality line
    """
    x = plot_df["Concentration"].astype(float).values
    y = plot_df["Corrected Mortality %"].astype(float).values

    summary_df = (
        plot_df.groupby("Concentration")["Corrected Mortality %"]
        .agg(["mean", "std"])
        .reset_index()
        .sort_values("Concentration")
    )

    summary_df["std"] = summary_df["std"].fillna(0)

    fig, ax = plt.subplots(figsize=(10, 5.8))

    x_jitter = np.zeros_like(x, dtype=float)

    for conc in sorted(np.unique(x)):
        idx = np.where(x == conc)[0]
        offsets = np.linspace(-0.8, 0.8, len(idx)) if len(idx) > 1 else np.array([0.0])
        x_jitter[idx] = offsets

    ax.scatter(
        x + x_jitter,
        y,
        s=70,
        alpha=0.85,
        label="Replicate data points"
    )

    ax.errorbar(
        summary_df["Concentration"],
        summary_df["mean"],
        yerr=summary_df["std"],
        fmt="s",
        capsize=5,
        markersize=7,
        linewidth=1.8,
        label="Mean ± SD"
    )

    x_min = max(0.1, min(x) - 5)
    x_max = max(x) + 5

    x_line = np.linspace(x_min, x_max, 400)
    y_line = sigmoid_model_percent(x_line, k, lc50)

    ax.plot(
        x_line,
        y_line,
        linewidth=2.2,
        label="Fitted concentration-response sigmoid curve"
    )

    ax.axhline(
        50,
        linestyle="--",
        linewidth=1.3,
        label="50% corrected mortality"
    )

    ax.axvline(
        lc50,
        linestyle="--",
        linewidth=1.3,
        label=f"LC50 = {lc50:.2f}%"
    )

    ax.set_title(
        f"Concentration-Response Sigmoid Curve ({sample_name}, {time} min)",
        fontsize=14
    )

    ax.set_xlabel("Sample Concentration (%)", fontsize=12)
    ax.set_ylabel("Corrected Mortality (%)", fontsize=12)
    ax.set_ylim(-5, 105)
    ax.set_xlim(x_min, x_max)
    ax.grid(True, linestyle="--", alpha=0.45)
    ax.legend(fontsize=9)

    ax.text(
        0.58,
        0.18,
        f"y = 100 / [1 + e^(-k(x - LC50))]\n"
        f"k = {k:.4f}\n"
        f"LC50 = {lc50:.2f}%\n"
        f"R² = {r2:.4f}",
        transform=ax.transAxes,
        bbox=dict(facecolor="white", edgecolor="gray", alpha=0.95)
    )

    fig.tight_layout()

    return fig, summary_df


# ================= SIDEBAR =================
st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Page",
    [
        "Home",
        "Input Data",
        "Overall Data",
        "Concentration-Response Sigmoid Graph",
        "LC50 Summary",
        "Prediction"
    ]
)

if st.sidebar.button("Reset All Data"):
    st.session_state.data_store = {}
    st.session_state.input_df = pd.DataFrame()
    st.success("All data cleared.")
    st.rerun()


# ================= TITLE =================
st.title("Brine Shrimp Ecotoxicity Data Analysis System")
st.caption("Streamlit Web Version | Concentration-Response Sigmoid Model")


# ================= HOME =================
if page == "Home":
    st.subheader("Main Dashboard")

    st.write("""
    This web system allows users to input brine shrimp ecotoxicity data and perform
    LC50 analysis using a simplified logistic sigmoid concentration-response model.

    Main modules:
    - Input experimental alive/dead data
    - Calculate mortality percentage
    - Optionally apply Abbott corrected mortality
    - View overall data
    - Generate concentration-response sigmoid graph with replicate points and error bars
    - Calculate LC50 using the logistic sigmoid model
    - Validate LC50 using manual linear interpolation
    - Predict LC50 by time
    """)

    st.info(
        "Selected model: y = 100 / [1 + e^(-k(x - LC50))], where y is corrected mortality (%) "
        "and x is sample concentration (%)."
    )


# ================= INPUT DATA =================
elif page == "Input Data":
    st.subheader("Input Experimental Data")

    data_type = st.radio(
        "Data Type",
        ["Control", "Treatment Sample"],
        horizontal=True
    )

    if data_type == "Control":
        sample_name = "Control"
        use_abbott = False
    else:
        sample_name = st.text_input(
            "Sample Name",
            value="Gold",
            help="You can enter any sample name such as Gold, Silver, Carbon, Copper, Zinc, etc."
        ).strip()

        use_abbott = st.checkbox(
            "Apply Abbott correction using saved control data",
            value=True
        )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        interval = st.number_input("Time Interval (min)", value=20, min_value=1)

    with col2:
        max_time = st.number_input("Maximum Time (min)", value=100, min_value=1)

    with col3:
        replicate = st.number_input(
            "Number of Replicates",
            value=2,
            min_value=1
        )

    with col4:
        initial = st.number_input("Default Initial Brine", value=10, min_value=1)

    if data_type == "Control":
        concentrations = [0.0]
        st.info("Control concentration is automatically set as 0%.")
    else:
        conc_text = st.text_input("Concentration (%)", "10,30,50")

        try:
            concentrations = [
                float(x.strip())
                for x in conc_text.split(",")
                if x.strip() != ""
            ]
        except ValueError:
            st.error("Please enter concentrations as numbers separated by commas, for example: 10,30,50")
            st.stop()

    if not sample_name:
        st.error("Please enter a sample name.")
        st.stop()

    if st.button("Generate Table"):
        rows = []
        times = list(range(0, int(max_time) + int(interval), int(interval)))

        for conc in concentrations:
            for rep in range(1, int(replicate) + 1):
                for time in times:
                    rows.append({
                        "Sample": sample_name,
                        "Data Type": data_type,
                        "Use Abbott Correction": use_abbott,
                        "Concentration": conc,
                        "Time": time,
                        "Replicate": rep,
                        "Initial": initial,
                        "Alive": initial
                    })

        st.session_state.input_df = pd.DataFrame(rows)

    if not st.session_state.input_df.empty:
        st.write("Edit Initial and Alive values:")

        edited_df = st.data_editor(
            st.session_state.input_df,
            num_rows="dynamic",
            use_container_width=True
        )

        if st.button("Calculate and Save Data"):
            df = edited_df.copy()

            required_columns = [
                "Sample",
                "Data Type",
                "Use Abbott Correction",
                "Concentration",
                "Time",
                "Replicate",
                "Initial",
                "Alive"
            ]

            missing_columns = [
                col for col in required_columns
                if col not in df.columns
            ]

            if missing_columns:
                st.error(f"Missing required columns: {missing_columns}")
                st.stop()

            df["Initial"] = pd.to_numeric(df["Initial"], errors="coerce")
            df["Alive"] = pd.to_numeric(df["Alive"], errors="coerce")
            df["Concentration"] = pd.to_numeric(df["Concentration"], errors="coerce")
            df["Time"] = pd.to_numeric(df["Time"], errors="coerce")
            df["Replicate"] = pd.to_numeric(df["Replicate"], errors="coerce")

            if df[["Initial", "Alive", "Concentration", "Time", "Replicate"]].isna().any().any():
                st.error("Please make sure all input values are numeric.")
                st.stop()

            if (df["Alive"] > df["Initial"]).any():
                st.error("Alive value cannot be greater than Initial value.")
                st.stop()

            if (df["Initial"] <= 0).any():
                st.error("Initial value must be greater than 0.")
                st.stop()

            df["Dead"] = df["Initial"] - df["Alive"]
            df["Mortality Decimal"] = df["Dead"] / df["Initial"]
            df["Mortality %"] = df["Mortality Decimal"] * 100

            control_values = []
            corrected_values = []

            control_df = get_control_data()

            for _, row in df.iterrows():
                mortality = float(row["Mortality Decimal"])
                time = float(row["Time"])
                rep = float(row["Replicate"])
                row_data_type = str(row["Data Type"])
                row_use_abbott = bool(row["Use Abbott Correction"])

                if row_data_type == "Control":
                    control = mortality
                    corrected = mortality

                else:
                    if row_use_abbott:
                        if control_df.empty:
                            st.error("Abbott correction selected. Please input and save Control data first.")
                            st.stop()

                        control = get_control_mortality(control_df, time, rep)
                        corrected = abbott_corrected(mortality, control)

                    else:
                        control = 0
                        corrected = mortality

                control_values.append(round(control * 100, 2))
                corrected_values.append(round(corrected * 100, 2))

            df["Control Mortality %"] = control_values
            df["Corrected Mortality %"] = corrected_values

            ordered_cols = [
                "Sample",
                "Data Type",
                "Use Abbott Correction",
                "Concentration",
                "Time",
                "Replicate",
                "Initial",
                "Alive",
                "Dead",
                "Mortality Decimal",
                "Mortality %",
                "Control Mortality %",
                "Corrected Mortality %"
            ]

            df = df[ordered_cols]

            save_data(sample_name, df)

            st.success(f"{sample_name} data saved successfully.")
            st.dataframe(df, use_container_width=True)


# ================= OVERALL DATA =================
elif page == "Overall Data":
    st.subheader("Overall Experimental Data")

    df = get_all_data()

    if df.empty:
        st.warning("No data found. Please input data first.")
    else:
        sample_options = ["All"] + get_sample_names(include_control=True)

        col1, col2 = st.columns(2)

        with col1:
            sample_filter = st.selectbox(
                "Filter Sample",
                sample_options
            )

        with col2:
            conc_filter = st.text_input("Filter Concentration", "")

        if sample_filter != "All":
            df = df[df["Sample"] == sample_filter]

        if conc_filter.strip() != "":
            try:
                df = df[df["Concentration"].astype(float) == float(conc_filter)]
            except ValueError:
                st.error("Concentration must be numeric.")

        st.dataframe(df, use_container_width=True)

        csv = df.to_csv(index=False).encode("utf-8")

        st.download_button(
            "Download CSV",
            csv,
            file_name="overall_data.csv",
            mime="text/csv"
        )


# ================= CONCENTRATION-RESPONSE SIGMOID GRAPH =================
elif page == "Concentration-Response Sigmoid Graph":
    st.subheader("Concentration-Response Sigmoid Graph")

    sample_options = get_sample_names(include_control=False)

    if not sample_options:
        st.warning("No treatment sample data found. Please input data first.")
    else:
        sample_name = st.selectbox("Sample", sample_options)

        df = read_data(sample_name)

        if df.empty:
            st.warning(f"No {sample_name} data found.")
        else:
            df = df[df["Sample"] == sample_name]
            time = st.selectbox("Time", sorted(df["Time"].dropna().unique()))

            plot_df = df[df["Time"] == time].copy()
            plot_df = plot_df[plot_df["Concentration"] > 0]

            try:
                plot_df, k, lc50, r2 = fit_sigmoid_lc50(plot_df)

                fig, summary_df = create_sigmoid_figure(
                    sample_name,
                    time,
                    plot_df,
                    k,
                    lc50,
                    r2
                )

                st.pyplot(fig)

                st.caption(
                    "Error bars represent standard deviation (SD) of corrected mortality "
                    "between replicates at each concentration."
                )

                col1, col2 = st.columns(2)

                col1.metric("LC50 (%)", f"{lc50:.4f}")
                col2.metric("R²", f"{r2:.4f}")

                st.write("Mean corrected mortality and error bars:")
                st.dataframe(summary_df, use_container_width=True)

                st.write("Replicate data used for fitting:")
                st.dataframe(plot_df, use_container_width=True)

            except Exception as e:
                st.error(f"Concentration-response sigmoid fitting failed: {e}")


# ================= LC50 SUMMARY =================
elif page == "LC50 Summary":
    st.subheader("LC50 Calculation Summary and Manual Validation")

    sample_options = get_sample_names(include_control=False)

    if not sample_options:
        st.warning("No treatment sample data found. Please input data first.")
    else:
        sample_name = st.selectbox("Sample", sample_options)

        df = read_data(sample_name)

        if df.empty:
            st.warning(f"No {sample_name} data found.")
        else:
            df = df[df["Sample"] == sample_name]
            time = st.selectbox("Time", sorted(df["Time"].dropna().unique()))

            temp = df[df["Time"] == time].copy()
            temp = temp[temp["Concentration"] > 0]

            if len(temp) < 3:
                st.warning("At least three data points are recommended.")
            else:
                try:
                    plot_df, k, lc50_gui, r2 = fit_sigmoid_lc50(temp)

                    lc50_manual, mean_df, message = manual_linear_interpolation_lc50(temp)

                    if pd.isna(lc50_manual):
                        difference = np.nan
                        deviation = np.nan
                    else:
                        difference = abs(lc50_gui - lc50_manual)
                        deviation = (difference / lc50_manual) * 100 if lc50_manual != 0 else np.nan

                    result_df = pd.DataFrame({
                        "Parameter": [
                            "Manual linear interpolation LC50 (%)",
                            "Python GUI concentration-response sigmoid LC50 (%)",
                            "Difference (%)",
                            "Deviation (%)",
                            "Slope parameter, k",
                            "R²",
                            "Toxicity Interpretation"
                        ],
                        "Value": [
                            round(lc50_manual, 4) if not pd.isna(lc50_manual) else "Not available",
                            round(lc50_gui, 4),
                            round(difference, 4) if not pd.isna(difference) else "Not available",
                            round(deviation, 4) if not pd.isna(deviation) else "Not available",
                            round(k, 4),
                            round(r2, 4),
                            toxicity_level(lc50_gui)
                        ]
                    })

                    st.dataframe(result_df, use_container_width=True)

                    st.write("Mean corrected mortality values used for manual interpolation:")
                    st.dataframe(mean_df, use_container_width=True)

                    if message != "OK":
                        st.warning(message)

                    st.info(
                        "Manual interpolation is used only for software validation. "
                        "The main LC50 output is generated using the concentration-response sigmoid model."
                    )

                except Exception as e:
                    st.error(f"LC50 calculation failed: {e}")


# ================= PREDICTION =================
elif page == "Prediction":
    st.subheader("LC50 Time Prediction")

    sample_options = get_sample_names(include_control=False)

    if not sample_options:
        st.warning("No treatment sample data found. Please input data first.")
    else:
        sample_name = st.selectbox("Sample", sample_options)

        df = read_data(sample_name)

        if df.empty:
            st.warning(f"No {sample_name} data found.")
        else:
            df = df[df["Sample"] == sample_name]

            results = []

            for time in sorted(df["Time"].dropna().unique()):
                if time == 0:
                    continue

                temp = df[df["Time"] == time].copy()
                temp = temp[temp["Concentration"] > 0]

                if len(temp) < 3:
                    continue

                try:
                    _, k, lc50, r2 = fit_sigmoid_lc50(temp)

                    results.append({
                        "Time": time,
                        "LC50 (%)": round(lc50, 4),
                        "R²": round(r2, 4)
                    })

                except Exception:
                    continue

            if len(results) < 2:
                st.warning("Not enough valid LC50 values for prediction.")
            else:
                lc50_df = pd.DataFrame(results)
                st.dataframe(lc50_df, use_container_width=True)

                target_time = st.number_input("Predict Time (min)", value=45.0)

                times = lc50_df["Time"].astype(float).values
                lc50_values = lc50_df["LC50 (%)"].astype(float).values

                if target_time < min(times) or target_time > max(times):
                    coeff = np.polyfit(times, lc50_values, 1)
                    predicted_lc50 = coeff[0] * target_time + coeff[1]
                    st.warning("This is outside experimental range. Result is extrapolated.")
                else:
                    predicted_lc50 = np.interp(target_time, times, lc50_values)

                st.success(f"Predicted LC50 at {target_time:.2f} min = {predicted_lc50:.4f}%")
                st.write(f"Toxicity Interpretation: **{toxicity_level(predicted_lc50)}**")
