import streamlit as st
import pandas as pd
import numpy as np
import os
from scipy.stats import norm
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

st.set_page_config(page_title="Ecotoxicity Analysis System", layout="wide")

DATA_FOLDER = "data"
os.makedirs(DATA_FOLDER, exist_ok=True)

CONTROL_FILE = os.path.join(DATA_FOLDER, "control_data.csv")
GOLD_FILE = os.path.join(DATA_FOLDER, "gold_data.csv")
SILVER_FILE = os.path.join(DATA_FOLDER, "silver_data.csv")


# ================= FUNCTIONS =================
def read_csv_safe(file_path):
    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
        return pd.read_csv(file_path)
    return pd.DataFrame()


def save_sample_data(sample, df):
    if sample == "Control":
        file_path = CONTROL_FILE
    elif sample == "Gold":
        file_path = GOLD_FILE
    else:
        file_path = SILVER_FILE

    old_df = read_csv_safe(file_path)

    if not old_df.empty:
        combined = pd.concat([old_df, df], ignore_index=True)
        combined = combined.drop_duplicates(
            subset=["Sample", "Concentration", "Time", "Replicate"],
            keep="last"
        )
    else:
        combined = df

    combined.to_csv(file_path, index=False)


def safe_probit(value):
    value = min(max(value, 0.001), 0.999)
    return norm.ppf(value) + 5


def abbott_corrected(treatment, control):
    if control >= 1:
        return 0
    corrected = (treatment - control) / (1 - control)
    return max(0, min(corrected, 1))


def sigmoid_model(x, k1, k2):
    return 1 / (1 + np.exp(-(k1 * x + k2)))


def get_control_mortality(time, replicate):
    control_df = read_csv_safe(CONTROL_FILE)

    if control_df.empty:
        return 0

    control_df["Time"] = pd.to_numeric(control_df["Time"], errors="coerce")
    control_df["Replicate"] = pd.to_numeric(control_df["Replicate"], errors="coerce")

    match = control_df[
        (control_df["Time"] == time) &
        (control_df["Replicate"] == replicate)
    ]

    if match.empty:
        return 0

    return float(match.iloc[0]["Mortality %"])


def process_dataframe(df):
    if df.empty:
        return df

    df["Initial"] = pd.to_numeric(df["Initial"], errors="coerce")
    df["Alive"] = pd.to_numeric(df["Alive"], errors="coerce")
    df["Time"] = pd.to_numeric(df["Time"], errors="coerce")
    df["Replicate"] = pd.to_numeric(df["Replicate"], errors="coerce")
    df["Concentration"] = pd.to_numeric(df["Concentration"], errors="coerce")

    df["Alive"] = df["Alive"].clip(lower=0, upper=df["Initial"])
    df["Dead"] = df["Initial"] - df["Alive"]

    df["Mortality Decimal"] = df["Dead"] / df["Initial"]
    df["Mortality %"] = df["Mortality Decimal"] * 100

    if df["Sample"].iloc[0] == "Control":
        df["Control Mortality %"] = 0
        df["Corrected Mortality %"] = 0
        df["Probit"] = df["Mortality Decimal"].apply(safe_probit)
        df["Log Conc"] = 0
    else:
        df["Control Mortality %"] = df.apply(
            lambda row: get_control_mortality(row["Time"], row["Replicate"]),
            axis=1
        )

        df["Corrected Mortality %"] = df.apply(
            lambda row: abbott_corrected(
                row["Mortality %"] / 100,
                row["Control Mortality %"] / 100
            ) * 100,
            axis=1
        )

        df["Probit"] = (df["Corrected Mortality %"] / 100).apply(safe_probit)
        df["Log Conc"] = df["Concentration"].apply(
            lambda x: np.log10(x) if x > 0 else 0
        )

    return df.round(4)


def toxicity_level(lc50):
    if lc50 < 1:
        return "Extremely toxic"
    elif lc50 < 10:
        return "Highly toxic"
    elif lc50 < 50:
        return "Moderately toxic"
    else:
        return "Low toxicity"


# ================= SIDEBAR =================
st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Page",
    [
        "Home",
        "Input Data",
        "Overall Data",
        "Probit Graph",
        "Sigmoid Graph",
        "LC50 Summary",
        "Prediction"
    ]
)

if st.sidebar.button("Reset All Data"):
    for file in [CONTROL_FILE, GOLD_FILE, SILVER_FILE]:
        if os.path.exists(file):
            os.remove(file)
    st.sidebar.success("All data has been reset.")


# ================= HEADER =================
st.title("Brine Shrimp Ecotoxicity Data Analysis System")
st.caption("Streamlit Web Version")


# ================= HOME =================
if page == "Home":
    st.subheader("Main Dashboard")
    st.write(
        """
        This web system allows users to input brine shrimp ecotoxicity data
        and perform toxicity analysis.

        Main modules:
        - Input experimental alive data
        - Automatically calculate dead brine shrimp
        - Calculate mortality percentage
        - Calculate Abbott corrected mortality
        - Generate Probit graph
        - Generate Sigmoid graph
        - Calculate LC50
        - Predict LC50 by time
        """
    )


# ================= INPUT DATA =================
elif page == "Input Data":
    st.subheader("Input Experimental Data")

    sample = st.selectbox("Sample Type", ["Control", "Gold", "Silver"])

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        time_interval = st.number_input("Time Interval (min)", min_value=1, value=20)

    with col2:
        max_time = st.number_input("Maximum Time (min)", min_value=1, value=100)

    with col3:
        number_of_replicates = st.number_input("Number of Replicates", min_value=1, value=2)

    with col4:
        initial = st.number_input("Initial Brine", min_value=1, value=20)

    if sample == "Control":
        concentration = 0
    else:
        concentration = st.number_input("Concentration (%)", min_value=0.0, value=10.0)

    times = list(range(0, int(max_time) + 1, int(time_interval)))

    if st.button("Generate Table"):
        rows = []

        for rep in range(1, int(number_of_replicates) + 1):
            for t in times:
                rows.append({
                    "Sample": sample,
                    "Concentration": concentration,
                    "Time": t,
                    "Replicate": rep,
                    "Initial": initial,
                    "Alive": initial
                })

        input_df = pd.DataFrame(rows)
        st.session_state["input_df"] = input_df

    if "input_df" in st.session_state:
        edited_df = st.data_editor(
            st.session_state["input_df"],
            use_container_width=True,
            num_rows="fixed",
            disabled=["Sample", "Concentration", "Time", "Replicate", "Initial"]
        )

        if st.button("Save Data"):
            edited_df = process_dataframe(edited_df)
            save_sample_data(sample, edited_df)
            st.success(f"{sample} data saved successfully.")
            st.dataframe(edited_df, use_container_width=True)


# ================= OVERALL DATA =================
elif page == "Overall Data":
    st.subheader("Overall Experimental Data")

    control_df = read_csv_safe(CONTROL_FILE)
    gold_df = read_csv_safe(GOLD_FILE)
    silver_df = read_csv_safe(SILVER_FILE)

    all_df = pd.concat([control_df, gold_df, silver_df], ignore_index=True)

    if all_df.empty:
        st.warning("No data found. Please input data first.")
    else:
        st.dataframe(all_df, use_container_width=True)


# ================= PROBIT GRAPH =================
elif page == "Probit Graph":
    st.subheader("Probit vs Log Concentration Graph")

    sample = st.selectbox("Sample", ["Gold", "Silver"])
    file_path = GOLD_FILE if sample == "Gold" else SILVER_FILE

    df = read_csv_safe(file_path)

    if df.empty:
        st.warning(f"No {sample} data found.")
    else:
        df = process_dataframe(df)
        df = df[(df["Sample"] == sample) & (df["Concentration"] > 0)]

        time = st.selectbox("Select Time (min)", sorted(df["Time"].dropna().unique()))
        plot_df = df[df["Time"] == time]

        if len(plot_df) < 3:
            st.warning("At least three data points are recommended for probit analysis.")
        else:
            x = plot_df["Log Conc"].astype(float).values
            y = plot_df["Probit"].astype(float).values

            slope, intercept = np.polyfit(x, y, 1)
            lc50 = 10 ** ((5 - intercept) / slope)

            fig, ax = plt.subplots(figsize=(10, 5.5))
            ax.scatter(x, y, label="Replicate Data Points")

            x_line = np.linspace(min(x) - 0.05, max(x) + 0.05, 100)
            y_line = slope * x_line + intercept

            ax.plot(x_line, y_line, label="Linear Regression")
            ax.axhline(5, linestyle="--", label="Probit 5")
            ax.axvline(np.log10(lc50), linestyle="--", label=f"LC50 = {lc50:.2f}%")

            ax.set_title(f"Probit vs Log Concentration ({sample}, {time} min)")
            ax.set_xlabel("Log Concentration")
            ax.set_ylabel("Probit")
            ax.grid(True, linestyle="--", alpha=0.5)
            ax.legend()

            ax.text(
                0.05,
                0.95,
                f"y = {slope:.4f}x + {intercept:.4f}\nLC50 = {lc50:.2f}%",
                transform=ax.transAxes,
                verticalalignment="top",
                bbox=dict(facecolor="white", edgecolor="gray")
            )

            st.pyplot(fig)
            st.dataframe(plot_df, use_container_width=True)


# ================= SIGMOID GRAPH =================
elif page == "Sigmoid Graph":
    st.subheader("Corrected Mortality Sigmoid Graph")

    sample = st.selectbox("Sample", ["Gold", "Silver"])
    file_path = GOLD_FILE if sample == "Gold" else SILVER_FILE

    df = read_csv_safe(file_path)

    if df.empty:
        st.warning(f"No {sample} data found.")
    else:
        df = process_dataframe(df)
        df = df[(df["Sample"] == sample) & (df["Concentration"] > 0)]

        time = st.selectbox("Select Time (min)", sorted(df["Time"].dropna().unique()))
        plot_df = df[df["Time"] == time]

        if len(plot_df) < 3:
            st.warning("At least three data points are recommended for sigmoid fitting.")
        else:
            x = plot_df["Log Conc"].astype(float).values
            y = plot_df["Corrected Mortality %"].astype(float).values / 100

            try:
                params, _ = curve_fit(
                    sigmoid_model,
                    x,
                    y,
                    p0=[5, -5],
                    maxfev=10000
                )

                k1, k2 = params
                y_pred = sigmoid_model(x, k1, k2)

                ss_res = np.sum((y - y_pred) ** 2)
                ss_total = np.sum((y - np.mean(y)) ** 2)
                r2 = 1 - (ss_res / ss_total) if ss_total != 0 else 0

                log_lc50 = -k2 / k1
                lc50 = 10 ** log_lc50

                fig, ax = plt.subplots(figsize=(10, 5.5))
                ax.scatter(x, y * 100, label="Replicate Data Points")

                x_line = np.linspace(min(x) - 0.05, max(x) + 0.05, 300)
                y_line = sigmoid_model(x_line, k1, k2) * 100

                ax.plot(x_line, y_line, label="Fitted Sigmoid Curve")
                ax.axhline(50, linestyle="--", label="50% Corrected Mortality")
                ax.axvline(log_lc50, linestyle="--", label=f"LC50 = {lc50:.2f}%")

                ax.set_title(f"Corrected Mortality Sigmoid Curve ({sample}, {time} min)")
                ax.set_xlabel("Log Concentration")
                ax.set_ylabel("Corrected Mortality (%)")
                ax.set_ylim(-5, 105)
                ax.grid(True, linestyle="--", alpha=0.5)
                ax.legend()

                ax.text(
                    0.70,
                    0.15,
                    f"k1 = {k1:.4f}\nk2 = {k2:.4f}\nR² = {r2:.4f}\nLC50 = {lc50:.2f}%",
                    transform=ax.transAxes,
                    bbox=dict(facecolor="white", edgecolor="gray")
                )

                st.pyplot(fig)
                st.dataframe(plot_df, use_container_width=True)

            except Exception as e:
                st.error(f"Sigmoid fitting failed: {e}")


# ================= LC50 SUMMARY =================
elif page == "LC50 Summary":
    st.subheader("LC50 Calculation Summary")

    sample = st.selectbox("Sample", ["Gold", "Silver"])
    file_path = GOLD_FILE if sample == "Gold" else SILVER_FILE

    df = read_csv_safe(file_path)

    if df.empty:
        st.warning(f"No {sample} data found.")
    else:
        df = process_dataframe(df)
        df = df[(df["Sample"] == sample) & (df["Concentration"] > 0)]

        time = st.selectbox("Select Time (min)", sorted(df["Time"].dropna().unique()))
        df = df[df["Time"] == time]

        if len(df) < 3:
            st.warning("At least three data points are recommended.")
        else:
            x = df["Log Conc"].astype(float).values

            probit_y = df["Probit"].astype(float).values
            slope, intercept = np.polyfit(x, probit_y, 1)
            lc50_probit = 10 ** ((5 - intercept) / slope)

            y = df["Corrected Mortality %"].astype(float).values / 100

            try:
                params, _ = curve_fit(
                    sigmoid_model,
                    x,
                    y,
                    p0=[5, -5],
                    maxfev=10000
                )

                k1, k2 = params
                lc50_sigmoid = 10 ** (-k2 / k1)

                summary_df = pd.DataFrame({
                    "Sample": [sample],
                    "Time (min)": [time],
                    "LC50 Probit (%)": [round(lc50_probit, 4)],
                    "LC50 Sigmoid (%)": [round(lc50_sigmoid, 4)],
                    "Toxicity Level": [toxicity_level(lc50_sigmoid)]
                })

                st.dataframe(summary_df, use_container_width=True)

            except Exception as e:
                st.error(f"LC50 calculation failed: {e}")


# ================= PREDICTION =================
elif page == "Prediction":
    st.subheader("LC50 Time Prediction")

    sample = st.selectbox("Sample", ["Gold", "Silver"])
    file_path = GOLD_FILE if sample == "Gold" else SILVER_FILE

    df = read_csv_safe(file_path)

    if df.empty:
        st.warning(f"No {sample} data found.")
    else:
        df = process_dataframe(df)
        df = df[(df["Sample"] == sample) & (df["Concentration"] > 0)]

        prediction_results = []

        for time in sorted(df["Time"].dropna().unique()):
            temp_df = df[df["Time"] == time]

            if len(temp_df) >= 3:
                try:
                    x = temp_df["Log Conc"].astype(float).values
                    y = temp_df["Corrected Mortality %"].astype(float).values / 100

                    params, _ = curve_fit(
                        sigmoid_model,
                        x,
                        y,
                        p0=[5, -5],
                        maxfev=10000
                    )

                    k1, k2 = params
                    lc50 = 10 ** (-k2 / k1)

                    prediction_results.append({
                        "Time (min)": time,
                        "Predicted LC50 (%)": round(lc50, 4),
                        "Toxicity Level": toxicity_level(lc50)
                    })

                except:
                    pass

        if prediction_results:
            result_df = pd.DataFrame(prediction_results)
            st.dataframe(result_df, use_container_width=True)

            fig, ax = plt.subplots(figsize=(10, 5.5))
            ax.plot(result_df["Time (min)"], result_df["Predicted LC50 (%)"], marker="o")
            ax.set_title(f"Predicted LC50 Over Time ({sample})")
            ax.set_xlabel("Time (min)")
            ax.set_ylabel("LC50 (%)")
            ax.grid(True, linestyle="--", alpha=0.5)
            st.pyplot(fig)
        else:
            st.warning("Not enough data for prediction.")
