import streamlit as st
import pandas as pd
import numpy as np
import os
from scipy.stats import norm
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Ecotoxicity Analysis System",
    layout="wide"
)

DATA_FOLDER = "data"
os.makedirs(DATA_FOLDER, exist_ok=True)

CONTROL_FILE = os.path.join(DATA_FOLDER, "control_data.csv")
GOLD_FILE = os.path.join(DATA_FOLDER, "gold_data.csv")
SILVER_FILE = os.path.join(DATA_FOLDER, "silver_data.csv")


# ================= FUNCTIONS =================
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


def toxicity_level(lc50):
    if lc50 < 1:
        return "Extremely toxic"
    elif lc50 < 10:
        return "Highly toxic"
    elif lc50 < 50:
        return "Moderately toxic"
    else:
        return "Low toxicity"


def read_csv_safe(file_path):
    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
        return pd.read_csv(file_path)
    return pd.DataFrame()


def save_sample_data(sample, df):
    if sample == "Control":
        path = CONTROL_FILE
    elif sample == "Gold":
        path = GOLD_FILE
    else:
        path = SILVER_FILE

    old_df = read_csv_safe(path)

    if not old_df.empty:
        combined = pd.concat([old_df, df], ignore_index=True)
        combined = combined.drop_duplicates(
            subset=["Sample", "Concentration", "Time", "Replicate"],
            keep="last"
        )
    else:
        combined = df

    combined.to_csv(path, index=False)


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

st.title("Brine Shrimp Ecotoxicity Data Analysis System")
st.caption("Streamlit Web Version")


# ================= HOME =================
if page == "Home":
    st.subheader("Main Dashboard")
    st.write(
        """
        This web version is a future enhancement of the Python desktop GUI system.

        Main modules:
        - Input experimental mortality data
        - Calculate Abbott corrected mortality
        - View overall saved data
        - Generate Probit vs Log Concentration graph
        - Generate Corrected Mortality sigmoid graph
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
        interval = st.number_input("Time Interval (min)", value=20, min_value=1)

    with col2:
        max_time = st.number_input("Maximum Time (min)", value=100, min_value=1)

    with col3:
        replicate = st.number_input(
            "Replicate",
            value=1 if sample == "Control" else 2,
            min_value=1
        )

    with col4:
        initial = st.number_input("Default Initial Brine", value=10, min_value=1)

    if sample == "Control":
        concentrations = [0.0]
        st.info("Control concentration is automatically set as 0%.")
    else:
        conc_text = st.text_input("Concentration (%)", "10,30,50")
        concentrations = [
            float(x.strip())
            for x in conc_text.split(",")
            if x.strip() != ""
        ]

    if st.button("Generate Table"):
        rows = []
        times = list(range(0, int(max_time) + int(interval), int(interval)))

        for conc in concentrations:
            for rep in range(1, int(replicate) + 1):
                for time in times:
                    rows.append({
                        "Sample": sample,
                        "Concentration": conc,
                        "Time": time,
                        "Replicate": rep,
                        "Initial": initial,
                        "Alive": initial
                    })

        st.session_state.input_df = pd.DataFrame(rows)

    if "input_df" in st.session_state:
        st.write("Edit Initial and Alive values:")

        edited_df = st.data_editor(
            st.session_state.input_df,
            num_rows="dynamic",
            use_container_width=True
        )

        if st.button("Calculate and Save Data"):
            df = edited_df.copy()

            df["Dead"] = df["Initial"] - df["Alive"]
            df["Mortality Decimal"] = df["Dead"] / df["Initial"]
            df["Mortality %"] = df["Mortality Decimal"] * 100

            control_dict = {}

            if sample == "Control":
                for _, row in df.iterrows():
                    control_dict[int(row["Time"])] = float(row["Mortality Decimal"])
            else:
                control_df = read_csv_safe(CONTROL_FILE)

                if control_df.empty:
                    st.error("Please input and save Control data first.")
                    st.stop()

                for _, row in control_df.iterrows():
                    control_dict[int(row["Time"])] = float(row["Mortality Decimal"])

            control_values = []
            corrected_values = []
            probit_values = []
            log_values = []

            for _, row in df.iterrows():
                mortality = float(row["Mortality Decimal"])
                time = int(row["Time"])

                if sample == "Control":
                    control = mortality
                    corrected = mortality
                else:
                    control = control_dict.get(time, 0)
                    corrected = abbott_corrected(mortality, control)

                control_values.append(round(control * 100, 2))
                corrected_values.append(round(corrected * 100, 2))
                probit_values.append(round(safe_probit(corrected), 4))

                if row["Concentration"] == 0:
                    log_values.append("")
                else:
                    log_values.append(round(np.log10(row["Concentration"]), 4))

            df["Control Mortality %"] = control_values
            df["Corrected Mortality %"] = corrected_values
            df["Probit"] = probit_values
            df["Log Conc"] = log_values

            save_sample_data(sample, df)

            st.success(f"{sample} data saved successfully.")
            st.dataframe(df, use_container_width=True)


# ================= OVERALL DATA =================
elif page == "Overall Data":
    st.subheader("Overall Experimental Data")

    dfs = []

    for path in [CONTROL_FILE, GOLD_FILE, SILVER_FILE]:
        temp_df = read_csv_safe(path)
        if not temp_df.empty:
            dfs.append(temp_df)

    if not dfs:
        st.warning("No saved data found.")
    else:
        df = pd.concat(dfs, ignore_index=True)

        col1, col2 = st.columns(2)

        with col1:
            sample_filter = st.selectbox(
                "Filter Sample",
                ["All", "Control", "Gold", "Silver"]
            )

        with col2:
            conc_filter = st.text_input("Filter Concentration", "")

        if sample_filter != "All":
            df = df[df["Sample"] == sample_filter]

        if conc_filter.strip() != "":
            try:
                df = df[df["Concentration"].astype(float) == float(conc_filter)]
            except:
                st.error("Concentration must be numeric.")

        st.dataframe(df, use_container_width=True)

        excel_path = os.path.join(DATA_FOLDER, "overall_data.xlsx")
        df.to_excel(excel_path, index=False)

        with open(excel_path, "rb") as file:
            st.download_button(
                "Download Excel",
                file,
                file_name="overall_data.xlsx"
            )


# ================= PROBIT GRAPH =================
elif page == "Probit Graph":
    st.subheader("Probit vs Log Concentration Graph")

    sample = st.selectbox("Sample", ["Gold", "Silver"])
    file_path = GOLD_FILE if sample == "Gold" else SILVER_FILE

    df = read_csv_safe(file_path)

    if df.empty:
        st.warning(f"No {sample} data found.")
    else:
        df = df[df["Sample"] == sample]

        time = st.selectbox("Time", sorted(df["Time"].dropna().unique()))

        plot_df = df[df["Time"] == time].copy()
        plot_df = plot_df[plot_df["Concentration"] > 0]

        plot_df["Log Conc"] = pd.to_numeric(plot_df["Log Conc"], errors="coerce")
        plot_df["Probit"] = pd.to_numeric(plot_df["Probit"], errors="coerce")
        plot_df = plot_df.dropna(subset=["Log Conc", "Probit"])

        if len(plot_df) < 2:
            st.warning("At least two data points are required.")
        else:
            x = plot_df["Log Conc"].astype(float).values
            y = plot_df["Probit"].astype(float).values

            slope, intercept = np.polyfit(x, y, 1)
            y_pred = slope * x + intercept

            ss_res = np.sum((y - y_pred) ** 2)
            ss_total = np.sum((y - np.mean(y)) ** 2)
            r2 = 1 - (ss_res / ss_total) if ss_total != 0 else 0

            log_lc50 = (5 - intercept) / slope
            lc50 = 10 ** log_lc50

            fig, ax = plt.subplots(figsize=(10, 5.5))
            ax.scatter(x, y, label="Replicate Data Points")

            x_line = np.linspace(min(x) - 0.05, max(x) + 0.05, 100)
            ax.plot(x_line, slope * x_line + intercept, label="Linear Regression Line")

            ax.axhline(5, linestyle="--", label="Probit 5 = 50% Mortality")
            ax.axvline(log_lc50, linestyle="--", label=f"LC50 = {lc50:.2f}%")

            ax.set_title(f"Probit vs Log Concentration ({sample}, {time} min)")
            ax.set_xlabel("Log Concentration")
            ax.set_ylabel("Probit")
            ax.grid(True, linestyle="--", alpha=0.5)
            ax.legend()

            ax.text(
                0.05,
                0.15,
                f"y = {slope:.4f}x + {intercept:.4f}\nR² = {r2:.4f}\nLC50 = {lc50:.2f}%",
                transform=ax.transAxes,
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
        df = df[df["Sample"] == sample]

        time = st.selectbox("Time", sorted(df["Time"].dropna().unique()))

        plot_df = df[df["Time"] == time].copy()
        plot_df = plot_df[plot_df["Concentration"] > 0]

        plot_df["Log Conc"] = pd.to_numeric(plot_df["Log Conc"], errors="coerce")
        plot_df["Corrected Mortality %"] = pd.to_numeric(
            plot_df["Corrected Mortality %"],
            errors="coerce"
        )

        plot_df = plot_df.dropna(subset=["Log Conc", "Corrected Mortality %"])

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
        df = df[df["Sample"] == sample]

        time = st.selectbox("Time", sorted(df["Time"].dropna().unique()))

        df = df[df["Time"] == time]
        df = df[df["Concentration"] > 0]

        df["Log Conc"] = pd.to_numeric(df["Log Conc"], errors="coerce")
        df["Probit"] = pd.to_numeric(df["Probit"], errors="coerce")
        df["Corrected Mortality %"] = pd.to_numeric(
            df["Corrected Mortality %"],
            errors="coerce"
        )

        df = df.dropna(subset=["Log Conc", "Probit", "Corrected Mortality %"])

        if len(df) < 3:
            st.warning("At least three data points are recommended.")
        else:
            x = df["Log Conc"].astype(float).values

            # Probit
            probit_y = df["Probit"].astype(float).values
            slope, intercept = np.polyfit(x, probit_y, 1)
            lc50_probit = 10 ** ((5 - intercept) / slope)

            # Sigmoid
            y = df["Corrected Mortality %"].astype(float).values / 100
            params, _ = curve_fit(sigmoid_model, x, y, p0=[5, -5], maxfev=10000)
            k1, k2 = params
            lc50_sigmoid = 10 ** (-k2 / k1)

            result_df = pd.DataFrame({
                "Method": ["Probit Analysis", "Sigmoid Analysis"],
                "LC50 (%)": [round(lc50_probit, 4), round(lc50_sigmoid, 4)],
                "Toxicity Interpretation": [
                    toxicity_level(lc50_probit),
                    toxicity_level(lc50_sigmoid)
                ]
            })

            st.dataframe(result_df, use_container_width=True)

            st.info(
                "Probit LC50 is recommended as the main toxicological LC50 value, "
                "while sigmoid LC50 is used for dose-response model comparison."
            )


# ================= PREDICTION =================
elif page == "Prediction":
    st.subheader("LC50 Time Prediction")

    sample = st.selectbox("Sample", ["Gold", "Silver"])
    file_path = GOLD_FILE if sample == "Gold" else SILVER_FILE

    df = read_csv_safe(file_path)

    if df.empty:
        st.warning(f"No {sample} data found.")
    else:
        df = df[df["Sample"] == sample]

        results = []

        for time in sorted(df["Time"].dropna().unique()):
            temp = df[df["Time"] == time].copy()
            temp = temp[temp["Concentration"] > 0]

            temp["Log Conc"] = pd.to_numeric(temp["Log Conc"], errors="coerce")
            temp["Corrected Mortality %"] = pd.to_numeric(
                temp["Corrected Mortality %"],
                errors="coerce"
            )

            temp = temp.dropna(subset=["Log Conc", "Corrected Mortality %"])

            if len(temp) < 3:
                continue

            try:
                x = temp["Log Conc"].astype(float).values
                y = temp["Corrected Mortality %"].astype(float).values / 100

                params, _ = curve_fit(sigmoid_model, x, y, p0=[5, -5], maxfev=10000)
                k1, k2 = params

                lc50 = 10 ** (-k2 / k1)

                results.append({
                    "Time": time,
                    "LC50 (%)": round(lc50, 4)
                })

            except:
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