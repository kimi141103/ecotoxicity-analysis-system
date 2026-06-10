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
            help="Enter any sample name such as Gold, Silver, Carbon, Copper, Zinc, etc."
        ).strip()

        use_abbott = st.checkbox(
            "Apply Abbott correction using saved control data",
            value=True
        )

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        time_unit = st.selectbox("Time Unit", ["min", "h"])

    with col2:
        interval = st.number_input("Time Interval", value=20, min_value=1)

    with col3:
        max_time = st.number_input("Maximum Time", value=100, min_value=1)

    with col4:
        replicate = st.number_input(
            "Number of Replicates",
            value=2,
            min_value=1
        )

    with col5:
        initial = st.number_input("Default Initial Brine", value=20, min_value=1)

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
                        "Time Unit": time_unit,
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
                "Time Unit",
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

            # Apply Abbott option once to the whole dataset
            df["Use Abbott Correction"] = use_abbott

            control_values = []
            analysis_values = []

            control_df = get_control_data()

            for _, row in df.iterrows():
                mortality = float(row["Mortality Decimal"])
                time = float(row["Time"])
                rep = float(row["Replicate"])
                row_data_type = str(row["Data Type"])

                if row_data_type == "Control":
                    control = mortality
                    analysis_mortality = mortality

                else:
                    if use_abbott:
                        if control_df.empty:
                            st.error("Abbott correction selected. Please input and save Control data first.")
                            st.stop()

                        control = get_control_mortality(control_df, time, rep)
                        analysis_mortality = abbott_corrected(mortality, control)

                    else:
                        control = 0
                        analysis_mortality = mortality

                control_values.append(round(control * 100, 2))
                analysis_values.append(round(analysis_mortality * 100, 2))

            df["Control Mortality %"] = control_values
            df["Analysis Mortality %"] = analysis_values

            ordered_cols = [
                "Sample",
                "Data Type",
                "Use Abbott Correction",
                "Time Unit",
                "Concentration",
                "Time",
                "Replicate",
                "Initial",
                "Alive",
                "Dead",
                "Mortality Decimal",
                "Mortality %",
                "Control Mortality %",
                "Analysis Mortality %"
            ]

            df = df[ordered_cols]

            save_data(sample_name, df)

            st.success(f"{sample_name} data saved successfully.")
            st.dataframe(df, use_container_width=True)
