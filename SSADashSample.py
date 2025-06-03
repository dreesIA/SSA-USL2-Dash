# VERY IMPORTANT
# source .venv/bin/activate
# TYPE THAT INTO TERMINAL

import streamlit as st
import pandas as pd
import altair as alt
import matplotlib.pyplot as plt
import numpy as np
from math import pi
from PIL import Image, ImageDraw

st.set_page_config(page_title="SSA Swarm Gameday Report", layout="wide")

# Colors
primary_color = "#D72638"
background_color = "#1A1A1D"
text_color = "#FFFFFF"

st.markdown(
    f"""
    <style>
        .reportview-container {{
            background-color: {background_color};
            color: {text_color};
        }}
        .sidebar .sidebar-content {{
            background-color: {background_color};
        }}
        h1, h2, h3, h4, h5, h6 {{
            color: {primary_color};
        }}
    </style>
    """,
    unsafe_allow_html=True
)

# File paths
match_files = {
    "5.17 vs Birmingham Legion 2": "SSA Swarm USL Mens 2 Games Statsports Reports - 5.17 SSA Swarm USL 2 vs Birmingham.csv",
    "5.21 vs Tennessee SC": "SSA Swarm USL Mens 2 Games Statsports Reports - 5.21 Swarm USL 2 Vs TN SC.csv",
    "5.25 vs East Atlanta FC": "SSA Swarm USL Mens 2 Games Statsports Reports - 5.25 SSA Swarm USL 2 vs East Atlanta.csv",
    
}

training_files = {
    "5.12 Training": "SSA Swarm USL 2 Mens Statsports Traning Report  - 5.12 Training.csv",
    "5.13 Training": "SSA Swarm USL 2 Mens Statsports Traning Report  - 5.13 Training .csv",
    "5.15 Training": "SSA Swarm USL 2 Mens Statsports Traning Report  - 5.15 Training.csv",
    "5.16 Training": "SSA Swarm USL 2 Mens Statsports Traning Report  - 5.16 Training.csv",
    "5.19 Training": "SSA Swarm USL 2 Mens Statsports Traning Report  - 5.19 Training.csv",
    "5.20 Training": "SSA Swarm USL 2 Mens Statsports Traning Report  - 5.20 Training.csv",
}

# Load data
def load_data(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df.dropna(subset=["Player Name"], inplace=True)
    df["Session Type"] = df["Session Type"].astype(str).str.strip().str.title()
    return df

# Radar chart with percentiles
def multi_player_percentile_radar(selected_df, full_df, metrics, title):
    percentile_df = selected_df.copy()
    for metric in metrics:
        all_values = full_df[metric]
        percentile_df[metric] = selected_df[metric].apply(
            lambda val: (all_values < val).sum() / len(all_values) * 100
        )

    players = percentile_df.index.tolist()
    values = percentile_df[metrics].astype(float).values.tolist()
    num_vars = len(metrics)

    angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

    for i, vals in enumerate(values):
        vals += vals[:1]
        ax.plot(angles, vals, label=players[i])
        ax.fill(angles, vals, alpha=0.1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels(["0", "25", "50", "75", "100"])
    ax.set_title(title, color=primary_color)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    return fig

# Constants
required_columns = [
    "Player Name", "Session Type", "Total Distance", "Max Speed", "No of Sprints",
    "Sprint Distance", "Accelerations", "Decelerations", "High Speed Running"
]
metrics = required_columns[2:]

#Sidebar Image
def crop_circle(image_path):
    img = Image.open(image_path).convert("RGBA")
    size = min(img.size)
    mask = Image.new('L', (size, size), 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse((0, 0, size, size), fill=255)

    # Center crop
    left = (img.width - size) // 2
    top = (img.height - size) // 2
    img_cropped = img.crop((left, top, left + size, top + size))
    img_cropped.putalpha(mask)
    return img_cropped

# Load and display in sidebar
logo = crop_circle("SSALogoTransparent.jpeg")
st.sidebar.image(logo, width=250)

# Match selection
match_options = ["All Matches (Average)"] + list(match_files.keys())
selected_match = st.sidebar.selectbox("Select Match", match_options)

if selected_match == "All Matches (Average)":
    df = pd.concat([load_data(path) for path in match_files.values()], ignore_index=True)
else:
    df = load_data(match_files[selected_match])

for col in metrics:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Sidebar report type
report_type = st.sidebar.selectbox("Select Report Type", [
    "Match Report", "Weekly Training Report", "Daily Training Report", "Compare Players"
])

def select_half():
    return st.sidebar.selectbox("Select Half", ["Total", "First Half", "Second Half"])

# Page title
st.title(f"SSA Swarm USL2 Performance Center - {selected_match}")

# Match Report
if report_type == "Match Report":
    half_option = select_half()
    players = df["Player Name"].unique().tolist()
    selected_player = st.sidebar.selectbox("Select a Player", ["All"] + players)

    match_df = df.copy()
    if selected_player != "All":
        match_df = match_df[match_df["Player Name"] == selected_player]
    if half_option != "Total":
        match_df = match_df[match_df["Session Type"] == half_option]

    st.subheader("Key Metrics")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Avg Distance (km)", round(match_df["Total Distance"].mean(), 2))
    col2.metric("Avg Max Speed (km/h)", round(match_df["Max Speed"].mean(), 2))
    col3.metric("Avg Sprints", round(match_df["No of Sprints"].mean(), 2))
    col4.metric("Avg High Speed Running (m)", round(match_df["High Speed Running"].mean(), 2))

    st.subheader("Performance Charts")
    tab1, tab2, tab3, tab4 = st.tabs(["Event Data", "Bar Chart", "Fluctuation", "Radar Chart"])

    with tab1:
        st.header("Match Events and Shot Map")

        # --- Match selection ---
        event_files = {
            "All Matches (Average)": None,  # placeholder to trigger multi-match behavior
            "5.17 vs Birmingham Legion": "SSA USL2 v BHM Legion2 05.17.25.xlsx",
            "5.21 vs Tennessee SC": "SSA USL2 v TSC 05.21.25.xlsx",
            "5.25 vs East Atlanta FC": "SSA USL2 v EAFC 05.25.25.xlsx",

            # Add more matches and their corresponding file names here
        }

        event_images = {
        "5.17 vs Birmingham Legion": "SSAvBHM2 Event Image.png",
        "5.21 vs Tennessee SC": "Match Event Image SSAvTSC.png",
        "5.25 vs East Atlanta FC": "East Atlanta Event Data Screenshot.png",
        
        }

        selected_match = st.selectbox("Select a Match", list(event_files.keys()))
        # Load the event data
        if selected_match == "All Matches (Average)":
            combined_event_df = []
            for match, path in event_files.items():
                if path:  # skip placeholder None
                    try:
                        xls = pd.ExcelFile(path)
                        df_temp = xls.parse("Nacsport")
                        df_temp["Match"] = match
                        combined_event_df.append(df_temp)
                    except Exception as e:
                        st.warning(f"Could not load event data for {match}: {e}")
            if not combined_event_df:
                st.error("No event data available across matches.")
                st.stop()
            df_events = pd.concat(combined_event_df, ignore_index=True)
            selected_image_path = None
        else:
            try:
                xls_path = event_files[selected_match]
                xls = pd.ExcelFile(xls_path)
                df_events = xls.parse("Nacsport")
                selected_image_path = event_images.get(selected_match)
            except Exception as e:
                st.error(f"Failed to load data for {selected_match}: {e}")
                st.stop()


           
        try:
            # Load the event data from the selected file
            df_events = event_xls.parse("Nacsport")

            # Display event image
            st.subheader("Event Table")
            image_path = event_images[selected_match]
            event_image = Image.open(image_path)
            st.image(event_image, caption=f"Event Table for {selected_match}", use_container_width=True)

            # --- Event Table Display ---
            core_cols = ["Category", "Start", "End"]
            descriptor_cols = [col for col in df_events.columns if "Des" in str(col)]
            non_empty_des = [col for col in descriptor_cols if df_events[col].notna().sum() > 0][:4]
            event_df = df_events[core_cols + non_empty_des].dropna(subset=["Category", "Start", "End"], how="all")

            # --- Shot Map Plotting ---
            st.subheader("Shot Map")

            import matplotlib.pyplot as plt
            from PIL import Image

            # Loading Field Images
            field_image_path = "NS_camp_soccer_H.png"
            field_image = Image.open(field_image_path)
            field_width, field_height = field_image.size
            midfield_x = field_width / 2

            field_image_thirds = Image.open("NS_camp_soccer_V_T.png")
            field_width_thirds, field_height_thirds = field_image_thirds.size

            df_xy = df_events.dropna(subset=["XY"]).copy()
            df_xy[["X", "Y"]] = df_xy["XY"].str.split(";", expand=True).astype(int)

            # Tag shots as Goal vs Other
            descriptor_text = df_xy[descriptor_cols].astype(str).agg(" ".join, axis=1)
            df_xy["is_goal"] = descriptor_text.str.contains("goal", case=False, na=False)

            # Colors: green for goals, red otherwise
            df_xy["color"] = df_xy["is_goal"].map({True: "lime", False: "red"})

            # Plot
            fig, ax = plt.subplots(figsize=(12, 7))
            ax.imshow(field_image, extent=[0, field_width, 0, field_height])

            # Draw shots
            ax.scatter(
                df_xy["X"],
                field_height - df_xy["Y"],
                c=df_xy["color"],
                edgecolors='white',
                s=60
            )

            # Midfield Label
            ax.set_xlim(0, field_width)
            ax.set_ylim(0, field_height)
            ax.set_title(f"Shot Locations – {selected_match} (Swarm → Right)", color=primary_color)
            ax.axis('off')

            st.pyplot(fig)


        except Exception as e:
            st.error(f"Failed to load data for {selected_match}: {e}")

        # --- Cross Map ---
        st.subheader("Cross Map")

        # Filter for crosses with coordinates
        df_cross = df_events[(df_events["Category"].str.lower() == "cross") & df_events["XY"].notna()].copy()

        if not df_cross.empty:
            df_cross[["X", "Y"]] = df_cross["XY"].str.split(";", expand=True).astype(int)

            # Check descriptor text for outcomes
            cross_desc = df_cross[descriptor_cols].astype(str).agg(" ".join, axis=1).str.lower()
            df_cross["is_goal"] = cross_desc.str.contains("goal")
            df_cross["is_contact"] = cross_desc.str.contains("contact") & ~df_cross["is_goal"]

            # Assign color
            def classify_cross(row):
                if row["is_goal"]:
                    return "lime"    # green
                elif row["is_contact"]:
                    return "yellow"
                else:
                    return "red"

            df_cross["color"] = df_cross.apply(classify_cross, axis=1)

            # Plot crosses
            fig, ax = plt.subplots(figsize=(12, 7))
            ax.imshow(field_image, extent=[0, field_width, 0, field_height])
            ax.scatter(
                df_cross["X"], field_height - df_cross["Y"],
                c=df_cross["color"], edgecolors='white', s=60
            )
            ax.axvline(midfield_x, color='white', linestyle='--', linewidth=1)
            ax.text(midfield_x - 150, field_height - 20, "OPP Attacking", color='lightgreen', fontsize=12)
            ax.text(midfield_x + 50, field_height - 20, "Swarm Attacking", color='lightblue', fontsize=12)

            ax.set_xlim(0, field_width)
            ax.set_ylim(0, field_height)
            ax.set_title(f"Cross Map – {selected_match} (Green = Goal, Yellow = Contact, Red = Unsuccessful)", color=primary_color)
            ax.axis('off')
            st.pyplot(fig)
        else:
            st.info("No cross events with coordinates found in this match.")

        #Attacking Third Entries
        field_image_thirds = Image.open("NS_camp_soccer_V_T.png")
        field_width_thirds, field_height_thirds = field_image_thirds.size

        def draw_entry_third_map(df_events, field_image_thirds, field_width_thirds, field_height_thirds, team="Swarm"):

            st.markdown(f"#### {team} Attacking Third Entry Map")

            # Filter rows with XY
            df = df_events[df_events["XY"].notna()].copy()
            df[["X", "Y"]] = df["XY"].str.split(";", expand=True).astype(int)

            # Attacking third for Swarm = high X (right), for Opponent = low X (left)
            if team.lower() == "swarm":
                df = df[df["X"] >= field_width * 0.66]
            else:
                df = df[df["X"] <= field_width * 0.33]

            # Use X axis to divide into vertical thirds
            def get_third(x):
                if x <= field_width / 3:
                    return "Left"
                elif x <= 2 * field_width / 3:
                    return "Center"
                else:
                    return "Right"

            df["VerticalThird"] = df["X"].apply(get_third)

            # Count and normalize to %
            third_counts = df["VerticalThird"].value_counts(normalize=True) * 100
            for zone in ["Left", "Center", "Right"]:
                if zone not in third_counts:
                    third_counts[zone] = 0

            # Draw the field and overlays
            fig, ax = plt.subplots(figsize=(6, 10))
            ax.imshow(field_image_thirds, extent=[0, field_width_thirds, 0, field_height_thirds])


            # Vertical lines to split thirds
            x1 = field_width_thirds / 3
            x2 = 2 * field_width_thirds / 3
            ax.axvline(x1, color="white", linestyle="--", linewidth=1)
            ax.axvline(x2, color="white", linestyle="--", linewidth=1)

            # Label %s in each third
            label_positions = {
                "Left": x1 / 2,
                "Center": (x1 + x2) / 2,
                "Right": (x2 + field_width) / 2
            }

            for zone in ["Left", "Center", "Right"]:
                percent = round(third_counts[zone], 1)
                x = label_positions[zone]
                ax.text(x, field_height / 2, f"{zone}\n{percent}%", color="white", fontsize=14, ha="center", va="center")

            ax.set_xlim(0, field_width_thirds)
            ax.set_ylim(0, field_height_thirds)
            ax.axis("off")
            ax.set_title(f"{team} Attacking Third Entry Distribution", color=primary_color)
            st.pyplot(fig)



        # --- Two Regain Maps: Team vs Opponent ---
        st.subheader("Regain Maps")

        # Filter regain rows with coordinates
        df_regain = df_events[
            (df_events["Category"].str.lower() == "regain") & df_events["XY"].notna()
        ].copy()

        if not df_regain.empty:
            df_regain[["X", "Y"]] = df_regain["XY"].str.split(";", expand=True).astype(int)

            # Create descriptor text field for classification
            desc_text = df_regain[descriptor_cols].astype(str).agg(" ".join, axis=1).str.lower()

            # Outcome classification
            df_regain["is_goal"] = desc_text.str.contains("goal")
            df_regain["is_shot"] = desc_text.str.contains("shot") | df_regain["is_goal"]
            df_regain["color"] = df_regain.apply(
                lambda row: "lime" if row["is_goal"] else ("yellow" if row["is_shot"] else "red"),
                axis=1
            )

            # Split: Team vs Opponent
            df_team = df_regain[~desc_text.str.contains("opp")]
            df_opp = df_regain[desc_text.str.contains("opp")]

            # Plotting function
            def plot_regain_map(df, title):
                fig, ax = plt.subplots(figsize=(12, 7))
                ax.imshow(field_image, extent=[0, field_width, 0, field_height])
                ax.scatter(
                    df["X"], field_height - df["Y"],
                    c=df["color"], edgecolors='white', s=60
                )
                ax.axvline(midfield_x, color='white', linestyle='--', linewidth=1)
                ax.text(midfield_x - 150, field_height - 20, "OPP Attacking", color='lightgreen', fontsize=12)
                ax.text(midfield_x + 50, field_height - 20, "Swarm Attacking", color='lightblue', fontsize=12)
                ax.set_xlim(0, field_width)
                ax.set_ylim(0, field_height)
                ax.set_title(title, color=primary_color)
                ax.axis('off')
                st.pyplot(fig)

            # Plot both maps
            st.markdown("#### Swarm Regains")
            plot_regain_map(df_team, f"Swarm Ball Regains – {selected_match}")

            st.markdown("#### Opponent Regains")
            plot_regain_map(df_opp, f"Opponent Ball Regains – {selected_match}")

        else:
            st.info("No regains with coordinates found in this match.")

    

    with tab2:
         # Player performance metrics
        for metric in metrics:
            chart_data = match_df.groupby("Player Name")[metric].mean().reset_index()
            st.write(f"### {metric}")
            st.altair_chart(
                alt.Chart(chart_data).mark_bar().encode(
                    x=alt.X("Player Name", sort="-y"),
                    y=metric,
                    tooltip=["Player Name", metric]
                ).properties(height=300),
                use_container_width=True
                )



    with tab3:  # Fluctuation tab
        if selected_match == "All Matches (Average)":
            st.markdown("### Team Metric Fluctuations Over Matches")

            # Load and prepare all match data
            df_all_matches = []
            for label, path in match_files.items():
                temp_df = load_data(path)
                temp_df["Match"] = label
                df_all_matches.append(temp_df)
            full_df = pd.concat(df_all_matches, ignore_index=True)

            # Clean numeric columns
            for col in metrics:
                full_df[col] = pd.to_numeric(full_df[col], errors='coerce')

            # Team Average Over Matches
            for metric in metrics:
                st.write(f"#### {metric} (Team Avg)")
                avg_data = full_df.groupby("Match")[metric].mean().reset_index()
                chart = alt.Chart(avg_data).mark_line(point=True).encode(
                    x=alt.X("Match:N", sort=list(match_files.keys())),
                    y=alt.Y(f"{metric}:Q"),
                    tooltip=["Match", metric]
                ).properties(height=300)
                st.altair_chart(chart, use_container_width=True)

            # Player-Specific Trends
            st.markdown("### Player Trends")
            selected_players = st.multiselect(
                "Select Players to Plot", sorted(full_df["Player Name"].unique()),
                default=sorted(full_df["Player Name"].unique())[:3],
                key="fluctuation_players"
            )
            if selected_players:
                for metric in metrics:
                    st.write(f"#### {metric} (By Player)")
                    player_data = full_df[full_df["Player Name"].isin(selected_players)]
                    player_avg = player_data.groupby(["Match", "Player Name"])[metric].mean().reset_index()

                    chart = alt.Chart(player_avg).mark_line(point=True).encode(
                        x=alt.X("Match:N", sort=list(match_files.keys())),
                        y=alt.Y(f"{metric}:Q"),
                        color="Player Name:N",
                        tooltip=["Match", "Player Name", metric]
                    ).properties(height=350)
                    st.altair_chart(chart, use_container_width=True)
            else:
                st.info("Select one or more players to show player-specific trends.")
        else:
            st.info("Fluctuation charts are only available for 'All Matches (Average)'.")


    with tab4:
        if selected_player == "All":
            st.info("Select a player to view radar chart.")
        else:
            full_team = df[df["Session Type"] == half_option] if half_option != "Total" else df
            full_team.columns = full_team.columns.str.strip()
            radar_df = full_team.groupby("Player Name")[metrics].mean()
            fig = multi_player_percentile_radar(
                radar_df.loc[[selected_player]], radar_df, metrics, f"{selected_player} Percentile Radar"
            )
            st.pyplot(fig)
            
    


# Weekly Training Report
elif report_type == "Weekly Training Report":
    st.subheader("Weekly Training Report")
    selected_sessions = st.sidebar.multiselect("Select Training Sessions", list(training_files.keys()), default=list(training_files.keys()))

    if not selected_sessions:
        st.info("Please select at least one session.")
    else:
        df_list = []
        for s in selected_sessions:
            training_df = load_data(training_files[s])
            training_df["Session"] = s
            df_list.append(training_df)

        weekly_df = pd.concat(df_list, ignore_index=True)
        for col in metrics:
            weekly_df[col] = pd.to_numeric(weekly_df[col], errors='coerce')

        view_mode = st.sidebar.radio("View Mode", ["Team Overview", "Individual", "Compare Players"], key="weekly_mode")

        if view_mode == "Team Overview":

            st.markdown("### Team Averages")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Avg Distance (km)", round(weekly_df["Total Distance"].mean(), 2))
            col2.metric("Avg Max Speed (km/h)", round(weekly_df["Max Speed"].mean(), 2))
            col3.metric("Avg Sprints", round(weekly_df["No of Sprints"].mean(), 2))
            col4.metric("Avg High Speed Running (m)", round(weekly_df["High Speed Running"].mean(), 2))

            st.markdown("### Overall Team Averages")
            team_averages = weekly_df[metrics].mean().round(2)
            avg_df = pd.DataFrame(team_averages).reset_index()
            avg_df.columns = ["Metric", "Team Average"]
            st.table(avg_df)
            for metric in metrics:
                
                st.write(f"### {metric}")
                chart_df = weekly_df.groupby("Player Name")[metric].mean().reset_index()
                chart_df = chart_df.sort_values(by=metric, ascending=False)

                chart = alt.Chart(chart_df).mark_bar().encode(
                    x=alt.X("Player Name", sort=None),
                    y=alt.Y(metric),
                    tooltip=["Player Name", metric]
                ).properties(height=300)

                st.altair_chart(chart, use_container_width=True)

            st.markdown("### Team Fluctuation Over Sessions")
            for metric in metrics:
                line_df = weekly_df.groupby("Session")[metric].mean().reset_index()
                st.altair_chart(
                    alt.Chart(line_df).mark_line(point=True).encode(
                        x="Session:N",
                        y=metric,
                        tooltip=["Session", metric]
                    ).properties(height=300, title=metric),
                    use_container_width=True
                )

        elif view_mode == "Individual":
            player = st.sidebar.selectbox("Select Player", weekly_df["Player Name"].unique())
            stats = weekly_df[weekly_df["Player Name"] == player][metrics].mean().to_frame(name=player)
            st.table(stats)
            st.bar_chart(stats)

            st.markdown("### Player Fluctuation Over Sessions")
            player_line = weekly_df[weekly_df["Player Name"] == player]
            for metric in metrics:
                metric_line = player_line[["Session", metric]].groupby("Session").mean().reset_index()
                st.altair_chart(
                    alt.Chart(metric_line).mark_line(point=True).encode(
                        x="Session:N",
                        y=metric,
                        tooltip=["Session", metric]
                    ).properties(height=300, title=f"{metric}"),
                    use_container_width=True
                )

            full = weekly_df.groupby("Player Name")[metrics].mean()
            fig = multi_player_percentile_radar(full.loc[[player]], full, metrics, f"{player} Weekly Percentile Radar")
            st.pyplot(fig)

        elif view_mode == "Compare Players":
            players = weekly_df["Player Name"].unique()
            selected = st.sidebar.multiselect("Select 2–3 Players", players, default=list(players)[:2])
            if len(selected) not in [2, 3]:
                st.warning("Please select 2 or 3 players.")
            else:
                comp = weekly_df[weekly_df["Player Name"].isin(selected)]
                summary = comp.groupby("Player Name")[metrics].mean()
                team = weekly_df.groupby("Player Name")[metrics].mean()
                st.dataframe(summary)

                st.markdown("### Player Fluctuation Over Sessions")
                for metric in metrics:
                    trend_df = comp[["Session", "Player Name", metric]]
                    agg_df = trend_df.groupby(["Session", "Player Name"])[metric].mean().reset_index()
                    chart = alt.Chart(agg_df).mark_line(point=True).encode(
                        x="Session:N",
                        y=metric,
                        color="Player Name:N",
                        tooltip=["Session", "Player Name", metric]
                    ).properties(height=300, title=metric)
                    st.altair_chart(chart, use_container_width=True)

                for metric in metrics:
                    st.write(f"### {metric}")
                    chart_df = summary[[metric]].reset_index()
                    chart = alt.Chart(chart_df).mark_bar().encode(
                        x=alt.X("Player Name", sort=chart_df["Player Name"].tolist()),
                        y=alt.Y(metric),
                        tooltip=["Player Name", metric]
                    ).properties(height=300)
                    st.altair_chart(chart, use_container_width=True)
                fig = multi_player_percentile_radar(summary, team, metrics, "Weekly Percentile Radar")
                st.pyplot(fig)


elif report_type == "Daily Training Report":
    session = st.sidebar.selectbox("Select Session", list(training_files.keys()))
    df_daily = load_data(training_files[session])

    if df_daily.empty:
        st.error("This session file is empty or invalid.")
        st.stop()

    for col in metrics:
        df_daily[col] = pd.to_numeric(df_daily[col], errors='coerce')

    view = st.sidebar.radio("View Mode", ["Team Overview", "Individual", "Compare Players"])

    if view == "Team Overview":
        st.title(f"Daily Training Report - {session}")
        st.subheader("Key Metrics")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Avg Distance (km)", round(df_daily["Total Distance"].mean(), 2))
        col2.metric("Avg Max Speed (km/h)", round(df_daily["Max Speed"].mean(), 2))
        col3.metric("Avg Sprints", round(df_daily["No of Sprints"].mean(), 2))
        col4.metric("Avg High Speed Running (m)", round(df_daily["High Speed Running"].mean(), 2))
        for metric in metrics:
            st.write(f"### {metric}")
            chart_df = df_daily.groupby("Player Name")[metric].mean().reset_index()
            chart_df = chart_df.sort_values(by=metric, ascending=False)

            chart = alt.Chart(chart_df).mark_bar().encode(
                x=alt.X("Player Name", sort=None),
                y=alt.Y(metric),
                tooltip=["Player Name", metric]
            ).properties(height=300)

            st.altair_chart(chart, use_container_width=True)

    elif view == "Individual":
        player = st.sidebar.selectbox("Select Player", df_daily["Player Name"].unique())
        st.subheader(f"{player} - Individual Report")

        stats = df_daily[df_daily["Player Name"] == player][metrics].mean().to_frame(name="Value").reset_index()
        stats.rename(columns={"index": "Metric"}, inplace=True)

        st.write("### Metric Table")
        st.table(stats.set_index("Metric"))

        st.write("### Performance Overview")
        chart = alt.Chart(stats).mark_bar().encode(
            x="Metric",
            y="Value",
            tooltip=["Metric", "Value"]
        ).properties(height=300)
        st.altair_chart(chart, use_container_width=True)

        # Radar chart
        radar_df = df_daily.groupby("Player Name")[metrics].mean()
        fig = multi_player_percentile_radar(radar_df.loc[[player]], radar_df, metrics, f"{player} Radar Chart")
        st.pyplot(fig)

        # Fluctuation chart (if Session column exists)
        if "Session" in df_daily.columns:
            st.write("### Fluctuation Chart")
            for metric in metrics:
                line_df = df_daily[df_daily["Player Name"] == player][["Session", metric]]
                if not line_df.empty:
                    line_df = line_df.groupby("Session")[metric].mean().reset_index()
                    chart = alt.Chart(line_df).mark_line(point=True).encode(
                        x="Session:N", y=metric, tooltip=["Session", metric]
                    ).properties(height=300, title=metric)
                    st.altair_chart(chart, use_container_width=True)

    elif view == "Compare Players":
        st.subheader("Compare Players")
        options = df_daily["Player Name"].unique()
        selected = st.sidebar.multiselect("Select 2–3 Players", options, default=list(options)[:2])
        if 2 <= len(selected) <= 3:
            summary = df_daily[df_daily["Player Name"].isin(selected)].groupby("Player Name")[metrics].mean()
            team = df_daily.groupby("Player Name")[metrics].mean()

            st.dataframe(summary)

            for metric in metrics:
                st.write(f"### {metric}")
                chart_df = summary[[metric]].reset_index()
                chart = alt.Chart(chart_df).mark_bar().encode(
                    x=alt.X("Player Name", sort=chart_df["Player Name"].tolist()),
                    y=alt.Y(metric),
                    tooltip=["Player Name", metric]
                ).properties(height=300)
                st.altair_chart(chart, use_container_width=True)

            fig = multi_player_percentile_radar(summary, team, metrics, "Daily Training Radar")
            st.pyplot(fig)

            # Fluctuation chart (if Session exists)
            if "Session" in df_daily.columns:
                st.write("### Fluctuation Chart")
                for metric in metrics:
                    trend_df = df_daily[df_daily["Player Name"].isin(selected)][["Session", "Player Name", metric]]
                    agg_df = trend_df.groupby(["Session", "Player Name"])[metric].mean().reset_index()
                    chart = alt.Chart(agg_df).mark_line(point=True).encode(
                        x="Session:N", y=metric, color="Player Name:N",
                        tooltip=["Session", "Player Name", metric]
                    ).properties(height=300, title=metric)
                    st.altair_chart(chart, use_container_width=True)
        else:
            st.warning("Please select 2 or 3 players.")


# Match Compare Players
elif report_type == "Compare Players":
    half_option = select_half()
    players = df["Player Name"].unique().tolist()
    selected = st.sidebar.multiselect("Select 2–3 Players", players, default=players[:2])
    if len(selected) not in [2, 3]:
        st.warning("Select 2 or 3 players.")
    else:
        comp_df = df[df["Player Name"].isin(selected)]
        if half_option != "Total":
            comp_df = comp_df[comp_df["Session Type"] == half_option]
        summary = comp_df.groupby("Player Name")[metrics].mean()
        team_df = df[df["Session Type"] == half_option] if half_option != "Total" else df
        team_avg = team_df.groupby("Player Name")[metrics].mean()

        st.dataframe(summary)
    for metric in metrics:
        st.write(f"### {metric}")
        chart_df = summary[[metric]].reset_index()  # Player Name becomes a column

        chart = alt.Chart(chart_df).mark_bar().encode(
            x=alt.X("Player Name", sort=chart_df["Player Name"].tolist()),  # preserve order
            y=alt.Y(metric),
            tooltip=["Player Name", metric]
        ).properties(height=300)

        st.altair_chart(chart, use_container_width=True)
    fig = multi_player_percentile_radar(summary, team_avg, metrics, "Match Percentile Radar")
    st.pyplot(fig)

# Footer
st.markdown(f"<p style='text-align:center; color:{text_color};'>Built for SSA Swarm</p>", unsafe_allow_html=True)
