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
import openai
from datetime import datetime

st.set_page_config(page_title="SSA Swarm Gameday Report", layout="wide")

# Colors
primary_color = "#D72638"
background_color = "#1A1A1D"
text_color = "#FFFFFF"

# Enhanced CSS with centered metrics for event analysis
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
        
        /* Center metrics in event analysis */
        div[data-testid="metric-container"] {{
            text-align: center;
        }}
        
        div[data-testid="metric-container"] > div {{
            justify-content: center;
            text-align: center;
        }}
        
        div[data-testid="metric-container"] label {{
            text-align: center;
            width: 100%;
        }}
        
        div[data-testid="metric-container"] > div > div {{
            text-align: center;
        }}
        
        /* Event analysis specific centering */
        .event-metric {{
            text-align: center;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
        }}
        
        /* AI Assistant styling */
        .ai-assistant-box {{
            background-color: #2D2D30;
            border: 1px solid {primary_color};
            border-radius: 10px;
            padding: 20px;
            margin: 10px 0;
        }}
        
        .ai-assistant-header {{
            color: {primary_color};
            font-weight: bold;
            margin-bottom: 10px;
        }}
    </style>
    """,
    unsafe_allow_html=True
)

# AI Assistant Coach Function
def get_ai_coach_insights(context, data_summary, api_key):
    """Get AI coach insights based on current context and data"""
    if not api_key:
        return "Please enter your OpenAI API key in the sidebar to enable AI Coach insights."
    
    try:
        from openai import OpenAI
        
        # Initialize client with API key
        client = OpenAI(api_key=api_key)
        
        prompt = f"""
        You are an expert soccer coach analyzing performance data for SSA Swarm USL2 team.
        
        Current Context: {context}
        
        Data Summary:
        {data_summary}
        
        Please provide:
        1. Key insights from this data
        2. Tactical recommendations
        3. Areas of concern
        4. Positive trends to reinforce
        
        Keep your response concise and actionable, focusing on practical coaching insights.
        """
        
        # Try GPT-4 first, fall back to GPT-3.5-turbo if not available
        try:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert soccer performance analyst and coach."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=500
            )
            return response.choices[0].message.content
        except Exception as e:
            # If GPT-4 fails, try GPT-3.5-turbo
            if "model" in str(e).lower() or "404" in str(e):
                try:
                    response = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": "You are an expert soccer performance analyst and coach."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.7,
                        max_tokens=500
                    )
                    return response.choices[0].message.content
                except Exception as fallback_error:
                    return f"AI Coach unavailable: {str(fallback_error)}. Please check your API key and ensure you have credits available."
            else:
                return f"AI Coach unavailable: {str(e)}. Please check your API key and ensure you have credits available."
    
    except Exception as e:
        return f"AI Coach unavailable: {str(e)}. Please ensure you have the latest OpenAI library installed: pip install --upgrade openai"

# AI Assistant Display Component
def display_ai_assistant(context, data_summary, api_key):
    """Display AI assistant coach insights"""
    with st.expander("🤖 AI Assistant Coach", expanded=False):
        st.markdown('<div class="ai-assistant-box">', unsafe_allow_html=True)
        st.markdown('<div class="ai-assistant-header">AI Coach Analysis</div>', unsafe_allow_html=True)
        
        if st.button("Get AI Insights", key=f"ai_button_{context}"):
            with st.spinner("Analyzing data..."):
                insights = get_ai_coach_insights(context, data_summary, api_key)
                st.markdown(insights)
        
        st.markdown('</div>', unsafe_allow_html=True)

# File paths
match_files = {
    "5.17 vs Birmingham Legion 2": "SSA Swarm USL Mens 2 Games Statsports Reports - 5.17 SSA Swarm USL 2 vs Birmingham.csv",
    "5.21 vs Tennessee SC": "SSA Swarm USL Mens 2 Games Statsports Reports - 5.21 Swarm USL 2 Vs TN SC.csv",
    "5.25 vs East Atlanta FC": "SSA Swarm USL Mens 2 Games Statsports Reports - 5.25 SSA Swarm USL 2 vs East Atlanta.csv",
    "5.31 vs Dothan United SC": "SSA Swarm USL Mens 2 Games Statsports Reports - 5.31 Swarm USL 2 vs Dothan.csv",
    "6.4 vs Asheville City SC": "SSA Swarm USL Mens 2 Games Statsports Reports - 6.4 Swarm USL 2 vs Asheville.csv",
    "6.7 vs Dothan United SC": "SSA Swarm USL Mens 2 Games Statsports Reports - 6.7 Swarm USL 2 vs Dothan FC .csv"
}

training_files = {
    "5.12 Training": "SSA Swarm USL 2 Mens Statsports Traning Report  - 5.12 Training.csv",
    "5.13 Training": "SSA Swarm USL 2 Mens Statsports Traning Report  - 5.13 Training .csv",
    "5.15 Training": "SSA Swarm USL 2 Mens Statsports Traning Report  - 5.15 Training.csv",
    "5.16 Training": "SSA Swarm USL 2 Mens Statsports Traning Report  - 5.16 Training.csv",
    "5.19 Training": "SSA Swarm USL 2 Mens Statsports Traning Report  - 5.19 Training.csv",
    "5.20 Training": "SSA Swarm USL 2 Mens Statsports Traning Report  - 5.20 Training.csv",
    "6.3 Training": "SSA Swarm USL 2 Mens Statsports Traning Report  - 6.3 Training.csv",
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

# API Key input in sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("### 🤖 AI Assistant Settings")
api_key = st.sidebar.text_input("Enter OpenAI API Key:", type="password", help="Your OpenAI API key for AI Coach insights")

# Sidebar report type (move this up)
report_type = st.sidebar.selectbox("Select Report Type", [
    "Match Report", "Weekly Training Report", "Daily Training Report", "Compare Players"
])

# Conditional match + half selectors
selected_match = None
half_option = None
df = pd.DataFrame()

if report_type in ["Match Report", "Compare Players"]:
    match_options = ["All Matches (Average)"] + list(match_files.keys())
    selected_match = st.sidebar.selectbox("Select Match", match_options)

    if selected_match == "All Matches (Average)":
        df = pd.concat([load_data(path) for path in match_files.values()], ignore_index=True)
    else:
        df = load_data(match_files[selected_match])

    for col in metrics:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Page title
st.title(f"SSA Swarm USL2 Performance Center - {selected_match}")

# Match Report
if report_type == "Match Report":
    half_option = st.sidebar.selectbox("Select Half", ["Total", "First Half", "Second Half"], key="half_selectbox")
    players = df["Player Name"].unique().tolist()
    selected_player = st.sidebar.selectbox("Select a Player", ["All"] + players)

    match_df = df.copy()
    if selected_player != "All":
        match_df = match_df[match_df["Player Name"] == selected_player]
    if half_option != "Total":
        match_df = match_df[match_df["Session Type"] == half_option]

    st.subheader("Key Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    avg_distance = round(match_df["Total Distance"].mean(), 2)
    avg_speed = round(match_df["Max Speed"].mean(), 2)
    avg_sprints = round(match_df["No of Sprints"].mean(), 2)
    avg_hsr = round(match_df["High Speed Running"].mean(), 2)
    
    col1.metric("Avg Distance (km)", avg_distance)
    col2.metric("Avg Max Speed (km/h)", avg_speed)
    col3.metric("Avg Sprints", avg_sprints)
    col4.metric("Avg High Speed Running (m)", avg_hsr)

    # AI Assistant for Match Report Overview
    data_summary = f"""
    Match: {selected_match}
    Half: {half_option}
    Player: {selected_player}
    Average Distance: {avg_distance} km
    Average Max Speed: {avg_speed} km/h
    Average Sprints: {avg_sprints}
    Average High Speed Running: {avg_hsr} m
    """
    
    display_ai_assistant("Match Report Overview", data_summary, api_key)

    st.subheader("Performance Charts")
    tab1, tab2, tab3, tab4 = st.tabs(["Event Data", "Bar Chart", "Fluctuation", "Radar Chart"])

    with tab1:
        st.header("Match Event Table and Maps")

        # --- Match selection ---
        event_files = {
            "All Matches (Average)": None,  # placeholder to trigger multi-match behavior
            "5.17 vs Birmingham Legion 2": "SSA USL2 v BHM Legion2 05.17.25.xlsx",
            "5.21 vs Tennessee SC": "SSA USL2 v TSC 05.21.25.xlsx",
            "5.25 vs East Atlanta FC": "SSA USL2 v EAFC 05.25.25.xlsx",
            "5.31 vs Dothan United SC": "SSA v Dothan USL2 05.31.25.xlsx",
            "6.4 vs Asheville City SC": "SSA USL2 v Asheville City SC 06.04.25.xlsx",
            "6.7 vs Dothan United SC": "SSA USL2 v Dothan2 06.07.25.xlsx",
        }

        event_images = {
            "5.17 vs Birmingham Legion 2": "SSAvBHM2 Event Image.png",
            "5.21 vs Tennessee SC": "TSC New Event Image.png",
            "5.25 vs East Atlanta FC": "East Atlanta Event Data Screenshot.png",
            "5.31 vs Dothan United SC": "Dothan Event Image.png",
            "6.4 vs Asheville City SC": "Asheville Event Image.png",
            "6.7 vs Dothan United FC": "Dothan2 Event Image.png",
        }

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
                descriptor_cols = [col for col in df_events.columns if col.startswith("Des")]
                selected_image_path = event_images.get(selected_match)
            except Exception as e:
                st.error(f"Failed to load data for {selected_match}: {e}")
                st.stop()

        # --- Dynamic Match Summary Generator ---
        def generate_match_summary(df_events):
            df = df_events.copy()
            df.columns = df.columns.str.strip()
            df["Category"] = df["Category"].astype(str).str.strip()
            descriptor_cols = [col for col in df.columns if col.startswith("Des")]
            desc_text = df[descriptor_cols].astype(str).agg(" ".join, axis=1).str.lower()
        
            def count_events(keyword, is_opp=False, descriptors_only=None):
                mask = df["Category"].str.lower().str.contains(keyword.lower())
                if descriptors_only:
                    mask &= desc_text.str.contains("|".join(descriptors_only))
                if is_opp:
                    mask &= desc_text.str.contains("opp") | df["Category"].str.lower().str.contains("opp")
                else:
                    mask &= ~desc_text.str.contains("opp") & ~df["Category"].str.lower().str.contains("opp")
        
                # If averaging, return per-match average
                if selected_match == "All Matches (Average)":
                    match_counts = df[mask].groupby("Match").size()
                    return round(match_counts.mean(), 1)
                else:
                    return mask.sum()
        
            summary = {
                "Score": [
                    count_events("Shot", descriptors_only=["goal"]),
                    count_events("Shot", is_opp=True, descriptors_only=["goal"])
                ],
                "Shots On Target": [
                    count_events("Shot", descriptors_only=["goal", "save"]),
                    count_events("Shot", is_opp=True, descriptors_only=["goal", "save"])
                ],
                "Shots": [count_events("Shot"), count_events("Shot", True)],
                "Blocked Shots": [
                    count_events("Shot", descriptors_only=["block"]),
                    count_events("Shot", is_opp=True, descriptors_only=["block"])
                ],
                "Finishing Zone Entries":[count_events("FZE"), count_events("FZE",True)],
                "PAZ Entries": [count_events("PAZ"), count_events("PAZ", True)],
                "Crosses": [count_events("Cross"), count_events("Cross", True)],
                "Zone 3 Entries": [
                    count_events("A3E"),
                    count_events("A3E", is_opp=True)
                ],
                "Regains": [count_events("Regain"), count_events("Regain", True)],
                "Fouls Won": [
                    count_events("Foul Won"),
                    count_events("Foul Won", is_opp=True)
                ],
                "Corner Kicks": [count_events("Corner"), count_events("Corner", True)],
                "Free Kicks": [count_events("Free Kick"), count_events("Free Kick", True)],
                "Goal Kicks": [count_events("Goal Kick"), count_events("Goal Kick", True)],
            }
        
            return summary

        # --- Show Dynamic Summary Table with Centered Numbers ---
        st.markdown("### Match Summary")
        summary_stats = generate_match_summary(df_events)
        
        # Add single header row with centered styling
        col1, col2, col3 = st.columns([1.5, 2, 1.5])
        with col1:
            st.markdown("<div style='text-align:center; font-weight:bold; font-size:16px;'>Swarm</div>", unsafe_allow_html=True)
        with col2:
            st.markdown("<div style='text-align:center; font-weight:bold; font-size:16px;'></div>", unsafe_allow_html=True)
        with col3:
            st.markdown("<div style='text-align:center; font-weight:bold; font-size:16px;'>Opponent</div>", unsafe_allow_html=True)
        
        # Loop through stats once with centered numbers
        for stat, (swarm, opp) in summary_stats.items():
            col1, col2, col3 = st.columns([1.5, 2, 1.5])
        
            color_swarm = "#3CB371" if swarm > opp else "#D72638" if swarm < opp else "#A9A9A9"
            color_opp = "#3CB371" if opp > swarm else "#D72638" if opp < swarm else "#A9A9A9"
        
            font_size = "36px" if stat.lower() == "score" else "20px"
            box_style = "padding:5px 10px; border-radius:10px; display:inline-block;"
        
            with col1:
                st.markdown(
                    f"<div style='text-align:center; width:100%;'>"
                    f"<span style='background-color:{color_swarm}; color:white; {box_style} font-size:{font_size}; font-weight:bold;'>{swarm}</span>"
                    f"</div>",
                    unsafe_allow_html=True
                )
            with col2:
                st.markdown(
                    f"<div style='text-align:center; font-weight:bold; font-size:18px; padding-top:10px;'>{stat}</div>",
                    unsafe_allow_html=True
                )
            with col3:
                st.markdown(
                    f"<div style='text-align:center; width:100%;'>"
                    f"<span style='background-color:{color_opp}; color:white; {box_style} font-size:{font_size}; font-weight:bold;'>{opp}</span>"
                    f"</div>",
                    unsafe_allow_html=True
                )

        # AI Assistant for Event Analysis
        event_summary = "\n".join([f"{stat}: Swarm {swarm} - Opponent {opp}" for stat, (swarm, opp) in summary_stats.items()])
        event_data_summary = f"""
        Match Event Summary:
        {event_summary}
        
        Total Events Analyzed: {len(df_events)}
        """
        
        display_ai_assistant("Match Event Analysis", event_data_summary, api_key)

        try:
            # Display event image
            st.subheader("Event Table")
            image_path = event_images[selected_match]
            event_image = Image.open(image_path)
            st.image(event_image, caption=f"Event Table for {selected_match}", use_container_width=True)

        except Exception as e:
            st.error(f"Error displaying event table: {e}")

        # --- Sub-Category Tables ---
        def show_event_subtable(df_events, category_keywords, title):
            df = df_events.copy()
            df.columns = df.columns.str.strip()
            df["Category"] = df["Category"].astype(str).str.strip()
            descriptor_cols = [col for col in df.columns if col.startswith("Des")]
            desc_text = df[descriptor_cols].astype(str).agg(" ".join, axis=1).str.lower()
        
            mask = pd.Series([False] * len(df))
            for kw in category_keywords:
                mask |= df["Category"].str.lower().str.contains(kw.lower())
                mask |= desc_text.str.contains(kw.lower())
        
            sub_df = df[mask].copy()
            core_cols = ["Category", "Start", "End"]
            des_cols = [col for col in df.columns if col.startswith("Des") and df[col].notna().sum() > 0][:4]
            sub_df = sub_df[core_cols + des_cols + (["XY"] if "XY" in df.columns else [])]
        
            if not sub_df.empty:
                with st.expander(f"▶️ {title} ({len(sub_df)} events)"):
                    st.dataframe(sub_df.reset_index(drop=True), use_container_width=True)
            else:
                with st.expander(f"▶️ {title} (No events)"):
                    st.info(f"No {title.lower()} found.")
        
        # --- Call for each sub-table ---
        st.markdown("### Event Sub-Tables")
        show_event_subtable(df_events, ["cross"], "Crosses")
        show_event_subtable(df_events, ["free kick"], "Free Kicks")
        show_event_subtable(df_events, ["corner"], "Corner Kicks")
        show_event_subtable(df_events, ["throw"], "Throw-ins")
        show_event_subtable(df_events, ["regain"], "Regains")
        show_event_subtable(df_events, ["paz"], "PAZ Entries")
        show_event_subtable(df_events, ["a3e"], "Zone 3 Entries")
        show_event_subtable(df_events, ["goal kick"], "Goal Kicks")

        # --- Shot Map Plotting ---
        st.subheader("Shot Map")

        import matplotlib.pyplot as plt
        from PIL import Image
        
        try:
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
                    
                # AI Assistant for Fluctuation Analysis
                player_trends_summary = f"""
                Players analyzed: {', '.join(selected_players)}
                Matches analyzed: {len(match_files)}
                Metrics tracked: {', '.join(metrics)}
                """
                
                display_ai_assistant("Player Fluctuation Analysis", player_trends_summary, api_key)
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
            
            # AI Assistant for Radar Analysis
            player_percentiles = []
            for metric in metrics:
                all_values = radar_df[metric]
                player_value = radar_df.loc[selected_player, metric]
                percentile = (all_values < player_value).sum() / len(all_values) * 100
                player_percentiles.append(f"{metric}: {percentile:.0f}th percentile")
            
            radar_summary = f"""
            Player: {selected_player}
            Percentile Rankings:
            {chr(10).join(player_percentiles)}
            """
            
            display_ai_assistant("Player Radar Analysis", radar_summary, api_key)

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
            
            team_avg_distance = round(weekly_df["Total Distance"].mean(), 2)
            team_avg_speed = round(weekly_df["Max Speed"].mean(), 2)
            team_avg_sprints = round(weekly_df["No of Sprints"].mean(), 2)
            team_avg_hsr = round(weekly_df["High Speed Running"].mean(), 2)
            
            col1.metric("Avg Distance (km)", team_avg_distance)
            col2.metric("Avg Max Speed (km/h)", team_avg_speed)
            col3.metric("Avg Sprints", team_avg_sprints)
            col4.metric("Avg High Speed Running (m)", team_avg_hsr)

            st.markdown("### Overall Team Averages")
            team_averages = weekly_df[metrics].mean().round(2)
            avg_df = pd.DataFrame(team_averages).reset_index()
            avg_df.columns = ["Metric", "Team Average"]
            st.table(avg_df)
            
            # AI Assistant for Team Overview
            team_overview_summary = f"""
            Weekly Training Overview:
            Sessions analyzed: {', '.join(selected_sessions)}
            Total players: {weekly_df['Player Name'].nunique()}
            Team averages:
            - Distance: {team_avg_distance} km
            - Max Speed: {team_avg_speed} km/h
            - Sprints: {team_avg_sprints}
            - High Speed Running: {team_avg_hsr} m
            """
            
            display_ai_assistant("Weekly Team Training Analysis", team_overview_summary, api_key)
            
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
            player_data = weekly_df[weekly_df["Player Name"] == player]
            stats = player_data[metrics].mean().to_frame(name=player)
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
            
            # AI Assistant for Individual Player
            player_stats_summary = f"""
            Player: {player}
            Weekly averages:
            {chr(10).join([f"- {metric}: {stats.loc[metric, player]:.2f}" for metric in metrics])}
            Sessions completed: {player_data['Session'].nunique()}
            """
            
            display_ai_assistant("Individual Player Analysis", player_stats_summary, api_key)

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
                
                # AI Assistant for Player Comparison
                comparison_summary = f"""
                Players compared: {', '.join(selected)}
                Key differences:
                {chr(10).join([f"- {metric}: " + ', '.join([f"{p} ({summary.loc[p, metric]:.1f})" for p in selected]) for metric in metrics[:3]])}
                """
                
                display_ai_assistant("Player Comparison Analysis", comparison_summary, api_key)

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
        
        daily_avg_distance = round(df_daily["Total Distance"].mean(), 2)
        daily_avg_speed = round(df_daily["Max Speed"].mean(), 2)
        daily_avg_sprints = round(df_daily["No of Sprints"].mean(), 2)
        daily_avg_hsr = round(df_daily["High Speed Running"].mean(), 2)
        
        col1.metric("Avg Distance (km)", daily_avg_distance)
        col2.metric("Avg Max Speed (km/h)", daily_avg_speed)
        col3.metric("Avg Sprints", daily_avg_sprints)
        col4.metric("Avg High Speed Running (m)", daily_avg_hsr)
        
        # AI Assistant for Daily Training
        daily_summary = f"""
        Training Session: {session}
        Players present: {df_daily['Player Name'].nunique()}
        Team averages:
        - Distance: {daily_avg_distance} km
        - Max Speed: {daily_avg_speed} km/h
        - Sprints: {daily_avg_sprints}
        - High Speed Running: {daily_avg_hsr} m
        """
        
        display_ai_assistant("Daily Training Analysis", daily_summary, api_key)
        
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
    half_option = st.sidebar.selectbox("Select Half", ["Total", "First Half", "Second Half"], key="half_compare")
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
        
        # AI Assistant for Match Comparison
        match_comparison_summary = f"""
        Match Player Comparison:
        Players: {', '.join(selected)}
        Half: {half_option}
        Performance comparison:
        {chr(10).join([f"- {metric}: " + ', '.join([f"{p} ({summary.loc[p, metric]:.1f})" for p in selected]) for metric in metrics[:4]])}
        """
        
        display_ai_assistant("Match Player Comparison", match_comparison_summary, api_key)
        
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
st.markdown(f"<p style='text-align:center; color:{text_color};'>Built for SSA Swarm | Enhanced with AI Assistant Coach</p>", unsafe_allow_html=True)
