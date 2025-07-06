# VERY IMPORTANT
# source .venv/bin/activate
# TYPE THAT INTO TERMINAL

from teams_config import TEAMS_CONFIG
import streamlit as st
import pandas as pd
import altair as alt
import matplotlib.pyplot as plt
import numpy as np
from math import pi
from PIL import Image, ImageDraw
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

try:
    import openai
except ImportError:
    st.error("Please install the OpenAI library: pip install openai")
    openai = None

# Page Configuration
st.set_page_config(
    page_title="Swarm Performance Analytics Hub", 
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "SSA Swarm Performance Analytics Dashboard"
    }
)

# Theme Configuration
class ThemeConfig:
    PRIMARY_COLOR = "#D72638"
    SECONDARY_COLOR = "#3CB371"
    BACKGROUND_COLOR = "#1A1A1D"
    CARD_BACKGROUND = "#2D2D30"
    TEXT_COLOR = "#FFFFFF"
    ACCENT_COLOR = "#FF6B6B"
    SUCCESS_COLOR = "#4ECDC4"
    WARNING_COLOR = "#FFE66D"

# Apply Custom CSS
st.markdown(f"""
    <style>
        /* Main container styling */
        .main {{
            background-color: {ThemeConfig.BACKGROUND_COLOR};
            color: {ThemeConfig.TEXT_COLOR};
        }}
        
        /* Sidebar styling */
        .css-1d391kg {{
            background-color: {ThemeConfig.CARD_BACKGROUND};
        }}
        
        /* Headers styling */
        h1, h2, h3, h4, h5, h6 {{
            color: {ThemeConfig.PRIMARY_COLOR};
            font-weight: 600;
        }}
        
        /* Metric cards */
        div[data-testid="metric-container"] {{
            background-color: {ThemeConfig.CARD_BACKGROUND};
            border: 1px solid {ThemeConfig.PRIMARY_COLOR};
            padding: 15px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }}
        
        /* Dataframe styling */
        .dataframe {{
            background-color: {ThemeConfig.CARD_BACKGROUND};
            color: {ThemeConfig.TEXT_COLOR};
        }}
        
        /* Tab styling */
        .stTabs [data-baseweb="tab-list"] {{
            background-color: {ThemeConfig.CARD_BACKGROUND};
            border-radius: 10px;
        }}
        
        /* Button styling */
        .stButton > button {{
            background-color: {ThemeConfig.PRIMARY_COLOR};
            color: white;
            border: none;
            border-radius: 5px;
            padding: 10px 20px;
            font-weight: 600;
            transition: all 0.3s;
        }}
        
        .stButton > button:hover {{
            background-color: {ThemeConfig.SECONDARY_COLOR};
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        }}
        
        /* Expander styling */
        .streamlit-expanderHeader {{
            background-color: {ThemeConfig.CARD_BACKGROUND};
            color: {ThemeConfig.TEXT_COLOR};
            border-radius: 5px;
        }}
        
        /* Custom info box */
        .info-box {{
            background-color: {ThemeConfig.CARD_BACKGROUND};
            padding: 20px;
            border-radius: 10px;
            border-left: 4px solid {ThemeConfig.PRIMARY_COLOR};
            margin: 10px 0;
        }}
        
        /* Custom metric box */
        .custom-metric {{
            text-align: center;
            padding: 20px;
            background-color: {ThemeConfig.CARD_BACKGROUND};
            border-radius: 10px;
            border: 1px solid {ThemeConfig.PRIMARY_COLOR};
            transition: all 0.3s;
        }}
        
        .custom-metric:hover {{
            transform: translateY(-5px);
            box-shadow: 0 5px 15px rgba(215, 38, 56, 0.3);
        }}
        
        .metric-value {{
            font-size: 2.5em;
            font-weight: bold;
            color: {ThemeConfig.PRIMARY_COLOR};
        }}
        
        .metric-label {{
            font-size: 1.1em;
            color: {ThemeConfig.TEXT_COLOR};
            opacity: 0.8;
        }}

        /* Team card styling */
        .team-card {{
            background-color: {ThemeConfig.CARD_BACKGROUND};
            border: 2px solid {ThemeConfig.PRIMARY_COLOR};
            border-radius: 15px;
            padding: 30px;
            text-align: center;
            transition: all 0.3s;
            cursor: pointer;
            margin: 10px;
        }}
        
        .team-card:hover {{
            transform: translateY(-10px);
            box-shadow: 0 10px 30px rgba(215, 38, 56, 0.4);
            border-color: {ThemeConfig.SECONDARY_COLOR};
        }}
        
        .team-logo {{
            width: 150px;
            height: 150px;
            margin: 0 auto 20px;
            border-radius: 50%;
            overflow: hidden;
            border: 3px solid {ThemeConfig.PRIMARY_COLOR};
        }}
        
        .team-name {{
            font-size: 1.8em;
            font-weight: bold;
            color: {ThemeConfig.PRIMARY_COLOR};
            margin-bottom: 10px;
        }}
        
        .team-description {{
            color: {ThemeConfig.TEXT_COLOR};
            opacity: 0.8;
            font-size: 1em;
        }}

        
    </style>
""", unsafe_allow_html=True)



# Constants
REQUIRED_COLUMNS = [
    "Player Name", "Session Type", "Total Distance", "Max Speed", "No of Sprints",
    "Sprint Distance", "Accelerations", "Decelerations", "High Speed Running"
]
METRICS = REQUIRED_COLUMNS[2:]

# Enhanced metric descriptions
METRIC_DESCRIPTIONS = {
    "Total Distance": "Total distance covered in kilometers",
    "Max Speed": "Maximum speed achieved in km/h",
    "No of Sprints": "Number of high-speed runs",
    "Sprint Distance": "Total distance covered while sprinting",
    "Accelerations": "Number of rapid speed increases",
    "Decelerations": "Number of rapid speed decreases",
    "High Speed Running": "Distance covered at high speed (meters)"
}


# Then define TEAMS_CONFIG using the above dictionaries
TEAMS_CONFIG = {
    "SSA Swarm USL2": {
        "logo": "SSALogoTransparent.jpeg",
        "description": "USL2 Performance Center",
        "match_files": MATCH_FILES,
        "training_files": TRAINING_FILES,
        "event_files": EVENT_FILES,
        "event_images": EVENT_IMAGES
    }
    # Add more teams here in the same format
}

# AI Assistant Coach Function
def get_ai_coach_insights(context, data_summary, api_key):
    """Get AI coach insights based on current context and data"""
    if not api_key:
        return "Please enter your OpenAI API key in the sidebar to enable AI Coach insights."
    
    if openai is None:
        return "OpenAI library not installed. Please run: pip install openai"
    
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
    with st.expander(" AI Assistant Coach", expanded=False):
        # Create a unique key based on context
        button_key = f"ai_button_{context.replace(' ', '_').replace('/', '_').replace(':', '_')}"
        
        if st.button("Get AI Insights", key=button_key):
            with st.spinner("Analyzing data..."):
                insights = get_ai_coach_insights(context, data_summary, api_key)
                # Store insights in session state
                st.session_state[f"insights_{button_key}"] = insights
        
        # Display insights if they exist in session state
        if f"insights_{button_key}" in st.session_state:
            st.markdown(st.session_state[f"insights_{button_key}"])

# Utility Functions
@st.cache_data
def load_data(path):
    """Load and preprocess CSV data with caching"""
    try:
        # Check if file exists
        if not os.path.exists(path):
            st.error(f"File not found: {path}")
            return pd.DataFrame()
            
        df = pd.read_csv(path)
        df.columns = df.columns.str.strip()
        df.dropna(subset=["Player Name"], inplace=True)
        df["Session Type"] = df["Session Type"].astype(str).str.strip().str.title()
        
        # Convert metrics to numeric
        for col in METRICS:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    except Exception as e:
        st.error(f"Error loading {path}: {str(e)}")
        return pd.DataFrame()

def create_circular_image(image_path):
    """Create a circular version of an image"""
    try:
        # Check if file exists first
        if not os.path.exists(image_path):
            print(f"Warning: Image not found at {image_path}")
            return None
            
        img = Image.open(image_path).convert("RGBA")
        size = min(img.size)
        mask = Image.new('L', (size, size), 0)
        draw = ImageDraw.Draw(mask)
        draw.ellipse((0, 0, size, size), fill=255)
        
        left = (img.width - size) // 2
        top = (img.height - size) // 2
        img_cropped = img.crop((left, top, left + size, top + size))
        img_cropped.putalpha(mask)
        return img_cropped
    except Exception as e:
        print(f"Error loading image {image_path}: {str(e)}")
        return None

def create_metric_card(label, value, delta=None, delta_color="normal"):
    """Create a custom metric card"""
    delta_html = ""
    if delta is not None:
        delta_symbol = "↑" if delta > 0 else "↓" if delta < 0 else "→"
        delta_color_code = ThemeConfig.SUCCESS_COLOR if delta > 0 else ThemeConfig.PRIMARY_COLOR if delta < 0 else ThemeConfig.TEXT_COLOR
        delta_html = f'<div style="color: {delta_color_code}; font-size: 0.9em;">{delta_symbol} {abs(delta):.1f}%</div>'
    
    return f"""
    <div class="custom-metric">
        <div class="metric-value">{value}</div>
        <div class="metric-label">{label}</div>
        {delta_html}
    </div>
    """

def create_plotly_radar(data, categories, title, names=None):
    """Create an enhanced Plotly radar chart"""
    fig = go.Figure()
    
    if names is None:
        names = data.index.tolist() if hasattr(data, 'index') else [f"Player {i+1}" for i in range(len(data))]
    
    colors = [ThemeConfig.PRIMARY_COLOR, ThemeConfig.SECONDARY_COLOR, ThemeConfig.ACCENT_COLOR]
    
    for i, (name, values) in enumerate(zip(names, data.values if hasattr(data, 'values') else data)):
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name=name,
            line_color=colors[i % len(colors)],
            fillcolor=colors[i % len(colors)],
            opacity=0.6
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickfont=dict(color=ThemeConfig.TEXT_COLOR)
            ),
            angularaxis=dict(
                tickfont=dict(color=ThemeConfig.TEXT_COLOR)
            ),
            bgcolor=ThemeConfig.BACKGROUND_COLOR
        ),
        showlegend=True,
        title=dict(
            text=title,
            font=dict(color=ThemeConfig.PRIMARY_COLOR, size=20)
        ),
        paper_bgcolor=ThemeConfig.BACKGROUND_COLOR,
        plot_bgcolor=ThemeConfig.BACKGROUND_COLOR,
        font=dict(color=ThemeConfig.TEXT_COLOR),
        height=500
    )
    
    return fig

def calculate_percentile_values(selected_df, full_df, metrics):
    """Calculate percentile values for radar charts"""
    percentile_df = selected_df.copy()
    for metric in metrics:
        all_values = full_df[metric].dropna()
        percentile_df[metric] = selected_df[metric].apply(
            lambda val: (all_values < val).sum() / len(all_values) * 100 if pd.notna(val) else 0
        )
    return percentile_df

def create_performance_summary(df, player_name=None):
    """Create a performance summary with key insights"""
    if player_name:
        df = df[df["Player Name"] == player_name]
    
    summary = {
        "Total Distance": f"{df['Total Distance'].mean():.2f} km",
        "Max Speed": f"{df['Max Speed'].mean():.2f} km/h",
        "Sprint Count": f"{df['No of Sprints'].mean():.0f}",
        "High Speed Running": f"{df['High Speed Running'].mean():.0f} m",
        "Work Rate": f"{(df['Accelerations'].mean() + df['Decelerations'].mean()):.0f} actions"
    }
    
    return summary

#Sidebar Configuration
def setup_sidebar(team_config):
    """Configure the sidebar with logo and navigation"""
    # Back to teams button
    if st.sidebar.button("← Back to Teams", use_container_width=True):
        st.session_state.selected_team = None
        st.rerun()
    
    st.sidebar.markdown("---")
    
    # Team logo
    try:
        logo = create_circular_image(team_config["logo"])
        if logo:
            st.sidebar.image(logo, width=200)
        else:
            st.sidebar.markdown(f"<h1 style='color: {ThemeConfig.PRIMARY_COLOR}; text-align: center;'>{st.session_state.selected_team}</h1>", unsafe_allow_html=True)
    except:
        st.sidebar.markdown(f"<h1 style='color: {ThemeConfig.PRIMARY_COLOR}; text-align: center;'>{st.session_state.selected_team}</h1>", unsafe_allow_html=True)
    
    st.sidebar.markdown("---")
    
    # API Key input
    st.sidebar.markdown("### 🤖 AI Assistant Settings")
    api_key = st.sidebar.text_input("Enter OpenAI API Key:", type="password", help="Your OpenAI API key for AI Coach insights")
    
    st.sidebar.markdown("---")
    
    # Report type selection
    report_type = st.sidebar.selectbox(
        "📊 Select Report Type",
        ["Match Report", "Weekly Training Report", "Daily Training Report", "Compare Players", "Player Profile"],
        help="Choose the type of analysis you want to view"
    )
    
    return report_type, api_key


# Landing Page Functions
def render_landing_page():
    """Render the landing page for team selection"""
    # Header
    st.markdown(f"""
    <h1 style='text-align: center; color: {ThemeConfig.PRIMARY_COLOR}; font-size: 3.5em; margin-bottom: 0;'>
        SSA Performance Analytics Hub
    </h1>
    <p style='text-align: center; color: {ThemeConfig.TEXT_COLOR}; opacity: 0.8; font-size: 1.3em; margin-bottom: 50px;'>
        Select a team to access their performance dashboard
    </p>
    """, unsafe_allow_html=True)
    
    # Calculate number of rows needed
    teams_list = list(TEAMS_CONFIG.items())
    num_teams = len(teams_list)
    teams_per_row = 3
    
    # Create rows of teams
    for row_idx in range(0, num_teams, teams_per_row):
        cols = st.columns(teams_per_row)
        
        # Get teams for this row
        row_teams = teams_list[row_idx:row_idx + teams_per_row]
        
        for col_idx, (team_name, team_config) in enumerate(row_teams):
            with cols[col_idx]:
                # Create team card button first (full width)
                if st.button(f"Select {team_name}", key=f"team_{team_name}", use_container_width=True):
                    st.session_state.selected_team = team_name
                    st.rerun()
            
            # Team logo - centered using columns
            logo_col1, logo_col2, logo_col3 = st.columns([1, 2, 1])
            with logo_col2:
                try:
                    logo = create_circular_image(team_config["logo"])
                    if logo:
                        st.image(logo, use_container_width=True)
                    else:
                        st.markdown(f"""
                        <div style='width: 150px; height: 150px; background-color: {ThemeConfig.PRIMARY_COLOR}; 
                                    border-radius: 50%; margin: 0 auto; display: flex; 
                                    align-items: center; justify-content: center;'>
                            <h2 style='color: white; margin: 0;'>{team_name[:3]}</h2>
                        </div>
                        """, unsafe_allow_html=True)
                except:
                    st.markdown(f"""
                    <div style='width: 150px; height: 150px; background-color: {ThemeConfig.PRIMARY_COLOR}; 
                                border-radius: 50%; margin: 0 auto; display: flex; 
                                align-items: center; justify-content: center;'>
                        <h2 style='color: white; margin: 0;'>{team_name[:3]}</h2>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Team info - ensure text is centered
            st.markdown(f"""
            <div style='text-align: center; width: 100%;'>
                <h3 style='color: {ThemeConfig.PRIMARY_COLOR}; margin: 10px 0;'>{team_name}</h3>
                <p style='color: {ThemeConfig.TEXT_COLOR}; opacity: 0.8; margin: 5px 0;'>{team_config['description']}</p>
                <p style='color: {ThemeConfig.ACCENT_COLOR}; margin: 5px 0;'>
                    {len(team_config['match_files'])} Matches | {len(team_config['training_files'])} Training Sessions
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)

# Main Application
def main():
    """Main application logic"""
    # Initialize session state
    if 'selected_team' not in st.session_state:
        st.session_state.selected_team = None
    
    # Show landing page or team dashboard
    if st.session_state.selected_team is None:
        render_landing_page()
    else:
        # Get team configuration
        team_config = TEAMS_CONFIG.get(st.session_state.selected_team)
        if not team_config:
            st.error(f"Team configuration not found for {st.session_state.selected_team}")
            st.session_state.selected_team = None
            st.rerun()
            return
        
        # Setup sidebar with team config
        report_type, api_key = setup_sidebar(team_config)
        
        # Header
        st.markdown(f"""
        <h1 style='text-align: center; color: {ThemeConfig.PRIMARY_COLOR}; font-size: 3em; margin-bottom: 0;'>
            {st.session_state.selected_team} Performance Analytics
        </h1>
        <p style='text-align: center; color: {ThemeConfig.TEXT_COLOR}; opacity: 0.8; font-size: 1.2em;'>
            {team_config['description']} - Advanced Analytics Dashboard with AI Insights
        </p>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Route to appropriate report with team config
        if report_type == "Match Report":
            render_match_report(api_key, team_config)
        elif report_type == "Weekly Training Report":
            render_weekly_training_report(api_key, team_config)
        elif report_type == "Daily Training Report":
            render_daily_training_report(api_key, team_config)
        elif report_type == "Compare Players":
            render_player_comparison(api_key, team_config)
        elif report_type == "Player Profile":
            render_player_profile(api_key, team_config)
        
        # Footer
        st.markdown("---")
        st.markdown(f"""
        <p style='text-align: center; color: {ThemeConfig.TEXT_COLOR}; opacity: 0.6;'>
            Built for SSA | Powered by Statsports & Nacsport | Enhanced with AI Insights | © 2025
        </p>
         """, unsafe_allow_html=True)

def render_match_report(api_key, team_config):
    """Render the match report section"""
    # Get team-specific files
    MATCH_FILES = team_config.get("match_files", {})
    EVENT_FILES = team_config.get("event_files", {})
    EVENT_IMAGES = team_config.get("event_images", {})
    
    if not MATCH_FILES:
        st.warning("No match files configured for this team.")
        return
        
    # Match selection
    match_files = team_config["match_files"]
    match_options = ["All Matches (Average)"] + list(match_files.keys())
    selected_match = st.sidebar.selectbox("Select Match", match_options)
    
    # Load data
    if selected_match == "All Matches (Average)":
        df = pd.concat([load_data(path) for path in MATCH_FILES.values()], ignore_index=True)
    else:
        df = load_data(MATCH_FILES[selected_match])
    
    if df.empty:
        st.error("No data available for the selected match.")
        return
    
    # Half selection
    half_option = st.sidebar.selectbox("Select Half", ["Total", "First Half", "Second Half"])
    
    # Player selection
    players = df["Player Name"].unique().tolist()
    selected_player = st.sidebar.selectbox("Select Player", ["All"] + players)
    
    # Filter data
    match_df = df.copy()
    if selected_player != "All":
        match_df = match_df[match_df["Player Name"] == selected_player]
    if half_option != "Total":
        match_df = match_df[match_df["Session Type"] == half_option]
    
    # Display match title
    st.markdown(f"<h2 style='color: {ThemeConfig.PRIMARY_COLOR};'> {selected_match}</h2>", unsafe_allow_html=True)
    
    # Key Metrics Dashboard
    st.markdown("### Key Performance Indicators")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    # Calculate metrics with comparisons
    avg_distance = match_df["Total Distance"].mean()
    avg_speed = match_df["Max Speed"].mean()
    avg_sprints = match_df["No of Sprints"].mean()
    avg_hsr = match_df["High Speed Running"].mean()
    work_rate = (match_df["Accelerations"].mean() + match_df["Decelerations"].mean())
    
    # Display metrics
    with col1:
        st.markdown(create_metric_card("Distance", f"{avg_distance:.2f} km"), unsafe_allow_html=True)
    with col2:
        st.markdown(create_metric_card("Max Speed", f"{avg_speed:.1f} km/h"), unsafe_allow_html=True)
    with col3:
        st.markdown(create_metric_card("Sprints", f"{avg_sprints:.0f}"), unsafe_allow_html=True)
    with col4:
        st.markdown(create_metric_card("HSR", f"{avg_hsr:.0f} m"), unsafe_allow_html=True)
    with col5:
        st.markdown(create_metric_card("Work Rate", f"{work_rate:.0f}"), unsafe_allow_html=True)
    
    # Add spacing between metrics and AI assistant
    st.markdown("<br>", unsafe_allow_html=True)
    
    # AI Assistant for Match Overview
    match_summary = f"""
    Match: {selected_match}
    Half: {half_option}
    Player: {selected_player}
    Team Averages:
    - Distance: {avg_distance:.2f} km
    - Max Speed: {avg_speed:.1f} km/h
    - Sprints: {avg_sprints:.0f}
    - High Speed Running: {avg_hsr:.0f} m
    - Work Rate: {work_rate:.0f} actions/game
    """
    
    display_ai_assistant("Match Overview Analysis", match_summary, api_key)
    
    # Add spacing between AI assistant and tabs
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Tabbed content
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["  Event Analysis", "Performance Charts", "Trends", "Radar Analysis", "Data Table"])
    
    with tab1:
        render_event_analysis(selected_match, api_key, EVENT_FILES, EVENT_IMAGES)
    
    with tab2:
        render_performance_charts(match_df, api_key)
    
    with tab3:
        render_trend_analysis(selected_match, match_df, api_key, MATCH_FILES)
    
    with tab4:
        render_radar_analysis(match_df, df, half_option, selected_player, api_key)
    
    with tab5:
        render_data_table(match_df)

def render_event_analysis(selected_match, api_key, EVENT_FILES, EVENT_IMAGES):
    """Render event analysis section - now works for all matches including average"""
    st.markdown("### Match Event Analysis")
    
    try:
        if selected_match == "All Matches (Average)":
            # Load all event data
            all_events = []
            for match, file_path in EVENT_FILES.items():
                try:
                    xls = pd.ExcelFile(file_path)
                    df_temp = xls.parse("Nacsport")
                    df_temp["Match"] = match
                    all_events.append(df_temp)
                except Exception as e:
                    st.warning(f"Could not load event data for {match}: {e}")
            
            if not all_events:
                st.error("No event data available across matches.")
                return
            
            df_events = pd.concat(all_events, ignore_index=True)
            
            # Generate aggregated match summary
            summary_stats = generate_enhanced_match_summary_aggregate(df_events)
            
            st.markdown("#### Average Match Summary (All Matches)")
            
        else:
            # Load single match event data
            xls_path = EVENT_FILES.get(selected_match)
            if not xls_path:
                st.warning("Event data not available for this match.")
                return
            
            xls = pd.ExcelFile(xls_path)
            df_events = xls.parse("Nacsport")
            
            # Generate match summary
            summary_stats = generate_enhanced_match_summary(df_events)
            
            st.markdown("#### Match Summary")
        
        # Display summary in a visually appealing way
        col_swarm, col_vs, col_opp = st.columns([2, 1, 2])
        
        with col_swarm:
            st.markdown(f"<h3 style='text-align: center; color: {ThemeConfig.PRIMARY_COLOR};'>SSA Swarm</h3>", unsafe_allow_html=True)
        with col_vs:
            st.markdown("<h3 style='text-align: center;'>vs</h3>", unsafe_allow_html=True)
        with col_opp:
            st.markdown(f"<h3 style='text-align: center; color: {ThemeConfig.SECONDARY_COLOR};'>Opponent</h3>", unsafe_allow_html=True)
        
        # Display stats
        for stat, (swarm, opp) in summary_stats.items():
            col1, col2, col3 = st.columns([2, 1, 2])
            
            # Determine colors
            if stat in ["Score", "Avg Goals"]:
                swarm_color = ThemeConfig.SUCCESS_COLOR if swarm > opp else ThemeConfig.PRIMARY_COLOR
                opp_color = ThemeConfig.SUCCESS_COLOR if opp > swarm else ThemeConfig.PRIMARY_COLOR
            else:
                swarm_color = ThemeConfig.PRIMARY_COLOR
                opp_color = ThemeConfig.SECONDARY_COLOR
            
            with col1:
                if stat in ["Score", "Avg Goals"]:
                    st.markdown(f"<div style='text-align: center; font-size: 3em; color: {swarm_color}; font-weight: bold;'>{swarm}</div>", unsafe_allow_html=True)
                else:
                    st.metric("", swarm)
            
            with col2:
                st.markdown(f"<div style='text-align: center; padding-top: 20px; font-weight: bold;'>{stat}</div>", unsafe_allow_html=True)
            
            with col3:
                if stat in ["Score", "Avg Goals"]:
                    st.markdown(f"<div style='text-align: center; font-size: 3em; color: {opp_color}; font-weight: bold;'>{opp}</div>", unsafe_allow_html=True)
                else:
                    st.metric("", opp)
        
        # AI Assistant for Event Analysis
        event_summary = "\n".join([f"{stat}: Swarm {swarm} - Opponent {opp}" for stat, (swarm, opp) in summary_stats.items()])
        event_data_summary = f"""
        Match Event Summary:
        {event_summary}
        
        Total Events Analyzed: {len(df_events)}
        Key Tactical Insights Needed: Formation effectiveness, possession patterns, attacking efficiency
        """
        
        display_ai_assistant("Match Event Tactical Analysis", event_data_summary, api_key)
        
        # Event categories breakdown
        if selected_match == "All Matches (Average)":
            with st.expander("Average Event Breakdown Across All Matches", expanded=False):
                event_categories = ["Crosses", "Free Kicks", "Corner Kicks", "Throw-ins", "Regains", "PAZ Entries", "Zone 3 Entries"]
                
                for category in event_categories:
                    show_enhanced_event_subtable_aggregate(df_events, [category.lower()], category)
            
            # Show match-by-match breakdown
            with st.expander("Match-by-Match Event Comparison", expanded=False):
                render_match_by_match_events(df_events)
        else:
            with st.expander("Detailed Event Breakdown", expanded=False):
                event_categories = ["Crosses", "Free Kicks", "Corner Kicks", "Throw-ins", "Regains", "PAZ Entries", "Zone 3 Entries"]
                
                for category in event_categories:
                    show_enhanced_event_subtable(df_events, [category.lower()], category)
            
            # Display event image if available
            if selected_match in EVENT_IMAGES:
                st.markdown("#### Event Visualization")
                try:
                    event_image = Image.open(EVENT_IMAGES[selected_match])
                    st.image(event_image, caption=f"Event Table - {selected_match}", use_container_width=True)
                except:
                    st.warning("Event visualization not available.")
        
        # Shot map
        render_enhanced_shot_map(df_events, selected_match)
        
    except Exception as e:
        st.error(f"Error loading event data: {str(e)}")

def generate_enhanced_match_summary(df_events):
    """Generate enhanced match summary statistics"""
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
        return mask.sum()
    
    summary = {
        "Score": [
            count_events("Shot", descriptors_only=["goal"]),
            count_events("Shot", is_opp=True, descriptors_only=["goal"])
        ],
        "Shots": [count_events("Shot"), count_events("Shot", True)],
        "On Target": [
            count_events("Shot", descriptors_only=["goal", "save"]),
            count_events("Shot", is_opp=True, descriptors_only=["goal", "save"])
        ],
        "Possession %": [55, 45],  # Placeholder - would need actual possession data
        "Pass Accuracy": [82, 78],  # Placeholder
        "Corners": [count_events("Corner"), count_events("Corner", True)],
        "Fouls": [count_events("Foul Won"), count_events("Foul Won", True)],
    }
    
    return summary

def generate_enhanced_match_summary_aggregate(df_events):
    """Generate aggregated match summary statistics for all matches"""
    df = df_events.copy()
    df.columns = df.columns.str.strip()
    df["Category"] = df["Category"].astype(str).str.strip()
    descriptor_cols = [col for col in df.columns if col.startswith("Des")]
    desc_text = df[descriptor_cols].astype(str).agg(" ".join, axis=1).str.lower()
    
    # Get unique matches
    matches = df["Match"].unique()
    num_matches = len(matches)
    
    def count_events_avg(keyword, is_opp=False, descriptors_only=None):
        total_count = 0
        for match in matches:
            match_df = df[df["Match"] == match]
            match_desc_text = match_df[descriptor_cols].astype(str).agg(" ".join, axis=1).str.lower()
            
            mask = match_df["Category"].str.lower().str.contains(keyword.lower())
            if descriptors_only:
                mask &= match_desc_text.str.contains("|".join(descriptors_only))
            if is_opp:
                mask &= match_desc_text.str.contains("opp") | match_df["Category"].str.lower().str.contains("opp")
            else:
                mask &= ~match_desc_text.str.contains("opp") & ~match_df["Category"].str.lower().str.contains("opp")
            total_count += mask.sum()
        
        return round(total_count / num_matches, 1)
    
    summary = {
        "Avg Goals": [
            count_events_avg("Shot", descriptors_only=["goal"]),
            count_events_avg("Shot", is_opp=True, descriptors_only=["goal"])
        ],
        "Avg Shots": [count_events_avg("Shot"), count_events_avg("Shot", True)],
        "Avg On Target": [
            count_events_avg("Shot", descriptors_only=["goal", "save"]),
            count_events_avg("Shot", is_opp=True, descriptors_only=["goal", "save"])
        ],
        "Avg Corners": [count_events_avg("Corner"), count_events_avg("Corner", True)],
        "Avg Fouls": [count_events_avg("Foul Won"), count_events_avg("Foul Won", True)],
        "Avg PAZ Entries": [count_events_avg("PAZ"), count_events_avg("PAZ", True)],
    }
    
    return summary

def show_enhanced_event_subtable_aggregate(df_events, keywords, title):
    """Show aggregated event subtable with match breakdown"""
    df = df_events.copy()
    df.columns = df.columns.str.strip()
    
    # Get events for this category across all matches
    mask = pd.Series([False] * len(df))
    for kw in keywords:
        mask |= df["Category"].str.lower().str.contains(kw.lower())
    
    sub_df = df[mask]
    
    if not sub_df.empty:
        # Show average count
        matches = sub_df["Match"].unique()
        avg_count = len(sub_df) / len(matches)
        
        st.markdown(f"**{title}** (Average: {avg_count:.1f} per match)")
        
        # Show breakdown by match
        match_counts = sub_df["Match"].value_counts().sort_index()
        
        # Create a simple bar chart
        fig = px.bar(
            x=match_counts.index,
            y=match_counts.values,
            labels={"x": "Match", "y": "Count"},
            title=f"{title} by Match",
            height=300
        )
        
        fig.update_layout(
            plot_bgcolor=ThemeConfig.CARD_BACKGROUND,
            paper_bgcolor=ThemeConfig.BACKGROUND_COLOR,
            font=dict(color=ThemeConfig.TEXT_COLOR),
            xaxis=dict(tickangle=-45),
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info(f"No {title.lower()} found.")

def render_match_by_match_events(df_events):
    """Render match-by-match event comparison"""
    st.markdown("#### Match-by-Match Event Statistics")
    
    # Create summary for each match
    matches = df_events["Match"].unique()
    match_summaries = []
    
    for match in matches:
        match_df = df_events[df_events["Match"] == match]
        summary = generate_enhanced_match_summary(match_df)
        
        match_data = {"Match": match}
        for stat, (swarm, opp) in summary.items():
            match_data[f"{stat} (Swarm)"] = swarm
            match_data[f"{stat} (Opp)"] = opp
        
        match_summaries.append(match_data)
    
    summary_df = pd.DataFrame(match_summaries)
    
    # Display as a styled table
    st.dataframe(
        summary_df.style.background_gradient(subset=[col for col in summary_df.columns if "Swarm" in col], cmap='RdYlGn'),
        use_container_width=True
    )

def show_enhanced_event_subtable(df_events, keywords, title):
    """Show enhanced event subtable with better formatting"""
    df = df_events.copy()
    df.columns = df.columns.str.strip()
    
    mask = pd.Series([False] * len(df))
    for kw in keywords:
        mask |= df["Category"].str.lower().str.contains(kw.lower())
    
    sub_df = df[mask]
    
    if not sub_df.empty:
        st.markdown(f"**{title}** ({len(sub_df)} events)")
        
        # Select relevant columns
        display_cols = ["Category", "Start", "End"]
        desc_cols = [col for col in df.columns if col.startswith("Des") and df[col].notna().sum() > 0][:3]
        
        display_df = sub_df[display_cols + desc_cols].reset_index(drop=True)
        
        # Style the dataframe
        st.dataframe(
            display_df,
            use_container_width=True,
            height=200
        )

def render_enhanced_shot_map(df_events, selected_match):
    """Render enhanced shot map with Plotly"""
    st.markdown("#### Shot Map Analysis")
    
    try:
        # Filter for shots with XY coordinates
        df_shots = df_events.dropna(subset=["XY"]).copy()
        df_shots = df_shots[df_shots["Category"].str.lower().str.contains("shot")]
        
        if df_shots.empty:
            st.info("No shot data available for this match.")
            return
        
        # Parse coordinates
        df_shots[["X", "Y"]] = df_shots["XY"].str.split(";", expand=True).astype(float)
        
        # Determine shot outcomes
        descriptor_cols = [col for col in df_events.columns if col.startswith("Des")]
        desc_text = df_shots[descriptor_cols].astype(str).agg(" ".join, axis=1).str.lower()
        
        df_shots["Outcome"] = "Shot"
        df_shots.loc[desc_text.str.contains("goal"), "Outcome"] = "Goal"
        df_shots.loc[desc_text.str.contains("save"), "Outcome"] = "Saved"
        df_shots.loc[desc_text.str.contains("block"), "Outcome"] = "Blocked"
        df_shots.loc[desc_text.str.contains("wide|over"), "Outcome"] = "Off Target"
        
        # Add match info for hover if aggregated
        if selected_match == "All Matches (Average)" and "Match" in df_shots.columns:
            hover_data = ["Match", "Outcome"]
            title = "Shot Locations - All Matches Combined"
        else:
            hover_data = ["Outcome"]
            title = f"Shot Locations - {selected_match}"
        
        # Create Plotly figure
        fig = px.scatter(
            df_shots,
            x="X",
            y="Y",
            color="Outcome",
            size=[20] * len(df_shots),
            hover_data=hover_data,
            color_discrete_map={
                "Goal": ThemeConfig.SUCCESS_COLOR,
                "Saved": ThemeConfig.WARNING_COLOR,
                "Blocked": ThemeConfig.SECONDARY_COLOR,
                "Off Target": ThemeConfig.PRIMARY_COLOR,
                "Shot": ThemeConfig.TEXT_COLOR
            },
            title=title,
            height=600
        )
        
        # Update layout
        fig.update_layout(
            plot_bgcolor=ThemeConfig.CARD_BACKGROUND,
            paper_bgcolor=ThemeConfig.BACKGROUND_COLOR,
            font=dict(color=ThemeConfig.TEXT_COLOR),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            showlegend=True,
            legend=dict(
                bgcolor=ThemeConfig.CARD_BACKGROUND,
                bordercolor=ThemeConfig.PRIMARY_COLOR,
                borderwidth=1
            )
        )
        
        # Add pitch outline (simplified)
        pitch_width, pitch_height = 100, 65
        
        # Pitch outline
        fig.add_shape(type="rect", x0=0, y0=0, x1=pitch_width, y1=pitch_height,
                      line=dict(color="white", width=2))
        
        # Penalty areas
        fig.add_shape(type="rect", x0=0, y0=15, x1=17, y1=50,
                      line=dict(color="white", width=2))
        fig.add_shape(type="rect", x0=83, y0=15, x1=100, y1=50,
                      line=dict(color="white", width=2))
        
        # Goal areas
        fig.add_shape(type="rect", x0=0, y0=25, x1=6, y1=40,
                      line=dict(color="white", width=2))
        fig.add_shape(type="rect", x0=94, y0=25, x1=100, y1=40,
                      line=dict(color="white", width=2))
        
        # Center circle
        fig.add_shape(type="circle", x0=40, y0=22.5, x1=60, y1=42.5,
                      line=dict(color="white", width=2))
        
        # Center line
        fig.add_shape(type="line", x0=50, y0=0, x1=50, y1=65,
                      line=dict(color="white", width=2))
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Shot statistics
        col1, col2, col3 = st.columns(3)
        
        total_shots = len(df_shots)
        goals = len(df_shots[df_shots["Outcome"] == "Goal"])
        on_target = len(df_shots[df_shots["Outcome"].isin(["Goal", "Saved"])])
        
        with col1:
            st.metric("Total Shots", total_shots)
        with col2:
            st.metric("Goals", goals)
        with col3:
            conversion_rate = (goals / total_shots * 100) if total_shots > 0 else 0
            st.metric("Conversion Rate", f"{conversion_rate:.1f}%")
        
    except Exception as e:
        st.error(f"Error creating shot map: {str(e)}"), 65
        
        # Pitch outline
        fig.add_shape(type="rect", x0=0, y0=0, x1=pitch_width, y1=pitch_height,
                      line=dict(color="white", width=2))
        
        # Penalty areas
        fig.add_shape(type="rect", x0=0, y0=15, x1=17, y1=50,
                      line=dict(color="white", width=2))
        fig.add_shape(type="rect", x0=83, y0=15, x1=100, y1=50,
                      line=dict(color="white", width=2))
        
        # Goal areas
        fig.add_shape(type="rect", x0=0, y0=25, x1=6, y1=40,
                      line=dict(color="white", width=2))
        fig.add_shape(type="rect", x0=94, y0=25, x1=100, y1=40,
                      line=dict(color="white", width=2))
        
        # Center circle
        fig.add_shape(type="circle", x0=40, y0=22.5, x1=60, y1=42.5,
                      line=dict(color="white", width=2))
        
        # Center line
        fig.add_shape(type="line", x0=50, y0=0, x1=50, y1=65,
                      line=dict(color="white", width=2))
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Shot statistics
        col1, col2, col3, col4 = st.columns(4)
        
        total_shots = len(df_shots)
        goals = len(df_shots[df_shots["Outcome"] == "Goal"])
        on_target = len(df_shots[df_shots["Outcome"].isin(["Goal", "Saved"])])
        
        if selected_match == "All Matches (Average)":
            num_matches = df_shots["Match"].nunique() if "Match" in df_shots.columns else 1
            avg_shots = total_shots / num_matches
            avg_goals = goals / num_matches
            
            with col1:
                st.metric("Avg Shots/Match", f"{avg_shots:.1f}")
            with col2:
                st.metric("Avg Goals/Match", f"{avg_goals:.1f}")
            with col3:
                st.metric("Total Goals", goals)
            with col4:
                conversion_rate = (goals / total_shots * 100) if total_shots > 0 else 0
                st.metric("Conversion Rate", f"{conversion_rate:.1f}%")
        else:
            with col1:
                st.metric("Total Shots", total_shots)
            with col2:
                st.metric("Goals", goals)
            with col3:
                st.metric("On Target", on_target)
            with col4:
                conversion_rate = (goals / total_shots * 100) if total_shots > 0 else 0
                st.metric("Conversion Rate", f"{conversion_rate:.1f}%")
        
        # Additional analysis for aggregated data
        if selected_match == "All Matches (Average)" and "Match" in df_shots.columns:
            with st.expander("Shot Analysis by Match"):
                shot_summary = df_shots.groupby("Match").agg({
                    "Outcome": "count",
                    "X": lambda x: len(x[df_shots.loc[x.index, "Outcome"] == "Goal"])
                }).rename(columns={"Outcome": "Total Shots", "X": "Goals"})
                
                shot_summary["Conversion %"] = (shot_summary["Goals"] / shot_summary["Total Shots"] * 100).round(1)
                
                st.dataframe(
                    shot_summary.style.background_gradient(subset=["Conversion %"], cmap='RdYlGn'),
                    use_container_width=True
                )
        
    except Exception as e:
        st.error(f"Error creating shot map: {str(e)}"), 65
        
        # Pitch outline
        fig.add_shape(type="rect", x0=0, y0=0, x1=pitch_width, y1=pitch_height,
                      line=dict(color="white", width=2))
        
        # Penalty areas
        fig.add_shape(type="rect", x0=0, y0=15, x1=17, y1=50,
                      line=dict(color="white", width=2))
        fig.add_shape(type="rect", x0=83, y0=15, x1=100, y1=50,
                      line=dict(color="white", width=2))
        
        # Goal areas
        fig.add_shape(type="rect", x0=0, y0=25, x1=6, y1=40,
                      line=dict(color="white", width=2))
        fig.add_shape(type="rect", x0=94, y0=25, x1=100, y1=40,
                      line=dict(color="white", width=2))
        
        # Center circle
        fig.add_shape(type="circle", x0=40, y0=22.5, x1=60, y1=42.5,
                      line=dict(color="white", width=2))
        
        # Center line
        fig.add_shape(type="line", x0=50, y0=0, x1=50, y1=65,
                      line=dict(color="white", width=2))
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Shot statistics
        col1, col2, col3 = st.columns(3)
        
        total_shots = len(df_shots)
        goals = len(df_shots[df_shots["Outcome"] == "Goal"])
        on_target = len(df_shots[df_shots["Outcome"].isin(["Goal", "Saved"])])
        
        with col1:
            st.metric("Total Shots", total_shots)
        with col2:
            st.metric("Goals", goals)
        with col3:
            conversion_rate = (goals / total_shots * 100) if total_shots > 0 else 0
            st.metric("Conversion Rate", f"{conversion_rate:.1f}%")
        
    except Exception as e:
        st.error(f"Error creating shot map: {str(e)}")

def render_performance_charts(match_df, api_key):
    """Render performance charts"""
    st.markdown("### Performance Metrics by Player")
    
    # Allow metric selection
    selected_metrics = st.multiselect(
        "Select metrics to display:",
        METRICS,
        default=["Total Distance", "Max Speed", "No of Sprints", "High Speed Running"]
    )
    
    if not selected_metrics:
        st.warning("Please select at least one metric to display.")
        return
    
    # Prepare detailed data for AI Assistant
    player_stats = match_df.groupby("Player Name")[selected_metrics].mean().round(2)
    
    # Create detailed summary for AI
    detailed_summary = f"""
    Performance Metrics Analysis:
    Metrics analyzed: {', '.join(selected_metrics)}
    Total players: {len(player_stats)}
    
    Player Performance Data:
    """
    
    # Add top and bottom performers for each metric
    for metric in selected_metrics:
        metric_data = player_stats[metric].sort_values(ascending=False)
        top_3 = metric_data.head(3)
        bottom_3 = metric_data.tail(3)
        team_avg = metric_data.mean()
        
        detailed_summary += f"\n\n{metric}:"
        detailed_summary += f"\n- Team Average: {team_avg:.2f}"
        detailed_summary += f"\n- Top Performers: {', '.join([f'{player} ({value:.2f})' for player, value in top_3.items()])}"
        detailed_summary += f"\n- Bottom Performers: {', '.join([f'{player} ({value:.2f})' for player, value in bottom_3.items()])}"
        detailed_summary += f"\n- Range: {metric_data.min():.2f} to {metric_data.max():.2f}"
    
    detailed_summary += "\n\nConsider position-specific requirements when evaluating metrics."
    
    display_ai_assistant("Performance Metrics Analysis", detailed_summary, api_key)
    
    # Create charts for each metric
    for metric in selected_metrics:
        with st.container():
            st.markdown(f"#### {metric}")
            
            # Add metric description
            if metric in METRIC_DESCRIPTIONS:
                st.caption(METRIC_DESCRIPTIONS[metric])
            
            # Prepare data
            chart_data = match_df.groupby("Player Name")[metric].mean().reset_index()
            chart_data = chart_data.sort_values(by=metric, ascending=False)
            
            # Create Plotly bar chart
            fig = px.bar(
                chart_data,
                x="Player Name",
                y=metric,
                color=metric,
                color_continuous_scale=["#1A1A1D", ThemeConfig.PRIMARY_COLOR],
                title=f"{metric} by Player",
                height=400
            )
            
            # Update layout
            fig.update_layout(
                plot_bgcolor=ThemeConfig.CARD_BACKGROUND,
                paper_bgcolor=ThemeConfig.BACKGROUND_COLOR,
                font=dict(color=ThemeConfig.TEXT_COLOR),
                xaxis=dict(tickangle=-45),
                showlegend=False
            )
            
            # Add average line
            avg_value = chart_data[metric].mean()
            fig.add_hline(
                y=avg_value,
                line_dash="dash",
                line_color=ThemeConfig.SECONDARY_COLOR,
                annotation_text=f"Team Avg: {avg_value:.1f}",
                annotation_position="top right"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Show top performers
            top_performers = chart_data.nlargest(3, metric)
            with st.expander(f" Top 3 Performers - {metric}"):
                for idx, (_, row) in enumerate(top_performers.iterrows(), 1):
                    medal = "1" if idx == 1 else "2" if idx == 2 else "3"
                    st.markdown(f"{medal} **{row['Player Name']}**: {row[metric]:.2f}")

def render_trend_analysis(selected_match, match_df, api_key, MATCH_FILES):
    """Render trend analysis"""
    if selected_match == "All Matches (Average)":
        st.markdown("### Team Performance Trends")
        
        # Load all match data
        all_matches_data = []
        for match_name, file_path in MATCH_FILES.items():
            df_temp = load_data(file_path)
            df_temp["Match"] = match_name
            all_matches_data.append(df_temp)
        
        full_df = pd.concat(all_matches_data, ignore_index=True)
        
        # Team trends
        st.markdown("#### Team Average Trends Across Matches")
        
        selected_metric = st.selectbox(
            "Select metric for trend analysis:",
            METRICS,
            key="trend_metric"
        )
        
        # Calculate team averages per match
        team_avg = full_df.groupby("Match")[selected_metric].mean().reset_index()
        team_avg["Match_Order"] = team_avg["Match"].map({match: i for i, match in enumerate(MATCH_FILES.keys())})
        team_avg = team_avg.sort_values("Match_Order")
        
        # Create trend chart
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=team_avg["Match"],
            y=team_avg[selected_metric],
            mode='lines+markers',
            name='Team Average',
            line=dict(color=ThemeConfig.PRIMARY_COLOR, width=3),
            marker=dict(size=10, color=ThemeConfig.PRIMARY_COLOR),
            hovertemplate='<b>%{x}</b><br>' + selected_metric + ': %{y:.2f}<extra></extra>'
        ))
        
        # Add trend line
        z = np.polyfit(team_avg["Match_Order"], team_avg[selected_metric], 1)
        p = np.poly1d(z)
        
        fig.add_trace(go.Scatter(
            x=team_avg["Match"],
            y=p(team_avg["Match_Order"]),
            mode='lines',
            name='Trend',
            line=dict(color=ThemeConfig.SECONDARY_COLOR, width=2, dash='dash'),
            hoverinfo='skip'
        ))
        
        fig.update_layout(
            title=f"{selected_metric} - Team Trend",
            xaxis_title="Match",
            yaxis_title=selected_metric,
            plot_bgcolor=ThemeConfig.CARD_BACKGROUND,
            paper_bgcolor=ThemeConfig.BACKGROUND_COLOR,
            font=dict(color=ThemeConfig.TEXT_COLOR),
            xaxis=dict(tickangle=-45),
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # AI Assistant for Trend Analysis
        trend_summary = f"""
        Trend Analysis Summary:
        Metric: {selected_metric}
        Matches covered: {len(MATCH_FILES)}
        
        Team average trend: {'Improving' if z[0] > 0 else 'Declining' if z[0] < 0 else 'Stable'}
        First match average: {team_avg.iloc[0][selected_metric]:.2f}
        Last match average: {team_avg.iloc[-1][selected_metric]:.2f}
        
        Key questions to consider:
        - Are trends consistent with training focus?
        - Do trends align with tactical changes?
        - Which players show improvement vs decline?
        """
        
        display_ai_assistant("Performance Trend Analysis", trend_summary, api_key)
        
        # Player comparison across matches
        st.markdown("#### Individual Player Trends")
        
        selected_players = st.multiselect(
            "Select players to compare:",
            sorted(full_df["Player Name"].unique()),
            default=sorted(full_df["Player Name"].unique())[:3],
            key="trend_players"
        )
        
        if selected_players:
            player_data = full_df[full_df["Player Name"].isin(selected_players)]
            player_avg = player_data.groupby(["Match", "Player Name"])[selected_metric].mean().reset_index()
            
            fig = px.line(
                player_avg,
                x="Match",
                y=selected_metric,
                color="Player Name",
                markers=True,
                title=f"{selected_metric} - Player Comparison",
                height=500
            )
            
            fig.update_layout(
                plot_bgcolor=ThemeConfig.CARD_BACKGROUND,
                paper_bgcolor=ThemeConfig.BACKGROUND_COLOR,
                font=dict(color=ThemeConfig.TEXT_COLOR),
                xaxis=dict(tickangle=-45)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Performance change analysis
            with st.expander("Performance Change Analysis"):
                for player in selected_players:
                    player_matches = player_avg[player_avg["Player Name"] == player].sort_values("Match")
                    if len(player_matches) > 1:
                        first_match_val = player_matches.iloc[0][selected_metric]
                        last_match_val = player_matches.iloc[-1][selected_metric]
                        change = ((last_match_val - first_match_val) / first_match_val) * 100
                        
                        change_color = ThemeConfig.SUCCESS_COLOR if change > 0 else ThemeConfig.PRIMARY_COLOR
                        change_symbol = "Up" if change > 0 else "Down"
                        
                        st.markdown(f"""
                        **{player}**: {change_symbol} {change:+.1f}% change
                        (From {first_match_val:.1f} to {last_match_val:.1f})
                        """)
    else:
        st.info("Trend analysis is only available when 'All Matches (Average)' is selected.")

def render_radar_analysis(match_df, full_df, half_option, selected_player, api_key):
    """Render radar analysis"""
    st.markdown("### Player Radar Analysis")
    
    if selected_player == "All":
        # Multi-player comparison
        st.markdown("#### Multi-Player Comparison")
        
        players_for_radar = st.multiselect(
            "Select 2-4 players for radar comparison:",
            match_df["Player Name"].unique(),
            default=list(match_df["Player Name"].unique())[:3],
            max_selections=4,
            key="radar_players"
        )
        
        if len(players_for_radar) >= 2:
            # Calculate percentiles
            full_team = full_df[full_df["Session Type"] == half_option] if half_option != "Total" else full_df
            radar_df = full_team.groupby("Player Name")[METRICS].mean()
            
            selected_radar_df = radar_df.loc[players_for_radar]
            percentile_df = calculate_percentile_values(selected_radar_df, radar_df, METRICS)
            
            # Create radar chart
            fig = create_plotly_radar(
                percentile_df[METRICS].values,
                METRICS,
                "Player Comparison - Percentile Rankings",
                players_for_radar
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Performance comparison table
            with st.expander("Detailed Comparison"):
                comparison_df = selected_radar_df.round(2)
                comparison_df["Overall Score"] = comparison_df.mean(axis=1).round(2)
                comparison_df = comparison_df.sort_values("Overall Score", ascending=False)
                
                st.dataframe(
                    comparison_df.style.background_gradient(cmap='RdYlGn', axis=1),
                    use_container_width=True
                )
        else:
            st.info("Please select at least 2 players for radar comparison.")
    else:
        # Single player analysis
        full_team = full_df[full_df["Session Type"] == half_option] if half_option != "Total" else full_df
        radar_df = full_team.groupby("Player Name")[METRICS].mean()
        
        if selected_player in radar_df.index:
            selected_radar_df = radar_df.loc[[selected_player]]
            percentile_df = calculate_percentile_values(selected_radar_df, radar_df, METRICS)
            
            # Create radar chart
            fig = create_plotly_radar(
                percentile_df[METRICS].values,
                METRICS,
                f"{selected_player} - Percentile Rankings",
                [selected_player]
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Player strengths and weaknesses
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Strengths")
                strengths = percentile_df[METRICS].T.nlargest(3, selected_player)
                for metric, value in strengths.iterrows():
                    st.markdown(f"**{metric}**: {value[selected_player]:.0f}th percentile")
            
            with col2:
                st.markdown("#### Areas for Improvement")
                weaknesses = percentile_df[METRICS].T.nsmallest(3, selected_player)
                for metric, value in weaknesses.iterrows():
                    st.markdown(f"**{metric}**: {value[selected_player]:.0f}th percentile")
            
            # AI Assistant for Individual Player Analysis
            player_percentiles = {metric: percentile_df[metric].iloc[0] for metric in METRICS}
            radar_summary = f"""
            Player Profile: {selected_player}
            
            Percentile Rankings:
            {chr(10).join([f"- {metric}: {player_percentiles[metric]:.0f}th percentile" for metric in METRICS])}
            
            Strengths: {', '.join([m for m, v in player_percentiles.items() if v >= 75])}
            Development Areas: {', '.join([m for m, v in player_percentiles.items() if v < 50])}
            """
            
            display_ai_assistant("Individual Player Profile Analysis", radar_summary, api_key)

def render_data_table(match_df):
    """Render data table with export options"""
    st.markdown("### Raw Data View")
    
    # Data filtering options
    col1, col2 = st.columns(2)
    
    with col1:
        show_cols = st.multiselect(
            "Select columns to display:",
            match_df.columns.tolist(),
            default=["Player Name", "Session Type"] + METRICS
        )
    
    with col2:
        sort_by = st.selectbox(
            "Sort by:",
            show_cols,
            index=0 if "Player Name" in show_cols else 0
        )
    
    if show_cols:
        display_df = match_df[show_cols].sort_values(by=sort_by)
        
        # Display statistics
        st.markdown("#### Summary Statistics")
        numeric_cols = [col for col in show_cols if col in METRICS]
        if numeric_cols:
            st.dataframe(
                display_df[numeric_cols].describe().round(2),
                use_container_width=True
            )
        
        # Display full data
        st.markdown("#### Full Dataset")
        st.dataframe(
            display_df.style.format({col: '{:.2f}' for col in numeric_cols if col in display_df.columns}),
            use_container_width=True,
            height=400
        )
        
        # Export options
        csv = display_df.to_csv(index=False)
        st.download_button(
            label="Download as CSV",
            data=csv,
            file_name=f"match_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

def render_weekly_training_report(api_key, team_config):
    """Render weekly training report"""
    st.markdown(f"<h2 style='color: {ThemeConfig.PRIMARY_COLOR};'> Weekly Training Report</h2>", unsafe_allow_html=True)
    
    # Session selection
    selected_sessions = st.sidebar.multiselect(
        "Select Training Sessions",
        list(TRAINING_FILES.keys()),
        default=list(TRAINING_FILES.keys())
    )
    
    if not selected_sessions:
        st.info("Please select at least one training session.")
        return
    
    # Load data
    training_data = []
    for session in selected_sessions:
        df_temp = load_data(TRAINING_FILES[session])
        df_temp["Session"] = session
        training_data.append(df_temp)
    
    weekly_df = pd.concat(training_data, ignore_index=True)
    
    # View mode selection
    view_mode = st.sidebar.radio(
        "View Mode",
        ["Team Overview", "Individual Analysis", "Session Comparison", "Load Management"],
        key="weekly_view"
    )
    
    if view_mode == "Team Overview":
        render_team_overview(weekly_df, api_key)
    elif view_mode == "Individual Analysis":
        render_individual_analysis(weekly_df, api_key)
    elif view_mode == "Session Comparison":
        render_session_comparison(weekly_df, api_key)
    elif view_mode == "Load Management":
        render_load_management(weekly_df, api_key)

def render_team_overview(weekly_df, api_key):
    """Render team overview for weekly training"""
    st.markdown("### Team Training Overview")
    
    # Calculate weekly totals and averages
    col1, col2, col3, col4 = st.columns(4)
    
    total_distance = weekly_df["Total Distance"].sum()
    avg_intensity = weekly_df["Max Speed"].mean()
    total_sprints = weekly_df["No of Sprints"].sum()
    avg_load = (weekly_df["Accelerations"].mean() + weekly_df["Decelerations"].mean())
    
    with col1:
        st.markdown(create_metric_card("Total Team Distance", f"{total_distance:.0f} km"), unsafe_allow_html=True)
    with col2:
        st.markdown(create_metric_card("Avg Intensity", f"{avg_intensity:.1f} km/h"), unsafe_allow_html=True)
    with col3:
        st.markdown(create_metric_card("Total Sprints", f"{total_sprints:.0f}"), unsafe_allow_html=True)
    with col4:
        st.markdown(create_metric_card("Avg Load", f"{avg_load:.0f}"), unsafe_allow_html=True)
    
    # AI Assistant for Team Overview
    team_summary = f"""
    Weekly Training Overview:
    Sessions completed: {weekly_df['Session'].nunique()}
    Total players: {weekly_df['Player Name'].nunique()}
    
    Team Statistics:
    - Total Distance: {total_distance:.0f} km
    - Average Intensity: {avg_intensity:.1f} km/h
    - Total Sprints: {total_sprints:.0f}
    - Average Load: {avg_load:.0f}
    
    Consider: Is the training load appropriate for match schedule? Are all players getting adequate volume?
    """
    
    display_ai_assistant("Weekly Team Training Analysis", team_summary, api_key)
    
    # Training load distribution
    st.markdown("#### Training Load Distribution")
    
    # Create heatmap of player loads across sessions
    pivot_df = weekly_df.pivot_table(
        values="Total Distance",
        index="Player Name",
        columns="Session",
        aggfunc="mean"
    ).fillna(0)
    
    fig = px.imshow(
        pivot_df,
        labels=dict(x="Session", y="Player", color="Distance (km)"),
        color_continuous_scale=["#1A1A1D", ThemeConfig.PRIMARY_COLOR],
        title="Player Load Heatmap",
        height=600
    )
    
    fig.update_layout(
        plot_bgcolor=ThemeConfig.CARD_BACKGROUND,
        paper_bgcolor=ThemeConfig.BACKGROUND_COLOR,
        font=dict(color=ThemeConfig.TEXT_COLOR)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Player rankings
    st.markdown("#### Weekly Player Rankings")
    
    ranking_metric = st.selectbox(
        "Select ranking metric:",
        METRICS,
        key="ranking_metric"
    )
    
    player_rankings = weekly_df.groupby("Player Name")[ranking_metric].mean().sort_values(ascending=False).reset_index()
    
    # Create horizontal bar chart for rankings
    fig = px.bar(
        player_rankings.head(10),
        y="Player Name",
        x=ranking_metric,
        orientation='h',
        color=ranking_metric,
        color_continuous_scale=["#1A1A1D", ThemeConfig.PRIMARY_COLOR],
        title=f"Top 10 Players - {ranking_metric}",
        height=400
    )
    
    fig.update_layout(
        plot_bgcolor=ThemeConfig.CARD_BACKGROUND,
        paper_bgcolor=ThemeConfig.BACKGROUND_COLOR,
        font=dict(color=ThemeConfig.TEXT_COLOR),
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)

def render_individual_analysis(weekly_df, api_key):
    """Render individual player analysis"""
    st.markdown("### Individual Player Analysis")
    
    selected_player = st.sidebar.selectbox(
        "Select Player",
        sorted(weekly_df["Player Name"].unique())
    )
    
    player_data = weekly_df[weekly_df["Player Name"] == selected_player]
    
    # Player summary cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_distance = player_data["Total Distance"].mean()
        st.markdown(create_metric_card("Avg Distance", f"{avg_distance:.2f} km"), unsafe_allow_html=True)
    
    with col2:
        max_speed = player_data["Max Speed"].max()
        st.markdown(create_metric_card("Peak Speed", f"{max_speed:.1f} km/h"), unsafe_allow_html=True)
    
    with col3:
        total_sprints = player_data["No of Sprints"].sum()
        st.markdown(create_metric_card("Total Sprints", f"{total_sprints:.0f}"), unsafe_allow_html=True)
    
    with col4:
        sessions_completed = player_data["Session"].nunique()
        st.markdown(create_metric_card("Sessions", f"{sessions_completed}"), unsafe_allow_html=True)
    
    # Performance trends
    st.markdown("#### Performance Trends")
    
    trend_metric = st.selectbox(
        "Select metric for trend analysis:",
        METRICS,
        key="individual_trend"
    )
    
    # Line chart for selected metric
    session_data = player_data.groupby("Session")[trend_metric].mean().reset_index()
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=session_data["Session"],
        y=session_data[trend_metric],
        mode='lines+markers',
        name=selected_player,
        line=dict(color=ThemeConfig.PRIMARY_COLOR, width=3),
        marker=dict(size=12, color=ThemeConfig.PRIMARY_COLOR)
    ))
    
    # Add team average line
    team_avg = weekly_df.groupby("Session")[trend_metric].mean()
    fig.add_trace(go.Scatter(
        x=team_avg.index,
        y=team_avg.values,
        mode='lines',
        name='Team Average',
        line=dict(color=ThemeConfig.SECONDARY_COLOR, width=2, dash='dash')
    ))
    
    fig.update_layout(
        title=f"{trend_metric} - {selected_player} vs Team Average",
        xaxis_title="Session",
        yaxis_title=trend_metric,
        plot_bgcolor=ThemeConfig.CARD_BACKGROUND,
        paper_bgcolor=ThemeConfig.BACKGROUND_COLOR,
        font=dict(color=ThemeConfig.TEXT_COLOR),
        xaxis=dict(tickangle=-45),
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # AI Assistant for Individual Analysis
    player_summary = f"""
    Player Analysis: {selected_player}
    
    Weekly Statistics:
    - Average Distance: {avg_distance:.2f} km
    - Peak Speed: {max_speed:.1f} km/h
    - Total Sprints: {total_sprints}
    - Sessions Completed: {sessions_completed}
    
    Current Trend: {trend_metric}
    Player Average: {session_data[trend_metric].mean():.2f}
    Team Average: {team_avg.mean():.2f}
    
    Consider: Training load appropriateness, injury risk, performance trajectory
    """
    
    display_ai_assistant("Individual Player Training Analysis", player_summary, api_key)
    
    # Percentile rankings
    st.markdown("#### Performance Rankings")
    
    # Calculate percentiles for all metrics
    percentile_data = []
    for metric in METRICS:
        all_values = weekly_df[metric].dropna()
        player_value = player_data[metric].mean()
        percentile = (all_values < player_value).sum() / len(all_values) * 100
        percentile_data.append({
            "Metric": metric,
            "Value": player_value,
            "Percentile": percentile,
            "Rank": f"{int(percentile)}th"
        })
    
    percentile_df = pd.DataFrame(percentile_data)
    
    # Create gauge charts for top metrics
    fig = go.Figure()
    
    for i, row in percentile_df.iterrows():
        fig.add_trace(go.Indicator(
            mode="gauge+number+delta",
            value=row["Percentile"],
            title={'text': row["Metric"]},
            delta={'reference': 50, 'relative': True},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': ThemeConfig.PRIMARY_COLOR},
                'steps': [
                    {'range': [0, 25], 'color': "#FFCCCC"},
                    {'range': [25, 50], 'color': "#FFFFCC"},
                    {'range': [50, 75], 'color': "#CCFFCC"},
                    {'range': [75, 100], 'color': "#CCFFFF"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            },
            domain={'row': i // 2, 'column': i % 2}
        ))
    
    fig.update_layout(
        grid={'rows': 4, 'columns': 2, 'pattern': "independent"},
        template="plotly_dark",
        showlegend=False,
        height=800,
        paper_bgcolor=ThemeConfig.BACKGROUND_COLOR
    )
    
    st.plotly_chart(fig, use_container_width=True)

def render_session_comparison(weekly_df, api_key):
    """Render session comparison analysis"""
    st.markdown("### Session Comparison")
    
    # Session metrics overview
    session_summary = weekly_df.groupby("Session")[METRICS].mean().round(2)
    
    st.markdown("#### Session Averages")
    st.dataframe(
        session_summary.style.background_gradient(cmap='RdYlGn', axis=0),
        use_container_width=True
    )
    
    # Metric comparison across sessions
    comparison_metric = st.selectbox(
        "Select metric for detailed comparison:",
        METRICS,
        key="session_comparison_metric"
    )
    
    # Box plot for distribution
    fig = px.box(
        weekly_df,
        x="Session",
        y=comparison_metric,
        color="Session",
        title=f"{comparison_metric} Distribution by Session",
        height=500
    )
    
    fig.update_layout(
        plot_bgcolor=ThemeConfig.CARD_BACKGROUND,
        paper_bgcolor=ThemeConfig.BACKGROUND_COLOR,
        font=dict(color=ThemeConfig.TEXT_COLOR),
        xaxis=dict(tickangle=-45),
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Player participation
    st.markdown("#### 👥 Player Participation")
    
    participation = weekly_df.groupby("Session")["Player Name"].nunique().reset_index()
    participation.columns = ["Session", "Players"]
    
    fig = px.bar(
        participation,
        x="Session",
        y="Players",
        color="Players",
        color_continuous_scale=["#1A1A1D", ThemeConfig.PRIMARY_COLOR],
        title="Player Attendance by Session",
        height=400
    )
    
    fig.update_layout(
        plot_bgcolor=ThemeConfig.CARD_BACKGROUND,
        paper_bgcolor=ThemeConfig.BACKGROUND_COLOR,
        font=dict(color=ThemeConfig.TEXT_COLOR),
        xaxis=dict(tickangle=-45),
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # AI Assistant for Session Comparison
    session_comparison_summary = f"""
    Session Comparison Analysis:
    Total sessions analyzed: {len(session_summary)}
    Average attendance: {participation['Players'].mean():.0f} players
    
    Metric being compared: {comparison_metric}
    Highest average: {session_summary[comparison_metric].max():.2f}
    Lowest average: {session_summary[comparison_metric].min():.2f}
    
    Consider: Session intensity variations, recovery between sessions, attendance patterns
    """
    
    display_ai_assistant("Training Session Comparison", session_comparison_summary, api_key)

def render_load_management(weekly_df, api_key):
    """Render load management analysis"""
    st.markdown("### Load Management & Recovery")
    
    # Calculate load metrics
    weekly_df["Load Score"] = (
        weekly_df["Total Distance"] * 0.3 +
        weekly_df["Sprint Distance"] * 0.2 +
        weekly_df["High Speed Running"] * 0.2 +
        weekly_df["Accelerations"] * 0.15 +
        weekly_df["Decelerations"] * 0.15
    )
    
    # Player load status
    st.markdown("#### Player Load Status")
    
    player_loads = weekly_df.groupby("Player Name")["Load Score"].agg(['mean', 'sum', 'std']).round(2)
    player_loads["Status"] = pd.cut(
        player_loads["mean"],
        bins=[0, player_loads["mean"].quantile(0.33), player_loads["mean"].quantile(0.67), float('inf')],
        labels=["Low Load", "Optimal Load", "High Load"]
    )
    
    # Create status cards
    col1, col2, col3 = st.columns(3)
    
    low_load = len(player_loads[player_loads["Status"] == "Low Load"])
    optimal_load = len(player_loads[player_loads["Status"] == "Optimal Load"])
    high_load = len(player_loads[player_loads["Status"] == "High Load"])
    
    with col1:
        st.markdown(f"""
        <div style='background-color: {ThemeConfig.SUCCESS_COLOR}; padding: 20px; border-radius: 10px; text-align: center;'>
            <h3>{low_load}</h3>
            <p>Low Load</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div style='background-color: {ThemeConfig.SECONDARY_COLOR}; padding: 20px; border-radius: 10px; text-align: center;'>
            <h3>{optimal_load}</h3>
            <p>Optimal Load</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div style='background-color: {ThemeConfig.WARNING_COLOR}; padding: 20px; border-radius: 10px; text-align: center;'>
            <h3>{high_load}</h3>
            <p>High Load</p>
        </div>
        """, unsafe_allow_html=True)
    
    # AI Assistant for Load Management
    load_summary = f"""
    Load Management Analysis:
    
    Player Distribution:
    - Low Load: {low_load} players
    - Optimal Load: {optimal_load} players
    - High Load: {high_load} players
    
    Key Considerations:
    - High load players need recovery focus
    - Low load players may need intensity increase
    - Monitor for injury risk indicators
    - Consider match schedule when planning loads
    """
    
    display_ai_assistant("Load Management Analysis", load_summary, api_key)
    
    # Detailed player load table
    st.markdown("#### Detailed Load Analysis")
    
    load_display = player_loads.reset_index()
    load_display.columns = ["Player", "Avg Load", "Total Load", "Load Variability", "Status"]
    
    st.dataframe(
        load_display.style.apply(
            lambda x: ['background-color: #28a745' if v == "Low Load" 
                      else 'background-color: #ffc107' if v == "High Load"
                      else 'background-color: #17a2b8' if v == "Optimal Load"
                      else '' for v in x],
            subset=['Status']
        ),
        use_container_width=True
    )
    
    # Load progression chart
    st.markdown("#### Load Progression")
    
    selected_players_load = st.multiselect(
        "Select players to track:",
        sorted(weekly_df["Player Name"].unique()),
        default=sorted(weekly_df["Player Name"].unique())[:5]
    )
    
    if selected_players_load:
        load_progression = weekly_df[weekly_df["Player Name"].isin(selected_players_load)]
        load_by_session = load_progression.groupby(["Session", "Player Name"])["Load Score"].mean().reset_index()
        
        fig = px.line(
            load_by_session,
            x="Session",
            y="Load Score",
            color="Player Name",
            markers=True,
            title="Load Progression by Player",
            height=500
        )
        
        fig.update_layout(
            plot_bgcolor=ThemeConfig.CARD_BACKGROUND,
            paper_bgcolor=ThemeConfig.BACKGROUND_COLOR,
            font=dict(color=ThemeConfig.TEXT_COLOR),
            xaxis=dict(tickangle=-45)
        )
        
        st.plotly_chart(fig, use_container_width=True)

def render_daily_training_report(api_key, team_config):
    """Render daily training report"""
    st.markdown(f"<h2 style='color: {ThemeConfig.PRIMARY_COLOR};'> Daily Training Report</h2>", unsafe_allow_html=True)
    
    # Session selection
    selected_session = st.sidebar.selectbox(
        "Select Training Session",
        list(TRAINING_FILES.keys())
    )
    
    # Load data
    df_daily = load_data(TRAINING_FILES[selected_session])
    
    if df_daily.empty:
        st.error("No data available for this session.")
        return
    
    st.markdown(f"### {selected_session}")
    
    # Session overview metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        participants = df_daily["Player Name"].nunique()
        st.markdown(create_metric_card("Participants", f"{participants}"), unsafe_allow_html=True)
    
    with col2:
        avg_distance = df_daily["Total Distance"].mean()
        st.markdown(create_metric_card("Avg Distance", f"{avg_distance:.2f} km"), unsafe_allow_html=True)
    
    with col3:
        max_speed_session = df_daily["Max Speed"].max()
        st.markdown(create_metric_card("Session Peak Speed", f"{max_speed_session:.1f} km/h"), unsafe_allow_html=True)
    
    with col4:
        total_sprints = df_daily["No of Sprints"].sum()
        st.markdown(create_metric_card("Total Sprints", f"{total_sprints}"), unsafe_allow_html=True)
    
    # AI Assistant for Daily Overview
    daily_summary = f"""
    Daily Training Session: {selected_session}
    
    Session Statistics:
    - Participants: {participants}
    - Average Distance: {avg_distance:.2f} km
    - Peak Speed: {max_speed_session:.1f} km/h
    - Total Sprints: {total_sprints}
    
    Consider: Was the session intensity appropriate? Any players showing fatigue? Recovery needs?
    """
    
    display_ai_assistant("Daily Training Overview", daily_summary, api_key)
    
    # View mode
    view_mode = st.radio(
        "Select View",
        ["Session Overview", "Individual Performance", "Comparative Analysis"],
        horizontal=True
    )
    
    if view_mode == "Session Overview":
        render_session_overview(df_daily, api_key)
    elif view_mode == "Individual Performance":
        render_individual_performance(df_daily, api_key)
    elif view_mode == "Comparative Analysis":
        render_comparative_analysis(df_daily, api_key)

def render_session_overview(df_daily, api_key):
    """Render session overview"""
    st.markdown("### Session Overview")
    
    # Metric distribution
    selected_metric = st.selectbox(
        "Select metric to analyze:",
        METRICS,
        key="session_metric"
    )
    
    # Create distribution plot
    fig = go.Figure()
    
    # Add histogram
    fig.add_trace(go.Histogram(
        x=df_daily[selected_metric],
        name="Distribution",
        nbinsx=20,
        marker_color=ThemeConfig.PRIMARY_COLOR,
        opacity=0.7
    ))
    
    # Add box plot on top
    fig.add_trace(go.Box(
        y=df_daily[selected_metric],
        name="Box Plot",
        boxpoints='all',
        jitter=0.3,
        pointpos=-1.8,
        marker_color=ThemeConfig.SECONDARY_COLOR,
        yaxis='y2'
    ))
    
    fig.update_layout(
        title=f"{selected_metric} Distribution",
        xaxis_title=selected_metric,
        yaxis_title="Count",
        yaxis2=dict(
            overlaying='y',
            side='right',
            showgrid=False
        ),
        plot_bgcolor=ThemeConfig.CARD_BACKGROUND,
        paper_bgcolor=ThemeConfig.BACKGROUND_COLOR,
        font=dict(color=ThemeConfig.TEXT_COLOR),
        height=500,
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Player rankings for the session
    st.markdown("### Session Rankings")
    
    ranking_data = df_daily.copy()
    ranking_data["Overall Score"] = ranking_data[METRICS].mean(axis=1)
    ranking_data = ranking_data.sort_values("Overall Score", ascending=False)
    
    # Top performers
    col1, col2, col3 = st.columns(3)
    
    top_3 = ranking_data.head(3)
    medals = ["1", "2", "3"]
    
    for i, (col, (_, player)) in enumerate(zip([col1, col2, col3], top_3.iterrows())):
        with col:
            st.markdown(f"""
            <div style='text-align: center; background-color: {ThemeConfig.CARD_BACKGROUND}; padding: 20px; border-radius: 10px;'>
                <h1>{medals[i]}</h1>
                <h4>{player['Player Name']}</h4>
                <p>Score: {player['Overall Score']:.2f}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Full ranking table
    with st.expander("View Complete Rankings"):
        display_cols = ["Player Name"] + METRICS + ["Overall Score"]
        st.dataframe(
            ranking_data[display_cols].reset_index(drop=True),
            use_container_width=True
        )

def render_individual_performance(df_daily, api_key):
    """Render individual performance for daily training"""
    st.markdown("### Individual Performance Analysis")
    
    selected_player = st.selectbox(
        "Select Player:",
        sorted(df_daily["Player Name"].unique())
    )
    
    player_data = df_daily[df_daily["Player Name"] == selected_player].iloc[0]
    
    # Player metrics overview
    st.markdown(f"#### {selected_player} - Performance Metrics")
    
    # Create metric cards in grid
    cols = st.columns(4)
    for i, metric in enumerate(METRICS):
        with cols[i % 4]:
            value = player_data[metric]
            avg_value = df_daily[metric].mean()
            delta = ((value - avg_value) / avg_value * 100) if avg_value > 0 else 0
            
            st.markdown(create_metric_card(
                metric.replace("_", " ").title(),
                f"{value:.2f}",
                delta
            ), unsafe_allow_html=True)
    
    # Percentile rankings
    st.markdown("#### Percentile Rankings")
    
    percentile_data = []
    for metric in METRICS:
        all_values = df_daily[metric]
        player_value = player_data[metric]
        percentile = (all_values < player_value).sum() / len(all_values) * 100
        percentile_data.append(percentile)
    
    # Create radar chart
    fig = create_plotly_radar(
        [percentile_data],
        METRICS,
        f"{selected_player} - Session Performance Percentiles",
        [selected_player]
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # AI Assistant for Individual Performance
    individual_summary = f"""
    Individual Performance Analysis: {selected_player}
    
    Session Metrics:
    {chr(10).join([f"- {metric}: {player_data[metric]:.2f} ({percentile_data[i]:.0f}th percentile)" 
                   for i, metric in enumerate(METRICS)])}
    
    Strengths: {', '.join([METRICS[i] for i, p in enumerate(percentile_data) if p >= 75])}
    Areas to monitor: {', '.join([METRICS[i] for i, p in enumerate(percentile_data) if p < 50])}
    """
    
    display_ai_assistant("Individual Session Performance", individual_summary, api_key)
    
    # Performance insights
    st.markdown("#### Performance Insights")
    
    # Find strengths and areas for improvement
    metric_percentiles = dict(zip(METRICS, percentile_data))
    strengths = sorted(metric_percentiles.items(), key=lambda x: x[1], reverse=True)[:3]
    improvements = sorted(metric_percentiles.items(), key=lambda x: x[1])[:3]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("** Top Strengths:**")
        for metric, percentile in strengths:
            st.markdown(f"- {metric}: {percentile:.0f}th percentile")
    
    with col2:
        st.markdown("** Areas for Development:**")
        for metric, percentile in improvements:
            st.markdown(f"- {metric}: {percentile:.0f}th percentile")

def render_comparative_analysis(df_daily, api_key):
    """Render comparative analysis for daily training"""
    st.markdown("### Comparative Analysis")
    
    # Player selection
    players = st.multiselect(
        "Select 2-4 players to compare:",
        sorted(df_daily["Player Name"].unique()),
        default=sorted(df_daily["Player Name"].unique())[:3],
        max_selections=4
    )
    
    if len(players) < 2:
        st.info("Please select at least 2 players for comparison.")
        return
    
    comparison_df = df_daily[df_daily["Player Name"].isin(players)]
    
    # Metric comparison
    st.markdown("#### Metric Comparison")
    
    # Create grouped bar chart
    metrics_for_comparison = st.multiselect(
        "Select metrics to compare:",
        METRICS,
        default=METRICS[:4]
    )
    
    if metrics_for_comparison:
        # Reshape data for plotting
        plot_data = comparison_df[["Player Name"] + metrics_for_comparison].melt(
            id_vars="Player Name",
            var_name="Metric",
            value_name="Value"
        )
        
        fig = px.bar(
            plot_data,
            x="Metric",
            y="Value",
            color="Player Name",
            barmode="group",
            title="Player Metric Comparison",
            height=500,
            color_discrete_sequence=[ThemeConfig.PRIMARY_COLOR, ThemeConfig.SECONDARY_COLOR, 
                                   ThemeConfig.ACCENT_COLOR, ThemeConfig.SUCCESS_COLOR]
        )
        
        fig.update_layout(
            plot_bgcolor=ThemeConfig.CARD_BACKGROUND,
            paper_bgcolor=ThemeConfig.BACKGROUND_COLOR,
            font=dict(color=ThemeConfig.TEXT_COLOR),
            xaxis=dict(tickangle=-45)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Radar comparison
    st.markdown("#### Multi-Player Radar Analysis")
    
    # Calculate percentiles
    percentile_data = []
    for player in players:
        player_data = comparison_df[comparison_df["Player Name"] == player].iloc[0]
        player_percentiles = []
        for metric in METRICS:
            all_values = df_daily[metric]
            player_value = player_data[metric]
            percentile = (all_values < player_value).sum() / len(all_values) * 100
            player_percentiles.append(percentile)
        percentile_data.append(player_percentiles)
    
    fig = create_plotly_radar(
        percentile_data,
        METRICS,
        "Player Comparison - Percentile Rankings",
        players
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # AI Assistant for Comparison
    comparison_summary = f"""
    Player Comparison Analysis:
    Players: {', '.join(players)}
    
    Key Observations:
    {chr(10).join([f"- {metric}: " + ', '.join([f"{p} ({comparison_df[comparison_df['Player Name']==p][metric].iloc[0]:.1f})" 
                                                for p in players]) 
                   for metric in metrics_for_comparison[:3]])}
    
    Consider: Position-specific requirements, tactical roles, individual development plans
    """
    
    display_ai_assistant("Player Comparison Analysis", comparison_summary, api_key)
    
    # Comparison table
    st.markdown("#### Detailed Comparison")
    
    comparison_display = comparison_df[["Player Name"] + METRICS].set_index("Player Name")
    
    # Add averages and rankings
    comparison_display.loc["Team Average"] = df_daily[METRICS].mean()
    
    st.dataframe(
        comparison_display.style.background_gradient(cmap='RdYlGn', axis=0),
        use_container_width=True
    )

def render_player_comparison(api_key, team_config):
    """Render player comparison across matches"""
    st.markdown(f"<h2 style='color: {ThemeConfig.PRIMARY_COLOR};'> Player Comparison Tool</h2>", unsafe_allow_html=True)
    
    # Data source selection
    data_source = st.sidebar.radio(
        "Select Data Source:",
        ["Match Data", "Training Data", "Combined"]
    )
    
    # Load appropriate data
    if data_source == "Match Data":
        df = pd.concat([load_data(path) for path in MATCH_FILES.values()], ignore_index=True)
    elif data_source == "Training Data":
        df = pd.concat([load_data(path) for path in TRAINING_FILES.values()], ignore_index=True)
    else:  # Combined
        match_df = pd.concat([load_data(path) for path in MATCH_FILES.values()], ignore_index=True)
        training_df = pd.concat([load_data(path) for path in TRAINING_FILES.values()], ignore_index=True)
        df = pd.concat([match_df, training_df], ignore_index=True)
    
    # Player selection
    available_players = sorted(df["Player Name"].unique())
    selected_players = st.multiselect(
        "Select players to compare (2-4):",
        available_players,
        default=available_players[:3],
        max_selections=4
    )
    
    if len(selected_players) < 2:
        st.info("Please select at least 2 players for comparison.")
        return
    
    # Filter data
    comparison_df = df[df["Player Name"].isin(selected_players)]
    
    # Comparison type
    comparison_type = st.radio(
        "Select Comparison Type:",
        ["Overall Performance", "Head-to-Head", "Trend Analysis", "Statistical Analysis"],
        horizontal=True
    )
    
    if comparison_type == "Overall Performance":
        render_overall_performance_comparison(comparison_df, selected_players, api_key)
    elif comparison_type == "Head-to-Head":
        render_head_to_head_comparison(comparison_df, selected_players, api_key)
    elif comparison_type == "Trend Analysis":
        render_trend_comparison(comparison_df, selected_players, api_key)
    elif comparison_type == "Statistical Analysis":
        render_statistical_comparison(comparison_df, selected_players, api_key)

def render_overall_performance_comparison(df, players, api_key):
    """Render overall performance comparison"""
    st.markdown("### Overall Performance Comparison")
    
    # Calculate averages
    player_averages = df.groupby("Player Name")[METRICS].mean()
    
    # Performance scores
    st.markdown("#### Performance Scores")
    
    # Calculate weighted performance score
    weights = {
        "Total Distance": 0.2,
        "Max Speed": 0.15,
        "No of Sprints": 0.15,
        "Sprint Distance": 0.15,
        "High Speed Running": 0.15,
        "Accelerations": 0.1,
        "Decelerations": 0.1
    }
    
    performance_scores = []
    for player in players:
        if player in player_averages.index:
            score = sum(player_averages.loc[player, metric] * weight 
                       for metric, weight in weights.items() if metric in player_averages.columns)
            performance_scores.append({"Player": player, "Score": score})
    
    scores_df = pd.DataFrame(performance_scores).sort_values("Score", ascending=False)
    
    # Display performance cards
    cols = st.columns(len(players))
    for i, (_, row) in enumerate(scores_df.iterrows()):
        with cols[i]:
            rank = i + 1
            medal = "1" if rank == 1 else "2" if rank == 2 else "3" if rank == 3 else " "
            
            st.markdown(f"""
            <div style='text-align: center; background-color: {ThemeConfig.CARD_BACKGROUND}; 
                        padding: 20px; border-radius: 10px; border: 2px solid {ThemeConfig.PRIMARY_COLOR};'>
                <h1>{medal}</h1>
                <h3>{row['Player']}</h3>
                <h2 style='color: {ThemeConfig.PRIMARY_COLOR};'>{row['Score']:.1f}</h2>
                <p>Performance Score</p>
            </div>
            """, unsafe_allow_html=True)
    
    # AI Assistant for Overall Comparison
    overall_summary = f"""
    Overall Performance Comparison:
    Players: {', '.join(players)}
    
    Performance Rankings:
    {chr(10).join([f"{i+1}. {row['Player']}: {row['Score']:.1f}" for i, (_, row) in enumerate(scores_df.iterrows())])}
    
    Key Insights: Consider position-specific requirements, playing time, and role in team tactics
    """
    
    display_ai_assistant("Overall Performance Comparison", overall_summary, api_key)
    
    # Detailed metrics comparison
    st.markdown("#### Detailed Metrics")
    
    # Create spider plot
    percentile_data = []
    for player in players:
        if player in player_averages.index:
            player_percentiles = []
            for metric in METRICS:
                all_values = df[metric].dropna()
                player_value = player_averages.loc[player, metric]
                percentile = (all_values < player_value).sum() / len(all_values) * 100
                player_percentiles.append(percentile)
            percentile_data.append(player_percentiles)
    
    fig = create_plotly_radar(
        percentile_data,
        METRICS,
        "Player Performance Comparison - Percentiles",
        players
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Strengths and weaknesses matrix
    st.markdown("#### Strengths & Weaknesses Matrix")
    
    # Create heatmap
    normalized_data = (player_averages.loc[players] - player_averages.min()) / (player_averages.max() - player_averages.min())
    
    fig = px.imshow(
        normalized_data.T,
        labels=dict(x="Player", y="Metric", color="Normalized Score"),
        color_continuous_scale="RdYlGn",
        aspect="auto",
        title="Performance Heatmap (Normalized)",
        height=400
    )
    
    fig.update_layout(
        plot_bgcolor=ThemeConfig.CARD_BACKGROUND,
        paper_bgcolor=ThemeConfig.BACKGROUND_COLOR,
        font=dict(color=ThemeConfig.TEXT_COLOR)
    )
    
    st.plotly_chart(fig, use_container_width=True)

def render_head_to_head_comparison(df, players, api_key):
    """Render head-to-head comparison"""
    st.markdown("### 👥 Head-to-Head Comparison")
    
    if len(players) != 2:
        st.info("Please select exactly 2 players for head-to-head comparison.")
        # Allow reselection
        player1 = st.selectbox("Select Player 1:", players, index=0)
        player2 = st.selectbox("Select Player 2:", [p for p in players if p != player1], index=0)
        players = [player1, player2]
    
    player1, player2 = players[:2]
    
    # Get player data
    p1_data = df[df["Player Name"] == player1]
    p2_data = df[df["Player Name"] == player2]
    
    # Head-to-head metrics
    st.markdown(f"#### {player1} vs {player2}")
    
    # Create comparison cards
    for metric in METRICS:
        p1_avg = p1_data[metric].mean()
        p2_avg = p2_data[metric].mean()
        
        diff = ((p1_avg - p2_avg) / p2_avg * 100) if p2_avg > 0 else 0
        
        col1, col2, col3 = st.columns([2, 1, 2])
        
        with col1:
            color = ThemeConfig.SUCCESS_COLOR if p1_avg > p2_avg else ThemeConfig.PRIMARY_COLOR
            st.markdown(f"""
            <div style='text-align: center; background-color: {color}; 
                        padding: 15px; border-radius: 10px; color: white;'>
                <h3>{p1_avg:.2f}</h3>
                <p>{player1}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div style='text-align: center; padding-top: 20px;'>
                <p><strong>{metric}</strong></p>
                <p style='color: {ThemeConfig.ACCENT_COLOR};'>{abs(diff):.1f}% diff</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            color = ThemeConfig.SUCCESS_COLOR if p2_avg > p1_avg else ThemeConfig.PRIMARY_COLOR
            st.markdown(f"""
            <div style='text-align: center; background-color: {color}; 
                        padding: 15px; border-radius: 10px; color: white;'>
                <h3>{p2_avg:.2f}</h3>
                <p>{player2}</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
    
    # Win/Loss summary
    wins_p1 = sum(p1_data[metric].mean() > p2_data[metric].mean() for metric in METRICS)
    wins_p2 = len(METRICS) - wins_p1
    
    st.markdown("#### Summary")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div style='text-align: center; background-color: {ThemeConfig.CARD_BACKGROUND}; 
                    padding: 20px; border-radius: 10px;'>
            <h2 style='color: {ThemeConfig.PRIMARY_COLOR};'>{wins_p1}/{len(METRICS)}</h2>
            <p>Metrics Won by {player1}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div style='text-align: center; background-color: {ThemeConfig.CARD_BACKGROUND}; 
                    padding: 20px; border-radius: 10px;'>
            <h2 style='color: {ThemeConfig.SECONDARY_COLOR};'>{wins_p2}/{len(METRICS)}</h2>
            <p>Metrics Won by {player2}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # AI Assistant for Head-to-Head
    h2h_summary = f"""
    Head-to-Head Analysis: {player1} vs {player2}
    
    Metrics won:
    - {player1}: {wins_p1}/{len(METRICS)}
    - {player2}: {wins_p2}/{len(METRICS)}
    
    Key differences:
    {chr(10).join([f"- {metric}: {player1} ({p1_data[metric].mean():.1f}) vs {player2} ({p2_data[metric].mean():.1f})" 
                   for metric in METRICS[:3]])}
    
    Consider: Playing positions, roles in team, age/experience differences
    """
    
    display_ai_assistant("Head-to-Head Comparison", h2h_summary, api_key)

def render_trend_comparison(df, players, api_key):
    """Render trend comparison analysis"""
    st.markdown("### Trend Comparison Analysis")
    
    # Check if we have session/match info
    if "Session" in df.columns:
        time_column = "Session"
    else:
        # Add match info from filename matching
        df["Match"] = "Unknown"
        for match, file in MATCH_FILES.items():
            # This is a simplified approach - you'd need proper matching logic
            df.loc[df.index, "Match"] = match
        time_column = "Match"
    
    # Metric selection
    trend_metric = st.selectbox(
        "Select metric for trend analysis:",
        METRICS,
        key="trend_comp_metric"
    )
    
    # Calculate moving averages
    st.markdown(f"#### {trend_metric} Trends")
    
    # Group by time and player
    trend_data = df.groupby([time_column, "Player Name"])[trend_metric].mean().reset_index()
    trend_data = trend_data[trend_data["Player Name"].isin(players)]
    
    # Create line chart
    fig = px.line(
        trend_data,
        x=time_column,
        y=trend_metric,
        color="Player Name",
        markers=True,
        title=f"{trend_metric} Evolution",
        height=500
    )
    
    # Add trend lines
    for player in players:
        player_data = trend_data[trend_data["Player Name"] == player]
        if len(player_data) > 1:
            # Calculate trend
            x_numeric = list(range(len(player_data)))
            y_values = player_data[trend_metric].values
            z = np.polyfit(x_numeric, y_values, 1)
            p = np.poly1d(z)
            
            fig.add_trace(go.Scatter(
                x=player_data[time_column],
                y=p(x_numeric),
                mode='lines',
                name=f'{player} (Trend)',
                line=dict(dash='dash'),
                showlegend=False
            ))
    
    fig.update_layout(
        plot_bgcolor=ThemeConfig.CARD_BACKGROUND,
        paper_bgcolor=ThemeConfig.BACKGROUND_COLOR,
        font=dict(color=ThemeConfig.TEXT_COLOR),
        xaxis=dict(tickangle=-45)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Growth analysis
    st.markdown("#### Growth Analysis")
    
    growth_data = []
    for player in players:
        player_trend = trend_data[trend_data["Player Name"] == player].sort_values(time_column)
        if len(player_trend) > 1:
            first_value = player_trend.iloc[0][trend_metric]
            last_value = player_trend.iloc[-1][trend_metric]
            growth = ((last_value - first_value) / first_value * 100) if first_value > 0 else 0
            
            growth_data.append({
                "Player": player,
                "Initial": first_value,
                "Final": last_value,
                "Growth": growth,
                "Status": "Improving" if growth > 0 else "Declining" if growth < 0 else "Stable"
            })
    
    if growth_data:
        growth_df = pd.DataFrame(growth_data)
        
        for _, row in growth_df.iterrows():
            color = ThemeConfig.SUCCESS_COLOR if row["Growth"] > 0 else ThemeConfig.PRIMARY_COLOR
            st.markdown(f"""
            <div style='background-color: {ThemeConfig.CARD_BACKGROUND}; padding: 15px; 
                        border-radius: 10px; margin: 10px 0; border-left: 4px solid {color};'>
                <h4>{row['Player']} {row['Status']}</h4>
                <p>Initial: {row['Initial']:.2f} → Final: {row['Final']:.2f}</p>
                <p style='color: {color}; font-weight: bold;'>Growth: {row['Growth']:+.1f}%</p>
            </div>
            """, unsafe_allow_html=True)
    
    # AI Assistant for Trend Analysis
    trend_summary = f"""
    Trend Comparison Analysis:
    Players: {', '.join(players)}
    Metric: {trend_metric}
    
    Growth Summary:
    {chr(10).join([f"- {row['Player']}: {row['Growth']:+.1f}% ({row['Status']})" for _, row in growth_df.iterrows()]) if growth_data else "Insufficient data for growth analysis"}
    
    Consider: Training adaptations, injury history, tactical changes affecting performance
    """
    
    display_ai_assistant("Trend Comparison Analysis", trend_summary, api_key)

def render_statistical_comparison(df, players, api_key):
    """Render statistical comparison"""
    st.markdown("### Statistical Analysis")
    
    # Prepare data
    player_stats = df[df["Player Name"].isin(players)]
    
    # Distribution analysis
    st.markdown("#### Distribution Analysis")
    
    selected_metric = st.selectbox(
        "Select metric for distribution analysis:",
        METRICS,
        key="stat_metric"
    )
    
    # Create violin plot
    fig = px.violin(
        player_stats,
        y=selected_metric,
        x="Player Name",
        color="Player Name",
        box=True,
        points="all",
        title=f"{selected_metric} Distribution",
        height=500
    )
    
    fig.update_layout(
        plot_bgcolor=ThemeConfig.CARD_BACKGROUND,
        paper_bgcolor=ThemeConfig.BACKGROUND_COLOR,
        font=dict(color=ThemeConfig.TEXT_COLOR),
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Statistical summary
    st.markdown("#### Statistical Summary")
    
    summary_stats = []
    for player in players:
        player_data = df[df["Player Name"] == player][selected_metric].dropna()
        if len(player_data) > 0:
            summary_stats.append({
                "Player": player,
                "Mean": player_data.mean(),
                "Median": player_data.median(),
                "Std Dev": player_data.std(),
                "Min": player_data.min(),
                "Max": player_data.max(),
                "CV%": (player_data.std() / player_data.mean() * 100) if player_data.mean() > 0 else 0
            })
    
    summary_df = pd.DataFrame(summary_stats)
    
    # Format and display
    st.dataframe(
        summary_df.style.format({
            "Mean": "{:.2f}",
            "Median": "{:.2f}",
            "Std Dev": "{:.2f}",
            "Min": "{:.2f}",
            "Max": "{:.2f}",
            "CV%": "{:.1f}"
        }).background_gradient(subset=["Mean", "Max"], cmap='RdYlGn'),
        use_container_width=True
    )
    
    # AI Assistant for Statistical Analysis
    stat_summary = f"""
    Statistical Analysis:
    Players: {', '.join(players)}
    Metric: {selected_metric}
    
    Key Statistics:
    {chr(10).join([f"- {row['Player']}: Mean {row['Mean']:.2f}, CV% {row['CV%']:.1f}" for _, row in summary_df.iterrows()])}
    
    Insights: Lower CV% indicates more consistent performance. Consider position-specific variability expectations.
    """
    
    display_ai_assistant("Statistical Comparison", stat_summary, api_key)
    
    # Consistency analysis
    st.markdown("#### Consistency Analysis")
    
    consistency_data = []
    for player in players:
        for metric in METRICS:
            player_metric_data = df[df["Player Name"] == player][metric].dropna()
            if len(player_metric_data) > 0:
                cv = (player_metric_data.std() / player_metric_data.mean() * 100) if player_metric_data.mean() > 0 else 0
                consistency_data.append({
                    "Player": player,
                    "Metric": metric,
                    "Consistency": 100 - min(cv, 100)  # Higher score = more consistent
                })
    
    consistency_df = pd.DataFrame(consistency_data)
    
    # Create heatmap
    pivot_consistency = consistency_df.pivot(index="Metric", columns="Player", values="Consistency")
    
    fig = px.imshow(
        pivot_consistency,
        labels=dict(x="Player", y="Metric", color="Consistency %"),
        color_continuous_scale="RdYlGn",
        title="Consistency Heatmap (Higher = More Consistent)",
        height=400
    )
    
    fig.update_layout(
        plot_bgcolor=ThemeConfig.CARD_BACKGROUND,
        paper_bgcolor=ThemeConfig.BACKGROUND_COLOR,
        font=dict(color=ThemeConfig.TEXT_COLOR)
    )
    
    st.plotly_chart(fig, use_container_width=True)

def render_player_profile(api_key, team_config):
    """Render comprehensive player profile"""
    MATCH_FILES = team_config["match_files"]
    TRAINING_FILES = team_config["training_files"]
    
    st.markdown(f"<h2 style='color: {ThemeConfig.PRIMARY_COLOR};'> Player Profile</h2>", unsafe_allow_html=True)
    
    # Load all available data
    all_match_data = pd.concat([load_data(path) for path in MATCH_FILES.values()], ignore_index=True)
    all_training_data = pd.concat([load_data(path) for path in TRAINING_FILES.values()], ignore_index=True)
    
    # Add data source column
    all_match_data["Data Source"] = "Match"
    all_training_data["Data Source"] = "Training"
    
    # Combine all data
    all_data = pd.concat([all_match_data, all_training_data], ignore_index=True)
    
    # Player selection
    selected_player = st.sidebar.selectbox(
        "Select Player:",
        sorted(all_data["Player Name"].unique())
    )
    
    # Filter player data
    player_data = all_data[all_data["Player Name"] == selected_player]
    
    # Display player header
    st.markdown(f"### {selected_player} - Complete Performance Profile")
    
    # Overview metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    total_sessions = player_data.shape[0]
    avg_distance = player_data["Total Distance"].mean()
    max_speed_recorded = player_data["Max Speed"].max()
    total_sprints = player_data["No of Sprints"].sum()
    avg_load = (player_data["Accelerations"].mean() + player_data["Decelerations"].mean())
    
    with col1:
        st.markdown(create_metric_card("Sessions", f"{total_sessions}"), unsafe_allow_html=True)
    with col2:
        st.markdown(create_metric_card("Avg Distance", f"{avg_distance:.2f} km"), unsafe_allow_html=True)
    with col3:
        st.markdown(create_metric_card("Top Speed", f"{max_speed_recorded:.1f} km/h"), unsafe_allow_html=True)
    with col4:
        st.markdown(create_metric_card("Total Sprints", f"{total_sprints}"), unsafe_allow_html=True)
    with col5:
        st.markdown(create_metric_card("Avg Load", f"{avg_load:.0f}"), unsafe_allow_html=True)
    
    # AI Assistant for Player Profile Overview
    profile_summary = f"""
    Player Profile: {selected_player}
    
    Career Statistics:
    - Total Sessions: {total_sessions}
    - Average Distance: {avg_distance:.2f} km
    - Top Speed Recorded: {max_speed_recorded:.1f} km/h
    - Total Sprints: {total_sprints}
    - Average Load: {avg_load:.0f}
    
    Data includes {len(player_data[player_data['Data Source']=='Match'])} matches and {len(player_data[player_data['Data Source']=='Training'])} training sessions.
    """
    
    display_ai_assistant("Player Profile Overview", profile_summary, api_key)
    
    # Profile tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview", "Performance Trends", "Strengths Analysis", "Session History", "Report"])
    
    with tab1:
        render_player_overview(player_data, all_data, api_key)
    
    with tab2:
        render_player_trends(player_data, api_key)
    
    with tab3:
        render_player_strengths(player_data, all_data, api_key)
    
    with tab4:
        render_session_history(player_data)
    
    with tab5:
        render_player_report(player_data, all_data, selected_player, api_key)

def render_player_overview(player_data, all_data, api_key):
    """Render player overview section"""
    st.markdown("### Performance Overview")
    
    # Split by data source
    match_data = player_data[player_data["Data Source"] == "Match"]
    training_data = player_data[player_data["Data Source"] == "Training"]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Match Performance")
        if not match_data.empty:
            match_metrics = match_data[METRICS].mean()
            
            fig = go.Figure(data=[
                go.Bar(
                    x=METRICS,
                    y=match_metrics.values,
                    marker_color=ThemeConfig.PRIMARY_COLOR,
                    text=[f"{v:.1f}" for v in match_metrics.values],
                    textposition='auto',
                )
            ])
            
            fig.update_layout(
                title="Average Match Metrics",
                plot_bgcolor=ThemeConfig.CARD_BACKGROUND,
                paper_bgcolor=ThemeConfig.BACKGROUND_COLOR,
                font=dict(color=ThemeConfig.TEXT_COLOR),
                height=400,
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No match data available")
    
    with col2:
        st.markdown("#### Training Performance")
        if not training_data.empty:
            training_metrics = training_data[METRICS].mean()
            
            fig = go.Figure(data=[
                go.Bar(
                    x=METRICS,
                    y=training_metrics.values,
                    marker_color=ThemeConfig.SECONDARY_COLOR,
                    text=[f"{v:.1f}" for v in training_metrics.values],
                    textposition='auto',
                )
            ])
            
            fig.update_layout(
                title="Average Training Metrics",
                plot_bgcolor=ThemeConfig.CARD_BACKGROUND,
                paper_bgcolor=ThemeConfig.BACKGROUND_COLOR,
                font=dict(color=ThemeConfig.TEXT_COLOR),
                height=400,
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No training data available")
    
    # Performance rating
    st.markdown("#### Performance Rating")
    
    # Calculate overall performance score
    weights = {
        "Total Distance": 0.2,
        "Max Speed": 0.15,
        "No of Sprints": 0.15,
        "Sprint Distance": 0.15,
        "High Speed Running": 0.15,
        "Accelerations": 0.1,
        "Decelerations": 0.1
    }
    
    player_avg = player_data[METRICS].mean()
    all_avg = all_data[METRICS].mean()
    
    performance_score = sum(
        (player_avg[metric] / all_avg[metric]) * weight 
        for metric, weight in weights.items() 
        if metric in player_avg.index
    ) * 100
    
    # Display rating
    rating_color = (
        ThemeConfig.SUCCESS_COLOR if performance_score > 110 else
        ThemeConfig.SECONDARY_COLOR if performance_score > 90 else
        ThemeConfig.WARNING_COLOR
    )
    
    st.markdown(f"""
    <div style='text-align: center; background-color: {ThemeConfig.CARD_BACKGROUND}; 
                padding: 30px; border-radius: 10px; border: 2px solid {rating_color};'>
        <h1 style='color: {rating_color}; font-size: 4em;'>{performance_score:.0f}</h1>
        <p style='font-size: 1.2em;'>Overall Performance Score</p>
        <p style='opacity: 0.7;'>(100 = Team Average)</p>
    </div>
    """, unsafe_allow_html=True)

def render_player_trends(player_data, api_key):
    """Render player performance trends"""
    st.markdown("### Performance Trends")
    
    # Add session numbers for trending
    player_data = player_data.sort_index()
    player_data["Session_Number"] = range(1, len(player_data) + 1)
    
    # Metric selection
    trend_metrics = st.multiselect(
        "Select metrics to analyze:",
        METRICS,
        default=["Total Distance", "Max Speed", "High Speed Running"]
    )
    
    if not trend_metrics:
        st.info("Please select at least one metric to display trends.")
        return
    
    # Create subplots
    for metric in trend_metrics:
        fig = go.Figure()
        
        # Add actual values
        fig.add_trace(go.Scatter(
            x=player_data["Session_Number"],
            y=player_data[metric],
            mode='lines+markers',
            name='Actual',
            line=dict(color=ThemeConfig.PRIMARY_COLOR, width=2),
            marker=dict(size=8)
        ))
        
        # Add moving average
        window = min(5, len(player_data) // 2)
        if window >= 2:
            player_data[f"{metric}_MA"] = player_data[metric].rolling(window=window, center=True).mean()
            
            fig.add_trace(go.Scatter(
                x=player_data["Session_Number"],
                y=player_data[f"{metric}_MA"],
                mode='lines',
                name=f'{window}-Session MA',
                line=dict(color=ThemeConfig.SECONDARY_COLOR, width=2, dash='dash')
            ))
        
        # Add trend line
        if len(player_data) > 1:
            z = np.polyfit(player_data["Session_Number"], player_data[metric].fillna(0), 1)
            p = np.poly1d(z)
            
            fig.add_trace(go.Scatter(
                x=player_data["Session_Number"],
                y=p(player_data["Session_Number"]),
                mode='lines',
                name='Trend',
                line=dict(color=ThemeConfig.ACCENT_COLOR, width=2, dash='dot')
            ))
        
        fig.update_layout(
            title=f"{metric} Over Time",
            xaxis_title="Session Number",
            yaxis_title=metric,
            plot_bgcolor=ThemeConfig.CARD_BACKGROUND,
            paper_bgcolor=ThemeConfig.BACKGROUND_COLOR,
            font=dict(color=ThemeConfig.TEXT_COLOR),
            height=400,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Trend analysis
        if len(player_data) > 1:
            slope = z[0]
            trend_direction = "Improving" if slope > 0 else "Declining" if slope < 0 else "Stable"
            change_per_session = abs(slope)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Trend", trend_direction)
            with col2:
                st.metric("Change/Session", f"{change_per_session:.3f}")
            with col3:
                recent_avg = player_data[metric].tail(5).mean()
                overall_avg = player_data[metric].mean()
                recent_vs_avg = ((recent_avg - overall_avg) / overall_avg * 100) if overall_avg > 0 else 0
                st.metric("Recent vs Average", f"{recent_vs_avg:+.1f}%")
    
    # AI Assistant for Trend Analysis
    trends_summary = f"""
    Performance Trend Analysis:
    Metrics analyzed: {', '.join(trend_metrics)}
    Total sessions: {len(player_data)}
    
    Key observations needed:
    - Are trends aligned with training periodization?
    - Any concerning declines that need intervention?
    - Positive trends that should be reinforced?
    """
    
    display_ai_assistant("Player Trend Analysis", trends_summary, api_key)

def render_player_strengths(player_data, all_data, api_key):
    """Render player strengths analysis"""
    st.markdown("### Strengths & Development Areas")
    
    # Calculate percentiles
    player_avg = player_data[METRICS].mean()
    percentiles = {}
    
    for metric in METRICS:
        all_values = all_data[metric].dropna()
        player_value = player_avg[metric]
        percentile = (all_values < player_value).sum() / len(all_values) * 100
        percentiles[metric] = percentile
    
    # Create radar chart
    fig = create_plotly_radar(
        [[percentiles[m] for m in METRICS]],
        METRICS,
        "Performance Percentiles",
        [player_data["Player Name"].iloc[0]]
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Identify strengths and weaknesses
    sorted_metrics = sorted(percentiles.items(), key=lambda x: x[1], reverse=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Core Strengths")
        for metric, percentile in sorted_metrics[:3]:
            if percentile >= 75:
                badge = "Elite"
                color = ThemeConfig.SUCCESS_COLOR
            elif percentile >= 50:
                badge = "Strong"
                color = ThemeConfig.SECONDARY_COLOR
            else:
                badge = "Average"
                color = ThemeConfig.ACCENT_COLOR
            
            st.markdown(f"""
            <div style='background-color: {ThemeConfig.CARD_BACKGROUND}; padding: 10px; 
                        border-radius: 5px; margin: 5px 0; border-left: 3px solid {color};'>
                <strong>{metric}</strong> {badge}<br>
                <span style='color: {color};'>{percentile:.0f}th percentile</span>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("#### Development Opportunities")
        for metric, percentile in sorted_metrics[-3:]:
            improvement_potential = 100 - percentile
            
            st.markdown(f"""
            <div style='background-color: {ThemeConfig.CARD_BACKGROUND}; padding: 10px; 
                        border-radius: 5px; margin: 5px 0; border-left: 3px solid {ThemeConfig.WARNING_COLOR};'>
                <strong>{metric}</strong><br>
                <span style='color: {ThemeConfig.WARNING_COLOR};'>
                    {percentile:.0f}th percentile 
                    ({improvement_potential:.0f}% improvement potential)
                </span>
            </div>
            """, unsafe_allow_html=True)
    
    # AI Assistant for Strengths Analysis
    strengths_summary = f"""
    Player Strengths & Development Analysis:
    
    Core Strengths (Top 3):
    {chr(10).join([f"- {metric}: {percentile:.0f}th percentile" for metric, percentile in sorted_metrics[:3]])}
    
    Development Areas (Bottom 3):
    {chr(10).join([f"- {metric}: {percentile:.0f}th percentile" for metric, percentile in sorted_metrics[-3:]])}
    
    Recommendations needed:
    - How to leverage strengths in tactical setup
    - Specific training focus for development areas
    - Position-specific considerations
    """
    
    display_ai_assistant("Strengths & Development Analysis", strengths_summary, api_key)
    
    # Performance consistency
    st.markdown("#### Performance Consistency")
    
    consistency_scores = {}
    for metric in METRICS:
        values = player_data[metric].dropna()
        if len(values) > 1 and values.mean() > 0:
            cv = values.std() / values.mean() * 100
            consistency = 100 - min(cv, 100)
            consistency_scores[metric] = consistency
    
    if consistency_scores:
        consistency_df = pd.DataFrame(
            list(consistency_scores.items()),
            columns=["Metric", "Consistency %"]
        ).sort_values("Consistency %", ascending=False)
        
        fig = px.bar(
            consistency_df,
            x="Consistency %",
            y="Metric",
            orientation='h',
            color="Consistency %",
            color_continuous_scale=["red", "yellow", "green"],
            range_color=[0, 100],
            title="Metric Consistency (Higher = More Consistent)",
            height=400
        )
        
        fig.update_layout(
            plot_bgcolor=ThemeConfig.CARD_BACKGROUND,
            paper_bgcolor=ThemeConfig.BACKGROUND_COLOR,
            font=dict(color=ThemeConfig.TEXT_COLOR),
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)

def render_session_history(player_data):
    """Render detailed session history"""
    st.markdown("### Session History")
    
    # Session filters
    col1, col2 = st.columns(2)
    
    with col1:
        data_type_filter = st.selectbox(
            "Filter by type:",
            ["All", "Match", "Training"]
        )
    
    with col2:
        metric_to_highlight = st.selectbox(
            "Highlight metric:",
            ["None"] + METRICS
        )
    
    # Filter data
    display_data = player_data.copy()
    if data_type_filter != "All":
        display_data = display_data[display_data["Data Source"] == data_type_filter]
    
    # Prepare display columns
    display_cols = ["Session Type", "Data Source"] + METRICS
    display_data = display_data[display_cols].reset_index(drop=True)
    
    # Apply highlighting
    if metric_to_highlight != "None":
        # Create style function
        def highlight_metric(df):
            styles = pd.DataFrame('', index=df.index, columns=df.columns)
            if metric_to_highlight in df.columns:
                max_val = df[metric_to_highlight].max()
                min_val = df[metric_to_highlight].min()
                
                for idx in df.index:
                    val = df.loc[idx, metric_to_highlight]
                    if val == max_val:
                        styles.loc[idx, metric_to_highlight] = f'background-color: {ThemeConfig.SUCCESS_COLOR}'
                    elif val == min_val:
                        styles.loc[idx, metric_to_highlight] = f'background-color: {ThemeConfig.WARNING_COLOR}'
            return styles
        
        styled_df = display_data.style.apply(highlight_metric, axis=None)
    else:
        styled_df = display_data.style.background_gradient(subset=METRICS, cmap='RdYlGn')
    
    # Display table
    st.dataframe(
        styled_df.format({col: '{:.2f}' for col in METRICS}),
        use_container_width=True,
        height=400
    )
    
    # Session statistics
    with st.expander("Session Statistics"):
        session_stats = display_data[METRICS].describe().round(2)
        st.dataframe(session_stats, use_container_width=True)

def render_player_report(player_data, all_data, player_name, api_key):
    """Generate comprehensive player report"""
    st.markdown("### Player Report")
    
    st.info("Generate a comprehensive PDF report with all player analytics.")
    
    report_period = st.selectbox(
        "Select report period:",
        ["All Time", "Last 30 Days", "Last 7 Days", "Custom"]
    )
    
    if report_period == "Custom":
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date")
        with col2:
            end_date = st.date_input("End Date")
    
    include_sections = st.multiselect(
        "Select sections to include:",
        ["Executive Summary", "Performance Metrics", "Trend Analysis", 
         "Strengths & Weaknesses", "Comparisons", "Recommendations"],
        default=["Executive Summary", "Performance Metrics", "Trend Analysis"]
    )
    
    if st.button("Generate Report", type="primary"):
        with st.spinner("Generating report..."):
            # Create report content
            report_content = f"""
# Performance Report: {player_name}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}

## Executive Summary

Total Sessions Analyzed: {len(player_data)}
- Matches: {len(player_data[player_data['Data Source'] == 'Match'])}
- Training: {len(player_data[player_data['Data Source'] == 'Training'])}

### Key Performance Indicators
"""
            
            # Add metrics
            for metric in METRICS:
                avg_value = player_data[metric].mean()
                percentile = (all_data[metric] < avg_value).sum() / len(all_data[metric]) * 100
                report_content += f"\n- **{metric}**: {avg_value:.2f} ({percentile:.0f}th percentile)"
            
            # Performance rating
            performance_score = sum(
                (player_data[metric].mean() / all_data[metric].mean()) 
                for metric in METRICS
            ) / len(METRICS) * 100
            
            report_content += f"\n\n### Overall Performance Score: {performance_score:.0f}/100"
            
            # AI-generated recommendations
            if "Recommendations" in include_sections:
                report_summary = f"""
                Player: {player_name}
                Performance Score: {performance_score:.0f}/100
                Sessions: {len(player_data)}
                
                Please provide:
                1. Training recommendations
                2. Tactical suggestions
                3. Recovery considerations
                4. Development priorities
                """
                
                recommendations = get_ai_coach_insights("Player Report Recommendations", report_summary, api_key)
                report_content += f"\n\n## AI Coach Recommendations\n\n{recommendations}"
            
            # Create download button
            st.download_button(
                label="Download Report",
                data=report_content,
                file_name=f"{player_name}_report_{datetime.now().strftime('%Y%m%d')}.md",
                mime="text/markdown"
            )
            
            st.success("Report generated successfully!")

# Run the application
if __name__ == "__main__":
    main()
        


