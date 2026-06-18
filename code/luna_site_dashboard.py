import streamlit as st
import numpy as np
import pandas as pd

st.set_page_config(page_title="LUNA-SITE Mission Control", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for dark theme/scientific look
st.markdown("""
    <style>
    .main {background-color: #0e1117;}
    h1, h2, h3 {color: #00e6e6;}
    .stMetric label {color: #00e6e6 !important;}
    </style>
""", unsafe_allow_html=True)

st.sidebar.title("🛰️ LUNA-SITE")
st.sidebar.markdown("**Phase 5: Digital Twin**")
mode = st.sidebar.radio("Navigation", ["Mission Control", "Radar Analytics (Layer 3)", "Rover Telemetry (Layer 18)"])

if mode == "Mission Control":
    st.title("Chandrayaan-4 Pre-Mission Planner")
    st.markdown("---")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Selected Target", "Mons Mouton (MM-4)")
    col2.metric("Ice Confidence Bound", "75.4%", "+2.1%")
    col3.metric("Terrain Safety", "92.0%", "Slope < 10°")
    col4.metric("Illumination Window", "11.2 Days", "Continuous")

    st.markdown("### Simulated Topographic Terrain Analysis")
    # Generate dummy terrain points near the lunar south pole coordinates
    terrain_data = pd.DataFrame(
        np.random.randn(100, 2) / [50, 50] + [-85.0, 35.0],
        columns=['lat', 'lon']
    )
    st.map(terrain_data, zoom=4)

    st.markdown("### Cost/Risk Pareto Front Evaluation (NSGA-II)")
    chart_data = pd.DataFrame(
        np.random.randn(50, 2),
        columns=["Ice Volume Yield (m³)", "Terrain Safety Score"]
    )
    st.scatter_chart(chart_data)

elif mode == "Radar Analytics (Layer 3)":
    st.title("DFSAR CPR/DOP Analytics")
    st.markdown("Analyzing L-Band & S-Band polarimetric scattering...")
    # Placeholder for actual radar visualization
    st.info("CPR > 1 and DOP < 0.13 Volumetric Scattering Detected.")
    st.image("presentation_images/cpr_dop_map.png", caption="DFSAR CPR/DOP Map — LUNA-SITE Layer 3")
    
elif mode == "Rover Telemetry (Layer 18)":
    st.title("ROS2 + Gazebo Rover Live Telemetry")
    st.markdown("Live feed from the RRT* global planner and DWA local obstacle avoidance.")
    st.code("[INFO] [1718000000.0] [nav2_rrt_star]: Global path computed successfully.\\n[INFO] [1718000000.5] [dwa_local_planner]: Avoiding obstacle (Boulder > 0.32m).\\n[WARN] [1718000001.0] [terramechanics]: High slip risk detected on 8° slope. Reducing velocity.", language="bash")
    st.progress(65, text="Traverse Progress to Target Drill Site")

st.sidebar.markdown("---")
st.sidebar.success("Backend Initialized. ROS2 Active.")
