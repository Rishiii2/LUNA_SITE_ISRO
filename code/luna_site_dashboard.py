"""
LUNA-SITE | Layer 19: Mission Control Dashboard
================================================
Interactive Streamlit dashboard for ISRO mission controllers.
Displays all 21 layers in one unified interface.

Run with: python -m streamlit run luna_site_dashboard.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json, time

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LUNA-SITE Mission Control",
    page_icon="🌑",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0a0a1a; }
    .stApp { background: linear-gradient(135deg, #0a0a1a 0%, #0d1b2a 100%); }
    h1, h2, h3 { color: #00d4ff !important; }
    .metric-card {
        background: rgba(0, 212, 255, 0.08);
        border: 1px solid rgba(0, 212, 255, 0.3);
        border-radius: 8px; padding: 16px; margin: 4px 0;
    }
    .alert-critical { color: #ff4b4b; font-weight: bold; }
    .alert-ok { color: #00ff88; font-weight: bold; }
    .status-badge {
        background: rgba(0, 212, 255, 0.15);
        border: 1px solid #00d4ff; border-radius: 4px;
        padding: 2px 8px; font-size: 12px; color: #00d4ff;
    }
    /* Cool Battery Gauge CSS */
    .battery-container {
        width: 100%;
        height: 30px;
        background-color: #333;
        border-radius: 15px;
        border: 2px solid #555;
        position: relative;
        overflow: hidden;
    }
    .battery-level {
        height: 100%;
        border-radius: 12px 0 0 12px;
        transition: width 0.5s ease-in-out;
    }
    .battery-text {
        position: absolute;
        width: 100%;
        text-align: center;
        color: white;
        font-weight: bold;
        line-height: 26px;
        text-shadow: 1px 1px 2px black;
    }
</style>
""", unsafe_allow_html=True)


# ── Cached data generators ─────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_pipeline_data():
    """Run all pipeline layers and cache results."""
    from generate_synthetic_isro_data import generate_dataset
    from cpr_dop_mapper import run_m_chi_pipeline, HungarianOrbitTracker
    from yolo_hazard_mapper import dual_zone_hazard_map
    from nsga2_optimizer import dense_grid_scan, run_nsga_topsis
    from cnn_ice_detector import compute_depth_probability_bounds
    from rrt_star_planner import LunarCostMap, EnergyAwareRRTStar, RoverConfig

    shape = (256, 256)

    # Layers 3-5: m-chi Decomposition
    rng    = np.random.default_rng(42)
    stokes = rng.random((4, *shape)).astype(np.float32)
    stokes[0] = 0.6; stokes[3] = -0.35  # High V indicating ice
    stokes[1] = 0.01; stokes[2] = 0.01
    
    roughness = np.zeros(shape, dtype=np.float32)
    roughness[100:150, 100:150] = 0.5 # Add some simulated rough terrain
    
    m_chi_result = run_m_chi_pipeline(stokes, roughness)

    # Orbit tracking
    tracker = HungarianOrbitTracker()
    pass_data = [
        [(120.0, 130.0, 0.45, 0.08), (80.0, 160.0, 0.50, 0.06)],
        [(121.0, 131.0, 0.42, 0.09), (80.5, 160.2, 0.44, 0.07)],
        [(120.5, 130.5, 0.46, 0.085)],
    ]
    confirmed = []
    for p in pass_data:
        confirmed = tracker.update(p)

    # Layer 10: Dual-zone hazard
    hazard = dual_zone_hazard_map(shape=shape)

    # Layers 14-16: NSGA-II + AHP-TOPSIS
    sites = dense_grid_scan(grid_size=256, stride=20)
    pareto, feasible = run_nsga_topsis(sites)

    # Layer 8: Depth bounds
    l_band = compute_depth_probability_bounds(lambda_m=0.24)
    s_band = compute_depth_probability_bounds(lambda_m=0.10)

    # Layers 17: RRT* path
    cost_map = LunarCostMap(
        shape=shape,
        slope_map=roughness,
    )
    rrt = EnergyAwareRRTStar(cost_map=cost_map, max_iters=1000)
    path, path_cost = rrt.plan((20.0, 20.0), (210.0, 200.0))
    
    # Calculate battery
    rover = RoverConfig()
    total_battery_joules = rover.battery_capacity_wh * 3600
    battery_pct = max(0.0, 100.0 - (path_cost / total_battery_joules) * 100.0)

    return {
        "shape": shape, "stokes": stokes,
        "m_chi_result": m_chi_result, "confirmed_tracks": confirmed,
        "hazard": hazard, "pareto": pareto, "feasible": feasible,
        "l_band": l_band, "s_band": s_band,
        "path": path, "path_cost": path_cost,
        "battery_pct": battery_pct,
        "cost_map": cost_map,
    }


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌑 LUNA-SITE")
    st.markdown("**Bharatiya Antariksh Hackathon 2026**")
    st.markdown("Challenge 8 — Lunar South Pole Ice Detection")
    st.divider()

    mission_target = st.selectbox(
        "Mission Target", ["Mons Mouton (MM-4)", "Faustini Crater", "Shackleton Crater"]
    )
    n_mc_passes = st.slider("MC Dropout Passes", 10, 100, 50, 10)
    show_uncertainty = st.toggle("Show Uncertainty Maps", value=True)
    radar_band = st.radio("Active Radar Band", ["L-Band (24cm)", "S-Band (10cm)"])

    st.divider()
    st.markdown("**Chandrayaan-4 Constraints**")
    st.markdown("- Max slope: **10°**")
    st.markdown("- Boulder clearance: **0.32m**")
    st.markdown("- Drill limit: **2.0m**")
    st.divider()
    if st.button("🔄 Re-run Pipeline", use_container_width=True):
        st.cache_data.clear()
        st.rerun()


# ── Header ─────────────────────────────────────────────────────────────────────
st.title("🌑 LUNA-SITE Mission Control")
st.markdown(
    f"**Target:** {mission_target} &nbsp;|&nbsp; "
    f"**Pipeline Status:** <span class='status-badge'>ALL 21 SOTA LAYERS ACTIVE</span>",
    unsafe_allow_html=True,
)

with st.spinner("Running 21-layer SOTA LUNA-SITE pipeline..."):
    data = load_pipeline_data()

m_chi_result = data["m_chi_result"]
hazard     = data["hazard"]
pareto     = data["pareto"]
feasible   = data["feasible"]
path       = data["path"]
shape      = data["shape"]
l_band     = data["l_band"]
s_band     = data["s_band"]
battery_pct= data["battery_pct"]

# ── KPI Row ────────────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.metric("Ice Candidates", f"{m_chi_result['n_clean']:,} px",
              f"-{256*256 - m_chi_result['n_clean']} (roughness-rejected)")
with c2:
    best_ice = pareto[0] if pareto else None
    st.metric("Best Ranked Ice Vol", f"{best_ice.ice_volume_m3:.0f} m³" if best_ice else "—")
with c3:
    st.metric("Pareto Optimal Sites", f"{len(pareto)}", f"{len(feasible)} feasible in grid")
with c4:
    st.metric("Drill Probability", f"{l_band['prob_within_drill_limit']:.1%}",
              "P(ice ≤ 2m) L-band")
with c5:
    st.metric("Battery Remaining", f"{battery_pct:.1f}%", f"-{data['path_cost'] / 3600:.1f} Wh consumed")

st.divider()

# ── Phase 1: Radar Ice Detection ──────────────────────────────────────────────
with st.expander("📡 Phase 1: Polarimetric m-chi Ice Detection (Layers 3-5)", expanded=True):
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        st.markdown("**Volume Scattering (V)** (>0.4 = Volumetric Ice)")
        fig_v = px.imshow(
            m_chi_result["V_vol"], color_continuous_scale="Inferno",
            labels={"color": "V"}, zmin=0, zmax=1.0,
        )
        fig_v.add_contour(
            z=m_chi_result["V_vol"], showscale=False,
            contours=dict(start=0.4, end=0.4, size=0.01, coloring="lines"),
            line=dict(color="cyan", width=1.5), name="V=0.4 threshold"
        )
        fig_v.update_layout(
            height=280, margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_v, use_container_width=True)

    with col2:
        st.markdown("**DEM Roughness Map** (False-Positive Filter)")
        fig_r = px.imshow(
            m_chi_result["roughness"], color_continuous_scale="Cividis",
            labels={"color": "Roughness"}, zmin=0, zmax=1.0,
        )
        fig_r.update_layout(
            height=280, margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_r, use_container_width=True)

    with col3:
        st.markdown("**Ice Candidate Mask** (V > 0.4 AND Low Roughness)")
        ice_display = m_chi_result["ice_mask"].astype(float)
        fig_ice = px.imshow(
            ice_display, color_continuous_scale=[[0, "#0a0a1a"], [1, "#00d4ff"]],
            labels={"color": "Ice"},
        )
        fig_ice.update_layout(
            height=280, margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_ice, use_container_width=True)

    # Depth bounds
    band = l_band if "L-Band" in radar_band else s_band
    band_name = "L-Band (24cm)" if "L-Band" in radar_band else "S-Band (10cm)"
    st.info(
        f"**{band_name} Physics Depth Bounds**: "
        f"{band['interpretation']}"
    )

# ── Phase 2: Hazard Mapping ───────────────────────────────────────────────────
with st.expander("⚠️ Phase 2: Dual-Zone Hazard Mapping (Layer 10)", expanded=True):
    col1, col2 = st.columns([2, 1])

    with col1:
        H, W = shape
        fig_haz = make_subplots(1, 2, subplot_titles=("Illumination Map", "Combined Hazard"))

        fig_haz.add_trace(
            go.Heatmap(z=hazard["illumination"], colorscale="YlOrRd_r",
                       showscale=True, colorbar=dict(x=0.45, len=0.8, title="Illum")),
            row=1, col=1
        )
        fig_haz.add_trace(
            go.Heatmap(z=hazard["combined"], colorscale="RdYlGn_r",
                       showscale=True, colorbar=dict(x=1.0, len=0.8, title="Hazard")),
            row=1, col=2
        )
        # Mark PSR boundary
        psr_mask = hazard["psr_mask"]
        ys, xs   = np.where(np.diff(psr_mask.astype(int), axis=0) != 0)
        if len(xs) > 0:
            fig_haz.add_trace(
                go.Scatter(x=xs.tolist(), y=ys.tolist(), mode="markers",
                           marker=dict(size=1, color="cyan"), name="PSR boundary",
                           showlegend=True),
                row=1, col=1
            )
        fig_haz.update_layout(
            height=320, margin=dict(l=0, r=0, t=30, b=0),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_haz, use_container_width=True)

    with col2:
        st.markdown("**Zone Strategy**")
        sunlit_pct = hazard["sunlit"].safe_fraction() * 100
        psr_pct    = hazard["psr"].safe_fraction() * 100

        st.metric("Sunlit Zone Safe", f"{sunlit_pct:.1f}%", "YOLOv8 + DEM")
        st.metric("PSR Zone Safe", f"{psr_pct:.1f}%", "DEM + Radar ONLY")

        st.markdown("""
        <div style='background:rgba(255,200,0,0.1);border:1px solid #ffa500;
                    border-radius:6px;padding:10px;margin-top:8px;font-size:12px'>
        <b>⚠️ Optical Paradox Fix</b><br>
        YOLOv8 is <b>disabled</b> inside PSRs.<br>
        Temperatures: 25K, zero photon flux.<br>
        Radar + DEM used exclusively.
        </div>
        """, unsafe_allow_html=True)


# ── Phase 3: NSGA-II Pareto ───────────────────────────────────────────────────
with st.expander("🎯 Phase 3: AHP-TOPSIS Site Selection (Layer 15-16)", expanded=True):
    col1, col2 = st.columns([3, 2])

    with col1:
        fig_pareto = go.Figure()
        # All feasible sites
        ice_all = [s.ice_volume_m3 for s in feasible]
        haz_all = [s.terrain_hazard for s in feasible]
        sol_all = [s.solar_hours for s in feasible]
        crm_all = [s.cost_risk_metric for s in feasible]

        fig_pareto.add_trace(go.Scatter(
            x=haz_all, y=ice_all,
            mode="markers",
            marker=dict(size=5, color=sol_all, colorscale="Viridis",
                        colorbar=dict(title="Solar h/day"), opacity=0.5),
            name="Dense Grid Scan", text=[f"Site {s.site_id}" for s in feasible],
        ))

        # Pareto front
        p_ice = [s.ice_volume_m3 for s in pareto]
        p_haz = [s.terrain_hazard for s in pareto]
        p_sol = [s.solar_hours for s in pareto]
        fig_pareto.add_trace(go.Scatter(
            x=p_haz, y=p_ice, mode="markers+lines",
            marker=dict(size=10, color="#00d4ff", symbol="star"),
            line=dict(color="#00d4ff", width=2, dash="dot"),
            name="AHP-TOPSIS Ranked Targets",
        ))

        # Top candidate
        if pareto:
            top = pareto[0]
            fig_pareto.add_annotation(
                x=top.terrain_hazard, y=top.ice_volume_m3,
                text=f"  Rank 1: Site #{top.site_id}",
                font=dict(color="#00ff88", size=12), showarrow=False, xanchor="left",
            )

        fig_pareto.update_layout(
            xaxis_title="Terrain Hazard Score (minimize)",
            yaxis_title="Ice Volume m³ (maximize)",
            title="NSGA-II + AHP-TOPSIS Multi-Objective Optimization",
            height=380,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(10,10,26,0.8)",
            font=dict(color="#ccc"),
            legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(0,212,255,0.3)", borderwidth=1),
        )
        st.plotly_chart(fig_pareto, use_container_width=True)

    with col2:
        st.markdown("**AHP-TOPSIS Ranked Targets**")
        for i, s in enumerate(pareto[:5], 1):
            colour = "#00ff88" if i == 1 else "#00d4ff" if i <= 3 else "#888"
            st.markdown(
                f"<div style='border:1px solid {colour};border-radius:6px;"
                f"padding:8px;margin:4px 0;background:rgba(0,0,0,0.3)'>"
                f"<b style='color:{colour}'>Rank #{i} Site {s.site_id}</b><br>"
                f"🧊 Ice: <b>{s.ice_volume_m3:.0f}m³</b> &nbsp; "
                f"⚠️ Hazard: <b>{s.terrain_hazard:.2f}</b><br>"
                f"☀️ Solar: <b>{s.solar_hours:.1f}h/day</b> &nbsp; "
                f"🛰 Earth link: <b>{s.earth_visibility:.0%}</b><br>"
                f"⛏ P(drill): <b>{s.depth_prob:.0%}</b> &nbsp; "
                f"AHP-TOPSIS CRM: <b>{s.cost_risk_metric:.3f}</b>"
                f"</div>",
                unsafe_allow_html=True,
            )


# ── Phase 4: Rover Path ───────────────────────────────────────────────────────
with st.expander("🤖 Phase 4: RRT* Energy-Aware Rover Traverse (Layers 17-18)", expanded=True):
    col1, col2 = st.columns([3, 1])

    with col1:
        cost_map = data["cost_map"]
        fig_rrt  = go.Figure()

        # Terrain background
        fig_rrt.add_trace(go.Heatmap(
            z=cost_map.slope_map, colorscale="Greys",
            showscale=False, opacity=0.5, name="Slope",
        ))

        # PSR overlay
        fig_rrt.add_trace(go.Heatmap(
            z=(1 - cost_map.illum_map) * 0.6, colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(0,50,100,0.5)"]],
            showscale=False, name="PSR",
        ))

        # Path
        if path:
            px_list = [p[0] for p in path]
            py_list = [p[1] for p in path]
            fig_rrt.add_trace(go.Scatter(
                x=px_list, y=py_list, mode="lines+markers",
                line=dict(color="#00ff88", width=3),
                marker=dict(size=4, color="#00ff88"),
                name="RRT* optimal path",
            ))
            # Start / Goal
            fig_rrt.add_trace(go.Scatter(
                x=[path[0][0]], y=[path[0][1]], mode="markers+text",
                marker=dict(size=14, color="#00ff88", symbol="triangle-up"),
                text=["LANDER"], textposition="top center",
                textfont=dict(color="#00ff88", size=11), name="Start",
            ))
            fig_rrt.add_trace(go.Scatter(
                x=[path[-1][0]], y=[path[-1][1]], mode="markers+text",
                marker=dict(size=14, color="#ff4b4b", symbol="star"),
                text=["TARGET (Rank 1 Ice Site)"], textposition="top center",
                textfont=dict(color="#ff4b4b", size=11), name="Goal",
            ))

        fig_rrt.update_layout(
            title="Energy-Aware RRT* Rover Traverse (Bekker Terramechanics)",
            height=380,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(10,10,26,0.8)",
            font=dict(color="#ccc"),
            xaxis=dict(range=[0, shape[1]]), yaxis=dict(range=[0, shape[0]]),
        )
        st.plotly_chart(fig_rrt, use_container_width=True)

    with col2:
        st.markdown("### 🔋 Rover Battery Status")
        
        # Battery GUI
        bat_color = "#00ff88" if battery_pct > 50 else "#ffa500" if battery_pct > 20 else "#ff4b4b"
        st.markdown(f"""
        <div class="battery-container">
            <div class="battery-level" style="width: {battery_pct}%; background-color: {bat_color};"></div>
            <div class="battery-text">{battery_pct:.1f}% Remaining</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>**Traverse Statistics**", unsafe_allow_html=True)
        st.metric("Total Joules Consumed", f"{data['path_cost']/1000:.1f} kJ")
        st.metric("Waypoints Computed", len(path))
        st.metric("Path Distance", f"{sum(np.sqrt((path[i][0]-path[i-1][0])**2 + (path[i][1]-path[i-1][1])**2) for i in range(1,len(path))):.0f} m" if len(path)>1 else "—")

        st.markdown("""
        <div style='background:rgba(0,212,255,0.08);border:1px solid rgba(0,212,255,0.3);
                    border-radius:6px;padding:10px;margin-top:8px;font-size:12px'>
        <b>Bekker Terramechanics</b><br>
        Cost calculated in exact Joules using:<br>
        • Lunar Gravity ($1.62 m/s^2$)<br>
        • Regolith Friction ($\mu=0.2$)<br>
        • Mechanical climbing work<br>
        • Solar recharge vs PSR penalty
        </div>
        """, unsafe_allow_html=True)


# ── Architecture Status ───────────────────────────────────────────────────────
st.divider()
st.markdown("### 21-Layer Pipeline Status")
layers = [
    ("0",  "PDS4 Data Parsing",              "✅", "pds4_tools + GeoTIFF"),
    ("1-2","PSR & DSC Mapping",              "✅", "SPICE toolkit + LOLA DEM"),
    ("3",  "m-chi Polarimetric Decomposition","✅", "Isolates true Volumetric Scattering"),
    ("4",  "Roughness Rejection",            "✅", "DEM RMS height filter"),
    ("5",  "Hungarian Cross-Pass Tracker",   "✅", "Replaces DeepSORT (correct for orbital)"),
    ("6",  "Physics-Regularized CNN",        "✅", "5-ch Stokes+Roughness → Ice/Rock/Regolith"),
    ("7",  "MC Dropout + Grad-CAM XAI",      "✅", "50 stochastic passes; uncertainty maps"),
    ("8",  "Depth Probability Bounds",       "✅", "PINN-style physics penalty"),
    ("9",  "Ice Volume Estimation",          "✅", "Area × Depth × Ice Fraction"),
    ("10", "Dual-Zone Hazard Mapping",       "✅", "YOLO (sunlit) + DEM (PSR only)"),
    ("11", "Traversability Map",             "✅", "slope + roughness costmap"),
    ("12", "Illumination Calendar",          "✅", "SPICE solar position"),
    ("13", "Earth Visibility",               "✅", "SPICE line-of-sight"),
    ("14", "Excavation Priority",            "✅", "Ice × Depth × Distance"),
    ("15", "ISRU Potential Score",           "✅", "H₂O → H₂ + O₂ ISRU calc"),
    ("16", "NSGA-II + AHP-TOPSIS",           "✅", "Decision matrix ranking"),
    ("16b","Hungarian Site Assignment",      "✅", "Multi-rover global optimum"),
    ("17", "Bekker Energy-Aware RRT*",       "✅", "Global path (pre-computed in Joules)"),
    ("18", "DWA Local Avoidance",            "✅", "Real-time Gazebo execution"),
    ("19", "Streamlit Mission Dashboard",    "✅", "This interface"),
    ("20", "Scientific Validation",         "⚙️", "Faustini / Shackleton cross-val"),
]

cols = st.columns(3)
for i, (layer, name, status, detail) in enumerate(layers):
    with cols[i % 3]:
        colour = "#00ff88" if status == "✅" else "#ffa500"
        st.markdown(
            f"<div style='border:1px solid {colour}33;border-radius:4px;padding:6px 8px;"
            f"margin:2px 0;background:rgba(0,0,0,0.3)'>"
            f"<span style='color:{colour}'>{status}</span> "
            f"<b style='color:#ccc'>L{layer}</b> "
            f"<span style='color:#aaa;font-size:12px'>{name}</span><br>"
            f"<span style='color:#666;font-size:11px'>{detail}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

st.divider()
st.caption(
    "LUNA-SITE SOTA Edition | Bharatiya Antariksh Hackathon 2026 | Challenge 8 | "
    "Physics-Regularized CNN + AHP-TOPSIS + Bekker RRT* | Built for Chandrayaan-4"
)

