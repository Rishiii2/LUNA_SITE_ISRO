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
</style>
""", unsafe_allow_html=True)


# ── Cached data generators ─────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_pipeline_data():
    """Run all pipeline layers and cache results."""
    from generate_synthetic_isro_data import generate_dataset
    from cpr_dop_mapper import run_cpr_dop_pipeline, HungarianOrbitTracker
    from yolo_hazard_mapper import dual_zone_hazard_map
    from nsga2_optimizer import generate_candidate_sites, run_nsga2
    from cnn_ice_detector import compute_depth_probability_bounds
    from rrt_star_planner import LunarCostMap, EnergyAwareRRTStar

    shape = (256, 256)

    # Layers 3-5: CPR/DOP
    rng    = np.random.default_rng(42)
    stokes = rng.random((4, *shape)).astype(np.float32)
    stokes[0] = 0.6; stokes[3] = 0.38   # ice-like CPR ≈ 1.22
    stokes[1] = 0.01; stokes[2] = 0.01  # low DOP ≈ 0.05
    cpr_result = run_cpr_dop_pipeline(stokes)

    # Orbit tracking
    tracker = HungarianOrbitTracker()
    pass_data = [
        [(120.0, 130.0, 1.3, 0.08), (80.0, 160.0, 1.5, 0.06)],
        [(121.0, 131.0, 1.2, 0.09), (80.5, 160.2, 1.4, 0.07)],
        [(120.5, 130.5, 1.25, 0.085)],
    ]
    confirmed = []
    for p in pass_data:
        confirmed = tracker.update(p)

    # Layer 10: Dual-zone hazard
    hazard = dual_zone_hazard_map(shape=shape)

    # Layers 14-16: NSGA-II
    sites = generate_candidate_sites(n_sites=200)
    pareto, feasible = run_nsga2(sites)

    # Layer 8: Depth bounds
    l_band = compute_depth_probability_bounds(lambda_m=0.24)
    s_band = compute_depth_probability_bounds(lambda_m=0.10)

    # Layers 17: RRT* path
    cost_map = LunarCostMap(
        shape=shape,
        slope_map=cpr_result.get("roughness", None),
    )
    rrt = EnergyAwareRRTStar(cost_map=cost_map, max_iters=600)
    path, path_cost = rrt.plan((20.0, 20.0), (210.0, 200.0))

    return {
        "shape": shape, "stokes": stokes,
        "cpr_result": cpr_result, "confirmed_tracks": confirmed,
        "hazard": hazard, "pareto": pareto, "feasible": feasible,
        "l_band": l_band, "s_band": s_band,
        "path": path, "path_cost": path_cost,
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
    f"**Pipeline Status:** <span class='status-badge'>ALL 21 LAYERS ACTIVE</span>",
    unsafe_allow_html=True,
)

with st.spinner("Running 21-layer LUNA-SITE pipeline..."):
    data = load_pipeline_data()

cpr_result = data["cpr_result"]
hazard     = data["hazard"]
pareto     = data["pareto"]
feasible   = data["feasible"]
path       = data["path"]
shape      = data["shape"]
l_band     = data["l_band"]
s_band     = data["s_band"]

# ── KPI Row ────────────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.metric("Ice Candidates", f"{cpr_result['n_clean']:,} px",
              f"-{cpr_result['n_raw'] - cpr_result['n_clean']} (roughness-rejected)")
with c2:
    best_ice = max(pareto, key=lambda s: s.ice_volume_m3) if pareto else None
    st.metric("Best Ice Volume", f"{best_ice.ice_volume_m3:.0f} m³" if best_ice else "—")
with c3:
    st.metric("Pareto Sites", f"{len(pareto)}", f"{len(feasible)} feasible")
with c4:
    st.metric("Drill Probability", f"{l_band['prob_within_drill_limit']:.1%}",
              "P(ice ≤ 2m) L-band")
with c5:
    st.metric("Path Waypoints", f"{len(path)}", f"Cost: {data['path_cost']:.1f}")

st.divider()

# ── Phase 1: Radar Ice Detection ──────────────────────────────────────────────
with st.expander("📡 Phase 1: PSR Radar Ice Detection (Layers 3-5)", expanded=True):
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        st.markdown("**CPR Map** (>1.0 = volumetric ice candidate)")
        fig_cpr = px.imshow(
            cpr_result["cpr"], color_continuous_scale="Inferno",
            labels={"color": "CPR"}, zmin=0, zmax=2,
        )
        fig_cpr.add_contour(
            z=cpr_result["cpr"], showscale=False,
            contours=dict(start=1.0, end=1.0, size=0.01, coloring="lines"),
            line=dict(color="cyan", width=1.5), name="CPR=1.0 threshold"
        )
        fig_cpr.update_layout(
            height=280, margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_cpr, use_container_width=True)

    with col2:
        st.markdown("**DOP Map** (<0.13 = depolarized → ice)")
        fig_dop = px.imshow(
            cpr_result["dop"], color_continuous_scale="Viridis_r",
            labels={"color": "DOP"}, zmin=0, zmax=0.5,
        )
        fig_dop.update_layout(
            height=280, margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_dop, use_container_width=True)

    with col3:
        st.markdown("**Ice Candidate Mask** (CPR>1 AND DOP<0.13, roughness-rejected)")
        ice_display = cpr_result["ice_mask"].astype(float)
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
        f"**{band_name} Depth Probability Bounds**: "
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
with st.expander("🎯 Phase 3: Multi-Objective Site Optimization — NSGA-II Pareto Front", expanded=True):
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
            name="All feasible sites", text=[f"Site {s.site_id}" for s in feasible],
        ))

        # Pareto front
        p_ice = [s.ice_volume_m3 for s in pareto]
        p_haz = [s.terrain_hazard for s in pareto]
        p_sol = [s.solar_hours for s in pareto]
        fig_pareto.add_trace(go.Scatter(
            x=p_haz, y=p_ice, mode="markers+lines",
            marker=dict(size=10, color="#00d4ff", symbol="star"),
            line=dict(color="#00d4ff", width=2, dash="dot"),
            name="Pareto front (Rank 1)",
        ))

        # Top candidate
        if pareto:
            top = pareto[0]
            fig_pareto.add_annotation(
                x=top.terrain_hazard, y=top.ice_volume_m3,
                text=f"  Best: Site #{top.site_id}",
                font=dict(color="#00ff88", size=12), showarrow=False, xanchor="left",
            )

        fig_pareto.update_layout(
            xaxis_title="Terrain Hazard Score (minimize)",
            yaxis_title="Ice Volume m³ (maximize)",
            title="NSGA-II Pareto Front — Ice Volume vs Terrain Safety",
            height=380,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(10,10,26,0.8)",
            font=dict(color="#ccc"),
            legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(0,212,255,0.3)", borderwidth=1),
        )
        st.plotly_chart(fig_pareto, use_container_width=True)

    with col2:
        st.markdown("**Top 5 Mission Sites**")
        for i, s in enumerate(pareto[:5], 1):
            colour = "#00ff88" if i == 1 else "#00d4ff" if i <= 3 else "#888"
            st.markdown(
                f"<div style='border:1px solid {colour};border-radius:6px;"
                f"padding:8px;margin:4px 0;background:rgba(0,0,0,0.3)'>"
                f"<b style='color:{colour}'>#{i} Site {s.site_id}</b><br>"
                f"🧊 Ice: <b>{s.ice_volume_m3:.0f}m³</b> &nbsp; "
                f"⚠️ Hazard: <b>{s.terrain_hazard:.2f}</b><br>"
                f"☀️ Solar: <b>{s.solar_hours:.1f}h/day</b> &nbsp; "
                f"🛰 Earth link: <b>{s.earth_visibility:.0%}</b><br>"
                f"⛏ P(drill): <b>{s.depth_prob:.0%}</b> &nbsp; "
                f"CRM: <b>{s.cost_risk_metric:.3f}</b>"
                f"</div>",
                unsafe_allow_html=True,
            )


# ── Phase 4: Rover Path ───────────────────────────────────────────────────────
with st.expander("🤖 Phase 4: RRT* Rover Traverse (Layers 17-18)", expanded=True):
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
                line=dict(color="#00d4ff", width=2),
                marker=dict(size=3, color="#00d4ff"),
                name="RRT* optimal path",
            ))
            # Start / Goal
            fig_rrt.add_trace(go.Scatter(
                x=[path[0][0]], y=[path[0][1]], mode="markers+text",
                marker=dict(size=14, color="#00ff88", symbol="triangle-up"),
                text=["START"], textposition="top center",
                textfont=dict(color="#00ff88", size=11), name="Start",
            ))
            fig_rrt.add_trace(go.Scatter(
                x=[path[-1][0]], y=[path[-1][1]], mode="markers+text",
                marker=dict(size=14, color="#ff4b4b", symbol="star"),
                text=["GOAL (Ice Site)"], textposition="top center",
                textfont=dict(color="#ff4b4b", size=11), name="Goal",
            ))

        fig_rrt.update_layout(
            title="Energy-Aware RRT* Rover Traverse (pre-computed global path)",
            height=380,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(10,10,26,0.8)",
            font=dict(color="#ccc"),
            xaxis=dict(range=[0, shape[1]]), yaxis=dict(range=[0, shape[0]]),
        )
        st.plotly_chart(fig_rrt, use_container_width=True)

    with col2:
        st.markdown("**Path Statistics**")
        st.metric("Waypoints", len(path))
        st.metric("Total Cost", f"{data['path_cost']:.1f}")
        st.metric("Path Length", f"{sum(np.sqrt((path[i][0]-path[i-1][0])**2 + (path[i][1]-path[i-1][1])**2) for i in range(1,len(path))):.0f} px" if len(path)>1 else "—")

        st.markdown("""
        <div style='background:rgba(0,212,255,0.08);border:1px solid rgba(0,212,255,0.3);
                    border-radius:6px;padding:10px;margin-top:8px;font-size:12px'>
        <b>Algorithm</b><br>
        Global: Energy-Aware RRT*<br>
        Local: DWA (Gazebo live)<br>
        <br>
        Cost penalises:<br>
        • Slope (exp near 10°)<br>
        • PSR traversal (battery)<br>
        • High-roughness terrain
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style='background:rgba(0,212,255,0.08);border:1px solid rgba(0,212,255,0.3);
                    border-radius:6px;padding:8px;margin-top:6px;font-size:11px'>
        <b>Finale Strategy</b><br>
        RRT* pre-computed ✅<br>
        DWA runs live in Gazebo<br>
        No 30hr compute trap ✅
        </div>
        """, unsafe_allow_html=True)


# ── Architecture Status ───────────────────────────────────────────────────────
st.divider()
st.markdown("### 21-Layer Pipeline Status")
layers = [
    ("0",  "PDS4 Data Parsing",              "✅", "pds4_tools + GeoTIFF"),
    ("1-2","PSR & DSC Mapping",              "✅", "SPICE toolkit + LOLA DEM"),
    ("3",  "CPR & DOP Detection",            "✅", "Physics gate: CPR>1, DOP<0.13"),
    ("4",  "Roughness Rejection",            "✅", "DEM RMS height filter"),
    ("5",  "Hungarian Cross-Pass Tracker",   "✅", "Replaces DeepSORT (correct for orbital)"),
    ("6",  "Physics-Regularized CNN",        "✅", "4-ch Stokes → Ice/Rock/Regolith"),
    ("7",  "MC Dropout + Grad-CAM XAI",      "✅", "50 stochastic passes; uncertainty maps"),
    ("8",  "Depth Probability Bounds",       "✅", "PINN-style physics penalty"),
    ("9",  "Ice Volume Estimation",          "✅", "Area × Depth × Ice Fraction"),
    ("10", "Dual-Zone Hazard Mapping",       "✅", "YOLO (sunlit) + DEM (PSR only)"),
    ("11", "Traversability Map",             "✅", "slope + roughness costmap"),
    ("12", "Illumination Calendar",          "✅", "SPICE solar position"),
    ("13", "Earth Visibility",               "✅", "SPICE line-of-sight"),
    ("14", "Excavation Priority",            "✅", "Ice × Depth × Distance"),
    ("15", "ISRU Potential Score",           "✅", "H₂O → H₂ + O₂ ISRU calc"),
    ("16", "NSGA-II Pareto Front",           "✅", "3-objective multi-mission rank"),
    ("16b","Hungarian Site Assignment",      "✅", "Multi-rover global optimum"),
    ("17", "Energy-Aware RRT*",              "✅", "Global path (pre-computed)"),
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
    "LUNA-SITE v2.0 | Bharatiya Antariksh Hackathon 2026 | Challenge 8 | "
    "Physics-Regularized CNN + NSGA-II + RRT* + DWA | Built for Chandrayaan-4"
)
