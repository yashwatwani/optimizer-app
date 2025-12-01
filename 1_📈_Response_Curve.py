import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---------------------------
# CONFIGURATION
# ---------------------------
FILE_PATH = "data/ROAS Data Frame.xlsx"

CHANNEL_COLORS = {"Google": "#fcde80", "Meta": "#8cbaf8"}
POINT_COLORS = {
    "Current Execution": "red",
    "Minimum Recommended": "green",
    "Maximum Recommended": "purple",
    "Most Efficient": "orange"
}
MARKERS = {
    "Current Execution": "triangle-up",
    "Minimum Recommended": "circle",
    "Maximum Recommended": "square",
    "Most Efficient": "diamond"
}

# ---------------------------
# DATA LOADING
# ---------------------------
@st.cache_data
def load_data(file_path):
    try:
        xls = pd.ExcelFile(file_path)
        sheet_names = xls.sheet_names
        all_dfs = {
            sheet: pd.read_excel(xls, sheet_name=sheet)
            for sheet in sheet_names
        }
        return pd.concat(
            [df.assign(Port=port) for port, df in all_dfs.items()],
            ignore_index=True
        )
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return pd.DataFrame()

# ---------------------------
# COMMON LEGEND
# ---------------------------
def show_common_legend():
    fig = go.Figure()

    # ROAS areas
    for channel, color in CHANNEL_COLORS.items():
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode="lines", line=dict(color=color, width=10),
            name=f"{channel} ROAS Area"
        ))

    # Revenue line
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode="lines", line=dict(color="black", width=3),
        name="Total Revenue"
    ))

    # Key points
    for label, color in POINT_COLORS.items():
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode="markers",
            marker=dict(symbol=MARKERS[label], color=color, size=12,
                        line=dict(color="black", width=1.2)),
            name=label
        ))

    # CLEAN LEGEND LAYOUT
    fig.update_layout(
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(size=14),
            bgcolor="rgba(0,0,0,0)",
            bordercolor="rgba(0,0,0,0)"
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        height=60,
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=dict(showticklabels=False, visible=False),
        yaxis=dict(showticklabels=False, visible=False)
    )

    st.plotly_chart(fig, use_container_width=True)

# ---------------------------
# ---------------------------
# PLOT FUNCTION (FIXED HOVER LABELS)
# ---------------------------
def plot_roas_curve(data_row, port, channel, month):
    channel_data = data_row.reset_index(drop=True)

    total_cost = channel_data['Total Cost']
    total_revenue = channel_data['Total Revenue']
    roas = channel_data['ROAS']
    incremental_roi = channel_data['Incremental ROI']

    current_execution = channel_data['Current Execution'].iloc[0]
    minimum_recommended = channel_data['Minimum Recommended'].iloc[0]
    most_efficient_idx = roas.idxmax()
    most_efficient_cost = total_cost[most_efficient_idx]

    max_recommended_cost = None
    roi_filtered = incremental_roi.iloc[15:]
    if not roi_filtered[roi_filtered < 1].empty:
        first_below_one_idx = roi_filtered[roi_filtered < 1].index[0]
        if first_below_one_idx - 1 >= 0:
            max_recommended_cost = total_cost.iloc[first_below_one_idx - 1]

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # 1. ROAS area (behind) - Name is needed for hover text context
    fig.add_trace(go.Scatter(
        x=total_cost, y=roas,
        fill='tozeroy', mode='lines',
        fillcolor=CHANNEL_COLORS.get(channel, "#C71585"),
        line=dict(width=0), opacity=0.75,
        name='ROAS', # Name trace for hover purposes
        hovertemplate='Spends: $%{x:.0f}<br>ROAS: %{y:.2f}<extra></extra>',
        showlegend=False
    ), secondary_y=False)

    # 2. Revenue line (front) - Name is crucial for hover text
    fig.add_trace(go.Scatter(
        x=total_cost, y=total_revenue,
        mode='lines', line=dict(color='black', width=3),
        name='Total Revenue', # Name trace for hover purposes
        #hovertemplate='Spends: $%{x:.0f}<br>Revenue: %{y:.0f}<extra></extra>',
        hovertemplate='Revenue: $%{y:.0f}<extra></extra>',
        showlegend=False
    ), secondary_y=True)

    # 3. Markers (Each point is a separate trace, name is crucial)
    points_to_plot = {
        "Current Execution": current_execution,
        "Minimum Recommended": minimum_recommended,
        "Most Efficient": most_efficient_cost
    }
    if max_recommended_cost is not None:
        points_to_plot["Maximum Recommended"] = max_recommended_cost

    for label, cost in points_to_plot.items():
        idx = (total_cost - cost).abs().idxmin()
        rev = total_revenue[idx]
        fig.add_trace(go.Scatter(
            x=[cost], y=[rev],
            mode="markers+text",
            name=label, # Name trace for hover purposes
            marker=dict(color=POINT_COLORS[label],
                        symbol=MARKERS[label],
                        size=14, line=dict(color='black', width=1.2)),
            text=[f"${int(round(cost)):,}"],
            textposition="top center",
            textfont=dict(size=13),
            # Custom hover template includes the label name via %{trace name}
            hovertemplate=f"{label}<br>Spends: ${int(round(cost)):,}<br>Revenue: %{{y:.0f}}<extra></extra>",
            showlegend=False
        ), secondary_y=True)

    # 4. Vertical line
    fig.add_vline(
        x=most_efficient_cost, line_dash="dash",
        line_color="orange", line_width=2,
        name='Most Efficient Line' # Name for hover purposes
    )

    # 5. Layout Customization
    fig.update_layout(
        title={
            'text': f"{port} - {channel} - {month}",
            'x': 0.5, 'xanchor': 'center', 'yanchor': 'top', 'font': dict(size=18, color="black")
        },
        height=600,
        plot_bgcolor="white",
        hovermode="x unified",
        # Custom hover label settings: makes the hover box cleaner
        hoverlabel=dict(namelength=0, font_size=18), 
        xaxis=dict(title="Spends (USD)",showgrid=False),
        yaxis=dict(title="ROAS", showgrid=False),
        yaxis2=dict(title="Revenue",showgrid=False)
    )

    st.plotly_chart(fig, use_container_width=True)
# ---------------------------
# MAIN STREAMLIT APP
# ---------------------------
def app_response_curve():
    st.set_page_config(page_title="Response Curves", layout="wide")
    st.title("📈 Channel Response Curve Viewer")

    full_df = load_data(FILE_PATH)
    if full_df.empty:
        st.warning("File not found or invalid format.")
        return

    # --- Sidebar filters ---
    st.sidebar.header("Filter Selection")
    selected_port = st.sidebar.selectbox("Select Port:", sorted(full_df['Port'].unique()))
    selected_month = st.sidebar.selectbox("Select Month:", sorted(full_df['Month'].unique()))

    filtered_data = full_df[
        (full_df['Port'] == selected_port) & (full_df['Month'] == selected_month)
    ]

    if filtered_data.empty:
        st.warning("No data found for this selection.")
        return

    # Show legend (clean horizontal)
    show_common_legend()

    # Plot for each channel
    available_channels = sorted(filtered_data['Channel'].unique().tolist())
    #st.subheader(f"Comparison for {selected_port} - {selected_month}")

    if len(available_channels) == 2:
        col1, col2 = st.columns(2)
        for i, channel in enumerate(available_channels):
            channel_data = filtered_data[filtered_data['Channel'] == channel]
            if not channel_data.empty:
                with (col1 if i == 0 else col2):
                    st.markdown(f"### 📊 {channel}")
                    plot_roas_curve(channel_data, selected_port, channel, selected_month)
    else:
        for channel in available_channels:
            channel_data = filtered_data[filtered_data['Channel'] == channel]
            if not channel_data.empty:
                st.markdown(f"### 📊 {channel} Channel")
                plot_roas_curve(channel_data, selected_port, channel, selected_month)

    st.markdown("---")
    #st.success("✅ All available channel curves displayed for comparison.")

# ---------------------------
# RUN APP
# ---------------------------
if __name__ == "__main__":
    app_response_curve()
