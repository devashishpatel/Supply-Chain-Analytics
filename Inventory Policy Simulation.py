import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from scipy.stats import norm # Used to convert Z-score to service level %

# --- Core Simulation Functions for Each Policy ---

def run_simulation(policy_type, params):
    """
    Dispatcher function to run the correct simulation based on policy type.
    """
    if policy_type == "(R, S) Periodic Review":
        return run_ts_simulation(params)
    elif policy_type == "(s, Q) Continuous Review":
        return run_sq_simulation(params)
    else:
        raise ValueError("Unknown policy type selected.")

def _base_simulation_logic(params, S_level, reorder_trigger, order_logic_func):
    """
    A base function containing the common daily simulation loop.
    """
    if params['random_seed'] is not None:
        np.random.seed(params['random_seed'])

    initial_inventory = S_level + params.get('initial_inventory_offset_from_S', 0)
    initial_inventory = max(0, initial_inventory)

    inventory_level_at_eod = [initial_inventory]
    inventory_level_at_sod = [initial_inventory]
    orders_in_transit = []
    daily_demand_realized = []
    orders_placed_log = []
    orders_received_log = []
    stock_out_days = 0
    inventory_position_log = [initial_inventory]
    total_unmet_demand = 0 # New variable to track fill rate

    for day in range(1, params['simulation_days'] + 1):
        # 1. Orders Arrive
        current_inventory_sod = inventory_level_at_eod[-1]
        quantity_received_today = 0
        remaining_orders_in_transit = []
        for order_qty, arrival_day in orders_in_transit:
            if day == arrival_day:
                quantity_received_today += order_qty
            else:
                remaining_orders_in_transit.append((order_qty, arrival_day))
        orders_in_transit = remaining_orders_in_transit
        current_inventory_sod += quantity_received_today
        inventory_level_at_sod.append(current_inventory_sod)
        if quantity_received_today > 0:
            orders_received_log.append((day, quantity_received_today, current_inventory_sod))

        # 2. Actual Demand Occurs
        actual_demand_today = max(0, int(np.random.normal(params['true_mean_daily_demand'], params['true_std_dev_daily_demand'])))
        daily_demand_realized.append(actual_demand_today)
        
        if current_inventory_sod >= actual_demand_today:
            inventory_after_demand = current_inventory_sod - actual_demand_today
        else:
            unmet_demand_today = actual_demand_today - current_inventory_sod
            total_unmet_demand += unmet_demand_today
            inventory_after_demand = 0
            stock_out_days += 1
        inventory_level_at_eod.append(inventory_after_demand)
        
        # 3. Calculate EOD Inventory Position
        total_on_order_quantity = sum(qty for qty, _ in orders_in_transit)
        current_inventory_position = inventory_after_demand + total_on_order_quantity
        inventory_position_log.append(current_inventory_position)

        # 4. Review & Place Order (using the specific policy's logic)
        new_order = order_logic_func(day, current_inventory_position, S_level, reorder_trigger, params)
        if new_order:
            quantity_to_order, inv_pos_at_order = new_order
            order_arrival_day = day + params['lead_time']
            orders_in_transit.append((quantity_to_order, order_arrival_day))
            orders_placed_log.append((day, quantity_to_order, inv_pos_at_order))
            
    # --- Generate summary statistics (common to all policies) ---
    total_actual_demand = sum(daily_demand_realized) if daily_demand_realized else 0
    service_level_days = (params['simulation_days'] - stock_out_days) / params['simulation_days'] * 100 if params['simulation_days'] > 0 else 100
    avg_eod_inventory = np.mean(inventory_level_at_eod[1:]) if len(inventory_level_at_eod) > 1 else initial_inventory
    fill_rate = ((total_actual_demand - total_unmet_demand) / total_actual_demand) * 100 if total_actual_demand > 0 else 100

    # --- Cost Calculations ---
    total_orders = len(orders_placed_log)
    total_ordering_cost = total_orders * params['order_cost']
    
    daily_holding_cost_rate = params['holding_cost_rate'] / 100 / 365
    total_holding_cost = avg_eod_inventory * params['unit_cost'] * daily_holding_cost_rate * params['simulation_days']
    
    total_inventory_cost = total_holding_cost + total_ordering_cost

    summary_stats = {
        "Stock Out Days": stock_out_days,
        "Service Level (Days)": service_level_days,
        "Fill Rate (Demand)": fill_rate, 
        "Average EOD Inventory": avg_eod_inventory,
        "Total Orders Placed": total_orders,
        "Average Order Quantity": np.mean([q for q, _, _ in orders_placed_log]) if orders_placed_log else 0,
        "Total Holding Cost": total_holding_cost,
        "Total Ordering Cost": total_ordering_cost,
        "Total Inventory Cost": total_inventory_cost,
    }
    
    return inventory_level_at_eod, inventory_position_log, daily_demand_realized, orders_placed_log, orders_received_log, inventory_level_at_sod, summary_stats

def get_common_safety_stock_params(params, risk_period):
    """Calculates forecast error std dev and safety stock."""
    if params['true_mean_daily_demand'] > 0:
        mean_absolute_error = (params['mape_forecast_error_percentage'] / 100.0) * params['true_mean_daily_demand']
        daily_forecast_error_std_dev = mean_absolute_error / 0.79788456
    else:
        daily_forecast_error_std_dev = 0.0
    daily_forecast_error_std_dev = max(0, daily_forecast_error_std_dev)
    
    combined_daily_std_dev = np.sqrt(params['true_std_dev_daily_demand']**2 + daily_forecast_error_std_dev**2)
    safety_stock_calculated = params['Z_score_for_service_level'] * combined_daily_std_dev * np.sqrt(risk_period)
    safety_stock_calculated = max(0, int(np.round(safety_stock_calculated)))
    
    return daily_forecast_error_std_dev, safety_stock_calculated

def run_ts_simulation(params):
    risk_period = params['T_review_period'] + params['lead_time']
    daily_forecast_error_std_dev, safety_stock_calculated = get_common_safety_stock_params(params, risk_period)
    
    S_order_up_to_level = int((params['true_mean_daily_demand'] * risk_period) + safety_stock_calculated)
    if S_order_up_to_level < 0: S_order_up_to_level = 0
    
    def order_logic(day, inv_pos, S_level, reorder_trigger, params):
        if day % params['T_review_period'] == 0:
            quantity_to_order = S_level - inv_pos
            if quantity_to_order > 0:
                return quantity_to_order, inv_pos
        return None

    fig_title = f"(R,S) Policy: T={params['T_review_period']}, S={S_order_up_to_level}, L={params['lead_time']}"
    reorder_point_line = None, "Reorder Point"
    
    inv_eod, inv_pos, demand, orders_placed, orders_received, inv_sod, summary = _base_simulation_logic(params, S_order_up_to_level, None, order_logic)

    summary.update({
        "Input MAPE (%)": params['mape_forecast_error_percentage'],
        "Derived Forecast Error StdDev": daily_forecast_error_std_dev,
        "Calculated Safety Stock": safety_stock_calculated,
        "Calculated Order-up-to Level (S)": S_order_up_to_level,
    })

    fig = create_simulation_plot(fig_title, params, inv_eod, inv_pos, demand, orders_placed, orders_received, inv_sod, safety_stock_calculated, S_order_up_to_level, reorder_point_line)
    return fig, summary


def run_sq_simulation(params):
    risk_period = params['lead_time']
    daily_forecast_error_std_dev, safety_stock_calculated = get_common_safety_stock_params(params, risk_period)
    
    s_reorder_point = int((params['true_mean_daily_demand'] * risk_period) + safety_stock_calculated)
    S_level_for_charting = s_reorder_point + params['Q_order_quantity']
    
    order_placed_yesterday = False
    def order_logic(day, inv_pos, S_level, reorder_trigger, params):
        nonlocal order_placed_yesterday
        if inv_pos <= reorder_trigger and not order_placed_yesterday:
            order_placed_yesterday = True
            return params['Q_order_quantity'], inv_pos
        elif inv_pos > reorder_trigger:
            order_placed_yesterday = False
        return None

    fig_title = f"(s,Q) Policy: s={s_reorder_point}, Q={params['Q_order_quantity']}, L={params['lead_time']}"
    reorder_point_line = s_reorder_point, f"Reorder Point (s = {s_reorder_point})"
    
    inv_eod, inv_pos, demand, orders_placed, orders_received, inv_sod, summary = _base_simulation_logic(params, S_level_for_charting, s_reorder_point, order_logic)

    summary.update({
        "Input MAPE (%)": params['mape_forecast_error_percentage'],
        "Derived Forecast Error StdDev": daily_forecast_error_std_dev,
        "Calculated Safety Stock": safety_stock_calculated,
        "Calculated Reorder Point (s)": s_reorder_point,
        "Order Quantity (Q)": params['Q_order_quantity'],
    })

    fig = create_simulation_plot(fig_title, params, inv_eod, inv_pos, demand, orders_placed, orders_received, inv_sod, safety_stock_calculated, S_level_for_charting, reorder_point_line)
    return fig, summary
    
def create_simulation_plot(title, params, inv_eod, inv_pos, demand, orders_placed, orders_received, inv_sod, ss_level, S_level, reorder_point_line):
    """Generic function to create the simulation plot."""
    sim_days = params['simulation_days']
    days_axis = list(range(sim_days + 1))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=days_axis, y=inv_eod, mode='lines+markers', name='Inventory Level (EOD)', line=dict(color='darkcyan', width=2)))
    fig.add_trace(go.Scatter(x=days_axis, y=inv_pos, mode='lines', name='Inventory Position (EOD)', line=dict(color='grey', dash='longdash')))
    fig.add_trace(go.Scatter(x=days_axis, y=[ss_level] * len(days_axis), mode='lines', name=f'Calculated Safety Stock ({ss_level})', line=dict(color='orange', dash='dot')))
    
    if reorder_point_line[0] is not None:
         fig.add_trace(go.Scatter(x=days_axis, y=[reorder_point_line[0]] * len(days_axis), mode='lines', name=reorder_point_line[1], line=dict(color='red', dash='dash')))

    fig.add_trace(go.Scatter(x=days_axis, y=[S_level] * len(days_axis), mode='lines', name=f'Max Level (S = {S_level})', line=dict(color='purple', dash='dashdot')))
    
    if demand:
        fig.add_trace(go.Bar(x=list(range(1, sim_days + 1)), y=demand, name='Actual Daily Demand', marker_color='lightcoral', opacity=0.6))
    if orders_placed:
        placed_days, placed_qtys, inv_pos_at_order = zip(*orders_placed)
        fig.add_trace(go.Scatter(x=placed_days, y=[pos + 2 for pos in inv_pos_at_order], mode='markers', name='Order Placed', marker=dict(color='magenta', size=10, symbol='triangle-down'), text=[f'Day: {d}<br>Qty: {q}' for d,q in zip(placed_days, placed_qtys)], hoverinfo='text+name'))
    if orders_received:
        received_days, received_qtys, _ = zip(*orders_received)
        valid_received_days = [d for d in received_days if d < len(inv_sod)]
        valid_received_y_values = [inv_sod[d] for d in valid_received_days]
        fig.add_trace(go.Scatter(x=valid_received_days, y=valid_received_y_values, mode='markers', name='Order Received', marker=dict(color='limegreen', size=10, symbol='star-diamond'), text=[f'Day: {d}<br>Qty: {q}' for d,q in zip(valid_received_days, received_qtys)], hoverinfo='text+name'))

    fig.update_layout(title_text=title, title_x=0.5, xaxis_title='Day', yaxis_title='Units', legend_title='Legend', hovermode='x unified', height=500)
    return fig

# --- Relationship Analysis Functions ---
@st.cache_data
def run_relationship_analysis(policy_type, params, mape_range, z_score_range):
    results = []
    total_runs = len(mape_range) * len(z_score_range)
    progress_bar = st.progress(0, text="Running relationship analysis...")
    run_count = 0
    
    for mape in mape_range:
        for z_score in z_score_range:
            current_params = params.copy()
            current_params['mape_forecast_error_percentage'] = mape
            current_params['Z_score_for_service_level'] = z_score
            
            _fig, stats = run_simulation(policy_type, current_params)
            results.append({
                'mape': mape,
                'z_score': z_score,
                'safety_stock': stats.get('Calculated Safety Stock', 0),
                'service_level': stats.get('Fill Rate (Demand)', 0),
                'total_cost': stats.get('Total Inventory Cost', 0)
            })
            run_count += 1
            progress_bar.progress(run_count / total_runs, text=f"Running analysis... ({run_count}/{total_runs})")
            
    progress_bar.empty()
    return pd.DataFrame(results)

# --- NEW: Policy Comparison Analysis Function ---
@st.cache_data
def run_policy_comparison_analysis(params, z_score_range):
    policy_types = ["(R, S) Periodic Review", "(s, Q) Continuous Review"]
    results = []
    total_runs = len(policy_types) * len(z_score_range)
    progress_bar = st.progress(0, text="Running policy comparison analysis...")
    run_count = 0

    for policy in policy_types:
        for z_score in z_score_range:
            current_params = params.copy()
            current_params['Z_score_for_service_level'] = z_score
            
            _fig, stats = run_simulation(policy, current_params)
            
            service_level_pct = norm.cdf(z_score) * 100

            results.append({
                'Policy': policy,
                'Target Service Level (%)': f"{service_level_pct:.1f}",
                'Safety Stock': stats.get('Calculated Safety Stock', 0),
                'Total Inventory Cost': stats.get('Total Inventory Cost', 0)
            })
            run_count += 1
            progress_bar.progress(run_count / total_runs, text=f"Comparing policies... ({run_count}/{total_runs})")
    
    progress_bar.empty()
    return pd.DataFrame(results)


def create_2d_relationship_plot(df):
    fig = go.Figure()
    unique_z_scores = sorted(df['z_score'].unique())
    for z_score in unique_z_scores:
        df_filtered = df[df['z_score'] == z_score]
        service_level_pct = norm.cdf(z_score) * 100
        fig.add_trace(go.Scatter(x=df_filtered['mape'], y=df_filtered['safety_stock'], mode='lines+markers', name=f'Target Service Level: {service_level_pct:.1f}% (Z={z_score})'))
    fig.update_layout(title="Impact of Forecast Accuracy on Required Safety Stock", xaxis_title='Forecast Accuracy (MAPE %)', yaxis_title='Required Safety Stock (units)', legend_title='Target Service Level', hovermode='x unified', height=600)
    return fig

def create_cost_relationship_plot(df):
    fig = go.Figure()
    unique_z_scores = sorted(df['z_score'].unique())
    for z_score in unique_z_scores:
        df_filtered = df[df['z_score'] == z_score]
        service_level_pct = norm.cdf(z_score) * 100
        fig.add_trace(go.Scatter(x=df_filtered['mape'], y=df_filtered['total_cost'], mode='lines+markers', name=f'Target Service Level: {service_level_pct:.1f}% (Z={z_score})', hovertemplate='<b>MAPE:</b> %{x:.1f}%<br><b>Total Cost:</b> $%{y:,.2f}<extra></extra>'))
    fig.update_layout(title="Impact of Forecast Accuracy on Total Inventory Cost", xaxis_title='Forecast Accuracy (MAPE %)', yaxis_title='Total Inventory Cost ($)', legend_title='Target Service Level', hovermode='x unified', height=600)
    return fig

# --- NEW: Policy Comparison Plotting Function ---
def create_policy_comparison_plot(df):
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Safety Stock Comparison", "Total Inventory Cost Comparison"))
    
    # Bar Chart for Safety Stock
    fig.add_trace(
        go.Bar(x=[df['Policy'], df['Target Service Level (%)']], y=df['Safety Stock'],
               text=df['Safety Stock'], textposition='auto'),
        row=1, col=1
    )
    
    # Bar Chart for Total Inventory Cost
    fig.add_trace(
        go.Bar(x=[df['Policy'], df['Target Service Level (%)']], y=df['Total Inventory Cost'],
               text=df['Total Inventory Cost'].apply(lambda x: f"${x:,.0f}"), textposition='auto'),
        row=1, col=2
    )

    fig.update_layout(
        title_text="Policy Comparison Across Different Service Levels",
        height=600,
        showlegend=False
    )
    fig.update_xaxes(title_text="Policy & Target Service Level", row=1, col=1)
    fig.update_xaxes(title_text="Policy & Target Service Level", row=1, col=2)
    fig.update_yaxes(title_text="Safety Stock (units)", row=1, col=1)
    fig.update_yaxes(title_text="Total Inventory Cost ($)", row=1, col=2)
    return fig


# --- Streamlit App Interface ---
st.set_page_config(layout="wide", page_title="Inventory Policy Simulation")

st.title("📦 Multi-Policy Inventory Simulation Tool")
st.markdown("""
This application simulates and compares three common inventory policies: **(R, S)** and **(s, Q)**.
It demonstrates how different policies, forecast accuracies (MAPE %), and costs impact inventory levels, service, and financial performance.
""")

# --- Sidebar for Inputs ---
st.sidebar.header("⚙️ Simulation Parameters")

policy = st.sidebar.selectbox("Select Inventory Policy", ["(R, S) Periodic Review", "(s, Q) Continuous Review"])

params = {}
st.sidebar.markdown("**Policy-Specific Parameters:**")
if policy == "(R, S) Periodic Review":
    params['T_review_period'] = st.sidebar.slider("Review Period (R days)", 1, 90, 7, 1, help="Inventory is reviewed every E days.")
elif policy == "(s, Q) Continuous Review":
    st.sidebar.markdown("For (s,Q), s is calculated, but Q is a fixed input.")
    params['Q_order_quantity'] = st.sidebar.number_input("Fixed Order Qty (Q)", 1, 5000, 80, 5, help="A fixed quantity to order when inventory position hits the reorder point s.")

st.sidebar.markdown("**Common Parameters:**")
params['lead_time'] = st.sidebar.slider("Lead Time (L days)", 0, 90, 7, 1)
params['simulation_days'] = st.sidebar.slider("Simulation Duration (days)", 30, 730, 360, 10)
params['Z_score_for_service_level'] = st.sidebar.slider("Z-score for Service Level", 0.1, 3.5, 1.65, 0.01, format="%.2f", help="Determines target service level. Used in Safety Stock calculation.")

st.sidebar.markdown("**True Demand Characteristics:**")
params['true_mean_daily_demand'] = st.sidebar.number_input("Mean Daily Demand (True)", 0.1, 1000.0, 20.0, 0.1, format="%.1f")
params['true_std_dev_daily_demand'] = st.sidebar.number_input("Std Dev of Daily Demand (True)", 0.0, 500.0, 5.0, 0.1, format="%.1f")

st.sidebar.markdown("**Forecast Accuracy (MAPE %):**")
params['mape_forecast_error_percentage'] = st.sidebar.number_input("Forecast MAPE (%)", 0.0, 100.0, 15.0, 0.1, format="%.1f")

st.sidebar.markdown("**Cost Parameters ($):**")
params['unit_cost'] = st.sidebar.number_input("Unit Cost ($)", 0.01, 10000.0, 50.0, 0.5, format="%.2f")
params['holding_cost_rate'] = st.sidebar.number_input("Annual Holding Cost Rate (%)", 0.0, 100.0, 25.0, 0.5, format="%.1f")
params['order_cost'] = st.sidebar.number_input("Cost per Order ($)", 0.0, 1000.0, 20.0, 1.0, format="%.2f")

st.sidebar.markdown("**Advanced:**")
params['initial_inventory_offset_from_S'] = st.sidebar.number_input("Initial Inventory Offset from S/Max Level", -1000, 1000, 0, 1)
params['random_seed'] = st.sidebar.number_input("Random Seed (0 for dynamic)", 0, 99999, 42, 1)

# --- Main Area for Output ---
main_tab, tab1, tab2, tab3 = st.tabs(["🚀 Single Simulation", "🔬 Accuracy vs. Stock", "💰 Accuracy vs. Cost", "📊 Policy Comparison"])

with main_tab:
    if st.sidebar.button("Run Single Simulation", type="primary", use_container_width=True):
        with st.spinner(f"Simulating {policy}..."):
            fig, stats = run_simulation(policy, params)
        st.header(f"Simulation Results for {policy}")
        st.plotly_chart(fig, use_container_width=True)
        st.subheader("Summary Statistics")
        formulae = {"Derived Forecast Error StdDev": "σ_fe ≈ ((MAPE/100) * μ_demand) / 0.798", "Calculated Safety Stock": "SS = Z * sqrt(σ_demand² + σ_fe²) * sqrt(Risk Period)", "Calculated Order-up-to Level (S)": "For (T,S), S = (μ_demand * (T+L)) + SS", "Calculated Reorder Point (R)": "For (R,S), R = (μ_demand * L) + SS", "Calculated Order-up-to Level (S = R+Q)": "For (R,S), S = R + Q", "Calculated Reorder Point (s)": "For (s,Q), s = (μ_demand * L) + SS", "Service Level (Days)": "100 * (Total Days - Stock Out Days) / Total Days", "Fill Rate (Demand)": "100 * (Total Demand - Unmet Demand) / Total Demand", "Average EOD Inventory": "Average of all end-of-day on-hand inventory levels.", "Total Holding Cost": "Avg EOD Inv * Unit Cost * (Holding Rate % / 365) * Sim Days", "Total Ordering Cost": "Total Orders Placed * Cost per Order", "Total Inventory Cost": "Total Holding Cost + Total Ordering Cost"}
        st.markdown("##### Financial Metrics")
        f_cols = st.columns(3)
        f_cols[0].metric("Total Inventory Cost ($)", f"${stats.get('Total Inventory Cost', 0):,.2f}", help=formulae.get('Total Inventory Cost'))
        f_cols[1].metric("Total Holding Cost ($)", f"${stats.get('Total Holding Cost', 0):,.2f}", help=formulae.get('Total Holding Cost'))
        f_cols[2].metric("Total Ordering Cost ($)", f"${stats.get('Total Ordering Cost', 0):,.2f}", help=formulae.get('Total Ordering Cost'))
        st.markdown("##### Service & Operational Metrics")
        o_cols = st.columns(4)
        o_cols[0].metric("Fill Rate (Demand %)", f"{stats.get('Fill Rate (Demand)', 0):.2f}%", help=formulae.get('Fill Rate (Demand)'))
        o_cols[1].metric("Service Level (Days %)", f"{stats.get('Service Level (Days)', 0):.2f}%", help=formulae.get('Service Level (Days)'))
        o_cols[2].metric("Avg EOD Inventory", f"{stats.get('Average EOD Inventory', 0):.2f}", help=formulae.get('Average EOD Inventory'))
        o_cols[3].metric("Total Orders Placed", f"{stats.get('Total Orders Placed', 0)}", help="Total number of replenishment orders.")
        st.markdown("##### Policy-Specific Calculations")
        p_cols = st.columns(4)
        col_idx = 0
        policy_metrics = ["Calculated Safety Stock", "Calculated Order-up-to Level (S)", "Calculated Reorder Point (R)", "Calculated Order-up-to Level (S = R+Q)", "Calculated Reorder Point (s)", "Order Quantity (Q)"]
        for key in policy_metrics:
            if key in stats:
                p_cols[col_idx % 4].metric(key, f"{stats[key]}", help=formulae.get(key, ''))
                col_idx += 1
    else:
        st.info("👈 Adjust parameters and click 'Run Single Simulation' on the sidebar to start.")

with tab1:
    st.header("🔬 The forecast Accuracy-Service-Stock Trade-off")
    st.markdown(f"This visualization explores the relationship between **Forecast Accuracy (MAPE)**, the **Safety Stock** you need to hold, and your **Target Service Level** for the **{policy}** policy.")
    if st.button("📈 Generate Stock vs. Accuracy Plot", use_container_width=True, key="stock_plot_btn"):
        with st.spinner("Running multiple simulations..."):
            mape_range = np.linspace(2, 35, 12)
            z_score_range = np.array([0.84, 1.28, 1.65, 1.96, 2.33])
            effective_seed = None if params['random_seed'] == 0 else params['random_seed']
            analysis_params = params.copy()
            if 'Q_order_quantity' not in analysis_params: analysis_params['Q_order_quantity'] = 80
            analysis_df = run_relationship_analysis(policy, analysis_params, mape_range=mape_range, z_score_range=z_score_range)
            relationship_fig_2d = create_2d_relationship_plot(analysis_df)
            st.plotly_chart(relationship_fig_2d, use_container_width=True)
            with st.expander("Show Raw Analysis Data"):
                st.dataframe(analysis_df.style.format({'mape': '{:.1f}%', 'z_score': '{:.2f}', 'safety_stock': '{:.0f} units', 'service_level': '{:.2f}%'}))

with tab2:
    st.header("💰 The Forecast Accuracy-Service-Cost Trade-off")
    st.markdown(f"This visualization shows the financial impact of the trade-offs for the **{policy}** policy.")
    if st.button("📊 Generate Cost vs. Accuracy Plot", use_container_width=True, key="cost_plot_btn"):
        with st.spinner("Running multiple simulations..."):
            mape_range = np.linspace(2, 35, 12)
            z_score_range = np.array([0.84, 1.28, 1.65, 1.96, 2.33])
            effective_seed = None if params['random_seed'] == 0 else params['random_seed']
            analysis_params = params.copy()
            if 'Q_order_quantity' not in analysis_params: analysis_params['Q_order_quantity'] = 80
            analysis_df = run_relationship_analysis(policy, analysis_params, mape_range=mape_range, z_score_range=z_score_range)
            cost_relationship_fig = create_cost_relationship_plot(analysis_df)
            st.plotly_chart(cost_relationship_fig, use_container_width=True)
            with st.expander("Show Raw Analysis Data"):
                st.dataframe(analysis_df.style.format({'mape': '{:.1f}%', 'z_score': '{:.2f}', 'safety_stock': '{:.0f}', 'service_level': '{:.2f}%', 'total_cost': '${:,.2f}'}))

with tab3:
    st.header("📊 Policy Comparison Analysis")
    st.markdown("""
    This analysis compares the performance of all three inventory policies across different service levels, using the **Forecast MAPE** and **Cost Parameters** set in the sidebar.
    This helps answer the strategic question: "Which policy is best for my business objectives?"
    - Note that continuous review policies ((R,S) and (s,Q)) generally require less safety stock than periodic review ((T,S)) because they are not exposed to demand uncertainty during the review period (T).
    """)
    if st.button("⚖️ Generate Policy Comparison Plot", use_container_width=True, key="policy_comp_btn"):
        with st.spinner("Running simulations for all policies..."):
            z_score_range = np.array([0.84, 1.28, 1.65, 2.33]) # 80%, 90%, 95%, 99%
            effective_seed = None if params['random_seed'] == 0 else params['random_seed']
            analysis_params = params.copy()
            if 'Q_order_quantity' not in analysis_params: analysis_params['Q_order_quantity'] = 80
            if 'T_review_period' not in analysis_params: analysis_params['T_review_period'] = 7
            
            comparison_df = run_policy_comparison_analysis(analysis_params, z_score_range)
            comparison_fig = create_policy_comparison_plot(comparison_df)
            st.plotly_chart(comparison_fig, use_container_width=True)
            with st.expander("Show Raw Comparison Data"):
                st.dataframe(comparison_df.style.format({'Safety Stock': '{:.0f}', 'Total Inventory Cost': '${:,.2f}'}))


st.markdown("---")
with st.expander("📘 Key Formulae Used in the Simulation"):
    st.markdown("""
    #### General Formulae
    - **Derived Forecast Error Std Dev (`σ_fe`)**: $\\sigma_{fe} \\approx \\frac{(\\frac{MAPE}{100} \\times \\mu_{demand})}{0.798}$
    - **Combined Std Dev (`σ_c`)**: $\\sigma_c = \\sqrt{\\sigma_{demand}^2 + \\sigma_{fe}^2}$
    - **Safety Stock (SS)**: $SS = Z \\times \\sigma_c \\times \\sqrt{\\text{Risk Period}}$
    - **Total Holding Cost**: $Avg\\ EOD\\ Inv \\times Unit\\ Cost \\times \\frac{Annual\\ Holding\\ Rate \\%}{365} \\times Sim\\ Days$
    - **Total Ordering Cost**: $Total\\ Orders\\ Placed \\times Cost\\ per\\ Order$
    - **Fill Rate (%)**: $100 \\times \\frac{Total\\ Demand - Unmet\\ Demand}{Total\\ Demand}$
    ---
    #### Policy-Specific Formulae
    - **(R, S) - Periodic Review**: **Risk Period** = T + L; **S** = $(\\mu_{demand} \\times (T+L)) + SS$
    - **(s, Q) - Continuous Review**: **Risk Period** = L; **s** = $(\\mu_{demand} \\times L) + SS$; **Q** is fixed.
    """)
