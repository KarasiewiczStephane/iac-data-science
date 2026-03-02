"""Streamlit dashboard for infrastructure cost analysis and projection.

Displays an infrastructure cost calculator with interactive sliders,
cost breakdown charts, monthly projections, and multi-cloud comparisons
using synthetic pricing data.

Run with: streamlit run src/dashboard/app.py
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

INSTANCE_TYPES = {
    "t3.medium": {"vcpus": 2, "memory_gb": 4, "hourly_rate": 0.0416},
    "t3.xlarge": {"vcpus": 4, "memory_gb": 16, "hourly_rate": 0.1664},
    "m5.2xlarge": {"vcpus": 8, "memory_gb": 32, "hourly_rate": 0.384},
    "c5.4xlarge": {"vcpus": 16, "memory_gb": 32, "hourly_rate": 0.68},
    "r5.2xlarge": {"vcpus": 8, "memory_gb": 64, "hourly_rate": 0.504},
    "p3.2xlarge": {"vcpus": 8, "memory_gb": 61, "hourly_rate": 3.06},
    "p3.8xlarge": {"vcpus": 32, "memory_gb": 244, "hourly_rate": 12.24},
    "g4dn.xlarge": {"vcpus": 4, "memory_gb": 16, "hourly_rate": 0.526},
    "g4dn.4xlarge": {"vcpus": 16, "memory_gb": 64, "hourly_rate": 1.204},
}

GPU_OPTIONS = {
    "None": {"hourly_rate": 0.0, "memory_gb": 0},
    "NVIDIA T4 (16GB)": {"hourly_rate": 0.526, "memory_gb": 16},
    "NVIDIA V100 (16GB)": {"hourly_rate": 2.14, "memory_gb": 16},
    "NVIDIA V100 (32GB)": {"hourly_rate": 3.06, "memory_gb": 32},
    "NVIDIA A100 (40GB)": {"hourly_rate": 4.10, "memory_gb": 40},
    "NVIDIA A100 (80GB)": {"hourly_rate": 5.80, "memory_gb": 80},
}

CLOUD_PROVIDERS = {
    "AWS": {"compute_multiplier": 1.0, "storage_per_gb": 0.023, "egress_per_gb": 0.09},
    "GCP": {"compute_multiplier": 0.95, "storage_per_gb": 0.020, "egress_per_gb": 0.085},
    "Azure": {"compute_multiplier": 1.02, "storage_per_gb": 0.021, "egress_per_gb": 0.087},
}


def calculate_costs(
    instance_type: str,
    gpu_type: str,
    num_instances: int,
    hours_per_day: float,
    days_per_month: int,
    storage_gb: int,
    egress_gb: int,
) -> dict[str, float]:
    """Calculate infrastructure costs based on configuration.

    Args:
        instance_type: EC2-style instance type name.
        gpu_type: GPU accelerator type name.
        num_instances: Number of compute instances.
        hours_per_day: Hours of usage per day.
        days_per_month: Working days per month.
        storage_gb: Storage capacity in GB.
        egress_gb: Data egress in GB per month.

    Returns:
        Dictionary with cost breakdown (compute, gpu, storage, egress, total).
    """
    instance = INSTANCE_TYPES[instance_type]
    gpu = GPU_OPTIONS[gpu_type]

    total_hours = hours_per_day * days_per_month
    compute_cost = instance["hourly_rate"] * total_hours * num_instances
    gpu_cost = gpu["hourly_rate"] * total_hours * num_instances
    storage_cost = storage_gb * CLOUD_PROVIDERS["AWS"]["storage_per_gb"]
    egress_cost = egress_gb * CLOUD_PROVIDERS["AWS"]["egress_per_gb"]
    total = compute_cost + gpu_cost + storage_cost + egress_cost

    return {
        "compute": round(compute_cost, 2),
        "gpu": round(gpu_cost, 2),
        "storage": round(storage_cost, 2),
        "egress": round(egress_cost, 2),
        "total": round(total, 2),
    }


def generate_multi_cloud_comparison(
    instance_type: str,
    gpu_type: str,
    num_instances: int,
    hours_per_day: float,
    days_per_month: int,
    storage_gb: int,
    egress_gb: int,
) -> pd.DataFrame:
    """Generate cost comparison across cloud providers.

    Args:
        instance_type: EC2-style instance type name.
        gpu_type: GPU accelerator type name.
        num_instances: Number of compute instances.
        hours_per_day: Hours of usage per day.
        days_per_month: Working days per month.
        storage_gb: Storage capacity in GB.
        egress_gb: Data egress in GB per month.

    Returns:
        DataFrame with provider, compute, gpu, storage, egress, total columns.
    """
    instance = INSTANCE_TYPES[instance_type]
    gpu = GPU_OPTIONS[gpu_type]
    total_hours = hours_per_day * days_per_month
    records = []

    for provider, rates in CLOUD_PROVIDERS.items():
        compute = instance["hourly_rate"] * rates["compute_multiplier"] * total_hours * num_instances
        gpu_cost = gpu["hourly_rate"] * rates["compute_multiplier"] * total_hours * num_instances
        storage = storage_gb * rates["storage_per_gb"]
        egress = egress_gb * rates["egress_per_gb"]
        total = compute + gpu_cost + storage + egress

        records.append(
            {
                "Provider": provider,
                "Compute": round(compute, 2),
                "GPU": round(gpu_cost, 2),
                "Storage": round(storage, 2),
                "Egress": round(egress, 2),
                "Total Monthly": round(total, 2),
            }
        )

    return pd.DataFrame(records)


def generate_monthly_projection(
    base_monthly_cost: float,
    months: int = 12,
    growth_rate: float = 0.05,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate monthly cost projection with growth.

    Args:
        base_monthly_cost: Starting monthly cost.
        months: Number of months to project.
        growth_rate: Monthly growth rate as a fraction.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with month, projected_cost, optimized_cost, and savings columns.
    """
    rng = np.random.default_rng(seed)
    records = []

    for month in range(1, months + 1):
        noise = rng.normal(0, 0.03)
        projected = base_monthly_cost * (1 + growth_rate) ** (month - 1) * (1 + noise)
        optimized = projected * (0.72 + rng.normal(0, 0.02))
        savings = projected - optimized

        records.append(
            {
                "month": month,
                "month_label": f"Month {month}",
                "projected_cost": round(projected, 2),
                "optimized_cost": round(optimized, 2),
                "savings": round(savings, 2),
            }
        )

    return pd.DataFrame(records)


def render_cost_calculator() -> dict[str, float]:
    """Render the cost calculator sidebar and return computed costs.

    Returns:
        Dictionary with cost breakdown values.
    """
    st.header("Infrastructure Cost Calculator")

    col1, col2 = st.columns(2)

    with col1:
        instance_type = st.selectbox(
            "Instance Type",
            options=list(INSTANCE_TYPES.keys()),
            index=2,
        )
        instance_info = INSTANCE_TYPES[instance_type]
        st.caption(
            f"{instance_info['vcpus']} vCPUs, {instance_info['memory_gb']} GB RAM, ${instance_info['hourly_rate']}/hr"
        )

        gpu_type = st.selectbox(
            "GPU Accelerator",
            options=list(GPU_OPTIONS.keys()),
            index=0,
        )

        num_instances = st.slider(
            "Number of Instances",
            min_value=1,
            max_value=32,
            value=4,
        )

    with col2:
        hours_per_day = st.slider(
            "Hours per Day",
            min_value=1.0,
            max_value=24.0,
            value=8.0,
            step=0.5,
        )

        days_per_month = st.slider(
            "Working Days per Month",
            min_value=1,
            max_value=31,
            value=22,
        )

        storage_gb = st.slider(
            "Storage (GB)",
            min_value=10,
            max_value=10000,
            value=500,
            step=10,
        )

        egress_gb = st.slider(
            "Data Egress (GB/month)",
            min_value=0,
            max_value=5000,
            value=100,
            step=10,
        )

    costs = calculate_costs(
        instance_type,
        gpu_type,
        num_instances,
        hours_per_day,
        days_per_month,
        storage_gb,
        egress_gb,
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Monthly Total", f"${costs['total']:,.2f}")
    col2.metric("Compute Cost", f"${costs['compute']:,.2f}")
    col3.metric("GPU Cost", f"${costs['gpu']:,.2f}")
    col4.metric("Storage + Egress", f"${costs['storage'] + costs['egress']:,.2f}")

    return {
        "instance_type": instance_type,
        "gpu_type": gpu_type,
        "num_instances": num_instances,
        "hours_per_day": hours_per_day,
        "days_per_month": days_per_month,
        "storage_gb": storage_gb,
        "egress_gb": egress_gb,
        **costs,
    }


def render_cost_breakdown(costs: dict[str, float]) -> None:
    """Render cost breakdown pie chart."""
    st.header("Cost Breakdown")

    breakdown_data = {
        "Compute": costs["compute"],
        "GPU": costs["gpu"],
        "Storage": costs["storage"],
        "Egress": costs["egress"],
    }
    breakdown_data = {k: v for k, v in breakdown_data.items() if v > 0}

    if not breakdown_data:
        st.info("No costs to display. Adjust the calculator settings.")
        return

    fig = go.Figure(
        data=[
            go.Pie(
                labels=list(breakdown_data.keys()),
                values=list(breakdown_data.values()),
                hole=0.4,
                textinfo="label+percent+value",
                texttemplate="%{label}<br>%{percent}<br>$%{value:,.2f}",
                marker={"colors": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]},
            )
        ]
    )
    fig.update_layout(
        title="Monthly Cost Distribution",
        height=450,
    )
    st.plotly_chart(fig, use_container_width=True)


def render_monthly_projection(base_cost: float) -> None:
    """Render monthly cost projection chart."""
    st.header("Monthly Cost Projection")

    col1, col2 = st.columns(2)
    with col1:
        months = st.slider("Projection Period (months)", 3, 24, 12)
    with col2:
        growth_rate = st.slider("Monthly Growth Rate (%)", 0.0, 20.0, 5.0, 0.5) / 100

    projection_df = generate_monthly_projection(base_cost, months, growth_rate)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=projection_df["month_label"],
            y=projection_df["projected_cost"],
            mode="lines+markers",
            name="Projected Cost",
            line={"color": "#d62728"},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=projection_df["month_label"],
            y=projection_df["optimized_cost"],
            mode="lines+markers",
            name="Optimized Cost",
            line={"color": "#2ca02c"},
        )
    )
    fig.add_trace(
        go.Bar(
            x=projection_df["month_label"],
            y=projection_df["savings"],
            name="Potential Savings",
            marker={"color": "#1f77b4", "opacity": 0.3},
        )
    )
    fig.update_layout(
        title="Cost Projection with Optimization Potential",
        xaxis_title="Month",
        yaxis_title="Cost ($)",
        height=450,
        barmode="overlay",
    )
    st.plotly_chart(fig, use_container_width=True)

    total_projected = projection_df["projected_cost"].sum()
    total_optimized = projection_df["optimized_cost"].sum()
    total_savings = projection_df["savings"].sum()

    col1, col2, col3 = st.columns(3)
    col1.metric(
        f"Total Projected ({months}mo)",
        f"${total_projected:,.2f}",
    )
    col2.metric(
        f"Total Optimized ({months}mo)",
        f"${total_optimized:,.2f}",
    )
    col3.metric(
        "Potential Savings",
        f"${total_savings:,.2f}",
        f"{total_savings / max(total_projected, 1) * 100:.1f}% reduction",
    )


def render_cloud_comparison(config: dict) -> None:
    """Render multi-cloud comparison table and chart."""
    st.header("Multi-Cloud Comparison")

    comparison_df = generate_multi_cloud_comparison(
        config["instance_type"],
        config["gpu_type"],
        config["num_instances"],
        config["hours_per_day"],
        config["days_per_month"],
        config["storage_gb"],
        config["egress_gb"],
    )

    st.dataframe(
        comparison_df.style.format(
            {
                "Compute": "${:,.2f}",
                "GPU": "${:,.2f}",
                "Storage": "${:,.2f}",
                "Egress": "${:,.2f}",
                "Total Monthly": "${:,.2f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    fig = px.bar(
        comparison_df,
        x="Provider",
        y=["Compute", "GPU", "Storage", "Egress"],
        title="Cost Comparison by Cloud Provider",
        labels={"value": "Monthly Cost ($)", "variable": "Cost Category"},
        barmode="stack",
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

    cheapest = comparison_df.loc[comparison_df["Total Monthly"].idxmin()]
    most_expensive = comparison_df.loc[comparison_df["Total Monthly"].idxmax()]
    savings = most_expensive["Total Monthly"] - cheapest["Total Monthly"]

    col1, col2, col3 = st.columns(3)
    col1.metric("Most Affordable", cheapest["Provider"])
    col2.metric(
        "Monthly Savings vs Most Expensive",
        f"${savings:,.2f}",
    )
    col3.metric(
        "Savings Percentage",
        f"{savings / max(most_expensive['Total Monthly'], 1) * 100:.1f}%",
    )


def main() -> None:
    """Run the Streamlit dashboard application."""
    st.set_page_config(
        page_title="Infrastructure Cost Dashboard",
        page_icon="💰",
        layout="wide",
    )

    st.title("Infrastructure Cost Dashboard")
    st.markdown(
        "Calculate and compare cloud infrastructure costs for data science "
        "workloads with interactive configuration and multi-cloud analysis."
    )

    config = render_cost_calculator()
    st.divider()
    render_cost_breakdown(config)
    st.divider()
    render_monthly_projection(config["total"])
    st.divider()
    render_cloud_comparison(config)


if __name__ == "__main__":
    main()
