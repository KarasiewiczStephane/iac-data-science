"""Tests for the infrastructure cost dashboard data generators."""

import pandas as pd

from src.dashboard.app import (
    CLOUD_PROVIDERS,
    GPU_OPTIONS,
    INSTANCE_TYPES,
    calculate_costs,
    generate_monthly_projection,
    generate_multi_cloud_comparison,
)


class TestCalculateCosts:
    """Tests for calculate_costs."""

    def test_returns_dict(self):
        """Returns a dictionary with expected keys."""
        costs = calculate_costs("t3.medium", "None", 1, 8.0, 22, 100, 50)
        expected_keys = {"compute", "gpu", "storage", "egress", "total"}
        assert expected_keys == set(costs.keys())

    def test_total_is_sum(self):
        """Total equals sum of components."""
        costs = calculate_costs("m5.2xlarge", "NVIDIA T4 (16GB)", 2, 8.0, 22, 500, 100)
        expected_total = costs["compute"] + costs["gpu"] + costs["storage"] + costs["egress"]
        assert abs(costs["total"] - expected_total) < 0.01

    def test_no_gpu_zero_cost(self):
        """GPU cost is zero when 'None' GPU selected."""
        costs = calculate_costs("t3.medium", "None", 1, 8.0, 22, 100, 50)
        assert costs["gpu"] == 0.0

    def test_gpu_adds_cost(self):
        """Selecting a GPU increases total cost."""
        costs_no_gpu = calculate_costs("m5.2xlarge", "None", 1, 8.0, 22, 100, 50)
        costs_with_gpu = calculate_costs("m5.2xlarge", "NVIDIA A100 (80GB)", 1, 8.0, 22, 100, 50)
        assert costs_with_gpu["total"] > costs_no_gpu["total"]

    def test_more_instances_cost_more(self):
        """More instances increase compute and GPU costs."""
        costs_1 = calculate_costs("m5.2xlarge", "NVIDIA T4 (16GB)", 1, 8.0, 22, 100, 50)
        costs_4 = calculate_costs("m5.2xlarge", "NVIDIA T4 (16GB)", 4, 8.0, 22, 100, 50)
        assert costs_4["compute"] > costs_1["compute"]
        assert costs_4["gpu"] > costs_1["gpu"]

    def test_all_values_non_negative(self):
        """All cost values are non-negative."""
        costs = calculate_costs("p3.8xlarge", "NVIDIA A100 (80GB)", 8, 24.0, 31, 10000, 5000)
        for value in costs.values():
            assert value >= 0

    def test_zero_hours_zero_compute(self):
        """Zero hours results in zero compute and GPU cost."""
        costs = calculate_costs("m5.2xlarge", "NVIDIA T4 (16GB)", 4, 0.0, 22, 100, 50)
        assert costs["compute"] == 0.0
        assert costs["gpu"] == 0.0


class TestGenerateMultiCloudComparison:
    """Tests for generate_multi_cloud_comparison."""

    def test_returns_dataframe(self):
        """Returns a DataFrame with expected columns."""
        df = generate_multi_cloud_comparison("t3.medium", "None", 1, 8.0, 22, 100, 50)
        assert isinstance(df, pd.DataFrame)
        assert "Provider" in df.columns
        assert "Total Monthly" in df.columns

    def test_all_providers_present(self):
        """All cloud providers are included."""
        df = generate_multi_cloud_comparison("t3.medium", "None", 1, 8.0, 22, 100, 50)
        assert set(df["Provider"]) == set(CLOUD_PROVIDERS.keys())

    def test_row_count(self):
        """One row per cloud provider."""
        df = generate_multi_cloud_comparison("m5.2xlarge", "NVIDIA T4 (16GB)", 2, 8.0, 22, 500, 100)
        assert len(df) == len(CLOUD_PROVIDERS)

    def test_all_totals_positive(self):
        """All total costs are positive."""
        df = generate_multi_cloud_comparison("m5.2xlarge", "NVIDIA T4 (16GB)", 2, 8.0, 22, 500, 100)
        assert (df["Total Monthly"] > 0).all()


class TestGenerateMonthlyProjection:
    """Tests for generate_monthly_projection."""

    def test_returns_dataframe(self):
        """Returns a DataFrame with expected columns."""
        df = generate_monthly_projection(1000.0)
        assert isinstance(df, pd.DataFrame)
        expected_cols = {
            "month",
            "month_label",
            "projected_cost",
            "optimized_cost",
            "savings",
        }
        assert expected_cols == set(df.columns)

    def test_month_count(self):
        """Generates correct number of months."""
        df = generate_monthly_projection(1000.0, months=6)
        assert len(df) == 6

    def test_default_12_months(self):
        """Default generates 12 months."""
        df = generate_monthly_projection(500.0)
        assert len(df) == 12

    def test_positive_costs(self):
        """All costs are positive."""
        df = generate_monthly_projection(1000.0)
        assert (df["projected_cost"] > 0).all()
        assert (df["optimized_cost"] > 0).all()

    def test_optimized_less_than_projected(self):
        """Optimized cost is generally less than projected."""
        df = generate_monthly_projection(1000.0, seed=42)
        assert (df["optimized_cost"] < df["projected_cost"]).all()

    def test_reproducible_with_seed(self):
        """Same seed produces identical output."""
        df1 = generate_monthly_projection(1000.0, seed=42)
        df2 = generate_monthly_projection(1000.0, seed=42)
        pd.testing.assert_frame_equal(df1, df2)

    def test_growth_rate_effect(self):
        """Higher growth rate leads to higher later costs."""
        df_low = generate_monthly_projection(1000.0, growth_rate=0.01, seed=42)
        df_high = generate_monthly_projection(1000.0, growth_rate=0.20, seed=42)
        assert df_high["projected_cost"].iloc[-1] > df_low["projected_cost"].iloc[-1]


class TestConstants:
    """Tests for module-level constants."""

    def test_instance_types_not_empty(self):
        """INSTANCE_TYPES contains entries."""
        assert len(INSTANCE_TYPES) > 0

    def test_instance_types_have_required_keys(self):
        """Each instance type has vcpus, memory_gb, hourly_rate."""
        for name, spec in INSTANCE_TYPES.items():
            assert "vcpus" in spec, f"{name} missing vcpus"
            assert "memory_gb" in spec, f"{name} missing memory_gb"
            assert "hourly_rate" in spec, f"{name} missing hourly_rate"
            assert spec["hourly_rate"] > 0

    def test_gpu_options_have_required_keys(self):
        """Each GPU option has hourly_rate and memory_gb."""
        for name, spec in GPU_OPTIONS.items():
            assert "hourly_rate" in spec, f"{name} missing hourly_rate"
            assert "memory_gb" in spec, f"{name} missing memory_gb"

    def test_cloud_providers_have_required_keys(self):
        """Each cloud provider has required pricing keys."""
        for name, rates in CLOUD_PROVIDERS.items():
            assert "compute_multiplier" in rates, f"{name} missing compute_multiplier"
            assert "storage_per_gb" in rates, f"{name} missing storage_per_gb"
            assert "egress_per_gb" in rates, f"{name} missing egress_per_gb"
