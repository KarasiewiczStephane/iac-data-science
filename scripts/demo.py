"""End-to-end demo script for iac-data-science."""

from pathlib import Path

from src.data.pipeline import DataPipeline
from src.models.evaluation import ModelEvaluator
from src.models.implementations import get_model
from src.models.reporting import generate_report
from src.models.trainer import ModelTrainer


def run_demo() -> None:
    """Run the full pipeline: load, preprocess, train, evaluate, save."""
    print("=== iac-data-science demo ===\n")

    # 1. Load data
    pipeline = DataPipeline()
    train_df = pipeline.run(Path("data/sample/train.csv"))
    test_df = pipeline.run(Path("data/sample/test.csv"))
    print(f"Loaded {len(train_df)} training rows, {len(test_df)} test rows")

    # 2. Preprocess (missing values + scaling, no outlier removal to keep alignment)
    from sklearn.pipeline import Pipeline

    from src.data.preprocessors import FeatureScaler, MissingValueHandler

    target_col = "target"
    X_train_raw = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]
    X_test_raw = test_df.drop(columns=[target_col])
    y_test = test_df[target_col]

    feat_pipeline = Pipeline(
        [
            ("missing", MissingValueHandler(strategy="mean")),
            ("scaler", FeatureScaler()),
        ]
    )
    X_train = feat_pipeline.fit_transform(X_train_raw)
    X_test = feat_pipeline.transform(X_test_raw)
    print(f"Preprocessing complete: {X_train.shape[1]} features")

    # 4. Train model
    model = get_model("random_forest", n_estimators=100, random_state=42)
    trainer = ModelTrainer(model)
    trained_model = trainer.train(X_train, y_train)
    print("Model trained")

    # 5. Evaluate
    evaluator = ModelEvaluator("classification")
    predictions = trained_model.predict(X_test)
    metrics = evaluator.evaluate(y_test, predictions)
    print(f"\nResults:\n  accuracy  = {metrics.accuracy:.4f}")
    print(f"  precision = {metrics.precision:.4f}")
    print(f"  recall    = {metrics.recall:.4f}")
    print(f"  f1        = {metrics.f1:.4f}")

    # 6. Save model and report
    model_path = Path("models/demo_model.joblib")
    trained_model.save(model_path)
    print(f"\nModel saved to {model_path}")

    report_path = Path("reports/demo_report.json")
    generate_report(metrics, report_path)
    print(f"Report saved to {report_path}")

    # 7. Cross-validation
    cv_results = evaluator.cross_validate(trained_model, X_train, y_train, cv=5)
    print(f"\nCross-validation: mean={cv_results['mean']:.4f} (+/- {cv_results['std']:.4f})")
    print("\n=== Demo complete ===")


if __name__ == "__main__":
    run_demo()
