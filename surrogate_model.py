"""Train and evaluate the deep surrogate model."""
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from .config import MODELS, RESULTS, SEED, INPUT_NAMES, OUTPUT_NAMES
from .preprocessing import load_data, split_scale

def build_tensorflow_model(n_inputs, n_outputs):
    import tensorflow as tf
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(n_inputs,)),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(n_outputs)
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="mse",
        metrics=["mae"]
    )
    return model

def train():
    MODELS.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    df = load_data()
    data = split_scale(df)
    Xtr, Xv, Xte, ytr, yv, yte, xs, ys = data

    history_dict = {}
    backend = "tensorflow"
    try:
        import tensorflow as tf
        tf.random.set_seed(SEED)
        model = build_tensorflow_model(Xtr.shape[1], ytr.shape[1])
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=25, restore_best_weights=True
            )
        ]
        history = model.fit(
            Xtr, ytr, validation_data=(Xv, yv),
            epochs=300, batch_size=64, verbose=0, callbacks=callbacks
        )
        model.save(MODELS / "surrogate_model.keras")
        history_dict = history.history
        pred_scaled = model.predict(Xte, verbose=0)
    except Exception as exc:
        backend = "sklearn"
        print("TensorFlow unavailable; using scikit-learn MLP fallback:", exc)
        model = MLPRegressor(
            hidden_layer_sizes=(64, 128, 64, 32),
            activation="relu", solver="adam", learning_rate_init=1e-3,
            max_iter=1000, early_stopping=True, random_state=SEED
        )
        model.fit(Xtr, ytr)
        joblib.dump(model, MODELS / "surrogate_model.joblib")
        pred_scaled = model.predict(Xte)

    pred = ys.inverse_transform(pred_scaled)
    actual = ys.inverse_transform(yte)

    metrics = []
    for i, name in enumerate(OUTPUT_NAMES):
        metrics.append({
            "output": name,
            "MAE": mean_absolute_error(actual[:, i], pred[:, i]),
            "RMSE": np.sqrt(mean_squared_error(actual[:, i], pred[:, i])),
            "R2": r2_score(actual[:, i], pred[:, i]),
        })
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(RESULTS / "surrogate_metrics.csv", index=False)
    joblib.dump({"x_scaler": xs, "y_scaler": ys, "backend": backend},
                MODELS / "scalers.joblib")

    # Save test predictions for plotting and validation.
    out = pd.DataFrame(actual, columns=[f"{x}_actual" for x in OUTPUT_NAMES])
    for i, name in enumerate(OUTPUT_NAMES):
        out[f"{name}_pred"] = pred[:, i]
    out.to_csv(RESULTS / "test_predictions.csv", index=False)

    if history_dict:
        pd.DataFrame(history_dict).to_csv(RESULTS / "training_history.csv", index=False)

    print(metrics_df)
    return model, metrics_df

if __name__ == "__main__":
    train()
