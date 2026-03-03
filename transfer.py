import os, re, json, gc, random
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from sklearn.metrics import roc_curve, roc_auc_score, classification_report, accuracy_score, confusion_matrix
from tensorflow.keras.optimizers import Adam
from keras import backend as K
from keras.models import Model, Sequential
from keras.layers import (Input, Dense, Dropout, Flatten, BatchNormalization,
                          Conv2D, MultiHeadAttention, concatenate, Multiply,
                          Lambda, Add)
from keras.regularizers import l2
import tensorflow as tf


def _build_and_load_model(mode, X1_example, X2_example, weights_path, lr=1e-5):
    model = multi_modal_model(mode, X1_example, X2_example)
    model.compile(optimizer=Adam(learning_rate=lr, decay=1e-8),
                  loss='sparse_categorical_crossentropy',
                  metrics=['sparse_categorical_accuracy'])
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weight file not found: {weights_path}")
    model.load_weights(weights_path)
    return model

def reset_random_seeds(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

# =============================
# New Data Loading and Alignment
# =============================
def extract_id_from_name(name):
    base = os.path.splitext(os.path.basename(name))[0]
    return base.lstrip("0")  # Remove leading zeros

def load_new_dataset(
    mod1_dir: str,
    mod2_dir: str,
    labels_csv: str,
    id_col: str = "ID",
    label_col: str = "Label",
    sort_by_id: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Reads .npy files from two modality directories, aligns them with the CSV labels by ID, 
    and returns X1, X2, y, id_list.
    Skips samples if they are missing in either modality.
    """
    df = pd.read_csv(labels_csv)
    # Uniform IDs to string format
    ids = df[id_col].astype(str).tolist()
    labels = df[label_col].values

    # Create index: ID -> File Path
    def index_dir(d: str) -> Dict[str, str]:
        mp = {}
        for fn in os.listdir(d):
            if fn.lower().endswith(".npy"):
                fid = extract_id_from_name(fn)
                mp[str(fid)] = os.path.join(d, fn)
        return mp

    idx1 = index_dir(mod1_dir)
    idx2 = index_dir(mod2_dir)

    X1_list, X2_list, y_list, keep_ids = [], [], [], []
    miss_1, miss_2 = 0, 0

    for sid, y in zip(ids, labels):
        if str(sid) not in idx1:
            miss_1 += 1
            continue
        if str(sid) not in idx2:
            miss_2 += 1
            continue
        
        x1 = np.load(idx1[str(sid)])
        x2 = np.load(idx2[str(sid)])
        
        # Force float32
        X1_list.append(x1.astype(np.float32, copy=False))
        X2_list.append(x2.astype(np.float32, copy=False))
        y_list.append(int(y))
        keep_ids.append(str(sid))

    if miss_1 or miss_2:
        print(f"[WARNING] Samples missing in Modality 1: {miss_1}, Modality 2: {miss_2} (Skipped)")

    X1 = np.stack(X1_list, axis=0)
    X2 = np.stack(X2_list, axis=0)
    y = np.array(y_list, dtype=np.int64)

    # Optional: Sort by ID for stable output
    if sort_by_id:
        order = np.argsort(np.array(keep_ids, dtype=object))
        X1, X2, y = X1[order], X2[order], y[order]
        keep_ids = [keep_ids[i] for i in order]

    print(f"[INFO] Aligned new data: N={len(keep_ids)}; Mod1 shape={X1.shape}; Mod2 shape={X2.shape}")
    return X1, X2, y, keep_ids

# =============================
# Evaluation: Test pretrained model
# =============================
def _binary_mean_roc_across_folds(fold_rocs):
    grid = np.linspace(0.0, 1.0, 101)
    tprs = []
    for fr in fold_rocs:
        fpr = np.asarray(fr["fpr"])
        tpr = np.asarray(fr["tpr"])
        tprs.append(np.interp(grid, fpr, tpr))
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[0] = 0.0
    mean_tpr[-1] = 1.0
    from numpy import trapz
    mean_auc = trapz(mean_tpr, grid)
    return {"fpr": grid.tolist(), "tpr": mean_tpr.tolist(), "auc": float(mean_auc)}

def evaluate_pretrained_on_new_data(
    mode: str,
    mod1_dir: str,
    mod2_dir: str,
    labels_csv: str,
    id_col: str = "ID",
    label_col: str = "Label",
    weights_pattern: str = "model_weights_fold_best{}.h5",
    folds: List[int] = [1,2,3,4,5],
    batch_size: int = 32,
    learning_rate: float = 1e-5,
    seed: int = 118,
    out_dir: str = "./eval_newdata_outputs"
):
    os.makedirs(out_dir, exist_ok=True)
    reset_random_seeds(seed)

    # Load new dataset
    X1, X2, y_true, id_list = load_new_dataset(
        mod1_dir, mod2_dir, labels_csv, id_col=id_col, label_col=label_col
    )

    accs, aucs = [], []
    fold_json_records = []
    preds_collection = []

    for fold in folds:
        weights_path = weights_pattern.format(fold)
        print(f"\n==== Evaluating Fold {fold} | Weights: {weights_path} ====")

        model = _build_and_load_model(mode, X1, X2, weights_path, lr=learning_rate)

        # Predict
        y_prob = model.predict([X1, X2], batch_size=batch_size, verbose=0)
        if y_prob.ndim == 1:
            y_prob = np.stack([1 - y_prob, y_prob], axis=1)
        y_pred = np.argmax(y_prob, axis=1)

        # Metrics
        acc = accuracy_score(y_true, y_pred)
        accs.append(float(acc))
        cr = classification_report(y_true, y_pred, digits=4)
        cm = confusion_matrix(y_true, y_pred)

        # ROC 
        classes_sorted = np.unique(y_true)
        if len(classes_sorted) == 2:
            pos_label = classes_sorted.max()
            y_bin = (y_true == pos_label).astype(int)
            fpr, tpr, thr = roc_curve(y_bin, y_prob[:,1])
            auc_val = roc_auc_score(y_bin, y_prob[:,1])
            aucs.append(float(auc_val))
            fold_roc = {
                "fold": fold,
                "test_acc": float(acc),
                "binary_roc": {
                    "fpr": fpr.tolist(),
                    "tpr": tpr.tolist(),
                    "thresholds": thr.tolist(),
                    "auc": float(auc_val)
                }
            }
        else:
            raise ValueError("Current evaluation script assumes binary classification (labels 0/1). Please check the label column.")

        # Save predictions for each fold
        df_pred = pd.DataFrame({
            "id": id_list,
            "y_true": y_true,
            "p0": y_prob[:,0],
            "p1": y_prob[:,1],
            "y_pred": y_pred
        })
        pred_csv = os.path.join(out_dir, f"predictions_fold{fold}.csv")
        df_pred.to_csv(pred_csv, index=False, encoding="utf-8-sig")
        
        print(f"[SAVE] Fold predictions: {pred_csv}")
        print(f"[RESULT] Fold {fold}: ACC={acc:.4f}, AUC={aucs[-1]:.4f}")
        print(cr)
        print("Confusion Matrix:\n", cm)

        fold_json_records.append(fold_roc)
        preds_collection.append(df_pred)

        # Clean session
        K.clear_session()
        del model, y_prob, y_pred
        gc.collect()

    # Summary
    summary_df = pd.DataFrame({
        "fold": folds,
        "ACC": accs,
        "AUC": aucs
    })
    summary_df.loc["mean"] = ["mean", np.mean(accs), np.mean(aucs)]
    summary_df.loc["std"]  = ["std",  np.std(accs),  np.std(aucs)]
    sum_csv = os.path.join(out_dir, "summary_metrics.csv")
    summary_df.to_csv(sum_csv, index=True, encoding="utf-8-sig")
    print(f"\n[SAVE] Summary metrics: {sum_csv}")

    # Mean ROC
    mean_bin = _binary_mean_roc_across_folds([f["binary_roc"] for f in fold_json_records])
    json_obj = {
        "mode": mode,
        "seed": int(seed),
        "folds": fold_json_records,
        "task_type": "binary",
        "mean_binary_roc": mean_bin
    }
    json_path = os.path.join(out_dir, "roc_values_newdata.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_obj, f, ensure_ascii=False, indent=2)
    print(f"[SAVE] ROC JSON: {json_path}")

    return sum_csv, json_path

# ========== Usage Example ==========
if __name__ == "__main__":
    
    MOD1_DIR   = smri_npy_pro"
    MOD2_DIR   = "fmri_npy_pro"
    LABELS_CSV = "tranfer/MDD_HC____.csv"

    evaluate_pretrained_on_new_data(
        mode='MM_SA_BA',
        mod1_dir=MOD1_DIR,
        mod2_dir=MOD2_DIR,
        labels_csv=LABELS_CSV,
        id_col="list",
        label_col="label",
        weights_pattern="model_weights_fold_best{}.h5",
        folds=[1,2,3,4,5],
        batch_size=32,
        learning_rate=1e-5,
        seed=118,
        out_dir="./eval_newdata_outputs"
    )