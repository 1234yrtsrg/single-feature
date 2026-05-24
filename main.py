import os
import gc
import sys
import argparse
import subprocess
import warnings

import pandas as pd
from sklearn.exceptions import ConvergenceWarning

from config import Config
from pipeline.orchestrator import NestedCVOrchestrator
from pipeline.reporter import ResultReporter
from utils.logger import Logger

os.environ["PYTHONWARNINGS"] = "ignore"

METAL_COLUMN_MAP = {
    "Sn": "118Sn (KED)",
    "Bi": "209Bi (KED)",
}


def suppress_warnings():
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    warnings.filterwarnings("ignore", message="^Objective did not converge.*")
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        message="Using a target size .*different to the input size.*"
    )
    warnings.filterwarnings("ignore", category=RuntimeWarning, message="overflow encountered in cast")
    warnings.filterwarnings("ignore", message=".*A worker stopped while some jobs were given to the executor.*")


def parse_int_csv(text):
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def split_contiguous(values, n_chunks):
    if n_chunks <= 0:
        raise ValueError("n_chunks must be positive.")

    chunk_size = (len(values) + n_chunks - 1) // n_chunks
    return [values[i:i + chunk_size] for i in range(0, len(values), chunk_size)]


def apply_runtime_config(args):
    Config.N_JOBS = int(args.n_jobs)


def build_log_name(args):
    if args.worker_max_features:
        return f"worker_{args.metal.lower()}"
    if args.sweep_shared_max_features:
        return f"sweep_controller_{args.metal.lower()}"
    return f"single_{args.metal.lower()}_k{int(args.shared_max_features)}"


def build_single_output_path(current_dir, metal, shared_max_features):
    base_name = f"nestedcv_oof_predictions_{metal}.xlsx"
    if shared_max_features != Config.SHARED_MAX_FEATURES:
        base_name = f"nestedcv_oof_predictions_{metal}_k{shared_max_features}.xlsx"
    return os.path.abspath(os.path.join(current_dir, base_name))


def build_sweep_output_path(current_dir, metal):
    return os.path.abspath(os.path.join(current_dir, f"shared_max_features_sweep_{metal}.xlsx"))


def sanitize_path_part(text):
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(text))


def build_sweep_checkpoint_dir(current_dir, metal, summary_model):
    safe_model = sanitize_path_part(summary_model)
    return os.path.abspath(os.path.join(current_dir, f"sweep_checkpoints_{metal}_{safe_model}"))


def read_sweep_summary_xlsx(path, target_metal=None, summary_model=None):
    try:
        df = pd.read_excel(path)
    except Exception as exc:
        Logger.log(f"[WARN] Skip unreadable sweep checkpoint: {path} | {exc}")
        return pd.DataFrame()

    if "SHARED_MAX_FEATURES" not in df.columns:
        Logger.log(f"[WARN] Skip sweep checkpoint without SHARED_MAX_FEATURES: {path}")
        return pd.DataFrame()

    if target_metal is not None and "Metal" in df.columns:
        df = df[df["Metal"] == target_metal]
    if summary_model is not None and "Model" in df.columns:
        df = df[df["Model"] == summary_model]
    return df


def merge_sweep_summaries(frames):
    valid_frames = [df for df in frames if df is not None and not df.empty]
    if not valid_frames:
        return pd.DataFrame()

    merged = pd.concat(valid_frames, ignore_index=True)
    merged = merged.dropna(subset=["SHARED_MAX_FEATURES"]).copy()
    merged["SHARED_MAX_FEATURES"] = merged["SHARED_MAX_FEATURES"].astype(int)
    return (
        merged
        .drop_duplicates(subset=["SHARED_MAX_FEATURES"], keep="last")
        .sort_values("SHARED_MAX_FEATURES")
        .reset_index(drop=True)
    )


def load_existing_sweep_results(sweep_output, checkpoint_dir, target_metal, summary_model):
    frames = []
    if os.path.exists(sweep_output):
        frames.append(read_sweep_summary_xlsx(sweep_output, target_metal, summary_model))

    if os.path.isdir(checkpoint_dir):
        for name in sorted(os.listdir(checkpoint_dir)):
            if name.lower().endswith(".xlsx"):
                frames.append(
                    read_sweep_summary_xlsx(
                        os.path.join(checkpoint_dir, name),
                        target_metal,
                        summary_model
                    )
                )

    return merge_sweep_summaries(frames)


def collect_target_metrics(result, target_metal, summary_model, shared_max_features):
    summary_df = result.get("summary")
    row = None
    if summary_df is not None:
        subset = summary_df[
            (summary_df["Target"] == target_metal) &
            (summary_df["Model"] == summary_model)
        ]
        if not subset.empty:
            rec = subset.iloc[0]
            row = {
                "Target": target_metal,
                "Model": summary_model,
                "Train R2": rec.get("Train R2"),
                "Train RMSE": rec.get("Train RMSE"),
                "Train RPD": rec.get("Train RPD"),
                "Test R2": rec.get("OOF R2"),
                "Test RMSE": rec.get("OOF RMSE"),
                "Test RPD": rec.get("OOF RPD"),
            }

    if row is None:
        row = ResultReporter.extract_train_test_metrics(
            result["fold_metrics"],
            target=target_metal,
            model=summary_model
        )

    return {
        "SHARED_MAX_FEATURES": int(shared_max_features),
        "Metal": target_metal,
        "Model": summary_model,
        "Train R2": row.get("Train R2"),
        "Train RMSE": row.get("Train RMSE"),
        "Train RPD": row.get("Train RPD"),
        "Test R2": row.get("Test R2"),
        "Test RMSE": row.get("Test RMSE"),
        "Test RPD": row.get("Test RPD"),
    }


def release_runtime_memory():
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def run_single_job(current_dir, data_path, args, shared_max_features=None, export_outputs=None):
    shared_max_features = int(args.shared_max_features if shared_max_features is None else shared_max_features)
    export_outputs = (not args.no_oof_export) if export_outputs is None else bool(export_outputs)
    target_metal = METAL_COLUMN_MAP[args.metal]
    output_path = build_single_output_path(current_dir, args.metal, shared_max_features) if export_outputs else None

    Logger.console(
        f"[TRAIN] START metal={args.metal} shared_max_features={shared_max_features}"
    )
    Logger.log(f"[*] Data Source: {data_path}")
    if output_path:
        Logger.log(f"[*] Output Target: {output_path}")
    Logger.log(f"[*] Predict Metal: {args.metal} -> {target_metal}")
    Logger.log(f"[*] SHARED_MAX_FEATURES: {shared_max_features}")
    Logger.log(f"[*] N_JOBS: {Config.N_JOBS}")

    runner = NestedCVOrchestrator(
        data_path=data_path,
        output_path=output_path,
        target_metals=[target_metal],
        max_features=shared_max_features,
        export_outputs=export_outputs
    )
    result = runner.run()
    Logger.console(
        f"[TRAIN] DONE  metal={args.metal} shared_max_features={shared_max_features}"
    )
    return result


def run_worker_jobs(current_dir, data_path, args):
    if not args.worker_max_features or not args.summary_xlsx:
        raise ValueError("Worker mode requires --worker-max-features and --summary-xlsx.")

    target_metal = METAL_COLUMN_MAP[args.metal]
    rows = []
    feature_values = parse_int_csv(args.worker_max_features)

    for idx, shared_max_features in enumerate(feature_values, start=1):
        Logger.log(
            f"[*] Worker GPU visible set={os.environ.get('CUDA_VISIBLE_DEVICES', 'CPU')} "
            f"| task {idx}/{len(feature_values)} | SHARED_MAX_FEATURES={shared_max_features}"
        )
        result = run_single_job(
            current_dir=current_dir,
            data_path=data_path,
            args=args,
            shared_max_features=shared_max_features,
            export_outputs=False
        )
        rows.append(
            collect_target_metrics(
                result=result,
                target_metal=target_metal,
                summary_model=args.summary_model,
                shared_max_features=shared_max_features
            )
        )
        ResultReporter.export_sweep_summary_to_excel(
            args.summary_xlsx,
            pd.DataFrame(rows).sort_values("SHARED_MAX_FEATURES"),
            sheet_name=f"{args.metal}_Sweep"
        )
        release_runtime_memory()

    return pd.DataFrame(rows).sort_values("SHARED_MAX_FEATURES")


def run_sweep_jobs(current_dir, data_path, args):
    sweep_values = list(
        range(
            Config.SHARED_MAX_FEATURES_SWEEP_START,
            Config.SHARED_MAX_FEATURES_SWEEP_END + 1,
            Config.SHARED_MAX_FEATURES_SWEEP_STEP
        )
    )
    sweep_output = (
        os.path.abspath(args.sweep_output)
        if args.sweep_output else build_sweep_output_path(current_dir, args.metal)
    )
    checkpoint_dir = build_sweep_checkpoint_dir(current_dir, args.metal, args.summary_model)
    os.makedirs(checkpoint_dir, exist_ok=True)

    target_metal = METAL_COLUMN_MAP[args.metal]
    existing_df = load_existing_sweep_results(
        sweep_output=sweep_output,
        checkpoint_dir=checkpoint_dir,
        target_metal=target_metal,
        summary_model=args.summary_model
    )
    completed_values = (
        set(existing_df["SHARED_MAX_FEATURES"].astype(int).tolist())
        if not existing_df.empty else set()
    )
    pending_values = [x for x in sweep_values if x not in completed_values]

    Logger.log(
        f"[*] Sweep resume scan: completed={len(completed_values)} "
        f"pending={len(pending_values)} checkpoint_dir={checkpoint_dir}"
    )
    if not existing_df.empty:
        ResultReporter.export_sweep_summary_to_excel(
            sweep_output,
            existing_df,
            sheet_name=f"{args.metal}_Sweep"
        )

    if not pending_values:
        Logger.log("[*] Sweep already complete. No pending SHARED_MAX_FEATURES values.")
        return existing_df

    gpu_ids = parse_int_csv(args.gpu_ids)
    if not gpu_ids:
        raise ValueError("--gpu-ids must contain at least one GPU id in sweep mode.")

    chunks = split_contiguous(pending_values, len(gpu_ids))
    run_id = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    worker_jobs = []

    for worker_id, (gpu_id, chunk) in enumerate(zip(gpu_ids, chunks), start=1):
        if not chunk:
            continue
        summary_xlsx = os.path.join(
            checkpoint_dir,
            f"run_{run_id}_worker_{worker_id:02d}_gpu_{gpu_id}.xlsx"
        )
        cmd = [
            sys.executable,
            os.path.abspath(__file__),
            "--metal", args.metal,
            "--worker-max-features", ",".join(str(x) for x in chunk),
            "--summary-xlsx", summary_xlsx,
            "--summary-model", args.summary_model,
            "--n-jobs", str(args.worker_n_jobs),
            "--no-oof-export",
        ]

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        env.setdefault("OMP_NUM_THREADS", "1")
        env.setdefault("MKL_NUM_THREADS", "1")
        env.setdefault("NUMEXPR_NUM_THREADS", "1")

        Logger.log(
            f"[*] Launch worker {worker_id}: GPU {gpu_id} "
            f"-> pending features {chunk[0]}..{chunk[-1]} ({len(chunk)} tasks)"
        )
        proc = subprocess.Popen(cmd, cwd=current_dir, env=env)
        worker_jobs.append((gpu_id, chunk, summary_xlsx, proc))

    failed = []
    for gpu_id, chunk, _, proc in worker_jobs:
        return_code = proc.wait()
        if return_code != 0:
            failed.append((gpu_id, chunk, return_code))

    frames = [existing_df]
    for _, _, summary_xlsx, _ in worker_jobs:
        if os.path.exists(summary_xlsx):
            frames.append(read_sweep_summary_xlsx(summary_xlsx, target_metal, args.summary_model))
        elif not failed:
            raise FileNotFoundError(f"Expected worker summary file not found: {summary_xlsx}")

    sweep_df = merge_sweep_summaries(frames)
    ResultReporter.export_sweep_summary_to_excel(sweep_output, sweep_df, sheet_name=f"{args.metal}_Sweep")

    if failed:
        raise RuntimeError(
            f"Sweep workers failed: {failed}. Partial results were saved to: {sweep_output}"
        )

    Logger.log(f"[*] Sweep complete: {len(sweep_df)}/{len(sweep_values)} feature counts saved.")
    return sweep_df


if __name__ == "__main__":
    suppress_warnings()

    parser = argparse.ArgumentParser(
        description="Run single-metal nested CV prediction with LassoCV-based feature selection."
    )
    parser.add_argument(
        "--metal",
        choices=sorted(METAL_COLUMN_MAP.keys()),
        required=True,
        help="Target metal to predict."
    )
    parser.add_argument(
        "--shared-max-features",
        type=int,
        default=Config.SHARED_MAX_FEATURES,
        help="Upper limit for the number of selected features in a single run."
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=Config.N_JOBS,
        help="CPU parallelism used inside one run."
    )
    parser.add_argument(
        "--summary-model",
        default="MoE",
        help="Model row to extract when summarizing metrics."
    )
    parser.add_argument(
        "--sweep-shared-max-features",
        action="store_true",
        help="Sweep SHARED_MAX_FEATURES from config start/end/step and aggregate one summary table."
    )
    parser.add_argument(
        "--gpu-ids",
        default="0,1,2,3,4,5,6,7",
        help="Comma-separated GPU ids used for sweep mode."
    )
    parser.add_argument(
        "--worker-n-jobs",
        type=int,
        default=1,
        help="CPU parallelism used inside each sweep worker."
    )
    parser.add_argument(
        "--sweep-output",
        default=None,
        help="Output XLSX path for the sweep summary table."
    )
    parser.add_argument(
        "--no-oof-export",
        action="store_true",
        help="Skip exporting per-run OOF prediction workbooks."
    )
    parser.add_argument("--worker-max-features", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--summary-xlsx", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    apply_runtime_config(args)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    Logger.init(
        log_dir=os.path.join(current_dir, "logs"),
        log_name=build_log_name(args)
    )
    data_path = os.path.abspath(os.path.join(current_dir, "data/Raman_spectroscopy_data_preprocessed.csv"))
    Logger.log(f"[*] Log File: {Logger.log_path()}")

    if args.worker_max_features:
        run_worker_jobs(current_dir, data_path, args)
    elif args.sweep_shared_max_features:
        run_sweep_jobs(current_dir, data_path, args)
    else:
        run_single_job(current_dir, data_path, args)
