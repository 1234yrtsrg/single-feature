import os
import gc
import sys
import argparse
import subprocess
import shutil
import warnings
import tempfile

import pandas as pd
from sklearn.exceptions import ConvergenceWarning

from config import Config
from pipeline.orchestrator import NestedCVOrchestrator
from pipeline.reporter import ResultReporter

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


def build_single_output_path(current_dir, metal, shared_max_features):
    base_name = f"nestedcv_oof_predictions_{metal}.xlsx"
    if shared_max_features != Config.SHARED_MAX_FEATURES:
        base_name = f"nestedcv_oof_predictions_{metal}_k{shared_max_features}.xlsx"
    return os.path.abspath(os.path.join(current_dir, base_name))


def build_sweep_output_path(current_dir, metal):
    return os.path.abspath(os.path.join(current_dir, f"shared_max_features_sweep_{metal}.xlsx"))


def collect_target_metrics(result, target_metal, summary_model, shared_max_features):
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

    print(f"[*] Data Source: {data_path}")
    if output_path:
        print(f"[*] Output Target: {output_path}")
    print(f"[*] Predict Metal: {args.metal} -> {target_metal}")
    print(f"[*] SHARED_MAX_FEATURES: {shared_max_features}")
    print(f"[*] N_JOBS: {Config.N_JOBS}")

    runner = NestedCVOrchestrator(
        data_path=data_path,
        output_path=output_path,
        target_metals=[target_metal],
        max_features=shared_max_features,
        export_outputs=export_outputs
    )
    return runner.run()


def run_worker_jobs(current_dir, data_path, args):
    if not args.worker_max_features or not args.summary_xlsx:
        raise ValueError("Worker mode requires --worker-max-features and --summary-xlsx.")

    target_metal = METAL_COLUMN_MAP[args.metal]
    rows = []
    feature_values = parse_int_csv(args.worker_max_features)

    for idx, shared_max_features in enumerate(feature_values, start=1):
        print(
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
    gpu_ids = parse_int_csv(args.gpu_ids)
    chunks = split_contiguous(sweep_values, len(gpu_ids))
    temp_dir = tempfile.mkdtemp(prefix=f"sweep_{args.metal.lower()}_", dir=current_dir)
    try:
        worker_jobs = []
        for worker_id, (gpu_id, chunk) in enumerate(zip(gpu_ids, chunks), start=1):
            if not chunk:
                continue
            summary_xlsx = os.path.join(temp_dir, f"worker_{worker_id:02d}_gpu_{gpu_id}.xlsx")
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

            print(f"[*] Launch worker {worker_id}: GPU {gpu_id} -> features {chunk[0]}..{chunk[-1]}")
            proc = subprocess.Popen(cmd, cwd=current_dir, env=env)
            worker_jobs.append((gpu_id, chunk, summary_xlsx, proc))

        failed = []
        for gpu_id, chunk, _, proc in worker_jobs:
            return_code = proc.wait()
            if return_code != 0:
                failed.append((gpu_id, chunk, return_code))

        if failed:
            raise RuntimeError(f"Sweep workers failed: {failed}")

        frames = []
        for _, _, summary_xlsx, _ in worker_jobs:
            if not os.path.exists(summary_xlsx):
                raise FileNotFoundError(f"Expected worker summary file not found: {summary_xlsx}")
            frames.append(pd.read_excel(summary_xlsx))

        sweep_df = pd.concat(frames, ignore_index=True).sort_values("SHARED_MAX_FEATURES").reset_index(drop=True)
        sweep_output = (
            os.path.abspath(args.sweep_output)
            if args.sweep_output else build_sweep_output_path(current_dir, args.metal)
        )

        ResultReporter.export_sweep_summary_to_excel(sweep_output, sweep_df, sheet_name=f"{args.metal}_Sweep")
        return sweep_df
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


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
    data_path = os.path.abspath(os.path.join(current_dir, "data/Raman_spectroscopy_data_preprocessed.csv"))

    if args.worker_max_features:
        run_worker_jobs(current_dir, data_path, args)
    elif args.sweep_shared_max_features:
        run_sweep_jobs(current_dir, data_path, args)
    else:
        run_single_job(current_dir, data_path, args)
