import numpy as np
import pandas as pd
from utils.logger import Logger
from utils.metrics import MetricsUtil

class ResultReporter:
    @staticmethod
    def print_selected_bands(selected_idx, wavelength_cols, metal):
        selected_waves = wavelength_cols[selected_idx]
        selected_waves_float = pd.to_numeric(pd.Index(selected_waves), errors="coerce").values

        order_by_wave = np.argsort(selected_waves_float)
        Logger.log(f"[SelectedBands] ===== {metal} bands in ascending wavelength order =====")
        for k, oi in enumerate(order_by_wave, start=1):
            Logger.log(
                f"[SelectedBands] #{k:02d} idx={selected_idx[oi]:4d} "
                f"wave={selected_waves_float[oi]:.2f} nm (col='{selected_waves[oi]}')"
            )

        return selected_waves, selected_waves_float

    @staticmethod
    def build_summary_table(fold_metrics_df):
        mean_metrics = fold_metrics_df.groupby(['Target', 'Model', 'Set'])[['R2', 'RMSE', 'RPD']].mean().reset_index()

        pivot_df = mean_metrics.pivot(index=['Target', 'Model'], columns='Set', values=['R2', 'RMSE', 'RPD'])

        pivot_df.columns = [f"{col[1]} {col[0]}" for col in pivot_df.columns]
        pivot_df = pivot_df.reset_index()

        if "Test R2" in pivot_df.columns:
            pivot_df = pivot_df.rename(columns={
                "Test R2": "OOF R2",
                "Test RMSE": "OOF RMSE",
                "Test RPD": "OOF RPD"
            })

        cols_order = ['Target', 'Model', 'Train R2', 'Train RMSE', 'Train RPD', 'OOF R2', 'OOF RMSE', 'OOF RPD']
        valid_cols = [c for c in cols_order if c in pivot_df.columns]
        df_summary = pivot_df[valid_cols]

        model_order = {"PLS": 1, "KRR": 2, "ElasticNet": 3, "RFRR": 4, "RSRidge": 5, "GlobalStack": 6, "MoE": 7}
        df_summary = df_summary.copy()
        df_summary['_sort_idx'] = df_summary['Model'].map(model_order).fillna(99)
        df_summary = df_summary.sort_values(by=['Target', '_sort_idx']).drop(columns=['_sort_idx'])
        return df_summary

    @staticmethod
    def extract_train_test_metrics(fold_metrics_df, target, model):
        mean_metrics = (
            fold_metrics_df.groupby(['Target', 'Model', 'Set'])[['R2', 'RMSE', 'RPD']]
            .mean()
            .reset_index()
        )

        subset = mean_metrics[
            (mean_metrics['Target'] == target) &
            (mean_metrics['Model'] == model)
        ]
        if subset.empty:
            raise ValueError(f"No metrics found for target={target}, model={model}.")

        row = {"Target": target, "Model": model}
        for set_name in ["Train", "Test"]:
            set_df = subset[subset["Set"] == set_name]
            if set_df.empty:
                continue
            rec = set_df.iloc[0]
            row[f"{set_name} R2"] = float(rec["R2"])
            row[f"{set_name} RMSE"] = float(rec["RMSE"])
            row[f"{set_name} RPD"] = float(rec["RPD"])
        return row

    @staticmethod
    def export_sweep_summary_to_excel(output_path, sweep_df, sheet_name="SweepSummary"):
        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            sweep_df.to_excel(writer, sheet_name=sheet_name, index=False)
        Logger.log(f"[SAVE] Sweep summary saved to: {output_path}")

    @staticmethod
    def export_oof_predictions_to_excel(output_path, sample_ids, outer_fold_ids, target_metals, Y_true_matrix, oof_pred_global_dict, oof_pred_moe_dict, fold_metrics_df):
        sample_ids = np.asarray(sample_ids)
        outer_fold_ids = np.asarray(outer_fold_ids)

        df_g = pd.DataFrame({"Sample": sample_ids, "OuterFold": outer_fold_ids.astype(int)})
        df_m = pd.DataFrame({"Sample": sample_ids, "OuterFold": outer_fold_ids.astype(int)})

        for j, metal in enumerate(target_metals):
            y_true = np.asarray(Y_true_matrix[:, j], dtype=float)
            pred_g = np.asarray(oof_pred_global_dict[metal], dtype=float)
            pred_m = np.asarray(oof_pred_moe_dict[metal], dtype=float)

            df_g[f"{metal}_true"] = y_true
            df_g[f"{metal}_pred"] = pred_g
            df_m[f"{metal}_true"] = y_true
            df_m[f"{metal}_pred"] = pred_m

        df_summary = ResultReporter.build_summary_table(fold_metrics_df)

        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            df_g.to_excel(writer, sheet_name="GlobalStack_Predictions", index=False)
            df_m.to_excel(writer, sheet_name="MoE_Predictions", index=False)
            fold_metrics_df.to_excel(writer, sheet_name="Fold_Performance", index=False)
            df_summary.to_excel(writer, sheet_name="Global_Summary", index=False)

        Logger.log(f"[SAVE] OOF predictions and summaries saved to: {output_path}")
