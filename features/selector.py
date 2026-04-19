import numpy as np
from sklearn.linear_model import LassoCV
from config import Config

class FeatureSelector:
    @staticmethod
    def select_bands_lasso(X_train, y_train, max_features=Config.SHARED_MAX_FEATURES):
        lasso = LassoCV(
            alphas=np.logspace(-4, 1, 15),
            cv=5,
            random_state=Config.RANDOM_STATE,
            n_jobs=Config.N_JOBS,
            max_iter=10000
        )
        lasso.fit(X_train, y_train)

        importance_all = np.abs(lasso.coef_)

        nonzero_idx = np.where(importance_all > 1e-6)[0]

        if len(nonzero_idx) == 0:
            selected_idx = np.argsort(importance_all)[::-1][:max_features]
        else:
            sorted_nonzero = np.argsort(importance_all[nonzero_idx])[::-1]
            selected_idx = nonzero_idx[sorted_nonzero]
            if len(selected_idx) > max_features:
                selected_idx = selected_idx[:max_features]

        return np.sort(selected_idx), importance_all
