import numpy as np

def get_top_samples(
    costs: np.ndarray,
    samples: np.ndarray,
    N_top_samples: float
    ) -> np.ndarray:
    assert costs.ndim == 2, "Expected costs of shape (N, T)."
    assert samples.ndim == 3, "Expected samples of shape (N, T, D)."
    assert costs.shape[:2] == samples.shape[:2], "Mismatched iteration/sample dimensions."

    D = samples.shape[2]
    # Flatten across all iterations
    costs_flat = costs.reshape(-1)
    samples_flat = samples.reshape(-1, D)
    
    # Remove samples in double (in case keep elites for instance)
    costs_flat_unique, arg_unique = np.unique(costs_flat, return_index=True, sorted=True)
    samples_flat_unique = samples_flat[arg_unique]

    return samples_flat_unique[:N_top_samples], costs_flat_unique[:N_top_samples]