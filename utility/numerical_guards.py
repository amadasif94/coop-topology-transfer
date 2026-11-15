"""Shared NaN/Inf guard for optimizer updates (F19). No effect on finite values."""
import torch


def require_finite(value, what, optimizer=None):
    """Fail explicitly instead of applying a NaN/Inf update.

    Called before backward() for losses and before optimizer.step() for gradient norms;
    on failure the pending gradients are discarded and FloatingPointError is raised, so
    the offending step is never applied.
    """
    if not bool(torch.isfinite(value).all()):
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        raise FloatingPointError(f"Non-finite {what} ({value.detach().cpu().tolist()}); "
                                 "optimizer step refused, parameters for this step left unchanged.")
