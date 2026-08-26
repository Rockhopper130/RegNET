"""Routing for the two seg inputs (torch-free so it is unit-testable)."""


def build_step_inputs(synthseg_seg, gt_seg, *, input_source, supervision_target):
    """Pick the model's 2nd input and the supervision target for one step.

    input_source:       'synthseg' (deployable, default) | 'gt' (sanity).
    supervision_target: 'gt' (supervised, default) | 'synthseg' (ablation).

    Returns (input_seg, target_seg).
    """
    if input_source not in ('synthseg', 'gt'):
        raise ValueError(f"input_source must be 'synthseg' or 'gt', got {input_source!r}")
    if supervision_target not in ('gt', 'synthseg'):
        raise ValueError(
            f"supervision_target must be 'gt' or 'synthseg', got {supervision_target!r}")

    input_seg = synthseg_seg if input_source == 'synthseg' else gt_seg
    target_seg = gt_seg if supervision_target == 'gt' else synthseg_seg
    return input_seg, target_seg
