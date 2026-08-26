# Archive — superseded pipelines

Earlier work, kept for provenance. **None of these is the current model.**
The current model is [`../KeyReg/`](../KeyReg/) — see the [root README](../README.md).

| Folder | Input | Why it was superseded |
|---|---|---|
| `S-RegNET/` | Segmentation | Seg-only registration with a bounded B-spline FFD cascade and a genus-0 WM mesh deliverable. Replaced by KeyReg's stationary velocity field, which is diffeomorphic by construction rather than by penalty. |
| `S-RegNET_exp4/` | Segmentation | Experiment-4 variant of S-RegNET. Same lineage. |
| `M-RegNET/` | MRI | Dense field on MRI intensities with segmentation-guided attention. The project moved to segmentation-only inputs. |

Each still has its own README, model, losses, training loop and `config.yaml`, and
should still run — only the paths have changed.
