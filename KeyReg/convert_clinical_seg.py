"""Turn recon-all-clinical asegs into model-ready 5-class one-hot volumes.

The KeyReg models consume (5, D, H, W) one-hot segmentations on the neurite-OASIS
`aligned_*` grid (160x192x224). recon-all-clinical writes its aseg on FreeSurfer's
own conformed grid, so each aseg is resampled onto the subject's aligned_seg4 grid
through the two world affines (nearest neighbour -- these are labels) before being
one-hot encoded. Both volumes describe the same scan, so this is a pure regrid, no
registration.

Label remapping reuses the project's existing definition in
archive/S-RegNET/label_mapping.py: 0 bg, 1 cortex, 2 subcortical GM, 3 WM, 4 CSF.

    python convert_clinical_seg.py --out_list clinical_all.txt

By default the moving/template subject is excluded. Use --source_list to create a
paired clinical list in exactly the same order as an existing GT split.
"""
import argparse, os, re, sys
import numpy as np
import nibabel as nib
from scipy.ndimage import map_coordinates

_D = "/shared/scratch/0/home/v_vijay_bala_mahalingam"
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "archive", "S-RegNET"))
from label_mapping import remap_to_5class          # noqa: E402

NUM_CLASSES = 5

ap = argparse.ArgumentParser()
ap.add_argument("--neurite", default=f"{_D}/neurite_oasis")
ap.add_argument("--surf_root", default=f"{_D}/oasis_mri_outputs")
ap.add_argument("--suffix", default="_clinical")
ap.add_argument("--name", default="seg4_onehot_clinical.npy",
                help="written next to each subject's neurite volumes")
ap.add_argument("--out_list", default=f"{_D}/neurite_oasis/clinical_all.txt")
ap.add_argument("--source_list", default=None,
                help="optional GT list defining the included subjects and their order")
ap.add_argument("--exclude_subject", default="OASIS_OAS1_0001_MR1",
                help="subject directory name to exclude as the moving/template image")
ap.add_argument("--require_surf", action="store_true", default=True,
                help="only include subjects that also have lh/rh.white")
ap.add_argument("--force", action="store_true", help="rebuild even if cached")
args = ap.parse_args()


def regrid(aseg_img, ref_img):
    """Sample aseg onto ref's voxel grid via world coordinates, nearest neighbour."""
    D, H, W = ref_img.shape
    ii, jj, kk = np.meshgrid(np.arange(D), np.arange(H), np.arange(W), indexing="ij")
    idx = np.stack([ii, jj, kk, np.ones_like(ii)], -1).reshape(-1, 4).astype(np.float64)
    world = idx @ ref_img.affine.T                                  # ref vox -> world
    src = world @ np.linalg.inv(aseg_img.affine).T                  # world -> aseg vox
    vals = map_coordinates(np.asarray(aseg_img.dataobj).astype(np.int32),
                           src[:, :3].T, order=0, mode="constant", cval=0)
    return vals.reshape(D, H, W)


def main():
    if args.source_list:
        subs = [os.path.basename(os.path.dirname(line.strip()))
                for line in open(args.source_list) if line.strip()]
    else:
        subs = sorted(s for s in os.listdir(args.neurite) if s.startswith("OASIS_"))
    subs = [s for s in subs if s != args.exclude_subject]
    written, skipped, bad = [], [], []
    for n, subj in enumerate(subs, 1):
        m = re.search(r"(OAS1_\d+)", subj)
        sd = os.path.join(args.surf_root, (m.group(1) if m else "") + args.suffix)
        aseg_p = os.path.join(sd, "mri", "aseg.mgz")
        ref_p = os.path.join(args.neurite, subj, "aligned_seg4.nii.gz")
        out_p = os.path.join(args.neurite, subj, args.name)
        has_surf = (os.path.exists(os.path.join(sd, "surf", "lh.white"))
                    and os.path.exists(os.path.join(sd, "surf", "rh.white")))
        if not (m and os.path.exists(aseg_p) and os.path.exists(ref_p)) or \
           (args.require_surf and not has_surf):
            skipped.append(subj); continue
        if os.path.exists(out_p) and not args.force:
            written.append(out_p); continue

        ref = nib.load(ref_p)
        seg = remap_to_5class(regrid(nib.load(aseg_p), ref))
        oh = np.zeros((NUM_CLASSES, *seg.shape), dtype=np.uint8)
        for c in range(NUM_CLASSES):
            oh[c] = (seg == c)
        chk = oh.sum(0)
        if not (chk.min() == 1 and chk.max() == 1):
            bad.append(subj); print(f"  BAD one-hot: {subj}", flush=True); continue
        np.save(out_p, oh)
        written.append(out_p)
        wm = int((seg == 3).sum())
        print(f"[{n}/{len(subs)}] {subj}: WM {wm:,} vox -> {os.path.basename(out_p)}", flush=True)

    with open(args.out_list, "w") as f:
        for p in written:
            f.write(p + "\n")
    print(f"\nwrote {len(written)} one-hot volumes; list -> {args.out_list}")
    print(f"skipped {len(skipped)} (no recon-all-clinical output yet), malformed {len(bad)}")


if __name__ == "__main__":
    main()
