"""Self-supervised loss terms for S-RegNET.

Respects the input contract: the network still sees exactly two inputs, the
genus-0 template and the SynthSeg one-hot. Nothing here loads new data. Both
terms act on the model's own output or on a transformation of an input that
has already been loaded.

1. mesh_injectivity_loss
   The repo's forensics (utils/mesh_flip_probe.py, hypothesis 2) found the
   lattice determinant test does not predict mesh failure: 0.013% voxel folding
   coexisted with 10.49% cortical triangle flips. It cannot predict it -- the
   pushed mesh carries ~0.8 mm triangles on a 2 mm lattice, so non-injectivity
   that lives between lattice samples is invisible to a test that only looks at
   those samples. This evaluates det(I + d flow/dp) analytically AT the genus-0
   template's vertex positions, at a sub-voxel step, on flow_rv -- the field the
   deliverable push actually uses. Same penalty shape as
   losses.jacobian_det_loss, so its weight reads the same way.

2. equivariance_loss
   A random smooth warp T is applied to the already-loaded SynthSeg input. The
   prediction on T.input should agree with T applied to the prediction on the
   original input. No label is involved. It penalises erratic, input-sensitive,
   high-frequency field behaviour -- the mechanism behind self-intersection.
   The transformed branch runs under no_grad as a stop-gradient target, so the
   second forward pass stores no activations; at 192^3 a second graph would not
   fit alongside the first.
"""
import torch
import torch.nn.functional as F


def sample_field(field, pts):
    """Sample (1,3,D,H,W) field, channel order (x,y,z), at pts (N,3) normalized.
    align_corners=False and padding_mode='border' match model.SpatialTransformer
    and train._sample_field; any other choice silently shifts the mesh."""
    g = pts.view(1, -1, 1, 1, 3)
    s = F.grid_sample(field, g, mode='bilinear', padding_mode='border',
                      align_corners=False)
    return s[0, :, :, 0, 0].T


def vertex_jacobian_det(flow, pts, h):
    """det(I + d flow/dp) at pts by central differences at spacing h.

    flow (1,3,D,H,W) displacement in normalized coords; pts (N,3) normalized;
    h is in NORMALIZED units and should be sub-voxel -- that is the point.
    det <= 0 means the map is locally non-injective there: the mesh folds."""
    cols = []
    for k in range(3):
        e = torch.zeros(3, device=pts.device, dtype=pts.dtype)
        e[k] = h
        cols.append((sample_field(flow, pts + e) - sample_field(flow, pts - e)) / (2 * h))
    J = torch.stack(cols, dim=2)
    J = J + torch.eye(3, device=pts.device, dtype=J.dtype)
    return torch.linalg.det(J)


def mesh_injectivity_loss(flow, pts, h, margin=0.1, topk_frac=0.001):
    """Linear mean + top-K hard mining + pre-fold barrier, on the mesh."""
    det = vertex_jacobian_det(flow, pts, h)
    neg = F.relu(-det)
    k = max(1, int(topk_frac * det.numel()))
    topk = torch.topk(neg, k).values.sum()
    barrier = (F.relu(margin - det) * (det > 0).to(det.dtype)) ** 2
    return neg.mean() + 2.0 * topk + 0.1 * barrier.mean(), det


def random_smooth_warp(shape, device, amp=0.03, ctrl=8, dtype=torch.float32):
    """Random displacement from a tiny upsampled control grid: band-limited by
    construction, so it cannot itself inject the high-frequency content the
    term is meant to discourage."""
    B, _, D, H, W = shape
    g = torch.randn(B, 3, ctrl, ctrl, ctrl, device=device, dtype=dtype)
    g = F.interpolate(g, size=(D, H, W), mode='trilinear', align_corners=False)
    return g / g.abs().amax().clamp_min(1e-8) * amp


def equivariance_loss(model, template_seg, input_seg, flow_fw, stn, amp=0.03, ctrl=8,
                      use_amp=True):
    """MSE between T.phi(tpl, x) and phi(tpl, T.x), the latter a stop-grad target.

    Reuses flow_fw from the main forward pass, so only ONE extra forward runs,
    and it runs without a graph."""
    with torch.no_grad():
        T = random_smooth_warp(input_seg.shape, input_seg.device, amp, ctrl,
                               dtype=torch.float32)
        x_t = stn(input_seg.float(), T)
        with torch.cuda.amp.autocast(enabled=use_amp):
            target = model(template_seg, x_t)[0]
        target = target.float()
    return F.mse_loss(stn(flow_fw.float(), T), target)
