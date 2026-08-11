"""
Seg-only registration network with a bounded cubic-B-spline FFD.

Both inputs are 5-channel one-hot segmentations. Architecture:

    template_seg (5ch) ─┐
    sample_seg   (5ch) ─┼─► [AffineNet] ──► 3×4 affine ──► warps template_seg
                        │                                          │
                        ▼                                          ▼
                   concat (10ch) ──► UNet backbone ──► decoder feature @ cp_spacing
                                                              │
                                                              ▼
                                          control-point heads (clamped)
                                                              │
                                                              ▼
                                          BoundedBSplineFFD cascade

A cubic-B-spline FFD whose control-point displacements are clamped below the
Choi-Lee injectivity bound is a diffeomorphism, and composing several stages
stays one. Training consumes the dense field as a pull field through the
SpatialTransformer (warped(x) = template(x + flow(x))) against a voxel Dice
loss. The genus-0 template mesh is carried template->subject afterwards via the
field's numerical inverse (push_points_to_sample -> invert_to_sample), so its
topology is preserved without any mesh or Jacobian loss term.

Warp order: affine first, then the FFD cascade. Volume consumers replay
affine-grid then SpatialTransformer(aligned, flow); the mesh push replays the
same order inverted (_apply_affine_inverse then invert_to_sample). forward()
returns the clamped control points per stage plus the affine; dense_flow_from_cps()
rebuilds the dense field for the warp and the folding diagnostics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Building Blocks
# =============================================================================

class AffineNet(nn.Module):
    """
    Predicts a 3x4 affine matrix from concat(template_seg, sample_seg).
    Identity-initialised so training starts from a no-op affine.
    """
    def __init__(self, in_channels=10):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, 16, kernel_size=3, padding=1),
            nn.InstanceNorm3d(16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool3d(2),
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.InstanceNorm3d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool3d(2),
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.InstanceNorm3d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool3d(1),
        )
        self.fc = nn.Linear(64, 12)
        self.fc.weight.data.zero_()
        self.fc.bias.data.copy_(torch.tensor([
            1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0,
        ], dtype=torch.float))

    def forward(self, template_seg, sample_seg):
        x = torch.cat([template_seg, sample_seg], dim=1)
        f = self.conv(x).view(x.size(0), -1)
        return self.fc(f).view(-1, 3, 4)


class ConvBlock(nn.Module):
    """Conv → InstanceNorm → LeakyReLU, twice."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


# =============================================================================
# Cubic B-spline upsampling
# =============================================================================

def _cubic_bspline_kernel_1d(stride):
    """1D uniform cubic B-spline kernel for integer upsampling `stride`.

    Returns a length-(4*stride - 1) tensor with w[m] = B(m/stride), B the uniform
    cubic B-spline basis (support |t| < 2). Convolving a stride-spaced control
    lattice with this kernel gives cubic B-spline interpolation.
    """
    radius = 2 * stride - 1
    m = torch.arange(-radius, radius + 1, dtype=torch.float32)
    t = (m / stride).abs()
    w = torch.zeros_like(t)
    near = t < 1
    far = (t >= 1) & (t < 2)
    w[near] = (3 * t[near] ** 3 - 6 * t[near] ** 2 + 4) / 6
    w[far] = (2 - t[far]) ** 3 / 6
    return w


class CubicBSplineUpsample3d(nn.Module):
    """Upsample a coarse control-point grid to a dense field by cubic B-spline
    interpolation, as a fixed (non-learnable) separable convolution on the
    lattice-stuffed tensor. Output size equals the full resolution exactly.

    The interpolant's spatial gradient is bounded by the control-point spacing,
    which is what lets a clamp on the control-point magnitude bound the Jacobian
    and keep the warp injective. The last ~2*stride voxels at each boundary taper
    toward zero (lattice support truncates there), which is harmless since the
    volume edge is background.
    """

    def __init__(self, stride):
        super().__init__()
        self.stride = int(stride)
        k1d = _cubic_bspline_kernel_1d(self.stride)
        L = k1d.numel()
        self.pad = (L - 1) // 2
        k3d = k1d[:, None, None] * k1d[None, :, None] * k1d[None, None, :]  # (L, L, L)
        weight = k3d[None, None].repeat(3, 1, 1, 1, 1)  # (3, 1, L, L, L) — groups=3
        self.register_buffer('weight', weight)

    def forward(self, control_points, out_size):
        """control_points: (B, 3, nd, nh, nw) on the stride lattice.
        out_size: (D, H, W). Returns the dense field (B, 3, D, H, W)."""
        s = self.stride
        stuffed = control_points.new_zeros(control_points.shape[0], 3, *out_size)
        stuffed[..., ::s, ::s, ::s] = control_points
        return F.conv3d(stuffed, self.weight, padding=self.pad, groups=3)


# cp_spacing → (UNet decoder feature key used as control-point source, its channels)
_BSPLINE_SOURCE = {2: ('d2', 64), 4: ('d3', 128), 8: ('d4', 256)}


# =============================================================================
# UNet backbone (decoder features only; heads live in BoundedBSplineFFD)
# =============================================================================

class UNet(nn.Module):
    """
    4-level UNet on concat(template_seg, sample_seg) (10 channels).

    Returns the four decoder features keyed by scale so the FFD can tap the one
    matching the control lattice (cp_spacing):

        d4: 16³ (256ch, stride 8)   d3: 32³ (128ch, stride 4)
        d2: 64³ (64ch,  stride 2)   d1: 128³ (32ch, stride 1)

    The control-point heads live in BoundedBSplineFFD.
    """
    def __init__(self, in_channels=10):
        super().__init__()

        # Encoder
        self.enc1 = ConvBlock(in_channels, 32)
        self.enc2 = ConvBlock(32, 64)
        self.enc3 = ConvBlock(64, 128)
        self.enc4 = ConvBlock(128, 256)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck = ConvBlock(256, 512)

        # Decoder
        self.up4 = nn.ConvTranspose3d(512, 256, kernel_size=2, stride=2)
        self.dec4 = ConvBlock(512, 256)
        self.up3 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(256, 128)
        self.up2 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(128, 64)
        self.up1 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(64, 32)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # Bottleneck
        b = self.bottleneck(self.pool(e4))

        # Decoder with skip connections
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return {'d4': d4, 'd3': d3, 'd2': d2, 'd1': d1}


# =============================================================================
# Bounded forward B-spline FFD transform
# =============================================================================

class BoundedBSplineFFD(nn.Module):
    """A cascade of bounded cubic-B-spline FFD stages on one control lattice.

    Each stage is a control-point head (near-zero init -> identity at start)
    whose raw output is clamped per control point to

        cp = delta_max * tanh(cp_raw / delta_max),
        delta_max = injectivity_k * cp_spacing * (2 / N)     [normalized units]

    The clamp keeps every stage's displacement below the Choi-Lee injectivity
    bound, so each stage and their composition stay diffeomorphic. delta_max is a
    scalar because the grid is isotropic; an anisotropic grid would need a
    per-axis value.
    """

    def __init__(self, in_channels, cp_spacing, n_stages, injectivity_k, target_size):
        super().__init__()
        self.cp_spacing = int(cp_spacing)
        self.n_stages = int(n_stages)
        N = int(target_size[0])
        assert tuple(target_size) == (N, N, N), \
            f"BoundedBSplineFFD assumes an isotropic grid; got {tuple(target_size)}"
        # Clamp magnitude in normalized units (control spacing = cp_spacing * 2/N).
        self.delta_max = float(injectivity_k) * self.cp_spacing * (2.0 / N)

        # One head per stage; near-zero init -> control points ~ 0 -> identity warp.
        self.heads = nn.ModuleList()
        for _ in range(self.n_stages):
            head = nn.Conv3d(in_channels, 3, kernel_size=3, padding=1)
            nn.init.normal_(head.weight, 0, 1e-4)
            nn.init.zeros_(head.bias)
            self.heads.append(head)

        self.upsample = CubicBSplineUpsample3d(self.cp_spacing)

    def control_points(self, feat):
        """Decoder feature (B, C, n, n, n) -> list of n_stages clamped control
        grids, each (B, 3, n, n, n)."""
        dm = self.delta_max
        return [dm * torch.tanh(head(feat) / dm) for head in self.heads]

    def dense_flow(self, cps_list, out_size):
        """Compose the cascade into one dense displacement field (B, 3, D, H, W).
        A single stage is just the B-spline interpolant; for a cascade the later
        stages are sampled at the running warped grid, so sampling the returned
        field at x reproduces the staged recurrence x -> x + u1(x) -> ... -> x_K."""
        flow = self.upsample(cps_list[0], out_size)
        if len(cps_list) == 1:
            return flow
        grid = _identity_grid(out_size, flow.device, flow.dtype)  # (1,D,H,W,3) x,y,z
        for cps in cps_list[1:]:
            u = self.upsample(cps, out_size)
            warped = (grid + flow.permute(0, 2, 3, 4, 1))           # sample u at id+flow
            sampled = F.grid_sample(u, warped, mode='bilinear',
                                    padding_mode='border', align_corners=False)
            flow = flow + sampled
        return flow

    def deform_points(self, points, cps_list, out_size):
        """Apply the FFD cascade directly to points (N, 3) in normalized (x, y, z):
        v_k = v_{k-1} + u_k(v_{k-1}), sampling each stage's dense field at the
        current positions."""
        v = points
        for cps in cps_list:
            u = self.upsample(cps, out_size)
            v = v + _sample_field(u, v)
        return v


def _identity_grid(size, device, dtype):
    """Pixel-centre identity grid (1, D, H, W, 3), channel order (x, y, z),
    matching `SpatialTransformer` / align_corners=False."""
    D, H, W = size
    lin_z = (2 * torch.arange(D, device=device, dtype=dtype) + 1) / D - 1
    lin_y = (2 * torch.arange(H, device=device, dtype=dtype) + 1) / H - 1
    lin_x = (2 * torch.arange(W, device=device, dtype=dtype) + 1) / W - 1
    zz, yy, xx = torch.meshgrid(lin_z, lin_y, lin_x, indexing='ij')
    return torch.stack((xx, yy, zz), dim=-1).unsqueeze(0)


def _sample_field(field, pts):
    """Sample (1, 3, D, H, W) field at pts (N, 3 normalized x,y,z) -> (N, 3).
    align_corners=False matches the model grid; padding_mode='border' so edge
    points read the boundary value rather than zero."""
    g = pts.view(1, -1, 1, 1, 3)
    s = F.grid_sample(field, g, mode='bilinear', padding_mode='border',
                      align_corners=False)               # (1, 3, N, 1, 1)
    return s.squeeze(0).squeeze(-1).squeeze(-1).permute(1, 0)   # (N, 3) order (x,y,z)


def invert_to_sample(flow, points, n_iter=50, alpha=0.5):
    """Damped fixed-point inverse of the dense pull field: find sample-space o
    with o + flow(o) = v for each template-space point v. This is the mesh-push
    direction (template->subject).

    Iterates o <- o + alpha*(v - flow(o) - o); alpha < 1 damps oscillation where
    the flow is locally non-contractive. Where the forward map folds, no o exists
    and that vertex's residual stays high (gate the mesh on the residual).

    Args:
        flow:   (1, 3, D, H, W) dense field, channel order (x, y, z).
        points: (N, 3) template-space points, normalized (x, y, z); numpy or torch.
    Returns:
        o:   (N, 3) inverse-mapped (subject-space) points.
        res: (N,) per-vertex residual ||o + flow(o) - v||.
    """
    v = torch.as_tensor(points, dtype=torch.float32, device=flow.device)
    o = v.clone()
    for _ in range(n_iter):
        o = o + alpha * (v - _sample_field(flow, o) - o)
    res = (o + _sample_field(flow, o) - v).norm(dim=1)
    return o, res


# =============================================================================
# Main Model
# =============================================================================

class SegRegistrationNet(nn.Module):
    """
    Seg-only registration network: optional affine pre-alignment then a bounded
    B-spline FFD cascade.

    forward(template_seg, sample_seg) -> (cps_list, affine_matrix):
        cps_list:      list of n_stages clamped control grids, each
                       (B, 3, n, n, n) with n = D / cp_spacing.
        affine_matrix: (B, 3, 4) or None.

    Use dense_flow_from_cps() to materialise the dense field (for the warp and
    folding diagnostics) and push_points_to_sample() to carry mesh vertices.
    """
    def __init__(self, seg_channels=5, use_affine=False,
                 cp_spacing=8, n_stages=1, injectivity_k=0.40,
                 target_size=(128, 128, 128)):
        super().__init__()
        self.use_affine = use_affine
        self.target_size = tuple(target_size)
        assert cp_spacing in _BSPLINE_SOURCE, \
            f"cp_spacing must be one of {sorted(_BSPLINE_SOURCE)}, got {cp_spacing}"
        self.cp_source_key, src_channels = _BSPLINE_SOURCE[cp_spacing]

        if use_affine:
            self.affine_net = AffineNet(in_channels=2 * seg_channels)

        self.unet = UNet(in_channels=2 * seg_channels)
        self.ffd = BoundedBSplineFFD(
            in_channels=src_channels, cp_spacing=cp_spacing, n_stages=n_stages,
            injectivity_k=injectivity_k, target_size=self.target_size,
        )

    def forward(self, template_seg, sample_seg):
        affine_matrix = None
        if self.use_affine:
            affine_matrix = self.affine_net(template_seg, sample_seg)
            affine_grid = F.affine_grid(affine_matrix, template_seg.size(), align_corners=False)
            template_seg = F.grid_sample(
                template_seg, affine_grid, mode='nearest',
                padding_mode='zeros', align_corners=False,
            )

        x = torch.cat([template_seg, sample_seg], dim=1)
        feats = self.unet(x)
        cps_list = self.ffd.control_points(feats[self.cp_source_key])
        return cps_list, affine_matrix

    def dense_flow_from_cps(self, cps_list):
        """Reconstruct the dense displacement field (B, 3, D, H, W)."""
        return self.ffd.dense_flow(cps_list, self.target_size)

    def deform_points(self, points, cps_list, affine_matrix=None):
        """Apply the warp directly to points (N, 3) in normalized (x, y, z),
        affine first then the FFD cascade. Affine on points uses theta^-1 (the
        affine warps the template volume, so a template point v at output p has
        theta*p = v, i.e. p = theta^-1 * v). Used by the smoke test; the
        deliverable mesh uses push_points_to_sample."""
        if affine_matrix is not None:
            v = _apply_affine_inverse(affine_matrix, points)
        else:
            v = points
        return self.ffd.deform_points(v, cps_list, self.target_size)

    def push_points_to_sample(self, points, cps_list, affine_matrix=None,
                              n_iter=50, alpha=0.5):
        """Carry template-space points (N, 3) into subject space: the numerical
        inverse of the warp, replaying affine-then-flow inverted (affine^-1 first,
        then the field's damped fixed-point inverse). Returns (pushed (N, 3),
        residual (N,)); gate the mesh on the residual."""
        v = (_apply_affine_inverse(affine_matrix, points)
             if affine_matrix is not None else points)
        flow = self.dense_flow_from_cps(cps_list)
        return invert_to_sample(flow, v, n_iter, alpha)


def _apply_affine_inverse(theta, pts):
    """Apply theta⁻¹ to points (N, 3) normalized (x, y, z). theta is (1, 3, 4)
    in the F.affine_grid convention (maps output coords → input coords)."""
    B = theta.shape[0]
    assert B == 1, "point-deformation affine replay supports batch size 1"
    bottom = theta.new_tensor([0.0, 0.0, 0.0, 1.0]).view(1, 1, 4).expand(B, 1, 4)
    full = torch.cat([theta, bottom], dim=1)                  # (1, 4, 4)
    inv = torch.linalg.inv(full)[0]                           # (4, 4)
    ones = pts.new_ones(pts.shape[0], 1)
    hom = torch.cat([pts, ones], dim=1)                       # (N, 4)
    out = hom @ inv.t()                                       # (N, 4)
    return out[:, :3]


# =============================================================================
# Spatial Transformer (dense volume warp)
# =============================================================================

class SpatialTransformer(nn.Module):
    """
    Dense STN for 3D volumes. The identity grid uses pixel-centre coords
    consistent with align_corners=False; padding_mode='zeros' avoids the
    boundary streak artifact border-replication produces.

    Applies flow with grid_sample backward semantics (output[p] = moving[p +
    flow(p)]), i.e. the FFD field is consumed as a pull field. This is the
    training warp path; the mesh is pushed the other way via invert_to_sample.
    """
    def __init__(self, size, device='cpu'):
        super().__init__()
        D, H, W = size

        # Pixel-centre coordinates under align_corners=False: (2i+1)/N - 1.
        lin_z = (2 * torch.arange(D, device=device).float() + 1) / D - 1
        lin_y = (2 * torch.arange(H, device=device).float() + 1) / H - 1
        lin_x = (2 * torch.arange(W, device=device).float() + 1) / W - 1
        zz, yy, xx = torch.meshgrid(lin_z, lin_y, lin_x, indexing='ij')

        id_grid = torch.stack((xx, yy, zz), dim=-1)
        self.register_buffer('id_grid', id_grid.unsqueeze(0))

    def forward(self, moving, flow):
        """
        Args:
            moving: (B, C, D, H, W)
            flow:   (B, 3, D, H, W) — channel order (x, y, z)
        Returns:
            warped: (B, C, D, H, W)
        """
        B = moving.shape[0]
        flow = flow.permute(0, 2, 3, 4, 1)
        grid = self.id_grid.expand(B, -1, -1, -1, -1)
        warped_grid = grid + flow

        return F.grid_sample(
            moving, warped_grid,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False,
        )
