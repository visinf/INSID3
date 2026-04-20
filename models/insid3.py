"""INSID3: In-context segmentation with a frozen DINOv3 encoder."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
import math
from PIL import Image

from utils.clustering import agglomerative_clustering, compute_cluster_prototypes
from utils.data import build_transform, downsample_mask, load_image, load_mask
from utils.keypoints import kernel_softargmax_get_matches_logits, rescale_points
from utils.refinement import upsample_mask, init_crf, crf_refine


class INSID3(nn.Module):
    """Training-free in-context segmentation using a frozen DINOv3 encoder."""

    def __init__(
        self,
        encoder: nn.Module,
        image_size: int = 1024,
        svd_components: int = 500,
        tau: float = 0.6,
        merge_threshold: float = 0.2,
        mask_refiner: str = "bilinear",
        resize_to_orig_size: bool = True,
        device: str = "cuda",
    ):
        super().__init__()
        self.encoder = encoder
        self.device = device
        self.image_size = image_size
        self.svd_components = svd_components
        self.tau = tau
        self.merge_threshold = merge_threshold
        self.mask_refiner = mask_refiner
        self.resize_to_orig_size = resize_to_orig_size

        self.positional_basis = self._build_positional_basis(self.device)

        if mask_refiner == 'crf':
            self._crf, self._crf_band_px, self._crf_p_core = init_crf(image_size, self.device)

        self._transform = build_transform(image_size)
        self.reset_state()

    def reset_state(self) -> None:
        """Clear cached reference/target inputs and their original sizes."""
        self._ref_images = None
        self._ref_masks = None
        self._tgt_image = None
        self._orig_tgt_size = None
        self._orig_ref_size = None

    # ──────── State setup ────────

    def set_reference(self, image: str | Image.Image | torch.Tensor,
                      mask: str | Image.Image | torch.Tensor | None = None) -> None:
        """Set reference image and optional mask from file paths, PIL Images, or tensors.

        Args:
            image: path (str), PIL Image, or tensor (*,C,H,W) normalized as the model expects.
            mask: path (str), PIL Image, tensor (*, H, W), or None for matching-only use.
        """
        img_tensor, self._orig_ref_size = load_image(image, self._transform, self.device)

        mask_tensor = None
        if mask is not None:
            mask_tensor = load_mask(mask, self.image_size, self.device)

        if self._ref_images is None:
            self._ref_images = img_tensor
            self._ref_masks = mask_tensor
        else:
            self._ref_images = torch.cat([self._ref_images, img_tensor], dim=0)
            self._ref_masks = torch.cat([self._ref_masks, mask_tensor], dim=0)

    def set_target(self, image: str | Image.Image | torch.Tensor) -> None:
        """Set target image from a file path, PIL Image, or tensor."""
        img_tensor, self._orig_tgt_size = load_image(image, self._transform, self.device)
        self._tgt_image = img_tensor

    # ──────── Public API ────────

    def segment(self) -> torch.Tensor:
        """Run prediction using previously set reference(s) and target.

        Returns:
            pred_mask: (H, W) boolean mask.
        """
        if self._ref_images is None or self._ref_masks is None or self._tgt_image is None:
            raise RuntimeError('segment() requires reference image(s), reference mask(s), and a target image.')
        pred = self.predict_mask(self._ref_images, self._ref_masks, self._tgt_image)
        
        self.reset_state()
        return pred
    
    def match(self, src_kps: torch.Tensor, use_debiased: bool = True) -> torch.Tensor:
        """Match source keypoints using previously set reference and target.
        
        Returns:
            pred_kps: (N, 2) predicted keypoint coordinates in the target image.
        """
        if self._ref_images is None or self._tgt_image is None:
            raise RuntimeError('match() requires a reference image and a target image.')
        pred = self.predict_keypoint(self._ref_images, src_kps, self._tgt_image, use_debiased=use_debiased)

        self.reset_state()
        return pred
    
    # ──────── Inference: segmentation ────────

    @torch.no_grad()
    def predict_mask(self, ref_images: torch.Tensor, ref_masks: torch.Tensor, tgt_image: torch.Tensor) -> torch.Tensor:
        """Segment the target image given reference image(s) and mask(s).

        Args:
            ref_images: (S, C, H, W) reference image(s).
            ref_masks:  (S, H, W) binary reference mask(s).
            tgt_image:  (1, C, H, W) target image.

        Returns:
            pred_mask: (H, W) boolean mask.
        """
        S = ref_images.shape[0]
        imgs = torch.cat([ref_images, tgt_image], dim=0).unsqueeze(0)

        # Feature extraction
        fmaps = self._extract_features(imgs)
        fmaps_norm = F.normalize(fmaps, p=2, dim=2)
        _, _, C, h, w = fmaps_norm.shape

        ref_masks = ref_masks.unsqueeze(1)
        feat_tgt = fmaps_norm[:, S]

        # Positional debiasing
        fmaps_debiased = self._debias_features(fmaps_norm)
        feat_refs_deb = fmaps_debiased[:, :S]
        feat_tgt_deb = fmaps_debiased[:, S]

        # Reference prototype (averaged across shots)
        ref_prototypes = []
        for s in range(S):
            mask_s = downsample_mask(ref_masks[s:s+1], h, w)
            fg = feat_refs_deb[0, s, :, mask_s]
            if fg.shape[1] > 0:
                ref_prototypes.append(fg.mean(dim=1))
        ref_prototype = F.normalize(
            torch.stack(ref_prototypes).mean(dim=0), p=2, dim=0
        ).unsqueeze(1)

        # Candidate localization (forward + backward matching)
        # Compute similarity maps between each reference and the target (debiased space)
        sim_maps = []
        for m in range(S):
            feat_ref_m = feat_refs_deb[:, m]
            sim_m = torch.einsum('bchw,bcxy->bhwxy', feat_ref_m, feat_tgt_deb)
            sim_maps.append(sim_m)
        candidate_mask = self._locate_candidates(
            sim_maps, ref_masks, feat_tgt_deb, ref_prototype, h, w
        )
        if candidate_mask.sum() == 0:
            return self._finalize_mask(candidate_mask, tgt_image)

        # Fine-grained clustering
        feat_tgt_flat = feat_tgt[0].reshape(C, -1).permute(1, 0)
        cluster_labels = agglomerative_clustering(feat_tgt_flat, self.tau).reshape(h, w)
        K = int(cluster_labels.max().item()) + 1

        feat_tgt_deb_flat = feat_tgt_deb[0].reshape(C, -1).permute(1, 0)
        cluster_protos = compute_cluster_prototypes(
            feat_tgt_deb_flat, cluster_labels.view(-1), K
        )

        # Seed selection and cluster aggregation
        pred_mask = self._seed_and_aggregate(
            candidate_mask, cluster_labels, cluster_protos, K,
            ref_prototype, feat_tgt, feat_tgt_deb, h, w
        )

        return self._finalize_mask(pred_mask, tgt_image)

    # ──────── Inference: semantic correspondence ────────

    @torch.no_grad()
    def predict_keypoint(self, ref_image: torch.Tensor, ref_kps: torch.Tensor, tgt_image: torch.Tensor, use_debiased: bool = False) -> torch.Tensor:
        """Predict target keypoints given a reference image and keypoints.

        Args:
            ref_image: (1, C, H, W) reference image.
            ref_kps: (K, 2) reference keypoints in original reference-image pixels.
            tgt_image: (1, C, H, W) target image.
            use_debiased: Whether to use debiased features for matching.

        Returns:
            pred_keypoints: (K, 2) predicted target keypoints in original target-image pixels.
        """

        if not isinstance(ref_kps, torch.Tensor):
            ref_kps = torch.tensor(ref_kps, dtype=torch.float32, device=tgt_image.device)
        else:
            ref_kps = ref_kps.to(tgt_image.device).float()

        imgs = torch.cat([ref_image, tgt_image], dim=0).unsqueeze(0)
        fmaps_norm = F.normalize(self._extract_features(imgs), p=2, dim=2)
        if use_debiased:
            fmaps_norm = self._debias_features(fmaps_norm)

        feat_ref = fmaps_norm[0, 0]
        feat_tgt = fmaps_norm[0, 1]
        _, h, w = feat_ref.shape

        src_kps_feat = rescale_points(ref_kps, self._orig_ref_size, (h, w))
        src_xy = src_kps_feat.round().long()
        src_x = src_xy[..., 0].clamp_(0, w - 1)
        src_y = src_xy[..., 1].clamp_(0, h - 1)
        src_desc = feat_ref[:, src_y, src_x].transpose(0, 1)
        sim_map = (src_desc @ feat_tgt.flatten(1)).view(1, -1, h, w)
        matches = kernel_softargmax_get_matches_logits(sim_map, 0.04, 7)
        return rescale_points(matches, (h, w), self._orig_tgt_size)[0]
    
    # ──────── Feature extraction ────────

    def _extract_features(self, imgs: torch.Tensor) -> torch.Tensor:
        B, T = imgs.shape[:2]
        x = einops.rearrange(imgs, 'b t c h w -> (b t) c h w')
        fmaps = self.encoder.get_intermediate_layers(x, n=1, reshape=True)[0]
        return einops.rearrange(fmaps, '(b t) c h w -> b t c h w', b=B)

    # ──────── Positional debiasing ────────

    @torch.no_grad()
    def _build_positional_basis(self, device: str) -> torch.Tensor:
        """Estimate the positional subspace from a noise image via SVD."""
        from torchvision.transforms.functional import normalize
        noise_img = normalize(
            torch.zeros(1, 3, self.image_size, self.image_size),
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
        ).to(device)
        noise_fmaps = self.encoder.to(device).get_intermediate_layers(
            noise_img, n=1, reshape=True
        )[0]
        noise_fmaps = F.normalize(noise_fmaps, p=2, dim=1)

        E = einops.rearrange(noise_fmaps, 'b c h w -> c (b h w)')
        E = E - E.mean(dim=1, keepdim=True)
        U, _, _ = torch.linalg.svd(E, full_matrices=False)
        return U[:, :self.svd_components].contiguous()

    def _debias_features(self, fmaps_norm: torch.Tensor) -> torch.Tensor:
        """Project features onto the orthogonal complement of the positional subspace."""
        B, T, C, H, W = fmaps_norm.shape
        X = fmaps_norm.reshape(B * T, C, H * W)

        basis = self.positional_basis.to(X.device)
        P_perp = torch.eye(C, device=X.device, dtype=X.dtype) - basis @ basis.T
        X_deb = torch.matmul(P_perp.unsqueeze(0), X).reshape(B, T, C, H, W)
        return F.normalize(X_deb, p=2, dim=2)

    # ──────── Candidate localization ────────

    def _locate_candidates(
        self,
        sim_maps: list,
        ref_masks: torch.Tensor,
        feat_tgt_deb: torch.Tensor,
        ref_prototype: torch.Tensor,
        h: int,
        w: int,
    ) -> torch.Tensor:
        """Find candidate target patches via forward and backward matching."""
        # Forward: positive similarity to aggregated reference prototype
        sim_fwd = torch.einsum('bchw,cd->bhw', feat_tgt_deb, ref_prototype).squeeze(0)
        forward_mask = sim_fwd > 0
        if forward_mask.sum() == 0:
            forward_mask = sim_fwd > float(torch.quantile(sim_fwd, 0.9))

        # Backward: majority-vote over nearest neighbours in each reference
        k = len(sim_maps)
        votes = torch.zeros((h, w), dtype=torch.int32, device=sim_maps[0].device)
        for m, sim_m in enumerate(sim_maps):
            sim0 = sim_m[0]  # (Hs, Ws, h, w)
            Hs, Ws = sim0.shape[:2]
            sim_t_to_r = sim0.permute(2, 3, 0, 1)  # (h, w, Hs, Ws)
            best_idx = sim_t_to_r.reshape(h, w, -1).argmax(dim=2)  # (h, w)
            rows = best_idx // Ws
            cols = best_idx % Ws
            ref_mask_m = downsample_mask(ref_masks[m:m+1], Hs, Ws).squeeze(0)  # (Hs, Ws)
            votes += ref_mask_m[rows, cols].to(torch.int32)

        majority_thresh = math.ceil(k / 2)
        backward_mask = votes >= majority_thresh

        return forward_mask & backward_mask

    # ──────── Seed selection and cluster aggregation ────────

    def _seed_and_aggregate(
        self,
        candidate_mask: torch.Tensor,
        cluster_labels: torch.Tensor,
        cluster_protos: torch.Tensor,
        K: int,
        ref_prototype: torch.Tensor,
        feat_tgt: torch.Tensor,
        feat_tgt_deb: torch.Tensor,
        h: int,
        w: int,
    ) -> torch.Tensor:
        """Select the seed cluster and aggregate remaining clusters."""
        matched_mask = candidate_mask & (cluster_labels >= 0)
        if matched_mask.sum() == 0:
            return candidate_mask

        matched_ids, n_pixels = cluster_labels[matched_mask].unique(return_counts=True)

        # Area weighting
        all_areas = cluster_labels[cluster_labels >= 0].unique(return_counts=True)[1]
        per_cluster = torch.zeros(K, device=cluster_labels.device)
        per_cluster[matched_ids] = n_pixels.float()
        area_weights = per_cluster / all_areas

        # Seed selection: cluster with highest cross-image similarity
        protos_matched = cluster_protos[matched_ids]
        cross_sim_matched = (ref_prototype.T @ protos_matched.T).squeeze(0)
        seed_idx = int(torch.argmax(cross_sim_matched).item())
        seed_id = matched_ids[seed_idx].item()

        # Intra-image similarity to seed (original feature space)
        feat_tgt_flat = feat_tgt[0].reshape(feat_tgt.shape[1], -1).permute(1, 0)
        orig_protos = compute_cluster_prototypes(
            feat_tgt_flat, cluster_labels.view(-1), K
        )
        intra_sim = torch.einsum('c,kc->k', orig_protos[seed_id], orig_protos)

        # Cross-image similarity per cluster (debiased space)
        fg_sim = torch.einsum('bchw,cd->bhw', feat_tgt_deb, ref_prototype).squeeze(0)
        cross_sim = torch.empty(K, device=fg_sim.device, dtype=fg_sim.dtype)
        for k in range(K):
            idx = (cluster_labels == k)
            cross_sim[k] = fg_sim[idx].mean() if idx.any() else 0.0

        # Combined score
        combined = cross_sim * intra_sim
        area_weights[seed_id] = 1.0
        combined *= area_weights

        final_mask = torch.zeros(h, w, dtype=torch.bool, device=cluster_labels.device)
        valid = cluster_labels >= 0
        final_mask[valid] = combined[cluster_labels[valid]] > self.merge_threshold
        return final_mask

    # ──────── Mask finalization ────────

    def _finalize_mask(self, mask: torch.Tensor, tgt_image: torch.Tensor) -> torch.Tensor:
        """Upsample feature-resolution mask, optionally with CRF refinement."""
        H, W = tgt_image.shape[-2:]
        up = upsample_mask(mask, H, W)
        if self.mask_refiner == 'crf':
            up = crf_refine(self._crf, self._crf_band_px, self._crf_p_core, tgt_image, up)
        # Resize to original target resolution
        if self.resize_to_orig_size:
            up = upsample_mask(up, self._orig_tgt_size[0], self._orig_tgt_size[1])
        return up
