import gc
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import onnxruntime
import soundfile as sf
import torch
from pydub import AudioSegment

from STFT_Process import STFT_Process
from modeling_modified.hgtcrn_optimized import GTCRN_IVA


model_path          = r"/home/DakeQQ/Downloads/H-GTCRN-main"                        # The H-GTCRN download path.
onnx_model_A        = r"/home/DakeQQ/Downloads/H_GTCRN_ONNX/H_GTCRN.onnx"           # The exported onnx model path.
test_noisy_audio    = r"./example/Samples2_noisy.wav"                               # The noisy audio path.
save_denoised_audio = r"./denoised.wav"                                             # The output denoised audio path.


DYNAMIC_AXES       = False                          # False exports a fixed windowed model; set True to keep dynamic audio length so WPE/AuxIVA can use full-sequence statistics.
IN_SAMPLE_RATE     = 16000                          # [8000, 16000, 22500, 24000, 44000, 48000]; It accepts various sample rates as input.
OUT_SAMPLE_RATE    = 16000                          # [8000, 16000, 22500, 24000, 44000, 48000]; It accepts various sample rates as input.
INPUT_AUDIO_LENGTH = 32000                          # Dummy export length when dynamic axes are enabled. Keep it as an integer multiple of HOP_LENGTH.
MAX_SIGNAL_LENGTH  = 4096 if DYNAMIC_AXES else 128  # Max frames for audio length after STFT processed. Set an appropriate larger value for long audio input, such as 4096.
WINDOW_TYPE        = 'hann'                         # Type of window function used in the STFT (matches original H-GTCRN).
PAD_MODE           = 'constant'                     # ['constant', 'reflect'] Match torch.stft default padding in the original repo.
NFFT               = 512                            # Number of FFT components for the STFT process.
WINDOW_LENGTH      = 512                            # Length of windowing, edit it carefully.
HOP_LENGTH         = 256                            # Number of samples between successive frames in the STFT.
N_CHANNELS         = 2                              # Number of input microphone channels.
WPE_RT60           = 0.3                            # WPE reverberation time parameter.
WPE_DELAY          = 2                              # WPE prediction delay parameter.
WPE_ITER           = 1                              # WPE number of iterations.
IVA_ITER           = 10                             # AuxIVA number of iterations (must match training: 10 iterations for proper source separation).
CG_SOLVE_ITER      = 6                              # Inner CG steps for the WPE linear solve.

NORMALIZE_AUDIO    = False                          # Normalize the input audio to a target RMS level (e.g., 8192) before processing. It can help improve the performance of the model, especially for low-volume audio. Set it to True if you want to enable it.
MAX_THREADS        = 4                              # Number of parallel threads for test audio denoising.
ORT_Accelerate_Providers = []                       # If you have accelerated devices for : ['CUDAExecutionProvider', 'TensorrtExecutionProvider', 'DmlExecutionProvider', 'OpenVINOExecutionProvider', 'MIGraphXExecutionProvider']
                                                    # else keep empty.


def normalise_audio(audio: np.ndarray, target_rms: float = 8192.0) -> np.ndarray:
    _audio = audio.astype(np.float32)
    rms = np.sqrt(np.mean(_audio * _audio, dtype=np.float32), dtype=np.float32)
    if rms > 0:
        _audio *= (target_rms / (rms + 1e-12))
        np.clip(_audio, -32768.0, 32767.0, out=_audio)
        return _audio.astype(np.int16)
    else:
        return audio


def pad_audio_tail_with_context(audio: np.ndarray, target_length: int) -> np.ndarray:
    current_length = audio.shape[-1]
    if current_length >= target_length:
        return audio

    pad_amount = target_length - current_length
    if current_length == 0:
        padding = np.zeros((*audio.shape[:-1], pad_amount), dtype=audio.dtype)
    elif current_length == 1:
        padding = np.repeat(audio[..., -1:], pad_amount, axis=-1)
    else:
        padding = np.pad(audio, ((0, 0), (0, 0), (0, pad_amount)), mode='reflect')[..., current_length:]

    return np.concatenate((audio, padding.astype(audio.dtype, copy=False)), axis=-1)


# ═══════════════════════════════════════════════════════════════════════════
# ONNX-friendly complex arithmetic helpers (real-valued representation)
# Complex tensors are represented as separate real/imag tensors or (..., 2) pairs
# ═══════════════════════════════════════════════════════════════════════════

def batched_complex_solve_cg(R_r, R_i, P_r, P_i, n_iter=36):
    """
    Solve R @ G = P for G using Conjugate Gradient (CG).
    ONNX-friendly: uses only matmul, element-wise ops, and reductions.
    Guaranteed to converge for Hermitian positive definite R.

    R: (B, F, N, N) complex Hermitian positive definite (R_r + j*R_i)
    P: (B, F, N, M) complex (P_r + j*P_i)
    Returns: G_r, G_i of same shape as P
    """
    # Initialize: x=0, r=P, p=r
    x_r = torch.zeros_like(P_r)
    x_i = torch.zeros_like(P_i)
    r_r = P_r.clone()
    r_i = P_i.clone()
    p_r = P_r.clone()
    p_i = P_i.clone()

    # rr = sum_n |r_n|^2 per (B, F, M) = r^H @ r per column
    rr = (r_r * r_r + r_i * r_i).sum(dim=-2) + 1e-12  # (B, F, M)

    for _ in range(n_iter):
        # Ap = R @ p (complex matmul)
        Ap_r = torch.matmul(R_r, p_r) - torch.matmul(R_i, p_i)  # (B, F, N, M)
        Ap_i = torch.matmul(R_r, p_i) + torch.matmul(R_i, p_r)

        # pAp = p^H @ Ap = sum_n conj(p_n)*Ap_n per column (real for HPD R)
        # For HPD R, p^H R p is real and positive
        pAp = (p_r * Ap_r + p_i * Ap_i).sum(dim=-2) + 1e-12  # (B, F, M)

        # alpha = rr / pAp
        alpha = rr / pAp # (B, F, M)
        alpha_expanded = alpha.unsqueeze(-2)  # (B, F, 1, M)

        # x = x + alpha * p
        x_r = x_r + alpha_expanded * p_r
        x_i = x_i + alpha_expanded * p_i

        # r = r - alpha * Ap
        r_r = r_r - alpha_expanded * Ap_r
        r_i = r_i - alpha_expanded * Ap_i

        # rr_new = r^H @ r
        rr_new = (r_r * r_r + r_i * r_i).sum(dim=-2) + 1e-12  # (B, F, M)

        # beta = rr_new / rr
        beta = rr_new / rr  # (B, F, M)
        beta_expanded = beta.unsqueeze(-2)  # (B, F, 1, M)

        # p = r + beta * p
        p_r = r_r + beta_expanded * p_r
        p_i = r_i + beta_expanded * p_i

        rr = rr_new

    return x_r, x_i


def solve_2x2_complex(A, b):
    """
    Solve A @ x = b for x, where A is (..., 2, 2, 2) complex and b is (..., 2, 2) complex.
    Uses Cramer's rule for 2x2 system. ONNX-friendly (no linalg ops).

    A: (..., 2, 2, 2) - 2x2 complex matrix (last dim is real/imag)
    b: (..., 2, 2)    - 2-element complex vector (last dim is real/imag)
    Returns: x (..., 2, 2) complex vector
    """
    # A = [[a, b], [c, d]], det = ad - bc
    A_row0, A_row1 = A.split(1, dim=-3)  # each (..., 1, 2, 2)
    a, b_mat = A_row0.squeeze(-3).split(1, dim=-2)  # each (..., 1, 2)
    a, b_mat = a.squeeze(-2), b_mat.squeeze(-2)  # each (..., 2)
    c, d = A_row1.squeeze(-3).split(1, dim=-2)  # each (..., 1, 2)
    c, d = c.squeeze(-2), d.squeeze(-2)  # each (..., 2)

    # Split all complex pairs into real/imag once to avoid repeated gather
    a_r, a_i = a.split(1, dim=-1)      # each (..., 1)
    b_mat_r, b_mat_i = b_mat.split(1, dim=-1)
    c_r, c_i = c.split(1, dim=-1)
    d_r, d_i = d.split(1, dim=-1)

    # det = a*d - b*c (complex)
    ad = torch.cat([a_r * d_r - a_i * d_i, a_r * d_i + a_i * d_r], dim=-1)
    bc = torch.cat([b_mat_r * c_r - b_mat_i * c_i, b_mat_r * c_i + b_mat_i * c_r], dim=-1)
    det = ad - bc  # (..., 2)

    # inv_det = conj(det) / |det|^2
    det_r, det_i = det.split(1, dim=-1)
    det_abs_sq = 1.0 / ((det_r ** 2 + det_i ** 2) + 1e-12)
    inv_det_r = det_r * det_abs_sq   # (..., 1)
    inv_det_i = -det_i * det_abs_sq  # (..., 1)

    # x0 = (d * b[0] - b_mat * b[1]) * inv_det
    # x1 = (a * b[1] - c * b[0]) * inv_det
    b0, b1 = b.split(1, dim=-2)  # each (..., 1, 2)
    b0, b1 = b0.squeeze(-2), b1.squeeze(-2)  # each (..., 2)
    b0_r, b0_i = b0.split(1, dim=-1)
    b1_r, b1_i = b1.split(1, dim=-1)

    # num0 = d * b0 - b_mat * b1
    num0_r = (d_r * b0_r - d_i * b0_i) - (b_mat_r * b1_r - b_mat_i * b1_i)
    num0_i = (d_r * b0_i + d_i * b0_r) - (b_mat_r * b1_i + b_mat_i * b1_r)

    # num1 = a * b1 - c * b0
    num1_r = (a_r * b1_r - a_i * b1_i) - (c_r * b0_r - c_i * b0_i)
    num1_i = (a_r * b1_i + a_i * b1_r) - (c_r * b0_i + c_i * b0_r)

    # Multiply by inv_det
    x0 = torch.cat([num0_r * inv_det_r - num0_i * inv_det_i,
                    num0_r * inv_det_i + num0_i * inv_det_r], dim=-1)
    x1 = torch.cat([num1_r * inv_det_r - num1_i * inv_det_i,
                    num1_r * inv_det_i + num1_i * inv_det_r], dim=-1)

    return torch.stack([x0, x1], dim=-2)  # (..., 2, 2)


class OnnxFriendlyWPE(torch.nn.Module):
    """
    ONNX-exportable Weighted Prediction Error (WPE) dereverberation.
    Replaces torch.linalg.inv with complex Conjugate Gradient.

    Optimizations:
      - Pre-compute: eye matrix and delay templates registered as buffers
      - Hoist loop-invariant: Xp transpose computed once outside iteration loop
    """
    def __init__(self, n_channels=2, rt60=0.3, hop_length=256, delay=2, sample_rate=16000, num_iter=1, ns_iter=36,
         n_freq_bins=NFFT // 2 + 1, max_frames=MAX_SIGNAL_LENGTH):
        super().__init__()
        self.M = n_channels
        self.Lg = int(rt60 * sample_rate / hop_length)
        self.D = delay
        self.num_iter = num_iter
        self.solve_iter = ns_iter
        self.MLg = self.M * self.Lg
        self.n_freq_bins = n_freq_bins
        self.max_frames = max_frames
        # Pre-compute identity matrix as buffer (avoids runtime torch.eye allocation)
        self.register_buffer('eye_MLg', torch.eye(self.MLg, dtype=torch.float32))
        self.register_buffer(
            'delay_template_r',
            torch.zeros(1, n_freq_bins, self.MLg, max_frames, dtype=torch.float32),
        )
        self.register_buffer(
            'delay_template_i',
            torch.zeros(1, n_freq_bins, self.MLg, max_frames, dtype=torch.float32),
        )

    def forward(self, X_real, X_imag):
        """
        X_real, X_imag: (B, M, F, T) — multi-channel STFT real/imag parts.
        Returns: Y_real, Y_imag of same shape — dereverberated.
        """
        _, M, F, T = X_real.shape
        # Permute to (B, F, M, T)
        Xp_r = X_real.permute(0, 2, 1, 3)  # (B, F, M, T)
        Xp_i = X_imag.permute(0, 2, 1, 3)

        # Build delay matrix: (B, F, M*Lg, T)
        if F != self.n_freq_bins or T > self.max_frames:
            raise ValueError(
                f"WPE delay buffers require F={self.n_freq_bins} and T<={self.max_frames}, got F={F}, T={T}."
            )

        MLg = self.MLg
        X_delay_r = self.delay_template_r[:, :, :, :T].clone()
        X_delay_i = self.delay_template_i[:, :, :, :T].clone()

        for l_idx in range(self.Lg):
            start_col = self.D + l_idx
            row_start = l_idx * M
            row_end = row_start + M
            X_delay_r[:, :, row_start:row_end, start_col:] = Xp_r[:, :, :, :-start_col]
            X_delay_i[:, :, row_start:row_end, start_col:] = Xp_i[:, :, :, :-start_col]

        # Compute eps matching original: 1e-3 * mean(max_per_batch(|X|^2))
        mag_sq = Xp_r * Xp_r + Xp_i * Xp_i  # (B, F, M, T)
        eps_val = 1e-3 * mag_sq.amax(dim=(-2, -1)).mean()

        # Y = Xp initially
        Y_r = Xp_r.clone()  # (B, F, M, T)
        Y_i = Xp_i.clone()

        # Hoist loop-invariant: Xp transposed doesn't change across iterations
        Xp_rT = Xp_r.transpose(-2, -1)  # (B, F, T, M)
        Xp_iT = Xp_i.transpose(-2, -1)

        for _ in range(self.num_iter):
            # lambda = mean(|Y|^2, dim=channels), shape (B, F, 1, T)
            Y_pow = (Y_r * Y_r + Y_i * Y_i).mean(dim=2, keepdim=True).clamp(min=eps_val)  # (B, F, 1, T)

            # temp = X_delay / lambda: (B, F, MLg, T)
            inv_lambda = 1.0 / Y_pow  # (B, F, 1, T)
            temp_r = X_delay_r * inv_lambda
            temp_i = X_delay_i * inv_lambda

            # R = temp @ conj(X_delay)^H: (B, F, MLg, MLg)
            Xd_rT = X_delay_r.transpose(-2, -1)  # (B, F, T, MLg)
            Xd_iT = X_delay_i.transpose(-2, -1)
            R_real = torch.matmul(temp_r, Xd_rT) + torch.matmul(temp_i, Xd_iT)
            R_imag = torch.matmul(temp_i, Xd_rT) - torch.matmul(temp_r, Xd_iT)

            # P = temp @ conj(Xp)^H: (B, F, MLg, M)
            P_real = torch.matmul(temp_r, Xp_rT) + torch.matmul(temp_i, Xp_iT)
            P_imag = torch.matmul(temp_i, Xp_rT) - torch.matmul(temp_r, Xp_iT)

            # Add eps * I to R for regularization (use pre-computed buffer)
            R_real = R_real + eps_val * self.eye_MLg

            # Solve R @ G = P using Conjugate Gradient.
            G_r, G_i = batched_complex_solve_cg(
                R_real,
                R_imag,
                P_real,
                P_imag,
                n_iter=self.solve_iter,
            )

            # Y = Xp - conj(G)^T @ X_delay
            G_conj_T_real = G_r.transpose(-2, -1)   # (B, F, M, MLg)
            G_conj_T_imag = -G_i.transpose(-2, -1)  # (B, F, M, MLg)

            pred_r = torch.matmul(G_conj_T_real, X_delay_r) - torch.matmul(G_conj_T_imag, X_delay_i)
            pred_i = torch.matmul(G_conj_T_imag, X_delay_r) + torch.matmul(G_conj_T_real, X_delay_i)

            Y_r = Xp_r - pred_r
            Y_i = Xp_i - pred_i

        # Permute back to (B, M, F, T)
        return Y_r.permute(0, 2, 1, 3), Y_i.permute(0, 2, 1, 3)


class OnnxFriendlyAuxIVA(torch.nn.Module):
    """
    ONNX-exportable AuxIVA source separation for 2-channel input.
    Replaces torch.linalg.solve with analytical 2x2 complex solve.

    Optimizations:
      - Pre-compute: eye_M, e_s unit vectors registered as buffers
      - Hoist: X transpose computed once outside iteration loop
      - Use split instead of slice for source channel extraction
    """
    def __init__(self, n_iter=10, n_channels=2, n_freq_bins=NFFT // 2 + 1):
        super().__init__()
        self.n_iter = n_iter
        self.M = n_channels
        self.n_freq_bins = n_freq_bins
        # Pre-computed constants as buffers
        eye_M = torch.eye(n_channels, dtype=torch.float32)
        self.register_buffer('eye_M', eye_M)
        self.register_buffer('proj_back_one', torch.ones(1, 1, dtype=torch.float32))
        self.register_buffer('proj_back_zero', torch.zeros(1, 1, dtype=torch.float32))
        # Pre-compute e_s unit vectors across the fixed frequency axis.
        e_s_all = torch.zeros(n_channels, n_freq_bins, n_channels, 2)
        for s in range(n_channels):
            e_s_all[s, :, s, 0] = 1.0
        self.register_buffer('e_s', e_s_all)
        self.eps = 1e-10
        self.register_buffer(
            'eps_eye',
            (self.eps * eye_M).view(1, 1, n_channels, n_channels).expand(1, n_freq_bins, n_channels, n_channels).clone(),
        )
        self.register_buffer(
            'init_W_r',
            eye_M.view(1, 1, n_channels, n_channels).expand(1, n_freq_bins, n_channels, n_channels).clone(),
        )
        self.register_buffer(
            'init_W_i',
            torch.zeros(1, n_freq_bins, n_channels, n_channels, dtype=torch.float32),
        )

    def forward(self, X_real, X_imag):
        """
        X_real, X_imag: (B, M=2, F, T) — dereverberated STFT.
        Returns: Y_real, Y_imag (B, M=2, F, T) — separated sources.
        """
        _, M, F, T = X_real.shape
        inv_T = 1.0 / T

        # Reshape to (B, F, M, T) for processing
        X_r = X_real.permute(0, 2, 1, 3)  # (B, F, M, T)
        X_i = X_imag.permute(0, 2, 1, 3)

        # Hoist loop-invariant: X transposed (constant across all iterations)
        X_rT = X_r.transpose(-2, -1)  # (B, F, T, M)
        X_iT = X_i.transpose(-2, -1)

        # Initialize W as identity: (B, F, M, M)
        W_r = self.init_W_r.clone()
        W_i = self.init_W_i.clone()

        # Y = W @ X (W starts as identity, so Y = X initially)
        Y_r = X_r.clone()
        Y_i = X_i.clone()

        for iter_idx in range(self.n_iter):
            # r = 2 * L2_norm(Y over F): (B, M, T)
            Y_pow = Y_r * Y_r + Y_i * Y_i  # (B, F, M, T)
            r = 2.0 * torch.sqrt(Y_pow.sum(dim=1) + self.eps)  # (B, M, T)
            r_inv = 1.0 / r  # (B, M, T)

            for s in range(M):
                # r_inv for source s: (B, 1, 1, T)
                w_s = r_inv[:, s:s+1, :].unsqueeze(1)  # (B, 1, 1, T)

                # weighted X: (B, F, M, T)
                wX_r = X_r * w_s
                wX_i = X_i * w_s

                # V = wX @ conj(X)^H / T: (B, F, M, M) complex
                V_r = (torch.matmul(wX_r, X_rT) + torch.matmul(wX_i, X_iT)) * inv_T
                V_i = (torch.matmul(wX_i, X_rT) - torch.matmul(wX_r, X_iT)) * inv_T

                # On the very first source-step, W starts as the identity and W_i is zero.
                if iter_idx == 0 and s == 0:
                    WV_r = V_r
                    WV_i = V_i
                else:
                    # WV = W @ V: (B, F, M, M)
                    WV_r = torch.matmul(W_r, V_r) - torch.matmul(W_i, V_i)
                    WV_i = torch.matmul(W_r, V_i) + torch.matmul(W_i, V_r)

                # Solve (WV + eps*I) @ w_new = e_s using pre-computed buffers
                WV_r_reg = WV_r + self.eps_eye

                # Stack WV as (B, F, 2, 2, 2) for solve_2x2_complex
                A_solve = torch.stack([WV_r_reg, WV_i], dim=-1)  # (B, F, M, M, 2)
                w_new = solve_2x2_complex(A_solve, self.e_s[s])  # (B, F, M, 2)

                # w_new split into real/imag
                w_new_r, w_new_i = w_new.split(1, dim=-1)  # each (B, F, M, 1)
                w_new_r = w_new_r.squeeze(-1)  # (B, F, M)
                w_new_i = w_new_i.squeeze(-1)

                # W[s] = conj(w_new)
                conj_w_r = w_new_r   # (B, F, M)
                conj_w_i = -w_new_i

                # Normalize: denom = conj(w) @ V @ w
                wn_r_col = w_new_r.unsqueeze(-1)  # (B, F, M, 1)
                wn_i_col = w_new_i.unsqueeze(-1)
                Vw_r = torch.matmul(V_r, wn_r_col) - torch.matmul(V_i, wn_i_col)  # (B, F, M, 1)
                Vw_i = torch.matmul(V_r, wn_i_col) + torch.matmul(V_i, wn_r_col)

                denom_r = (conj_w_r * Vw_r.squeeze(-1) - conj_w_i * Vw_i.squeeze(-1)).sum(dim=-1)

                # For HPD V, w^H V w is real and non-negative; normalizing with the
                # real scalar avoids phase drift from a noisy complex sqrt.
                norm_scale = torch.rsqrt(denom_r.clamp(min=0.0) + self.eps).unsqueeze(-1)
                final_r = conj_w_r * norm_scale
                final_i = conj_w_i * norm_scale

                # Update W[s]
                W_r[:, :, s, :] = final_r
                W_i[:, :, s, :] = final_i

            # Recompute Y = W @ X
            Y_r = torch.matmul(W_r, X_r) - torch.matmul(W_i, X_i)
            Y_i = torch.matmul(W_r, X_i) + torch.matmul(W_i, X_r)

        # Projection back to align with reference channel (channel 0)
        ref_r = X_r[:, :, 0, :]  # (B, F, T)
        ref_i = X_i[:, :, 0, :]

        # Use split for channel extraction instead of indexing + zeros_like
        Y_s_list_r = Y_r.split(1, dim=2)  # list of (B, F, 1, T)
        Y_s_list_i = Y_i.split(1, dim=2)

        out_r_list = []
        out_i_list = []
        for s in range(M):
            Ys_r = Y_s_list_r[s].squeeze(2)  # (B, F, T)
            Ys_i = Y_s_list_i[s].squeeze(2)

            num_r = (ref_r * Ys_r + ref_i * Ys_i).sum(dim=-1)
            num_i = (ref_r * Ys_i - ref_i * Ys_r).sum(dim=-1)
            denom = (Ys_r * Ys_r + Ys_i * Ys_i).sum(dim=-1)
            valid = denom > 0.0
            safe_denom = 1.0 / torch.where(valid, denom, self.proj_back_one)

            c_r = torch.where(valid, num_r * safe_denom, self.proj_back_one).unsqueeze(-1)  # (B, F, 1)
            c_i = torch.where(valid, num_i * safe_denom, self.proj_back_zero).unsqueeze(-1)

            out_r_list.append(c_r * Ys_r + c_i * Ys_i)
            out_i_list.append(c_r * Ys_i - c_i * Ys_r)

        Y_out_r = torch.stack(out_r_list, dim=2)
        Y_out_i = torch.stack(out_i_list, dim=2)

        return Y_out_r.permute(0, 2, 1, 3), Y_out_i.permute(0, 2, 1, 3)


class H_GTCRN_CUSTOM(torch.nn.Module):
    """
    Fully fused H-GTCRN pipeline for end-to-end ONNX export:
      int16 2-channel audio → STFT → WPE → AuxIVA → feature construction → GTCRN → CRM → iSTFT → int16 mono audio.

    All preprocessing (WPE, AuxIVA) is implemented with ONNX-friendly operators
    (no linalg.inv/solve — uses Jacobi iteration and analytical 2x2 Cramer's rule).

    Optimizations:
      - Weight fusion: BN already fused in GTCRN via fuse_bn_()
      - Scale fusion: inv_int16 / output_pcm_scale as registered buffers
      - Reduce dim changes: view instead of squeeze+unsqueeze; single transpose for feature layout
      - Use split instead of slice for channel extraction
      - Pre-compute: n_freq_bins constant avoids runtime .shape; log10 fused with sqrt
      - Fuse concat+transpose: build all 6 features in (F,T) then single transpose to (T,F)

    Input:  noisy_audio (1, N_CHANNELS, audio_len) int16
    Output: denoised_audio (1, 1, audio_len) int16
    """
    def __init__(self, gtcrn_core, stft_model, istft_model, wpe_module, iva_module, n_fft=512, in_sample_rate=16000, out_sample_rate=16000):
        super(H_GTCRN_CUSTOM, self).__init__()
        self.gtcrn = gtcrn_core
        self.stft_model = stft_model
        self.istft_model = istft_model
        self.wpe = wpe_module
        self.iva = iva_module
        # Pre-computed constants as buffers (no runtime computation)
        self.register_buffer('inv_int16', torch.tensor(1.0 / 32768.0))
        self.register_buffer('output_pcm_scale', torch.tensor(32767.0))
        self.in_sample_rate = in_sample_rate
        self.out_sample_rate = out_sample_rate
        self.in_sample_rate_scale = in_sample_rate / 16000.0
        self.out_sample_rate_scale = out_sample_rate / 16000.0
        self.model_rate_scale = 1.0 / self.in_sample_rate_scale
        self.resample_before_centering = self.in_sample_rate_scale > 1.0
        self.resample_after_centering = self.in_sample_rate_scale < 1.0
        self.output_resample_before_pcm = self.out_sample_rate_scale > 1.0
        self.output_resample_after_pcm = self.out_sample_rate_scale < 1.0
        self.n_freq_bins = n_fft // 2 + 1  # Avoid runtime .shape query for F dimension

    def forward(self, audio):
        """
        audio: (1, 2, audio_len) int16 — 2-channel raw audio
        """
        # ─── 1. Resample to the 16 kHz model rate and normalize ──────────
        audio_f = audio.float()  # (1, 2, L)
        if self.resample_before_centering:
            audio_f = torch.nn.functional.interpolate(
                audio_f,
                scale_factor=self.model_rate_scale,
                mode='linear',
                align_corners=False
            )
        audio_f = audio_f * self.inv_int16
        audio_f = audio_f - torch.mean(audio_f)
        if self.resample_after_centering:
            audio_f = torch.nn.functional.interpolate(
                audio_f,
                scale_factor=self.model_rate_scale,
                mode='linear',
                align_corners=False
            )

        # ─── 2. Multi-channel STFT via Conv1d ─────────────────────────────
        # view replaces squeeze(0)+unsqueeze(1): (1,2,L) → (2,1,L) in one op
        real_parts, imag_parts = self.stft_model(audio_f.view(2, 1, -1))  # each (2, F, T)

        # ─── 3. WPE dereverberation ──────────────────────────────────────
        # unsqueeze(0) adds batch dim: (2,F,T) → (1,2,F,T)
        drb_real, drb_imag = self.wpe(real_parts.unsqueeze(0), imag_parts.unsqueeze(0))

        # ─── 4. AuxIVA source separation ─────────────────────────────────
        iva_real, iva_imag = self.iva(drb_real, drb_imag)  # each (1, 2, F, T)

        # ─── 5. Channel selection via torch.where (no float mul) ──────────
        # Use split instead of slice for ONNX-friendly channel extraction
        iva_r0, iva_r1 = iva_real.split(1, dim=1)  # each (1, 1, F, T)
        iva_i0, iva_i1 = iva_imag.split(1, dim=1)
        energy = (iva_real * iva_real + iva_imag * iva_imag).sum(dim=(2, 3))  # (1, 2)
        # pred broadcast shape fused: use split instead of gather for ONNX-friendly extraction
        energy_0, energy_1 = energy.split(1, dim=1)  # each (1, 1)
        pred = (energy_0 < energy_1).view(1, 1, 1, 1)
        sel_real = torch.where(pred, iva_r0, iva_r1)      # (1, 1, F, T)
        sel_imag = torch.where(pred, iva_i0, iva_i1)
        unsel_real = torch.where(pred, iva_r1, iva_r0)
        unsel_imag = torch.where(pred, iva_i1, iva_i0)

        # ─── 6. Fused log-magnitude: log10(sqrt(x)) = 0.5*log10(x) ──────
        # Saves 2 sqrt ops; clamp adjusted: (1e-6)^2 = 1e-12
        sel_log = 0.5 * torch.log10((sel_real * sel_real + sel_imag * sel_imag + 1e-12))
        unsel_log = 0.5 * torch.log10((unsel_real * unsel_real + unsel_imag * unsel_imag + 1e-12))

        # ─── 7. Feature construction: single stack+cat+transpose ─────────
        # Stack real/imag interleaved → (2,2,F,T) → view (1,4,F,T)
        # Ordering: [ch0_real, ch0_imag, ch1_real, ch1_imag] (matches original)
        spec_4ch = torch.stack([real_parts, imag_parts], dim=1).view(1, 4, self.n_freq_bins, -1)
        # Combine all 6 features in (F,T) layout, then single transpose to (T,F)
        spec_features = torch.cat([spec_4ch, sel_log, unsel_log], dim=1).transpose(-1, -2)  # (1, 6, T, F)

        # ─── 8. GTCRN network (ERB + Encoder + DPGRNN + Decoder + CRM) ───
        s_real, s_imag = self.gtcrn(spec_features)  # each (1, F, T)

        # ─── 9. iSTFT → time-domain audio ────────────────────────────────
        audio_out = self.istft_model(s_real, s_imag)  # (1, 1, audio_samples)

        # ─── 10. Resample output and scale to int16 PCM ──────────────────
        if self.output_resample_before_pcm:
            audio_out = torch.nn.functional.interpolate(
                audio_out,
                scale_factor=self.out_sample_rate_scale,
                mode='linear',
                align_corners=False
            )
        audio_out = audio_out * self.output_pcm_scale
        if self.output_resample_after_pcm:
            audio_out = torch.nn.functional.interpolate(
                audio_out,
                scale_factor=self.out_sample_rate_scale,
                mode='linear',
                align_corners=False
            )
        return audio_out.clamp(-32768.0, 32767.0).to(torch.int16)


print('Export start ...')
with torch.inference_mode():
    custom_stft = STFT_Process(model_type='stft_B', n_fft=NFFT, hop_len=HOP_LENGTH, win_length=WINDOW_LENGTH, max_frames=0, window_type=WINDOW_TYPE, center_pad=True, pad_mode=PAD_MODE).eval()
    custom_istft = STFT_Process(model_type='istft_B', n_fft=NFFT, hop_len=HOP_LENGTH, win_length=WINDOW_LENGTH, max_frames=MAX_SIGNAL_LENGTH, window_type=WINDOW_TYPE, center_pad=True, pad_mode=PAD_MODE).eval()
    wpe_module = OnnxFriendlyWPE(
        n_channels=N_CHANNELS,
        rt60=WPE_RT60,
        hop_length=HOP_LENGTH,
        delay=WPE_DELAY,
        sample_rate=IN_SAMPLE_RATE,
        num_iter=WPE_ITER,
        ns_iter=CG_SOLVE_ITER,
        n_freq_bins=NFFT // 2 + 1,
        max_frames=MAX_SIGNAL_LENGTH,
    ).eval()
    iva_module = OnnxFriendlyAuxIVA(n_iter=IVA_ITER, n_channels=N_CHANNELS).eval()
    gtcrn_iva = GTCRN_IVA().eval()
    ckpt = torch.load(model_path + "/checkpoints/best_model_0121.tar", map_location='cpu')
    gtcrn_iva.load_state_dict(ckpt['model'], strict=False)
    gtcrn_iva.fuse_bn_()  # Fuse BatchNorm into Conv weights for optimized inference
    model = H_GTCRN_CUSTOM(
        gtcrn_iva,
        custom_stft,
        custom_istft,
        wpe_module,
        iva_module,
        n_fft=NFFT,
        in_sample_rate=IN_SAMPLE_RATE,
        out_sample_rate=OUT_SAMPLE_RATE,
    ).eval()
    audio = torch.ones((1, N_CHANNELS, INPUT_AUDIO_LENGTH), dtype=torch.int16)
    os.makedirs(os.path.dirname(onnx_model_A), exist_ok=True)
    torch.onnx.export(
        model,
        (audio,),
        onnx_model_A,
        input_names=['noisy_audio'],
        output_names=['denoised_audio'],
        do_constant_folding=True,
        dynamic_axes={
            'noisy_audio': {2: 'audio_len'},
            'denoised_audio': {2: 'audio_len'}
        } if DYNAMIC_AXES else None,
        opset_version=17,
        dynamo=False
    )
    # If torch.onnx.export produced external data, re-save as a single self-contained file.
    import onnx
    data_file = os.path.basename(onnx_model_A) + ".data"
    data_path = os.path.join(os.path.dirname(onnx_model_A), data_file)
    if os.path.exists(data_path):
        onnx_model = onnx.load(onnx_model_A, load_external_data=True)
        for tensor in onnx_model.graph.initializer:
            if tensor.external_data:
                del tensor.external_data[:]
            tensor.data_location = onnx.TensorProto.DEFAULT
        onnx.save(onnx_model, onnx_model_A)
        os.remove(data_path)
        del onnx_model
    del model
    del gtcrn_iva
    del custom_stft
    del custom_istft
    del wpe_module
    del iva_module
    del audio
    gc.collect()
print('\nExport done!\n\nStart to run H-GTCRN by ONNX Runtime.\n\nNow, loading the model...')


# ONNX Runtime settings
session_opts = onnxruntime.SessionOptions()
session_opts.log_severity_level = 4         # Fatal level, it an adjustable value.
session_opts.inter_op_num_threads = 1       # Run different nodes with num_threads. Set 0 for auto.
session_opts.intra_op_num_threads = 1       # Under the node, execute the operators with num_threads. Set 0 for auto.
session_opts.enable_cpu_mem_arena = True    # True for execute speed; False for less memory usage.
session_opts.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
session_opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL  # ORT 1.26 bug: runtime constant folding creates tensors it cannot handle; torch already did constant folding at export.
session_opts.add_session_config_entry("session.intra_op.allow_spinning", "1")
session_opts.add_session_config_entry("session.inter_op.allow_spinning", "1")
session_opts.add_session_config_entry("session.set_denormal_as_zero", "1")


ort_session_A = onnxruntime.InferenceSession(onnx_model_A, sess_options=session_opts, providers=ORT_Accelerate_Providers)
print(f"\nUsable Providers: {ort_session_A.get_providers()}")
in_name_A = ort_session_A.get_inputs()
out_name_A = ort_session_A.get_outputs()
in_name_A0 = in_name_A[0].name
out_name_A0 = out_name_A[0].name


# Load the input audio (2-channel)
print(f"\nTest Input Audio: {test_noisy_audio}")
audio = np.array(AudioSegment.from_file(test_noisy_audio).set_channels(N_CHANNELS).set_frame_rate(IN_SAMPLE_RATE).get_array_of_samples(), dtype=np.int16)
audio = audio.reshape(-1, N_CHANNELS)
if NORMALIZE_AUDIO:
    audio_int16_ch0 = normalise_audio(audio[:, 0])
    audio_int16_ch1 = normalise_audio(audio[:, 1])
    audio = np.stack([audio_int16_ch0, audio_int16_ch1], axis=-1)
audio_len = audio.shape[0]

# Reshape audio to (1, N_CHANNELS, audio_len) for model input
audio = audio.T.reshape(1, N_CHANNELS, -1)

shape_value_in = ort_session_A._inputs_meta[0].shape[-1]
shape_value_out = ort_session_A._outputs_meta[0].shape[-1]
if isinstance(shape_value_in, str):
    INPUT_AUDIO_LENGTH = audio_len  # Preserve full-sequence WPE/AuxIVA statistics by default when the model is dynamic.
else:
    INPUT_AUDIO_LENGTH = shape_value_in
stride_step = INPUT_AUDIO_LENGTH

if audio_len > INPUT_AUDIO_LENGTH:
    if (shape_value_in != shape_value_out) & isinstance(shape_value_in, int) & isinstance(shape_value_out, int):
        stride_step = shape_value_out
    num_windows = int(np.ceil((audio_len - INPUT_AUDIO_LENGTH) / stride_step)) + 1
    total_length_needed = (num_windows - 1) * stride_step + INPUT_AUDIO_LENGTH
    audio = pad_audio_tail_with_context(audio, total_length_needed)
elif audio_len < INPUT_AUDIO_LENGTH:
    audio = pad_audio_tail_with_context(audio, INPUT_AUDIO_LENGTH)

aligned_len = audio.shape[-1]
inv_audio_len = float(100.0 / aligned_len)
output_audio_len = int(audio_len * OUT_SAMPLE_RATE / IN_SAMPLE_RATE)


def process_segment(_inv_audio_len, _slice_start, _slice_end, _audio):
    return _slice_start * _inv_audio_len, ort_session_A.run(
        [out_name_A0],
        {
            in_name_A0: _audio[:, :, _slice_start: _slice_end],
        }
    )[0]


# Start to run H-GTCRN
print("\nRunning the H-GTCRN by ONNX Runtime.")
results = []
start_time = time.time()
with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:  # Parallel denoised the audio.
    futures = []
    slice_start = 0
    slice_end = INPUT_AUDIO_LENGTH
    while slice_end <= aligned_len:
        futures.append(executor.submit(process_segment, inv_audio_len, slice_start, slice_end, audio))
        slice_start += stride_step
        slice_end = slice_start + INPUT_AUDIO_LENGTH
    for future in futures:
        results.append(future.result())
        print(f"Complete: {results[-1][0]:.3f}%")
results.sort(key=lambda x: x[0])
saved = [result[1] for result in results]
denoised_wav = np.concatenate(saved, axis=-1).reshape(-1)[:output_audio_len]
end_time = time.time()
elapsed = end_time - start_time
audio_duration = output_audio_len / OUT_SAMPLE_RATE if OUT_SAMPLE_RATE > 0 else 0.0
rtf = elapsed / audio_duration if audio_duration > 0 else 0.0
print(f"Complete: 100.00%")

# Save the denoised wav.
sf.write(save_denoised_audio, denoised_wav, OUT_SAMPLE_RATE, format='WAVEX')
print(f"\nDenoise Process Complete.\n\nSaving to: {save_denoised_audio}.\n\nTime Cost: {elapsed:.3f} Seconds\nAudio Duration: {audio_duration:.3f} Seconds\nRTF: {rtf:.4f}")
