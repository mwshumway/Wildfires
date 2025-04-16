import numpy as np

def weno5_left(f):
    eps = 1e-6
    v0, v1, v2, v3, v4 = f[:-4], f[1:-3], f[2:-2], f[3:-1], f[4:]

    beta0 = (13/12)*(v0 - 2*v1 + v2)**2 + (1/4)*(v0 - 4*v1 + 3*v2)**2
    beta1 = (13/12)*(v1 - 2*v2 + v3)**2 + (1/4)*(v1 - v3)**2
    beta2 = (13/12)*(v2 - 2*v3 + v4)**2 + (1/4)*(3*v2 - 4*v3 + v4)**2

    d0, d1, d2 = 0.1, 0.6, 0.3
    alpha0 = d0 / (eps + beta0)**2
    alpha1 = d1 / (eps + beta1)**2
    alpha2 = d2 / (eps + beta2)**2
    alpha_sum = alpha0 + alpha1 + alpha2

    w0 = alpha0 / alpha_sum
    w1 = alpha1 / alpha_sum
    w2 = alpha2 / alpha_sum

    f0 = (1/3)*v0 - (7/6)*v1 + (11/6)*v2
    f1 = -(1/6)*v1 + (5/6)*v2 + (1/3)*v3
    f2 = (1/3)*v2 + (5/6)*v3 - (1/6)*v4

    return w0*f0 + w1*f1 + w2*f2

def weno5_right(f):
    return weno5_left(f[::-1])[::-1]

def convection_weno2D(v, wx, wy, dx, dy):
    """
    Computes convection term (w · ∇v) using WENO5 and Lax-Friedrichs splitting,
    with zero-flux (Neumann) boundary conditions via symmetric padding.
    """
    Nx, Ny = v.shape
    conv = np.zeros_like(v)

    pad = 3  # WENO5 needs 5 points per stencil => pad 3

    # Symmetric padding for Neumann BCs (zero normal gradient)
    vp  = np.pad(v, pad_width=pad, mode='reflect')
    wxp = np.pad(wx, pad_width=pad, mode='reflect')
    wyp = np.pad(wy, pad_width=pad, mode='reflect')

    # x-direction (loop over rows)
    for j in range(pad, pad + Ny):
        v_line = vp[:, j]
        w_line = wxp[:, j]
        alpha = np.max(np.abs(w_line))

        f_plus  = 0.5 * (w_line * v_line + alpha * v_line)
        f_minus = 0.5 * (w_line * v_line - alpha * v_line)

        f_hat = weno5_left(f_plus) + weno5_right(f_minus)
        df_dx = (f_hat[1:] - f_hat[:-1]) / dx
        conv[:, j - pad] += df_dx[pad - 1 : -(pad - 1)]

    # y-direction (loop over columns)
    for i in range(pad, pad + Nx):
        v_line = vp[i, :]
        w_line = wyp[i, :]
        alpha = np.max(np.abs(w_line))

        f_plus  = 0.5 * (w_line * v_line + alpha * v_line)
        f_minus = 0.5 * (w_line * v_line - alpha * v_line)

        f_hat = weno5_left(f_plus) + weno5_right(f_minus)
        df_dy = (f_hat[1:] - f_hat[:-1]) / dy
        conv[i - pad, :] += df_dy[pad - 1 : -(pad - 1)]

    return conv
