# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ sysrem.py ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# tools for running SYSREM algorithm on clean data
# by Emily Deibert (upgraded)
# following Tamuz, Mazeh, & Zucker 2005, MNRAS, 356, 1466

import numpy as np

_LOG10 = np.log(10.0)
MAG_ERR_FACTOR = 2.5 / _LOG10  # 2.5 / ln(10)

# ---------------------------------------------------------------------
# Magnitude conversion
# ---------------------------------------------------------------------

def flux_to_mag(flux, err, min_flux=1e-12):
    """
    Convert flux to magnitudes and propagate errors.

    flux, err: arrays of same shape (n_exp, n_pix) or (n_ord, n_exp, n_pix)
    Returns:
      mag, magerr (same shape), with invalid entries as NaN.
    """
    flux = np.asarray(flux)
    err  = np.asarray(err)

    mag = np.full_like(flux, np.nan, dtype=np.float64)
    magerr = np.full_like(err, np.nan, dtype=np.float64)

    good = np.isfinite(flux) & np.isfinite(err) & (err > 0) & (flux > min_flux)
    # magnitude
    mag[good] = -2.5 * np.log10(flux[good])
    # magnitude uncertainty (positive)
    magerr[good] = MAG_ERR_FACTOR * (err[good] / flux[good])

    return mag, magerr


def remove_pixel_mean_weighted(mag, magerr):
    """
    Subtract weighted mean over time for each pixel (column).
    mag, magerr shape: (n_exp, n_pix)
    """
    w = np.zeros_like(mag, dtype=np.float64)
    ok = np.isfinite(mag) & np.isfinite(magerr) & (magerr > 0)
    w[ok] = 1.0 / (magerr[ok] ** 2)

    # weighted mean per pixel
    wsum = np.nansum(w, axis=0)  # (n_pix,)
    msum = np.nansum(w * mag, axis=0)

    mu = np.full(mag.shape[1], np.nan, dtype=np.float64)
    goodpix = wsum > 0
    mu[goodpix] = msum[goodpix] / wsum[goodpix]

    resid = mag - mu[None, :]
    return resid


# ---------------------------------------------------------------------
# Core SYSREM 
# ---------------------------------------------------------------------

def _compute_c_i(r_ij, a_j, w_ij):
    """
    r_ij: (n_pix, n_exp)
    a_j:  (n_exp,)
    w_ij: (n_pix, n_exp) weights
    """
    aj = a_j[None, :]  # (1, n_exp)
    num = np.nansum(w_ij * r_ij * aj, axis=1)          # (n_pix,)
    den = np.nansum(w_ij * (aj ** 2), axis=1)          # (n_pix,)
    c = np.full(r_ij.shape[0], np.nan, dtype=np.float64)
    good = den > 0
    c[good] = num[good] / den[good]
    return c


def _compute_a_j(r_ij, c_i, w_ij):
    """
    r_ij: (n_pix, n_exp)
    c_i:  (n_pix,)
    w_ij: (n_pix, n_exp)
    """
    ci = c_i[:, None]  # (n_pix, 1)
    num = np.nansum(w_ij * r_ij * ci, axis=0)          # (n_exp,)
    den = np.nansum(w_ij * (ci ** 2), axis=0)          # (n_exp,)
    a = np.full(r_ij.shape[1], np.nan, dtype=np.float64)
    good = den > 0
    a[good] = num[good] / den[good]
    return a


def sysrem_one(mag, magerr, airmass, tol=1e-4, max_iter=50):
    """
    One SYSREM application for one order.

    Inputs:
      mag, magerr: (n_exp, n_pix) magnitudes + mag errors
      airmass:     (n_exp,) initial vector (can be airmass, time, etc.)

    Returns:
      resid: (n_exp, n_pix) after removing one SYSREM component
      optionally (a_j, c_i) if return_trend=True
    """
    mag = np.asarray(mag, dtype=np.float64)
    magerr = np.asarray(magerr, dtype=np.float64)
    airmass = np.asarray(airmass, dtype=np.float64)

    ok = np.isfinite(mag) & np.isfinite(magerr) & (magerr > 0) & np.isfinite(airmass)[:, None]

    sigma = magerr.copy()

    # Weights
    w = np.zeros_like(mag, dtype=np.float64)
    w[ok] = 1.0 / (sigma[ok] ** 2)

    # Mean-center per pixel with weights
    resid = remove_pixel_mean_weighted(mag, sigma)

    r_ij = resid.T  # (n_pix, n_exp)
    w_ij = w.T      # (n_pix, n_exp)

    # Initialize
    a_j = airmass.copy()
    # make sure it's finite
    a_j = np.where(np.isfinite(a_j), a_j, 0.0)

    c_i = np.ones(r_ij.shape[0], dtype=np.float64)

    # ALS loop
    for _ in range(max_iter):
        c_new = _compute_c_i(r_ij, a_j, w_ij)
        a_new = _compute_a_j(r_ij, c_new, w_ij)

        # convergence on finite components only
        dc = np.nanmax(np.abs(c_new - c_i))
        da = np.nanmax(np.abs(a_new - a_j))
        c_i, a_j = c_new, a_new

        if np.isfinite(dc) and np.isfinite(da) and max(dc, da) < tol:
            break

    # Subtract best-fit rank-1 systematic
    model = c_i[:, None] * a_j[None, :]     # (n_pix, n_exp)
    out = (r_ij - model).T                  # (n_exp, n_pix)

    return out


def sysrem_multi(mag, magerr, airmass, n_app=15, **kwargs):
    """
    Apply SYSREM repeatedly (n_app times) to one order.
    Each iteration recomputes the best rank-1 component on the residuals.

    Returns final residual magnitudes (n_exp, n_pix).
    """
    results_cube = np.zeros((n_app, np.shape(mag)[0], np.shape(mag)[1]))
    r = mag
    for k in range(int(n_app)):
        r = sysrem_one(r, magerr, airmass, **kwargs)
        results_cube[k,:,:] = r
        print(k, 'iterations')
    return results_cube


def sysrem_total(mag_cube, magerr_cube, airmass, n_app=15, orders=None, **kwargs):
    """
    Apply SYSREM to multiple orders.

    mag_cube, magerr_cube: (n_ord, n_exp, n_pix)
    orders: None or list/array of order indices to process
    """
    results_cube = np.zeros((n_app, np.shape(mag_cube)[0], np.shape(mag_cube)[1], np.shape(mag_cube)[2]))
    mag_cube = np.asarray(mag_cube, dtype=np.float64)
    magerr_cube = np.asarray(magerr_cube, dtype=np.float64)

    if orders is not None:
        mag_cube = mag_cube[orders, :, :]
        magerr_cube = magerr_cube[orders, :, :]

    for o in range(mag_cube.shape[0]):
        print('order: ', o)
        order_result = sysrem_multi(mag_cube[o], magerr_cube[o], airmass, n_app=n_app, **kwargs)
        results_cube[:,o,:,:] = order_result
    return results_cube




