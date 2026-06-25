# distutils: language = c
# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True

"""Cython backward tracker — Cython hot loop (nogil grid + wall + RNG + CX rejection)."""

from libc.math cimport M_PI, NAN, cos, exp, fabs, fmax, fmin, hypot, log, pow, sqrt
from cython cimport boundscheck, wraparound
from cython.parallel cimport prange

import numpy as np

cimport numpy as cnp

cnp.import_array()

CYTHON_AVAILABLE = True

cdef struct GridMeta:
    double r_min
    double r_max
    double z_min
    double z_max
    int n_r
    int n_z

cdef struct Pcg64State:
    unsigned long long state
    unsigned long long inc

cdef unsigned long long PCG64_MULT = 6364136223846793005ULL
cdef unsigned long long PCG64_MASK = 0xFFFFFFFFFFFFFFFFULL

cdef unsigned long long rotl64(unsigned long long x, unsigned long long k) nogil noexcept:
    return (x << k) | (x >> (64 - k))

cdef unsigned long long pcg64_next(Pcg64State* st) nogil noexcept:
    # PCG-XSH-RR with 64-bit state and 32-bit output (a.k.a. pcg32).
    cdef unsigned long long oldstate = st.state
    st.state = oldstate * PCG64_MULT + st.inc
    cdef unsigned int xorshifted = <unsigned int>(((oldstate >> 18) ^ oldstate) >> 27)
    cdef unsigned int rot = <unsigned int>(oldstate >> 59)
    return <unsigned long long>((xorshifted >> rot) | (xorshifted << ((-rot) & 31)))

cdef void pcg64_seed(Pcg64State* st, unsigned long long seed) nogil noexcept:
    st.state = 0
    st.inc = (PCG64_MULT << 1) | 1
    st.state = st.state * PCG64_MULT + st.inc
    st.state += seed
    st.state = st.state * PCG64_MULT + st.inc

cdef void pcg64_advance(Pcg64State* st, unsigned long long seed, unsigned long long inc) nogil noexcept:
    st.state = seed
    st.inc = inc | 1

cdef double pcg64_random(Pcg64State* st) nogil noexcept:
    # 32-bit output mapped to [0, 1).
    return (<double>(pcg64_next(st) & 0xFFFFFFFFULL)) * (1.0 / 4294967296.0)

cdef double pcg64_normal(Pcg64State* st) nogil noexcept:
    cdef double u1 = pcg64_random(st)
    cdef double u2 = pcg64_random(st)
    if u1 < 1e-15:
        u1 = 1e-15
    return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2)

# --- grid implementation (grid.pxd) ---

cdef int cell_index(const double[:] coords, double value) nogil noexcept:
    cdef int n = coords.shape[0]
    cdef int idx
    if value < coords[0] or value > coords[n - 1]:
        return -1
    if fabs(value - coords[n - 1]) < 1e-15:
        return n - 2
    idx = 0
    while idx < n - 1 and coords[idx + 1] < value:
        idx += 1
    if idx >= n - 1:
        idx = n - 2
    if idx < 0:
        return -1
    return idx

cdef double bilinear_field(
    const double[:, :] values,
    const unsigned char[:, :] mask,
    const double[:] r_coords,
    const double[:] z_coords,
    GridMeta meta,
    double r,
    double z,
) nogil noexcept:
    cdef int i = cell_index(r_coords, r)
    cdef int j = cell_index(z_coords, z)
    cdef double r0, r1, z0, z1, tx, ty
    cdef double v00, v10, v01, v11
    if i < 0 or j < 0:
        return NAN
    if not (mask[j, i] and mask[j, i + 1] and mask[j + 1, i] and mask[j + 1, i + 1]):
        return NAN
    r0 = r_coords[i]
    r1 = r_coords[i + 1]
    z0 = z_coords[j]
    z1 = z_coords[j + 1]
    tx = 0.0 if r1 == r0 else (r - r0) / (r1 - r0)
    ty = 0.0 if z1 == z0 else (z - z0) / (z1 - z0)
    v00 = values[j, i]
    v10 = values[j, i + 1]
    v01 = values[j + 1, i]
    v11 = values[j + 1, i + 1]
    return (1.0 - tx) * (1.0 - ty) * v00 + tx * (1.0 - ty) * v10 + (1.0 - tx) * ty * v01 + tx * ty * v11

cdef bint grid_in_plasma(
    const double[:, :] n_e,
    const unsigned char[:, :] mask,
    const double[:] r_coords,
    const double[:] z_coords,
    GridMeta meta,
    double r,
    double z,
) nogil noexcept:
    cdef double nval
    if r < meta.r_min or r > meta.r_max or z < meta.z_min or z > meta.z_max:
        return False
    nval = bilinear_field(n_e, mask, r_coords, z_coords, meta, r, z)
    return nval == nval and nval > 0.0

cdef double grid_local_ti(
    const double[:, :] ti,
    const unsigned char[:, :] mask,
    const double[:] r_coords,
    const double[:] z_coords,
    GridMeta meta,
    double r,
    double z,
    double default_ti,
) nogil noexcept:
    cdef double val = bilinear_field(ti, mask, r_coords, z_coords, meta, r, z)
    if val != val or val <= 0.0:
        return default_ti
    return val

cdef double grid_sigma_ion(
    const double[:, :] n_e,
    const double[:, :] iz_rate,
    const unsigned char[:, :] mask,
    const double[:] r_coords,
    const double[:] z_coords,
    GridMeta meta,
    double r,
    double z,
    double speed_m_s,
) nogil noexcept:
    cdef double ne, iz
    if speed_m_s <= 0.0:
        return 0.0
    ne = bilinear_field(n_e, mask, r_coords, z_coords, meta, r, z)
    iz = bilinear_field(iz_rate, mask, r_coords, z_coords, meta, r, z)
    if ne != ne or iz != iz:
        return 0.0
    return fmax(0.0, ne * iz / speed_m_s)

cdef double grid_sigma_cx(
    const double[:, :] n_e,
    const double[:, :] cx_rate,
    const unsigned char[:, :] mask,
    const double[:] r_coords,
    const double[:] z_coords,
    GridMeta meta,
    double r,
    double z,
    double speed_m_s,
) nogil noexcept:
    cdef double ne, cx
    if speed_m_s <= 0.0:
        return 0.0
    ne = bilinear_field(n_e, mask, r_coords, z_coords, meta, r, z)
    cx = bilinear_field(cx_rate, mask, r_coords, z_coords, meta, r, z)
    if ne != ne or cx != cx:
        return 0.0
    return fmax(0.0, ne * cx / speed_m_s)

# --- CX kernel (Janev H.2 + rejection) ---

cdef double CX_AMU = 1.66053906660e-27
cdef double CX_CHARGE = 1.602176634e-19
cdef double CX_M_REF_AMU = 1.0
cdef double CX_JANEV_A0 = 3.2345
cdef double CX_JANEV_A1 = 2.3588e2
cdef double CX_JANEV_A2 = 2.3713
cdef double CX_JANEV_A3 = 3.8371e-2
cdef double CX_JANEV_A4 = 3.8068e-6
cdef double CX_JANEV_A5 = 1.1832e-10

cdef double cx_relative_energy_kev(double v_rel_m_s) nogil noexcept:
    cdef double mu = 0.5 * CX_M_REF_AMU * CX_AMU
    return 0.5 * mu * v_rel_m_s * v_rel_m_s / (CX_CHARGE * 1000.0)

cdef double cx_sigma_janev_m2(double energy_kev_amu) nogil noexcept:
    cdef double ehat = fmax(energy_kev_amu, 1.0e-12)
    cdef double numer = CX_JANEV_A0 * log(CX_JANEV_A1 / ehat + CX_JANEV_A2)
    cdef double denom = (
        1.0
        + CX_JANEV_A3 * ehat
        + CX_JANEV_A4 * pow(ehat, 3.5)
        + CX_JANEV_A5 * pow(ehat, 5.4)
    )
    return 1.0e-20 * numer / denom

cdef double cx_calibration_factor(
    double ti_ev,
    const double[:] cal_log_t,
    const double[:] cal_log_f,
) nogil noexcept:
    cdef int n = cal_log_t.shape[0]
    cdef int i
    cdef double logv, t0, t1, f0, f1, alpha
    if n < 2 or ti_ev <= 0.0:
        return 1.0
    logv = log(fmax(ti_ev, exp(cal_log_t[0])))
    if logv <= cal_log_t[0]:
        return exp(cal_log_f[0])
    if logv >= cal_log_t[n - 1]:
        return exp(cal_log_f[n - 1])
    for i in range(n - 1):
        if logv <= cal_log_t[i + 1]:
            t0 = cal_log_t[i]
            t1 = cal_log_t[i + 1]
            f0 = cal_log_f[i]
            f1 = cal_log_f[i + 1]
            alpha = 0.0 if t1 == t0 else (logv - t0) / (t1 - t0)
            return exp(f0 + alpha * (f1 - f0))
    return exp(cal_log_f[n - 1])

cdef double cx_sigma_v(
    double v_rel_m_s,
    double ti_ev,
    bint calibrate_hdg,
    const double[:] cal_log_t,
    const double[:] cal_log_f,
) nogil noexcept:
    cdef double sigma = cx_sigma_janev_m2(cx_relative_energy_kev(v_rel_m_s))
    if calibrate_hdg:
        sigma *= cx_calibration_factor(ti_ev, cal_log_t, cal_log_f)
    return sigma * v_rel_m_s

cdef double cx_majorant_sv(
    double ti_ev,
    double v_n_x,
    double v_n_y,
    double v_n_z,
    bint calibrate_hdg,
    const double[:] cal_log_t,
    const double[:] cal_log_f,
) nogil noexcept:
    cdef double v_n_norm = hypot(v_n_x, hypot(v_n_y, v_n_z))
    cdef double std = sqrt(ti_ev * CX_CHARGE / (CX_M_REF_AMU * CX_AMU)) if ti_ev > 0.0 else 0.0
    cdef double v_max = fmax(v_n_norm + 50.0 * std, 2.0e6)
    cdef int n_scan = 512
    cdef int k
    cdef double speed, v_rel, sv, best = 0.0
    if ti_ev <= 0.0:
        return 0.0
    for k in range(n_scan + 1):
        speed = v_max * k / n_scan
        v_rel = v_n_norm + speed
        sv = cx_sigma_v(v_rel, ti_ev, calibrate_hdg, cal_log_t, cal_log_f)
        if sv > best:
            best = sv
    return best

cdef bint cx_rejection_resample_velocity(
    Pcg64State* rng,
    double ti_ev,
    double fallback_speed,
    double* vx,
    double* vy,
    double* vz,
    bint calibrate_hdg,
    const double[:] cal_log_t,
    const double[:] cal_log_f,
    int max_trials,
) nogil noexcept:
    cdef double vnx = vx[0]
    cdef double vny = vy[0]
    cdef double vnz = vz[0]
    cdef double majorant, std, vix, viy, viz, v_rel, sv
    cdef int trial
    if ti_ev <= 0.0:
        return False
    majorant = cx_majorant_sv(ti_ev, vnx, vny, vnz, calibrate_hdg, cal_log_t, cal_log_f)
    if majorant <= 0.0:
        return False
    std = sqrt(ti_ev * CX_CHARGE / (CX_M_REF_AMU * CX_AMU))
    for trial in range(max_trials):
        vix = pcg64_normal(rng) * std
        viy = pcg64_normal(rng) * std
        viz = pcg64_normal(rng) * std
        v_rel = hypot(vnx - vix, hypot(vny - viy, vnz - viz))
        sv = cx_sigma_v(v_rel, ti_ev, calibrate_hdg, cal_log_t, cal_log_f)
        if pcg64_random(rng) < sv / majorant:
            vx[0] = vix
            vy[0] = viy
            vz[0] = viz
            return True
    return False

cdef bint apply_cx_velocity_update(
    Pcg64State* rng,
    const double[:, :] ti,
    const unsigned char[:, :] mask,
    const double[:] r_coords,
    const double[:] z_coords,
    GridMeta meta,
    double r,
    double z,
    double fallback_speed,
    double* vx,
    double* vy,
    double* vz,
    bint cx_rejection,
    bint calibrate_hdg,
    const double[:] cal_log_t,
    const double[:] cal_log_f,
    int max_trials,
) nogil noexcept:
    cdef double ti_ev = grid_local_ti(ti, mask, r_coords, z_coords, meta, r, z, 10.0)
    if cx_rejection:
        return cx_rejection_resample_velocity(
            rng, ti_ev, fallback_speed, vx, vy, vz,
            calibrate_hdg, cal_log_t, cal_log_f, max_trials,
        )
    sample_maxwellian_velocity(rng, ti_ev, fallback_speed, vx, vy, vz)
    return True

# --- wall ray intersection (simplified port of geometry/wall.py) ---

cdef struct RayHitC:
    double t
    double hit_r
    double hit_z
    int segment_index

cdef void solve_ray_radius(
    double ox, double oy, double dx, double dy, double target_r, double t_min,
    double* out_t, int* out_n,
) nogil noexcept:
    cdef double a, b, c, disc, sqrt_disc, t_val
    cdef int count = 0
    out_n[0] = 0
    if target_r < 0.0:
        return
    a = dx * dx + dy * dy
    if a < 1e-15:
        if fabs(hypot(ox, oy) - target_r) <= 1e-9:
            out_t[0] = t_min
            out_n[0] = 1
        return
    b = 2.0 * (ox * dx + oy * dy)
    c = ox * ox + oy * oy - target_r * target_r
    disc = b * b - 4.0 * a * c
    if disc < 0.0:
        return
    sqrt_disc = sqrt(fmax(disc, 0.0))
    t_val = (-b - sqrt_disc) / (2.0 * a)
    if t_val >= t_min:
        out_t[count] = t_val
        count += 1
    t_val = (-b + sqrt_disc) / (2.0 * a)
    if t_val >= t_min:
        out_t[count] = t_val
        count += 1
    out_n[0] = count

cdef RayHitC intersect_segment(
    double ox, double oy, double oz,
    double dx, double dy, double dz,
    double r0, double z0, double r1, double z1,
    int seg_index,
    double t_min,
    double t_max,
) nogil noexcept:
    cdef RayHitC best
    cdef RayHitC hit
    cdef double dr, dz_seg, u_hit, rt, zt_plane, t_hit
    cdef double t_hits[2]
    cdef int n_hits, i, k, scan, bis
    cdef bint found = False
    cdef double u_a, u_b, t_a, t_b, res_a, res_b, lo, hi, mid, t_mid, res_mid, res_lo, zt_scan, rt2
    best.t = 1e300
    best.segment_index = -1
    dr = r1 - r0
    dz_seg = z1 - z0
    if fabs(dz) <= 1e-9:
        if fabs(dz_seg) <= 1e-9:
            if fabs(oz - z0) > 1e-6:
                return best
            for k in range(33):
                u_hit = k / 32.0
                rt = r0 + u_hit * dr
                solve_ray_radius(ox, oy, dx, dy, rt, t_min, t_hits, &n_hits)
                for i in range(n_hits):
                    t_hit = t_hits[i]
                    if t_hit <= t_max and t_hit < best.t:
                        best.t = t_hit
                        best.hit_r = hypot(ox + t_hit * dx, oy + t_hit * dy)
                        best.hit_z = oz + t_hit * dz
                        best.segment_index = seg_index
                        found = True
            return best
        u_hit = (oz - z0) / dz_seg
        if u_hit < -1e-9 or u_hit > 1.0 + 1e-9:
            return best
        u_hit = fmin(fmax(u_hit, 0.0), 1.0)
        rt = r0 + u_hit * dr
        solve_ray_radius(ox, oy, dx, dy, rt, t_min, t_hits, &n_hits)
        for i in range(n_hits):
            t_hit = t_hits[i]
            if t_hit <= t_max and t_hit < best.t:
                best.t = t_hit
                best.hit_r = hypot(ox + t_hit * dx, oy + t_hit * dy)
                best.hit_z = oz + t_hit * dz
                best.segment_index = seg_index
    else:
        for scan in range(32):
            u_a = scan / 32.0
            u_b = (scan + 1) / 32.0
            zt_scan = z0 + u_a * dz_seg
            rt2 = r0 + u_a * dr
            t_a = (zt_scan - oz) / dz
            if t_a < t_min:
                continue
            res_a = hypot(ox + t_a * dx, oy + t_a * dy) - rt2
            zt_scan = z0 + u_b * dz_seg
            rt2 = r0 + u_b * dr
            t_b = (zt_scan - oz) / dz
            if t_b < t_min:
                continue
            res_b = hypot(ox + t_b * dx, oy + t_b * dy) - rt2
            if res_a == 0.0 or res_b == 0.0 or res_a * res_b < 0.0:
                lo = u_a
                hi = u_b
                for bis in range(50):
                    mid = 0.5 * (lo + hi)
                    zt_scan = z0 + mid * dz_seg
                    rt2 = r0 + mid * dr
                    t_mid = (zt_scan - oz) / dz
                    if t_mid < t_min:
                        break
                    res_mid = hypot(ox + t_mid * dx, oy + t_mid * dy) - rt2
                    if fabs(res_mid) < 1e-10:
                        if t_mid <= t_max and t_mid < best.t:
                            best.t = t_mid
                            best.hit_r = hypot(ox + t_mid * dx, oy + t_mid * dy)
                            best.hit_z = oz + t_mid * dz
                            best.segment_index = seg_index
                        break
                    zt_scan = z0 + lo * dz_seg
                    rt2 = r0 + lo * dr
                    t_a = (zt_scan - oz) / dz
                    if t_a < t_min:
                        break
                    res_lo = hypot(ox + t_a * dx, oy + t_a * dy) - rt2
                    if res_lo * res_mid <= 0.0:
                        hi = mid
                    else:
                        lo = mid
    return best

cdef RayHitC intersect_wall(
    double ox, double oy, double oz,
    double dx, double dy, double dz,
    const double[:] seg_r0,
    const double[:] seg_z0,
    const double[:] seg_r1,
    const double[:] seg_z1,
    double t_min,
    double t_max,
) nogil noexcept:
    cdef RayHitC best
    cdef RayHitC hit
    cdef int n = seg_r0.shape[0]
    cdef int i
    cdef double norm = hypot(dx, hypot(dy, dz))
    best.t = 1e300
    best.segment_index = -1
    if norm < 1e-15:
        return best
    dx /= norm
    dy /= norm
    dz /= norm
    for i in range(n):
        hit = intersect_segment(
            ox, oy, oz, dx, dy, dz,
            seg_r0[i], seg_z0[i], seg_r1[i], seg_z1[i],
            i, t_min, t_max,
        )
        if hit.segment_index >= 0 and hit.t < best.t:
            best = hit
    return best

cdef void sample_maxwellian_velocity(
    Pcg64State* rng,
    double ti_ev,
    double fallback_speed,
    double* vx,
    double* vy,
    double* vz,
) nogil noexcept:
    cdef double mass = 1.66053906660e-27
    cdef double charge = 1.602176634e-19
    cdef double std, speed
    if ti_ev <= 0.0:
        std = 0.0
    else:
        std = sqrt(ti_ev * charge / mass)
    if std <= 0.0:
        vx[0] = fallback_speed
        vy[0] = 0.0
        vz[0] = 0.0
        return
    vx[0] = pcg64_normal(rng) * std
    vy[0] = pcg64_normal(rng) * std
    vz[0] = pcg64_normal(rng) * std
    speed = hypot(vx[0], hypot(vy[0], vz[0]))
    if speed < 100.0:
        vx[0] = fallback_speed
        vy[0] = 0.0
        vz[0] = 0.0

cdef int track_one(
    double seed_x, double seed_y, double seed_z_cart, double seed_r, double seed_z_plane,
    const double[:, :] n_e,
    const double[:, :] iz_rate,
    const double[:, :] cx_rate,
    const double[:, :] ti,
    const unsigned char[:, :] mask,
    const double[:] r_coords,
    const double[:] z_coords,
    GridMeta meta,
    const double[:] seg_r0,
    const double[:] seg_z0,
    const double[:] seg_r1,
    const double[:] seg_z1,
    Pcg64State* rng,
    double tau_max,
    double max_step_m,
    double vacuum_wall_search_m,
    double max_path_m,
    int max_steps,
    double fallback_speed,
    bint enable_cx,
    bint cx_rejection,
    bint calibrate_hdg,
    const double[:] cal_log_t,
    const double[:] cal_log_f,
    int cx_max_trials,
    bint survival_weight,
    double* out_weight,
    int* out_termination,
    int* out_segment,
    double* out_path,
    int* out_n_steps,
    int* out_n_cx,
    double* out_hit_r,
    double* out_hit_z,
) nogil noexcept:
    cdef double px = seed_x
    cdef double py = seed_y
    cdef double pz = seed_z_cart
    cdef double vx, vy, vz, speed, r, z, log_w, path_m, ds, sigma_ion, sigma_cx, sigma_eff
    cdef double bx, by, bz, p_cx
    cdef double ion_weight_sign = -1.0 if survival_weight else 1.0
    cdef int step, n_cx = 0
    cdef RayHitC hit
    cdef int TERM_WALL = 0
    cdef int TERM_LOST = 1
    cdef int TERM_MAX_PATH = 2
    cdef int TERM_MAX_STEPS = 3

    out_n_cx[0] = 0
    if not grid_in_plasma(n_e, mask, r_coords, z_coords, meta, seed_r, seed_z_plane):
        out_weight[0] = 1.0
        out_termination[0] = TERM_LOST
        out_segment[0] = -1
        out_path[0] = 0.0
        out_n_steps[0] = 0
        out_hit_r[0] = NAN
        out_hit_z[0] = NAN
        return 0

    sample_maxwellian_velocity(rng, grid_local_ti(ti, mask, r_coords, z_coords, meta, seed_r, seed_z_plane, 0.0), fallback_speed, &vx, &vy, &vz)
    speed = hypot(vx, hypot(vy, vz))
    if speed <= 0.0:
        sample_maxwellian_velocity(rng, 0.0, fallback_speed, &vx, &vy, &vz)
        speed = hypot(vx, hypot(vy, vz))
    log_w = 0.0
    path_m = 0.0

    for step in range(max_steps):
        r = hypot(px, py)
        z = pz
        speed = hypot(vx, hypot(vy, vz))
        if speed <= 0.0:
            out_weight[0] = exp(log_w)
            out_termination[0] = TERM_LOST
            out_segment[0] = -1
            out_path[0] = path_m
            out_n_steps[0] = step
            out_n_cx[0] = n_cx
            out_hit_r[0] = NAN
            out_hit_z[0] = NAN
            return 0
        bx = -vx / speed
        by = -vy / speed
        bz = -vz / speed

        if not grid_in_plasma(n_e, mask, r_coords, z_coords, meta, r, z):
            hit = intersect_wall(px, py, pz, bx, by, bz, seg_r0, seg_z0, seg_r1, seg_z1, 1e-12, vacuum_wall_search_m)
            if hit.segment_index >= 0:
                sigma_ion = grid_sigma_ion(n_e, iz_rate, mask, r_coords, z_coords, meta, r, z, speed)
                log_w += ion_weight_sign * sigma_ion * hit.t
                out_weight[0] = exp(log_w)
                out_termination[0] = TERM_WALL
                out_segment[0] = hit.segment_index
                out_path[0] = path_m + hit.t
                out_n_steps[0] = step
                out_n_cx[0] = n_cx
                out_hit_r[0] = hit.hit_r
                out_hit_z[0] = hit.hit_z
                return 0
            out_weight[0] = exp(log_w)
            out_termination[0] = TERM_LOST
            out_segment[0] = -1
            out_path[0] = path_m
            out_n_steps[0] = step
            out_n_cx[0] = n_cx
            out_hit_r[0] = NAN
            out_hit_z[0] = NAN
            return 0

        sigma_ion = grid_sigma_ion(n_e, iz_rate, mask, r_coords, z_coords, meta, r, z, speed)
        sigma_cx = grid_sigma_cx(n_e, cx_rate, mask, r_coords, z_coords, meta, r, z, speed) if enable_cx else 0.0
        sigma_eff = sigma_ion + sigma_cx
        if sigma_eff > 0.0:
            ds = tau_max / sigma_eff
        else:
            ds = max_step_m
        if ds > max_step_m:
            ds = max_step_m
        if ds > max_path_m - path_m:
            ds = max_path_m - path_m
        if ds <= 0.0:
            out_weight[0] = exp(log_w)
            out_termination[0] = TERM_MAX_PATH
            out_segment[0] = -1
            out_path[0] = path_m
            out_n_steps[0] = step
            out_n_cx[0] = n_cx
            out_hit_r[0] = NAN
            out_hit_z[0] = NAN
            return 0

        hit = intersect_wall(px, py, pz, bx, by, bz, seg_r0, seg_z0, seg_r1, seg_z1, 1e-12, ds)
        if hit.segment_index >= 0:
            log_w += ion_weight_sign * sigma_ion * hit.t
            if enable_cx and sigma_cx > 0.0:
                p_cx = 1.0 - exp(-sigma_cx * hit.t)
                if pcg64_random(rng) < p_cx:
                    if apply_cx_velocity_update(
                        rng, ti, mask, r_coords, z_coords, meta, r, z, fallback_speed,
                        &vx, &vy, &vz, cx_rejection, calibrate_hdg, cal_log_t, cal_log_f, cx_max_trials,
                    ):
                        n_cx += 1
            out_weight[0] = exp(log_w)
            out_termination[0] = TERM_WALL
            out_segment[0] = hit.segment_index
            out_path[0] = path_m + hit.t
            out_n_steps[0] = step + 1
            out_n_cx[0] = n_cx
            out_hit_r[0] = hit.hit_r
            out_hit_z[0] = hit.hit_z
            return 0

        log_w += ion_weight_sign * sigma_ion * ds
        if enable_cx and sigma_cx > 0.0:
            p_cx = 1.0 - exp(-sigma_cx * ds)
            if pcg64_random(rng) < p_cx:
                if apply_cx_velocity_update(
                    rng, ti, mask, r_coords, z_coords, meta, r, z, fallback_speed,
                    &vx, &vy, &vz, cx_rejection, calibrate_hdg, cal_log_t, cal_log_f, cx_max_trials,
                ):
                    n_cx += 1
        speed = hypot(vx, hypot(vy, vz))
        if speed <= 0.0:
            out_weight[0] = exp(log_w)
            out_termination[0] = TERM_LOST
            out_segment[0] = -1
            out_path[0] = path_m + ds
            out_n_steps[0] = step + 1
            out_n_cx[0] = n_cx
            out_hit_r[0] = NAN
            out_hit_z[0] = NAN
            return 0
        bx = -vx / speed
        by = -vy / speed
        bz = -vz / speed
        px += bx * ds
        py += by * ds
        pz += bz * ds
        path_m += ds
        if path_m >= max_path_m - 1e-12:
            out_weight[0] = exp(log_w)
            out_termination[0] = TERM_MAX_PATH
            out_segment[0] = -1
            out_path[0] = path_m
            out_n_steps[0] = step + 1
            out_n_cx[0] = n_cx
            out_hit_r[0] = NAN
            out_hit_z[0] = NAN
            return 0

    out_weight[0] = exp(log_w)
    out_termination[0] = TERM_MAX_STEPS
    out_segment[0] = -1
    out_path[0] = path_m
    out_n_steps[0] = max_steps
    out_n_cx[0] = n_cx
    out_hit_r[0] = NAN
    out_hit_z[0] = NAN
    return 0


cdef void track_one_history_parallel(
    int i,
    int history_start_index,
    unsigned long long batch_seed,
    double sx_i,
    double sy_i,
    double sz_i,
    double sr_i,
    double szz_i,
    double[:, :] n_e,
    double[:, :] iz_rate,
    double[:, :] cx_rate,
    double[:, :] ti,
    unsigned char[:, :] mask,
    double[:] r_coords,
    double[:] z_coords,
    GridMeta meta,
    double[:] seg_r0,
    double[:] seg_z0,
    double[:] seg_r1,
    double[:] seg_z1,
    double tau_max,
    double max_step_m,
    double vacuum_wall_search_m,
    double max_path_m,
    int max_steps,
    double fallback_speed_m_s,
    bint enable_cx,
    bint cx_rejection,
    bint calibrate_hdg,
    double[:] cal_t,
    double[:] cal_f,
    int cx_max_trials,
    bint survival_weight,
    double* out_weight,
    int* out_termination,
    int* out_segment,
    double* out_path_m,
    int* out_n_steps,
    int* out_n_cx,
    double* out_hit_r,
    double* out_hit_z,
) nogil noexcept:
    """Run one history with a thread-local RNG (OpenMP-safe)."""
    cdef Pcg64State rng
    pcg64_seed(
        &rng,
        batch_seed
        ^ ((<unsigned long long>history_start_index + <unsigned long long>i + 1)
           * PCG64_MULT),
    )
    track_one(
        sx_i, sy_i, sz_i, sr_i, szz_i,
        n_e, iz_rate, cx_rate, ti, mask, r_coords, z_coords, meta,
        seg_r0, seg_z0, seg_r1, seg_z1,
        &rng,
        tau_max, max_step_m, vacuum_wall_search_m, max_path_m, max_steps, fallback_speed_m_s, enable_cx,
        cx_rejection, calibrate_hdg, cal_t, cal_f, cx_max_trials, survival_weight,
        out_weight, out_termination, out_segment, out_path_m, out_n_steps, out_n_cx,
        out_hit_r, out_hit_z,
    )


cdef void store_parallel_history_result(
    int i,
    int history_start_index,
    unsigned long long batch_seed,
    double sx_i,
    double sy_i,
    double sz_i,
    double sr_i,
    double szz_i,
    double[:, :] n_e,
    double[:, :] iz_rate,
    double[:, :] cx_rate,
    double[:, :] ti,
    unsigned char[:, :] mask,
    double[:] r_coords,
    double[:] z_coords,
    GridMeta meta,
    double[:] seg_r0,
    double[:] seg_z0,
    double[:] seg_r1,
    double[:] seg_z1,
    double tau_max,
    double max_step_m,
    double vacuum_wall_search_m,
    double max_path_m,
    int max_steps,
    double fallback_speed_m_s,
    bint enable_cx,
    bint cx_rejection,
    bint calibrate_hdg,
    double[:] cal_t,
    double[:] cal_f,
    int cx_max_trials,
    bint survival_weight,
    double[:] weights_mv,
    int[:] term_codes_mv,
    int[:] segments_mv,
    double[:] paths_mv,
    int[:] step_counts_mv,
    int[:] cx_counts_mv,
    double[:] hit_rs_mv,
    double[:] hit_zs_mv,
) nogil noexcept:
    cdef double weight, path_m, hit_r, hit_z
    cdef int termination, segment, n_steps, n_cx
    track_one_history_parallel(
        i,
        history_start_index,
        batch_seed,
        sx_i, sy_i, sz_i, sr_i, szz_i,
        n_e, iz_rate, cx_rate, ti, mask, r_coords, z_coords, meta,
        seg_r0, seg_z0, seg_r1, seg_z1,
        tau_max, max_step_m, vacuum_wall_search_m, max_path_m, max_steps, fallback_speed_m_s, enable_cx,
        cx_rejection, calibrate_hdg, cal_t, cal_f, cx_max_trials, survival_weight,
        &weight, &termination, &segment, &path_m, &n_steps, &n_cx, &hit_r, &hit_z,
    )
    weights_mv[i] = weight
    term_codes_mv[i] = termination
    segments_mv[i] = segment
    paths_mv[i] = path_m
    step_counts_mv[i] = n_steps
    cx_counts_mv[i] = n_cx
    hit_rs_mv[i] = hit_r
    hit_zs_mv[i] = hit_z


def run_backward_batch_cython(
    object seeds,
    object grid_arrays,
    object wall_arrays,
    *,
    int seed=42,
    unsigned long long rng_state=0,
    unsigned long long rng_inc=0,
    bint sync_numpy_rng=False,
    double tau_max=0.1,
    double max_step_m=0.02,
    double vacuum_wall_search_m=0.15,
    double max_path_m=5.0,
    int max_steps=20000,
    double fallback_speed_m_s=10000.0,
    bint enable_cx=True,
    bint cx_rejection=True,
    bint calibrate_hdg=True,
    object cx_calibration=None,
    int cx_max_trials=10000,
    int n_threads=0,
    int history_start_index=0,
    bint survival_weight=False,
):
    """
    Run a batch of backward histories in Cython.

    ``cx_rejection=True`` uses the Janev + rejection kernel.
    ``n_threads > 1`` enables OpenMP over histories (ignored when ``sync_numpy_rng``).
    """
    cdef int n = seeds["x"].shape[0]
    cdef Pcg64State rng
    cdef GridMeta meta
    cdef int i
    cdef double weight, path_m, hit_r, hit_z
    cdef int termination, segment, n_steps, n_cx
    cdef bint use_parallel = n_threads != 1 and not sync_numpy_rng and n > 1
    cdef int num_threads = n_threads if n_threads > 1 else 0

    cdef cnp.ndarray weights = np.zeros(n, dtype=np.float64)
    cdef cnp.ndarray term_codes = np.zeros(n, dtype=np.int32)
    cdef cnp.ndarray segments = np.full(n, -1, dtype=np.int32)
    cdef cnp.ndarray paths = np.zeros(n, dtype=np.float64)
    cdef cnp.ndarray step_counts = np.zeros(n, dtype=np.int32)
    cdef cnp.ndarray cx_counts = np.zeros(n, dtype=np.int32)
    cdef cnp.ndarray hit_rs = np.full(n, np.nan, dtype=np.float64)
    cdef cnp.ndarray hit_zs = np.full(n, np.nan, dtype=np.float64)

    cdef double[:] sx = seeds["x"]
    cdef double[:] sy = seeds["y"]
    cdef double[:] sz = seeds["z"]
    cdef double[:] sr = seeds["r"]
    cdef double[:] szz = seeds["z_plane"]

    cdef double[:, :] n_e = grid_arrays["n"]
    cdef double[:, :] iz_rate = grid_arrays["iz_rate"]
    cdef double[:, :] cx_rate = grid_arrays["cx_rate"]
    cdef double[:, :] ti = grid_arrays["ti"]
    cdef unsigned char[:, :] mask = grid_arrays["mask"]
    cdef double[:] r_coords = grid_arrays["r_coords"]
    cdef double[:] z_coords = grid_arrays["z_coords"]

    cdef double[:] seg_r0 = wall_arrays["r0"]
    cdef double[:] seg_z0 = wall_arrays["z0"]
    cdef double[:] seg_r1 = wall_arrays["r1"]
    cdef double[:] seg_z1 = wall_arrays["z1"]

    if cx_calibration is None:
        cal_log_t = np.zeros(0, dtype=np.float64)
        cal_log_f = np.zeros(0, dtype=np.float64)
    else:
        cal_log_t = cx_calibration["log_t"]
        cal_log_f = cx_calibration["log_f"]
    cdef double[:] cal_t = cal_log_t
    cdef double[:] cal_f = cal_log_f

    cdef double[:] weights_mv = weights
    cdef int[:] term_codes_mv = term_codes
    cdef int[:] segments_mv = segments
    cdef double[:] paths_mv = paths
    cdef int[:] step_counts_mv = step_counts
    cdef int[:] cx_counts_mv = cx_counts
    cdef double[:] hit_rs_mv = hit_rs
    cdef double[:] hit_zs_mv = hit_zs

    meta.r_min = grid_arrays["r_min"]
    meta.r_max = grid_arrays["r_max"]
    meta.z_min = grid_arrays["z_min"]
    meta.z_max = grid_arrays["z_max"]
    meta.n_r = grid_arrays["n_r"]
    meta.n_z = grid_arrays["n_z"]

    if use_parallel:
        for i in prange(n, nogil=True, schedule="dynamic", num_threads=num_threads):
            store_parallel_history_result(
                i,
                history_start_index,
                <unsigned long long>seed,
                sx[i], sy[i], sz[i], sr[i], szz[i],
                n_e, iz_rate, cx_rate, ti, mask, r_coords, z_coords, meta,
                seg_r0, seg_z0, seg_r1, seg_z1,
                tau_max, max_step_m, vacuum_wall_search_m, max_path_m, max_steps, fallback_speed_m_s, enable_cx,
                cx_rejection, calibrate_hdg, cal_t, cal_f, cx_max_trials, survival_weight,
                weights_mv, term_codes_mv, segments_mv, paths_mv,
                step_counts_mv, cx_counts_mv, hit_rs_mv, hit_zs_mv,
            )
    else:
        if sync_numpy_rng:
            pcg64_advance(&rng, rng_state, rng_inc)
        else:
            pcg64_seed(&rng, <unsigned long long>seed)
        for i in range(n):
            track_one(
                sx[i], sy[i], sz[i], sr[i], szz[i],
                n_e, iz_rate, cx_rate, ti, mask, r_coords, z_coords, meta,
                seg_r0, seg_z0, seg_r1, seg_z1,
                &rng,
                tau_max, max_step_m, vacuum_wall_search_m, max_path_m, max_steps, fallback_speed_m_s, enable_cx,
                cx_rejection, calibrate_hdg, cal_t, cal_f, cx_max_trials, survival_weight,
                &weight, &termination, &segment, &path_m, &n_steps, &n_cx, &hit_r, &hit_z,
            )
            weights_mv[i] = weight
            term_codes_mv[i] = termination
            segments_mv[i] = segment
            paths_mv[i] = path_m
            step_counts_mv[i] = n_steps
            cx_counts_mv[i] = n_cx
            hit_rs_mv[i] = hit_r
            hit_zs_mv[i] = hit_z

    return {
        "weights": weights,
        "termination_codes": term_codes,
        "segment_index": segments,
        "path_m": paths,
        "n_steps": step_counts,
        "n_cx_events": cx_counts,
        "hit_r": hit_rs,
        "hit_z": hit_zs,
        "seed_r": np.asarray(seeds["r"], dtype=np.float64),
        "seed_z": np.asarray(seeds["z_plane"], dtype=np.float64),
    }


def debug_intersect_wall(
    double ox, double oy, double oz,
    double dx, double dy, double dz,
    object wall_arrays,
    double t_min=1e-12,
    double t_max=1e30,
):
    """Expose intersect_wall for validation against the Python ray tracer."""
    cdef double[:] seg_r0 = wall_arrays["r0"]
    cdef double[:] seg_z0 = wall_arrays["z0"]
    cdef double[:] seg_r1 = wall_arrays["r1"]
    cdef double[:] seg_z1 = wall_arrays["z1"]
    cdef RayHitC hit = intersect_wall(
        ox, oy, oz, dx, dy, dz, seg_r0, seg_z0, seg_r1, seg_z1, t_min, t_max
    )
    if hit.segment_index < 0:
        return None
    return (hit.t, hit.segment_index, hit.hit_r, hit.hit_z)


def debug_sample_speeds(double ti_ev, int n, int seed=0):
    """Expose the Cython Maxwellian sampler: return speeds for validation."""
    cdef Pcg64State rng
    cdef double vx, vy, vz
    cdef int i
    cdef cnp.ndarray out = np.zeros(n, dtype=np.float64)
    cdef double[:] o = out
    pcg64_seed(&rng, <unsigned long long>seed)
    for i in range(n):
        sample_maxwellian_velocity(&rng, ti_ev, 10000.0, &vx, &vy, &vz)
        o[i] = hypot(vx, hypot(vy, vz))
    return out

