# cython: language_level=3

cdef struct GridMeta:
    double r_min
    double r_max
    double z_min
    double z_max
    int n_r
    int n_z

cdef int cell_index(const double[:] coords, double value) nogil noexcept

cdef double bilinear_field(
    const double[:, :] values,
    const unsigned char[:, :] mask,
    const double[:] r_coords,
    const double[:] z_coords,
    GridMeta meta,
    double r,
    double z,
) nogil noexcept

cdef bint grid_in_plasma(
    const double[:, :] n_e,
    const unsigned char[:, :] mask,
    const double[:] r_coords,
    const double[:] z_coords,
    GridMeta meta,
    double r,
    double z,
) nogil noexcept

cdef double grid_local_ti(
    const double[:, :] ti,
    const unsigned char[:, :] mask,
    const double[:] r_coords,
    const double[:] z_coords,
    GridMeta meta,
    double r,
    double z,
    double default_ti,
) nogil noexcept

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
) nogil noexcept

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
) nogil noexcept
