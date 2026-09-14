"""Lat/lon <-> grid-index transforms for the rain-gauge raster.

The prior project trained for months on imagery centred 1,100 km from Singapore
because a hard-coded tile index was never checked against a map. Every transform
here is therefore paired with an executable round-trip check (`self_test`), and
`tests/test_geometry.py` asserts real gauge coordinates land where they should.

Convention
----------
Cell (row=j, col=i) has its CENTRE at:
    lon = lon_min + (i + 0.5) * dlon,   dlon = (lon_max - lon_min) / nx
    lat = lat_min + (j + 0.5) * dlat,   dlat = (lat_max - lat_min) / ny

Row 0 is the SOUTHERNMOST row (lat increases with j). Any plotting code that
wants north-up must flip explicitly - see `GridSpec.to_display()`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

EARTH_R_KM = 6371.0088


@dataclass(frozen=True)
class GridSpec:
    lon_min: float
    lon_max: float
    lat_min: float
    lat_max: float
    nx: int
    ny: int

    @classmethod
    def from_config(cls, cfg: dict) -> "GridSpec":
        g = cfg["grid"]
        return cls(g["lon_min"], g["lon_max"], g["lat_min"], g["lat_max"], g["nx"], g["ny"])

    # ---------------------------------------------------------------- spacing
    @property
    def dlon(self) -> float:
        return (self.lon_max - self.lon_min) / self.nx

    @property
    def dlat(self) -> float:
        return (self.lat_max - self.lat_min) / self.ny

    # ---------------------------------------------------------- centre arrays
    def lon_centres(self) -> np.ndarray:
        return self.lon_min + (np.arange(self.nx) + 0.5) * self.dlon

    def lat_centres(self) -> np.ndarray:
        return self.lat_min + (np.arange(self.ny) + 0.5) * self.dlat

    def mesh(self) -> tuple[np.ndarray, np.ndarray]:
        """(ny, nx) arrays of cell-centre lon and lat."""
        return np.meshgrid(self.lon_centres(), self.lat_centres())

    # ------------------------------------------------------------ transforms
    def lonlat_to_ij(self, lon, lat):
        """Continuous (fractional) grid coordinates. Inverse of ij_to_lonlat."""
        lon = np.asarray(lon, dtype=float)
        lat = np.asarray(lat, dtype=float)
        i = (lon - self.lon_min) / self.dlon - 0.5
        j = (lat - self.lat_min) / self.dlat - 0.5
        return i, j

    def ij_to_lonlat(self, i, j):
        i = np.asarray(i, dtype=float)
        j = np.asarray(j, dtype=float)
        lon = self.lon_min + (i + 0.5) * self.dlon
        lat = self.lat_min + (j + 0.5) * self.dlat
        return lon, lat

    def nearest_ij(self, lon, lat):
        """Integer indices of the containing cell (clipped to the grid)."""
        i, j = self.lonlat_to_ij(lon, lat)
        i = np.clip(np.rint(i).astype(int), 0, self.nx - 1)
        j = np.clip(np.rint(j).astype(int), 0, self.ny - 1)
        return i, j

    def contains(self, lon, lat) -> np.ndarray:
        lon = np.asarray(lon, dtype=float)
        lat = np.asarray(lat, dtype=float)
        return ((lon >= self.lon_min) & (lon <= self.lon_max)
                & (lat >= self.lat_min) & (lat <= self.lat_max))

    # ------------------------------------------------------------------ misc
    def km_per_deg(self) -> tuple[float, float]:
        """(km per degree lon, km per degree lat) at the grid's centre latitude."""
        lat0 = 0.5 * (self.lat_min + self.lat_max)
        km_lat = np.pi * EARTH_R_KM / 180.0
        km_lon = km_lat * np.cos(np.radians(lat0))
        return float(km_lon), float(km_lat)

    def cell_size_km(self) -> tuple[float, float]:
        km_lon, km_lat = self.km_per_deg()
        return self.dlon * km_lon, self.dlat * km_lat

    def to_display(self, grid: np.ndarray) -> np.ndarray:
        """Flip row axis so index 0 is the NORTH edge (image/plot convention)."""
        return grid[..., ::-1, :]

    def bounds_leaflet(self) -> list[list[float]]:
        """[[south, west], [north, east]] - Leaflet imageOverlay bounds."""
        return [[self.lat_min, self.lon_min], [self.lat_max, self.lon_max]]

    # ------------------------------------------------------------- self-test
    def self_test(self, tol: float = 1e-9) -> None:
        """Round-trip known points and assert agreement. Raises on failure."""
        # 1. cell centres must round-trip exactly through both directions
        i0 = np.arange(self.nx)
        j0 = np.arange(self.ny)
        lon, lat = self.ij_to_lonlat(i0, np.zeros_like(i0))
        i_back, _ = self.lonlat_to_ij(lon, lat)
        if not np.allclose(i_back, i0, atol=tol):
            raise AssertionError(f"lon round-trip failed: max err {np.abs(i_back - i0).max()}")
        lon, lat = self.ij_to_lonlat(np.zeros_like(j0), j0)
        _, j_back = self.lonlat_to_ij(lon, lat)
        if not np.allclose(j_back, j0, atol=tol):
            raise AssertionError(f"lat round-trip failed: max err {np.abs(j_back - j0).max()}")

        # 2. corners map to the corner cells, in the orientation we claim
        i, j = self.nearest_ij(self.lon_min + 1e-6, self.lat_min + 1e-6)
        if (int(i), int(j)) != (0, 0):
            raise AssertionError(f"SW corner -> ({i},{j}), expected (0,0)")
        i, j = self.nearest_ij(self.lon_max - 1e-6, self.lat_max - 1e-6)
        if (int(i), int(j)) != (self.nx - 1, self.ny - 1):
            raise AssertionError(f"NE corner -> ({i},{j}), expected ({self.nx-1},{self.ny-1})")

        # 3. latitude must increase with j (guards a silent flip)
        lat_lo = self.ij_to_lonlat(0, 0)[1]
        lat_hi = self.ij_to_lonlat(0, self.ny - 1)[1]
        if not lat_hi > lat_lo:
            raise AssertionError("latitude does not increase with row index j")

        # 4. mesh must agree with the scalar transform
        LON, LAT = self.mesh()
        lon_c, lat_c = self.ij_to_lonlat(5 % self.nx, 7 % self.ny)
        if not (np.isclose(LON[7 % self.ny, 5 % self.nx], lon_c)
                and np.isclose(LAT[7 % self.ny, 5 % self.nx], lat_c)):
            raise AssertionError("mesh() disagrees with ij_to_lonlat()")


def haversine_km(lon1, lat1, lon2, lat2):
    """Great-circle distance in km. Broadcasts."""
    lon1, lat1, lon2, lat2 = map(np.radians, (np.asarray(lon1, dtype=float),
                                              np.asarray(lat1, dtype=float),
                                              np.asarray(lon2, dtype=float),
                                              np.asarray(lat2, dtype=float)))
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_R_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
