"""Generate a Hovmöller diagram of equatorial Pacific rsut anomaly from CMIP6 CESM2.

Shows top-of-atmosphere outgoing shortwave radiation anomaly (W m-2) as a
longitude-time diagram for 5S-5N, 1950-2014. El Nino events appear as
red bands of elevated rsut (less cloud cover -> more solar escape) in the
central and eastern Pacific.
"""

from pathlib import Path

import gcsfs
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import xarray as xr

CATALOG_URL = "https://storage.googleapis.com/cmip6/cmip6-zarr-consolidated-stores.csv"
OUT_DIR = Path("imgs")
PDF_PATH = OUT_DIR / "hovmoller_rsut_equatorial_pacific.pdf"

LON_MIN, LON_MAX = 150, 280   # 150E to 80W in 0-360 coords
LAT_MIN, LAT_MAX = -5, 5
TIME_START, TIME_END = "1950", "2014"

# Strong El Nino peak years (DJF Jan/Feb year) for annotation markers
EL_NINO_PEAKS = [1957, 1965, 1972, 1982, 1987, 1991, 1994, 1997, 2002, 2009]


def open_cesm2_rsut(df, gcs):
    """Open CESM2 r1i1p1f1 historical rsut from CMIP6, trying gn then gr grid."""
    for grid_label in ("gn", "gr"):
        matches = df.query(
            "activity_id == 'CMIP' "
            "& experiment_id == 'historical' "
            "& source_id == 'CESM2' "
            "& member_id == 'r1i1p1f1' "
            "& variable_id == 'rsut' "
            "& table_id == 'Amon' "
            f"& grid_label == '{grid_label}'"
        )
        if not matches.empty:
            break
    if matches.empty:
        raise ValueError("No CESM2 historical rsut zarr store found in catalog.")
    print(f"  Found rsut store with grid_label='{grid_label}'")
    mapper = gcs.get_mapper(matches.iloc[0].zstore)
    return xr.open_zarr(mapper, consolidated=True).rsut


def weighted_lat_mean(da):
    """Area-weighted mean over latitude using cosine weights."""
    weights = np.cos(np.deg2rad(da.lat))
    return da.weighted(weights).mean("lat")


def format_lon_label(lon):
    if lon == 180:
        return "180°"
    if lon < 180:
        return f"{int(lon)}°E"
    return f"{int(360 - lon)}°W"


def build_hovmoller():
    print("Loading CMIP6 catalog...")
    df = pd.read_csv(CATALOG_URL)
    gcs = gcsfs.GCSFileSystem(token="anon")

    print("Opening CESM2 rsut dataset...")
    rsut = open_cesm2_rsut(df, gcs).sel(
        time=slice(TIME_START, TIME_END),
        lat=slice(LAT_MIN, LAT_MAX),
        lon=slice(LON_MIN, LON_MAX),
    )

    print("Computing climatology and anomalies...")
    clim = rsut.groupby("time.month").mean("time")
    anomaly = rsut.groupby("time.month") - clim

    print("Meridionally averaging and smoothing...")
    hovmoller = weighted_lat_mean(anomaly)
    hovmoller = hovmoller.rolling(time=3, center=True).mean()

    print("Loading into memory (this may take a minute)...")
    hovmoller = hovmoller.load()
    return hovmoller


def plot_hovmoller(hovmoller):
    year_frac = hovmoller.time.dt.year + (hovmoller.time.dt.month - 1) / 12
    lons = hovmoller.lon.values
    data = hovmoller.values  # shape: (time, lon)

    fig, ax = plt.subplots(figsize=(14, 9))

    mesh = ax.pcolormesh(
        lons,
        year_frac,
        data,
        cmap="RdBu_r",
        vmin=-20,
        vmax=20,
        shading="auto",
    )

    # Dashed horizontal lines at known strong El Nino peak years.
    # Draw black shadow first, then white on top so lines pop against all colormap values.
    ax.text(
        LON_MAX + 1.5, float(year_frac.min()) - 0.8,
        "El Niño\nyears →",
        va="top", ha="left", fontsize=8.5, color="#111111", style="italic",
    )
    for peak_year in EL_NINO_PEAKS:
        ax.axhline(peak_year, color="black", linewidth=2.0, linestyle="--", alpha=0.9)
        ax.axhline(peak_year, color="white", linewidth=1.2, linestyle="--", alpha=0.9)
        ax.text(
            LON_MAX + 1.5,
            peak_year,
            str(peak_year),
            va="center",
            ha="left",
            fontsize=10,
            color="#111111",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.75),
        )

    ax.set_xlim(LON_MIN, LON_MAX)
    ax.set_ylim(float(year_frac.min()), float(year_frac.max()))
    ax.invert_yaxis()

    xticks = [150, 160, 170, 180, 200, 220, 240, 260, 280]
    ax.set_xticks(xticks)
    ax.set_xticklabels([format_lon_label(t) for t in xticks], fontsize=9)
    ax.set_xlabel("Longitude", fontsize=11, labelpad=6)

    ax.set_yticks(range(1950, 2015, 5))
    ax.set_yticklabels([str(y) for y in range(1950, 2015, 5)], fontsize=9)
    ax.yaxis.set_minor_locator(mticker.MultipleLocator(1))
    ax.set_ylabel("Year", fontsize=11, labelpad=6)

    ax.grid(axis="x", color="white", linewidth=0.5, alpha=0.3)

    cbar = fig.colorbar(mesh, ax=ax, orientation="horizontal", pad=0.09, shrink=0.68, aspect=35)
    cbar.set_label("rsut anomaly (W m⁻²)", fontsize=10)

    ax.annotate(
        "El Niño 1997–98\n(strongest in record)",
        xy=(235, 1997.5),
        xytext=(165, 1993.5),
        arrowprops=dict(arrowstyle="->", color="#111111", lw=1.1),
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cfcfcf", lw=0.8),
        fontsize=9,
        ha="left",
    )

    fig.suptitle(
        "Equatorial Pacific Top-of-Atmosphere Shortwave Anomaly (5°S–5°N)",
        x=0.075,
        y=0.975,
        ha="left",
        fontsize=15,
        weight="bold",
    )
    fig.text(
        0.075,
        0.945,
        "CESM2 historical r1i1p1f1, 1950–2014  •  Hovmöller: Longitude × Time  "
        "•  3-month smoothed  •  Red = more outgoing SW (El Niño clearing)",
        ha="left",
        va="top",
        fontsize=9.5,
        color="#555555",
    )
    fig.text(
        0.075,
        0.022,
        "Data: Google Cloud CMIP6 zarr catalog, CESM2 historical r1i1p1f1. "
        "rsut = toa_outgoing_shortwave_flux (W m⁻²). "
        "Dashed lines mark strong El Niño peak years.",
        fontsize=8,
        color="#555555",
    )

    fig.tight_layout(rect=[0, 0.07, 0.97, 0.91])
    OUT_DIR.mkdir(exist_ok=True)
    fig.savefig(PDF_PATH, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {PDF_PATH}")


def main():
    hovmoller = build_hovmoller()
    plot_hovmoller(hovmoller)


if __name__ == "__main__":
    main()
