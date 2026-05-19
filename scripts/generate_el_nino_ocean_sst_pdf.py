"""Generate a static PDF chart about El Nino's effect on ocean conditions.

This follows the CMIP6 Google Cloud loading pattern from the course notebook:
read the public zarr catalog, open stores with gcsfs, and use xarray for the
subsetting and averaging.
"""

from pathlib import Path

import gcsfs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr


CATALOG_URL = "https://storage.googleapis.com/cmip6/cmip6-zarr-consolidated-stores.csv"
OUT_DIR = Path("imgs")
DATA_DIR = Path("data")
PDF_PATH = OUT_DIR / "thomas_deitel_el_nino_tropical_pacific_sst_anomaly.pdf"
CSV_PATH = DATA_DIR / "cesm2_el_nino_tropical_pacific_sst_anomaly_1951_2014.csv"


def open_cesm2_tos(df, gcs):
    """Open CESM2 r1i1p1f1 historical sea-surface temperature from CMIP6."""
    matches = df.query(
        "activity_id == 'CMIP' "
        "& experiment_id == 'historical' "
        "& source_id == 'CESM2' "
        "& member_id == 'r1i1p1f1' "
        "& variable_id == 'tos' "
        "& table_id == 'Omon' "
        "& grid_label == 'gr'"
    )
    if matches.empty:
        raise ValueError("No CESM2 historical tos zarr store found.")

    mapper = gcs.get_mapper(matches.iloc[0].zstore)
    return xr.open_zarr(mapper, consolidated=True).tos


def weighted_box_mean(da, lat_min, lat_max, lon_min, lon_max):
    """Area-weighted regional mean using cosine latitude weights."""
    subset = da.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))
    weights = np.cos(np.deg2rad(subset.lat))
    return subset.weighted(weights).mean(("lat", "lon"))


def djf_mean_by_winter_year(da):
    """Group December-January-February values by the year of Jan/Feb."""
    djf = da.where(da.time.dt.month.isin([12, 1, 2]), drop=True)
    winter_year = xr.where(djf.time.dt.month == 12, djf.time.dt.year + 1, djf.time.dt.year)
    djf = djf.assign_coords(winter_year=("time", winter_year.data))
    return djf.groupby("winter_year").mean("time")


def build_dataset():
    df = pd.read_csv(CATALOG_URL)
    gcs = gcsfs.GCSFileSystem(token="anon")
    tos = open_cesm2_tos(df, gcs).sel(time=slice("1950", "2014"))

    # Nino 3.4 region: 5S-5N, 170W-120W. CMIP longitudes are 0-360.
    nino34 = weighted_box_mean(tos, lat_min=-5, lat_max=5, lon_min=190, lon_max=240)
    nino34_clim = nino34.groupby("time.month").mean("time")
    nino34_anomaly = nino34.groupby("time.month") - nino34_clim
    nino34_djf = djf_mean_by_winter_year(nino34_anomaly).sel(winter_year=slice(1951, 2014)).load()

    tropical_pacific = tos.sel(lat=slice(-25, 25), lon=slice(120, 290))
    monthly_climatology = tropical_pacific.groupby("time.month").mean("time")
    sst_anomaly = tropical_pacific.groupby("time.month") - monthly_climatology
    sst_djf = djf_mean_by_winter_year(sst_anomaly).sel(winter_year=slice(1951, 2014))

    el_nino_years = nino34_djf.where(nino34_djf >= 0.5, drop=True).winter_year
    el_nino_sst_anomaly = sst_djf.sel(winter_year=el_nino_years).mean("winter_year").load()

    nino34_el_nino_mean = float(nino34_djf.sel(winter_year=el_nino_years).mean().load())
    output = el_nino_sst_anomaly.to_dataframe(name="sst_anomaly_c").reset_index()
    output["mean_el_nino_nino34_anomaly_c"] = nino34_el_nino_mean
    output["el_nino_winter_count"] = len(el_nino_years)
    return el_nino_sst_anomaly, nino34_el_nino_mean, len(el_nino_years), output


def format_lon_label(lon):
    if lon == 180:
        return "180"
    if lon < 180:
        return f"{int(lon)}E"
    return f"{int(360 - lon)}W"


def plot_pdf(sst_anomaly, nino34_mean, winter_count):
    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    levels = np.linspace(-1.8, 1.8, 19)
    mesh = ax.contourf(
        sst_anomaly.lon,
        sst_anomaly.lat,
        sst_anomaly,
        levels=levels,
        cmap="RdBu_r",
        extend="both",
    )
    contours = ax.contour(
        sst_anomaly.lon,
        sst_anomaly.lat,
        sst_anomaly,
        levels=[0.5, 1.0, 1.5],
        colors="#222222",
        linewidths=0.7,
        alpha=0.65,
    )
    ax.clabel(contours, inline=True, fontsize=8, fmt="%.1fC")

    # Mark the Nino 3.4 region used to classify El Nino winters.
    ax.plot([190, 240, 240, 190, 190], [-5, -5, 5, 5, -5], color="#111111", lw=1.4)
    ax.text(192, 7.5, "Nino 3.4 region", fontsize=10, weight="bold", color="#111111")

    ax.annotate(
        f"El Nino winters warm this ocean region\nby {nino34_mean:.1f}C on average",
        xy=(216, 0),
        xytext=(126, 17),
        arrowprops=dict(arrowstyle="->", color="#222222", lw=1.2),
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#cfcfcf", lw=0.8),
        fontsize=10.5,
        ha="left",
        va="center",
    )

    cbar = fig.colorbar(mesh, ax=ax, pad=0.025, shrink=0.9)
    cbar.set_label("Sea-surface temperature anomaly (C)")

    fig.suptitle(
        "El Nino Warms the Central and Eastern Tropical Pacific",
        x=0.085,
        y=0.985,
        ha="left",
        fontsize=17,
        weight="bold",
    )
    fig.text(
        0.085,
        0.94,
        f"Average DJF sea-surface temperature anomaly during {winter_count} CESM2 El Nino winters, 1951-2014",
        ha="left",
        va="top",
        fontsize=10.5,
        color="#555555",
    )
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    xticks = [120, 150, 180, 210, 240, 270, 290]
    ax.set_xticks(xticks)
    ax.set_xticklabels([format_lon_label(tick) for tick in xticks])
    ax.set_yticks([-20, -10, 0, 10, 20])
    ax.set_yticklabels(["20S", "10S", "0", "10N", "20N"])
    ax.set_xlim(120, 290)
    ax.set_ylim(-25, 25)
    ax.grid(color="white", linewidth=0.8, alpha=0.45)

    fig.text(
        0.01,
        0.02,
        "Data: Google Cloud CMIP6 zarr catalog, CESM2 historical r1i1p1f1. "
        "El Nino winters are classified when DJF Nino 3.4 SST anomaly >= 0.5C.",
        fontsize=8.5,
        color="#555555",
    )
    fig.tight_layout(rect=[0, 0.05, 1, 0.90])
    OUT_DIR.mkdir(exist_ok=True)
    fig.savefig(PDF_PATH, format="pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    DATA_DIR.mkdir(exist_ok=True)
    sst_anomaly, nino34_mean, winter_count, output = build_dataset()
    output.to_csv(CSV_PATH, index=False)
    plot_pdf(sst_anomaly, nino34_mean, winter_count)
    print(f"Wrote {PDF_PATH}")
    print(f"Wrote {CSV_PATH}")


if __name__ == "__main__":
    main()
