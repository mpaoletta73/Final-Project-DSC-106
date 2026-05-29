"""Build compact data for the interactive El Nino dashboard.

The dashboard needs one shared data file rather than five static figures. This
script follows the existing project notebooks/scripts, then exports small
Pacific-grid layers for SST, precipitation, cloud cover, radiation, and a
combined signal.
"""

from __future__ import annotations

import json
from pathlib import Path

import gcsfs
import numpy as np
import pandas as pd
import xarray as xr


CATALOG_URL = "https://storage.googleapis.com/cmip6/cmip6-zarr-consolidated-stores.csv"
MODEL = "CESM2"
MEMBER = "r1i1p1f1"
EXPERIMENT = "historical"
START_YEAR = "1950"
END_YEAR = "2014"
OUT_PATH = Path("data/interactive_el_nino_dashboard.json")


def find_store(df: pd.DataFrame, variable_id: str, table_id: str) -> pd.Series:
    matches = df.query(
        "activity_id == 'CMIP' and "
        "source_id == @MODEL and "
        "experiment_id == @EXPERIMENT and "
        "member_id == @MEMBER and "
        "variable_id == @variable_id and "
        "table_id == @table_id"
    ).copy()
    if matches.empty:
        raise ValueError(f"No CMIP6 store found for {variable_id} / {table_id}.")
    return matches.sort_values("version", ascending=False).iloc[0]


def open_store(gcs: gcsfs.GCSFileSystem, row: pd.Series) -> xr.Dataset:
    return xr.open_zarr(gcs.get_mapper(row.zstore), consolidated=True)


def area_weighted_mean_latlon(da: xr.DataArray) -> xr.DataArray:
    weights = np.cos(np.deg2rad(da.lat))
    weights = weights / weights.mean()
    return da.weighted(weights).mean(("lat", "lon"))


def year_month_labels(time_coord: xr.DataArray) -> xr.DataArray:
    labels = time_coord.dt.strftime("%Y-%m").values
    return xr.DataArray(labels, coords={"time": time_coord}, dims=("time",))


def format_model_month(value) -> str:
    return f"{int(value.year):04d}-{int(value.month):02d}"


def convert_units(da: xr.DataArray, variable_id: str) -> xr.DataArray:
    if variable_id == "pr":
        return da * 86400
    return da


def global_view(da: xr.DataArray) -> xr.DataArray:
    return da.sel(lat=slice(-65, 65))


def coarsen_layer(da: xr.DataArray, lat_step: int = 2, lon_step: int = 2) -> xr.DataArray:
    """Average nearby model cells into a smaller browser-friendly grid."""
    return da.coarsen(lat=lat_step, lon=lon_step, boundary="trim").mean()


def layer_records(da: xr.DataArray, value_key: str) -> list[dict[str, float | None]]:
    frame = da.to_dataframe(name=value_key).reset_index()
    records = []
    for row in frame.itertuples(index=False):
        value = getattr(row, value_key)
        records.append(
            {
                "lat": round(float(row.lat), 3),
                "lon": round(float(row.lon), 3),
                value_key: None if pd.isna(value) else round(float(value), 5),
            }
        )
    return records


def normalize(da: xr.DataArray) -> xr.DataArray:
    limit = float(np.nanpercentile(np.abs(da.values), 98))
    if not np.isfinite(limit) or limit == 0:
        return xr.zeros_like(da)
    return (da / limit).clip(-1, 1)


def composite_difference(ds: xr.Dataset, variable_id: str, el_keys, neutral_keys) -> xr.DataArray:
    da = convert_units(ds[variable_id].sel(time=slice(START_YEAR, END_YEAR)), variable_id)
    da = global_view(da)
    da = da.assign_coords(year_month=year_month_labels(da.time))
    el = da.where(da.year_month.isin(el_keys), drop=True).mean("time")
    neutral = da.where(da.year_month.isin(neutral_keys), drop=True).mean("time")
    return coarsen_layer((el - neutral).load())


def build() -> dict:
    print("Loading CMIP6 catalog...")
    df = pd.read_csv(CATALOG_URL)
    gcs = gcsfs.GCSFileSystem(token="anon")

    stores = {
        "tos": find_store(df, "tos", "Omon"),
        "pr": find_store(df, "pr", "Amon"),
        "clt": find_store(df, "clt", "Amon"),
        "rsut": find_store(df, "rsut", "Amon"),
    }

    print("Opening datasets...")
    ds_tos = open_store(gcs, stores["tos"])
    ds_pr = open_store(gcs, stores["pr"])
    ds_clt = open_store(gcs, stores["clt"])
    ds_rsut = open_store(gcs, stores["rsut"])

    print("Computing Nino 3.4 monthly index...")
    tos = ds_tos["tos"].sel(time=slice(START_YEAR, END_YEAR))
    nino34 = area_weighted_mean_latlon(tos.sel(lat=slice(-5, 5), lon=slice(190, 240)))
    nino34_anom = nino34.groupby("time.month") - nino34.groupby("time.month").mean("time")
    nino34_anom = nino34_anom.load()
    nino34_anom = nino34_anom.assign_coords(year_month=year_month_labels(nino34_anom.time))
    el_keys = nino34_anom.where(nino34_anom >= 0.5, drop=True).year_month.values
    neutral_keys = nino34_anom.where(abs(nino34_anom) < 0.5, drop=True).year_month.values

    print("Building map layers...")
    sst_layer = composite_difference(ds_tos, "tos", el_keys, neutral_keys)
    pr_layer = composite_difference(ds_pr, "pr", el_keys, neutral_keys)
    clt_layer = composite_difference(ds_clt, "clt", el_keys, neutral_keys)

    print("Building radiation layer...")
    rsut_layer = composite_difference(ds_rsut, "rsut", el_keys, neutral_keys)

    print("Building combined signal...")
    pr_on_sst = pr_layer.interp(lat=sst_layer.lat, lon=sst_layer.lon)
    clt_on_sst = clt_layer.interp(lat=sst_layer.lat, lon=sst_layer.lon)
    rsut_on_sst = rsut_layer.interp(lat=sst_layer.lat, lon=sst_layer.lon)
    combined = (
        normalize(sst_layer) * 0.35
        + normalize(pr_on_sst) * 0.25
        + normalize(clt_on_sst) * 0.2
        + normalize(rsut_on_sst) * 0.2
    )

    print("Building timeline...")
    timeline = (
        nino34_anom.to_dataframe(name="nino34_anomaly_c")
        .reset_index()[["time", "nino34_anomaly_c"]]
    )
    timeline["date"] = timeline["time"].map(format_model_month)
    timeline["year"] = timeline["time"].map(lambda value: int(value.year))
    timeline["phase"] = np.select(
        [
            timeline["nino34_anomaly_c"] >= 0.5,
            timeline["nino34_anomaly_c"] <= -0.5,
        ],
        ["el_nino", "la_nina"],
        default="neutral",
    )

    regions = {
        "West Pacific": {"lon": slice(120, 170), "lat": slice(-15, 15)},
        "Central Pacific": {"lon": slice(170, 220), "lat": slice(-15, 15)},
        "East Pacific": {"lon": slice(220, 280), "lat": slice(-15, 15)},
    }
    regional = []
    for region, bounds in regions.items():
        regional.append(
            {
                "region": region,
                "sst_anomaly_c": round(float(area_weighted_mean_latlon(sst_layer.sel(**bounds))), 4),
                "precip_mm_day": round(float(area_weighted_mean_latlon(pr_layer.sel(**bounds))), 4),
                "cloud_pct_points": round(float(area_weighted_mean_latlon(clt_layer.sel(**bounds))), 4),
                "rsut_w_m2": round(float(area_weighted_mean_latlon(rsut_layer.sel(**bounds))), 4),
                "combined": round(float(area_weighted_mean_latlon(combined.sel(**bounds))), 4),
            }
        )

    return {
        "meta": {
            "model": MODEL,
            "member": MEMBER,
            "experiment": EXPERIMENT,
            "years": "1950-2014",
            "el_nino_months": int(len(el_keys)),
            "neutral_months": int(len(neutral_keys)),
            "nino34_threshold_c": 0.5,
        },
        "layers": {
            "combined": layer_records(combined, "combined"),
            "sst": layer_records(sst_layer, "sst_anomaly_c"),
            "precip": layer_records(pr_layer, "precip_mm_day"),
            "cloud": layer_records(clt_layer, "cloud_pct_points"),
            "rsut": layer_records(rsut_layer, "rsut_w_m2"),
        },
        "timeline": [
            {
                "date": row.date,
                "year": int(row.year),
                "nino34_anomaly_c": round(float(row.nino34_anomaly_c), 4),
                "phase": row.phase,
            }
            for row in timeline.itertuples(index=False)
        ],
        "regional": regional,
    }


def main() -> None:
    OUT_PATH.parent.mkdir(exist_ok=True)
    data = build()
    OUT_PATH.write_text(json.dumps(data, separators=(",", ":")))
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
