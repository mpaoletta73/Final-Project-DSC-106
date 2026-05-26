# Final-Project-DSC-106

Static graph for the final project question: How does El Nino affect ocean weather conditions?

Generated output:

- `imgs/thomas_deitel_el_nino_tropical_pacific_sst_anomaly.pdf`
- `index.html` combines the five project graph ideas into one data-driven interactive visualization.
- `data/interactive_el_nino_dashboard.json` stores the compact browser dataset for the map layers, timeline, and regional summaries.

View the interactive dashboard:

```bash
python3 -m http.server 8000
```

Then open `http://127.0.0.1:8000/index.html`.

Reproduce the graph:

```bash
python scripts/generate_el_nino_ocean_sst_pdf.py
```

Rebuild the interactive dashboard data:

```bash
python scripts/generate_interactive_dashboard_data.py
```

The script follows the CMIP6 Google Cloud workflow from the course notebook: it reads the public zarr catalog, opens CESM2 historical sea-surface temperature data with `gcsfs` and `xarray`, classifies winters by the Nino 3.4 sea-surface-temperature anomaly, then maps the average tropical Pacific ocean warming pattern during El Nino winters.
