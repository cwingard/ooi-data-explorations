# OOI Data Quality Review Dashboard

A browser-based, interactive dashboard for Human-in-the-Loop (HITL) quality
control reviews of Ocean Observatories Initiative (OOI) data. Built with
[Panel](https://panel.holoviz.org/), [hvplot](https://hvplot.holoviz.org/),
and [Bokeh](https://bokeh.org/).

The dashboard supports visual inspection of time series and 2D (depth/spectral)
data against QARTOD gross range and climatology test limits, overlay of discrete
water sample comparisons, and authoring of quality annotations for submission
back to the OOI annotation system.

---

## Requirements

The dashboard requires the `dashboard` optional dependency group. Install it
alongside the main package:

```shell
# From the python/ directory, using pip:
pip install -e ".[dashboard]"

# Or with the full environment:
pip install -e ".[all]"
```

Using conda (recommended):

```shell
conda env create -f environment.yml
conda activate ooi
pip install -e ".[dashboard]"
```

**Dashboard dependencies** (installed automatically with `[dashboard]`):

| Package   | Minimum version |
|-----------|----------------|
| bokeh     | 3.4            |
| cmocean   | 3.0            |
| hvplot    | 0.10           |
| panel     | 1.4            |
| param     | 2.0            |

---

## Running the Dashboard

After installation, launch from any terminal with:

```shell
ooi-dashboard
```

This starts a local Panel server on port 5006 and opens the dashboard
automatically in your default browser at `http://localhost:5006/dashboard`.

Alternatively, run the module directly:

```shell
python -m ooi_data_explorations.apps.dashboard
```

---

## Interface Overview

The dashboard is divided into a **sidebar** (controls) and a **main panel**
(plots and annotations).

<!-- TODO: Insert screenshot of full dashboard layout here -->

### Sidebar

The sidebar is organized into collapsible accordion sections:

#### Select Data

Load data from the OOI Gold Copy THREDDS server or from a local NetCDF file.

- **GC THREDDS** -- Select site, node, sensor, delivery method, stream, and
  deployment from cascading dropdowns populated live from the M2M API. Adjust
  the file tag (regex) if needed, then click **Load from GC THREDDS**.
- **Local Files** -- Browse for a local `.nc` file with **Load data...**. Use
  **Append data...** to extend the time range by concatenating a second file.

<!-- TODO: Insert screenshot of Select Data section here -->

#### Variables & Display

Control what is plotted and how.

- **Plot Type** -- Switch between `Time Series` and `Heatmap` (heatmap requires
  a 2D variable with a depth or spectral dimension).
- **Variables** -- Select one or more science variables to plot. Use **All** /
  **None** to quickly select or deselect everything.
- **Color Variable** -- (Heatmap mode only) Choose the variable to encode as
  color.
- **Y Axis** -- (Heatmap mode only) Choose the dimension to place on the y-axis.
- **Standardize (z-score)** -- Normalize all selected variables to z-scores for
  cross-variable comparison on a shared axis.
- **Show QARTOD flags** -- Overlay scatter points (yellow = suspect, red = fail)
  where the embedded QARTOD result flags in the dataset indicate suspect or fail
  values.
- **Column range** -- (Time series mode with 2D variables) Set start, stride,
  and stop indices to control which channels/depth bins are plotted as separate
  lines.
- **Colormap** -- Select from cmocean and matplotlib colormaps.

<!-- TODO: Insert screenshot of Variables & Display section here -->

#### QARTOD

- **QARTOD Limits** -- Load a gross range CSV and/or a climatology CSV with
  their respective **Load...** buttons.
- **Map Variables** -- Map dataset variable names to their corresponding QARTOD test variable names.

<!-- TODO: Insert screenshot of QARTOD section here -->

#### Discrete Samples

Fetch discrete water sample data from the OOI raw data server or load from a
local CSV. Map sample columns to dataset variables for overlay on the time
series plots.

<!-- TODO: Insert screenshot of Discrete Samples section here -->

#### Configuration

Save and restore the full dashboard state (selected site/node/sensor, loaded
file paths, variable selections, and QARTOD mappings) to a JSON configuration
file. Useful for resuming a review session without re-selecting everything.

<!-- TODO: Insert screenshot of Configuration section here -->

---

### Main Panel

#### Plots

<!-- TODO: Insert screenshot of time series plot here -->

<!-- TODO: Insert screenshot of heatmap plot here -->

Time series mode displays one subplot per selected variable, all sharing a
linked x (time) axis. Each subplot shows:

- **Data lines** -- One line per deployment, color-coded; multiple delivery
  methods distinguished by line style.
- **Gross range limits** -- Horizontal dashed (suspect) and solid (fail) lines
  at the QARTOD limit values (1D variables only).
- **Climatology lines** -- Dotted step-function lines showing monthly min/max
  climatology bounds (1D variables only).
- **Limit scatter** -- Yellow (suspect) and red (fail) scatter points for values
  that exceed the loaded QARTOD limits.
- **Discrete sample overlay** -- Scatter points from bottle or CTD cast data.
- **Annotation shading** -- Colored background regions corresponding to existing
  quality annotations (green = pass, yellow = suspect, red = fail, etc.).

Heatmap mode renders a single 2D color plot for the selected variable.
Annotation shading is drawn on the heatmap; QARTOD overlays are not supported
in heatmap mode.

#### Annotations

<!-- TODO: Insert screenshot of annotation table here -->

The annotation table below the plots displays all quality annotations for the
selected reference designator. The toolbar provides:

**Row 1 -- Fetch and gap detection:**

| Control         | Action                                                             |
|-----------------|--------------------------------------------------------------------|
| Fetch Annotations | Pull existing annotations from the M2M API for the loaded RefDes |
| Min gap (hours) | Minimum gap duration to detect (default 72 h)                    |
| Find Gaps       | Scan the loaded dataset for gaps and auto-create annotation rows   |

**Row 2 -- File operations:**

| Control         | Action                                                                   |
|-----------------|--------------------------------------------------------------------------|
| (path display)  | Shows the path of the last loaded or exported annotation CSV             |
| Load CSV        | Load a previously exported annotation CSV; updates the path display      |
| Export CSV      | Save adds/modifications as a CSV for M2M submission; prompts for path    |
| Export Deletes  | Save annotation IDs flagged for deletion to a text file; prompts for path|

**Row 3 -- Table editing:**

| Button       | Action                                                        |
|--------------|---------------------------------------------------------------|
| Annotate     | Toggle tap-to-create mode; click start then end time on plot  |
| Add Row      | Insert a blank annotation row for manual entry                |
| Mark Deleted | Flag the selected row for deletion (greyed out, preserved)    |

Rows marked for deletion are preserved in the table and are included in the
Export Deletes output. They are not submitted via Export CSV.

---

## Workflow

A typical HITL review session follows this sequence:

1. **Load data** -- Use GC THREDDS or a local file.
2. **Load QARTOD limits** -- Load gross range and climatology CSVs if available.
3. **Select variables** -- Choose the science variables to review.
4. **Inspect plots** -- Look for anomalies relative to the limit lines and
   scatter overlays.
5. **Fetch annotations** -- Pull existing annotations from the M2M API.
6. **Author annotations** -- Use tap-to-create or Add Row to mark problem
   periods; set the qcFlag and annotation text.
7. **Export** -- Export the annotation CSV and/or delete list for submission.
8. **Save config** -- Save the session configuration for the next deployment.

---

## Notes

- M2M API access requires credentials in `~/.netrc`. See the main package
  `README.md` for setup instructions.
- Dashboard state (last configuration path) is persisted in
  `~/.ooidata/dashboard_config.json`.
- For large multi-deployment datasets, initial file load may take several
  seconds. All subsequent plot interactions are typically sub-second.
- 2D variables (e.g., ADCP velocity profiles, SPKIR spectral irradiance) are
  supported in both time series (per-channel lines) and heatmap modes. QARTOD
  limit lines are not drawn for multi-channel variables; limit scatter overlays
  are used instead.

---

## Credits

Designed and implemented by Christopher Wingard with Claude (Anthropic).
Inspired by the original MATLAB `data_reviews.mlapp`.
