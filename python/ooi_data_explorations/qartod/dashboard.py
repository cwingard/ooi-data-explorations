#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author Christopher Wingard
@brief Panel-based interactive dashboard for HITL review of QARTOD test
    limits and annotations for OOI data. Run with:

        cd python
        python ooi_data_explorations/qartod/dashboard.py
"""
import json
import os
from pathlib import Path

import pandas as pd
import panel as pn
import param
import xarray as xr

from ooi_data_explorations.common import (
    get_annotations,
    get_vocabulary,
    list_deployments,
    list_methods,
    list_nodes,
    list_sensors,
    list_sites,
    list_streams,
    load_gc_thredds,
)
from ooi_data_explorations.qartod.qc_processing import ANNO_HEADER

pn.extension("tabulator", sizing_mode="stretch_width")

CONFIG_PATH = Path.home() / ".ooidata" / "dashboard_config.json"
DRAFT_PATH = Path.home() / ".ooidata" / "dashboard_draft.json"
EXPORT_DIR = Path.home() / "ooidata"

_DEFAULT_CONFIG: dict = {
    "site": None,
    "node": None,
    "sensor": None,
    "methods": [],
    "deployment": 1,
    "data_path": "",
    "limits_path": "",
    "colormap": "cmo.balance",
}

_CASCADE_ORDER = ("node", "sensor", "method", "stream")

_VAR_EXCLUDE_PATTERNS = (
    "qartod", "qc_", "quality_flag", "annotations_qc", "rollup_", "provenance",
)
_VAR_EXCLUDE_NAMES = frozenset({
    "time", "obs", "lat", "lon", "latitude", "longitude",
    "depth", "pressure", "deployment", "id",
})

_SAMPLE_META_COLS = frozenset({
    "Start Time [UTC]", "Latitude [degrees_north]", "Longitude [degrees_east]",
    "Start Latitude [degrees]", "Start Longitude [degrees]",
    "CTD Depth [m]", "Cast", "Station", "Niskin Bottle",
})

_CMOCEAN_CMAPS = [
    "balance", "delta", "curl", "diff", "tarn",
    "thermal", "haline", "solar", "ice", "oxy", "deep", "dense",
    "algae", "matter", "turbid", "speed", "amp", "tempo", "rain",
    "phase", "topo", "gray",
]
_MPL_CMAPS = [
    "RdBu_r", "Spectral_r", "coolwarm", "seismic",
    "viridis", "plasma", "inferno", "magma", "cividis",
]

_ANNO_COLS = ANNO_HEADER + ["deleted"]
_QC_FLAG_STRINGS = [None, "pass", "suspect", "fail", "not_operational", "not_available"]


def _available_cmaps() -> list[str]:
    """
    Build colormap list. Prefers cmocean (prefixed 'cmo.' for hvplot
    compatibility) when installed; appends matplotlib fallbacks regardless.
    """
    cmaps: list[str] = []
    try:
        import cmocean  # noqa: F401

        cmaps.extend(f"cmo.{n}" for n in _CMOCEAN_CMAPS)
    except ImportError:
        pass
    cmaps.extend(_MPL_CMAPS)
    return cmaps


_CMAPS = _available_cmaps()
_DEFAULT_CMAP = "cmo.balance" if "cmo.balance" in _CMAPS else (_CMAPS[0] if _CMAPS else "RdBu_r")


def load_config() -> dict:
    """Load persisted dashboard state, falling back to defaults on failure."""
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH) as f:
                return {**_DEFAULT_CONFIG, **json.load(f)}
        except (json.JSONDecodeError, OSError):
            pass
    return dict(_DEFAULT_CONFIG)


def save_config(cfg: dict) -> None:
    """Write dashboard state to disk."""
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_PATH, "w") as f:
        json.dump(cfg, f, indent=2)


def _browse_file(
    title: str,
    filetypes: list[tuple[str, str]],
    initial_dir: str | None = None,
) -> str | None:
    """
    Open a native OS file picker dialog via tkinter. Blocking call -- safe
    for a single-user local Panel server where brief IOLoop blocking is
    acceptable (no other sessions to serve while the dialog is open).
    Returns the selected path, or None if cancelled.
    """
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.wm_attributes("-topmost", 1)
        path = filedialog.askopenfilename(
            title=title,
            initialdir=initial_dir or str(Path.home()),
            filetypes=filetypes,
        )
        root.destroy()
        return path or None
    except Exception:
        return None


def _default_method(available: list[str]) -> str | None:
    """
    Prefer 'streamed' (cabled array) then 'telemetered'; fall back to the
    first available method.
    """
    for preferred in ("streamed", "telemetered"):
        if preferred in available:
            return preferred
    return available[0] if available else None


def _science_variables(ds: xr.Dataset) -> list[str]:
    """Return science variable names, excluding QC flags and metadata."""
    result = []
    for v in ds.data_vars:
        if v in _VAR_EXCLUDE_NAMES:
            continue
        if any(p in v for p in _VAR_EXCLUDE_PATTERNS):
            continue
        result.append(v)
    return sorted(result)


def _sample_measurement_cols(df: pd.DataFrame) -> list[str]:
    """Return measurement column names from a discrete samples DataFrame."""
    return sorted(c for c in df.columns if c not in _SAMPLE_META_COLS)


def _array_from_site(site: str | None) -> str:
    """Map an OOI site code prefix to its array name for discrete samples."""
    if site is None:
        return "Endurance"
    prefixes = {
        "CE": "Endurance",
        "CP": "Pioneer",
        "GA": "Global_Argentine",
        "GI": "Global_Irminger",
        "GP": "Global_Station_Papa",
        "GS": "Global_Southern",
        "RS": "Cabled",
    }
    for prefix, array in prefixes.items():
        if site.startswith(prefix):
            return array
    return "Endurance"


class OOIDashboard(param.Parameterized):
    """
    Stateful Panel dashboard for reviewing OOI QARTOD test limits and
    annotations. All cascade logic and data loading lives here; layout
    methods assemble widgets into Panel components.
    """

    # Reference designator cascade
    site = param.Selector(default=None, objects=[None])
    node = param.Selector(default=None, objects=[None])
    sensor = param.Selector(default=None, objects=[None])
    method = param.Selector(default=None, objects=[None])
    stream = param.Selector(default=None, objects=[None])
    deployment = param.Selector(default=None, objects=[None])

    # File paths persisted to config; updated by the browse callbacks
    data_path = param.String(default="")
    limits_path = param.String(default="")

    # Display controls
    variables = param.ListSelector(default=[], objects=[])
    plot_type = param.Selector(default="timeseries", objects=["timeseries", "heatmap"])
    normalize = param.Boolean(default=False)
    color_var = param.Selector(default=None, objects=[None])
    colormap = param.Selector(default=_DEFAULT_CMAP, objects=_CMAPS)

    # Discrete samples
    sample_columns = param.ListSelector(default=[], objects=[])

    def __init__(self, **params):
        super().__init__(**params)
        self._cfg = load_config()
        self._data: dict[str, xr.Dataset] = {}
        self._limits = None
        self._samples: pd.DataFrame | None = None
        self._samples_csv_path: str = ""
        self._append_path: str = ""
        self._anno_df: pd.DataFrame = pd.DataFrame(columns=_ANNO_COLS)
        self._deleted_ids: set[int] = set()

        saved_cmap = self._cfg.get("colormap", _DEFAULT_CMAP)
        if saved_cmap in _CMAPS:
            self.colormap = saved_cmap

        # -- Status --
        self._status = pn.pane.Alert("Initializing...", alert_type="info")
        self._spinner = pn.indicators.LoadingSpinner(
            value=False, size=20, visible=False
        )

        # -- Cascade widgets --
        self._w_site = pn.widgets.Select.from_param(self.param.site, name="Site")
        self._w_node = pn.widgets.Select.from_param(self.param.node, name="Node")
        self._w_sensor = pn.widgets.Select.from_param(
            self.param.sensor, name="Sensor"
        )
        self._w_method = pn.widgets.Select.from_param(
            self.param.method, name="Method"
        )
        self._w_stream = pn.widgets.Select.from_param(
            self.param.stream, name="Stream"
        )
        self._w_deploy = pn.widgets.Select.from_param(
            self.param.deployment, name="Deployment"
        )
        self._w_tag = pn.widgets.TextInput(
            name="File tag (regex)",
            value=".*\\.nc$",
            sizing_mode="stretch_width",
        )

        # -- File path displays (disabled TextInput) + Browse buttons --
        self._w_data_path_display = pn.widgets.TextInput(
            name="",
            placeholder="No file selected",
            disabled=True,
            sizing_mode="stretch_width",
        )
        self._btn_browse_data = pn.widgets.Button(
            name="Browse...", button_type="light", width=90
        )
        self._btn_browse_data.on_click(self._on_browse_data)

        self._w_limits_path_display = pn.widgets.TextInput(
            name="",
            placeholder="No file selected",
            disabled=True,
            sizing_mode="stretch_width",
        )
        self._btn_browse_limits = pn.widgets.Button(
            name="Browse...", button_type="light", width=90
        )
        self._btn_browse_limits.on_click(self._on_browse_limits)

        self._w_samples_path_display = pn.widgets.TextInput(
            name="",
            placeholder="No file selected",
            disabled=True,
            sizing_mode="stretch_width",
        )
        self._btn_browse_samples = pn.widgets.Button(
            name="Browse...", button_type="light", width=90
        )
        self._btn_browse_samples.on_click(self._on_browse_samples)

        self._w_append_path_display = pn.widgets.TextInput(
            name="",
            placeholder="No file selected",
            disabled=True,
            sizing_mode="stretch_width",
        )
        self._btn_browse_append = pn.widgets.Button(
            name="Browse...", button_type="light", width=90
        )
        self._btn_browse_append.on_click(self._on_browse_append)

        # -- Gap detection controls --
        self._w_gap_threshold = pn.widgets.FloatInput(
            name="Min gap (hours)", value=72.0, step=1.0, width=160
        )
        self._btn_find_gaps = pn.widgets.Button(
            name="Find Gaps", button_type="default"
        )
        self._btn_find_gaps.on_click(self._find_gaps)

        # -- Display control widgets --
        self._w_variables = pn.widgets.CheckBoxGroup.from_param(
            self.param.variables, name="Variables"
        )
        self._w_plot_type = pn.widgets.RadioButtonGroup(
            name="Plot Type",
            options={"Time Series": "timeseries", "Heatmap": "heatmap"},
            value=self.plot_type,
            button_type="default",
            button_style="outline",
        )
        self._w_plot_type.param.watch(
            lambda e: setattr(self, "plot_type", e.new), "value"
        )
        self._w_normalize = pn.widgets.Toggle.from_param(
            self.param.normalize, name="Normalize (0-1)", button_type="default"
        )
        self._w_color_var = pn.widgets.Select.from_param(
            self.param.color_var, name="Color Variable"
        )
        self._w_color_var.visible = False
        self._w_colormap = pn.widgets.Select.from_param(
            self.param.colormap, name="Colormap"
        )

        self._btn_var_all = pn.widgets.Button(
            name="All", button_type="light", width=60
        )
        self._btn_var_all.on_click(
            lambda e: setattr(
                self, "variables", list(self.param["variables"].objects)
            )
        )
        self._btn_var_none = pn.widgets.Button(
            name="None", button_type="light", width=60
        )
        self._btn_var_none.on_click(lambda e: setattr(self, "variables", []))

        # -- Discrete sample filter widgets --
        self._w_depth_min = pn.widgets.FloatInput(
            name="Min depth (m)", value=0.0, step=1.0, width=130
        )
        self._w_depth_max = pn.widgets.FloatInput(
            name="Max depth (m)", value=6000.0, step=1.0, width=130
        )
        self._chk_loc_filter = pn.widgets.Checkbox(
            name="Apply location filter", value=False
        )
        self._w_sample_lat = pn.widgets.FloatInput(
            name="Latitude", value=0.0, step=0.0001, width=130, visible=False
        )
        self._w_sample_lon = pn.widgets.FloatInput(
            name="Longitude", value=0.0, step=0.0001, width=130, visible=False
        )
        self._w_sample_radius = pn.widgets.FloatInput(
            name="Radius (km)", value=5.0, step=0.5, width=130, visible=False
        )
        self._chk_loc_filter.param.watch(self._on_loc_filter_toggle, "value")

        self._w_sample_cols = pn.widgets.CheckBoxGroup.from_param(
            self.param.sample_columns, name="Sample Columns"
        )
        self._btn_sample_all = pn.widgets.Button(
            name="All", button_type="light", width=60
        )
        self._btn_sample_all.on_click(
            lambda e: setattr(
                self, "sample_columns", list(self.param["sample_columns"].objects)
            )
        )
        self._btn_sample_none = pn.widgets.Button(
            name="None", button_type="light", width=60
        )
        self._btn_sample_none.on_click(lambda e: setattr(self, "sample_columns", []))

        # -- Annotation Tabulator --
        self._anno_table = pn.widgets.Tabulator(
            self._anno_df,
            show_index=False,
            pagination="local",
            page_size=15,
            sizing_mode="stretch_width",
            editors=self._anno_editors(),
            widths={"annotation": 350, "source": 180, "parameters": 120},
            formatters={"annotation": "textarea"},
        )

        # -- Buttons: data loading --
        self._btn_m2m = pn.widgets.Button(
            name="Load from GC THREDDS", button_type="primary"
        )
        self._btn_m2m.on_click(self._load_from_m2m)
        self._btn_file = pn.widgets.Button(
            name="Load file", button_type="default"
        )
        self._btn_file.on_click(self._load_from_file)
        self._btn_append_file = pn.widgets.Button(
            name="Append file", button_type="default"
        )
        self._btn_append_file.on_click(self._append_from_file)
        self._btn_limits = pn.widgets.Button(
            name="Load limits", button_type="default"
        )
        self._btn_limits.on_click(self._load_limits)

        # -- Buttons: discrete samples --
        self._btn_fetch_samples = pn.widgets.Button(
            name="Fetch from Alfresco", button_type="default"
        )
        self._btn_fetch_samples.on_click(self._fetch_discrete_samples)
        self._btn_load_samples_csv = pn.widgets.Button(
            name="Load from CSV", button_type="default"
        )
        self._btn_load_samples_csv.on_click(self._load_samples_from_file)

        # -- Buttons: annotations --
        self._btn_fetch_anno = pn.widgets.Button(
            name="Fetch Annotations", button_type="default"
        )
        self._btn_fetch_anno.on_click(self._fetch_annotations)
        self._btn_add_anno = pn.widgets.Button(
            name="Add Row", button_type="success", width=95
        )
        self._btn_add_anno.on_click(self._add_anno_row)
        self._btn_del_anno = pn.widgets.Button(
            name="Mark Deleted", button_type="warning", width=110
        )
        self._btn_del_anno.on_click(self._mark_anno_deleted)
        self._btn_save_draft = pn.widgets.Button(
            name="Save Draft", button_type="default", width=95
        )
        self._btn_save_draft.on_click(self._save_draft)
        self._btn_export_anno = pn.widgets.Button(
            name="Export CSV", button_type="primary", width=95
        )
        self._btn_export_anno.on_click(self._export_annotations)
        self._btn_export_dels = pn.widgets.Button(
            name="Export Deletes", button_type="primary", width=110
        )
        self._btn_export_dels.on_click(self._export_deletes)

        self._plot_pane = pn.pane.Markdown(
            "*Load data to display plots.*", styles={"color": "#888"}
        )

        self._init_sites()

    # ------------------------------------------------------------------
    # Cascade
    # ------------------------------------------------------------------

    def _clear_from(self, start: str) -> None:
        """
        Reset cascade selectors at and downstream of start without
        triggering downstream watchers for selectors already at None.
        Also resets deployment when sensor or above is cleared.
        """
        clearing = False
        for name in _CASCADE_ORDER:
            if name == start:
                clearing = True
            if clearing:
                self.param[name].objects = [None]
                if getattr(self, name) is not None:
                    setattr(self, name, None)
        # Deployment options are populated at sensor level; reset them
        # whenever sensor or anything above in the cascade is cleared.
        if start in ("node", "sensor"):
            self.param["deployment"].objects = [None]
            if self.deployment is not None:
                self.deployment = None

    def _init_sites(self) -> None:
        """
        Populate the site selector on startup and kick off cascade
        restoration from the persisted config.
        """
        self.data_path = self._cfg.get("data_path", "")
        self.limits_path = self._cfg.get("limits_path", "")

        # Restore last-used paths into the display widgets
        if self.data_path:
            self._w_data_path_display.value = self.data_path
        if self.limits_path:
            self._w_limits_path_display.value = self.limits_path

        self._set_status("Loading sites...", "info")
        try:
            sites = sorted(list_sites())
        except Exception as e:
            self._set_status(f"Failed to load sites: {e}", "danger")
            return

        self.param["site"].objects = [None] + sites
        saved = self._cfg.get("site")
        if saved in sites:
            self.site = saved
        else:
            self._set_status("Ready.", "info")

    @param.depends("site", watch=True)
    def _on_site(self) -> None:
        self._clear_from("node")
        if self.site is None:
            self._set_status("Ready.", "info")
            return
        self._set_status(f"Loading nodes for {self.site}...", "info")
        try:
            nodes = sorted(list_nodes(self.site))
        except Exception as e:
            self._set_status(f"Failed to load nodes: {e}", "danger")
            return
        self.param["node"].objects = [None] + nodes
        saved = self._cfg.get("node")
        if saved in nodes:
            self.node = saved
        else:
            self._set_status("Ready.", "info")

    @param.depends("node", watch=True)
    def _on_node(self) -> None:
        self._clear_from("sensor")
        if self.node is None:
            return
        self._set_status(
            f"Loading sensors for {self.site}/{self.node}...", "info"
        )
        try:
            sensors = sorted(list_sensors(self.site, self.node))
        except Exception as e:
            self._set_status(f"Failed to load sensors: {e}", "danger")
            return
        self.param["sensor"].objects = [None] + sensors
        saved = self._cfg.get("sensor")
        if saved in sensors:
            self.sensor = saved
        else:
            self._set_status("Ready.", "info")

    @param.depends("sensor", watch=True)
    def _on_sensor(self) -> None:
        """Load methods and deployment list when sensor is selected."""
        self._clear_from("method")
        if self.sensor is None:
            return

        self._set_status("Loading delivery methods...", "info")
        try:
            available = sorted(list_methods(self.site, self.node, self.sensor))
        except Exception as e:
            self._set_status(f"Failed to load methods: {e}", "danger")
            return

        if not available:
            self._set_status("No delivery methods found.", "warning")
            return

        # Deployment selector: "All" + each numbered deployment
        try:
            deploys = list_deployments(self.site, self.node, self.sensor)
            deploy_opts = ["All"] + [str(d) for d in sorted(int(d) for d in deploys)]
        except Exception:
            deploy_opts = ["All"]
        self.param["deployment"].objects = deploy_opts
        saved_d = self._cfg.get("deployment")
        self.deployment = saved_d if saved_d in deploy_opts else "All"

        self.param["method"].objects = [None] + available
        saved_m = self._cfg.get("method")
        self.method = saved_m if saved_m in available else _default_method(available)
        self._prefill_sample_filters()
        # Explicitly invoke _on_method as a safety net in case the param
        # watch does not fire when method is set to its current value.
        if self.method is not None:
            self._on_method()

    @param.depends("method", watch=True)
    def _on_method(self) -> None:
        """Load streams for the selected method."""
        self.param["stream"].objects = [None]
        if self.stream is not None:
            self.stream = None
        if self.method is None:
            return

        self._set_status(f"Loading streams for {self.method}...", "info")
        try:
            streams = sorted(list_streams(
                self.site, self.node, self.sensor, self.method
            ))
        except Exception as e:
            self._set_status(f"Failed to load streams: {e}", "danger")
            return

        if not streams:
            self._set_status("No streams found for this method.", "warning")
            return

        self.param["stream"].objects = [None] + streams
        saved_s = self._cfg.get("stream")
        self.stream = saved_s if saved_s in streams else streams[0]
        self._set_status("Ready.", "info")

    @param.depends("deployment", watch=True)
    def _on_deployment(self) -> None:
        """Auto-fill the tag field when deployment changes."""
        if self.deployment is None or self.deployment == "All":
            self._w_tag.value = ".*\\.nc$"
        else:
            self._w_tag.value = "deployment{:04d}.*\\.nc$".format(
                int(self.deployment)
            )

    # ------------------------------------------------------------------
    # File browse callbacks
    # ------------------------------------------------------------------

    def _on_browse_data(self, event=None) -> None:
        initial = str(Path(self.data_path).parent) if self.data_path else None
        path = _browse_file(
            title="Select NetCDF data file",
            filetypes=[("NetCDF files", "*.nc"), ("All files", "*.*")],
            initial_dir=initial,
        )
        if path:
            self.data_path = path
            self._w_data_path_display.value = path

    def _on_browse_limits(self, event=None) -> None:
        initial = str(Path(self.limits_path).parent) if self.limits_path else None
        path = _browse_file(
            title="Select QARTOD limits CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initial_dir=initial,
        )
        if path:
            self.limits_path = path
            self._w_limits_path_display.value = path

    def _on_browse_samples(self, event=None) -> None:
        path = _browse_file(
            title="Select discrete samples CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if path:
            self._samples_csv_path = path
            self._w_samples_path_display.value = path

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_from_m2m(self, event=None) -> None:
        """
        Fetch the selected deployment from the Gold Copy THREDDS server
        for each chosen delivery method.
        """
        if not all([self.site, self.node, self.sensor, self.method, self.stream]):
            self._set_status(
                "Complete the cascade: site, node, sensor, method, and stream "
                "must all be selected.",
                "warning",
            )
            return

        self._set_busy(True)
        tag = self._w_tag.value.strip() or ".*\\.nc$"
        catalog_id = "-".join([
            self.site, self.node, self.sensor, self.method, self.stream
        ])
        self._set_status(
            f"Loading {catalog_id} (deployment: {self.deployment})...", "info"
        )
        try:
            ds = load_gc_thredds(
                self.site, self.node, self.sensor,
                self.method, self.stream, tag,
            )
            if ds is None or len(ds.data_vars) == 0:
                self._set_status(
                    f"No files matched for {catalog_id} "
                    f"(deployment: {self.deployment}). "
                    "Check that the stream name is correct and the deployment exists.",
                    "warning",
                )
                return

            self._data = {self.method: ds}
            self._populate_variables()
            self._persist_config()
            t0 = pd.Timestamp(ds.time.values[0]).strftime("%Y-%m-%d")
            t1 = pd.Timestamp(ds.time.values[-1]).strftime("%Y-%m-%d")
            self._set_status(
                f"Loaded {catalog_id} -- {t0} to {t1}.", "success"
            )
        except Exception as e:
            self._set_status(f"Load failed: {e}", "danger")
        finally:
            self._set_busy(False)

    def _load_from_file(self, event=None) -> None:
        """Open a local NetCDF file as the active dataset."""
        path = self.data_path.strip()
        if not path or not os.path.exists(path):
            self._set_status("Browse for a NetCDF file first.", "warning")
            return
        self._set_busy(True)
        try:
            self._data = {"file": xr.open_dataset(path)}
            self._populate_variables()
            self._persist_config()
            self._set_status(f"Loaded {os.path.basename(path)}.", "success")
        except Exception as e:
            self._set_status(f"File load failed: {e}", "danger")
        finally:
            self._set_busy(False)

    def _load_limits(self, event=None) -> None:
        """Load a QARTOD gross-range or climatology limits CSV."""
        path = self.limits_path.strip()
        if not path or not os.path.exists(path):
            self._set_status("Browse for a limits CSV file first.", "warning")
            return
        try:
            self._limits = pd.read_csv(path)
            self._persist_config()
            self._set_status(
                f"Limits loaded from {os.path.basename(path)}.", "success"
            )
        except Exception as e:
            self._set_status(f"Limits load failed: {e}", "danger")

    def _on_browse_append(self, event=None) -> None:
        """Open a native file dialog to select a NetCDF file to append."""
        path = _browse_file(
            "Select NetCDF file to append",
            [("NetCDF files", "*.nc"), ("All files", "*.*")],
        )
        if path:
            self._append_path = path
            self._w_append_path_display.value = path

    def _append_from_file(self, event=None) -> None:
        """
        Load a second NetCDF and concatenate it with the first loaded dataset
        entry, extending the time range without discarding the original data.
        """
        path = self._append_path
        if not path or not os.path.exists(path):
            self._set_status("Browse for a NetCDF file to append first.", "warning")
            return
        if not self._data:
            self._set_status("Load a primary dataset before appending.", "warning")
            return
        self._set_busy(True)
        try:
            new_ds = xr.open_dataset(path)
            key = next(iter(self._data))
            combined = xr.concat([self._data[key], new_ds], dim="time").sortby("time")
            self._data[key] = combined
            self._populate_variables()
            t0 = pd.Timestamp(combined.time.values[0]).strftime("%Y-%m-%d")
            t1 = pd.Timestamp(combined.time.values[-1]).strftime("%Y-%m-%d")
            self._set_status(
                f"Appended {os.path.basename(path)} to '{key}'. "
                f"Combined time range: {t0} -- {t1}.",
                "success",
            )
        except Exception as e:
            self._set_status(f"Append failed: {e}", "danger")
        finally:
            self._set_busy(False)

    def _find_gaps(self, event=None) -> None:
        """
        Scan the loaded dataset for time gaps >= threshold hours and
        auto-create annotation rows (qcFlag='not_available') for each.
        """
        if not self._data:
            self._set_status("Load data before searching for gaps.", "warning")
            return

        threshold_hours = self._w_gap_threshold.value
        ds = next(iter(self._data.values()))
        times = pd.DatetimeIndex(ds["time"].values)
        if len(times) < 2:
            self._set_status("Not enough data points to check for gaps.", "warning")
            return

        diffs = times[1:] - times[:-1]
        gap_mask = diffs > pd.Timedelta(hours=threshold_hours)
        gap_starts = times[:-1][gap_mask]
        gap_ends = times[1:][gap_mask]
        gap_durations = diffs[gap_mask]

        if len(gap_starts) == 0:
            self._set_status(
                f"No gaps >= {threshold_hours:.0f} h found in the loaded data.", "info"
            )
            return

        if "id" not in self._anno_df.columns:
            self._anno_df = pd.DataFrame(columns=_ANNO_COLS)
            self._anno_table.value = self._anno_df
            self._anno_table.editors = self._anno_editors()

        new_rows = []
        for gs, ge, dur in zip(gap_starts, gap_ends, gap_durations):
            gap_hours = dur.total_seconds() / 3600
            blank: dict = {col: None for col in _ANNO_COLS}
            blank["subsite"] = self.site
            blank["node"] = self.node
            blank["sensor"] = self.sensor
            blank["method"] = self.method
            blank["stream"] = self.stream
            blank["beginDate"] = gs.strftime("%Y-%m-%dT%H:%M:%S")
            blank["endDate"] = ge.strftime("%Y-%m-%dT%H:%M:%S")
            blank["exclusionFlag"] = False
            blank["qcFlag"] = "not_available"
            blank["annotation"] = f"Data gap of {gap_hours:.1f} hours."
            blank["deleted"] = False
            new_rows.append(blank)

        self._anno_df = pd.concat(
            [self._anno_df, pd.DataFrame(new_rows)], ignore_index=True
        )
        self._anno_table.value = self._anno_df
        self._set_status(
            f"{len(new_rows)} gap annotation(s) added "
            f"(threshold: {threshold_hours:.0f} h).",
            "success",
        )

    # ------------------------------------------------------------------
    # Variable population and plot type detection
    # ------------------------------------------------------------------

    def _populate_variables(self) -> None:
        """
        After data loads, populate variable selectors from the dataset and
        auto-detect whether 1D or 2D mode is appropriate. Variables are
        deselected by default.
        """
        if not self._data:
            return
        ds = next(iter(self._data.values()))
        sci_vars = _science_variables(ds)
        if not sci_vars:
            return
        prev_selection = [v for v in self.variables if v in sci_vars]
        self.param["variables"].objects = sci_vars
        self.variables = prev_selection
        self.param["color_var"].objects = [None] + sci_vars
        self._detect_plot_type(ds, sci_vars[0])

    def _detect_plot_type(self, ds: xr.Dataset, var: str) -> None:
        """Set plot_type based on the leading variable's dimensionality."""
        if var not in ds:
            return
        extra_dims = [d for d in ds[var].dims if "time" not in d and d != "obs"]
        detected = "heatmap" if extra_dims else "timeseries"
        self.plot_type = detected
        self._w_plot_type.value = detected

    @param.depends("plot_type", watch=True)
    def _on_plot_type(self) -> None:
        """Show/hide controls appropriate for the active plot type."""
        is_heatmap = self.plot_type == "heatmap"
        self._w_variables.visible = not is_heatmap
        self._w_normalize.visible = not is_heatmap
        self._w_color_var.visible = is_heatmap

    # ------------------------------------------------------------------
    # Discrete samples
    # ------------------------------------------------------------------

    def _on_loc_filter_toggle(self, event) -> None:
        """Show or hide location filter inputs when the checkbox changes."""
        visible = event.new
        self._w_sample_lat.visible = visible
        self._w_sample_lon.visible = visible
        self._w_sample_radius.visible = visible

    def _prefill_sample_filters(self) -> None:
        """Pre-fill depth and location filter defaults from the sensor vocabulary."""
        if not all([self.site, self.node, self.sensor]):
            return
        try:
            vocab = get_vocabulary(self.site, self.node, self.sensor)[0]
            depth = vocab.get("maxdepth")
            lat = vocab.get("lat")
            lon = vocab.get("lon")
            if depth is not None:
                self._w_depth_min.value = max(0.0, float(depth) - 3.0)
                self._w_depth_max.value = float(depth) + 3.0
            if lat is not None:
                self._w_sample_lat.value = float(lat)
            if lon is not None:
                self._w_sample_lon.value = float(lon)
        except Exception:
            pass

    def _apply_sample_filters(self, samples: pd.DataFrame) -> pd.DataFrame:
        """Apply depth range and optional location filter to discrete samples."""
        if "CTD Depth [m]" in samples.columns:
            samples = samples[
                (samples["CTD Depth [m]"] >= self._w_depth_min.value) &
                (samples["CTD Depth [m]"] <= self._w_depth_max.value)
            ]
        if (self._chk_loc_filter.value
                and "Start Latitude [degrees]" in samples.columns
                and "Start Longitude [degrees]" in samples.columns):
            from ooi_data_explorations.qartod.discrete_samples import distance_to_cast

            dist = distance_to_cast(
                samples,
                self._w_sample_lat.value,
                self._w_sample_lon.value,
            )
            samples = samples[dist <= self._w_sample_radius.value]
        return samples

    def _finish_sample_load(self, samples: pd.DataFrame, source: str) -> None:
        """Shared post-load logic for Alfresco and CSV sample paths."""
        self._samples = self._apply_sample_filters(samples).reset_index(drop=True)
        meas_cols = _sample_measurement_cols(self._samples)
        self.param["sample_columns"].objects = meas_cols
        self.sample_columns = []
        self._set_status(
            f"{len(self._samples)} discrete samples loaded from {source}.", "success"
        )

    def _fetch_discrete_samples(self, event=None) -> None:
        """Fetch discrete water samples from Alfresco for the active site."""
        if self.site is None:
            self._set_status("Select a site before fetching samples.", "warning")
            return
        self._set_busy(True)
        try:
            from ooi_data_explorations.qartod.discrete_samples import (
                get_discrete_samples,
            )

            samples = get_discrete_samples(_array_from_site(self.site))
            self._finish_sample_load(samples, "Alfresco")
        except Exception as e:
            self._set_status(f"Sample fetch failed: {e}", "danger")
        finally:
            self._set_busy(False)

    def _load_samples_from_file(self, event=None) -> None:
        """Load discrete samples from a local CSV file."""
        path = self._samples_csv_path
        if not path or not os.path.exists(path):
            self._set_status(
                "Browse for a discrete samples CSV file first.", "warning"
            )
            return
        self._set_busy(True)
        try:
            samples = pd.read_csv(path)
            self._finish_sample_load(samples, os.path.basename(path))
        except Exception as e:
            self._set_status(f"Sample CSV load failed: {e}", "danger")
        finally:
            self._set_busy(False)

    # ------------------------------------------------------------------
    # Annotations
    # ------------------------------------------------------------------

    def _fetch_annotations(self, event=None) -> None:
        """
        Fetch all annotations from M2M for the selected reference designator.
        qcFlag is preserved as the raw string value returned by the API so
        it can be posted back without remapping.
        """
        if not all([self.site, self.node, self.sensor]):
            self._set_status(
                "Select site, node, and sensor before fetching annotations.",
                "warning",
            )
            return
        self._set_busy(True)
        try:
            raw = get_annotations(self.site, self.node, self.sensor)
            if not raw:
                anno = pd.DataFrame(columns=_ANNO_COLS)
            else:
                anno = pd.DataFrame(raw)
                anno = anno.drop(columns=["@class"], errors="ignore")
                anno["beginDate"] = pd.to_datetime(
                    anno["beginDT"], unit="ms"
                ).dt.strftime("%Y-%m-%dT%H:%M:%S")
                anno["endDate"] = pd.to_datetime(
                    anno["endDT"], unit="ms"
                ).dt.strftime("%Y-%m-%dT%H:%M:%S")
                for col in ANNO_HEADER:
                    if col not in anno.columns:
                        anno[col] = None
                anno["deleted"] = False
                anno = anno[_ANNO_COLS].copy()
            self._anno_df = anno
            self._deleted_ids = set()
            self._anno_table.value = self._anno_df
            self._anno_table.editors = self._anno_editors()
            self._set_status(f"{len(anno)} annotations fetched.", "success")
        except Exception as e:
            self._set_status(f"Annotation fetch failed: {e}", "danger")
        finally:
            self._set_busy(False)

    @staticmethod
    def _anno_editors() -> dict:
        """
        Tabulator editor config for ANNO_HEADER columns. 'id' is absent
        intentionally -- read-only: blank for new rows, populated for M2M records.
        """
        return {
            "subsite": {"type": "input"},
            "node": {"type": "input"},
            "sensor": {"type": "input"},
            "stream": {"type": "input"},
            "method": {"type": "input"},
            "parameters": {"type": "input"},
            "beginDate": {"type": "input"},
            "endDate": {"type": "input"},
            "exclusionFlag": {"type": "tickCross"},
            "qcFlag": {"type": "list", "values": _QC_FLAG_STRINGS},
            "source": {"type": "input"},
            "annotation": {"type": "textarea"},
            "deleted": {"type": "tickCross"},
        }

    def _add_anno_row(self, event=None) -> None:
        """
        Append a blank annotation row pre-filled with the current reference
        designator. Works before a fetch (initializes the table on first call).
        """
        if self._anno_df.empty and "id" not in self._anno_df.columns:
            self._anno_df = pd.DataFrame(columns=_ANNO_COLS)
            self._anno_table.value = self._anno_df
            self._anno_table.editors = self._anno_editors()

        blank: dict = {col: None for col in _ANNO_COLS}
        blank["subsite"] = self.site
        blank["node"] = self.node
        blank["sensor"] = self.sensor
        blank["method"] = self.method
        blank["stream"] = self.stream
        blank["qcFlag"] = "suspect"
        blank["deleted"] = False

        self._anno_df = pd.concat(
            [self._anno_df, pd.DataFrame([blank])], ignore_index=True
        )
        self._anno_table.value = self._anno_df

    def _mark_anno_deleted(self, event=None) -> None:
        """Mark selected annotation rows as deleted without removing them."""
        selected = self._anno_table.selection
        if not selected:
            self._set_status("Select rows to mark as deleted.", "warning")
            return
        self._anno_df.loc[selected, "deleted"] = True
        for idx in selected:
            row_id = self._anno_df.loc[idx, "id"]
            if pd.notna(row_id):
                self._deleted_ids.add(int(row_id))
        self._anno_table.value = self._anno_df

    def _save_draft(self, event=None) -> None:
        """Persist current annotation table and deletion IDs to a draft JSON."""
        if self._anno_df.empty:
            self._set_status("No annotations to save.", "warning")
            return
        DRAFT_PATH.parent.mkdir(parents=True, exist_ok=True)
        draft = {
            "annotations": self._anno_df.to_dict(orient="records"),
            "deleted_ids": sorted(self._deleted_ids),
        }
        with open(DRAFT_PATH, "w") as f:
            json.dump(draft, f, indent=2, default=str)
        self._set_status(f"Draft saved to {DRAFT_PATH}.", "success")

    def _export_annotations(self, event=None) -> None:
        """Export added/modified annotations (non-deleted rows) to CSV."""
        if self._anno_df.empty:
            self._set_status("No annotations to export.", "warning")
            return
        EXPORT_DIR.mkdir(parents=True, exist_ok=True)
        out = self._anno_df[~self._anno_df["deleted"]][ANNO_HEADER]
        out_path = EXPORT_DIR / "annotations_export.csv"
        out.to_csv(out_path, index=False)
        self._set_status(f"Annotations exported to {out_path}.", "success")

    def _export_deletes(self, event=None) -> None:
        """Write annotation deletion IDs to a text file, one ID per line."""
        if not self._deleted_ids:
            self._set_status("No rows marked for deletion.", "warning")
            return
        EXPORT_DIR.mkdir(parents=True, exist_ok=True)
        out_path = EXPORT_DIR / "annotation_deletions.txt"
        with open(out_path, "w") as f:
            f.write("\n".join(str(i) for i in sorted(self._deleted_ids)))
        self._set_status(
            f"{len(self._deleted_ids)} deletion IDs exported to {out_path}.",
            "success",
        )

    # ------------------------------------------------------------------
    # Config persistence
    # ------------------------------------------------------------------

    def _persist_config(self) -> None:
        save_config(
            {
                "site": self.site,
                "node": self.node,
                "sensor": self.sensor,
                "method": self.method,
                "stream": self.stream,
                "deployment": self.deployment,
                "data_path": self.data_path,
                "limits_path": self.limits_path,
                "colormap": self.colormap,
            }
        )

    # ------------------------------------------------------------------
    # UI helpers
    # ------------------------------------------------------------------

    def _set_status(self, message: str, alert_type: str = "info") -> None:
        self._status.object = message
        self._status.alert_type = alert_type

    def _set_busy(self, busy: bool) -> None:
        self._spinner.value = busy
        self._spinner.visible = busy

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def _build_sidebar(self) -> pn.Column:
        refdes_section = pn.Column(
            self._w_site,
            self._w_node,
            self._w_sensor,
            self._w_method,
            self._w_stream,
            self._w_deploy,
            self._w_tag,
            self._btn_m2m,
        )
        files_section = pn.Column(
            pn.pane.Markdown("**Data file (.nc)**"),
            pn.Row(self._w_data_path_display, self._btn_browse_data),
            self._btn_file,
            pn.layout.Divider(),
            pn.pane.Markdown("**Append data (.nc)**"),
            pn.Row(self._w_append_path_display, self._btn_browse_append),
            self._btn_append_file,
            pn.layout.Divider(),
            pn.pane.Markdown("**QARTOD limits (.csv)**"),
            pn.Row(self._w_limits_path_display, self._btn_browse_limits),
            self._btn_limits,
        )
        display_section = pn.Column(
            self._w_plot_type,
            pn.pane.Markdown("**Variables**"),
            pn.Row(self._btn_var_all, self._btn_var_none),
            self._w_variables,
            self._w_color_var,
            self._w_normalize,
            self._w_colormap,
        )
        samples_section = pn.Column(
            pn.pane.Markdown("**Fetch from Alfresco**"),
            self._btn_fetch_samples,
            pn.layout.Divider(),
            pn.pane.Markdown("**Or load from CSV**"),
            pn.Row(self._w_samples_path_display, self._btn_browse_samples),
            self._btn_load_samples_csv,
            pn.layout.Divider(),
            pn.pane.Markdown("**Depth Filter**"),
            pn.Row(self._w_depth_min, self._w_depth_max),
            pn.layout.Divider(),
            pn.pane.Markdown("**Location Filter**"),
            self._chk_loc_filter,
            pn.Row(self._w_sample_lat, self._w_sample_lon),
            self._w_sample_radius,
            pn.layout.Divider(),
            pn.pane.Markdown("**Sample Columns**"),
            pn.Row(self._btn_sample_all, self._btn_sample_none),
            self._w_sample_cols,
        )
        anno_section = pn.Column(
            self._btn_fetch_anno,
            pn.layout.Divider(),
            pn.pane.Markdown("**Find Data Gaps**"),
            pn.Row(self._w_gap_threshold, self._btn_find_gaps),
        )
        return pn.Column(
            pn.Accordion(
                ("Reference Designator", refdes_section),
                ("Local Files", files_section),
                ("Variables & Display", display_section),
                ("Discrete Samples", samples_section),
                ("Annotations", anno_section),
                active=[0],
            ),
            pn.layout.Divider(),
            pn.Row(self._spinner, self._status),
        )

    def _build_main(self) -> pn.Column:
        anno_toolbar = pn.Row(
            self._btn_add_anno,
            self._btn_del_anno,
            self._btn_save_draft,
            self._btn_export_anno,
            self._btn_export_dels,
        )
        return pn.Column(
            pn.Card(
                self._plot_pane,
                title="Time Series",
                sizing_mode="stretch_width",
            ),
            pn.Card(
                pn.Column(anno_toolbar, self._anno_table),
                title="Annotations",
                sizing_mode="stretch_width",
            ),
        )

    def servable(self) -> pn.template.FastListTemplate:
        """Assemble and return the Panel template for serving."""
        return pn.template.FastListTemplate(
            title="OOI Data Review Dashboard",
            sidebar=[self._build_sidebar()],
            main=[self._build_main()],
            sidebar_width=360,
            accent_base_color="#1976d2",
            header_background="#1976d2",
        )


if __name__ == "__main__":
    pn.serve(
        {"dashboard": lambda: OOIDashboard().servable()},
        port=5006,
        show=True,
        autoreload=False,
    )
