#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author Christopher Wingard
@brief Panel-based interactive dashboard for HITL review of OOI data,
    associated QARTOD test limits, discrete samples, and annotations
    (with the ability to create and edit annotations). Run with:

        ooi-dashboard
"""
import ast
import base64
import json
import operator
import os
import time
from contextlib import contextmanager
from functools import reduce
from pathlib import Path

import numpy as np
import pandas as pd
import panel as pn
import param
import xarray as xr
from loguru import logger
from bokeh.models import BoxAnnotation, ColumnDataSource, CustomJS, Span, TapTool
from bokeh.models import Line as BokehLine
from bokeh.plotting import figure as BokehFigure

from ooi_data_explorations.common import (
    get_annotations,
    get_sensor_information,
    get_vocabulary,
    list_deployments,
    list_methods,
    list_nodes,
    list_sensors,
    list_sites,
    list_streams,
    load_gc_thredds,
)
from ooi_data_explorations.qartod.discrete_samples import (
    distance_to_cast,
    get_discrete_samples,
)
from ooi_data_explorations.qartod.qc_processing import ANNO_HEADER

try:
    import tkinter as tk
    from tkinter import filedialog
    _HAS_TKINTER = True
except ImportError:
    _HAS_TKINTER = False

try:
    import hvplot.pandas  # noqa: F401
    import hvplot.xarray  # noqa: F401
    _HAS_HVPLOT = True
except ImportError:
    _HAS_HVPLOT = False

pn.extension("tabulator", sizing_mode="stretch_width")

LAST_CONFIG_PATH_FILE = Path.home() / ".ooidata" / "last_config_path.txt"
EXPORT_DIR = Path.home() / "ooidata"
DEBUG: bool = False  # set True to enable timing logs at DEBUG level

_LOGO_PATH = Path(__file__).parent / "ooi-logo.png"
_LOGO_B64: str = (
    base64.b64encode(_LOGO_PATH.read_bytes()).decode()
    if _LOGO_PATH.exists() else ""
)

_CASCADE_ORDER = ("node", "sensor", "method", "stream")

_VAR_EXCLUDE_PATTERNS = (
    "qartod", "qc_", "quality_flag", "annotations_qc", "rollup_", "provenance"
)
_VAR_EXCLUDE_NAMES = frozenset({
    "time", "obs", "lat", "lon", "latitude", "longitude", "deployment", "id"
})

_SAMPLE_META_COLS = frozenset({
    "Start Time [UTC]", "Latitude [degrees_north]", "Longitude [degrees_east]",
    "Start Latitude [degrees]", "Start Longitude [degrees]",
    "CTD Depth [m]", "Cast", "Station", "Niskin Bottle",
})

_SAMPLE_ARRAYS: list[str] = [
    "Argentine_Basin", "Cabled", "Endurance", "Irminger_Sea",
    "Pioneer-MAB", "Pioneer-NES", "Southern_Ocean", "Station_Papa",
]

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

_METHOD_DASH: dict[str, str] = {
    "telemetered": "solid",
    "recovered_cspp": "dashed",
    "recovered_host": "dashed",
    "recovered_wfp": "dashed",
    "recovered_inst": "dotted",
    "streamed": "solid",
    "file": "solid"
}


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


_PLOT_COLORS: dict[str, dict[str, str]] = {
    "default": {
        "scatter": "black",
        "suspect_line": "grey",
        "fail_line": "red",
        "clim_line": "grey",
    },
    "dark": {
        "scatter": "lightgrey",
        "suspect_line": "orange",
        "fail_line": "red",
        "clim_line": "orange",
    },
}


def _is_nested_span(span: list | None) -> bool:
    """Return True if span is a list of per-channel [lo, hi] pairs, not a flat [lo, hi]."""
    return bool(span and isinstance(span[0], (list, tuple)))


@contextmanager
def _timed(label: str):
    """Log elapsed wall-clock time for a code block at DEBUG level."""
    t0 = time.perf_counter()
    yield
    logger.debug("TIMER | {:40s} | {:.3f}s", label, time.perf_counter() - t0)


def _theme_colors() -> dict[str, str]:
    """Return plot colors appropriate for the active Panel theme."""
    theme = getattr(pn.config, "theme", "default") or "default"
    return _PLOT_COLORS.get(theme, _PLOT_COLORS["default"])


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
    if not _HAS_TKINTER:
        return None
    try:
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


def _browse_save_file(
    title: str,
    filetypes: list[tuple[str, str]],
    default_ext: str = ".json",
    initial_dir: str | None = None,
    initial_file: str = "",
) -> str | None:
    """Open a native OS save-file dialog via tkinter."""
    if not _HAS_TKINTER:
        return None
    try:
        root = tk.Tk()
        root.withdraw()
        root.wm_attributes("-topmost", 1)
        path = filedialog.asksaveasfilename(
            title=title,
            initialdir=initial_dir or str(Path.home()),
            initialfile=initial_file,
            defaultextension=default_ext,
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
    """Return measurement column names, preserving the original file ordering."""
    return [c for c in df.columns if c not in _SAMPLE_META_COLS]


def _array_from_site(site: str | None) -> str:
    """Map an OOI site code prefix to its Raw Data Server array name."""
    if site is None:
        return "Endurance"
    prefixes = {
        "CE": "Endurance",
        "CP0": "Pioneer-NES",
        "CP1": "Pioneer-MAB",
        "GA": "Argentine_Basin",
        "GI": "Irminger_Sea",
        "GP": "Station_Papa",
        "GS": "Southern_Ocean",
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
    gross_range_path = param.String(default="")

    # Display controls
    variables = param.ListSelector(default=[], objects=[])
    plot_type = param.Selector(default="timeseries", objects=["timeseries", "heatmap"])
    normalize = param.Boolean(default=False)
    show_qartod_flags = param.Boolean(default=False)
    color_var = param.Selector(default=None, objects=[None])
    y_dim = param.Selector(default=None, objects=[None])
    colormap = param.Selector(default=_DEFAULT_CMAP, objects=_CMAPS)
    col_start = param.Integer(default=0)
    col_stride = param.Integer(default=1)
    col_stop = param.Integer(default=-1)

    # Internal triggers: bumped to force plot re-render after data/annotation changes
    _data_gen = param.Integer(default=0, precedence=-1)
    _anno_gen = param.Integer(default=0, precedence=-1)

    def __init__(self, **params):
        super().__init__(**params)
        self._cfg: dict = {}
        self._loading_config: bool = False
        self._current_config_path: str = ""
        self._data: dict[str, xr.Dataset] = {}
        self._gross_range: pd.DataFrame | None = None
        self._climatologies: dict[str, pd.DataFrame] = {}
        self._var_map: dict[str, list[str]] = {}
        self._clim_paths: list[str] = []
        self._sample_var_map: dict[str, list[str]] = {}
        self._samples_raw: pd.DataFrame | None = None
        self._samples: pd.DataFrame | None = None
        self._samples_csv_path: str = ""
        self._append_path: str = ""
        self._anno_df: pd.DataFrame = pd.DataFrame(columns=_ANNO_COLS)
        self._deleted_ids: set[int] = set()
        self._anno_del_path: str = ""
        self._annotation_mode: bool = False
        self._tap_clicks: list[float] = []
        self._tap_source = ColumnDataSource(data={"x": [0.0]})
        self._tap_source.on_change("data", self._on_tap_data_change)

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

        # -- File path displays (disabled TextInput, right-aligned text) --
        self._w_data_path_display = pn.widgets.TextInput(
            name="",
            placeholder="No file selected",
            disabled=True,
            sizing_mode="stretch_width",
        )
        self._w_gr_path_display = pn.widgets.TextInput(
            name="",
            placeholder="No file selected",
            disabled=True,
            sizing_mode="stretch_width",
        )
        self._w_clim_path_display = pn.widgets.TextInput(
            name="",
            placeholder="No files selected",
            disabled=True,
            sizing_mode="stretch_width",
        )
        self._var_map_container = pn.Column(
            sizing_mode="stretch_width", visible=False
        )

        self._w_samples_path_display = pn.widgets.TextInput(
            name="",
            placeholder="No file selected",
            disabled=True,
            sizing_mode="stretch_width",
        )
        self._w_append_path_display = pn.widgets.TextInput(
            name="",
            placeholder="No file selected",
            disabled=True,
            sizing_mode="stretch_width",
        )


        # -- Gap detection controls --
        self._w_gap_threshold = pn.widgets.FloatInput(
            name="", value=72.0, step=1.0, width=75
        )
        self._btn_find_gaps = pn.widgets.Button(
            name="Find Gaps", button_type="default", width=140
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
        self._w_normalize = pn.widgets.Checkbox(
            name="Standardize (z-score)",
            value=self.normalize,
        )
        self._w_normalize.param.watch(
            lambda e: setattr(self, "normalize", e.new), "value"
        )
        self._w_show_qartod = pn.widgets.Checkbox(
            name="Show QARTOD flags",
            value=self.show_qartod_flags,
        )
        self._w_show_qartod.param.watch(
            lambda e: setattr(self, "show_qartod_flags", e.new), "value"
        )
        self._w_color_var = pn.widgets.Select.from_param(
            self.param.color_var, name="Color Variable"
        )
        self._w_color_var.visible = False
        self._w_y_dim = pn.widgets.Select.from_param(
            self.param.y_dim, name="Y Axis"
        )
        self._w_y_dim.visible = False
        self._w_col_start = pn.widgets.IntInput(
            name="Start", value=0, step=1, width=85
        )
        self._w_col_stride = pn.widgets.IntInput(
            name="Stride", value=1, step=1, width=85
        )
        self._w_col_stop = pn.widgets.IntInput(
            name="Stop", value=-1, step=1, width=85
        )
        for _w, _attr in (
            (self._w_col_start, "col_start"),
            (self._w_col_stride, "col_stride"),
            (self._w_col_stop, "col_stop"),
        ):
            _w.param.watch(lambda e, a=_attr: setattr(self, a, e.new), "value")
        self._col_range_section = pn.Column(
            pn.pane.Markdown("**2D Column Range** (stop: -1 = last)"),
            pn.Row(self._w_col_start, self._w_col_stride, self._w_col_stop),
            visible=False,
        )
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
        for _w in (
            self._w_depth_min, self._w_depth_max,
            self._w_sample_lat, self._w_sample_lon, self._w_sample_radius,
        ):
            _w.param.watch(lambda e: self._reapply_sample_filters(), "value")
        self._chk_loc_filter.param.watch(
            lambda e: self._reapply_sample_filters(), "value"
        )

        self._w_sample_arrays = pn.widgets.CheckBoxGroup(
            name="Arrays", options=_SAMPLE_ARRAYS, value=[]
        )

        self._sample_var_map_container = pn.Column(
            sizing_mode="stretch_width", visible=False
        )
        self.param.watch(self._on_variables_for_samples, "variables")
        self.param.watch(self._on_variables_for_qartod, "variables")

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
            name="Load data", button_type="primary"
        )
        self._btn_file.on_click(self._load_from_file)
        self._btn_append_file = pn.widgets.Button(
            name="Append data", button_type="primary"
        )
        self._btn_append_file.on_click(self._append_from_file)
        self._btn_save_data = pn.widgets.Button(
            name="Save data (.nc)", button_type="primary"
        )
        self._btn_save_data.on_click(self._save_data_nc)
        self._btn_load_gr = pn.widgets.Button(
            name="Load gross range", button_type="primary"
        )
        self._btn_load_gr.on_click(self._load_gross_range)

        self._btn_load_clim = pn.widgets.Button(
            name="Load climatology", button_type="primary"
        )
        self._btn_load_clim.on_click(self._load_climatology)

        # -- Buttons: discrete samples --
        self._btn_fetch_samples = pn.widgets.Button(
            name="Fetch discrete samples", button_type="primary"
        )
        self._btn_fetch_samples.on_click(self._fetch_discrete_samples)
        self._btn_load_samples_csv = pn.widgets.Button(
            name="Load CSV", button_type="primary"
        )
        self._btn_load_samples_csv.on_click(self._load_samples_from_file)
        self._btn_save_samples_csv = pn.widgets.Button(
            name="Save CSV", button_type="primary"
        )
        self._btn_save_samples_csv.on_click(self._save_samples_csv)

        # -- Buttons: annotations --
        self._btn_fetch_anno = pn.widgets.Button(
            name="Fetch annotations", button_type="primary", width=140
        )
        self._btn_fetch_anno.on_click(self._fetch_annotations)
        self._btn_load_anno_csv = pn.widgets.Button(
            name="Load CSV", button_type="primary", width=110
        )
        self._btn_load_anno_csv.on_click(self._load_annotations_csv)
        self._btn_add_anno = pn.widgets.Button(
            name="Add row", button_type="primary", width=95
        )
        self._btn_add_anno.on_click(self._add_anno_row)
        self._btn_del_anno = pn.widgets.Button(
            name="Mark Deleted", button_type="warning", width=110
        )
        self._btn_del_anno.on_click(self._mark_anno_deleted)
        self._w_anno_path_display = pn.widgets.TextInput(
            name="",
            placeholder="No annotation CSV selected",
            disabled=True,
            width=350,
        )
        self._btn_export_anno = pn.widgets.Button(
            name="Export CSV", button_type="primary", width=95
        )
        self._btn_export_anno.on_click(self._export_annotations)
        self._btn_export_dels = pn.widgets.Button(
            name="Export Deletes", button_type="primary", width=110
        )
        self._btn_export_dels.on_click(self._export_deletes)
        self._btn_annotate = pn.widgets.Toggle(
            name="Annotate", value=False, button_type="primary", width=95
        )
        self._btn_annotate.param.watch(self._on_annotate_toggle, "value")

        # -- Config buttons --
        self._w_config_path_display = pn.widgets.TextInput(
            name="",
            placeholder="No config loaded",
            disabled=True,
            sizing_mode="stretch_width",
        )
        self._btn_load_config = pn.widgets.Button(
            name="Load Config", button_type="primary"
        )
        self._btn_load_config.on_click(self._load_config)
        self._btn_save_config = pn.widgets.Button(
            name="Save config", button_type="primary"
        )
        self._btn_save_config.on_click(self._save_config)
        self._theme_pref: str = pn.config.theme
        self._w_theme = pn.widgets.RadioButtonGroup(
            name="Theme",
            options=["default", "dark"],
            value=pn.config.theme,
            button_type="primary",
            sizing_mode="stretch_width",
        )
        self._w_theme.param.watch(self._on_theme_change, "value")
        self._btn_reset = pn.widgets.Button(
            name="Reset dashboard", button_type="warning",
            sizing_mode="stretch_width",
        )
        self._btn_reset.on_click(self._reset_dashboard)

        self._startup()

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

    def _startup(self) -> None:
        """
        Populate the site selector on startup, then autoload the last
        used config file if one is recorded (handles theme-change recovery
        and session restart).
        """
        self._set_status("Loading sites...", "info")
        try:
            sites = sorted(list_sites())
        except Exception as e:
            self._set_status(f"Failed to load sites: {e}", "danger")
            return

        self.param["site"].objects = [None] + sites

        if LAST_CONFIG_PATH_FILE.exists():
            try:
                last_path = LAST_CONFIG_PATH_FILE.read_text().strip()
                if last_path and Path(last_path).exists():
                    self._load_config_from_path(last_path)
                    return
            except Exception:
                pass

        self._set_status(
            "Ready. Use 'Load Config' to restore a session.", "info"
        )

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
        if self._loading_config:
            saved = self._cfg.get("node")
            if saved in nodes:
                self.node = saved
                return
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
        if self._loading_config:
            saved = self._cfg.get("sensor")
            if saved in sensors:
                self.sensor = saved
                return
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
        saved_d = self._cfg.get("deployment") if self._loading_config else None
        self.deployment = saved_d if saved_d in deploy_opts else "All"

        self.param["method"].objects = [None] + available
        self._prefill_sample_filters()
        saved_m = self._cfg.get("method") if self._loading_config else None
        self.method = (
            saved_m if (saved_m and saved_m in available) else _default_method(available)
        )

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
        if self._loading_config:
            saved_s = self._cfg.get("stream")
            self.stream = (
                saved_s if (saved_s and saved_s in streams) else streams[0]
            )
            self._restore_non_cascade_settings()
            self._loading_config = False
        else:
            self.stream = streams[0]
        self._set_status("Ready.", "info")

    @param.depends("deployment", watch=True)
    def _on_deployment(self) -> None:
        """Autofill the tag field when deployment changes."""
        if self.deployment is None or self.deployment == "All":
            self._w_tag.value = ".*\\.nc$"
        else:
            self._w_tag.value = "deployment{:04d}.*\\.nc$".format(
                int(self.deployment)
            )

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
            with _timed("load_gc_thredds"):
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
            with _timed("_populate_variables (m2m)"):
                self._populate_variables()
            self._data_gen += 1
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
        """Browse for and open a local NetCDF file as the active dataset."""
        if event is not None:
            initial = str(Path(self.data_path).parent) if self.data_path else None
            picked = _browse_file(
                title="Select NetCDF data file",
                filetypes=[("NetCDF files", "*.nc"), ("All files", "*.*")],
                initial_dir=initial,
            )
            if not picked:
                return
            self.data_path = picked
            self._w_data_path_display.value = os.path.basename(picked)
        path = self.data_path.strip()
        if not path or not os.path.exists(path):
            self._set_status("Select a NetCDF file first.", "warning")
            return
        self._set_busy(True)
        try:
            with _timed("xr.open_dataset"):
                self._data = {"file": xr.open_dataset(path)}
            with _timed("_populate_variables (file)"):
                self._populate_variables()
            self._data_gen += 1
            self._set_status(f"Loaded {os.path.basename(path)}.", "success")
        except Exception as e:
            self._set_status(f"File load failed: {e}", "danger")
        finally:
            self._set_busy(False)

    def _load_gross_range(self, event=None) -> None:
        """Browse for and load a QARTOD gross range CSV."""
        if event is not None:
            initial = (
                str(Path(self.gross_range_path).parent)
                if self.gross_range_path
                else None
            )
            picked = _browse_file(
                title="Select QARTOD gross range CSV",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
                initial_dir=initial,
            )
            if not picked:
                return
            self.gross_range_path = picked
            self._w_gr_path_display.value = os.path.basename(picked)
        path = self.gross_range_path.strip()
        if not path or not os.path.exists(path):
            self._set_status("Select a gross range CSV first.", "warning")
            return
        try:
            df = pd.read_csv(path)
            if self.site and self.node and self.sensor:
                df = df[
                    (df["subsite"] == self.site) &
                    (df["node"] == self.node) &
                    (df["sensor"] == self.sensor)
                ].copy()

            def _get_var(s: str) -> str | None:
                try:
                    return ast.literal_eval(s).get("inp")
                except Exception:
                    return None

            def _get_span(s: str, key: str) -> list | None:
                try:
                    return (
                        ast.literal_eval(s)["qartod"]["gross_range_test"][key]
                    )
                except Exception:
                    return None

            df["_var"] = df["parameters"].apply(_get_var)
            df["_suspect_span"] = df["qcConfig"].apply(
                lambda x: _get_span(x, "suspect_span")
            )
            df["_fail_span"] = df["qcConfig"].apply(
                lambda x: _get_span(x, "fail_span")
            )
            df = df.dropna(subset=["_var"]).drop_duplicates(subset=["_var"])
            self._gross_range = df
            self._auto_populate_var_map()
            self._data_gen += 1
            self._set_status(
                f"Gross range loaded: {len(df)} variable(s) from "
                f"{os.path.basename(path)}.",
                "success",
            )
        except Exception as e:
            self._set_status(f"Gross range load failed: {e}", "danger")

    def _load_climatology(self, event=None) -> None:
        """Browse for and load one or more climatology CSVs (one file per variable)."""
        if event is not None:
            if not _HAS_TKINTER:
                paths = ()
            else:
                try:
                    root = tk.Tk()
                    root.withdraw()
                    root.wm_attributes("-topmost", 1)
                    paths = filedialog.askopenfilenames(
                        title="Select climatology CSV files",
                        initialdir=str(Path.home()),
                        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
                    )
                    root.destroy()
                except Exception:
                    paths = ()
            if not paths:
                return
            self._clim_paths = list(paths)
            n = len(paths)
            self._w_clim_path_display.value = (
                os.path.basename(paths[0]) if n == 1 else f"{n} files selected"
            )
        if not self._clim_paths:
            self._set_status("Select climatology CSV files first.", "warning")
            return
        loaded = 0
        has_depth = False
        for path in self._clim_paths:
            try:
                stem = Path(path).stem
                parts = stem.split("-", 4)
                if len(parts) < 5:
                    self._set_status(
                        f"Cannot parse variable name from filename: {stem}", "warning"
                    )
                    continue
                var_name = parts[4]
                raw = pd.read_csv(path, index_col=0, header=0)
                raw.index = pd.Index(
                    [tuple(ast.literal_eval(str(i))) for i in raw.index]
                )
                raw.columns = pd.Index(
                    [tuple(ast.literal_eval(str(c))) for c in raw.columns]
                )
                parsed = raw.map(
                    lambda x: ast.literal_eval(str(x)) if isinstance(x, str) else x
                )
                self._climatologies[var_name] = parsed
                if any(list(idx) != [0, 0] for idx in parsed.index):
                    has_depth = True
                loaded += 1
            except Exception as e:
                self._set_status(
                    f"Climatology parse failed for {Path(path).name}: {e}", "warning"
                )
        if loaded:
            self._auto_populate_var_map(has_depth=has_depth)
            self._data_gen += 1
            self._set_status(f"{loaded} climatology table(s) loaded.", "success")

    def _auto_populate_var_map(self, has_depth: bool = False) -> None:
        """
        Build per-variable MultiSelect widgets from loaded QARTOD limits.
        Options are all QARTOD names from gross range and climatologies.
        Existing selections in _var_map are preserved.
        """
        qnames: list[str] = []
        if self._gross_range is not None and "_var" in self._gross_range.columns:
            qnames.extend(self._gross_range["_var"].dropna().unique().tolist())
        for k in self._climatologies:
            if k not in qnames:
                qnames.append(k)
        if not qnames or not self.variables:
            self._var_map_container.visible = False
            return
        widgets = []
        for v in self.variables:
            existing = self._var_map.get(v, [])
            if not existing:
                existing = [q for q in qnames if q == v]
            resolved = [q for q in existing if q in qnames]
            self._var_map[v] = resolved
            w = pn.widgets.MultiSelect(
                name=v,
                options=qnames,
                value=resolved,
                size=min(5, max(3, len(qnames))),
                sizing_mode="stretch_width",
            )
            w.param.watch(
                lambda e, var=v: self._on_var_map_change(var, e.new),
                "value",
            )
            widgets.append(w)
        self._var_map_container.objects = widgets
        self._var_map_container.visible = True

    def _on_var_map_change(self, var: str, value: list[str]) -> None:
        self._var_map[var] = value
        self._data_gen += 1

    def _on_theme_change(self, event) -> None:
        self._theme_pref = event.new

    def _on_variables_for_qartod(self, event=None) -> None:
        self._auto_populate_var_map()

    def _append_from_file(self, event=None) -> None:
        """
        Browse for and append a second NetCDF to the active dataset, extending
        the time range without discarding the original data.
        """
        if event is not None:
            initial = (
                str(Path(self._append_path).parent) if self._append_path else None
            )
            picked = _browse_file(
                title="Select NetCDF file to append",
                filetypes=[("NetCDF files", "*.nc"), ("All files", "*.*")],
                initial_dir=initial,
            )
            if not picked:
                return
            self._append_path = picked
            self._w_append_path_display.value = os.path.basename(picked)
        path = self._append_path
        if not path or not os.path.exists(path):
            self._set_status("Select a NetCDF file to append first.", "warning")
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
            self._data_gen += 1
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

    def _save_data_nc(self, event=None) -> None:
        """Save the loaded dataset to a NetCDF file chosen by the user."""
        if not self._data:
            self._set_status("No data loaded to save.", "warning")
            return
        ds = next(iter(self._data.values()))
        refdes = "-".join(
            p for p in [self.site, self.node, self.sensor, self.method] if p
        )
        default_name = f"{refdes}.nc" if refdes else "data.nc"
        initial_dir = str(Path(self.data_path).parent) if self.data_path else None
        path = _browse_save_file(
            title="Save dataset as NetCDF",
            filetypes=[("NetCDF files", "*.nc"), ("All files", "*.*")],
            default_ext=".nc",
            initial_dir=initial_dir,
            initial_file=default_name,
        )
        if not path:
            return
        self._set_busy(True)
        try:
            ds.to_netcdf(path)
            self.data_path = path
            self._w_data_path_display.value = os.path.basename(path)
            self._set_status(f"Data saved to {os.path.basename(path)}.", "success")
        except Exception as e:
            self._set_status(f"Save failed: {e}", "danger")
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
        self._anno_gen += 1
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
        detected = "timeseries"
        self.plot_type = detected
        self._w_plot_type.value = detected
        self.param["y_dim"].objects = [None] + extra_dims
        self.y_dim = extra_dims[0] if extra_dims else None

    @param.depends("plot_type", watch=True)
    def _on_plot_type(self) -> None:
        """Show/hide controls appropriate for the active plot type."""
        is_heatmap = self.plot_type == "heatmap"
        self._w_variables.visible = not is_heatmap
        self._w_normalize.visible = not is_heatmap
        self._w_color_var.visible = is_heatmap
        self._w_y_dim.visible = is_heatmap
        self._col_range_section.visible = not is_heatmap

    @param.depends("color_var", watch=True)
    def _on_color_var(self) -> None:
        """Update y_dim options when color variable changes."""
        if self.color_var is None or not self._data:
            return
        ds = next(iter(self._data.values()))
        if self.color_var not in ds:
            return
        extra_dims = [
            d for d in ds[self.color_var].dims
            if "time" not in d and d != "obs"
        ]
        self.param["y_dim"].objects = [None] + extra_dims
        if self.y_dim not in extra_dims:
            self.y_dim = extra_dims[0] if extra_dims else None

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
        """Pre-fill depth and array defaults from the sensor vocabulary."""
        default_array = _array_from_site(self.site)
        if default_array not in self._w_sample_arrays.value:
            self._w_sample_arrays.value = [default_array]
        if not all([self.site, self.node, self.sensor]):
            return
        try:
            vocab = get_vocabulary(self.site, self.node, self.sensor)[0]
            depth = vocab.get("maxdepth")
            if depth is not None:
                d = float(depth)
                if d <= 0:
                    self._w_depth_min.value = 0.0
                    self._w_depth_max.value = 20.0
                else:
                    self._w_depth_min.value = max(0.0, d - 3.0)
                    self._w_depth_max.value = d + 3.0
        except Exception:
            pass
        self._sync_location_from_data()

    def _sync_location_from_data(self) -> None:
        """Fetch mooring lat/lon by averaging across all deployment records."""
        if not all([self.site, self.node, self.sensor]):
            return
        try:
            deploys = list_deployments(self.site, self.node, self.sensor)
            lats, lons = [], []
            for d in deploys:
                info = get_sensor_information(self.site, self.node, self.sensor, d)
                if info:
                    lats.append(info[0]["location"]["latitude"])
                    lons.append(info[0]["location"]["longitude"])
            if lats:
                self._w_sample_lat.value = float(np.mean(lats))
                self._w_sample_lon.value = float(np.mean(lons))
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
            dist = distance_to_cast(
                samples,
                self._w_sample_lat.value,
                self._w_sample_lon.value,
            )
            samples = samples[dist <= self._w_sample_radius.value]
        return samples

    def _reapply_sample_filters(self) -> None:
        """Re-filter from the raw sample set and trigger a plot update."""
        if self._samples_raw is None:
            return
        self._samples = self._apply_sample_filters(
            self._samples_raw
        ).reset_index(drop=True)
        self._data_gen += 1

    def _finish_sample_load(self, samples: pd.DataFrame, source: str) -> None:
        """Shared post-load logic for all discrete sample load paths."""
        self._samples_raw = samples.reset_index(drop=True)
        self._samples = self._apply_sample_filters(
            self._samples_raw
        ).reset_index(drop=True)
        self._auto_populate_sample_var_map()
        self._data_gen += 1
        self._set_status(
            f"{len(self._samples)} discrete samples loaded from {source}.", "success"
        )

    def _on_variables_for_samples(self, event) -> None:
        self._auto_populate_sample_var_map()

    def _auto_populate_sample_var_map(self) -> None:
        """Rebuild per-variable MultiSelect widgets, preserving existing selections."""
        if self._samples_raw is None or not self.variables:
            self._sample_var_map_container.visible = False
            return
        col_opts = _sample_measurement_cols(self._samples_raw)
        widgets = []
        for v in self.variables:
            existing = self._sample_var_map.get(v, [])
            w = pn.widgets.MultiSelect(
                name=v,
                options=col_opts,
                value=[c for c in existing if c in col_opts],
                size=min(5, max(3, len(col_opts))),
                sizing_mode="stretch_width",
            )
            w.param.watch(
                lambda e, var=v: self._on_sample_var_map_change(var, e.new),
                "value",
            )
            widgets.append(w)
        self._sample_var_map_container.objects = widgets
        self._sample_var_map_container.visible = True

    def _on_sample_var_map_change(self, var: str, value: list[str]) -> None:
        self._sample_var_map[var] = value
        self._data_gen += 1

    def _build_sample_scatter(self, var: str) -> list:
        """Return scatter elements for discrete samples mapped to var."""
        if self._samples is None:
            return []
        cols = self._sample_var_map.get(var, [])
        if not cols:
            return []
        time_col = "Start Time [UTC]"
        if time_col not in self._samples.columns:
            return []
        results = []
        for col in cols:
            if col not in self._samples.columns:
                continue
            df = self._samples[[time_col, col]].dropna(subset=[col]).copy()
            df = df.rename(columns={time_col: "time", col: var})
            if df.empty:
                continue
            if not pd.api.types.is_datetime64_any_dtype(df["time"]):
                df["time"] = pd.to_datetime(df["time"], utc=True)
            results.append(
                df.hvplot.scatter(
                    x="time", y=var,
                    color=_theme_colors()["scatter"], alpha=0.6, size=14,
                    marker="triangle", label=col,
                )
            )
        return results

    def _fetch_discrete_samples(self, event=None) -> None:
        """Fetch discrete water samples from the OOI Raw Data Server."""
        arrays = self._w_sample_arrays.value
        if not arrays:
            self._set_status("Select at least one array.", "warning")
            return
        self._set_busy(True)
        try:
            frames = [get_discrete_samples(arr) for arr in arrays]
            frames = [f for f in frames if f is not None]
            if not frames:
                self._set_status("No sample data found for selected arrays.", "warning")
                return
            samples = pd.concat(frames, ignore_index=True)
            self._finish_sample_load(samples, ", ".join(arrays))
        except Exception as e:
            self._set_status(f"Sample fetch failed: {e}", "danger")
        finally:
            self._set_busy(False)

    def _load_samples_from_file(self, event=None) -> None:
        """Browse for and load discrete samples from a local CSV file."""
        if event is not None:
            picked = _browse_file(
                title="Select discrete samples CSV",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            )
            if not picked:
                return
            self._samples_csv_path = picked
            self._w_samples_path_display.value = os.path.basename(picked)
        path = self._samples_csv_path
        if not path or not os.path.exists(path):
            self._set_status("Select a discrete samples CSV file first.", "warning")
            return
        self._set_busy(True)
        try:
            samples = pd.read_csv(path)
            self._finish_sample_load(samples, os.path.basename(path))
        except Exception as e:
            self._set_status(f"Sample CSV load failed: {e}", "danger")
        finally:
            self._set_busy(False)

    def _save_samples_csv(self, event=None) -> None:
        """Save the raw (unfiltered) discrete samples to a CSV file."""
        if self._samples_raw is None:
            self._set_status("No discrete samples loaded to save.", "warning")
            return
        refdes = "-".join(p for p in [self.site, self.node, self.sensor] if p)
        default_name = f"{refdes}_discrete_samples.csv" if refdes else "discrete_samples.csv"
        path = _browse_save_file(
            title="Save discrete samples as CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            default_ext=".csv",
            initial_file=default_name,
        )
        if not path:
            return
        try:
            self._samples_raw.to_csv(path, index=False)
            self._samples_csv_path = path
            self._w_samples_path_display.value = os.path.basename(path)
            self._set_status(
                f"{len(self._samples_raw)} samples saved to {os.path.basename(path)}.",
                "success",
            )
        except Exception as e:
            self._set_status(f"Sample save failed: {e}", "danger")

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
            self._anno_gen += 1
            self._set_status(f"{len(anno)} annotations fetched.", "success")
        except Exception as e:
            self._set_status(f"Annotation fetch failed: {e}", "danger")
        finally:
            self._set_busy(False)

    def _apply_anno_csv(self, path: str) -> bool:
        """
        Load annotations from CSV at path. After loading, apply any known
        deletion IDs: checks self._anno_del_path first, then falls back to
        annotation_deletions.txt in the same directory as the CSV.
        Returns True on success.
        """
        try:
            raw = pd.read_csv(path)
            missing = [c for c in ANNO_HEADER if c not in raw.columns]
            if missing:
                self._set_status(
                    f"CSV is missing required columns: {missing}", "warning"
                )
                return False
            for col in ANNO_HEADER:
                if col not in raw.columns:
                    raw[col] = None
            if "deleted" not in raw.columns:
                raw["deleted"] = False
            anno = raw[_ANNO_COLS].copy()
            anno["deleted"] = anno["deleted"].fillna(False).astype(bool)
            # Resolve deletion ID file: saved path first, sibling fallback
            del_file = Path(self._anno_del_path) if self._anno_del_path else None
            if del_file is None or not del_file.exists():
                sibling = Path(path).parent / "annotation_deletions.txt"
                if sibling.exists():
                    del_file = sibling
            if del_file is not None and del_file.exists():
                del_ids: set[int] = set()
                with open(del_file) as f:
                    for line in f:
                        stripped = line.strip()
                        if stripped.isdigit():
                            del_ids.add(int(stripped))
                if del_ids and "id" in anno.columns:
                    mask = anno["id"].apply(
                        lambda v: pd.notna(v) and str(v).isdigit() and int(v) in del_ids
                    )
                    anno.loc[mask, "deleted"] = True
                self._deleted_ids = del_ids | set(
                    int(i) for i in anno.loc[anno["deleted"], "id"].dropna()
                )
            else:
                self._deleted_ids = set(
                    int(i) for i in anno.loc[anno["deleted"], "id"].dropna()
                )
            self._anno_df = anno
            self._anno_table.value = self._anno_df
            self._anno_table.editors = self._anno_editors()
            self._anno_gen += 1
            self._w_anno_path_display.value = path
            return True
        except Exception as e:
            self._set_status(f"Annotation CSV load failed: {e}", "danger")
            return False

    def _load_annotations_csv(self, event=None) -> None:
        """Load annotations from a previously exported CSV file."""
        path = _browse_file(
            title="Select annotation CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if not path:
            return
        if self._apply_anno_csv(path):
            self._set_status(
                f"{len(self._anno_df)} annotations loaded from CSV.", "success"
            )

    @staticmethod
    def _anno_editors() -> dict:
        """
        Tabulator editor config for ANNO_HEADER columns. 'id' is read-only:
        blank for new rows, populated for M2M records -- no user control.
        """
        return {
            "id": None,
            "subsite": {"type": "input"},
            "node": {"type": "input"},
            "sensor": {"type": "input"},
            "stream": {"type": "input"},
            "method": {"type": "input"},
            "parameters": {"type": "input"},
            "beginDate": {"type": "input"},
            "endDate": {"type": "input"},
            "exclusionFlag": {"type": "tickCross"},
            "qcFlag": {"type": "list", "values": [s for s in _QC_FLAG_STRINGS if s is not None]},
            "source": {"type": "input"},
            "annotation": "textarea",
            "deleted": {"type": "tickCross"},
        }

    def _add_anno_row(self, event=None) -> None:
        """
        Append a blank annotation row pre-filled with the current reference
        designator. Works before a fetch (initializes the table on first call).
        """
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
        self._anno_gen += 1

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
        self._anno_gen += 1

    def _export_annotations(self, event=None) -> None:
        """Export added/modified annotations (non-deleted rows) to CSV."""
        if self._anno_df.empty:
            self._set_status("No annotations to export.", "warning")
            return
        out_path = _browse_save_file(
            title="Save annotation CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            default_ext=".csv",
            initial_dir=str(EXPORT_DIR),
            initial_file="annotations_export.csv",
        )
        if not out_path:
            return
        out = self._anno_df[~self._anno_df["deleted"]][ANNO_HEADER]
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(out_path, index=False)
        self._w_anno_path_display.value = out_path
        self._set_status(f"Annotations exported to {out_path}.", "success")

    def _export_deletes(self, event=None) -> None:
        """Write annotation deletion IDs to a text file, one ID per line."""
        if not self._deleted_ids:
            self._set_status("No rows marked for deletion.", "warning")
            return
        out_path = _browse_save_file(
            title="Save deletion ID list",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
            default_ext=".txt",
            initial_dir=str(EXPORT_DIR),
            initial_file="annotation_deletions.txt",
        )
        if not out_path:
            return
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            f.write("\n".join(str(i) for i in sorted(self._deleted_ids)))
        self._anno_del_path = out_path
        self._set_status(
            f"{len(self._deleted_ids)} deletion IDs exported to {out_path}.",
            "success",
        )

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    @param.depends(
        "variables", "plot_type", "normalize", "show_qartod_flags", "color_var",
        "y_dim", "colormap", "col_start", "col_stride", "col_stop",
        "_data_gen", "_anno_gen",
    )
    def _plot_view(self):
        if not _HAS_HVPLOT:
            return pn.pane.Markdown(
                "*hvplot is required for plotting. Install with: pip install hvplot*",
                styles={"color": "#888"},
            )
        if not self._data:
            return pn.pane.Markdown(
                "*Load data to display plots.*", styles={"color": "#888"}
            )
        if self.plot_type == "heatmap":
            return self._build_heatmap_plot()
        if not self.variables:
            return pn.pane.Markdown(
                "*Select variables to plot.*", styles={"color": "#888"}
            )
        if self.normalize:
            return self._build_standardize_overlay()
        return self._build_stacked_subplots()

    def _make_annotation_hook(self, var: str | None):
        """
        Return a Bokeh finalize hook that draws annotation spans and (when
        var is given) QARTOD suspect/fail shading behind the data curves.
        """
        _ANNO_COLORS = {
            "pass": "#2ecc71",
            "suspect": "#f1c40f",
            "fail": "#e74c3c",
            "not_operational": "#00bcd4",
            "not_available": "#2196f3",
        }


        def hook(plot, element):
            _t_hook = time.perf_counter()
            bokeh_plot = plot.handles["plot"]

            if not self._anno_df.empty:
                for _, row in self._anno_df.iterrows():
                    if row.get("deleted", False):
                        continue
                    try:
                        t0 = pd.Timestamp(row["beginDate"]).timestamp() * 1000
                        t1 = pd.Timestamp(row["endDate"]).timestamp() * 1000
                    except Exception:
                        continue
                    color = _ANNO_COLORS.get(row.get("qcFlag"), "#888888")
                    bokeh_plot.add_layout(BoxAnnotation(
                        left=t0, right=t1,
                        fill_color=color, fill_alpha=0.15,
                        line_color=None,
                    ))

            _OOI_TAG = "__ooi_annotate__"
            if not any(_OOI_TAG in list(t.tags) for t in bokeh_plot.tools):
                _ttool = TapTool(behavior="inspect")
                _ttool.tags = [_OOI_TAG]
                bokeh_plot.add_tools(_ttool)
                bokeh_plot.js_on_event(
                    "tap",
                    CustomJS(
                        args={"source": self._tap_source},
                        code="source.data = {x: [cb_obj.x]};",
                    ),
                )

            if var is None or not self._data:
                logger.debug(
                    "TIMER | {:<40s} | {:.3f}s",
                    f"annotation_hook[annotations only]",
                    time.perf_counter() - _t_hook,
                )
                return
            # --- Gross range limit lines (Span; does not affect y-axis auto-range) ---
            if (
                self._gross_range is not None
                and "_var" in self._gross_range.columns
            ):
                qnames = self._var_map.get(var, [var])
                matches = self._gross_range[self._gross_range["_var"].isin(qnames)]
                if not matches.empty:
                    gr = matches.iloc[0]
                    sus = gr.get("_suspect_span")
                    fail = gr.get("_fail_span")
                    tc = _theme_colors()
                    if not _is_nested_span(sus) and not _is_nested_span(fail):
                        for val, dash, color in [
                            (sus[0] if sus else None, "dashed", tc["suspect_line"]),
                            (sus[1] if sus else None, "dashed", tc["suspect_line"]),
                            (fail[0] if fail else None, "solid", tc["fail_line"]),
                            (fail[1] if fail else None, "solid", tc["fail_line"]),
                        ]:
                            if val is not None:
                                bokeh_plot.add_layout(Span(
                                    location=float(val),
                                    dimension="width",
                                    line_color=color,
                                    line_dash=dash,
                                    line_width=1.25,
                                ))

            # --- Climatology step-function lines ---
            qnames = self._var_map.get(var, [var])
            clim_df = next(
                (self._climatologies[q] for q in qnames if q in self._climatologies),
                self._climatologies.get(var),
            )
            if clim_df is not None:
                # Per-channel tables (non-[0,0] rows) don't map to a single y-value;
                # the limit scatter already marks out-of-range points per channel.
                is_multichan = any(list(idx) != [0, 0] for idx in clim_df.index)
                if not is_multichan:
                    zero_rows = [idx for idx in clim_df.index if list(idx) == [0, 0]]
                    if zero_rows:
                        clim_row = clim_df.loc[zero_rows[0]]
                        for _, ds in self._data.items():
                            if var not in ds:
                                continue
                            times = pd.DatetimeIndex(ds["time"].values)
                            times_ms = [
                                pd.Timestamp(t).timestamp() * 1000 for t in times
                            ]
                            months = times.month
                            clim_mins: list[float] = []
                            clim_maxs: list[float] = []
                            for m in months:
                                matched = False
                                for col_span, val in clim_row.items():
                                    if col_span[0] <= m <= col_span[1]:
                                        clim_mins.append(float(val[0]))
                                        clim_maxs.append(float(val[1]))
                                        matched = True
                                        break
                                if not matched:
                                    clim_mins.append(float("nan"))
                                    clim_maxs.append(float("nan"))
                            for yvals in (clim_mins, clim_maxs):
                                src = ColumnDataSource({"x": times_ms, "y": yvals})
                                bokeh_plot.add_glyph(src, BokehLine(
                                    x="x", y="y",
                                    line_color=_theme_colors()["clim_line"],
                                    line_dash="dotted",
                                    line_width=1.25,
                                ))
                            break  # use first dataset only

            logger.debug(
                "TIMER | {:<40s} | {:.3f}s",
                f"annotation_hook total({var})",
                time.perf_counter() - _t_hook,
            )

        return hook

    def _build_stacked_subplots(self):
        multi_method = len(self._data) > 1
        subplot_h = max(200, min(350, 700 // len(self.variables)))

        subplots = []
        for var in self.variables:
            curves = []
            is_2d = False
            for method, ds in self._data.items():
                if var not in ds:
                    continue
                extra_dims = [
                    d for d in ds[var].dims if "time" not in d and d != "obs"
                ]
                if extra_dims:
                    is_2d = True
                    dim = extra_dims[0]
                    n = ds.dims[dim]
                    dim_coords = (
                        ds.coords[dim].values if dim in ds.coords
                        else np.arange(n)
                    )
                    start = max(0, self.col_start)
                    stop = n if self.col_stop < 0 else min(n, self.col_stop + 1)
                    stride = max(1, self.col_stride)
                    for idx in range(start, stop, stride):
                        col_val = dim_coords[idx]
                        col_label = (
                            f"{col_val:.4g}"
                            if isinstance(
                                col_val, (int, float, np.integer, np.floating)
                            )
                            else str(col_val)
                        )
                        if multi_method:
                            col_label = f"{col_label} ({method})"
                        df = (
                            ds[var].isel({dim: idx})
                            .to_dataframe()
                            .reset_index()
                            .dropna(subset=[var])
                        )
                        if df.empty:
                            continue
                        p = df.hvplot.line(
                            x="time",
                            y=var,
                            label=col_label,
                            line_dash=_METHOD_DASH.get(method, "solid"),
                            ylabel=var,
                            grid=True,
                        )
                        curves.append(p)
                else:
                    has_dep = (
                        "deployment" in ds.data_vars or "deployment" in ds.coords
                    )
                    cols = [var] + (["deployment"] if has_dep else [])
                    df = (
                        ds[cols].to_dataframe().reset_index().dropna(subset=[var])
                    )
                    if df.empty:
                        continue
                    if has_dep and "deployment" in df.columns:
                        df = df.copy()
                        suffix = f" ({method})" if multi_method else ""
                        df["label"] = df["deployment"].apply(
                            lambda d: f"dep {int(d)}{suffix}"
                        )
                        by_kw: dict = {"by": "label"}
                    elif multi_method:
                        by_kw = {"label": method}
                    else:
                        by_kw = {}
                    p = df.hvplot.line(
                        x="time",
                        y=var,
                        **by_kw,
                        line_dash=_METHOD_DASH.get(method, "solid"),
                        ylabel=var,
                        grid=True,
                    )
                    curves.append(p)

            if not curves:
                continue

            with _timed(f"build_limit_scatter({var})"):
                for sc in self._build_limit_scatter(var):
                    curves.append(sc)

            if self.show_qartod_flags:
                for sc in self._build_qartod_scatter(var):
                    curves.append(sc)

            if not is_2d:
                for sc in self._build_sample_scatter(var):
                    curves.append(sc)

            with _timed(f"subplot_reduce({var})"):
                subplot = reduce(operator.mul, curves).opts(
                    show_legend=is_2d or multi_method,
                    legend_position="bottom_right",
                    hooks=[self._make_annotation_hook(var)],
                )
            subplots.append(subplot)

        if not subplots:
            return pn.pane.Markdown(
                "*No data for selected variables.*", styles={"color": "#888"}
            )

        panes = [
            pn.pane.HoloViews(sp, sizing_mode="stretch_width", height=subplot_h)
            for sp in subplots
        ]
        return pn.Column(*panes, sizing_mode="stretch_width")

    def _build_standardize_overlay(self):
        """
        Single overlay plot: all selected variables z-scored onto one axis.
        Color by variable; line style by method. Legend identifies variables.
        """
        multi_method = len(self._data) > 1
        curves = []
        for var in self.variables:
            for method, ds in self._data.items():
                if var not in ds:
                    continue
                extra_dims = [
                    d for d in ds[var].dims if "time" not in d and d != "obs"
                ]
                if extra_dims:
                    continue  # 2D variables not supported in normalize mode
                df = (
                    ds[[var]].to_dataframe().reset_index().dropna(subset=[var])
                )
                if df.empty:
                    continue
                s = df[var].std()
                if s > 0:
                    df[var] = (df[var] - df[var].mean()) / s
                label = f"{var} ({method})" if multi_method else var
                p = df.hvplot.line(
                    x="time",
                    y=var,
                    label=label,
                    line_dash=_METHOD_DASH.get(method, "solid"),
                    ylabel="standardized",
                    responsive=True,
                    height=400,
                    grid=True,
                )
                curves.append(p)

        if not curves:
            return pn.pane.Markdown(
                "*No data for selected variables.*", styles={"color": "#888"}
            )

        plot = reduce(operator.mul, curves).opts(
            show_legend=len(self.variables) > 1,
            legend_position="bottom_right",
            hooks=[self._make_annotation_hook(None)],
        )
        return pn.pane.HoloViews(plot, sizing_mode="stretch_width")

    def _build_heatmap_plot(self):
        if self.color_var is None:
            return pn.pane.Markdown(
                "*Select a Color Variable for heatmap mode.*",
                styles={"color": "#888"},
            )
        ds = next(iter(self._data.values()))
        if self.color_var not in ds:
            return pn.pane.Markdown(
                "*Selected variable not in dataset.*", styles={"color": "#888"}
            )
        extra_dims = [
            d for d in ds[self.color_var].dims
            if "time" not in d and d != "obs"
        ]
        if not extra_dims:
            return pn.pane.Markdown(
                "*Variable has no depth/bin dimension for heatmap.*",
                styles={"color": "#888"},
            )
        y_dim = self.y_dim
        if y_dim is not None and y_dim not in ds[self.color_var].dims:
            return pn.pane.Markdown(
                f"*Y Axis dimension '{y_dim}' is not available for "
                f"'{self.color_var}'. Select a compatible Y Axis.*",
                styles={"color": "#888"},
            )
        if y_dim is None:
            y_dim = extra_dims[0]
        p = ds[self.color_var].hvplot.quadmesh(
            x="time",
            y=y_dim,
            cmap=self.colormap,
            responsive=True,
            height=400,
            colorbar=True,
        ).opts(hooks=[self._make_annotation_hook(None)])
        return pn.pane.HoloViews(p, sizing_mode="stretch_width")

    def _build_qartod_scatter(self, var: str) -> list:
        """
        Return scatter elements for data points flagged suspect (3) or fail (4)
        by the QARTOD results variable embedded in the dataset.
        """
        if not self._data:
            return []
        _FLAG_STYLES: dict[int, tuple[str, str]] = {
            3: ("#f1c40f", "qc_suspect"),
            4: ("#e74c3c", "qc_fail"),
        }
        results: list = []
        for ds in self._data.values():
            qnames = self._var_map.get(var, [var])
            qartod_var = next(
                (f"{q}_qartod_results" for q in qnames if f"{q}_qartod_results" in ds),
                f"{var}_qartod_results" if f"{var}_qartod_results" in ds else None,
            )
            if qartod_var is None or var not in ds:
                break
            times = ds["time"].values
            flags = ds[qartod_var].values
            extra_dims = [d for d in ds[var].dims if "time" not in d and d != "obs"]
            if extra_dims and flags.ndim > 1:
                dim = extra_dims[0]
                n = ds.dims[dim]
                start = max(0, self.col_start)
                stop = n if self.col_stop < 0 else min(n, self.col_stop + 1)
                stride = max(1, self.col_stride)
                for idx in range(start, stop, stride):
                    if idx >= flags.shape[1]:
                        continue
                    chan_flags = flags[:, idx]
                    chan_vals = ds[var].isel({dim: idx}).values
                    for flag_val, (color, label) in _FLAG_STYLES.items():
                        mask = chan_flags == flag_val
                        if not mask.any():
                            continue
                        df = pd.DataFrame({"time": times[mask], var: chan_vals[mask]})
                        results.append(df.hvplot.scatter(
                            x="time", y=var, color=color, alpha=0.5, size=6, label=label,
                        ))
            else:
                if flags.ndim > 1:
                    flags = flags.max(axis=tuple(range(1, flags.ndim)))
                vals = ds[var].values
                for flag_val, (color, label) in _FLAG_STYLES.items():
                    mask = flags == flag_val
                    if not mask.any():
                        continue
                    df = pd.DataFrame({"time": times[mask], var: vals[mask]})
                    results.append(df.hvplot.scatter(
                        x="time", y=var, color=color, alpha=0.5, size=6, label=label,
                    ))
            break  # use first dataset
        return results

    def _build_limit_scatter(self, var: str) -> list:
        """
        Return up to two hvplot scatter elements (suspect=orange, fail=red)
        for data points outside gross range or climatology limits for var.
        Dispatches to a numpy 2D path or a pandas 1D path based on variable shape.
        """
        if not self._data:
            return []

        qnames = self._var_map.get(var, [var])
        gr_row = None
        if self._gross_range is not None and "_var" in self._gross_range.columns:
            matches = self._gross_range[self._gross_range["_var"].isin(qnames)]
            if not matches.empty:
                gr_row = matches.iloc[0]

        clim_df = next(
            (self._climatologies[q] for q in qnames if q in self._climatologies),
            self._climatologies.get(var),
        )
        if gr_row is None and clim_df is None:
            return []

        dim: str | None = None
        for ds in self._data.values():
            if var in ds:
                extra = [d for d in ds[var].dims if "time" not in d and d != "obs"]
                if extra:
                    dim = extra[0]
                break

        if dim is not None:
            return self._limit_scatter_2d(var, gr_row, clim_df)
        return self._limit_scatter_1d(var, gr_row, clim_df)

    def _limit_scatter_2d(self, var: str, gr_row, clim_df) -> list:
        """
        Limit scatter for 2D (time x channel) variables. Works entirely in numpy
        on the raw (n_time, n_chan) array to avoid the cost of to_dataframe().
        """
        val_arrays: list[np.ndarray] = []
        time_arrays: list[np.ndarray] = []
        for ds in self._data.values():
            if var in ds:
                val_arrays.append(ds[var].values)
                time_arrays.append(ds["time"].values)

        if not val_arrays:
            return []

        vals = np.concatenate(val_arrays, axis=0) if len(val_arrays) > 1 else val_arrays[0]
        times = np.concatenate(time_arrays) if len(time_arrays) > 1 else time_arrays[0]
        n_time, n_chan = vals.shape

        fail_2d = np.zeros((n_time, n_chan), dtype=bool)
        suspect_2d = np.zeros((n_time, n_chan), dtype=bool)

        _t_gr = time.perf_counter()
        if gr_row is not None:
            fail_span: list | None = gr_row.get("_fail_span")
            suspect_span: list | None = gr_row.get("_suspect_span")
            if fail_span is not None:
                if _is_nested_span(fail_span):
                    lo = np.array([fail_span[i][0] if i < len(fail_span) else np.nan for i in range(n_chan)])
                    hi = np.array([fail_span[i][1] if i < len(fail_span) else np.nan for i in range(n_chan)])
                    fail_2d = (vals < lo[np.newaxis, :]) | (vals > hi[np.newaxis, :])
                else:
                    fail_2d = (vals < fail_span[0]) | (vals > fail_span[1])
            if suspect_span is not None:
                if _is_nested_span(suspect_span):
                    lo = np.array([suspect_span[i][0] if i < len(suspect_span) else np.nan for i in range(n_chan)])
                    hi = np.array([suspect_span[i][1] if i < len(suspect_span) else np.nan for i in range(n_chan)])
                    suspect_2d = ((vals < lo[np.newaxis, :]) | (vals > hi[np.newaxis, :])) & ~fail_2d
                else:
                    suspect_2d = ((vals < suspect_span[0]) | (vals > suspect_span[1])) & ~fail_2d
        logger.debug(
            "TIMER | {:<40s} | {:.3f}s",
            f"limit_scatter 2d gross_range({var})",
            time.perf_counter() - _t_gr,
        )

        _t_cl = time.perf_counter()
        if clim_df is not None:
            months = pd.DatetimeIndex(times).month.to_numpy()
            is_multichan = any(list(idx) != [0, 0] for idx in clim_df.index)
            if is_multichan:
                lo_arr = np.full((n_chan, 13), np.nan)
                hi_arr = np.full((n_chan, 13), np.nan)
                for idx in clim_df.index:
                    lo_idx, hi_idx = list(idx)
                    clim_row = clim_df.loc[idx]
                    for col_span, val in clim_row.items():
                        for pos in range(lo_idx, hi_idx + 1):
                            for m in range(col_span[0], col_span[1] + 1):
                                lo_arr[pos, m] = val[0]
                                hi_arr[pos, m] = val[1]
                # row_lo[t, c] = lo_arr[c, months[t]] via numpy broadcast
                chan_idx = np.arange(n_chan)
                row_lo = lo_arr[chan_idx[np.newaxis, :], months[:, np.newaxis]]
                row_hi = hi_arr[chan_idx[np.newaxis, :], months[:, np.newaxis]]
                suspect_2d |= ((vals < row_lo) | (vals > row_hi)) & ~fail_2d
            else:
                zero_rows = [idx for idx in clim_df.index if list(idx) == [0, 0]]
                if zero_rows:
                    clim_row = clim_df.loc[zero_rows[0]]
                    lo_1d = np.full(13, np.nan)
                    hi_1d = np.full(13, np.nan)
                    for col_span, val in clim_row.items():
                        for m in range(col_span[0], col_span[1] + 1):
                            lo_1d[m] = val[0]
                            hi_1d[m] = val[1]
                    months = pd.DatetimeIndex(times).month.to_numpy()
                    row_lo_1d = lo_1d[months]
                    row_hi_1d = hi_1d[months]
                    clim_out = (vals < row_lo_1d[:, np.newaxis]) | (vals > row_hi_1d[:, np.newaxis])
                    suspect_2d |= (clim_out & ~fail_2d)
        logger.debug(
            "TIMER | {:<40s} | {:.3f}s",
            f"limit_scatter 2d climatology({var})",
            time.perf_counter() - _t_cl,
        )

        results: list = []
        for mask_2d, color, label in [
            (fail_2d, "red", "fail"),
            (suspect_2d, "orange", "suspect"),
        ]:
            t_idx = np.where(mask_2d)[0]
            if len(t_idx) == 0:
                continue
            df = pd.DataFrame({"time": times[t_idx], var: vals[mask_2d]})
            results.append(df.hvplot.scatter(
                x="time", y=var, color=color, alpha=0.6, size=8, label=label,
            ))
        return results

    def _limit_scatter_1d(self, var: str, gr_row, clim_df) -> list:
        """Limit scatter for 1D variables; pandas path."""
        dfs: list[pd.DataFrame] = []
        with _timed(f"limit_scatter 1d to_dataframe({var})"):
            for ds in self._data.values():
                if var not in ds:
                    continue
                df = ds[[var]].to_dataframe().reset_index().dropna(subset=[var]).copy()
                dfs.append(df)

        if not dfs:
            return []

        all_df = pd.concat(dfs, ignore_index=True)
        fail_mask = pd.Series(False, index=all_df.index)
        suspect_mask = pd.Series(False, index=all_df.index)

        _t_gr = time.perf_counter()
        if gr_row is not None:
            fail_span: list | None = gr_row.get("_fail_span")
            suspect_span: list | None = gr_row.get("_suspect_span")
            if fail_span is not None and not _is_nested_span(fail_span):
                fail_mask = (all_df[var] < fail_span[0]) | (all_df[var] > fail_span[1])
            if suspect_span is not None and not _is_nested_span(suspect_span):
                suspect_mask = (
                    (all_df[var] < suspect_span[0]) | (all_df[var] > suspect_span[1])
                ) & ~fail_mask
        logger.debug(
            "TIMER | {:<40s} | {:.3f}s",
            f"limit_scatter 1d gross_range({var})",
            time.perf_counter() - _t_gr,
        )

        _t_cl = time.perf_counter()
        if clim_df is not None:
            month_arr = pd.DatetimeIndex(all_df["time"]).month.to_numpy()
            zero_rows = [idx for idx in clim_df.index if list(idx) == [0, 0]]
            if zero_rows:
                clim_row = clim_df.loc[zero_rows[0]]
                lo_arr = np.full(13, np.nan)
                hi_arr = np.full(13, np.nan)
                for col_span, val in clim_row.items():
                    for m in range(col_span[0], col_span[1] + 1):
                        lo_arr[m] = val[0]
                        hi_arr[m] = val[1]
                vals = all_df[var].to_numpy()
                out = (vals < lo_arr[month_arr]) | (vals > hi_arr[month_arr])
                suspect_mask |= pd.Series(out, index=all_df.index) & ~fail_mask
        logger.debug(
            "TIMER | {:<40s} | {:.3f}s",
            f"limit_scatter 1d climatology({var})",
            time.perf_counter() - _t_cl,
        )

        results: list = []
        fail_df = all_df[fail_mask]
        sus_df = all_df[suspect_mask]
        if not fail_df.empty:
            results.append(fail_df.hvplot.scatter(
                x="time", y=var, color="red", alpha=0.6, size=8, label="fail",
            ))
        if not sus_df.empty:
            results.append(sus_df.hvplot.scatter(
                x="time", y=var, color="orange", alpha=0.6, size=8, label="suspect",
            ))
        return results

    # ------------------------------------------------------------------
    # Config persistence
    # ------------------------------------------------------------------

    def _build_config_dict(self) -> dict:
        """Collect all current dashboard state into a serialisable dict."""
        return {
            # Select Data
            "site": self.site,
            "node": self.node,
            "sensor": self.sensor,
            "method": self.method,
            "stream": self.stream,
            "deployment": self.deployment,
            "file_regex": self._w_tag.value,
            "data_path": self.data_path,
            # Variables & Display
            "variables": list(self.variables),
            "colormap": self.colormap,
            "col_start": self.col_start,
            "col_stride": self.col_stride,
            "col_stop": self.col_stop,
            "theme": self._theme_pref,
            # QARTOD
            "gross_range_path": self.gross_range_path,
            "clim_paths": self._clim_paths,
            "qartod_var_map": self._var_map,
            # Annotations
            "anno_csv_path": self._w_anno_path_display.value,
            "anno_del_path": self._anno_del_path,
            # Discrete Samples
            "sample_arrays": list(self._w_sample_arrays.value),
            "samples_csv_path": self._samples_csv_path,
            "depth_min": self._w_depth_min.value,
            "depth_max": self._w_depth_max.value,
            "loc_filter": self._chk_loc_filter.value,
            "sample_lat": self._w_sample_lat.value,
            "sample_lon": self._w_sample_lon.value,
            "sample_radius": self._w_sample_radius.value,
            "sample_var_map": self._sample_var_map,
        }

    def _restore_non_cascade_settings(self) -> None:
        """
        Apply all non-cascade settings from self._cfg. Called at the end
        of _on_method once the full cascade has completed during a config
        load. Order matters: var_map must be set before any limits reload.
        """
        cfg = self._cfg
        cmap = cfg.get("colormap", _DEFAULT_CMAP)
        if cmap in _CMAPS:
            self.colormap = cmap
        # Restore var maps before reloading limits so saved mappings survive
        self._var_map = cfg.get("qartod_var_map", {})
        self._sample_var_map = cfg.get("sample_var_map", {})
        # Gross range
        gr_path = cfg.get("gross_range_path", "")
        if gr_path and Path(gr_path).exists():
            self.gross_range_path = gr_path
            self._w_gr_path_display.value = os.path.basename(gr_path)
            self._load_gross_range()
        # Climatology
        clim_paths = [p for p in cfg.get("clim_paths", []) if Path(p).exists()]
        if clim_paths:
            self._clim_paths = clim_paths
            n = len(clim_paths)
            self._w_clim_path_display.value = (
                os.path.basename(clim_paths[0]) if n == 1 else f"{n} files selected"
            )
            self._load_climatology()
        # Local data file -- autoload so variables can be restored and plots fire
        saved_vars: list[str] = cfg.get("variables", [])
        dp = cfg.get("data_path", "")
        if dp and Path(dp).exists():
            self.data_path = dp
            self._w_data_path_display.value = os.path.basename(dp)
            self._load_from_file()
            # _populate_variables() ran inside _load_from_file; now apply saved selection
            valid_vars = [v for v in saved_vars if v in self.param["variables"].objects]
            if valid_vars:
                self.variables = valid_vars
        elif dp:
            # Path saved but file not found -- just restore the display
            self.data_path = dp
            self._w_data_path_display.value = os.path.basename(dp)
        # Annotations -- restore deletion path first so _apply_anno_csv can use it
        self._anno_del_path = cfg.get("anno_del_path", "")
        anno_path = cfg.get("anno_csv_path", "")
        if anno_path and Path(anno_path).exists():
            if self._apply_anno_csv(anno_path):
                self._set_status(
                    f"{len(self._anno_df)} annotations restored from CSV.", "success"
                )
        elif anno_path:
            self._w_anno_path_display.value = anno_path
        # Discrete samples CSV -- prefer saved file over re-fetching
        sc_path = cfg.get("samples_csv_path", "")
        if sc_path and Path(sc_path).exists():
            self._samples_csv_path = sc_path
            self._w_samples_path_display.value = os.path.basename(sc_path)
            self._load_samples_from_file()
        # Sample arrays (used when fetching fresh; preserve for reference)
        saved_arrays = [
            a for a in cfg.get("sample_arrays", []) if a in _SAMPLE_ARRAYS
        ]
        self._w_sample_arrays.value = saved_arrays
        # Depth filter
        self._w_depth_min.value = float(cfg.get("depth_min", 0.0))
        self._w_depth_max.value = float(cfg.get("depth_max", 6000.0))
        # Location filter (sets widget visibility via watcher)
        self._chk_loc_filter.value = bool(cfg.get("loc_filter", False))
        self._w_sample_lat.value = float(cfg.get("sample_lat", 0.0))
        self._w_sample_lon.value = float(cfg.get("sample_lon", 0.0))
        self._w_sample_radius.value = float(cfg.get("sample_radius", 5.0))
        # Column range for 2D timeseries
        self.col_start = int(cfg.get("col_start", 0))
        self._w_col_start.value = self.col_start
        self.col_stride = max(1, int(cfg.get("col_stride", 1)))
        self._w_col_stride.value = self.col_stride
        self.col_stop = int(cfg.get("col_stop", -1))
        self._w_col_stop.value = self.col_stop
        # File tag regex -- overrides the auto-fill from _on_deployment
        self._w_tag.value = cfg.get("file_regex", self._w_tag.value)
        # Theme preference (applied at next launch via __main__)
        theme = cfg.get("theme", "default")
        if theme in ("default", "dark"):
            self._theme_pref = theme
            self._w_theme.value = theme
    def _load_config_from_path(self, path: str) -> None:
        """Parse a config JSON and trigger the cascade to restore full state."""
        try:
            with open(path) as f:
                cfg = json.load(f)
        except Exception as e:
            self._set_status(f"Config load failed: {e}", "danger")
            return
        self._cfg = cfg
        self._loading_config = True
        self._current_config_path = path
        self._w_config_path_display.value = os.path.basename(path)
        LAST_CONFIG_PATH_FILE.parent.mkdir(parents=True, exist_ok=True)
        LAST_CONFIG_PATH_FILE.write_text(path)
        saved_site = cfg.get("site")
        site_options = [s for s in self.param["site"].objects if s is not None]
        if saved_site and saved_site in site_options:
            self.site = saved_site
        else:
            # No valid site in config; restore non-cascade settings only
            self._restore_non_cascade_settings()
            self._loading_config = False
            msg = (
                f"Config loaded (site '{saved_site}' not found; select manually)."
                if saved_site else "Config loaded."
            )
            self._set_status(msg, "warning" if saved_site else "success")

    def _load_config(self, event=None) -> None:
        """Open a file picker and load the selected config JSON."""
        initial = (
            str(Path(self._current_config_path).parent)
            if self._current_config_path else None
        )
        path = _browse_file(
            title="Load dashboard config",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            initial_dir=initial,
        )
        if path:
            self._load_config_from_path(path)

    def _save_config(self, event=None) -> None:
        """Open a save dialog and write current state to a config JSON."""
        initial_dir = (
            str(Path(self._current_config_path).parent)
            if self._current_config_path else None
        )
        initial_file = (
            Path(self._current_config_path).name
            if self._current_config_path else "dashboard_config.json"
        )
        path = _browse_save_file(
            title="Save dashboard config",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            default_ext=".json",
            initial_dir=initial_dir,
            initial_file=initial_file,
        )
        if not path:
            return
        try:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w") as f:
                json.dump(self._build_config_dict(), f, indent=2)
            self._current_config_path = path
            self._w_config_path_display.value = os.path.basename(path)
            LAST_CONFIG_PATH_FILE.parent.mkdir(parents=True, exist_ok=True)
            LAST_CONFIG_PATH_FILE.write_text(path)
            self._set_status(f"Config saved to {path}.", "success")
        except Exception as e:
            self._set_status(f"Config save failed: {e}", "danger")

    def _reset_dashboard(self, event=None) -> None:
        """Reset all dashboard state to defaults."""
        # Data
        self._data = {}
        self.data_path = ""
        self._w_data_path_display.value = ""
        self._append_path = ""
        self._w_append_path_display.value = ""
        # QARTOD limits
        self._gross_range = None
        self.gross_range_path = ""
        self._w_gr_path_display.value = ""
        self._climatologies = {}
        self._clim_paths = []
        self._w_clim_path_display.value = ""
        self._var_map = {}
        self._var_map_container.objects = []
        self._var_map_container.visible = False
        # Variables
        self.param["variables"].objects = []
        self.variables = []
        # Discrete samples
        self._samples = None
        self._samples_raw = None
        self._samples_csv_path = ""
        self._w_samples_path_display.value = ""
        self._sample_var_map = {}
        self._sample_var_map_container.objects = []
        self._sample_var_map_container.visible = False
        # Annotations
        self._anno_df = pd.DataFrame(columns=_ANNO_COLS)
        self._deleted_ids = set()
        self._anno_del_path = ""
        self._w_anno_path_display.value = ""
        self._anno_table.value = self._anno_df
        self._annotation_mode = False
        self._btn_annotate.value = False
        self._tap_clicks = []
        # Config
        self._cfg = {}
        self._loading_config = False
        self._current_config_path = ""
        self._w_config_path_display.value = ""
        # Cascade reset -- _on_site handles None and sets status "Ready."
        self.site = None
        # Force plot/annotation refresh; override "Ready." with final message
        self._data_gen += 1
        self._anno_gen += 1
        self._set_status("Dashboard reset.", "info")

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
        select_data_section = pn.Column(
            self._w_site,
            self._w_node,
            self._w_sensor,
            self._w_method,
            self._w_stream,
            self._w_deploy,
            self._w_tag,
            self._btn_m2m,
            pn.layout.Divider(),
            pn.pane.Markdown("**Data file (.nc)**"),
            self._w_data_path_display,
            pn.Row(self._btn_file, self._btn_save_data),
            pn.layout.Divider(),
            pn.pane.Markdown("**Append data (.nc)**"),
            self._w_append_path_display,
            self._btn_append_file,
        )
        qartod_section = pn.Column(
            pn.pane.Markdown("**Gross Range (.csv)**"),
            pn.Row(self._w_gr_path_display, self._btn_load_gr),
            pn.layout.Divider(),
            pn.pane.Markdown("**Climatology (.csv)**"),
            pn.Row(self._w_clim_path_display, self._btn_load_clim),
            pn.layout.Divider(),
            pn.pane.Markdown("**Variable Mapping**"),
            self._var_map_container,
        )
        display_section = pn.Column(
            self._w_plot_type,
            pn.pane.Markdown("**Variables**"),
            pn.Row(self._btn_var_all, self._btn_var_none),
            self._w_variables,
            self._w_color_var,
            self._w_y_dim,
            self._w_normalize,
            self._w_show_qartod,
            self._col_range_section,
            self._w_colormap,
        )
        samples_section = pn.Column(
            pn.pane.Markdown("**Fetch from Raw Data Server**"),
            self._w_sample_arrays,
            self._btn_fetch_samples,
            pn.layout.Divider(),
            pn.pane.Markdown("**Or load from CSV**"),
            self._w_samples_path_display,
            pn.Row(self._btn_load_samples_csv, self._btn_save_samples_csv),
            pn.layout.Divider(),
            pn.pane.Markdown("**Depth Filter**"),
            pn.Row(self._w_depth_min, self._w_depth_max),
            pn.layout.Divider(),
            pn.pane.Markdown("**Location Filter**"),
            self._chk_loc_filter,
            pn.Row(self._w_sample_lat, self._w_sample_lon),
            self._w_sample_radius,
            pn.layout.Divider(),
            pn.pane.Markdown("**Variable Mapping**"),
            self._sample_var_map_container,
        )
        config_section = pn.Column(
            pn.Row(self._w_config_path_display),
            pn.Row(self._btn_load_config, self._btn_save_config),
            pn.layout.Divider(),
            pn.pane.Markdown("**Theme** (applies on next launch)"),
            self._w_theme,
        )
        return pn.Column(
            pn.Row(self._spinner, self._status),
            pn.layout.Divider(),
            pn.Accordion(
                ("Select Data", select_data_section),
                ("Variables & Display", display_section),
                ("QARTOD", qartod_section),
                ("Discrete Samples", samples_section),
                ("Configuration", config_section),
                active=[],
            ),
            pn.layout.Divider(),
            self._btn_reset,
            pn.layout.Divider(),
            pn.pane.HTML(
                (
                    "<div style='text-align:center; padding:6px 0;'>"
                    + (
                        f"<img src='data:image/png;base64,{_LOGO_B64}' "
                        "style='width:225px; margin-bottom:6px;'/><br>"
                        if _LOGO_B64 else ""
                    )
                    + "<span style='font-size:1em; color:#888;'>"
                    "Designed and implemented by<br>"
                    "Christopher Wingard with Claude (Anthropic)"
                    "</span></div>"
                ),
                sizing_mode="stretch_width",
            ),
        )

    def _on_annotate_toggle(self, event) -> None:
        """Activate or deactivate tap-to-create annotation mode."""
        self._annotation_mode = event.new
        self._tap_clicks = []
        if event.new:
            self._set_status("Click start time on any subplot.", "info")
        # TapTool is added unconditionally at plot render time in the finalize
        # hook. No dynamic tool injection here -- _on_tap_data_change checks
        # _annotation_mode and ignores taps when the mode is off.

    def _on_tap_data_change(self, attr: str, old, new) -> None:
        """
        Called by ColumnDataSource.on_change when JS writes a tap x-coordinate
        into _tap_source. Accumulates clicks; creates an annotation row on the
        second click.
        """
        if not self._annotation_mode:
            return
        x = new["x"][0]
        self._tap_clicks.append(x)
        n = len(self._tap_clicks)
        if n == 1:
            ts = pd.Timestamp(int(x), unit="ms")
            self._set_status(
                f"Start: {ts.strftime('%Y-%m-%d %H:%M')}. Click end time.", "info"
            )
        elif n >= 2:
            x0, x1 = sorted(self._tap_clicks[-2:])
            begin_dt = pd.Timestamp(int(x0), unit="ms").strftime("%Y-%m-%dT%H:%M:%S")
            end_dt = pd.Timestamp(int(x1), unit="ms").strftime("%Y-%m-%dT%H:%M:%S")
            self._create_anno_from_taps(begin_dt, end_dt)
            self._btn_annotate.value = False

    def _create_anno_from_taps(self, begin_dt: str, end_dt: str) -> None:
        """
        Insert a new annotation row populated from tap coordinates. Caller is
        responsible for exiting annotation mode after this returns.
        """
        blank: dict = {col: None for col in _ANNO_COLS}
        blank["subsite"] = self.site
        blank["node"] = self.node
        blank["sensor"] = self.sensor
        blank["method"] = self.method
        blank["stream"] = self.stream
        blank["beginDate"] = begin_dt
        blank["endDate"] = end_dt
        blank["exclusionFlag"] = False
        blank["deleted"] = False
        self._anno_df = pd.concat(
            [self._anno_df, pd.DataFrame([blank])], ignore_index=True
        )
        self._anno_table.value = self._anno_df
        self._set_status(
            f"Annotation added {begin_dt} -- {end_dt}. Edit qcFlag and notes in table.",
            "success",
        )
        # Inject BoxAnnotation directly into existing figures to avoid a full
        # plot re-render (which would reset zoom). The full re-render hook will
        # re-draw all spans correctly if the plot is later rebuilt.
        t0 = pd.Timestamp(begin_dt).timestamp() * 1000
        t1 = pd.Timestamp(end_dt).timestamp() * 1000
        doc = pn.state.curdoc
        if doc is not None:
            for fig in doc.select({"type": BokehFigure}):
                fig.add_layout(BoxAnnotation(
                    left=t0, right=t1,
                    fill_color="#888888", fill_alpha=0.15,
                    line_color=None,
                ))

    def _build_main(self) -> pn.Column:
        anno_toolbar = pn.Column(
            pn.Row(
                self._btn_fetch_anno,
                self._btn_load_anno_csv,
                self._w_anno_path_display,
                self._btn_export_anno,
                self._btn_export_dels,
            ),
            pn.Row(
                self._btn_annotate,
                self._btn_add_anno,
                self._btn_del_anno,
                self._btn_find_gaps,
                self._w_gap_threshold,
                pn.pane.Markdown("Min gap (hours)", margin=(-12, 0, 0, 0)),
            ),
        )
        return pn.Column(
            pn.Card(
                self._plot_view,
                title="Data View",
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
            title="OOI Data Quality Reviews",
            sidebar=[self._build_sidebar()],
            main=[self._build_main()],
            sidebar_width=360,
            accent_base_color="#1976d2",
            header_background="#1976d2",
            theme=self._theme_pref,
        )


def main() -> None:
    """Launch the OOI HITL QC dashboard."""
    import sys
    logger.remove()
    logger.add(sys.stderr, level="DEBUG" if DEBUG else "INFO")
    if LAST_CONFIG_PATH_FILE.exists():
        try:
            _last = LAST_CONFIG_PATH_FILE.read_text().strip()
            if _last and Path(_last).exists():
                with open(_last) as _f:
                    _theme = json.load(_f).get("theme", "default")
                if _theme in ("default", "dark"):
                    pn.config.theme = _theme
        except Exception:
            pass
    pn.serve(
        {"dashboard": lambda: OOIDashboard().servable()},
        port=5006,
        show=True,
        autoreload=False,
    )


if __name__ == "__main__":
    main()
