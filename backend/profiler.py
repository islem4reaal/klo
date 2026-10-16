import datetime
import json
import math
import warnings

import numpy as np
import pandas as pd

__all__ = ["profile_dataframe", "format_profile_for_llm"]

# Tunable thresholds (kept together so they are easy to change later)

MAX_SAMPLE_VALUES = 5          
TOP_N = 5                      
DATE_SAMPLE_SIZE = 200         
DATE_PARSE_THRESHOLD = 0.9     
CATEGORICAL_MAX_UNIQUE = 20    
CATEGORICAL_MAX_RATIO = 0.5    
MISSING_WARN_PCT = 50         
IQR_MULTIPLIER = 1.5          


def _to_native(value):
    """Convert one value to something json.dumps() accepts.

    None / NaN / NaT / +-inf -> None, numpy scalars -> int / float / bool,
    Timestamp / datetime / date -> ISO-8601 string, anything else unchanged.
    """
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return None if (math.isnan(number) or math.isinf(number)) else number
    if isinstance(value, (pd.Timestamp, datetime.datetime, datetime.date)):
        return None if pd.isna(value) else value.isoformat()   # NaT -> None
    try:
        if pd.isna(value):          # pd.NA and friends
            return None
    except (TypeError, ValueError):  # list-like input makes pd.isna return an array
        pass
    return value


def _parse_dates(values: pd.Series) -> pd.Series:
    """Parse to datetimes, turning failures into NaT, without noisy warnings."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")   # pandas warns when it must guess a format
        return pd.to_datetime(values, errors="coerce")


def _infer_kind(series: pd.Series) -> str:
    """Classify a column as numeric / categorical / text / datetime / boolean / empty.

    The dtype says how pandas STORES the data; the kind says what it MEANS.
    The order of the checks matters.
    """
    non_null = series.dropna()
    if non_null.empty:
        return "empty"

    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"


    if set(non_null.unique()) <= {True, False}:
        return "boolean"

    sample = non_null.astype(str).head(DATE_SAMPLE_SIZE)
    if _parse_dates(sample).notna().mean() >= DATE_PARSE_THRESHOLD:
        return "datetime"

    n_unique = non_null.nunique()
    if n_unique <= CATEGORICAL_MAX_UNIQUE or n_unique / len(non_null) <= CATEGORICAL_MAX_RATIO:
        return "categorical"
    return "text"


def _numeric_stats(series: pd.Series) -> dict:
    values = series.dropna().astype(float)

    q25 = values.quantile(0.25)
    q75 = values.quantile(0.75)
    iqr = q75 - q25
    lower_fence = q25 - IQR_MULTIPLIER * iqr
    upper_fence = q75 + IQR_MULTIPLIER * iqr

    return {
        "min": _to_native(values.min()),
        "max": _to_native(values.max()),
        "mean": _to_native(values.mean()),
        "median": _to_native(values.median()),
        "std": _to_native(values.std()),          # NaN for a single value -> None
        "q25": _to_native(q25),
        "q75": _to_native(q75),
        "zeros": int((values == 0).sum()),
        "negatives": int((values < 0).sum()),
        "outliers_iqr": int(((values < lower_fence) | (values > upper_fence)).sum()),
    }


def _categorical_stats(series: pd.Series, top_n: int = TOP_N) -> dict:
    counts = series.dropna().value_counts().head(top_n)   # already most-frequent first
    return {
        "top_values": [
            {"value": _to_native(value), "count": int(count)}
            for value, count in counts.items()
        ]
    }


def _datetime_stats(series: pd.Series) -> dict:
    dates = _parse_dates(series).dropna()
    if dates.empty:
        return {"min": None, "max": None, "span_days": None}
    first, last = dates.min(), dates.max()
    return {
        "min": _to_native(first),
        "max": _to_native(last),
        "span_days": int((last - first).days),
    }


def _boolean_stats(series: pd.Series) -> dict:
    values = series.dropna().astype(bool)
    true_count = int(values.sum())
    return {"true_count": true_count, "false_count": int(len(values) - true_count)}


def _profile_column(series: pd.Series) -> dict:
    """Build one entry of profile["columns"]."""
    total = len(series)
    non_null = series.dropna()
    kind = _infer_kind(series)
    missing = int(series.isna().sum())

    if kind == "numeric":
        stats = _numeric_stats(series)
    elif kind in ("categorical", "text"):
        stats = _categorical_stats(series)
    elif kind == "datetime":
        stats = _datetime_stats(series)
    elif kind == "boolean":
        stats = _boolean_stats(series)
    else:                                   # "empty"
        stats = {}

    return {
        "name": str(series.name),          
        "kind": kind,
        "dtype": str(series.dtype),
        "missing": missing,
        "missing_pct": round(100 * missing / total, 1) if total else 0.0,
        "unique": int(non_null.nunique()),
        "sample_values": [_to_native(v) for v in non_null.head(MAX_SAMPLE_VALUES).tolist()],
        "stats": stats,
    }


def profile_dataframe(df: pd.DataFrame) -> dict:
    """Profile a whole DataFrame (see the contract at the top of this file).

    Raises
    ------
    TypeError  if df is not a pandas DataFrame
    ValueError if df has zero columns
    A DataFrame with columns but zero rows is valid (every column is "empty").
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"df must be a pandas DataFrame, got {type(df).__name__}")
    if df.shape[1] == 0:
        raise ValueError("DataFrame has no columns")


    columns = [_profile_column(df.iloc[:, i]) for i in range(df.shape[1])]
    duplicate_rows = int(df.duplicated().sum())

    found = []
    for col in columns:
        if col["kind"] == "empty":
            found.append(f"Column '{col['name']}' is completely empty")
            continue
        if col["missing_pct"] >= MISSING_WARN_PCT:
            found.append(f"Column '{col['name']}' is {col['missing_pct']}% missing")
        if col["unique"] == 1:
            found.append(f"Column '{col['name']}' has a single constant value")
    if duplicate_rows:
        found.append(f"{duplicate_rows} duplicate row(s) found")

    return {
        "overview": {
            "rows": int(len(df)),
            "columns": int(df.shape[1]),
            # deep=True measures the real size of string contents, not just pointers
            "memory_mb": round(float(df.memory_usage(deep=True).sum()) / 1e6, 3),
            "duplicate_rows": duplicate_rows,
            "missing_cells": int(df.isna().sum().sum()),
        },
        "columns": columns,
        "warnings": found,
    }

def _fmt_number(x) -> str:
    """12345.6789 -> '12,345.68', 9800.0 -> '9,800', None -> 'n/a'."""
    if x is None:
        return "n/a"
    if isinstance(x, float):
        return f"{x:,.2f}".rstrip("0").rstrip(".")
    return str(x)


def _describe_column(col: dict) -> str:
    """One line of text for one column."""
    kind, stats = col["kind"], col["stats"]

    if kind == "numeric":
        detail = (f"min {_fmt_number(stats['min'])}, median {_fmt_number(stats['median'])}, "
                  f"mean {_fmt_number(stats['mean'])}, max {_fmt_number(stats['max'])}, "
                  f"outliers {stats['outliers_iqr']}")
    elif kind in ("categorical", "text"):
        top = ", ".join(f"{t['value']} ({t['count']})" for t in stats["top_values"][:3])
        detail = f"{col['unique']} unique | top: {top}"
    elif kind == "datetime":
        if stats["min"] is None:
            detail = "no parseable dates"
        else:
            detail = f"{stats['min'][:10]} to {stats['max'][:10]} ({stats['span_days']} days)"
    elif kind == "boolean":
        detail = f"true {stats['true_count']}, false {stats['false_count']}"
    else:
        detail = "no values"

    missing = f" | {col['missing']} missing ({col['missing_pct']}%)" if col["missing"] else ""
    return f"- {col['name']} [{kind}]: {detail}{missing}"


def format_profile_for_llm(profile: dict, max_columns: int = 40) -> str:
    """Render a profile as compact plain text for a small model's prompt.

    One line per column (at most max_columns), then any warnings. No raw rows.
    """
    o = profile["overview"]
    lines = [
        f"Dataset: {o['rows']} rows x {o['columns']} columns | "
        f"{o['duplicate_rows']} duplicate rows | {o['missing_cells']} missing cells",
        "Columns:",
    ]
    lines += [_describe_column(c) for c in profile["columns"][:max_columns]]

    hidden = len(profile["columns"]) - max_columns
    if hidden > 0:
        lines.append(f"... and {hidden} more columns")

    if profile["warnings"]:
        lines.append("Warnings:")
        lines += [f"- {w}" for w in profile["warnings"]]
    return "\n".join(lines)