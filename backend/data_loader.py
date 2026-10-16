import os
 
import duckdb
import pandas as pd
 
# One in-memory DuckDB connection for the app's lifetime. Re-registering
# DATASET_TABLE on every load means later tools can always query
# "dataset" without needing to know which file is currently active.
_con = duckdb.connect(database=":memory:")
DATASET_TABLE = "dataset"
 
 
def _cell_to_json_safe(value):
    """Convert one pandas/numpy cell value into a plain JSON-safe type."""
    if pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "item"):  # numpy scalars: int64, float64, bool_, ...
        return value.item()
    return value
 
 
def load_filesub(path: str, preview_rows: int = 50) -> dict:
    if not os.path.isfile(path):
        raise ValueError(f"File not found: {path}")
 
    ext = os.path.splitext(path)[1].lower()
    if ext not in (".xlsx", ".xls", ".csv"):
        raise ValueError(f"Unsupported file type '{ext}' — expected .xlsx, .xls, or .csv")

    try:
        if ext == ".csv":
            df = pd.read_csv(path)
        if ext in (".xlsx", ".xls"):
            df = pd.read_excel(path, engine="openpyxl" if ext == ".xlsx" else None)
    except ImportError as exc:
        raise ValueError(
            f"Missing dependency to read '{ext}' files ({exc}). "
            "Old .xls files need: pip install xlrd"
        ) from exc
    except Exception as exc:
        raise ValueError(f"Could not read Excel file: {exc}") from exc
 
    df.columns = [str(c) for c in df.columns]
    if df.shape[1] == 0:
        raise ValueError(f"'{os.path.basename(path)}' has no columns")
 
    _con.register(DATASET_TABLE, df)
 
    columns = list(df.columns)
    row_count = int(len(df))
    col_count = len(columns)
 
    preview = df.head(preview_rows)
    rows = [
        [_cell_to_json_safe(cell) for cell in row]
        for row in preview.itertuples(index=False, name=None)
    ]
 
    return {
        "columns": columns,
        "rows": rows,
        "row_count": row_count,
        "col_count": col_count,
    }
 
 
def query_dataset(sql: str) -> pd.DataFrame:
    """
    Run a SQL query against the currently loaded dataset and return a DataFrame of results.
    """
    return _con.execute(sql).df()