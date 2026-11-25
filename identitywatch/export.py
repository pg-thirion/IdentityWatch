"""Spreadsheet-safe export of untrusted log fields."""
def safe_csv(frame):
    out = frame.copy()
    for col in out.select_dtypes(include=["object", "string"]).columns:
        out[col] = out[col].map(lambda v: "'" + v if isinstance(v, str) and
            (v.lstrip().startswith(("=", "+", "-", "@")) or v.startswith(("\t", "\r", "\n"))) else v)
    return out.to_csv(index=False)
