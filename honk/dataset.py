"""Dataset loading and sequential row cursor."""

import os

import pandas as pd


class DatasetCursor:
    """Loads Parquet/CSV files and provides sequential row consumption."""

    def __init__(self, file_paths: list[str]):
        dfs = []
        for path in file_paths:
            ext = os.path.splitext(path)[1].lower()
            if ext == ".parquet":
                dfs.append(pd.read_parquet(path))
            elif ext == ".csv":
                dfs.append(pd.read_csv(path))
            else:
                raise ValueError(f"Unsupported file format: {ext} ({path})")

        self._df = pd.concat(dfs, ignore_index=True)
        self._offset = 0

    @property
    def total_rows(self) -> int:
        return len(self._df)

    @property
    def remaining(self) -> int:
        return self.total_rows - self._offset

    @property
    def dataframe(self) -> pd.DataFrame:
        """Full dataframe (for distribution learning)."""
        return self._df

    def consume(self, n: int) -> pd.DataFrame:
        """Return next n rows and advance the cursor."""
        if n > self.remaining:
            raise ValueError(
                f"Cannot consume {n} rows, only {self.remaining} remaining"
            )
        chunk = self._df.iloc[self._offset : self._offset + n]
        self._offset += n
        return chunk
