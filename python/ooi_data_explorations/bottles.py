import re
import numpy as np
import pandas as pd
import numpy.typing as npt


class QualityFlags:
    """
    QARTOD QC-flag definitions.

    The assigned flag values are:

        1 = Pass
        2 = Unknown
        3 = Suspect or of High Interest
        4 = Fail
        9 = Missing
    """
    GOOD = 1
    UNKNOWN = 2
    SUSPECT = 3
    BAD = 4
    MISSING = 9


FLAGS = QualityFlags


def check_fill(flag: str | float | None) -> bool:
    """
    Check if an OOI discrete sample flag is a fill value.

    Parameters
    ----------
    flag : str or float or None
        The flag value to check.

    Returns
    -------
    bool
        True if the flag is considered a fill or missing value, False otherwise.

    Notes
    -----
    Fill values are:
      - NaN or None
      - The string "-9999999"
      - Any string not containing the digit '1'
    """
    if pd.isna(flag):
        return True
    flag_str = str(flag)
    if flag_str == "-9999999":
        return True
    if "1" not in flag_str:
        return True
    return False


def parse_flag(flag: str) -> list[int]:
    """
    Parse an OOI discrete sample flag string and return indices of all '1' bits read right-to-left.

    Parameters
    ----------
    flag : str
        A string representing the flag bits.

    Returns
    -------
    list[int]
        List of indices where '1' occurs in the reversed flag string.

    Examples
    --------
    >>> parse_flag("0101")
    [0, 2]
    """
    locs = [match.start() for match in re.finditer("1", flag[::-1])]
    return locs


def map_ctd_flag(flag: str | float | None) -> int | float | None:
    """
    Map OOI discrete sample CTD flags to QARTOD-style QC flags.

    Parameters
    ----------
    flag : str or float or None
        The CTD flag to map.

    Returns
    -------
    int or float or None
        Mapped QARTOD flag value or original fill value.
    """
    if check_fill(flag):
        return flag
    parsed_flag = parse_flag(str(flag))
    max_bit = max(parsed_flag) if parsed_flag else -1

    return {
        1: QualityFlags.MISSING,
        2: QualityFlags.GOOD,
        3: QualityFlags.SUSPECT,
        4: QualityFlags.BAD,
    }.get(max_bit, QualityFlags.UNKNOWN)


def map_discrete_flag(flag: str | float | None) -> int | float | None:
    """
    Map OOI discrete sample Bottle flags to QARTOD-style QC flags.

    Parameters
    ----------
    flag : str or float or None
        The discrete flag to map.

    Returns
    -------
    int or float or None
        Mapped QARTOD flag value or original fill value.
    """
    # Logic is same as map_ctd_flag, so re-use that function:
    return map_ctd_flag(flag)


def map_replicate_flag(flag: str | float | None) -> bool | float | None:
    """
    Interpret OOI discrete sample Bottle replicate flags.

    Parameters
    ----------
    flag : str or float or None
        The replicate flag to interpret.

    Returns
    -------
    bool or float or None
        True if the sample is a replicate (flag bit 3 or 4 set), False otherwise.
        Returns original fill value if applicable.
    """
    if check_fill(flag):
        return flag
    parsed_flag = parse_flag(str(flag))
    max_bit = max(parsed_flag) if parsed_flag else -1
    return max_bit in (3, 4)


def map_niskin_flag(flag: str | float | None) -> int | float | None:
    """
    Map OOI discrete Niskin Bottle flags to QARTOD-style QC flags.

    Parameters
    ----------
    flag : str or float or None
        The Niskin flag to map.

    Returns
    -------
    int or float or None
        Mapped QARTOD flag value or original fill value.
    """
    if check_fill(flag):
        return flag
    parsed_flag = parse_flag(str(flag))
    max_bit = max(parsed_flag) if parsed_flag else -1

    if max_bit == 1:
        return QualityFlags.MISSING
    elif max_bit == 2:
        return QualityFlags.GOOD
    elif max_bit in (3, 4, 5):
        return QualityFlags.SUSPECT
    else:
        return QualityFlags.UNKNOWN


def convert_times(x: str | pd.Timestamp) -> pd.Timestamp | str:
    """
    Convert OOI discrete summary spreadsheet times to pandas Timestamps.

    Parameters
    ----------
    x : str or pandas.Timestamp
        Time value as a string or Timestamp.

    Returns
    -------
    pandas.Timestamp or str
        Converted pandas Timestamp or original input if not string.
    """
    if isinstance(x, str):
        x_clean = x.replace(" ", "")
        return pd.to_datetime(x_clean, utc=False)
    return x


def not_statistically_sigificant(x: str | float | int) -> float | str | int:
    """
    Replace OOI discrete nutrient sample values that are not statistically significant with zero.

    Parameters
    ----------
    x : str or float or int
        Sample value.

    Returns
    -------
    float or str or int
        0 if input contains '<' indicating non-significant value, else original value.
    """
    if isinstance(x, str) and "<" in x:
        return 0
    return x


def clean_data(bottle_data: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and convert OOI Discrete Sampling summary data.

    This function processes the OOI Discrete Sample summary sheets by replacing fill values,
    converting date/time strings to pandas datetime objects, converting non-statistically
    significant values to zero, and mapping bit flag columns to QARTOD-style QC flags.

    Parameters
    ----------
    bottle_data : pandas.DataFrame
        DataFrame containing OOI Discrete Sampling summary data.

    Returns
    -------
    pandas.DataFrame
        Cleaned DataFrame with standardized QC flags and datetime columns.
    """
    # Replace fill values with NaN
    bottle_data = bottle_data.replace(to_replace=["-9999999", -9999999], value=np.nan)

    # Convert time columns to datetime
    for col in bottle_data.columns:
        if "time" in col.lower():
            bottle_data[col] = bottle_data[col].apply(convert_times)

    # Replace nutrient values with '<' by zero
    bottle_data = bottle_data.applymap(not_statistically_sigificant)

    # Map flag columns to QARTOD flags
    for col in bottle_data.columns:
        if "flag" in col.lower():
            col_lower = col.lower()
            if "ctd" in col_lower and "file" not in col_lower:
                bottle_data[col] = bottle_data[col].apply(map_ctd_flag)
            elif "discrete" in col_lower:
                if "replicate" in col_lower:
                    bottle_data[col] = bottle_data[col].apply(map_replicate_flag)
                else:
                    bottle_data[col] = bottle_data[col].apply(map_discrete_flag)
            elif "niskin" in col_lower:
                bottle_data[col] = bottle_data[col].apply(map_niskin_flag)
            else:
                # default fallback
                bottle_data[col] = bottle_data[col].apply(map_discrete_flag)

    return bottle_data