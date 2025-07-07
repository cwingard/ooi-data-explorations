import re
import numpy as np
import pandas as pd
import numpy.typing as npt


class QualityFlags():
    """
    QARTOD QC-flag definitions. The assigned flag values are:

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

def check_fill(flag: str) -> bool:
    """Check if an OOI discrete sample flag is a fill value (-9999999)"""
    if pd.isna(flag):
        return True
    elif str(flag) == "-9999999":
        return True
    elif "1" not in str(flag):
        return True
    else:
        return False


def parse_flag(flag: str) -> list[int]:
    """
    Parse an OOI discrete sample flag and returns index of all 1 bits
    read right-to-left. Returns fill or nan when appropriate.
    """
    locs=[]
    for match in re.finditer("1", flag[::-1], re.S):
        locs.append(match.span()[0])
    return locs


def map_ctd_flag(flag: str) -> int:
    """Map OOI discrete sample CTD flags to QARTOD-style QC-flags."""
    # First filter for fill
    if check_fill(flag):
        return flag
    else:
        parsed_flag = parse_flag(flag)
        max_bit = max(parsed_flag)
        if max_bit == 1:
            return QualityFlags.MISSING
        elif max_bit == 2:
            return QualityFlags.GOOD
        elif max_bit == 3:
            return QualityFlags.SUSPECT
        elif max_bit == 4:
            return QualityFlags.BAD
        else:
            return QualityFlags.UNKNOWN


def map_discrete_flag(flag):
    """Map OOI discrete sample Bottle flags to QARTOD-style QC-flags."""
    # First filter for fill values
    if check_fill(flag):
        return flag
    else:
        parsed_flag = parse_flag(flag)
        max_bit = max(parsed_flag)
        if max_bit == 1:
            return QualityFlags.MISSING
        elif max_bit == 2:
            return QualityFlags.GOOD
        elif max_bit == 3:
            return QualityFlags.SUSPECT
        elif max_bit == 4:
            return QualityFlags.BAD
        else:
            return QualityFlags.UNKNOWN


def map_replicate_flag(flag: str) -> int:
    """
    Interpret OOI discrete sample Bottle replicate flags. Returns a boolean
    if a sample is a duplicate/replicate sample.
    """
    # First filter for fill values
    if check_fill(flag):
        return flag
    else:
        parsed_flag = parse_flag(flag)
        max_bit = max(parsed_flag)
        if max_bit == 3 or max_bit == 4:
            return True
        else:
            return False


def map_niskin_flag(flag: str) -> int:
    """Map OOI discrete Niskin Bottle flags to QARTOD-style QC-flags."""
    if check_fill(flag):
        return flag
    else:
        parsed_flag = parse_flag(flag)
        max_bit = max(parsed_flag)
        if max_bit == 1:
            return QualityFlags.MISSING
        elif max_bit == 2:
            return QualityFlags.GOOD
        elif max_bit == 3 or max_bit == 4 or max_bit == 5:
            return QualityFlags.SUSPECT
        else:
            return QualityFlags.UNKNOWN


def convert_times(x: str) -> pd.Timestamp:
    """Parse times in OOI discrete summary spreadsheets to python datetimes."""
    if type(x) is str:
        x = x.replace(" ","")
        x = pd.to_datetime(x, utc=False)
    else:
        pass
    return x


def not_statistically_sigificant(x: str) -> float:
    """
    Replace OOI discrete nutrient sample values that are not statistically
    significant with zero.
    """
    if type(x) is str:
        if "<" in x:
            x = 0
    return x


def clean_data(bottle_data: pd.DataFrame) -> pd.DataFrame:
    """
    Process, clean, and convert the OOI Discrete Sampling summary sheets.
    
    This function takes the Discrete Sample summary sheets provided by OOI
    and cleans up the spreadsheets, converts data types to be more useable,
    and intrepts the bit flag-maps into QARTOD-type flags.
    
    Parameters
    ----------
    bottle_data: pandas.DataFrame
        A dataframe containing the loaded OOI Discrete Sampling summary data.
        
    Returns
    -------
    bottle_data: pandas.DataFrame
        The discrete sampling data cleaned up and standardized with the quality flags
        mapped to QARTOD-style QC flags and replicate flags mapped to TRUE/FALSE.
    """
    # Replace -9999999 with NaNs
    bottle_data = bottle_data.replace(to_replace="-9999999", value=np.nan)
    bottle_data = bottle_data.replace(to_replace=-9999999, value=np.nan)
    
    # Convert times from strings to pandas datetime objects
    bottle_data["Start Time [UTC]"] = bottle_data["Start Time [UTC]"].apply(lambda x: convert_times(x))
    bottle_data["CTD Bottle Closure Time [UTC]"] = bottle_data["CTD Bottle Closure Time [UTC]"].apply(lambda x: convert_times(x))

    # Convert any values with a "<", which indicates a value not statistically significant from zero, with zero
    bottle_data = bottle_data.applymap(not_statistically_sigificant)
    
    # Interpret the quality flags to QARTOD flag values
    for col in bottle_data.columns:
        if "Flag" in col:
            if "CTD" in col and "File" not in col:
                bottle_data[col] = bottle_data[col].apply(lambda x: map_ctd_flag(x))
            elif "Discrete" in col:
                bottle_data[col] = bottle_data[col].apply(lambda x: map_discrete_flag(x))
            elif "Replicate" in col:
                bottle_data[col] = bottle_data[col].apply(lambda x: map_replicate_flag(x))
            elif "Niskin" in col:
                bottle_data[col] = bottle_data[col].apply(lambda x: map_niskin_flag(x))
            else:
                pass
            
    return bottle_data