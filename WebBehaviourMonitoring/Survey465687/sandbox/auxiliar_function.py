from pylab import *
import os


def files_with_pattern(directory):
    for path, dirs, files in os.walk(directory):
        for i in files:
            yield os.path.join(path, i)


def get_files(directory, l_strings=''):
    """ This function receives one string or a list of strings.

    Parameters
    ----------
    directory: string
      the file directory.
    l_strings: string
      code in file.

    Returns
    -------
    list_files: array-like
      the files
    """
    if is_string_like(l_strings):
        return [i for i in files_with_pattern(directory) if l_strings in i]
    else:
        list_files = [i for i in files_with_pattern(directory)]
        for include_string in l_strings[:]:
            list_files = [i for i in list_files if include_string in i]
    return list_files


def crop_data(track_variables, time):
    limit = find((cumsum(diff(track_variables['t'][0])[1:])) <= time)[-1]
    for i in track_variables.columns:
        track_variables[i][0] = track_variables[i][0][:limit+1]

    return track_variables


def get_statistics(variable, zero=0):
    """ This function analyse statistically variables.

    Parameters
    ----------
    variable: array
      variable to analyse.
    zero: int
      consider values equal to 0.

    Returns
    -------
    var[argmax(var)]: float
      maximum value of array
    min_value: float
      minimum value of array
    mean(var): float
      mean value of array
    std(var): float
      standard deviation of array
    """
    variable = array(variable)
    if zero == 0:
        min_value = variable[argmin(variable)]
    else:
        min_value = variable[variable > 0][argmin(array(variable[variable > 0]))]

    return variable[argmax(variable)], min_value, mean(variable), std(variable)


def round_dig(x):
    """ This function returns the number of significant numbers. To round values.

    Parameters
    ----------
    x: float
      value to analyse

    Returns
    -------
    int significant digits
    """
    return -int(floor(log10(abs(x))))


def reject_outliers(data, m=1):
    """ This function reject outliers based on mean and std

    Parameters
    ----------
    data: list
      list to reject outliers
    m: int
      factor to remove the outliers

    Returns
    -------
    data: array
      list without outliers
    """
    data = array(data)
    return data[abs(data - mean(data)) < m * std(data)]
