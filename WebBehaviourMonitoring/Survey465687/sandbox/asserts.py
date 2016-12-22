from pylab import *


def is_new_step(server_info, context_variables, _ip, _group):
    """ This function verifies is new file is or not a new group.
     To concatenate fragmented files in the same info

    Parameters
    ----------
    server_info: dataframe
      to use index step and person_ip
    context_variables: dataframe
      to use index group
    _ip: int
      previous ip
    _group: string
      previous group.

    Returns
    -------
    True: bool
      if is a new step.
    False: bool
      if is the same step.
    """
    s = server_info['step'][0]
    ip = server_info['person_ip'][0]
    group = context_variables['group'][0]

    if group == 0:
        return True
    else:
        if ip == _ip and s != -1 and group == _group:
            return False
        else:
            return True


def is_tablet(track_variables):
    """ This function verifies if the subject used an touch device.
    Compare movement (ev=0) with click (ev=1). If tablet they are similar.
    In this case we lose information.

    Parameters
    ----------
    track_variables: dataframe
      to use index events

    Returns
    -------
    True: bool
      if is tablet.
    False: bool
      if is not tablet.
    """

    ev = track_variables['events'][0]
    if len(find(ev == 1)) > 0:
        ratio = abs(len(find(ev == 0)) - len(find(ev == 1)))
        ratio2 = len(find(ev == 0)) / len(find(ev == 1))
        if ratio <= 1 or ratio2 <= 2:
            return True
        else:
            return False


def assert_answer_position(track_variables, group):
    """ This function asserts that the answers positions is correct.

    Parameters
    ----------
    track_variables: dataframe
      to use index answers and x
    group: string
      to identify answers positions

    Returns
    -------
    file_error: int
      number of times that position is wrong
    """

    ans = track_variables['answers'][0].tolist()
    x = track_variables['x'][0].tolist()
    ans = array(ans)
    x = array(x)
    unknown_error = 0
    file_error = 0
    if group == "NewMaximizer" or group=="OldMaximizer":
        pos1 = x[find(ans == 1)]
        pos2 = x[find(ans == 5)]
        pos3 = x[find(ans == 4)]
        pos4 = x[find(ans == 3)]
        pos5 = x[find(ans == 2)]

    if group == "SPF":
        pos1 = x[find(ans == 1)]
        pos2 = x[find(ans == 2)]
        pos3 = x[find(ans == 5)]
        pos4 = x[find(ans == 4)]
        pos5 = x[find(ans == 3)]

    if group == "ARES":
        pos1 = x[find(ans == 1)]
        pos2 = x[find(ans == 2)]
        pos3 = x[find(ans == 3)]
        pos4 = x[find(ans == 4)]

    if group == "NEO":
        pos1 = x[find(ans == 1)]
        pos2 = x[find(ans == 2)]
        pos3 = x[find(ans == 3)]
        pos4 = x[find(ans == 4)]
        pos5 = x[find(ans == 5)]

    if len(pos1) > 0 and len(pos2) > 0:
        if mean(pos2) > mean(pos1):
            unknown_error += 1
        else:
            file_error += 1
            # print("Pos 1 and 2 not correct")
    if len(pos2) > 0 and len(pos3) > 0:
        if mean(pos3) > mean(pos2):
            unknown_error += 1
        else:
            file_error += 1
            # print("Pos 2 and 3 not correct")
    if len(pos3) > 0 and len(pos4) > 0:
        if mean(pos4) > mean(pos3):
            unknown_error += 1
        else:
            file_error += 1
            # print("Pos 3 and 4 not correct")
    if len(pos4) > 0 and len(pos5) > 0:
        if mean(pos5) > mean(pos4):
            unknown_error += 1
        else:
            file_error += 1
            # print("Pos 4 and 5 not correct")

    return file_error
