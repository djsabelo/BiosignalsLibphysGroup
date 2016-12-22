from WebBehaviourMonitoring.Survey465687.sandbox import *
from pylab import *
import pandas
from scipy import signal
from novainstrumentation.smooth import smooth

# sample_freq = 0.5     time between velocity - tested 0.1 and 1


def parameters_analysis(i_change_question, track_variables, context_variables, survey):
    """ This function computes context variables.

    Parameters
    ----------
    i_change_question: array
      int index of item change.
    track_variables: dataframe
      to use indexes items, events and time
    context_variables: dataframe
      to use index nr_items

    Returns
    -------
    new_i_change_question: array
      int index of question change wout scroll
    nr_items_orig: int
      number of items in the beginning of processing
    abandon_t_quest: int
      value to consider an abandon event
    track_variables: dataframe
      change index items_order (array exclude scroll items)
    context_variables: dataframe
      add index t_quest (array time per question wout scroll (seconds))
      add index nr_abandons (nr of times that people spend t_abandon times the mean time per question (#/items))
      add index nr_scroll (nr of times that people scroll the page (#/items))
      add index nr_items_scroll (total number of items scrolled (#/items))
      add index nr_revisit (nr of times that people go back to old question (#/items))
      add index nr_correct_within_item (nr of times of corrections inside the question (#/items))
      add index nr_correct_between_item (nr of times that go back and correct a previous question (#/items))
      add index inter_items_interval (interval of time between the last click and enter the next question (sec))
      add index time_click (interval of time between click in and click out (sec))
    """

    items = track_variables['items'][0]
    ev = track_variables['events'][0]
    t = track_variables['t'][0]
    items = array(items)
    # time per question
    end = 0  # for the first question
    n_quest = len(i_change_question)
    t_quest = []

    context_variables = count_items(items[items != -1], context_variables, survey)
    nr_items_orig = context_variables['nr_items'][0]

    for i in arange(0, n_quest):  # time and x in the beginning of answer
        start = end
        end = t[i_change_question[i]]
        t_quest.append(end - start)
    t_quest_f = signal.medfilt(t_quest)  # filter of time and items

    # abandonment and scroll
    new_t_quest = []
    abandon = []
    items_order = []
    items_to_rev = []
    new_i_change_question = []
    nr_scroll = 0
    items_scroll = []
    nr_items_scroll = 0
    scroll_t_quest = 0.1  # mean(t_quest_f)/4
    abandon_t_quest = 10 * mean(t_quest_f)
    t_review = 1

    # abandon events
    # scroll events
    for i in arange(0, len(t_quest)):
        if t_quest[i] > abandon_t_quest:
            abandon.append(t[i_change_question[i]])
            items_order.append(items[i_change_question[i]].tolist())
            new_i_change_question.append(i_change_question[i])
            if t_quest[i] > t_review:
                items_to_rev.append(items[i_change_question[i] - 1].tolist())
        elif t_quest[i] < scroll_t_quest:
            items_scroll.append(t[i_change_question[i]])
        else:
            new_i_change_question.append(i_change_question[i])
            items_order.append(items[i_change_question[i] - 1].tolist())
            new_t_quest.append(t_quest[i])
            if t_quest[i] > t_review:
                items_to_rev.append(items[i_change_question[i] - 1].tolist())
    items_order.append(items[-1].tolist())
    items_to_rev.append(items[-1].tolist())
    items_scroll.sort()
    i_change_scroll = [i for i in arange(0, len(items_scroll) - 1)
                       if items_scroll[i + 1] - items_scroll[i] > scroll_t_quest]
    i_change_scroll.append(len(items_scroll) - 1)
    begin = 0
    for i in arange(0, len(i_change_scroll)):
        part_scroll = items_scroll[begin:i_change_scroll[i] + 1]
        begin = i_change_scroll[i] + 1
        if len(part_scroll) > 1:
            nr_scroll += 1
            nr_items_scroll += len(part_scroll)

    # Revisit
    nr_revisit = 0
    items_order = array(items_order)
    new_i_change_question = array(new_i_change_question)
    items_order_t = concatenate((items_order, [5000]))
    items_order = items_order[diff(items_order_t) != 0]
    items_order = list(items_order)
    new_i_change_question = new_i_change_question[diff(items_order_t[:-1]) != 0]
    my_dict = {i: items_order.count(i) for i in items_order}  # item: nr times
    if -1 in my_dict:
        del my_dict[-1]
    nr_revisit = [nr_revisit + my_dict[i] - 1 for i in my_dict if my_dict[i] > 1]

    # Return nr_correct_within_item (int)
    # nr_correct_between_item (int)
    it_click = items[ev == 1]
    it_click = it_click[find(it_click != -1)]
    nr_correct_within_item = len(find(diff(it_click) == 0))  # click in the same item

    nr_correct_between_item = [0]
    it_click = it_click.tolist()
    it_click_corr = {i: it_click.count(i) for i in it_click}  # item: nr times clicked

    nr_correct_between_item = [nr_correct_between_item[-1] + it_click_corr[i] - 1 for i in it_click_corr
                               if it_click_corr[i] > 1]
    if len(nr_correct_between_item) > 0:
        nr_correct_between_item = sum(nr_correct_between_item, 0) - nr_correct_within_item
    else:
        nr_correct_between_item = 0
    if len(nr_revisit) > 0:
        nr_revisit = sum(nr_revisit, 0) - nr_correct_between_item - nr_correct_within_item
    else:
        nr_revisit = 0

    # Return inter_items_interval (sec)
    inter_items_interval = []
    t_click = [t[i] for i in find(ev == 1) if items[i] != -1]

    t_new_item = [t[i] for i in new_i_change_question if items[i] != -1]
    for i in t_click:
        t_dif = t_new_item - i
        t_dif = t_dif[t_dif > 0]
        if len(t_dif) > 0:
            inter_items_interval.append(min(t_dif))
    inter_items_interval = array(inter_items_interval)
    inter_items_interval = inter_items_interval[inter_items_interval < abandon_t_quest]

    # time click
    click_in = [t[i] for i in find(ev == 1) if items[i] != -1]
    click_out = [t[i] for i in find(ev == 4) if items[i] != -1]
    if len(click_in) >= len(click_out):
        time_click = array(click_out) - array(click_in[:len(click_out)])
    else:
        time_click = array(click_out[:len(click_in)]) - array(click_in)

    time_click = time_click[find(time_click > 0)]
    if len(find(time_click < 0)) > 0:
        print("ERROR", len(find(time_click < 0)))
        print("ERROR", time_click)

    track_variables['items_order'] = [items_order]
    context_variables['t_quest'] = [array(new_t_quest)]
    context_variables['nr_abandons'] = [len(abandon)]
    context_variables['nr_scroll'] = [nr_scroll]
    context_variables['nr_items_scroll'] = [nr_items_scroll]
    context_variables['nr_revisit'] = [nr_revisit]
    context_variables['nr_correct_within_item'] = [nr_correct_within_item]
    context_variables['nr_correct_between_item'] = [nr_correct_between_item]
    context_variables['inter_item_interval'] = [array(inter_items_interval)]
    context_variables['time_click'] = [time_click]

    return new_i_change_question, nr_items_orig, abandon_t_quest, track_variables, context_variables


def interpolate_data(track_variables, context_variables, t_abandon, t_crop=Inf, begin=0, end=-1):
    """ This function compute the spatial and temporal data.

    Parameters
    ----------
    track_variables: dataframe
      to use indexes time, x and y
    context_variables: dataframe
      to join features
    t_abandon: float
      time established to consider an abandon event

    Returns
    -------
    time_variables: dataframe
      indexes
      xt: array (x interpolated in time (px))
      yt: array (y interpolated in time (px))
      tt: array (time interpolated (sec))
      vt: array (velocity in time (px/sec))
      vx: array (horizontal velocity in time (px/sec))
      vy: array (vertical velocity in time (px/sec))
      a: array (acceleration (px/sec²))
      jerk: array (jerk (px/sec³))
    space_variables: dataframe
      indexes
      xs: array (x interpolated in space (px))
      ys: array (y interpolated in space (px))
      l_strokes: array (distance for stroke (px/items))
      straightness: array (real distance/shorter distance (px/px))
      jitter: array (tremors analysis, relation between original and smooth path)
      s: array (cumulative distance with mouse (px))
      angles: array (angle spacial movement (rad))
      w: array (angular velocity (rad/sec))
      curvatures: array (1/R curvature spatial movement (rad/px))
      var_curvatures: array (curvature variation in space (rad/px²))
    context_variables: dataframe
      add index nr_pauses: int total number of pauses
      add index t_pauses: array time of each pause
    """

    t = track_variables['t'][0].tolist()[begin:end]
    x = track_variables['x'][0].tolist()[begin:end]
    y = track_variables['y'][0].tolist()[begin:end]

    x = array(x)
    y = array(y)
    _s = get_s(x, y)
    x_f = smooth(x)
    y_f = smooth(y)
    s_f = get_s(x_f, y_f)
    jitter = s_f[-1] / _s[-1]

    t_temp = t

    # Detect time interpolation factor by mean of dt
    dt = diff(t)
    dt = dt[dt != 0]
    min_t = min(dt)
    dig_int = round_dig(min_t)
    interp_f = round(min_t, dig_int)
    if t[-1] / interp_f > 200000000:  # 200000000 memory error
        interp_f = t[-1] / 200000000

    # Detect spatial interpolation factor by min of ds
    _s = get_s(x, y)
    ds = diff(_s)
    ds = ds[ds != 0]
    min_s = min(ds)
    dig_int_s = round_dig(min_s)
    interp_s = round(min_s, dig_int_s)

    # find pauses
    #i_inter = list(find(diff(t) <= 1))
    i_inter = arange(0, len(t))
    t_pauses, t_all, xt, yt, xs, ys, ts, angles, w, curvatures, var_curvatures, l_strokes, straightness = ([] for _ in
                                                                                                           range(13))
    inter = []
    t_inter_total = []
    i_inter_total = []
    for i in arange(0, len(i_inter)-1):
        if t[i_inter[i+1]]-t[i_inter[i]] < 1:
            inter.append(i_inter[i+1])
            if i == len(i_inter)-2:
                inter.insert(0, inter[0] - 1)
                t_inter_total.append(array(t)[inter])
                i_inter_total.append(inter)
        elif len(inter) > 0:
            inter.insert(0, inter[0] - 1)
            t_inter_total.append(array(t)[inter])
            i_inter_total.append(inter)
            inter = []
    for i in arange(0, len(t_inter_total)-1):
        if len(t_pauses) == 0:
            t_pauses.append(t_inter_total[i][0])
        else:
            t_pauses.append(t_inter_total[i+1][0] - t_inter_total[i][-1])

    nr_pauses = len(t_pauses)
    # interpolate/stroke
    t_interac_acum = 0
    for i in arange(0, len(t_inter_total)):
        if (i_inter_total[i][-1] - i_inter_total[i][0]) > 2:
            begin_t = i_inter_total[i][0]
            end_t = i_inter_total[i][-1]
            t_interac_acum += t[end_t]-t[begin_t]
            if t_interac_acum > 0:
            # if 10 < t_interac_acum < 10+t_crop:
                t_slice, xt_slice, yt_slice = get_path_smooth_t(t, x, y, begin_t, end_t+1, ttol=interp_f)
                t_all = concatenate((t_all, t_slice))
                xt = concatenate((xt, xt_slice))
                yt = concatenate((yt, yt_slice))
                xs_slice, ys_slice, ts_slice = \
                    get_path_smooth_s(t_temp, x, y, begin_t, end_t+1, stol=interp_s)
                xs = concatenate((xs, xs_slice))
                ys = concatenate((ys, ys_slice))
                ts = concatenate((ts, ts_slice))
                s_strokes = get_s(xs_slice, ys_slice)
                if s_strokes[-1] > 0:
                    l_strokes.append(s_strokes[-1])
                    straightness.append((sqrt(((ys_slice[-1] - ys_slice[0]) ** 2) + ((xs_slice[-1] - xs_slice[0]) ** 2))) /
                                            s_strokes[-1])
    ss = get_s(xs, ys)
    # angle = atan(dy/dx)
    # unwrap removes discontinuities.
    angle_value = smooth(unwrap(arctan2(diff(ys) / diff(ss), diff(xs) / diff(ss))))

    # angular velocity
    w = angle_value / (diff(ss) / diff(ts))

    # c = (dx * ddy - dy * ddx) / ((dx^2+dy^2)^(3/2))
    curvature_top = (diff(xs) / diff(ss))[:-1] * diff(diff(ys) / diff(ss)) / (diff(ss)[:-1]) - \
                    (diff(ys) / diff(ss))[:-1] * diff(diff(xs) / diff(ss)) / (diff(ss)[:-1])
    curvature_bottom = ((diff(xs) / diff(ss)) ** 2 + (diff(ys) / diff(ss)) ** 2) ** (3 / 2.0)
    curvature = array(curvature_top / curvature_bottom[:-1])
    var_curvature = array(diff(curvature) / diff(ss)[:-2])

    st = get_s(xt, yt)
    _t = around(t_all, dig_int)

    # Save t_pauses > 1 sec
    # Save t_pauses < abandon
    t_pauses = array(t_pauses)
    t_pauses = t_pauses[t_pauses > 1.]
    t_pauses = t_pauses[t_pauses < t_abandon]

    # velocity moving and total
    vt_moving = get_v(t_all, st)

    t = arange(t_temp[0], t_temp[-1], interp_f)
    vt_moving = list(vt_moving)
    vx = list(abs(diff(xt) / diff(t_all)))
    vy = list(abs(diff(yt) / diff(t_all)))

    for i in arange(0, len(i_inter_total)-1):
        if i == 0:
            begin = [0]
            end = find(t - t_temp[i_inter_total[i][0]] > interp_f)
        else:
            begin = find(t - t_temp[i_inter_total[i-1][-1]] > interp_f)
            end = find(t - t_temp[i_inter_total[i][0]] > interp_f)
        zero_v = arange(begin[0], end[0])
        for j in zero_v:
            vt_moving.insert(j, 0)
            vx.insert(j, 0)
            vy.insert(j, 0)

    vt_moving.extend(zeros(len(t) - len(vt_moving)))
    vx.extend(zeros(len(t) - len(vx)))
    vy.extend(zeros(len(t) - len(vy)))
    vt = array(vt_moving)
    vx = array(vx)
    vy = array(vy)

    a = diff(vt) / diff(t)
    jerk = diff(a) / diff(t[:-1])

    space_variables = pandas.DataFrame({'xs': [xs],
                                        'ys': [ys],
                                        'l_strokes': [array(l_strokes)],
                                        'straightness': [array(straightness)],
                                        'jitter': [jitter],
                                        's': [st],
                                        'ss': [ss],
                                        'angles': [angle_value],
                                        'w': [w],
                                        'curvatures': [curvature],
                                        'var_curvatures': [var_curvature]
                                        })

    time_variables = pandas.DataFrame({'xt': [xt],
                                       'yt': [yt],
                                       'tt': [t_all],
                                       'ttv': [t],
                                       'vt': [vt],
                                       'vx': [vx],
                                       'vy': [vy],
                                       'a': [a],
                                       'jerk': [jerk]
                                       })

    context_variables['nr_pauses'] = {nr_pauses /
                                      float(context_variables['nr_items'][0])}
    context_variables['t_pauses'] = [t_pauses]

    return time_variables, space_variables, context_variables
