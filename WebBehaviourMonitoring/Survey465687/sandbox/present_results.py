from WebBehaviourMonitoring.Survey465687.sandbox import *
from math import *
from pylab import *
from matplotlib.lines import Line2D
from matplotlib import animation
#import seaborn as sns


def movement_prototype(track_variables, context_variables, t_abandon):
    ev = track_variables['events'][0]
    p = find(ev == 1)
    begin = p[3]
    end = p[4]
    # begin = p[3]
    # end = int(p[3] + (p[4]-p[3])/3)
    # end = p[4]
    ax = figure(1)
    time_variables, space_variables, context_variables = interpolate_data(track_variables, context_variables, t_abandon,
                                                                          begin=begin, end=end)

    x = track_variables['x'][0].tolist()[begin:end-2]
    y = track_variables['y'][0].tolist()[begin:end-2]

    xs = space_variables['xs'][0].tolist()
    ys = space_variables['ys'][0].tolist()

    t = track_variables['t'][0].tolist()[begin:end-2]
    tt = time_variables['tt'][0].tolist()
    xt = time_variables['xt'][0].tolist()
    yt = time_variables['yt'][0].tolist()

    ss = space_variables['ss'][0].tolist()
    angle = space_variables['angles'][0].tolist()
    w = space_variables['w'][0].tolist()
    curvature = space_variables['curvatures'][0].tolist()
    var_curvature = space_variables['var_curvatures'][0].tolist()

    ttv = time_variables['ttv'][0].tolist()
    vt = time_variables['vt'][0].tolist()
    vx = time_variables['vx'][0].tolist()
    vy = time_variables['vy'][0].tolist()
    a = time_variables['a'][0].tolist()
    jerk = time_variables['jerk'][0].tolist()

    # x and y visualization
    xlabel('x (px)', fontsize=15)
    xticks(fontsize=15)
    ylabel('y (px)', fontsize=15)
    yticks(fontsize=15)
    plot(xs, ys, marker='o', color='none',
         markeredgecolor='#0080FF', markersize=5, markeredgewidth=1)
    plot(x, y, 'o',
         markerfacecolor='orange', markeredgecolor='orange', markersize=5)
    axis('equal')
    plot(x[0], y[0], marker='o', markerfacecolor='k', markeredgecolor='k', markersize=5)
    gca().invert_yaxis()
    gca().invert_xaxis()
    show()

    subplot(121)
    xlabel('time (s)')
    ylabel('x (px)')
    plot(tt, xt, 'o', color='none', markeredgecolor='#0080FF', markersize=5, markeredgewidth=1)
    plot(t, x, 'o', markerfacecolor='orange', markeredgecolor='orange', markersize=4)

    #
    subplot(122)
    xlabel('time (s)')
    ylabel('y (px)')
    plot(tt, yt, 'o',  color='none', markeredgecolor='#0080FF', markersize=5, markeredgewidth=1)
    plot(t, y, 'o', markerfacecolor='orange', markeredgecolor='orange', markersize=4)
    show()

    subplot(221)
    xlabel('s (px)')
    ylabel('$\Theta$')
    plot(ss[:-1], angle, 'k')
    subplot(222)
    xlabel('s (px)')
    ylabel('w')
    plot(ss[:-1], w, 'k')
    subplot(223)
    xlabel('s (px)')
    ylabel('c')
    plot(ss[:-2], curvature, 'k')
    subplot(224)
    xlabel('s (px)')
    ylabel("c'")
    plot(ss[:-3], var_curvature, 'k')
    show()

    xlabel('time (s)')
    ylabel('velocity (px/s)')
    plot(ttv, vt, 'k')
    show()

    subplot(221)
    xlabel('time (s)')
    ylabel('horizontal velocity (px/s)')
    plot(ttv, vx, 'k')
    subplot(222)
    xlabel('time (s)')
    ylabel('vertical velocity (px/s)')
    plot(ttv, vy, 'k')
    subplot(223)
    xlabel('time (s)')
    ylabel('a (px/s)')
    plot(ttv[:-1], a, 'k')
    subplot(224)
    xlabel('time (s)')
    ylabel('jerk (px/s)')
    plot(ttv[:-2], jerk, 'k')
    show()


def save_results(df, subj_pos, time_variables, space_variables, context_variables, variables_range):
    """ This function saves parameters in an empty dataframe.

    Parameters
    ----------
    df: dataframe
      with id and survey scales, to join all the results.
    subj_pos: int
      position in doc to get the line to add info
    time_variables: dataframe
      to save temporal features
    space_variables: dataframe
      to save spatial features
    context_variables: dataframe
      to save contextual features
    variables_range: dataframe
      join every people features to further plot violin

    Returns
    -------
    df: dataframe
      with columns and respective results
    """

    space_variables['s'] = space_variables['s'][0].tolist()[-1]
    space_variables = space_variables.drop(['xs', 'ys', 'ss'], 1)
    time_variables = time_variables.drop(['xt', 'yt', 'tt', 'ttv'], 1)
    context_variables = context_variables.drop(['nr_items', 'group'], 1)

    # to form variables_range - further violin analysis
    dframes = [context_variables, time_variables, space_variables]
    for d in dframes:
        for i in variables_range.columns.tolist():
            if i in d.columns.tolist():
                if len(variables_range[i]) == 0 or (size(variables_range[i][0]) == 1 and isnan(variables_range[i][0])):
                    if size(d[i][0]) == 1:
                        variables_range[i] = [[d[i][0]]]
                    else:
                        variables_range[i] = d[i]
                else:
                    if size(d[i][0]) == 1:
                        variables_range[i] = [array(variables_range[i][0]).tolist() + [d[i][0]]]
                    else:
                        variables_range[i] = [array(variables_range[i][0]).tolist() + d[i][0].tolist()]

    colname = []
    col = []

    for d in dframes:
        for i in d.columns.tolist():
            if i == 'vt' or i == 'vx' or i == 'vy':
                zero = 1
            else:
                zero = 0
            if i in ['nr_abandons', 'nr_scroll', 'nr_items_scroll', 'nr_revisit', 'nr_correct_within_item',
                     'nr_correct_between_item', 'nr_pauses', 'jitter', 's']:
                colname += [i]
                col += [d[i][0].tolist()]
            else:
                max_value, min_value, mean_value, std_value = get_statistics(d[i][0].tolist(), zero=zero)
                colname += [i + '_max']
                col += [max_value]
                colname += [i + '_min']
                col += [min_value]
                colname += [i + '_mean']
                col += [mean_value]
                colname += [i + '_std']
                col += [std_value]

    for i, j in zip(colname, col):
        df.loc[subj_pos, i] = j

    return df, variables_range


def multilineplot_zones(subj_pos, _t, time_variables, track_variables, ei, _title=None):
    """ This function plots in a pdf a representation of the signal, events and questions.

    Parameters
    ----------
    subj_pos: int
      identification of person, to name figure
    _t: array
      original time (seconds)
    time_variables: dataframe
      to use tt time interpolated (seconds), sig float signal to plot
    track_variables: dataframe
      to use events int mouse events, items int items order
    ei: array
      index of item change (int)
    _title: str

    Returns
    -------
    file: pdf
      representation of the signal
    """
    t = time_variables['ttv'][0]
    sig = time_variables['vt'][0]
    events = find(track_variables['events'][0] != 0)
    items = track_variables['items_order'][0]
    ei = concatenate(([0], ei))
    lt = t[len(t) - 1]
    nplots = int(lt // 20 + 1)
    ma_x = sig[argmax(sig)]
    mi_x = sig[argmin(sig)]
    figure(figsize=(20, 1.5 * nplots), dpi=120)
    title(_title)
    end = 0
    for i in range(nplots):
        subplot(nplots, 1, i + 1)
        start = end
        istart = find(t >= start)[0]
        end = start + 20
        iend = find(t <= end)[-1]
        plot(t[istart:iend], sig[istart:iend])
        grid(False)
        axis((start, end, 0, ma_x))
        ax = gca()
        ax.yaxis.set_visible(False)
        tevents = _t[events]
        click_e = tevents[(tevents >= start) & (tevents < end)]  # click events
        if len(click_e) > 0:
            # vlines(e,mi_x,ma_x-(ma_x-mi_x)/4.*2., lw=2)
            vlines(click_e, 0, ma_x, lw=2)

        _color = ['#9ABFFF', '#99CCFF']
        _alpha = [1, 0.5]
        if len(ei) > 0:
            for j in arange(1, len(ei)):
                ax.add_patch(
                    Rectangle((_t[ei[j - 1]], mi_x), _t[ei[j]] - _t[ei[j - 1]], ma_x - mi_x, color=_color[j % 2],
                              alpha=_alpha[j % 2]))
                ax.text(_t[ei[j - 1]] + 0.03, (ma_x - mi_x) / 2, items[j - 1], fontsize=20)
            ax.text(_t[ei[-1]] + 0.03, (ma_x - mi_x) / 2, items[-1], fontsize=20)
    tight_layout()
    #show()
    savefig('../data/multiplotline/' + str(subj_pos) + '_v_multiplotline.pdf')
    close()


def plot_path(server_info, track_variables, t):
    """ This function plots the mouse with events in realtime.

    Parameters
    ----------
    server_info: dataframe
      to use person_ip
    track_variables: dataframe
      to use x: int with mouse cursor x position (px)
      y: int with mouse cursor y position (px)
      ev: int with number of event
    t: array
      float of time (sec)

    Returns
    -------
    figure with plot of xy and events
    """

    person_id = server_info['person_ip'][0]
    x = track_variables['x'][0]
    y = track_variables['y'][0]
    ev = track_variables['events'][0]

    if person_id == '130.60.69.150':
        # 1	1	2	2	2	1	3	3	1	1	3	2	2	2	1	1	3	4
        y = y[:] * 4 - 500
        x = x[:] + 500
    if person_id == '84.226.30.100':
        # FIRST
        # 1	1	2	2	2	1	3	3	1	1	3	2	2	2	1	1	3	4
        y = y[:] * 1.55 + 15
        x = x[:] + 500
    if person_id == '178.197.226.48':
        # SECOND
        # 2	1	1	2	3	1	1	4	2	4	1	1	2	1	1	2	1	3
        y = y[:] * 1.55
        x = x[:] + 550
    if person_id == '178.197.233.170':
        # 5	2	5	4	5	4	4	2	4	4	2	3	3	4	3	2	3	3
        y = y[:] * 1.55 + 20
        x = x[:] * 1.1 + 380
    if person_id == '178.192.253.94':
        # 5	2	5	4	5	4	4	2	4	4	2	3	3	4	3	2	3	3
        y = y[:] * 1.55 + 20
        x = x[:] * 1.1 + 350
    if person_id == '89.206.92.230':
        # 2	3	1	4	4	5	4	2	4	3	5	4	4	4	3	4	4	2
        y = y[:] * 1.55 + 20
        x = x[:] * 1.1 + 370
    if person_id == '89.206.65.132':
        # THIRD
        # 5	4	5	5	3	5	5	5	5	5	3	2	1	2	2	1	3	1
        y = y[:] * 1.45
        x = x[:] * 1.1 + 650
    if person_id == '178.82.234.24':
        # 5	2	1	3	3	5	5	4	4	1	5	3	5	5	5	4	5	1
        y = y[:] * 1.53 + 40
        x = x[:] * 1.1 + 460
    if person_id == '80.219.210.230':
        # FOURTH
        # 1	1	2	4	2	1	4	2	2	4	3	2	2	2	3	3	1	4
        y = y[:] * 1.53 + 40
        x = x[:] * 1.1 + 300

    # Complete plot
    img = imread("../images/old_max_image.png")
    plt.axis([450, 1910, 10, 2155])
    plt.imshow(img, extent=(450, 1910, 2155, 10))
    plot(x, y, 'k')
    p = find(ev == 1)
    scatter(x[p], y[p], color='red', marker='o')
    gca().invert_yaxis()
    gca().axes.get_xaxis().set_ticks([])
    gca().axes.get_yaxis().set_ticks([])
    show()

    x = x[find(diff(t) > 0)]
    y = y[find(diff(t) > 0)]
    ev = ev[find(diff(t) > 0)]
    t = t[find(diff(t) > 0)]

    class SubplotAnimation(animation.TimedAnimation):
        def __init__(self):
            self.x = x
            self.y = y
            self.t = t
            self.markersx = []
            self.markersy = []
            backimg = imread("../images/old_max_image.png")
            fig = plt.figure(figsize=(100, 150))
            ax1 = fig.add_subplot(1, 2, 1)

            gca().axes.get_xaxis().set_ticks([])
            gca().axes.get_yaxis().set_ticks([])
            # self.line1 = Line2D([], [], color='black')  # plot line
            self.line1 = Line2D([], [], color='black', marker='.', markersize=2, lw=0)  # plot dots
            self.line1e = Line2D(
                [], [], color='red', marker='o', markeredgecolor='r', markersize=4, lw=0)
            ax1.add_line(self.line1)
            ax1.add_line(self.line1e)
            ax1.imshow(backimg, extent=(450, 1910, 2155, 10))

            ax1.set_xlim(450, 1910)
            ax1.set_ylim(10, 2155)
            gca().invert_yaxis()
            animation.TimedAnimation.__init__(self, fig, interval=50, blit=True, repeat=False)

        def _draw_frame(self, framedata):
            i = framedata

            animation.TimedAnimation._interval = diff(self.t)[i] * 1000

            v = find(t < 5)[-1]
            if t[i] < 5:
                self.line1.set_data(self.x[:i], self.y[:i])
            elif 5 <= t[i] <= t[-1] - 5:
                self.line1.set_data(self.x[i - v:i], self.y[i - v:i])
            else:
                self.line1.set_data(self.x[i - v:i], self.y[i - v:i])

            pp = find(ev == 1)
            if i in pp:
                self.markersx.append(self.x[i])
                self.markersy.append(self.y[i])
                self.line1e.set_data(self.markersx, self.markersy)

            self._drawn_artists = [self.line1, self.line1e]

        def new_frame_seq(self):
            return iter(range(self.x.size))

        def _init_draw(self):
            lines = [self.line1, self.line1e]
            for l in lines:
                l.set_data([], [])

    SubplotAnimation()
    # plt.show()


def plot_path_frac(subj_pos, space_variables, track_variables):
    """ This function plots a fraction of the mouse with events.

    Parameters
    ----------
    subj_pos: int
      identification of person, to name figure
    space_variables: dataframe
      to use _x and _y (x and y position px)
    track_variables: dataframe
      to use x and y (x and y position interpolated px)
      to use ev (events)

    Returns
    -------
    save figure with plot of xy and events in data - interpolation plot folder
    """

    x = space_variables['xs'][0]
    y = space_variables['ys'][0]
    _x = track_variables['x'][0]
    _y = track_variables['y'][0]

    ev = track_variables['events'][0]
    p = find(ev == 1)
    use_x = []
    use_y = []
    for k, l in zip((arange(0, len(_x[p[12]:p[13]]))), (arange(0, len(_y[p[12]:p[13]])))):
        for i, j in zip((arange(0, len(x))), (arange(0, len(y)))):
            if abs(_x[p[12]:p[13]][k] - x[i]) < 0.3 and abs(_y[p[12]:p[13]][l] - y[j]) < 0.3:
                use_x.append(i)
                use_y.append(j)
    figure(1)
    plot(x[use_x], y[use_y], marker='o', mfc='0.9', mec='k', mew=0.5, markersize=4, lw=0)
    plot(_x[p[12]:p[13]], _y[p[12]:p[13]], marker='.', color="orange", markersize=5, lw=0)
    plot(_x[p[12]], _y[p[12]], color='red', marker='*', markeredgecolor='r', markersize=6, lw=0)
    plot(_x[p[13]], _y[p[13]], color='red', marker='o', markeredgecolor='r', markersize=4, lw=0)
    gca().invert_yaxis()
    gca().invert_xaxis()
    savefig('../data/interpolation_plot/' + str(subj_pos) + '_interpolation_plot.pdf')
    close()


def violin_results(variables_range):
    """ This function plots the violins of each feacture.

        Parameters
        ----------
        variables_range: dataframe
          identification of person, to name figure

        Returns
        -------
        save figures with plots
    """
    text_file = open("../data/violin/featuresRange.txt", "w")
    range_str = ""

    m_01 = ['vt', 'vx', 'vy', 'jerk', 'inter_item_interval']
    m_001 = ['w', 'curvatures']
    m_00001 = ['var_curvatures']

    for i in variables_range.columns.tolist():
        if i in m_01:
            m = 0.1
        elif i in m_001:
            m = 0.01
        elif i in m_00001:
            m = 0.0001
        else:
            m = 1

        sns.set_style("white")
        range_str += i + "\n" + str(sort(variables_range[i][0])[0]) + "\n" + str(sort(variables_range[i][0])[-1]) + "\n"
        sns.violinplot(reject_outliers(variables_range[i][0], m=m))
        sns.despine(left=True, bottom=True)
        tick_params(labelbottom='off')
        savefig('../data/violin/' + i + '.pdf')
        close()

    text_file.write(range_str)
    text_file.close()
