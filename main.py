from bokeh.models import ColumnDataSource, Panel, Whisker, Legend, LegendItem
from bokeh.layouts import row, column
from bokeh.models import BoxSelectTool, LassoSelectTool, CDSView, BooleanFilter, \
    Span
from bokeh.plotting import curdoc, figure
from bokeh.models.widgets import Select, Tabs, Slider, RangeSlider
from bokeh.models.tools import HoverTool
from modules.utils import *
from scripts.data_opener import wind, cn2_file2, time_series_data, max_corr, correlations, cn2_file1
from modules.config import file1_name, file2_name

date_keys = time_series_data['date_key'].unique().tolist()


select_bin_width = Slider(start=2, end=10, value=2, step=1, title="How many max mean periods to use as bin width", width=900)

date_keys_options = list(time_series_data['date_key'].unique())
date_keys_options.sort()
select_date_key = Select(title='Select day to inspect',
                     value='2009-04-01',
                     options=date_keys_options
                     )


a, b = max_corr.n_points.min(), max_corr.n_points.max()


select_n_samples = RangeSlider(start=a, end=b, value=(a, b), step=10, title="n_samples", width=900)





# -----------INTERVALS FOR CONTRIBUTIONS (in meters)--------------------

interval1 = (0, .75) * 1000
interval2 = (.75, 1.5) * 1000
interval3 = (1.5, 3) * 1000
interval4 = (3, 6) * 1000
interval5 = (6, 12) * 1000
interval6 = (12, 1000000000000000000000) * 1000
# ------------------SHIFT_MAX_CORR---------------------------



def make_dataset_for_scatter():
    """
    this function formats the data based on the sampling_rate
    :param sampling_rate: sampling_rate to use
    :return: filtered ColumnDataSource
    """
    condition1 = max_corr['bin_width'] == select_bin_width.value
    condition2 = select_n_samples.value[0] <= max_corr['n_points']
    condition3 = max_corr['n_points'] <= select_n_samples.value[1]
    by_bin = max_corr[condition1 & condition2 & condition3]
    return ColumnDataSource(by_bin)




def make_plot_shift_max_corr(src):
        # Blank plot with correct labels

        TOOLTIPS = [
            ("day", "@date_key"),
            ("lag in minutes", "@lag_in_minutes"),
            ("max cross_corr", "@cross_corr"),
            ("cross_corr_err", "@cross_corr_err")
        ]


        TOOLS = "pan,wheel_zoom,box_select,lasso_select,reset, hover"
        p = figure(tools=TOOLS, plot_width=900, plot_height=700, min_border=10,
                   min_border_left=50, tooltips=TOOLTIPS,
                   toolbar_location="above", y_axis_label='maximum cross corr value',
                   x_axis_label='corresponding shift in minutes',
                   title="Shift vs max_corr", y_range=None, x_range=None)

        p.background_fill_color = "#fafafa"
        p.select(BoxSelectTool).select_every_mousemove = False
        p.select(LassoSelectTool).select_every_mousemove = False

        joint = p.scatter(source=src, x='lag_in_minutes', y='cross_corr')





        return p, joint


def update_scatter_data_based_on_bin(attr, old, new):
    """
    it updates the plot based on the bin selector
    :param attr: value
    :param old: old bin
    :param new: new bin
    :return: None
    """
    # make new dataset
    new_src = make_dataset_for_scatter()

    #update the data
    scatter_src.data.update(new_src.data)

    return





def make_plot_hist_npoints():
        # Blank plot with correct labels


        TOOLTIPS = [
            ("count bin", "@top"),
            ("interval left", "@left"),
            ("interval right", "@right")
        ]

        TOOLS = "pan,wheel_zoom,box_select,lasso_select,reset,hover"
        p = figure(tools=TOOLS, plot_width=900, plot_height=700, min_border=10,
                   min_border_left=50, tooltips=TOOLTIPS,
                   toolbar_location="above", y_axis_label='frequency',
                   x_axis_label='number of points for one day',
                   title="Shift vs max_corr", y_range=None, x_range=None)

        p.background_fill_color = "#fafafa"
        p.select(BoxSelectTool).select_every_mousemove = False
        p.select(LassoSelectTool).select_every_mousemove = False

        vals, edges = np.histogram(max_corr[max_corr['bin_width'] == 2]['n_points'], bins=30)

        df = pd.DataFrame({'top': vals, 'bottom': 0, 'left':edges[:-1],
                           'right': edges[1:]})
        src = ColumnDataSource(df)

        p.quad(top='top', bottom='bottom', left='left', right='right', line_color = "#033649", source=src)


        return p




# ---------------------CORRELATIONS-----------------------



def make_correlation_src(date_key, bin_w):
    condition1 = correlations['date_key'] == date_key
    condition2 = correlations['bin_width'] == bin_w

    df = correlations[condition1 & condition2]
    return ColumnDataSource(df)

def make_plot_corr(src):

    TOOLS = "pan,wheel_zoom,box_select,lasso_select,reset,hover,save"
    p = figure(tools=TOOLS, plot_width=2*900, plot_height=400, min_border=10,
               min_border_left=50,
               toolbar_location="above",
               y_axis_label='cross corr value',
               x_axis_label='corresponding shift in minutes',
               title="Shift vs cross correlation", y_range=None, x_range=None)

    p.line(x='lag_in_minutes', y='cross_corr', source=src)


    p.add_layout(
        Whisker(source=src, base="lag_in_minutes", upper="upper", lower="lower")
    )


    return p


def update_corr_plot_based_on_bin(attr, old, new):
    bin_w = new
    date_key = select_date_key.value

    new_src = make_correlation_src(date_key, bin_w)
    corr_src.data.update(new_src.data)
    return


def update_corr_plot_based_on_datekey(attr, old, new):
    bin_w = select_bin_width.value
    date_key = new

    new_src = make_correlation_src(date_key, bin_w)
    corr_src.data.update(new_src.data)
    return


#-------------------Time Series---------------------------


def make_dataset_for_time_series(date_key):
    """
    This functions filter the data for the time series
    :param sampling_rate: sampling rato to filter
    :param date_key: date key to use
    :return: data filtered
    """
    by_date_key = time_series_data[time_series_data['date_key'] == date_key]
    by_date_key.sort_values('datetime', inplace=True)
    return ColumnDataSource(by_date_key), ColumnDataSource(by_date_key.interpolate('slinear'))


def make_plot_timeseries(src):
    TOOLS = "pan,wheel_zoom,box_select,lasso_select,reset,save"
    src1, src2 = src

    p = figure(tools=TOOLS, plot_width=900*2, plot_height=400, min_border=10,
               min_border_left=50,
               toolbar_location="above", y_axis_label='seeing',
               x_axis_label='UT time',
               title="Seeing time series", x_axis_type='datetime')
    hover = HoverTool(
        tooltips=[("datetime", "@datetime{%F %T}"),
                  (file1_name, "@seeing_file1"),
                  (file2_name, "@seeing_file2"),
                  ],
        formatters={'datetime': 'datetime'},
        mode='mouse',
        line_policy='nearest')

    p.tools.append(hover)

    p.background_fill_color = "#fafafa"
    p.select(BoxSelectTool).select_every_mousemove = False
    p.select(LassoSelectTool).select_every_mousemove = False



    p.line('datetime', 'seeing_file1', source=src2, legend_label=file1_name + ' raw data', color='blue', line_dash='dashed')
    p.line('datetime', 'seeing_file2', source=src2, legend_label=file2_name + ' raw data', color='red', line_dash='dashed')
    p.circle('datetime', 'seeing_file1', source=src1, legend_label=file1_name + ' raw data', color='blue')
    p.circle('datetime', 'seeing_file2', source=src1, legend_label=file2_name + ' raw data', color='red')

    p.legend.location = "top_right"
    p.legend.click_policy = "hide"


    return p



def update_select_date_key_based_on_selection(attr, old, new):
    indices = new
    if len(indices)==0:

        new_options = list(time_series_data[time_series_data['bin_width'] == select_bin_width.value]['date_key'].unique())
        new_options.sort()
        select_date_key.options = new_options
        select_date_key.value = select_date_key.options[0]
    else:
        new_options = list(set(scatter_src.data['date_key'][indices]))
        new_options.sort()
        select_date_key.options = new_options
    date_key = select_date_key.value
    new_src = make_dataset_for_time_series(date_key)
    time_series_src[0].data.update(new_src[0].data)
    time_series_src[1].data.update(new_src[1].data)
    return


def update_time_series_based_on_date_key(attr, old, new):
    date_key = new
    new_src1, new_src2 = make_dataset_for_time_series(date_key)
    time_series_src[0].data.update(new_src1.data)
    time_series_src[1].data.update(new_src2.data)




# -------------------contributions cn2---------------------------

def make_cn2_data(contibution_df):
    """
    This function creates a column data sourced from the contribution df
    ussing the actual date_key
    :return: ColumnDataSource
    """
    date_key = select_date_key.value
    cols = ['datetime', 'fixed_cn2_1', 'fixed_cn2_2', 'fixed_cn2_3',
            'fixed_cn2_4', 'fixed_cn2_5', 'fixed_cn2_6']
    df = contibution_df[contibution_df['date_key'] == date_key]
    return ColumnDataSource(df)

def make_cn2_plot_file1(src, name):
    cols = ['fixed_cn2_1', 'fixed_cn2_2', 'fixed_cn2_3',
            'fixed_cn2_4', 'fixed_cn2_5', 'fixed_cn2_6']
    TOOLS = "pan,wheel_zoom,box_select,lasso_select,reset,save"

    p = figure(width=2*900, height=400, x_axis_type="datetime",
               title=name + ' Cn2 profile', tools=TOOLS,
               x_range=cn2_file2_plot.x_range,
               y_range=cn2_file2_plot.y_range
               )
    hover = HoverTool(
        tooltips=[("datetime", "@datetime{%F %T}"),
                  ("fixed_cn2_1", "@fixed_cn2_1"),
                  ("fixed_cn2_2", "@fixed_cn2_2"),
                  ("fixed_cn2_3", "@fixed_cn2_3"),
                  ("fixed_cn2_4", "@fixed_cn2_4"),
                  ("fixed_cn2_5", "@fixed_cn2_5"),
                  ("fixed_cn2_6", "@fixed_cn2_6"),
                  ],
        formatters={'datetime': 'datetime'},
        mode='mouse',
        line_policy='nearest')
    p.tools.append(hover)
    mypalette = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#525252', '#a65628', '#f781bf', '#999999']

    for i, col in enumerate(cols):
        p.line(x='datetime', y=col, source=src, color=mypalette[i], line_width=2, legend_label=col)


    return p


def make_cn2_plot_file2(src, name):
    cols = ['fixed_cn2_1', 'fixed_cn2_2', 'fixed_cn2_3',
            'fixed_cn2_4', 'fixed_cn2_5', 'fixed_cn2_6']
    TOOLS = "pan,wheel_zoom,box_select,lasso_select,reset,save"

    p = figure(width=2*900, height=400, x_axis_type="datetime",
               title=name + ' Cn2 profile', tools=TOOLS)
    hover = HoverTool(
        tooltips=[("datetime", "@datetime{%F %T}"),
                  ("fixed_cn2_1", "@fixed_cn2_1"),
                  ("fixed_cn2_2", "@fixed_cn2_2"),
                  ("fixed_cn2_3", "@fixed_cn2_3"),
                  ("fixed_cn2_4", "@fixed_cn2_4"),
                  ("fixed_cn2_5", "@fixed_cn2_5"),
                  ("fixed_cn2_6", "@fixed_cn2_6"),
                  ],
        formatters={'datetime': 'datetime'},
        mode='mouse',
        line_policy='nearest')
    p.tools.append(hover)
    mypalette = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#525252', '#a65628', '#f781bf', '#999999']

    for i, col in enumerate(cols):
        p.line(x='datetime', y=col, source=src, color=mypalette[i], line_width=2, legend_label=col)


    return p

def update_cn2_plot_file2(attr, old, new):
    new_src = make_cn2_data(cn2_file2)
    cn2_file2_src.data.update(new_src.data)


def update_cn2_plot_file1(attr, old, new):
    new_src = make_cn2_data(cn2_file1)
    cn2_file1_src.data.update(new_src.data)


# wind information







def make_wind_data():
    date_key = select_date_key.value
    where_date_key = np.where(wind['date_key'] == date_key)
    ddtt = wind['datetime'][where_date_key]
    _, indexes_unique = np.unique(ddtt, return_index=True)
    mypalette = ['#440154', '#30678D', '#35B778', '#FDE724']

    ddtt = ddtt[indexes_unique]
    time_predicted = wind['time_predicted'][where_date_key][indexes_unique]
    height_m_s_n_m = wind['height_wind'][where_date_key][indexes_unique]
    actual_time = wind['max_corr_lag'][where_date_key][indexes_unique]
    dfs = []
    i = 0
    for height, tp, at, utdate in zip(height_m_s_n_m, time_predicted, actual_time, ddtt):
        utdate = pd.to_datetime(pd.to_datetime(utdate)).strftime('%Y-%m-%d %H:%M:%S')
        dfs.append(pd.DataFrame({'color_line': mypalette[i],'datetime':utdate ,'time_predicted': tp.reshape(-1), 'max_corr_lag': at.reshape(-1), 'height_over_sea_level -2600m': height.reshape(-1) - 2600}))
        i += 1
    if len(dfs)==0:
        return ColumnDataSource(pd.DataFrame({'color_line': [], 'datetime': [], 'time_predicted': [], 'max_corr_lag': [], 'height_over_sea_level -2600m': []}))
    return ColumnDataSource(pd.concat(dfs))



def make_wind_plot(src):
    TOOLS = "pan,wheel_zoom,box_select,lasso_select,reset,save,box_zoom"
    p = figure(width=2 * 900, height=900,
               title='Layer travel time from '+file1_name+' to '+ file2_name,
               y_axis_label='Time in minutes',
               x_axis_label='Height from sea level -2600m (approx Paranal) in meters',
               tools=TOOLS)


    mypalette = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#525252', '#a65628', '#f781bf', '#999999']

    legend = []
    for i, date_time in enumerate(np.unique(src.data['datetime'])):
        boolean = np.zeros((len(src.data['datetime'])), dtype=bool)
        filter_bool = np.where(src.data['datetime'] == date_time)
        colors = np.unique(src.data['color_line'][filter_bool]).tolist()[0]

        boolean[filter_bool] = True
        view = CDSView(source=src, filters=[BooleanFilter(boolean)])
        p.line(x='height_over_sea_level -2600m', y='time_predicted',
               line_width=2, view=view, source=src, color=colors)
        legend.append(LegendItem(label=src.data['datetime'][boolean][0], renderers=[p.renderers[i]]))

    legend1 = Legend(items=legend, location='top_right')
    p.add_layout(legend1)
    print(src.data['max_corr_lag'].shape)
    hline = Span(location=src.data['max_corr_lag'][0], dimension='width', line_color='black', line_width=1.5, line_dash='dashed')
    p.add_layout(hline)



    for i, h in enumerate([500, 1000, 2000, 4000, 8000, 16000]):
        vline = Span(location=h, dimension='height', line_color=mypalette[i], line_width=2.5, line_dash='dashed')
        p.add_layout(vline)


    return p


def update_wind_plot_based_on_date_key(attr, old, new):
    new_src = make_wind_data()
    wind_src.data.update(new_src.data)
    legend = []
    for i, date_time in enumerate(np.unique(new_src.data['datetime'])):
        boolean = np.zeros((len(new_src.data['datetime'])), dtype=bool)
        filter_bool = np.where(new_src.data['datetime'] == date_time)
        boolean[filter_bool] = True
        legend.append(LegendItem(label=new_src.data['datetime'][boolean][0], renderers=[wind_plot.renderers[i]]))
    wind_plot.legend.items = legend
    return



# END OF DEFINITIONS


# SCATTER LAYOUT

scatter_src = make_dataset_for_scatter()

# std_plot = make_plot_std(scatter_src)

shift_max_corr, for_scatter_selection = make_plot_shift_max_corr(scatter_src)

histogram = make_plot_hist_npoints()

#histogram = make_plot_hist_npoints(scatter_src)


# shift_vs_sampling_rate_src = make_dataset_shift_vs_sampling_rate(select_date_key.value)

# shift_vs_sampling_rate_plot = make_plot_sampling_rate_vs_shift(shift_vs_sampling_rate_src)

layout_scatter = row(column(shift_max_corr, select_bin_width), column(histogram, select_n_samples))#, histogram)

scatter_tab = Panel(child=layout_scatter, title='Scatter Plots')

# CORR

corr_src = make_correlation_src(select_date_key.value, select_bin_width.value)

corr_plot = make_plot_corr(corr_src)

# TIME SERIES LAYOUT

time_series_src = make_dataset_for_time_series(select_date_key.value)

time_series_plot = make_plot_timeseries(time_series_src)

layout_time_series = column(select_date_key, time_series_plot, corr_plot)#, filtered_table_scatter_data, shift_vs_sampling_rate_plot)

time_series_tab = Panel(child=layout_time_series, title='Time series and correlations')

# cn2 LAYOUT

cn2_file2_src = make_cn2_data(cn2_file2)
cn2_file2_plot = make_cn2_plot_file2(cn2_file2_src, file2_name)
cn2_file1_src = make_cn2_data(cn2_file1)
cn2_file1_plot = make_cn2_plot_file1(cn2_file1_src, file1_name)
layout_cn2 = column(select_date_key, cn2_file1_plot, cn2_file2_plot)#,# corr_plot)#, filtered_table_scatter_data, shift_vs_sampling_rate_plot)
cn2_tab = Panel(child=layout_cn2, title='Cn2 information')

# wind_layout
wind_src = make_wind_data()
wind_plot = make_wind_plot(wind_src)
wind_layout = column(select_date_key, wind_plot)
wind_tab = Panel(child=wind_layout, title='Wind information')

# CALLBACKS

# update scatter plots based on sampling rate
select_bin_width.on_change('value', update_scatter_data_based_on_bin)
select_n_samples.on_change('value', update_scatter_data_based_on_bin)

#update time series based on select_date_key
select_date_key.on_change('value', update_time_series_based_on_date_key)


#update corr plot based on datekey
select_date_key.on_change('value', update_corr_plot_based_on_datekey)
#update corr plot based on bin
select_bin_width.on_change('value', update_corr_plot_based_on_bin)

# update profiles
select_date_key.on_change('value', update_cn2_plot_file2)
select_date_key.on_change('value', update_cn2_plot_file1)

#update wind
select_date_key.on_change('value', update_wind_plot_based_on_date_key)

#hist
#histogram.on_change('', update_indices_based_on_selection_histogram)


# update the options for the date_keys in select_date_key
for_scatter_selection.data_source.selected.on_change('indices', update_select_date_key_based_on_selection)
# FINAL LAYOUT
layout = column(Tabs(tabs=[scatter_tab, time_series_tab, cn2_tab, wind_tab]))









curdoc().add_root(layout)
curdoc().title = "NOWCAST DASHBOARD"




