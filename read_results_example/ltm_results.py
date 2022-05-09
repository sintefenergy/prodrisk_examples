import pandas as pd
import numpy as np
import plotly.express as px
import h5py
import os
from bisect import bisect_right
import read_prodrisk_data_from_LTM_folder as ltm_input


#Used to convert volumes to head
class Interpolate:
    def __init__(self, x_list, y_list):
        if any(y - x <= 0 for x, y in zip(x_list, x_list[1:])):
            raise ValueError("x_list must be in strictly ascending order!")
        self.x_list = x_list
        self.y_list = y_list
        intervals = zip(x_list, x_list[1:], y_list, y_list[1:])
        self.slopes = [(y2 - y1) / (x2 - x1) for x1, x2, y1, y2 in intervals]

    def __call__(self, x):
        if not (self.x_list[0] <= x <= self.x_list[-1]):
            raise ValueError("x out of bounds!")
        if x == self.x_list[-1]:
            return self.y_list[-1]
        i = bisect_right(self.x_list, x) - 1
        return self.y_list[i] + self.slopes[i] * (x - self.x_list[i])


def get_head_from_vol(ltm_dir, mod_name, vol):
    vol_head = get_vol_head_curve(ltm_dir, mod_name)
    vol_head_interpolate = Interpolate(vol_head.x, vol_head.y)
    return vol_head_interpolate(vol)


def get_vol_head_curve(data_dir, mod_name):
    detsimres = data_dir + '/detsimres.h5'
    f1 = h5py.File(detsimres, 'r+')  # open the file

    area_name = f1['result_description/AreaData/']['OMRNAVN']
    vol_head_dict = ltm_input.get_vol_head_dict(data_dir, area_name)

    return vol_head_dict[mod_name]


def plot_percentiles(result_serie: pd.DataFrame, y_axis, plot_title, percentiles_limits=[0, 25, 50, 75, 100], plot_path=''):
    t_values = result_serie.index

    # Always include average?
    percentiles = np.zeros((t_values.size, len(percentiles_limits) + 1))

    i = 0
    columns = []
    for limit in percentiles_limits:
        percentiles[:, i] = np.percentile(result_serie.values, limit, axis=1)
        columns.append(str(limit) + "%")
        i += 1

    percentiles[:, i] = np.average(result_serie.values, axis=1)
    columns.append("Average")

    percentiles_ts = pd.DataFrame(data=percentiles, index=t_values, columns=columns)

    fig = px.line(percentiles_ts, labels={
        "index": "Date",
        "value": y_axis,
        "variable": "Legend"
    }, title=plot_title)


    if plot_path == '':
        fig.show()
    else:
        fig.write_image(f'{plot_path}/{plot_title}.png')

    return


def plot_xy(series, x_axis='x values', y_axis='y_values', plot_title='', plot_path=''):

    fig = px.scatter(series, labels={"index": x_axis, "value": y_axis}, title=plot_title)

    fig.update_traces(marker=dict(size=12,
                                  line=dict(width=2,
                                            color='DarkSlateGrey')),
                      selector=dict(mode='markers'),
                      showlegend=False)

    if plot_path == '':
        fig.show()
    else:
        fig.write_image(f'{plot_path}/{plot_title}.png')


def get_txy_series(data_dir, module_name, series_name, scenarios=None, start_time=None, hourly=True):
    # This routine reads detailed hydro results from the file detsimres.h5 in a EOPS/ProdRisk datasets.
    # The returned time series has the same format as a stochastic time series read from pyProdrisk (panda dataframe)

    only_pos = False
    only_neg = False
    vol_to_head = False

    if series_name == "production":
        only_pos = True
        series_name = "hydro_prod"
    elif series_name == "discharge":
        only_pos = True
        series_name = "draw_down"
    elif series_name == "consumption":
        only_neg = True
        series_name = "hydro_prod"
    elif series_name == "upflow":
        only_neg = True
        series_name = "draw_down"
    if series_name == "head":
        vol_to_head = True
        series_name = "reservoir"


    detsimres = data_dir + '/detsimres.h5'
    f1 = h5py.File(detsimres, 'r+')  # open the file
    values = f1['hydro_module_results/' + series_name + '/val']

    if scenarios == None:
        scenarios = range(values.shape[0])
    n_time_steps = values.shape[2]

    if start_time == None:
        start_time = pd.Timestamp('2022-01-03')

    start_time = start_time.to_datetime64()

    # Map placement on detsimres to module name (to input results from shop-objects in correct place)
    mapping_table = f1['result_description/mapping/hydro_module_results']
    mapping = dict(zip(mapping_table['hydro_module_id'], mapping_table['loc_hydro_module_id']))
    module_data = f1['result_description/ModulData']
    mod_names = [mod_name.decode('utf-8') for mod_name in module_data['Modulnavn']]
    mod_name_to_nr = dict(zip(mod_names, module_data['ModulNr']))

    internal_module_number = mapping[mod_name_to_nr[module_name]]-1

    y_values = np.zeros((len(scenarios), n_time_steps))

    for scen in scenarios:
        y_values[scen] = values[scen][internal_module_number][:] #dim0: scen, dim1: module, dim2: time step

    ntimen_u = f1['result_description/PRISAVSNITT.DATA/NTIMEN_U']
    series_info = f1['result_description/time_series_info/hydro_module_results']

    resolution = -1

    for i in range(series_info.size):
        if series_info[i]['name_of_result'].decode("iso-8859-1") == series_name:
            resolution = series_info[i]['time_series_type']
            break

    if resolution == -1:
        print(f"Not able to get time resolution for series {series_name} for module {module_name}. Aborts reading txy.")
        return False

    if resolution == 122:
        # 122. Price period resolution
        n_weeks = int(n_time_steps / ntimen_u.size)
        seq_price_period_start_hours = np.zeros(ntimen_u.size)
        seq_price_period_start_hours[0] = 0
        for period in range(ntimen_u.size - 1):
            seq_price_period_start_hours[period + 1] = seq_price_period_start_hours[period] + ntimen_u[period]

        if series_name == "reservoir":
            t_values = np.tile(seq_price_period_start_hours, n_weeks) + np.array([168*int(val / ntimen_u.size)
                                                                              for val in range(n_time_steps)])
            t_values = np.append(t_values, n_weeks * 168)
        else:
            t_values = np.tile(seq_price_period_start_hours, n_weeks) + np.array([168 * int(val / ntimen_u.size)
                                                                                  for val in range(n_time_steps)])

    elif resolution == 123:
        # 123: Weekly resolution
        n_weeks = n_time_steps
        if series_name == "reservoir":
            t_values = np.array([168 * i for i in range(n_time_steps + 1)])
        else:
            t_values = np.array([168 * i for i in range(n_time_steps)])
    else:
        print(f"Time series type {series_info['time_series_type']} not supported; "
              f"should not be too difficult to fix.")
        return

    if series_name == "reservoir":
        start_vols = get_start_vols(data_dir)
        start_vol_module = np.zeros(len(scenarios))
        for scen in scenarios:
            start_vol_module[scen] = start_vols[scen][module_name]
        y_values = np.insert(y_values, 0, start_vol_module, axis=1)
    elif series_name == "spillage" or series_name == "reservoir_inflow":
        t_values = np.array([168*i for i in range(n_time_steps)])

        # Spillage and inflows are scaled to unit m3/s instead of Mm3.
        y_values = y_values*1e6/(168*3600)

    else:
        # Discharge and bypass are scaled to unit m3/s instead of Mm3
        if series_name == "draw_down" or series_name == "bypass":
            y_values = (1e6/3600)*y_values/np.repeat(ntimen_u, n_weeks)

    if only_pos:
        y_values[y_values < 0] = 0.0
    elif only_neg:
        y_values[y_values > 0] = 0.0
        y_values = -1*y_values





    if hourly:
        if t_values[-1] != n_weeks * 168-1:
            t_values = np.append(t_values, n_weeks*168-1)
            a = y_values[:, -1]
            y_values = np.append(y_values, a.reshape(a.shape[0], 1), axis=1)

    delta = pd.Timedelta(hours=1)
    t_values = np.repeat(start_time, t_values.size) + t_values * delta

    if vol_to_head:
        vol_head = get_head_from_vol(data_dir, module_name)
        VH = Interpolate(vol_head.x, vol_head.y)
        new_y_values = 0.0*y_values
        size = y_values.shape
        for scen in range(size[0]):
            for time_step in range(size[1]):
                new_y_values[scen][time_step] = VH(y_values[scen][time_step])
        y_values = new_y_values



    time_series = pd.DataFrame(data=y_values.transpose(), index=t_values)

    if hourly:
        time_series = time_series.resample('H')

        if series_name == 'reservoir':
            time_series = time_series.interpolate()
        else:
            time_series = time_series.pad()[:]

    return time_series


def get_area_txy_series(data_dir, result_type, series_name, scenarios=None, start_time=None, hourly=True):

    if result_type == "market":
        result_type = "market_results"
    if result_type == "hydro":
        result_type = "hydro_results"

    enmres = data_dir + '/enmres.h5'
    f1 = h5py.File(enmres, 'r+')  # open the file
    values = f1[result_type + '/' + series_name + '/val']

    if scenarios == None:
        scenarios = range(values.shape[1])
    n_time_steps = values.shape[2]

    if start_time == None:
        start_time = pd.Timestamp('2021-01-04')

    start_time = start_time.to_datetime64()

    y_values = np.zeros((len(scenarios), n_time_steps))

    for scen in scenarios:
        y_values[scen] = values[0][scen][:] #dim0: scen, dim1: time step

    ntimen_u = f1['result_description/PRISAVSNITT.DATA/NTIMEN_U']
    seq_price_period_start_hours = np.zeros(ntimen_u.size)
    seq_price_period_start_hours[0] = 0
    for period in range(ntimen_u.size-1):
        seq_price_period_start_hours[period+1] = seq_price_period_start_hours[period]+ntimen_u[period]
    n_weeks = int(n_time_steps/ntimen_u.size)
    t_values = np.tile(seq_price_period_start_hours, n_weeks) + np.array([168*int(val / ntimen_u.size) for val in range(n_time_steps)])

    if hourly:
        if t_values[-1] != n_weeks * 168-1:
            t_values = np.append(t_values, n_weeks*168-1)
            a = y_values[:,-1]
            y_values = np.append(y_values, a.reshape(a.shape[0],1), axis=1)

    delta = pd.Timedelta(hours=1)
    t_values = np.repeat(start_time, t_values.size) + t_values * delta

    time_series = pd.DataFrame(data=y_values.transpose(), index=t_values)

    time_series = time_series.resample('H')
    time_series = time_series.pad()[:]

    if series_name != "reservoir":
        new_datetime_range = pd.date_range(start=time_series.index.min(), freq="H", periods=n_weeks * 168)
        time_series = time_series.reindex(new_datetime_range)

    return time_series


def get_start_vols(data_dir):

    # Check if results are from series- og parallell simulation.
    detsimres = data_dir + '/detsimres.h5'
    f1 = h5py.File(detsimres, 'r+')  # open the file
    sim_data = f1['result_description/SimData']
    is_serie_sim = sim_data["SERIE"]

    area_name = f1['result_description/AreaData/']['OMRNAVN'][0].decode("iso-8859-1")

    if is_serie_sim:
        start_vols = get_start_vols_from_stmag_serie_sddp(data_dir)
    else:
        start_vols = get_start_vols_from_smag(data_dir, area_name, sim_data["NSIM"][0])

    return start_vols


def get_start_vols_from_smag(data_dir, system_name, n_scen):
    smag_vols = ltm_input.get_start_volumes(data_dir, system_name)

    start_vols = []

    for scen in range(n_scen):
        start_vols.append(smag_vols)

    return start_vols


def get_start_vols_from_stmag_serie_sddp(data_dir):
    # For series simulations in ProdRisk, the start volumes of each scenario is only available on the binary file
    # StMagSerie.SDDP.
    # The ordering of these start volumes per module is not stored in any metadata on any of the result files
    # (potential future improvement...).
    # Reservoirs with cuts (>2 Mm3?) have the same ordering on StMagSerie.SDDP as on KUTT.SDDP, where the module
    # sorting metadata is stored in the file head.
    # For the remaining reservoirs, it is random if the correct start volume is selected or not from StMagSerie.SDDP.
    # This script writes out the selected module sorting to a file 'ordered_module_names_stmagserie_sddp.txt'.
    # This file may then be used to manually reorder the small reservoirs (end of the list) if necessary.

    start_vols = []

    with open(data_dir + '\StMagSerie.SDDP', "rb") as f:
        # First 4 ints are dimensioning quantities.
        first_4_ints = np.fromfile(f, dtype=np.int32, count=4)
        n_scen = first_4_ints[2]
        n_mod = first_4_ints[1]

        mod_names = get_stmagserie_mod_name_order(data_dir)
        for scen in range(n_scen):
            scen_start_vol = {}
            for mod in range(n_mod):
                scen_start_vol[mod_names[mod]] = np.sum(np.fromfile(f, dtype=np.double, count=1))
            start_vols.append(scen_start_vol)

        f.close()

    return start_vols


def get_stmagserie_mod_name_order(data_dir):
    mod_numbers = []

    file_name = "ordered_module_names_stmagserie_sddp.txt"

    if os.path.exists(data_dir + '/KUTT.SDDP'):

        with open(data_dir + '/KUTT.SDDP', "rb") as f:
            # Blokk 1
            first15Ints = np.fromfile(f, dtype=np.int32, count=15)
            for i in range(first15Ints[3]):
                mod_numbers.append(np.fromfile(f, dtype=np.int32, count=1)[0])

        f1 = h5py.File(data_dir + '\\detsimres.h5', 'r+')  # open the file

        # Map placement on detsimres to module name (to input results from shop-objects in correct place)
        mapping_table = f1['result_description/mapping/hydro_module_results']
        mapping = dict(zip(mapping_table['loc_hydro_module_id'], mapping_table['hydro_module_id']))
        module_data = f1['result_description/ModulData']
        mod_nr_to_module_names = dict(zip(module_data['ModulNr'], module_data['Modulnavn']))

        mod_nr_h5 = module_data['ModulNr']

        buffer_modules = list(set(mod_nr_h5) - set(mod_numbers))

        for mod_nr in buffer_modules:
            mod_numbers.append(mod_nr)

        ordered_module_names = [mod_nr_to_module_names[mod_nr].decode("iso-8859-1") for mod_nr in mod_numbers]

        write_string_list_to_file(data_dir, file_name, ordered_module_names)

    else:
        ordered_module_names = read_string_list_from_file(data_dir, file_name)

    return ordered_module_names


def write_string_list_to_file(data_dir, file_name, ordered_module_names):
    output_file = open(data_dir + '/' + file_name, "w")

    for mod_name in ordered_module_names:
        output_file.write(mod_name + "\n")

    output_file.close()

    return


def read_string_list_from_file(data_dir, file_name):
    ordered_module_names = []

    with open(data_dir + '/' + file_name, "r") as file:
        ordered_module_names = [line.rstrip() for line in file]

    return ordered_module_names