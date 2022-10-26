import os
import sys
import h5py
import numpy as np
import itertools
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import shutil
import plotly.graph_objects as go
import detd_parser as DETD

#from wrapper import Objects as OBJ
#from wrapper import GetSetFunctions as IO

from pyprodrisk import ProdriskSession


class Area(object):

    def __init__(self, name):
        self.name = name
        self.max_vol_txy = pd.Series(dtype=np.float64)
        self.min_vol_txy = pd.Series(dtype=np.float64)
        self.qmax_txy = pd.Series(dtype=np.float64)
        self.qmin_txy = pd.Series(dtype=np.float64)

# The functions below are used to load input from a v10 EOPS folder into a ProdriskSession
# The following input is read:
# Module data from model.h5
# Price forecast from PRISREKKE.PRI
# Price periods from PRISAVSNITT.DATA
# Inflow scenarios from ScenarioData.h5
# Settings from PRODRISK.CPAR (note: settings might be re-set by python code).
# Time dependent module restrictions from DYNMODELL.SIMT:
### max/min for reservoir, discharge and bypass, reference volume and energy equivalents.
# Water values from VVERD_000.EOPS


def set_topology_modnrs(areas):
    for name, area in areas.items():
        for module in area.modules:
            module.topology_modnr = []
            for i in range(3):
                if module.topology[i] == 0:
                    module.topology_modnr.append(0)
                else:
                    module.topology_modnr.append(area.modules[module.topology[i]-1].mod_nr)

    return


def add_default_inflow_series(areas, series_name_to_index):
    for name, area in areas.items():
        for mod in area.modules:
            if mod.inflow_series not in series_name_to_index.keys():
                series_name_to_index[mod.inflow_series] = 1
                print(f"Missing inflow series {mod.inflow_series} for module {mod.name}. Connects the modules reg inflow to first available inflow series instead.")

            if mod.unreg_inflow_series not in series_name_to_index.keys():
                series_name_to_index[mod.unreg_inflow_series] = 1
                print(f"Missing unreg inflow series {mod.unreg_inflow_series} for module {mod.name}. Connects the modules unreg inflow to first available inflow series instead.")

    return

def add_detd_modules(prodrisk, data_dir, area_names, series_name_to_index):

    areas = DETD.parse_area_modules(data_dir, area_names)

    set_topology_modnrs(areas)

    counter = 1
    for name, area in areas.items():
        # --- add a module to the session ---
        for module in area.modules:
            mod_name = module.name.strip("'").strip()

            mod = prodrisk.model.module.add_object(mod_name)
            mod.name.set(mod_name)
            mod.plantName.set(f"{module.name} prod")
            mod.number.set(module.mod_nr)
            mod.ownerShare.set(1.0)
            mod.regulationType.set(1)

            # ID number should match reg- and unreg series set in add_inflow_series()
            if module.inflow_series in series_name_to_index.keys():
                mod.connectedSeriesId.set(series_name_to_index[module.inflow_series])
            else:
                print(f"The inflow series {module.inflow_series} for module {module.name} is missing on the "
                      f"historical.h5 file. Will use the first series added to the API session for this module's "
                      f"regulated inflow(that is a random one).")
                mod.connectedSeriesId.set(1)

            if module.unreg_inflow_series in series_name_to_index.keys():
                mod.connected_unreg_series_id.set(series_name_to_index[module.unreg_inflow_series])
            else:
                print(f"The inflow series {module.inflow_series} for module {module.name} is missing on the "
                      f"historical.h5 file. Will use the first series added to the API session for this module's "
                      f"unregulated inflow(that is a random one).")
                mod.connected_unreg_series_id.set(1)

            # Used together with the attribute histAverageInflow on the connected series to scale inflow to each module.
            mod.meanRegInflow.set(module.mean_reg_inflow)
            mod.meanUnregInflow.set(module.mean_unreg_inflow)

            mod.rsvMax.set(module.max_vol)

            if module.min_vol_txy is not None:
                mod.minVol.set(set_up_scenarios_from_yearly(prodrisk, module.min_vol_txy))
            if module.max_vol_txy is not None:
                mod.maxVol.set(set_up_scenarios_from_yearly(prodrisk, module.max_vol_txy))
            if module.qmin_txy is not None:
                mod.minDischarge.set(set_up_scenarios_from_yearly(prodrisk, module.qmin_txy))
            if module.qmax_txy is not None:
                mod.maxDischarge.set(set_up_scenarios_from_yearly(prodrisk, module.qmax_txy))
            if module.qfomin_txy is not None:
                mod.minBypass.set(set_up_scenarios_from_yearly(prodrisk, module.qfomin_txy))


            mod.PQcurve.set(pd.Series(name=0.0, index=[0, module.qmax * module.enekv], data=[0, module.qmax]))
            mod.energyEquivalentConst.set(module.global_enekv)
            mod.maxDischargeConst.set(module.qmax)
            mod.maxProd.set(module.qmax * module.enekv)
            mod.maxBypassConst.set(module.qfomax)
            mod.topology.set(module.topology_modnr)

            mod.startVol.set(0.6*mod.rsvMax.get())

            counter = counter + 1

    hist_avg_infl = {}
    for series_name in series_name_to_index.keys():
        hist_avg_infl[series_name] = 0.0

    for mod in prodrisk.model.module:
        for series_name in hist_avg_infl.keys():
            if mod.connectedSeriesId.get() == series_name_to_index[series_name]:
                hist_avg_infl[series_name] += mod.meanRegInflow.get()
            if mod.connected_unreg_series_id.get() == series_name_to_index[series_name]:
                hist_avg_infl[series_name] += mod.meanUnregInflow.get()

    #for name, hist_infl in hist_avg_infl.items():
    #    if hist_infl > 0:
    #        prodrisk.model.inflowSeries[name].histAverageInflow.set(hist_infl)

    return True


def build_prodrisk_model(data_dir, area_names, n_weeks=156, start_time="2030-01-07", deterministic_scen=-1):

    model_name = "SYSTEM"
    # INITIALIZE PRODRISK API #
    prodrisk = ProdriskSession(license_path='/prodrisk/lib', solver_path='/prodrisk/lib', silent=False, log_file='')
    prodrisk.set_optimization_period(pd.Timestamp(start_time), n_weeks=n_weeks)

    prodrisk.n_scenarios = 50
    prodrisk.n_price_levels = 1

    # BUILD MODEL#
    set_price_periods(prodrisk, res="weekly")

    series_name_to_index = get_scenarios_from_historical_h5(prodrisk, data_dir)
    add_detd_modules(prodrisk, data_dir, area_names, series_name_to_index)

    add_area_object(prodrisk, data_dir, deterministic_scen=deterministic_scen)


    return prodrisk


def set_price_periods(prodrisk, res="weekly"):
    # The time resolution is specified by the price period time series.
    # Currently only values for the first week (168 hours) are used.
    # For each week, the average of the hourly price values are used as the price for each period.

    if res == "weekly":
        prodrisk.price_periods = pd.Series(
            index=[prodrisk.start_time],
            data=np.arange(1, 2)
        )
    elif res == "3H":
        prodrisk.price_periods = pd.Series(
            index=[prodrisk.start_time + pd.Timedelta(hours=3 * i) for i in range(56)],
            data=np.arange(1, 57)
        )
    elif res == "6H":
        prodrisk.price_periods = pd.Series(
            index=[prodrisk.start_time + pd.Timedelta(hours=6 * i) for i in range(28)],
            data=np.arange(1, 29)
        )



def set_start_vols(prodrisk, LTM_input_folder, model_name):

    start_vols = get_start_volumes(LTM_input_folder, model_name)
    for mod, vol in start_vols.items():
        prodrisk.model.module[mod].startVol.set(vol)

    return True


def add_modules(prodrisk, data_dir, model_name):
    model_file = h5py.File(data_dir + "model.h5", 'r')

    system_path = 'hydro_data/' + model_name.upper()

    PQcurves = {}
    energy_equivalents = {}
    pq = model_file.get(system_path + '/PQ_curve')
    for key in pq.keys():
        if "curve" in key:
            Q = np.array(pq[key]["M3s"])
            P = np.array(pq[key]["MW"])
            PQcurves[int(pq[key]["id"][0])] = [P, Q]
            # energy_equivalents[int(pq[key]["id"][0])] = float(pq[key]["local_conv_"][0])

    res_curves = {}
    res = model_file.get(system_path + '/res_curves')

    for key in res.keys():
        try:
            int(key)
        except ValueError:
            continue
        if ("Kote" in res[key]):
            res_curves[int(key)] = [np.array(res[key]["Vol"]), np.array(res[key]["Kote"])]

    module_data = model_file.get(system_path + '/Module_data')
    for i in range(module_data.size):

        # From Bernt: Our H5-files should be encoded with iso-8859-1
        mod = prodrisk.model.module.add_object(module_data[i]['res_name'].decode("iso-8859-1"))

        mod.name.set(module_data[i]['res_name'].decode("iso-8859-1"))
        mod.plantName.set(module_data[i]['plant_name'].decode("iso-8859-1"))
        mod.number.set(module_data[i]['res_id'])
        mod.ownerShare.set(module_data[i]['owner_share'])
        mod.regulationType.set(module_data[i]['reg_res'])

        mod.rsvMax.set(module_data[i]['max_res'])
        mod.connectedSeriesId.set(prodrisk.model.inflowSeries[module_data[i]['r_infl_name'].decode("iso-8859-1")].seriesId.get())
        mod.meanRegInflow.set(module_data[i]['r_infl_rvol'])
        mod.meanUnregInflow.set(module_data[i]['u_infl_rvol'])
        mod.nominalHead.set(module_data[i]['nom_elevation'])
        mod.submersion.set(module_data[i]["Undervannstand"])
        if(module_data[i]['res_id'] in res_curves.keys()):
            mod.volHeadCurve.set(pd.Series(name=0.0, index=res_curves[module_data[i]['res_id']][0], data=res_curves[module_data[i]['res_id']][1]))

        mod.reservoirMaxRestrictionType.set(module_data[i]['res_up_lim_type'])
        mod.reservoirMinRestrictionType.set(module_data[i]['res_low_lim_type'])
        mod.regulationDegree.set(module_data[i]["reg_level"])

        # Set refVol to improve convergence of first main iteration.
        # Default value for this attribute is 0% filling for all weeks, which gives a large gap between the converging F- and K-cost.

        if(module_data[i]['res_id'] in PQcurves.keys()):
            mod.PQcurve.set(pd.Series(name=module_data[i]['nom_elevation'], index=PQcurves[module_data[i]['res_id']][0], data=PQcurves[module_data[i]['res_id']][1]))

        mod.energyEquivalentConst.set(module_data[i]['conv_factor'])
        mod.maxDischargeConst.set(module_data[i]['max_flow'])
        mod.maxProd.set(module_data[i]['prod_cap'])
        mod.maxBypassConst.set(module_data[i]['max_bypass'])
        mod.topology.set([module_data[i]['flow_to'], module_data[i]['bypass_to'], module_data[i]['spill_to']])

    # Get module time series for modules from dynmodell.SIMT
    #getdynmodellSeries(data_dir, modules) TODO!!
    #getStraffdotCPAR() TODO

    return True


def add_inflow_series(prodrisk, data_dir, model_name, area_names, read_csv=False):
    
    # Inflow series for regulated inflow
    counter = 1
    for area in area_names:
        for inflow_type in ["R", "U"]:
            inflow_serie = prodrisk.model.inflowSeries.add_object(f"{area}-{inflow_type}")
            inflow_serie.seriesId.set(counter)
            # Used together with module attributes meanRegInflow and meanUnregInflow in
            # scaling of series to modules if one series is used to specify inflow profile for several modules:
            inflow_serie.histAverageInflow.set(1.0)

            counter = counter + 1

    for serie_name in prodrisk.model.inflowSeries.get_object_names():
        if read_csv:
            inflow_scenarios_52 = read_inflow_series_from_csv(prodrisk, data_dir, serie_name)
        else:
            inflow_scenarios_52 = get_inflow_scenarios(prodrisk, data_dir, serie_name, model_name, n_weeks=52)
        inflow_scenarios = set_up_scenarios_from_yearly(prodrisk, inflow_scenarios_52)
        prodrisk.model.inflowSeries[serie_name].inflowScenarios.set(inflow_scenarios)

    return True


def set_up_scenarios_from_yearly(prodrisk, series):
    indices = pd.DatetimeIndex([])
    next_indices = series.index

    n_years = int(prodrisk.n_weeks/52)

    series_size = series.values.shape
    steps_in_year = series_size[0]

    if len(series_size) > 1:
        n_scen = series_size[1]
    else:
        n_scen = 1

    if n_scen > 1:
        data = np.zeros((steps_in_year*n_years, n_scen))
    else:
        data = np.zeros(steps_in_year * n_years)

    for year in range(n_years):
        indices = indices.append(next_indices)
        next_indices = next_indices + pd.offsets.DateOffset(days=364)

        if n_scen > 1:
            data[year*steps_in_year:(year+1)*steps_in_year, 0:n_scen-year] = series.values[0:steps_in_year, year:n_scen]

            for y in range(year):
                data[year * steps_in_year:(year + 1) * steps_in_year, n_scen - year + y] = series.values[
                                                                                           0:steps_in_year, y]
        else:
            data[year*steps_in_year:(year+1)*steps_in_year] = series.values[0:steps_in_year]

    if n_scen > 1:
        scenarios = pd.DataFrame(data=data, index=indices)
    else:
        scenarios = pd.Series(data=data, index=indices)

    return scenarios


def get_inflow_scenarios(prodrisk, data_dir, serie_name, model_name, n_weeks=52):
    scenario_data_file = h5py.File(data_dir + "ScenarioData.h5", 'r')
    data = {}

    counter = 0
    for key in scenario_data_file.get(model_name.upper() + '/' + serie_name).keys():
        h5scenario = scenario_data_file.get(model_name.upper() + '/' + serie_name)[key]
        daily_index = [prodrisk.start_time + pd.Timedelta(days=i) for i in range(n_weeks * 7)]

        data[f"scen{counter}"] = np.array(h5scenario)[0:n_weeks*7]
        counter += 1
        if counter == prodrisk.n_scenarios:
            break

    return pd.DataFrame(index=daily_index, data=data)


def add_area_object(prodrisk, LTM_input_folder, deterministic_scen=-1):
    area = prodrisk.model.area.add_object("my_area")

    price_52w = get_yearly_price_ts(prodrisk, LTM_input_folder, deterministic_scen=deterministic_scen)
    price = set_up_scenarios_from_yearly(prodrisk, price_52w)

    # fig = px.line(price, labels={
    #     "index": "Date",
    #     "value": "Price [EUR/MWh]",
    #     "variable": "Scenario"
    # })
    # fig.show()

    area.price.set(price)

    #water_values = get_water_values(LTM_input_folder, 156)
    water_values = [pd.Series(name=ref, index=[np.real(100 - n * 2) for n in range(51)], data=[price.mean().mean() for n in range(51)]) for ref in range(prodrisk.n_price_levels.get())]
    area.waterValue.set(water_values)

    return True


def get_yearly_price_ts(prodrisk, LTM_input_folder, deterministic_scen=-1):
    price_df = get_price_scenarios(prodrisk.start_time, LTM_input_folder, "prisrekke.PRI", n_weeks=52, n_scen=prodrisk.n_scenarios, deterministic_scen=deterministic_scen)

    return price_df


def get_start_volumes(data_dir, model_name):
    fileNameStartVolumes = data_dir + model_name + ".SMAG"
    startVolumes = {}
    with open(fileNameStartVolumes, 'r') as volume_file:
        data = volume_file.readlines()
        dummy = data[0]
        for line in data[1:]:
            line = line.split(',')
            modName = line[0].replace("'", "").replace(" ", '').strip()
            startVolumes[modName] = float(line[1])
    return startVolumes


def get_sequential_price_periods(data_dir):
    detsimres = data_dir + '/detsimres.h5'
    f1 = h5py.File(detsimres, 'r+')  # open the file
    n_hours_per_seq_period = f1['result_description/PRISAVSNITT.DATA/NTIMEN_U']

    seq_to_akk = f1['result_description/PRISAVSNITT.DATA/KryssTilLAkkAvsn']

    start_hours = [sum(n_hours_per_seq_period[:i]) for i in range(len(n_hours_per_seq_period))]

    return dict(zip(start_hours, seq_to_akk))


def get_price_scenarios(start_time, data_dir, priceFileName, n_weeks=-1, n_scen=-1, deterministic_scen=-1):

    price_periods = get_sequential_price_periods(data_dir)
    start_hours = price_periods.keys()
    seq_periods = price_periods.values()

    fileNamePrisrekke = data_dir + '/' + priceFileName
    price_scenarios = []
    with open(fileNamePrisrekke, 'r') as prisrekke_file:
        allData = prisrekke_file.readlines()
        separator = allData[0][1]
        if n_weeks == -1:
            n_weeks = int(allData[4].split(separator)[0])
        n_price_periods = int(allData[6].split(separator)[0])
        prisData = allData[8:]
        priceData = [line.split(separator) for line in prisData]
        for i in range(0, len(priceData), n_price_periods):
            scenario = []
            listcompr = [priceData[i+j][2:] for j in range(n_price_periods)]
            merged = []

            for liste in zip(*listcompr):
                ny_liste = []
                for elem in liste:
                    try:
                        ny_liste.append(float(elem.replace(allData[0][2], ".")))
                    except ValueError:
                        continue
                merged.append(ny_liste)

            #merged = [[float(i.replace(allData[0][2], ".")) for i in liste] for liste in zip(*listcompr)]
            for week_no in range(n_weeks):
                week = merged[week_no]
                seq_week = [week[i-1] for i in seq_periods]
                scenario.extend(seq_week)
            price_scenarios.append(scenario[0:n_weeks*len(seq_periods)])

    if n_scen == -1:
        n_scen = len(price_scenarios)

    if deterministic_scen > 0:
        price_df = pd.DataFrame(
            #index=[start_time + pd.Timedelta(hours=3 * i) for i in range(len(price_scenarios[0]))],
            index=[start_time + pd.Timedelta(hours=168*w + h) for w in range(n_weeks) for h in start_hours],
            data={
                f"scen{i}": price_scenarios[deterministic_scen] for i in range(n_scen)
            },
        )
    else:
        price_df = pd.DataFrame(
            # index=[start_time + pd.Timedelta(hours=3 * i) for i in range(len(price_scenarios[0]))],
            index=[start_time + pd.Timedelta(hours=168 * w + h) for w in range(n_weeks) for h in start_hours],
            data={
                f"scen{i}": price_scenarios[i % 30] for i in range(n_scen)
            },
        )

    return price_df

# Read settings from PRODRISK.CPAR
def add_settings(prodrisk, data_dir):
    fileNameSettings = data_dir + "PRODRISK.CPAR"
    try:
        with open(fileNameSettings, 'r') as settings_file:
            allData = settings_file.readlines()
            data = [line.split(',')[0].split() for line in allData]
            dataDict = {line[0]:line[1] for line in data}

            settingInfoDict = {
                "STAITER": ("max_iterations", int),
                "MINITER": ("min_iterations", int),
                "STAITER1": ("max_iterations_first_run", int),
                "MINITER1": ("min_iterations_first_run", int),
                "FKONV": ("convergence_criteria", float),
                "STOR": ("inf", float),
                "ALFASTOR": ("alfa_max", float),
                "CTANK": ("water_ration_cost", float),
                "CFORB_STYR": ("bypass_cost", float),
                "CFLOM_STYR": ("overflow_cost", float),
                "TOMMAX": ("max_relaxation_iterations", int),
                "HALDKUT": ("max_cuts_created", int),
                "STR_MAGBR": ("reservoir_soft_restriction_cost", float),
                "ANTBRU1": ("first_relaxation_parameter", int),
                "SLETTE_FREKV": ("second_relaxation_parameter", int),
                "SLETTE_TOL": ("relaxation_tolerance", float),
                "RESSTOY": ("residual_model", int),
                "JUKE_AGGR_PRAVSN": ("aggregated_price_period_start_week", int),
                "JSEKV_STARTUKE": ("sequential_price_period_start_week", int),
                "JSEKV_SLUTTUKE": ("sequential_price_period_end_week", int),
                "PQValg": ("use_input_pq_curve", int),
                "MagBal": ("reservoir_balance_option", int),
                "FramSomSluttsim": ("forward_model_option", int),
                "PrisScenStrategi": ("price_scenario_option", int)
            }

            for key, value in dataDict.items():
                try:
                    getattr(prodrisk, settingInfoDict[key][0]).set(settingInfoDict[key][1](value))
                except KeyError:
                    continue

    except FileNotFoundError:
        print("File 'PRODRISK.CPAR' not found")

    return


def set_price_periods_from_file(prodrisk, data_dir):
    price_periods = getPricePeriods(data_dir)
    prodrisk.price_periods.set(pd.Series(data=np.array(price_periods),
                                         index = [prodrisk.start_time + pd.Timedelta(hours=i) for i in range(168)]))

    return


def getPricePeriods(data_dir):
    fileNamePrisavsnitt = data_dir + "PRISAVSNITT.DATA"
    prisavsnitt = []
    with open(fileNamePrisavsnitt, 'r') as prisavsnitt_file:
        allData = prisavsnitt_file.readlines()
        nPriceLevels = int(allData[1].split(',')[0])
        pricePeriodsData = allData[nPriceLevels+2:]
        for line in pricePeriodsData:
            line = list(map(int, line.split(',')[:-2]))
            prisavsnitt.extend(line)
    return prisavsnitt

# Read module restrictions from the binary file DYNMODELL.SIMT. To understand this, read the documentation for this file (AN Filstruktur_V10).
def add_module_restrictions(prodrisk, data_dir):
    dynmodell = read_dynmodell(data_dir)

    n_mod = dynmodell['Blokk1']['first6Ints'][2]
    n_weeks = dynmodell['Blokk1']['first6Ints'][4]

    if n_weeks < prodrisk.n_weeks:
        print(f"WARNING: Restrictions from DYNMODELL.SIMT read for {n_weeks} weeks. Simulation period is set to {prodrisk.n_weeks} weeks. Restrictions for the last {prodrisk.n_weeks - n_weeks} weeks of the simulation period may be set incorrectly?")

    module_indices = dynmodell['Blokk'+str(n_weeks+2)][0:n_mod]

    MAMAX = {mod: np.zeros(n_weeks) for mod in module_indices}
    MAMIN = {mod: np.zeros(n_weeks) for mod in module_indices}
    MAGREF = {mod: np.zeros(n_weeks) for mod in module_indices}
    ENEKV = {mod: np.zeros(n_weeks) for mod in module_indices}
    QMAX = {mod: np.zeros(n_weeks) for mod in module_indices}
    QMIN = {mod: np.zeros(n_weeks) for mod in module_indices}
    QFOMAX = {mod: np.zeros(n_weeks) for mod in module_indices}
    QFOMIN = {mod: np.zeros(n_weeks) for mod in module_indices}


    # First read out to python dicts, to ensure correct sorting is maintained
    for w in range(n_weeks):
        for i in range(n_mod):
            week_block = dynmodell['Blokk'+str(w+2)]
            mod = module_indices[i]
            MAMAX[mod][w] = week_block[i*8]
            MAMIN[mod][w] = week_block[i * 8+1]
            MAGREF[mod][w] = week_block[i*8+2]
            ENEKV[mod][w] = week_block[i*8+3]
            QMAX[mod][w] = week_block[i*8+4]
            QMIN[mod][w] = week_block[i*8+5]
            QFOMAX[mod][w] = week_block[i*8+6]
            QFOMIN[mod][w] = week_block[i*8+7]


    weekly_indices = [prodrisk.start_time + pd.Timedelta(weeks=w) for w in range(n_weeks)]

    # Add information to module objects
    for module_name in prodrisk.model.module.get_object_names():
        mod = prodrisk.model.module[module_name]
        mod_number = mod.number.get()
        if(mod_number in module_indices):
            if np.min(MAMAX[mod_number]) < mod.rsvMax.get():
                mod.maxVol.set(pd.Series(data=MAMAX[mod_number], index=weekly_indices))
            if np.max(MAMIN[mod_number]) > 0.0:
                mod.minVol.set(pd.Series(data=MAMIN[mod_number], index=weekly_indices))

            mod.refVol.set(pd.Series(data=MAGREF[mod_number], index=weekly_indices))

            if np.any(ENEKV[mod_number] != mod.energyEquivalentConst.get()):
                mod.energyEquivalent.set(pd.Series(data=ENEKV[mod_number], index=weekly_indices))

            if np.min(QMAX[mod_number]) < mod.maxDischargeConst.get():
                mod.maxDischarge.set(pd.Series(data=QMAX[mod_number], index=weekly_indices))
            if np.max(QMIN[mod_number]) > 0.0:
                mod.minDischarge.set(pd.Series(data=QMIN[mod_number], index=weekly_indices))

            if np.min(QFOMAX[mod_number]) < mod.maxBypassConst.get():
                mod.maxBypass.set(pd.Series(data=QFOMAX[mod_number], index=weekly_indices))
            if np.max(QFOMIN[mod_number]) > 0.0:
                mod.minBypass.set(pd.Series(data=QFOMIN[mod_number], index=weekly_indices))


    return

# Currently not in use...
def get_hydro_parameters(data_dir):
    model_file = h5py.File(data_dir + "model.h5", 'r')
    hydroparametersdata = model_file.get('hydro_data/hydro_parameters')
    hydro_parameters = {}
    for i in range(hydroparametersdata.size):
        hydro_parameters[hydroparametersdata[i]['parameter'].decode("utf-8")] = hydroparametersdata[i]['value']
    return hydro_parameters

# Currently not in use...
def getNumModules(data_dir, model_name):
    fileNameModel = data_dir + "model.h5"
    model_file = h5py.File(fileNameModel, 'r')
    numModules = model_file.get('hydro_data/' + model_name + '/numb_of_modules')[0]
    return numModules

# Currently not in use (no pumps in basic python example)...
def get_pumps(data_dir, model_name):
    model_file = h5py.File(data_dir + "model.h5", 'r')

    pumps = []
    pump_data = model_file.get('hydro_data/' + model_name + '/Pumpe_data')
    if pump_data is not None:
        for i in range(pump_data.size):
            pump = {
                "name": pump_data[i]['name'].decode("utf-8"),
                "ownerShare": pump_data[i]['owner_share'],
                "maxPumpHeight": 1.0,
                "minPumpHeight": 0.0,
                "maxHeightUpflow": pump_data[i]['Pumpekap_1']+pump_data[i]['Pumpekap_2'],
                "minHeightUpflow": pump_data[i]['Pumpekap_2'],
                "averagePower": pump_data[i]['Pumpekap_3'],
                "topology": [pump_data[i]['con_to_pla'], pump_data[i]['to_res'], pump_data[i]['from_res']]
            }
            pumps.append(pump)

    return pumps


def duration_curve(time_series, plot_title="Duration curve", y_axis='', plot_path='', y_axis_range=None, x_range=None):

    fig = go.Figure()

    line_styles = ["solid", "dash"]
    i = 0
    for name, time_serie in time_series.items():
        y = np.sort(time_serie.values.flatten("F"))[::-1]
        x = np.arange(0, 100, 100 / y.size)

        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=name, line={'dash': line_styles[i]}))
        ++i

    fig.update_layout(
        #title=plot_title,
        xaxis_title="%",
        yaxis_title=y_axis,
        font=dict(
            size=12,
        )
    )

    if y_axis_range is not None:
        fig.update_layout(yaxis_range=[y_axis_range[0], y_axis_range[1]])

    if x_range is not None:
        fig.update_layout(xaxis_range=[x_range[0], x_range[1]])
        plot_title = f'{plot_title}_{x_range[0]}_{x_range[1]}'
    if plot_path == '':
        fig.show()
    else:
        fig.write_image(f'{plot_path}/{plot_title}.svg')
    return


def get_scenarios_from_historical_h5(prodrisk, LTM_input_folder):
    n_days = 29220
    historical = h5py.File(os.path.join(LTM_input_folder+'historical.h5'), 'r')

    series_name_to_index = {}

    counter = 0
    for series_name in historical['historical_series'].keys():
        counter = counter + 1

        daily_inflow = historical[f'historical_series/{series_name}'][0:n_days]

        inflow_serie = prodrisk.model.inflowSeries.add_object(series_name)
        inflow_serie.seriesId.set(counter)
        series_name_to_index[series_name] = counter

        inflow_serie.inflowScenarios.set(setup_inflow_scenarios(prodrisk, daily_inflow))

    return series_name_to_index


def remove_unused_series_from_historical_h5(LTM_input_folder, area_names):

    historical = h5py.File(os.path.join(LTM_input_folder+'historical.h5'), 'a')
    unused_series = list(historical['historical_series'].keys())

    areas = DETD.parse_area_modules(LTM_input_folder, area_names)

    for name, area in areas.items():
        for mod in area.modules:
            if mod.inflow_series not in unused_series:
                print(f"Inflow series for module {mod.name} missing on historical file!")
            else:
                unused_series.remove(mod.inflow_series)

            if mod.unreg_inflow_series not in unused_series:
                print(f"Unreg inflow series for module {mod.name} missing on historical file!")
            else:
                unused_series.remove(mod.unreg_inflow_series)

            print(f"Series {mod.inflow_series} and {mod.unreg_inflow_series} will be kept on the historical file.")

    for series_name in unused_series:
        print(f"Remove unused series {series_name}")
        historical['historical_series'][series_name][:] = 0.0*historical['historical_series'][series_name][:]
        del historical['historical_series'][series_name]

    historical.close()

    return


def setup_inflow_scenarios(prodrisk, daily_inflow):
    scens = np.zeros((prodrisk.n_scenarios, prodrisk.n_weeks*7))
    daily_inflow_extended = np.zeros(prodrisk.n_scenarios * prodrisk.n_weeks*7)

    for d in range(prodrisk.n_scenarios * prodrisk.n_weeks*7):
        daily_inflow_extended[d] = daily_inflow[d % len(daily_inflow)]

    for scen in range(prodrisk.n_scenarios):
        scens[scen] = (1.0e6 / (3600*24))*daily_inflow_extended[scen*52*7:scen*52*7+prodrisk.n_weeks*7]

    return pd.DataFrame(index=[prodrisk.start_time + pd.Timedelta(days=i) for i in range(prodrisk.n_weeks*7)], data=scens.transpose())

if __name__ == "__main__":
    # SET UP WORKING DIR #
    workin_dir = os.getcwd() + "/"

    # --- create a new session ---

    data_dir = workin_dir + "subareas/"
    NO1 = ["OSTLAND"]
    NO2 = ["HAUGESUND", "SORLAND", "TELEMARK", "SOROST"]
    NO3 = ["MOERE", "NORGEMIDT", "NORDVEST"]
    NO4 = ["HELGELAND", "SVARTISEN", "TROMS", "FINNMARK"]
    NO5 = ["VESTSYD", "HALLINGDAL", "VESTMIDT"]
    SE1 = ["SVER-SE1"]
    SE2 = ["SVER-SE2"]
    SE3 = ["SVER-SE3"]
    SE4 = ["SVER-SE4"]


    #read_area_txys_from_hdf5(data_dir, NO1+NO2+NO3+NO4+NO4+SE1+SE2+SE3+SE4)
    #copy_inflow_from_text_file_to_scenariodata_h5("C:/Users/Hansha/Documents/GitHub/prodrisk_examples/south_norway_one_reservoir_model/south_norway/")
    
    a = 5


