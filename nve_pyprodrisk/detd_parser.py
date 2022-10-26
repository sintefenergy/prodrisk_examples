import os
import numpy as np
import pandas as pd
import h5py

# giving directory name
dirname = '.'

# giving file extension
ext = ('.DETD')

start_time = pd.Timestamp("2030-01-07")

class Module(object):

    def __init__(self, mod_nr, name):
        self.mod_nr = mod_nr
        self.name = name
        self.enekv = 0.0
        self.global_enekv = 0.0
        self.qmax = 0.0
        self.qfomax = 0.0
        self.max_vol = 0.0
        self.max_vol_txy = None
        self.min_vol_txy = None
        self.qmax_txy = None
        self.qmin_txy = None
        self.qfomin_txy = None

        self.reg_degree = 0.0
        self.mean_reg_inflow = 0.0
        self.mean_unreg_inflow = 0.0
        self.inflow_series = ""
        self.unreg_inflow_series = ""

        self.qmin_inflow_series = None

        self.hyd_kobl = None
        self.upstream_hyd_kobl_modules = []
        self.coupled_module = False
        self.topology = [0, 0, 0]
        self.topology_modnr = [0, 0, 0]

class Area(object):

    def __init__(self, name):
        self.name = name
        self.modules = []
        self.qmax = 0.0
        self.max_vol = 0.0
        self.max_vol_txy = pd.Series(index=[start_time], data=[0])
        self.min_vol_txy = pd.Series(index=[start_time], data=[0])
        self.qmax_txy = pd.Series(index=[start_time], data=[0])
        self.qmin_txy = pd.Series(index=[start_time], data=[0])



def parse_module(modules, Lines, next_line):
    end_string = '10,   0,   0,'
    hyd_kobl_string = 'Kode hydraulisk kobling'
    utlop_string = 'Utlopekote stasjon, Nominell brutto fallhoyde'
    qmin_from_series_string = 'Navn vannmerke, Midlere aarlig vassforing'
    topology_string = 'Lopenr. magasin for; Stasjonstapping, Forbitapping, Flom'
    unreg_infl_string = 'Midlere uregulerbart aarstilsig, Navn vannmerke'


    id_line = Lines[next_line].split(',')
    mod_nr = int(id_line[2])
    mod = Module(mod_nr, id_line[1])

    next_line = next_line + 1
    line = Lines[next_line].split(',')
    mod.max_vol = float(line[0])
    mod.reg_degree = float(line[1])
    mod.mean_reg_inflow = float(line[2])
    mod.inflow_series = line[3].strip("'").strip()

    next_line = next_line + 1

    enekv_line = Lines[next_line].split(',')
    mod.qmax = float(enekv_line[0])
    mod.qfomax = float(enekv_line[1])
    mod.enekv = float(enekv_line[2])

    while unreg_infl_string not in Lines[next_line]:
        next_line += 1

    unreg_infl_line = Lines[next_line].split(',')

    mod.mean_unreg_inflow = float(unreg_infl_line[0])
    mod.unreg_inflow_series = unreg_infl_line[1].strip("'").strip()

    while topology_string not in Lines[next_line]:
        next_line = next_line + 1

    topology_line = Lines[next_line].split(',')
    mod.topology = [int(topology_line[0]), int(topology_line[1]), int(topology_line[2])]
    next_line = next_line + 1

    hyd_kobl_line = Lines[next_line].split(',')
    if int(hyd_kobl_line[0]) != 0:
        mod.hyd_kobl = [int(hyd_kobl_line[0]), int(hyd_kobl_line[1]), int(hyd_kobl_line[2])]

    next_line = next_line + 1

    while True:
        if utlop_string in Lines[next_line]:
            next_line = next_line + 1

        if end_string in Lines[next_line]:
            break

        potential_restr = Lines[next_line].split(',')
        restr_type = int(potential_restr[0])
        if restr_type == 1:
            # print('Restriksjonstype max_vol')
            n_points = int(potential_restr[1])
            next_line = next_line + 1
            restr_line = Lines[next_line].split(',')
            weeks = []
            vals = []
            for p in range(n_points):
                weeks.append(int(restr_line[2*p]))
                vals.append(float(restr_line[2*p + 1]))
            mod.max_vol_txy = pd.Series(index=[start_time + pd.Timedelta(weeks=i) for i in weeks], data=vals)

        elif restr_type == 2:
            # print('Restriksjonstype min_vol')
            n_points = int(potential_restr[1])
            next_line = next_line + 1
            restr_line = Lines[next_line].split(',')
            weeks = []
            vals = []
            for p in range(n_points):
                weeks.append(int(restr_line[2*p]))
                vals.append(float(restr_line[2*p + 1]))
            mod.min_vol_txy = pd.Series(index=[start_time + pd.Timedelta(weeks=i) for i in weeks], data=vals)

        elif restr_type == 3:
            # print('Restriksjonstype Qmax')
            n_points = int(potential_restr[1])
            next_line = next_line + 1
            restr_line = Lines[next_line].split(',')
            weeks = []
            vals = []
            for p in range(n_points):
                weeks.append(int(restr_line[2*p]))
                vals.append(float(restr_line[2*p + 1]))
            mod.qmax_txy = pd.Series(index=[start_time + pd.Timedelta(weeks=i) for i in weeks], data=vals)
        elif restr_type == 4:
            # print('Restriksjonstype Qmin')
            n_points = int(potential_restr[1])
            next_line = next_line + 1

            if qmin_from_series_string in Lines[next_line]:
                #print(f"Qmin from series for mod {mod_nr}. {Lines[next_line]}")
                qmin_from_series_line = Lines[next_line].split(',')
                mod.qmin_inflow_series = qmin_from_series_line[0].strip()
                mod.qmin_inflow_series_yearly_mean = float(qmin_from_series_line[1])
            else:
                restr_line = Lines[next_line].split(',')
                weeks = []
                vals = []
                for p in range(n_points):
                    weeks.append(int(restr_line[2*p]))
                    vals.append(float(restr_line[2*p + 1]))
                mod.qmin_txy = pd.Series(index=[start_time + pd.Timedelta(weeks=i) for i in weeks], data=vals)
        elif restr_type == 5:
            # print('Restriksjonstype MIN FORBITAPPING')
            n_points = int(potential_restr[1])
            next_line = next_line + 1

            if qmin_from_series_string in Lines[next_line]:
                # print(f"Qfomin from series for mod {mod_nr}. {Lines[next_line]}")
                qmin_from_series_line = Lines[next_line].split(',')
                mod.qmin_inflow_series = qmin_from_series_line[0].strip()
                mod.qmin_inflow_series_yearly_mean = float(qmin_from_series_line[1])
            else:
                restr_line = Lines[next_line].split(',')
                weeks = []
                vals = []
                for p in range(n_points):
                    weeks.append(int(restr_line[2 * p]))
                    vals.append(float(restr_line[2 * p + 1]))
                mod.qfomin_txy = pd.Series(index=[start_time + pd.Timedelta(weeks=i) for i in weeks], data=vals)
        elif restr_type in range(6, 10):
            next_line = next_line + 1

        next_line = next_line + 1

    next_line = next_line + 1

    modules.append(mod)
    return next_line

def get_area_names(dir_name):
    area_names = []

    # iterating over all files
    for files in os.listdir(dir_name):
        if files.endswith(ext):
            area_names.append(os.path.splitext(files)[0])  # printing file name of desired extension
        else:
            continue
    return area_names

def parse_area_modules(dir_name, area_names_in_run):

    areas = {}

    for area_name in area_names_in_run:
        file1 = open(f'{dir_name}/{area_name}.DETD', 'r', errors='ignore')
        Lines = file1.readlines()

        second_line = Lines[1].split(',')
        n_mod = int(second_line[0])
        #print(f'{area_name} has {n_mod} modules.\n')

        areas[area_name] = Area(area_name)

        next_line = 2
        for mod in range(n_mod):
            next_line = parse_module(areas[area_name].modules, Lines, next_line)

    set_default_module_txys(areas)
    sum_area_txys(areas)

    return areas


def set_default_module_txys(areas):

    for name, area in areas.items():
        #print(f'{area.name} has {len(area.modules)} modules.\n')

        int_nr = 0
        for mod in area.modules:
            if mod.hyd_kobl is not None:
                int_nr_2 = 0
                for mod2 in area.modules:
                    if mod2.topology[0] - 1 == int_nr:
                        mod.upstream_hyd_kobl_modules.append(int_nr_2)
                        mod2.coupled_module = True

                    int_nr_2 = int_nr_2 + 1

            int_nr = int_nr + 1

            if mod.max_vol_txy is None:
                mod.max_vol_txy = pd.Series(index=[start_time], data=[mod.max_vol])
            if mod.min_vol_txy is None:
                mod.min_vol_txy = pd.Series(index=[start_time], data=[0])
            if mod.qmax_txy is None and mod.hyd_kobl is None:
                mod.qmax_txy = pd.Series(index=[start_time], data=[mod.qmax])
            if mod.qmin_txy is None and mod.hyd_kobl is None:
                mod.qmin_txy = pd.Series(index=[start_time], data=[0])
            if mod.qfomin_txy is None:
                mod.qfomin_txy = pd.Series(index=[start_time], data=[0])

            # if mod.qmin_inflow_series is not None:
            #    print(f"Qmin from series for mod {mod.mod_nr}. {mod.qmin_inflow_series}, {mod.qmin_inflow_series_yearly_mean}")


def sum_area_txys(areas):
    for name, area in areas.items():
        #print(name)
        for mod in area.modules:

            next_mod = mod
            while next_mod.topology[0] > 0:
                mod.global_enekv = mod.global_enekv + next_mod.enekv
                next_mod = area.modules[next_mod.topology[0] - 1]
            mod.global_enekv = mod.global_enekv + next_mod.enekv

            area.max_vol_txy = area.max_vol_txy + mod.max_vol_txy * mod.global_enekv
            area.min_vol_txy = area.min_vol_txy + mod.min_vol_txy * mod.global_enekv

            if mod.hyd_kobl is not None:
                #print(f"Module {mod.name} is a gathering module of hydraulig coupled modules: {mod.upstream_hyd_kobl_modules}. Coupling code {mod.hyd_kobl[0]}")
                if mod.qmax_txy is not None:
                    print(f"This module has a time dependent Qmax constraint which is not included in the one reservoir model!!!")
                if mod.qmin_txy is not None:
                    print(f"This module has a time dependent Qmin constraint which is not included in the one reservoir model!!!")

                hc_qmax = pd.Series(index=[start_time], data=[0])
                hc_qmax_const = 0.0
                for us_int_nr in mod.upstream_hyd_kobl_modules:

                    us_mod = area.modules[us_int_nr]

                    hc_qmax_const = max(hc_qmax_const, mod.qmax * us_mod.enekv)
                    if us_mod.qmax_txy is not None:
                        hc_qmax = hc_qmax + us_mod.qmin_txy * us_mod.enekv
                        # print(f"Upstream module {us_mod.mod_nr} has a time dependent Qmax constraint!!!")

                    if us_mod.qmin_txy is not None:
                        area.qmin_txy = area.qmin_txy + us_mod.qmin_txy * us_mod.enekv
                        #print(f"Upstream module {us_mod.mod_nr} has a time dependent Qmin constraint!!!")

                hc_qmax = np.clip(hc_qmax, 0.0, hc_qmax_const)
                area.qmax_txy = area.qmax_txy + hc_qmax

            elif not mod.coupled_module:
                area.qmax_txy = area.qmax_txy + mod.qmax_txy * mod.enekv
                area.qmin_txy = area.qmin_txy + mod.qmin_txy * mod.enekv


def write_area_txys_to_hdf5(areas, area_h5_folder):
    if not os.path.exists(area_h5_folder):
        os.makedirs(area_h5_folder)
    for name, area in areas.items():

        area_file = os.path.join(area_h5_folder, f"{name}.h5")

        min_vol = pd.Series(index=[start_time + pd.Timedelta(weeks=i) for i in range(52)], data=area.min_vol_txy.vals)
        max_vol = pd.Series(index=[start_time + pd.Timedelta(weeks=i) for i in range(52)], data=area.max_vol_txy.vals)
        qmax = pd.Series(index=[start_time + pd.Timedelta(weeks=i) for i in range(52)], data=area.qmax_txy.vals)
        qmin = pd.Series(index=[start_time + pd.Timedelta(weeks=i) for i in range(52)], data=area.qmin_txy.vals)

        min_vol.to_hdf(area_file, key='min_vol', mode='a')
        max_vol.to_hdf(area_file, key='max_vol', mode='a')
        qmax.to_hdf(area_file, key='qmax', mode='a')
        qmin.to_hdf(area_file, key='qmin', mode='a')
    return


def read_area_txys_from_hdf5(area_names, area_h5_folder):
    if not os.path.exists(area_h5_folder):
        exit("Could not find h5-dir with area txys...")

    areas = {}

    for name in area_names:

        area_file = os.path.join(area_h5_folder, f"{name}.h5")

        area = Area(name)
        area.min_vol = pd.read_hdf(area_file, 'min_vol')
        area.max_vol = pd.read_hdf(area_file, 'max_vol')
        area.qmax = pd.read_hdf(area_file, 'qmax')
        area.qmin = pd.read_hdf(area_file, 'qmin')

        areas[name] = area


    return areas


def copy_inflow_from_text_file_to_scenariodata_h5(data_dir, area_names, n_weeks=52, n_scen=50):

    scenario_data_file = h5py.File("C:\\areas\\ScenarioData.h5", 'a')

    for area in area_names:
        for inflow_type in ["R", "U"]:
            series_path = f"{area}-{inflow_type}"
            scenario_data_file["SYSTEM"].copy("I-1", series_path)
    scenario_data_file.close()
    scenario_data_file = h5py.File("C:\\areas\\ScenarioData.h5", 'a')

    for area in area_names:
        for inflow_type in ["R", "U"]:
            inflow_file_name = f"{data_dir}/{area}_{inflow_type}.ascii"
            inflow_scenarios = np.zeros((n_scen, n_weeks))
            with open(inflow_file_name, 'r') as inflow_file:
                allData = inflow_file.readlines()
                for scen in range(n_scen):
                    inflow_scenarios[scen][0:n_weeks] = allData[4 + scen * n_weeks:4 + (scen + 1) * n_weeks]
            series_path = f"SYSTEM/{area}-{inflow_type}"
            counter = 0
            for key in scenario_data_file.get("SYSTEM/I-1").keys():
                scenario_data_file[series_path][key][0:n_weeks * 7] = (1000 / (168 * 3.6)) * np.repeat(
                    inflow_scenarios[counter][0:n_weeks], 7, axis=0)
                counter = counter + 1
    del scenario_data_file["SYSTEM"]["I-1"]
    scenario_data_file.close()

    return


if __name__ == "__main__":
    NO1 = ["OSTLAND"]
    NO2 = ["HAUGESUND", "SORLAND", "TELEMARK", "SOROST"]
    NO5 = ["VESTSYD", "HALLINGDAL", "VESTMIDT"]
    NO3 = ["MOERE", "NORGEMIDT", "NORDVEST"]
    NO4 = ["HELGELAND", "SVARTISEN", "TROMS", "FINNMARK"]
    SE1 = ["SVER-SE1"]
    SE2 = ["SVER-SE2"]
    SE3 = ["SVER-SE3"]
    SE4 = ["SVER-SE4"]

    areas = parse_area_modules(dirname)


