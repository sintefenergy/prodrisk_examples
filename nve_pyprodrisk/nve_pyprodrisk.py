import os
import pandas as pd
import numpy as np
import plotly.express as px
import ltm_results as ltm_res
import read_prodrisk_data_from_LTM_folder as ltm_input

from pyprodrisk import ProdriskSession

# SET UP WORKING DIR #
workin_dir = os.getcwd()+"/"

# --- create a new session ---

data_dir = workin_dir + "subareas/"
NO1 = ["OSTLAND"]
NO2 = ["HAUGESUND", "SORLAND", "TELEMARK", "SOROST"]
NO3 = ["MOERE", "NORGEMIDT", "NORDVEST"]
NO4 = ["HELGELAND", "SVARTISEN", "TROMS", "FINNMARK"]
NO5 = ["VESTSYD", "HALLINGDAL", "VESTMIDT"]

# deterministic_scen: use specified price scenario from PRISREKKE.PRI as the price for all inflow scenarios.
# If not specified: the first price scenarios is used together with the first inflow scenario, etc.
prodrisk = ltm_input.build_prodrisk_model(data_dir, ["HAUGESUND"], n_weeks=52*6, deterministic_scen=0)


# --- run prodrisk session ---
prodrisk.use_coin_osi = True
prodrisk.command_line_option = "-NOHEAD"
prodrisk.deficit_power_cost = 200.0         # Rationing cost. All input prices will be capped at this value.

#prodrisk.residual_model = 1

prodrisk.keep_working_directory = True   # Keep temp run-folder for debugging purposes.
prodrisk.temp_dir = "/home/jovyan/work/temp"
prodrisk.log_file_path = "/home/jovyan/work/LogFiles"
prodrisk.n_processes = 1    # number of mpi processes
prodrisk.mpi_path = "/opt/intel/oneapi/mpi/latest/bin"

prodrisk.prodrisk_path = "/prodrisk/ltm_core_bin"
prodrisk.prodrisk_variant = "prodrisk"


prodrisk.max_iterations = 5
prodrisk.aggregated_price_period_start_week = 1

status = prodrisk.run()

my_area = prodrisk.model.area["my_area"]

expected_objective_val_kkr = my_area.expected_objective_value.get()
print(f"Expected objective value: {expected_objective_val_kkr} kkr")


ltm_res.plot_iteration_costs(prodrisk)


#mod = prodrisk.model.module['South_Norway']

# Market results. Input and output price
price = my_area.price.get()
output_price = my_area.output_price.get()

### Available area results:
# "water_value_result", "total_reservoir_volume", "total_reservoir_overflow", "total_production",
# "total_discharge", "total_energy_consumed", "total_energy_pumped", "total_storable_inflow", "total_nonstorable_inflow"

# Dict of results to plot. Keys: result attribute names, values: y-axis name in plot.
selected_area_results = {"water_value_result":          "Water values",
                         "total_reservoir_volume":      "Total reservoir volume",
                         "total_reservoir_overflow":    "Total reservoir overflow",     #Sum of water through bypass and spillage waterroutes, scaled with local energy equivalent
                         "total_production":            "Total production",
                         "total_storable_inflow":       "Total inflow"
                         }

# Aggregated hydro results
for result_attr, plot_name in selected_area_results.items():
    result_ts = my_area[result_attr].get()

    unit = prodrisk._pb_api.GetAttributeInfo("area", result_attr, "yUnit")

    ltm_res.plot_percentiles(result_ts, f"{plot_name} [{unit}]", "", percentiles_limits=[0, 25, 50, 75, 100])








