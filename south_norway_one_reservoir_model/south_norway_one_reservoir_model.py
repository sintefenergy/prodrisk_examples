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

LTM_input_folder = workin_dir + "south_norway/"
prodrisk = ltm_input.build_prodrisk_model(LTM_input_folder, n_weeks=52*6, start_time="2022-01-03")

# This input should be set based on for example the dimension of the water value matrix in build_prodrisk_model:
prodrisk.n_price_levels = 7
prodrisk.n_processes = 7 # number of mpi processes

# --- run prodrisk session ---
prodrisk.use_coin_osi = False
#prodrisk.aggregated_price_period_start_week = 1
prodrisk.command_line_option = "-NOHEAD"
prodrisk.deficit_power_cost = 200.0

#prodrisk.residual_model=1

status = prodrisk.run()

my_area = prodrisk.model.area["my_area"]

expected_objective_val_kkr = my_area.expected_objective_value.get()
print(f"Expected objective value: {expected_objective_val_kkr} kkr")


fcost = my_area.forward_cost.get()
kcost = my_area.backward_cost.get()
iteration_numbers = range(1, len(fcost)+1)

df = pd.DataFrame({"F-cost": pd.Series(data=fcost, index=iteration_numbers),
                   "K-cost": pd.Series(data=kcost, index=iteration_numbers),
                   })



fig = px.line(df, labels={
                     "index": "Iteration number",
                     "value": "Cost"
                 })
fig.show()


mod = prodrisk.model.module['South_Norway']

rsv_vols = mod.reservoirVolume.get()
inflow = mod.localInflow.get()
production = mod.production.get()
price = my_area.price.get()

ltm_res.plot_percentiles(rsv_vols, "Volume [Mm3]", "", percentiles_limits=[0, 25, 50, 75, 100])
ltm_res.plot_percentiles(inflow, "Reservoir inflow [m3/s]", "", percentiles_limits=[0, 25, 50, 75, 100])
ltm_res.plot_percentiles(production, "Production [MW]", "", percentiles_limits=[0, 25, 50, 75, 100])
ltm_res.plot_percentiles(price, "Price [EUR/MWh]", "", percentiles_limits=[0, 25, 50, 75, 100])








