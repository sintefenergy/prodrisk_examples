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

LTM_input_folder = workin_dir + "session_2021-12-07-12-03-41/"
prodrisk = ltm_input.build_prodrisk_model(LTM_input_folder, n_weeks=156, start_time="2021-01-04")

# This input should be set based on for example the dimension of the water value matrix in build_prodrisk_model:
prodrisk.n_price_levels = 7

# --- run prodrisk session ---

status = prodrisk.run()

my_area = prodrisk.model.area["my_area"]

expected_objective_val_kkr = my_area.expected_objective_value.get()
print(f"Expected objective value: {expected_objective_val_kkr} kkr")


fcost_first_run = my_area.forward_cost_first_run.get()
kcost_first_run = my_area.backward_cost_first_run.get()
iteration_numbers_first_run = range(1, len(fcost_first_run)+1)

fcost = my_area.forward_cost.get()
kcost = my_area.backward_cost.get()
iteration_numbers = range(1, len(fcost)+1)

df = pd.DataFrame({"F-cost": pd.Series(data=fcost, index=iteration_numbers),
                   "K-cost": pd.Series(data=kcost, index=iteration_numbers),
                   "F-cost first run": pd.Series(data=fcost_first_run, index=iteration_numbers_first_run),
                   "K-cost first run": pd.Series(data=kcost_first_run, index=iteration_numbers_first_run),
                   })



fig = px.line(df, labels={
                     "index": "Iteration number",
                     "value": "Cost"
                 })
fig.show()


mod = prodrisk.model.module['ModuleA']

rsv_vols = mod.reservoirVolume.get()
inflow = mod.localInflow.get()
production = mod.production.get()
price = my_area.price.get()

ltm_res.plot_percentiles(rsv_vols, "Volume [Mm3]", "", percentiles_limits=[0, 25, 50, 75, 100])
ltm_res.plot_percentiles(inflow, "Reservoir inflow [m3/s]", "", percentiles_limits=[0, 25, 50, 75, 100])
ltm_res.plot_percentiles(production, "Production [MW]", "", percentiles_limits=[0, 25, 50, 75, 100])
ltm_res.plot_percentiles(price, "Price [EUR/MWh]", "", percentiles_limits=[0, 25, 50, 75, 100])








