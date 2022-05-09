import os
import ltm_results as ltm_res
import pandas as pd

# SET UP WORKING DIR #
workin_dir = os.getcwd()+"/"

# --- create a new session ---

LTM_input_folder = workin_dir + "prodrisk_case/"

rsv_vols = ltm_res.get_txy_series(LTM_input_folder, "ModuleA", "reservoir", start_time=pd.Timestamp('2022-01-03'))
inflow = ltm_res.get_txy_series(LTM_input_folder, "ModuleA", "reservoir_inflow")
production = ltm_res.get_txy_series(LTM_input_folder, "ModuleA", "production")
price = ltm_res.get_area_txy_series(LTM_input_folder, "market", "price")



ltm_res.plot_percentiles(rsv_vols, "Volume [Mm3]", "", percentiles_limits=[0, 25, 50, 75, 100])
ltm_res.plot_percentiles(inflow, "Reservoir inflow [m3/s]", "", percentiles_limits=[0, 25, 50, 75, 100])
ltm_res.plot_percentiles(production, "Production [MW]", "", percentiles_limits=[0, 100])
ltm_res.plot_percentiles(price, "Price [EUR/MWh]", "", percentiles_limits=[0, 100])








