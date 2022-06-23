import pandas as pd
import numpy as np
import plotly.express as px
import ltm_results as ltm_res

from pyprodrisk import ProdriskSession

# --- create a new session ---
prodrisk = ProdriskSession(license_path='', silent=False, log_file='')

# --- configure settings for the session ---

prodrisk.set_optimization_period(
    pd.Timestamp("2021-01-04"),
    n_weeks=156
)

prodrisk.keep_working_directory = True
prodrisk.use_coin_osi = True
prodrisk.temp_dir = "C:/temp/"
prodrisk.prodrisk_path = "C:/PRODRISK/ltm_core_bin_r18408/"

prodrisk.write_penalty_logfiles = True

prodrisk.n_scenarios = 10
prodrisk.command_line_option = "-SEKV"
prodrisk.min_iterations = 1  # default 1
prodrisk.max_iterations = 5
prodrisk.min_iterations_first_run = 1  # default 1
prodrisk.max_iterations_first_run = 5

prodrisk.n_price_levels = 7  # number of levels in discrete price model (include max and min)
#prodrisk.max_allowed_scens_per_node = 1

prodrisk.n_processes = 1  # number of mpi processes
prodrisk.price_periods = pd.Series(
    index = [prodrisk.start_time + pd.Timedelta(days=i) for i in range(7)],
    data=[1, 2, 3, 4, 5, 6, 7]
    #data=[1, 2, 1, 2, 1, 2, 2]
)

prodrisk.deficit_power_cost = 500.0
prodrisk.surplus_power_cost = 0.02
prodrisk.water_ration_cost = 1000.0     # CTANK

prodrisk.suppress_seq_res = 0
prodrisk.aggregated_price_period_start_week = 104
prodrisk.sequential_price_period_start_week = 1
prodrisk.sequential_price_period_end_week = 40
prodrisk.reservoir_balance_option = 1

# SHOP file settings
# prodrisk.prepare_shop_input = True                # Create shop files
# prodrisk.shop_directory = "C:/SHOP/basic/today/"  # Copy SHOP files to this directory (should already exist!)
# prodrisk.shop_file_name = "basic123"              # preface of shop file names (default: SHOP-SYSTEM)
# prodrisk.shop_cut_weeks = [1, 2]                  # Create extended cut files for week 1 and 2.

# prodrisk.residual_model = 1


# --- add a module to the session ---

mod = prodrisk.model.module.add_object('ModuleA')
mod.name.set('ModuleA')
mod.plantName.set('PlantA')
mod.number.set(1001)
mod.ownerShare.set(1.0)
mod.regulationType.set(1)

mod.rsvMax.set(200.0)
mod.connectedSeriesId.set(1)
mod.meanRegInflow.set(630.0)
mod.meanUnregInflow.set(0.0)
mod.nominalHead.set(700.0)
mod.submersion.set(15.0)
mod.volHeadCurve.set(pd.Series(name=0.0, index=[0.0, 200.0], data=[600.0, 1000.0]))

# Set refVol to improve convergence of first main iteration.
# Default value for this attribute is a typical series (based on the start date) with high volume in fall and low volume in spring
mod.refVol.set(pd.Series(name=0.0,
                         index=[prodrisk.start_time + pd.Timedelta(weeks=i)
                                for i in [0, 18, 24, 52, 70, 76, 104, 122, 128, 156]],
                         data=[45.0, 10.0, 90.0, 100.0, 10.0, 90.0, 100.0, 10.0, 90.0, 100.0]))


mod.PQcurve.set(pd.Series(name=50.0, index=[0, 20.0, 40.0, 80.0], data=[0, 20.0, 30.0, 65.0]))
mod.energyEquivalentConst.set(0.341)
mod.maxDischargeConst.set(65.0)
mod.maxProd.set(80.0)
mod.maxBypassConst.set(10000.0)
mod.topology.set([1002, 1002, 0])

mod.startVol.set(90.0)

mod.minVol.set(pd.Series(name=0.0,
                         index=[prodrisk.start_time + pd.Timedelta(weeks=i)
                                for i in [0, 17, 18, 40, 41, 69, 70, 92, 93, 121, 122, 144, 145, 156]],
                         data=[0.0, 0.0, 150.0, 150.0, 0.0, 0.0, 150.0, 150.0, 0.0, 0.0, 150.0, 150.0, 0.0, 0.0]))

mod.MinReservoirCostVolume.set(pd.Series( name=0.0,
                                            index=[prodrisk.start_time + pd.Timedelta(weeks=i) for i in range(156)],
                                            data=100*np.ones(156)))

# --- add a second module to the session ---

mod = prodrisk.model.module.add_object('ModuleB')
mod.name.set('ModuleB')
mod.plantName.set('PlantB')
mod.number.set(1002)
mod.ownerShare.set(1.0)
mod.regulationType.set(1)

mod.rsvMax.set(50.0)
mod.connectedSeriesId.set(1)
mod.meanRegInflow.set(20.0)
mod.meanUnregInflow.set(30.0)
mod.nominalHead.set(15.0)
mod.submersion.set(0.0)
mod.volHeadCurve.set(pd.Series(name=0.0, index=[0.0, 50.0], data=[10.0, 15.0]))

mod.PQcurve.set(pd.Series(name=50.0, index=[0, 50.0], data=[0, 70.0]))
mod.energyEquivalentConst.set(0.341)
mod.maxDischargeConst.set(70.0)
mod.maxProd.set(50.0)
mod.maxBypassConst.set(100.0)
mod.topology.set([0, 0, 0])

mod.startVol.set(45.0)

mod.minBypass.set(pd.Series(name=0.0,
                         index=[prodrisk.start_time + pd.Timedelta(weeks=i)
                                for i in [0, 17, 18, 40, 41, 69, 70, 92, 93, 121, 122, 144, 145, 156]],
                         data=[0.0, 0.0, 30.0, 30.0, 0.0, 0.0, 30.0, 30.0, 0.0, 0.0, 30.0, 30.0, 0.0, 0.0]))

mod.MinBypassCostVolume.set(pd.Series( name=0.0,
                                            index=[prodrisk.start_time + pd.Timedelta(weeks=i) for i in [40, 92, 144]],
                                            data=[100, 100, 100]))

# --- add inflow series to the session ---

y1 = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 100, 100, 100, 30, 30, 30, 30, 30, 30, 30, 30, 30,
        30, 30, 30, 30, 30, 30, 30, 30, 30, 45, 45, 45, 30, 30, 30, 30, 30, 30, 30, 30, 10, 10]
y2 = [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 50, 50, 50, 50, 10, 10, 10, 10, 10, 10, 10, 10,
        10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 1, 1, 1, 1, 1]
y3 = [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 60, 60, 60, 60, 15, 15, 15, 15, 15, 15,
        15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5]

inflow_df = pd.DataFrame(
    index=[prodrisk.start_time + pd.Timedelta(weeks=i) for i in range(prodrisk.n_weeks)],
    data={
        "scen0": y1 + y2 + y3,
        "scen1": y2 + y3 + y1,
        "scen2": y3 + y1 + y2,
        "scen3": y1 + y1 + y2,
        "scen4": y1 + y2 + y2,
        "scen5": y2 + y2 + y3,
        "scen6": y2 + y3 + y3,
        "scen7": y3 + y1 + y1,
        "scen8": y3 + y3 + y1,
        "scen9": y2 + y1 + y3,
    },
)

ser = prodrisk.model.inflowSeries.add_object('Serie1')
ser.seriesId.set(1)
ser.inflowScenarios.set(inflow_df)

# --- add area object (price and watervalue) to the session ---

area = prodrisk.model.area.add_object('my_area')
seasonPrice = [20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 15, 15, 15, 15, 15, 15, 8, 8, 8, 8, 8, 8, 8, 8,
                8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 10, 10, 10, 10, 10]
np.random.seed(5)
randomPrice = np.random.rand(prodrisk.n_scenarios * prodrisk.n_weeks, 1)
priceScenarios = [0.3 * randomPrice[i] * seasonPrice[i % 52] for i in range(len(randomPrice))]

priceScenIndexed = np.array(priceScenarios).reshape((prodrisk.n_scenarios, prodrisk.n_weeks))
price_df = pd.DataFrame(
    index=[prodrisk.start_time + pd.Timedelta(weeks=i) for i in range(prodrisk.n_weeks)],
    data={
        f"scen{i}": ps for i, ps in enumerate(priceScenIndexed)
    },
)
area.price.set(price_df)

# --- add simple water value matrix to the session ---
refs = []
nPoints = []
x = []
y = []

for i in range(prodrisk.n_price_levels.get()):
    refs.append(i)
    nPoints.append(51)

    for n in range(51):
        x.append(np.real(100 - n * 2))
        #y.append(np.real((10.0 + n * 0.3) * i * 0.5))
        y.append(0.0)

x_values = np.array(x).reshape((prodrisk.n_price_levels.get(), 51))
y_values = np.array(y).reshape((prodrisk.n_price_levels.get(), 51))
area.waterValue.set([
    pd.Series(name=ref, index=x_val, data=y_val) for ref, x_val, y_val in zip(refs, x_values, y_values)
])

# --- run prodrisk session ---

#prodrisk.model.build_connection_tree(write_file=True)

status = prodrisk.run()

if status is False:
    exit()

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


if prodrisk.write_penalty_logfiles.get() > 0:

    modA = prodrisk.model.module['ModuleA']
    modB = prodrisk.model.module['ModuleB']

    ltm_res.plot_percentiles(modA.reservoirVolume.get(), "Module A - Reservoir volume", "")
    ltm_res.plot_percentiles(modA.min_vol_violation.get(), "Module A - Min volume violation [Mm3]", "",
                             percentiles_limits=[0, 100])
    ltm_res.plot_txy(modA.min_vol_violation_last_backward.get(), "Module A - Min volume violation last backward [Mm3]", "")

    ltm_res.plot_percentiles(modB.bypass.get(), "Module B - Bypass [m3/s]", "")
    ltm_res.plot_percentiles(modB.min_bypass_violation.get(), "Module B - Min bypass violation [m3/s]", "",
                             percentiles_limits=[0, 100])
    ltm_res.plot_txy(modB.min_bypass_violation_last_backward.get(), "Module B - Min bypass violation last backward [m3/s]", "")

    ltm_res.plot_txy(modA.n_different_inflows.get(), "Module A - Number of calculated inflows in last backward recursion", "")
    ltm_res.plot_txy(modA.n_negative_inflows_back.get(), "Module A - Number of occurances of negative inflows in last backward iteration", "")
    ltm_res.plot_txy(modA.probability_of_negative_inflow_back.get(), "Module A - Probability of occurances of negative inflows in last backward iteration", "")
    ltm_res.plot_txy(modA.average_neg_inflow_back.get(), "Module A - Average negative inflow in last backward iteration", "")






