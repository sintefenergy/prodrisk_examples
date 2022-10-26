import pandas as pd
import numpy as np
import plotly.express as px


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


def plot_iteration_costs(prodrisk):
    my_area = prodrisk.model.area["my_area"]
    fcost = my_area.forward_cost.get()
    kcost = my_area.backward_cost.get()

    if fcost is not None:
        iteration_numbers = range(1, len(fcost) + 1)

        df = pd.DataFrame({"F-cost": pd.Series(data=fcost, index=iteration_numbers),
                           "K-cost": pd.Series(data=kcost, index=iteration_numbers),
                           })

        fig = px.line(df, labels={
            "index": "Iteration number",
            "value": "Cost"
        })
        fig.show()
    
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

