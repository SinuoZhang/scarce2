from typing import Tuple
from scarce2.sensor import Sensor

import fipy
import numpy as np
from numba import njit


def collected_charge_vs_time(
    sensor: Sensor, initial_pos: Tuple[float, float] = (0, 0), timestep: float = 0.1
) -> dict:
    """Propagate an electron hole pair through the sensor and calculate the charge signal induced by each type of charge carrier.

    Args:
        sensor (Sensor): A sensor object with electric and weighting field on a numpy meshgrid.
        initial_pos (Tuple[float, float], optional): Initial position of th electron hole pair in um. Defaults to (0, 0).
        timestep (float, optional): Time step for the propagation. Defaults to 0.1.

    Returns:
        dict: Dictionary with keys "electrons" and "holes" for the respective time series of induced charge
    """

    # Take inverse polarity of holes into account via mobility constant.
    # Only valid for silicon. Ignores (small) dependency on electircal field and temperature.
    mobility = {"electrons": 140, "holes": -50}  # µm / V / ns
    current_pos = {"electrons": initial_pos, "holes": initial_pos}
    charge = {"electrons": [], "holes": []}

    e_field_x = sensor.griddata["electric"]["field_x"]
    e_field_y = sensor.griddata["electric"]["field_y"]
    pot = (
        1000 * sensor.griddata["weighting"]["potential"]
    )  # scale to larger values for numeric stability of propagation

    for charge_type in ["electrons", "holes"]:
        current_wf = _calculate_field(initial_pos, pot, (sensor.n_pixel * sensor.pitch, sensor.thickness))
        while True:
            e_field_at_pos = (
                _calculate_field(current_pos[charge_type], e_field_x, (sensor.n_pixel * sensor.pitch, sensor.thickness)),
                _calculate_field(current_pos[charge_type], e_field_y, (sensor.n_pixel * sensor.pitch, sensor.thickness)),
            )
            current_pos[charge_type] = _propagate_charge(
                current_pos[charge_type], e_field_at_pos, mobility[charge_type], timestep
            )

            if charge_type == "electrons":
                stop_cond_pos = (
                    current_pos[charge_type][1] >= sensor.thickness - 0.0025
                )  # electron reached frontside of sensor
            elif charge_type == "holes":
                stop_cond_pos = current_pos[charge_type][1] <= 0.0025  # hole reached backside of sensor

            stop_cond_pos |= (
                abs(current_pos[charge_type][0] - (sensor.n_pixel * sensor.pitch)) <= 0.0025
            )  # charge reached side of sensor

            new_wf = _calculate_field(current_pos[charge_type], pot, (sensor.n_pixel * sensor.pitch, sensor.thickness))
            charge_per_step = 0.001 * (new_wf - current_wf)  # scale down to proper value after propagation

            if stop_cond_pos:
                break

            charge[charge_type].append(charge_per_step if charge_type == "electrons" else -1 * charge_per_step)
            current_wf = new_wf

    return charge


@njit
def _calculate_field(position: Tuple[float, float], field, sensor_dim: Tuple[float, float]) -> float:
    bin_width_x = sensor_dim[0] / 500  # TODO: take binning from the sensor object
    bin_width_y = sensor_dim[1] / 500  # TODO: take binning from the sensor object
    field_at_pos = field[
        int(np.floor(position[1] / bin_width_y)), int(np.floor((position[0] + 0.5 * sensor_dim[0]) / bin_width_x))
    ]

    return field_at_pos


@njit
def _propagate_charge(start_loc: Tuple[float, float], field, mobility, timestep: float = 0.1) -> float:
    x_start, y_start = start_loc
    field_x, field_y = field
    x_final = x_start + mobility * field_x * timestep
    y_final = y_start + mobility * field_y * timestep
    return (x_final, y_final)
