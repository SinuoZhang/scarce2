import fipy
import gmsh
import os
import numpy as np
from fipy.solvers import Solver as solver_t  # typing
from scipy import constants, interpolate

from scarce2 import plotting, solver

EPSILON_SI = 1.04e-10  # Permittivity of silicon [F/m]
DENSITY_SI = 2.3290  # Density of silicon [g cm^-3]

SOLVER = solver.LinearLUSolver


class Sensor(object):
    def __init__(
        self,
        n_pixel: int = 7,
        pitch: int | float = 50,
        electrode_size: int | float = 10,
        thickness: int | float = 100,
    ):
        """Base class for different sensors

        Args:
            n_pixel (int, optional): Number of pixels or strips, should be odd number since regarding central electrode only. Defaults to 7.
            pitch (int | float, optional): Pitch (width) of the pixel or strip in µm. Defaults to 50.
            electrode_size (int | float, optional): Width of the readout electrode in µm. Defaults to 30.
            thickness (int | float, optional): Thickness of the sensor in µm. Defaults to 100.
        """
        self.n_pixel = n_pixel
        self.pitch = pitch
        self.electrode_size = electrode_size
        self.thickness = thickness

        self.n_eff = 2.7e12

        self.mesh_file_path = "/tmp"
        self.structure_name =  f"n{n_pixel}_p{pitch}_e{electrode_size}_d{thickness}"
        self.mesh_file_name = f"{self.structure_name}_mesh"
        self.mesh_file_ext = "msh2"
        self.mesh_file = os.path.join(self.mesh_file_path, f"{self.mesh_file_name}.{self.mesh_file_ext}")

        self.griddata = {}

    def generate_mesh(self, mesh_density=1, file_path=None):
        """Generate mesh to solve equations on.

        The labeling (tags) of the points is as follows:
        8 --- 7 --3*pitch-- 6 --- 5
        |     |             |     |
        |     |  fine mesh  |     |
        |     |             |     |
        1 --- 2 --3*pitch-- 3 --- 4

        Points 2, 3, 6, 7 have higher mesh density for higher precision for the central electrode
        """
        if file_path == None:
            file_path = self.mesh_file_path

        width = self.n_pixel * self.pitch
        outside_roi = (
            width - 3 * self.pitch
        ) / 2  # width of region left and right of central part (defined by 3 * pixel pitch)

        points_xyz = [
            [-width / 2, 0, 0],
            [-width / 2 + outside_roi, 0, 0],
            [width / 2 - outside_roi, 0, 0],
            [width / 2, 0, 0],
            [width / 2, self.thickness, 0],
            [width / 2 - outside_roi, self.thickness, 0],
            [-width / 2 + outside_roi, self.thickness, 0],
            [-width / 2, self.thickness, 0],
        ]

        gmsh.initialize()
        m = gmsh.model
        m.add("planar")

        points = []
        for pnt_i, pnt in enumerate(points_xyz):
            if pnt_i in [5, 6]:  # close to central readout electrode
                mesh_spacing = 1.0 / mesh_density
            elif pnt_i in [1, 2]:  # central region on backplane
                mesh_spacing = 2.0 / mesh_density
            else:
                mesh_spacing = 5.0 / mesh_density  # outer corners
            points.append(m.geo.addPoint(*pnt, mesh_spacing, pnt_i + 1))

        lines = [m.geo.addLine(points[i], points[i + 1]) for i in range(len(points) - 1)]
        lines.append(m.geo.addLine(points[-1], points[0]))

        line_loop = m.geo.addCurveLoop(lines)
        m.geo.addPlaneSurface([line_loop])

        m.geo.synchronize()
        m.mesh.generate(dim=2)
        mesh_file_tmp = os.path.join(file_path, f"{self.mesh_file_name}.{self.mesh_file_ext}")
        mesh_obj = os.path.join(file_path, f"{self.mesh_file_name}.obj")
        self.mesh_file = mesh_file_tmp
        print(self.mesh_file)
        gmsh.write(mesh_obj)
        gmsh.write(self.mesh_file)  # fipy can only read msh version 2

    def setup_e_potential(self):
        """Define electric potential"""
        self.mesh = fipy.Gmsh2D(self.mesh_file)

        self.e_potential = fipy.CellVariable(mesh=self.mesh, name="potential", value=0.0)
        electrons = fipy.CellVariable(mesh=self.mesh, name="e-")
        electrons.valence = -1
        self.charge = electrons * electrons.valence
        self.charge.name = "charge"

        epsilon = EPSILON_SI * 1e-6  # convert to um units
        rho = constants.elementary_charge * self.n_eff * (1e-4) ** 3  # Charge density in C / um3
        rho_epsilon = rho / epsilon  # compute rho/epsilon for better solvability.
        # In this case, no coefficient in the diffusion term is needed.
        electrons.setValue(rho_epsilon)  # Because of scaling, use rho_epsilon

    def setup_w_potential(self):
        """Define weighting potential"""
        self.mesh = fipy.Gmsh2D(self.mesh_file)

        self.w_potential = fipy.CellVariable(mesh=self.mesh, name="weighting_potential", value=0.0)

    def solve_e_potential(self, V_bias: int | float = -100):
        """Solve the electric potential equation.

        Args:
            V_bias (int | float, optional): (negative) bias potential at the backside of the sensor
        """
        backplane = self.mesh.facesBottom
        readout_plane = self.mesh.facesTop

        # Boundary conditions. Set potential to 0 at readout electrodes
        x, _ = np.array(self.e_potential.mesh.faceCenters)
        for pixel in range(self.n_pixel):
            pixel_pos = self.pitch * (pixel + 0.5) - self.pitch * self.n_pixel / 2
            self.e_potential.constrain(
                value=0.0,
                where=readout_plane
                & (x > pixel_pos - self.electrode_size / 2)
                & (x < pixel_pos + self.electrode_size / 2),
            )

        # Bias voltage applied on the backside
        self.e_potential.constrain(value=V_bias, where=backplane)

        self.e_potential.equation = fipy.DiffusionTerm(coeff=1.0) + self.charge == 0.0
        self.e_potential.equation.solve(var=self.e_potential, solver=SOLVER)

        self.e_potential.solved = True

    def solve_w_potential(self):
        """Solve the weighting potential equation."""
        backplane = self.mesh.facesBottom
        readout_plane = self.mesh.facesTop

        x, _ = np.array(self.w_potential.mesh.faceCenters)
        for pixel in range(self.n_pixel):
            pixel_pos = self.pitch * (pixel + 0.5) - self.pitch * self.n_pixel / 2
            self.w_potential.constrain(
                value=1.0 if pixel_pos == 0 else 0.0,
                where=readout_plane
                & (x > pixel_pos - self.electrode_size / 2)
                & (x < pixel_pos + self.electrode_size / 2),
            )
        self.w_potential.constrain(value=0.0, where=backplane)

        self.w_potential.equation = fipy.DiffusionTerm(coeff=1.0) == 0.0
        self.w_potential.equation.solve(var=self.w_potential, solver=SOLVER)

        self.w_potential.solved = True

    def convert_to_numpy(self, points=500, which: str = "both"):
        if which == "both":
            self.convert_to_numpy(which="electric")
            self.convert_to_numpy(which="weighting")
        else:
            if which == "electric":
                pot = self.e_potential
            elif which == "weighting":
                pot = self.w_potential
            else:
                raise RuntimeError("Illegal potential specification (supported are 'weighting' and 'electric')")

            # Interpolation
            X = np.linspace(min(pot.mesh.x), max(pot.mesh.x), points)
            Y = np.linspace(min(pot.mesh.y), max(pot.mesh.y), points)
            xx, yy = np.meshgrid(X, Y)
            potential = interpolate.griddata(
                np.transpose(pot.mesh.faceCenters),
                pot.arithmeticFaceValue,
                (xx, yy),
                method="linear",
            )

            field_x = interpolate.griddata(
                np.transpose(pot.mesh.faceCenters),
                pot.grad.arithmeticFaceValue[0],
                (xx, yy),
                method="linear",
            )
            field_y = interpolate.griddata(
                np.transpose(pot.mesh.faceCenters),
                pot.grad.arithmeticFaceValue[1],
                (xx, yy),
                method="linear",
            )

            # If grid has nan values from interpolation, fill with closest finite value
            for arr in [potential, field_x, field_y]:
                nan_mask = np.isnan(arr)
                arr[nan_mask] = np.interp(np.flatnonzero(nan_mask), np.flatnonzero(~nan_mask), arr[~nan_mask])

            self.griddata[which] = {"potential": potential, "field_x": field_x, "field_y": field_y}


if __name__ == "__main__":
    s = Sensor(n_pixel=7, thickness=150)
    s.generate_mesh()
    s.setup_e_potential()
    s.setup_w_potential()
    s.solve_e_potential(V_bias=-100)
    s.solve_w_potential()
    s.convert_to_numpy()
    # s.plot_potential(s.potential, plot_title="Electric potential", colorbar_label="Potential [V]")
    plotting.plot_potential(s.w_potential, n_pixel=s.n_pixel, plot_title="Weighting potential")
    plotting.plot_field(
        s.e_potential,
        n_pixel=s.n_pixel,
        plot_title="Electric field",
        colorbar_label="E [V / µm]",
    )
