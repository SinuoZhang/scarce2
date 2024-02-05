import fipy
import gmsh
import numpy as np
from fipy.solvers import Solver as solver_t  # typing
from matplotlib import pyplot as plt
from scipy import constants, interpolate

import solver

EPSILON_SI = 1.04e-10  # Permittivity of silicon [F/m]
DENSITY_SI = 2.3290  # Density of silicon [g cm^-3]


class Sensor(object):
    def __init__(
        self, n_pixel: int = 7, pitch: int | float = 50, electrode_size: int | float = 30, thickness: int | float = 100
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

        self.n_eff = 1.45e12
        self.mesh_file = "/tmp/mesh.msh2"

    def generate_mesh(self):
        """Generate mesh to solve equations on.

        The labeling (tags) of the points is as follows:
        8 --- 7 --3*pitch-- 6 --- 5
        |     |             |     |
        |     |  fine mesh  |     |
        |     |             |     |
        1 --- 2 --3*pitch-- 3 --- 4

        Points 2, 3, 6, 7 have higher mesh density for higher precision for the central electrode
        """
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
                mesh_spacing = 1
            elif pnt_i in [1, 2]:  # central region on backplane
                mesh_spacing = 2
            else:
                mesh_spacing = 5  # outer corners
            points.append(m.geo.addPoint(*pnt, mesh_spacing, pnt_i + 1))

        lines = [m.geo.addLine(points[i], points[i + 1]) for i in range(len(points) - 1)]
        lines.append(m.geo.addLine(points[-1], points[0]))

        line_loop = m.geo.addCurveLoop(lines)
        m.geo.addPlaneSurface([line_loop])

        m.geo.synchronize()
        m.mesh.generate(dim=3)
        gmsh.write("/tmp/mesh.msh2")  # fipy can only read msh version 2

    def setup_e_potential(self):
        """Define potential and fields"""
        self.mesh = fipy.Gmsh2D(self.mesh_file)

        self.potential = fipy.CellVariable(mesh=self.mesh, name="potential", value=0.0)
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
        """Define weighting potential and field"""
        self.mesh = fipy.Gmsh2D(self.mesh_file)

        self.w_potential = fipy.CellVariable(mesh=self.mesh, name="weighting_potential", value=0.0)

    def solve_e_potential(self, V_bias: int | float=-100, solver: solver_t = solver.LinearLUSolver):
        """Solve the electric potential equation.

        Args:
            solver (solver_t, optional): fipy solver to use. Defaults to solver.LinearLUSolver.
        """
        backplane = self.mesh.facesBottom
        readout_plane = self.mesh.facesTop

        # Boundary conditions. Set potential to 0 at readout electrodes (i.e. connected to GND)
        x, _ = np.array(self.potential.mesh.faceCenters)
        for pixel in range(self.n_pixel):
            pixel_pos = self.pitch * (pixel + 0.5) - self.pitch * self.n_pixel / 2
            self.potential.constrain(
                value=0.0,
                where=readout_plane
                & (x > pixel_pos - self.electrode_size / 2)
                & (x < pixel_pos + self.electrode_size / 2),
            )

        # Bias voltage applied on the backside
        self.potential.constrain(value=V_bias, where=backplane)

        solver = solver
        self.potential.equation = fipy.DiffusionTerm(coeff=1.0) + self.charge == 0.0
        self.potential.equation.solve(var=self.potential, solver=solver)

        self.potential.solved = True

    def solve_w_potential(self, solver: solver_t = solver.LinearLUSolver):
        """Solve the weighting potential equation.

        Args:
            solver (solver_t, optional): fipy solver to use. Defaults to solver.LinearLUSolver.
        """
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
        self.w_potential.equation.solve(var=self.w_potential, solver=solver)

        self.w_potential.solved = True

    def plot_potential(self, pot: fipy.CellVariable, plot_title: str = "Potential", colorbar_label=""):
        if not pot.solved:
            raise RuntimeWarning("Potential has not been solved yet!")
        # Interpolation
        X = np.linspace(min(pot.mesh.x), max(pot.mesh.x), 250)
        Y = np.linspace(min(pot.mesh.y), max(pot.mesh.y), 250)
        xx, yy = np.meshgrid(X, Y)
        grid = interpolate.griddata(
            np.transpose(pot.mesh.faceCenters), pot.arithmeticFaceValue, (xx, yy), method="linear"
        )

        aspect = pot.mesh.aspect2D
        fig_width = 9
        fig, ax = plt.subplots(figsize=(fig_width, aspect * fig_width))

        ax.set_title(plot_title)
        im = ax.pcolormesh(xx, yy, grid)
        cbar = plt.colorbar(im)
        cbar.set_label("Potential [V]")
        plt.show()


if __name__ == "__main__":
    s = Sensor()
    s.generate_mesh()
    s.setup_e_potential()
    s.setup_w_potential()
    s.solve_e_potential(V_bias=-100)
    s.solve_w_potential()
    s.plot_potential(s.potential, plot_title="Potential", colorbar_label="Potential [V]")
    s.plot_potential(s.w_potential, plot_title="Weighting potential")
