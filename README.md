# 🌪️ Curled Wake Model

A fast, modular, and extensible Python solver for wind farm flows using the **Curled Wake Model (CWM)**, based on the method introduced by [Martínez-Tossas et al. (2021)](https://wes.copernicus.org/articles/6/555/2021/). This tool enables rapid simulation and analysis of wind turbine wake interactions in atmospheric boundary layer (ABL) flows.

---

## 📘 Reference

The curled wake model was first introduced in:

> **Martínez-Tossas, L. A., Annoni, J., Fleming, P. A., and Churchfield, M. J. (2019).**  
> The aerodynamics of the curled wake: a simplified model in view of flow control.  
> Wind Energy Science, 2019.  
> [https://doi.org/10.5194/wes-4-127-2019](https://doi.org/10.5194/wes-4-127-2019)


This latest implementation follows the methodology in:

> **Martínez-Tossas, L. A., King, J., Quon, E., Bay, C. J., Mudafort, R., Hamilton, N., Howland, M. F., and Fleming, P. A. (2021).**  
> The curled wake model: a three-dimensional and extremely fast steady-state wake solver for wind plant flows.  
> Wind Energy Science, 2021.  
> [https://doi.org/10.5194/wes-6-555-2021](https://doi.org/10.5194/wes-6-555-2021)

---

## 🚀 Features

- ✅ Steady-state and unsteady wake solvers using Runge-Kutta integration  
- ✅ Support for yawed turbines and multiple wake interaction models  
- ✅ Modular turbulence model options (`standard`, `kl`, etc.)  
- ✅ Flexible boundary layer initialization (power-law, log-law, TI-derived)  
- ✅ Wake veer and dynamic inflow support  
- ✅ Time-resolved output for comparison with SCADA or LES  
- ✅ Plotting tools and error analysis against SCADA data  

---

## 🧠 Structure

```
curled_wake_model/
├── wind_farm.py                  # Main driver and solver orchestrator
├── wind_farm_plot.py               # Visualization methods (attached as mixins)
├── wind_farm_solvers/
│   ├── solver_steady.py            # Steady-state RK solver
│   ├── solver_time.py              # Full unsteady RK solver
│   └── solver_quasisteady.py       # Quasi-steady solver (fast approximated time evolution)
├── turbine_model.py                # Turbine class including actuator and wake properties
├── examples/                       # Example wind farm layouts and runs
├── requirements.txt                # Dependencies
└── README.md
```

---

## 📦 Installation

Install dependencies via conda and pip:

```bash
# 1. Create a conda environment with Python >= 3.12
conda create -n cwm_env python=3.12

# 2. Activate the environment
conda activate cwm_env

# 3. Clone the rk4_solvers branch of the repository
git clone -b rk4_solvers https://github.com/tonyinme/curled_wake_model.git
cd curled_wake_model

# 4. Install dependencies
pip install -r requirements.txt

# 5. Install the package in development mode
pip install -e .
```

If using Windows this can be done using Powershell
```bash
:: 1. Make sure Python 3.12 or newer is installed
::    You can check with:
python --version

:: 2. (Optional) Create a virtual environment
python -m venv cwm_env

:: 3. Activate the virtual environment
cwm_env\Scripts\activate

:: 4. Clone the rk4_solvers branch of the repository
git clone -b rk4_solvers https://github.com/tonyinme/curled_wake_model.git
cd curled_wake_model

:: 5. Install dependencies
pip install -r requirements.txt

:: 6. Install the package in development mode
pip install -e .
```

You may also want to install:

- `matplotlib`
- `scipy`
- `pandas`
- `numba`
- `ffmpeg` (optional, for animation exports)

---

## 🌀 Quick Start

```python
from curled_wake_model import turbine_model as tm
from curled_wake_model.wind_farm import *

# Define turbines
turbines = [tm.turbine_model_class(D=126, th=90, alpha=20, location=(100, 250, 90))]

# Create wind farm
wf = wind_farm_class(Lx=1000, Ly=500, Lz=300, Nx=100, Ny=50, Nz=30, turbines=turbines)

# Set boundary layer
wf.add_boundary_layer(alpha_shear=0.2)

# Run solver
wf.solve()

# Plot
wf.plot_streamwise()
```

---

## 📊 Output

- Wake field components: `U`, `V`, `W`, and deficits `uw`, `vw`, `ww`
- Time-resolved wake snapshots: `wf.u_time`, `wf.time_video`
- Turbine-specific diagnostics: power, Ct, Cp, time series

---

## 📁 Example Run

```bash
python wind_farm.py
```

This executes a sample 6-turbine layout, computes the wake fields, and saves downstream plane plots.

---

## 📜 License

This project is licensed under the **GNU General Public License v2.0**.  
See the [LICENSE](https://www.gnu.org/licenses/old-licenses/gpl-2.0.html) file for details.

© 2025 Luis A. Martínez Tossas

---

## 🤝 Acknowledgments

This model is based on the **Curled Wake Model** developed at the National Renewable Energy Laboratory.

See the original papers:
  
[https://wes.copernicus.org/articles/6/555/2021/](https://wes.copernicus.org/articles/6/555/2021/)
