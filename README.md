# Distributed MPC for cooperative tasks at equilibria

![Python](https://img.shields.io/badge/Python-3.13-blue)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](./LICENSE)

This repository contains the simulation code used for the numerical example in Section 4.4.2 of my submission for a PhD thesis at the University of Stuttgart.

The implemented algorithm is adapted from
> M. Köhler, M. A. Müller, and F. Allgöwer, 'Distributed MPC for Self-Organized Cooperation of Multiagent Systems,' *IEEE Trans. Autom. Control*, vol. 69 (11), 7988--7995, 2024. doi: [10.1109/TAC.2024.3407633](https://doi.org/10.1109/TAC.2024.3407633)

## Usage

The code was run using *Python 3.13.1*, with the required packages listed in `requirements.txt`.

To run the code:

1. Install Python 3.13.1 (or a compatible version).
2. Create a virtual environment, activate it, and install the dependencies using (on Windows):

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## Notes on implementation

Please note that this is conceptual code, implemented without consideration for modularity, flexibility, or computational efficiency.

Terminal constraints for the quadrotor example are computed following the procedure presented in
> J. Köhler, M. A. Müller, and F. Allgöwer, 'A nonlinear model predictive control framework using reference generic terminal ingredients,' *IEEE Trans. Autom. Control*, vol. 65 (3), 3576--3583, 2019. doi: [10.1109/TAC.2019.2949350](https://doi.org/10.1109/TAC.2019.2949350)

This implementation uses [CasADi](https://web.casadi.org/docs/), [CVXPY](https://www.cvxpy.org/), and [IPOPT](https://github.com/coin-or/Ipopt).
