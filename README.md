# Convergence theory for Hermite approximations under adaptive coordinate transformations 

This repository contains source code and results of calculations accompanying the manuscript:

> Y. Saleh: Convergence theory for Hermite approximations under adaptive
> coordinate transformations. arXiv. 7494109 (2026).
> [DOI:tba](https://doi.org/10.1007/s11785-025-01874-5).

Contents
---
The experiments pertaining to interpolation problems are provided in
`Hermite_approx.ipynb`, `Hermite_lin_approx.ipynb`,
`Hermite_iResNet_approx.ipynb` and `Hermite_sublin_approx.ipynb` for Hermite
approximations employing identity, linear, iResNet, and power-law
transformations, respectively. The scripts save data in `simulations_data`.

In `plot_results.ipynb`, the simulation data are loaded and used to produce the
figures of the manuscript.

The experiments pertaining to the Morse oscillator are provided in
`Morse_Hermite.ipynb`, `Morse_scaled_Hermite.ipynb`, `Morse_NF_Hermite.ipynb`. Analysis is
performed in `Morse_analysis.py` and results are plotted in `plot_results_Morse.ipynb`

Citation
---
If you use this code in your research, please cite:

> Y. Saleh: Convergence theory for Hermite approximations under adaptive
> coordinate transformations. arXiv. 7494109 (2026).
> [DOI:tba](https://doi.org/10.1007/s11785-025-01874-5).


```bibtex
@article{Saleh:CAOT20:21,
	author = {Saleh, Yahya},
	year   = {2025},
	isbn   = {1661-8262},
	journal= {...},
	number = {1},
	pages  = {21},
	title  = {Convergence theory for Hermite approximations under adaptive coordinate transformations},
	url    = {https://doi.org/10.1007/s11785-025-01874-5},
	volume = {20},
	doi    = {10.1007/s11785-025-01874-5},
  }
```

Contact
---
For questions or feedback, feel free to open an issue or reach out to the authors directly __via__ yahya.saleh@mathmods.eu.
