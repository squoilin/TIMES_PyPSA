### TIMES_PyPSA
Soft-linking between the TIMES‑WAL and PyPSA‑WAL models. This repository includes a parser/visualizer that reads TIMES `.vd` output files and generates interactive Sankey energy-flow diagrams and a power capacity bar chart.

### Requirements
- **Python**: 3.9+ recommended
- **Libraries**: `pandas`, `plotly` (and `jupyter` if using the notebook)

Install with pip:
```bash
python -m pip install --upgrade pip
pip install pandas plotly jupyter
```

### Data
Files under `data/`:
- `demos_004_0209.vd`: example TIMES output file parsed by the script
- `commodities.csv`, `technologies.csv`, `times_variables.csv`: reference/mapping CSVs (inferred values)

### Outputs
Generated files under `output/` after running the script:
- [interactive_energy_sankey_pj.html](https://squoilin.github.io/TIMES_PyPSA/output/interactive_energy_sankey_pj.html)
- [interactive_energy_sankey_twh.html](https://squoilin.github.io/TIMES_PyPSA/output/interactive_energy_sankey_twh.html)
- [power_capacity_by_technology.html](https://squoilin.github.io/TIMES_PyPSA/output/power_capacity_by_technology.html)
- `sankey_data_2020_pj.csv`, `sankey_data_2020_twh.csv`: exported flows/nodes for year 2020 (year can be changed in the script)

The output is available through github pages at `https://squoilin.github.io/TIMES_PyPSA/`.

### How to run
Run the Python script (recommended):
```bash
cd scripts
python sankey_diagram.py
```
This reads `data/demos_004_0209.vd`, creates interactive Sankey diagrams in PJ and TWh, a capacity bar plot, and exports CSVs to `output/`.

Run the notebook:
```bash
jupyter lab  # or: jupyter notebook
```
Then open `scripts/run_sankey.ipynb` and run all cells.

### View the notebook online
If you prefer not to run locally, view the notebook on nbviewer to display the plotly content:
[View on nbviewer](https://nbviewer.org/github/squoilin/TIMES_PyPSA/blob/main/scripts/run_sankey.ipynb)
