# Alternatives to Contour Visualizations for Power Systems Data

The scripts in this repo are for handling DSS distribution grid data, primarily from the SFO Smart Grid. The various directories in src/ are from:
1. p13u: This is the code to go with the paper "Alternatives to Contour Visualizations for Power Systems Data" by Isaiah Lyons-Galante, Morteza Karimzadeh, Samantha Molnar, Graham Johnson, and Kenny Gruchalla. It uses a feeder within p13u with about 24,000 buses.
2. smbl_envc_6: this is another section of the SFO grid, but this one includes more types of grid components such as transformers, batteries, and solar panels. It also has time series of bus voltages.
3. evsatscale: this contains some of the code for processing the EV charging simulations. 
4. 10x: This code is for scaling up the analysis in p13u to a 10x larger grid with about 300,0000 buses. 

Please note that many of the data paths in the scripts will need to be updated to match the new folder structure in this repo.

## Environment setup (conda + R + Python)
1. Create the env (installs R 4.3, geo stack, and Python with boto3/jupyter):  
   `mamba env create -f environment.yml -p ../.conda/envs/voltage-tiling`
2. Activate it when working here:  
   `mamba activate /Users/korey417/Desktop/Scientific\ Visualization/cs237-transmission-grid-project/.conda/envs/voltage-tiling`
3. Register the Jupyter kernel (already installed as `Python 3.11 (voltage-tiling)`; rerun if needed):  
   `python -m ipykernel install --user --name voltage-tiling --display-name "Python 3.11 (voltage-tiling)"`
4. Install a few R packages not available via conda (run inside the env):  
   `Rscript -e 'install.packages(c("ggvoronoi","h3","h3jsr","vioplot","dotenv"), repos="https://cloud.r-project.org")'`
5. Update data paths in the Rmd/ipynb files to point at your local `data/` if needed.

For the S3 download notebooks in `src/10x/`, select the `Python 3.11 (voltage-tiling)` kernel in Jupyter/VS Code and configure your AWS credentials if you need to pull from S3.

## Julia environment setup

A Julia 1.12 environment is configured for data analysis and visualization. The environment includes:
- **DataFrames, CSV, Parquet** - data manipulation
- **GeoStats** - geostatistical analysis
- **Plots, Makie** - visualization

### Using Julia in Jupyter

1. Select the **Julia 1.12 (voltage-tiling)** kernel in Jupyter/VS Code
2. The kernel automatically uses the project's Julia depot and packages

### Running Julia from command line

```bash
# Set environment and run Julia
export JULIA_DEPOT_PATH="/Users/korey417/Desktop/Scientific Visualization/cs237-transmission-grid-project/.julia_depot"
/Users/korey417/Desktop/Scientific\ Visualization/cs237-transmission-grid-project/.juliaup/juliaup/julia-1.12.1+0.aarch64.apple.darwin14/bin/julia \
  --project="/Users/korey417/Desktop/Scientific Visualization/cs237-transmission-grid-project/.julia_depot/environments/v1.12"
```

### Reinstalling Julia packages (if needed)

```julia
using Pkg
Pkg.add(["IJulia", "DataFrames", "CSV", "Plots", "GeoStats", "Makie", "Parquet"])
```

### Registering the Julia kernel (if needed)

```julia
using IJulia
installkernel("Julia 1.12 (voltage-tiling)",
    env=Dict("JULIA_DEPOT_PATH" => "/Users/korey417/Desktop/Scientific Visualization/cs237-transmission-grid-project/.julia_depot"))
```
