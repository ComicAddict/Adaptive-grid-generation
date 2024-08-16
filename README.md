# Adaptive Grid Generation for Discretizing Implicit Complexes

![](https://github.com/user-attachments/assets/c102e033-dbee-4fe5-942f-953cd21f0f14)

This code implements the ACM SIGGRAPH 2024 paper: [Adaptive grid generation for discretizing implicit complexes](https://dl.acm.org/doi/10.1145/3658215). Given one or multiple functions in 3D and desired type of implicit complex defined by these functions (e.g., implicit arrangement, material interface, CSG, or curve network), this algorithm generates an adaptive simplicial (tetrahedral) grid, which can then be used to discretize the implicit complex using [this robust method](https://github.com/duxingyi-charles/Robust-Implicit-Surface-Networks/tree/main).

## Build

Use the following command to build: 

```bash
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make
```
The program `gridgen` will be generated in the build file. 

### Dependency

Currently, all the packages dependencies are available. These are currently in a release process and will be available shortly.

## Usage

To use the `gridgen` tool, you must provide an initial grid file and implicit function file as arguments, along with any desired options.

```bash
./gridgen <grid> <function> [OPTIONS]
```

### Positional Arguments

- `grid` : The path to the initial grid file that will be used for gridgen. This file can either be a `.msh` or `.json` file. 
Examples of grid files can be found in the `data/grid` directory.
- `function` : The path to the implicit function file that is used to evaluate the isosurface.

### Options

- `-h, --help` : Show the help message and exit the program.
- `-t, --threshold` : Set the threshold value for the isosurface generation. This is a `DOUBLE` value that defines the precision level of the gridgen.
- `-o, --option` : Set the type of the implicit complexes from Implicit Arrangement(IA), Material Interface(MI), or Constructive Solid Geometry(CSG). This is a `STRING` value that takes input from "IA", "MI", and "CSG". The default type is "IA".
- `--tree` : The path to the CSG tree file that defines the set of boolean operations on the functions. Only required if the option is set to be "CSG".
- `-c, --curve_network` : Set the switch of extracting only the Curve Network. Notice that Curve Network of all the above implicit complexes are different given the same set of input functions. This is a `BOOLEAN` type that takes in 1 or 0.
- `-m, --max-elements` : Set the maximum number of elements in the grid after refinement. This is an `INT` value that limits the size of the generated grid. If this value is a **negative** number, the grid will be refined until the threshold value is reached.
- `-s,--shortest-edge` : Set the shortest length of edges in the grid after refinement. This is a `DOUBLE` value that defines the shortest edge length.

## Example

The following is an example of how to use the `gridgen` tool with all available options:

```bash
./gridgen ../data/grid/cube6.msh ../data/function_examples/1-sphere.json -t 0.01 -o "IA" -m 10000 -s 0.05
```

In this example, `cube6.msh` is the initial grid file, `sphere.json` is the implicit function file, `0.01` is the threshold value, "IA" is the type of implicit complexes, `10000` is the maximum number of elements, and `0.05` is the shortest edge length.

Examples from the paper can be found in the `data` folder, where each example contains a grid file and a function file. If the example is a CSG, then it also includes a CSG tree file. Each figure folder also contains a `figure.sh` and a `figure.bat` file containing the exact commandline to produce the examples from the paper.

## Help

You can always run `./gridgen -h` to display the help message which provides usage information and describes all the options available.

## Output and After Grid Generation

The complete set of output files include data files (`tet_mesh.msh`, `active_tets.msh`, `grid.json`, and `function_value.json`) and information files (`timings.json` and `stats.json`). The first two files can be viewed using [Gmsh](https://gmsh.info/) software showing the entire background grid or only the grid elements containing the surfaces. The last two files are for the later tool to discretize the implicit complex. 

We have an off-the-shelf algorithm that extracts the isosurfacs robustly from the grid for implicit complexes. First download and build [this discretization method](https://github.com/duxingyi-charles/Robust-Implicit-Surface-Networks/tree/main) following its instructions. 

After generating the grid using our method, please use the above discretization tool by replacing its `config_file` with `data/config.json` according its usage example: 

```bash
./impl_arrangement [OPTIONS] config_file
```

 Currently, this discretization tool supports implicit arrangement, material interface, and CSG. For CSG, please specify the same path to the CSG tree file using the same command.
 
 ## Information Files
 
timing.json: timing of different stages of our pipeline. Details can be found in the paper.

stats.json: statistics of the background grid, e.g., the worst tetrahedral quality (radius ratio).
