# HierarchicalGeodesicModeling

## What is it?
HierarchicalGeodesicModeling is a 3D Slicer module is for computation and visualization/analysis of longtiudinal shape 
models using geodesic based shape modeling methodology. It is part of the newly added functionalities in [SlicerSALT 5.0.0 release](https://salt.slicer.org/).

![Applying HGM module to the longitudinal analysis of shape change in L1L2 intervertebral disc w.r.t. patient reported initial VAS.](/HGM_Screenshot.png)

This module implements a hierarchical geodesic modeling method based on our [IPMI paper](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10323213/).
It allows user to perform longitudinal analysis on anatomies of interest in shape space, which is a core aspect of many medical applications for understanding the relationship between an anatomical subject’s functions and its trajectory of shape change over time.

Given a set of correspondence established shapes of subjects, this module computes
1. a continuous trajectory of the shape change of individual subjects 
2. a mean trajectory of all input subjects
3. a hierarchical geodesic model (HGM) based on selected covariates

## How to use
1. Two inputs are required for using this module:
   - An input `directory` that contains all the shape files in ".vtk" format.
   - A `demographics(.csv)` file that has the information for each shape file including `filenames`, `subject index`, `time variables` and `covariates`.
   

2. After setting the inputs, click `Load data` to load the `demographics(.csv)` into the module. The first three columns of `demographics(.csv)` should be `filename`, `subject index` and `time variable`. The rest columns are `covariates` which can be clinical assessments or other functional values associated with the input shapes.


3. The loaded data will populate the module table and the option/parameter entries. The options/parameters are briefly described below:
   - Subject Level Longitudinal Model
     - Model Degree: Degree of individual subject's longitudinal geodesic polynomial model. Model degree should be lower than number of time points in the inputs (e.g. 2nd order model requires 3 or more input time points).
     - Id Type: Use `prefix` or `suffix` of the filename as the subject id for visualization and export, depending on user's file namings. If `index` is selected, subject id will be shown as "Subject `index`".
     - Id Length: Number of characters in `prefix` or `suffix`. 
   - Population Level Covariate Model
     - Select Covariate(s): If one or multiple covariates are selected, the module will compute a hierarchical geodesic model. Otherwise, only the mean and individual subjects' trajectories will be computed.
     - Covariates Time Index: Use the covariates at the selected shapes of a subject for HGM computation. Selecting 0 means using the covariates at the first shapes of all subjects.
     - Model Degree: Degree of population level covariate model, which is currently limited to linear geodesic model (degree = 1).


4. After confirming the options/parameters, user may click the `Compute HGM` button to fit geodesics models to the input shapes. Three types of geodesic models will be generated and can be visualized using the module's interactive sliders in the visualization section.
   - Mean trajectory: Geodesic polynomial model obtained by averaging fitted model parameters from all subjects. The mean time 0 point is the Fréchet mean of all subjects' time 0 points. The tangent vectors (model parameters) are the mean tangent vectors of all post-parallel-translation tangent vectors of all subjects. 
   - HGM: The hierarchical geodesic model as described in our [IPMI paper](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10323213/), which can show how shape changes with respect to both time variable and selected covariate(s).
   - Subject trajectory: Geodesic polynomial model representing the continuous trajectory of the shape change over time of individual subjects.


5. The successfully computed results can be exported to a user selected directory, with a metadata ".json" file, a reference polydata and all the geodesic model parameters.


6. Load existing model: the saved computation results can be reloaded to the module by selecting the exported ".json" file and clicking the `Load model` button. 


7. The results can also be exported for visualization in ShapePopulationViewer (SPV) using the 'SPV Visualization' subsection under 'Export'. Users may check `Mean`, `Subjects`, and/or `HGM` to export trajectories of interest to the selected output directory and visualize them in SPV.

A simple example dataset is can be downloaded [here](https://raw.githubusercontent.com/KitwareMedical/HierarchicalGeodesicModeling/refs/heads/main/HGM_example.zip), which is a sphere being deformed into an ellipse with sampled shapes at three time points without covariate information.

### Additional notes
- Shape modeling is performed in shape space (the pre-space of Kendall space). All input shapes' scale and rotation will be removed by partial procrustes alignment as part of preprocessing. And all the result shapes have size "1" (squared sum of point coordinates).
- The input shapes should be correspondence established both within and across individual subjects. All `.vtk` files should have the same number of points, and each point corresponds to the point of the same index in the other input shapes. Correspondence between meshes of different topologies can be established by using existing mesh-to-mesh registration tools in [SlicerSALT](https://salt.slicer.org/).
- `subject index` in the input `demographics(.csv)` file should start from 0.
- The time variables can be different across individual subjects (e.g. T_subject0 = [0, 1, 2] and T_subject1 = [0.1, 1.2, 2.3]).
But each input subject should contain the same number of time points as the others.

## Acknowledgement
 This work is supported by NIH NIBIB awards R01EB021391 (PI. Beatriz Paniagua, Shape Analysis Toolbox for Medical Image Computing Projects)
