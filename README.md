# Unsupervised Green Object Tracking

This repo contains implementation and demos of the green object tracker (GOT).
While the DCF-based global correlator used in the paper is written in Matlab,
the full version of the demo will require related installation.
To facilitate custom usage of other DCF trackers as the global correlator,
a demo using KCF is included as the reference.


# Installation via git and conda

1. Clone the GIT repository

2. Run the following command to create virtual environment according to the environment.yml.  
   Reference to [here](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file).
```
conda env create -f environment.yml
```

3. Add the created venv named got to jupyter notebook
```
conda activate got
python -m ipykernel install --name got
```

4. Run the baseline demo (demo_kcf) to verify that all python libs are good

5. To run full version of GOT, download Matlab2021a. The Signal Processing Toolbox is needed.

6. Install MATLAB Engine API for Python according to the instruction [here](https://www.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html).

7. Setup the corresponding python interpreter in matlab.

    Instruction refers to [here](https://www.mathworks.com/help/matlab/matlab_external/install-supported-python-implementation.html).

8. Run the demo (demo_gusot).

   Functions on how to run the tracker and examples are included in the demo.

The steps above have been verified on Windows 10 and Ubuntu 16.04.

You may stop at step 4 if you cannot or do not want to install matlab on your machine. While the KCF baseline only provides limited performance, you can substitute it with any DCF tracker, with minor modification to the code.

