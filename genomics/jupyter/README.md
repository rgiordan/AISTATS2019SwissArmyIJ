These notebooks provide examples and exposition of the analyses that underlie
the genomics experiments in our paper, "A Swiss Army Infinitesimal Jackknife".
The actual full experimental results are run by the scripts in the directory
``cluster_scripts``.


### Virtual environments and Jupyter.

If you are using a virtual environment for Python (as we recommend),
you will need to define a jupyter kernel for your virtual
environment.  With your virtual environment activated, run

```
python3 -m pip install ipykernel
python3 -m ipykernel install --user --name=aistats2019_ij_paper
```

Then use this kernel when executing the analysis scripts (including
the R script, which also uses Python through the ``reticulate``
package).

To delete this kernel when you are through, you can run
``jupyter kernelspec list`` and delete the
directory corresponding to the ``aistats2019_ij_paper`` kernel.


### Jupyter notebook extensions.

The python notebooks require Jupyter notebook extensions. See
[jupyter_contrib_nbextensions](https://github.com/ipython-contrib/jupyter_contrib_nbextensions)
for installation details.

After installing ``jupyter_contrib_nbextensions``, run

```
jupyter nbextension enable python-markdown/main
```

before opening the jupyter notebook.

You may also have to go to `File` at the top left of the notebook, and click
`Trust Notebook`.

### Running the notebooks

The three notebooks in this directory run an example analysis on a single degree
of freedom and left-out point.  They are intended for exposition; the full
analysis to produce the results in the paper is run using the scripts in
``cluster_scripts``.  The notebooks should be run in the following order:

1. Run ``fit_model_and_save.ipynb``.  This will perform the initial fit.
1. Run ``load_and_refit.ipynb``.  This will load the initial fit and
   perform exact CV.
1. Run ``calculate_prediction_errors.ipynb``.  This will load the initail fit,
   the exact CV results, peform the IJ and
   calculate the prediction error as well as other diagnostics.

For a detailed analysis of all the results in the paper, including more
extensive metrics measuring the accuracy of the IJ, see the R notebook
``R/examine_and_save_results.ipynb``.  This notebook does not use the results
of the three expository scripts.  Rather, it consumes the output of the
scripts in ``cluster_scripts``.

### Converting the notebooks to pdf

To convert the three notebooks to pdf form for inclusion in the appendix
of the paper, run the script convert_notebooks_to_pdf.sh.
