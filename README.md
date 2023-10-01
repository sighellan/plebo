# Prior Learning for Bayesian Optimisation (PLeBO)

This repository hosts the code used for *Data-driven Prior Learning for Bayesian Optimisation*. 

`requirements.txt` has the package requirements for the optimisation step.

The core files for the optimisation are `optimisation.py` and `do_BO.py`. They also use the following source files
* `synthetic.py`
* `jam.py`
* `hbo.py`

The preprocessing for PLeBO happens in the `mcmc` folder. 
* `requirement-numpyro.txt` has the requirements for this step (which are different to those for the optimisation step). The preprocessing files are stored in the `candidates` folder.
* `prior_learning.py` has source files.
* `learn_candidates_numpyro.py` is the script for learning candidate hyperparameters, as well as the Gamma prior.

Some of the baselines have preprocessing steps.
* `learn_supermodel.py` is used to learn the set of Shared GP hyperparameters.
* `learn_initial_points.py` and `learn_initial_points_satellite.py` is used to learn initial points for Initial.

The experiments were run on a cluster, using 
* `write_exps.sh`
* `cluster_script.sh`
* `cluster_wrapper.sh`


The figures are stored in the `Figures` folder.
To plot and print the results reported, use
* `plot_cluster_grouped_results.py`
* `plot_example_tasks.py`
* `plot_motivation_together.py`
* `plot_prior_fit.py`
* `plot_results_summary.py`
* `print_durations.py`

The synthetic optimisation tasks were generated using `generate_synth_problem_set.py`. These are stored in the `problems` folder.
