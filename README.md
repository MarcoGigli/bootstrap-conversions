# README

## Structure
The code is structured as follows:
- In the outer folder, the `*_simulation.py` files can be executed to run the simulations
- In the outer folder, `delays.yml` contains the required packages
- `/delay_mle`: contains the EM model, both linear and univariate.
- `/linear_delayed`: contains the strategies under scrutiny. de_lin_ts and de_lin_ucb are taken from Vernade et al - Linear bandits with Stochastic Delayed Feedback - 2020 and are here for comparison
- `/criteo_distribution`: contains the distribution of Criteo delays.

The original code (i.e., the core of the contribution) is the content of `/delay_mle` and of `/linear_delayed/de_bootstrap_lin_ts.py`.

`/linear_delayed/de_lin_ts.py` and `/linear_delayed/de_lin_ucb.py` are taken from Vernade et al - Linear bandits with Stochastic Delayed Feedback - 2020 and are here for comparison

For a fair comparison, also the Criteo distribution files are generated with the code accompanying Vernade et al.
