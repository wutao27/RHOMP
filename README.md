# Retrospective Higher-Order Markov Processes for User Trails (RHOMP)

#### Tao Wu
#### David F. Gleich
------
### Folders and Files
* `data` contains the sequence datasets used in the paper
* `demo.ipynb` contains IJulia notebook codes for running the experiments.
* `initialization.jl` contains functions for initializations of RHOMP models and MC models
* `solver_second.jl` contains functions for solving a second-order RHOMP
* `solver_high.jl` contains functions for solving higher-order (order > 3) models
* `evaluation_second.jl` are implementations for computing MRR and Precision for second-order methods
* `evaluation_high.jl` are implementations for computing MRR and Precision for high-order (order > 3) methods
* `kneser_ney.jl` contains the implementations for Kneser Ney smoothing methods
* `ritf.jl` contains the implementations for RITF
* `LME.jl` are codes that take the output from LME software and evaluation the MRR and Precisions
* `util.jl` utility functions

### Usage
Please refer to the demos.