WEATHER RIESZ: FIRST CONTROLLED TEMPERATURE EXPERIMENT
===================================================

Paper: Chen, Janke, Steinke & Lerch (2024), Generative machine learning
methods for multivariate ensemble post-processing, arXiv:2211.01345.
https://arxiv.org/pdf/2211.01345
Official code: https://github.com/jieyu97/mvpp
Pinned official revision: 091ac33b613b7d79b4686d11eae81b48a5728257

This is a NEW, separate experiment. No Hudson or existing image repository is
modified. The initial experiment is temperature, D=10, official station group 0.
It compares energy-score training with Sophia's existing Riesz-softmax stopped
field target. It does not implement diffusion, learn MAE features, or pool
unrelated dates as reference particles.

QUICK START ON YOUR GPU MACHINE
------------------------------
This small model also runs quickly on CPU; a GPU is optional for the first group.
Unzip weather_riesz_experiment.zip and enter weather_riesz/.
Use a Python 3.11 environment with a working CUDA PyTorch installation.
Existing PyTorch >=2.5 is supported; no TensorFlow is needed.

    conda activate dmmd
    python -m pip install -r requirements.txt
    python -m pytest -q

The bundle includes the small PREPARED public station-group-0 dataset in data/.
If it is absent, download and prepare the full source once (1.36 GB):

    python -u prepare_data.py --download

Two-epoch real-data checks in fresh workdirs:

    python -u train.py --method energy --device cuda --epochs 2 --workdir runs/check_energy
    python -u train.py --method riesz_softmax --device cuda --epochs 2 --workdir runs/check_riesz

Full paired comparison for this ONE station group (20 fits, 10 per method):

    nohup bash scripts/run_comparison.sh > weather_comparison.log 2>&1 &
    tail -f weather_comparison.log

Or on UCL SGE, activate your compatible environment and submit from this folder:

    qsub scripts/submit_sge.sh

The SGE template follows your aihub_ucl GPU job format and requests ONE H100;
it has not been submitted or tested on the cluster. It uses the activated
environment through -V. No fixed SAN paths are required. Outputs default to
runs/temperature10_group0/. Set WEATHER_RUNS or WEATHER_DATA if needed.
WEATHER_DEVICE=cpu bash scripts/run_comparison.sh is also supported.
Rerunning run_comparison.sh resumes latest checkpoints, preserving optimizer,
sample generator and shuffle state, and skips finished evaluation outputs.

READ RESULTS
------------
    cat runs/temperature10_group0/energy/evaluation/metrics.json
    cat runs/temperature10_group0/riesz_softmax/evaluation/metrics.json

Each file gives energy score, variogram score (p=.5), mean marginal CRPS, the
raw 50-member forecast scores, seed/checkpoint provenance, and sample counts.
Smaller scores are better. Predictions and per-case scores are also saved.
The paper's D=10 temperature averages (100 station groups) are ES 2.91 and
VS 16.9 for CGM. These are context, NOT pass/fail targets for this one group.
Comparisons of our two objectives use exactly the same group and output count.

WHAT MATCHES
------------
* Public temperature data: 48-hour ECMWF forecasts, 50 raw members.
* Supplied 10-station geographically neighbouring subset, first row of CSV.
* Only dates with all 10 unique stations; explicit station-coordinate alignment.
* Training 2007-2014, validation 2015, test 2016; no random date split.
* Official feature columns and train-only scalers. As in the released script,
  mean/scale modules receive raw temperature forecast mean/std; decoder receives
  standardized target forecasts and auxiliary features; observed targets remain
  in the original physical units. Sample std uses ddof=1; scaler std ddof=0.
* Generator: separate per-station linear mean, exp(linear(flat std inputs))
  multiplying 10 Gaussian noise coordinates, decoder Dense100 ELU -> Dense100
  ELU -> DenseD. Same input order and no sqrt applied to noise multiplier.
* Batch 64, Adam lr .001, beta .9/.999, epsilon 1e-7; no decay/warmup/clipping.
* 50 generated predictions per forecast case during training; at most 300 epochs,
  patience 10, best checkpoint selected by validation energy score for BOTH.
* Final 50-member evaluation = 10 independently trained models x 5 scenarios.

EXPLICIT DIFFERENCES / PITFALLS
-----------------------------
* PyTorch port, not bitwise TensorFlow 2.3 reproduction. Random initial weights,
  floating-point kernels and Adam implementation may differ. We use the same
  architecture and Glorot-style initialization; equivalence of trained results
  to the original TensorFlow implementation has NOT been established.
* Fixed, recorded seeds; validation noise is fixed across epochs and objectives
  to make selection less noisy (official code resamples). Both methods share
  initial parameters, training sample streams and permutations per seed.
* Original training loss uses M(M-1), with sqrt(epsilon) diagonal constants.
  Original evaluation ES uses M^2, with zero self distances. Both are preserved.
* Released script concatenates 50 scenarios from EACH of 10 models (500 total).
  Main-paper protocol is 5 from each, 50 total; our evaluation follows the PAPER.
* This first comparison covers ONE of the paper's 100 station groups. It must
  be labelled a pilot until repeated across groups. ES/VS/CRPS use the original
  physical units; no extra normalization of target coordinates is introduced.
* Learning rate is initially identical for a controlled test. Objective-specific
  tuning, if needed, must use validation data and equal tuning budgets, not test.
* No EMOS+ECC/DRN baseline fitting or paper skill-score replication yet; the raw
  ensemble is scored and the central comparison is energy vs Riesz-softmax.

RIESZ ADAPTATION
----------------
weather/riesz_topk_softmax.py is an UNMODIFIED copy of Sophia's local
riesz_flow_sophia/riesz_topk_softmax.py, with SHA256 recorded in provenance.json.
JIT compilation is disabled by default in the wrapper for these small tensors.
Input gen [B,50,10], positive [B,1,10]: each date is a condition and its one
observed 10-station vector is its only positive. Generated self-interactions
exclude the particle itself. There are no CFG/fixed negatives or memory banks.
Tau [.2,.05,.02], per-temperature RMS normalization, epsilon 1e-6 and direction
epsilon 1e-8 match the inspected ImageNet softmax recipe. The copied code uses
detached global distance/RMS statistics across a minibatch; these scalar scales
are shared, but particle neighbour/reference sets never mix forecast cases.
With one positive, positive softmax weight is always one; temperature still
changes generated repulsion. This is a conditional adaptation, NOT the original
image recipe with 64 generated and 128 independent real images per class.
The Riesz surrogate's numeric value is not an energy score. Validation/test
comparisons therefore use proper forecasting scores, not training-loss values.

DATA ATTRIBUTION
----------------
Chen, Jieyu; Lerch, Sebastian (2022), Full data for temperature post-processing.
https://doi.org/10.6084/m9.figshare.19453580
License CC BY 4.0: https://creativecommons.org/licenses/by/4.0/
Source file temperature_data_cgm_std.feather (1,360,331,336 bytes),
MD5 76b1615b7a9b3e774c5e8ee98ae39ccc. Prepared data is a transformed subset:
selected official station group, complete-case filtering and train-only scaling.
Full metadata, scaler values and station order are alongside the prepared NPZ.
Original official code retains its MIT license in third_party/mvpp/LICENSE.

TESTS
-----
    python -m pytest -q

Tests cover independent formulas/gradients, reference stop-gradient, particle
permutation, per-condition gradient grouping, explicit three-module forward,
and direct parity with the authors' ES and VS evaluation functions.

Optional full preprocessing parity audit (needs scikit-learn):

    python -m pip install scikit-learn
    python check_preprocessing.py --source data/temperature_data_cgm_std.feather

This executes the pinned authors' preprocessing statements, with only flattening
of scaler results for pandas 3 column-assignment compatibility. Saved audit
results are in validation/. The public source uses legacy Feather V1; PyArrow
is capped below 27 to retain its reader. Deprecation warnings are expected.
