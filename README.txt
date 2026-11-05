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

REPEAT OVER THE FIRST TEN OFFICIAL STATION GROUPS
-----------------------------------------------
    nohup python -u run_groups.py --device cpu > weather_10groups.log 2>&1 &
    tail -f weather_10groups.log

The runner uses supplied group indices 0-9, ten fits per method per group, and
50 final scenarios. It reuses the completed runs/temperature10_group0 fits and
the existing prepared group-0 data. It downloads the verified 1.36 GB source
once to prepare groups 1-9; allow time for this download. No GPU is required.
No hyperparameters change. Per-fit logs are in each seed's runner.log.

Partial and final equal-weight group means, per-group scores and paired
Riesz-minus-energy differences are written to:
    runs/temperature10_groups0_9/summary.json

Rerun the same command to resume. Existing manifests/evaluations are checked
against the selected group, config, seeds and prepared-data hash before reuse.
Ten groups are still a subset of the paper's 100; neighbouring groups may overlap.
The descriptive spread across groups is not an independent-sample significance
test. For a quick cached check: python run_groups.py --groups 0 --workdir runs/group0_reuse_check

VALIDATION-ONLY TEMPERATURE SWEEP
--------------------------------
    nohup python -u tune_temperature.py --device cpu > weather_temperature_sweep.log 2>&1 &
    tail -f weather_temperature_sweep.log

Default grid multiplies the original [.2,.05,.02] by 1, .25, 4 and 16.
Each candidate still sums THREE RMS-normalized fields. Testing one field
against three would also change the aggregate update scale; we avoid that
confound. Learning rate, architecture, particle count and other settings stay
fixed. We fit seeds 42-44 on station groups 0-2: 36 trial cells. The nine
original-setting fits are reused from the completed group runs where present,
leaving 27 new fits. Missing group data is prepared from the same source file.

Selection minimizes the equal-group average of best per-seed 2015 validation
energy scores (the training-form U-denominator score used for early stopping).
The same validation set therefore selects both checkpoints and temperatures;
there is no separate nested validation split or claim of unbiased validation
performance. This is an inexpensive exploratory search over three groups.
The runner never calls evaluate.py or reads test metrics. The 2016 test results
have already been inspected in the original comparison, so they are not a
previously untouched holdout. Do not choose a candidate by evaluating all of
them on 2016. Freeze the validation choice before the next test comparison.

Progress and final ranked selection are saved to:
    runs/temperature_validation_sweep/progress.json
    runs/temperature_validation_sweep/selection.json
    runs/temperature_validation_sweep/selected_config.yaml

Rerunning resumes incomplete fits and checks the prior grid, config, data hashes
and model provenance before reuse. An existing final selection cannot silently
change. Finishing this command does NOT launch test evaluation; review the
validation ranking first. No energy-baseline hyperparameters are tuned by it.

TEST THE FROZEN VALIDATION-SELECTION
-----------------------------------
An optional modest extension ABOVE scale 16 can reuse the prior sweep:

    nohup python -u tune_temperature.py --device cpu --multipliers 16 24 32 \
      --reuse-sweep runs/temperature_validation_sweep \
      --workdir runs/temperature_validation_higher > weather_temperature_higher.log 2>&1 &

This compares [.2,.05,.02] times 16, 24 and 32, i.e. [3.2,.8,.32],
[4.8,1.2,.48], and [6.4,1.6,.64]. Default groups/seeds stay 0-2 / 42-44.
The nine scale-16 fits are reused after config, data and checkpoint-hash checks;
only the 18 scale-24/32 fits are new. The original sweep/selection stays intact.
The new selection is in runs/temperature_validation_higher/selection.json.
This is another exploratory validation search; it does not run test evaluation.

After the ORIGINAL sweep finishes, keep its selection.json unchanged and run:

    nohup python -u evaluate_selected.py --device cpu > weather_selected_test.log 2>&1 &
    tail -f weather_selected_test.log

The default above reads the ORIGINAL selection. To evaluate a later higher-grid
selection, explicitly use --selection runs/temperature_validation_higher/selection.json
and a NEW --workdir runs/temperature_higher_selected_test. Review/freeze the new
validation choice first; the higher-grid tuning command does not run this test.

This recomputes the winner from the complete saved validation ranking, freezes
the selection file/hash in a new output directory, and evaluates only that
selected setting. Original energy and untuned-Riesz scores/fits are reused from
the completed group-0 and ten-group runs, with exact config/data checks. The
selected sweep fits (normally nine) are reused with checkpoint SHA256 checks;
only missing seeds/groups at the selected temperatures are trained. Ten models
times five samples give 50 test scenarios per group. No other hyperparameters
change and no test score selects a temperature.

Results, including the original Riesz scores for context, accumulate in:
    runs/temperature_selected_test/summary.json

Rerun the same command to resume. Changing the selection/config requires a new
workdir; it cannot overwrite the frozen run. Originals are never overwritten.
This remains an exploratory follow-up on the previously inspected 2016 test
year, not a new untouched holdout. Do not keep choosing temperatures from test
results. The default test scope is the same ten station groups (0-9).

VALIDATION-ONLY LEARNING-RATE SEARCH FOR BOTH METHODS
----------------------------------------------------
After freezing the extended temperature choice, run:

    nohup python -u tune_learning_rate.py --device cpu > weather_learning_rate.log 2>&1 &
    tail -f weather_learning_rate.log

By default this reads runs/temperature_validation_extended/selection.json and
holds its winning Riesz temperatures fixed (scale 512 in the current experiment).
It tries learning rates .0005, .001 and .002 for BOTH energy and Riesz, using
the same groups 0-2, seeds 42-44, patience, maximum epochs and validation score.
There are 54 trial cells; the nine original energy fits and nine selected Riesz
fits at LR .001 are reused where available, leaving 36 new fits. The algorithm,
architecture, particle count, temperature normalization and Adam settings stay
unchanged. The original fits are not resumed/modified by this search.

Each method selects its own LR from equal-group mean 2015 validation energy.
Both methods get the same LR search budget, but Riesz has had ADDITIONAL prior
temperature tuning; this is not equal total historical tuning effort. As before,
validation also selects checkpoints, and 2016 has already been inspected in
earlier experiments. No test metrics/predictions are used by this LR runner.

The ranking, checkpoint hashes and frozen choices are saved to:
    runs/temperature_learning_rate_sweep/lr_selection.json
    runs/temperature_learning_rate_sweep/selected_energy.yaml
    runs/temperature_learning_rate_sweep/selected_riesz_softmax.yaml

Rerun the same command to resume. Different grids or temperature selections need
a new workdir. Read the validation ranking before planning test evaluation.
The LR selection artifact is a distinct format from temperature selection.json;
evaluate_selected.py accepts temperature selections only, not lr_selection.json.
Use the runner below after freezing the independent LR choices.

TEST THE FROZEN LEARNING-RATE CHOICES
------------------------------------
    nohup python -u evaluate_learning_rate.py --device cpu > weather_lr_selected_test.log 2>&1 &
    tail -f weather_lr_selected_test.log

Defaults read runs/temperature_learning_rate_sweep/lr_selection.json and verify
its full validation ranking plus the underlying frozen temperature choice.
The winning LR is applied separately for energy and Riesz; neither choice is
made using test scores. Each method is evaluated with ten seeds x five samples
on groups 0-9. The 18 selected LR-sweep fits are reused where available, and only
missing seeds/groups are trained. If the original LR wins, compatible completed
fits are reused for all seeds. All reuses check config, data and provenance.

Context columns come from the original energy fits and the fixed-temperature
Riesz results BEFORE LR tuning, i.e. the earlier scale-512 run in the current
experiment. The latter defaults to runs/temperature_extended_selected_test;
override --temperature-test-runs if those results live elsewhere. Original
fits, validation selections and test results are not overwritten.

New results accumulate in:
    runs/temperature_learning_rate_selected_test/summary.json

Rerun to resume with identical frozen selections. A changed LR/temperature
selection or group list requires a new workdir. Checkpoint SHA256 identities
and evaluation records are validated before reuse. The comparison is still
an exploratory follow-up on the previously inspected 2016 test year; the two
methods have matched LR grids but unequal total historical tuning effort.

EXACT UNIFORM-WEIGHT ABLATION
----------------------------
After the tuned-LR ten-group comparison has finished:

    nohup python -u compare_uniform.py --device cpu > weather_uniform.log 2>&1 &
    tail -f weather_uniform.log

This reads the frozen LR selection and reuses the energy and Riesz results in
runs/temperature_learning_rate_selected_test. It trains only the uniform-weight
Riesz variant: same architecture, seeds, data, 50 training particles, learning
rate, validation checkpoint selection and normalization as the chosen softmax
run. In the current experiment the LR is .0005 and the three temperatures are
[102.4,25.6,10.24]. There is no additional uniform-specific hyperparameter search.

Each generated particle's repulsion weights become exactly 1/(M-1) across all
OTHER generated particles; the diagonal remains excluded. The sole observed
positive already has weight 1. Unit direction vectors, detached global distance
scale, coordinate floor, three 1/tau prefactors, per-field RMS normalization and
its numerical floor, field summation and stopped-target regression are retained.
The three uniform fields normally share a direction, but all three contributions
are retained so this does not reduce the summed update by a factor of three.
With uniform weights tau no longer controls neighbour concentration; its stored
prefactors remain only to hold the rest of the arithmetic/flooring fixed.

The upstream weather/riesz_topk_softmax.py file is unchanged. The uniform
implementation is separate, and tests compare its losses/gradients with the
original implementation when only finite softmax logits are replaced by zero.
The training method is explicitly recorded as riesz_uniform in manifests.

Ten seeds per group give 50 final scenarios, as before. Reference fits/results
are validated and read without modification. Only new uniform fits are trained
(100 across the default ten groups); output lives separately at:
    runs/temperature_uniform_comparison/summary.json

Rerun the same command to resume. The table reports energy, riesz_softmax and
riesz_uniform scores, plus raw forecasts. Per-group uniform-minus-softmax score
differences are also stored. This is a controlled diagnostic at inherited
settings, on the previously inspected 2016 test year, not a new untouched test.
There is no automatic choice to replace softmax based on these test scores.

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
