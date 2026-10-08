# Experimental configuration mapping

All tags below map to existing study results in results/summary.csv. Exact recorded arguments and seeds are retained in configs/run_arguments.json.

| Tag | Model / training variant | Seeds |
|---|---|---:|
| voltkd | Complete VOLT-KD: mV, superclass + subclass labels, cross-fitted teacher | 5 |
| abl_arch_z | VOLT-KD superclass-only student, z-score input | 5 |
| abl_plain | VOLT-KD superclass-only student, mV input | 5 |
| abl_sub | VOLT-KD superclass + subclass student, mV, no distillation | 5 |
| abl_zscore | Complete supervision, z-score input | 5 |
| abl_nosub | No subclass hard-label or subclass distillation losses | 5 |
| abl_insample | In-sample eight-fold teacher | 5 |
| abl_noisyor | Noisy-OR classification head | 3 |
| abl_nornn | No recurrent layer | 3 |
| abl_half | Reduced width: c1=32, c2=64, hid=40 | 3 |
| lead_limb6 | Six limb leads, unused channels zeroed during training and evaluation | 3 |
| lead_ii | Lead II only, unused channels zeroed during training and evaluation | 3 |
| ref_msca_mamba | Local MSCA-Mamba, per-lead z-score | 5 |
| ref_msca_mamba_mv | Local MSCA-Mamba, millivolt input | 5 |

Five-seed configurations use 42, 123, 456, 789, and 2025; three-seed variants use 42, 123, and 456. The label-mapping files are generated from PTB-XL rather than supplied as patient metadata.

Reduced-lead training is distinct from randomly zeroing channels in a fixed twelve-lead model during robustness evaluation. Published comparison values are stored separately from local measurements.

The complete student has 112,364 trainable parameters, including its auxiliary subclass head. Its 107,189 deployed parameters exclude that head. Teacher parameters are not student deployment parameters.
