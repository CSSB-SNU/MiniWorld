# Antibody-antigen overfitting analysis (v1.2.0 / v2.4.0 phase-1 distogram models), 2026-11-04/05

One-off probes that were run as 10-minute, 1-GPU Slurm jobs on node02 (training data on the local NVMe `/local_data1`).
Outputs went to `~/MiniWorld-scratch/fb-v240/<name>/` (not versioned); the scripts hardcode those paths and the checkpoint paths below.

Common set-up of the `eval_*.py` scripts: the dataset is built from `phase1a_distogram_medium_v240_local` (the catalog is memory-mapped, never rebuilt), `antibody_protein` rows are drawn with the
sampler's weights (rows cache `trainabag_rows.npy`), every item is preprocessed with `Preprocessor.process` with the crop budget raised to 2400 tokens (whole assemblies),
the metric is FoldBench's contact AP (rep-atom distance < 8 A, score = P(bin <= 23)) over the (antibody chain, antigen chain) pair of the item.
Checkpoints: v1.2.0 `runs/v1.2.0/phase1a/medium_launch_20260917_130505/.../epoch=0100..0400.pt` and `runs/v1.2.0/phase1b/main_20260930/.../epoch=0900.pt` (model config `phase1a_distogram_medium_v120`,
EMA weights); v2.4.0 `runs/v2.4.0/phase1a/medium_distogram_L384_scratch/2026-11-01/234255_5040bd46/checkpoints/epoch=0100..0600.pt`.
`name=path@model_config` on the command line builds that model from its own config while the data always comes from `--config`.

## Scripts

| script | what it does |
|---|---|
| `fb_epoch_sweep.py`, `merge_fb_sweep.py` | FoldBench distogram AP per interface class for a list of checkpoints; `merge_fb_sweep.py` pools the per-target npz into AP / F1 / AUROC |
| `eval_train_abag.py` | AP on antibody-antigen items drawn from the TRAINING database (whole assemblies), several checkpoints on the same batch |
| `iface_diag.py` | where the top-k predicted pairs fall (true distance buckets, index miss) from the FoldBench npz |
| `why_low.py` | one FoldBench target: AP at 8 / 10 / 12 / 16 A, rank of the true contacts, figures |
| `viz_dist50.py` | true vs predicted L x L distograms (data for the artifact page) |
| `crop_abag.py`, `crop_viz.py` | how the training crop (`Preprocessor.get_crop_indices`, the function the dataloader calls) cuts antibody-antigen items; 3-D data for the viewer |
| `tpl_diag.py` | FoldBench template slots filled / empty per target and why (key missing, zero templates) |
| `probe_2yze.py` | preprocess every training item of one PDB id (the `disordered_pdb` copy of 2yze has an empty coordinate field) |
| `eval_ablate_msa.py` | antibody MSA removed |
| `eval_cdr_mask.py`, `eval_cdr_alanine.py`, `eval_h3_unk.py` | all CDRs / CDR-H3 set to UNK, or all CDRs to ALA (sequence, MSA columns, side-chain atoms, template) |
| `eval_epitope_unk.py` | epitope residues of the antigen set to UNK |
| `eval_pairdep.py` | cross tests: CDR/epitope marginals, CDR swapped for another antibody's CDRs, epitope identities swapped |
| `eval_fwswap.py` | another antibody's framework, original CDRs kept (with the processing control `fw_self`) |
| `eval_msagap.py`, `eval_msaswap.py`, `eval_msainfo.py` | MSA at the CDR / epitope columns: gap, another antibody's MSA swapped in, and the conservation statistics of the MSA the model receives |
| `eval_h3mut.py` | k random point mutations in CDR-H3 on top of a donor CDR MSA |
| `eval_posscan.py` | one substitution at a time over CDR1-3, the epitope and 6 control antigen residues (radical vs conservative) |
| `example_10min_gpu_job.sbatch` | the job header used for all of them: `--time=00:10:00`, 1 GPU, `--nodelist=node02` |

## What was found (v1.2.0 unless noted; small samples, see the caveats in each script's summary)

- FoldBench distogram AP (EMA, corrected nucleic inputs, templates on). v2.4.0 ep100/200/300/400: protein-RNA 0.143/0.139/0.142/0.171, protein-DNA 0.346/0.357/0.359/0.369, antibody-antigen (113 targets) 0.049/0.052/0.053/0.045.
- Antibody-antigen AP, training items (158, whole assemblies) vs FoldBench (112/113 targets), ep100/200/300/400 (+900 for v1.2.0):
  v1.2.0 train 0.273/0.466/0.550/0.587 (ep900 0.845), FoldBench 0.079/0.076/0.067/0.074 (ep900 0.054);
  v2.4.0 train 0.207/0.355/0.405/0.524, FoldBench 0.049/0.052/0.053/0.045. The gap opens from ep100 in both models and grows with the epoch.
- The FoldBench top-k predictions on antibody-antigen interfaces are far from the true contacts: 52-59% of the top-k pairs are >= 20 A apart, median index miss 10-16 residues (protein-DNA: 1).
- What the training-item score depends on (ep900, train-DB antibody-antigen): the identities of the CDR residues and of the epitope residues (CDRs to UNK -81%, to ALA -85..96%, CDR-H3 only -30%, epitope to UNK -79%;
  same number of control residues -4..15%), not the framework (another antibody's framework + original CDRs: 0.828 -> 0.702 vs 0.710 for the processing control), not the templates.
  Gapping the MSA at the CDR columns costs 47%, but swapping in ANOTHER antibody's CDR-column MSA costs 6%: the gap result is an out-of-distribution effect, not information in the MSA
  (the CDR columns of the MSA are close to random: 10-20% identity to the query, no row shares all CDRs with it).
- Random mutations in CDR-H3 decay smoothly (about 1.5% per mutated residue, 30% only when most of H3 is changed); single substitutions are mostly harmless: 5.7% of the scanned positions drop the AP by more than 0.05, the 5 most sensitive positions of an item carry 55% of its total drop,
  75% of the sensitive positions are contacting positions, and about 40% of them also react to a conservative substitution.
- Open: whether this is memorisation of training complexes (needs the same tests on unseen antibodies, the similarity-stratified curve, and a data-side intervention).

## Operational notes found on the way

- `disordered_pdb` rows of 2yze have an empty coordinate field (`could not convert string to float: ''`); `BioMolData.__getitem__` draws another item (27 times in the 600-epoch phase-1a run, together with 97 draws for the 66 long-distillation records missing from their LMDB).
- The phase-2 graph trainer must capture with `capture_error_mode="thread_local"` (the pin-memory thread invalidates a global capture with 32 workers on 4 GPUs).
