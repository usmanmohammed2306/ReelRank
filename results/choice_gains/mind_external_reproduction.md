# Published MIND baselines and matched-history comparisons

This study reranks every article shown in every labelled impression of MIND-small validation and MIND-large validation. The original labels supply the clicked positions. It also repeats the comparisons on 259,161 large impressions from accounts that passed cross-release overlap screening. All three evaluations concern the same logged day. A hit counts one impression when its highest-ranked article was clicked. It cannot measure clicks caused by changing a ranking. [Microsoft MIND release](https://msnews.github.io/) · [MIND paper](https://aclanthology.org/2020.acl-main.331/)

## Which models are compared

The neural baselines use the NewsReX authors' released PyTorch GloVe weights and model classes for three published architectures. NRMS uses self-attention on headlines and browsing history. NAML also uses abstracts and topical metadata. LSTUR combines the recent reading sequence with a long-term account embedding. The checkpoint publisher is the NewsReX research team. Each of the three architecture papers predates these specific checkpoints. [NRMS](https://aclanthology.org/D19-1671/) · [NAML](https://www.ijcai.org/proceedings/2019/536) · [LSTUR](https://aclanthology.org/P19-1033/) · [NewsReX](https://arxiv.org/abs/2508.21572)

Each checkpoint comes from the MIND-small random-split release. NewsReX reserved 5% of the original 156,965 training impressions for internal validation, sampled negatives during neural training and called Microsoft's public small validation file its test set. The score replay presents each entire logged candidate list. The large run uses these small-trained weights on the public large validation release. LSTUR receives the actual small-release user ID on small validation. On large validation, it receives an all-zero long-term state because the releases provide no verified identity crosswalk. The released GRU still uses the full supplied sequence up to its 50-click history cap. [NRMS checkpoints](https://huggingface.co/newsrex/NRMS-PYTORCH-MIND-small-glove-random/tree/0f7f34ba5b6c0266304751fe54f7901085ca186c) · [NAML checkpoints](https://huggingface.co/newsrex/NAML-PYTORCH-MIND-small-glove-random/tree/a44ff655ebb58a5f6d956163f3c9be66eabc5169) · [LSTUR checkpoints](https://huggingface.co/newsrex/LSTUR-PYTORCH-MIND-small-glove-random/tree/44c54c7dc29a6bc6846015f614893e7f8a03ff04)

The nine-checkpoint baseline averages the within-list rank percentiles of the individual NRMS, NAML and LSTUR checkpoints. The delayed-feedback variant uses MIND-small impressions from earlier that day. Its logistic model chooses between the first headline from the frozen ranker and the first headline from the feedback-adjusted ranker. Static configurations use only pre-validation feedback.

The replay reconstructs the publisher's 23,266-entry vocabulary from the released small training and validation article text. Thresholding the vocabulary uses article metadata. Submitted scores break neural ties using a fixed hash of impression ID and article ID. NewsReX shuffles its evaluation candidates, so the reference and replay can differ slightly on tied cases. The project used the small validation release during method development. The large file serves as a transfer evaluation on the same date, and the screened accounts address detectable overlap.

The frozen configuration uses all supplied history in its six-input component. On the large file, 75,014 of 376,471 impressions contain more than 50 history articles. NoStream50 refits and evaluates that component with the latest 50 history clicks. Matched50 additionally excludes the exact 7,848 impressions reserved by NewsReX from both local model fitting and aggregate article statistics. That leaves 149,117 local fitting impressions and aligns the serving history cap, the source of the training impressions and the availability of validation-day feedback. Model architecture, use of full training slates versus sampled negatives and use of the author's internal validation for neural early stopping still differ. The frozen mixing weights are identical across the three static controls. [Publisher split implementation](https://github.com/igor17400/NewsReX/blob/6f10b83a613be934a679aeae1cef595c39b636c0/src/core/data/processing/interactions/behaviors.py#L655)

## Observed results

| Cohort | Logged lists | Nine-checkpoint ensemble first hits | Matched50 first hits | Switch first hits | Ensemble nDCG@10 | Matched50 nDCG@10 | Switch nDCG@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Complete small validation | 73,152 | 14,562 | 15,020 | 15,176 | 0.42264 | 0.42649 | 0.43271 |
| Complete large validation | 376,471 | 75,245 | 77,348 | 78,190 | 0.42257 | 0.42685 | 0.43227 |
| Screened large accounts | 259,161 | 51,555 | 52,952 | 53,512 | 0.42017 | 0.42451 | 0.42969 |

With the same training impressions and 50-click history cap, Matched50 put a recorded click first on 2,103 more large lists than the nine-checkpoint ensemble. The paired difference was 0.559 percentage points, with a 95% interval of 0.462 to 0.655 points. Its nDCG@10 increased by 0.00428, with an interval of 0.00379 to 0.00476. With delayed same-day feedback available, Switch gained 2,945 first hits over the same ensemble, or 0.782 points with an interval of 0.677 to 0.888. The large cohort with overlap screening had gains of 1,397 lists for Matched50 and 1,957 for Switch. All nine checkpoints reproduced the NewsReX small-validation AUC, MRR and nDCG values to within 0.000198. [Reported rankings, individual seeds, controls and intervals](mind_external_baselines.json)

## Reproduction

Use a temporary directory for data and individual predictions. Run numerical scoring with Python 3.10, NumPy 1.23.5, SciPy 1.9.3, scikit-learn 1.3.2, pandas and Numba 0.56.3. For neural inference, use Python 3.12 with the dependencies from the pinned NewsReX revision, including PyTorch and safetensors. Set the two Python paths before running the commands.

```sh
mind="$(mktemp -d)"
small_python=/path/to/mind-3.10/bin/python
neural_python=/path/to/newsrex-3.12/bin/python
archive=https://huggingface.co/datasets/Recommenders/MIND/resolve/2115068f457b577f93a268fffe22373e45727d89
revision=6f10b83a613be934a679aeae1cef595c39b636c0
export OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 WANDB_MODE=disabled
for data in MINDsmall_train MINDsmall_dev MINDlarge_dev
do
  curl -fLsS "$archive/$data.zip" -o "$mind/$data.zip"
done
curl -fLsS https://raw.githubusercontent.com/msnews/MIND/47fb9852c97814d80ccf9c658d6b81f5c930b510/evaluate.py -o "$mind/evaluate.py"
curl -fLsS "https://codeload.github.com/igor17400/NewsReX/tar.gz/$revision" -o "$mind/newsrex.tar.gz"
tar -xzf "$mind/newsrex.tar.gz" -C "$mind"
"$small_python" scripts/choice_external_transfer_data.py --data "$mind" --cohort "$mind/large_cohort.zip" --audit "$mind/overlap-audit.json"

naml=https://huggingface.co/newsrex/NAML-PYTORCH-MIND-small-glove-random/resolve/a44ff655ebb58a5f6d956163f3c9be66eabc5169
for seed in 42 123 456
do
  curl -fLsS "$naml/seed_$seed/model.safetensors" -o "$mind/NAML_${seed}_model.safetensors"
  curl -fLsS "$naml/seed_$seed/test_results.json" -o "$mind/NAML_${seed}_test_results.json"
done
"$neural_python" scripts/choice_neural_gain_naml.py --data "$mind" --upstream "$mind/NewsReX-$revision"
"$neural_python" scripts/choice_neural_gain_naml.py --data "$mind" --upstream "$mind/NewsReX-$revision" --cache-convolutions

nrms=https://huggingface.co/newsrex/NRMS-PYTORCH-MIND-small-glove-random/resolve/0f7f34ba5b6c0266304751fe54f7901085ca186c
lstur=https://huggingface.co/newsrex/LSTUR-PYTORCH-MIND-small-glove-random/resolve/44c54c7dc29a6bc6846015f614893e7f8a03ff04
curl -fLsS "$nrms/seed_123/model.safetensors" -o "$mind/nrms.safetensors"
curl -fLsS "$nrms/seed_123/test_results.json" -o "$mind/nrms_test.json"
"$neural_python" scripts/choice_external_nrms.py --data "$mind" --upstream "$mind/NewsReX-$revision" --weights "$mind/nrms.safetensors" --scores "$mind/nrms_scores.npy"
"$neural_python" scripts/choice_external_nrms.py --data "$mind" --upstream "$mind/NewsReX-$revision" --weights "$mind/nrms.safetensors" --scores "$mind/gain_full_nrms.npy" --cohort "$mind/gain_full_large.zip"
for seed in 42 456
do
  prefix="$nrms/seed_$seed"
  curl -fLsS "$prefix/model.safetensors" -o "$mind/NRMS_${seed}_model.safetensors"
  curl -fLsS "$prefix/test_results.json" -o "$mind/NRMS_${seed}_test_results.json"
  "$neural_python" scripts/choice_external_nrms.py --data "$mind" --upstream "$mind/NewsReX-$revision" --seed "$seed" --weights "$mind/NRMS_${seed}_model.safetensors" --scores "$mind/gain_small_nrms_$seed.npy"
  "$neural_python" scripts/choice_external_nrms.py --data "$mind" --upstream "$mind/NewsReX-$revision" --seed "$seed" --weights "$mind/NRMS_${seed}_model.safetensors" --scores "$mind/gain_full_nrms_$seed.npy" --cohort "$mind/gain_full_large.zip"
done
for seed in 42 123 456
do
  prefix="$lstur/seed_$seed"
  curl -fLsS "$prefix/model.safetensors" -o "$mind/LSTUR_${seed}_model.safetensors"
  curl -fLsS "$prefix/training_run_summary.json" -o "$mind/LSTUR_${seed}_summary.json"
  "$neural_python" scripts/choice_external_lstur.py --data "$mind" --upstream "$mind/NewsReX-$revision" --seed "$seed"
done

"$small_python" scripts/choice_external_transfer_score.py --data "$mind" --cohort "$mind/MINDsmall_dev.zip" --private-ranks "$mind/full_ranks.npz"
"$small_python" scripts/choice_external_transfer_score.py --data "$mind" --cohort "$mind/gain_full_large.zip" --private-ranks "$mind/gain_full_local.npz"
"$small_python" scripts/choice_external_history50.py --data "$mind"
"$small_python" scripts/choice_external_history50.py --data "$mind" --publisher-split
"$small_python" scripts/choice_neural_gain_stream.py --data "$mind"
cp results/choice_gains/neural_first_coefficients.json "$mind/"
for part in small full
do
  "$small_python" scripts/choice_neural_gain_first.py --data "$mind" --part "$part" --out "$mind" --use-fitted
done
result="$mind/mind_external_baselines.json"
"$small_python" scripts/choice_external_benchmark.py --data "$mind" --lstur --history50 --save-ranks --out "$result"
"$small_python" scripts/choice_external_metric_audit.py --data "$mind" --results "$result" --output "$result"
"$small_python" scripts/choice_external_benchmark_provenance.py --data "$mind" --results "$result" --full-scorer NAML-456
```

The first scoring command for the delayed-feedback model can also refit the time-ordered coefficients on MIND-small by omitting `--use-fitted`. The MIND-large step uses coefficients prepared on the small release.

## What the checks establish

The benchmark reads clicked flags from the original publisher archives. It compares each ordered large candidate's article URL with the remapped article at that position and checks impression IDs, account IDs, timestamps at hour resolution, list boundaries and click labels. For every ranked variant it checks that each list's ranks are a permutation and counts first-position hits directly.

The four official measures are impression-averaged AUC, MRR, nDCG@5 and nDCG@10. MIND's MRR averages reciprocal ranks over **all** clicked articles in an impression. For every model, the unmodified Microsoft scoring function grades the same 512 pseudorandom lists from each complete validation file. A second check evaluates all lists using the original four metric functions. It caches cases that yield identical metric values. For a strict ranking, AUC depends on list length, positive count and positive-rank sum. MRR depends on the positive ranks. nDCG at a cutoff depends on clicked positions above that cutoff and the capped ideal positive count. The program rejects disagreements above 1e-10. [Microsoft evaluator at the verified revision](https://github.com/msnews/MIND/blob/47fb9852c97814d80ccf9c658d6b81f5c930b510/evaluate.py)

The November 23, 2026 run reproduced all four original metric function means for the reported rankings on all three cohorts, with a maximum absolute difference of zero at the stored precision. A separate replay fed NAML seed 456's predictions and the original public click labels through Microsoft's complete, unmodified evaluator. Its largest discrepancy on either full validation file was below 2.26 × 10⁻¹³. Reapplying both history-capped controls with the six-signal implementation reproduced the four small and large rank files byte for byte.

The JSON contains one row per method and cohort, exact first-hit counts, original-metric cross-checks, per-seed comparisons to the publisher's small-validation scores and differences paired by impression with 95% intervals clustered by account. These intervals condition on the fitted weights and this service day. Comparing the largest observed seed or ensemble is descriptive. The intervals are pointwise and do not account for selecting that row after evaluation. [Machine-readable results](mind_external_baselines.json)
