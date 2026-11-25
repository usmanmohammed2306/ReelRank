# Reproducing the ReelRank external benchmark study

The paper evaluates one practical decision. Given titles that a person has already been offered, how often can the ranker place a title they selected in the first position? A separate experiment asks the corresponding question about clicked news articles. All comparisons in the course materials use a public algorithm, released model weights, or the originally displayed order.

## Movie and book choices

The benchmark authors released the underlying choices, the original notebook, the fitted recommendation engines and the canonical evaluator. We ran their Task 3 archive on all 9,371 prefix cases. We then evaluated all seven rankings on every one of the 6,892 genuine ten-item recommendation rounds. These protocols produce different numerical scores. The archived notebook and its printed precision denominator are reproduced separately in the paper. The primary ranking result compares exactly the same ten items and raw selected flags for 349 held-out people on 5,510 rounds. A round with no selection remains in the denominator. The 25,000,095 GroupLens ratings supply catalog features. The labels come from participants in SM-RS. [Published benchmark and evaluator](https://github.com/pdokoupil/SM-RS) · [Author artifacts](https://osf.io/wsakx/) · [MovieLens 25M](https://grouplens.org/datasets/movielens/25m/)

Use Python 3.10 and install the versions in `requirements-choice-baselines.txt`. The public model checkpoints require the pinned LensKit, Numba and TensorFlow versions. Use a new private directory for data and predictions and keep it outside the research package.

```sh
python3.10 -m venv .venv-choice
. .venv-choice/bin/activate
python -m pip install -r requirements-choice-baselines.txt
python -m pip install --no-deps recommenders==1.2.1 lightgbm==4.6.0
export CHOICE_DATA="$(mktemp -d)"
export PYTHONPATH="$CHOICE_DATA/benchmark/src"
export OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMBA_NUM_THREADS=1
export TF_NUM_INTEROP_THREADS=1 TF_NUM_INTRAOP_THREADS=1
git clone https://github.com/pdokoupil/SM-RS.git "$CHOICE_DATA/benchmark"
git -C "$CHOICE_DATA/benchmark" checkout --detach 3dcc7d36facf026798696d7b891663ca40ecb511
python -m scripts.fetch_choice_osf "$CHOICE_DATA/data" --core
python -m scripts.fetch_choice_osf "$CHOICE_DATA/artifacts"
python -m scripts.choice_baselines "$CHOICE_DATA" --out "$CHOICE_DATA/publisher_results"
curl -fLsS https://files.grouplens.org/datasets/movielens/ml-25m.zip -o "$CHOICE_DATA/ml-25m.zip"
books=https://raw.githubusercontent.com/zygmuntz/goodbooks-10k/6dd165b555a7b47b2dd36743a425776e641ff50c
curl -fLsS "$books/books.csv" -o "$CHOICE_DATA/goodbooks-meta.csv"
curl -fLsS "$books/book_tags.csv" -o "$CHOICE_DATA/goodbooks-tags.csv"
out="$CHOICE_DATA/external"
private="$CHOICE_DATA/predictions"
mkdir -p "$out" "$private"
cp results/choice_gains/personal_protocol.json results/choice_gains/lambdamart_protocol.json "$out/"
python -m scripts.choice_personal_ranker --root "$CHOICE_DATA" --movie-zip "$CHOICE_DATA/ml-25m.zip" --out "$out"
python -m scripts.choice_frozen_refit --root "$CHOICE_DATA" --movie-zip "$CHOICE_DATA/ml-25m.zip" --private "$private" --lock "$out/personal_lock.json"
python -m scripts.choice_external_lambdamart --root "$CHOICE_DATA" --movie-zip "$CHOICE_DATA/ml-25m.zip" --private "$private" --out "$out"
python -m scripts.choice_released_benchmark --root "$CHOICE_DATA" --movie-zip "$CHOICE_DATA/ml-25m.zip" --private "$private" --out "$out/external_baseline_audit.json"
```

The frozen movie model needs the six checkpoint outputs as inputs and runs directly from those files. To replay the selected blend, supply the distributed `results/choice_gains/personal_lock.json` in the refit command. Separate runs of the standard VAE can produce different orders because its released inference graph samples a latent vector. We preserve the original reported run and an independently rerun score table. A second LambdaMART experiment repeated all twelve development settings and every held-out and calendar ranking exactly. Printed scores and both reproductions appear in `external_benchmark_verification.json`.

### Matched supervised baseline

The independent supervised comparator is LightGBM 4.6.0's LambdaMART implementation. Training uses all 1,382 development rounds from 86 people, with groups of ten candidates. The ranker receives the same 352 input columns and binary labels as ReelRank. Four participant-disjoint validation folds evaluate twelve settings. The selection rule maximizes mean first-position hits, using nDCG and model simplicity for ties. The winner had 31 leaves, 200 rounds and truncation level 3. The aligned confirmation interval is clustered on participants. The calendar sensitivity refits both rankers on earlier people and transfers the hyperparameters selected under the original participant design. These calendar results do not constitute a prospective study. [LightGBM LGBMRanker](https://lightgbm.readthedocs.io/en/v4.6.0/pythonapi/lightgbm.LGBMRanker.html)

## Microsoft News

The neural reference experiment uses the NewsReX authors' published weights and encoders. We compare original NRMS, NAML and LSTUR architectures on complete labelled validation files. MIND-small and MIND-large share a validation day. The large archive uses different item identifiers, so the pipeline matches publisher URLs and verifies candidate order. We separately evaluate a conservative reader-overlap screen. A common tie rule makes scores comparable within each list. The fixed 50-click model uses only pre-impression information. A separate feedback experiment uses earlier same-day small-release impressions; the source records impression time and does not measure real production click latency. [Microsoft benchmark](https://msnews.github.io/) · [NewsReX publication](https://doi.org/10.1145/3799682.3840209) · [Released weights](https://huggingface.co/newsrex)

Follow the [MIND reproduction guide](results/choice_gains/mind_external_reproduction.md) for the pinned publisher sources, scoring commands, cohort checks and results. Microsoft reports AUC, MRR and nDCG at 5 and 10. First-position hit is an additional endpoint. Hidden-test leaderboard results use a different split.

### Complete MIND-large training experiment

A separate experiment reads every training impression and every labelled validation impression in the large release. The training data contain 2,232,748 lists with 83,507,374 displayed candidates and 3,383,656 recorded clicks from 711,222 readers. The published CPU implicit ALS implementation trains on 16,532,504 distinct positive user-article pairs from the training histories and clicked candidates. Each development impression supplies up to 50 preceding clicks for the official fold-in solver. Articles without training interactions receive zero collaborative score. Two further controls use training click count and empirical-Bayes click-through rate. One fixed ReelRank extension combines a 20% training-only click-rate rank with an 80% capped-history model rank. The input weight was recorded before scoring this extension. The already studied validation date makes it a secondary result. [Official implicit ALS API](https://benfred.github.io/implicit/api/models/cpu/als.html) · [Pinned training archive](https://huggingface.co/datasets/Recommenders/MIND/blob/2115068f457b577f93a268fffe22373e45727d89/MINDlarge_train.zip)

After building the capped-history ranking, use a Python environment with the numerical dependencies above and install `implicit==0.7.2` and `tqdm==4.67.1`. The script verifies the training and validation SHA-256 digests. The full development click file is public. The official large test file withholds the labels. Run `python scripts/verify_mind_large_training.py --data "$mind" --report "$mind/mind_full_training.json" --out "$mind/mind_full_training_verification.json"` to check every validation list using the original Microsoft metric functions.

```sh
curl -fLsS https://huggingface.co/datasets/Recommenders/MIND/resolve/2115068f457b577f93a268fffe22373e45727d89/MINDlarge_train.zip -o "$mind/MINDlarge_train.zip"
python scripts/choice_mind_full_training.py --data "$mind" --protocol results/choice_gains/mind_full_training_protocol.json --out "$mind/mind_full_training.json"
```

## Reading the evidence

The movie audit checks each predicted order against the original ten IDs and all raw selected flags. The news audit computes original Microsoft metrics and independently counts first-position hits. The release contains source code, aggregate outputs and verification tests. Individual histories and recommendations stay in the licensed datasets and private run directory. Team responsibilities in the paper are assigned work areas.

The independent re-fit reproduced exactly all 5,510 primary orders, all 6,892 out-of-fold orders and all 2,358 calendar orders from the original frozen system using predictions from the six released engines. The second raw-label audit reproduced every reported count. The largest numeric change across the aggregate file was 2.8 × 10⁻¹⁷. The source template used to build the PowerPoint was the supplied ASU 2026 deck, SHA-256 `a5d92e5a6fbacb990e0231341a0ccbece8080d07348034177abddb897be47c1a`.
