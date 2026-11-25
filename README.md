# ReelRank

ReelRank studies which title to put first in a recommendation list. For movies and books, it learns from both the titles viewers selected and the alternatives they were offered. A separate news experiment tests the same first-position question on recorded headline clicks.

## Final materials

- [Project report](output/pdf/ReelRank_Project_Report.pdf) and [LaTeX source](reelrank_project_report.tex)
- [PowerPoint](output/pptx/ReelRank_ASU_Presentation.pptx) and [slide PDF](output/pdf/ReelRank_ASU_Presentation.pdf)
- [Speaking script](output/pdf/ReelRank_Presentation_Script.pdf)

## Results

| Evaluation | ReelRank first hits | Comparison first hits |
|---|---:|---:|
| 5,510 held-out SM-RS movie and book lists | 2,920 | 2,578 for UserKNN |
| Same SM-RS lists, with matched features | 2,920 | 2,882 for LambdaMART |
| 376,471 MIND-large validation lists, with matched training impressions and history limits | 77,348 | 75,245 for the nine-checkpoint average |

ReelRank gained 6.21 percentage points over UserKNN (95% confidence interval 4.83 to 7.55). The same-feature LambdaMART difference was statistically inconclusive (95% interval -0.36 to 1.75 percentage points). Both studies evaluate rankings against recorded actions. A future user study could measure how the new ordering affects choice time and viewing.

## Reproduce the experiments

Start with the [reproduction guide](external_benchmarks_reproduction.md). It covers environment setup, official dataset and checkpoint downloads, and the final movie/book and news evaluations. The repository includes aggregate results and verification tests. Download the third-party inputs from the linked publishers and keep them outside the checkout.

The report documents the data sources, published methods, team roles, and evaluation limits.
