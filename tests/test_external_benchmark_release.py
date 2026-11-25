"""Evidence checks for the external-comparator course release."""
import hashlib
import json
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]
EVIDENCE=ROOT/"results/choice_gains"


def read(name):
    return json.loads((EVIDENCE/(name+".json")).read_text())


class ExternalBenchmarkRelease(unittest.TestCase):
    def test_movie_comparators_share_outcomes(self):
        baseline=read("external_baseline_audit")
        matched=read("lambdamart_results")
        checked=read("external_benchmark_verification")
        for cohort in ("confirmation","forward"):
            cases=baseline[cohort]
            learned=matched[cohort]
            self.assertEqual(cases["lists"],learned["slates"])
            self.assertEqual(cases["people"],learned["participants"])
            hits=cases["hits"]
            self.assertEqual(set(hits),set(baseline["released_orders"])|{"personal50"})
            self.assertEqual(hits["personal50"],learned["first_hits"]["personal50"])
            for method,change in cases["transitions"].items():
                self.assertEqual(hits["personal50"]-hits[method],change["gained"]-change["lost"])
            for method,total in learned["first_hits"].items():
                audited=checked["raw_label_regrade"][cohort][method]
                self.assertEqual(total,audited["first_hits"])
                self.assertEqual(audited["lists"],cases["lists"])
        self.assertEqual(matched["features"],352)
        self.assertEqual(len(matched["grid"]),12)
        self.assertEqual(matched["protocol_sha256"],hashlib.sha256((EVIDENCE/"lambdamart_protocol.json").read_bytes()).hexdigest())

    def test_full_large_training_scores_and_source(self):
        report=read("mind_full_training")
        protocol=read("mind_full_training_protocol")
        self.assertEqual(report["training_sha256"],protocol["train_sha256"])
        self.assertEqual(report["validation_sha256"],protocol["validation_sha256"])
        self.assertGreater(report["training"]["candidate_exposures"],80_000_000)
        total=report["confirmation_impressions"]
        self.assertEqual(total,read("mind_external_baselines")["cohorts"]["full"]["impressions"])
        for score in report["scores"].values():
            self.assertAlmostEqual(score["first_hits"]/total,score["Hit@1"],places=12)
            self.assertLess(score["official_random_audit"]["max_abs_error"],1e-10)
        self.assertEqual(report["protocol_sha256"],hashlib.sha256((EVIDENCE/"mind_full_training_protocol.json").read_bytes()).hexdigest())


if __name__=="__main__":
    unittest.main()
