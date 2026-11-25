"""Independently check all published MIND metrics on the full-training comparisons."""
import argparse
from importlib import util
import json
from pathlib import Path

import numpy as np

from choice_external_benchmark import digest, ci, grade
from choice_external_metric_audit import original_functions, evaluate_functions
from choice_external_mind import METRICS
from choice_external_verify import EXPECTED
from choice_neural_gain_stream import read


def verify(data, report, out):
    if digest(data/"evaluate.py")!=EXPECTED:
        raise ValueError("Unexpected publisher evaluator")
    spec=util.spec_from_file_location("microsoft_evaluator",data/"evaluate.py")
    module=util.module_from_spec(spec)
    spec.loader.exec_module(module)
    functions=original_functions(module)
    original=read(data/"MINDlarge_dev.zip")
    result={}
    for name,published in json.loads(report.read_text())["scores"].items():
        ranks=np.load(data/f"ext_ranks_full_{name}.npy",mmap_mode="r")
        values=evaluate_functions(ranks,original.labels,original.offsets,functions).mean(axis=0)
        expected=[published[k] for k in METRICS]
        np.testing.assert_allclose(values,expected,rtol=0,atol=1e-10)
        hits=int(original.labels[ranks==1].sum())
        if hits!=published["first_hits"] or int(np.count_nonzero(ranks==1))!=len(original.ids):
            raise ValueError("Raw first-position count discrepancy")
        result[name]=dict(lists=len(original.ids),hits=hits,official=dict(zip(METRICS,values)),
                          maximum_error=float(np.max(abs(values-expected))))
    prior=np.load(data/"ext_ranks_full_ReelRank-large-prior.npy",mmap_mode="r")
    ensemble=np.load(data/"ext_ranks_full_Neural-9.npy",mmap_mode="r")
    diff=grade(original.labels,original.offsets,prior)-grade(original.labels,original.offsets,ensemble)
    _,groups=np.unique(original.users,return_inverse=True)
    paired={"extra_first_hits":int(diff[:,4].sum()),"paired_reader_ci95":ci(diff,groups)}
    out.write_text(json.dumps({"evaluator_sha256":EXPECTED,"full_training_methods":result,
                               "large_prior_minus_nine_external_checkpoints":paired},indent=2)+"\n")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ("data","report","out"):
        parser.add_argument("--"+key,required=True,type=Path)
    args=parser.parse_args()
    verify(args.data,args.report,args.out)
