"""Refit the locked ReelRank movie/book model using publisher files."""
import argparse
import json
from pathlib import Path

import pandas as pd
from threadpoolctl import threadpool_limits

from scripts import choice_content_ranker as base
from scripts import choice_fusion as fusion
from scripts import choice_personal_ranker as personal
from scripts.choice_personal_forward import time_split


def run(args):
    lock = json.loads(args.lock.read_text())
    if base.digest(Path(personal.__file__)) != lock["source_sha256"]:
        raise ValueError("Model source differs from development lock")
    users = pd.read_parquet(args.root/"data/users.parquet").userId.tolist()
    cases = personal.assemble(args.root,args.movie_zip,users)
    winner = f"personal{lock['fraction']}"

    def score(train,test):
        predictions = personal.fit_split(train,test,fractions=(lock["fraction"],))
        ranked = base.rows_for(test,{winner:predictions[winner]})
        base.canonical(test,ranked)
        return ranked

    args.private.mkdir(exist_ok=True,parents=True)
    dev = [c for c in cases if fusion.participant_fold(c.user)==0]
    holdout = [c for c in cases if fusion.participant_fold(c.user)!=0]
    score(dev,holdout).to_csv(args.private/"personal_confirmation.csv.gz",index=False)
    print("Participant confirmation complete",flush=True)
    _, earlier, later, _ = time_split(pd.read_parquet(args.root/"data/behaviors.parquet"))
    score([c for c in cases if c.user in earlier],[c for c in cases if c.user in later]).to_csv(
        args.private/"personal_calendar.csv.gz",index=False)
    print("Calendar refit complete",flush=True)
    grouped = []
    for fold in range(5):
        grouped.append(score([c for c in cases if fusion.participant_fold(c.user)!=fold],
                             [c for c in cases if fusion.participant_fold(c.user)==fold]))
    pd.concat(grouped,ignore_index=True).to_csv(args.private/"personal_grouped_cv.csv.gz",index=False)
    print("Five held-out participant folds complete",flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for k in ("root","movie-zip","private","lock"):
        parser.add_argument("--"+k,type=Path,required=True)
    with threadpool_limits(limits=2):
        run(parser.parse_args())
