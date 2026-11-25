"""Build the participant-disjoint calendar split for SM-RS."""
import numpy as np


def time_split(behaviors):
    periods = {}
    for u, group in behaviors.groupby("userId"):
        first=group.timestamp.min().value/1e9
        last=group.timestamp.max().value/1e9
        for r in group.itertuples():
            for o in r.impressions:
                if o["selected"] and np.isfinite(o["timestamp"]):
                    last=max(last,float(o["timestamp"]))
        periods[int(u)]=(first,last)
    starts=sorted(t[0] for t in periods.values())
    cutoff=starts[int(np.ceil(.7*(len(starts)-1)))]
    train={u for u,(start,end) in periods.items() if end<cutoff}
    test={u for u,(start,end) in periods.items() if start>=cutoff}
    if train & test:
        raise ValueError("Cohort crossing")
    return cutoff,train,test,len(periods)-len(train)-len(test)
