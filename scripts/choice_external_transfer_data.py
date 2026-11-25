"""Build a private, conservatively de-duplicated MIND-large validation cohort."""

import argparse
import hashlib
import json
from array import array
from collections import Counter
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

ROOT = Path(__file__).resolve().parents[1]


def signature(ids):
    return hashlib.blake2b(array("i", ids).tobytes(), digest_size=20).digest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument(
        "--audit",
        type=Path,
        default=ROOT / "results/choice_gains/mind_overlap_audit.json",
    )
    args = parser.parse_args()
    paths = [
        args.data / name
        for name in ("MINDsmall_train.zip", "MINDsmall_dev.zip", "MINDlarge_dev.zip")
    ]
    protocol = json.loads(
        (ROOT / "results/choice_gains/mind_inputs.json").read_text()
    )
    if hashlib.sha256(paths[2].read_bytes()).hexdigest() != protocol["raw_sha256"]:
        raise ValueError("MIND-large validation digest mismatch")
    articles, raw_id_urls = {}, {}
    for path in paths[:2]:
        with ZipFile(path) as z:
            for line in z.open("news.tsv"):
                raw, cat, sub, title, abst, url, *_ = line.rstrip(b"\r\n").split(b"\t")
                if raw in raw_id_urls and raw_id_urls[raw] != url:
                    raise ValueError("Ambiguous small article identifier")
                if url in articles and articles[url][0] != int(raw[1:]):
                    raise ValueError("Multiple small IDs for one URL")
                articles[url] = (int(raw[1:]), cat, sub, title, abst)
                raw_id_urls[raw] = url
    small_ids = {raw: articles[url][0] for raw, url in raw_id_urls.items()}
    largest = max(small_ids.values())
    large_ids, large_urls, news_rows = {}, set(), []
    coverage = Counter()
    with ZipFile(paths[2]) as z:
        for row in z.open("news.tsv"):
            fields = row.rstrip(b"\r\n").split(b"\t")
            raw, cat, sub, title, abst, url = fields[:6]
            if url in large_urls or raw in large_ids:
                raise ValueError("Nonunique large URL or ID")
            large_urls.add(url)
            if raw in raw_id_urls:
                coverage["large_news_ids_also_seen_in_small"] += 1
                coverage["same_url_at_coincident_raw_id"] += int(
                    url == raw_id_urls[raw]
                )
            if url in articles:
                canonical, cat0, sub0, title0, abst0 = articles[url]
                coverage["shared_article_urls"] += 1
                coverage["shared_urls_matching_title_category_subcategory"] += int(
                    (title, cat, sub) == (title0, cat0, sub0)
                )
                coverage["shared_urls_matching_abstract"] += int(abst == abst0)
            else:
                largest += 1
                canonical = largest
                coverage["large_only_urls"] += 1
            large_ids[raw] = canonical
            fields[0] = b"N" + str(canonical).encode()
            news_rows.append(b"\t".join(fields) + b"\n")
    all_small_users = set()
    known_histories = set()
    exposures = {}

    def history_fingerprint(history, lookup, cache):
        if history not in cache:
            cache[history] = signature(lookup[t] for t in history.split())
        return cache[history]

    history_cache = {}
    for path in paths[:2]:
        with ZipFile(path) as z:
            for row in z.open("behaviors.tsv"):
                _, user, ts, history, slate = row.rstrip(b"\r\n").split(b"\t")
                all_small_users.add(user)
                if history:
                    known_histories.add(
                        history_fingerprint(history, small_ids, history_cache)
                    )
                items = slate.split()
                key = (ts, signature(sorted(small_ids[t[:-2]] for t in items)))
                clicked = signature(
                    sorted(small_ids[t[:-2]] for t in items if t.endswith(b"-1"))
                )
                exposures.setdefault(key, set()).add((user, clicked))
    by_id, by_history, by_slate, all_users = set(), set(), set(), set()
    large_history_cache = {}
    counts = Counter()
    with ZipFile(paths[2]) as z:
        for row in z.open("behaviors.tsv"):
            _, user, ts, history, slate = row.rstrip(b"\r\n").split(b"\t")
            all_users.add(user)
            counts["source_impressions"] += 1
            if user in all_small_users:
                by_id.add(user)
            if (
                history
                and history_fingerprint(history, large_ids, large_history_cache)
                in known_histories
            ):
                by_history.add(user)
                counts["impressions_matching_nonempty_history"] += 1
            items = slate.split()
            counts["source_candidates"] += len(items)
            key = (ts, signature(sorted(large_ids[t[:-2]] for t in items)))
            if key in exposures:
                by_slate.add(user)
                counts["impressions_matching_timestamp_and_slate"] += 1
                clicked = signature(
                    sorted(large_ids[t[:-2]] for t in items if t.endswith(b"-1"))
                )
                counts["matched_slates_with_identical_clicked_articles"] += int(
                    any(v == clicked for _, v in exposures[key])
                )
                counts["matched_slates_with_identical_raw_user_id"] += int(
                    any(u == user for u, _ in exposures[key])
                )
    excluded = by_id | by_history | by_slate
    counts.update(
        source_users=len(all_users),
        known_small_users=len(all_small_users),
        excluded_users=len(excluded),
        excluded_by_raw_user_id=len(by_id),
        excluded_by_nonempty_history=len(by_history),
        excluded_by_timestamp_and_slate=len(by_slate),
        eligible_users=len(all_users - excluded),
    )
    known_ids = set(small_ids.values())
    large_bytes = {key: b"N" + str(v).encode() for key, v in large_ids.items()}
    known_in_train = set()
    with ZipFile(paths[0]) as z:
        for row in z.open("behaviors.tsv"):
            known_in_train.update(
                small_ids[t[:-2]] for t in row.rstrip(b"\r\n").split(b"\t")[-1].split()
            )
    relative = {}
    with (
        ZipFile(paths[2]) as z,
        ZipFile(args.cohort, "w", compression=ZIP_DEFLATED, compresslevel=1) as target,
    ):
        target.writestr(
            ZipInfo("news.tsv"),
            b"".join(news_rows),
            compress_type=ZIP_DEFLATED,
            compresslevel=1,
        )
        with target.open("behaviors.tsv", "w") as out:
            for row in z.open("behaviors.tsv"):
                sid, user, ts, hist, slate = row.rstrip(b"\r\n").split(b"\t")
                if user in excluded:
                    continue
                if hist not in relative:
                    relative[hist] = b" ".join(large_bytes[t] for t in hist.split())
                items = slate.split()
                numbers = [large_ids[t[:-2]] for t in items]
                counts["eligible_impressions"] += 1
                counts["eligible_candidates"] += len(items)
                counts["eligible_clicks"] += sum(t.endswith(b"-1") for t in items)
                counts["eligible_candidates_with_small_metadata"] += sum(
                    n in known_ids for n in numbers
                )
                counts["eligible_candidates_exposed_in_small_training"] += sum(
                    n in known_in_train for n in numbers
                )
                counts["eligible_impressions_with_empty_history"] += int(not hist)
                restored = b" ".join(large_bytes[t[:-2]] + t[-2:] for t in items)
                out.write(b"\t".join((sid, user, ts, relative[hist], restored)) + b"\n")
    with ZipFile(args.cohort) as derived:
        content_digests = {}
        for name in ("news.tsv", "behaviors.tsv"):
            digest = hashlib.sha256()
            with derived.open(name) as source:
                for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
                    digest.update(chunk)
            content_digests[name] = digest.hexdigest()
    output = dict(
        expanded_cohort_sha256=content_digests,
        archive_sha256=protocol["raw_sha256"],
        article_crosswalk=dict(coverage),
        complete_source_and_exclusions=dict(counts),
        cohort_private_zip_sha256=hashlib.sha256(args.cohort.read_bytes()).hexdigest(),
    )
    args.audit.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
