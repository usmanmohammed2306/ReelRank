"""Draw the course-report figures from the audited external comparisons."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results" / "choice_gains"
OUTPUT = ROOT / "figures"
WIDTH = 468
MAROON = "#8C1D40"
DARK = "#252427"
MUTED = "#53535A"
SILVER = "#E8EAEB"
LINE = "#CCC8C4"
CREAM = "#FAF4E8"
GOLD = "#FFC627"
ROSE = "#F7EDF0"
WHITE = "#FFFFFF"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "pdf.fonttype": 42,
    "axes.unicode_minus": False,
})


def canvas(height):
    """Make a figure whose coordinate units are physical points."""
    fig = plt.figure(figsize=(WIDTH / 72, height / 72), facecolor=WHITE)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, WIDTH)
    ax.set_ylim(height, 0)
    ax.axis("off")
    return fig, ax


def label(ax, x, y, value, size=9, *, color=DARK, bold=False, align="left", **kwargs):
    return ax.text(x, y, value, ha=align, va="center", fontsize=size,
                   color=color, fontweight="bold" if bold else "normal", **kwargs)


def rounded(ax, x, y, w, h, face, edge="none", radius=4, lw=0.7):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={radius}",
                                facecolor=face, edgecolor=edge, linewidth=lw))


def arrow(ax, x1, y1, x2, y2, color=MUTED):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=9, linewidth=1.2, edgecolor=color,
                                 facecolor=color, shrinkA=0, shrinkB=0))


def save(fig, name):
    OUTPUT.mkdir(exist_ok=True)
    fig.savefig(OUTPUT / f"choice_pitch_{name}.pdf",
                metadata={"Title": f"ReelRank {name}", "CreationDate": None, "ModDate": None})
    plt.close(fig)


def check(ax, x, y):
    ax.add_patch(Circle((x, y), 5.4, facecolor=GOLD, edgecolor=WHITE, lw=.7))
    ax.plot([x-2.5, x-.6, x+2.8], [y, y+2.0, y-2.0], color=DARK, lw=1.0,
            solid_capstyle="round", solid_joinstyle="round")


def film(ax, x, y, thriller, chosen):
    """A generic film tile with a visible T for thriller and a separate choice mark."""
    w, h = 21, 27
    rounded(ax, x, y, w, h, MAROON if thriller else SILVER,
            edge=MAROON if thriller else "#A8A9AC", radius=2, lw=.5)
    label(ax, x+w/2, y+11, "T" if thriller else "•", size=9.3, align="center",
          color=WHITE if thriller else MUTED, bold=True)
    ax.plot([x+5, x+w-5], [y+21.5, y+21.5], color="#D99BAC" if thriller else "#A3A4A7", lw=1.2)
    if chosen:
        check(ax, x+w-.8, y+1)


def opportunity():
    fig, ax = canvas(158)
    label(ax, 5, 10, "The viewer chooses the same two thrillers", 12, bold=True)
    label(ax, 462, 10, "Illustrative example", 8.1, align="right", color=MUTED)
    selected = {1, 7}
    for row_y, mixed in ((31, False), (89, True)):
        label(ax, 5, row_y+8, "Mixed genres" if mixed else "All thrillers", 10.6, bold=True)
        label(ax, 5, row_y+23, "2 of 10 are thrillers" if mixed else "10 of 10 are thrillers",
              8.4, color=MUTED)
        for j in range(10):
            film(ax, 145+32*j, row_y, j in selected or not mixed, j in selected)
    ax.plot([5, 461], [72, 72], lw=.65, color=LINE)
    ax.plot([5, 461], [129, 129], lw=.65, color=LINE)
    rounded(ax, 6, 141, 10, 10, MAROON, radius=1.5)
    label(ax, 11, 146, "T", 6.4, align="center", bold=True, color=WHITE)
    label(ax, 22, 146, "Thriller", 8.4)
    rounded(ax, 105, 141, 10, 10, SILVER, edge="#A8A9AC", radius=1.5)
    label(ax, 110, 146, "•", 7, align="center", color=MUTED)
    label(ax, 121, 146, "Other genre", 8.4)
    check(ax, 232, 146)
    label(ax, 244, 146, "Marked for consideration", 8.4)
    save(fig, "opportunity")


def method():
    fig, ax = canvas(168)
    label(ax, 5, 10, "How ReelRank orders the next ten titles", 12, bold=True)
    rounded(ax, 5, 31, 123, 83, CREAM)
    label(ax, 13, 46, "Earlier displays", 10.5, bold=True)
    label(ax, 13, 65, "Which titles appeared?", 8.8)
    label(ax, 13, 80, "Which did this viewer", 8.8)
    label(ax, 13, 93, "choose?", 8.8)
    arrow(ax, 131, 72, 146, 72)
    rounded(ax, 149, 31, 130, 83, ROSE)
    label(ax, 157, 46, "Preference profile", 10.5, bold=True)
    label(ax, 157, 66, "Compare features of", 8.8)
    label(ax, 157, 80, "chosen titles with those", 8.8)
    label(ax, 157, 94, "of all titles shown.", 8.8)
    arrow(ax, 282, 72, 297, 72)
    rounded(ax, 300, 31, 163, 83, MAROON)
    label(ax, 310, 47, "Score each title", 10.7, color=WHITE, bold=True)
    label(ax, 310, 69, "50% conditional logit", 9.3, color=WHITE)
    label(ax, 310, 86, "50% gradient boosting", 9.3, color=WHITE)
    label(ax, 5, 143, "Catalog features and six recommendation scores", 8.8, color=MUTED)
    ax.plot([237, 286, 286], [143, 143, 101], c=MUTED, lw=1.05)
    arrow(ax, 286, 101, 297, 101)
    arrow(ax, 385, 117, 385, 132, MAROON)
    label(ax, 384, 146, "Rank all ten titles", 10.1, align="center", bold=True, color=MAROON)
    save(fig, "method")


def evidence(n, winner, comparisons):
    fig, ax = canvas(174)
    rounded(ax, 4, 2, 460, 38, ROSE)
    label(ax, 13, 20, "ReelRank", 11, bold=True)
    label(ax, 451, 20, f"{winner:,} / {n:,} correct first choices", 11,
          color=MAROON, align="right", bold=True)
    label(ax, 5, 55, "Comparison model", 9.2, color=MUTED, bold=True)
    label(ax, 143, 55, "First hits", 9.2, color=MUTED, align="center", bold=True)
    label(ax, 191, 55, "ReelRank gain in percentage points", 9.2, color=MUTED, bold=True)
    min_x, max_x = 193, 454
    plot_x = lambda pp: min_x + (max_x-min_x)*(pp+2)/10
    for tick in range(-2, 9, 2):
        x = plot_x(tick)
        ax.plot([x, x], [67, 139], c="#9B9B9F" if tick==0 else SILVER,
                lw=.85 if tick==0 else .55, zorder=0)
        label(ax, x, 148, str(tick), 8, align="center", color=MUTED)
    for y, (title, hits, gain, low, high) in zip((83, 122), comparisons):
        label(ax, 5, y, title, 8.8)
        label(ax, 143, y, f"{hits:,}", 10.0, align="center", bold=True)
        ax.plot([plot_x(low), plot_x(high)], [y, y], c=MAROON, lw=1.65, solid_capstyle="round")
        ax.plot([plot_x(low), plot_x(low)], [y-3, y+3], color=MAROON, lw=.9)
        ax.plot([plot_x(high), plot_x(high)], [y-3, y+3], color=MAROON, lw=.9)
        ax.scatter([plot_x(gain)], [y], s=34, c=MAROON, zorder=3)
        label(ax, plot_x(gain), y-10, f"+{gain:.2f}", 8.7, color=MAROON, align="center", bold=True)
    label(ax, 5, 168, "Each line shows the paired 95% interval from resampling participants.",
          8.2, color=MUTED)
    save(fig, "evidence")


def positions(steps):
    fig, ax = canvas(179)
    total = sum(steps)
    label(ax, 5, 10, "Where the additional correct first choices came from", 12, bold=True)
    label(ax, 5, 27, "Change versus UserKNN, the strongest SM-RS first-position baseline", 8.8, color=MUTED)
    baseline, unit, bar_width = 139, 86 / total, 51
    centers = (73, 182, 291, 400)
    previous = 0
    for i, delta in enumerate(steps + [total]):
        x = centers[i]
        start = previous if i < len(steps) else 0
        end = start+delta
        y1, y2 = baseline-start*unit, baseline-end*unit
        color = DARK if i == len(steps) else (MAROON if delta>=0 else "#777981")
        ax.add_patch(Rectangle((x-bar_width/2, min(y1,y2)), bar_width, abs(y2-y1), color=color))
        if i < len(steps):
            ax.plot([x+bar_width/2, centers[i+1]-bar_width/2], [y2, y2],
                    lw=.85, c=MUTED, linestyle=(0,(2.5,2)))
        label_y = min(y1,y2)-9 if delta>0 else y2+9
        label(ax, x, label_y, f"{delta:+d}", 11.3, align="center", bold=True, color=color)
        previous = end
    ax.plot([29, 446], [baseline, baseline], color="#A2A1A5", lw=.6, zorder=0)
    label(ax, 19, baseline, "0", 8, align="right", color=MUTED)
    for x, t in zip(centers, ("Originally first", "Originally 2nd–5th", "Originally 6th–10th", "Net gain")):
        label(ax, x, 159, t, 8.7, align="center", bold=(x==centers[-1]))
    save(fig, "positions")


def news(cohorts):
    fig, ax = canvas(95)
    methods = cohorts["full"]["models"]
    external = [k for k in methods if k.split("-")[0] in ("NRMS", "NAML", "LSTUR")
                and k.split("-")[-1] in ("42", "123", "456")]
    best = max(external, key=lambda name: methods[name]["first_hits"])
    label(ax, 5, 9, "Additional clicked headlines placed first per 1,000 lists", 11.5, bold=True)
    label(ax, 5, 25, "Frozen ReelRank on all 376,471 public validation lists", 8.4, color=MUTED)
    left, right, max_rate = 147, 413, 15
    for y, name, comparator in ((48, best.replace("-", ", seed "), best), (72, "Nine-model average", "Neural-9")):
        rate, lo, hi = cohorts["full"]["paired"]["Matched50"][comparator]["clustered_95"]["Hit@1"]
        rate, lo, hi = (1000 * value for value in (rate, lo, hi))
        if not 0 < lo < rate < hi < max_rate:
            raise ValueError("The audited range is outside this labeled plotting scale.")
        scale = (right - left) / max_rate
        label(ax, 5, y, name, 10, bold=True)
        ax.plot([left, right], [y, y], color=SILVER, lw=1)
        rounded(ax, left, y-4, scale*rate, 8, ROSE, radius=2)
        ax.plot([left+scale*lo, left+scale*hi], [y, y], color=MAROON, lw=1.35)
        for end in (lo, hi):
            ax.plot([left+scale*end, left+scale*end], [y-3.6, y+3.6], color=MAROON, lw=.9)
        ax.scatter([left+scale*rate], [y], s=22, c=MAROON, zorder=3)
        label(ax, 462, y, f"+{rate:.1f}", 10.1, color=MAROON, bold=True, align="right")
    for tick in (0, 5, 10, 15):
        label(ax, left+(right-left)*tick/max_rate, 89, str(tick), 7.8, align="center", color=MUTED)
    save(fig, "news")


def main():
    raw = json.loads((DATA / "external_baseline_audit.json").read_text())["confirmation"]
    matched = json.loads((DATA / "lambdamart_results.json").read_text())["confirmation"]
    if (raw["lists"], raw["people"]) != (matched["slates"], matched["participants"]):
        raise ValueError("The external comparisons use different cohorts.")
    count = raw["hits"]["personal50"]
    comparisons = []
    for title, hits, scores in (
        ("UserKNN", raw["hits"]["user_knn"], raw["paired"]["user_knn"]["hit1"]),
        ("LambdaMART", matched["first_hits"]["lambdamart"], matched["paired"]["hit1"]),
    ):
        lo, hi = scores["ci95"]
        comparisons.append((title, hits, 100*scores["difference"], 100*lo, 100*hi))
    origin = raw["by_original_position"]
    steps = [new-old for new,old in zip(origin["personal50"], origin["user_knn"])]
    if sum(steps) != count - raw["hits"]["user_knn"]:
        raise ValueError("Original-position changes do not sum to the measured difference.")
    opportunity()
    method()
    evidence(raw["lists"], count, comparisons)
    positions(steps)
    data = json.loads((DATA / "mind_external_baselines.json").read_text())
    news(data["cohorts"])


if __name__ == "__main__":
    main()
