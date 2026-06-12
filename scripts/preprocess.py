"""
preprocess.py — 各実データセットを統一カスケードテーブルに変換する。

出力: data/processed/cascades_unified.csv
  columns: platform, cascade_id, label, size
出力: data/processed/higgs_hourly_volume.csv (Higgsの時系列、参考用)

各データセットのカスケード定義:
- Higgs:     RTイベント (userA が userB をRT) から情報流方向 B->A の有向グラフを構築し、
             弱連結成分 (WCC) を1カスケードとみなす近似。
             ※ツイートIDが匿名化で落ちているため厳密なルート単位の分割は不可能。
             この近似の限界は DESIGN.md の「方法論上の注意」を参照。
- Twitter15/16: 1行 = 1カスケード (root_tweet_id \t "uid:weight uid:weight ...")
             参加ユーザー数 = カスケードサイズ。ラベル (true/false/unverified/non-rumor) を結合。
- Weibo:     同形式。ラベルは rumor / non-rumor。
"""
import gzip
import io
from collections import defaultdict
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"
OUT = DATA / "processed"
OUT.mkdir(exist_ok=True)


# ---------- Union-Find (Higgs WCC用) ----------
class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, x):
        root = x
        while self.parent.setdefault(root, root) != root:
            root = self.parent[root]
        while self.parent[x] != root:  # 経路圧縮
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def process_higgs():
    """RTイベントのみ抽出し、WCCをカスケードとみなす。時系列ボリュームも出力。"""
    uf = UnionFind()
    hourly = defaultdict(int)
    n_rt = 0
    with gzip.open(DATA / "higgs" / "higgs-activity_time.txt.gz", "rt") as f:
        for line in f:
            a, b, ts, kind = line.split()
            if kind != "RT":
                continue
            n_rt += 1
            uf.union(a, b)
            hourly[int(ts) // 3600 * 3600] += 1

    sizes = defaultdict(int)
    for node in list(uf.parent):
        sizes[uf.find(node)] += 1

    rows = [
        ("higgs", f"higgs_wcc_{i}", "science_news", s)
        for i, s in enumerate(sorted(sizes.values(), reverse=True))
    ]
    with open(OUT / "higgs_hourly_volume.csv", "w") as f:
        f.write("unix_hour,retweets\n")
        for ts in sorted(hourly):
            f.write(f"{ts},{hourly[ts]}\n")
    print(f"[higgs] RT events: {n_rt:,}  cascades(WCC): {len(rows):,}  "
          f"max size: {rows[0][3]:,}")
    return rows


def load_labels_tsv(path, label_col=2):
    """tweet_id \t text \t label 形式。"""
    labels = {}
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) > label_col:
                labels[parts[0]] = parts[label_col].strip()
    return labels


def process_graph_file(platform, graph_path, labels, open_fn=open):
    rows = []
    with open_fn(graph_path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            cid = parts[0]
            size = len(parts[1].split())
            rows.append((platform, cid, labels.get(cid, "unknown"), size))
    known = sum(1 for r in rows if r[2] != "unknown")
    print(f"[{platform}] cascades: {len(rows):,}  labeled: {known:,}")
    return rows


def main():
    all_rows = []
    all_rows += process_higgs()
    t15_labels = load_labels_tsv(DATA / "twitter15" / "twitter15_labels.tsv")
    all_rows += process_graph_file(
        "twitter15", DATA / "twitter15" / "twitter15_graph.txt", t15_labels)
    t16_labels = load_labels_tsv(DATA / "twitter16" / "twitter16_labels.tsv")
    all_rows += process_graph_file(
        "twitter16", DATA / "twitter16" / "twitter16_graph.txt", t16_labels)
    wb_labels = load_labels_tsv(DATA / "weibo" / "weibo_labels.tsv", label_col=1)
    all_rows += process_graph_file(
        "weibo", DATA / "weibo" / "weibo_graph.txt.gz", wb_labels,
        open_fn=lambda p, *a, **k: io.TextIOWrapper(gzip.open(p, "rb"), **{kk: vv for kk, vv in k.items() if kk in ("encoding", "errors")}))

    with open(OUT / "cascades_unified.csv", "w") as f:
        f.write("platform,cascade_id,label,size\n")
        for r in all_rows:
            f.write(",".join(map(str, r)) + "\n")
    print(f"\n=> {OUT / 'cascades_unified.csv'}  total rows: {len(all_rows):,}")


if __name__ == "__main__":
    main()
