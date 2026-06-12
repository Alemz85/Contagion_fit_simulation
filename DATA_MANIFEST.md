# DATA_MANIFEST.md — データの出所と追加データカタログ

## 同梱データ（取得・検証済み 2026-06-12）

| パス | 出所 | 中身 | 検証結果 |
|---|---|---|---|
| `data/higgs/higgs-activity_time.txt.gz` | SNAP公式データのGitHubミラー (AndyVale/Network_Analysis-Higgs_Twitter_Dataset) | 563,069イベント（RT 354,930 / MT 171,237 / RE 36,902）、形式: `userA userB unixtime type` | 行数・形式がSNAP公式仕様と一致 |
| `data/higgs/higgs-retweet_network.edgelist.gz` | 同上 | RTネットワーク（重み付きエッジリスト） | 形式確認済み |
| `data/twitter15/twitter15_graph.txt` ほか | OwenLeng/rumor-detection-include-twitter15-twitter16data- | 1,489カスケード（1行=1カスケード、`root_id \t uid:weight ...`）+ ラベルtsv | 全件ラベル結合成功 |
| `data/twitter15/rvnn_trees_TD.txt.gz` | majingCUHK/Rumor_RvNN（原著者公式リポジトリ） | カスケードの親子ツリー構造（深さ解析用） | 形式確認済み |
| `data/twitter16/` | 同上 | 817カスケード | 全件ラベル結合成功 |
| `data/weibo/weibo_graph.txt.gz` | OwenLeng同上 | 4,664カスケード + rumor/non-rumorラベル | 全件ラベル結合成功 |
| `data/processed/cascades_unified.csv` | `scripts/preprocess.py` で生成 | 統一テーブル 20,169行: platform, cascade_id, label, size | サニティチェック済（DESIGN.md §3の表） |
| `data/processed/higgs_hourly_volume.csv` | 同上 | Higgs RTの時間別件数（発表瞬間のバーストが見える。プレゼン導入図に使える） | 生成済み |

## 原典の引用義務

- **Higgs**: M. De Domenico, A. Lima, P. Mougel, M. Musolesi, "The Anatomy of a Scientific Rumor," *Scientific Reports* 3, 2980 (2013). 原配布: https://snap.stanford.edu/data/higgs-twitter.html
- **Twitter15/16**: J. Ma, W. Gao, K.-F. Wong, "Detect Rumors in Microblog Posts Using Propagation Structure via Kernel Learning," *ACL* (2017)
- **Weibo**: J. Ma et al., "Detecting Rumors from Microblogs with Recurrent Neural Networks," *IJCAI* (2016)

いずれも研究目的の利用が前提。プレゼン・READMEに必ず引用を入れること。

## 追加で取るならここ（カスケード/ネットワークデータの集積地カタログ）

| サイト | URL | 特徴 |
|---|---|---|
| **SNAP** (Stanford) | snap.stanford.edu/data | 定番。Higgs全体（フォロワー網14.8Mエッジ含む）、memetracker等 |
| **Netzschleuder** | networks.skewed.de | 最大級のネットワークカタログ。graph-toolから1行でロード可。Higgsもミラーあり |
| **KONECT** (Koblenz) | konect.cc | 数百のネットワーク、統計量付き |
| **ICON** (Colorado) | icon.colorado.edu | ネットワークデータの横断インデックス |
| **AMiner** | aminer.org/data-sna | Weibo大規模カスケード（DeepHawkes用約30万件）等。要登録のものあり |
| **SocioPatterns** | sociopatterns.org | 対面接触ネットワーク（学校・病院）。複雑伝播の実証研究でよく使われる |

### 追加するなら優先順位
1. **Higgsフォロワー網全体** (`higgs-social_network.edgelist.gz`, ~80MB, SNAP) — 基盤ネットワークを合成から実物に差し替えられる（DESIGN.md Phase 2 選択肢C）
2. **AMiner Weibo (DeepHawkes版)** — カスケード数が桁違い（~30万）。裾の統計が安定する
3. **SocioPatterns** — 「対面ネットワーク上では複雑伝播がどう違うか」の拡張用

## 取得時の注意（今回の実体験から）

- SNAP本体・Netzschleuder・KaggleはこのClaude環境からは直接落とせない（ネットワーク制限）。
  ローカルのClaude Codeなら制限なく取得可能。
- GitHubミラーは形式が原典と一致するか必ず検証する（今回は行数・カラム形式・タイプ別件数で照合した）。
- Twitter15/16の「graph.txt」形式はタイムスタンプを含まない処理済み版。時刻つき生ツリーが
  必要になったら原著者配布版（Ma氏のDropbox、Rumor_RvNN READMEにリンク）を取得する。
