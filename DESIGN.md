# DESIGN.md — バイラル拡散シミュレーション × 実データフィット

**プロジェクト名**: `contagion-fit` — 単純伝播 vs 複雑伝播、どちらのメカニズムが実SNSカスケードを説明するか
**目的**: ESADE Python課題のプレゼン（ルーブリック: 課題定義15 / 技術25 / 価値15 / 苦労15 / 次のステップ10 / プレゼン10）
**実装担当**: Claude Code　**設計**: 本ドキュメント

---

## 0. リサーチクエスチョン（これが背骨。全コードはこの問いに奉仕する）

> 実SNSの情報カスケードのサイズ分布は、**単純伝播（IC: 1回の接触で確率的に伝播）** と
> **複雑伝播（閾値: 複数の隣人からの補強が必要）** のどちらのモデルでよりよく再現できるか？
> そしてその答えは**コンテンツの種類**（科学ニュース / 真の噂 / 偽の噂 / 非噂）と
> **プラットフォーム**（Twitter / Weibo）で変わるか？

**理論的背景**（プレゼンで引用する対立軸）:
- Centola & Macy (2007): 情報は単純伝播、行動は複雑伝播。弱い紐帯は単純を加速し複雑を阻害する。
- Watts & Dodds (2007): 大規模カスケードはインフルエンサーではなく「影響されやすい人々の臨界量」が駆動。
- 本プロジェクトは両論争を**実データへのフィット**という土俵で突き合わせる。

**ビジネス翻訳**（プレゼンの締め）: あなたの商材の拡散が単純型ならインフルエンサー起点が有効、
複雑型ならクラスタ化した一般ユーザー群への分散シードが有効。型の見極めが予算配分を決める。

---

## 1. 成果物（プレゼンに載せる図。これが完成条件）

| # | 図 | 生成モジュール |
|---|---|---|
| F1 | 実データのカスケードサイズCCDF（4データセット重ね描き、log-log） | `viz.py` |
| F2 | ベストフィット比較: 実CCDF vs IC-best vs Threshold-best（データセット別、2×2グリッド） | `viz.py` |
| F3 | フィット距離ヒートマップ: モデル×データセットのKS距離マトリクス | `viz.py` |
| F4 | シード戦略実験: ハブ起点 vs ランダム起点の期待リーチ（モデル別、エラーバー付き） | `viz.py` |
| F5 | 反転境界ヒートマップ: (伝播確率p × 閾値φ)平面で「ハブ優位/分散優位」が反転する境界 | `viz.py` |
| F6 | ワークフロー図（手描きでも可、ルーブリック「flowchart」要件） | 手動 |

---

## 2. リポジトリ構成

```
contagion-fit/
├── data/                      # 同梱済み（本パッケージ）
│   ├── higgs/                 #   higgs-activity_time.txt.gz, higgs-retweet_network.edgelist.gz
│   ├── twitter15/             #   twitter15_graph.txt, twitter15_labels.tsv, rvnn_trees_TD.txt.gz, Twitter15_label_All.txt
│   ├── twitter16/             #   同上
│   ├── weibo/                 #   weibo_graph.txt.gz, weibo_labels.tsv
│   └── processed/             #   cascades_unified.csv（検証済み・生成済み）, higgs_hourly_volume.csv
├── scripts/
│   └── preprocess.py          # 同梱済み・実行済み（再現可能）
├── src/contagion_fit/
│   ├── __init__.py
│   ├── config.py              # 全パラメータの一元管理（dataclass）
│   ├── data.py                # cascades_unified.csv のロード、CCDF計算
│   ├── network.py             # 基盤ネットワーク生成（BA / config model / 実Higgsフォロワー網）
│   ├── models.py              # SimpleContagion(IC), ComplexContagion(threshold) — 共通インターフェース
│   ├── simulate.py            # モンテカルロエンジン（並列化、シード固定）
│   ├── fit.py                 # グリッドサーチ + KS距離によるモデル選択
│   ├── experiments.py         # シード戦略2×2実験、反転境界スイープ
│   └── viz.py                 # F1〜F5の図生成
├── tests/
│   ├── test_models.py
│   ├── test_simulate.py
│   └── test_fit.py
├── results/                   # 図とフィット結果のJSON
├── pyproject.toml             # 依存: networkx, numpy, scipy, matplotlib, pytest（最小限に保つ）
└── README.md
```

**コーディング規約**: 型ヒント必須。乱数は全て `numpy.random.Generator` をDI（seed固定で全再現可能）。
ルーブリックが「functions, classes, modules」構造を25点配点で見るため、**モジュール分割そのものが採点対象**。

---

## 3. データ仕様（取得・検証済み）

`data/processed/cascades_unified.csv` — **20,169カスケード、検証済み**:

| platform | label | n | median | max |
|---|---|---|---|---|
| higgs | science_news | 13,199 | 2 | 223,833 |
| twitter15 | false/true/unverified/non-rumor | 1,489 | 5–24 | 136 |
| twitter16 | 同上 | 817 | 9–27 | 165 |
| weibo | false/non-rumor | 4,664 | 8–13 | 4,237 |

- **Higgs**: 2012年ヒッグス粒子発見発表（7/1–7/7）のTwitter拡散。RT 354,930件、タイムスタンプ付き。
  カスケード単位 = RTグラフの弱連結成分（**方法論上の注意①参照**）。
- **Twitter15/16**: 噂検証データセット（Ma et al. 2017）。1カスケード=1ソースツイートの伝播ツリー。
  ラベル: true / false / unverified / non-rumor。`rvnn_trees_TD.txt.gz` に親子構造があり、**深さ・幅の構造解析に拡張可**。
- **Weibo**: 中国Weiboの噂/非噂4,664カスケード（Ma et al. 2016）。プラットフォーム横断の頑健性チェック用。

**追加ダウンロード（任意、ローカルで）**: Higgsフォロワー網全体（456,626ノード/14.8Mエッジ）は
`https://snap.stanford.edu/data/higgs-twitter.html` の `higgs-social_network.edgelist.gz` (~80MB)。
ステップ4のSubstrate選択肢Cで使う。なくてもプロジェクトは完結する。

---

## 4. パイプライン設計

### Phase 1 — `data.py`
```python
def load_cascades(platform: str | None = None, label: str | None = None) -> np.ndarray
    # cascades_unified.csv からサイズ配列を返す
def ccdf(sizes: np.ndarray) -> tuple[np.ndarray, np.ndarray]
    # P(X >= x)。log-logプロット用
```
受け入れ基準: 上の検証テーブルと件数・中央値が一致すること。

### Phase 2 — `network.py` 基盤ネットワーク
シミュレーションの土俵。3つの選択肢を実装し、configで切替:
- **A (デフォルト)**: `nx.barabasi_albert_graph(n=50_000, m=3)` — スケールフリー
- **B**: `nx.watts_strogatz_graph(n=50_000, k=6, p=0.1)` — 高クラスタ（複雑伝播が有利な構造）
- **C (任意)**: 実Higgsフォロワー網のサンプル（追加DL時のみ）

AとBの両方で結果を出すこと自体が感度分析になる（基盤構造への依存性チェック）。

### Phase 3 — `models.py` 伝播モデル
共通インターフェース:
```python
class ContagionModel(Protocol):
    def run(self, G: nx.Graph, seeds: set[int], rng: np.random.Generator) -> set[int]:
        """最終的に活性化したノード集合を返す"""

class SimpleContagion:   # Independent Cascade
    p: float             # 各エッジで1回だけ試行する伝播確率

class ComplexContagion:  # Fractional Threshold (Watts 2002系)
    phi: float           # 隣人のうち活性化済みの割合がphi以上で活性化
    p: float = 1.0       # 閾値超過時の確率的採用（=1で決定的）
```
実装注意: ICはBFSフロンティアで実装（エッジごと1回試行）。閾値モデルは同期更新で
変化がなくなるまで反復。どちらも `run()` は活性ノード集合を返すだけにし、副作用なし。

### Phase 4 — `simulate.py` モンテカルロ
```python
def cascade_size_distribution(
    G, model, n_runs: int = 1000,
    seed_strategy: Literal["random", "hub"] = "random",
    seed_count: int = 1, rng=...
) -> np.ndarray   # shape (n_runs,) 各ランの最終サイズ
```
- 1ラン = ランダム（または次数上位）に種を1つ置き、モデルを走らせ、最終サイズを記録。
- これが「シミュレーション側のカスケードサイズ分布」になり、実データ分布と比較可能になる。
- `multiprocessing.Pool` で並列化。**n_runsは収束チェック**（500と1000でCCDFが視覚的に一致するか）。

### Phase 5 — `fit.py` フィットとモデル選択（プロジェクトの心臓部）
```python
def ks_distance(observed: np.ndarray, simulated: np.ndarray) -> float
    # log10(size)上の2標本KS統計量。裾の重さの違いに感度を持たせるため対数スケール必須

def grid_search(G, model_cls, param_grid: dict, observed, n_runs=1000) -> FitResult
    # FitResult: best_params, best_distance, 全グリッドの距離テーブル

def compare_models(dataset: str) -> ComparisonResult
    # IC best-fit vs Threshold best-fit のKS距離を比較し勝者を判定
```
グリッド（初期値、収束を見て調整可）:
- IC: `p ∈ {0.001, 0.003, 0.01, 0.03, 0.05, 0.1, 0.2}`
- Threshold: `phi ∈ {0.05, 0.1, 0.15, 0.2, 0.3, 0.4}` × `p ∈ {0.5, 1.0}`

**判定の出し方（重要）**: 「IC距離 0.04 vs Threshold距離 0.19 → このデータは単純伝播型」のように
データセット×ラベルごとに勝敗表を作る。これがF3になる。
予想される結果（仮説として明記、外れたらそれ自体が発見）:
science_news（Higgs）→ 単純型が勝つ / 噂・行動系 → 閾値型が競る、ラベル間で差が出る。

### Phase 6 — `experiments.py` シード戦略と反転境界
```python
def seeding_experiment(G, model, n_runs=1000) -> dict
    # {"hub": sizes, "random": sizes} 期待リーチと分布を比較 → F4

def flip_boundary(G, p_range, phi_range, n_runs=300) -> np.ndarray
    # 各(p, phi)でハブ優位度 = mean(hub) - mean(random) を計算 → F5ヒートマップ
```
ここがWatts-Dodds vs インフルエンサー神話の直接対決。フィット済みパラメータ近傍で
「自分のデータに最も近い世界では、どちらのシード戦略が正しいか」を答える。

### Phase 7 — `viz.py`
matplotlib のみ（seaborn不要）。全図はSVGとPNG両方を `results/` に保存。
log-log CCDF が基本表現。図には必ずn_runs、ネットワーク種別、パラメータを注記。

---

## 5. 方法論上の注意（プレゼンの「苦労」「限界」セクションの素材。隠さず明記する）

1. **HiggsのWCC近似**: ツイートIDが匿名化されているため、厳密な「1ツイート=1カスケード」分割は不可能。
   弱連結成分による近似は、別ルートのカスケードが共通ユーザーで融合するため**サイズを過大評価**する
   （max 223,833 はほぼ「発表直後の巨大連結成分」）。対処: (a) 発表前(7/1–7/3)のみのサブセットで
   頑健性確認、(b) この限界をプレゼンで明示。誠実さがルーブリックの得点源。
2. **識別可能性**: 異なる(モデル, パラメータ)が似たサイズ分布を生むことがある（equifinality）。
   サイズ分布だけでなく、余裕があればTwitter15のツリー深さ（structural virality）を第2の判別軸に追加。
3. **有限サイズ効果**: 基盤ネットワークn=50,000に対しHiggsの最大カスケード223,833は載らない。
   比較は**正規化サイズ**（カスケードサイズ/ネットワークサイズ）またはHiggs側を発表前サブセットに
   限定して行う。
4. **モンテカルロ収束**: n_runsを変えてCCDFの安定性を確認し、その図も残す（「苦労」素材）。
5. **基盤ネットワークの仮定**: BAとWSの両方で主結論（モデル勝敗、シード戦略反転）が変わらないか
   確認する。変わるならそれ自体を報告する（robust vs fragile の分離）。

---

## 6. テスト（`pytest`、ルーブリック「次のステップ: testing」の先取り実装）

- `test_models.py`: p=0でカスケードサイズ=seed数 / p=1で連結成分全体 / phi=0で全活性化、
  phi>1で種のみ。既知の小グラフ（パス、スター、完全グラフ）で手計算と一致。
- `test_simulate.py`: seed固定で完全再現。スターグラフのIC期待サイズが解析値 1+k*p と一致（許容誤差内）。
- `test_fit.py`: 自作データ（IC自身で生成）をグリッドサーチに食わせ、真のpが回収できる
  （**パラメータ回収テスト** — これが通れば手法の妥当性が示せる。プレゼンの強い1枚になる）。

---

## 7. 実行順序（Claude Codeへの指示）

1. `pip install networkx numpy scipy matplotlib pytest` （最小依存）
2. `scripts/preprocess.py` を実行し、検証テーブルの数値一致を確認
3. Phase 2→3→4 を実装、`tests/` を書きながら進める（特にパラメータ回収テストを先に）
4. Phase 5 を4データセット×ラベルで回す（計算が重ければ n_runs=500 に落とし、収束図を残す）
5. Phase 6 のシード実験と反転境界
6. Phase 7 で F1〜F5 を `results/` に出力
7. `results/summary.json` に全フィット結果（best_params, 距離、勝者）を保存

**スコープ厳守**: ベイズ推定(ABC)、時系列フィット、GNN等への拡張は実装しない。
「次のステップ」スライドに書く素材として温存する（complex centrality, CCIM強化学習,
Cencetti+ 2023の判別手法、ハイパーグラフ）。

---

## 8. 引用（プレゼン・README用）

- De Domenico et al. (2013) "The Anatomy of a Scientific Rumor" *Sci. Rep.* — Higgsデータ
- Ma et al. (2016, IJCAI / 2017, ACL) — Weibo / Twitter15/16 噂データセット
- Centola & Macy (2007) *AJS* — 複雑伝播と弱い紐帯
- Watts & Dodds (2007) *J. Consumer Research* — インフルエンサー仮説の検証
- Watts (2002) *PNAS* — 閾値モデルのカスケード理論
