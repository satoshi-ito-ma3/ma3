"""MA³ 実験001（v2）— 交絡を切り分ける（死 vs 忘却）

【v1 の反省】初版はこう主張していた：「変化する世界では有限性が不死に勝つ」。
だが独立レビューで交絡が判明した。mortal と immortal の違いは "いつ止まるか" だけ
ではなく、実際には3つの変数が同時に動いていた：
  (1) 推定に使う履歴窓の長さ（8 vs 200）
  (2) 観測サンプル数（8 vs 200）
  (3) 答え合わせの時刻（t=7 vs t=199、θ* がどれだけドリフトした後に採点するか）
コードは「死」を一度も表現しておらず、勝敗を分けていたのは「短い窓・近い的」か
「長い窓・遠い的」かだけ。"死" はその交絡に後から貼ったラベルだった。

【v2 の目的】"死" を独立変数に近づけるため、第3の群を足して変数を1本ずつ分離する。
結論を「死が効く」に寄せるためではない——混ざった要因を切り分けるためである。

────────────────────────────────────────────────────────────────
3群（他をそろえ、1要素ずつ変える）：

  群                  寿命/観測   記憶窓    評価時刻
  mortal              短(life)    短(life)  t=life-1   … 短命・直近のみ・早く答える
  immortal            長(horizon) 長(全部)  t=horizon-1… 長命・全履歴・遅く答える（＝v1のimmortal）
  immortal-forgetful  長(horizon) 短(life)  t=horizon-1… 長命だが "直近 life 個" だけで答える（新）

読み取る2つの対比：
  ・immortal vs immortal-forgetful … 寿命も評価時刻も同じ。違いは記憶窓だけ。
        → 「忘却（直近重視）」の効果を単独で測る。
  ・mortal vs immortal-forgetful … 記憶窓は同じ(life)。違いは寿命と評価時刻だけ。
        → 「死／早期コミットそのもの」の効果を測る。

判定の意味：
  ・immortal-forgetful ≈ mortal ≪ immortal なら → 勝因は "忘却" であって "死" ではない。
        v1 の「有限性が勝つ」は誤帰属で、正しくは「直近重視が勝つ」。
  ・mortal ≪ immortal-forgetful なら → 窓をそろえてもなお死が効く＝有限性そのものに手がかり。

依存：標準ライブラリのみ（numpy 不要・API キー不要、無料で動く）。
実行：python -m src.experiment_immortality  （リポジトリ直下から）
"""

from __future__ import annotations

import random
import statistics
import sys


def _use_utf8_output() -> None:
    """Windows コンソールでも日本語・記号を化けさせず出力する。"""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8")
            except Exception:
                pass


def simulate_world(rng: random.Random, horizon: int, drift: float, noise: float,
                   theta0: float = 0.0) -> tuple[list[float], list[float]]:
    """変化する世界を1本生成する。

    θ* はランダムウォークでドリフトし（世界は変わる）、各ステップで観測には
    ノイズが乗る。truths[t] = その時刻の真値、obs[t] = 観測。
    """
    theta = theta0
    truths: list[float] = []
    obs: list[float] = []
    for _ in range(horizon):
        theta += rng.gauss(0.0, drift)          # 世界がドリフトする
        truths.append(theta)
        obs.append(theta + rng.gauss(0.0, noise))  # 観測にノイズ
    return truths, obs


def run_one(seed: int, life: int, horizon: int, drift: float, noise: float
            ) -> tuple[float, float, float]:
    """同一の世界で3群を走らせ、それぞれの誤差（真値との距離）を返す。

    全群とも推定は「持っている履歴の単純平均」。違いは寿命・記憶窓・評価時刻だけ。
      mortal             … obs[:life] を平均、t=life-1 の真値で採点。
      immortal           … obs[:horizon] を平均、t=horizon-1 の真値で採点。
      immortal-forgetful … obs[horizon-life:horizon]（直近 life 個）を平均、
                            t=horizon-1 の真値で採点。immortal と寿命・評価時刻は
                            同一、違いは "記憶窓" だけ（＝忘却の効果を分離する群）。
    """
    rng = random.Random(seed)
    truths, obs = simulate_world(rng, horizon, drift, noise)

    mortal_est = statistics.fmean(obs[:life])
    mortal_err = abs(mortal_est - truths[life - 1])

    immortal_est = statistics.fmean(obs[:horizon])
    immortal_err = abs(immortal_est - truths[horizon - 1])

    # 長命だが直近 life 個だけ覚えている不死（評価時刻は immortal と同じ t=horizon-1）
    forgetful_est = statistics.fmean(obs[horizon - life:horizon])
    forgetful_err = abs(forgetful_est - truths[horizon - 1])

    return mortal_err, immortal_err, forgetful_err


def experiment(drifts: list[float], seeds: int = 2000, life: int = 8,
               horizon: int = 200, noise: float = 1.0) -> None:
    """ドリフトを変えながら3群の誤差を比較し、死と忘却の効果を切り分ける。"""
    _use_utf8_output()
    print("════════ 実験001 v2：交絡を切り分ける（死 vs 忘却）════════")
    print(f"設定：mortalの寿命/窓={life} ／ 不死の地平(horizon)={horizon} ／ "
          f"忘却窓={life} ／ 観測ノイズ={noise} ／ 試行={seeds}seed")
    print("推定はどれも『持っている履歴の単純平均』。誤差 = 答えた瞬間の真値との距離（小さいほど良い）。\n")

    header = (f"{'ドリフト':>8} | {'mortal':>8} | {'immortal':>9} | "
              f"{'immortal-忘却':>13}")
    print(header)
    print("-" * len(header))

    rows = []  # (drift, m_mean, i_mean, f_mean)
    for drift in drifts:
        m_errs, i_errs, f_errs = [], [], []
        for s in range(seeds):
            me, ie, fe = run_one(s, life, horizon, drift, noise)
            m_errs.append(me)
            i_errs.append(ie)
            f_errs.append(fe)
        m_mean = statistics.fmean(m_errs)
        i_mean = statistics.fmean(i_errs)
        f_mean = statistics.fmean(f_errs)
        rows.append((drift, m_mean, i_mean, f_mean))
        print(f"{drift:>8.3f} | {m_mean:>8.4f} | {i_mean:>9.4f} | {f_mean:>13.4f}")

    # ── データ駆動の判定（事前に結論は決めない）──────────────
    drifted = [r for r in rows if r[0] > 0.0]
    print("\n──── 切り分け（非定常域 drift>0 の平均で評価）────")
    if drifted:
        # 対比1：忘却の効果（immortal vs immortal-forgetful、寿命・評価時刻は同一）
        forget_gain = statistics.fmean(r[2] / r[3] for r in drifted)  # immortal / forgetful
        # 対比2：死そのものの効果（mortal vs immortal-forgetful、窓は同一）
        death_gain = statistics.fmean(r[3] / r[1] for r in drifted)   # forgetful / mortal

        print(f"・対比1【忘却の効果】immortal誤差 ÷ immortal-忘却誤差 ＝ {forget_gain:.2f} 倍")
        if forget_gain > 1.2:
            print("    → 寿命・評価時刻を固定し記憶窓だけ短くすると誤差が大きく下がる。")
            print("      ＝勝っていたのは『忘却（直近重視）』。長命でも忘れれば改善する。")
        else:
            print("    → 記憶窓を短くしても大きな改善はない。忘却は主因ではない。")

        print(f"・対比2【死そのものの効果】immortal-忘却誤差 ÷ mortal誤差 ＝ {death_gain:.2f} 倍")
        if death_gain <= 1.15:
            print("    → 記憶窓をそろえると、死なない忘却型が mortal とほぼ同等。")
            print("      ＝『死』そのものは独立した効果を持っていない（窓の差だった）。")
        else:
            print("    → 窓をそろえてもなお mortal が有意に良い。")
            print("      ＝『死／早期コミット』そのものに、忘却では説明できない効果がある。")

        print("\n──── 結論 ────")
        if forget_gain > 1.2 and death_gain <= 1.15:
            print("v1 の主張『有限性が不死に勝つ』は誤帰属。正しくは『直近重視（忘却）が勝つ』。")
            print("死なない不死でも、直近だけ見れば mortal と同等に振る舞える＝勝因は死ではない。")
            print("→ README は『有限性が勝つ』を『忘却（直近重視）が勝つ／死は未分離』に訂正すべき。")
        elif death_gain > 1.15:
            print("窓をそろえてもなお mortal が勝つ＝有限性そのものに手がかりがある可能性。")
            print("→ さらなる切り分け（評価時刻のみを変える対照など）が必要。")
        else:
            print("効果が弱く判然としない。パラメータ（life/horizon/noise）を変えて再検証が必要。")
    else:
        print("（drift>0 の行がないため切り分け不可。drifts に正のドリフトを入れること。）")


if __name__ == "__main__":
    experiment(drifts=[0.0, 0.01, 0.03, 0.05, 0.1, 0.2, 0.5])
