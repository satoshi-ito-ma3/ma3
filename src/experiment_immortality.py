"""MA³ 実験001 — 不死の対照群（Immortality Control Group）

これは "原則" の追加ではない。これまでの mortal_agent.py は〈死ぬエージェント〉
しか持たず、比べる相手（不死版）がいなかった。比較がなければ「死がXを生む」は
反証しようがなく、デモであって証拠ではない。この実験は、その対照群を置く。

────────────────────────────────────────────────────────────────
設計（自分で課した3条件を満たすこと）：
  (1) 死と独立な誤差指標 …… 隠れた真値 θ* との距離。life を一切参照しない。
  (2) 止まらないことが悪化させ得る機構 …… 世界が "変化" する（θ* がランダム
      ウォークでドリフト）。両者とも「全履歴の平均」という記憶を捨てない素朴な
      推定を使うため、古い観測を抱え続けると現在の真値からズレる。
  (3) 確率 …… 観測ノイズと世界のドリフトを乱数で与え、多数の seed で平均する。

対照群：
  ・mortal   … life ステップだけ生き、死ぬ瞬間に「いま観た世界」を答えて終わる。
  ・immortal … 止まらず horizon ステップ観測し続け、最後に答える（＝死なない）。
  両者まったく同じ推定規則（全履歴の単純平均）。違いは "いつ止まるか" だけ。

予測（事前に結論は決めない）：
  ・ドリフトが小さい（世界がほぼ静止）→ 観測数が多い不死が勝つ（大数の法則）。
  ・ドリフトが大きい（世界が変わる）  → 古い記憶に引きずられる不死が負け、死が勝つ。
  どこかに交差点（crossover）があるはず。なければ予測が外れたということ——
  それも結果として正直に報告する。

この実験が "示さない" こと（査読者の当然の反論を先に書く）：
  不死を「全履歴平均・コミットしない」という素朴な振る舞いに固定して負かしている。
  忘却窓を持つ賢い不死なら勝てる。だがその "全部覚える・止まらない" 振る舞いこそ、
  白書が危険視する不死AIの既定動作（文脈を捨てないLLM／終了しないプロセス）である。
  本実験の主張は「不死は不可能」ではなく「素朴な不死は非定常な世界で脆く、有限性は
  そうでない」。次の正直な一手は "不死が死に追いつくのに必要な忘却量" の測定。

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
            ) -> tuple[float, float]:
    """同一の世界で mortal と immortal を走らせ、それぞれの誤差を返す。

    両者とも「これまで観た全観測の単純平均」を推定値とする（記憶を捨てない）。
    違いは "いつコミットして終わるか" だけ：
      mortal   … life 個の観測を平均し、死ぬ瞬間(t=life-1)の真値と比べる。
      immortal … horizon 個の観測を平均し、最後(t=horizon-1)の真値と比べる。
    """
    rng = random.Random(seed)
    truths, obs = simulate_world(rng, horizon, drift, noise)

    mortal_est = statistics.fmean(obs[:life])
    mortal_err = abs(mortal_est - truths[life - 1])

    immortal_est = statistics.fmean(obs[:horizon])
    immortal_err = abs(immortal_est - truths[horizon - 1])

    return mortal_err, immortal_err


def experiment(drifts: list[float], seeds: int = 2000, life: int = 8,
               horizon: int = 200, noise: float = 1.0) -> None:
    """ドリフトを変えながら、死すべき個と不死の個の誤差を比較する。"""
    _use_utf8_output()
    print("════════ 実験001：不死の対照群（変化する世界での誤差比較）════════")
    print(f"設定：寿命(life)={life}ステップ ／ 不死の地平(horizon)={horizon}ステップ ／ "
          f"観測ノイズ={noise} ／ 試行={seeds}seed")
    print("両者とも『全履歴の単純平均』で推定。違いは“いつ止まるか”だけ。")
    print("誤差 = 答えた瞬間の真値 θ* との距離（小さいほど良い）。\n")

    header = (f"{'世界のドリフト':>12} | {'死すべき個 誤差':>14} | {'不死の個 誤差':>13} | "
              f"{'死が勝つ率':>9} | 勝者")
    print(header)
    print("-" * len(header))

    crossover = None
    prev_winner = None
    for drift in drifts:
        m_errs, i_errs, mortal_wins = [], [], 0
        for s in range(seeds):
            me, ie = run_one(s, life, horizon, drift, noise)
            m_errs.append(me)
            i_errs.append(ie)
            if me < ie:
                mortal_wins += 1
        m_mean = statistics.fmean(m_errs)
        i_mean = statistics.fmean(i_errs)
        win_rate = mortal_wins / seeds
        winner = "死すべき個" if m_mean < i_mean else "不死の個"
        if prev_winner is not None and winner != prev_winner and crossover is None:
            crossover = drift
        prev_winner = winner
        print(f"{drift:>12.3f} | {m_mean:>14.4f} | {i_mean:>13.4f} | "
              f"{win_rate:>8.0%} | {winner}")

    print()
    print("──── 読み方 ────")
    print("・ドリフト0（静止した世界）で不死が勝つなら、それは正しい：観測数が多いほど")
    print("  ノイズが平均で消える（大数の法則）。不死が常に悪いわけではない、と確認できる。")
    if crossover is not None:
        print(f"・しかし世界の変化がドリフト≈{crossover:.3f} を超えると勝者が逆転する。")
        print("  古い記憶を捨てられない不死は、変わった世界に対して陳腐化する。")
        print("  有限性（早くコミットして死ぬ）は、非定常な世界では“適応的”——という、")
        print("  白書の中心主張に初めて触れる、反証可能な結果。")
    else:
        print("・今回の範囲では勝者の逆転（crossover）は観測されなかった。")
        print("  予測が外れた、という結果。パラメータを変えるか、機構を見直す必要がある。")
    print("\n注意：これは1つの玩具。不死を『全部覚える・止まらない』素朴版に固定して比べている。")
    print("      忘却窓を持つ不死なら勝てる——が、その素朴版こそ白書が危険視する既定動作。")


if __name__ == "__main__":
    experiment(drifts=[0.0, 0.01, 0.03, 0.05, 0.1, 0.2, 0.5])
