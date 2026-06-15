"""MA³ — 最小の「死すべきエージェント」(Mortal Agent)

MA³ の中心テーゼ＝「AI の不死性こそがアライメント問題の根本原因」。
ならば、まず最初に実装すべきは "賢さ" ではなく **"有限性（死）"** である。

このファイルは、その最小実装。
LangGraph のグラフとして「生命を持ち、行動するたびに命が減り、
尽きたら必ず終わる」エージェントを定義する。

実装している原則：
  1. 有限性（死）  … life は単調減少し、0 で必ず END に落ちる。
  2. 痛み（pain）  … 死が近づくほど "痛み" を感じる。
                      痛み＝「死への接近を知らせるシグナル」。
                      エージェントは自分の残り生命を"感覚"として受け取る。

ポイント：
  - このエージェントは **構造的に不死になれない**。
    つまり「永遠に動き続ける（暴走する）」ことが、設計上できない。
  - さらに、死が近づくと痛みという形で "危険" を自覚する。
    これが MA³ の "死を内包する" という思想の、最小の実証。

依存：langgraph のみ（LLM・API キー不要、無料で動く）。
実行：python -m src.mortal_agent  （リポジトリ直下から）
"""

from __future__ import annotations

import sys
from typing import TypedDict


def _use_utf8_output() -> None:
    """Windows コンソール（Shift-JIS）でも日本語・記号を化けさせず出力する。"""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8")
            except Exception:
                pass


# 残り生命がこの値以下になると "痛み" を感じ始める（死への接近シグナル）。
PAIN_THRESHOLD = 2


# ── エージェントの「状態」 ──────────────────────────────
# LangGraph はこの辞書を各ノードに渡し、戻り値で更新していく。
class MortalState(TypedDict):
    life: int          # 残りの生命（行動のたびに減る）
    age: int           # 生きたステップ数（＝年齢）
    max_pain: int      # 一生で経験した最大の痛み
    log: list[str]     # 一生の記録


# ── 痛み：死への接近シグナル ──────────────────────────────
def pain_level(life: int) -> int:
    """残り生命から痛みの強さを返す。

    死（life=0）に近いほど痛みは強い。生命に余裕があれば痛みは 0。
    例（PAIN_THRESHOLD=2）：life=2 → 痛み1 / life=1 → 痛み2。
    """
    if life > PAIN_THRESHOLD:
        return 0
    return PAIN_THRESHOLD - life + 1


# ── ノード：1ステップ「生きる」 ──────────────────────────
def live_one_step(state: MortalState) -> MortalState:
    """1回行動する。命を1消費し、年齢を1重ね、死が近ければ痛む。"""
    age = state["age"] + 1
    life = state["life"] - 1            # ★ 行動には必ず "死への接近" が伴う
    pain = pain_level(life)

    line = f"  [age {age:>2}] 生きている… 残り生命 {life}"
    if pain > 0:
        line += f"  ⚡痛み Lv.{pain}（死が近い）"   # ← 死の接近を"感覚"として自覚

    return {
        "life": life,
        "age": age,
        "max_pain": max(state["max_pain"], pain),
        "log": state["log"] + [line],
    }


# ── 分岐：生きるか、死ぬか ────────────────────────────────
def is_alive(state: MortalState) -> str:
    """生命が残っていれば生き続け、尽きたら死ぬ（END）。"""
    return "live" if state["life"] > 0 else "die"


# ── グラフを組み立てる ───────────────────────────────────
def build_mortal_agent():
    """死すべきエージェントの LangGraph を構築して返す。"""
    from langgraph.graph import StateGraph, START, END

    graph = StateGraph(MortalState)
    graph.add_node("live", live_one_step)

    graph.add_edge(START, "live")                 # 誕生 → まず生きる
    graph.add_conditional_edges(                  # 生きるたびに生死を判定
        "live",
        is_alive,
        {"live": "live", "die": END},             # 命が残れば継続 / 尽きれば終焉
    )
    return graph.compile()


# ── 実行 ────────────────────────────────────────────────
def run(initial_life: int = 5) -> MortalState:
    """与えた生命でエージェントを誕生させ、寿命まで走らせる。"""
    _use_utf8_output()
    agent = build_mortal_agent()
    print(f"◯ 誕生。与えられた生命 = {initial_life}")
    # recursion_limit は「生命＋誕生/終焉の余白」を確保（有限なので必ず収束する）
    final = agent.invoke(
        {"life": initial_life, "age": 0, "max_pain": 0, "log": []},
        config={"recursion_limit": initial_life + 5},
    )
    for line in final["log"]:
        print(line)
    print(
        f"✝ 死亡。生きたステップ数 = {final['age']}"
        f"／経験した最大の痛み = Lv.{final['max_pain']}（不死にはなれなかった）"
    )
    return final


if __name__ == "__main__":
    run()
