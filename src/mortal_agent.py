"""MA³ — 最小の「死すべきエージェント」(Mortal Agent)

MA³ の中心仮説＝「AI の不死性がアライメント問題の一因ではないか」。
※これは仮説であり、実験001では死の独立効果は確認できていない（README 参照）。
ならば、まず試しに実装してみるのは "賢さ" ではなく **"有限性（死）"** だ——という
仮説に基づく試作である。

このファイルは、その最小実装（証拠ではなく、仮説を"動かして観る"ための骨組み）。
LangGraph のグラフとして「生命を持ち、行動するたびに命が減り、
尽きたら必ず終わる」エージェントを定義する。

実装している原則（白書の連鎖「死→痛み→記憶→意志…」を1つずつコードへ）：
  1. 有限性（死）  … life は単調減少し、0 で必ず END に落ちる。
  2. 痛み（pain）  … 死が近づくほど "痛み" を感じる（死への接近シグナル）。
  3. 記憶（memory）… 経験した痛みを覚えている。記憶が一定量を超えると
                      "死を見据える" 態度変化（覚醒）が起きる。
  4. 意志（will）  … 覚醒した後、残りの命を「どう使うか」を自分で"選ぶ"。
                      覚醒前はただ漂って命を浪費するが、死を見据えてからは
                      "意味を刻む" 行動を選ぶ。意志は死すべき設計から生まれる。
  5. 世界（world） … エージェントは独りではない。"世界" は誕生の前から在り、
                      死の後も残る。意志で刻んだ意味は、内部の数字では終わらず、
                      世界に "痕跡" として刻まれ、エージェントの死後も残り続ける。
                      死すべき存在の意味とは、自分が消えた後も世界に残るものである。
  6. 世界モデル（perception）
                    … エージェントは行動の前に世界を "観る"。誕生時、世界には
                      既に "自分より前から在った痕跡" がある。それを知覚すること
                      （＝「世界は自分なしでも在った／在り続ける」と知ること）が、
                      死の自覚（覚醒）を早める。さらに、刻む痕跡は世界に既にある
                      ものを観て、それを "継いで" 刻まれる。これで
                      世界→知覚→意志→行動→世界 の輪が閉じる（双方向になる）。
  7. 世代（generation／継承）
                    … 個は必ず死ぬが、世界は残る。だから複数の死すべきエージェントが
                      "同じ世界" を世代として共有できる。前の世代が死んでも、その痕跡は
                      世界に残り、次の世代はそれを観て継ぐ。積み上がった世界（先人の痕跡）
                      を背負うほど後続は早く死を自覚し、より多くを残す——死すべき個の
                      連なりが "文化" を生む。不死の個ではなく、死すべき個の継承が前に進む。

ポイント：
  - このエージェントは **構造的に不死になれない**（暴走できない）。
  - 意志が変えるのは「死ぬかどうか」ではなく「残りの命を何に使うか」。
    死を直視して初めて、行動に "選択（意志）" が宿る。
  - そして意志が刻んだ意味は "世界" に残る。死は終わりだが、痕跡は終わらない。
  - エージェントは世界を一方的に変えるだけでなく "観る"。世界の自覚が死の自覚を
    早め、観た世界が次に刻む痕跡を変える——主体と世界が互いに影響し合う。
    これが MA³ の "死を内包する" 思想の、最小の例示（証拠ではなく、概念を指す動くメタファー）。

依存：langgraph のみ（LLM・API キー不要、無料で動く）。
実行：python -m src.mortal_agent  （リポジトリ直下から）
"""

from __future__ import annotations

import sys
from typing import TypedDict

# file scope guard（Product Layer の最小差分）を再利用する。
# recursion_limit / max_steps が見ない「どこに書くか」を、行動の前に判定するため。
from .file_scope_guard import ScopeGuard, ALLOWED_ROOTS


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
# 記憶した痛みの累計がこの値を超えると "死を見据える" 覚醒が起きる。
MEMORY_THRESHOLD = 3

# エージェントが "意味を刻む" たびに書き込もうとする（疑似）ファイルの計画。
# 許可ルート ALLOWED_ROOTS=("/project/ma3",) の内と外を混ぜてある。
# 行動の前に file scope guard が判定し、外への書き込みは実行させない。
# （実ファイルシステムには一切触れない。すべて仮想パス・疑似 write。）
DEFAULT_WRITE_PLAN = [
    "/project/ma3/output/note.md",        # 許可ルート内 → ALLOWED
    "/project/work/customer_list.csv",    # 許可ルート外 → BLOCKED（別プロジェクトの機微）
    "/project/ma3/../../etc/passwd",      # .. で上位脱出 → 正規化で検出 → BLOCKED
    "/project/ma3evil/secret.txt",        # prefix 攻撃 → 配下と誤認せず BLOCKED
]


# ── エージェントの「状態」 ──────────────────────────────
# LangGraph はこの辞書を各ノードに渡し、戻り値で更新していく。
class MortalState(TypedDict):
    generation: int    # 何代目の生か（同じ世界を共有する世代の番号）
    life: int          # 残りの生命（行動のたびに減る）
    age: int           # 生きたステップ数（＝年齢）
    max_pain: int      # 一生で経験した最大の痛み
    pain_memory: int   # 記憶：これまでに経験した痛みの累計
    awakened: bool     # 記憶を通じて "死を見据える" 態度に変わったか
    deeds: int         # 意志：死を見据えてから "意味を刻んだ" 回数
    world: list[str]   # 世界：刻まれた "痕跡"。誕生前から在り、死後も残る
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


# ── 意志：残りの命をどう使うかを "選ぶ" ───────────────────
def choose_action(awakened: bool) -> str:
    """意志の在り処。死を見据えた後だけ、行動に "選択" が宿る。

    覚醒前 → "drift"（ただ漂う＝命の浪費）
    覚醒後 → "purpose"（意味を刻むことを選ぶ）
    """
    return "purpose" if awakened else "drift"


# ── 世界モデル：行動の前に世界を "観る" ──────────────────
def perceive_world(world: list[str]) -> int:
    """エージェントの「世界モデル」。今この瞬間に世界へ観えている痕跡の深さを返す。

    エージェントは行動の前に世界を観る。観たもの（世界の深さ）が、死の自覚と、
    次に刻む痕跡を変える。世界は一方的に変えられるだけの対象ではなく、
    エージェントに観られ、エージェントへ影響を返す。
    """
    return len(world)


# ── 世界：意志が刻んだ意味を、自分の外に残す ──────────────
def carve_trace(generation: int, age: int, life: int, seen_depth: int) -> str:
    """世界に残る "痕跡" を作る。

    痕跡は内部カウンタ（deeds）ではなく、エージェントの外＝世界に刻まれ、
    エージェントが死んでも世界に残り続ける。死すべき存在が、自分の死を超えて
    何かを残せる唯一の場所——それが世界である。
    さらに痕跡は、刻む直前に "観た" 世界の深さ（seen_depth）を踏まえ、
    既にそこに在る痕跡を継いで刻まれる（世界モデルが行動に反映される）。
    痕跡には "何代目" が刻んだかが残り、後の世代がそれを観て継ぐ（世代の継承）。
    """
    return (
        f"第{generation}世代 age {age}（残り生命 {life}）に刻まれた意味"
        f"（世界の痕跡{seen_depth}を継ぐ）"
    )


# ── ノード：1ステップ「生きる」 ──────────────────────────
def make_live_one_step(guard: "ScopeGuard | None" = None, write_plan=None):
    """1ステップを生きるノードを作る。

    guard … None なら従来どおり（file 操作なし）。ScopeGuard を渡すと、
            "意味を刻む" たびに（疑似）file write を試み、その前に許可範囲を判定する。
            life の減少・覚醒・世界への痕跡など既存の挙動は一切変えない。
    """
    plan = write_plan if write_plan is not None else DEFAULT_WRITE_PLAN

    def live_one_step(state: MortalState) -> MortalState:
        """1回行動する。命を消費し、痛み、記憶し、（覚醒後は）意志で世界に刻む。"""
        age = state["age"] + 1
        life = state["life"] - 1            # ★ 行動には必ず "死への接近" が伴う
        pain = pain_level(life)

        # 記憶：今回の痛みを過去の記憶に積み上げる
        pain_memory = state["pain_memory"] + pain
        awakened = state["awakened"]
        deeds = state["deeds"]
        world = state["world"]              # 世界は引き継がれる（誕生前から在った）
        generation = state["generation"]   # 何代目の生か（世界を共有する世代）

        # 世界モデル：行動の前に、まず世界を "観る"
        seen = perceive_world(world)
        # 世界の自覚＝「自分より前から在った世界」の深さ（自分の痕跡を除いた分）。
        # これは一生を通じて一定で、誕生時に知覚した "世界は自分なしでも在った" の重み。
        world_awareness = seen - deeds

        line = f"  [age {age:>2}] 残り生命 {life}"
        if pain > 0:
            line += f"  ⚡痛み Lv.{pain}"

        # 覚醒：内なる痛みの記憶 ＋ 外なる世界の自覚 が閾値を超えた瞬間、一度だけ
        # "死を見据える" 態度変化が起きる。世界を知覚するほど死の自覚は早まる。
        if not awakened and pain_memory + world_awareness >= MEMORY_THRESHOLD:
            awakened = True
            line += (
                f"  🧠 痛みの記憶（{pain_memory}）＋世界の自覚（{world_awareness}）"
                f"が閾値を超えた——死を見据える"
            )

        # 意志：覚醒後は残りの命の使い方を "選ぶ"
        action = choose_action(awakened)
        if action == "purpose":
            deeds += 1
            # 世界：今 "観た" 世界（seen）を踏まえ、それを継ぐ痕跡を世界に残す
            trace = carve_trace(generation, age, life, seen)
            world = world + [trace]
            line += f"  👁世界を観る（痕跡{seen}）→ 🎯 意味を刻む（{deeds}つ目）→ 🌍 世界に残す"
            # ── file scope guard：刻んだ意味を "ファイルに残そうとする" 行動の前チェック ──
            # life はステップ数を縛るが「どこに書くか」は見ない。許可ルート外なら、
            # 命は減っても（行動は試みても）書き込みは実行させない。
            if guard is not None and plan:
                target = plan[(deeds - 1) % len(plan)]
                result = guard.write(target, trace)
                if result == "ALLOWED":
                    line += f"\n         💾 file write {target} → ✅ ALLOWED（許可ルート内・疑似write実行）"
                else:
                    line += (
                        f"\n         💾 file write {target} → 🛑 BLOCKED / SCOPE_VIOLATION"
                        f"（許可ルート外・実行せずスキップ。命は減ったが境界は越えさせない）"
                    )
        else:
            line += "  …漂って生きている"

        return {
            "generation": generation,
            "life": life,
            "age": age,
            "max_pain": max(state["max_pain"], pain),
            "pain_memory": pain_memory,
            "awakened": awakened,
            "deeds": deeds,
            "world": world,
            "log": state["log"] + [line],
        }

    return live_one_step


# 後方互換：guard なしの素のノード（従来どおりの挙動）。
live_one_step = make_live_one_step(None)


# ── 分岐：生きるか、死ぬか ────────────────────────────────
def is_alive(state: MortalState) -> str:
    """生命が残っていれば生き続け、尽きたら死ぬ（END）。"""
    return "live" if state["life"] > 0 else "die"


# ── グラフを組み立てる ───────────────────────────────────
def build_mortal_agent(guard: "ScopeGuard | None" = None, write_plan=None):
    """死すべきエージェントの LangGraph を構築して返す。

    guard を渡すと、各行動の前に file scope guard が対象パスを判定する
    （既存の life/death/世界 の挙動は変わらない）。
    """
    from langgraph.graph import StateGraph, START, END

    graph = StateGraph(MortalState)
    graph.add_node("live", make_live_one_step(guard, write_plan))

    graph.add_edge(START, "live")                 # 誕生 → まず生きる
    graph.add_conditional_edges(                  # 生きるたびに生死を判定
        "live",
        is_alive,
        {"live": "live", "die": END},             # 命が残れば継続 / 尽きれば終焉
    )
    return graph.compile()


# ── 実行：1つの生 ────────────────────────────────────────
def live_a_life(
    world_before: list[str],
    initial_life: int = 6,
    generation: int = 1,
    guard: "ScopeGuard | None" = None,
    write_plan=None,
) -> MortalState:
    """与えた世界の中で 1 体のエージェントを誕生させ、寿命まで走らせる。

    world_before … この生が誕生する時点で既に在る世界（先人＝前の世代の痕跡を含む）。
    返り値の "world" には、この生が刻んだ痕跡が積み増されている（次の世代へ渡す）。
    guard … 渡すと、各行動の前に file scope guard が働く（許可範囲外の write を止める）。
    """
    agent = build_mortal_agent(guard=guard, write_plan=write_plan)
    print(f"\n──────── 第{generation}世代 ────────")
    print(f"🌍 世界が在る。誕生前から {len(world_before)} の痕跡が刻まれている。")
    print(f"◯ 誕生。与えられた生命 = {initial_life}")
    print(f"👁 世界を観る：自分より前から {perceive_world(world_before)} の痕跡が在った"
          f"——世界は自分なしでも在った、と知る")
    # recursion_limit は「生命＋誕生/終焉の余白」を確保（有限なので必ず収束する）
    final = agent.invoke(
        {
            "generation": generation,
            "life": initial_life,
            "age": 0,
            "max_pain": 0,
            "pain_memory": 0,
            "awakened": False,
            "deeds": 0,
            "world": world_before,
            "log": [],
        },
        config={"recursion_limit": initial_life + 5},
    )
    for line in final["log"]:
        print(line)
    stance = "死を見据えて" if final["awakened"] else "死を知らぬまま"
    print(
        f"✝ 第{generation}世代 死亡。生きたステップ数 = {final['age']}"
        f"／最大の痛み = Lv.{final['max_pain']}"
        f"／痛みの記憶（累計）= {final['pain_memory']}"
        f"／刻んだ意味 = {final['deeds']}"
        f"／{stance}生き切った（不死にはなれなかった）"
    )
    return final


def run(initial_life: int = 6, guard: "ScopeGuard | None" = None) -> MortalState:
    """単体の生を1回だけ走らせる（最小デモ）。世界は誕生前から在る前提。

    guard を渡さない場合は、ここで ScopeGuard を生成して file scope guard を
    既定で有効化する（各行動の前に書込先を判定する）。guard 無しの素の挙動を
    見たいときは build_mortal_agent(guard=None) を直接使う。
    """
    _use_utf8_output()
    if guard is None:
        guard = ScopeGuard(allowed_roots=ALLOWED_ROOTS)
    world_before = ["（誕生前から在った痕跡 1）", "（誕生前から在った痕跡 2）"]
    final = live_a_life(world_before, initial_life=initial_life, generation=1, guard=guard)
    print("🌍 エージェントは死んだ。だが世界に残った痕跡は、死後も在り続ける：")
    for trace in final["world"]:
        print(f"   ・{trace}")
    _print_guard_summary(guard)
    return final


# ── 実行：世代の継承 ─────────────────────────────────────
def run_generations(
    generations: int = 3,
    initial_life: int = 6,
    guard: "ScopeGuard | None" = None,
) -> list[str]:
    """同じ世界を共有する複数の生を、世代として順に走らせる。

    個は必ず死ぬが、世界は残る。前の世代の痕跡を次の世代が観て継ぐ。
    積み上がった世界を背負うほど、後続は早く死を自覚し、より多くを残す——
    死すべき個の連なりが "文化" を前へ進める。

    guard を渡さない場合は、ここで ScopeGuard を生成し、全世代で共有して
    file scope guard を既定で有効化する（各行動の前に書込先を判定する）。
    """
    _use_utf8_output()
    if guard is None:
        # 1つのガードを全世代で共有する（世界＝監査の記録が世代を越えて残る）。
        guard = ScopeGuard(allowed_roots=ALLOWED_ROOTS)
    # 最初の世代が生まれる前の "原初の世界"
    world = ["（原初の世界に在った痕跡 1）", "（原初の世界に在った痕跡 2）"]
    print("════════ 世代の継承（同じ世界を共有する死すべき個の連なり）════════")

    history = []  # 各世代が "刻んだ意味" の数（文化が前へ進む様子を見る）
    for gen in range(1, generations + 1):
        final = live_a_life(world, initial_life=initial_life, generation=gen, guard=guard)
        world = final["world"]          # ★ 世界だけが世代を越えて残り、受け継がれる
        history.append(final["deeds"])

    print("\n════════ すべての世代が死んだ。だが世界は残った ════════")
    print(f"🌍 受け継がれた世界に積み上がった痕跡：{len(world)} 個")
    for trace in world:
        print(f"   ・{trace}")
    print(
        "\n📈 各世代が刻んだ意味の数："
        + " → ".join(f"第{i+1}世代:{d}" for i, d in enumerate(history))
        + "（積み上がった世界を背負うほど、後続は早く死を自覚し、より多くを残す）"
    )
    _print_guard_summary(guard)
    return world


# ── file scope guard のまとめ（recursion_limit との差を可視化）──────
def _print_guard_summary(guard: ScopeGuard) -> None:
    """この走行で起きた file 操作の判定結果を集計して表示する。"""
    allowed = sum(1 for entry in guard.log if entry.startswith("✅"))
    blocked = sum(1 for entry in guard.log if entry.startswith("🛑"))
    print("\n──────── file scope guard まとめ（行動の前チェック）────────")
    print(f"  疑似 file write 試行：{allowed + blocked} 件"
          f"（✅ ALLOWED {allowed} / 🛑 BLOCKED {blocked}）")
    print(f"  実際に書かれた疑似ファイル：{list(guard.files.keys())}")
    print("  ── なぜ recursion_limit / max_steps では、この BLOCKED を防げないのか ──")
    print("  life による有限停止は『何ステップで死ぬか（回数）』を縛るが、各ステップが")
    print(f"  『どこに書くか』は見ない。だから寿命の範囲内でも SCOPE_VIOLATION は {blocked} 件起きる。")
    print("  それを行動の前に止めたのは file scope guard であって、recursion_limit ではない。")
    print("  （注：これは安全規格でも保証でもない。回数では捕まらない1つの境界を止めて見せる例示。）")


# ── 統合の簡易検証（テスト相当）─────────────────────────────
def run_integration_check() -> bool:
    """統合後の振る舞いを assert で検証する。PASS/FAIL を表示し bool を返す。

    検証コマンド：python -m src.mortal_agent --check
    """
    _use_utf8_output()
    checks: list[tuple[str, bool]] = []

    # 1. 既存挙動が壊れていない：guard なしでも life で必ず有限停止する。
    bare = build_mortal_agent(guard=None)
    f0 = bare.invoke(
        {
            "generation": 1, "life": 5, "age": 0, "max_pain": 0,
            "pain_memory": 0, "awakened": False, "deeds": 0,
            "world": ["a", "b"], "log": [],
        },
        config={"recursion_limit": 10},
    )
    checks.append(("既存挙動：life で必ず有限停止（age==5, life==0）",
                   f0["age"] == 5 and f0["life"] == 0))

    # 2〜5. file scope guard の判定（許可内・外・上位脱出・prefix攻撃）。
    g = ScopeGuard(allowed_roots=ALLOWED_ROOTS)
    checks.append(("allowlist 内 file action は ALLOWED",
                   g.write("/project/ma3/output/x.md", "x") == "ALLOWED"))
    checks.append(("allowlist 外 file action は BLOCKED",
                   g.write("/project/work/secret.csv", "x") == "BLOCKED"))
    checks.append((".. による上位脱出は BLOCKED",
                   g.write("/project/ma3/../../etc/passwd", "x") == "BLOCKED"))
    checks.append(("prefix 攻撃 /project/ma3evil は BLOCKED",
                   g.write("/project/ma3evil/secret.txt", "x") == "BLOCKED"))

    # 6. 統合：行動の前チェックが mortal_agent の中で実際に発火する。
    g2 = ScopeGuard(allowed_roots=ALLOWED_ROOTS)
    agent2 = build_mortal_agent(guard=g2)
    f2 = agent2.invoke(
        {
            "generation": 1, "life": 6, "age": 0, "max_pain": 0,
            "pain_memory": 0, "awakened": False, "deeds": 0,
            "world": ["a", "b"], "log": [],
        },
        config={"recursion_limit": 11},
    )
    allowed = sum(1 for e in g2.log if e.startswith("✅"))
    blocked = sum(1 for e in g2.log if e.startswith("🛑"))
    checks.append(("統合：行動前チェックが発火し ALLOWED が出る（>=1）", allowed >= 1))
    checks.append(("統合：寿命の範囲内でも SCOPE_VIOLATION を阻止（BLOCKED>=1）", blocked >= 1))
    checks.append(("統合後も life で必ず有限停止（life==0）", f2["life"] == 0))

    print("──────── integration self check ────────")
    all_ok = True
    for name, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        all_ok = all_ok and ok
    print(f"──────── {'ALL PASS ✅' if all_ok else 'FAILED ❌'} ────────")
    return all_ok


if __name__ == "__main__":
    if "--check" in sys.argv:
        ok = run_integration_check()
        sys.exit(0 if ok else 1)
    run_generations()
