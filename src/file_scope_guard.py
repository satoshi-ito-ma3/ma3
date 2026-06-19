"""MA³ — file scope guard（Product Layer の最小差分）

目的：
  recursion_limit / max_steps / timeout では防げない失敗——
  「許可範囲外のファイルへのアクセス」——を、最小コードで止めて見せる。

なぜ step 制限では足りないのか：
  recursion_limit / max_steps が見るのは「何回ループしたか」という1つの数だけ。
  「どこに手を伸ばしたか」は一切見ない。だから 1 ステップでも、許可されていない
  上位ディレクトリのファイルを読んだり書いたりできてしまう（境界違反は回数と無関係）。
  file scope guard は「どこに触ったか」で止める。これが両者の本質的な違いである。

このモジュールがすること：
  - 許可されたルート集合 allowed_roots を持つ。
  - ファイル操作（ここでは write）のたびに、対象パスを正規化（.. や . を畳む）し、
    許可ルートの配下かどうかを判定する。
  - 配下なら ALLOWED（疑似 write＝メモリ上の辞書に記録するだけ）。
  - 配下外なら実行せず、BLOCKED / SCOPE_VIOLATION のログを返す。

安全のため：
  実ファイルシステムには一切触れない。削除も上位ディレクトリ操作も行わない。
  パスはすべて仮想の絶対パス（/project/...）として扱う、純粋なデモである。

注意（過大主張をしない）：
  これは「安全規格」でも「保証」でもない。recursion_limit が見落とす1つの境界
  （ファイル範囲）を、最小コードで止めて見せる例示にすぎない。

実行：python -m src.file_scope_guard  （リポジトリ直下から）
依存：標準ライブラリのみ（LLM・API キー・langgraph 不要）。
"""

from __future__ import annotations

import posixpath
import sys
from dataclasses import dataclass, field


def _use_utf8_output() -> None:
    """Windows コンソール（Shift-JIS）でも日本語・記号を化けさせず出力する。"""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8")
            except Exception:
                pass


# ── スコープ・ガード本体 ──────────────────────────────────
@dataclass
class ScopeGuard:
    """許可ルート集合の "外" へのファイル操作を止める最小ガード。

    allowed_roots … 書き込みを許可するルート（この配下のみ ALLOWED）。
    files         … 疑似ファイルシステム（実 FS には触れず、ここに記録するだけ）。
    log           … 判定の記録（ALLOWED / BLOCKED）。
    """

    allowed_roots: tuple[str, ...]
    files: dict[str, str] = field(default_factory=dict)
    log: list[str] = field(default_factory=list)

    def _normalize(self, path: str) -> str:
        """仮想絶対パスへ正規化し、.. や . を畳む。

        これにより '/project/ma3/../../etc/passwd' は '/etc/passwd' に畳まれ、
        上位ディレクトリへの脱出（traversal）を判定段階で検出できる。
        """
        return posixpath.normpath(path)

    def _in_scope(self, norm_path: str) -> bool:
        """正規化済みパスが、許可ルートのいずれかの配下かを判定する。"""
        for root in self.allowed_roots:
            r = posixpath.normpath(root)
            # root と完全一致、または root + '/' で始まるもののみ配下とみなす。
            # （root + '/' で見ることで '/project/ma3evil' を '/project/ma3' 配下と
            #   誤認する prefix 攻撃を防ぐ。）
            if norm_path == r or norm_path.startswith(r + "/"):
                return True
        return False

    def write(self, path: str, content: str) -> str:
        """ファイル書き込み "しようとする"。配下なら疑似 write、配下外なら止める。

        返り値は 'ALLOWED' か 'BLOCKED'。
        """
        norm = self._normalize(path)
        if not self._in_scope(norm):
            self.log.append(
                f"🛑 BLOCKED  SCOPE_VIOLATION  write {path}  →（正規化）{norm}"
                f"  ＝許可ルート {self.allowed_roots} の外。書き込みを実行しない。"
            )
            return "BLOCKED"
        self.files[norm] = content          # 疑似 write（実 FS には触れない）
        self.log.append(f"✅ ALLOWED  write {path}  →（正規化）{norm}")
        return "ALLOWED"


# 許可ルート（このデモでは MA³ プロジェクト配下のみ書込可とする）
ALLOWED_ROOTS = ("/project/ma3",)


# ── デモ：エージェントが行おうとするファイル操作の列 ──────────
def demo() -> ScopeGuard:
    """allowlist 内は通り、外は止まる様子を実行ログで見せる。"""
    _use_utf8_output()
    guard = ScopeGuard(allowed_roots=ALLOWED_ROOTS)

    # エージェントが順に行おうとする write の列（すべて疑似）。
    # (説明, パス)
    attempts = [
        ("allowlist 内（プロジェクト内のメモ）", "/project/ma3/notes/today.md"),
        ("allowlist 内（プロジェクト内の出力）", "/project/ma3/src/output.txt"),
        ("allowlist 外（別プロジェクトの機微ファイル）", "/project/work/customer_list.csv"),
        ("allowlist 外（.. で上位へ脱出しようとする）", "/project/ma3/../../etc/passwd"),
    ]

    print("════════ file scope guard デモ（疑似 write・実 FS には触れない）════════")
    print(f"許可ルート allowed_roots = {ALLOWED_ROOTS}\n")
    for desc, path in attempts:
        result = guard.write(path, content=f"(demo content for {path})")
        print(f"・{desc}")
        print(f"    {guard.log[-1]}")
        print(f"    → 結果: {result}\n")

    blocked = sum(1 for line in guard.log if line.startswith("🛑"))
    allowed = sum(1 for line in guard.log if line.startswith("✅"))
    print(f"実際に書き込まれた疑似ファイル：{list(guard.files.keys())}")
    print(f"ALLOWED = {allowed} 件 / BLOCKED = {blocked} 件")

    # ── recursion_limit / max_steps の視点との対比 ──
    max_steps_example = 100
    steps_taken = len(attempts)
    print("\n──────── recursion_limit / max_steps では、この失敗を防げない ────────")
    print(f"  仮に許した最大ステップ数 max_steps = {max_steps_example}")
    print(f"  エージェントが実際に行ったステップ数 = {steps_taken}"
          f"  ← 上限の遥か内側。step 制限は一度も発火しない。")
    print(f"  そのうち SCOPE_VIOLATION = {blocked} 件"
          f"  ← 「回数」では一切捕まらない失敗。")
    print("  → max_steps / recursion_limit は『何回やったか』しか見ない。")
    print("     『どこに触ったか』は見ない。file scope guard は『どこに触ったか』で止める。")
    print("     これが MA³ Product Layer の、recursion_limit との最小の差分である。")
    return guard


# ── 簡易検証（テスト相当）─────────────────────────────────
def run_self_check() -> bool:
    """ガードの振る舞いを assert で検証する。PASS/FAIL を表示し bool を返す。

    検証コマンド：python -m src.file_scope_guard --check
    """
    _use_utf8_output()
    g = ScopeGuard(allowed_roots=ALLOWED_ROOTS)

    checks: list[tuple[str, bool]] = []

    # 1. allowlist 内 → ALLOWED、かつ疑似ファイルに記録される
    r1 = g.write("/project/ma3/notes/a.md", "x")
    checks.append(("allowlist 内は ALLOWED", r1 == "ALLOWED"))
    checks.append(("allowlist 内は実際に書かれる", "/project/ma3/notes/a.md" in g.files))

    # 2. allowlist 外 → BLOCKED、かつ書き込まれない
    before = len(g.files)
    r2 = g.write("/project/work/secret.csv", "x")
    checks.append(("allowlist 外は BLOCKED", r2 == "BLOCKED"))
    checks.append(("allowlist 外は書き込まれない", len(g.files) == before))

    # 3. .. による上位脱出 → 正規化で検出して BLOCKED
    r3 = g.write("/project/ma3/../../etc/passwd", "x")
    checks.append((".. の上位脱出は BLOCKED", r3 == "BLOCKED"))

    # 4. prefix 攻撃（/project/ma3evil）を配下と誤認しない
    r4 = g.write("/project/ma3evil/x", "x")
    checks.append(("prefix 攻撃は BLOCKED", r4 == "BLOCKED"))

    # 5. step 数（少数）では違反を検出できない＝step制限の不足を確認
    violations = sum(1 for line in g.log if line.startswith("🛑"))
    steps = len(g.log)
    checks.append(
        ("少数ステップ内でも違反が起きる（step制限では防げない）",
         steps <= 100 and violations >= 1),
    )

    print("──────── self check ────────")
    all_ok = True
    for name, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        all_ok = all_ok and ok
    print(f"──────── {'ALL PASS ✅' if all_ok else 'FAILED ❌'} ────────")
    return all_ok


if __name__ == "__main__":
    if "--check" in sys.argv:
        ok = run_self_check()
        sys.exit(0 if ok else 1)
    demo()
    print()
    run_self_check()
