# -*- coding: utf-8 -*-
"""
===============================================================================
ĐỒ ÁN TRÍ TUỆ NHÂN TẠO — ĐỀ TÀI 5
Game Tic-Tac-Toe (Người vs Máy) với thuật toán Minimax & Adversarial Search

Ứng dụng Python ĐỘC LẬP (không cần Jupyter notebook), có GIAO DIỆN tkinter.

CÁCH CHẠY:
    python3 tictactoe_minimax.py

Yêu cầu: Python 3 (đã có sẵn thư viện chuẩn: tkinter, math, random, time).
Không cần cài thêm gì.

NỘI DUNG:
    - Engine Tic-Tac-Toe n x n (k quân liên tiếp để thắng).
    - 7 giải thuật tìm kiếm: AND-OR Search, Minimax, Alpha-Beta, Alpha-Beta +
      Move Ordering, Alpha-Beta + Bảng chuyển vị + Đối xứng, Heuristic
      Alpha-Beta (cut-off + Eval h1/h2 + forward pruning), Pure Monte Carlo
      và Monte Carlo Tree Search (UCB1).
    - Giao diện 3 CHẾ ĐỘ CHƠI: Người vs Máy, Người vs Người, Máy vs Máy.
    - Chọn ĐỘ KHÓ (Dễ / Vừa / Khó) hoặc chọn thẳng từng thuật toán.
===============================================================================
"""

import math
import time
import random
import tracemalloc
import threading
from collections import namedtuple
from dataclasses import dataclass

# =============================================================================
#  PHẦN A — ENGINE TRÒ CHƠI (Definition of a Game)
# =============================================================================
EMPTY, X, O = '.', 'X', 'O'
State = namedtuple('State', ['board', 'to_move'])   # trạng thái s = (bàn cờ, người đến lượt)
COLORS = {X: '#1f5fbf', O: '#d62728'}


def other(p):
    """Đối thủ của p (X <-> O)."""
    return O if p == X else X


class TicTacToe:
    """Bàn cờ n x n, thắng khi có k quân liên tiếp (hàng/cột/chéo)."""

    def __init__(self, n=3, k=None, first=X, fast_win=False):
        k = n if k is None else k
        if not (3 <= k <= n):
            raise ValueError('Cần 3 <= k <= n')
        self.n, self.k, self.first, self.fast_win = n, k, first, fast_win
        # fast_win=True -> Utility = ±(1 + số ô trống): ưu tiên thắng nhanh
        self.max_utility = (1 + n * n) if fast_win else 1
        self.lines = self._build_lines()                 # mọi "đường thắng"
        self.cell_lines = [[] for _ in range(n * n)]
        for li, line in enumerate(self.lines):
            for c in line:
                self.cell_lines[c].append(li)
        self.position_weight = [len(ls) for ls in self.cell_lines]
        # Move ordering: ô nằm trên nhiều đường thắng (giữa, góc...) xét trước
        self.move_order = sorted(range(n * n), key=lambda i: -self.position_weight[i])
        self.symmetries = self._build_symmetries()       # 8 phép đối xứng hình vuông

    def _build_lines(self):
        n, k = self.n, self.k
        lines = []
        for r in range(n):
            for c in range(n):
                for dr, dc in [(0, 1), (1, 0), (1, 1), (1, -1)]:
                    er, ec = r + dr * (k - 1), c + dc * (k - 1)
                    if 0 <= er < n and 0 <= ec < n:
                        lines.append(tuple((r + dr * i) * n + (c + dc * i) for i in range(k)))
        return lines

    def _build_symmetries(self):
        n = self.n
        transforms = [
            lambda r, c: (r, c),
            lambda r, c: (c, n - 1 - r),
            lambda r, c: (n - 1 - r, n - 1 - c),
            lambda r, c: (n - 1 - c, r),
            lambda r, c: (r, n - 1 - c),
            lambda r, c: (n - 1 - r, c),
            lambda r, c: (c, r),
            lambda r, c: (n - 1 - c, n - 1 - r),
        ]
        perms = []
        for t in transforms:
            perm = [0] * (n * n)
            for r in range(n):
                for c in range(n):
                    nr, nc = t(r, c)
                    perm[nr * n + nc] = r * n + c
            perms.append(perm)
        return perms

    # ---- Definition of a Game ----
    def initial_state(self):                      # s0
        return State(EMPTY * (self.n * self.n), self.first)

    def empty_cells(self, board):
        return [i for i, ch in enumerate(board) if ch == EMPTY]

    def actions(self, s):                         # Actions(s)
        if self.terminal(s):
            return []
        return self.empty_cells(s.board)

    def result(self, s, a):                       # Result(s, a) - transition model
        if s.board[a] != EMPTY:
            raise ValueError('Hành động không hợp lệ: ô %d đã có quân' % a)
        new_board = s.board[:a] + s.to_move + s.board[a + 1:]
        return State(new_board, other(s.to_move))

    def winning_line(self, board):
        for line in self.lines:
            ch = board[line[0]]
            if ch != EMPTY:
                for i in line[1:]:
                    if board[i] != ch:
                        break
                else:
                    return line
        return None

    def winner(self, board):
        line = self.winning_line(board)
        return board[line[0]] if line else None

    def terminal(self, s):                        # Terminal(s)
        return EMPTY not in s.board or self.winner(s.board) is not None

    def utility(self, s, player):                 # Utility(s) cho 'player' (Max)
        w = self.winner(s.board)
        if w is None:
            return 0
        score = (1 + s.board.count(EMPTY)) if self.fast_win else 1
        return score if w == player else -score

    def canonical(self, board):
        """Dạng chuẩn: chuỗi nhỏ nhất trong 8 ảnh đối xứng (gộp trạng thái đối xứng)."""
        return min(''.join(board[p] for p in perm) for perm in self.symmetries)

    def rc(self, a):
        return divmod(a, self.n)

    def pretty(self, board):
        n = self.n
        return '\n'.join(' '.join(board[r * n:(r + 1) * n]) for r in range(n))


# =============================================================================
#  PHẦN B — CÔNG CỤ ĐO LƯỜNG
# =============================================================================
INF = math.inf


class SearchLimitExceeded(Exception):
    """Ném ra khi vượt quá số nút cho phép (tránh treo ở bàn lớn)."""


class Stats:
    def __init__(self, max_nodes=None):
        self.nodes = 0
        self.max_depth = 0
        self.cutoffs = 0
        self.tt_hits = 0
        self.tt_size = 0
        self.max_nodes = max_nodes
        self.time_ms = 0.0
        self.peak_kb = None

    def visit(self, depth):
        self.nodes += 1
        if depth > self.max_depth:
            self.max_depth = depth
        if self.max_nodes is not None and self.nodes > self.max_nodes:
            raise SearchLimitExceeded('Vượt quá %d nút' % self.max_nodes)


@dataclass
class SearchResult:
    algorithm: str
    move: int
    value: float
    move_values: dict
    stats: Stats


# =============================================================================
#  PHẦN C — HAI HÀM HEURISTIC (Eval)
# =============================================================================
def h1_open_lines(game, board, player):
    """Số đường còn mở cho player trừ số đường còn mở cho đối thủ (chuẩn hóa về (-1,1))."""
    opp = other(player)
    mine = theirs = 0
    for line in game.lines:
        cells = [board[i] for i in line]
        if opp not in cells:
            mine += 1
        if player not in cells:
            theirs += 1
    return (mine - theirs) / (len(game.lines) + 1)


def h2_weighted(game, board, player):
    """Đường chỉ có c quân một bên được 10^(c-1) điểm cho bên đó (nhận ra mối đe dọa)."""
    opp = other(player)
    score = 0
    for line in game.lines:
        cells = [board[i] for i in line]
        cp, co = cells.count(player), cells.count(opp)
        if cp and not co:
            score += 10 ** (cp - 1)
        elif co and not cp:
            score -= 10 ** (co - 1)
    return score / (len(game.lines) * 10 ** (game.k - 2) + 1)


HEURISTICS = {'h1': h1_open_lines, 'h2': h2_weighted}


# =============================================================================
#  PHẦN D — CÁC GIẢI THUẬT TÌM KIẾM
# =============================================================================

# --- 1) AND-OR Search (coi nước đối thủ là hành động không xác định) ----------
def and_or_search(game, state, goal='win', stats=None):
    """Trả về kế hoạch có điều kiện [a, {b: kế_hoạch_con}] hoặc None (thất bại)."""
    stats = stats or Stats()
    player = state.to_move
    success = (lambda u: u > 0) if goal == 'win' else (lambda u: u >= 0)

    def or_search(s, path, depth):                 # lượt Max: chọn 1 hành động
        stats.visit(depth)
        if game.terminal(s):
            return [] if success(game.utility(s, player)) else None
        if s in path:
            return None
        for a in game.empty_cells(s.board):
            plan = and_search(game.result(s, a), path | {s}, depth + 1)
            if plan is not None:
                return [a, plan]
        return None

    def and_search(s, path, depth):                # lượt Min: xét MỌI nước
        stats.visit(depth)
        if game.terminal(s):
            return {} if success(game.utility(s, player)) else None
        plans = {}
        for b in game.empty_cells(s.board):
            p = or_search(game.result(s, b), path | {s}, depth + 1)
            if p is None:
                return None
            plans[b] = p
        return plans

    return or_search(state, frozenset(), 0)


# --- 2) Minimax Search --------------------------------------------------------
def minimax_search(game, state, stats):
    player = state.to_move

    def max_value(s, depth):
        stats.visit(depth)
        if game.terminal(s):
            return game.utility(s, player)
        v = -INF
        for a in game.empty_cells(s.board):
            v = max(v, min_value(game.result(s, a), depth + 1))
        return v

    def min_value(s, depth):
        stats.visit(depth)
        if game.terminal(s):
            return game.utility(s, player)
        v = INF
        for a in game.empty_cells(s.board):
            v = min(v, max_value(game.result(s, a), depth + 1))
        return v

    stats.visit(0)
    move_values = {}
    for a in game.empty_cells(state.board):
        move_values[a] = (min_value(game.result(state, a), 1), True)
    best = max(move_values, key=lambda a: move_values[a][0])
    return best, move_values[best][0], move_values


# --- 3 & 4) Alpha-Beta Pruning (+ Move Ordering) ------------------------------
def alphabeta_search(game, state, stats, move_ordering=False):
    player = state.to_move
    if move_ordering:
        moves = lambda board: [a for a in game.move_order if board[a] == EMPTY]
    else:
        moves = game.empty_cells

    def alpha_beta(s, alpha, beta, max_player, depth):
        stats.visit(depth)
        if game.terminal(s):
            return game.utility(s, player)
        if max_player:
            max_val = -INF
            for a in moves(s.board):
                max_val = max(max_val, alpha_beta(game.result(s, a), alpha, beta, False, depth + 1))
                alpha = max(alpha, max_val)
                if alpha >= beta:
                    stats.cutoffs += 1
                    break
            return max_val
        else:
            min_val = INF
            for a in moves(s.board):
                min_val = min(min_val, alpha_beta(game.result(s, a), alpha, beta, True, depth + 1))
                beta = min(beta, min_val)
                if alpha >= beta:
                    stats.cutoffs += 1
                    break
            return min_val

    stats.visit(0)
    alpha, best, best_v, move_values = -INF, None, -INF, {}
    for a in moves(state.board):
        v = alpha_beta(game.result(state, a), alpha, INF, False, 1)
        move_values[a] = (v, v > alpha)
        if v > best_v:
            best, best_v = a, v
        alpha = max(alpha, v)
    return best, best_v, move_values


# --- 5) Alpha-Beta + Move Ordering + Bảng chuyển vị (TT) + Đối xứng -----------
EXACT, LOWER, UPPER = 0, 1, 2


def alphabeta_tt_search(game, state, stats, use_symmetry=True):
    player = state.to_move
    tt = {}
    key_of = game.canonical if use_symmetry else (lambda b: b)

    def ordered(board):
        return [a for a in game.move_order if board[a] == EMPTY]

    def alpha_beta(s, alpha, beta, depth):
        stats.visit(depth)
        if game.terminal(s):
            return game.utility(s, player)
        key = (key_of(s.board), s.to_move)
        alpha0, beta0 = alpha, beta
        entry = tt.get(key)
        if entry is not None:
            stats.tt_hits += 1
            v, flag = entry
            if flag == EXACT:
                return v
            if flag == LOWER:
                alpha = max(alpha, v)
            else:
                beta = min(beta, v)
            if alpha >= beta:
                return v
        if s.to_move == player:
            v = -INF
            for a in ordered(s.board):
                v = max(v, alpha_beta(game.result(s, a), alpha, beta, depth + 1))
                alpha = max(alpha, v)
                if alpha >= beta:
                    stats.cutoffs += 1
                    break
        else:
            v = INF
            for a in ordered(s.board):
                v = min(v, alpha_beta(game.result(s, a), alpha, beta, depth + 1))
                beta = min(beta, v)
                if alpha >= beta:
                    stats.cutoffs += 1
                    break
        flag = UPPER if v <= alpha0 else LOWER if v >= beta0 else EXACT
        tt[key] = (v, flag)
        return v

    stats.visit(0)
    groups = {}                                           # gộp nước cho ra bàn cờ đối xứng
    for a in ordered(state.board):
        groups.setdefault(key_of(game.result(state, a).board), []).append(a)
    alpha, best, best_v, move_values = -INF, None, -INF, {}
    for group in groups.values():
        rep = group[0]
        v = alpha_beta(game.result(state, rep), alpha, INF, 1)
        for a in group:
            move_values[a] = (v, v > alpha)
        if v > best_v:
            best, best_v = rep, v
        alpha = max(alpha, v)
    stats.tt_size = len(tt)
    return best, best_v, move_values


# --- 6) Heuristic Alpha-Beta (cut-off + Eval + forward pruning/beam) ----------
def heuristic_alphabeta_search(game, state, stats, depth_limit=3, heuristic='h2', beam=0):
    player = state.to_move
    h = HEURISTICS[heuristic] if isinstance(heuristic, str) else heuristic

    def candidates(s, max_player):
        moves = [a for a in game.move_order if s.board[a] == EMPTY]
        if beam and len(moves) > beam:                    # forward pruning
            scored = []
            for a in moves:
                t = game.result(s, a)
                v = game.utility(t, player) if game.terminal(t) else h(game, t.board, player)
                scored.append((v, a))
            scored.sort(key=lambda p: -p[0] if max_player else p[0])
            moves = [a for _, a in scored[:beam]]
        return moves

    def alpha_beta(s, depth, alpha, beta, max_player):
        stats.visit(depth_limit - depth)
        if game.terminal(s):
            return game.utility(s, player)
        if depth == 0:                                    # cut-off -> Eval
            return h(game, s.board, player)
        if max_player:
            max_val = -INF
            for a in candidates(s, True):
                max_val = max(max_val, alpha_beta(game.result(s, a), depth - 1, alpha, beta, False))
                alpha = max(alpha, max_val)
                if alpha >= beta:
                    stats.cutoffs += 1
                    break
            return max_val
        else:
            min_val = INF
            for a in candidates(s, False):
                min_val = min(min_val, alpha_beta(game.result(s, a), depth - 1, alpha, beta, True))
                beta = min(beta, min_val)
                if alpha >= beta:
                    stats.cutoffs += 1
                    break
            return min_val

    stats.visit(0)
    alpha, best, best_v, move_values = -INF, None, -INF, {}
    for a in candidates(state, True):
        v = alpha_beta(game.result(state, a), depth_limit - 1, alpha, INF, False)
        move_values[a] = (v, v > alpha)
        if v > best_v:
            best, best_v = a, v
        alpha = max(alpha, v)
    return best, best_v, move_values


# --- 7) Pure Monte Carlo & Monte Carlo Tree Search (UCB1) --------------------
def _rng(seed):
    return random.Random(seed if seed is not None else random.getrandbits(32))


def _playout(game, s, rng, stats, depth):
    while not game.terminal(s):
        s = game.result(s, rng.choice(game.empty_cells(s.board)))
        depth += 1
        stats.visit(depth)
    return s


def pure_monte_carlo_search(game, state, stats, n_playouts=1000, seed=None):
    rng, player = _rng(seed), state.to_move
    moves = game.empty_cells(state.board)
    total = {a: 0.0 for a in moves}
    count = {a: 0 for a in moves}
    stats.visit(0)
    for i in range(n_playouts):
        a = moves[i % len(moves)]
        s = game.result(state, a)
        stats.visit(1)
        end = _playout(game, s, rng, stats, 1)
        total[a] += game.utility(end, player) / game.max_utility
        count[a] += 1
    move_values = {a: (total[a] / count[a], True) for a in moves if count[a]}
    best = max(move_values, key=lambda a: move_values[a][0])
    return best, move_values[best][0], move_values


class MCTSNode:
    __slots__ = ('state', 'parent', 'action', 'children', 'untried', 'N', 'U')

    def __init__(self, game, state, parent=None, action=None):
        self.state, self.parent, self.action = state, parent, action
        self.children = []
        self.untried = [] if game.terminal(state) else game.empty_cells(state.board)
        self.N = 0
        self.U = 0.0


def mcts_search(game, state, stats, n_playouts=1000, c=math.sqrt(2), seed=None):
    rng, player = _rng(seed), state.to_move
    root = MCTSNode(game, state)
    size = 1

    def ucb1(nd):
        return nd.U / nd.N + c * math.sqrt(math.log(nd.parent.N) / nd.N)

    for _ in range(n_playouts):
        node, depth = root, 0
        while not node.untried and node.children:          # Selection
            node = max(node.children, key=ucb1)
            depth += 1
        if node.untried:                                   # Expansion
            a = node.untried.pop(rng.randrange(len(node.untried)))
            child = MCTSNode(game, game.result(node.state, a), node, a)
            node.children.append(child)
            node, depth, size = child, depth + 1, size + 1
        stats.visit(depth)
        end = _playout(game, node.state, rng, stats, depth)  # Simulation
        u = game.utility(end, player) / game.max_utility
        while node is not None:                            # Backpropagation
            node.N += 1
            if node.parent is not None:
                node.U += u if node.parent.state.to_move == player else -u
            node = node.parent
    stats.tt_size = size
    move_values = {ch.action: (ch.U / ch.N, True) for ch in root.children}
    best = max(root.children, key=lambda ch: ch.N).action
    return best, move_values[best][0], move_values


# =============================================================================
#  PHẦN E — HÀM solve() CHUNG + ĐỘ KHÓ
# =============================================================================
ALGORITHMS = {
    'minimax': 'Minimax Search',
    'alphabeta': 'Alpha-Beta Pruning',
    'alphabeta_mo': 'Alpha-Beta + Move Ordering',
    'alphabeta_tt': 'Alpha-Beta + Move Ordering + TT + Đối xứng',
    'heuristic_ab': 'Heuristic Alpha-Beta Tree Search',
    'pure_mc': 'Pure Monte Carlo Search',
    'mcts': 'Monte Carlo Tree Search (UCB1)',
}
EXACT_ALGOS = {'minimax', 'alphabeta', 'alphabeta_mo', 'alphabeta_tt'}
MC_ALGOS = {'pure_mc', 'mcts'}


def solve(game, state, algo='alphabeta_tt', depth_limit=3, heuristic='h2', beam=0,
          n_playouts=1000, c=math.sqrt(2), seed=None, max_nodes=None,
          measure_memory=False, use_symmetry=True):
    stats = Stats(max_nodes)
    runners = {
        'minimax': lambda: minimax_search(game, state, stats),
        'alphabeta': lambda: alphabeta_search(game, state, stats),
        'alphabeta_mo': lambda: alphabeta_search(game, state, stats, move_ordering=True),
        'alphabeta_tt': lambda: alphabeta_tt_search(game, state, stats, use_symmetry),
        'heuristic_ab': lambda: heuristic_alphabeta_search(game, state, stats, depth_limit, heuristic, beam),
        'pure_mc': lambda: pure_monte_carlo_search(game, state, stats, n_playouts, seed),
        'mcts': lambda: mcts_search(game, state, stats, n_playouts, c, seed),
    }
    if measure_memory:
        tracemalloc.start()
    t0 = time.perf_counter()
    try:
        move, value, move_values = runners[algo]()
    finally:
        stats.time_ms = (time.perf_counter() - t0) * 1000
        if measure_memory:
            stats.peak_kb = tracemalloc.get_traced_memory()[1] / 1024
            tracemalloc.stop()
    name = ALGORITHMS[algo]
    if algo == 'heuristic_ab':
        name += ' (d=%d, %s%s)' % (depth_limit, heuristic, (', beam=%d' % beam) if beam else '')
    elif algo in MC_ALGOS:
        name += ' (N=%d)' % n_playouts
    return SearchResult(name, move, value, move_values, stats)


def difficulty_config(n, level):
    """Quy đổi mức độ khó -> cấu hình AI (kèm 'blunder' = tỉ lệ AI đi ngẫu nhiên)."""
    if level == 'easy':
        return dict(algo='alphabeta_tt' if n == 3 else 'heuristic_ab', depth_limit=1, heuristic='h1',
                    beam=0, n_playouts=200, c=1.41, blunder=0.5, label='🟢 Dễ')
    if level == 'medium':
        return dict(algo='heuristic_ab', depth_limit=3 if n == 3 else 2, heuristic='h2',
                    beam=0, n_playouts=800, c=1.41, blunder=0.18, label='🟡 Vừa')
    return dict(algo='alphabeta_tt' if n == 3 else 'heuristic_ab', depth_limit=4, heuristic='h2',
                beam=0, n_playouts=3000, c=1.41, blunder=0.0, label='🔴 Khó')


def fmt_value(v, exact=True):
    p = '' if exact else '≤'
    if v == 0:
        return p + '0'
    if float(v).is_integer():
        return p + ('%+g' % v)
    return p + ('%+.2f' % v)


def value_color(v, exact, approx=False):
    if not exact:
        return '#eef0f3'
    if approx:
        return '#dcfce7' if v > 0 else '#fee2e2' if v < 0 else '#e5e7eb'
    return '#86efac' if v > 0 else '#fca5a5' if v < 0 else '#e5e7eb'


# =============================================================================
#  PHẦN F — GIAO DIỆN tkinter (3 chế độ chơi + độ khó)
# =============================================================================
import tkinter as tk
from tkinter import ttk, messagebox

# Danh sách lựa chọn đối thủ: 3 mức độ khó + 7 thuật toán cụ thể
DIFF_ITEMS = [('🟢 Dễ', ('diff', 'easy')),
              ('🟡 Vừa', ('diff', 'medium')),
              ('🔴 Khó (tối ưu)', ('diff', 'hard'))]
ALGO_ITEMS = [('⚙ ' + ALGORITHMS[k], ('algo', k)) for k in ALGORITHMS]
OPP_ITEMS = DIFF_ITEMS + ALGO_ITEMS
OPP_MAP = dict(OPP_ITEMS)
OPP_DISPLAYS = [d for d, _ in OPP_ITEMS]

BG = '#eef2ff'       # nền cửa sổ
CARD = '#ffffff'     # nền thẻ
ACCENT = '#4f46e5'   # màu nhấn
BOARDBG = '#c7d2fe'  # nền bàn cờ
MODES = {'Người vs Máy': 'pvc', 'Người vs Người': 'pvp', 'Máy vs Máy': 'cvc'}


class TicTacToeGUI:
    def __init__(self, root):
        self.root = root
        root.title('Tic-Tac-Toe — Minimax & Adversarial Search')
        root.configure(bg=BG)
        self.auto = False
        self.busy = False

        # ttk + theme 'clam' tự tô màu -> hiển thị ổn định cả Light lẫn Dark mode (macOS/Win/Linux)
        style = ttk.Style(root)
        try:
            style.theme_use('clam')
        except tk.TclError:
            pass
        INK, SUB = '#111827', '#475569'
        style.configure('App.TFrame', background=BG)
        style.configure('Card.TFrame', background=CARD)
        style.configure('Head.TFrame', background=ACCENT)
        style.configure('Board.TFrame', background=BOARDBG)
        style.configure('App.TLabel', background=BG, foreground=INK)
        style.configure('Card.TLabel', background=CARD, foreground=INK)
        style.configure('Muted.TLabel', background=CARD, foreground='#64748b')
        style.configure('Accent.TLabel', background=CARD, foreground=ACCENT)
        style.configure('Head.TLabel', background=ACCENT, foreground='white')
        style.configure('Status.TLabel', background=BG, foreground='#1e293b')
        style.configure('Legend.TLabel', background=BG, foreground=SUB)
        style.configure('Info.TLabel', background='#f1f5f9', foreground='#1f2937', padding=10)
        style.configure('TCheckbutton', background=CARD, foreground=INK)
        style.map('TCheckbutton', background=[('active', CARD)])
        style.configure('TCombobox', padding=3)
        style.configure('TButton', font=('Helvetica', 10), padding=4)
        style.configure('Big.TButton', font=('Helvetica', 11, 'bold'), padding=5)
        root.configure(bg=BG)
        root.geometry('1060x720')
        root.minsize(900, 600)

        outer = ttk.Frame(root, style='App.TFrame')
        outer.pack(fill='both', expand=True)
        outer.grid_columnconfigure(1, weight=1)
        outer.grid_rowconfigure(1, weight=1)

        # ---- Header ----
        header = ttk.Frame(outer, style='Head.TFrame')
        header.grid(row=0, column=0, columnspan=2, sticky='we')
        ttk.Label(header, text='🎲 TIC-TAC-TOE', style='Head.TLabel',
                  font=('Helvetica', 20, 'bold')).pack(side='left', padx=16, pady=10)
        ttk.Label(header, text='Minimax · Alpha-Beta · Heuristic · Monte Carlo',
                  style='Head.TLabel', font=('Helvetica', 11)).pack(side='left', pady=10)

        # ---- Vùng điều khiển (trái) ----
        self.panel = ttk.Frame(outer, style='Card.TFrame')
        self.panel.grid(row=1, column=0, sticky='nw', padx=12, pady=12)
        self._build_controls()

        # ---- Vùng bàn cờ + thông tin (phải) ----
        right = ttk.Frame(outer, style='App.TFrame')
        right.grid(row=1, column=1, sticky='nw', padx=(0, 12), pady=12)
        self.status = ttk.Label(right, text='', style='Status.TLabel', font=('Helvetica', 14, 'bold'))
        self.status.pack(anchor='w')
        self.boardf = ttk.Frame(right, style='Board.TFrame')
        self.boardf.pack(pady=8, anchor='w')
        self.legend = ttk.Label(right, style='Legend.TLabel', justify='left', font=('Helvetica', 9),
                                text='Số trên ô trống = giá trị AI tính:  🟩 +thắng   ⬜ 0 hòa   '
                                     '🟥 −thua   ≤ cắt tỉa   số lẻ = ước lượng')
        self.legend.pack(anchor='w')
        self.score = ttk.Label(right, style='Status.TLabel', font=('Helvetica', 11, 'bold'))
        self.score.pack(anchor='w', pady=(6, 0))
        self.info = ttk.Label(right, style='Info.TLabel', justify='left', anchor='nw',
                              font=('Menlo', 10))
        self.info.pack(anchor='w', pady=8, fill='x')

        self.tally = {X: 0, O: 0, 'draw': 0}
        self.new_game()

    # ------------------------------------------------------------------
    def _build_controls(self):
        p = self.panel
        self.v_mode = tk.StringVar(value='Người vs Máy')
        self.v_n = tk.IntVar(value=3)
        self.v_k = tk.IntVar(value=3)
        self.v_first = tk.StringVar(value='X')
        self.v_human = tk.StringVar(value='X')
        self.v_opp = tk.StringVar(value='🔴 Khó (tối ưu)')
        self.v_ai_x = tk.StringVar(value='🔴 Khó (tối ưu)')
        self.v_ai_o = tk.StringVar(value='🟡 Vừa')
        self.v_heur = tk.StringVar(value='h2')
        self.v_depth = tk.IntVar(value=3)
        self.v_playouts = tk.IntVar(value=1000)
        self.v_speed = tk.DoubleVar(value=0.6)
        self.v_fast = tk.BooleanVar(value=True)
        self.v_show = tk.BooleanVar(value=True)

        r = [0]

        def add(label, widget):
            ttk.Label(p, text=label, style='Card.TLabel', font=('Helvetica', 10)).grid(
                row=r[0], column=0, sticky='w', padx=8, pady=3)
            widget.grid(row=r[0], column=1, sticky='we', padx=8, pady=3)
            r[0] += 1

        ttk.Label(p, text='🎮 Thiết lập', style='Accent.TLabel',
                  font=('Helvetica', 13, 'bold')).grid(row=r[0], column=0, columnspan=2,
                                                       sticky='w', padx=8, pady=(8, 2))
        r[0] += 1
        self.cb_mode = ttk.Combobox(p, values=list(MODES), textvariable=self.v_mode,
                                    state='readonly', width=20)
        add('Chế độ chơi:', self.cb_mode)
        add('Kích thước n:', ttk.Spinbox(p, from_=3, to=5, textvariable=self.v_n, width=8,
                                         command=self._on_size))
        self.sp_k = ttk.Spinbox(p, from_=3, to=5, textvariable=self.v_k, width=8, command=self.new_game)
        add('Số quân thắng k:', self.sp_k)
        add('Quân đi trước:', ttk.Combobox(p, values=['X', 'O'], textvariable=self.v_first,
                                           state='readonly', width=8))
        self.cb_human = ttk.Combobox(p, values=['X', 'O'], textvariable=self.v_human,
                                     state='readonly', width=8)
        add('Bạn cầm quân:', self.cb_human)
        self.cb_opp = ttk.Combobox(p, values=OPP_DISPLAYS, textvariable=self.v_opp,
                                   state='readonly', width=28)
        add('Đối thủ AI:', self.cb_opp)
        self.cb_ai_x = ttk.Combobox(p, values=OPP_DISPLAYS, textvariable=self.v_ai_x,
                                    state='readonly', width=28)
        add('AI quân X:', self.cb_ai_x)
        self.cb_ai_o = ttk.Combobox(p, values=OPP_DISPLAYS, textvariable=self.v_ai_o,
                                    state='readonly', width=28)
        add('AI quân O:', self.cb_ai_o)
        self.sc_speed = ttk.Scale(p, from_=0.1, to=2.0, variable=self.v_speed, orient='horizontal')
        add('Tốc độ tự chạy (s):', self.sc_speed)

        # Nâng cao
        ttk.Label(p, text='⚙️ Nâng cao (cho thuật toán cụ thể)', style='Muted.TLabel',
                  font=('Helvetica', 10, 'italic')).grid(row=r[0], column=0, columnspan=2,
                                                         sticky='w', padx=8, pady=(10, 2))
        r[0] += 1
        add('Heuristic Eval:', ttk.Combobox(p, values=['h1', 'h2'], textvariable=self.v_heur,
                                            state='readonly', width=8))
        add('Độ sâu cut-off d:', ttk.Spinbox(p, from_=1, to=25, textvariable=self.v_depth, width=8))
        add('Số playout N:', ttk.Spinbox(p, from_=50, to=10000, increment=50,
                                         textvariable=self.v_playouts, width=8))
        ttk.Checkbutton(p, text='Ưu tiên thắng nhanh', variable=self.v_fast,
                        command=self.new_game).grid(row=r[0], column=0, columnspan=2,
                                                    sticky='w', padx=8, pady=2)
        r[0] += 1
        ttk.Checkbutton(p, text='Hiện điểm AI tính cho từng ô', variable=self.v_show,
                        command=self.render).grid(row=r[0], column=0, columnspan=2,
                                                  sticky='w', padx=8, pady=2)
        r[0] += 1

        # Nút bấm
        bf = ttk.Frame(p, style='Card.TFrame')
        bf.grid(row=r[0], column=0, columnspan=2, pady=10)
        ttk.Button(bf, text='🔄 Ván mới', style='Big.TButton', command=self.new_game).grid(row=0, column=0, padx=3, pady=3)
        ttk.Button(bf, text='↩️ Hoàn tác', command=self.undo).grid(row=0, column=1, padx=3, pady=3)
        self.b_hint = ttk.Button(bf, text='💡 Gợi ý', command=self.hint)
        self.b_hint.grid(row=0, column=2, padx=3, pady=3)
        self.b_step = ttk.Button(bf, text='▶ Đi 1 nước', command=self.cvc_step)
        self.b_step.grid(row=1, column=0, padx=3, pady=3)
        self.b_auto = ttk.Button(bf, text='⏯ Tự chạy', command=self.toggle_auto)
        self.b_auto.grid(row=1, column=1, padx=3, pady=3)
        self.b_runall = ttk.Button(bf, text='⏭ Chạy hết', command=self.run_all)
        self.b_runall.grid(row=1, column=2, padx=3, pady=3)

        self.cb_mode.bind('<<ComboboxSelected>>', lambda e: self.new_game())
        self.cb_human.bind('<<ComboboxSelected>>', lambda e: self.new_game())
        p.grid_columnconfigure(1, weight=1)

    # ------------------------------------------------------------------
    @property
    def mode(self):
        return MODES[self.v_mode.get()]

    def _on_size(self):
        n = self.v_n.get()
        self.sp_k.configure(from_=3, to=n)
        self.v_k.set({3: 3, 4: 4, 5: 4}[n])
        self.new_game()

    def _adv(self):
        return dict(depth_limit=self.v_depth.get(), heuristic=self.v_heur.get(), beam=0,
                    n_playouts=self.v_playouts.get(), c=1.41)

    def _resolve(self, display):
        kind, key = OPP_MAP[display]
        if kind == 'diff':
            return difficulty_config(self.v_n.get(), key)
        cfg = dict(algo=key, blunder=0.0, label=ALGORITHMS[key])
        cfg.update(self._adv())
        return cfg

    def _ai_cfg(self):
        if self.mode == 'pvc':
            return self._resolve(self.v_opp.get())
        return self._resolve(self.v_ai_x.get() if self.state.to_move == X else self.v_ai_o.get())

    def _role(self, sym):
        if self.mode == 'pvc':
            return 'Bạn' if sym == self.human else 'Máy'
        if self.mode == 'pvp':
            return 'Người 1' if sym == X else 'Người 2'
        return 'AI ' + sym

    # ------------------------------------------------------------------
    def new_game(self):
        self.auto = False
        self.b_auto.configure(text='⏯ Tự chạy')
        n = self.v_n.get()
        k = min(max(3, self.v_k.get()), n)
        self.v_k.set(k)
        self.human = self.v_human.get()
        self.game = TicTacToe(n, k, self.v_first.get(), fast_win=self.v_fast.get())
        self.state = self.game.initial_state()
        self.moves = []
        self.overlay = None
        self.last = None
        self.over = False

        for w in self.boardf.winfo_children():
            w.destroy()
        cell = {3: 110, 4: 88, 5: 72}[n]
        gap = 10
        self.geo = (cell, gap)
        side = n * cell + (n + 1) * gap
        self.cv = tk.Canvas(self.boardf, width=side, height=side, bg=BOARDBG,
                            highlightthickness=0, bd=0)
        self.cv.pack(padx=4, pady=4)
        self.cv.bind('<Button-1>', self._on_canvas_click)

        self._sync_mode()
        self.render()
        self.info.configure(text='Bắt đầu! ' + ('Bấm vào ô trống để đánh.'
                            if self.mode != 'cvc' else 'Bấm ▶ / ⏯ / ⏭ để hai máy đấu nhau.'))
        if self.mode == 'pvc' and self.state.to_move != self.human:
            self.root.after(150, self.ai_step)

    def _sync_mode(self):
        m = self.mode

        def en(widget, on):
            try:
                widget.configure(state='readonly' if on else 'disabled')
            except tk.TclError:
                widget.configure(state='normal' if on else 'disabled')
        en(self.cb_opp, m == 'pvc')
        en(self.cb_human, m == 'pvc')
        en(self.cb_ai_x, m == 'cvc')
        en(self.cb_ai_o, m == 'cvc')
        try:
            self.sc_speed.state(['!disabled'] if m == 'cvc' else ['disabled'])
        except tk.TclError:
            pass
        self.b_hint.configure(state='normal' if m != 'cvc' else 'disabled')
        for b in (self.b_step, self.b_auto, self.b_runall):
            b.configure(state='normal' if m == 'cvc' else 'disabled')

    # ------------------------------------------------------------------
    def on_cell(self, i):
        if self.over or self.busy or self.mode == 'cvc':
            return
        if self.mode == 'pvc' and self.state.to_move != self.human:
            return
        if self.state.board[i] != EMPTY:
            return
        self.overlay = None
        self._play(i)
        if self.mode == 'pvc' and not self.over and self.state.to_move != self.human:
            self.root.after(120, self.ai_step)

    def _play(self, a):
        sym = self.state.to_move
        self.state = self.game.result(self.state, a)
        self.moves.append((sym, a))
        self.last = a
        if self.game.terminal(self.state):
            self.over = True
            w = self.game.winner(self.state.board)
            self.tally['draw' if w is None else w] += 1
        self.render()

    def undo(self):
        self.auto = False
        if not self.moves:
            return
        back = 2 if (self.mode == 'pvc' and len(self.moves) >= 2) else 1
        self.moves = self.moves[:-back]
        self.state = self.game.initial_state()
        for _, a in self.moves:
            self.state = self.game.result(self.state, a)
        self.over = self.game.terminal(self.state)
        self.last = self.moves[-1][1] if self.moves else None
        self.overlay = None
        self.render()

    def _search(self, state, cfg):
        try:
            res = solve(self.game, state, cfg['algo'], cfg.get('depth_limit', 3),
                        cfg.get('heuristic', 'h2'), cfg.get('beam', 0),
                        cfg.get('n_playouts', 1000), cfg.get('c', 1.41), max_nodes=1_000_000)
            note = ''
        except SearchLimitExceeded:
            res = solve(self.game, state, 'heuristic_ab', 2, cfg.get('heuristic', 'h2'))
            note = 'Vượt giới hạn nút -> tạm dùng Heuristic Alpha-Beta d=2.'
        played = res.move
        legal = self.game.empty_cells(state.board)
        if cfg.get('blunder') and len(legal) > 1 and random.random() < cfg['blunder']:
            played = random.choice([a for a in legal if a != res.move])
            note = (note + '  ' if note else '') + '🎲 AI đi ngẫu nhiên (độ khó %s).' % cfg['label']
        return res, note, played

    def ai_step(self):
        if self.over or self.busy:
            return
        if self.mode == 'pvc' and self.state.to_move == self.human:
            return                          # không đi thay lượt của người
        if self.mode == 'pvp':
            return                          # chế độ người-người: không có AI
        self.busy = True
        cfg = self._ai_cfg()
        before = self.state
        self.status.configure(text='🤖 %s đang suy nghĩ…' % self._role(before.to_move))
        self.root.update_idletasks()
        res, note, played = self._search(before, cfg)
        self.overlay = (res.move_values, self._role(before.to_move), played,
                        cfg['algo'] not in EXACT_ALGOS)
        self._play(played)
        self._show_info(res, note, before, cfg, played)
        self.busy = False
        if self.mode == 'cvc' and self.auto and not self.over:
            self.root.after(int(self.v_speed.get() * 1000), self.ai_step)

    def cvc_step(self):
        if self.mode == 'cvc' and not self.over:
            self.ai_step()

    def run_all(self):
        self.auto = False
        while self.mode == 'cvc' and not self.over and not self.busy:
            self.ai_step()

    def toggle_auto(self):
        if self.mode != 'cvc':
            return
        self.auto = not self.auto
        self.b_auto.configure(text='⏸ Dừng' if self.auto else '⏯ Tự chạy')
        if self.auto and not self.over:
            self.ai_step()

    def hint(self):
        if self.over or self.busy or self.mode == 'cvc':
            return
        cfg = difficulty_config(self.v_n.get(), 'hard')
        res, note, _ = self._search(self.state, cfg)
        self.overlay = (res.move_values, self._role(self.state.to_move), res.move,
                        cfg['algo'] not in EXACT_ALGOS)
        self.render()
        self._show_info(res, note, self.state, cfg, res.move, hint=True)

    # ------------------------------------------------------------------
    def _show_info(self, res, note, state, cfg, played, hint=False):
        s, g = res.stats, self.game
        v, exact = res.move_values.get(played, (res.value, True))
        approx = (cfg['algo'] not in EXACT_ALGOS) and not float(v).is_integer()
        who = self._role(state.to_move)
        if approx:
            pred = 'ước lượng %+.3f' % v
        elif v > 0:
            pred = '%s sẽ THẮNG nếu chơi đúng' % who
        elif v < 0:
            pred = '%s sẽ THUA nếu đối thủ hoàn hảo' % who
        else:
            pred = 'HÒA nếu cả hai tối ưu'
        title = '💡 GỢI Ý' if hint else '🤖 LƯỢT SUY NGHĨ VỪA RỒI'
        lines = [
            title,
            '─' * 46,
            'Độ khó/Thuật toán : %s' % cfg['label'],
            '                    %s' % res.algorithm,
            'Nước chọn (h,c)   : %s' % str(g.rc(played)),
            'Giá trị           : %s   (%s)' % (fmt_value(v, exact), pred),
            'Số nút duyệt      : %s' % format(s.nodes, ','),
            'Số lần cắt tỉa    : %s' % format(s.cutoffs, ','),
            'Tra bảng TT       : %s' % format(s.tt_hits, ','),
            'TT / cây MCTS     : %s' % format(s.tt_size, ','),
            'Độ sâu lớn nhất   : %d' % s.max_depth,
            'Thời gian         : %.1f ms' % s.time_ms,
        ]
        if note:
            lines.append('⚠️ ' + note)
        self.info.configure(text='\n'.join(lines))

    def _cell_xy(self, i):
        cell, gap = self.geo
        r, c = divmod(i, self.game.n)
        x0 = gap + c * (cell + gap)
        y0 = gap + r * (cell + gap)
        return x0, y0, x0 + cell, y0 + cell

    def _on_canvas_click(self, e):
        cell, gap = self.geo
        n = self.game.n
        c = int((e.x - gap) // (cell + gap))
        r = int((e.y - gap) // (cell + gap))
        if 0 <= r < n and 0 <= c < n:
            x0, y0, x1, y1 = self._cell_xy(r * n + c)
            if x0 <= e.x <= x1 and y0 <= e.y <= y1:
                self.on_cell(r * n + c)

    def _round_rect(self, x0, y0, x1, y1, rad, **kw):
        pts = [x0 + rad, y0, x1 - rad, y0, x1, y0, x1, y0 + rad, x1, y1 - rad, x1, y1,
               x1 - rad, y1, x0 + rad, y1, x0, y1, x0, y1 - rad, x0, y0 + rad, x0, y0]
        return self.cv.create_polygon(pts, smooth=True, **kw)

    def render(self):
        g, b = self.game, self.state.board
        win = set(g.winning_line(b) or [])
        show = self.v_show.get()
        mv, who, chosen, approx = self.overlay if (self.overlay and show) else ({}, '', None, False)
        cv = self.cv
        cv.delete('all')
        cell, gap = self.geo
        lw = max(5, int(cell * 0.13))
        for i in range(g.n * g.n):
            x0, y0, x1, y1 = self._cell_xy(i)
            ch = b[i]
            if i in win:
                fill = '#fde68a'
            elif ch == EMPTY and i in mv:
                v, exact = mv[i]
                fill = value_color(v, exact, approx and not float(v).is_integer())
            elif ch != EMPTY and i == self.last:
                fill = '#dbeafe'
            else:
                fill = '#ffffff'
            outline = '#16a34a' if i == chosen else '#dbe0ea'
            self._round_rect(x0, y0, x1, y1, cell * 0.16, fill=fill,
                             outline=outline, width=(4 if i == chosen else 1))
            if ch == X:
                ins = cell * 0.27
                cv.create_line(x0 + ins, y0 + ins, x1 - ins, y1 - ins, fill=COLORS[X],
                               width=lw, capstyle='round')
                cv.create_line(x0 + ins, y1 - ins, x1 - ins, y0 + ins, fill=COLORS[X],
                               width=lw, capstyle='round')
            elif ch == O:
                ins = cell * 0.25
                cv.create_oval(x0 + ins, y0 + ins, x1 - ins, y1 - ins, outline=COLORS[O], width=lw)
            elif i in mv:
                v, exact = mv[i]
                cv.create_text((x0 + x1) / 2, (y0 + y1) / 2, text=fmt_value(v, exact),
                               fill='#1f2937', font=('Helvetica', max(12, int(cell * 0.22)), 'bold'))
        if self.over:
            w = g.winner(b)
            if w is None:
                self.status.configure(text='🤝 HÒA!   (bấm “Ván mới” để chơi tiếp)', foreground='#475569')
            else:
                col = '#15803d' if (self.mode == 'pvc' and w == self.human) else (
                    '#b91c1c' if self.mode == 'pvc' else ACCENT)
                self.status.configure(text='🏆 %s (%s) THẮNG!' % (self._role(w), w), foreground=col)
        else:
            cur = self.state.to_move
            self.status.configure(text='Lượt: %s — quân %s    ·    thắng khi có %d quân liên tiếp'
                                  % (self._role(cur), cur, g.k), foreground=COLORS[cur])
        t = self.tally
        self.score.configure(text='Tỉ số:  %s (X) %d   ·   %s (O) %d   ·   Hòa %d'
                             % (self._role(X), t[X], self._role(O), t[O], t['draw']))


def main():
    root = tk.Tk()
    TicTacToeGUI(root)
    root.mainloop()


# =============================================================================
#  PHẦN G — BÁO CÁO PHÂN TÍCH (chạy: python3 tictactoe_minimax.py --report)
#  In toàn bộ nội dung đồ án ra màn hình, bằng Python thuần (không cần thư viện ngoài).
# =============================================================================
def _board1(g, b):
    """Bàn cờ dạng một dòng: 'XOX/.O./..X'."""
    n = g.n
    return '/'.join(b[r * n:(r + 1) * n] for r in range(n))


def _table(headers, rows):
    """In bảng chữ căn cột gọn gàng."""
    cols = len(headers)
    w = [len(str(headers[i])) for i in range(cols)]
    srows = [[str(c) for c in row] for row in rows]
    for row in srows:
        for i in range(cols):
            w[i] = max(w[i], len(row[i]))
    line = '+' + '+'.join('-' * (w[i] + 2) for i in range(cols)) + '+'
    out = [line, '| ' + ' | '.join(str(headers[i]).ljust(w[i]) for i in range(cols)) + ' |', line]
    for row in srows:
        out.append('| ' + ' | '.join(row[i].ljust(w[i]) for i in range(cols)) + ' |')
    out.append(line)
    print('\n'.join(out))


def _h(title):
    print('\n' + '=' * 78)
    print(title)
    print('=' * 78)


def _exact_value(game, state, player):
    """Minimax value chính xác của 'state' theo góc nhìn 'player'."""
    if game.terminal(state):
        return game.utility(state, player)
    v = solve(game, state, 'alphabeta_tt').value
    return v if state.to_move == player else -v


def _explore_state_space(game):
    """BFS đếm mọi trạng thái đạt được từ s0."""
    from collections import deque
    start = game.initial_state()
    visited = {start}
    queue = deque([start])
    term = {'X': 0, 'O': 0, 'draw': 0}
    while queue:
        s = queue.popleft()
        if game.terminal(s):
            w = game.winner(s.board)
            term['draw' if w is None else w] += 1
            continue
        for a in game.actions(s):
            t = game.result(s, a)
            if t not in visited:
                visited.add(t)
                queue.append(t)
    return visited, term


# --- cây số học cho bài tập slide 2-ply ---
class _Num:
    def __init__(self, name, *kids):
        self.name = name
        self.kids = list(kids)


def _count_leaves(node):
    if not isinstance(node, _Num):
        return 1
    return sum(_count_leaves(k) for k in node.kids)


def _mm(node, mx):
    if not isinstance(node, _Num):
        return node
    vals = [_mm(k, not mx) for k in node.kids]
    return max(vals) if mx else min(vals)


def _ab(node, a, b, mx, seen):
    if not isinstance(node, _Num):
        seen[0] += 1
        return node
    best = -INF if mx else INF
    for k in node.kids:
        v = _ab(k, a, b, not mx, seen)
        if mx:
            best = max(best, v); a = max(a, best)
        else:
            best = min(best, v); b = min(b, best)
        if a >= b:
            break
    return best


def run_report():
    g = TicTacToe(3, 3)
    print('#' * 78)
    print('#  ĐỒ ÁN AI — TIC-TAC-TOE VỚI MINIMAX & ADVERSARIAL SEARCH (BÁO CÁO)')
    print('#  Chạy: python3 tictactoe_minimax.py --report   |   Giao diện: python3 tictactoe_minimax.py')
    print('#' * 78)

    # ----- PHẦN 1: PEAS -----
    _h('PHẦN 1 — ĐẶC TẢ PEAS & PHÂN LOẠI MÔI TRƯỜNG')
    _table(['Thành phần', 'Mô tả cho Tic-Tac-Toe'], [
        ['P - Performance', 'Thắng = +1, Hòa = 0, Thua = -1 (Utility của Max). Bàn 3x3: AI không thua.'],
        ['E - Environment', 'Bàn n x n, đi luân phiên, đối thủ (người) là một phần môi trường.'],
        ['A - Actuators', 'Đặt 1 quân (X/O) vào 1 ô trống.'],
        ['S - Sensors', 'Đọc toàn bộ bàn cờ và biết đến lượt ai.'],
    ])
    _table(['Tiêu chí', 'Kết luận'], [
        ['Quan sát', 'Đầy đủ (perfect information)'],
        ['Xác định/ngẫu nhiên', 'Xác định (deterministic); đối thủ => strategic'],
        ['Tĩnh/động', 'Tĩnh'], ['Rời rạc/liên tục', 'Rời rạc'],
        ['Số tác nhân', 'Đa tác nhân, cạnh tranh, zero-sum'],
    ])

    # ----- PHẦN 2: BÀI TOÁN TÌM KIẾM -----
    _h('PHẦN 2 — ĐỊNH NGHĨA BÀI TOÁN TÌM KIẾM')
    s0 = g.initial_state()
    print('(1) Trạng thái s = (board, to_move). Bàn 3x3 đánh số ô 0..8 (i = hàng*3 + cột).')
    print('(2) Trạng thái đầu s0 = bàn trống, X đi trước:', _board1(g, s0.board), '| lượt', s0.to_move)
    print('(3) Actions(s0) =', g.actions(s0))
    s1 = g.result(s0, 4)
    print('(4) Result(s0, 4):', _board1(g, s0.board), '--X vào ô 4-->', _board1(g, s1.board))
    print('(5) Terminal(s): có người đạt k quân liên tiếp, hoặc bàn đầy.')
    print('(6) Utility(s): +1 nếu Max thắng, -1 nếu thua, 0 nếu hòa (zero-sum).')
    states, term = _explore_state_space(g)
    canon = {g.canonical(s.board) for s in states}
    _h('  2.1 — KÍCH THƯỚC KHÔNG GIAN TRẠNG THÁI (tính bằng BFS)')
    _table(['Đại lượng', 'Giá trị'], [
        ['Cận trên 3^9 (slide tr.7)', format(3 ** 9, ',')],
        ['Số trạng thái hợp lệ (BFS)', format(len(states), ',')],
        ['  X thắng', term['X']], ['  O thắng', term['O']], ['  Hòa', term['draw']],
        ['Sau khi gộp đối xứng', format(len(canon), ',')],
    ])

    # ----- PHẦN 3: TRANSITION FUNCTION -----
    _h('PHẦN 3 — TRANSITION FUNCTION (bảng, sinh tự động bằng Result)')
    rows, s = [], s0
    for a in [4, 1, 0, 8, 6, 3, 2]:
        t = g.result(s, a)
        note = ''
        if g.terminal(t):
            w = g.winner(t.board)
            note = ('%s thắng' % w) if w else 'Hòa'
        rows.append([_board1(g, s.board), s.to_move, '%s->%s' % (s.to_move, g.rc(a)),
                     _board1(g, t.board), note])
        s = t
    for a in [0, 8]:                      # vài nhánh khác từ s0
        rows.append([_board1(g, s0.board), 'X', 'X->%s' % str(g.rc(a)),
                     _board1(g, g.result(s0, a).board), 'nhánh khác từ s0'])
    s4 = g.result(s0, 4)
    for a in [2, 7]:
        rows.append([_board1(g, s4.board), 'O', 'O->%s' % str(g.rc(a)),
                     _board1(g, g.result(s4, a).board), 'nhánh khác sau X giữa'])
    _table(['s (hiện tại)', 'lượt', 'hành động a', "s' = Result(s,a)", 'ghi chú'], rows)
    print('Bảng có %d dòng (yêu cầu >= 10).' % len(rows))

    # ----- PHẦN 4: CÂY TÌM KIẾM -----
    _h('PHẦN 4 — CÂY TÌM KIẾM TỪ s0 (kèm minimax value, dạng cây ASCII)')
    count = 1
    print('s0 %s  [MV=%+g]  (MAX, X đi)' % (_board1(g, s0.board), _exact_value(g, s0, X)))
    children = g.actions(s0)
    for idx, a in enumerate(children):
        c = g.result(s0, a)
        count += 1
        mv = _exact_value(g, c, X)
        print('  |- X->%s  %s  [MV=%+g]%s'
              % (str(g.rc(a)), _board1(g, c.board), mv, '  (MIN, O đi)' if idx == 0 else ''))
        if idx < 3:
            for b in g.actions(c)[:2]:
                gc = g.result(c, b)
                count += 1
                print('  |    |- O->%s  %s  [MV=%+g]' % (str(g.rc(b)), _board1(g, gc.board),
                                                         _exact_value(g, gc, X)))
    print('Cây hiển thị %d trạng thái (yêu cầu 10..30). MV(s0)=0 => chơi tối ưu luôn HÒA.' % count)

    # ----- PHẦN 5: HAI HEURISTIC -----
    _h('PHẦN 5 — HAI HÀM HEURISTIC (Eval) trên bàn ví dụ "XX..O...."')
    ex = 'XX..O....'
    print(g.pretty(ex))
    print('h1 (số đường còn mở)      = %.4f' % h1_open_lines(g, ex, X))
    print('h2 (trọng số đe dọa)      = %.4f  (nhận ra đường 0-1-2 có 2 quân X)' % h2_weighted(g, ex, X))

    # ----- PHẦN 5B: SO SÁNH 7 GIẢI THUẬT (E1, bàn 3x3 trống) -----
    _h('PHẦN 5B — SO SÁNH 7 GIẢI THUẬT (E1: bàn 3x3 trống)')
    configs = [('minimax', {}), ('alphabeta', {}), ('alphabeta_mo', {}), ('alphabeta_tt', {}),
               ('heuristic_ab', dict(depth_limit=2, heuristic='h1')),
               ('heuristic_ab', dict(depth_limit=4, heuristic='h2')),
               ('pure_mc', dict(n_playouts=1000, seed=0)), ('mcts', dict(n_playouts=1000, seed=0))]
    rows = []
    for algo, kw in configs:
        r = solve(g, s0, algo, **kw)
        rows.append([r.algorithm[:34], str(g.rc(r.move)), round(r.value, 3), format(r.stats.nodes, ','),
                     format(r.stats.cutoffs, ','), format(r.stats.tt_size, ','),
                     r.stats.max_depth, '%.1f' % r.stats.time_ms])
    _table(['Giải thuật', 'Nước', 'Giá trị', 'Số nút', 'Cắt tỉa', 'TT/MCTS', 'Sâu', 'ms'], rows)
    print('Minimax 549,946 nút -> Alpha-Beta ~18k -> +Move ordering ~7k -> +TT+đối xứng vài trăm.')

    # ----- PHẦN 5C: THU HẸP KHÔNG GIAN (E2) -----
    _h('PHẦN 5C — SỐ NÚT THEO SỐ Ô TRỐNG (E2, các giải thuật tối ưu)')
    algos = ['minimax', 'alphabeta', 'alphabeta_mo', 'alphabeta_tt']
    random.seed(3)
    rows = []
    for empties in range(4, 10):
        pos = None
        for _ in range(400):
            s = g.initial_state()
            for _ in range(9 - empties):
                if g.terminal(s):
                    break
                s = g.result(s, random.choice(g.actions(s)))
            if not g.terminal(s) and s.board.count(EMPTY) == empties:
                pos = s
                break
        if pos is None:
            continue
        rows.append([empties] + [format(solve(g, pos, a).stats.nodes, ',') for a in algos])
    _table(['Ô trống', 'Minimax', 'Alpha-Beta', '+Move order', '+TT+đxứng'], rows)

    # ----- PHẦN 5D: KIỂM TRA TÍNH ĐÚNG -----
    _h('PHẦN 5D — KIỂM TRA TÍNH ĐÚNG ĐẮN')
    random.seed(1)
    checked = mismatch = 0
    for _ in range(120):
        s = g.initial_state()
        for _ in range(random.randint(1, 5)):
            if g.terminal(s):
                break
            s = g.result(s, random.choice(g.actions(s)))
        if g.terminal(s):
            continue
        checked += 1
        if len({solve(g, s, a).value for a in algos}) != 1:
            mismatch += 1
    print('4 giải thuật tối ưu cho CÙNG giá trị trên %d thế cờ: %d sai lệch.' % (checked, mismatch))
    optimal = lambda st: solve(g, st, 'alphabeta_tt').move
    rnd = lambda st: random.choice(g.empty_cells(st.board))

    def play(a1, a2):
        s = g.initial_state(); ags = {X: a1, O: a2}
        while not g.terminal(s):
            s = g.result(s, ags[s.to_move](s))
        return g.winner(s.board) or 'draw'
    s = g.initial_state()
    while not g.terminal(s):
        s = g.result(s, optimal(s))
    print('Hai AI tối ưu đấu nhau: %s' % ('HÒA' if g.winner(s.board) is None else 'THẮNG'))
    random.seed(0)
    res = [play(optimal, rnd) for _ in range(50)] + [play(rnd, optimal) for _ in range(50)]
    print('AI tối ưu vs Ngẫu nhiên (100 ván): thắng %d, hòa %d, THUA %d'
          % (sum(r in (X, O) and ((r == X) == (i < 50)) for i, r in enumerate(res)),
             res.count('draw'),
             sum(r in (X, O) and ((r == X) != (i < 50)) for i, r in enumerate(res))))

    # ----- PHẦN 5E: BÀI TẬP SLIDE 2-PLY -----
    _h('PHẦN 5E — GIẢI LẠI BÀI TẬP TRONG SLIDE (kiểm chứng Minimax/Alpha-Beta)')
    ex1 = _Num('Max', _Num('a1', 2, 0, 5), _Num('a2', -5, -2, 7), _Num('a3', 5, -7, 4))
    ex2 = _Num('Max', _Num('a1', -5, 2, 5), _Num('a2', 7, 0, 2), _Num('a3', 5, -7, -4))
    ex3 = _Num('Max', _Num('a2', 7, 0, 2), _Num('a1', -5, 2, 5), _Num('a3', 5, -7, -4))
    exAG = _Num('A', _Num('B', _Num('D', 2, 3), _Num('E', 5, 9)), _Num('C', _Num('F', 0, 1), _Num('G', 7, 5)))
    for name, tr, slide in [('Bài 1 (tr.18)', ex1, 0), ('Bài 2 a1,a2,a3 (tr.24)', ex2, 0),
                            ('Bài 3 move-order a2,a1,a3 (tr.26)', ex3, 0), ('Ví dụ A-G (examples)', exAG, 3)]:
        seen = [0]
        v = _ab(tr, -INF, INF, True, seen)
        pruned = _count_leaves(tr) - seen[0]
        mm = _mm(tr, True)
        print('%-34s Minimax=%2d  Alpha-Beta=%2d (slide %d)  | lá bị cắt tỉa=%d'
              % (name, mm, v, slide, pruned))

    _h('KẾT LUẬN')
    print('- Minimax/Alpha-Beta cho nước đi TỐI ƯU; Move ordering + TT + đối xứng thu hẹp mạnh số nút.')
    print('- Heuristic Alpha-Beta và Monte Carlo (UCB1) xử lý được bàn lớn, đổi tối ưu lấy tốc độ.')
    print('- Chạy giao diện để chơi:  python3 tictactoe_minimax.py')
    print('- Xuất hình ảnh (cần matplotlib):  python3 tictactoe_minimax.py --figures')


# =============================================================================
#  PHẦN H — XUẤT HÌNH ẢNH (tùy chọn, cần: pip install matplotlib)
# =============================================================================
def save_figures(outdir='hinh_ket_qua'):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
    except ImportError:
        print('⚠️ Chưa có matplotlib. Cài bằng:  pip3 install matplotlib  rồi chạy lại --figures')
        return
    import os
    os.makedirs(outdir, exist_ok=True)
    g = TicTacToe(3, 3)

    def draw_board(ax, board, x, y, size=1.0, win=(), last=None):
        n = g.n; cell = size / n; x0, y0 = x - size / 2, y - size / 2
        ax.add_patch(Rectangle((x0, y0), size, size, facecolor='white', edgecolor='#333', lw=1.3, zorder=2))
        for idx, ch in enumerate(board):
            r, c = divmod(idx, n)
            if idx in win or idx == last:
                ax.add_patch(Rectangle((x0 + c * cell, y0 + size - (r + 1) * cell), cell, cell,
                                       facecolor='#ffd54f' if idx in win else '#cfe8ff', edgecolor='none', zorder=2.5))
            if ch != EMPTY:
                ax.text(x0 + (c + .5) * cell, y0 + size - (r + .5) * cell, ch, ha='center', va='center',
                        fontsize=40 / n, color=COLORS[ch], weight='bold', zorder=4)
        for i in range(1, n):
            ax.plot([x0 + i * cell] * 2, [y0, y0 + size], color='#999', lw=.8, zorder=3)
            ax.plot([x0, x0 + size], [y0 + i * cell] * 2, color='#999', lw=.8, zorder=3)

    # Hình 1: chuỗi một ván X thắng
    seq_moves = [4, 1, 0, 8, 6, 3, 2]
    s = g.initial_state(); seq = [s]
    for a in seq_moves:
        s = g.result(s, a); seq.append(s)
    fig, ax = plt.subplots(figsize=(1.8 * len(seq), 2.2))
    for i, st in enumerate(seq):
        draw_board(ax, st.board, i * 1.8, 0, win=g.winning_line(st.board) or (),
                   last=seq_moves[i - 1] if i else None)
        ax.text(i * 1.8, -.72, 's%d' % i, ha='center', fontsize=9)
    ax.set_xlim(-.7, (len(seq) - 1) * 1.8 + .7); ax.set_ylim(-.95, .7)
    ax.set_aspect('equal'); ax.axis('off'); ax.set_title('Chuỗi transition: X thắng bằng nước đôi', weight='bold')
    plt.tight_layout(); plt.savefig('%s/1_chuoi_transition.png' % outdir, dpi=130); plt.close()

    # Hình 2: biểu đồ E2 (số nút theo số ô trống, thang log)
    algos = [('minimax', 'Minimax'), ('alphabeta', 'Alpha-Beta'),
             ('alphabeta_mo', '+Move order'), ('alphabeta_tt', '+TT+đối xứng')]
    random.seed(3)
    data = {name: ([], []) for _, name in algos}
    for empties in range(4, 10):
        pos = None
        for _ in range(400):
            st = g.initial_state()
            for _ in range(9 - empties):
                if g.terminal(st):
                    break
                st = g.result(st, random.choice(g.actions(st)))
            if not g.terminal(st) and st.board.count(EMPTY) == empties:
                pos = st; break
        if pos is None:
            continue
        for key, name in algos:
            data[name][0].append(empties); data[name][1].append(solve(g, pos, key).stats.nodes)
    fig, ax = plt.subplots(figsize=(7, 4.3))
    for _, name in algos:
        ax.plot(data[name][0], data[name][1], 'o-', label=name)
    ax.set_yscale('log'); ax.set_xlabel('Số ô trống (độ sâu còn lại)'); ax.set_ylabel('Số nút duyệt (log)')
    ax.set_title('E2 — Số nút theo kích thước bài toán'); ax.grid(alpha=.3, which='both'); ax.legend()
    plt.tight_layout(); plt.savefig('%s/2_bieu_do_so_nut.png' % outdir, dpi=130); plt.close()

    print('✅ Đã lưu hình vào thư mục "%s/".' % outdir)


# =============================================================================
#  PHẦN I — KIỂM TRA NHANH (chạy: python3 tictactoe_minimax.py --selftest)
# =============================================================================
def _selftest():
    g = TicTacToe(3, 3)
    assert len(g.lines) == 8
    # 4 giải thuật exact phải cho cùng giá trị trên vài thế cờ
    random.seed(1)
    for _ in range(60):
        s = g.initial_state()
        for _ in range(random.randint(1, 5)):
            if g.terminal(s):
                break
            s = g.result(s, random.choice(g.actions(s)))
        if g.terminal(s):
            continue
        vals = {solve(g, s, a).value for a in ('minimax', 'alphabeta', 'alphabeta_mo', 'alphabeta_tt')}
        assert len(vals) == 1, (s, vals)
    # hai AI tối ưu đấu nhau phải hòa
    s = g.initial_state()
    while not g.terminal(s):
        s = g.result(s, solve(g, s, 'alphabeta_tt').move)
    assert g.winner(s.board) is None
    # heuristic + monte carlo chạy được
    for algo in ('heuristic_ab', 'pure_mc', 'mcts'):
        r = solve(g, g.initial_state(), algo, seed=0)
        assert r.move in range(9)
    # and-or search: bàn trống không có kế hoạch thắng chắc
    assert and_or_search(g, g.initial_state(), 'win') is None
    print('✅ SELFTEST OK — engine và 7 giải thuật hoạt động đúng.')


if __name__ == '__main__':
    import sys
    args = sys.argv[1:]
    if '--help' in args or '-h' in args:
        print('Cách dùng:')
        print('  python3 tictactoe_minimax.py            # Mở GIAO DIỆN để chơi (mặc định)')
        print('  python3 tictactoe_minimax.py --report   # In BÁO CÁO phân tích ra màn hình')
        print('  python3 tictactoe_minimax.py --figures  # Xuất hình ảnh (cần matplotlib)')
        print('  python3 tictactoe_minimax.py --selftest # Kiểm tra nhanh engine')
    elif '--selftest' in args:
        _selftest()
    elif '--report' in args:
        run_report()
        if '--figures' in args:
            save_figures()
    elif '--figures' in args:
        save_figures()
    else:
        main()
