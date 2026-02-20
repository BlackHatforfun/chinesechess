"""
\u4e2d\u56fd\u8c61\u68cb - \u5f3a\u5316\u7248 AI \u5f15\u64ce\uff08PGZero\uff09
"""

import pgzrun
import pygame
import random
import time
import math
from array import array
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

try:
    import winsound
except Exception:
    winsound = None

# ==================== \u5e38\u91cf\u5b9a\u4e49 ====================
WIDTH = 900
HEIGHT = 1050
TITLE = "\u601d\u8fdc\u8c61\u68cb"
BOARD_SIZE = 10
BOARD_WIDTH = 9
MARGIN = 50

Move = Tuple[Tuple[int, int], Tuple[int, int]]

HEADER_HEIGHT = 120
BOARD_SCALE = 0.90
CONTROL_BAR_TOP = HEADER_HEIGHT + 8
CONTROL_BUTTON_HEIGHT = 34
CONTROL_BAR_ROW_GAP = 8
CONTROL_BAR_HEIGHT = CONTROL_BUTTON_HEIGHT * 2 + CONTROL_BAR_ROW_GAP
BOARD_OUTER_FRAME_MARGIN = 26
BOARD_MIN_GAP_TO_CONTROLS = 8
PLAY_AREA_TOP = CONTROL_BAR_TOP + CONTROL_BAR_HEIGHT + 14
PLAY_AREA_BOTTOM = HEIGHT - 110
BOARD_VERTICAL_SHIFT = 41
BASE_BOARD_WIDTH = (WIDTH - 78 * 2)
BASE_BOARD_HEIGHT = (HEIGHT - 130) - (HEADER_HEIGHT + 28)
BOARD_PIXEL_WIDTH = int(round(BASE_BOARD_WIDTH * BOARD_SCALE))
BOARD_PIXEL_HEIGHT = int(round(BASE_BOARD_HEIGHT * BOARD_SCALE))
BOARD_LEFT = (WIDTH - BOARD_PIXEL_WIDTH) // 2
BOARD_RIGHT = BOARD_LEFT + BOARD_PIXEL_WIDTH
BOARD_TOP_BASE = PLAY_AREA_TOP + (PLAY_AREA_BOTTOM - PLAY_AREA_TOP - BOARD_PIXEL_HEIGHT) // 2
BOARD_MIN_TOP = CONTROL_BAR_TOP + CONTROL_BAR_HEIGHT + BOARD_MIN_GAP_TO_CONTROLS + BOARD_OUTER_FRAME_MARGIN
BOARD_TOP = min(
    PLAY_AREA_BOTTOM - BOARD_PIXEL_HEIGHT,
    max(BOARD_MIN_TOP, BOARD_TOP_BASE + BOARD_VERTICAL_SHIFT),
)
BOARD_BOTTOM = BOARD_TOP + BOARD_PIXEL_HEIGHT
BOARD_PIXEL_WIDTH = BOARD_RIGHT - BOARD_LEFT
BOARD_PIXEL_HEIGHT = BOARD_BOTTOM - BOARD_TOP
X_STEP = BOARD_PIXEL_WIDTH / (BOARD_WIDTH - 1)
Y_STEP = BOARD_PIXEL_HEIGHT / (BOARD_SIZE - 1)


def board_to_screen(row: int, col: int) -> Tuple[int, int]:
    x = BOARD_LEFT + col * X_STEP
    y = BOARD_TOP + row * Y_STEP
    return int(round(x)), int(round(y))


def screen_to_board(pos: Tuple[int, int]) -> Optional[Tuple[int, int]]:
    x, y = pos
    col = int(round((x - BOARD_LEFT) / X_STEP))
    row = int(round((y - BOARD_TOP) / Y_STEP))

    if not (0 <= row < BOARD_SIZE and 0 <= col < BOARD_WIDTH):
        return None

    cx, cy = board_to_screen(row, col)
    tolerance = min(X_STEP, Y_STEP) * 0.45
    if (x - cx) ** 2 + (y - cy) ** 2 > tolerance * tolerance:
        return None

    return row, col


class Difficulty(Enum):
    """AI \u96be\u5ea6\u7b49\u7ea7"""

    EASY = 2
    NORMAL = 3
    HARD = 4
    MASTER = 5


class PieceType(Enum):
    GENERAL = "G"
    ADVISOR = "A"
    ELEPHANT = "E"
    HORSE = "H"
    CHARIOT = "R"
    CANNON = "C"
    PAWN = "P"


class Player(Enum):
    RED = 1
    BLACK = -1


USE_ASCII_PIECE_LABELS = False

UI_FONT_CJK_CANDIDATES = [
    "microsoftyahei",
    "simhei",
    "simsun",
    "nsimsun",
    "fangsong",
    "dengxian",
    "kaiti",
]
UI_FONT_FORCE = ""

_UI_FONT_NAME_CACHE: Optional[str] = None
_FONT_CACHE: Dict[Tuple[str, int, bool], pygame.font.Font] = {}
_FONT_CJK_SUPPORT_CACHE: Dict[str, bool] = {}
_SFX_READY = False
_MOVE_SOUND: Optional[pygame.mixer.Sound] = None
_CAPTURE_SOUND: Optional[pygame.mixer.Sound] = None

ASCII_PIECE_LABELS = {
    PieceType.GENERAL: "K",
    PieceType.ADVISOR: "A",
    PieceType.ELEPHANT: "E",
    PieceType.HORSE: "H",
    PieceType.CHARIOT: "R",
    PieceType.CANNON: "C",
    PieceType.PAWN: "P",
}

RED_ZH_PIECE_LABELS = {
    PieceType.GENERAL: "\u5e05",
    PieceType.ADVISOR: "\u4ed5",
    PieceType.ELEPHANT: "\u76f8",
    PieceType.HORSE: "\u9a6c",
    PieceType.CHARIOT: "\u8f66",
    PieceType.CANNON: "\u70ae",
    PieceType.PAWN: "\u5175",
}

BLACK_ZH_PIECE_LABELS = {
    PieceType.GENERAL: "\u5c06",
    PieceType.ADVISOR: "\u58eb",
    PieceType.ELEPHANT: "\u8c61",
    PieceType.HORSE: "\u9a6c",
    PieceType.CHARIOT: "\u8f66",
    PieceType.CANNON: "\u70ae",
    PieceType.PAWN: "\u5352",
}


@dataclass
class TTEntry:
    depth: int
    score: int
    flag: str
    best_move: Optional[Move]
    age: int


class Piece:
    def __init__(self, piece_type: PieceType, player: Player):
        self.type = piece_type
        self.player = player

    def __repr__(self):
        prefix = "R" if self.player == Player.RED else "B"
        return f"{prefix}{self.type.value}"

    def copy(self):
        return Piece(self.type, self.player)


def piece_label(piece: Piece) -> str:
    force_ascii = USE_ASCII_PIECE_LABELS
    if not force_ascii and pygame.font.get_init():
        force_ascii = not font_supports_cjk(get_ui_font_name())

    if force_ascii:
        label = ASCII_PIECE_LABELS[piece.type]
        return label if piece.player == Player.RED else label.lower()
    if piece.player == Player.RED:
        return RED_ZH_PIECE_LABELS[piece.type]
    return BLACK_ZH_PIECE_LABELS[piece.type]


def font_supports_cjk(font_name: str) -> bool:
    cached = _FONT_CJK_SUPPORT_CACHE.get(font_name)
    if cached is not None:
        return cached

    try:
        font = pygame.font.SysFont(font_name, 28)
    except Exception:
        _FONT_CJK_SUPPORT_CACHE[font_name] = False
        return False

    probe = "\u5e05\u5c06\u695a\u6cb3\u6c49\u754c"
    metrics = font.metrics(probe)
    if not metrics or any(m is None for m in metrics):
        _FONT_CJK_SUPPORT_CACHE[font_name] = False
        return False

    # \u82e5\u5b57\u5f62\u4e0e\u901a\u7528\u65b9\u5757\u5b8c\u5168\u4e00\u81f4\uff0c\u901a\u5e38\u662f\u7f3a\u5b57\u5f62\u56de\u9000
    probe_surf = font.render("\u5e05", True, (0, 0, 0))
    tofu_surf = font.render("\u25a1", True, (0, 0, 0))
    same_size = probe_surf.get_size() == tofu_surf.get_size()
    if same_size:
        probe_rgba = pygame.image.tostring(probe_surf, "RGBA")
        tofu_rgba = pygame.image.tostring(tofu_surf, "RGBA")
        if probe_rgba == tofu_rgba:
            _FONT_CJK_SUPPORT_CACHE[font_name] = False
            return False

    _FONT_CJK_SUPPORT_CACHE[font_name] = True
    return True


def get_ui_font_name() -> str:
    global _UI_FONT_NAME_CACHE
    if _UI_FONT_NAME_CACHE is not None:
        return _UI_FONT_NAME_CACHE

    if not pygame.font.get_init():
        pygame.font.init()

    installed = set(pygame.font.get_fonts())

    if UI_FONT_FORCE:
        forced = UI_FONT_FORCE.strip().lower().replace(" ", "")
        if forced in installed:
            _UI_FONT_NAME_CACHE = forced
            return forced

    for name in UI_FONT_CJK_CANDIDATES:
        if name in installed and font_supports_cjk(name):
            _UI_FONT_NAME_CACHE = name
            return name

    for name in UI_FONT_CJK_CANDIDATES:
        if name in installed:
            _UI_FONT_NAME_CACHE = name
            return name

    _UI_FONT_NAME_CACHE = "arial"
    return _UI_FONT_NAME_CACHE


def get_ui_font(size: int, bold: bool = False) -> pygame.font.Font:
    font_name = get_ui_font_name()
    key = (font_name, size, bold)
    if key not in _FONT_CACHE:
        _FONT_CACHE[key] = pygame.font.SysFont(font_name, size, bold=bold)
    return _FONT_CACHE[key]


def draw_ui_text(
    text: str,
    pos: Tuple[int, int],
    color: Tuple[int, int, int],
    size: int,
    bold: bool = False,
    center: bool = False,
):
    font = get_ui_font(size, bold=bold)
    surf = font.render(text, True, color)
    rect = surf.get_rect()
    if center:
        rect.center = pos
    else:
        rect.topleft = pos
    screen.surface.blit(surf, rect)


def _make_tone_sound(frequency_hz: int, duration_sec: float, volume: float) -> Optional[pygame.mixer.Sound]:
    if pygame.mixer.get_init() is None:
        return None

    sample_rate = 22050
    num_samples = max(1, int(sample_rate * duration_sec))
    amplitude = int(32767 * max(0.0, min(1.0, volume)))
    samples = array("h")

    for i in range(num_samples):
        t = i / sample_rate
        envelope = max(0.0, 1.0 - i / num_samples)
        value = int(amplitude * math.sin(2 * math.pi * frequency_hz * t) * envelope)
        samples.append(value)

    try:
        return pygame.mixer.Sound(buffer=samples.tobytes())
    except Exception:
        return None


def ensure_sfx_ready():
    global _SFX_READY, _MOVE_SOUND, _CAPTURE_SOUND
    if _SFX_READY:
        return

    try:
        if pygame.mixer.get_init() is None:
            pygame.mixer.init(frequency=22050, size=-16, channels=1, buffer=512)
        _MOVE_SOUND = _make_tone_sound(760, 0.065, 0.35)
        _CAPTURE_SOUND = _make_tone_sound(510, 0.12, 0.42)
    except Exception:
        _MOVE_SOUND = None
        _CAPTURE_SOUND = None
    finally:
        _SFX_READY = True


def play_move_sound(captured: bool):
    ensure_sfx_ready()

    tone = _CAPTURE_SOUND if captured else _MOVE_SOUND
    if tone is not None:
        try:
            tone.play()
            return
        except Exception:
            pass

    if winsound is not None:
        try:
            winsound.MessageBeep(-1 if captured else 0)
        except Exception:
            pass


class ChessBoard:
    def __init__(self):
        self.board = [[None for _ in range(BOARD_WIDTH)] for _ in range(BOARD_SIZE)]
        self._init_piece_index()
        self.initialize_board()

    def initialize_board(self):
        self.board[0] = [
            Piece(PieceType.CHARIOT, Player.BLACK),
            Piece(PieceType.HORSE, Player.BLACK),
            Piece(PieceType.ELEPHANT, Player.BLACK),
            Piece(PieceType.ADVISOR, Player.BLACK),
            Piece(PieceType.GENERAL, Player.BLACK),
            Piece(PieceType.ADVISOR, Player.BLACK),
            Piece(PieceType.ELEPHANT, Player.BLACK),
            Piece(PieceType.HORSE, Player.BLACK),
            Piece(PieceType.CHARIOT, Player.BLACK),
        ]
        self.board[2] = [
            None,
            Piece(PieceType.CANNON, Player.BLACK),
            None,
            None,
            None,
            None,
            None,
            Piece(PieceType.CANNON, Player.BLACK),
            None,
        ]
        for col in range(BOARD_WIDTH):
            if col % 2 == 0:
                self.board[3][col] = Piece(PieceType.PAWN, Player.BLACK)

        self.board[9] = [
            Piece(PieceType.CHARIOT, Player.RED),
            Piece(PieceType.HORSE, Player.RED),
            Piece(PieceType.ELEPHANT, Player.RED),
            Piece(PieceType.ADVISOR, Player.RED),
            Piece(PieceType.GENERAL, Player.RED),
            Piece(PieceType.ADVISOR, Player.RED),
            Piece(PieceType.ELEPHANT, Player.RED),
            Piece(PieceType.HORSE, Player.RED),
            Piece(PieceType.CHARIOT, Player.RED),
        ]
        self.board[7] = [
            None,
            Piece(PieceType.CANNON, Player.RED),
            None,
            None,
            None,
            None,
            None,
            Piece(PieceType.CANNON, Player.RED),
            None,
        ]
        for col in range(BOARD_WIDTH):
            if col % 2 == 0:
                self.board[6][col] = Piece(PieceType.PAWN, Player.RED)

        self._rebuild_piece_index()

    def copy(self):
        new_board = ChessBoard.__new__(ChessBoard)
        new_board.board = [
            [self.board[r][c].copy() if self.board[r][c] else None for c in range(BOARD_WIDTH)]
            for r in range(BOARD_SIZE)
        ]
        new_board._init_piece_index()
        new_board._rebuild_piece_index()
        return new_board

    def _init_piece_index(self):
        self.piece_positions: Dict[Player, set] = {
            Player.RED: set(),
            Player.BLACK: set(),
        }
        self.general_positions: Dict[Player, Optional[Tuple[int, int]]] = {
            Player.RED: None,
            Player.BLACK: None,
        }

    def _index_add_piece(self, row: int, col: int, piece: Piece):
        self.piece_positions[piece.player].add((row, col))
        if piece.type == PieceType.GENERAL:
            self.general_positions[piece.player] = (row, col)

    def _index_remove_piece(self, row: int, col: int, piece: Piece):
        self.piece_positions[piece.player].discard((row, col))
        if piece.type == PieceType.GENERAL and self.general_positions[piece.player] == (row, col):
            self.general_positions[piece.player] = None

    def _rebuild_piece_index(self):
        self._init_piece_index()
        for r in range(BOARD_SIZE):
            for c in range(BOARD_WIDTH):
                piece = self.board[r][c]
                if piece:
                    self._index_add_piece(r, c, piece)

    @staticmethod
    def is_in_bounds(row: int, col: int) -> bool:
        return 0 <= row < BOARD_SIZE and 0 <= col < BOARD_WIDTH

    def get_piece(self, row: int, col: int) -> Optional[Piece]:
        if self.is_in_bounds(row, col):
            return self.board[row][col]
        return None

    def set_piece(self, row: int, col: int, piece: Optional[Piece]):
        if self.is_in_bounds(row, col):
            existing = self.board[row][col]
            if existing is not None:
                self._index_remove_piece(row, col, existing)
            self.board[row][col] = piece
            if piece is not None:
                self._index_add_piece(row, col, piece)

    @staticmethod
    def is_in_player_palace(player: Player, row: int, col: int) -> bool:
        if not (3 <= col <= 5):
            return False
        if player == Player.RED:
            return 7 <= row <= 9
        return 0 <= row <= 2

    def find_general(self, player: Player) -> Optional[Tuple[int, int]]:
        return self.general_positions[player]

    def get_piece_positions(self, player: Player):
        return self.piece_positions[player]

    def make_move(self, move: Move) -> Tuple[Optional[Piece], Optional[Piece]]:
        from_pos, to_pos = move
        moving_piece = self.get_piece(from_pos[0], from_pos[1])
        captured_piece = self.get_piece(to_pos[0], to_pos[1])
        self.set_piece(from_pos[0], from_pos[1], None)
        self.set_piece(to_pos[0], to_pos[1], moving_piece)
        return moving_piece, captured_piece

    def unmake_move(self, move: Move, moving_piece: Optional[Piece], captured_piece: Optional[Piece]):
        from_pos, to_pos = move
        self.set_piece(to_pos[0], to_pos[1], captured_piece)
        self.set_piece(from_pos[0], from_pos[1], moving_piece)


class MoveValidator:
    @staticmethod
    def is_valid_move(board: ChessBoard, from_pos: Tuple[int, int], to_pos: Tuple[int, int]) -> bool:
        r1, c1 = from_pos
        r2, c2 = to_pos

        if from_pos == to_pos:
            return False
        if not board.is_in_bounds(r1, c1) or not board.is_in_bounds(r2, c2):
            return False

        piece = board.get_piece(r1, c1)
        target = board.get_piece(r2, c2)
        if piece is None:
            return False
        if target and target.player == piece.player:
            return False

        if not MoveValidator._is_piece_pattern_valid(board, piece, from_pos, to_pos, target):
            return False

        moving_piece, captured_piece = board.make_move((from_pos, to_pos))
        legal = not MoveValidator.is_in_check(board, piece.player)
        board.unmake_move((from_pos, to_pos), moving_piece, captured_piece)
        return legal

    @staticmethod
    def is_in_check(board: ChessBoard, player: Player) -> bool:
        general_pos = board.find_general(player)
        if general_pos is None:
            return True

        enemy = Player.BLACK if player == Player.RED else Player.RED
        return MoveValidator._is_square_attacked(board, general_pos, enemy)

    @staticmethod
    def _is_square_attacked(board: ChessBoard, target: Tuple[int, int], attacker: Player) -> bool:
        tr, tc = target

        # 鐩寸嚎鏀诲嚮锛氳溅/鐐?椋炲皢
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            r, c = tr + dr, tc + dc
            blocker_seen = False

            while board.is_in_bounds(r, c):
                piece = board.get_piece(r, c)
                if piece is None:
                    r += dr
                    c += dc
                    continue

                if not blocker_seen:
                    if piece.player == attacker:
                        if piece.type == PieceType.CHARIOT:
                            return True
                        if dc == 0 and piece.type == PieceType.GENERAL:
                            return True
                    blocker_seen = True
                else:
                    if piece.player == attacker and piece.type == PieceType.CANNON:
                        return True
                    break

                r += dr
                c += dc

        if MoveValidator._is_horse_attack(board, target, attacker):
            return True
        if MoveValidator._is_pawn_attack(board, target, attacker):
            return True

        return False

    @staticmethod
    def _is_horse_attack(board: ChessBoard, target: Tuple[int, int], attacker: Player) -> bool:
        tr, tc = target
        horse_attackers = [
            (tr - 2, tc - 1, tr - 1, tc),
            (tr - 2, tc + 1, tr - 1, tc),
            (tr + 2, tc - 1, tr + 1, tc),
            (tr + 2, tc + 1, tr + 1, tc),
            (tr - 1, tc - 2, tr, tc - 1),
            (tr + 1, tc - 2, tr, tc - 1),
            (tr - 1, tc + 2, tr, tc + 1),
            (tr + 1, tc + 2, tr, tc + 1),
        ]

        for hr, hc, lr, lc in horse_attackers:
            if not board.is_in_bounds(hr, hc) or not board.is_in_bounds(lr, lc):
                continue
            piece = board.get_piece(hr, hc)
            if piece and piece.player == attacker and piece.type == PieceType.HORSE:
                if board.get_piece(lr, lc) is None:
                    return True
        return False

    @staticmethod
    def _is_pawn_attack(board: ChessBoard, target: Tuple[int, int], attacker: Player) -> bool:
        tr, tc = target

        if attacker == Player.RED:
            origins = [(tr + 1, tc)]
            if tr <= 4:
                origins.extend([(tr, tc - 1), (tr, tc + 1)])
        else:
            origins = [(tr - 1, tc)]
            if tr >= 5:
                origins.extend([(tr, tc - 1), (tr, tc + 1)])

        for pr, pc in origins:
            if not board.is_in_bounds(pr, pc):
                continue
            piece = board.get_piece(pr, pc)
            if piece and piece.player == attacker and piece.type == PieceType.PAWN:
                return True
        return False

    @staticmethod
    def _is_piece_pattern_valid(
        board: ChessBoard,
        piece: Piece,
        from_pos: Tuple[int, int],
        to_pos: Tuple[int, int],
        target: Optional[Piece],
    ) -> bool:
        if piece.type == PieceType.GENERAL:
            return MoveValidator._validate_general(board, piece, from_pos, to_pos, target)
        if piece.type == PieceType.ADVISOR:
            return MoveValidator._validate_advisor(piece, from_pos, to_pos)
        if piece.type == PieceType.ELEPHANT:
            return MoveValidator._validate_elephant(board, piece, from_pos, to_pos)
        if piece.type == PieceType.HORSE:
            return MoveValidator._validate_horse(board, from_pos, to_pos)
        if piece.type == PieceType.CHARIOT:
            return MoveValidator._validate_chariot(board, from_pos, to_pos)
        if piece.type == PieceType.CANNON:
            return MoveValidator._validate_cannon(board, from_pos, to_pos, target)
        if piece.type == PieceType.PAWN:
            return MoveValidator._validate_pawn(piece, from_pos, to_pos)
        return False

    @staticmethod
    def _validate_general(
        board: ChessBoard,
        piece: Piece,
        from_pos: Tuple[int, int],
        to_pos: Tuple[int, int],
        target: Optional[Piece],
    ) -> bool:
        r1, c1 = from_pos
        r2, c2 = to_pos

        if target and target.type == PieceType.GENERAL and c1 == c2:
            return MoveValidator._count_pieces_between(board, from_pos, to_pos) == 0

        if not board.is_in_player_palace(piece.player, r2, c2):
            return False
        return abs(r1 - r2) + abs(c1 - c2) == 1

    @staticmethod
    def _validate_advisor(piece: Piece, from_pos: Tuple[int, int], to_pos: Tuple[int, int]) -> bool:
        r1, c1 = from_pos
        r2, c2 = to_pos
        return (
            ChessBoard.is_in_player_palace(piece.player, r2, c2)
            and abs(r1 - r2) == 1
            and abs(c1 - c2) == 1
        )

    @staticmethod
    def _validate_elephant(
        board: ChessBoard, piece: Piece, from_pos: Tuple[int, int], to_pos: Tuple[int, int]
    ) -> bool:
        r1, c1 = from_pos
        r2, c2 = to_pos

        if abs(r1 - r2) != 2 or abs(c1 - c2) != 2:
            return False
        if piece.player == Player.RED and r2 < 5:
            return False
        if piece.player == Player.BLACK and r2 > 4:
            return False

        eye_r, eye_c = (r1 + r2) // 2, (c1 + c2) // 2
        return board.get_piece(eye_r, eye_c) is None

    @staticmethod
    def _validate_horse(board: ChessBoard, from_pos: Tuple[int, int], to_pos: Tuple[int, int]) -> bool:
        r1, c1 = from_pos
        r2, c2 = to_pos
        dr, dc = r2 - r1, c2 - c1
        abs_dr, abs_dc = abs(dr), abs(dc)

        if abs_dr == 2 and abs_dc == 1:
            leg_r, leg_c = r1 + (1 if dr > 0 else -1), c1
            return board.get_piece(leg_r, leg_c) is None
        if abs_dr == 1 and abs_dc == 2:
            leg_r, leg_c = r1, c1 + (1 if dc > 0 else -1)
            return board.get_piece(leg_r, leg_c) is None
        return False

    @staticmethod
    def _validate_chariot(board: ChessBoard, from_pos: Tuple[int, int], to_pos: Tuple[int, int]) -> bool:
        r1, c1 = from_pos
        r2, c2 = to_pos
        if r1 != r2 and c1 != c2:
            return False
        return MoveValidator._count_pieces_between(board, from_pos, to_pos) == 0

    @staticmethod
    def _validate_cannon(
        board: ChessBoard,
        from_pos: Tuple[int, int],
        to_pos: Tuple[int, int],
        target: Optional[Piece],
    ) -> bool:
        r1, c1 = from_pos
        r2, c2 = to_pos
        if r1 != r2 and c1 != c2:
            return False

        between = MoveValidator._count_pieces_between(board, from_pos, to_pos)
        if target is None:
            return between == 0
        return between == 1

    @staticmethod
    def _validate_pawn(piece: Piece, from_pos: Tuple[int, int], to_pos: Tuple[int, int]) -> bool:
        r1, c1 = from_pos
        r2, c2 = to_pos

        direction = -1 if piece.player == Player.RED else 1
        dr, dc = r2 - r1, c2 - c1

        if dr == direction and dc == 0:
            return True

        crossed_river = (r1 <= 4) if piece.player == Player.RED else (r1 >= 5)
        if crossed_river and dr == 0 and abs(dc) == 1:
            return True

        return False

    @staticmethod
    def _count_pieces_between(board: ChessBoard, from_pos: Tuple[int, int], to_pos: Tuple[int, int]) -> int:
        r1, c1 = from_pos
        r2, c2 = to_pos

        if r1 == r2:
            step = 1 if c2 > c1 else -1
            return sum(1 for c in range(c1 + step, c2, step) if board.get_piece(r1, c) is not None)

        if c1 == c2:
            step = 1 if r2 > r1 else -1
            return sum(1 for r in range(r1 + step, r2, step) if board.get_piece(r, c1) is not None)

        return -1

class AdvancedAIEngine:
    PIECE_VALUES = {
        PieceType.GENERAL: 100000,
        PieceType.CHARIOT: 900,
        PieceType.CANNON: 470,
        PieceType.HORSE: 430,
        PieceType.ELEPHANT: 220,
        PieceType.ADVISOR: 220,
        PieceType.PAWN: 130,
    }

    PIECE_INDEX = {
        PieceType.GENERAL: 0,
        PieceType.ADVISOR: 1,
        PieceType.ELEPHANT: 2,
        PieceType.HORSE: 3,
        PieceType.CHARIOT: 4,
        PieceType.CANNON: 5,
        PieceType.PAWN: 6,
    }

    INF = 10**9
    MATE_SCORE = 10**7
    MAX_PLY = 64
    TT_LIMIT = 300000
    QUIESCENCE_DELTA_MARGIN = 70
    LMR_MIN_DEPTH = 3
    LMR_MIN_MOVE_INDEX = 3
    NULL_MIN_DEPTH = 3
    LMP_MAX_DEPTH = 3
    IID_MIN_DEPTH = 5
    TT_AGE_REPLACE = 4
    TT_AGE_DROP = 7
    HISTORY_MAX = 200_000
    HISTORY_DECAY_SHIFT = 2
    CAPTURE_HISTORY_MAX = 220_000
    CAPTURE_HISTORY_DECAY_SHIFT = 2
    COUNTERMOVE_BONUS = 780_000
    PV_MOVE_BONUS = 860_000
    SEE_ORDER_SCALE = 36
    SEE_BAD_CAPTURE_THRESHOLD = -80
    SEE_BAD_CAPTURE_PENALTY = 260_000
    SEE_QS_PRUNE_MARGIN = -140
    CHECK_EXTENSION_DEPTH_LIMIT = 2
    RECAPTURE_EXTENSION_DEPTH_LIMIT = 3
    MAX_EXTENSIONS_PER_PATH = 1
    RECAPTURE_SEE_MIN = -10
    EXTENSION_TIME_RATIO_LIMIT = 0.65
    SINGULAR_MIN_DEPTH = 7
    SINGULAR_VERIFY_REDUCTION = 4
    SINGULAR_MARGIN_BASE_EXACT = 18
    SINGULAR_MARGIN_BASE_LOWER = 30
    SINGULAR_MARGIN_PER_DEPTH = 10
    SINGULAR_MARGIN_MIN = 16
    SINGULAR_MARGIN_MAX = 140
    SINGULAR_ALT_LIMIT = 4
    SINGULAR_TIME_RATIO_LIMIT = 0.30

    def __init__(self, difficulty: Difficulty = Difficulty.NORMAL):
        depth_map = {
            Difficulty.EASY: 3,
            Difficulty.NORMAL: 4,
            Difficulty.HARD: 6,
            Difficulty.MASTER: 7,
        }
        time_map = {
            Difficulty.EASY: 0.25,
            Difficulty.NORMAL: 0.8,
            Difficulty.HARD: 4.0,
            Difficulty.MASTER: 8.0,
        }

        self.max_depth = depth_map[difficulty]
        self.time_limit = time_map[difficulty]
        self.quiescence_depth = 5 if difficulty in (Difficulty.HARD, Difficulty.MASTER) else 3

        self.transposition_table: Dict[int, TTEntry] = {}
        self.history: Dict[Tuple[int, Move], int] = {}
        self.capture_history: Dict[Tuple[int, int, int, int, int], int] = {}
        self.counter_moves: Dict[Tuple[int, Move], Move] = {}
        self.killer_moves: List[List[Optional[Move]]] = [[None, None] for _ in range(self.MAX_PLY)]
        self.move_stack: List[Tuple[Move, Optional[Piece], Optional[Piece]]] = []
        self.pv_table: List[List[Optional[Move]]] = [
            [None for _ in range(self.MAX_PLY)] for _ in range(self.MAX_PLY)
        ]
        self.pv_length: List[int] = [0 for _ in range(self.MAX_PLY + 1)]
        self.pv_hint: List[Optional[Move]] = [None for _ in range(self.MAX_PLY)]

        rng = random.Random(20260220)
        self.zobrist = [
            [[rng.getrandbits(64) for _ in range(14)] for _ in range(BOARD_WIDTH)]
            for _ in range(BOARD_SIZE)
        ]
        self.side_hash = rng.getrandbits(64)

        self.nodes = 0
        self.stop_search = False
        self.search_start = 0.0
        self.tt_generation = 0

        self.last_search_nodes = 0
        self.last_search_depth = 0
        self.last_search_time = 0.0
        self.last_search_score = 0

    def get_all_legal_moves(self, board: ChessBoard, player: Player) -> List[Move]:
        moves = self._generate_legal_moves(board, player)
        return self._order_moves(board, moves, player, 0, None, None, None)

    def find_best_move(self, board: ChessBoard, player: Player) -> Optional[Move]:
        legal_moves = self._generate_legal_moves(board, player)
        if not legal_moves:
            self.last_search_nodes = 0
            self.last_search_depth = 0
            self.last_search_time = 0.0
            self.last_search_score = -self.MATE_SCORE
            return None

        root_hash = self._compute_hash(board, player)
        best_move = legal_moves[0]
        best_score = -self.INF

        self.nodes = 0
        self.move_stack.clear()
        self.stop_search = False
        self.search_start = time.perf_counter()
        self.tt_generation = (self.tt_generation + 1) & 0xFFFF
        self._decay_history()
        self._decay_capture_history()
        self.pv_hint = [None for _ in range(self.MAX_PLY)]
        completed_depth = 0

        for depth in range(1, self.max_depth + 1):
            alpha = -self.INF
            beta = self.INF

            if depth >= 3 and best_score > -self.INF // 2:
                window = 120
                alpha = best_score - window
                beta = best_score + window

            self._clear_pv_lengths()
            score, move = self._negamax(board, depth, alpha, beta, player, 0, root_hash, True)
            if self.stop_search:
                break

            if score <= alpha or score >= beta:
                self._clear_pv_lengths()
                score, move = self._negamax(board, depth, -self.INF, self.INF, player, 0, root_hash, True)
                if self.stop_search:
                    break

            if move is None and self.pv_length[0] > 0:
                move = self.pv_table[0][0]

            if move is not None:
                best_move = move
                best_score = score
                completed_depth = depth
                self._refresh_pv_hint()

        self.last_search_time = time.perf_counter() - self.search_start
        self.last_search_nodes = self.nodes
        self.last_search_depth = completed_depth
        self.last_search_score = best_score if best_score > -self.INF // 2 else 0
        self._prune_transposition_table_if_needed()
        return best_move

    def _negamax(
        self,
        board: ChessBoard,
        depth: int,
        alpha: int,
        beta: int,
        player: Player,
        ply: int,
        hash_key: int,
        allow_null: bool,
        allow_iid: bool = True,
        prev_move: Optional[Move] = None,
        extensions_used: int = 0,
        allow_singular: bool = True,
    ) -> Tuple[int, Optional[Move]]:
        if self.stop_search:
            return 0, None

        if ply <= self.MAX_PLY:
            self.pv_length[ply] = ply

        self.nodes += 1
        if (self.nodes & 1023) == 0 and (time.perf_counter() - self.search_start) >= self.time_limit:
            self.stop_search = True
            return 0, None

        alpha_orig = alpha
        beta_orig = beta

        entry = self.transposition_table.get(hash_key)
        if entry is not None:
            entry.age = self.tt_generation
        if entry and entry.depth >= depth:
            if entry.flag == "EXACT":
                if entry.best_move is not None:
                    self._update_pv_line(ply, entry.best_move)
                return entry.score, entry.best_move
            if entry.flag == "LOWER":
                alpha = max(alpha, entry.score)
            else:
                beta = min(beta, entry.score)
            if alpha >= beta:
                if entry.best_move is not None:
                    self._update_pv_line(ply, entry.best_move)
                return entry.score, entry.best_move

        in_check = MoveValidator.is_in_check(board, player)
        if depth <= 1 and in_check:
            depth += 1

        if depth <= 0:
            score = self._quiescence(board, alpha, beta, player, ply, hash_key, self.quiescence_depth)
            return score, None

        opponent = Player.BLACK if player == Player.RED else Player.RED
        is_pv_node = (beta - alpha) > 1

        # Null Move Pruning
        if (
            allow_null
            and not is_pv_node
            and not in_check
            and depth >= self.NULL_MIN_DEPTH
            and self._has_major_material(board, player)
        ):
            reduction = 2 + depth // 4
            null_depth = max(0, depth - 1 - reduction)
            null_score, _ = self._negamax(
                board,
                null_depth,
                -beta,
                -(beta - 1),
                opponent,
                ply + 1,
                hash_key ^ self.side_hash,
                False,
                False,
                None,
                extensions_used,
                False,
            )
            null_score = -null_score
            if self.stop_search:
                return 0, None
            if null_score >= beta:
                self._store_tt(hash_key, depth, null_score, "LOWER", None)
                return null_score, None

        moves = self._generate_legal_moves(board, player)
        if not moves:
            return -self.MATE_SCORE + ply, None

        tt_move = entry.best_move if entry else None
        if tt_move is None and allow_iid and is_pv_node and depth >= self.IID_MIN_DEPTH:
            iid_depth = max(1, depth - 2)
            _, iid_move = self._negamax(
                board,
                iid_depth,
                alpha,
                beta,
                player,
                ply,
                hash_key,
                False,
                False,
                prev_move,
                extensions_used,
                False,
            )
            if self.stop_search:
                return 0, None
            if iid_move is not None:
                tt_move = iid_move

        countermove = self.counter_moves.get((player.value, prev_move)) if prev_move is not None else None
        pv_move = None
        if tt_move is None and ply < self.MAX_PLY:
            pv_move = self.pv_hint[ply]
        ordered = self._order_moves(board, moves, player, ply, tt_move, countermove, pv_move)

        best_score = -self.INF
        best_move = None
        allow_extensions_here = (time.perf_counter() - self.search_start) < (
            self.time_limit * self.EXTENSION_TIME_RATIO_LIMIT
        )
        allow_singular_here = False
        if allow_singular and depth == self.max_depth:
            allow_singular_here = (time.perf_counter() - self.search_start) < (
                self.time_limit * self.SINGULAR_TIME_RATIO_LIMIT
            )
        singular_move: Optional[Move] = None
        if allow_singular_here and self._should_try_singular_extension(
            depth,
            is_pv_node,
            in_check,
            entry,
            tt_move,
            extensions_used,
            ordered,
        ):
            assert entry is not None and tt_move is not None
            singular_margin = self._adaptive_singular_margin(depth, entry)
            singular_beta = max(-self.INF + 2, min(self.INF - 2, entry.score - singular_margin))
            if self._verify_singular_move(
                board,
                depth,
                singular_beta,
                singular_margin,
                entry,
                player,
                opponent,
                ply,
                hash_key,
                ordered,
                tt_move,
                extensions_used,
            ):
                singular_move = tt_move

        for move_index, move in enumerate(ordered):
            moving_piece = board.get_piece(move[0][0], move[0][1])
            captured_piece = board.get_piece(move[1][0], move[1][1])
            recapture_ext = False
            if allow_extensions_here:
                recapture_ext = self._should_extend_recapture(
                    board,
                    depth,
                    move,
                    prev_move,
                    moving_piece,
                    captured_piece,
                    extensions_used,
                )
            next_hash = self._hash_after_move(hash_key, move, moving_piece, captured_piece)

            captured = self._make_move(board, move, moving_piece, captured_piece)
            gives_check = MoveValidator.is_in_check(board, opponent)
            check_ext = False
            if allow_extensions_here:
                check_ext = self._should_extend_check(
                    depth,
                    ply,
                    gives_check,
                    extensions_used,
                    captured_piece is not None,
                )
            if move_index > 1:
                check_ext = False
            singular_ext = singular_move == move
            extension = 1 if (check_ext or recapture_ext or singular_ext) else 0
            child_extensions_used = extensions_used + extension
            base_child_depth = depth - 1 + extension

            if move_index == 0:
                score, _ = self._negamax(
                    board,
                    base_child_depth,
                    -beta,
                    -alpha,
                    opponent,
                    ply + 1,
                    next_hash,
                    True,
                    True,
                    move,
                    child_extensions_used,
                )
                score = -score
            else:
                child_depth = base_child_depth
                reduced = False

                if self._should_apply_lmp(
                    depth,
                    move_index,
                    captured_piece is not None,
                    in_check,
                    gives_check,
                ):
                    self._unmake_move(board)
                    continue

                if self._should_apply_lmr(
                    depth,
                    move_index,
                    captured_piece is not None,
                    in_check,
                    gives_check,
                ):
                    reduction = self._lmr_reduction(depth, move_index)
                    if reduction > 0:
                        child_depth = max(1, child_depth - reduction)
                        reduced = True

                # PVS: 鍚庣画鐫€娉曞厛鍋氶浂绐楀彛鎼滅储
                score, _ = self._negamax(
                    board,
                    child_depth,
                    -(alpha + 1),
                    -alpha,
                    opponent,
                    ply + 1,
                    next_hash,
                    True,
                    True,
                    move,
                    child_extensions_used,
                )
                score = -score

                # LMR 鍛戒腑鍚庤嫢绐佺牬 alpha锛屽厛琛ヤ竴娆℃甯告繁搴﹂浂绐楀彛
                if reduced and score > alpha:
                    score, _ = self._negamax(
                        board,
                        base_child_depth,
                        -(alpha + 1),
                        -alpha,
                        opponent,
                        ply + 1,
                        next_hash,
                        True,
                        True,
                        move,
                        child_extensions_used,
                    )
                    score = -score

                # PVS 澶辫触楂橈紝琛ュ叏绐楀彛閲嶆悳
                if score > alpha and score < beta:
                    score, _ = self._negamax(
                        board,
                        base_child_depth,
                        -beta,
                        -alpha,
                        opponent,
                        ply + 1,
                        next_hash,
                        True,
                        True,
                        move,
                        child_extensions_used,
                    )
                    score = -score

            self._unmake_move(board)

            if self.stop_search:
                return 0, None

            if score > best_score:
                best_score = score
                best_move = move

            if score > alpha:
                alpha = score
                self._update_pv_line(ply, move)
                if captured is None:
                    self._record_history(player, move, depth)
                else:
                    self._record_capture_history(player, move, moving_piece, captured, depth)

            if alpha >= beta:
                if captured is None:
                    self._record_killer(ply, move)
                    self._record_countermove(player, prev_move, move)
                else:
                    self._record_capture_history(player, move, moving_piece, captured, depth + 1)
                break

        flag = "EXACT"
        if best_score <= alpha_orig:
            flag = "UPPER"
        elif best_score >= beta_orig:
            flag = "LOWER"

        self._store_tt(hash_key, depth, best_score, flag, best_move)
        return best_score, best_move

    def _quiescence(
        self,
        board: ChessBoard,
        alpha: int,
        beta: int,
        player: Player,
        ply: int,
        hash_key: int,
        depth_left: int,
    ) -> int:
        if self.stop_search:
            return 0
        if (self.nodes & 1023) == 0 and (time.perf_counter() - self.search_start) >= self.time_limit:
            self.stop_search = True
            return 0

        stand_pat = self.evaluate_position(board, player)
        if stand_pat >= beta:
            return stand_pat
        if stand_pat > alpha:
            alpha = stand_pat
        if depth_left <= 0:
            return stand_pat

        capture_moves = self._generate_legal_moves(board, player, captures_only=True)
        if not capture_moves:
            return stand_pat

        ordered = self._order_moves(board, capture_moves, player, ply, None, None, None)
        opponent = Player.BLACK if player == Player.RED else Player.RED

        for move in ordered:
            moving_piece = board.get_piece(move[0][0], move[0][1])
            captured_piece = board.get_piece(move[1][0], move[1][1])
            if captured_piece is None:
                continue

            if captured_piece.type != PieceType.GENERAL:
                gain_cap = self.PIECE_VALUES[captured_piece.type] + self.QUIESCENCE_DELTA_MARGIN
                if stand_pat + gain_cap <= alpha:
                    continue
                see_score = self._see_capture(board, move, moving_piece, captured_piece)
                if see_score < self.SEE_QS_PRUNE_MARGIN:
                    continue

            next_hash = self._hash_after_move(hash_key, move, moving_piece, captured_piece)

            self._make_move(board, move, moving_piece, captured_piece)
            score = -self._quiescence(
                board,
                -beta,
                -alpha,
                opponent,
                ply + 1,
                next_hash,
                depth_left - 1,
            )
            self._unmake_move(board)

            if self.stop_search:
                return 0
            if score >= beta:
                return score
            if score > alpha:
                alpha = score

        return alpha

    def evaluate_position(self, board: ChessBoard, player: Player) -> int:
        red_general = board.find_general(Player.RED)
        black_general = board.find_general(Player.BLACK)
        if red_general is None:
            return -self.MATE_SCORE if player == Player.RED else self.MATE_SCORE
        if black_general is None:
            return self.MATE_SCORE if player == Player.RED else -self.MATE_SCORE

        red_score = 0
        black_score = 0
        red_guards = 0
        black_guards = 0

        for r, c in board.get_piece_positions(Player.RED):
            piece = board.get_piece(r, c)
            if not piece:
                continue
            value = self.PIECE_VALUES[piece.type] + self._piece_square_bonus(piece, r, c)
            red_score += value
            if piece.type in (PieceType.ADVISOR, PieceType.ELEPHANT):
                red_guards += 1

        for r, c in board.get_piece_positions(Player.BLACK):
            piece = board.get_piece(r, c)
            if not piece:
                continue
            value = self.PIECE_VALUES[piece.type] + self._piece_square_bonus(piece, r, c)
            black_score += value
            if piece.type in (PieceType.ADVISOR, PieceType.ELEPHANT):
                black_guards += 1

        red_score += red_guards * 18
        black_score += black_guards * 18

        total = red_score - black_score
        return total if player == Player.RED else -total

    def _piece_square_bonus(self, piece: Piece, row: int, col: int) -> int:
        r = row if piece.player == Player.RED else 9 - row
        center = 4 - abs(col - 4)

        if piece.type == PieceType.GENERAL:
            return center * 8 - abs(r - 8) * 6
        if piece.type == PieceType.ADVISOR:
            if (r, col) in {(9, 4), (8, 3), (8, 5), (7, 4)}:
                return 14
            return 4
        if piece.type == PieceType.ELEPHANT:
            if r in (9, 7, 5):
                return 10
            return 4
        if piece.type == PieceType.HORSE:
            return center * 7 + (9 - r) * 2
        if piece.type == PieceType.CHARIOT:
            return center * 3 + (9 - r)
        if piece.type == PieceType.CANNON:
            return center * 5 + (9 - r)

        advance = 9 - r
        bonus = advance * 10
        if r <= 4:
            bonus += 30
        if r <= 2:
            bonus += 18
        if abs(col - 4) <= 1:
            bonus += 6
        return bonus

    def _generate_legal_moves(
        self, board: ChessBoard, player: Player, captures_only: bool = False
    ) -> List[Move]:
        legal_moves: List[Move] = []
        for move in self._generate_pseudo_legal_moves(board, player, captures_only):
            self._make_move(board, move)
            if not MoveValidator.is_in_check(board, player):
                legal_moves.append(move)
            self._unmake_move(board)
        return legal_moves

    def _generate_pseudo_legal_moves(
        self, board: ChessBoard, player: Player, captures_only: bool = False
    ) -> List[Move]:
        moves: List[Move] = []

        for r, c in board.get_piece_positions(player):
            piece = board.get_piece(r, c)
            if piece is None:
                continue

            if piece.type == PieceType.GENERAL:
                self._gen_general_moves(board, piece, r, c, moves, captures_only)
            elif piece.type == PieceType.ADVISOR:
                self._gen_advisor_moves(board, piece, r, c, moves, captures_only)
            elif piece.type == PieceType.ELEPHANT:
                self._gen_elephant_moves(board, piece, r, c, moves, captures_only)
            elif piece.type == PieceType.HORSE:
                self._gen_horse_moves(board, piece, r, c, moves, captures_only)
            elif piece.type == PieceType.CHARIOT:
                self._gen_chariot_moves(board, piece, r, c, moves, captures_only)
            elif piece.type == PieceType.CANNON:
                self._gen_cannon_moves(board, piece, r, c, moves, captures_only)
            else:
                self._gen_pawn_moves(board, piece, r, c, moves, captures_only)

        return moves

    def _append_move(
        self,
        board: ChessBoard,
        piece: Piece,
        from_pos: Tuple[int, int],
        to_pos: Tuple[int, int],
        moves: List[Move],
        captures_only: bool,
    ):
        tr, tc = to_pos
        if not board.is_in_bounds(tr, tc):
            return

        target = board.get_piece(tr, tc)
        if target and target.player == piece.player:
            return
        if captures_only and target is None:
            return

        moves.append((from_pos, to_pos))

    def _gen_general_moves(
        self,
        board: ChessBoard,
        piece: Piece,
        r: int,
        c: int,
        moves: List[Move],
        captures_only: bool,
    ):
        from_pos = (r, c)

        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if ChessBoard.is_in_player_palace(piece.player, nr, nc):
                self._append_move(board, piece, from_pos, (nr, nc), moves, captures_only)

        for dr in (-1, 1):
            nr = r + dr
            while 0 <= nr < BOARD_SIZE:
                target = board.get_piece(nr, c)
                if target:
                    if target.player != piece.player and target.type == PieceType.GENERAL:
                        moves.append((from_pos, (nr, c)))
                    break
                nr += dr

    def _gen_advisor_moves(
        self,
        board: ChessBoard,
        piece: Piece,
        r: int,
        c: int,
        moves: List[Move],
        captures_only: bool,
    ):
        from_pos = (r, c)
        for dr, dc in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
            nr, nc = r + dr, c + dc
            if ChessBoard.is_in_player_palace(piece.player, nr, nc):
                self._append_move(board, piece, from_pos, (nr, nc), moves, captures_only)

    def _gen_elephant_moves(
        self,
        board: ChessBoard,
        piece: Piece,
        r: int,
        c: int,
        moves: List[Move],
        captures_only: bool,
    ):
        from_pos = (r, c)
        for dr, dc in ((2, 2), (2, -2), (-2, 2), (-2, -2)):
            nr, nc = r + dr, c + dc
            if not board.is_in_bounds(nr, nc):
                continue
            if piece.player == Player.RED and nr < 5:
                continue
            if piece.player == Player.BLACK and nr > 4:
                continue

            eye_r, eye_c = r + dr // 2, c + dc // 2
            if board.get_piece(eye_r, eye_c) is not None:
                continue

            self._append_move(board, piece, from_pos, (nr, nc), moves, captures_only)

    def _gen_horse_moves(
        self,
        board: ChessBoard,
        piece: Piece,
        r: int,
        c: int,
        moves: List[Move],
        captures_only: bool,
    ):
        from_pos = (r, c)
        horse_steps = [
            ((2, 1), (1, 0)),
            ((2, -1), (1, 0)),
            ((-2, 1), (-1, 0)),
            ((-2, -1), (-1, 0)),
            ((1, 2), (0, 1)),
            ((-1, 2), (0, 1)),
            ((1, -2), (0, -1)),
            ((-1, -2), (0, -1)),
        ]

        for (dr, dc), (lr, lc) in horse_steps:
            leg_r, leg_c = r + lr, c + lc
            if not board.is_in_bounds(leg_r, leg_c):
                continue
            if board.get_piece(leg_r, leg_c) is not None:
                continue
            self._append_move(board, piece, from_pos, (r + dr, c + dc), moves, captures_only)

    def _gen_chariot_moves(
        self,
        board: ChessBoard,
        piece: Piece,
        r: int,
        c: int,
        moves: List[Move],
        captures_only: bool,
    ):
        from_pos = (r, c)
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            while board.is_in_bounds(nr, nc):
                target = board.get_piece(nr, nc)
                if target is None:
                    if not captures_only:
                        moves.append((from_pos, (nr, nc)))
                else:
                    if target.player != piece.player:
                        moves.append((from_pos, (nr, nc)))
                    break
                nr += dr
                nc += dc

    def _gen_cannon_moves(
        self,
        board: ChessBoard,
        piece: Piece,
        r: int,
        c: int,
        moves: List[Move],
        captures_only: bool,
    ):
        from_pos = (r, c)
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            jumped = False

            while board.is_in_bounds(nr, nc):
                target = board.get_piece(nr, nc)
                if not jumped:
                    if target is None:
                        if not captures_only:
                            moves.append((from_pos, (nr, nc)))
                    else:
                        jumped = True
                else:
                    if target is not None:
                        if target.player != piece.player:
                            moves.append((from_pos, (nr, nc)))
                        break
                nr += dr
                nc += dc

    def _gen_pawn_moves(
        self,
        board: ChessBoard,
        piece: Piece,
        r: int,
        c: int,
        moves: List[Move],
        captures_only: bool,
    ):
        from_pos = (r, c)
        direction = -1 if piece.player == Player.RED else 1
        self._append_move(board, piece, from_pos, (r + direction, c), moves, captures_only)

        crossed_river = (r <= 4) if piece.player == Player.RED else (r >= 5)
        if crossed_river:
            self._append_move(board, piece, from_pos, (r, c - 1), moves, captures_only)
            self._append_move(board, piece, from_pos, (r, c + 1), moves, captures_only)

    def _order_moves(
        self,
        board: ChessBoard,
        moves: List[Move],
        player: Player,
        ply: int,
        tt_move: Optional[Move],
        countermove: Optional[Move],
        pv_move: Optional[Move],
    ) -> List[Move]:
        return sorted(
            moves,
            key=lambda move: self._move_priority(board, move, player, ply, tt_move, countermove, pv_move),
            reverse=True,
        )

    def _move_priority(
        self,
        board: ChessBoard,
        move: Move,
        player: Player,
        ply: int,
        tt_move: Optional[Move],
        countermove: Optional[Move],
        pv_move: Optional[Move],
    ) -> int:
        if tt_move == move:
            return 10_000_000

        score = 0
        if pv_move == move:
            score += self.PV_MOVE_BONUS

        from_pos, to_pos = move
        moving = board.get_piece(from_pos[0], from_pos[1])
        target = board.get_piece(to_pos[0], to_pos[1])

        if moving and target:
            score += 1_000_000 + 10 * self.PIECE_VALUES[target.type] - self.PIECE_VALUES[moving.type]
            key = self._capture_history_key(player, move, moving, target)
            score += self.capture_history.get(key, 0)
            see_score = self._see_capture(board, move, moving, target)
            score += see_score * self.SEE_ORDER_SCALE
            if see_score <= self.SEE_BAD_CAPTURE_THRESHOLD:
                score -= self.SEE_BAD_CAPTURE_PENALTY

        if ply < self.MAX_PLY:
            if move == self.killer_moves[ply][0]:
                score += 900_000
            elif move == self.killer_moves[ply][1]:
                score += 800_000

        if countermove == move:
            score += self.COUNTERMOVE_BONUS

        score += self.history.get((player.value, move), 0)
        return score

    def _record_killer(self, ply: int, move: Move):
        if ply >= self.MAX_PLY:
            return
        if self.killer_moves[ply][0] != move:
            self.killer_moves[ply][1] = self.killer_moves[ply][0]
            self.killer_moves[ply][0] = move

    def _record_history(self, player: Player, move: Move, depth: int):
        key = (player.value, move)
        bonus = depth * depth
        self.history[key] = min(self.HISTORY_MAX, self.history.get(key, 0) + bonus)

    def _capture_history_key(
        self,
        player: Player,
        move: Move,
        moving_piece: Piece,
        captured_piece: Piece,
    ) -> Tuple[int, int, int, int, int]:
        _, to_pos = move
        return (
            player.value,
            self.PIECE_INDEX[moving_piece.type],
            self.PIECE_INDEX[captured_piece.type],
            to_pos[0],
            to_pos[1],
        )

    def _record_capture_history(
        self,
        player: Player,
        move: Move,
        moving_piece: Optional[Piece],
        captured_piece: Optional[Piece],
        depth: int,
    ):
        if moving_piece is None or captured_piece is None:
            return
        key = self._capture_history_key(player, move, moving_piece, captured_piece)
        bonus = depth * depth
        self.capture_history[key] = min(
            self.CAPTURE_HISTORY_MAX,
            self.capture_history.get(key, 0) + bonus,
        )

    def _record_countermove(self, player: Player, prev_move: Optional[Move], move: Move):
        if prev_move is None:
            return
        self.counter_moves[(player.value, prev_move)] = move

    def _decay_history(self):
        if not self.history:
            return

        stale_keys: List[Tuple[int, Move]] = []
        for key, value in self.history.items():
            decayed = value - (value >> self.HISTORY_DECAY_SHIFT)
            if decayed <= 1:
                stale_keys.append(key)
            else:
                self.history[key] = decayed

        for key in stale_keys:
            self.history.pop(key, None)

    def _decay_capture_history(self):
        if not self.capture_history:
            return

        stale_keys: List[Tuple[int, int, int, int, int]] = []
        for key, value in self.capture_history.items():
            decayed = value - (value >> self.CAPTURE_HISTORY_DECAY_SHIFT)
            if decayed <= 1:
                stale_keys.append(key)
            else:
                self.capture_history[key] = decayed

        for key in stale_keys:
            self.capture_history.pop(key, None)

    def _clear_pv_lengths(self):
        for i in range(self.MAX_PLY + 1):
            self.pv_length[i] = 0

    def _update_pv_line(self, ply: int, move: Move):
        if ply >= self.MAX_PLY:
            return
        self.pv_table[ply][ply] = move
        child_ply = ply + 1
        child_len = self.pv_length[child_ply] if child_ply <= self.MAX_PLY else child_ply
        if child_len < child_ply:
            child_len = child_ply
        for idx in range(child_ply, min(child_len, self.MAX_PLY)):
            self.pv_table[ply][idx] = self.pv_table[child_ply][idx]
        self.pv_length[ply] = min(child_len, self.MAX_PLY)

    def _refresh_pv_hint(self):
        self.pv_hint = [None for _ in range(self.MAX_PLY)]
        pv_end = min(self.pv_length[0], self.MAX_PLY)
        for idx in range(pv_end):
            move = self.pv_table[0][idx]
            if move is None:
                break
            self.pv_hint[idx] = move

    def _should_try_singular_extension(
        self,
        depth: int,
        is_pv_node: bool,
        in_check: bool,
        entry: Optional[TTEntry],
        tt_move: Optional[Move],
        extensions_used: int,
        ordered_moves: List[Move],
    ) -> bool:
        if self.max_depth <= 6:
            return False
        if depth < self.max_depth:
            return False
        if depth < self.SINGULAR_MIN_DEPTH:
            return False
        if is_pv_node or in_check:
            return False
        if extensions_used >= self.MAX_EXTENSIONS_PER_PATH:
            return False
        if entry is None or tt_move is None:
            return False
        if entry.flag == "UPPER":
            return False
        if entry.depth < depth - 1:
            return False
        if tt_move not in ordered_moves:
            return False
        if len(ordered_moves) <= 1:
            return False
        return True

    def _adaptive_singular_margin(self, depth: int, entry: TTEntry) -> int:
        if entry.flag == "EXACT":
            base = self.SINGULAR_MARGIN_BASE_EXACT
        elif entry.flag == "LOWER":
            base = self.SINGULAR_MARGIN_BASE_LOWER
        else:
            base = (self.SINGULAR_MARGIN_BASE_EXACT + self.SINGULAR_MARGIN_BASE_LOWER) // 2

        margin = base + depth * self.SINGULAR_MARGIN_PER_DEPTH
        depth_cover = entry.depth - depth
        if depth_cover >= 2:
            margin -= 6
        elif depth_cover <= 0:
            margin += 4

        if margin < self.SINGULAR_MARGIN_MIN:
            margin = self.SINGULAR_MARGIN_MIN
        if margin > self.SINGULAR_MARGIN_MAX:
            margin = self.SINGULAR_MARGIN_MAX
        return margin

    def _adaptive_singular_verify_depth(self, depth: int, margin: int, entry: TTEntry) -> int:
        reduction = self.SINGULAR_VERIFY_REDUCTION

        if entry.flag == "EXACT":
            reduction -= 1
        elif entry.flag == "LOWER":
            reduction += 1

        if margin <= 28:
            reduction -= 1
        elif margin >= 88:
            reduction += 1

        depth_cover = entry.depth - depth
        if depth_cover >= 2:
            reduction += 1
        elif depth_cover <= 0:
            reduction -= 1

        if reduction < 1:
            reduction = 1
        if reduction > depth - 1:
            reduction = depth - 1

        return max(1, depth - 1 - reduction)

    def _verify_singular_move(
        self,
        board: ChessBoard,
        depth: int,
        singular_beta: int,
        singular_margin: int,
        entry: TTEntry,
        player: Player,
        opponent: Player,
        ply: int,
        hash_key: int,
        ordered_moves: List[Move],
        tt_move: Move,
        extensions_used: int,
    ) -> bool:
        verify_depth = self._adaptive_singular_verify_depth(depth, singular_margin, entry)
        tested = 0

        for move in ordered_moves:
            if move == tt_move:
                continue
            if tested >= self.SINGULAR_ALT_LIMIT:
                break

            moving_piece = board.get_piece(move[0][0], move[0][1])
            captured_piece = board.get_piece(move[1][0], move[1][1])
            next_hash = self._hash_after_move(hash_key, move, moving_piece, captured_piece)

            self._make_move(board, move, moving_piece, captured_piece)
            score, _ = self._negamax(
                board,
                verify_depth,
                -singular_beta,
                -(singular_beta - 1),
                opponent,
                ply + 1,
                next_hash,
                True,
                False,
                move,
                extensions_used,
                False,
            )
            score = -score
            self._unmake_move(board)

            if self.stop_search:
                return False

            tested += 1
            if score >= singular_beta:
                return False

        return tested > 0

    def _is_immediate_recapture(
        self,
        move: Move,
        prev_move: Optional[Move],
        captured_piece: Optional[Piece],
    ) -> bool:
        if prev_move is None or captured_piece is None:
            return False
        return move[1] == prev_move[1]

    def _should_extend_check(
        self,
        depth: int,
        ply: int,
        gives_check: bool,
        extensions_used: int,
        is_capture: bool,
    ) -> bool:
        if self.max_depth <= 6:
            return False
        if not gives_check:
            return False
        if is_capture:
            return False
        if extensions_used >= self.MAX_EXTENSIONS_PER_PATH:
            return False
        if depth > self.CHECK_EXTENSION_DEPTH_LIMIT:
            return False
        if ply >= self.MAX_PLY - 2:
            return False
        return True

    def _should_extend_recapture(
        self,
        board: ChessBoard,
        depth: int,
        move: Move,
        prev_move: Optional[Move],
        moving_piece: Optional[Piece],
        captured_piece: Optional[Piece],
        extensions_used: int,
    ) -> bool:
        if self.max_depth <= 6:
            return False
        if extensions_used >= self.MAX_EXTENSIONS_PER_PATH:
            return False
        if depth > self.RECAPTURE_EXTENSION_DEPTH_LIMIT:
            return False
        if not self._is_immediate_recapture(move, prev_move, captured_piece):
            return False
        if moving_piece is None or captured_piece is None:
            return False
        if captured_piece.type == PieceType.GENERAL:
            return True

        see_score = self._see_capture(board, move, moving_piece, captured_piece)
        if see_score < self.RECAPTURE_SEE_MIN:
            return False
        return True

    def _least_valuable_attacker(
        self,
        board: ChessBoard,
        player: Player,
        target_pos: Tuple[int, int],
    ) -> Tuple[Optional[Move], Optional[Piece]]:
        tr, tc = target_pos
        best_move: Optional[Move] = None
        best_piece: Optional[Piece] = None
        best_value = self.INF

        for fr, fc in tuple(board.get_piece_positions(player)):
            if fr == tr and fc == tc:
                continue
            piece = board.get_piece(fr, fc)
            if piece is None:
                continue

            piece_value = self.PIECE_VALUES[piece.type]
            if piece_value > best_value:
                continue
            if not MoveValidator.is_valid_move(board, (fr, fc), (tr, tc)):
                continue

            if piece_value < best_value or best_move is None:
                best_value = piece_value
                best_move = ((fr, fc), (tr, tc))
                best_piece = piece

        return best_move, best_piece

    def _see_capture(
        self,
        board: ChessBoard,
        move: Move,
        moving_piece: Optional[Piece] = None,
        captured_piece: Optional[Piece] = None,
    ) -> int:
        from_pos, to_pos = move
        if moving_piece is None:
            moving_piece = board.get_piece(from_pos[0], from_pos[1])
        if captured_piece is None:
            captured_piece = board.get_piece(to_pos[0], to_pos[1])
        if moving_piece is None or captured_piece is None:
            return 0

        gains = [self.PIECE_VALUES[captured_piece.type]]
        undo_count = 0

        self._make_move(board, move, moving_piece, captured_piece)
        undo_count += 1

        side = Player.BLACK if moving_piece.player == Player.RED else Player.RED
        depth = 0

        while True:
            reply_move, reply_piece = self._least_valuable_attacker(board, side, to_pos)
            if reply_move is None or reply_piece is None:
                break

            depth += 1
            gains.append(self.PIECE_VALUES[reply_piece.type] - gains[depth - 1])

            occupant = board.get_piece(to_pos[0], to_pos[1])
            if occupant is None:
                break

            self._make_move(board, reply_move, reply_piece, occupant)
            undo_count += 1
            side = Player.BLACK if side == Player.RED else Player.RED

        while undo_count > 0:
            self._unmake_move(board)
            undo_count -= 1

        for idx in range(len(gains) - 2, -1, -1):
            gains[idx] = max(gains[idx], -gains[idx + 1])

        return gains[0]

    def _lmr_reduction(self, depth: int, move_index: int) -> int:
        reduction = 1
        if depth >= 6 and move_index >= 6:
            reduction += 1
        if depth >= 10 and move_index >= 12:
            reduction += 1
        return min(reduction, max(0, depth - 2))

    def _lmp_limit(self, depth: int) -> int:
        if depth <= 1:
            return 8
        if depth == 2:
            return 12
        return 18

    def _should_apply_lmr(
        self,
        depth: int,
        move_index: int,
        is_capture: bool,
        in_check: bool,
        gives_check: bool,
    ) -> bool:
        if depth < self.LMR_MIN_DEPTH:
            return False
        if move_index < self.LMR_MIN_MOVE_INDEX:
            return False
        if is_capture:
            return False
        if in_check or gives_check:
            return False
        return True

    def _should_apply_lmp(
        self,
        depth: int,
        move_index: int,
        is_capture: bool,
        in_check: bool,
        gives_check: bool,
    ) -> bool:
        if depth > self.LMP_MAX_DEPTH:
            return False
        if is_capture:
            return False
        if in_check or gives_check:
            return False
        return move_index >= self._lmp_limit(depth)

    def _has_major_material(self, board: ChessBoard, player: Player) -> bool:
        for r, c in board.get_piece_positions(player):
            piece = board.get_piece(r, c)
            if not piece:
                continue
            if piece.type in (PieceType.CHARIOT, PieceType.CANNON, PieceType.HORSE):
                return True
        return False

    def _tt_age_delta(self, entry_age: int) -> int:
        return (self.tt_generation - entry_age) & 0xFFFF

    def _should_replace_tt_entry(
        self,
        old_entry: TTEntry,
        new_depth: int,
        new_flag: str,
        new_best_move: Optional[Move],
    ) -> bool:
        age_delta = self._tt_age_delta(old_entry.age)
        if age_delta >= self.TT_AGE_REPLACE:
            return True
        if new_depth > old_entry.depth:
            return True
        if new_depth == old_entry.depth and new_flag == "EXACT" and old_entry.flag != "EXACT":
            return True
        if old_entry.best_move is None and new_best_move is not None:
            return True
        return False

    def _store_tt(
        self,
        hash_key: int,
        depth: int,
        score: int,
        flag: str,
        best_move: Optional[Move],
    ):
        old_entry = self.transposition_table.get(hash_key)
        if old_entry is None:
            self.transposition_table[hash_key] = TTEntry(depth, score, flag, best_move, self.tt_generation)
            return

        if self._should_replace_tt_entry(old_entry, depth, flag, best_move):
            self.transposition_table[hash_key] = TTEntry(depth, score, flag, best_move, self.tt_generation)
        else:
            old_entry.age = self.tt_generation

    def _prune_transposition_table_if_needed(self):
        if len(self.transposition_table) <= self.TT_LIMIT:
            return

        stale_keys = [
            key
            for key, entry in self.transposition_table.items()
            if self._tt_age_delta(entry.age) >= self.TT_AGE_DROP and entry.depth <= 3
        ]
        for key in stale_keys:
            self.transposition_table.pop(key, None)

        if len(self.transposition_table) <= self.TT_LIMIT:
            return

        overflow = len(self.transposition_table) - self.TT_LIMIT
        scored = sorted(
            self.transposition_table.items(),
            key=lambda kv: (
                self._tt_age_delta(kv[1].age),
                0 if kv[1].flag != "EXACT" else -1,
                -kv[1].depth,
            ),
            reverse=True,
        )
        for key, _ in scored[:overflow]:
            self.transposition_table.pop(key, None)

    def _piece_hash_index(self, piece: Piece) -> int:
        return self.PIECE_INDEX[piece.type] + (0 if piece.player == Player.RED else 7)

    def _compute_hash(self, board: ChessBoard, player: Player) -> int:
        h = 0
        for r in range(BOARD_SIZE):
            for c in range(BOARD_WIDTH):
                piece = board.get_piece(r, c)
                if piece:
                    h ^= self.zobrist[r][c][self._piece_hash_index(piece)]
        if player == Player.BLACK:
            h ^= self.side_hash
        return h

    def _hash_after_move(
        self,
        hash_key: int,
        move: Move,
        moving_piece: Optional[Piece],
        captured_piece: Optional[Piece],
    ) -> int:
        if moving_piece is None:
            return hash_key

        (fr, fc), (tr, tc) = move
        moving_idx = self._piece_hash_index(moving_piece)
        next_hash = hash_key
        next_hash ^= self.zobrist[fr][fc][moving_idx]
        if captured_piece is not None:
            next_hash ^= self.zobrist[tr][tc][self._piece_hash_index(captured_piece)]
        next_hash ^= self.zobrist[tr][tc][moving_idx]
        next_hash ^= self.side_hash
        return next_hash

    def _make_move(
        self,
        board: ChessBoard,
        move: Move,
        moving_piece: Optional[Piece] = None,
        captured_piece: Optional[Piece] = None,
    ) -> Optional[Piece]:
        from_pos, to_pos = move
        if moving_piece is None:
            moving_piece = board.get_piece(from_pos[0], from_pos[1])
        if captured_piece is None:
            captured_piece = board.get_piece(to_pos[0], to_pos[1])

        board.set_piece(from_pos[0], from_pos[1], None)
        board.set_piece(to_pos[0], to_pos[1], moving_piece)
        self.move_stack.append((move, moving_piece, captured_piece))
        return captured_piece

    def _unmake_move(self, board: ChessBoard):
        move, moving_piece, captured_piece = self.move_stack.pop()
        from_pos, to_pos = move
        board.set_piece(to_pos[0], to_pos[1], captured_piece)
        board.set_piece(from_pos[0], from_pos[1], moving_piece)


class ChessGame:
    def __init__(self, difficulty: Difficulty = Difficulty.NORMAL, ai_first: bool = False):
        self.difficulty = difficulty
        self.human_player = Player.BLACK if ai_first else Player.RED
        self.board = ChessBoard()
        self.ai = AdvancedAIEngine(difficulty)
        self.current_player = Player.RED
        self.selected_pos: Optional[Tuple[int, int]] = None
        self.valid_moves: List[Tuple[int, int]] = []
        self.ai_thinking = False
        self.game_over = False
        self.winner: Optional[Player] = None
        self.move_history: List[Tuple[Tuple[int, int], Tuple[int, int], Optional[Piece]]] = []
        self.ai_time = 0.0
        self.ai_nodes = 0
        self.ai_depth = 0
        self.in_check_player: Optional[Player] = None

    @property
    def ai_player(self) -> Player:
        return Player.BLACK if self.human_player == Player.RED else Player.RED

    def reset(self, ai_first: Optional[bool] = None):
        if ai_first is not None:
            self.human_player = Player.BLACK if ai_first else Player.RED

        self.board = ChessBoard()
        self.ai = AdvancedAIEngine(self.difficulty)
        self.current_player = Player.RED
        self.selected_pos = None
        self.valid_moves = []
        self.ai_thinking = False
        self.game_over = False
        self.winner = None
        self.move_history = []
        self.ai_time = 0.0
        self.ai_nodes = 0
        self.ai_depth = 0
        self.in_check_player = None

    def make_move(self, from_pos: Tuple[int, int], to_pos: Tuple[int, int]) -> bool:
        piece = self.board.get_piece(from_pos[0], from_pos[1])
        if piece is None or piece.player != self.current_player:
            return False
        if not MoveValidator.is_valid_move(self.board, from_pos, to_pos):
            return False

        captured = self.board.get_piece(to_pos[0], to_pos[1])
        self.board.set_piece(from_pos[0], from_pos[1], None)
        self.board.set_piece(to_pos[0], to_pos[1], piece)
        self.move_history.append((from_pos, to_pos, captured))
        play_move_sound(captured is not None)

        if captured and captured.type == PieceType.GENERAL:
            self.game_over = True
            self.winner = self.current_player
            self.in_check_player = None
            return True

        opponent = self.get_opponent()
        if self._is_checkmate(opponent):
            self.game_over = True
            self.winner = self.current_player
            self.in_check_player = None
            return True

        self.current_player = opponent
        self._refresh_check_state()
        return True

    def get_opponent(self) -> Player:
        return Player.BLACK if self.current_player == Player.RED else Player.RED

    def _is_checkmate(self, player: Player) -> bool:
        return len(self.ai.get_all_legal_moves(self.board, player)) == 0

    def set_difficulty(self, difficulty: Difficulty) -> bool:
        if self.ai_thinking:
            return False
        if difficulty == self.difficulty:
            return False

        self.difficulty = difficulty
        self.ai = AdvancedAIEngine(difficulty)
        self.ai_time = 0.0
        self.ai_nodes = 0
        self.ai_depth = 0
        self.selected_pos = None
        self.valid_moves = []
        return True

    def _refresh_check_state(self):
        if self.game_over:
            self.in_check_player = None
            return
        if MoveValidator.is_in_check(self.board, self.current_player):
            self.in_check_player = self.current_player
        else:
            self.in_check_player = None

    def refresh_valid_moves(self):
        self.valid_moves = []
        if self.selected_pos is None:
            return
        for to_r in range(BOARD_SIZE):
            for to_c in range(BOARD_WIDTH):
                if MoveValidator.is_valid_move(self.board, self.selected_pos, (to_r, to_c)):
                    self.valid_moves.append((to_r, to_c))

    def _undo_one_move(self) -> bool:
        if not self.move_history:
            return False

        from_pos, to_pos, captured = self.move_history[-1]
        moving_piece = self.board.get_piece(to_pos[0], to_pos[1])
        if moving_piece is None:
            return False
        self.move_history.pop()

        self.board.set_piece(to_pos[0], to_pos[1], captured)
        self.board.set_piece(from_pos[0], from_pos[1], moving_piece)
        self.current_player = Player.BLACK if self.current_player == Player.RED else Player.RED
        self.selected_pos = None
        self.valid_moves = []
        self.game_over = False
        self.winner = None
        self.ai_time = 0.0
        self.ai_nodes = 0
        self.ai_depth = 0
        self._refresh_check_state()
        return True

    def undo_last_turn(self) -> bool:
        if self.ai_thinking or not self.move_history:
            return False

        undone = 0
        while self.move_history and undone < 2:
            if not self._undo_one_move():
                break
            undone += 1
            if self.current_player == self.human_player:
                break
        if undone > 0:
            play_move_sound(False)
        return undone > 0

    def get_ai_move(self) -> Optional[Move]:
        start_time = time.perf_counter()
        move = self.ai.find_best_move(self.board, self.current_player)
        self.ai_time = time.perf_counter() - start_time
        self.ai_nodes = self.ai.last_search_nodes
        self.ai_depth = self.ai.last_search_depth
        return move


game = ChessGame(Difficulty.NORMAL)


def get_undo_button_rect() -> pygame.Rect:
    total_width = 118 + 12 + 164 + 12 + 164
    left = (WIDTH - total_width) // 2
    return pygame.Rect(left, CONTROL_BAR_TOP, 118, CONTROL_BUTTON_HEIGHT)


def get_ai_first_button_rect() -> pygame.Rect:
    undo_rect = get_undo_button_rect()
    return pygame.Rect(undo_rect.right + 12, CONTROL_BAR_TOP, 164, CONTROL_BUTTON_HEIGHT)


def get_human_first_button_rect() -> pygame.Rect:
    ai_rect = get_ai_first_button_rect()
    return pygame.Rect(ai_rect.right + 12, CONTROL_BAR_TOP, 164, CONTROL_BUTTON_HEIGHT)


def get_difficulty_button_rects() -> Dict[Difficulty, pygame.Rect]:
    btn_w = 136
    gap = 12
    total_width = btn_w * 3 + gap * 2
    left = (WIDTH - total_width) // 2
    y = CONTROL_BAR_TOP + CONTROL_BUTTON_HEIGHT + CONTROL_BAR_ROW_GAP
    return {
        Difficulty.EASY: pygame.Rect(left, y, btn_w, CONTROL_BUTTON_HEIGHT),
        Difficulty.NORMAL: pygame.Rect(left + btn_w + gap, y, btn_w, CONTROL_BUTTON_HEIGHT),
        Difficulty.MASTER: pygame.Rect(left + (btn_w + gap) * 2, y, btn_w, CONTROL_BUTTON_HEIGHT),
    }


def point_in_rect(pos: Tuple[int, int], rect: pygame.Rect) -> bool:
    return rect.collidepoint(pos[0], pos[1])


def draw_cross_mark(center: Tuple[int, int], color: Tuple[int, int, int] = (55, 35, 20)):
    x, y = center
    arm = int(min(X_STEP, Y_STEP) * 0.16)
    gap = int(min(X_STEP, Y_STEP) * 0.08)
    w = 2

    pygame.draw.line(screen.surface, color, (x - gap - arm, y - gap), (x - gap, y - gap), w)
    pygame.draw.line(screen.surface, color, (x - gap, y - gap), (x - gap, y - gap - arm), w)

    pygame.draw.line(screen.surface, color, (x + gap, y - gap), (x + gap + arm, y - gap), w)
    pygame.draw.line(screen.surface, color, (x + gap, y - gap), (x + gap, y - gap - arm), w)

    pygame.draw.line(screen.surface, color, (x - gap - arm, y + gap), (x - gap, y + gap), w)
    pygame.draw.line(screen.surface, color, (x - gap, y + gap), (x - gap, y + gap + arm), w)

    pygame.draw.line(screen.surface, color, (x + gap, y + gap), (x + gap + arm, y + gap), w)
    pygame.draw.line(screen.surface, color, (x + gap, y + gap), (x + gap, y + gap + arm), w)


def draw_board_marks():
    cannon_marks = [(2, 1), (2, 7), (7, 1), (7, 7)]
    pawn_marks = [
        (3, 0),
        (3, 2),
        (3, 4),
        (3, 6),
        (3, 8),
        (6, 0),
        (6, 2),
        (6, 4),
        (6, 6),
        (6, 8),
    ]

    for r, c in cannon_marks + pawn_marks:
        draw_cross_mark(board_to_screen(r, c))


def get_capture_counts(move_history: List[Tuple[Tuple[int, int], Tuple[int, int], Optional[Piece]]]):
    red_captures: Dict[PieceType, int] = {}
    black_captures: Dict[PieceType, int] = {}

    for move_index, move_entry in enumerate(move_history):
        captured = move_entry[2]
        if captured is None:
            continue

        # 绾㈡柟鍏堟墜锛屽伓鏁版涓虹孩鏂硅蛋瀛?        if move_index % 2 == 0:
            red_captures[captured.type] = red_captures.get(captured.type, 0) + 1
        else:
            black_captures[captured.type] = black_captures.get(captured.type, 0) + 1

    return red_captures, black_captures


def format_capture_text(captures: Dict[PieceType, int], captured_side: Player) -> str:
    if not captures:
        return "\u65e0"

    piece_order = [
        PieceType.CHARIOT,
        PieceType.HORSE,
        PieceType.CANNON,
        PieceType.ELEPHANT,
        PieceType.ADVISOR,
        PieceType.PAWN,
        PieceType.GENERAL,
    ]

    parts: List[str] = []
    for piece_type in piece_order:
        count = captures.get(piece_type, 0)
        if count <= 0:
            continue
        label = piece_label(Piece(piece_type, captured_side))
        parts.append(f"{label}{count}")

    return " ".join(parts) if parts else "\u65e0"


def draw():
    for y in range(HEIGHT):
        ratio = y / HEIGHT
        color = (
            int(236 - 20 * ratio),
            int(210 - 20 * ratio),
            int(165 - 16 * ratio),
        )
        pygame.draw.line(screen.surface, color, (0, y), (WIDTH, y), 1)

    header_rect = pygame.Rect(0, 0, WIDTH, HEADER_HEIGHT)
    screen.draw.filled_rect(header_rect, (244, 222, 184))
    pygame.draw.line(screen.surface, (132, 93, 52), (0, HEADER_HEIGHT), (WIDTH, HEADER_HEIGHT), 2)

    red_captures, black_captures = get_capture_counts(game.move_history)
    red_capture_text = format_capture_text(red_captures, Player.BLACK)
    black_capture_text = format_capture_text(black_captures, Player.RED)

    capture_panel_w = 280
    capture_panel_h = 74
    left_panel = pygame.Rect(14, 18, capture_panel_w, capture_panel_h)
    right_panel = pygame.Rect(WIDTH - 14 - capture_panel_w, 18, capture_panel_w, capture_panel_h)

    screen.draw.filled_rect(left_panel, (238, 212, 170))
    screen.draw.filled_rect(right_panel, (238, 212, 170))
    pygame.draw.rect(screen.surface, (140, 98, 56), left_panel, 1)
    pygame.draw.rect(screen.surface, (140, 98, 56), right_panel, 1)

    draw_ui_text(
        "\u7ea2\u65b9\u5403\u5b50",
        (left_panel.x + 10, left_panel.y + 8),
        color=(130, 30, 30),
        size=20,
        bold=True,
    )
    draw_ui_text(
        red_capture_text,
        (left_panel.x + 10, left_panel.y + 40),
        color=(65, 32, 18),
        size=22 if USE_ASCII_PIECE_LABELS else 24,
    )

    draw_ui_text(
        "\u9ed1\u65b9\u5403\u5b50",
        (right_panel.x + 10, right_panel.y + 8),
        color=(28, 28, 28),
        size=20,
        bold=True,
    )
    draw_ui_text(
        black_capture_text,
        (right_panel.x + 10, right_panel.y + 40),
        color=(65, 32, 18),
        size=22 if USE_ASCII_PIECE_LABELS else 24,
    )

    undo_button = get_undo_button_rect()
    ai_first_button = get_ai_first_button_rect()
    human_first_button = get_human_first_button_rect()
    difficulty_buttons = get_difficulty_button_rects()
    all_button_rects = [undo_button, ai_first_button, human_first_button] + list(difficulty_buttons.values())
    buttons_left = min(rect.left for rect in all_button_rects)
    buttons_right = max(rect.right for rect in all_button_rects)
    buttons_top = min(rect.top for rect in all_button_rects)
    buttons_bottom = max(rect.bottom for rect in all_button_rects)
    control_bar = pygame.Rect(
        buttons_left - 10,
        buttons_top - 4,
        buttons_right - buttons_left + 20,
        buttons_bottom - buttons_top + 8,
    )
    screen.draw.filled_rect(control_bar, (239, 216, 178))
    pygame.draw.rect(screen.surface, (150, 109, 65), control_bar, 1)
    screen.draw.filled_rect(undo_button, (232, 204, 164))
    screen.draw.filled_rect(ai_first_button, (232, 204, 164))
    screen.draw.filled_rect(human_first_button, (232, 204, 164))
    pygame.draw.rect(screen.surface, (120, 86, 48), undo_button, 1)
    pygame.draw.rect(screen.surface, (120, 86, 48), ai_first_button, 1)
    pygame.draw.rect(screen.surface, (120, 86, 48), human_first_button, 1)
    draw_ui_text("\u6094\u68cb(U)", undo_button.center, color=(40, 28, 18), size=18, bold=True, center=True)
    draw_ui_text(
        "AI\u5148\u624b\u65b0\u5c40(A)",
        ai_first_button.center,
        color=(40, 28, 18),
        size=18,
        bold=True,
        center=True,
    )
    draw_ui_text(
        "\u4eba\u5148\u624b\u65b0\u5c40(R)",
        human_first_button.center,
        color=(40, 28, 18),
        size=18,
        bold=True,
        center=True,
    )

    difficulty_labels = {
        Difficulty.EASY: "\u7b80\u5355(E)",
        Difficulty.NORMAL: "\u6b63\u5e38(N)",
        Difficulty.MASTER: "\u5927\u5e08(M)",
    }
    for diff, rect in difficulty_buttons.items():
        active = game.difficulty == diff
        fill_color = (214, 176, 120) if active else (232, 204, 164)
        border_color = (150, 98, 48) if active else (120, 86, 48)
        text_color = (72, 36, 16) if active else (40, 28, 18)
        screen.draw.filled_rect(rect, fill_color)
        pygame.draw.rect(screen.surface, border_color, rect, 1)
        draw_ui_text(
            difficulty_labels[diff],
            rect.center,
            color=text_color,
            size=18,
            bold=active,
            center=True,
        )

    frame_outer = pygame.Rect(
        int(BOARD_LEFT - 26),
        int(BOARD_TOP - 26),
        int(BOARD_PIXEL_WIDTH + 52),
        int(BOARD_PIXEL_HEIGHT + 52),
    )
    frame_inner = pygame.Rect(
        int(BOARD_LEFT - 14),
        int(BOARD_TOP - 14),
        int(BOARD_PIXEL_WIDTH + 28),
        int(BOARD_PIXEL_HEIGHT + 28),
    )
    screen.draw.filled_rect(frame_outer, (112, 78, 42))
    screen.draw.filled_rect(frame_inner, (223, 191, 138))
    pygame.draw.rect(screen.surface, (72, 44, 18), frame_outer, 3)
    pygame.draw.rect(screen.surface, (141, 100, 60), frame_inner, 2)

    if game.in_check_player is not None and not game.game_over:
        checked_text = "\u7ea2\u65b9" if game.in_check_player == Player.RED else "\u9ed1\u65b9"
        pulse = 0.5 + 0.5 * math.sin(time.perf_counter() * 8.5)
        banner_h = 30
        banner_w = min(420, int(BOARD_PIXEL_WIDTH + 8))
        banner_x = WIDTH // 2 - banner_w // 2
        banner_y = max(control_bar.bottom + 4, frame_outer.top - banner_h - 4)
        banner_rect = pygame.Rect(banner_x, banner_y, banner_w, banner_h)

        banner = pygame.Surface((banner_rect.width, banner_rect.height), pygame.SRCALPHA)
        banner.fill((190, 18, 18, int(92 + 120 * pulse)))
        screen.surface.blit(banner, banner_rect.topleft)
        pygame.draw.rect(screen.surface, (252, 228, 228), banner_rect, 2)
        draw_ui_text(
            f"\u5c06\u519b\uff01{checked_text}\u88ab\u5c06",
            banner_rect.center,
            color=(255, 246, 246),
            size=22,
            bold=True,
            center=True,
        )

    river_top_y = BOARD_TOP + 4 * Y_STEP
    river_bottom_y = BOARD_TOP + 5 * Y_STEP
    river_rect = pygame.Rect(
        int(BOARD_LEFT),
        int(river_top_y),
        int(BOARD_PIXEL_WIDTH),
        int(river_bottom_y - river_top_y),
    )
    screen.draw.filled_rect(river_rect, (207, 171, 118))

    for r in range(BOARD_SIZE):
        y = BOARD_TOP + r * Y_STEP
        line_width = 2 if r in (0, BOARD_SIZE - 1) else 1
        pygame.draw.line(screen.surface, (40, 22, 12), (BOARD_LEFT, y), (BOARD_RIGHT, y), line_width)

    for c in range(BOARD_WIDTH):
        x = BOARD_LEFT + c * X_STEP
        line_width = 2 if c in (0, BOARD_WIDTH - 1) else 1
        if c in (0, BOARD_WIDTH - 1):
            pygame.draw.line(screen.surface, (40, 22, 12), (x, BOARD_TOP), (x, BOARD_BOTTOM), line_width)
        else:
            pygame.draw.line(screen.surface, (40, 22, 12), (x, BOARD_TOP), (x, river_top_y), line_width)
            pygame.draw.line(screen.surface, (40, 22, 12), (x, river_bottom_y), (x, BOARD_BOTTOM), line_width)

    palace_lines = [
        ((0, 3), (2, 5)),
        ((0, 5), (2, 3)),
        ((7, 3), (9, 5)),
        ((7, 5), (9, 3)),
    ]
    for start, end in palace_lines:
        x1, y1 = board_to_screen(start[0], start[1])
        x2, y2 = board_to_screen(end[0], end[1])
        pygame.draw.line(screen.surface, (72, 42, 22), (x1, y1), (x2, y2), 2)

    draw_board_marks()

    river_text_y = int((river_top_y + river_bottom_y) / 2 - 18)
    draw_ui_text(
        "\u695a\u6cb3",
        (int(BOARD_LEFT + BOARD_PIXEL_WIDTH * 0.26), river_text_y),
        color=(95, 46, 26),
        size=34,
        bold=True,
    )
    draw_ui_text(
        "\u6c49\u754c",
        (int(BOARD_LEFT + BOARD_PIXEL_WIDTH * 0.62), river_text_y),
        color=(95, 46, 26),
        size=34,
        bold=True,
    )

    if game.move_history:
        last_from, last_to, _ = game.move_history[-1]
        fx, fy = board_to_screen(last_from[0], last_from[1])
        tx, ty = board_to_screen(last_to[0], last_to[1])
        pygame.draw.line(screen.surface, (66, 126, 86), (fx, fy), (tx, ty), 2)

    piece_radius = int(min(X_STEP, Y_STEP) * 0.46)
    for r in range(BOARD_SIZE):
        for c in range(BOARD_WIDTH):
            piece = game.board.get_piece(r, c)
            if not piece:
                continue

            x, y = board_to_screen(r, c)
            text_color = (188, 32, 32) if piece.player == Player.RED else (28, 28, 28)
            shadow = (x + 2, y + 2)
            screen.draw.filled_circle(shadow, piece_radius, (116, 92, 56))
            screen.draw.filled_circle((x, y), piece_radius, (244, 224, 188))
            pygame.draw.circle(screen.surface, (92, 62, 32), (x, y), piece_radius, 2)
            pygame.draw.circle(screen.surface, (140, 98, 56), (x, y), int(piece_radius * 0.78), 1)

            label = piece_label(piece)
            font_size = 26 if USE_ASCII_PIECE_LABELS else 30
            draw_ui_text(
                label,
                (x, y),
                color=text_color,
                size=font_size,
                bold=True,
                center=True,
            )

    if game.selected_pos:
        r, c = game.selected_pos
        x, y = board_to_screen(r, c)
        pygame.draw.circle(screen.surface, (255, 218, 80), (x, y), int(piece_radius * 1.12), 3)

        for to_r, to_c in game.valid_moves:
            tx, ty = board_to_screen(to_r, to_c)
            screen.draw.filled_circle((tx, ty), max(4, piece_radius // 5), (26, 148, 80))

    diff_map = {
        "EASY": "\u521d\u7ea7",
        "NORMAL": "\u4e2d\u7ea7",
        "HARD": "\u9ad8\u7ea7",
        "MASTER": "\u5927\u5e08",
    }
    side_text = "\u7ea2\u65b9" if game.current_player == Player.RED else "\u9ed1\u65b9"
    human_text = "\u7ea2\u65b9" if game.human_player == Player.RED else "\u9ed1\u65b9"
    status_text = (
        f"\u96be\u5ea6: {diff_map.get(game.difficulty.name, game.difficulty.name)}"
        f" | \u5f53\u524d: {side_text}"
        f" | \u73a9\u5bb6: {human_text}"
    )
    if game.ai_thinking:
        status_text += " | AI\u601d\u8003\u4e2d..."
    elif game.ai_depth > 0:
        status_text += f" | \u6df1\u5ea6:{game.ai_depth} \u8282\u70b9:{game.ai_nodes} \u7528\u65f6:{game.ai_time:.2f}s"

    if game.in_check_player is not None and not game.game_over:
        checked_text = "\u7ea2\u65b9" if game.in_check_player == Player.RED else "\u9ed1\u65b9"
        status_text += f" | \u5c06\u519b\uff01{checked_text}\u88ab\u5c06"

    if game.game_over:
        winner_text = "\u7ea2\u65b9" if game.winner == Player.RED else "\u9ed1\u65b9"
        status_text = f"\u6e38\u620f\u7ed3\u675f\uff01{winner_text}\u83b7\u80dc\uff01"

    if not USE_ASCII_PIECE_LABELS and not font_supports_cjk(get_ui_font_name()):
        status_text += " | \u672a\u627e\u5230\u4e2d\u6587\u5b57\u4f53"

    title_text = "\u601d\u8fdc\u8c61\u68cb"
    draw_ui_text(
        title_text,
        (WIDTH // 2, HEADER_HEIGHT // 2 - 2),
        color=(72, 40, 16),
        size=36,
        bold=True,
        center=True,
    )

    status_rect = pygame.Rect(12, HEIGHT - 48, WIDTH - 24, 36)
    screen.draw.filled_rect(status_rect, (236, 214, 176))
    pygame.draw.rect(screen.surface, (120, 86, 48), status_rect, 1)
    draw_ui_text(
        status_text,
        (22, HEIGHT - 43),
        color=(24, 24, 24),
        size=18,
    )

    if game.game_over:
        overlay_w = 360
        overlay_h = 118
        overlay_rect = pygame.Rect((WIDTH - overlay_w) // 2, (HEIGHT - overlay_h) // 2, overlay_w, overlay_h)
        screen.draw.filled_rect(overlay_rect, (245, 223, 188))
        pygame.draw.rect(screen.surface, (120, 86, 48), overlay_rect, 2)
        winner_text = "\u7ea2\u65b9" if game.winner == Player.RED else "\u9ed1\u65b9"
        draw_ui_text(
            f"\u6e38\u620f\u7ed3\u675f\uff1a{winner_text}\u83b7\u80dc",
            (overlay_rect.centerx, overlay_rect.y + 34),
            color=(72, 40, 16),
            size=30,
            bold=True,
            center=True,
        )
        draw_ui_text(
            "\u6309 R \u4eba\u5148\u624b\u65b0\u5c40 | \u6309 A AI\u5148\u624b\u65b0\u5c40",
            (overlay_rect.centerx, overlay_rect.y + 78),
            color=(48, 36, 24),
            size=18,
            center=True,
        )


def on_mouse_down(pos):
    for diff, rect in get_difficulty_button_rects().items():
        if point_in_rect(pos, rect):
            game.set_difficulty(diff)
            return

    if point_in_rect(pos, get_undo_button_rect()):
        game.undo_last_turn()
        return
    if point_in_rect(pos, get_ai_first_button_rect()):
        game.reset(ai_first=True)
        return
    if point_in_rect(pos, get_human_first_button_rect()):
        game.reset(ai_first=False)
        return

    if game.game_over or game.ai_thinking or game.current_player != game.human_player:
        return

    board_pos = screen_to_board(pos)
    if board_pos is None:
        return

    r, c = board_pos
    piece = game.board.get_piece(r, c)

    if game.selected_pos is None:
        if piece and piece.player == game.current_player:
            game.selected_pos = (r, c)
            game.refresh_valid_moves()
    else:
        if game.selected_pos == (r, c):
            game.selected_pos = None
            game.valid_moves = []
        elif (r, c) in game.valid_moves:
            if game.make_move(game.selected_pos, (r, c)):
                game.selected_pos = None
                game.valid_moves = []
        else:
            game.selected_pos = (r, c) if piece and piece.player == game.current_player else None
            game.refresh_valid_moves()


def on_key_down(key):
    if key == keys.U:
        game.undo_last_turn()
    elif key == keys.A:
        game.reset(ai_first=True)
    elif key == keys.R:
        game.reset(ai_first=False)
    elif key == keys.E:
        game.set_difficulty(Difficulty.EASY)
    elif key == keys.N:
        game.set_difficulty(Difficulty.NORMAL)
    elif key == keys.M:
        game.set_difficulty(Difficulty.MASTER)
    elif key == keys.H:
        game.set_difficulty(Difficulty.MASTER)


def update():
    if game.game_over or game.ai_thinking:
        return

    if game.current_player == game.ai_player:
        game.ai_thinking = True
        move = game.get_ai_move()
        if move:
            game.make_move(move[0], move[1])
        else:
            # 防御性兜底：若当前局面 AI 无合法着法，直接判负结束，避免空转。
            legal = game.ai.get_all_legal_moves(game.board, game.current_player)
            if not legal:
                game.game_over = True
                game.winner = game.human_player
                game.in_check_player = None
        game.ai_thinking = False


pgzrun.go()

