
import numpy as np
import random

N_STEPS = 3
A = 1e6
B = 100
C = 10
D = -50
E = -1000

def agent(obs, config):
    def drop_piece(grid, col, mark, config):
        next_grid = grid.copy()
        for row in range(config.rows - 1, -1, -1):
            if next_grid[row][col] == 0:
                next_grid[row][col] = mark
                return next_grid, row
        return next_grid, -1

    def get_valid_moves(grid, config):
        return [c for c in range(config.columns) if grid[0][c] == 0]

    def check_window(window, num_discs, piece, config):
        return window.count(piece) == num_discs and window.count(0) == config.inarow - num_discs

    def count_windows(grid, num_discs, piece, config):
        num_windows = 0
        for row in range(config.rows):
            for col in range(config.columns - (config.inarow - 1)):
                window = list(grid[row, col:col + config.inarow])
                if check_window(window, num_discs, piece, config):
                    num_windows += 1
        for row in range(config.rows - (config.inarow - 1)):
            for col in range(config.columns):
                window = list(grid[row:row + config.inarow, col])
                if check_window(window, num_discs, piece, config):
                    num_windows += 1
        for row in range(config.rows - (config.inarow - 1)):
            for col in range(config.columns - (config.inarow - 1)):
                window = list(grid[range(row, row + config.inarow), range(col, col + config.inarow)])
                if check_window(window, num_discs, piece, config):
                    num_windows += 1
        for row in range(config.inarow - 1, config.rows):
            for col in range(config.columns - (config.inarow - 1)):
                window = list(grid[range(row, row - config.inarow, -1), range(col, col + config.inarow)])
                if check_window(window, num_discs, piece, config):
                    num_windows += 1
        return num_windows

    def get_heuristic(grid, mark, config):
        num_twos = count_windows(grid, 2, mark, config)
        num_threes = count_windows(grid, 3, mark, config)
        num_fours = count_windows(grid, 4, mark, config)
        num_twos_opp = count_windows(grid, 2, 3 - mark, config)
        num_threes_opp = count_windows(grid, 3, 3 - mark, config)
        center_col = config.columns // 2
        center_count = sum(1 for row in range(config.rows) if grid[row, center_col] == mark)
        score = A * num_fours + B * num_threes + C * num_twos + D * num_twos_opp + E * num_threes_opp + center_count * 5
        return score

    def is_terminal_window(window, config):
        return window.count(1) == config.inarow or window.count(2) == config.inarow

    def is_terminal_node(grid, config):
        if list(grid[0, :]).count(0) == 0:
            return True
        for row in range(config.rows):
            for col in range(config.columns - (config.inarow - 1)):
                window = list(grid[row, col:col + config.inarow])
                if is_terminal_window(window, config):
                    return True
        for row in range(config.rows - (config.inarow - 1)):
            for col in range(config.columns):
                window = list(grid[row:row + config.inarow, col])
                if is_terminal_window(window, config):
                    return True
        for row in range(config.rows - (config.inarow - 1)):
            for col in range(config.columns - (config.inarow - 1)):
                window = list(grid[range(row, row + config.inarow), range(col, col + config.inarow)])
                if is_terminal_window(window, config):
                    return True
        for row in range(config.inarow - 1, config.rows):
            for col in range(config.columns - (config.inarow - 1)):
                window = list(grid[range(row, row - config.inarow, -1), range(col, col + config.inarow)])
                if is_terminal_window(window, config):
                    return True
        return False

    def minimax(grid, depth, alpha, beta, maximizingPlayer, mark, config):
        if is_terminal_node(grid, config):
            if count_windows(grid, config.inarow, mark, config) > 0:
                return 1e6
            elif count_windows(grid, config.inarow, 3 - mark, config) > 0:
                return -1e6
            return 0
        if depth == 0:
            return get_heuristic(grid, mark, config)
        valid_moves = get_valid_moves(grid, config)
        if not valid_moves:
            return 0
        if maximizingPlayer:
            value = -np.inf
            for col in valid_moves:
                next_grid, row = drop_piece(grid, col, mark, config)
                if row != -1:
                    value = max(value, minimax(next_grid, depth - 1, alpha, beta, False, mark, config))
                    alpha = max(alpha, value)
                    if alpha >= beta:
                        break
            return value
        else:
            value = np.inf
            for col in valid_moves:
                next_grid, row = drop_piece(grid, col, 3 - mark, config)
                if row != -1:
                    value = min(value, minimax(next_grid, depth - 1, alpha, beta, True, mark, config))
                    beta = min(beta, value)
                    if alpha >= beta:
                        break
            return value

    def score_move(grid, col, mark, config, nsteps):
        next_grid, row = drop_piece(grid, col, mark, config)
        if row == -1:
            return -np.inf
        return minimax(next_grid, nsteps - 1, -np.inf, np.inf, False, mark, config)

    try:
        grid = np.asarray(obs.board).reshape(config.rows, config.columns)
        valid_moves = get_valid_moves(grid, config)
        if not valid_moves:
            return 0
        for col in valid_moves:
            next_grid, row = drop_piece(grid, col, obs.mark, config)
            if row != -1 and count_windows(next_grid, config.inarow, obs.mark, config) > 0:
                return col
        for col in valid_moves:
            next_grid, row = drop_piece(grid, col, 3 - obs.mark, config)
            if row != -1 and count_windows(next_grid, config.inarow, 3 - obs.mark, config) > 0:
                return col
        scores = dict(zip(valid_moves, [score_move(grid, col, obs.mark, config, N_STEPS) for col in valid_moves]))
        max_cols = [col for col in scores if scores[col] == max(scores.values())]
        return random.choice(max_cols)
    except Exception:
        return random.choice(valid_moves) if valid_moves else 0
    