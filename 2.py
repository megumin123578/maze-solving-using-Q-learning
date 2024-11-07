import numpy as np
import pygame

# Khởi tạo Pygame
pygame.init()

# Kích thước ô (cell) của mê cung
CELL_SIZE = 50
MAZE_WIDTH = 20
MAZE_HEIGHT = 20

# Màu sắc
WHITE = (255, 255, 255)  # Đường đi
BLACK = (0, 0, 0)        # Tường
BLUE = (0, 0, 255)       # Điểm bắt đầu (S)
GREEN = (0, 255, 0)      # Đích (G)
RED = (255, 0, 0)        # Robot

# Thông số
alpha = 0.1        # Tốc độ học (learning rate)
gamma = 0.9        # Hệ số chiết khấu (discount factor)
epsilon = 1.0      # Epsilon cho epsilon-greedy (ban đầu 100% khám phá)
epsilon_decay = 0.995  # Tỷ lệ giảm epsilon sau mỗi tập
min_epsilon = 0.01     # Giá trị nhỏ nhất của epsilon
episodes = 10000    # Số tập huấn luyện

# Tạo cửa sổ hiển thị
screen = pygame.display.set_mode((MAZE_WIDTH * CELL_SIZE, MAZE_HEIGHT * CELL_SIZE))
pygame.display.set_caption("Maze Visualization")

# Mê cung
maze = np.array([
    ['S', 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0],
    [1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0],
    [1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1],
    [1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1],
    [1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0],
    [0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0],
    [1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0],
    [1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0],
    [0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
    [1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0],
    [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    [1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0],
    [1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
    [1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0],
    [1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 'G']
])

# Khởi tạo Q-Table
q_table = np.zeros((20, 20, 4))

# Chuyển đổi các giá trị tường và đường đi về kiểu int
maze = np.where(maze == 'S', -1, maze)  # -1 đại diện cho điểm bắt đầu
maze = np.where(maze == 'G', 2, maze)   # 2 đại diện cho điểm đích
maze = maze.astype(int)                 # Chuyển tất cả thành int

# Định nghĩa các giá trị cho trạng thái
GOAL_REWARD = 10   # Phần thưởng khi đến đích
STEP_PENALTY = -1   # Phạt cho mỗi bước đi
WALL_PENALTY = -10  # Phạt khi đâm vào tường

def draw_agent(state):
    row, col = state
    pygame.draw.rect(screen, RED, (col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE, CELL_SIZE))

def get_reward(maze, position):
    row, col = position
    if maze[row][col] == 2:  # Nếu đạt đích
        return GOAL_REWARD
    elif maze[row][col] == 1:  # Nếu đâm vào tường
        return WALL_PENALTY
    else:
        return STEP_PENALTY     # Phần thưởng thông thường cho mỗi bước đi

def epsilon_greedy_policy(q_table, state, epsilon):
    if np.random.uniform(0, 1) < epsilon:
        return np.random.randint(0, 4)  # Khám phá (hành động ngẫu nhiên)
    else:
        return np.argmax(q_table[state[0], state[1], :])  # Khai thác (hành động tốt nhất)

def draw_maze(maze):
    for row in range(len(maze)):
        for col in range(len(maze[row])):
            cell_value = maze[row][col]
            if cell_value == 1:
                pygame.draw.rect(screen, BLACK, (col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE, CELL_SIZE))
            elif cell_value == 0:
                pygame.draw.rect(screen, WHITE, (col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE, CELL_SIZE))
            elif cell_value == -1:
                pygame.draw.rect(screen, BLUE, (col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE, CELL_SIZE))
            elif cell_value == 2:
                pygame.draw.rect(screen, GREEN, (col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE, CELL_SIZE))

def take_action(state, action):
    row, col = state
    if action == 0:  # Lên
        return (max(row - 1, 0), col)
    elif action == 1:  # Xuống
        return (min(row + 1, MAZE_HEIGHT-1), col)
    elif action == 2:  # Trái
        return (row, max(col - 1, 0))
    elif action == 3:  # Phải
        return (row, min(col + 1, MAZE_WIDTH - 1))

# Huấn luyện Q-Learning
for episode in range(episodes):
    state = (0, 0)
    done = False
    while not done:
        action = epsilon_greedy_policy(q_table, state, epsilon)
        next_state = take_action(state, action)
        reward = get_reward(maze, next_state)
        old_q_value = q_table[state[0], state[1], action]
        next_max_q = np.max(q_table[next_state[0], next_state[1], :])
        new_q_value = old_q_value + alpha * (reward + gamma * next_max_q - old_q_value)
        q_table[state[0], state[1], action] = new_q_value
        state = next_state
        if reward == GOAL_REWARD or reward == WALL_PENALTY:
            done = True
    epsilon = max(min_epsilon, epsilon * epsilon_decay)

# Tìm đường đi tối ưu
def find_optimal_path(q_table, start, goal):
    path = [start]
    state = start
    while state != goal:
        action = np.argmax(q_table[state[0], state[1], :])
        state = take_action(state, action)
        path.append(state)
        if len(path) > 100:
            break
    return path


optimal_path = find_optimal_path(q_table, (0, 0), (19, 19))

# Hiển thị con đường tối ưu sau khi huấn luyện
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill(WHITE)
    draw_maze(maze)

    # Vẽ con đường tối ưu
    for state in optimal_path:
        draw_agent(state)

    pygame.display.update()

pygame.quit()
