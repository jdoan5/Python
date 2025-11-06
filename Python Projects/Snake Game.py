import pygame
import time
import random

# Initialize pygame
pygame.init()

# Set up display
WIDTH, HEIGHT = 600, 400
CELL_SIZE = 20
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Snake Game")

# Colors
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLACK = (0, 0, 0)

# Clock
clock = pygame.time.Clock()
SPEED = 15

# Snake
snake_pos = [100, 50]
snake_body = [[100, 50], [90, 50], [80, 50]]
direction = 'RIGHT'
change_to = direction

# Food
food_pos = [random.randrange(1, (WIDTH//CELL_SIZE)) * CELL_SIZE,
            random.randrange(1, (HEIGHT//CELL_SIZE)) * CELL_SIZE]
food_spawn = True

# Score
def show_score():
    font = pygame.font.SysFont('Arial', 24)
    score = font.render(f'Score: {len(snake_body) - 3}', True, WHITE)
    screen.blit(score, (10, 10))

# Main loop
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            quit()
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP and direction != 'DOWN':
                change_to = 'UP'
            elif event.key == pygame.K_DOWN and direction != 'UP':
                change_to = 'DOWN'
            elif event.key == pygame.K_LEFT and direction != 'RIGHT':
                change_to = 'LEFT'
            elif event.key == pygame.K_RIGHT and direction != 'LEFT':
                change_to = 'RIGHT'

    direction = change_to

    if direction == 'UP':
        snake_pos[1] -= CELL_SIZE
    if direction == 'DOWN':
        snake_pos[1] += CELL_SIZE
    if direction == 'LEFT':
        snake_pos[0] -= CELL_SIZE
    if direction == 'RIGHT':
        snake_pos[0] += CELL_SIZE

    # Snake body growing mechanism
    snake_body.insert(0, list(snake_pos))
    if snake_pos == food_pos:
        food_spawn = False
    else:
        snake_body.pop()

    if not food_spawn:
        food_pos = [random.randrange(1, (WIDTH//CELL_SIZE)) * CELL_SIZE,
                    random.randrange(1, (HEIGHT//CELL_SIZE)) * CELL_SIZE]
        food_spawn = True

    # Game Over conditions
    if (snake_pos[0] < 0 or snake_pos[0] >= WIDTH or
        snake_pos[1] < 0 or snake_pos[1] >= HEIGHT):
        break

    for block in snake_body[1:]:
        if snake_pos == block:
            break

    screen.fill(BLACK)

    for pos in snake_body:
        pygame.draw.rect(screen, GREEN, pygame.Rect(pos[0], pos[1], CELL_SIZE, CELL_SIZE))

    pygame.draw.rect(screen, RED, pygame.Rect(food_pos[0], food_pos[1], CELL_SIZE, CELL_SIZE))

    show_score()

    pygame.display.update()
    clock.tick(SPEED)

pygame.quit()
quit()
