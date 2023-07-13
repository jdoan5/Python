import pygame

from pygame import display, event

pygame.init()

display.set_caption('My Game')

#set the screen size
screen = display.set_mode((512, 512))

matched = image.load('other_assets/matched.png')
screen.blit(matched, (0, 0))
display.flip()

# set the game "true" to continue or "false" to stop
running = True

while running:
    current_events = event.get()

    for e in current_events:
        if e.type == pygame.QUIT:
            running = False
print('Goodbye')