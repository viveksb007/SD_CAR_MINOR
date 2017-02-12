import pygame
import cv2

pygame.init()
size = (320, 240)
screen = pygame.display.set_mode(size)
img = cv2.imread('mypic.jpg')
img = cv2.resize(img, (320, 240))
pygame.surfarray.blit_array(screen, img)

while True:
    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            key_input = pygame.key.get_pressed()
            if key_input[pygame.K_UP] and key_input[pygame.K_RIGHT]:
                print "Forward Right"
            elif key_input[pygame.K_UP] and key_input[pygame.K_LEFT]:
                print "Forward Left"
            elif key_input[pygame.K_DOWN] and key_input[pygame.K_RIGHT]:
                print "Reverse Right"
            elif key_input[pygame.K_DOWN] and key_input[pygame.K_LEFT]:
                print "Reverse Left"
            elif key_input[pygame.K_UP]:
                print "Forward"
            elif key_input[pygame.K_DOWN]:
                print "Reverse"
            elif key_input[pygame.K_RIGHT]:
                print "Right"
            elif key_input[pygame.K_LEFT]:
                print "Left"
            elif key_input[pygame.K_x] or key_input[pygame.K_q]:
                print 'exit'
                exit()
            break
        elif event.type == pygame.KEYUP:
            pass
