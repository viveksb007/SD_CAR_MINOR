import pygame
import cv2
import numpy as np

#pygame.init()
size = (320, 240)
#screen = pygame.display.set_mode(size)
cap = cv2.VideoCapture(0)



while True:
    _, frame = cap.read()
    cv2.imshow("yo", frame)
    cv2.waitKey(1)
    '''

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
            '''
