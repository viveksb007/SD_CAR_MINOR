import pygame
import serial
import cv2
import numpy as np
import sys
import threading
import urllib2
from socket import error as SocketError


class AndroidCamFeed:
    __bytes = ''
    __stream = None
    __isOpen = False
    __feed = None
    __bytes = ''
    __noStreamCount = 1
    __loadCode = cv2.IMREAD_COLOR if sys.version_info[0] > 2 else 1

    def __init__(self, host):
        self.hoststr = 'http://' + host + '/video'
        try:
            AndroidCamFeed.__stream = urllib2.urlopen(self.hoststr, timeout=3)
            AndroidCamFeed.__isOpen = True
        except (SocketError, urllib2.URLError) as err:
            print "Failed to connect to stream. \nError: " + str(err)
            self.__close()
        t = threading.Thread(target=self.__captureFeed)
        t.start()

    def __captureFeed(self):
        while AndroidCamFeed.__isOpen:
            newbytes = AndroidCamFeed.__stream.read(1024)
            if not newbytes:
                self.__noStream()
                continue
            AndroidCamFeed.__bytes += newbytes
            self.a = AndroidCamFeed.__bytes.find('\xff\xd8')
            self.b = AndroidCamFeed.__bytes.find('\xff\xd9')
            if self.a != -1 and self.b != -1:
                self.jpg = AndroidCamFeed.__bytes[self.a: self.b + 2]
                AndroidCamFeed.__bytes = AndroidCamFeed.__bytes[self.b + 2:]
                AndroidCamFeed.__feed = cv2.imdecode(np.fromstring(self.jpg,
                                                                   dtype=np.uint8),
                                                     AndroidCamFeed.__loadCode)
        return

    def __close(self):
        AndroidCamFeed.__isOpen = False
        AndroidCamFeed.__noStreamCount = 1

    def __noStream(self):
        AndroidCamFeed.__noStreamCount += 1
        if AndroidCamFeed.__noStreamCount > 10:
            try:
                AndroidCamFeed.__stream = urllib2.urlopen(
                    self.hoststr, timeout=3)
            except (SocketError, urllib2.URLError) as err:
                print "Failed to connect to stream: Error: " + str(err)
                self.__close()

    def isOpened(self):
        return AndroidCamFeed.__isOpen

    def read(self):
        if AndroidCamFeed.__feed is not None:
            return True, AndroidCamFeed.__feed
        else:
            return False, None

    def release(self):
        self.__close()


class CollectTrainingData:
    def __init__(self):
        self.host = "192.168.43.1:8080"
        self.connected = False

        ## Pygame stuff
        pygame.init()
        size = (320, 240)
        pygame.display.set_mode(size)

        self.k = np.zeros((4, 4), 'float')
        for i in range(4):
            self.k[i, i] = 1

        '''
        # Connect to serial port for arduino
        self.ser = serial.Serial('/dev/tty.usbmodem1421', 115200, timeout=1)
        '''

        self.send_inst = True
        self.collect_image()

    def collect_image(self):
        saved_frames = 0
        total_frames = 0

        acf = AndroidCamFeed(self.host)
        self.connected = True
        cv2.namedWindow('Android Feed', cv2.WINDOW_AUTOSIZE)

        print "Start collecting images .. \n"

        number_pic = 0

        image_array = np.zeros((1, 38400))
        label_array = np.zeros((1, 4), 'float')

        if not self.connected:
            print "Error in connection"
            return
        else:
            try:
                while self.send_inst and acf.isOpened():
                    ret, frame = acf.read()
                    if ret:
                        cv2.imshow('Android Feed', frame)
                        cv2.waitKey(1)
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        '''
                        image = cv2.GaussianBlur(frame, (5, 5), 0)
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                        image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
                        file_name = 'train' + str(number_pic) + '.jpg'
                        cv2.imwrite(file_name, image)
                        '''
                        roi = frame[120:240, :]
                        # image_temp = np.asarray(image)  # copy image so that original does not get destroyed

                        total_frames += 1

                        temp_array = roi.reshape(1, 38400).astype(np.float32)

                    for event in pygame.event.get():
                        if event.type == pygame.KEYDOWN:
                            key_input = pygame.key.get_pressed()
                            number_pic += 1
                            filename = 'train' + str(number_pic) + '.jpg'
                            cv2.imwrite('training_images/' + filename, frame)

                            # complex orders
                            if key_input[pygame.K_UP] and key_input[pygame.K_RIGHT]:
                                print "Forward Right"
                                image_array = np.vstack((image_array, temp_array))
                                label_array = np.vstack((label_array, self.k[1]))
                                saved_frames += 1
                            # self.ser.write(chr(6))
                            elif key_input[pygame.K_UP] and key_input[pygame.K_LEFT]:
                                print "Forward Left"
                                image_array = np.vstack((image_array, temp_array))
                                label_array = np.vstack((label_array, self.k[0]))
                                saved_frames += 1
                            # self.ser.write(chr(7))
                            elif key_input[pygame.K_DOWN] and key_input[pygame.K_RIGHT]:
                                print "Reverse Right"
                            # self.ser.write(chr(8))
                            elif key_input[pygame.K_DOWN] and key_input[pygame.K_LEFT]:
                                print "Reverse Left"
                            # self.ser.write(chr(9))

                            # simple orders
                            elif key_input[pygame.K_UP]:
                                print "Forward"
                                saved_frames += 1
                                image_array = np.vstack((image_array, temp_array))
                                label_array = np.vstack((label_array, self.k[2]))
                            # self.ser.write(chr(1))
                            elif key_input[pygame.K_DOWN]:
                                print "Reverse"
                                saved_frames += 1
                                image_array = np.vstack((image_array, temp_array))
                                label_array = np.vstack((label_array, self.k[3]))
                            # self.ser.write(chr(2))
                            elif key_input[pygame.K_RIGHT]:
                                print "Right"
                                saved_frames += 1
                                image_array = np.vstack((image_array, temp_array))
                                label_array = np.vstack((label_array, self.k[1]))
                            # self.ser.write(chr(3))
                            elif key_input[pygame.K_LEFT]:
                                print "Left"
                                saved_frames += 1
                                image_array = np.vstack((image_array, temp_array))
                                label_array = np.vstack((label_array, self.k[0]))
                            # self.ser.write(chr(4))
                            elif key_input[pygame.K_x] or key_input[pygame.K_q]:
                                print 'exit'
                                self.send_inst = False
                                # self.ser.write(chr(0))
                                break
                        elif event.type == pygame.KEYUP:
                            pass  # self.ser.write(chr(0))

                '''
                Save training images and labels
                '''
                train = image_array[1:, :]
                train_labels = label_array[1:, :]
                np.savez('training_data/training.npz', train=train, train_labels=train_labels)
                print "Training Completed \n"
                print(train.shape)
                print(train_labels.shape)
                print "Total frames : ", total_frames
                print "Saved frames : ", saved_frames
                print "Dropped frames : ", total_frames - saved_frames

            finally:
                pygame.quit()
                acf.release()
                cv2.destroyAllWindows()


if __name__ == '__main__':
    CollectTrainingData()
