from queue import Queue
import queue
from threading import Thread

import cv2


class Stream:
    def __init__(self, path, size=128):
        """Initalise Stream.

        Args:
            path ([type]): path to source.
            size (int, optional): queue size. Defaults to 128.
        """
        self.stream = cv2.VideoCapture(path)
        self.stopped = False

        self.queue = Queue(maxsize=size)

    def start(self):
        """Start a thread for reading of frames from stream."""
        thread = Thread(target=self.update, args=())
        thread.daemon = True
        thread.start()

        return self

    def update(self):
        """Streams frames from source to queue."""
        while True:
            if self.stopped:
                return

            if not self.queue.full():
                # read next frame
                success, image = self.stream.read()

                # end of stream
                if not success:
                    self.stop()
                    return

                # add frame to queue
                self.queue.put(image)

    def read(self):
        """Returns the next frame in the queue.

        Returns:
            [type]: returns a single frame.
        """
        return self.queue.get()

    def is_empty(self):
        """Check if queue is empty.

        Returns:
            boolean: True if queue is empty, False otherwise.
        """
        if self.queue.qsize() == 0:
            print("Empty quiting")
        return self.queue.qsize() == 0

    def stop(self):
        """Signal thread to stop."""
        self.stopped = True
