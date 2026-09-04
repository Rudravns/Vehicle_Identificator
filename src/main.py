"""Application entry point."""

from get_path import get_test_image_path
import detector, visualizer
import os


class Main:
    def __init__(self):
        #reset  
        os.system('cls' if os.name == 'nt' else 'clear')
        #print("Cleared")

        self.test = detector.test()
        visualizer.draw_rects(get_test_image_path(), self.test)

if __name__ == "__main__":
    main = Main()

# use this to run:  .venv\Scripts\python.exe src\main.py