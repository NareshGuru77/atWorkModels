import cv2
import PIL.Image
import PIL.ImageTk


class MyLoadImage:

    def __init__(self, img_path, resize_to=None):
        self.img = cv2.imread(img_path)

        if self.img is not None:
            if resize_to is not None:
                self.img = cv2.resize(self.img, resize_to)

            self.img_height = self.img.shape[0]
            self.img_width = self.img.shape[1]

        else:
            raise ValueError('No image found in the path provided')

    def get_image(self):
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        return PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(self.img))
