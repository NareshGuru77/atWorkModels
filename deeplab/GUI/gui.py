import Tkinter
import cv2
import PIL.Image, PIL.ImageTk
import time
from video_capture import MyVideoCapture
from load_img import MyLoadImage
from deeplab import inference_graph
import os


# https://solarianprogrammer.com/2018/04/21/python-opencv-show-video-tkinter-window/
# https://stackoverflow.com/questions/34276663/tkinter-gui-layout-using-frames-and-grid

class App:
    def __init__(self, window, window_title, video_source=0):

        self.window = window
        self.window.title(window_title)
        self.video_source = video_source
        self.resize_to_fit = (320, 240)

        self.current_variant = "full"
        self.current_encoder = "MobileNetv2"
        self.image_path = None

        frame_0_0 = Tkinter.Frame(window, width=450, height=50, padx=3, pady=3)
        frame_0_0.grid(row=0, sticky="ew")
        frame_0_1 = Tkinter.Frame(window, width=450, height=50, padx=3, pady=3)
        frame_0_1.grid(row=0, column=1, sticky="ew")
        frame_1_0 = Tkinter.Frame(window, width=450, height=50, padx=3, pady=3)
        frame_1_0.grid(row=1, column=0, sticky="ew")
        frame_1_1 = Tkinter.Frame(window, width=450, height=50, padx=3, pady=3)
        frame_1_1.grid(row=1, column=1, sticky="ew")
        frame_2_0 = Tkinter.Frame(window, width=450, height=50, padx=3, pady=3)
        frame_2_0.grid(row=2, column=0, sticky="ew")
        self.frame_2_1 = Tkinter.Frame(window, width=450, height=50, padx=3, pady=3)
        self.frame_2_1.grid(row=2, column=1, sticky="ew")
        self.frame_3_0 = Tkinter.Frame(window, width=450, height=50, padx=3, pady=3)
        self.frame_3_0.grid(row=3, column=0, sticky="ew")
        self.frame_3_1 = Tkinter.Frame(window, width=450, height=50, padx=3, pady=3)
        self.frame_3_1.grid(row=3, column=1, sticky="ew")

        self.write_text(frame_0_0, "Dataset variant:")
        self.write_text(frame_0_1, "DeepLabv3+ Encoder:")
        self.write_text(frame_1_0, "Color Guide:")
        self.write_text(frame_1_1, "?????:")
        self.write_text(frame_2_0, "Live Video:")
        self.write_text(self.frame_2_1, "Current Image:")
        self.write_text(self.frame_3_0, "Segmented result:")
        self.write_text(self.frame_3_1, "Segmentation Overlayed:")

        btn_variant = Tkinter.Button(frame_0_0,
                                     text="full", width=10, command=
                                     lambda: self.display_image("./images/full_guide.png",
                                                                frame_1_0, 2, 0, (600, 100)))
        btn_variant.grid(row=1, column=0)

        btn_variant = Tkinter.Button(frame_0_0,
                                     text="size_invariant", width=10, command=
                                     lambda: self.display_image("./images/size_guide.png",
                                                                frame_1_0, 2, 0, (600, 100)))
        btn_variant.grid(row=1, column=1)

        btn_variant = Tkinter.Button(frame_0_0,
                                     text="similar_shapes", width=10, command=
                                     lambda: self.display_image("./images/shape_guide.png",
                                                                frame_1_0, 2, 0, (600, 100)))
        btn_variant.grid(row=2, column=0)

        btn_variant = Tkinter.Button(frame_0_0,
                                     text="binary", width=10, command=
                                     lambda: self.display_image("./images/binary_guide.png",
                                                                frame_1_0, 2, 0, (600, 100)))
        btn_variant.grid(row=2, column=1)

        btn_encoder = Tkinter.Button(frame_0_1, text="MobileNetv2", width=10,
                                     command=lambda: self.set_encoder("MobileNetv2"))
        btn_encoder.grid(row=1, column=0)
        btn_encoder = Tkinter.Button(frame_0_1, text="Xception", width=10,
                                     command=lambda: self.set_encoder("Xception"))
        btn_encoder.grid(row=1, column=1)

        # open video source (by default this will try to open the computer webcam)
        self.vid = MyVideoCapture(self.video_source)

        # Create a canvas that can fit the above video source size
        self.canvas_vid = Tkinter.Canvas(frame_2_0, width=self.resize_to_fit[0],
                                         height=self.resize_to_fit[1])
        self.canvas_vid.grid(row=1, column=0)

        # Button that lets the user take a snapshot
        btn_snapshot = Tkinter.Button(frame_2_0, text="Snapshot", width=10, command=self.snapshot)
        btn_snapshot.grid(row=2, column=0)
        # self.btn_snapshot.pack(anchor=tkinter.CENTER, expand=False)

        # After it is called once, the update method will be automatically called every delay milliseconds
        self.delay = 15
        self.update()

        self.window.mainloop()

    def set_encoder(self, name):
        self.current_encoder = name

    def write_text(self, frame, text):
        label_text = Tkinter.Label(frame, text=text,
                                   font=("Helvetica", 16), foreground="green")
        label_text.grid(row=0, column=0)

    def display_image(self, img_path, frame, row, column, resize_to=None):
        img = MyLoadImage(img_path, resize_to=resize_to)
        image = img.get_image()
        label_var = Tkinter.Label(frame, image=image)
        label_var.image = image
        label_var.grid(row=row, column=column)

    def segment_image(self):

        inference_dir = "./inference"
        graph_paths = "../checkpoints_logs/train_logs/mobileNet/full_final_01/mobileNet_full.pb"
        inference_graph.FLAGS.image_path = self.image_path
        inference_graph.FLAGS.graph_path = graph_paths
        inference_graph.FLAGS.inference_dir = inference_dir
        inference_graph.main(None)
        self.display_results(self.image_path, inference_dir)

    def display_results(self, image_path, inference_dir):
        seg_img_path = (inference_graph._PREDICTION_FORMAT % os.path.join(inference_dir,
                        image_path.split('/')[-1].split('.')[0]) + '.png')
        self.display_image(seg_img_path, self.frame_3_0, 1, 0,
                           resize_to=self.resize_to_fit)

    def snapshot(self):
        # Get a frame from the video source
        ret, frame = self.vid.get_frame()

        if ret:
            img_path = "./snapshots/frame-" + time.strftime("%d-%m-%Y-%H-%M-%S") + ".jpg"
            cv2.imwrite(img_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            self.display_image(img_path, self.frame_2_1, 1, 0,
                               resize_to=self.resize_to_fit)
            btn_snapshot = Tkinter.Button(self.frame_2_1, text="Segment",
                                          width=10, command=self.segment_image)
            btn_snapshot.grid(row=2, column=0)
            self.image_path = img_path

    def update(self):
        # Get a frame from the video source
        ret, frame = self.vid.get_frame()

        if ret:
            frame = cv2.resize(frame, self.resize_to_fit)
            self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame))
            self.canvas_vid.create_image(0, 0, image=self.photo, anchor=Tkinter.NW)

        self.window.after(self.delay, self.update)


# Create a window and pass it to the Application object
App(Tkinter.Tk(), "Segmentation demo", "/home/nareshguru77/Desktop/m20_100_2.avi")