import Tkinter
import cv2
import PIL.Image
import PIL.ImageTk
import time
from video_capture import MyVideoCapture
from utils import display_image
from utils import write_text
from utils import undistort
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
        self.frame_1_0 = Tkinter.Frame(window, width=450, height=50, padx=3, pady=3)
        self.frame_1_0.grid(row=1, column=0, sticky="ew")
        self.frame_1_1 = Tkinter.Frame(window, width=450, height=50, padx=3, pady=3)
        self.frame_1_1.grid(row=1, column=1, sticky="ew")
        frame_2_0 = Tkinter.Frame(window, width=450, height=50, padx=3, pady=3)
        frame_2_0.grid(row=2, column=0, sticky="ew")
        self.frame_2_1 = Tkinter.Frame(window, width=450, height=50, padx=3, pady=3)
        self.frame_2_1.grid(row=2, column=1, sticky="ew")
        self.frame_3_0 = Tkinter.Frame(window, width=450, height=50, padx=3, pady=3)
        self.frame_3_0.grid(row=3, column=0, sticky="ew")
        self.frame_3_1 = Tkinter.Frame(window, width=450, height=50, padx=3, pady=3)
        self.frame_3_1.grid(row=3, column=1, sticky="ew")

        write_text(frame_0_0, "Dataset variant:")
        write_text(frame_0_1, "DeepLabv3+ Encoder:")
        write_text(self.frame_1_0, "Color Guide:")
        write_text(frame_2_0, "Live Video:")
        write_text(self.frame_2_1, "Current Image:")
        write_text(self.frame_3_0, "Segmented result:")
        write_text(self.frame_3_1, "Segmentation Overlayed:")

        btn_variant = Tkinter.Button(frame_0_0,
                                     text="full", width=10, command=lambda:
                                     self.set_variant("full"))
        btn_variant.grid(row=1, column=0)

        btn_variant = Tkinter.Button(frame_0_0,
                                     text="size_invariant", width=10, command=lambda:
                                     self.set_variant("size_invariant"))
        btn_variant.grid(row=1, column=1)

        btn_variant = Tkinter.Button(frame_0_0,
                                     text="similar_shapes", width=10, command=lambda:
                                     self.set_variant("similar_shapes"))
        btn_variant.grid(row=2, column=0)

        btn_variant = Tkinter.Button(frame_0_0,
                                     text="binary", width=10, command=lambda:
                                     self.set_variant("binary"))
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

        # After it is called once, the update method will be automatically called every delay milliseconds
        self.delay = 15
        self.update()

        self.window.mainloop()

    def update_info(self):
        for widget in self.frame_1_1.winfo_children():
            widget.destroy()
        write_text(self.frame_1_1, "Dataset variant: {}, \n DeepLabv3+ encoder: {}".format(
                        self.current_variant, self.current_encoder))

    def set_variant(self, name):
        self.current_variant = name
        variant_to_guide = {"full": "./images/full_guide.png",
                            "size_invariant": "./images/size_guide.png",
                            "similar_shapes": "./images/shape_guide.png",
                            "binary": "./images/binary_guide.png"}
        display_image(variant_to_guide[name], self.frame_1_0, 2, 0, (600, 100))
        self.update_info()

    def set_encoder(self, name):
        self.current_encoder = name
        self.update_info()

    def segment_image(self):

        inference_dir = "./inference"
        common_path = "../checkpoints_logs/train_logs/"

        # graphs = {"MobileNetv2": {"full": "mobileNet/full_final_01/mobileNet_full.pb",
        #                           "size_invariant": "mobileNet/size_invariant_01/mobileNet_size.pb",
        #                           "similar_shapes": "mobileNet/similar_shapes_01/mobileNet_shape.pb",
        #                           "binary": "mobileNet/binary_final_02/mobileNet_binary.pb"},
        #           "Xception": {"full": "xception/full_final_01/xception_full.pb",
        #                        "size_invariant": "xception/size_invariant_final_01/xception_size.pb",
        #                        "similar_shapes": "xception/similar_shapes_final_01/xception_shape.pb",
        #                        "binary": "xception/binary_final_01/xception_binary.pb"}}

        graphs = {"MobileNetv2": {"full": "mobileNet/alldata/full_01/mobileNet_full.pb",
                                  "size_invariant": "mobileNet/alldata/size_01/mobileNet_size.pb",
                                  "similar_shapes": "mobileNet/alldata/shape_01/mobileNet_shape.pb",
                                  "binary": "mobileNet/alldata/binary_01/mobileNet_binary.pb"},
                  "Xception": {"full": "xception/alldata/full_01/xception_full.pb",
                               "size_invariant": "xception/alldata/size_01/xception_size.pb",
                               "similar_shapes": "xception/alldata/shape_01/xception_shape.pb",
                               "binary": "xception/alldata/binary_01/xception_binary.pb"}}
        inference_graph.FLAGS.image_path = self.image_path
        inference_graph.FLAGS.graph_path = os.path.join(common_path,
                                                        graphs[self.current_encoder]
                                                        [self.current_variant])
        inference_graph.FLAGS.inference_dir = inference_dir
        inference_time = inference_graph.main(None)
        write_text(self.window, "Inference time: {0:.2f} s".format(inference_time), row=4, column=1)
        self.display_results(self.image_path, inference_dir)

    def display_results(self, image_path, inference_dir):
        seg_img_path = (inference_graph.get_prediction_format() % os.path.join(inference_dir,
                        image_path.split('/')[-1].split('.')[0]) + ".png")
        display_image(seg_img_path, self.frame_3_0, 1, 0,
                      resize_to=self.resize_to_fit)
        image = cv2.imread(image_path)
        mask = cv2.imread(seg_img_path)
        alpha = 0.6
        cv2.addWeighted(mask, alpha, image, 1 - alpha,
                        0, image)
        overlay_save_path = os.path.join(inference_dir, image_path.split('/')[-1].split('_')[0] +
                                         "_overlay.png")
        cv2.imwrite(overlay_save_path, image)
        display_image(overlay_save_path, self.frame_3_1, 1, 0,
                      resize_to=self.resize_to_fit)

    def snapshot(self):
        # Get a frame from the video source
        ret, frame = self.vid.get_frame()

        if ret:
            frame = undistort(frame)
            img_path = "./snapshots/frame-" + time.strftime("%d-%m-%Y-%H-%M-%S") + ".jpg"
            cv2.imwrite(img_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            display_image(img_path, self.frame_2_1, 1, 0,
                          resize_to=self.resize_to_fit)
            btn_snapshot = Tkinter.Button(self.frame_2_1, text="Segment",
                                          width=10, command=self.segment_image)
            btn_snapshot.grid(row=2, column=0)
            self.image_path = img_path

    def update(self):
        # Get a frame from the video source
        ret, frame = self.vid.get_frame()

        if ret:
            frame = undistort(frame)
            frame = cv2.resize(frame, self.resize_to_fit)
            photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame))
            self.canvas_vid.create_image(0, 0, image=photo, anchor=Tkinter.NW)
            self.canvas_vid.image = photo

        self.window.after(self.delay, self.update)


# Create a window and pass it to the Application object
App(Tkinter.Tk(), "Segmentation demo", 0) #"/home/nareshguru77/Desktop/m20_100_2.avi")