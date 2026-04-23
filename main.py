from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.uix.spinner import Spinner
from kivy.uix.popup import Popup
from kivy.graphics.texture import Texture
from kivy.clock import Clock
from kivy.core.window import Window
from kivy.uix.camera import Camera
from kivy.utils import platform
import cv2
import numpy as np
import os

click_points = []
warped_image = None
current_frame = None

def four_point_transform(image, pts):
    pts = np.array(pts, dtype=np.float32).reshape((4, 2))
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    (tl, tr, br, bl) = rect
    w1, w2 = abs(br[0] - bl[0]), abs(tr[0] - tl[0])
    h1, h2 = abs(tr[1] - br[1]), abs(tl[1] - bl[1])
    maxW, maxH = max(int(w1), int(w2)), max(int(h1), int(h2))
    dst = np.array([[0, 0], [maxW-1, 0], [maxW-1, maxH-1], [0, maxH-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (maxW, maxH))

def enhance_sharpen(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (51, 51), 0)
    return cv2.divide(gray, blur, scale=255)

def black_white(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 6)

def brighten(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = np.clip(v * 1.4, 0, 255).astype(np.uint8)
    return cv2.cvtColor(cv2.merge((h, s, v)), cv2.COLOR_HSV2BGR)

def gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def save_ink(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.convertScaleAbs(gray, alpha=0.7, beta=30)

def cv2_to_texture(cv_img):
    if len(cv_img.shape) == 2:
        buf = cv2.flip(cv_img, 0).tobytes()
        texture = Texture.create(size=(cv_img.shape[1], cv_img.shape[0]), colorfmt='luminance')
        texture.blit_buffer(buf, colorfmt='luminance', bufferfmt='ubyte')
    else:
        buf = cv2.flip(cv_img, 0).tobytes()
        texture = Texture.create(size=(cv_img.shape[1], cv_img.shape[0]), colorfmt='bgr')
        texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
    return texture

class ScanScreen(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(orientation='vertical', **kwargs)
        self.camera = None
        self.current_frame_cv = None
        self.mode = 'preview'
        self.current_result = None

        self.status_label = Label(text="准备扫描文档", size_hint=(1, 0.1))
        self.add_widget(self.status_label)

        self.image_widget = Image()
        self.add_widget(self.image_widget)

        btn_layout = BoxLayout(size_hint=(1, 0.15), spacing=5)

        self.capture_btn = Button(text="拍照")
        self.capture_btn.bind(on_press=self.capture_image)
        btn_layout.add_widget(self.capture_btn)

        self.select_btn = Button(text="选点", disabled=True)
        self.select_btn.bind(on_press=self.start_select_points)
        btn_layout.add_widget(self.select_btn)

        self.filter_spinner = Spinner(text='选择滤镜', values=('增强锐化', '黑白', '增亮', '灰度', '保存墨迹'), disabled=True)
        self.filter_spinner.bind(text=self.apply_filter)
        btn_layout.add_widget(self.filter_spinner)

        self.save_btn = Button(text="保存", disabled=True)
        self.save_btn.bind(on_press=self.save_result)
        btn_layout.add_widget(self.save_btn)

        self.add_widget(btn_layout)
        self.start_camera()

    def start_camera(self):
        try:
            self.camera = Camera(resolution=(640, 480), play=True)
            Clock.schedule_interval(self.update_frame, 1.0/30.0)
        except Exception as e:
            self.status_label.text = f"摄像头启动失败: {e}"

    def update_frame(self, dt):
        if self.camera and self.camera.texture:
            pixels = self.camera.texture.pixels
            if pixels:
                w, h = self.camera.texture.size
                arr = np.frombuffer(pixels, dtype=np.uint8).reshape(h, w, 4)
                self.current_frame_cv = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
                if self.mode == 'preview':
                    texture = cv2_to_texture(self.current_frame_cv)
                    self.image_widget.texture = texture

    def capture_image(self, instance):
        if self.current_frame_cv is not None:
            self.camera.play = False
            self.status_label.text = "已拍照，请点击'选点'标记文档四角"
            self.capture_btn.disabled = True
            self.select_btn.disabled = False
            texture = cv2_to_texture(self.current_frame_cv)
            self.image_widget.texture = texture

    def start_select_points(self, instance):
        global click_points
        click_points = []
        self.mode = 'select_points'
        self.status_label.text = "请依次点击文档的四个角（左上、右上、右下、左下）"
        self.select_btn.disabled = True
        self.filter_spinner.disabled = True
        Window.bind(on_touch_down=self.on_touch_down)

    def on_touch_down(self, window, touch):
        if self.mode != 'select_points':
            return False
        if not self.image_widget.collide_point(*touch.pos):
            return False

        x, y = touch.pos
        img = self.image_widget
        if self.current_frame_cv is not None and img.texture:
            scale_x = self.current_frame_cv.shape[1] / img.width
            scale_y = self.current_frame_cv.shape[0] / img.height
            img_x = max(0, min(int((x - img.x) * scale_x), self.current_frame_cv.shape[1]-1))
            img_y = max(0, min(int((y - img.y) * scale_y), self.current_frame_cv.shape[0]-1))
        else:
            return False

        global click_points
        click_points.append([img_x, img_y])
        self.status_label.text = f"已选择 {len(click_points)}/4 个点"

        frame = self.current_frame_cv.copy()
        for pt in click_points:
            cv2.circle(frame, tuple(pt), 10, (0, 0, 255), -1)
        self.image_widget.texture = cv2_to_texture(frame)

        if len(click_points) == 4:
            self.perform_warp()
        return True

    def perform_warp(self):
        global click_points, warped_image
        if self.current_frame_cv is None or len(click_points) != 4:
            return
        try:
            warped_image = four_point_transform(self.current_frame_cv, click_points)
            self.status_label.text = "矫正完成，请选择滤镜效果"
            self.mode = 'result'
            self.image_widget.texture = cv2_to_texture(warped_image)
            self.filter_spinner.disabled = False
            Window.unbind(on_touch_down=self.on_touch_down)
        except Exception as e:
            self.status_label.text = f"矫正失败: {e}"

    def apply_filter(self, spinner, text):
        global warped_image
        if warped_image is None:
            return
        filter_map = {'增强锐化': enhance_sharpen, '黑白': black_white, '增亮': brighten, '灰度': gray, '保存墨迹': save_ink}
        func = filter_map.get(text)
        if func:
            try:
                result = func(warped_image)
                self.current_result = result
                self.image_widget.texture = cv2_to_texture(result)
                self.save_btn.disabled = False
                self.status_label.text = f"已应用滤镜: {text}"
            except Exception as e:
                self.status_label.text = f"滤镜应用失败: {e}"

    def save_result(self, instance):
        if not hasattr(self, 'current_result'):
            return
        if platform == 'android':
            from android.storage import primary_external_storage_path
            from android.permissions import request_permissions, Permission
            request_permissions([Permission.WRITE_EXTERNAL_STORAGE, Permission.READ_EXTERNAL_STORAGE])
            save_dir = os.path.join(primary_external_storage_path(), 'Documents')
        else:
            save_dir = os.getcwd()
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        filename = os.path.join(save_dir, f"scan_{np.random.randint(1000,9999)}.jpg")
        cv2.imwrite(filename, self.current_result)
        self.status_label.text = f"已保存至: {filename}"
        popup = Popup(title='保存成功', content=Label(text=f'图片已保存至:\n{filename}'), size_hint=(0.8, 0.3))
        popup.open()

class ScannerApp(App):
    def build(self):
        if platform == 'android':
            from android.permissions import request_permissions, Permission
            request_permissions([Permission.CAMERA, Permission.WRITE_EXTERNAL_STORAGE, Permission.READ_EXTERNAL_STORAGE])
        return ScanScreen()

    def on_pause(self):
        if self.root.camera:
            self.root.camera.play = False
        return True

    def on_resume(self):
        if self.root.camera:
            self.root.camera.play = True

if __name__ == '__main__':
    ScannerApp().run()
