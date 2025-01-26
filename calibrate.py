import pyautogui
import time
import threading
from pynput.keyboard import Key, Controller
import os
import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
from PIL import Image

class RikkuImplementation:
    def __init__(self):
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.image_path = os.path.join(self.script_dir, "images")
        
        self.keys = ["c", "d", "down", "left", "right", "s", "up", "v", "x", "z"]
        self.confidence_level = 0.9
        self.key_hold_duration = 0.05
        
        self.key_mapping = {
            "c": 'c',
            "d": 'd',
            "down": Key.down,
            "left": Key.left,
            "right": Key.right,
            "s": 's',
            "up": Key.up,
            "v": 'v',
            "x": 'x',
            "z": 'z',
        }
        
        self.keyboard_controller = Controller()

    def check_for_key(self, key, detected_keys):
        try:
            location = pyautogui.locateOnScreen(f"{self.image_path}/{key}.png", confidence=self.confidence_level)
            if location:
                detected_keys.append((key, location))
        except Exception as e:
            pass

    def press_and_hold_keys(self, keys_to_press):
        if not keys_to_press:
            return
            
        for key in keys_to_press:
            self.keyboard_controller.press(key)
            print(f"Pressed {key}")

        time.sleep(self.key_hold_duration)

        for key in keys_to_press:
            self.keyboard_controller.release(key)

    def monitor_screen_for_keys(self):
        detected_keys = []

        threads = []
        for key in self.keys:
            thread = threading.Thread(target=self.check_for_key, args=(key, detected_keys))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        detected_keys = sorted(detected_keys, 
                             key=lambda x: x[1][2] * x[1][3], 
                             reverse=True)

        return [key for key, _ in detected_keys]

    def run(self):
        print("Starting screen monitoring... Press Ctrl+C to exit.")
        try:
            last_keys = []
            
            while True:
                detected_keys = self.monitor_screen_for_keys()

                if len(detected_keys) == 1:
                    time.sleep(0.05)
                    detected_keys_verification = self.monitor_screen_for_keys()

                    if len(detected_keys_verification) > 1:
                        self.press_and_hold_keys([self.key_mapping[key] for key in detected_keys_verification])
                        last_keys = detected_keys_verification
                elif len(detected_keys) > 1:
                    self.press_and_hold_keys([self.key_mapping[key] for key in detected_keys])
                    last_keys = detected_keys

                time.sleep(0.05)

        except KeyboardInterrupt:
            print("Exiting script.")

@dataclass
class IconMatch:
    key: str
    timestamp: float
    position: tuple

class SequenceTracker:
    def __init__(self):
        self.keyboard = Controller()
        self.running = True
        self.sequence: List[str] = []
        self.templates: Dict[str, np.ndarray] = {}
        self.key_mapping = {
            "c": 'c', "d": 'd', "down": Key.down,
            "left": Key.left, "right": Key.right,
            "s": 's', "up": Key.up, "v": 'v',
            "x": 'x', "z": 'z',
        }
        
        self.load_templates()
        self.position_tolerance = 50
        self.last_positions = {}

    def load_templates(self):
        """Load and preprocess template images"""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        image_path = os.path.join(script_dir, "images")
        keys = ["c", "d", "down", "left", "right", "s", "up", "v", "x", "z"]
        THRESHOLD_VALUE = 200
        
        start_template = cv2.imread(f"{image_path}/yuna_start.png", cv2.IMREAD_GRAYSCALE)
        if start_template is not None:
            _, start_template = cv2.threshold(start_template, THRESHOLD_VALUE, 255, cv2.THRESH_BINARY)
            self.templates["yuna_start"] = start_template
            print("Successfully loaded yuna_start template")
        else:
            print("Failed to load yuna_start template")

        trigger_template = cv2.imread(f"{image_path}/yuna_trigger.png", cv2.IMREAD_GRAYSCALE)
        if trigger_template is not None:
            _, trigger_template = cv2.threshold(trigger_template, THRESHOLD_VALUE, 255, cv2.THRESH_BINARY)
            self.templates["yuna_trigger"] = trigger_template
            print("Successfully loaded yuna_trigger template")
        else:
            print("Failed to load yuna_trigger template")

        for key in keys:
            template = cv2.imread(f"{image_path}/{key}.png", cv2.IMREAD_GRAYSCALE)
            if template is not None:
                _, template = cv2.threshold(template, THRESHOLD_VALUE, 255, cv2.THRESH_BINARY)
                self.templates[key] = template
                print(f"Successfully loaded template for key: {key}")
            else:
                print(f"Failed to load template for key: {key}")

    def check_for_icon(self, key: str) -> tuple[bool, tuple[int, int]]:
        """Check if an icon is present on screen, returns (found, position)"""
        try:
            screenshot = np.array(pyautogui.screenshot())
            gray_screenshot = cv2.cvtColor(screenshot, cv2.COLOR_RGB2GRAY)
            _, binary = cv2.threshold(gray_screenshot, 200, 255, cv2.THRESH_BINARY)
            
            template = self.templates.get(key)
            if template is None:
                return False, (0, 0)
            
            result = cv2.matchTemplate(binary, template, cv2.TM_CCOEFF_NORMED)
            locations = np.where(result >= 0.75)
            
            for pt in zip(*locations[::-1]):
                w, h = template.shape[::-1]
                roi = binary[pt[1]:pt[1]+h, pt[0]:pt[0]+w]
                contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if any(cv2.contourArea(contour) > 100 for contour in contours):
                    return True, (pt[0], pt[1])
            
            return False, (0, 0)
            
        except Exception as e:
            print(f"Error checking for icon {key}: {e}")
            return False, (0, 0)

    def is_new_position(self, key: str, position: tuple) -> bool:
        """Check if the position is significantly different from last known position"""
        if key not in self.last_positions:
            return True
            
        last_pos = self.last_positions[key]
        distance = ((position[0] - last_pos[0]) ** 2 + (position[1] - last_pos[1]) ** 2) ** 0.5
        return distance > self.position_tolerance

    def press_key(self, key: str):
        """Press a key with the specified duration"""
        mapped_key = self.key_mapping.get(key, key)
        try:
            print(f"Pressing {key}")
            self.keyboard.press(mapped_key)
            time.sleep(0.1)
            self.keyboard.release(mapped_key)
            time.sleep(0.5)
        except Exception as e:
            print(f"Error pressing key {key}: {e}")
            self.keyboard.release(mapped_key)

    def replay_sequence(self):
        """Replay the stored sequence"""
        print(f"Replaying sequence: {self.sequence}")
        for key in self.sequence:
            self.press_key(key)
        self.sequence.clear()
        print("Sequence cleared from memory")
        time.sleep(0.1)

    def monitor_screen(self):
        """Monitor the screen for icons and trigger sequence"""
        try:
            if self.check_for_icon("yuna_start")[0]:
                print("Start trigger detected - beginning sequence recording")
                self.sequence.clear()
                self.last_positions.clear()
                time.sleep(0.3)
                
                recording = True
                last_icon_time = time.time()
                while recording:
                    current_time = time.time()
                    
                    if current_time - last_icon_time > 2.0:
                        print("Sequence timeout - no icons detected")
                        self.sequence.clear()
                        recording = False
                        return

                    if self.check_for_icon("yuna_trigger")[0]:
                        print("End trigger detected")
                        if self.sequence:
                            time.sleep(0.3)
                            self.replay_sequence()
                        self.sequence.clear()
                        self.last_positions.clear()
                        time.sleep(0.15)
                        recording = False
                        return
                    
                    for key in ["c", "d", "down", "left", "right", "s", "up", "v", "x", "z"]:
                        found, position = self.check_for_icon(key)
                        if found and self.is_new_position(key, position):
                            print(f"Adding {key} to sequence at position {position}")
                            self.sequence.append(key)
                            self.last_positions[key] = position
                            last_icon_time = time.time()

        except Exception as e:
            print(f"Error in monitor_screen: {e}")
            self.sequence.clear()

def run_yuna():
    print("Starting script.")
    tracker = SequenceTracker()
    
    try:
        while tracker.running:
            tracker.monitor_screen()
            time.sleep(0.01)
            
    except KeyboardInterrupt:
        print("Exiting script.")
        tracker.running = False

@dataclass
class FallingNote:
    key: str
    lane_x: int
    last_y: int
    first_seen: float
    active: bool = True

@dataclass
class NoteMatch:
    key: str
    count: int = 0
    last_y: int = 0
    first_seen: float = 0.0

class PaineTracker:
    def __init__(self):
        self.keyboard_controller = Controller()
        self.lock = threading.Lock()
        self.lane_matches = defaultdict(lambda: defaultdict(lambda: NoteMatch(key="")))
        self.active_notes: List[FallingNote] = []
        self.key_mapping = {
            "c": 'c',
            "d": 'd',
            "down": Key.down,
            "left": Key.left,
            "right": Key.right,
            "s": 's',
            "up": Key.up,
            "v": 'v',
            "x": 'x',
            "z": 'z',
        }
        self.LANE_TOLERANCE = 100
        self.sparkle_thread = None
        self.running = True
        self.sparkle_region = (0, 400, 1920, 900)
        self.templates = {}
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        image_path = os.path.join(script_dir, "images")
        keys = ["c", "d", "down", "left", "right", "s", "up", "v", "x", "z"]
        
        for key in keys:
            template = cv2.imread(f"{image_path}/{key}.png", cv2.IMREAD_GRAYSCALE)
            if template is not None:
                _, template = cv2.threshold(template, 200, 255, cv2.THRESH_BINARY)
                self.templates[key] = template
                print(f"Successfully loaded template for key: {key}")
            else:
                print(f"Failed to load template for key: {key}")
                
            small_template = cv2.imread(f"{image_path}/{key}_small.png", cv2.IMREAD_GRAYSCALE)
            if small_template is not None:
                _, small_template = cv2.threshold(small_template, 200, 255, cv2.THRESH_BINARY)
                self.templates[f"{key}_small"] = small_template
                print(f"Successfully loaded small template for key: {key}")
            else:
                print(f"Failed to load small template for key: {key}")
        self.match_count = 0
        self.last_reset_time = time.time()

    def get_lane_key(self, x_pos: int) -> int:
        for lane_x in self.lane_matches.keys():
            if abs(lane_x - x_pos) <= self.LANE_TOLERANCE:
                return lane_x
        return x_pos

    def update_note_matches(self, key: str, location) -> None:
        x, y, w, h = location
        center_x = x + w//2
        current_time = time.time()
        
        with self.lock:
            lane_x = self.get_lane_key(center_x)
            
            note_match = self.lane_matches[lane_x][key]
            if not note_match.key:
                note_match.key = key
                note_match.first_seen = current_time
            note_match.count += 1
            note_match.last_y = y

            self.update_active_notes()

    def update_active_notes(self):
        self.active_notes.clear()
        
        for lane_x, matches in self.lane_matches.items():
            if matches:
                best_match = max(matches.values(), key=lambda m: m.count)
                if best_match.count > 0:
                    note = FallingNote(
                        key=best_match.key,
                        lane_x=lane_x,
                        last_y=best_match.last_y,
                        first_seen=best_match.first_seen
                    )
                    self.active_notes.append(note)

    def check_for_key(self, key: str) -> None:
        try:
            screenshot = np.array(pyautogui.screenshot())
            if screenshot is None or screenshot.size == 0:
                print("Warning: Invalid screenshot")
                return

            gray_screenshot = cv2.cvtColor(screenshot, cv2.COLOR_RGB2GRAY)
            _, binary = cv2.threshold(gray_screenshot, 200, 255, cv2.THRESH_BINARY)
            
            template = self.templates.get(key)
            small_template = self.templates.get(f"{key}_small")
            
            if template is None and small_template is None:
                print(f"Warning: No templates available for key {key}")
                return
            
            if template is not None and template.size > 0:
                result = cv2.matchTemplate(binary, template, cv2.TM_CCOEFF_NORMED)
                locations = np.where(result >= 0.80)
                
                for pt in zip(*locations[::-1]):
                    w, h = template.shape[::-1]
                    x, y = pt[0], pt[1]
                    roi = binary[y:y+h, x:x+w]
                    contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    valid_match = any(cv2.contourArea(contour) > 100 for contour in contours)
                    if valid_match:
                        self.update_note_matches(key, (x, y, w, h))
            
            if small_template is not None and small_template.size > 0:
                result = cv2.matchTemplate(binary, small_template, cv2.TM_CCOEFF_NORMED)
                locations = np.where(result >= 0.80)
                
                for pt in zip(*locations[::-1]):
                    w, h = small_template.shape[::-1]
                    x, y = pt[0], pt[1]
                    roi = binary[y:y+h, x:x+w]
                    contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    valid_match = any(cv2.contourArea(contour) > 100 for contour in contours)
                    if valid_match:
                        self.update_note_matches(key, (x, y, w, h))
                    
        except Exception as e:
            print(f"Error in check_for_key for {key}: {e}")
            pass

    def reset_state(self):
        with self.lock:
            self.lane_matches.clear()
            self.active_notes.clear()
            self.match_count += 1
            
            if self.match_count % 5 == 0:
                import gc
                gc.collect()

    def find_sparkle(self, screenshot: Image.Image) -> Optional[Tuple[int, int]]:
        region_screenshot = screenshot.crop(self.sparkle_region)
        img_array = np.array(region_screenshot)
        
        SPARKLE_COLORS = [
            (255, 232, 149),
            (255, 153, 230),
            (255, 255, 176),
            (255, 168, 193),
            (255, 168, 255),
            (219, 150, 101),
            (255, 190, 255),
            (255, 192, 181)
        ]
        
        for color in SPARKLE_COLORS:
            r_mask = abs(img_array[:, :, 0] - color[0]) <= 25
            g_mask = abs(img_array[:, :, 1] - color[1]) <= 25
            b_mask = abs(img_array[:, :, 2] - color[2]) <= 25
            
            color_mask = r_mask & g_mask & b_mask
            
            if np.any(color_mask):
                y_coords, x_coords = np.where(color_mask)
                if len(y_coords) > 0:
                    sparkle_pos = (
                        x_coords[0] + self.sparkle_region[0],
                        y_coords[0] + self.sparkle_region[1]
                    )
                    print(f"Sparkle found at {sparkle_pos} matching color {color}")
                    return sparkle_pos
        return None

    def find_nearest_note(self, sparkle_x: int, sparkle_y: int) -> Optional[str]:
        if len(self.active_notes) != 3:
            return None

        min_distance = float('inf')
        nearest_note = None

        for note in self.active_notes:
            distance = ((note.lane_x - sparkle_x) ** 2 + (note.last_y - sparkle_y) ** 2) ** 0.5
            if distance < min_distance:
                min_distance = distance
                nearest_note = note

        if nearest_note:
            for note in self.active_notes:
                dist = ((note.lane_x - sparkle_x) ** 2 + (note.last_y - sparkle_y) ** 2) ** 0.5
                print(f"  {note.key}: {dist:.2f} pixels (at x={note.lane_x}, y={note.last_y})")

        return nearest_note.key if nearest_note else None

    def press_key(self, key: str) -> None:
        mapped_key = self.key_mapping.get(key, key)
        try:
            print(f"Pressing {key} for {0.1}s")
            time.sleep(0.05)
            self.keyboard_controller.release(mapped_key)
            time.sleep(0.05)
            self.keyboard_controller.press(mapped_key)
            time.sleep(0.1)
            self.keyboard_controller.release(mapped_key)
            time.sleep(0.1)
        except Exception as e:
            print(f"Error pressing key {key}: {e}")
            self.keyboard_controller.release(mapped_key)

    def sparkle_monitor(self):
        last_detection_time = time.time()
        while self.running:
            try:
                current_time = time.time()
                if len(self.active_notes) == 3:
                    screenshot = pyautogui.screenshot()
                    sparkle_pos = self.find_sparkle(screenshot)
                    
                    if sparkle_pos:
                        sparkle_x, sparkle_y = sparkle_pos
                        nearest_key = self.find_nearest_note(sparkle_x, sparkle_y)
                        if nearest_key:
                            self.press_key(nearest_key)
                            last_detection_time = current_time
                            time.sleep(0.2)
                            self.reset_state()
                            time.sleep(0.5)
                
            except Exception as e:
                print(f"Error in sparkle_monitor: {e}")
            
            time.sleep(0.01)

    def monitor_screen(self) -> None:
        keys = ["c", "d", "down", "left", "right", "s", "up", "v", "x", "z"]
        for key in keys:
            self.check_for_key(key)
        time.sleep(0.01)

def run_paine():
    tracker = PaineTracker()
    
    try:
        tracker.sparkle_thread = threading.Thread(target=tracker.sparkle_monitor)
        tracker.sparkle_thread.daemon = True
        tracker.sparkle_thread.start()
        
        while True:
            tracker.monitor_screen()
            
    except KeyboardInterrupt:
        print("Exiting script.")
        tracker.running = False
        if tracker.sparkle_thread:
            tracker.sparkle_thread.join(timeout=1.0)

@dataclass
class ReverseSequenceTracker:
    def __init__(self):
        self.keyboard = Controller()
        self.running = True
        self.sequence: List[str] = []
        self.templates: Dict[str, np.ndarray] = {}
        self.key_mapping = {
            "c": 'c', "d": 'd', "down": Key.down,
            "left": Key.left, "right": Key.right,
            "s": 's', "up": Key.up, "v": 'v',
            "x": 'x', "z": 'z',
            "pgup": Key.page_up,
            "pgdn": Key.page_down,
        }
        
        self.load_templates()
        self.position_tolerance = 50
        self.last_positions = {}

    def load_templates(self):
        """Load and preprocess template images"""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        image_path = os.path.join(script_dir, "images")
        keys = ["c", "d", "down", "left", "right", "s", "up", "v", "x", "z", "pgup", "pgdn"]
        THRESHOLD_VALUE = 200
        
        start_template = cv2.imread(f"{image_path}/yuna_start_rev.png", cv2.IMREAD_GRAYSCALE)
        if start_template is not None:
            _, start_template = cv2.threshold(start_template, THRESHOLD_VALUE, 255, cv2.THRESH_BINARY)
            self.templates["yuna_start_rev"] = start_template
            print("Successfully loaded yuna_start_rev template")
        else:
            print("Failed to load yuna_start_rev template")

        trigger_template = cv2.imread(f"{image_path}/yuna_trigger_rev.png", cv2.IMREAD_GRAYSCALE)
        if trigger_template is not None:
            _, trigger_template = cv2.threshold(trigger_template, THRESHOLD_VALUE, 255, cv2.THRESH_BINARY)
            self.templates["yuna_trigger_rev"] = trigger_template
            print("Successfully loaded yuna_trigger_rev template")
        else:
            print("Failed to load yuna_trigger_rev template")

        for key in keys:
            template = cv2.imread(f"{image_path}/{key}.png", cv2.IMREAD_GRAYSCALE)
            if template is not None:
                _, template = cv2.threshold(template, THRESHOLD_VALUE, 255, cv2.THRESH_BINARY)
                self.templates[key] = template
                print(f"Successfully loaded template for key: {key}")
            else:
                print(f"Failed to load template for key: {key}")

    def check_for_icon(self, key: str) -> tuple[bool, tuple[int, int]]:
        """Check if an icon is present on screen, returns (found, position)"""
        try:
            screenshot = np.array(pyautogui.screenshot())
            gray_screenshot = cv2.cvtColor(screenshot, cv2.COLOR_RGB2GRAY)
            _, binary = cv2.threshold(gray_screenshot, 200, 255, cv2.THRESH_BINARY)
            
            template = self.templates.get(key)
            if template is None:
                return False, (0, 0)
            
            confidence_threshold = 0.85 if key in ["pgup", "pgdn"] else 0.75
            
            result = cv2.matchTemplate(binary, template, cv2.TM_CCOEFF_NORMED)
            locations = np.where(result >= confidence_threshold)
            
            for pt in zip(*locations[::-1]):
                w, h = template.shape[::-1]
                roi = binary[pt[1]:pt[1]+h, pt[0]:pt[0]+w]
                contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                min_contour_area = 150 if key in ["pgup", "pgdn"] else 100
                if any(cv2.contourArea(contour) > min_contour_area for contour in contours):
                    if key in ["pgup", "pgdn"]:
                        match_score = result[pt[1], pt[0]]
                        print(f"Match score for {key} at {pt}: {match_score}")
                        
                        if pt[1] < 100 or pt[1] > screenshot.shape[0] - 100:
                            continue
                            
                    return True, (pt[0], pt[1])
            
            return False, (0, 0)
            
        except Exception as e:
            print(f"Error checking for icon {key}: {e}")
            return False, (0, 0)

    def is_new_position(self, key: str, position: tuple) -> bool:
        """Check if the position is significantly different from last known position"""
        if key not in self.last_positions:
            return True
            
        last_pos = self.last_positions[key]
        distance = ((position[0] - last_pos[0]) ** 2 + (position[1] - last_pos[1]) ** 2) ** 0.5
        return distance > self.position_tolerance

    def press_key(self, key: str):
        """Press a key with the specified duration"""
        mapped_key = self.key_mapping.get(key, key)
        try:
            print(f"Pressed {key}")
            self.keyboard.press(mapped_key)
            time.sleep(0.1)
            self.keyboard.release(mapped_key)
            time.sleep(0.25)
        except Exception as e:
            print(f"Error pressing key {key}: {e}")
            self.keyboard.release(mapped_key)

    def replay_sequence(self):
        """Replay the stored sequence in reverse order"""
        reversed_sequence = self.sequence[::-1]
        print(f"Replaying reversed sequence: {reversed_sequence}")
        for key in reversed_sequence:
            self.press_key(key)
        self.sequence.clear()
        print("Sequence cleared from memory")
        time.sleep(0.1)

    def monitor_screen(self):
        """Monitor the screen for icons and trigger sequence"""
        try:
            if self.check_for_icon("yuna_start_rev")[0]:
                print("Start trigger detected - beginning sequence recording")
                self.sequence.clear()
                self.last_positions.clear()
                time.sleep(0.3)
                
                recording = True
                last_icon_time = time.time()
                while recording:
                    current_time = time.time()
                    
                    if current_time - last_icon_time > 2.0:
                        print("Sequence timeout - no icons detected")
                        self.sequence.clear()
                        recording = False
                        return

                    if self.check_for_icon("yuna_trigger_rev")[0]:
                        print("End trigger detected")
                        if self.sequence:
                            time.sleep(0.95)
                            self.replay_sequence()
                        self.sequence.clear()
                        self.last_positions.clear()
                        time.sleep(0.15)
                        recording = False
                        return
                    
                    for key in ["c", "d", "down", "left", "right", "s", "up", "v", "x", "z", "pgup", "pgdn"]:
                        found, position = self.check_for_icon(key)
                        if found and self.is_new_position(key, position):
                            print(f"Adding {key} to sequence at position {position}")
                            self.sequence.append(key)
                            self.last_positions[key] = position
                            last_icon_time = time.time()

        except Exception as e:
            print(f"Error in monitor_screen: {e}")
            self.sequence.clear()

def run_yuna_reverse():
    print("Starting reverse Yuna implementation.")
    tracker = ReverseSequenceTracker()
    
    try:
        while tracker.running:
            tracker.monitor_screen()
            time.sleep(0.01)
            
    except KeyboardInterrupt:
        print("Exiting script.")
        tracker.running = False

if __name__ == "__main__":
    print("Select which tower calibration to run:")
    print("1. Rikku tower calibration")
    print("2. Yuna tower calibration")
    print("3. Paine tower calibration")
    print("4. Yuna reverse tower calibration (tower 10)")
    
    while True:
        choice = input("Enter your choice (1, 2, 3, or 4): ")
        if choice == "1":
            rikku = RikkuImplementation()
            rikku.run()
            break
        elif choice == "2":
            run_yuna()
            break
        elif choice == "3":
            run_paine()
            break
        elif choice == "4":
            run_yuna_reverse()
            break
        else:
            print("Invalid choice. Please enter 1, 2, 3, or 4.") 