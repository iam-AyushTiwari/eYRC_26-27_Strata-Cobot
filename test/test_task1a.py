#!/usr/bin/env python3

import unittest
import numpy as np
import cv2
import os
import sys

# Add task1a script path to sys.path so ore_detector can be imported
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TASK1A_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '../algorithms/scripts/task1a'))
if TASK1A_DIR not in sys.path:
    sys.path.insert(0, TASK1A_DIR)

from ore_detector import detect_ores


class TestTask1A(unittest.TestCase):
    """Unit test suite for e-Yantra Task 1A ore detection vision module."""

    def test_empty_image(self):
        """Test detection on a blank black image (no ores)."""
        black_img = np.zeros((480, 640, 3), dtype=np.uint8)
        centers, ore_types = detect_ores(black_img)
        self.assertEqual(len(centers), 0)
        self.assertEqual(len(ore_types), 0)

    def test_azurite_detection(self):
        """Test detection of Azurite ore (Blue color range in HSV)."""
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        # Create blue patch (HSV H=120, S=200, V=200 -> BGR)
        hsv_patch = np.full((50, 50, 3), (120, 200, 200), dtype=np.uint8)
        bgr_patch = cv2.cvtColor(hsv_patch, cv2.COLOR_HSV2BGR)
        
        # Place patch at center (u=320, v=240) -> y: 215..265, x: 295..345
        img[215:265, 295:345] = bgr_patch

        centers, ore_types = detect_ores(img)
        self.assertEqual(len(ore_types), 1)
        self.assertEqual(ore_types[0], 'azurite_ore')
        
        # Centroid should be approximately (320, 240)
        cX, cY = centers[0]
        self.assertAlmostEqual(cX, 320, delta=5)
        self.assertAlmostEqual(cY, 240, delta=5)

    def test_malachite_detection(self):
        """Test detection of Malachite ore (Green color range in HSV)."""
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        # Create green patch (HSV H=60, S=200, V=200 -> BGR)
        hsv_patch = np.full((50, 50, 3), (60, 200, 200), dtype=np.uint8)
        bgr_patch = cv2.cvtColor(hsv_patch, cv2.COLOR_HSV2BGR)
        
        img[100:150, 100:150] = bgr_patch

        centers, ore_types = detect_ores(img)
        self.assertEqual(len(ore_types), 1)
        self.assertEqual(ore_types[0], 'malachite_ore')
        
        cX, cY = centers[0]
        self.assertAlmostEqual(cX, 125, delta=5)
        self.assertAlmostEqual(cY, 125, delta=5)

    def test_vanadinite_detection(self):
        """Test detection of Vanadinite ore (Orange color range in HSV)."""
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        # Create orange patch (HSV H=15, S=200, V=200 -> BGR)
        hsv_patch = np.full((50, 50, 3), (15, 200, 200), dtype=np.uint8)
        bgr_patch = cv2.cvtColor(hsv_patch, cv2.COLOR_HSV2BGR)
        
        img[300:350, 400:450] = bgr_patch

        centers, ore_types = detect_ores(img)
        self.assertEqual(len(ore_types), 1)
        self.assertEqual(ore_types[0], 'vanadinite_ore')
        
        cX, cY = centers[0]
        self.assertAlmostEqual(cX, 425, delta=5)
        self.assertAlmostEqual(cY, 325, delta=5)

    def test_multiple_distinct_ores(self):
        """Test detection when multiple distinct ores are present in a single image."""
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Azurite (Blue)
        hsv_az = np.full((40, 40, 3), (120, 200, 200), dtype=np.uint8)
        img[50:90, 50:90] = cv2.cvtColor(hsv_az, cv2.COLOR_HSV2BGR)

        # Malachite (Green)
        hsv_mal = np.full((40, 40, 3), (60, 200, 200), dtype=np.uint8)
        img[50:90, 200:240] = cv2.cvtColor(hsv_mal, cv2.COLOR_HSV2BGR)

        # Vanadinite (Orange)
        hsv_van = np.full((40, 40, 3), (15, 200, 200), dtype=np.uint8)
        img[50:90, 400:440] = cv2.cvtColor(hsv_van, cv2.COLOR_HSV2BGR)

        centers, ore_types = detect_ores(img)
        self.assertEqual(len(ore_types), 3)
        self.assertIn('azurite_ore', ore_types)
        self.assertIn('malachite_ore', ore_types)
        self.assertIn('vanadinite_ore', ore_types)

    def test_duplicate_same_type_ores(self):
        """Test detection when multiple ores of the SAME type (e.g., 2 Azurites) are present."""
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # First Azurite patch
        hsv_az1 = np.full((40, 40, 3), (120, 200, 200), dtype=np.uint8)
        img[50:90, 50:90] = cv2.cvtColor(hsv_az1, cv2.COLOR_HSV2BGR)

        # Second Azurite patch
        hsv_az2 = np.full((40, 40, 3), (120, 200, 200), dtype=np.uint8)
        img[200:240, 300:340] = cv2.cvtColor(hsv_az2, cv2.COLOR_HSV2BGR)

        centers, ore_types = detect_ores(img)
        self.assertEqual(len(ore_types), 2)
        self.assertEqual(ore_types.count('azurite_ore'), 2)

    def test_noise_filtering(self):
        """Test that small color specks (< 100 px contour area) are filtered out as noise."""
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Tiny 5x5 blue speck (area = 25 pixels, below threshold)
        hsv_speck = np.full((5, 5, 3), (120, 200, 200), dtype=np.uint8)
        img[100:105, 100:105] = cv2.cvtColor(hsv_speck, cv2.COLOR_HSV2BGR)

        centers, ore_types = detect_ores(img)
        self.assertEqual(len(centers), 0, "Specks smaller than min contour area should be ignored")

    def test_unregistered_color(self):
        """Test that non-target colors (e.g., Purple) produce no detections."""
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Purple patch (Hue ~150, outside target ore ranges)
        hsv_purple = np.full((50, 50, 3), (150, 200, 200), dtype=np.uint8)
        img[200:250, 200:250] = cv2.cvtColor(hsv_purple, cv2.COLOR_HSV2BGR)

        centers, ore_types = detect_ores(img)
        self.assertEqual(len(centers), 0)


if __name__ == '__main__':
    unittest.main()