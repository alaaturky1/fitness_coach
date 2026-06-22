from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np

from app.core.models import Joint


@dataclass
class PoseDetectionResult:
    joints: List[Joint]
    confidence: float
    error: Optional[str] = None
    error_code: Optional[str] = None
    image_width: Optional[int] = None
    image_height: Optional[int] = None
    detector: str = "mediapipe"


class PoseDetector:
    """Pose detector that prefers MediaPipe and falls back safely."""

    def __init__(self) -> None:
        self.use_mediapipe = False
        self.mp_pose = None
        self.pose = None
        self.init_error: Optional[str] = None

        try:
            import mediapipe as mp

            self.mp_pose = mp.solutions.pose
            self.pose = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                enable_segmentation=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self.use_mediapipe = True
        except Exception as exc:
            self.init_error = f"MediaPipe unavailable: {exc}"
    
    def detect_from_base64(self, image_b64: str) -> PoseDetectionResult:
        """Detect pose from base64 image"""
        try:
            # Decode base64 image
            img_data = base64.b64decode(image_b64)
            nparr = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                return PoseDetectionResult(
                    joints=[],
                    confidence=0.0,
                    error="Failed to decode image",
                    error_code="pose_image_decode_failed",
                    detector="opencv",
                )

            result = self._detect_from_image(img)
            result.image_height = int(img.shape[0])
            result.image_width = int(img.shape[1])
            return result
            
        except Exception as e:
            return PoseDetectionResult(
                joints=[],
                confidence=0.0,
                error=f"Image processing error: {str(e)}",
                error_code="pose_image_decode_failed",
                detector="opencv",
            )
    
    def _detect_from_image(self, img: np.ndarray) -> PoseDetectionResult:
        """Detect pose from numpy image"""
        if self.use_mediapipe:
            try:
                return self._detect_with_mediapipe(img)
            except Exception as exc:
                self.use_mediapipe = False
                self.init_error = f"MediaPipe inference failed: {exc}"
        return self._detect_fallback(img)
    
    def _detect_with_mediapipe(self, img: np.ndarray) -> PoseDetectionResult:
        """Use MediaPipe for pose detection"""
        rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_image)
        
        if not results.pose_landmarks:
            return PoseDetectionResult(
                joints=[],
                confidence=0.0,
                error="No pose detected",
                error_code="pose_not_detected",
            )
        
        landmarks = results.pose_landmarks.landmark
        joints = []
        
        # Map MediaPipe landmarks to our joint names (backend expects _l, _r format)
        joint_mappings = {
            'shoulder_l': 11,
            'shoulder_r': 12,
            'elbow_l': 13,
            'elbow_r': 14,
            'wrist_l': 15,
            'wrist_r': 16,
            'hip_l': 23,
            'hip_r': 24,
            'knee_l': 25,
            'knee_r': 26,
            'ankle_l': 27,
            'ankle_r': 28,
        }
        
        img_height, img_width = img.shape[:2]
        total_confidence = 0.0
        joint_count = 0
        
        for joint_name, landmark_idx in joint_mappings.items():
            landmark = landmarks[landmark_idx]
            
            # Convert normalized coordinates to pixel coordinates
            x = landmark.x * img_width
            y = landmark.y * img_height
            confidence = landmark.visibility
            
            joint = Joint(
                name=joint_name,
                x=x,
                y=y,
                confidence=confidence
            )
            joints.append(joint)
            
            total_confidence += confidence
            joint_count += 1
        
        avg_confidence = total_confidence / joint_count if joint_count > 0 else 0.0
        
        return PoseDetectionResult(
            joints=joints,
            confidence=avg_confidence,
        )
    
    def _detect_fallback(self, img: np.ndarray) -> PoseDetectionResult:
        """Return a safe failure instead of generating fake body landmarks."""
        return PoseDetectionResult(
            joints=[],
            confidence=0.0,
            error=self.init_error or "Using fallback detection (MediaPipe not available)",
            error_code="pose_detector_unavailable",
            detector="fallback",
        )


# Global pose detector instance
pose_detector = PoseDetector()


def detect_pose_from_image(image_b64: str) -> PoseDetectionResult:
    """Convenience function to detect pose from base64 image"""
    return pose_detector.detect_from_base64(image_b64)
