# 🚘 Modular Hybrid ADAS Prototype

A real-time **Advanced Driver Assistance System (ADAS)** leveraging Computer Vision and Deep Learning for intelligent lane perception, vehicle tracking, and forward collision warning. Built natively in Python, this project mirrors the core sensor-fusion architectures used in early production autonomous vehicles.

---

## ✨ Core Features

*   **Intelligent Lane Perception (YOLOP):** Dynamically tracks lane boundaries, calculates real-time vehicle deviation (drift percentage), and predicts upcoming road curvature (Left/Right/Straight).
*   **Obstacle Tracking & FCW (YOLOv8):** Tracks vehicles, pedestrians, and infrastructure (traffic lights/stop signs). Calculates distance using focal-length projection and triggers a Forward Collision Warning (FCW) based on an Exponential Moving Average (EMA) of relative velocity and Time-To-Collision (TTC).
*   **Advanced Sensor Fusion (Occlusion Handling):** YOLO bounding boxes are fed directly into the YOLOP lane detection module. The system actively "blindfolds" the lane-search algorithm exactly where vehicles are detected, mathematically preventing the AI from hallucinating false lane lines on the bumpers or shadows of blocking trucks.
*   **Dynamic Inverse Visibility Scaling:** The system's curve-prediction sensitivity dynamically scales based on how far it can see down the road. If line-of-sight is blocked by traffic, it mathematically increases the dead-zone to ignore short-line noise. If the road is clear, it shrinks the dead-zone to accurately track mild, sweeping highway curves.
*   **Automotive-Grade Telemetry HUD:**
    *   **Bird's-Eye View (BEV) Radar:** A top-down 2D grid projecting the distance and lateral position of forward targets.
    *   **Steering Correction Arc:** A dynamic visual steering wheel that physically twists to guide the driver back to the lane center.
    *   **Distance-Graded Targeting:** Vehicles directly in the ego-path are color-graded (Green $\rightarrow$ Yellow $\rightarrow$ Red) based on closing distance, while adjacent lane traffic remains a passive Cyan to reduce visual clutter.

---

## 🧠 Architectural Highlights & Math Concepts

### Geometric "Bulge" Curve Detection
Instead of blindly trusting the polynomial coefficients of a fitted parabola (which often overfit on short line segments), this system uses pure geometry. It draws an invisible straight line from the bottom of the lane to the top of the lane, and measures the mathematical "bulge" (the pixel distance the physical lane pulls away from that straight string). 

### Temporal Memory State
To prevent the UI from flickering due to 1-frame camera glitches, the curvature math utilizes an **Exponential Moving Average (EMA)** memory state. A single-frame anomaly is completely absorbed, while a sustained physical curve smoothly glides the UI prediction up to its true percentage.

---

## 🛠️ Installation & Setup

**1. Clone the repository**
```bash
git clone [https://github.com/yourusername/modular-hybrid-adas.git](https://github.com/yourusername/modular-hybrid-adas.git)
cd modular-hybrid-adas