When discussing cube-recognition confidence, distinguish the pose model's
per-object detection/class score (`result.boxes.conf`) from its
keypoint-presence channel (`result.keypoints.conf`) and from the separate
cube bounding-box detector's score.

The independent renderer's landmark thresholds of 0.25/0.40/0.25 and sticker
threshold of 0.90 apply to the pose model's per-object detection/class scores.
Each independent landmark or sticker is one detected object with one keypoint.

For an object containing multiple keypoints, its detection confidence is one
overall object score, not a separate localization confidence for each point.
Keypoint-presence scores use positive targets for both visibility 1 and 2 and
can saturate near one. Describe them explicitly as presence scores whenever
they are used. Check the actual tensor and threshold before interpreting a
displayed confidence.
