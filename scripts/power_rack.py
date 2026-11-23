"""Extract paired-view squat features, fit a research model, or verify metrics."""

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path

LABELS = ("PASS", "FAIL")
RETAINED_COLUMNS = [0, 1, 2, 3, 5, 6, 7, 8, 10, 11, 12, 13]
WEIGHT_HASHES = {
    "detector": "ceb11c07298f95c50d7c5abeb906d03340c85f23aa79e3e66966e7fb6c307250",
    "pose": "9aeb635b83f86aea45cf45d85798f7eba1a162de8e0d721c44e54fe5eebaf47d",
}


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def extract(video, detector_path, pose_path):
    import cv2
    import numpy as np
    from rtmlib import Body

    for name, path in (("detector", detector_path), ("pose", pose_path)):
        if digest(path) != WEIGHT_HASHES[name]:
            raise ValueError(f"Unexpected {name} weights: {path}")
    cv2.setNumThreads(1)
    body = Body(det=str(detector_path), det_input_size=(416, 416), pose=str(pose_path),
                pose_input_size=(192, 256), mode="lightweight", backend="onnxruntime",
                device="cpu", to_openpose=False)
    width, height = 768, 432
    frame_size = width * height * 3
    command = ["ffmpeg", "-v", "error", "-nostdin", "-threads", "1", "-filter_threads", "1",
               "-i", str(video), "-an", "-vf", f"fps=4,scale={width}:{height}",
               "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    frames, values, visibility = 0, [], []
    side_indices = ([5, 11, 13, 15], [6, 12, 14, 16])
    with subprocess.Popen(command, stdout=subprocess.PIPE) as decoder:
        while raw := decoder.stdout.read(frame_size):
            if len(raw) != frame_size:
                raise ValueError("Truncated decoded frame")
            frames += 1
            frame = np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3)
            front, front_score = body(frame[:, :width // 2].copy())
            side, side_score = body(frame[:, width // 2:].copy())
            if len(front) != 1 or len(side) != 1:
                continue
            joints = max(side_indices, key=lambda group: min(side_score[0][group]))
            score = min(float(min(front_score[0][[11, 12, 13, 14, 15, 16]])), float(min(side_score[0][joints])))
            if not np.isfinite(score):
                continue
            visibility.append(score)
            if score < 0.4:
                continue
            shoulder, hip, knee, ankle = side[0][joints]
            front_pose = front[0]
            thigh, shin = hip - knee, ankle - knee
            thigh_len, shin_len = np.linalg.norm(thigh), np.linalg.norm(shin)
            leg = thigh_len + shin_len
            ankle_width = abs(front_pose[15, 0] - front_pose[16, 0])
            if leg <= 1.0 or ankle_width <= 1.0:
                continue
            knee_cos = thigh.dot(shin) / max(1e-8, thigh_len * shin_len)
            trunk = shoulder - hip
            row = [abs(front_pose[13, 0] - front_pose[14, 0]) / ankle_width,
                   np.degrees(np.arctan2(abs(trunk[0]), abs(trunk[1]))),
                   (hip[1] - knee[1]) / leg,
                   np.degrees(np.arccos(np.clip(knee_cos, -1, 1))),
                   (height - ankle[1]) / leg]
            if np.isfinite(row).all():
                values.append(row)
        if decoder.wait():
            raise ValueError(f"Could not decode {video}")
    median = float(np.median(visibility)) if visibility else None
    coverage = {"frames": frames, "valid": len(values), "visible_median": median}
    usable = frames > 0 and len(values) >= 12 and len(values) / frames >= 0.5 and median is not None and median >= 0.4
    vector = np.quantile(values, [0.1, 0.5, 0.9], axis=0).reshape(-1).tolist() if usable else None
    return {"coverage": coverage, "vector": vector}


def rates(matrix):
    if len(matrix) != 2 or any(len(row) != 2 or any(type(n) is not int or n < 0 for n in row) for row in matrix):
        raise ValueError("Expected a 2×2 confusion matrix of nonnegative counts")
    if not all(sum(row) for row in matrix):
        raise ValueError("Both PASS and FAIL must be represented in evaluation")
    recall = [matrix[i][i] / sum(matrix[i]) for i in range(2)]
    return {"pass_recall": recall[0], "fail_recall": recall[1], "balanced_accuracy": sum(recall) / 2}


def load_features(path):
    import numpy as np

    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"Expected a nonempty JSON array in {path}")
    people = [row["person"] for row in rows]
    if any(not isinstance(person, str) or not person.strip() for person in people) or len(set(people)) != len(rows):
        raise ValueError(f"Expected one valid row per participant in {path}")
    y = [row["label"] for row in rows]
    if any(type(label) is not int or label not in (0, 1) for label in y) or min(y.count(0), y.count(1)) < 3:
        raise ValueError(f"Expected at least three rows per class, labeled 0 or 1, in {path}")
    x = np.asarray([row["vector"] for row in rows], dtype=float)
    if x.shape != (len(rows), 15) or not np.isfinite(x).all():
        raise ValueError(f"Expected 15 finite features per row in {path}")
    return set(people), x[:, RETAINED_COLUMNS], np.array(y)


def fit(train_path, test_path, output_model):
    import joblib
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import confusion_matrix
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    train_people, x_train, y_train = load_features(train_path)
    test_people, x_test, y_test = load_features(test_path)
    if train_people & test_people:
        raise ValueError("Training and evaluation participants overlap")
    model = make_pipeline(StandardScaler(), LogisticRegression(C=0.25, max_iter=1000, random_state=20261123))
    model.fit(x_train, y_train)
    prediction = (model.predict_proba(x_test)[:, 1] >= 0.5).astype(int)
    matrix = confusion_matrix(y_test, prediction, labels=[0, 1]).tolist()
    joblib.dump(model, output_model, compress=3)
    np.testing.assert_array_equal(joblib.load(output_model).predict_proba(x_test), model.predict_proba(x_test))
    return {"training_people": len(train_people), "evaluation_people": len(test_people),
            "labels": LABELS, "matrix": matrix, **rates(matrix)}


def verify(path):
    report = json.loads(path.read_text(encoding="utf-8"))
    if tuple(report["labels"]) != LABELS:
        raise ValueError("Metrics must use PASS and FAIL in that order")
    result = {}
    for name, metrics in report["models"].items():
        matrix = metrics["matrix"]
        calculated = rates(matrix)
        if [sum(row) for row in matrix] != report["evaluation_class_counts"]:
            raise ValueError(f"Unexpected class counts: {name}")
        if any(not math.isclose(value, metrics[key]) for key, value in calculated.items()):
            raise ValueError(f"Metric mismatch: {name}")
        result[name] = calculated
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    extraction = commands.add_parser("extract")
    for name in ("video", "detector", "pose", "output"):
        extraction.add_argument(f"--{name}", type=Path, required=True)
    training = commands.add_parser("fit")
    for name in ("train", "test", "model"):
        training.add_argument(f"--{name}", type=Path, required=True)
    commands.add_parser("verify").add_argument("report", type=Path)
    args = parser.parse_args()
    if args.command == "extract":
        result = extract(args.video, args.detector, args.pose)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    elif args.command == "fit":
        result = fit(args.train, args.test, args.model)
    else:
        result = verify(args.report)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
