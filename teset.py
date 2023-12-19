import cv2
import inference
import supervision as sv
from collections import Counter

# set ROBOFLOW_API_KEY=CsK9Wrm4sbp5UWhPyYoU
# python teset.py

annotator = sv.BoxAnnotator()

def on_prediction(predictions, image):
    labels = [p["class"] for p in predictions["predictions"]]
    label_counts = Counter(labels)  # 감지된 레이블 카운트
    detections = sv.Detections.from_roboflow(predictions)

    annotated_image = annotator.annotate(
        scene=image,
        detections=detections,
        labels=labels
    )

    # 감지된 각 레이블과 개수를 화면에 표시
    y_offset = 23

    # 더 큰 출력 창을 위한 창 생성
    cv2.namedWindow("Prediction", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Prediction", 1280, 600)  # 원하는 크기로 조절

    for label, count in label_counts.items():
        cv2.putText(annotated_image, f"{label}: {count}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 5, 0), 2)
        y_offset += 23

    cv2.imshow("Prediction", annotated_image)
    cv2.waitKey(1)

inference.Stream(
    source=1,
    model="yogurt_detection/3",
    output_channel_order="BGR",
    use_main_thread=True,
    on_prediction=on_prediction,
)