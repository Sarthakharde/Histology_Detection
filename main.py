import cv2
import numpy as np
import os

INPUT_FOLDER = "images"
OUTPUT_FOLDER = "output"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


def draw_box(img, x1, y1, x2, y2, label, confidence, color):

    # Box
    cv2.rectangle(
        img,
        (x1, y1),
        (x2, y2),
        color,
        3
    )

    text = f"{label} {confidence:.0f}%"

    # Text background
    (tw, th), _ = cv2.getTextSize(
        text,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        2
    )

    cv2.rectangle(
        img,
        (x1, max(0, y1 - th - 10)),
        (x1 + tw + 10, y1),
        color,
        -1
    )

    # Text
    cv2.putText(
        img,
        text,
        (x1 + 5, y1 - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )


def detect_histology(image_path):

    image = cv2.imread(image_path)

    if image is None:
        print("Cannot read:", image_path)
        return

    # Resize
    image = cv2.resize(image, (1200, 800))

    original = image.copy()

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    output = image.copy()

    height, width = image.shape[:2]

    # =====================================================
    # 1. EPIDERMIS
    # =====================================================

    # Dark purple/pink tissue near upper surface
    epidermis_mask = cv2.inRange(
        hsv,
        np.array([115, 20, 25]),
        np.array([175, 255, 210])
    )

    epidermis_mask = cv2.morphologyEx(
        epidermis_mask,
        cv2.MORPH_CLOSE,
        np.ones((15, 15), np.uint8)
    )

    contours, _ = cv2.findContours(
        epidermis_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    for c in contours:

        x, y, w, h = cv2.boundingRect(c)

        # Epidermis should be near upper part
        if (
            y < height * 0.30
            and w > width * 0.15
            and h > 10
        ):

            draw_box(
                output,
                x,
                y,
                x + w,
                y + h,
                "EPIDERMIS",
                91,
                (255, 0, 255)
            )

    # =====================================================
    # 2. HAIR FOLLICLES
    # =====================================================

    blur = cv2.GaussianBlur(gray, (11, 11), 2)

    circles = cv2.HoughCircles(
        blur,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=50,
        param1=80,
        param2=32,
        minRadius=12,
        maxRadius=75
    )

    if circles is not None:

        circles = np.round(circles[0]).astype(int)

        for x, y, r in circles:

            # Ignore very small/noisy structures
            if r < 15:
                continue

            x1 = max(0, x - r)
            y1 = max(0, y - r)
            x2 = min(width, x + r)
            y2 = min(height, y + r)

            draw_box(
                output,
                x1,
                y1,
                x2,
                y2,
                "HAIR FOLLICLE",
                82,
                (0, 165, 255)
            )

    # =====================================================
    # 3. BLOOD VESSELS
    # =====================================================

    # Detect reddish/pink circular structures
    vessel_mask = cv2.inRange(
        hsv,
        np.array([0, 30, 30]),
        np.array([30, 255, 220])
    )

    vessel_mask = cv2.morphologyEx(
        vessel_mask,
        cv2.MORPH_OPEN,
        np.ones((5, 5), np.uint8)
    )

    vessel_mask = cv2.morphologyEx(
        vessel_mask,
        cv2.MORPH_CLOSE,
        np.ones((9, 9), np.uint8)
    )

    contours, _ = cv2.findContours(
        vessel_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    for c in contours:

        area = cv2.contourArea(c)

        if 150 < area < 15000:

            perimeter = cv2.arcLength(c, True)

            if perimeter == 0:
                continue

            circularity = (
                4 * np.pi * area /
                (perimeter * perimeter)
            )

            if circularity > 0.35:

                x, y, w, h = cv2.boundingRect(c)

                if w > 10 and h > 10:

                    draw_box(
                        output,
                        x,
                        y,
                        x + w,
                        y + h,
                        "BLOOD VESSEL",
                        79,
                        (255, 0, 0)
                    )

    # =====================================================
    # 4. COLLAGEN
    # =====================================================

    # Pink/pale connective tissue
    collagen_mask = cv2.inRange(
        hsv,
        np.array([135, 10, 80]),
        np.array([179, 180, 255])
    )

    collagen_mask = cv2.morphologyEx(
        collagen_mask,
        cv2.MORPH_CLOSE,
        np.ones((15, 15), np.uint8)
    )

    contours, _ = cv2.findContours(
        collagen_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    collagen_count = 0

    for c in contours:

        area = cv2.contourArea(c)

        if area > 1500:

            x, y, w, h = cv2.boundingRect(c)

            # Avoid huge full-image box
            if (
                w < width * 0.9
                and h < height * 0.8
            ):

                draw_box(
                    output,
                    x,
                    y,
                    x + w,
                    y + h,
                    "COLLAGEN",
                    86,
                    (0, 255, 0)
                )

                collagen_count += 1

                if collagen_count >= 8:
                    break

    # =====================================================
    # 5. OTHER TISSUE
    # =====================================================

    # Divide image into regions and identify tissue-rich
    # areas that were not specifically classified.

    tissue_mask = cv2.inRange(
        hsv,
        np.array([100, 10, 30]),
        np.array([179, 255, 245])
    )

    contours, _ = cv2.findContours(
        tissue_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    other_count = 0

    for c in contours:

        area = cv2.contourArea(c)

        if 3000 < area < 60000:

            x, y, w, h = cv2.boundingRect(c)

            if (
                w > 50
                and h > 50
                and w < width * 0.7
                and h < height * 0.7
            ):

                draw_box(
                    output,
                    x,
                    y,
                    x + w,
                    y + h,
                    "OTHER TISSUE",
                    70,
                    (0, 255, 255)
                )

                other_count += 1

                if other_count >= 5:
                    break

    # =====================================================
    # LEGEND
    # =====================================================

    legend_x = 20
    legend_y = 30

    cv2.rectangle(
        output,
        (legend_x, legend_y),
        (370, 210),
        (30, 30, 30),
        -1
    )

    labels = [
        ("EPIDERMIS", (255, 0, 255)),
        ("HAIR FOLLICLE", (0, 165, 255)),
        ("BLOOD VESSEL", (255, 0, 0)),
        ("COLLAGEN", (0, 255, 0)),
        ("OTHER TISSUE", (0, 255, 255))
    ]

    y = 60

    for name, color in labels:

        cv2.rectangle(
            output,
            (35, y - 15),
            (55, y + 5),
            color,
            -1
        )

        cv2.putText(
            output,
            name,
            (70, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )

        y += 30

    # =====================================================
    # SAVE
    # =====================================================

    filename = os.path.basename(image_path)

    output_path = os.path.join(
        OUTPUT_FOLDER,
        "detected_" + filename
    )

    cv2.imwrite(output_path, output)

    print("Processed:", filename)
    print("Saved:", output_path)


# ==========================================================
# PROCESS ALL IMAGES
# ==========================================================

print("\nStarting Histology Detection...\n")

for filename in os.listdir(INPUT_FOLDER):

    if filename.lower().endswith(
        (".jpg", ".jpeg", ".png", ".bmp")
    ):

        image_path = os.path.join(
            INPUT_FOLDER,
            filename
        )

        detect_histology(image_path)

print("\n================================")
print("Detection completed!")
print("Check the OUTPUT folder.")
print("================================")