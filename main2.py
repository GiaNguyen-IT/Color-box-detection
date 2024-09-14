import cv2
import numpy as np

def define_color_ranges():
    color_ranges = {
        'red': ([0, 100, 100], [10, 255, 255]),
        'blue': ([100, 100, 100], [130, 255, 255]),
        'yellow': ([20, 100, 100], [30, 255, 255]),
    }
    return {k: (np.array(v[0], dtype="uint8"), np.array(v[1], dtype="uint8")) for k, v in color_ranges.items()}

def detect_colors(frame, color_ranges):
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    for color_name, (lower, upper) in color_ranges.items():
        mask = cv2.inRange(hsv_frame, lower, upper)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if cv2.contourArea(contour) > 500:  # Lọc các khối nhỏ
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, color_name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

def main():
    image_path = 'box2.png' 
    frame = cv2.imread(image_path)
    
    if frame is None:
        print("Không thể đọc được hình ảnh")
        return

    frame = cv2.resize(frame, (500, 500))
    
    color_ranges = define_color_ranges()

    detect_colors(frame, color_ranges)
    
    cv2.imshow("Color Detection", frame)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
