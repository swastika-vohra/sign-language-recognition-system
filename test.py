import cv2
import numpy as np
import math
import tensorflow as tf
from cvzone.HandTrackingModule import HandDetector
import os 


MODEL_TFLITE_PATH = "C://Users//Swastika//OneDrive//Desktop//converted_tflite (3)//model_unquant.tflite"
LABELS_PATH = "C://Users//Swastika//OneDrive//Desktop//converted_tflite (3)//labels.txt"
CAMERA_INDEX = 0
OFFSET = 20 
FONT = cv2.FONT_HERSHEY_COMPLEX

try:
    with open(LABELS_PATH, 'r') as f:
        
        LABELS = [line.strip().split(' ', 1)[-1] for line in f if line.strip()]
    if not LABELS:
        LABELS = ["hello","i love you","no","sorry","thank you","where"] 
except FileNotFoundError:
    print(f"Warning: Label file not found at {LABELS_PATH}. Using hardcoded labels.")
    LABELS = ["hello","i love you","no","sorry","thank you","where"]
except Exception as e:
    print(f"Warning: Failed to load labels. Using hardcoded labels. Error: {e}")
    LABELS = ["hello","i love you","no","sorry","thank you","where"]



def initialize_model():
    """Loads the TFLite model and allocates tensors."""
    try:
        
        interpreter = tf.lite.Interpreter(model_path=MODEL_TFLITE_PATH)
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        MODEL_IMG_SIZE = input_details[0]['shape'][1] 
        print(f"TFLite Model Loaded. Expected Input Size: {MODEL_IMG_SIZE}x{MODEL_IMG_SIZE}")
        
        return interpreter, input_details, output_details, MODEL_IMG_SIZE

    except Exception as e:
        print(f"\nFATAL ERROR: Failed to load TFLite model or interpreter.")
        print(f"Please check the path: {MODEL_TFLITE_PATH}")
        print(f"Error details: {e}")
        raise SystemExit(1)

def predict_tflite(img_white, interpreter, input_details, output_details):
    """
    Runs inference on the pre-processed white image canvas.
    """
    
    input_data = np.asarray(img_white, dtype=np.float32)
    input_data = (input_data / 255.0) 
    
    input_data = np.expand_dims(input_data, axis=0)

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    prediction = output_data[0]
    index = np.argmax(prediction)
    
    return prediction[index], index

def main():
    """Main function to run the video stream and classification."""
    
    interpreter, input_details, output_details, imgSize = initialize_model()
    
    
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW) 

    if not cap.isOpened():
        print(f"\nERROR: Failed to open video stream/camera at index {CAMERA_INDEX}.")
        print("Suggestions: 1. Ensure no other app is using the camera. 2. Try index 1 or 2.")
        raise SystemExit(1)

    detector = HandDetector(maxHands=1)
    
    while True:
        success, img = cap.read()
        if not success:
            print("Failed to read frame from camera. Exiting.")
            break
            
        imgOutput = img.copy()
        
        hands, img = detector.findHands(img, draw=False)
        
        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']

            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            
            y_start = max(0, y - OFFSET)
            y_end = min(img.shape[0], y + h + OFFSET)
            x_start = max(0, x - OFFSET)
            x_end = min(img.shape[1], x + w + OFFSET)
            
            imgCrop = img[y_start:y_end, x_start:x_end]
            
            aspectRatio = imgCrop.shape[0] / imgCrop.shape[1] 

            if aspectRatio > 1:
                k = imgSize / imgCrop.shape[0] 
                wCal = math.ceil(k * imgCrop.shape[1])
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap: wCal + wGap] = imgResize
            else:
                
                k = imgSize / imgCrop.shape[1] 
                hCal = math.ceil(k * imgCrop.shape[0])
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap: hCal + hGap, :] = imgResize
                
            prediction, index = predict_tflite(imgWhite, interpreter, input_details, output_details)
            
            if 0 <= index < len(LABELS):
                confidence_text = f"{LABELS[index]} ({prediction*100:.1f}%)"
                
                (text_w, text_h), baseline = cv2.getTextSize(confidence_text, FONT, 1.5, 2)
                
                cv2.rectangle(imgOutput, 
                              (x - OFFSET, y - OFFSET - text_h - baseline - 10), 
                              (x - OFFSET + text_w + 10, y - OFFSET), 
                              (0, 255, 0), cv2.FILLED)
                
                cv2.putText(imgOutput, confidence_text, (x - OFFSET + 5, y - OFFSET - 5), 
                            FONT, 1.5, (0, 0, 0), 2)
                
            cv2.rectangle(imgOutput, 
                          (x - OFFSET, y - OFFSET), 
                          (x + w + OFFSET, y + h + OFFSET), 
                          (0, 255, 0), 4) 
            
            cv2.imshow('Hand Crop (Original Ratio)', imgCrop)
            cv2.imshow('Canvas (Input to Model)', imgWhite)

        cv2.imshow('ASL Recognizer', imgOutput)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
