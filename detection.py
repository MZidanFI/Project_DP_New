import cv2

def video_detection(cap, model, enhance_image, enhance_type, confidence, writer, stframe, progress, total_frames):
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        # 1. Image Enhancement
        frame_proc = enhance_image(frame, enhance_type)

        # 2. YOLOv11 Detection
        results = model.predict(
            source=frame_proc, 
            conf=confidence, 
            imgsz=640, 
            verbose=False
        )
        res = results[0]

        # 3. Visualisasi (Bounding Boxes)
        # Menggunakan plot() bawaan YOLO, lalu ditimpa dengan info tambahan
        annotated = res.plot() if hasattr(res, "plot") else frame_proc.copy()
        count = len(res.boxes) if hasattr(res, "boxes") else 0

        # 4. Tambahan HUD 
        # Background Transparan Hitam
        overlay = annotated.copy()
        cv2.rectangle(overlay, (10, 10), (280, 70), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, annotated, 0.5, 0, annotated)

        # Teks Jumlah Deteksi
        text_info = f"Detected: {count}"
        cv2.putText(annotated, text_info, (25, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        # 5. Render ke Streamlit
        stframe.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)
        
        # Update Progress Bar (Safe Check)
        if total_frames > 0:
            progress.progress(min(frame_idx / total_frames, 1.0))

        # 6. Simpan Video (Jika mencentang Save)
        if writer:
            writer.write(annotated)