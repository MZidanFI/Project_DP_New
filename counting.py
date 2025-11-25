import cv2
import numpy as np

def video_counting(cap, model, enhance_image, enhance_type, confidence, writer, stframe, progress, total_frames):
    frame_idx = 0
    
    # --- 1. KONFIGURASI KELAS ---
    class_mapping = {
        'car': 'Mobil',
        'motorcycle': 'Motor',
        'motorbike': 'Motor',
        'bus': 'Bus',
        'truck': 'Truk',
        'bicycle': 'Sepeda'
    }
    
    # Inisialisasi hitungan berdasarkan values unik
    unique_names = list(set(class_mapping.values()))
    counts = {name: {'in': 0, 'out': 0} for name in unique_names}
    
    counted_ids = set()
    previous_positions = {}

    # --- 2. LOOPING VIDEO ---
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        # Pre-processing
        frame_proc = enhance_image(frame, enhance_type)
        height, width = frame_proc.shape[:2]
        line_y = int(height / 2) # Garis tengah

        # Tracking YOLO
        results = model.track(
            source=frame_proc,
            conf=confidence,
            imgsz=640,
            verbose=False,
            persist=True
        )

        res = results[0]
        annotated = res.plot() if hasattr(res, "plot") else frame_proc.copy()

        # --- 3. LOGIKA PENGHITUNGAN ---
        if getattr(res, "boxes", None) is not None and getattr(res.boxes, "id", None) is not None:
            boxes = res.boxes
            ids = boxes.id.cpu().numpy().tolist() if hasattr(boxes.id, "cpu") else list(boxes.id)
            xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, "cpu") else np.array(boxes.xyxy)
            classes = boxes.cls.cpu().numpy() if hasattr(boxes.cls, "cpu") else np.array(boxes.cls)

            for obj_id, box, cls_id in zip(ids, xyxy, classes):
                raw_name = model.names[int(cls_id)]
                model_class_name = raw_name.lower()
                
                if model_class_name in class_mapping:
                    display_name = class_mapping[model_class_name]
                    x1, y1, x2, y2 = box
                    cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

                    if obj_id in previous_positions:
                        prev_cy = previous_positions[obj_id]

                        if obj_id not in counted_ids:
                            # Logika: Melewati garis dari Atas ke Bawah (Turun/Masuk)
                            if prev_cy < line_y and cy >= line_y:
                                counts[display_name]['in'] += 1
                                counted_ids.add(obj_id)
                                cv2.line(annotated, (0, line_y), (width, line_y), (0, 255, 0), 5)
                            
                            # Logika: Melewati garis dari Bawah ke Atas (Naik/Keluar)
                            elif prev_cy > line_y and cy <= line_y:
                                counts[display_name]['out'] += 1
                                counted_ids.add(obj_id)
                                cv2.line(annotated, (0, line_y), (width, line_y), (255, 255, 0), 5)

                    previous_positions[obj_id] = cy
                    cv2.circle(annotated, (cx, cy), 6, (0, 255, 0), -1)

        # --- 4. TAMPILAN HUD ---
        # Garis Batas
        cv2.line(annotated, (0, line_y), (width, line_y), (0, 0, 255), 2)

        # Konfigurasi Tampilan
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.8
        thick = 2
        color_txt = (255, 255, 255)
        
        # Koordinat Kolom
        pos_nama    = 20
        pos_titik   = 140
        pos_angka1  = 170  # Angka Turun
        pos_garis   = 220  # Garis Pemisah
        pos_angka2  = 250  # Angka Naik

        # Background Kotak Transparan
        overlay = annotated.copy()
        cv2.rectangle(overlay, (10, 10), (320, 290), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, annotated, 0.5, 0, annotated)

        # Gambar Header
        y_head = 40
        cv2.putText(annotated, "TIPE", (pos_nama, y_head), font, scale, color_txt, thick)
        cv2.putText(annotated, "TRN",  (pos_angka1 - 10, y_head), font, scale, color_txt, thick)
        cv2.putText(annotated, "|",    (pos_garis, y_head), font, scale, color_txt, thick)
        cv2.putText(annotated, "NAIK", (pos_angka2 - 10, y_head), font, scale, color_txt, thick)
        
        # Garis Bawah Header
        cv2.line(annotated, (20, 50), (310, 50), color_txt, 2)

        # Gambar Data (Looping per Baris)
        y_offset = 80
        display_order = ['Mobil', 'Bus', 'Truk', 'Motor', 'Sepeda']
        
        for name in display_order:
            if name in counts:
                data = counts[name]

                cv2.putText(annotated, name,              (pos_nama, y_offset),   font, scale, color_txt, thick)
                cv2.putText(annotated, ":",               (pos_titik, y_offset),  font, scale, color_txt, thick)
                cv2.putText(annotated, str(data['in']),   (pos_angka1, y_offset), font, scale, color_txt, thick)
                cv2.putText(annotated, "|",               (pos_garis, y_offset),  font, scale, color_txt, thick)
                cv2.putText(annotated, str(data['out']),  (pos_angka2, y_offset), font, scale, color_txt, thick)

                y_offset += 40

        # Output ke Streamlit
        stframe.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)
        
        if total_frames > 0:
            progress.progress(min(frame_idx / total_frames, 1.0))

        if writer:
            writer.write(annotated)