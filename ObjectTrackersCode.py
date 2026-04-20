import cv2
import time
import os
import json
import psutil

# --- OMGEVINGSINSTELLINGEN ---
os.environ["QT_QPA_PLATFORM"] = "xcb"
os.environ["QT_QPA_FONTDIR"] = "/usr/share/fonts/truetype/dejavu"

# =====================================================
# CONFIGURATIE
# =====================================================
tracker_type       = "VITTRACK"  # "KCF", "CSRT", "MOSSE", "NANOTRACK", "VITTRACK"
video_path         = "/home/arne/Videos/Final/Verberg.mp4"
gt_dir             = "/home/arne/tracking_project/groundTruth"
visualiseer        = False
aantal_runs        = 5
pauze_tussen_runs  = 10  # seconden

models = {
    "vittrack":           "object_tracking_vittrack_2023sep.onnx",
    "nanotrack_backbone": "object_tracking_nanotrack_backbone_2021sep.onnx",
    "nanotrack_head":     "object_tracking_nanotrack_head_2021sep.onnx"
}
# =====================================================

video_name = os.path.splitext(os.path.basename(video_path))[0]
gt_path = os.path.join(gt_dir, f"{video_name}_gt.json")

clicked_point = None

def click_event(event, x, y, flags, param):
    global clicked_point
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_point = (x, y)

def get_tracker(t_type):
    t_type = t_type.upper()
    if t_type == "KCF":       return cv2.TrackerKCF_create()
    elif t_type == "CSRT":    return cv2.TrackerCSRT_create()
    elif t_type == "MOSSE":   return cv2.legacy.TrackerMOSSE_create()
    elif t_type == "VITTRACK":
        params = cv2.TrackerVit_Params()
        params.net = models["vittrack"]
        return cv2.TrackerVit_create(params)
    elif t_type == "NANOTRACK":
        params = cv2.TrackerNano_Params()
        params.backbone = models["nanotrack_backbone"]
        params.neckhead = models["nanotrack_head"]
        return cv2.TrackerNano_create(params)

def select_bbox(frame, gt_bbox=None):
    global clicked_point
    sizex, sizey = (gt_bbox[2], gt_bbox[3]) if gt_bbox else (75, 75)
    clicked_point = (gt_bbox[0] + sizex // 2, gt_bbox[1] + sizey // 2) if gt_bbox else None

    cv2.namedWindow("Select Object")
    cv2.setMouseCallback("Select Object", click_event)
    print("Klik op object, beweeg met Z/S/Q/D, bevestig met SPACE.")

    while True:
        display = frame.copy()
        key = cv2.waitKey(1) & 0xFF

        if clicked_point is not None:
            cx, cy = clicked_point
            x = max(0, min(int(cx - sizex / 2), frame.shape[1] - sizex))
            y = max(0, min(int(cy - sizey / 2), frame.shape[0] - sizey))

            if gt_bbox:
                gx, gy, gw, gh = gt_bbox
                cv2.rectangle(display, (gx, gy), (gx + gw, gy + gh), (255, 100, 0), 2)
                cv2.putText(display, "GT (blauw)", (gx, gy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 0), 1)

            cv2.rectangle(display, (x, y), (x + sizex, y + sizey), (0, 255, 0), 2)
            cv2.putText(display, "Selectie - SPACE=ok", (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            if key == ord('z'): clicked_point = (cx, cy - 1)
            elif key == ord('s'): clicked_point = (cx, cy + 1)
            elif key == ord('q'): clicked_point = (cx - 1, cy)
            elif key == ord('d'): clicked_point = (cx + 1, cy)

            if key == 32:
                cv2.destroyWindow("Select Object")
                return (x, y, sizex, sizey)

        cv2.imshow("Select Object", display)
        if key == 27:
            cv2.destroyWindow("Select Object")
            return None

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
    intersection = max(0, xB - xA) * max(0, yB - yA)
    union = boxA[2] * boxA[3] + boxB[2] * boxB[3] - intersection
    return intersection / union if union > 0 else 0.0

def run_benchmark(frames, frame_tijden, bbox, ground_truth):
    """Voer Ã©Ã©n benchmark run uit en geef de resultaten terug."""
    tracker = get_tracker(tracker_type)
    tracker.init(frames[0], bbox)

    time_algo_list     = []
    time_pipeline_list = []
    iou_list          = []
    cpu_list          = []
    success_frames    = 0
    total_frames      = 0
    lost_counter      = 0
    was_tracking      = True
    process           = psutil.Process(os.getpid())

    for frame_id, frame in enumerate(frames[1:], start=1):
        # --- Puur algoritme ---
        t_start = time.perf_counter()
        success, result_bbox = tracker.update(frame)
        t_end = time.perf_counter()

        time_algo = t_end - t_start
        time_pipeline = time_algo + frame_tijden[frame_id]  # algo + laadtijd van dit frame

        time_algo_list.append(time_algo)
        time_pipeline_list.append(time_pipeline)
        cpu_list.append(process.cpu_percent(interval=0.0) / psutil.cpu_count())

        total_frames += 1
        if success:
            success_frames += 1
        else:
            if was_tracking:
                lost_counter += 1
        was_tracking = success

        current_iou = None
        if ground_truth and frame_id < len(ground_truth):
            gt_box = ground_truth[frame_id]
            if gt_box is not None:
                current_iou = iou(list(result_bbox), gt_box) if success else 0.0
                iou_list.append(current_iou)

        if visualiseer:
            display = frame.copy()
            if ground_truth and frame_id < len(ground_truth):
                gt_box = ground_truth[frame_id]
                if gt_box is not None:
                    gx, gy, gw, gh = gt_box
                    cv2.rectangle(display, (gx, gy), (gx + gw, gy + gh), (255, 100, 0), 2)
                    cv2.putText(display, "GT", (gx, gy - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 0), 1)

            if success:
                x, y, w, h = [int(v) for v in result_bbox]
                cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(display, tracker_type, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            else:
                cv2.putText(display, "LOST", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            fps_algo_now = len(time_algo_list) / sum(time_algo_list)
            fps_pipeline_now = len(time_pipeline_list) / sum(time_pipeline_list)
            cv2.putText(display, f"FPS algo:     {int(fps_algo_now)}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            cv2.putText(display, f"FPS pipeline: {int(fps_pipeline_now)}", (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            cv2.putText(display, f"CPU: {cpu_list[-1]:.1f}%", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            if current_iou is not None:
                kleur = (0, 255, 0) if current_iou > 0.5 else (0, 165, 255) if current_iou > 0.2 else (0, 0, 255)
                cv2.putText(display, f"IoU: {current_iou:.2f}", (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.6, kleur, 2)

            cv2.imshow("Benchmark", display)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cv2.destroyAllWindows()

    active_ious = [i for i in iou_list if i > 0]
    return {
        "fps_algo":      len(time_algo_list) / sum(time_algo_list) if sum(time_algo_list) > 0 else 0,
        "fps_pipeline":  len(time_pipeline_list) / sum(time_pipeline_list) if sum(time_pipeline_list) > 0 else 0,
        "cpu":           sum(cpu_list) / len(cpu_list) if cpu_list else 0,
        "success_rate":  (success_frames / total_frames) * 100 if total_frames else 0,
        "lost":          lost_counter,
        "iou_vot":       sum(iou_list) / len(iou_list) if iou_list else 0,
        "iou_otb":       sum(active_ious) / len(active_ious) if active_ious else 0,
    }

# --- Laad ground truth ---
ground_truth = None
if os.path.exists(gt_path):
    with open(gt_path, "r") as f:
        ground_truth = json.load(f)
    print(f"Ground truth geladen: {gt_path} ({len(ground_truth)} frames)")
else:
    print(f"Geen ground truth gevonden op: {gt_path} â€” benchmark zonder IoU.")

# --- Frames vooraf inladen in RAM + laadtijd per frame bijhouden ---
print("Frames inladen in RAM...")
frames = []
frame_tijden = []  # laadtijd per frame in seconden
video = cv2.VideoCapture(video_path)
t_load_totaal_start = time.perf_counter()
while True:
    t_frame_start = time.perf_counter()
    ret, frame = video.read()
    if not ret:
        break
    frame = cv2.resize(frame, (640, 360))
    t_frame_end = time.perf_counter()
    frames.append(frame)
    frame_tijden.append(t_frame_end - t_frame_start)
video.release()
t_load_totaal_end = time.perf_counter()
laadtijd = t_load_totaal_end - t_load_totaal_start
print(f"{len(frames)} frames ingeladen in {laadtijd:.2f} seconden.")

# --- Eenmalige selectie voor alle runs ---
first_gt = next((e for e in ground_truth if e is not None), None) if ground_truth else None
bbox = select_bbox(frames[0], first_gt)

if bbox is None:
    print("Geen selectie gemaakt, stoppen.")
    cv2.destroyAllWindows()
    exit()

# --- Meerdere runs ---
print(f"\n-- Benchmark gestart: {tracker_type} | {aantal_runs} runs | {pauze_tussen_runs}s pauze --")
alle_resultaten = []

for run in range(1, aantal_runs + 1):
    print(f"\n  Run {run}/{aantal_runs}...")
    resultaat = run_benchmark(frames, frame_tijden, bbox, ground_truth)
    alle_resultaten.append(resultaat)

    print(f"  {'='*42}")
    print(f"  RESULTATEN RUN {run}: {tracker_type}")
    print(f"  {'='*42}")
    print(f"  FPS puur algoritme:     {resultaat['fps_algo']:.2f}")
    print(f"  FPS volledige pipeline: {resultaat['fps_pipeline']:.2f}")
    print(f"  Gemiddeld CPU:          {resultaat['cpu']:.2f}%")
    print(f"  Tracking success rate:  {resultaat['success_rate']:.2f}%")
    print(f"  Aantal keer verloren:   {resultaat['lost']}")
    print(f"  IoU VOT (strict):       {resultaat['iou_vot']:.3f}")
    print(f"  IoU OTB (actief):       {resultaat['iou_otb']:.3f}")
    print(f"  {'='*42}")

    if run < aantal_runs:
        print(f"\n  Pauze van {pauze_tussen_runs} seconden voor CPU afkoeling...")
        time.sleep(pauze_tussen_runs)

# --- Gemiddelde over alle runs ---
def gem(key): return sum(r[key] for r in alle_resultaten) / len(alle_resultaten)

print(f"\n{'='*45}")
print(f"  GEMIDDELDE RESULTATEN: {tracker_type} ({aantal_runs} runs)")
print(f"{'='*45}")
print(f"  Frames ingeladen:         {len(frames)}")
print(f"  Laadtijd video (RAM):     {laadtijd:.2f} sec")
print(f"  ---")
print(f"  FPS puur algoritme:       {gem('fps_algo'):.2f}")
print(f"  FPS volledige pipeline:   {gem('fps_pipeline'):.2f}")
print(f"  Gemiddeld CPU verbruik:   {gem('cpu'):.2f}%")
print(f"  Tracking success rate:    {gem('success_rate'):.2f}%")
print(f"  Aantal keer verloren:     {gem('lost'):.1f}")
print(f"  ---")
print(f"  IoU VOT (strict):         {gem('iou_vot'):.3f}")
print(f"  IoU OTB (actief):         {gem('iou_otb'):.3f}")
print(f"{'='*45}")
