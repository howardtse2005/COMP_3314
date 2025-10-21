import numpy as np
import torch
from scipy import ndimage
from sklearn.metrics import adjusted_rand_score
from scipy.ndimage import distance_transform_edt, binary_erosion, label

#--------------------- Benchmark Metric Calculation Functions ---------------------

def _to_binary(prediction, ground_truth, threshold):
    pred_binary = (prediction > threshold).astype(np.uint8)
    gt_binary = (ground_truth > 0.5).astype(np.uint8)
    return pred_binary, gt_binary

def _boundary_from_mask(mask):
    # Morphological gradient (XOR with eroded) as boundary
    eroded = binary_erosion(mask, structure=np.ones((3, 3)))
    return (mask ^ eroded).astype(np.bool_)

def calculate_warping_error(prediction, ground_truth, threshold=0.5):
    """Symmetric average boundary distance (normalized) as warping error proxy."""
    pred_binary, gt_binary = _to_binary(prediction, ground_truth, threshold)
    pred_edges = _boundary_from_mask(pred_binary)
    gt_edges = _boundary_from_mask(gt_binary)

    # Handle empty edges
    if gt_edges.sum() == 0 and pred_edges.sum() == 0:
        return 0.0
    if gt_edges.sum() == 0 and pred_edges.sum() > 0:
        return 1.0
    if gt_edges.sum() > 0 and pred_edges.sum() == 0:
        return 1.0

    # Distance transforms on complement of edges
    dt_gt = distance_transform_edt(~gt_edges)
    dt_pred = distance_transform_edt(~pred_edges)

    # Mean min distance from pred->gt and gt->pred
    d_pred_to_gt = dt_gt[pred_edges].mean() if pred_edges.any() else 0.0
    d_gt_to_pred = dt_pred[gt_edges].mean() if gt_edges.any() else 0.0
    d_sym = 0.5 * (d_pred_to_gt + d_gt_to_pred)

    # Normalize by image diagonal to keep in [0,1] scale
    h, w = prediction.shape[:2]
    diag = np.sqrt(h * h + w * w)
    return float(d_sym / (diag / 2.0))  # conservative normalization

def calculate_rand_error(prediction, ground_truth, threshold=0.5):
    """Foreground-restricted Rand error via ARI on connected components (regions)."""
    pred_binary, gt_binary = _to_binary(prediction, ground_truth, threshold)

    # Connected components on inverse (treat non-boundary as regions)
    pred_labels, _ = label(1 - pred_binary)
    gt_labels, _ = label(1 - gt_binary)

    # ARI on labelings
    ari = adjusted_rand_score(gt_labels.flatten(), pred_labels.flatten())
    rand_error = 1.0 - ari
    return max(0.0, float(rand_error))

def calculate_pixel_error(prediction, ground_truth, threshold=0.5):
    """Per-pixel misclassification rate."""
    pred_binary, gt_binary = _to_binary(prediction, ground_truth, threshold)
    incorrect_pixels = np.sum(pred_binary != gt_binary)
    total_pixels = pred_binary.size
    return float(incorrect_pixels / total_pixels)

def calculate_iou(prediction, ground_truth, threshold=0.5):
    """Calculate IoU for a single image
    Args:
        prediction: Prediction array (H, W) with values 0-1
        ground_truth: Ground truth array (H, W) with values 0-1
    """
    pred_binary = (prediction > threshold).astype(np.uint8)
    gt_binary = (ground_truth > 0.5).astype(np.uint8)
    
    intersection = np.sum((pred_binary == 1) & (gt_binary == 1))
    union = np.sum((pred_binary == 1) | (gt_binary == 1))
    
    return intersection / union if union > 0 else 1.0

def calculate_confusion_matrix(prediction, ground_truth, threshold=0.5):
    """Calculate confusion matrix values for a single image
    Args:
        prediction: Prediction array (H, W) with values 0-1
        ground_truth: Ground truth array (H, W) with values 0-1
    """
    pred_binary = (prediction > threshold).astype(np.uint8)
    gt_binary = (ground_truth > 0.5).astype(np.uint8)
    
    tp = np.sum((pred_binary == 1) & (gt_binary == 1))
    tn = np.sum((pred_binary == 0) & (gt_binary == 0))
    fp = np.sum((pred_binary == 1) & (gt_binary == 0))
    fn = np.sum((pred_binary == 0) & (gt_binary == 1))

    return {'TP': int(tp), 'TN': int(tn), 'FP': int(fp), 'FN': int(fn)}

def _errors_over_thresholds(pred, gt, thresholds):
    """
    Compute warping, rand and pixel error over thresholds and return arrays + best per metric.
    """
    warps, rands, pixels = [], [], []
    for t in thresholds:
        warps.append(calculate_warping_error(pred, gt, t))
        rands.append(calculate_rand_error(pred, gt, t))
        pixels.append(calculate_pixel_error(pred, gt, t))
    warps = np.array(warps)
    rands = np.array(rands)
    pixels = np.array(pixels)
    best_warp_idx = int(np.argmin(warps))
    best_rand_idx = int(np.argmin(rands))
    best_pixel_idx = int(np.argmin(pixels))
    return {
        'warps': warps, 'rands': rands, 'pixels': pixels,
        'best_warp_err': float(warps[best_warp_idx]), 'best_warp_th': float(thresholds[best_warp_idx]),
        'best_rand_err': float(rands[best_rand_idx]), 'best_rand_th': float(thresholds[best_rand_idx]),
        'best_pixel_err': float(pixels[best_pixel_idx]), 'best_pixel_th': float(thresholds[best_pixel_idx]),
    }

# Calculate threshold-based metrics
def calculate_metrics(predictions, groundtruths, threshold=0.5):
    """
    Calculate metrics; for warping/rand/pixel, sweep 10 thresholds and pick best (per U-Net paper).
    Args:
        predictions: List of prediction arrays, each (H, W) with values 0-1
        groundtruths: List of ground truth arrays, each (H, W) with values 0-1
        threshold: Kept for backward-compatibility (used for IoU/confusion-only)
    """
    if not predictions or not groundtruths:
        print("Warning: No prediction or ground truth data for metric calculation")
        return {}
    
    # Threshold sweep: 10 levels (0.05, 0.15, ..., 0.95)
    sweep_thresholds = np.linspace(0.05, 0.95, 10)

    per_image_metrics = []
    total_tp = total_fp = total_tn = total_fn = 0

    # Collect per-threshold errors to derive dataset-level best thresholds
    all_warp_errors = []  # list of arrays shape [10]
    all_rand_errors = []
    all_pixel_errors = []

    for idx, (pred, gt) in enumerate(zip(predictions, groundtruths)):
        if isinstance(pred, torch.Tensor):
            pred = pred.cpu().numpy()
        if isinstance(gt, torch.Tensor):
            gt = gt.cpu().numpy()

        # Errors at single threshold for legacy fields (kept at provided 'threshold')
        warping_error_single = calculate_warping_error(pred, gt, threshold)
        rand_error_single = calculate_rand_error(pred, gt, threshold)
        pixel_error_single = calculate_pixel_error(pred, gt, threshold)
        iou = calculate_iou(pred, gt, threshold)
        confusion = calculate_confusion_matrix(pred, gt, threshold)

        # Sweep thresholds and pick best per metric
        sweep = _errors_over_thresholds(pred, gt, sweep_thresholds)
        all_warp_errors.append(sweep['warps'])
        all_rand_errors.append(sweep['rands'])
        all_pixel_errors.append(sweep['pixels'])

        total_tp += confusion['TP']
        total_fp += confusion['FP']
        total_tn += confusion['TN']
        total_fn += confusion['FN']

        per_image_metrics.append({
            # Best-over-threshold metrics (requested)
            'warping_error': sweep['best_warp_err'],
            'warping_best_threshold': sweep['best_warp_th'],
            'rand_error': sweep['best_rand_err'],
            'rand_best_threshold': sweep['best_rand_th'],
            'pixel_error': sweep['best_pixel_err'],
            'pixel_best_threshold': sweep['best_pixel_th'],
            # Also keep IoU and confusion computed at the provided threshold
            'iou': iou,
            'TP': confusion['TP'],
            'TN': confusion['TN'],
            'FP': confusion['FP'],
            'FN': confusion['FN'],
            # Optional reference: errors at the provided single threshold
            'warping_error@fixed': warping_error_single,
            'rand_error@fixed': rand_error_single,
            'pixel_error@fixed': pixel_error_single,
        })
    
    num_images = len(per_image_metrics)
    avg_metrics = {
        # Averages of best-per-image errors
        'warping_error': float(sum(m['warping_error'] for m in per_image_metrics) / num_images),
        'rand_error': float(sum(m['rand_error'] for m in per_image_metrics) / num_images),
        'pixel_error': float(sum(m['pixel_error'] for m in per_image_metrics) / num_images),
        'iou': float(sum(m['iou'] for m in per_image_metrics) / num_images),
    }

    # Global metrics from fixed-threshold confusion (unchanged)
    total_pixels = total_tp + total_fp + total_tn + total_fn
    global_pixel_error = (total_fp + total_fn) / total_pixels if total_pixels > 0 else 0.0
    global_iou = total_tp / (total_tp + total_fp + total_fn) if (total_tp + total_fp + total_fn) > 0 else 0.0
    global_metrics = { 'pixel_error': float(global_pixel_error), 'iou': float(global_iou) }

    # Dataset-level best thresholds (min average error across images)
    all_warp_errors = np.stack(all_warp_errors, axis=0)  # [N, 10]
    all_rand_errors = np.stack(all_rand_errors, axis=0)  # [N, 10]
    all_pixel_errors = np.stack(all_pixel_errors, axis=0)  # [N, 10]

    mean_warp_per_th = all_warp_errors.mean(axis=0)
    mean_rand_per_th = all_rand_errors.mean(axis=0)
    mean_pixel_per_th = all_pixel_errors.mean(axis=0)

    warp_best_idx = int(np.argmin(mean_warp_per_th))
    rand_best_idx = int(np.argmin(mean_rand_per_th))
    pixel_best_idx = int(np.argmin(mean_pixel_per_th))

    best_errors_over_thresholds = {
        'thresholds': sweep_thresholds.tolist(),
        'warping': {
            'mean_per_threshold': mean_warp_per_th.tolist(),
            'best_threshold': float(sweep_thresholds[warp_best_idx]),
            'best_error': float(mean_warp_per_th[warp_best_idx]),
        },
        'rand': {
            'mean_per_threshold': mean_rand_per_th.tolist(),
            'best_threshold': float(sweep_thresholds[rand_best_idx]),
            'best_error': float(mean_rand_per_th[rand_best_idx]),
        },
        'pixel': {
            'mean_per_threshold': mean_pixel_per_th.tolist(),
            'best_threshold': float(sweep_thresholds[pixel_best_idx]),
            'best_error': float(mean_pixel_per_th[pixel_best_idx]),
        },
    }

    # Keep PR metrics as before (IoU-based)
    thresholds_pr = np.linspace(0.01, 0.99, 99)
    benchmark_metrics = calculate_pr_metrics_at_thresholds(predictions, groundtruths, thresholds_pr)
    
    return {
        'per_image_metrics': per_image_metrics,
        'average_metrics': avg_metrics,                      # averages of best-per-image errors
        'global_metrics': global_metrics,
        'total_confusion_matrix': {
            'TP': int(total_tp),
            'FP': int(total_fp),
            'TN': int(total_tn),
            'FN': int(total_fn)
        },
        'benchmark_metrics': benchmark_metrics,             # IoU ODS/OIS as before
        'best_errors_over_thresholds': best_errors_over_thresholds,  # dataset-level bests for errors
        'total_images': num_images
    }

# Calculate ODS and OIS metrics for multiple thresholds
def calculate_pr_metrics_at_thresholds(predictions, groundtruths, thresholds):
    """
    Calculate ODS and OIS benchmark metrics for multiple thresholds based on IoU
    
    Args:
        predictions: List of prediction arrays, each (H, W) with values 0-1
        groundtruths: List of ground truth arrays, each (H, W) with values 0-1
        thresholds: List of thresholds to evaluate
        
    Returns:
        dict: Dictionary containing ODS and OIS metrics
    """
    # Check if we have data to process
    if not predictions or not groundtruths:
        print("Warning: No prediction or ground truth data for benchmark calculation")
        return {
            'ODS': 0.0,
            'ODS_threshold': thresholds[0] if thresholds else 0.5,
            'OIS': 0.0
        }
        
    num_thresholds = len(thresholds)
    iou_scores = np.zeros(num_thresholds)
    
    # For ODS: aggregate IoU across all images
    total_intersection = np.zeros(num_thresholds)
    total_union = np.zeros(num_thresholds)
    
    # For OIS: store best IoU for each image
    image_best_ious = []
    
    # For each image
    for pred, gt in zip(predictions, groundtruths):
        image_ious = np.zeros(num_thresholds)
        
        for i, threshold in enumerate(thresholds):
            # Calculate IoU at this threshold
            iou = calculate_iou(pred, gt, threshold)
            image_ious[i] = iou
            
            # Update for ODS calculation
            pred_binary = (pred > threshold).astype(np.uint8)
            gt_binary = (gt > 0.5).astype(np.uint8)
            
            intersection = np.sum((pred_binary == 1) & (gt_binary == 1))
            union = np.sum((pred_binary == 1) | (gt_binary == 1))
            
            total_intersection[i] += intersection
            total_union[i] += union
        
        # Find best IoU for this image (for OIS)
        best_iou = np.max(image_ious)
        image_best_ious.append(best_iou)
    
    # Calculate ODS IoU scores
    for i in range(num_thresholds):
        iou_scores[i] = total_intersection[i] / total_union[i] if total_union[i] > 0 else 0
    
    # Calculate OIS (average of best IoUs per image)
    ois = np.mean(image_best_ious)
    
    # Get ODS (optimal dataset scale - best IoU across all thresholds)
    ods_idx = np.argmax(iou_scores)
    ods_threshold = thresholds[ods_idx]
    ods = iou_scores[ods_idx]
    
    return {
        'ODS': ods,
        'ODS_threshold': ods_threshold,
        'OIS': ois
    }
    
    return {
        'ODS': ods,
        'ODS_threshold': ods_threshold,
        'OIS': ois
    }
