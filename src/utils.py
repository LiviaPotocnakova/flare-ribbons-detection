import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from metrics import dice_np, iou_np
import math
from math import sqrt


def plot_imgs(imgs, masks, predictions=None, n_imgs=10):
    """
    Plots images, masks, prediction masks and overlays side by side. If predictions aren't provided, masks are used as
    overlays.
    :param numpy.array imgs: array of images
    :param numpy.array masks: array of masks
    :param numpy.array predictions: array of predictions
    :param int n_imgs: number of images to plot
    :return matplotlib.pyplot: pyplot of images side by side
    """
    if predictions is None:
        fig, axes = plt.subplots(n_imgs, 3, figsize=(12, n_imgs * 4), squeeze=False)
        overlap = masks
        ii = 2
    else:
        fig, axes = plt.subplots(n_imgs, 4, figsize=(16, n_imgs * 4), squeeze=False)
        axes[0, 2].set_title("Prediction", fontsize=22)
        overlap = predictions
        ii = 3

    # Set titles
    axes[0, 0].set_title("Image", fontsize=22)
    axes[0, 1].set_title("Mask", fontsize=22)
    axes[0, ii].set_title("Overlay", fontsize=22)

    # Plot imgs
    for i in range(n_imgs):
        # masked = np.ma.masked_where(overlap[i] == 0, overlap[i])
        zero = np.zeros((overlap[i].shape[0], overlap[i].shape[1]))
        one = overlap[i].reshape((overlap[i].shape[0], overlap[i].shape[1]))
        masked = np.stack((one, zero, zero, one), axis=-1)
        # Show imgs
        axes[i, 0].imshow(imgs[i], cmap="gray", interpolation=None)
        axes[i, 1].imshow(masks[i], cmap="gray", interpolation=None)
        axes[i, ii].imshow(imgs[i], cmap="gray", interpolation=None)
        axes[i, ii].imshow(masked, cmap="jet", alpha=0.5)

        if predictions is not None:
            axes[i, 2].imshow(predictions[i], cmap="gray", interpolation=None)
            # Show metrics - dice, iou
            dice = np.round(dice_np(y_true=masks[i], y_pred=predictions[i]), 4)
            iou = np.round(iou_np(y_true=masks[i], y_pred=predictions[i]), 4)
            axes[i, 3].text(
                0.1,
                0.9,
                f"Dice: {dice}\nIoU: {iou}",
                fontsize=15,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.5),
            )
            axes[i, 3].set_axis_off()
        # Hide axis
        axes[i, 0].set_axis_off()
        axes[i, 1].set_axis_off()
        axes[i, 2].set_axis_off()
    return plt


def plot_metrics(model):
    """
    Plots training history of Keras model.
    :param model: trained model
    :return matplotlib.pyplot: pyplot of training history
    """
    plt.style.use("ggplot")
    fig, axs = plt.subplots(1, 2, figsize=(18, 5))
    # Plot metrics
    axs[0].plot(model.history["iou"])
    axs[0].plot(model.history["dice"])
    axs[0].plot(model.history["val_iou"])
    axs[0].plot(model.history["val_dice"])
    axs[1].plot(model.history["loss"])
    axs[1].plot(model.history["val_loss"])
    # Set titles
    axs[1].set_title("Loss over epochs", fontsize=20)
    axs[1].set_ylabel("loss", fontsize=20)
    axs[1].set_xlabel("epochs", fontsize=20)
    axs[0].set_title("Metrics over epochs", fontsize=20)
    axs[0].set_ylabel("metrics", fontsize=20)
    axs[0].set_xlabel("epochs", fontsize=20)
    # Set legend
    axs[1].legend(["loss", "val_loss"], loc="center right", fontsize=15)
    axs[0].legend(
        ["iou", "dice", "val_iou", "val_dice"], loc="center right", fontsize=15
    )
    return plt


def plot_top(imgs, y_true, y_pred, best=True, n_imgs=10):
    """
    Plots best|worst images, masks, prediction masks according to dice coefficient.
    :param numpy.array imgs: array of images
    :param numpy.array y_true: array of masks
    :param numpy.array y_pred: array of predictions
    :param bool best: whether to plot best or worst results
    :param int n_imgs: number of images to plot
    :return matplotlib.pyplot: pyplot of images side by side
    """
    dice_list = []
    for y_t, y_p in zip(y_true, y_pred):
        dice_coef = round(dice_np(y_t, y_p), 4)
        dice_list.append(dice_coef)
    dice_list = np.array(dice_list)
    # Sort list by dice_coef
    idx = dice_list.argsort()
    imgs = imgs[idx]
    y_true = y_true[idx]
    y_pred = y_pred[idx]

    if best:
        return plot_imgs(imgs[-n_imgs:], y_true[-n_imgs:], y_pred[-n_imgs:])
    else:
        return plot_imgs(imgs[:n_imgs], y_true[:n_imgs], y_pred[:n_imgs])


def create_contours(y_pred, target_size=(4096, 4096)):
    """
    Create contours coordinates from binary mask.
    :param numpy.array y_pred: array of binary mask
    :param target_size: size of image we will draw these coordinates
    :return list: list containing list of tuples with x and y coordinates [[(x,y), (x,y)]]
    """
    # (w, h)
    if isinstance(y_pred, Image.Image):
        mask = np.array(y_pred)
    elif isinstance(y_pred, np.ndarray):
        one = y_pred.reshape((y_pred.shape[0], y_pred.shape[1])) * 255
        one = np.array(one, dtype=np.uint8)
        mask = np.stack((one, one, one), axis=-1)
        mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
    else:
        raise TypeError(
            f"Expected class: {np.ndarray} or {Image.Image} but got {type(y_pred)}"
        )

    width = mask.shape[1]
    height = mask.shape[0]

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    polygons = []
    for object in contours:
        coords = []
        for i, point in enumerate(object):
            new_x = (int(point[0][0]) - 0) / (width - 0) * (target_size[0] - 0) - 0
            new_y = (int(point[0][1]) - 0) / (height - 0) * (target_size[1] - 0) - 0
            coords.append((new_x, new_y))
            # To make sure that polygon is fully connected
            if i == len(object) - 1:
                coords.append(coords[0])
        polygons.append(coords)
    return polygons
  
# function for finding line's slope
def getSlope(x1, y1, x2, y2):
    if (x2 - x1) == 0:
        return 0
    else: 
        return (y2 - y1) / (x2 - x1)
      
# function for checking whether the lines are intersecting
def intersects(x1, y1, x2, y2, x3, y3, x4, y4):
    dx1, dy1 = x2 - x1, y2 - y1
    dx2, dy2 = x4 - x3, y4 - y3
    delta = dx1 * dy2 - dy1 * dx2
    if delta == 0:
        return False
    s = (dx2 * (y1 - y3) - dy2 * (x1 - x3)) / delta
    t = (dx1 * (y1 - y3) - dy1 * (x1 - x3)) / delta
    
    return 0 <= s <= 1 and 0 <= t <= 1
 
# function for finding line's length
def getLength(x1, y1, x2, y2):
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
  
# function for finding line's angle
def getAngle(x1, y1, x2, y2):
    return abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)    
  
# function for getting a perpendicular distance of a line and a point
# line is set by its point coordinates - (x1, y1), (x2, y2)
# a point is set by coordinates (px, py)
def perpendicularDistance(x1,y1,x2,y2, px, py):
    AB = [None, None]
    AE = [None, None]
    
    AB[0] = x2 - x1
    AB[1] = y2 - y1
    AE[0] = px - x1
    AE[1] = py - y1
    
    dx1 = AB[0]
    dy1 = AB[1]
    dx2 = AE[0]
    dy2 = AE[1]
    mod = sqrt(dx1 * dx1 + dy1 * dy1)
    
    return abs(dx1 * dy2 - dy1 * dx2) / mod

# function for getting a shortest distance of a line and a point
# line is set by its point coordinates - (x1, y1), (x2, y2)
# a point is set by coordinates (px, py)
def minDistance(x1, y1, x2, y2, px, py) :
 
    # vector AB
    AB = [None, None]
    AB[0] = x2 - x1
    AB[1] = y2 - y1
 
    # vector BP
    BE = [None, None]
    BE[0] = px - x2
    BE[1] = py - y2
 
    # vector AP
    AE = [None, None]
    AE[0] = px - x1
    AE[1] = py - y1
 
    # Variables to store dot product
 
    # Calculating the dot product
    AB_BE = AB[0] * BE[0] + AB[1] * BE[1]
    AB_AE = AB[0] * AE[0] + AB[1] * AE[1]
 
    # Minimum distance from
    # point E to the line segment
    reqAns = 0
 
    # Case 1
    if (AB_BE > 0) :
 
        # Finding the magnitude
        y = py - y2
        x = px - x2
        reqAns = sqrt(x * x + y * y)
 
    # Case 2
    elif (AB_AE < 0) :
        y = py - y1
        x = px - x1
        reqAns = sqrt(x * x + y * y)
 
    # Case 3
    else:
        reqAns = perpendicularDistance(x1, y1, x2, y2, px, py)
     
    return reqAns
  
  # function for getting cleaner results from Hough transform
# lines from Hough transform are compared and combined if possible
def cleanLines(lines):
    
    new_lines = None
    for line in lines:
        x1, y1, x2, y2 = line[0]

        # get slope and angle of the first line
        slope = getSlope(x1, y1, x2, y2)
        angle = getAngle(x1, y1, x2, y2)

        for line2 in lines:
            # in case we are comparing the same line, skip it
            if np.array_equal(line, line2):
                continue

            x3, y3, x4, y4 = line2[0]
            # get slope of the second line
            slope2 = getSlope(x3, y3, x4, y4)

            # if the slopes are too different, skip them
            if abs(slope-slope2) >= 1 or (slope > 0 and slope2 < 0 or slope < 0 and slope2 > 0):
                continue

            # get the smallest distance of two lines 
            point_distance = min(minDistance(x1,y1,x2,y2,x3,y3), minDistance(x1,y1,x2,y2,x4,y4), minDistance(x3,y3,x4,y4,x1,y1), minDistance(x3,y3,x4,y4,x2,y2))
            
            # if the lines are too far from each other, skip them
            if point_distance > 5:
                continue
            
            if angle > 45 and angle < 135:
            # Lines are mostly vertical
                coords = {
                    y1: x1,
                    y2: x2,
                    y3: x3,
                    y4: x4}

                # create coordinates of the new line, that should be the logest possible combination
                # get smallest y and corresponding x
                new_y1 = min(y1,y2,y3,y4)
                new_x1 = coords[new_y1]
                # get biggest y and corresponding x
                new_y2 = max(y1,y2,y3,y4)
                new_x2 = coords[new_y2]

            elif (angle <= 45 or angle >= 135):
            # Lines are mostly horizontal
                coords = {
                    x1: y1,
                    x2: y2,
                    x3: y3,
                    x4: y4}
                # create coordinates of the new line, that should be the logest possible combination
                # get smallest x and corresponding y
                new_x1 = min(x1,x2,x3,x4)
                new_y1 = coords[new_x1]
                # get biggest x and corresponding y
                new_x2 = max(x1,x2,x3,x4)
                new_y2 = coords[new_x2]

            # create new line out of new coords
            new_line = [[[new_x1, new_y1, new_x2, new_y2]]]
            
            # remove old lines from the list of lines are append a new one
            indices = np.where(np.all(lines == line, axis=2))[0]
            new_lines = np.delete(lines, indices, axis=0)
            indices = np.where(np.all(new_lines == line2, axis=2))[0]
            new_lines = np.delete(new_lines, indices, axis=0)
            new_lines = np.append(new_lines, new_line, axis=0)
            lines = new_lines
            break

        # if a new line has been created, call the function recursively
        if new_lines is not None:
            return cleanLines(new_lines)
    
    return lines
