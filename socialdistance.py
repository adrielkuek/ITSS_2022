"""
Author: Adriel Kuek
Date Created: 22 Feb 2022
Version: 0.1
Email: adrielkuek@gmail.com
Status: Devlopment

Description:
socialdistance takes in all person bounding boxes from the output from a tracker, and iteratively computes a euclidean distance proximity score value 
based on the centroid. It is calibrated against a heuristic approach for real-world distance to pixel mapping using an average human height of 170cm.
Additional, to work around the internal camera configurations which may not be made available to us, we implement a special depth_score calibration value
to penalise distance separation of objects that appear further in depth. As we are working on a 2D image plane with no 3D information, we would require these
additional heuristic assupmtions to ensure a robust model framework.

The framework considers 2 factors before making a decision for violation alert:
1. The minimum distance separation institutionalised based on prevailing SMM restrictions (default: 1m)
2. The duration at which the 2 close contacts maintain this separation, which constitutes to higher risks of infection.

"""

import numpy as np
import math
import cv2
import time, datetime

# Global Dictionaries
# socialdistance_dict = {}
# printed_tracks = []
# track_pair_list = []
# count = 0
# filter inaccurate yolo bboxs
# seen_tracks = {}

#########################################################################
# Configuration Values
#########################################################################
calibration_value1 = 150
calibration_value2 = 0.4

# dist_thres = 1.0
# violation_thres = 3
# DepthControlFactor = 0.3

class socialdistance(object):
    def __init__(self, dist_thres, violation_thres, DepthControlFactor, humanheight, socialdistance_dict, seen_tracks, count, track_pair_list, printed_tracks):

        self.dist_thres = dist_thres
        self.violation_thres = violation_thres
        self.DepthControlFactor = DepthControlFactor
        self.humanHT = humanheight
        self.socialdistance_dict = socialdistance_dict
        self.seen_tracks = seen_tracks
        self.count = count
        self.track_pair_list = track_pair_list
        self.printed_tracks = printed_tracks

    def calibrated_dist(self, p1, p2):
        global calibration_value1
        return ((p1[0] - p2[0]) ** 2 + calibration_value1 / ((p1[1] + p2[1]) / 2) * (p1[1] - p2[1]) ** 2) ** calibration_value2


    def proximity_score(self, p1, p2, target_height_list):
        global dist_thres

        # Adriel Info - Addition of Depth Control Factor to control FAs arising from targets along the depth planar axis
        delta_x = abs(p1[0] - p2[0])
        delta_y = abs(p1[1] - p2[1])
        if delta_x and delta_y != 0:
            angle_deg = (math.atan2(delta_y, delta_x)) * (180/math.pi)
        else:
            angle_deg = 0

        avg_height = round(np.mean(target_height_list))
        # score = calibrated_dist(p1, p2)
        # Calculate euclid dist
        score = np.sqrt(np.square(p1[0] - p2[0]) + np.square(p1[1] - p2[1]))
        # calib = (p1[1] + p2[1]) / 2

        # Apply depth control constant when targets are placed in planar depth
        if angle_deg > 30:

            # Assuming average height of human is approx. 1.7m
            # if 0 < score < 0.15 * calib:
            if 0 < score < ((avg_height/self.humanHT) * self.dist_thres * self.DepthControlFactor):
                # print(f'Depth Control Factor Applied!')
                # Danger Close Pairs
                return 1
            else:
                return 0
        else:
            # Assuming average height of human is approx. 1.7m
            # if 0 < score < 0.15 * calib:
            if 0 < score < ((avg_height/self.humanHT) * self.dist_thres):
                # Danger Close Pairs
                return 1
            else:
                return 0

    def euclid_dist(self, emb1, emb2):
        a, b = np.asarray(emb1), np.asarray(emb2)
        if len(a) == 0 or len(b) == 0:
            return np.zeros((len(a), len(b)))
        c = a - b
        c_sq = np.square(c)
        dists = np.sum(c_sq)
        return dists

    def proximity_evaluation(self, trackid_bbox_centroid, canvas, cam):
        # global track_pair_list, count
        avg_height = []

        # if USE_ROI:
        #     roi = ROIS[str(cam)]
        #     roi = roi[0]

        # if USE_ROI:
        #     # Draw Polygon lines on canvas
        #     cv2.polylines(canvas, np.int32([roi]), True, (0,255,255))

        for outer_idx in range(len(trackid_bbox_centroid)):
            bbox_to_draw_outeridx = (np.asarray(trackid_bbox_centroid[outer_idx][2])).astype(int)
            cv2.putText(canvas,str(trackid_bbox_centroid[outer_idx][0]),(bbox_to_draw_outeridx[0],bbox_to_draw_outeridx[1]+bbox_to_draw_outeridx[3]+20), 0, 400*1e-3*2, [255,255,255], 2)

            for inner_idx in range(len(trackid_bbox_centroid)):
                bbox_to_draw_inneridx = (np.asarray(trackid_bbox_centroid[inner_idx][2])).astype(int)
                cv2.putText(canvas,str(trackid_bbox_centroid[inner_idx][0]),(bbox_to_draw_inneridx[0],bbox_to_draw_inneridx[1]+bbox_to_draw_inneridx[3]+20), 0, 400*1e-3*2, [255,255,255], 2)

                avg_height = [trackid_bbox_centroid[outer_idx][4], trackid_bbox_centroid[inner_idx][4]]
                p_score = self.proximity_score(trackid_bbox_centroid[outer_idx][3], trackid_bbox_centroid[inner_idx][3], avg_height)
                
                if p_score == 1:
                    track_pair_meta_1 = "{}_{}".format(trackid_bbox_centroid[outer_idx][0], trackid_bbox_centroid[inner_idx][0])

                    start_time = time.time()

                    if any(key.startswith(track_pair_meta_1) for key in list(self.socialdistance_dict)):
                        # track pairs already inside socialdistance dictionary
                        try:
                            initial_time, violation_time = self.socialdistance_dict[track_pair_meta_1]
                            if violation_time - initial_time < self.violation_thres:
                                curr_time = time.time()
                                self.socialdistance_dict[track_pair_meta_1] = [initial_time, curr_time]
                            else:
                                # if USE_ROI:
                                # Check if targets are within polygon ROI
                                # pt = (int(bbox_to_draw_outeridx[0] + bbox_to_draw_outeridx[2]/2.0), int(bbox_to_draw_outeridx[1] + bbox_to_draw_outeridx[3]))
                                
                                # in_roi = cv2.pointPolygonTest(np.int32([roi]), (pt[0],pt[1]), False)

                                # if in_roi >= 0:
                                # Draw red bounding box for violations
                                cv2.rectangle(canvas,(bbox_to_draw_outeridx[0],bbox_to_draw_outeridx[1]),(bbox_to_draw_outeridx[0]+bbox_to_draw_outeridx[2],bbox_to_draw_outeridx[1]+bbox_to_draw_outeridx[3]), [0, 0, 255], 2)
                                cv2.rectangle(canvas,(bbox_to_draw_inneridx[0],bbox_to_draw_inneridx[1]),(bbox_to_draw_inneridx[0]+bbox_to_draw_inneridx[2],bbox_to_draw_inneridx[1]+bbox_to_draw_inneridx[3]), [0, 0, 255], 2)
                                # Draw blue line connection between 2 centroids
                                cv2.line(canvas, tuple(trackid_bbox_centroid[outer_idx][3]), tuple(trackid_bbox_centroid[inner_idx][3]), (255, 0, 0), 2)

                                # Remove duplicates
                                self.track_pair_list.append([int(trackid_bbox_centroid[outer_idx][0]), int(trackid_bbox_centroid[inner_idx][0])])
                                data = {tuple(item) for item in map(sorted, self.track_pair_list)}
                                self.track_pair_list = list(data)

                                track_key = "{}_{}".format(cam, track_pair_meta_1)

                                track_pair = (int(trackid_bbox_centroid[outer_idx][0]), int(trackid_bbox_centroid[inner_idx][0]))

                                if track_key in self.seen_tracks:
                                    self.seen_tracks[track_key] += 1
                                else:
                                    # First time seeing the track - initialise
                                    self.seen_tracks[track_key] = 1

                                data_set = set(self.track_pair_list)
                                # Track pair must appear consecutively more than 3 hits then count
                                if self.seen_tracks[track_key] > 3 and track_pair in data_set:
                                    if track_key not in self.printed_tracks:
                                        date_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                                        self.count += 1
                                # else:
                                #     # Draw red bounding box for violations
                                #     cv2.rectangle(canvas,(bbox_to_draw_outeridx[0],bbox_to_draw_outeridx[1]),(bbox_to_draw_outeridx[0]+bbox_to_draw_outeridx[2],bbox_to_draw_outeridx[1]+bbox_to_draw_outeridx[3]), [255, 0, 0], 2)
                                #     cv2.rectangle(canvas,(bbox_to_draw_inneridx[0],bbox_to_draw_inneridx[1]),(bbox_to_draw_inneridx[0]+bbox_to_draw_inneridx[2],bbox_to_draw_inneridx[1]+bbox_to_draw_inneridx[3]), [255, 0, 0], 2)
                                #     # Draw red line connection between 2 centroids
                                #     cv2.line(canvas, tuple(trackid_bbox_centroid[outer_idx][3]), tuple(trackid_bbox_centroid[inner_idx][3]), (255, 0, 0), 2)

                                #     # Remove duplicates
                                #     track_pair_list.append([int(trackid_bbox_centroid[outer_idx][0]), int(trackid_bbox_centroid[inner_idx][0])])
                                #     data = {tuple(item) for item in map(sorted, track_pair_list)}
                                #     track_pair_list = list(data)

                                #     # Save Screen Capture and Logfile workflow
                                #     if not USE_OFFLINE_VIDEO:
                                #         track_key = "{}_{}".format(RTSP_POLCAM_ID[cam], track_pair_meta_1)
                                #     else:
                                #         track_key = "{}_{}".format(cam, track_pair_meta_1)

                                #     track_pair = (int(trackid_bbox_centroid[outer_idx][0]), int(trackid_bbox_centroid[inner_idx][0]))

                                #     if track_key in seen_tracks:
                                #         seen_tracks[track_key] += 1
                                #     else:
                                #         # First time seeing the track - initialise
                                #         seen_tracks[track_key] = 1

                                #     data_set = set(track_pair_list)
                                #     # Track pair must appear consecutively more than 3 hits then count
                                #     if seen_tracks[track_key] > 3 and track_pair in data_set:
                                #         if track_key not in printed_tracks:
                                #             date_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                                #             count += 1
                                #             save_screenshot(cam, canvas, track_key, date_timestamp, count)
                                #             save_log(cam, trackid_bbox_centroid[outer_idx][0], trackid_bbox_centroid[inner_idx][0], violation_time, bbox_to_draw_outeridx)
                        except KeyError:
                            pass
                    else:
                        # Initialising violation times
                        self.socialdistance_dict[track_pair_meta_1] = [start_time, start_time] 

        return canvas

def main():

    print(f'insert main test code here')


if __name__ == "__main__":
    main()