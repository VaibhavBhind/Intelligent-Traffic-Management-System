# TechVidvan Vehicle-Tracker

import math

class EuclideanDistTracker:
    def __init__(self):
        # Store the center positions of the objects
        self.center_points = {}
        self.expected_point= {}
        self.cnt = {}
        # Keep the count of the IDs
        # each time a new object id detected, the count will increase by one
        self.id_count = 0


    def update(self, objects_rect):
        # Objects boxes and ids
        objects_bbs_ids = []
        new_center_point = {}
        # Get center point of new object
        for rect in objects_rect:
            x, y, w, h, index = rect
            cx = (x + x + w) // 2
            cy = (y + y + h) // 2

            # Find out if that object was detected already
            same_object_detected = False
            for id, pt in self.center_points.items():
                dist = math.hypot(cx - pt[0], cy - pt[1])

                if dist < 40:
                    del self.center_points[id]
                    new_center_point[id]=(cx,cy)
                    # print(self.center_points)
                    objects_bbs_ids.append([x, y, w, h, id, index])
                    same_object_detected = True
                    break
            if(same_object_detected==True):
                continue

            for id, pt in self.expected_point.items():
                dist = math.hypot(cx - pt[0], cy - pt[1])

                if dist < 40:
                    del self.expected_point[id]
                    del self.cnt[id]
                    new_center_point[id]=(cx,cy)
                    # print(self.center_points)
                    objects_bbs_ids.append([x, y, w, h, id, index])
                    same_object_detected = True
                    break

            # New object is detected we assign the ID to that object
            if same_object_detected is False:
                new_center_point[self.id_count] = (cx, cy)
                objects_bbs_ids.append([x, y, w, h, self.id_count, index])
                self.id_count += 1

        expected_points_nw={}
        cnt_nw={}
        for id,pt in self.expected_point.items():
            if self.cnt[id]<3:
                expected_points_nw[id]=(pt[0],pt[0]+1)
                cnt_nw[id]=self.cnt[id]+1
        for id,pt in self.center_points.items():
            expected_points_nw[id]=(pt[0],pt[1]+1)
            cnt_nw[id]=1
        self.expected_point = expected_points_nw.copy()
        self.cnt = cnt_nw.copy()
        # Update dictionary with IDs not used removed
        self.center_points = new_center_point.copy()
        return objects_bbs_ids



def ad(a, b):
    return a+b