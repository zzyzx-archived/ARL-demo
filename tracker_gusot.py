import cv2
import numpy as np
from collections import deque
import matlab.engine

from pyutils.GOT_ import *
from pyutils.motion_estimation import GMC_total_all
from pyutils.mrfBCD import MRF_BCD


class GotTrack:
    def __init__(self, eng):
        self.is_deterministic = True
        self.name = 'GOT'
        self.eng = eng

    def init(self, frame, init_bb, max_length=2000, add_self_gmc=False):
        img = cv2.imread(frame)
        gt_bbs = [init_bb]

        # global branch
        self.init_bb = init_bb
        self.final_pred = init_bb
        self.bb2ml = gt_bbs[0]
        self.flag_must_use_clf = 0

        # local branch
        self.flag_slim = gt_bbs[0][3] <= 70 or gt_bbs[0][2] <= 70
        self.parJointSaab, self.sel_feat_idx, total_feat, y_dft = init([frame], gt_bbs,
                                                                       numKernel=[5], sizeKernel=5, stride=1,
                                                                       energyTh=[1e-2, 1e-2])
        self.pos_init = total_feat[y_dft > 0.5][:, self.sel_feat_idx]
        self.neg_init = total_feat[y_dft <= 0.5][:, self.sel_feat_idx]
        self.clf = get_trained_clf(self.pos_init, self.neg_init)
        self.clf_init = self.clf

        # control signal
        self.flag_seg_adv = False
        self.flag_bad_clf = False
        self.flag_no_shape = False
        self.flag_strike = True
        self.flag_continue = False
        self.area_vs_init = 1
        self.q_isgood = deque()
        self.mrf_ratio = deque()
        self.seg_ious = list()
        self.min_area_ratio = 0.25
        self.max_area_ratio = 4
        self.prob_boxes = [gt_bbs[0]] * 5
        self.seg_boxes = [gt_bbs[0]] * 5

        # global motion compensation
        self.add_self_gmc = add_self_gmc
        if self.add_self_gmc:
            self.gmc_pred = init_bb
            self.motion_mag = 0
            self.flag_lv = False
            self.no_gmc_cnt = 0
            self.last_im_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if max(img.shape) > 1000:
                self.dif_ds_stride = int(3) if np.prod(init_bb[-2:]) > 6000 else int(2)
            else:
                self.dif_ds_stride = int(1)
            if self.dif_ds_stride > 1:
                self.last_im_gray = self.last_im_gray[::self.dif_ds_stride, ::self.dif_ds_stride]

        # initialize DCF tracker
        bs_bb_last, bs_bb_ori, bs_bb, _, _, _, _, _ = \
            self.eng.run_GUSOT(int(1), str(frame),
                               matlab.double(self.bb2ml.tolist()),
                               int(0), int(max_length),
                               str('./gusot/ss_wk_space'),
                               nargout=8)
        self.bs_bb_last = np.array(bs_bb_last).flatten() + np.array([-1, -1, 0, 0])
        self.bs_bb_ori = np.array(bs_bb_ori).flatten() + np.array([-1, -1, 0, 0])
        self.bs_bb = np.array(bs_bb).flatten() + np.array([-1, -1, 0, 0])

        self.cnt = 0

    def update(self, frame):
        self.cnt += 1
        img = cv2.imread(frame)

        # global branch
        bs_bb_last, bs_bb_ori, bs_bb, cf_sim_score, gmc_pred, gmc_sim_score, xy_pred, _ = \
            self.eng.run_GUSOT(int(self.cnt + 1), str(frame),
                               matlab.double(self.bb2ml.tolist()),
                               int(self.flag_must_use_clf), int(2000),
                               str('./gusot/ss_wk_space'),
                               nargout=8)
        self.bs_bb_last = np.array(bs_bb_last).flatten() + np.array([-1, -1, 0, 0])
        self.bs_bb_ori = np.array(bs_bb_ori).flatten() + np.array([-1, -1, 0, 0])
        self.bs_bb = np.array(bs_bb).flatten() + np.array([-1, -1, 0, 0])
        gmc_pred = np.array(gmc_pred).flatten()
        xy_pred = np.array(xy_pred).flatten()

        # local branch
        self.flag_must_use_clf = 0
        if not self.flag_slim:
            patch_draw_box = self.bs_bb_last
            pred, prob_n, bb_scaled, clf_new, clf_bb_scaled, self.bb2ml, clf_bbt, viz_data, debug_data, update_data = \
                update(img, patch_draw_box, self.parJointSaab, self.sel_feat_idx, self.clf,
                       self.bs_bb, self.pos_init, self.neg_init, self.flag_seg_adv, verbose=False)

            # noise suppression
            if not self.flag_bad_clf:
                current_prob = prob_n
                if self.cnt > 1:
                    if prob_n.shape[0] > self.template_prob.shape[0]:
                        current_prob = current_prob[10:70, 10:70]
                    dy, dx = circonv2D(current_prob, self.template_prob)
                    self.template_prob = np.roll(self.template_prob, (dy, dx), axis=(0, 1))
                    self.template_prob = reg_map(P=current_prob, Q=self.template_prob, lbda=5)
                    self.template_prob /= np.max(self.template_prob)
                    if prob_n.shape[0] == self.template_prob.shape[0]:
                        prob_n[self.template_prob < 0.5] = prob_n[self.template_prob < 0.5] * self.template_prob[
                            self.template_prob < 0.5]
                else:
                    self.template_prob = prob_n if prob_n.shape[0] < 80 else prob_n[10:70, 10:70]

            # parse / process output
            if len(clf_bbt) > 0:
                clf_bbt_frame = getFrameBox(clf_bbt, viz_data[3], viz_data[4], viz_data[0].shape[:2])
            patch = viz_data[0]
            bs_bb_ori_scaled = getScaledBox(viz_data[4], self.bs_bb_ori, viz_data[3], viz_data[0].shape[:2])
            bb2ml_ori = self.bb2ml.copy()
            self.bb2ml = clip_box_to_frame(self.bb2ml, img.shape[0], img.shape[1])
            ratio_now = max(bb2ml_ori[-2:]) / min(bb2ml_ori[-2:])
            ratio_prev = max(self.final_pred[-2:]) / min(self.final_pred[-2:])
            self.area_vs_init_prev = self.area_vs_init
            self.area_vs_init = np.prod(bb2ml_ori[-2:]) / np.prod(self.init_bb[-2:])
            clf_sz_ratio = np.prod(debug_data[1][-2:]) / np.prod(bb_scaled[-2:])
            tmp = [np.prod(self.final_pred[-2:]), np.prod(bb2ml_ori[-2:])]
            sz_change_ratio = max(tmp) / min(tmp)
            bb2ml_ori_ct = convert_bbox_format(bb2ml_ori, 'center')[:2]
            im_h, im_w = img.shape[:2]
            flag_near_border = bb2ml_ori_ct[0] < 0.15 * im_w or bb2ml_ori_ct[0] > 0.85 * im_w or \
                               bb2ml_ori_ct[1] < 0.15 * im_h or bb2ml_ori_ct[1] > 0.85 * im_h
            bs_scaled = getScaledBox(self.bs_bb_last, self.bs_bb_ori, viz_data[3], viz_data[0].shape[:2])
            if len(clf_bbt) > 0:
                score_bs = calc_rect_int(clf_bbt, bs_scaled)[0]
                score_clf = calc_rect_int(clf_bbt, clf_bb_scaled)[0]
            else:
                score_bs = getScoreInBox(prob_n, bs_scaled)
                score_clf = getScoreInBox(prob_n, clf_bb_scaled)

            # quality assess
            if np.prod(bb2ml_ori[-2:]) > np.prod(img.shape[:2]) or \
                    sz_change_ratio > 1.5 and flag_near_border and abs(ratio_now - ratio_prev) > 0.3 or \
                    self.area_vs_init < self.min_area_ratio and flag_near_border or \
                    self.area_vs_init > self.max_area_ratio and not clf_sz_ratio < 0.5 or \
                    abs(self.area_vs_init - self.area_vs_init_prev) > 1 and clf_sz_ratio > 1.5 or \
                    15 < self.cnt < 20 and abs(ratio_now - ratio_prev) > 0.5:
                self.flag_bad_clf = True
            if (self.cnt > 15 and self.flag_strike) and (clf_sz_ratio < 0.4 or clf_sz_ratio > 1.5):
                self.flag_bad_clf = True
            if self.cnt < 20 and (viz_data[6] > 2 or self.flag_strike and viz_data[6] > 1 and \
                                  (viz_data[5] < 0.8 or clf_sz_ratio < 0.5 or clf_sz_ratio > 1.3)):
                self.flag_bad_clf = True
            self.q_isgood.append(
                len(clf_bbt) < 1 or (len(clf_bbt) > 0 and np.prod(clf_bbt_frame[-2:]) > np.prod(img.shape[:2])))
            if len(self.q_isgood) > 5:
                self.q_isgood.popleft()
            if np.sum(self.q_isgood) >= 5:
                self.flag_bad_clf = True
            if sz_change_ratio > 5:
                self.flag_bad_clf = True

            # run global branch only for seqs that cannot do shape estimation
            if self.cnt <= 10:
                self.seg_ious.append([calc_rect_int(debug_data[0], debug_data[1])[0],
                                      calc_rect_int(debug_data[0], debug_data[2])[0],
                                      calc_rect_int(debug_data[1], debug_data[2])[0]])
            if self.cnt == 10:
                tmp = np.mean(self.seg_ious, axis=0)
                self.stats10 = [tmp[0], tmp[1], tmp[2], np.mean(tmp)]
                flag_continue = self.stats10[-1] > 0.6
                if not flag_continue or self.flag_slim:
                    self.flag_no_shape = True
                    self.flag_bad_clf = True
                if self.stats10[0] < self.stats10[1] and self.stats10[0] < self.stats10[2]:
                    self.flag_seg_adv = True

            # MRF fusion
            mrf_bb_scaled = []
            mrf_bb = []
            boxes = [bs_scaled, clf_bbt, viz_data[8]]
            tmp = [calc_rect_int(boxes[0], boxes[1])[0],
                   calc_rect_int(boxes[0], boxes[2])[0],
                   calc_rect_int(boxes[1], boxes[2])[0]]
            tmp = np.array(tmp)
            bs_bb_clip = clip_box_to_frame(self.bs_bb, img.shape[0], img.shape[1])
            inside_frame_ratio = calc_rect_int(self.bs_bb, bs_bb_clip)[0]

            if self.cnt > 10 and (cf_sim_score < 0.15 or cf_sim_score >= 0.15 and inside_frame_ratio < 0.9) and (
            not self.flag_bad_clf):
                flag_mrf = False
                if np.sum(tmp > 0.7) < 3:
                    flag_normal, fg_coords, bg_coords = get_coords_prior(patch.shape[:2], bs_bb_ori_scaled,
                                                                         fnum=100, bnum=400, prob=prob_n)
                    try:
                        bayes_seg, final_seg, gmms = MRF_BCD(patch, fg_coords, bg_coords,
                                                             n_compo=2, max_iter=20, lmbd=10, lmbd2=2,
                                                             gmms_ref=None,
                                                             prob=prob_n,
                                                             iter=1,
                                                             verbose=False)
                        max_blob, _, mainland_ratio = findmaxisland(final_seg, return_ratio=True)
                        self.mrf_ratio.append(mainland_ratio < 0.9)
                        if len(self.mrf_ratio) > 5:
                            self.mrf_ratio.popleft()
                        mrf_bb_scaled = seg2box(max_blob, [])
                        mrf_bb = getFrameBox(mrf_bb_scaled, viz_data[3], viz_data[4], viz_data[0].shape[:2])
                    except Exception as e:
                        pass

                    if len(mrf_bb) > 0 and 0.75 * np.prod(bb_scaled[-2:]) < np.prod(
                            mrf_bb_scaled[-2:]) < 1.25 * np.prod(bb_scaled[-2:]):
                        candidates = viz_data[7]
                        candidates.append(bs_scaled)
                        candidates.append(clf_bbt)
                        tmp = [calc_rect_int(mrf_bb_scaled, v)[0] for v in candidates]
                        bb2ml_scaled = candidates[np.argmax(tmp)]
                        self.bb2ml = getFrameBox(bb2ml_scaled, viz_data[3], viz_data[4], viz_data[0].shape[:2])
                        self.flag_must_use_clf = 1
                        flag_mrf = True
                        if clf_sz_ratio < 0.8 or clf_sz_ratio > 1.2:
                            clf = update_clf(update_data)

                if not flag_mrf:
                    bb2ml_scaled = clf_bb_scaled
                    if 0.5 < (ratio_now / ratio_prev) < 1.5 and \
                            not ((clf_sz_ratio < 0.1 or viz_data[5] < 0.1) and np.prod(clf_bb_scaled[-2:]) > np.prod(
                                bs_scaled[-2:])) and \
                            (score_clf > score_bs or \
                             self.flag_seg_adv and calc_rect_int(bs_scaled, clf_bb_scaled)[0] > 0.8):
                        self.flag_must_use_clf = 1
                        if clf_sz_ratio < 0.8 or clf_sz_ratio > 1.2:
                            self.clf = update_clf(update_data)

            # motion based checking
            if len(gmc_pred) > 0:
                bs_trans = np.sqrt((self.bs_bb[0] - xy_pred[0]) ** 2 + (self.bs_bb[1] - xy_pred[1]) ** 2) / np.sqrt(
                    img.shape[0] ** 2 + img.shape[1] ** 2)
                gmc_trans = np.sqrt((gmc_pred[0] - xy_pred[0]) ** 2 + (gmc_pred[1] - xy_pred[1]) ** 2) / np.sqrt(
                    img.shape[0] ** 2 + img.shape[1] ** 2)
            tmp = getScoreInBox(prob_n, bs_scaled)
            flag_loss_obj = tmp < 0.2 and len(clf_bbt) < 1 or tmp < 0.1
            if self.flag_bad_clf and len(gmc_pred) > 0 and flag_loss_obj:
                _, prob_n1, gmc_pred_scaled, _, gmc_pred_adj_scaled, gmc_pred_adj, _, viz_data1, debug_data1, update_data1 = \
                    update(img, gmc_pred, self.parJointSaab, self.sel_feat_idx, self.clf, gmc_pred, self.pos_init,
                           self.neg_init,
                           self.flag_seg_adv, verbose=False)
                gmc_pred_adj_ct = convert_bbox_format(gmc_pred_adj, 'center')[:2]
                gmc_pred_adj_ct = np.array([gmc_pred_adj_ct[0], gmc_pred_adj_ct[1], gmc_pred[2], gmc_pred[3]])
                gmc_pred_adj = convert_bbox_format(gmc_pred_adj_ct, 'topleft')
                gmc_pred_adj_scaled = getScaledBox(gmc_pred, gmc_pred_adj, viz_data1[3], viz_data1[0].shape[:2])
                tmp1 = getScoreInBox(prob_n1, gmc_pred_adj_scaled)

                loss_ct_bs = 0.2 * bs_trans - 2 * cf_sim_score - np.exp(
                    np.sign(tmp - 0.7) * np.sqrt(abs(tmp - 0.7))) * tmp
                loss_ct_gmc = 0.2 * gmc_trans - 2 * gmc_sim_score - np.exp(
                    np.sign(tmp - 0.7) * np.sqrt(abs(tmp - 0.7))) * tmp1

                if loss_ct_gmc < loss_ct_bs:
                    self.bb2ml = np.array(gmc_pred_adj)
                    self.flag_must_use_clf = 1

            # resume shape estimation
            self.prob_boxes[self.cnt % 5] = clf_bbt_frame if len(clf_bbt) > 0 else [0, 0, 0, 0]
            self.seg_boxes[self.cnt % 5] = self.bb2ml
            if self.cnt > 20 and not self.flag_no_shape and self.flag_bad_clf and len(clf_bbt) > 0:
                tmp = [v[-2] for v in self.prob_boxes]
                w1_std = np.std(tmp, ddof=1) / np.mean(tmp)
                tmp = [v[-1] for v in self.prob_boxes]
                h1_std = np.std(tmp, ddof=1) / np.mean(tmp)
                tmp = [v[-2] for v in self.seg_boxes]
                w2_std = np.std(tmp, ddof=1) / np.mean(tmp)
                tmp = [v[-1] for v in self.seg_boxes]
                h2_std = np.std(tmp, ddof=1) / np.mean(tmp)
                if cf_sim_score > 0.09 and \
                        not (clf_sz_ratio < 0.2 or clf_sz_ratio > 2 and calc_rect_int(clf_bbt, viz_data[8])[0] < 0.9):
                    if w2_std < 0.08 and h2_std < 0.08 and \
                            (self.seg_boxes[-1][2] < 0.95 * img.shape[1] and self.seg_boxes[-1][3] < 0.95 * img.shape[
                                0]):
                        self.flag_must_use_clf = 1
                        self.flag_bad_clf = False
                    elif w1_std < 0.08 and h1_std < 0.08:
                        self.flag_must_use_clf = 1
                        self.flag_bad_clf = False
                        self.bb2ml = clf_bbt_frame
                    if self.flag_must_use_clf:
                        if self.area_vs_init < self.min_area_ratio:
                            self.min_area_ratio /= 2
                        if self.area_vs_init > self.max_area_ratio:
                            self.max_area_ratio *= 2

            # quality assess
            if self.cnt > 10 and self.flag_strike and cf_sim_score < 0.15:
                self.flag_strike = False

            if not self.flag_must_use_clf:
                bs_bb_scaled = getScaledBox(viz_data[4], self.bs_bb, viz_data[3], viz_data[0].shape[:2])
                if calc_rect_int(bs_bb_scaled, [0, 0, viz_data[0].shape[0], viz_data[0].shape[1]])[0] < 0.01:
                    self.flag_bad_clf = True

        self.final_pred = self.bb2ml if self.flag_must_use_clf else self.bs_bb
        # terminate clf if very poor
        if not self.flag_slim:
            if self.cnt < 10 and self.flag_bad_clf:
                self.flag_slim = True
                # return False,[]
            if self.flag_strike and self.final_pred[-1] < 70 and self.final_pred[-2] < 70:
                self.flag_slim = True
            if 10 < self.cnt < 100 and self.flag_bad_clf and self.area_vs_init - self.area_vs_init_prev > 2 and \
                    self.stats10[-1] < 0.75:
                self.flag_slim = True
            if self.cnt > 0 and self.flag_strike and clf_sz_ratio > 2:
                self.flag_slim = True
            if np.sum(self.mrf_ratio) == 5 and self.stats10[-1] < 0.75:
                self.flag_slim = True
            if self.cnt <= 20 and self.flag_bad_clf:
                self.flag_no_shape = True
        return np.array(self.final_pred)

    def track(self, frames, init_bb):
        res = [init_bb]
        self.init(frames[0], init_bb)
        for idx, frame in enumerate(frames[1:]):
            res.append(self.update(frame))
        return np.array(res)

    def get_motion(self, img):
        if self.cnt == 0:
            return list()
        if self.flag_lv and self.no_gmc_cnt > 0:
            if self.no_gmc_cnt == 10:
                self.flag_lv = False
                self.no_gmc_cnt = 0
            else:
                self.no_gmc_cnt += 1
                return list()

        curr_im_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if self.dif_ds_stride > 1:
            curr_im_gray = curr_im_gray[::self.dif_ds_stride, ::self.dif_ds_stride]

        hit_obj_radius = int(-1)
        th = 0.1
        if np.prod(self.final_pred[-2:]) < 1000 and self.motion_mag > 1:
            hit_obj_radius = int(min([max([max(self.final_pred[-2:]), 30]), 50]))
            th = 0
        gmc_success, dif_int, gmc_pos_ori, dif_map = GMC_total_all(
            self.last_im_gray,
            curr_im_gray,
            (int(self.final_pred[3] // self.dif_ds_stride), int(self.final_pred[2] // self.dif_ds_stride)),
            1,
            0.2,
            th,
            self.gmc_pred[1] // self.dif_ds_stride,
            self.gmc_pred[0] // self.dif_ds_stride,
            hit_obj_radius)[:4]

        if not gmc_success or len(gmc_pos_ori) < 1:
            self.flag_lv = True
            self.no_gmc_cnt = 1
            return list()
        gmc_pos_ori = np.array(gmc_pos_ori[0])
        gmc_pos = self.dif_ds_stride * gmc_pos_ori
        self.gmc_pred = [gmc_pos[1], gmc_pos[0], self.final_pred[2], self.final_pred[3]]
        self.motion_mag = dif_int[int(gmc_pos_ori[0])][int(gmc_pos_ori[1])]
        self.last_im_gray = curr_im_gray

        return self.gmc_pred
