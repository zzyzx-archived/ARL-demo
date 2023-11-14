import numpy as np

class GMM:
    def __init__(self, lr=0.009, max_num_samp=30):
        self.num_samples = 0
        self.samples = list()
        self.centroids = list()
        self.lr = lr
        self.max_num_samp = max_num_samp
        self._distance_matrix = np.ones((self.max_num_samp, self.max_num_samp), dtype=np.float32) * np.inf
        self._gram_matrix = np.ones((self.max_num_samp, self.max_num_samp), dtype=np.float32) * np.inf
        self.prior_weights = np.zeros((self.max_num_samp, 1), dtype=np.float32)
        # find the minimum allowed sample weight. samples are discarded if their weights become lower
        self.minimum_sample_weight = self.lr * (1 - self.lr) ** (2 * self.max_num_samp)

    def _find_gram_vector(self, new_sample):
        gram_vector = np.inf * np.ones((self.max_num_samp))
        if self.num_samples > 0:
            #print(np.array(self.centroids).shape, new_sample.shape)
            if len(np.array(self.centroids).shape)<2:
                print([np.array(v).shape for v in self.centroids])
            ip = np.array(self.centroids) @ new_sample
            gram_vector[:self.num_samples] = ip.ravel()
        return gram_vector

    def _merge_samples(self, sample1, sample2, w1, w2, sample_merge_type):
        alpha1 = w1 / (w1 + w2)
        alpha2 = 1 - alpha1
        if sample_merge_type == 'replace':
            merged_sample = sample1
        elif sample_merge_type == 'merge':
            merged_sample = alpha1 * sample1 + alpha2 * sample2
        return merged_sample

    def _update_distance_matrix(self, gram_vector, new_sample_norm, id1, id2, w1, w2):
        """
            update the distance matrix
        """
        alpha1 = w1 / (w1 + w2)
        alpha2 = 1 - alpha1
        if id2 < 0:
            norm_id1 = self._gram_matrix[id1, id1]

            # udpate the gram matrix
            if alpha1 == 0:
                self._gram_matrix[:, id1] = gram_vector
                self._gram_matrix[id1, :] = self._gram_matrix[:, id1]
                self._gram_matrix[id1, id1] = new_sample_norm
            elif alpha2 == 0:
                # new sample is discard
                pass
            else:
                # new sample is merge with an existing sample
                self._gram_matrix[:, id1] = alpha1 * self._gram_matrix[:, id1] + alpha2 * gram_vector
                self._gram_matrix[id1, :] = self._gram_matrix[:, id1]
                self._gram_matrix[id1, id1] = alpha1 ** 2 * norm_id1 + alpha2 ** 2 * new_sample_norm + 2 * alpha1 * alpha2 * gram_vector[id1]

            # udpate distance matrix
            self._distance_matrix[:, id1] = np.maximum(self._gram_matrix[id1, id1] + np.diag(self._gram_matrix) - 2 * self._gram_matrix[:, id1], 0)
            # self._distance_matrix[:, id1][np.isnan(self._distance_matrix[:, id1])] = 0
            self._distance_matrix[id1, :] = self._distance_matrix[:, id1]
            self._distance_matrix[id1, id1] = np.inf
        else:
            if alpha1 == 0 or alpha2 == 0:
                raise("Error!")

            norm_id1 = self._gram_matrix[id1, id1]
            norm_id2 = self._gram_matrix[id2, id2]
            ip_id1_id2 = self._gram_matrix[id1, id2]

            # handle the merge of existing samples
            self._gram_matrix[:, id1] = alpha1 * self._gram_matrix[:, id1] + alpha2 * self._gram_matrix[:, id2]
            # self._distance_matrix[:, id1][np.isnan(self._distance_matrix[:, id1])] = 0
            self._gram_matrix[id1, :] = self._gram_matrix[:, id1]
            self._gram_matrix[id1, id1] = alpha1 ** 2 * norm_id1 + alpha2 ** 2 * norm_id2 + 2 * alpha1 * alpha2 * ip_id1_id2
            gram_vector[id1] = alpha1 * gram_vector[id1] + alpha2 * gram_vector[id2]

            # handle the new sample
            self._gram_matrix[:, id2] = gram_vector
            self._gram_matrix[id2, :] = self._gram_matrix[:, id2]
            self._gram_matrix[id2, id2] = new_sample_norm

            # update the distance matrix
            self._distance_matrix[:, id1] = np.maximum(self._gram_matrix[id1, id1] + np.diag(self._gram_matrix) - 2 * self._gram_matrix[:, id1], 0)
            self._distance_matrix[id1, :] = self._distance_matrix[:, id1]
            self._distance_matrix[id1, id1] = np.inf
            self._distance_matrix[:, id2] = np.maximum(self._gram_matrix[id2, id2] + np.diag(self._gram_matrix) - 2 * self._gram_matrix[:, id2], 0)
            self._distance_matrix[id2, :] = self._distance_matrix[:, id2]
            self._distance_matrix[id2, id2] = np.inf

    def update_sample_space_model(self, new_train_sample, img=None, verbose=False):
        # find the inner product of the new sample with existing samples
        gram_vector = self._find_gram_vector(new_train_sample)

        # find the inner product of the new sample with existing samples
        new_train_sample_norm = np.vdot(new_train_sample, new_train_sample)

        dist_vector = np.maximum(new_train_sample_norm + np.diag(self._gram_matrix) - 2 * gram_vector, 0)
        dist_vector[self.num_samples:] = np.inf

        merged_sample = []
        new_sample = []
        merged_sample_id = -1
        new_sample_id = -1

        if self.num_samples == self.max_num_samp:
            min_sample_id = np.argmin(self.prior_weights)
            min_sample_weight = self.prior_weights[min_sample_id]
            if min_sample_weight < self.minimum_sample_weight:
                # if any prior weight is less than the minimum allowed weight
                # replace the sample with the new sample
                # udpate distance matrix and the gram matrix
                self._update_distance_matrix(gram_vector, new_train_sample_norm, min_sample_id, -1, 0, 1)

                # normalize the prior weights so that the new sample gets weight as the learning rate
                self.prior_weights[min_sample_id] = 0
                self.prior_weights = self.prior_weights * (1 - self.lr) / np.sum(self.prior_weights)
                self.prior_weights[min_sample_id] = self.lr

                # set the new sample and new sample position in the samplesf
                new_sample_id = min_sample_id
                new_sample = new_train_sample

                # update
                if verbose:
                    print('~1~')
                self.centroids[new_sample_id] = new_train_sample
                if img is None:
                    self.samples[new_sample_id] = [new_train_sample]
                else:
                    self.samples[new_sample_id] = [img]
            else:
                # if no sample has low enough prior weight, then we either merge the new sample with
                # an existing sample, or merge two of the existing samples and insert the new sample
                # in the vacated position
                closest_sample_to_new_sample = np.argmin(dist_vector)
                new_sample_min_dist = dist_vector[closest_sample_to_new_sample]

                # find the closest pair amongst existing samples
                closest_existing_sample_idx = np.argmin(self._distance_matrix.flatten())
                closest_existing_sample_pair = np.unravel_index(closest_existing_sample_idx, self._distance_matrix.shape)
                existing_samples_min_dist = self._distance_matrix[closest_existing_sample_pair[0], closest_existing_sample_pair[1]]
                closest_existing_sample1, closest_existing_sample2 = closest_existing_sample_pair
                if closest_existing_sample1 == closest_existing_sample2:
                    raise("Score matrix diagnoal filled wrongly")

                if new_sample_min_dist < existing_samples_min_dist:
                    # if the min distance of the new sample to the existing samples is less than the
                    # min distance amongst any of the existing samples, we merge the new sample with
                    # the nearest existing sample

                    # renormalize prior weights
                    self.prior_weights = self.prior_weights * (1 - self.lr)

                    # set the position of the merged sample
                    merged_sample_id = closest_sample_to_new_sample

                    # extract the existing sample the merge
                    existing_sample_to_merge = self.centroids[merged_sample_id]

                    # merge the new_training_sample with existing sample
                    merged_sample = self._merge_samples(existing_sample_to_merge,
                                                        new_train_sample,
                                                        self.prior_weights[merged_sample_id],
                                                        self.lr,
                                                        'merge')

                    # update distance matrix and the gram matrix
                    self._update_distance_matrix(gram_vector,
                                                 new_train_sample_norm,
                                                 merged_sample_id,
                                                 -1,
                                                 self.prior_weights[merged_sample_id, 0],
                                                 self.lr)

                    # udpate the prior weight of the merged sample
                    self.prior_weights[closest_sample_to_new_sample] = self.prior_weights[closest_sample_to_new_sample] + self.lr

                    # update
                    if verbose:
                        print('~2~')
                    self.centroids[merged_sample_id] = merged_sample
                    if img is None:
                        self.samples[merged_sample_id].append(new_train_sample)
                    else:
                        self.samples[merged_sample_id].append(img)
                else:
                    # if the min distance amongst any of the existing samples is less than the
                    # min distance of the new sample to the existing samples, we merge the nearest
                    # existing samples and insert the new sample in the vacated position

                    # renormalize prior weights
                    self.prior_weights = self.prior_weights * (1 - self.lr)

                    if self.prior_weights[closest_existing_sample2] > self.prior_weights[closest_existing_sample1]:
                        tmp = closest_existing_sample1
                        closest_existing_sample1 = closest_existing_sample2
                        closest_existing_sample2 = tmp

                    sample_to_merge1 = self.centroids[closest_existing_sample1]
                    sample_to_merge2 = self.centroids[closest_existing_sample2]

                    # merge the existing closest samples
                    merged_sample = self._merge_samples(sample_to_merge1,
                                                        sample_to_merge2,
                                                        self.prior_weights[closest_existing_sample1],
                                                        self.prior_weights[closest_existing_sample2],
                                                        'merge')

                    # update distance matrix and the gram matrix
                    self._update_distance_matrix(gram_vector,
                                                new_train_sample_norm,
                                                closest_existing_sample1,
                                                closest_existing_sample2,
                                                self.prior_weights[closest_existing_sample1, 0],
                                                self.prior_weights[closest_existing_sample2, 0])

                    # update prior weights for the merged sample and the new sample
                    self.prior_weights[closest_existing_sample1] = self.prior_weights[closest_existing_sample1] + self.prior_weights[closest_existing_sample2]
                    self.prior_weights[closest_existing_sample2] = self.lr

                    # set the mreged sample position and new sample position
                    merged_sample_id = closest_existing_sample1
                    new_sample_id = closest_existing_sample2

                    new_sample = new_train_sample

                    # update
                    if verbose:
                        print('~3~')
                    self.centroids[merged_sample_id] = merged_sample
                    self.centroids[new_sample_id] = new_train_sample
                    self.samples[closest_existing_sample1].extend(self.samples[closest_existing_sample2])
                    if img is None:
                        self.samples[new_sample_id] = [new_train_sample]
                    else:
                        self.samples[new_sample_id] = [img]
        else:
            # if the memory is not full, insert the new sample in the next empty location
            sample_position = self.num_samples

            # update the distance matrix and the gram matrix
            self._update_distance_matrix(gram_vector, new_train_sample_norm,sample_position, -1, 0, 1)

            # update the prior weight
            if sample_position == 0:
                self.prior_weights[sample_position] = 1
            else:
                self.prior_weights = self.prior_weights * (1 - self.lr)
                self.prior_weights[sample_position] = self.lr

            new_sample_id = sample_position
            new_sample = new_train_sample

            # update
            if verbose:
                print('~4~')
            self.centroids.append(new_train_sample)
            if img is None:
                self.samples.append([new_train_sample])
            else:
                self.samples.append([img])
            self.num_samples += 1

        if abs(1 - np.sum(self.prior_weights)) > 1e-5:
            raise("weights not properly udpated")

        return merged_sample, new_sample, merged_sample_id, new_sample_id