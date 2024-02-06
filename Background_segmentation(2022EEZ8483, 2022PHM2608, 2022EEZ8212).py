'''importing required packages'''

import numpy as np
import cv2
import os
import shutil
from tqdm import tqdm

'''Class is responsible  for background segmentation from an inout video feed'''
class Background_segmentation():
    '''This class is responsible for Foreground-Background Segmentation of Video feed'''

    def __init__(self):
        self.alpha_value = 0.01
        self.T_min_portion_background = 0.7

    def sort(self, mean, covariance_matrix, weights, index):
        return np.take_along_axis(mean, index, axis=0), np.take_along_axis(covariance_matrix, index,axis=0), np.take_along_axis(weights, index,axis=0)

    def extract_image_features(self, captured_video):
        _, captured_frame = captured_video.read()
        captured_frame = cv2.cvtColor(captured_frame, cv2.COLOR_BGR2GRAY)  # RGB to grayscale
        captured_frame_height, captured_frame_width = captured_frame.shape
        return captured_frame, captured_frame_height, captured_frame_width

    def initialize_weights_frame(self, height, width):
        frame_weight = np.zeros([3, height, width], np.double)
        frame_weight[0, :, :], frame_weight[1, :, :], frame_weight[2, :, :] = 0.30, 0.35, 0.35  # Sum of weights is equal to 1
        return frame_weight

    def initialize_frame_mean(self, height, width, extracted_frame_mean):
        mean = np.zeros([3, height, width], np.double)
        mean[1, :, :] = extracted_frame_mean
        return mean

    def initialize_covariance_matrix_frame(self, height, width):
        covariance_matrix_frame = np.zeros([3, height, width], np.double)
        covariance_matrix_frame[:, :, :] = 5
        return covariance_matrix_frame

    def main(self):
        if os.path.isdir("background"):
            shutil.rmtree("background")
        if os.path.isdir("foreground"):
            shutil.rmtree("foreground")

        os.mkdir("background")
        os.mkdir("foreground")

        captured_video = cv2.VideoCapture(r"C:\Users\DELL\Downloads\ML Assignment video.mpg")

        main_loop_count = 0
        main_ret_boolean = True

        extracted_frame, frame_height, frame_width = self.extract_image_features(captured_video)
        mean = self.initialize_frame_mean(frame_height, frame_width, extracted_frame)
        frame_covariance_matrix = self.initialize_covariance_matrix_frame(frame_height, frame_width)
        weights_frame = self.initialize_weights_frame(frame_height, frame_width)
        normalized_weights = np.zeros([3, frame_height, frame_width], np.double)

        frame_background = np.zeros([frame_height, frame_width], np.uint8)

        frame_standard_deviation = np.sqrt(frame_covariance_matrix)

        while captured_video.isOpened():
            # Counter to save images with index for later video transformation.
            main_loop_count = main_loop_count + 1
            main_ret_boolean, extracted_frame = captured_video.read()
            if main_ret_boolean == True:
                different_gaussians, same_gaussians = [], []
                different_gaussians_index = np.zeros([3, frame_height, frame_width])
                same_gaussians_index = np.zeros([frame_height, frame_width])
                try:
                    extracted_frame = cv2.cvtColor(extracted_frame, cv2.COLOR_BGR2GRAY).astype(np.double)
                except:
                    pass

                # Check if the frame x^t is within 2.5 standard deviations. If yes, it matches the kth gaussian
                for i in range(3):
                    same_gaussians.append(
                        np.where(np.abs(extracted_frame - mean[i]) / frame_standard_deviation[i] <= 2.5))
                    different_gaussians.append(
                        np.where(np.abs(extracted_frame - mean[i]) / frame_standard_deviation[i] > 2.5))

                same_gaussians_index[same_gaussians[0]], same_gaussians_index[same_gaussians[1]
                ], same_gaussians_index[same_gaussians[
                    2]] = 1, 1, 1  # Assign pixel value of 1 to the pixels of frame x^t which match the gaussians.

                # Extract non-matching pixel values as well for creation of new gaussian for these pixels later on
                different_gaussians_index = np.where(same_gaussians_index == 0)

                for j in range(3):
                    weights_frame[j][same_gaussians[j]] = (1 - self.alpha_value) * weights_frame[j][same_gaussians[
                                                                  j]] + self.alpha_value  # Update weights of pixels which match the gaussians.

                    seg_rho = self.alpha_value * (
                            1 / (np.sqrt(2 * np.pi) * frame_standard_deviation[j][same_gaussians[j]])) * (
                                  np.exp(-0.5 * (
                                          ((extracted_frame[same_gaussians[j]] - mean[j][same_gaussians[j]]) /
                                           frame_standard_deviation[j][
                                               same_gaussians[j]]) ** 2)))  # Find ρ for mean and covariance updation.

                    mean[j][same_gaussians[j]] = (1 - seg_rho) * mean[j][same_gaussians[j]] + seg_rho * \
                                                 extracted_frame[same_gaussians[j]]  # Update mean for matching gaussians

                    frame_covariance_matrix[j][same_gaussians[j]] = (1 - seg_rho) * frame_covariance_matrix[j][same_gaussians[j]] + seg_rho * ((
                                                                                                                extracted_frame[same_gaussians[j]] -mean[j][
                                                                                                                    same_gaussians[j]]) ** 2)  # Update covariance for matching gaussians

                # standardise the updated weights
                weights_frame = weights_frame / np.sum(weights_frame, axis=0)
                # Find normalized weights Σw/σ for sorting mean, weights and covariance wrt to it.
                normalized_weights = weights_frame / frame_standard_deviation

                seg_index = np.argsort(normalized_weights, axis=0)
                mean, frame_covariance_matrix, weights_frame = self.sort(mean, frame_covariance_matrix, weights_frame,
                                                                         seg_index)

                # update not matching gaussians with new least probable gaussian values.
                mean[0][different_gaussians_index] = extracted_frame[different_gaussians_index]
                frame_covariance_matrix[0][different_gaussians_index] = 10
                weights_frame[0][different_gaussians_index] = 0.1

                extracted_frame = extracted_frame.astype(np.uint8)

                # select background if last gaussian from sorted weights crosses the threshold and its gaussian values match as well
                temp = np.zeros([frame_height, frame_width])
                bg_gauss_pixels = np.where(weights_frame[2] > self.T_min_portion_background)
                temp[bg_gauss_pixels] = 1
                temp[same_gaussians[2]] = temp[same_gaussians[2]] + 1
                frame_background[np.where(temp == 2)] = extracted_frame[np.where(temp == 2)]

                # select background if last two gaussians from sorted weights crosses the threshold and their gaussian values match as well
                temp = np.zeros([frame_height, frame_width])
                bg_gauss_pixels = np.where(
                    ((weights_frame[2] + weights_frame[1]) > self.T_min_portion_background) & (
                            weights_frame[2] < self.T_min_portion_background))
                temp[bg_gauss_pixels] = 1
                temp[same_gaussians[1]] = temp[same_gaussians[1]] + 1
                temp[same_gaussians[2]] = temp[same_gaussians[2]] + 1
                frame_background[np.where(temp == 2)] = extracted_frame[np.where(temp == 2)]

                frame_concatenated = np.concatenate(
                    (frame_background, cv2.absdiff(extracted_frame, frame_background)),
                    axis=1)  # Concatenate frames for visualization

                # Save frames as images
                cv2.imwrite(r"background/background_frame_save%d.jpg" % main_loop_count, frame_background)
                cv2.imwrite(r"foreground/foreground_frame_save%d.jpg" %
                            main_loop_count, cv2.absdiff(extracted_frame, frame_background))

                cv2.imshow('Video', frame_concatenated)
                if cv2.waitKey(25) & 0xFF == 27:
                    break
            else:
                break
        captured_video.release()
        cv2.destroyAllWindows()

        background_writer = cv2.VideoWriter(
            'background_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30,
            (frame_width, frame_height))  # Write background images to video
        foreground_writer = cv2.VideoWriter(
            'foreground_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30,
            (frame_width, frame_height))  # Write background images to video

        tqdm_count = 1
        for tqdm_count in tqdm(range(1, main_loop_count)):
            background_image = cv2.imread(r"background/background_frame_save%d.jpg" % tqdm_count, 1)
            foreground_image = cv2.imread(r"foreground/foreground_frame_save%d.jpg" % tqdm_count, 1)

            # background_image = cv2.resize(background_image, (frame_width, frame_height))
            background_writer.write(background_image)
            # foreground_image = cv2.resize(foreground_image, (frame_width, frame_height))
            foreground_writer.write(foreground_image)
            main_loop_count += 1
        background_writer.release()
        foreground_writer.release()

        # Delete image directories
        if os.path.isdir("background"):
            shutil.rmtree("background")
        if os.path.isdir("foreground"):
            shutil.rmtree("foreground")


if __name__ == '__main__':
    '''Creating background_segmentation class object'''
    background_segmentation_obj = Background_segmentation()

    '''calling main function of the class'''
    background_segmentation_obj.main()
