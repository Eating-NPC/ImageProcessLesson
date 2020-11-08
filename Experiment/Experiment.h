#pragma once

#include <QtWidgets/QMainWindow>
#include "ui_Experiment.h"


class Experiment : public QMainWindow
{
    Q_OBJECT

public:
    Experiment(QWidget *parent = Q_NULLPTR);

private:
    Ui::ExperimentClass ui;

private slots:
    void on_Load_clicked();
    void on_Capture_clicked();
    void on_grayImage_clicked();
    void on_Hist_clicked();
    void on_equalizeHist_clicked();
    void on_equalizeHistCV_clicked();
    void on_GradientSharp_clicked();
    void on_Laplace_clicked();
    void on_SaltAndPepper_clicked();
    void on_Monocular_clicked();
    void on_Binocular_clicked();
    void on_Stereo_clicked();
    void on_Robert_clicked();
    void on_Soble_clicked();
    void on_Canny_clicked();
    void on_Gauss_clicked();
    void on_LinearFilter_clicked();
    void on_MorphologicalFilter_clicked();
    void on_EdgeFilter_clicked();
    void on_AffineTrans_clicked();
    void on_PerspectiveTrans_clicked();
    void on_ThresholdSeg_clicked();
    void on_OTSU_clicked();
    void on_Kittle_clicked();
    void on_ThresholdSegCV_clicked();
    void on_InterframeDiff_clicked();
    void on_MixedGauss_clicked();
    void on_MixedGaussVideo_clicked();
    void on_SIFT_clicked();
    void on_Brisk_clicked();
    void on_Brisk1_clicked();
    void on_ORB_clicked();
    void on_ORB1_clicked();
    void on_haar_clicked();
    void on_ssd_clicked();
    void on_yolo_clicked();
    void on_svm_clicked();
    void on_carcascade_clicked();
    void on_num_clicked();
    void on_tran_clicked();
    void on_camshift_clicked();
    void on_singleGauss_clicked();
};
