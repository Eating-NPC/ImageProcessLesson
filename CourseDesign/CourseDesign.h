#pragma once

#include <QtWidgets/QMainWindow>
#include "ui_CourseDesign.h"

class CourseDesign : public QMainWindow
{
    Q_OBJECT

public:
    CourseDesign(QWidget *parent = Q_NULLPTR);

private:
    Ui::CourseDesignClass ui;
};
