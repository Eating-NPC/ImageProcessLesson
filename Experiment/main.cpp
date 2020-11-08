#include "Experiment.h"
#include <QtWidgets/QApplication>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    Experiment w;
    w.show();
    return a.exec();
}
