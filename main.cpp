#include "mainwindow.h"
#include <QApplication>

/* TODO:
 *  - detect object
 *  - draw bounding box
 *  - write points to csv file
 */

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    MainWindow w;
    w.show();
    return a.exec();
}
