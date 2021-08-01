#ifndef INTERACTIONS_H
#define INTERACTIONS_H
#define W 1600
#define H 1600
#define DELTA 5 // pixel increment for arrow keys
#define TITLE_STRING "flashlight: distance image display app"
int2 loc = {W / 2, H / 2};
bool dragMode = false; // mouse tracking mode

void keyboard(unsigned char key, int x, int y) {
    if (key == 'a') dragMode = !dragMode; // toggle tracking mode
    if (key == 27) exit(0);
    glutPostRedisplay();
}

void mouseMove(int x, int y) {
    if (dragMode) return;
    loc.x = W * x / glutGet(GLUT_WINDOW_WIDTH);
    loc.y = H * y / glutGet(GLUT_WINDOW_HEIGHT);
    glutPostRedisplay();
}

void mouseDrag(int x, int y) {
    if (!dragMode) return;
    loc.x = W * x / glutGet(GLUT_WINDOW_WIDTH);
    loc.y = H * y / glutGet(GLUT_WINDOW_HEIGHT);
    glutPostRedisplay();
}

bool leftMouseButtonDown = 0;
bool rightMouseButtonDown = 0;
bool middleMouseButtonDown = 0;

void mouse(int button, int state, int x, int y) {
    // Save the left button state
    if (button == GLUT_LEFT_BUTTON) {
        leftMouseButtonDown = (state == GLUT_DOWN);
    } else if (button == GLUT_RIGHT_BUTTON) {
        // right MouseButton
        rightMouseButtonDown = (state == GLUT_DOWN);
    } else if (button == GLUT_MIDDLE_BUTTON) {
        // middle MouseButton
        middleMouseButtonDown = (state == GLUT_DOWN);
    }


    // Save the mouse position
//    mousePos.x = x;
//    mousePos.y = y;
}

void mouseWheel(int button, int dir, int x, int y) {
//    if(button&1<<0) printf("Left Mouse Button\n");
//    if(button&1<<1) printf("right Mouse Button\n");
//    //if(button&1<<2) printf("??1 Mouse Button\n");
//    //if(button&1<<3) printf("??2 Mouse Button\n");
//    if(button&1<<4) printf("Middle Mouse Button\n");
    if (dir > 0) {
        // Zoom in
    } else {
        // Zoom out
    }

    return;
}

void handleSpecialKeypress(int key, int x, int y) {
    if (key == GLUT_KEY_LEFT) loc.x -= DELTA;
    if (key == GLUT_KEY_RIGHT) loc.x += DELTA;
    if (key == GLUT_KEY_UP) loc.y -= DELTA;
    if (key == GLUT_KEY_DOWN) loc.y += DELTA;
    glutPostRedisplay();
}

void printInstructions() {
    printf("flashlight interactions\n");
    printf("a: toggle mouse tracking mode\n");
    printf("arrow keys: move ref location\n");
    printf("esc: close graphics window\n");
}

#endif