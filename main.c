// Fluid Simulator (2D Smoke) in C using OpenGL and GLFW
// Modularized implementation of Jos Stam’s "Stable Fluids" with gravity

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <GLFW/glfw3.h>

#define N 128
#define ITER 20
#define DT 0.1f
#define DIFF 0.0001f
#define VISC 0.0001f
#define FORCE 100.0f
#define SOURCE 100.0f
#define GRAVITY -9.8f

static int size;
static float *u, *v, *u_prev, *v_prev;
static float *dens, *dens_prev;
static unsigned int densityTex;

static int winWidth = 600, winHeight = 600;

void add_source(float *x, float *s);
void set_bnd(int b, float *x);
void lin_solve(int b, float *x, float *x0, float a, float c);
void diffuse(int b, float *x, float *x0, float diff);
void advect(int b, float *d, float *d0, float *u, float *v);
void project(float *u, float *v, float *p, float *div);
void step();
void initTexture();
void renderTexture();
void allocate_fields();
void free_fields();

void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
    winWidth = width;
    winHeight = height;
    glViewport(0, 0, winWidth, winHeight);
}

void add_source(float *x, float *s) {
    for (int i = 0; i < size; i++) x[i] += DT * s[i];
}

void set_bnd(int b, float *x) {
    for (int i = 1; i < N-1; i++) {
        x[i] = b==2 ? -x[i+N] : x[i+N];
        x[(N-1)*N + i] = b==2 ? -x[(N-2)*N + i] : x[(N-2)*N + i];
        x[i*N] = b==1 ? -x[i*N + 1] : x[i*N + 1];
        x[i*N + N-1] = b==1 ? -x[i*N + N-2] : x[i*N + N-2];
    }
    x[0] = 0.5f*(x[1] + x[N]);
    x[N-1] = 0.5f*(x[N-2] + x[2*N-1]);
    x[(N-1)*N] = 0.5f*(x[(N-2)*N] + x[(N-1)*N + 1]);
    x[N*N-1] = 0.5f*(x[N*N-2] + x[(N-1)*N - 1]);
}

void lin_solve(int b, float *x, float *x0, float a, float c) {
    for (int k = 0; k < ITER; k++) {
        for (int j = 1; j < N-1; j++)
            for (int i = 1; i < N-1; i++)
                x[i+j*N] = (x0[i+j*N] + a*(x[(i-1)+j*N] + x[(i+1)+j*N] + x[i+(j-1)*N] + x[i+(j+1)*N]))/c;
        set_bnd(b, x);
    }
}

void diffuse(int b, float *x, float *x0, float diff) {
    float a = DT * diff * (N-2) * (N-2);
    lin_solve(b, x, x0, a, 1 + 4*a);
}

void advect(int b, float *d, float *d0, float *u, float *v) {
    float dt0 = DT * (N-2);
    for (int j = 1; j < N-1; j++) for (int i = 1; i < N-1; i++) {
        float x = i - dt0 * u[i+j*N]; float y = j - dt0 * v[i+j*N];
        x = fmin(fmax(x,0.5f), N-2+0.5f); y = fmin(fmax(y,0.5f), N-2+0.5f);
        int i0 = (int)floorf(x), i1 = i0+1, j0 = (int)floorf(y), j1 = j0+1;
        float s1=x-i0, s0=1-s1, t1=y-j0, t0=1-t1;
        d[i+j*N] = s0*(t0*d0[i0+j0*N]+t1*d0[i0+j1*N]) + s1*(t0*d0[i1+j0*N]+t1*d0[i1+j1*N]);
    }
    set_bnd(b, d);
}

void project(float *u, float *v, float *p, float *div) {
    for (int j = 1; j < N-1; j++) for (int i = 1; i < N-1; i++) {
        div[i+j*N] = -0.5f*(u[(i+1)+j*N]-u[(i-1)+j*N]+v[i+(j+1)*N]-v[i+(j-1)*N])/N;
        p[i+j*N] = 0;
    }
    set_bnd(0,div); set_bnd(0,p);
    lin_solve(0,p,div,1,4);
    for (int j = 1; j < N-1; j++) for (int i = 1; i < N-1; i++) {
        u[i+j*N] -= 0.5f * N * (p[(i+1)+j*N]-p[(i-1)+j*N]);
        v[i+j*N] -= 0.5f * N * (p[i+(j+1)*N]-p[i+(j-1)*N]);
    }
    set_bnd(1,u); set_bnd(2,v);
}

void step() {
    add_source(u, u_prev); add_source(v, v_prev); add_source(dens, dens_prev);
    for (int i = 0; i < size; i++) v_prev[i] += GRAVITY * DT;
    memcpy(u_prev, u, size*sizeof(float)); diffuse(1, u, u_prev, VISC);
    memcpy(v_prev, v, size*sizeof(float)); diffuse(2, v, v_prev, VISC);
    project(u, v, u_prev, v_prev);
    memcpy(u_prev, u, size*sizeof(float)); memcpy(v_prev, v, size*sizeof(float));
    advect(1, u, u_prev, u_prev, v_prev); advect(2, v, v_prev, u_prev, v_prev);
    project(u, v, u_prev, v_prev);
    memcpy(dens_prev, dens, size*sizeof(float)); diffuse(0, dens, dens_prev, DIFF);
    memcpy(dens_prev, dens, size*sizeof(float)); advect(0, dens, dens_prev, u, v);
}

void initTexture() {
    glGenTextures(1, &densityTex); glBindTexture(GL_TEXTURE_2D, densityTex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, N, N, 0, GL_RED, GL_FLOAT, NULL);
}

void renderTexture() {
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, winWidth, 0, winHeight, -1, 1);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glBindTexture(GL_TEXTURE_2D, densityTex);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, N, N, GL_RED, GL_FLOAT, dens);
    glEnable(GL_TEXTURE_2D);
    glBegin(GL_QUADS);
      glTexCoord2f(0,0); glVertex2f(0, 0);
      glTexCoord2f(1,0); glVertex2f(winWidth, 0);
      glTexCoord2f(1,1); glVertex2f(winWidth, winHeight);
      glTexCoord2f(0,1); glVertex2f(0, winHeight);
    glEnd();
    glDisable(GL_TEXTURE_2D);
}

void allocate_fields() {
    size = N*N;
    u = calloc(size, sizeof(float)); v = calloc(size, sizeof(float));
    u_prev = calloc(size, sizeof(float)); v_prev = calloc(size, sizeof(float));
    dens = calloc(size, sizeof(float)); dens_prev = calloc(size, sizeof(float));
}

void free_fields() {
    free(u); free(v); free(u_prev); free(v_prev); free(dens); free(dens_prev);
}

int main() {
    allocate_fields();
    if (!glfwInit()) return -1;
    GLFWwindow* w = glfwCreateWindow(winWidth, winHeight, "Fluid Simulator", NULL, NULL);
    glfwMakeContextCurrent(w); glfwSwapInterval(1);
    glfwSetFramebufferSizeCallback(w, framebuffer_size_callback);
    initTexture(); glClearColor(0, 0, 0, 1);
    while (!glfwWindowShouldClose(w)) {
        memset(u_prev, 0, size*sizeof(float));
        memset(v_prev, 0, size*sizeof(float));
        memset(dens_prev, 0, size*sizeof(float));

        double xpos, ypos;
        glfwGetCursorPos(w, &xpos, &ypos);
        int cx = (int)((xpos / winWidth) * N);
        int cy = (int)(((winHeight - ypos) / winHeight) * N);
        if (cx >= 1 && cx < N-1 && cy >= 1 && cy < N-1) {
            v_prev[cx+cy*N] = FORCE;
            dens_prev[cx+cy*N] = SOURCE;
        }

        step();
        glClear(GL_COLOR_BUFFER_BIT);
        renderTexture();
        glfwSwapBuffers(w);
        glfwPollEvents();
    }
    free_fields(); glfwTerminate(); return 0;
}