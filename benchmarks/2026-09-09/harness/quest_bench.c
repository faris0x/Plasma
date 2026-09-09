/* QuEST (v3.x) benchmark driver for the cross-platform suite.
 * Reads a gate file (opcode args per line) and reports:
 *   construct_ms  construction (gate application) time
 *   sample_ms     sampling time (shots x all-qubit measure)
 * Gate file format:
 *   h <q> | x <q> | y <q> | z <q> | s <q> | t <q>
 *   cx <c> <t> | cz <a> <b> | swap <a> <b>
 *   rz <q> <th> | ry <q> <th> | rx <q> <th> | p <q> <th>
 *   cp <c> <t> <th>
 * Two-qubit matrix index: c|t as LSB|MSB (QuEST applyMatrix4 target1=LSB).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "QuEST.h"

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

static ComplexMatrix2 m2(double r00,double i00,double r01,double i01,
                         double r10,double i10,double r11,double i11) {
    ComplexMatrix2 m;
    m.real[0][0]=r00; m.imag[0][0]=i00; m.real[0][1]=r01; m.imag[0][1]=i01;
    m.real[1][0]=r10; m.imag[1][0]=i10; m.real[1][1]=r11; m.imag[1][1]=i11;
    return m;
}
static ComplexMatrix4 m4(double r[16], double i[16]) {
    ComplexMatrix4 m;
    for (int a=0;a<4;a++) for (int b=0;b<4;b++) { m.real[a][b]=r[a*4+b]; m.imag[a][b]=i[a*4+b]; }
    return m;
}

int main(int argc, char** argv) {
    if (argc < 4) { fprintf(stderr, "usage: %s <n> <gates> <shots>\n", argv[0]); return 1; }
    int n = atoi(argv[1]);
    int shots = atoi(argv[3]);
    FILE* f = fopen(argv[2], "r");
    if (!f) { fprintf(stderr, "cannot open %s\n", argv[2]); return 1; }

    QuESTEnv env = createQuESTEnv();
    Qureg q = createQureg(n, env);

    double t0 = now_ms();
    char op[8];
    const double sq = 1.0 / sqrt(2.0);
    const double pi4 = M_PI / 4.0;
    while (fscanf(f, "%7s", op) == 1) {
        if (!strcmp(op, "h")) { int a; fscanf(f,"%d",&a); ComplexMatrix2 u=m2(sq,0,sq,0, sq,0,-sq,0); applyMatrix2(q,a,u); }
        else if (!strcmp(op, "x")) { int a; fscanf(f,"%d",&a); ComplexMatrix2 u=m2(0,0,1,0, 1,0,0,0); applyMatrix2(q,a,u); }
        else if (!strcmp(op, "y")) { int a; fscanf(f,"%d",&a); ComplexMatrix2 u=m2(0,0,0,-1, 0,1,0,0); applyMatrix2(q,a,u); }
        else if (!strcmp(op, "z")) { int a; fscanf(f,"%d",&a); ComplexMatrix2 u=m2(1,0,0,0, 0,0,-1,0); applyMatrix2(q,a,u); }
        else if (!strcmp(op, "s")) { int a; fscanf(f,"%d",&a); ComplexMatrix2 u=m2(1,0,0,0, 0,0,0,1); applyMatrix2(q,a,u); }
        else if (!strcmp(op, "t")) { int a; fscanf(f,"%d",&a); ComplexMatrix2 u=m2(1,0,0,0, 0,0,cos(pi4),sin(pi4)); applyMatrix2(q,a,u); }
        else if (!strcmp(op, "rz")) { int a; double th; fscanf(f,"%d %lf",&a,&th);
            ComplexMatrix2 u=m2(cos(th/2),-sin(th/2),0,0, 0,0,cos(th/2),sin(th/2)); applyMatrix2(q,a,u); }
        else if (!strcmp(op, "ry")) { int a; double th; fscanf(f,"%d %lf",&a,&th);
            ComplexMatrix2 u=m2(cos(th/2),0,-sin(th/2),0, sin(th/2),0,cos(th/2),0); applyMatrix2(q,a,u); }
        else if (!strcmp(op, "rx")) { int a; double th; fscanf(f,"%d %lf",&a,&th);
            ComplexMatrix2 u=m2(cos(th/2),0,0,-sin(th/2), 0,-sin(th/2),cos(th/2),0); applyMatrix2(q,a,u); }
        else if (!strcmp(op, "p")) { int a; double th; fscanf(f,"%d %lf",&a,&th);
            ComplexMatrix2 u=m2(1,0,0,0, 0,0,cos(th),sin(th)); applyMatrix2(q,a,u); }
        else if (!strcmp(op, "cx")) { int a,b; fscanf(f,"%d %d",&a,&b);
            double r[16]={1,0,0,0, 0,0,1,0, 0,1,0,0, 0,0,0,1}; double i[16]={0}; applyMatrix4(q,a,b,m4(r,i)); }
        else if (!strcmp(op, "cz")) { int a,b; fscanf(f,"%d %d",&a,&b);
            double r[16]={1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,-1}; double i[16]={0}; applyMatrix4(q,a,b,m4(r,i)); }
        else if (!strcmp(op, "swap")) { int a,b; fscanf(f,"%d %d",&a,&b);
            double r[16]={1,0,0,0, 0,0,1,0, 0,1,0,0, 0,0,0,1}; double i[16]={0}; applyMatrix4(q,a,b,m4(r,i)); }
        else if (!strcmp(op, "cp")) { int a,b; double th; fscanf(f,"%d %d %lf",&a,&b,&th);
            double r[16]={1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,cos(th)}; double i[16]={0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,sin(th)}; applyMatrix4(q,a,b,m4(r,i)); }
        else { fprintf(stderr, "unknown op %s\n", op); return 1; }
    }
    double t1 = now_ms();
    printf("construct_ms %.3f\n", t1 - t0);

    if (shots > 0) {
        t0 = now_ms();
        for (int s = 0; s < shots; s++) {
            for (int qbit = 0; qbit < n; qbit++) (void)measure(q, qbit);
        }
        t1 = now_ms();
        printf("sample_ms %.3f\n", t1 - t0);
    }

    destroyQureg(q, env);
    destroyQuESTEnv(env);
    fclose(f);
    return 0;
}