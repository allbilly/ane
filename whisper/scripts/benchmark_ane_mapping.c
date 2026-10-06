// Compare copy instructions on the real driver's uncached BO mapping.
// This allocates a BO but submits no ANE work. Hold the shared hardware locks.
#define _GNU_SOURCE
#include <fcntl.h>
#include <linux/types.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>
#define DRM_COMMAND_BASE 0x40
#define DRM_IOWR(nr, type) _IOWR('d', nr, type)
#include "../../kmod/uapi/drm/ane_accel.h"

static uint64_t now_ns(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return (uint64_t)t.tv_sec*1000000000ull + t.tv_nsec;
}
static void libc_copy(void *dst, const void *src, size_t bytes) { memcpy(dst,src,bytes); }
static void ld1_64(void *dst, const void *src, size_t bytes) {
    for (size_t i=0;i<bytes;i+=64) {
        asm volatile("ld1 {v0.16b, v1.16b, v2.16b, v3.16b}, [%0]\n\t"
                     "st1 {v0.16b, v1.16b, v2.16b, v3.16b}, [%1]"
                     : : "r"((const char *)src+i), "r"((char *)dst+i)
                     : "v0","v1","v2","v3","memory");
    }
}
static void ld1_256(void *dst, const void *src, size_t bytes) {
    for (size_t i=0;i<bytes;i+=256) {
        const char *s=(const char *)src+i; char *d=(char *)dst+i;
        asm volatile("ld1 {v0.16b, v1.16b, v2.16b, v3.16b}, [%0], #64\n\t"
                     "ld1 {v4.16b, v5.16b, v6.16b, v7.16b}, [%0], #64\n\t"
                     "ld1 {v16.16b, v17.16b, v18.16b, v19.16b}, [%0], #64\n\t"
                     "ld1 {v20.16b, v21.16b, v22.16b, v23.16b}, [%0]\n\t"
                     "st1 {v0.16b, v1.16b, v2.16b, v3.16b}, [%1], #64\n\t"
                     "st1 {v4.16b, v5.16b, v6.16b, v7.16b}, [%1], #64\n\t"
                     "st1 {v16.16b, v17.16b, v18.16b, v19.16b}, [%1], #64\n\t"
                     "st1 {v20.16b, v21.16b, v22.16b, v23.16b}, [%1]"
                     : "+r"(s), "+r"(d) : : "v0","v1","v2","v3","v4","v5","v6","v7",
                       "v16","v17","v18","v19","v20","v21","v22","v23","memory");
    }
}
static void ldnp_256(void *dst, const void *src, size_t bytes) {
    for (size_t i=0;i<bytes;i+=256) {
        const char *s=(const char *)src+i; char *d=(char *)dst+i;
        asm volatile("ldnp q0, q1, [%0]\n\tldnp q2, q3, [%0,#32]\n\t"
                     "ldnp q4, q5, [%0,#64]\n\tldnp q6, q7, [%0,#96]\n\t"
                     "ldnp q16, q17, [%0,#128]\n\tldnp q18, q19, [%0,#160]\n\t"
                     "ldnp q20, q21, [%0,#192]\n\tldnp q22, q23, [%0,#224]\n\t"
                     "stp q0, q1, [%1]\n\tstp q2, q3, [%1,#32]\n\t"
                     "stp q4, q5, [%1,#64]\n\tstp q6, q7, [%1,#96]\n\t"
                     "stp q16, q17, [%1,#128]\n\tstp q18, q19, [%1,#160]\n\t"
                     "stp q20, q21, [%1,#192]\n\tstp q22, q23, [%1,#224]"
                     : : "r"(s), "r"(d) : "v0","v1","v2","v3","v4","v5","v6","v7",
                       "v16","v17","v18","v19","v20","v21","v22","v23","memory");
    }
}
int main(void) {
    const size_t capacity=131072; const int repeats=400;
    int fd=open("/dev/accel/accel0",O_RDWR|O_CLOEXEC);
    struct drm_ane_bo_init bo={.size=capacity};
    if (fd<0 || ioctl(fd,DRM_IOCTL_ANE_BO_INIT,&bo) || !bo.handle) { perror("ANE BO"); return 1; }
    void *src=mmap(NULL,capacity,PROT_READ|PROT_WRITE,MAP_SHARED,fd,bo.offset);
    unsigned char *expected=malloc(capacity), *dst=malloc(capacity);
    if (src==MAP_FAILED || !expected || !dst) { perror("mapping allocation"); close(fd); return 1; }
    for (size_t i=0;i<capacity;i++) expected[i]=(unsigned char)(i*37+(i>>8));
    memcpy(src,expected,capacity); asm volatile("dsb sy" ::: "memory");
    void (*copies[])(void *,const void *,size_t)={libc_copy,ld1_64,ld1_256,ldnp_256};
    const char *names[]={"libc_memcpy","ld1_64","ld1_256","ldnp_256"};
    const size_t sizes[]={24576,98304};
    for (int round=0;round<3;round++) for (int size=0;size<2;size++) for (int mode=0;mode<4;mode++) {
        const size_t bytes=sizes[size];
        for (int i=0;i<4;i++) copies[mode](dst,src,bytes);
        uint64_t start=now_ns();
        for (int i=0;i<repeats;i++) copies[mode](dst,src,bytes);
        const double ms=(now_ns()-start)/1e6;
        if (memcmp(dst,expected,bytes)) { fprintf(stderr,"copy mismatch\n"); return 2; }
        printf("{\"mode\":\"%s\",\"bytes\":%zu,\"repeats\":%d,\"round\":%d,\"ms\":%.6f,\"MB_s\":%.3f}\n",
               names[mode],bytes,repeats,round+1,ms,(double)bytes*repeats/ms/1000);
    }
    munmap(src,capacity); close(fd); free(expected); free(dst);
    return 0;
}
