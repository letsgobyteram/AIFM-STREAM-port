/*-----------------------------------------------------------------------*/
/* Program: STREAM                                                       */
/* Revision: $Id: stream.c,v 5.10 2013/01/17 16:01:06 mccalpin Exp mccalpin $ */
/* Original code developed by John D. McCalpin                           */
/* Programmers: John D. McCalpin                                         */
/*              Joe R. Zagar                                             */
/*                                                                       */
/* This program measures memory transfer rates in MB/s for simple        */
/* computational kernels coded in C.                                     */
/*-----------------------------------------------------------------------*/
/* Copyright 1991-2013: John D. McCalpin                                 */
/*-----------------------------------------------------------------------*/
/* License:                                                              */
/*  1. You are free to use this program and/or to redistribute           */
/*     this program.                                                     */
/*  2. You are free to modify this program for your own use,             */
/*     including commercial use, subject to the publication              */
/*     restrictions in item 3.                                           */
/*  3. You are free to publish results obtained from running this        */
/*     program, or from works that you derive from this program,         */
/*     with the following limitations:                                   */
/*     3a. In order to be referred to as "STREAM benchmark results",     */
/*         published results must be in conformance to the STREAM        */
/*         Run Rules, (briefly reviewed below) published at              */
/*         http://www.cs.virginia.edu/stream/ref.html                    */
/*         and incorporated herein by reference.                         */
/*         As the copyright holder, John McCalpin retains the            */
/*         right to determine conformity with the Run Rules.             */
/*     3b. Results based on modified source code or on runs not in       */
/*         accordance with the STREAM Run Rules must be clearly          */
/*         labelled whenever they are published.  Examples of            */
/*         proper labelling include:                                     */
/*           "tuned STREAM benchmark results"                            */
/*           "based on a variant of the STREAM benchmark code"           */
/*         Other comparable, clear, and reasonable labelling is          */
/*         acceptable.                                                   */
/*     3c. Submission of results to the STREAM benchmark web site        */
/*         is encouraged, but not required.                              */
/*  4. Use of this program or creation of derived works based on this    */
/*     program constitutes acceptance of these licensing restrictions.   */
/*  5. Absolutely no warranty is expressed or implied.                   */
/*-----------------------------------------------------------------------*/

extern "C"
{
#include <runtime/runtime.h>
}

# include <float.h>

# include "array.hpp"
# include "device.hpp"
# include "manager.hpp"

# include <cstdint>
# include <iostream>
# include <memory>
# include <random>

constexpr uint64_t kCacheSize = (128ULL << 20);
constexpr uint64_t kFarMemSize = (4ULL << 30);
constexpr uint32_t kNumGCThreads = 12;
constexpr uint32_t kNumEntries = 10;

/*-----------------------------------------------------------------------
 * INSTRUCTIONS:
 *
 *      1) STREAM requires different amounts of memory to run on different
 *           systems, depending on both the system cache size(s) and the
 *           granularity of the system timer.
 *     You should adjust the value of 'STREAM_ARRAY_SIZE' (below)
 *           to meet *both* of the following criteria:
 *       (a) Each array must be at least 4 times the size of the
 *           available cache memory. I don't worry about the difference
 *           between 10^6 and 2^20, so in practice the minimum array size
 *           is about 3.8 times the cache size.
 *           Example 1: One Xeon E3 with 8 MB L3 cache
 *               STREAM_ARRAY_SIZE should be >= 4 million, giving
 *               an array size of 30.5 MB and a total memory requirement
 *               of 91.5 MB.  
 *           Example 2: Two Xeon E5's with 20 MB L3 cache each (using OpenMP)
 *               STREAM_ARRAY_SIZE should be >= 20 million, giving
 *               an array size of 153 MB and a total memory requirement
 *               of 458 MB.  
 *       (b) The size should be large enough so that the 'timing calibration'
 *           output by the program is at least 20 clock-ticks.  
 *           Example: most versions of Windows have a 10 millisecond timer
 *               granularity.  20 "ticks" at 10 ms/tic is 200 milliseconds.
 *               If the chip is capable of 10 GB/s, it moves 2 GB in 200 msec.
 *               This means the each array must be at least 1 GB, or 128M elements.
 *
 *      Version 5.10 increases the default array size from 2 million
 *          elements to 10 million elements in response to the increasing
 *          size of L3 caches.  The new default size is large enough for caches
 *          up to 20 MB. 
 *      Version 5.10 changes the loop index variables from "register int"
 *          to "ssize_t", which allows array indices >2^32 (4 billion)
 *          on properly configured 64-bit systems.  Additional compiler options
 *          (such as "-mcmodel=medium") may be required for large memory runs.
 *
 *      Array size can be set at compile time without modifying the source
 *          code for the (many) compilers that support preprocessor definitions
 *          on the compile line.  E.g.,
 *                gcc -O -DSTREAM_ARRAY_SIZE=100000000 stream.c -o stream.100M
 *          will override the default size of 10M with a new size of 100M elements
 *          per array.
 */

#ifndef STREAM_ARRAY_SIZE
#   define STREAM_ARRAY_SIZE kNumEntries
#endif

/*  2) STREAM runs each kernel "NTIMES" times and reports the *best* result
 *         for any iteration after the first, therefore the minimum value
 *         for NTIMES is 2.
 *      There are no rules on maximum allowable values for NTIMES, but
 *         values larger than the default are unlikely to noticeably
 *         increase the reported performance.
 *      NTIMES can also be set on the compile line without changing the source
 *         code using, for example, "-DNTIMES=7".
 */

#ifdef NTIMES
#if NTIMES<=1
#   define NTIMES       10
#endif
#endif
#ifndef NTIMES
#   define NTIMES       10
#endif

/*  Users are allowed to modify the "OFFSET" variable, which *may* change the
 *         relative alignment of the arrays (though compilers may change the 
 *         effective offset by making the arrays non-contiguous on some systems). 
 *      Use of non-zero values for OFFSET can be especially helpful if the
 *         STREAM_ARRAY_SIZE is set to a value close to a large power of 2.
 *      OFFSET can also be set on the compile line without changing the source
 *         code using, for example, "-DOFFSET=56".
 */

#ifndef OFFSET
#   define OFFSET       0
#endif

/*
 *      3) Compile the code with optimization.  Many compilers generate
 *       unreasonably bad code before the optimizer tightens things up.  
 *     If the results are unreasonably good, on the other hand, the
 *       optimizer might be too smart for me!
 *
 *     For a simple single-core version, try compiling with:
 *            cc -O stream.c -o stream
 *     This is known to work on many, many systems....
 *
 *     To use multiple cores, you need to tell the compiler to obey the OpenMP
 *       directives in the code.  This varies by compiler, but a common example is
 *            gcc -O -fopenmp stream.c -o stream_omp
 *       The environment variable OMP_NUM_THREADS allows runtime control of the 
 *         number of threads/cores used when the resulting "stream_omp" program
 *         is executed.
 *
 *     To run with single-precision variables and arithmetic, simply add
 *         -DSTREAM_TYPE=float
 *     to the compile line.
 *     Note that this changes the minimum array sizes required --- see (1) above.
 *
 *     The preprocessor directive "TUNED" does not do much -- it simply causes the 
 *       code to call separate functions to execute each kernel.  Trivial versions
 *       of these functions are provided, but they are *not* tuned -- they just 
 *       provide predefined interfaces to be replaced with tuned code.
 *
 *
 *      4) Optional: Mail the results to mccalpin@cs.virginia.edu
 *         Be sure to include info that will help me understand:
 *              a) the computer hardware configuration (e.g., processor model, memory type)
 *              b) the compiler name/version and compilation flags
 *      c) any run-time information (such as OMP_NUM_THREADS)
 *              d) all of the output from the test case.
 *
 * Thanks!
 *
 *-----------------------------------------------------------------------*/


# define HLINE "-------------------------------------------------------------\n"

# ifndef MIN2
# define MIN2(x,y) ((x)<(y)?(x):(y))
# endif
# ifndef MAX2
# define MAX2(x,y) ((x)>(y)?(x):(y))
# endif


#ifndef STREAM_TYPE
#define STREAM_TYPE double
#endif


//static auto a, b, c;

static double   avgtime[4] = {0}, maxtime[4] = {0},
                mintime[4] = {FLT_MAX,FLT_MAX,FLT_MAX,FLT_MAX};

static char     *label[4] = {"Copy:      ", "Scale:     ",
    "Add:       ", "Triad:     "};

static double   bytes[4] = {
    2 * sizeof(STREAM_TYPE) * STREAM_ARRAY_SIZE,
    2 * sizeof(STREAM_TYPE) * STREAM_ARRAY_SIZE,
    3 * sizeof(STREAM_TYPE) * STREAM_ARRAY_SIZE,
    3 * sizeof(STREAM_TYPE) * STREAM_ARRAY_SIZE
    };


void do_work(far_memory::FarMemManager *manager);

extern double mysecond();
extern void checkSTREAMresults();

#ifdef TUNED
extern void tuned_STREAM_Copy();
extern void tuned_STREAM_Scale(STREAM_TYPE scalar);
extern void tuned_STREAM_Add();
extern void tuned_STREAM_Triad(STREAM_TYPE scalar);
#endif

#ifdef _OPENMP
extern int omp_get_num_threads();
#endif

void _main(void *arg)
{
        std::cout << HLINE;
        std::cout << "Cache Size (in GiB): " << ( kCacheSize / 1024.0/1024.0/1024.0) << "\n";
        std::cout << "Far Memory Size (in GiB): " << ( kFarMemSize / 1024.0/1024.0/1024.0)  << "\n";
        std::cout << "Number of Threads: " << kNumGCThreads << "\n";
        std::cout << "Number of Elements in array: " << kNumEntries << "\n";

        auto manager = std::unique_ptr<far_memory::FarMemManager>(far_memory::FarMemManagerFactory::build(
                                        kCacheSize, kNumGCThreads, new far_memory::FakeDevice(kFarMemSize)));
        do_work(manager.get());
}

int main(int argc, char *argv[])
{
        int ret;

        if (argc < 2) {
                std::cerr << "usage: [cfg_file]" << std::endl;
                return -EINVAL;
        }

        ret = runtime_init(argv[1], _main, NULL);
        if (ret) {
                std::cerr << "failed to start runtime" << std::endl;
                return ret;
        }
        return 0;
}

void do_work(far_memory::FarMemManager *manager)
{
        int quantum, checktick();
        int BytesPerWord;
        int k;
        ssize_t j;
        STREAM_TYPE scalar;
        double t, times[4][NTIMES];

        auto a = manager->allocate_array<STREAM_TYPE, STREAM_ARRAY_SIZE+OFFSET>(),
        b = manager->allocate_array<STREAM_TYPE, STREAM_ARRAY_SIZE+OFFSET>(),
        c = manager->allocate_array<STREAM_TYPE, STREAM_ARRAY_SIZE+OFFSET>();


        /* --- SETUP --- determine precision and check timing --- */
        BytesPerWord = sizeof(STREAM_TYPE);

        std::cout << HLINE;
        std::cout << "STREAM version $Revision: 5.10 $\n";
        std::cout << HLINE;
        std::cout << "This system uses " << BytesPerWord << " bytes per array element.\n";
        std::cout << HLINE;

        #ifdef N
        std::cout << "*****  WARNING: ******\n";
        std::cout << "      It appears that you set the preprocessor variable N when compiling this code.\n";
        std::cout << "      This version of the code uses the preprocessor variable STREAM_ARRAY_SIZE to control the array size\n";
        std::cout << "      Reverting to default value of STREAM_ARRAY_SIZE=" << (unsigned long long) STREAM_ARRAY_SIZE << "\n";
        std::cout << "*****  WARNING: ******\n";
        #endif

        std::cout << std::fixed << std::setprecision(1) <<  "Array size = " << (unsigned long long) STREAM_ARRAY_SIZE << " (elements), Offset = " << OFFSET << " (elements)\n";
        std::cout << std::fixed << std::setprecision(1) << "Memory per array = " << BytesPerWord * ( (double) STREAM_ARRAY_SIZE / 1024.0/1024.0) << std::setprecision(1) <<
                " MiB (= " << BytesPerWord * ( (double) STREAM_ARRAY_SIZE / 1024.0/1024.0/1024.0) << " GiB).\n";
        std::cout << std::fixed << std::setprecision(1) << "Total memory required = " << std::setprecision(1) << (3.0 * BytesPerWord) * ( (double) STREAM_ARRAY_SIZE / 1024.0/1024.) <<
                " MiB (= " << (3.0 * BytesPerWord) * ( (double) STREAM_ARRAY_SIZE / 1024.0/1024./1024.) << " GiB).\n";
        std::cout << std::fixed << std::setprecision(1) << "Each kernel will be executed " << NTIMES << " times.\n";
        std::cout << std::fixed << std::setprecision(1) << " The *best* time for each kernel (excluding the first iteration)\n";
        std::cout << std::fixed << std::setprecision(1) << " will be used to compute the reported bandwidth.\n";

        #ifdef _OPENMP
        std::cout << HLINE;
        #pragma omp parallel
        {
                #pragma omp master
                {
                        k = omp_get_num_threads();
                        std::cout << "Number of Threads requested = " << k << "\n";
                }
        }
        #endif

        #ifdef _OPENMP
        k = 0;
        #pragma omp parallel
        #pragma omp atomic
        k++;
        std::cout << "Number of Threads counted = " << k << "\n";
        #endif

        /* Get initial value for system clock. */
        #pragma omp parallel for
        for (j=0; j<STREAM_ARRAY_SIZE; j++)
        {
                far_memory::DerefScope scope;
                a.at_mut(scope, j) = 1.0;
                b.at_mut(scope, j) = 2.0;
                c.at_mut(scope, j) = 0.0;
        }

        printf(HLINE);

        if  ( (quantum = checktick()) >= 1) {
                std::cout << "Your clock granularity/precision appears to be "
                << quantum << " microseconds.\n";
        }
        else {
                std::cout << "Your clock granularity appears to be "
                "less than one microsecond.\n";
                quantum = 1;
        }

        t = mysecond();
        #pragma omp parallel for
        for (j = 0; j < STREAM_ARRAY_SIZE; j++)
        {
                far_memory::DerefScope scope;
                a.at_mut(scope, j) = 2.0E0 * a.at(scope, j);
        }
        t = 1.0E6 * (mysecond() - t);

        std::cout << "Each test below will take on the order"
                " of " << (int) t << " microseconds.\n";
        std::cout << "   (= " << (int) (t/quantum) << " clock ticks)\n";
        std::cout << "Increase the size of the arrays if this shows that\n";
        std::cout << "you are not getting at least 20 clock ticks per test.\n";

        printf(HLINE);

        std::cout << "WARNING -- The above is only a rough guideline.\n";
        std::cout << "For best results, please be sure you know the\n";
        std::cout << "precision of your system timer.\n";
        std::cout << HLINE;

        /*      --- MAIN LOOP --- repeat test cases NTIMES times --- */

        scalar = 3.0;
        for (k=0; k<NTIMES; k++)
        {
                times[0][k] = mysecond();
                #ifdef TUNED
                tuned_STREAM_Copy();
                #else
                #pragma omp parallel for
                for (j=0; j<STREAM_ARRAY_SIZE; j++)
                {
                        far_memory::DerefScope scope;
                        c.at_mut(scope,j) = a.at(scope, j);
                }
                #endif
                times[0][k] = mysecond() - times[0][k];

                times[1][k] = mysecond();
                #ifdef TUNED
                tuned_STREAM_Scale(scalar);
                #else
                #pragma omp parallel for
                for (j=0; j<STREAM_ARRAY_SIZE; j++){
                    far_memory::DerefScope scope;
                    b.at_mut(scope, j)= scalar*c.at(scope, j);
                }
                #endif
                times[1][k] = mysecond() - times[1][k];

                times[2][k] = mysecond();
                #ifdef TUNED
                tuned_STREAM_Add();
                #else
                #pragma omp parallel for
                for (j=0; j<STREAM_ARRAY_SIZE; j++) {
                        far_memory::DerefScope scope;
                        c.at_mut(scope, j) = a.at(scope, j) + b.at(scope, j);
                }
                #endif
                times[2][k] = mysecond() - times[2][k];

                times[3][k] = mysecond();
                #ifdef TUNED
                tuned_STREAM_Triad(scalar);
                #else
                #pragma omp parallel for
                for (j=0; j<STREAM_ARRAY_SIZE; j++) {
                    far_memory::DerefScope scope;
                    a.at_mut(scope, j) = b.at(scope, j)+scalar*c.at(scope, j);
                }
                #endif
                times[3][k] = mysecond() - times[3][k];
        }

        /*      --- SUMMARY --- */

        for (k=1; k<NTIMES; k++) { /* note -- skip first iteration */
                for (j=0; j<4; j++) {
                        avgtime[j] = avgtime[j] + times[j][k];
                        mintime[j] = MIN2(mintime[j], times[j][k]);
                        maxtime[j] = MAX2(maxtime[j], times[j][k]);
                }
        }

        std::cout << "Function    Best Rate MB/s  Avg time     Min time     Max time\n";
        int precision = 6;
        for (j=0; j<4; j++)
        {
                avgtime[j] = avgtime[j]/(double)(NTIMES-1);

                std::cout << std::fixed << std::left << std::setprecision(1) << std::setw(12) << label[j] <<
                        std::setw(16) << 1.0E-06 * bytes[j]/mintime[j] << std::setprecision(precision) <<
                        std::setw(13) << avgtime[j]  <<  std::setprecision(precision) <<
                        std::setw(13) << mintime[j] << std::setprecision(precision) <<
                        std::setw(20) << maxtime[j] <<  std::setprecision(precision) << "\n";
        }
        std::cout << HLINE;

        /* --- Check Results --- */

        STREAM_TYPE aj,bj,cj;
        scalar = 0;
        STREAM_TYPE aSumErr,bSumErr,cSumErr;
        STREAM_TYPE aAvgErr,bAvgErr,cAvgErr;

        //far_memory::DerefScope scope0;

        double epsilon;
        j = 0;
        int     ierr,err;
        k = 0;

        /* reproduce initialization */
        aj = 1.0;
        bj = 2.0;
        cj = 0.0;
        /* a[] is modified during timing check */
        aj = 2.0E0 * aj;
        /* now execute timing loop */
        scalar = 3.0;
        for (k=0; k<NTIMES; k++)
        {
            cj = aj;
            bj = scalar*cj;
            cj = aj+bj;
            aj = bj+scalar*cj;
        }
        aSumErr = 0.0;
        bSumErr = 0.0;
        cSumErr = 0.0;
        for (j=0; j<STREAM_ARRAY_SIZE; j++)
        {
                far_memory::DerefScope scope;
                aSumErr += abs(a.at(scope, j) - aj);
                bSumErr += abs(b.at(scope, j) - bj);
                cSumErr += abs(c.at(scope, j) - cj);
                //if (j == 417) printf("Index 417: c[j]: %f, cj: %f\n",c[j],cj);        // MCCALPIN
        }
        aAvgErr = aSumErr / (STREAM_TYPE) STREAM_ARRAY_SIZE;
        bAvgErr = bSumErr / (STREAM_TYPE) STREAM_ARRAY_SIZE;
        cAvgErr = cSumErr / (STREAM_TYPE) STREAM_ARRAY_SIZE;

        if (sizeof(STREAM_TYPE) == 4) {
                epsilon = 1.e-6;
        }
        else if (sizeof(STREAM_TYPE) == 8) {
                epsilon = 1.e-13;
        }
        else {
                std::cout << "WEIRD: sizeof(STREAM_TYPE) = " << sizeof(STREAM_TYPE) << "\n";
                epsilon = 1.e-6;
        }

        err = 0;
        if (abs(aAvgErr/aj) > epsilon) {
                err++;
                std::cout << "Failed Validation on array a[], AvgRelAbsErr > epsilon (" << epsilon << ")\n";
                std::cout << "     Expected Value: " << aj <<
                                ", AvgAbsErr: " << aAvgErr <<
                                ", AvgRelAbsErr: " << abs(aAvgErr)/aj << "\n";
                ierr = 0;
                for (j=0; j<STREAM_ARRAY_SIZE; j++)
                {
                        far_memory::DerefScope scope1;
                        if (abs(a.at(scope1, j)/aj-1.0) > epsilon) {
                                ierr++;
                                #ifdef VERBOSE
                                if (ierr < 10) {
                                        far_memory::DerefScope scope2;
                                        std::cout << "         array a: index: " << j <<
                                                ", expected: " << aj <<
                                                ", observed: " << a.at(scope2, j) <<
                                                ", relative error: " << abs((aj-a.at(scope2, j))/aAvgErr)<< "\n";
                                }
                                #endif
                        }
                }
                std::cout << "     For array a[], " << ierr << " errors were found.\n";
        }
        if (abs(bAvgErr/bj) > epsilon) {
                err++;
                std::cout << "Failed Validation on array b[], AvgRelAbsErr > epsilon (" << epsilon << ")\n";
                std::cout << "     Expected Value: " << bj <<
                                ", AvgAbsErr: " << bAvgErr <<
                                ", AvgRelAbsErr: " << abs(bAvgErr)/bj << "\n";
                std::cout << "     AvgRelAbsErr > Epsilon (" << epsilon << ")\n";
                ierr = 0;
                for (j=0; j<STREAM_ARRAY_SIZE; j++)
                {
                        far_memory::DerefScope scope1;
                        if (abs(b.at(scope1, j)/bj-1.0) > epsilon) {
                                ierr++;
                                #ifdef VERBOSE
                                if (ierr < 10) {
                                        far_memory::DerefScope scope2;
                                        std::cout << "         array b: index: " << j <<
                                                ", expected: " << bj <<
                                                ", observed: " << b.at(scope2, j) <<
                                                ", relative error: " << abs((bj-b.at(scope2, j))/bAvgErr) << "\n";
                                }
                                #endif
                        }
                }
                std::cout << "     For array b[], " << ierr << " errors were found.\n";
        }
        if (abs(cAvgErr/cj) > epsilon) {
                err++;
                std::cout << "Failed Validation on array c[], AvgRelAbsErr > epsilon (" << epsilon << ")\n";
                std::cout << "     Expected Value: " << cj <<
                                ", AvgAbsErr: " << cAvgErr <<
                                ", AvgRelAbsErr: " << abs(cAvgErr)/cj << "\n";
                std::cout << "     AvgRelAbsErr > Epsilon (" << epsilon << ")\n";
                ierr = 0;
                for (j=0; j<STREAM_ARRAY_SIZE; j++)
                {
                        far_memory::DerefScope scope1;
                        if (abs(c.at(scope1, j)/cj-1.0) > epsilon) {
                                ierr++;
                                #ifdef VERBOSE
                                if (ierr < 10) {
                                        far_memory::DerefScope scope2;
                                        std::cout << "         array c: index: " << j <<
                                                ", expected: " << cj <<
                                                ", observed: " << c.at(scope2, j) <<
                                                ", relative error: " << abs(cj-c.at(scope2, j))/cAvgErr << "\n";
                                }
                                #endif
                        }
                }
                std::cout << "     For array c[], " << ierr << " errors were found.\n";
        }
        if (err == 0) {
                std::cout << "Solution Validates: avg error less than " << epsilon << " on all three arrays\n";
        }
        #ifdef VERBOSE
        std::cout << "Results Validation Verbose Results: \n";
        std::cout << "    Expected a(1), b(1), c(1): " << aj << " " << bj << " " << cj << " \n";
        std::cout << "    Observed a(1), b(1), c(1): " << a.at(scope0,1) << " " << b.at(scope0,1) << " " << c.at(scope0,1) << " \n";
        std::cout << "    Rel Errors on a, b, c:     " << abs(aAvgErr/aj) << " " << abs(bAvgErr/bj) << " " << abs(cAvgErr/cj) << " \n";
        #endif
        std::cout << HLINE;
}

# define M 20

int checktick()
{
        int     i, minDelta, Delta;
        double  t1, t2, timesfound[M];

        /*  Collect a sequence of M unique time values from the system. */

        for (i = 0; i < M; i++)
        {
                t1 = mysecond();
                while( ((t2=mysecond()) - t1) < 1.0E-6 );
                timesfound[i] = t1 = t2;
        }

        /*
        * Determine the minimum difference between these M values.
        * This result will be our estimate (in microseconds) for the
        * clock granularity.
        */

        minDelta = 1000000;
        for (i = 1; i < M; i++)
        {
                Delta = (int)( 1.0E6 * (timesfound[i]-timesfound[i-1]));
                minDelta = MIN2(minDelta, MAX2(Delta,0));
        }

        return(minDelta);
}

/* A gettimeofday routine to give access to the wall
   clock timer on most UNIX-like systems.  */

#include <sys/time.h>

double mysecond()
{
        struct timeval tp;
        struct timezone tzp;
        int i;

        i = gettimeofday(&tp,&tzp);
        return ( (double) tp.tv_sec + (double) tp.tv_usec * 1.e-6 );
}

#ifndef abs
#define abs(a) ((a) >= 0 ? (a) : -(a))
#endif

#ifdef TUNED
/* stubs for "tuned" versions of the kernels */
void tuned_STREAM_Copy()
{
        ssize_t j;
        #pragma omp parallel for
        for (j=0; j<STREAM_ARRAY_SIZE; j++)
        {
            DerefScope scope;
            c.at_mut(scope, j) = a.at(scope, j);
        }
}

void tuned_STREAM_Scale(STREAM_TYPE scalar)
{
        ssize_t j;
        #pragma omp parallel for
        for (j=0; j<STREAM_ARRAY_SIZE; j++)
        {
                DerefScope scope;
                b.at_mut(scope, j) = scalar*c.at(scope, j);
        }
}

void tuned_STREAM_Add()
{
        ssize_t j;
        #pragma omp parallel for
        for (j=0; j<STREAM_ARRAY_SIZE; j++)
        {
                DerefScope scope;
                c.at_mut(scope, j) = a.at(scope, j)+b.at(scope, j);
        }
}

void tuned_STREAM_Triad(STREAM_TYPE scalar)
{
        ssize_t j;
        #pragma omp parallel for
        for (j=0; j<STREAM_ARRAY_SIZE; j++)
        {
                DerefScope scope;
                a.at_mut(scope, j) = b.at(scope,j) + scalar * c.at(scope, j);
        }
}
/* end of stubs for the "tuned" versions of the kernels */
#endif
