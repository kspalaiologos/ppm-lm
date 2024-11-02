/*  TANGELO file compressor 1.0 based on PAQ8 by Matt Mahoney
    Release by Jan Ondrus, June. 17, 2013
 
    Copyright (C) 2013 Matt Mahoney, Serge Osnach, Alexander Ratushnyak,
    Bill Pettis, Przemyslaw Skibinski, Matthew Fite, wowtiger, Andrew Paterson,
    Jan Ondrus, Andreas Morphis, Pavel L. Holoborodko, KZ., Simon Berger,
    Neill Corlett, Marwijn Hessel
    
    LICENSE

    This program is free software; you can redistribute it and/or
    modify it under the terms of the GNU General Public License as
    published by the Free Software Foundation; either version 2 of
    the License, or (at your option) any later version.

    This program is distributed in the hope that it will be useful, but
    WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
    General Public License for more details at
    Visit <http://www.gnu.org/copyleft/gpl.html>.

Usage: TANGELO <command> <infile> <outfile>

<Commands>
 c       Compress
 d       Decompress

Recommended compiler command for MINGW g++:
 g++ tangelo.cpp -Wall -Wextra -O3 -s -march=pentium4 -mtune=pentiumpro -fomit-frame-pointer -o tangelo.exe

*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <ctype.h>

#define MEM (0x10000 << 7)

typedef unsigned char U8;
typedef unsigned short U16;
typedef unsigned int U32;
typedef unsigned long long U64;

inline int min(int a, int b) { return a < b ? a : b; }
inline int max(int a, int b) { return a < b ? b : a; }

inline U32 ilog2(U32 x) {
  if(x!=0)x=31-__builtin_clz(x);
  return x;
}

// Used memory and time
class Stats {
  int memused, maxmem;
  clock_t start_time;
public:
  void alloc(int n) {
    memused += n;
    if (memused > maxmem) maxmem = memused;
  }
  Stats():memused(0), maxmem(0) {
    start_time = clock();
  }
  void print() const {
    printf("Time %1.2f sec, used %d bytes of memory\n",
           double(clock() - start_time) / CLOCKS_PER_SEC, maxmem);
  }
} stats;

// Array
template<class T,int ALIGN=0> class Array {
private:
  int n, reserved;
  char *ptr;
  T *data;
  void create(int i);
public:
  explicit Array(int i = 0) { create(i); }
  ~Array();
  T&operator[](int i) { return data[i]; }
  const T&operator[](int i) const { return data[i]; }
  int size() const { return n; }
private:
  Array(const Array&);
  Array&operator=(const Array&);
};
template<class T,int ALIGN> void Array<T,ALIGN>::create(int i) {
  n = reserved = i;
  if (i <= 0) {
    data = 0, ptr = 0;
    return;
  }
  const int sz = ALIGN + n * sizeof(T);
  stats.alloc(sz);
  ptr = new char[sz];
  if (!ptr) throw "Out of memory";
  data = (ALIGN ? (T*)(ptr + ALIGN - (((long)ptr) & (ALIGN - 1))):(T*)ptr);
}
template<class T,int ALIGN> Array<T,ALIGN>::~Array() {
  stats.alloc(-ALIGN - n * sizeof(T));
  // free(ptr);
}

// Random generator
static class Random {
  Array<U32> table;
  int i;
public:
  Random():table(64) {
    table[0] = 123456789;
    table[1] = 987654321;
    for (int j = 0; j < 62; j++) {
      table[j + 2] = table[j + 1] * 11 + table[j] * 23 / 16;
    }
    i = 0;
  }
  U32 operator()() {
    return ++i, table[i & 63] = table[(i - 24) & 63] ^ table[(i - 55) & 63];
  }
} rnd;

// Buffer - array of n last bytes
static int pos;
class Buf {
  Array<U8> b;
public:
  Buf(int i = 0):b(i) {}
  U8& operator[](int i) {
    return b[i & (b.size() - 1)];
  }
  int operator()(int i) const {
    return b[(pos - i) & (b.size() - 1)];
  }
  int size() const{
    return b.size();
  }
};

// Global variables
static FILE* infile, *outfile;
static int y = 0, c0 = 1, bpos = 0;
static U32 c4 = 0;
static U8 grp0; // Quantized partial byte as ASCII group
static Buf buf(MEM * 8);

// Logarithm
static class Ilog {
  Array<U8> t;
public:
  int operator()(U16 x) const { return t[x]; }
  Ilog():t(65536) {
    U32 x = 14155776;
    for (int i = 2; i < 65536; ++i) {
      x += 774541002 / (i * 2 - 1);
      t[i] = x >> 24;
  }
}
} ilog;

// Precomputed table for pt(i) = 16384 / (i + i + 3)
static class Ptable {
  Array<int> t;
public:
  int operator()(U16 x) const { return t[x]; }
  Ptable():t(1024) {
    for (int i = 0; i < 1024; ++i) {
      t[i] = 16384 / (i + i + 3);
    }
  }
} pt;

inline int llog(U32 x) {
  if (x>=0x1000000)
    return 256+ilog(x>>16);
  else if (x>=0x10000)
    return 128+ilog(x>>8);
  else
    return ilog(x);
}

// Hash

/////////////////////////////////////////////////////////////////////////////////////////
#include <stdint.h>
#define PHI32 UINT32_C(0x9E3779B9) // 2654435769
#define PHI64 UINT64_C(0x9E3779B97F4A7C15) // 11400714819323198485
#define MUL64_1  UINT64_C(0x993DDEFFB1462949)
#define MUL64_2  UINT64_C(0xE9C91DC159AB0D2D)
#define MUL64_3  UINT64_C(0x83D6A14F1B0CED73)
#define MUL64_4  UINT64_C(0xA14F1B0CED5A841F)
#define MUL64_5  UINT64_C(0xC0E51314A614F4EF) 
#define MUL64_6  UINT64_C(0xDA9CC2600AE45A27)
#define MUL64_7  UINT64_C(0x826797AA04A65737)
#define MUL64_8  UINT64_C(0x2375BE54C41A08ED)
#define MUL64_9  UINT64_C(0xD39104E950564B37)
#define MUL64_10 UINT64_C(0x3091697D5E685623)
#define MUL64_11 UINT64_C(0x20EB84EE04A3C7E1)
#define MUL64_12 UINT64_C(0xF501F1D0944B2383)
#define MUL64_13 UINT64_C(0xE3E4E8AA829AB9B5)

static inline U32 finalize64(const U64 hash, const int hashbits) {
  return U32(hash>>(64-hashbits));
}

static inline U64 checksum64(const U64 hash, const int hashbits, const int checksumbits) {
  return hash>>(64-hashbits-checksumbits); 
}

static inline U64 hash(const U64 x0) {
  return (x0+1)*PHI64;
}
static inline U64 hash(const U64 x0, const U64 x1) {
  return (x0+1)*PHI64   + (x1+1)*MUL64_1;
}
static inline U64 hash(const U64 x0, const U64 x1, const U64 x2) {
  return (x0+1)*PHI64   + (x1+1)*MUL64_1 + (x2+1)*MUL64_2;
}
static inline U64 hash(const U64 x0, const U64 x1, const U64 x2, const U64 x3) {
  return (x0+1)*PHI64   + (x1+1)*MUL64_1 + (x2+1)*MUL64_2 +
         (x3+1)*MUL64_3;
}
static inline U64 hash(const U64 x0, const U64 x1, const U64 x2, const U64 x3, const U64 x4) {
  return (x0+1)*PHI64   + (x1+1)*MUL64_1 + (x2+1)*MUL64_2 +
         (x3+1)*MUL64_3 + (x4+1)*MUL64_4;
}
static inline U64 hash(const U64 x0, const U64 x1, const U64 x2, const U64 x3, const U64 x4,
                  const U64 x5) {
  return (x0+1)*PHI64   + (x1+1)*MUL64_1 + (x2+1)*MUL64_2 +
         (x3+1)*MUL64_3 + (x4+1)*MUL64_4 + (x5+1)*MUL64_5;
}
static inline U64 hash(const U64 x0, const U64 x1, const U64 x2, const U64 x3, const U64 x4,
                  const U64 x5, const U64 x6) {
  return (x0+1)*PHI64   + (x1+1)*MUL64_1 + (x2+1)*MUL64_2 +
         (x3+1)*MUL64_3 + (x4+1)*MUL64_4 + (x5+1)*MUL64_5 +
         (x6+1)*MUL64_6;
}
static inline U64 hash(const U64 x0, const U64 x1, const U64 x2, const U64 x3, const U64 x4,
                  const U64 x5, const U64 x6, const U64 x7) {
  return (x0+1)*PHI64   + (x1+1)*MUL64_1 + (x2+1)*MUL64_2 +
         (x3+1)*MUL64_3 + (x4+1)*MUL64_4 + (x5+1)*MUL64_5 +
         (x6+1)*MUL64_6 + (x7+1)*MUL64_7;
}
static inline U64 combine64(const U64 seed, const U64 x) {
  return hash(seed+x);
}

// State table
static const U8 State_table[256][2] = {
  {1,2},{3,5},{4,6},{7,10},{8,12},{9,13},{11,14},{15,19},{16,23},{17,24},
  {18,25},{20,27},{21,28},{22,29},{26,30},{31,33},{32,35},{32,35},{32,35},
  {32,35},{34,37},{34,37},{34,37},{34,37},{34,37},{34,37},{36,39},{36,39},
  {36,39},{36,39},{38,40},{41,43},{42,45},{42,45},{44,47},{44,47},{46,49},
  {46,49},{48,51},{48,51},{50,52},{53,43},{54,57},{54,57},{56,59},{56,59},
  {58,61},{58,61},{60,63},{60,63},{62,65},{62,65},{50,66},{67,55},{68,57},
  {68,57},{70,73},{70,73},{72,75},{72,75},{74,77},{74,77},{76,79},{76,79},
  {62,81},{62,81},{64,82},{83,69},{84,71},{84,71},{86,73},{86,73},{44,59},
  {44,59},{58,61},{58,61},{60,49},{60,49},{76,89},{76,89},{78,91},{78,91},
  {80,92},{93,69},{94,87},{94,87},{96,45},{96,45},{48,99},{48,99},{88,101},
  {88,101},{80,102},{103,69},{104,87},{104,87},{106,57},{106,57},{62,109},
  {62,109},{88,111},{88,111},{80,112},{113,85},{114,87},{114,87},{116,57},
  {116,57},{62,119},{62,119},{88,121},{88,121},{90,122},{123,85},{124,97},
  {124,97},{126,57},{126,57},{62,129},{62,129},{98,131},{98,131},{90,132},
  {133,85},{134,97},{134,97},{136,57},{136,57},{62,139},{62,139},{98,141},
  {98,141},{90,142},{143,95},{144,97},{144,97},{68,57},{68,57},{62,81},{62,81},
  {98,147},{98,147},{100,148},{149,95},{150,107},{150,107},{108,151},{108,151},
  {100,152},{153,95},{154,107},{108,155},{100,156},{157,95},{158,107},{108,159},
  {100,160},{161,105},{162,107},{108,163},{110,164},{165,105},{166,117},
  {118,167},{110,168},{169,105},{170,117},{118,171},{110,172},{173,105},
  {174,117},{118,175},{110,176},{177,105},{178,117},{118,179},{110,180},
  {181,115},{182,117},{118,183},{120,184},{185,115},{186,127},{128,187},
  {120,188},{189,115},{190,127},{128,191},{120,192},{193,115},{194,127},
  {128,195},{120,196},{197,115},{198,127},{128,199},{120,200},{201,115},
  {202,127},{128,203},{120,204},{205,115},{206,127},{128,207},{120,208},
  {209,125},{210,127},{128,211},{130,212},{213,125},{214,137},{138,215},
  {130,216},{217,125},{218,137},{138,219},{130,220},{221,125},{222,137},
  {138,223},{130,224},{225,125},{226,137},{138,227},{130,228},{229,125},
  {230,137},{138,231},{130,232},{233,125},{234,137},{138,235},{130,236},
  {237,125},{238,137},{138,239},{130,240},{241,125},{242,137},{138,243},
  {130,244},{245,135},{246,137},{138,247},{140,248},{249,135},{250,69},{80,251},
  {140,252},{249,135},{250,69},{80,251},{140,252}};
              
static int squash(int d) {
  static const int t[33] = {
    1,2,3,6,10,16,27,45,73,120,194,310,488,747,1101,1546,2047,2549,2994,3348,
    3607,3785,3901,3975,4022,4050,4068,4079,4085,4089,4092,4093,4094};
  if (d > 2047) return 4095;
  if (d < -2047) return 0;
  int w = d & 127;
  d = (d >> 7) + 16;
  return (t[d] * (128 - w) + t[(d + 1)] * w + 64) >> 7;
}

class Stretch {
  Array<short> t;
public:
  int operator()(int x) const { return t[x]; }
  Stretch():t(4096) {
    int j = 0;
    for (int x = -2047; x <= 2047; ++x) {
      int i = squash(x);
      while (j <= i) t[j++] = x;
    }
    t[4095] = 2047;
  }
} stretch;


#if !defined(__GNUC__)
#if (2 == _M_IX86_FP)
# define __SSE2__
#endif
#endif
#if defined(__SSE2__)
#include <emmintrin.h>

static int dot_product (const short* const t, const short* const w, int n) {
  __m128i sum = _mm_setzero_si128 ();
  while ((n -= 8) >= 0) {
    __m128i tmp = _mm_madd_epi16 (*(__m128i *) &t[n], *(__m128i *) &w[n]);
    tmp = _mm_srai_epi32 (tmp, 8);
    sum = _mm_add_epi32 (sum, tmp);
  }
  sum = _mm_add_epi32 (sum, _mm_srli_si128 (sum, 8));
  sum = _mm_add_epi32 (sum, _mm_srli_si128 (sum, 4));
  return _mm_cvtsi128_si32 (sum);
}

static void train (const short* const t, short* const w, int n, const int e) {
  if (e) {
    const __m128i one = _mm_set1_epi16 (1);
    const __m128i err = _mm_set1_epi16 (short(e));
    while ((n -= 8) >= 0) {
      __m128i tmp = _mm_adds_epi16 (*(__m128i *) &t[n], *(__m128i *) &t[n]);
      tmp = _mm_mulhi_epi16 (tmp, err);
      tmp = _mm_adds_epi16 (tmp, one);
      tmp = _mm_srai_epi16 (tmp, 1);
      tmp = _mm_adds_epi16 (tmp, *(__m128i *) &w[n]);
      *(__m128i *) &w[n] = tmp;
    }
  }
}
#else

static int dot_product (const short* const t, const short* const w, int n) {
  int sum = 0;
  while ((n -= 2) >= 0) {
    sum += (t[n] * w[n] + t[n + 1] * w[n + 1]) >> 8;
  }
  return sum;
}

static void train (const short* const t, short* const w, int n, const int err) {
  if (err) {
    while ((n -= 1) >= 0) {
      int wt = w[n] + ((((t[n] * err * 2) >> 16) + 1) >> 1);
      if (wt < -32768) {
        w[n] = -32768;
      } else if (wt > 32767) {
        w[n] = 32767;
      } else {
        w[n] = wt;
      }
    }
  }
}
#endif

// Mixer - combines models using neural networkss
class Mixer {
  const int N, M, S;
  Array<short,16> tx, wx;
  Array<int> cxt, pr;
  int ncxt, base, nx;
  Mixer *mp;
public:
  Mixer(int n, int m, int s = 1, int w = 0):
  N((n + 7) & -8), M(m), S(s), tx(N), wx(N * M),
  cxt(S), pr(S), ncxt(0), base(0), nx(0), mp(0) {
    for (int i = 0; i < S; ++i) {
      pr[i] = 2048;
    }
    for (int j = 0; j < N * M; ++j) {
      wx[j] = w;
    }
    if (S > 1) mp = new Mixer(S, 1, 1);
  }
  void update() {
    for (int i = 0; i < ncxt; ++i) {
      int err = ((y << 12) - pr[i]) * 7;
      train(&tx[0], &wx[cxt[i] * N], nx, err);
    }
    nx = base = ncxt = 0;
  }
  void add(int x) { tx[nx++] = x; }
  void set(int cx, int range) {
    cxt[ncxt++] = base + cx;
    base += range;
  }
  int p() {
    while (nx & 7) tx[nx++] = 0;
    if (mp) {
      mp->update();
      for (int i = 0; i < ncxt; ++i) {
        pr[i] = squash(dot_product(&tx[0], &wx[cxt[i] * N], nx) >> 5);
        mp->add(stretch(pr[i]));
      }
      mp->set(0, 1);
      return mp->p();
    } else {
      return pr[0] = squash(dot_product(&tx[0], &wx[0], nx) >> 8);
    }
  }
  ~Mixer() { /* delete mp; */ }
};

// APM - stationary map combining a context and an input probability.
class APM {
  int index;
  const int N;
  Array<U16> t;
public:
  APM(int n):index(0), N(n), t(n * 33) {
    for (int i = 0; i < N; ++i) {
      for (int j = 0; j < 33; ++j) {
        t[i * 33 + j] = i == 0 ? squash((j - 16) * 128) * 16 : t[j];
      }
    }
  }
  int p(int pr = 2048, int cxt = 0, int rate = 7) {
    pr = stretch(pr);
    int g = (y << 16) + (y << rate) - y - y;
    t[index] += (g - t[index]) >> rate;
    t[index + 1] += (g - t[index + 1]) >> rate;
    const int w = pr & 127;
    index = ((pr + 2048) >> 7) + cxt * 33;
    return(t[index] * (128 - w) + t[index + 1] * w) >> 11;
  }
};

//  StateMap - maps a context to a probability               
class StateMap {
protected:
  const int N;
  int cxt;
  Array<U32> t;  
public:
  StateMap(int n = 256):N(n), cxt(0), t(n) {
    for (int i = 0; i < N; ++i) {
      t[i] = 1 << 31;
    }
  }
  int p(int cx) {
    U32 *p = &t[cxt], p0 = p[0];
    int n = p0 & 1023, pr = p0 >> 10;
    if (n < 1023) ++p0;
    else p0 = (p0 & 0xfffffc00) | 1023;
    p0 += (((y << 22) - pr) >> 3) * pt(n) & 0xfffffc00;
    p[0] = p0;
    return t[cxt = cx] >> 20;
  }
};

class SmallStationaryContextMap {
  Array<U16> t;
  int cxt;
  U16 *cp;
public:
  SmallStationaryContextMap(int m):t(m / 2), cxt(0) {
    for (int i = 0; i < t.size(); ++i) {
      t[i] = 32768;
    }
    cp = &t[0];
  }
  void set(U32 cx) { cxt = cx * 256 & (t.size() - 256); }
  void mix(Mixer&m, int rate = 7) {
    *cp += ((y << 16) - (*cp) + (1 << (rate - 1))) >> rate;
    cp = &t[cxt + c0];
    m.add(stretch((*cp) >> 4));
  }
};

class ContextMap {
  const int C;
  class E {
    U16 chk[7];
    U8 last;
  public:
    U8 bh[7][7];
    U8* get(U16 chk);
  };
  Array<E, 64> t;
  Array<U8*> cp, cp0, runp;
  Array<U32> cxt;
  StateMap *sm;
  int cn;
  void update(U32 cx, int c);
public:
  ContextMap(int m, int c = 1);
  ~ContextMap();
  void set(U32 cx, int next = -1);
  int mix(Mixer&m);
};
inline U8*ContextMap::E::get(U16 ch) {
  if (chk[last & 15] == ch) return &bh[last & 15][0];
  int b = 0xffff, bi = 0;
  for (int i = 0; i < 7; ++i) {
    if (chk[i] == ch) return last = last << 4 | i, (U8*) &bh[i][0];
    int pri = bh[i][0];
    if (pri < b && (last & 15) != i && last >> 4 != i) b = pri, bi = i;
  }
  return last = 0xf0 | bi, chk[bi] = ch, (U8*)memset(&bh[bi][0], 0, 7);
}
ContextMap::ContextMap(int m, int c):C(c), t(m>>6), cp(c), cp0(c), runp(c),
    cxt(c), cn(0) {
  sm = new StateMap[C];
  for (int i = 0; i < C; ++i) {
    cp0[i] = cp[i] = &t[0].bh[0][0];
    runp[i] = cp[i] + 3;
  }
}
ContextMap::~ContextMap() {
  /* delete[] sm; */
}
inline void ContextMap::set(U32 cx, int next) {
  int i = cn++;
  i &= next;
  cx = cx * 987654323 + i;
  cx = cx << 16 | cx >> 16;
  cxt[i] = cx * 123456791 + i;
}
int ContextMap::mix(Mixer &m) {  
  int result = 0;
  for (int i = 0; i < cn; ++i) {
    if (cp[i]) {
      int ns = State_table[*cp[i]][y];
      if (ns >= 204 && rnd() << ((452 - ns) >> 3)) ns -= 4;
      *cp[i] = ns;
    }
    if (bpos > 1 && runp[i][0] == 0) {
      cp[i] = 0;
    } else {
      switch(bpos) {
        case 1: case 3: case 6: cp[i] = cp0[i] + 1 + (c0 & 1); break;
        case 4: case 7: cp[i] = cp0[i] + 3 + (c0 & 3); break;
        case 2: case 5: cp0[i] = cp[i] = t[(cxt[i] + c0) & (t.size() - 1)].get(cxt[i] >> 16); break;
        default: {
          cp0[i] = cp[i] = t[(cxt[i] + c0) & (t.size() - 1)].get(cxt[i] >> 16);
          if (cp0[i][3] == 2) {
            const int c = cp0[i][4] + 256;
            U8 *p = t[(cxt[i] + (c >> 6)) & (t.size() - 1)].get(cxt[i] >> 16);
            p[0] = 1 + ((c >> 5) & 1);
            p[1 + ((c >> 5) & 1)] = 1 + ((c >> 4) & 1);
            p[3 + ((c >> 4) & 3)] = 1 + ((c >> 3) & 1);
            p = t[(cxt[i] + (c >> 3)) & (t.size() - 1)].get(cxt[i] >> 16);
            p[0] = 1 + ((c >> 2) & 1);
            p[1 + ((c >> 2) & 1)] = 1 + ((c >> 1) & 1);
            p[3 + ((c >> 1) & 3)] = 1 + (c & 1);
            cp0[i][6] = 0;
          }
          int c1 = buf(1);
          if (runp[i][0] == 0) {
            runp[i][0] = 2, runp[i][1] = c1;
          } else if (runp[i][1] != c1) {
            runp[i][0] = 1, runp[i][1] = c1;
          } else if (runp[i][0] < 254) {
            runp[i][0] += 2;
          }
          runp[i] = cp0[i] + 3;
        } break;
      }
    }
    if ((runp[i][1] + 256) >> (8 - bpos) == c0) {
      int rc = runp[i][0];
      int b = (runp[i][1] >> (7 - bpos) & 1) * 2 - 1;
      int c = ilog(rc+1) << (2 + (~rc & 1));
      m.add(b * c);
    } else {
      m.add(0);
    }
    int p;
    if (cp[i]) {
      result += (*cp[i] > 0);
      p = sm[i].p(*cp[i]);
    } else {
      p = sm[i].p(0);
    }
    m.add(stretch(p));
  }
  if (bpos == 7) cn=0;
  return result;
}

// Match submodel
static int matchModel(Mixer& m) {
  const int MAXLEN = 0xfffe;
  static Array<int> t(MEM);
  static int h = 0, ptr = 0, len = 0, result = 0, posnl = 0;
  static SmallStationaryContextMap scm1(0x20000), scm2(0x20000);
  if (!bpos) {
    h = (h * 997 * 8 + buf(1) + 1) & (t.size() - 1);
    if (len) {
      ++len, ++ptr;
    } else {
      ptr = t[h];
      if (ptr && pos - ptr < buf.size()) {
        while (buf(len + 1) == buf[ptr - len - 1] && len < MAXLEN) ++len;
      }
    }
    t[h] = pos;
    result = len;
    scm1.set(pos);
    if (buf(1) == 0xff || buf(1) == '\r' || buf(1) == '\n') posnl = pos;
    scm2.set(min(pos - posnl, 255));
  }
  if (len) {
    if (buf(1) == buf[ptr - 1] && c0 == (buf[ptr] + 256) >> (8 - bpos)) {
      if (len > MAXLEN) len = MAXLEN;
      if (buf[ptr] >> (7 - bpos) & 1) {
        m.add(ilog(len) << 2);
        m.add(min(len, 32) << 6);
      } else {
        m.add(-(ilog(len) << 2));
        m.add(-(min(len, 32) << 6));
      }
    } else {
      len=0;
      m.add(0);
      m.add(0);
    }
  } else {
    m.add(0);
    m.add(0);
  }
  scm1.mix(m);
  scm2.mix(m);
  return result;
}

//////////////////////////// dmcModel //////////////////////////

// Model using DMC (Dynamic Markov Compression).
//
// The bitwise context is represented by a state graph.
//
// See the original paper: http://webhome.cs.uvic.ca/~nigelh/Publications/DMC.pdf
// See the original DMC implementation: http://maveric0.uwaterloo.ca/ftp/dmc/
//
// Main differences:
// - Instead of floats we use fixed point arithmetic.
// - For probability estimation each state maintains both a 0,1 count ("c0" and "c1") 
//   and a bit history ("state"). The bit history is mapped to a probability adaptively using 
//   a StateMap. The two computed probabilities are emitted to the Mixer to be combined.
// - All counts are updated adaptively.
// - The "dmcModel" is used in "dmcForest". See below.


class dmcNode {
private:
struct DMCNode { // 12 bytes
private:
  // c0,c1: adaptive counts of zeroes and ones; 
  //   fixed point numbers with 4 integer and 8 fractional bits, i.e. scaling factor=256;
  //   thus the counts 0.0 .. 15.996 are represented by 0 .. 4095
  // state: bit history state - as in a contextmodel
  U32 state_c0_c1;  // 8 + 12 + 12 = 32 bits
public:
  U32 nx0,nx1;     //indexes of next DMC nodes in the state graph
  U8   get_state() const {return state_c0_c1>>24;}
  void set_state(U8 state) {state_c0_c1=(state_c0_c1 & 0x00FFFFFF)|(state<<24);}
  U32 get_c0() const {return (state_c0_c1>>12) & 0xFFF;}
  void set_c0(U32 c0) {state_c0_c1=(state_c0_c1 &0xFF000FFF) | (c0<<12);}
  U32 get_c1() const {return state_c0_c1 & 0xFFF;}
  void set_c1(U32 c1) {state_c0_c1=(state_c0_c1 &0xFFFFF000) | c1;}
};
  U32 top, curr;     // index of first unallocated node (i.e. number of allocated nodes); index of current node
  U32 threshold;     // cloning threshold parameter: fixed point number as c0,c1
  Array<DMCNode> t;  // state graph
  StateMap sm;
  // Initialize the state graph to a bytewise order 1 model
  // See an explanation of the initial structure in:
  // http://wing.comp.nus.edu.sg/~junping/docs/njp-icita2005.pdf
  
  void resetstategraph() {
    for (int i=0; i<255; ++i) { //255 nodes in each tree
      for (int j=0; j<256; ++j) { //256 trees
        int node_idx=j*255+i;
        if (i<127) { //internal tree nodes
          t[node_idx].nx0=node_idx+i+1; // left node 
          t[node_idx].nx1=node_idx+i+2; // right node
        }
        else { // 128 leaf nodes - they each references a root node of tree(i)
          t[node_idx].nx0=(i-127)*255; // left node -> root of tree 0,1,2,3,... 
          t[node_idx].nx1=(i+1)*255;   // right node -> root of tree 128,129,...
        }
        t[node_idx].set_c0(128); //0.5
        t[node_idx].set_c1(128); //0.5
        t[node_idx].set_state(0);
      }
    }
    top=65280;
    curr=0;
  }

  // helper function: adaptively increment a counter
  U32 increment_counter (const U32 x, const U32 increment) const { //"*x" is a fixed point number as c0,c1 ; "increment"  is 0 or 1
    return (((x<<4)-x)>>4)+(increment<<8); // x * (1-1/16) + increment*256
  }

  //update stategraph
  void processbit(int y) {

    U32 c0=t[curr].get_c0();
    U32 c1=t[curr].get_c1();
    const U32 n = y ==0 ? c0 : c1;

    // update counts, state
    t[curr].set_c0(increment_counter(c0,1-y));
    t[curr].set_c1(increment_counter(c1,y));

    t[curr].set_state(State_table[t[curr].get_state()][y]);

    // clone next state when threshold is reached
    const U32 next = y==0 ? t[curr].nx0 : t[curr].nx1;
    c0=t[next].get_c0();
    c1=t[next].get_c1();
    const U32 nn=c0+c1;
    if(n>=threshold && nn>=n+threshold && top<t.size()) {
      U32 c0_top=U64(c0)*n/nn;
      U32 c1_top=U64(c1)*n/nn;
      c0-=c0_top;
      c1-=c1_top;

      t[top].set_c0(c0_top);
      t[top].set_c1(c1_top);
      t[next].set_c0(c0);
      t[next].set_c1(c1);
      
      t[top].nx0=t[next].nx0;
      t[top].nx1=t[next].nx1;
      t[top].set_state(t[next].get_state());
      if(y==0) t[curr].nx0=top;
      else t[curr].nx1=top;
      ++top;
    }

    if(y==0) curr=t[curr].nx0;
    else     curr=t[curr].nx1;
  }

public: 
  dmcNode(U32 mem, U32 th) : top(0),threshold(th),t(mem+(255*256)),  sm() {resetstategraph();  }//min(mem+(255*256),((U64(1)<<31)/sizeof(DMCNode)))

  bool isfull() {return bpos==1 && top==t.size();}
  bool isalmostfull() {return bpos==1 && top>=t.size()*15 >>4;} // *15/16
  void reset() {resetstategraph();}
  void mix(Mixer& m, bool activate) {
    processbit(y);
    if(activate) {
      const U32 n0=t[curr].get_c0()+1;
      const U32 n1=t[curr].get_c1()+1;
      const int pr1=(n1<<12)/(n0+n1);
      const int pr2=sm.p(t[curr].get_state());
      m.add(stretch(pr1)>>2);
      m.add(stretch(pr2)>>2);
    }
  }
};

// This class solves two problems of the DMC model
// 1) The DMC model is a memory hungry algorighm. In theory it works best when it can clone
//    nodes forever. But memory is a limited resource. When the state graph is full you can't
//    clone nodes anymore. You can either i) reset the model (the state graph) and start over
//    or ii) you can keep updating the counts forever in the already fixed state graph. Both
//    choices are troublesome: i) resetting the model degrades the predictive power significantly
//    until the graph becomes large enough again and ii) a fixed structure can't adapt anymore.
//    To solve this issue:
//    Two models with the same parameters work in tandem. Always both models are updated but
//    only one model (the larger, mature one) is active (predicts) at any time. When one model
//    needs resetting the other one feeds the mixer with predictions until the first one
//    becomes mature (nearly full) again.
//    Disadvantages: with the same memory reuirements we have just half of the number of nodes
//    in each model. Also keeping two models updated at all times requires 2x as much
//    calculations as updating one model only.
//    Advantage: stable and better compression - even with reduced number of nodes.
// 2) The DMC model is sensitive to the cloning threshold parameter. Some files prefer
//    a smaller threshold other files prefer a larger threshold.
//    The difference in terms of compression is significant.
//    To solve this issue:
//    Three models with different thresholds are used and their predictions are emitted to 
//    the mixer. This way the model with the better threshold will be favored naturally.
//    Disadvantage: same as in 1) just the available number of nodes is 1/3 of the 
//    one-model case.
//    Advantage: same as in 1).

class dmcModel1 {
private:
  U32 mem;
  dmcNode dmcmodel1a; // models a and b have the same parameters and work in tandem
  dmcNode dmcmodel1b;
  dmcNode dmcmodel2a; // models 1,2,3 have different threshold parameters
  dmcNode dmcmodel2b;
  dmcNode dmcmodel3a;
  dmcNode dmcmodel3b;
  int model1_state=0; // initial state, model (a) is active, both models are growing
  int model2_state=0; // model (a) is full and active, model (b) is reset and growing
  int model3_state=0; // model (b) is full and active, model (a) is reset and growing
public:
  dmcModel1(U32 val=0):
  mem(MEM),
  dmcmodel1a(mem,240),
  dmcmodel1b(mem,240),
  dmcmodel2a(mem,480),
  dmcmodel2b(mem,480),
  dmcmodel3a(mem,720),
  dmcmodel3b(mem,720){}
  int inputs() {return 2;}
  int nets() {return 0;}
  int netcount() {return 0;}
  int p(Mixer& m,int val1=0,int val2=0){

    switch(model1_state) {
      case 0:
        dmcmodel1a.mix(m,true);
        dmcmodel1b.mix(m,false);
        if(dmcmodel1a.isalmostfull()){dmcmodel1b.reset();model1_state++;}
        break;
      case 1:
        dmcmodel1a.mix(m, true);
        dmcmodel1b.mix(m, false);
        if(dmcmodel1a.isfull() && dmcmodel1b.isalmostfull()){dmcmodel1a.reset();model1_state++;}
        break;
      case 2:
        dmcmodel1b.mix(m,true);
        dmcmodel1a.mix(m,false);
        if(dmcmodel1b.isfull() && dmcmodel1a.isalmostfull()){dmcmodel1b.reset();model1_state--;}
        break;
    }
    
    switch(model2_state) {
    case 0:
      dmcmodel2a.mix(m,true);
      dmcmodel2b.mix(m,false);
      if(dmcmodel2a.isalmostfull()){dmcmodel2b.reset();model2_state++;}
      break;
    case 1:
      dmcmodel2a.mix(m,true);
      dmcmodel2b.mix(m,false);
      if(dmcmodel2a.isfull() && dmcmodel2b.isalmostfull()){dmcmodel2a.reset();model2_state++;}
      break;
    case 2:
      dmcmodel2b.mix(m,true);
      dmcmodel2a.mix(m,false);
      if(dmcmodel2b.isfull() && dmcmodel2a.isalmostfull()){dmcmodel2b.reset();model2_state--;}
      break;
    }

    switch(model3_state) {
    case 0:
      dmcmodel3a.mix(m,true);
      dmcmodel3b.mix(m,false);
      if(dmcmodel3a.isalmostfull()){dmcmodel3b.reset();model3_state++;}
      break;
    case 1:
      dmcmodel3a.mix(m,true);
      dmcmodel3b.mix(m,false);
      if(dmcmodel3a.isfull() && dmcmodel3b.isalmostfull()){dmcmodel3a.reset();model3_state++;}
      break;
    case 2:
      dmcmodel3b.mix(m,true);
      dmcmodel3a.mix(m,false);
      if(dmcmodel3b.isfull() && dmcmodel3a.isalmostfull()){dmcmodel3b.reset();model3_state--;}
      break;
    }
    return 0;
  }
  virtual ~dmcModel1(){ }
};

class nestModel1 {
  int ic, bc, pc,vc, qc, lvc, wc,ac, ec, uc, sense1, sense2, w;
  const int N;
  ContextMap cm;
public:
  nestModel1(U32 val=0): ic(0), bc(0),
   pc(0),vc(0), qc(0), lvc(0), wc(0),ac(0), ec(0), uc(0), sense1(0), sense2(0), w(0), N(12),
   cm(MEM, 10)  {
  }
int p(Mixer& m,int val1=0,int val2=0){
    if (bpos==0) {
    int c=c4&255, matched=1, vv;
    w*=((vc&7)>0 && (vc&7)<3);
    if (c&0x80) w = w*11*32 + c;
    const int lc = (c >= 'A' && c <= 'Z'?c+'a'-'A':c) ;
    if (lc == 'a' || lc == 'e' || lc == 'i' || lc == 'o' || lc == 'u'){ vv = 1; w = w*997*8 + (lc/4-22); } else
    if (lc >= 'a' && lc <= 'z' || c>128){ vv = 2; w = w*271*32 + lc-97; } else
    if (lc == ' ' || lc == '.' || lc == ',' || lc == '\n'|| lc == 5) vv = 3; else
    if (lc >= '0' && lc <= '9') vv = 4; else
    if (lc == 'y') vv = 5; else
    if (lc == '\'') vv = 6; else vv=(c&32)?7:0;
    vc = (vc << 3) | vv;
    if (vv != lvc) {
      wc = (wc << 3) | vv;
      lvc = vv;
    }
    switch(c) {
      case ' ': qc = 0; break;
      case '(': ic += 31; break;
      case ')': ic -= 31; break;
      case '[': ic += 11; break;
      case ']': ic -= 11; break;
      case '<': ic += 23; qc += 34; break;
      case '>': ic -= 23; qc /= 5; break;
      case ':': pc = 20; break;
      case '{': ic += 17; break;
      case '}': ic -= 17; break;
      case '|': pc += 223; break;
      case '"': pc += 0x40; break;
      case '\'': pc += 0x42; if (c!=(U8)(c4>>8)) sense2^=1; else ac+=(2*sense2-1); break;
      case 5: 
      case '\n': pc = qc = 0; break;
      case '.': pc = 0; break;
      case '!': pc = 0; break;
      case '?': pc = 0; break;
      case '#': pc += 0x08; break;
      case '%': pc += 0x76; break;
      case '$': pc += 0x45; break;
      case '*': pc += 0x35; break;
      case '-': pc += 0x3; break;
      case '@': pc += 0x72; break;
      case '&': qc += 0x12; break;
      case ';': qc /= 3; break;
      case '\\': pc += 0x29; break;
      case '/': pc += 0x11;
                if (buf.size() > 1 && buf(1) == '<') qc += 74;
                break;
      case '=': pc += 87; if (c!=(U8)(c4>>8)) sense1^=1; else ec+=(2*sense1-1); break;
      default: matched = 0;
    }
    if (c4==0x266C743B) uc=min(7,uc+1);
    else if (c4==0x2667743B) uc-=(uc>0);
    if (matched) bc = 0; else bc += 1;
    if (bc > 300) bc = ic = pc = qc = uc = 0;
if (val2==-1) return 1;
    cm.set(hash(ic, w, ilog2(bc+1)));
    
    cm.set(U32((3*vc+77*pc+373*ic+qc)&0xffff));
    cm.set(U32((31*vc+27*pc+281*qc)&0xffff));
    cm.set(U32((13*vc+271*ic+qc+bc)&0xffff));
    cm.set(U32((13*vc+ic)&0xffff));
    cm.set(U32((vc/3+pc)&0xffff));
    cm.set(U32((17*pc+7*ic)&0xffff));
    cm.set(U32((7*wc+qc)&0xffff));
  }
    
    cm.mix(m);
    m.set(vc&511,512);
  return 0;
}
virtual ~nestModel1(){ }
};

//////////////////////////// ppmdModel //////////////////////////

#define NOINLINE __attribute__((__noinline__))
template <class T> T Min( T x, T y ) { return (x<y) ? x : y; }
template <class T> T Max( T x, T y ) { return (x>y) ? x : y; }

#pragma pack(1)

typedef unsigned short word;
//typedef unsigned int   uint;
//typedef unsigned char  byte;
//typedef unsigned long long qword;

//--- #include "libpmd/model_def.inc"

//#pragma pack(1)

const int ORealMAX=256;

static signed char EscCoef[12] = { 16, -10, 1, 51, 14, 89, 23, 35, 64, 26, -42, 43  };

// Tabulated escapes for exponential symbol distribution
static const U8 ExpEscape[16]={ 51,43,18,12,11,9,8,7,6,5,4,3,3,2,2,2 };

struct ppmd_Model {

    typedef unsigned short word;

    enum{ SCALE=1<<15 };

    enum{
        UNIT_SIZE=12,
        N1=4, N2=4, N3=4, N4=(128+3-1*N1-2*N2-3*N3)/4,
        N_INDEXES=N1+N2+N3+N4
    };

    U8* HeapStart;
    typedef U8* pbyte;
    U32  Ptr2Indx( void* p ) { return pbyte(p)-HeapStart; }
    void* Indx2Ptr(U32 indx) { return indx + HeapStart; }
   /* U32  Ptr2Indx( void* p ) {
        U64 addr = ((U8*)p)-HeapStart;
        U32 lim = (UnitsStart-HeapStart);
        U32 indx = (addr>=lim) ? (addr-lim)/UNIT_SIZE+lim : addr;
        return indx;
    }

    void* Indx2Ptr( U32 indx ) {
        U32 lim = (UnitsStart-HeapStart);
        U64 addr = (indx>=lim) ? U64(indx-lim)*UNIT_SIZE+lim : indx;
        return HeapStart + addr;
    }*/

    //---   #include "alloc_node.inc"
    struct _MEM_BLK {
        U32 Stamp;
        U32 NextIndx;
        U32 NU;
    };

    struct BLK_NODE {
        U32 Stamp;
        U32 NextIndx;
        int avail() const { return (NextIndx!=0); }
    };

    BLK_NODE* getNext( BLK_NODE* This ) {
        return (BLK_NODE*)Indx2Ptr(This->NextIndx);
    }

    void setNext( BLK_NODE* This, BLK_NODE* p ) {
        This->NextIndx = Ptr2Indx(p);
    }

    void link( BLK_NODE* This, BLK_NODE* p ) {
        p->NextIndx = This->NextIndx;
        setNext( This, p );
    }

    void unlink( BLK_NODE* This ) {
        This->NextIndx = getNext(This)->NextIndx;
    }

    void* remove( BLK_NODE* This ) {
        BLK_NODE* p = getNext(This);
        unlink(This);
        This->Stamp--;
        return p;
    }

    void insert( BLK_NODE* This, void* pv, int NU ) {
        BLK_NODE* p = (BLK_NODE*)pv;
        link(This,p);
        p->Stamp = ~U32(0);
        ((_MEM_BLK&)*p).NU = NU;
        This->Stamp++;
    }

    struct MEM_BLK : public BLK_NODE {
        U32 NU;
    };

    typedef BLK_NODE* pBLK_NODE;

    typedef MEM_BLK* pMEM_BLK;

    BLK_NODE BList[N_INDEXES+1];

    U32  GlueCount;
    U32  GlueCount1;
    U64 SubAllocatorSize;
    U8* pText;
    U8* UnitsStart;
    U8* LoUnit;
    U8* HiUnit;
    U8* AuxUnit;

    //---   #include "alloc_units.inc"

    U32 U2B( U32 NU ) {
        return 8*NU+4*NU;
    }

    int StartSubAllocator( U64 SASize ) {
        U64 t = SASize << 20U;
        HeapStart = new U8[t];
        //  HeapStart = mAlloc<U8>(t);
        //  HeapStart = (U8*)VirtualAlloc( 0, t, MEM_COMMIT, PAGE_READWRITE );
        if( HeapStart==NULL ) return 0;
        SubAllocatorSize = t;
        return 1;
    }

    void InitSubAllocator() {
        memset( BList, 0, sizeof(BList) );
        HiUnit = (pText=HeapStart) + SubAllocatorSize;
        U64 Diff = SubAllocatorSize/8/UNIT_SIZE*7 *UNIT_SIZE ; //U2B(SubAllocatorSize/8/UNIT_SIZE*7);
        LoUnit = UnitsStart = HiUnit-Diff;
        GlueCount=GlueCount1=0;
        //printf( "HeapStart=%I64X\n", HeapStart );
        //printf( "UnitsStart=%I64X\n", UnitsStart-HeapStart );
        //printf( "SubAllocatorSize=%I64X\n", SubAllocatorSize );
    }

    U64 GetUsedMemory() {
        int i;
        U64 RetVal = SubAllocatorSize - (HiUnit-LoUnit) - (UnitsStart-pText);
        for( i=0; i<N_INDEXES; i++ ) {
            //    RetVal -= U2B( Indx2Units[i]*BList[i].Stamp );
            RetVal -= U64(Indx2Units[i]*BList[i].Stamp)*12;
        }
        return RetVal;
    }

    void StopSubAllocator() {
        // if( SubAllocatorSize ) SubAllocatorSize=0, delete HeapStart;
        //  if( SubAllocatorSize ) SubAllocatorSize=0, VirtualFree(HeapStart, 0, MEM_RELEASE);
    }

    //----------------------------------------

    void GlueFreeBlocks() {
        U32 i, k, sz;
        MEM_BLK s0;
        pMEM_BLK p, p0=&s0, p1;

        if( LoUnit!=HiUnit ) LoUnit[0]=0;

        for( p0->NextIndx=0,i=0; i<=N_INDEXES; i++ ) {
            while( BList[i].avail() ) {
                p = (MEM_BLK*)remove(&BList[i]);
                if( p->NU ) {
                    while( p1 = p + p->NU, p1->Stamp==~U32(0) ) {
                        p->NU += p1->NU;
                        p1->NU = 0;
                    }
                    link(p0,p); p0=p;
                }
            }
        }

        while( s0.avail() ) {
            p = (MEM_BLK*)remove(&s0);
            sz= p->NU;
            if( sz ) {
                for(; sz>128; sz-=128, p+=128 ) insert(&BList[N_INDEXES-1],p,128);
                i = Units2Indx[sz-1];
                if( Indx2Units[i] != sz ) {
                    k = sz - Indx2Units[--i];
                    insert( &BList[k-1], p+(sz-k) , k );
                }
                insert( &BList[i], p, Indx2Units[i] );
            }
        }

        GlueCount = 1 << (13+GlueCount1++);
    }

    void SplitBlock( void* pv, U32 OldIndx, U32 NewIndx ) {
        U32 i, k, UDiff=Indx2Units[OldIndx]-Indx2Units[NewIndx];
        U8* p = ((U8*)pv)+U2B(Indx2Units[NewIndx]);
        i = Units2Indx[UDiff-1];
        if( Indx2Units[i]!=UDiff ) {
            k=Indx2Units[--i];
            insert(&BList[i],p,k);
            p += U2B(k);
            UDiff -= k;
        }
        insert( &BList[Units2Indx[UDiff-1]], p, UDiff );
    }

    void* AllocUnitsRare( U32 indx ) {
        U32 i = indx;
        do {
            if( ++i == N_INDEXES ) {
                if( !GlueCount-- ) {
                    GlueFreeBlocks();
                    if( BList[i=indx].avail() ) return remove(&BList[i]);
                } else {
                    i = U2B(Indx2Units[indx]);
                    return (UnitsStart-pText>i) ? UnitsStart-=i : NULL;
                }
            }
        } while( !BList[i].avail() );

        void* RetVal=remove(&BList[i]);
        SplitBlock( RetVal, i, indx );

        return RetVal;
    }

    void* AllocUnits( U32 NU ) {
        U32 indx = Units2Indx[NU-1];
        if( BList[indx].avail() ) return remove(&BList[indx]);
        void* RetVal=LoUnit;
        LoUnit += U2B(Indx2Units[indx]);
        if( LoUnit<=HiUnit ) return RetVal;
        LoUnit -= U2B(Indx2Units[indx]);
        return AllocUnitsRare(indx);
    }

    void* AllocContext() {
        if( HiUnit!=LoUnit ) return HiUnit-=UNIT_SIZE;
        return BList->avail() ? remove(BList) : AllocUnitsRare(0);
    }

    void FreeUnits( void* ptr, U32 NU ) {
        U32 indx = Units2Indx[NU-1];
        insert( &BList[indx], ptr, Indx2Units[indx] );
    }

    void FreeUnit( void* ptr ) {
        int i = (U8*)ptr > UnitsStart+128*1024 ? 0 : N_INDEXES;
        insert( &BList[i], ptr, 1 );
    }

    //----------------------------------------

    void UnitsCpy( void* Dest, void* Src, U32 NU ) {
        memcpy( Dest, Src, 12*NU );
    }

    void* ExpandUnits( void* OldPtr, U32 OldNU ) {
        U32 i0 = Units2Indx[OldNU-1];
        U32 i1 = Units2Indx[OldNU-1+1];
        if( i0==i1 ) return OldPtr;
        void* ptr = AllocUnits(OldNU+1);
        if( ptr ) {
            UnitsCpy( ptr, OldPtr, OldNU );
            insert( &BList[i0], OldPtr, OldNU );
        }
        return ptr;
    }

    void* ShrinkUnits( void* OldPtr, U32 OldNU, U32 NewNU ) {
        U32 i0 = Units2Indx[OldNU-1];
        U32 i1 = Units2Indx[NewNU-1];
        if( i0==i1 ) return OldPtr;
        if( BList[i1].avail() ) {
            void* ptr = remove(&BList[i1]);
            UnitsCpy( ptr, OldPtr, NewNU );
            insert( &BList[i0], OldPtr, Indx2Units[i0] );
            return ptr;
        } else {
            SplitBlock(OldPtr,i0,i1);
            return OldPtr;
        }
    }

    void* MoveUnitsUp( void* OldPtr, U32 NU ) {
        U32 indx = Units2Indx[NU-1];
       // PrefetchData(OldPtr);
        if( (U8*)OldPtr > UnitsStart+128*1024 ||
                (BLK_NODE*)OldPtr > getNext(&BList[indx]) ) return OldPtr;

        void* ptr = remove(&BList[indx]);
        UnitsCpy( ptr, OldPtr, NU );

        insert( &BList[N_INDEXES], OldPtr, Indx2Units[indx] );

        return ptr;
    }

    void PrepareTextArea() {
        AuxUnit = (U8*)AllocContext();
        if( !AuxUnit ) {
            AuxUnit = UnitsStart;
        } else {
            if( AuxUnit==UnitsStart) AuxUnit = (UnitsStart+=UNIT_SIZE);
        }
    }

    void ExpandTextArea() {
        BLK_NODE* p;
        U32 Count[N_INDEXES], i=0;
        memset( Count, 0, sizeof(Count) );
        if( AuxUnit!=UnitsStart ) {
            if( *(U32*)AuxUnit != ~U32(0) )
            UnitsStart += UNIT_SIZE;
            else
            insert( BList, AuxUnit, 1 );
        }
        while( (p=(BLK_NODE*)UnitsStart)->Stamp == ~U32(0) ) {
            MEM_BLK* pm = (MEM_BLK*)p;
            UnitsStart = (U8*)(pm + pm->NU);
            Count[Units2Indx[pm->NU-1]]++;
            i++;
            pm->Stamp = 0;
        }
        if( i ) {
            for( p=BList+N_INDEXES; p->NextIndx; p=getNext(p) ) {
                while( p->NextIndx && !getNext(p)->Stamp ) {
                    Count[Units2Indx[((MEM_BLK*)getNext(p))->NU-1]]--;
                    unlink(p);
                    BList[N_INDEXES].Stamp--;
                }
                if( !p->NextIndx ) break;
            }
            for( i=0; i<N_INDEXES; i++ ) {
                for( p=BList+i; Count[i]!=0; p=getNext(p) ) {
                    while( !getNext(p)->Stamp ) {
                        unlink(p); BList[i].Stamp--;
                        if ( !--Count[i] ) break;
                    }
                }
            }
        }
    }


    //---   #include "ppmd_init.inc"

    static const int MAX_O=ORealMAX;  // maximum allowed model order

    //enum { FALSE=0,TRUE=1 };

    template <class T>
    T CLAMP( const T& X, const T& LoX, const T& HiX ) { return (X >= LoX)?((X <= HiX)?(X):(HiX)):(LoX); }

    template <class T>
    void SWAP( T& t1, T& t2 ) { T tmp=t1; t1=t2; t2=tmp; }

   // void PrefetchData(void* Addr) {
        //U8 Prefetchbyte = *(volatile U8*)Addr;
  //  }

    enum {
        UP_FREQ     = 5
    };

    U8 Indx2Units[N_INDEXES];
    U8 Units2Indx[128]; // constants
    U8 NS2BSIndx[256];
    U8 QTable[260];

    // constants initialization
    void PPMD_STARTUP( void ) {
        int i, k, m, Step;
        for( i=0,k=1; i<N1         ;i++,k+=1 ) Indx2Units[i]=k;
        for( k++;     i<N1+N2      ;i++,k+=2 ) Indx2Units[i]=k;
        for( k++;     i<N1+N2+N3   ;i++,k+=3 ) Indx2Units[i]=k;
        for( k++;     i<N1+N2+N3+N4;i++,k+=4 ) Indx2Units[i]=k;
        for( k=0,i=0; k<128; k++ ) {
            i += Indx2Units[i]<k+1;
            Units2Indx[k]=i;
        }
        NS2BSIndx[0] = 2*0; //-V525
        NS2BSIndx[1] = 2*1;
        NS2BSIndx[2] = 2*1;
        memset(NS2BSIndx+3,  2*2, 26);
        memset(NS2BSIndx+29, 2*3, 256-29);
        for( i=0; i<UP_FREQ; i++ ) QTable[i]=i;
        for( m=i=UP_FREQ, k=Step=1; i<260; i++ ) {
            QTable[i] = m;
            if( !--k ) k = ++Step, m++;
        }
    }

    //---   #include "mod_context.inc"

    enum {
        MAX_FREQ    = 124,
        O_BOUND     = 9
    };

    struct PPM_CONTEXT;
    struct STATE {
        U8 Symbol;
        U8 Freq;
        U32 iSuccessor;
    };

    PPM_CONTEXT* getSucc( STATE* This ) {
        return (PPM_CONTEXT*)Indx2Ptr( This->iSuccessor );
    }

    void SWAP( STATE& s1, STATE& s2 ) {
        word t1       = (word&)s1;
        U32 t2       = s1.iSuccessor;
        (word&)s1     = (word&)s2;
        s1.iSuccessor = s2.iSuccessor;
        (word&)s2     = t1;
        s2.iSuccessor = t2;
    }

    struct PPM_CONTEXT {
        U8 NumStats;
        U8 Flags;
        word SummFreq;
        U32 iStats;
        U32 iSuffix;
        STATE& oneState() const { return (STATE&) SummFreq; }
    };

    STATE* getStats( PPM_CONTEXT* This ) { return (STATE*)Indx2Ptr(This->iStats); }
    PPM_CONTEXT* suff( PPM_CONTEXT* This ) { return (PPM_CONTEXT*)Indx2Ptr(This->iSuffix); }

    int _MaxOrder, _CutOff, _MMAX;
    U32 _filesize;
    int OrderFall;

    STATE* FoundState; // found next state transition
    PPM_CONTEXT* MaxContext;

    U32 EscCount;
    U32 CharMask[256];

    int  BSumm;
    int  RunLength;
    int  InitRL;

    //---   #include "mod_see.inc"

    enum {
        INT_BITS    = 7,
        PERIOD_BITS = 7,
        TOT_BITS    = INT_BITS + PERIOD_BITS,
        INTERVAL    = 1 << INT_BITS,
        BIN_SCALE   = 1 << TOT_BITS,
        ROUND       = 16
    };

    // SEE-contexts for PPM-contexts with masked symbols
    struct SEE2_CONTEXT {
        word Summ;
        U8 Shift;
        U8 Count;

        void init( U32 InitVal ) {
            Shift = PERIOD_BITS-4;
            Summ  = InitVal << Shift;
            Count = 7;
        }

        U32 getMean() {
            return Summ >> Shift;
        }

        void update() {
            if( --Count==0 ) setShift_rare();
        }

        void setShift_rare() {
            U32 i = Summ >> Shift;
            i = PERIOD_BITS - (i>40) - (i>280) - (i>1020);
            if( i<Shift ) { Summ >>= 1; Shift--; } else
            if( i>Shift ) { Summ <<= 1; Shift++; }
            Count = 5 << Shift;
        }
    };

    int  NumMasked;

    //---   #include "mod_rescale.inc"

    STATE* rescale( PPM_CONTEXT& q, int OrderFall, STATE* FoundState ) {
        STATE tmp; STATE* p; STATE* p1;
        q.Flags &= 0x14;
        // move the current node to rank0
        p1 = getStats(&q);
        tmp = FoundState[0];
        for( p=FoundState; p!=p1; p-- ) p[0]=p[-1];
        p1[0] = tmp;

        int of = (OrderFall != 0);
        int a, i;
        int f0 = p->Freq;
        int sf = q.SummFreq;
        int EscFreq = sf-f0;
        q.SummFreq = p->Freq = (f0+of)>>1;

        // sort symbols by freqs
        for( i=0; i<q.NumStats; i++ ) {
            p++;
            a = p->Freq;
            EscFreq  -= a;
            a = (a+of)>>1;
            p->Freq = a;
            q.SummFreq += a;
            if( a ) q.Flags |= 0x08*(p->Symbol>=0x40);
            if( a > p[-1].Freq ) {
                tmp = p[0];
                for( p1=p; tmp.Freq>p1[-1].Freq; p1-- ) p1[0]=p1[-1];
                p1[0] = tmp;
            }
        }

        // remove the zero freq nodes
        if( p->Freq==0 ) {
            for( i=0; p->Freq==0; i++,p-- );
            EscFreq += i;
            a = (q.NumStats+2) >> 1;
            if( (q.NumStats-=i)==0 ) {
                tmp = getStats(&q)[0];
                tmp.Freq = Min( MAX_FREQ/3, (2*tmp.Freq+EscFreq-1)/EscFreq );
                q.Flags &= 0x18;
                FreeUnits( getStats(&q), a );
                q.oneState() = tmp;
                FoundState = &q.oneState();
                return FoundState;
            }
            q.iStats = Ptr2Indx( ShrinkUnits(getStats(&q),a,(q.NumStats+2)>>1) );
        }

        // some weird magic
        q.SummFreq += (EscFreq+1) >> 1;
        if( OrderFall || (q.Flags & 0x04)==0 ) {
            a = (sf-=EscFreq) - f0;
            a = CLAMP( U32( ( f0*q.SummFreq - sf*getStats(&q)->Freq + a-1 ) / a ), 2U, MAX_FREQ/2U-18U );
        } else {
            a = 2;
        }

        (FoundState=getStats(&q))->Freq += a;
        q.SummFreq += a;
        q.Flags |= 0x04;

        return FoundState;
    }
    //---   #include "mod_cutoff.inc"

    void AuxCutOff( STATE* p, int Order, int MaxOrder ) {
        if( Order<MaxOrder ) {
            //PrefetchData( getSucc(p) );
            p->iSuccessor = cutOff( getSucc(p)[0], Order+1,MaxOrder);
        } else {
            p->iSuccessor=0;
        }
    }

    U32 cutOff( PPM_CONTEXT& q, int Order, int MaxOrder ) {
        int i, tmp, EscFreq, Scale;
        STATE* p;
        STATE* p0;

        // for binary context, just cut off the successors
        if( q.NumStats==0 ) {
            int flag = 1;
            p = &q.oneState();
            if( (U8*)getSucc(p) >= UnitsStart ) {
                AuxCutOff( p, Order, MaxOrder );
                if( p->iSuccessor || Order<O_BOUND ) flag=0;
            }
            if( flag ) {
                FreeUnit( &q );
                return 0;
            }
        } else {
            tmp = (q.NumStats+2)>>1;
            p0 = (STATE*)MoveUnitsUp(getStats(&q),tmp);
            q.iStats = Ptr2Indx(p0);

            // cut the branches with links to text
            for( i=q.NumStats, p=&p0[i]; p>=p0; p-- ) {
                if( (U8*)getSucc(p) < UnitsStart ) {
                    p[0].iSuccessor=0;
                    SWAP( p[0], p0[i--] );
                } else AuxCutOff( p, Order, MaxOrder );
            }
            // if something was cut
            if( i!=q.NumStats && Order>0 ) {
                q.NumStats = i;
                p = p0;
                if( i<0 ) {
                    FreeUnits( p, tmp );
                    FreeUnit( &q );
                    return 0;
                }
                if( i==0 ) {
                    q.Flags = (q.Flags & 0x10) + 0x08*(p[0].Symbol>=0x40);
                    p[0].Freq = 1+(2*(p[0].Freq-1))/(q.SummFreq-p[0].Freq);
                    q.oneState() = p[0];
                    FreeUnits( p, tmp );
                } else {
                    p = (STATE*)ShrinkUnits( p0, tmp, (i+2)>>1 );
                    q.iStats = Ptr2Indx(p);
                    Scale = (q.SummFreq>16*i); // av.freq > 16
                    q.Flags = (q.Flags & (0x10+0x04*Scale));
                    if( Scale ) {
                        EscFreq = q.SummFreq;
                        q.SummFreq = 0;
                        for( i=0; i<=q.NumStats; i++ ) {
                            EscFreq  -= p[i].Freq;
                            p[i].Freq = (p[i].Freq+1)>>1;
                            q.SummFreq += p[i].Freq;
                            q.Flags |= 0x08*(p[i].Symbol>=0x40);
                        };
                        EscFreq = (EscFreq+1)>>1;
                        q.SummFreq += EscFreq;
                    } else {
                        for( i=0; i<=q.NumStats; i++ ) q.Flags |= 0x08*(p[i].Symbol>=0x40);
                    }
                }
            }
        }
        if( (U8*)&q==UnitsStart ) {
            // if this is a root, copy it
            UnitsCpy( AuxUnit, &q, 1 );
            return Ptr2Indx(AuxUnit);
        } else {
            // if suffix is root, switch the pointer
            if( (U8*)suff(&q)==UnitsStart ) q.iSuffix=Ptr2Indx(AuxUnit);
        }
        return Ptr2Indx(&q);
    }

    //---   #include "ppmd_flush.inc"

    NOINLINE
    void StartModelRare( void ) {
        int i, k, s;
        U8 i2f[25];
        memset( CharMask, 0, sizeof(CharMask) );
        EscCount=1;
        // we are in solid mode
        if( _MaxOrder<2 ) {
            OrderFall = _MaxOrder;
            for( PPM_CONTEXT* pc=MaxContext; pc->iSuffix!=0; pc=suff(pc) ) OrderFall--;
            return;
        }
        OrderFall = _MaxOrder;
        InitSubAllocator();
        InitRL = -( (_MaxOrder<13) ? _MaxOrder : 13 );
        RunLength = InitRL;
        // alloc and init order0 context
        MaxContext = (PPM_CONTEXT*)AllocContext();
        MaxContext->NumStats = 255;
        MaxContext->SummFreq = 255+2;
        MaxContext->iStats   = Ptr2Indx(AllocUnits(256/2));
        MaxContext->Flags    = 0;
        MaxContext->iSuffix  = 0;
        PrevSuccess          = 0;
        for( i=0; i<256; i++ ) {
            getStats(MaxContext)[i].Symbol     = i;
            getStats(MaxContext)[i].Freq       = 1;
            getStats(MaxContext)[i].iSuccessor = 0;
        }
        // _InitSEE
        if( 1 ) {
            // a freq for quant?
            for( k=i=0; i<25; i2f[i++]=k+1 ) while( QTable[k]==i ) k++;
            // bin SEE init
            for( k=0; k<64; k++ ) {
                for( s=i=0; i<6; i++ ) s += EscCoef[2*i+((k>>i)&1)];
                s = 128*CLAMP( s, 32, 256-32 );
                for( i=0; i<25; i++ ) BinSumm[i][k] = BIN_SCALE - s/i2f[i];
            }
            // masked SEE init
            for( i=0; i<23; i++ ) for( k=0; k<32; k++ ) SEE2Cont[i][k].init(8*i+5);
        }
    }

    // model flush
    NOINLINE
    void RestoreModelRare( void ) {
        STATE* p;
        pText = HeapStart;
        PPM_CONTEXT* pc = saved_pc;
        // from maxorder down, while there 2 symbols and 2nd symbol has a text pointer
        for(;; MaxContext=suff(MaxContext) ) {
            if( (MaxContext->NumStats==1) && (MaxContext!=pc) ) {
                p = getStats(MaxContext);
                if( (U8*)(getSucc(p+1))>=UnitsStart ) break;
            } else break;
            // turn a context with 2 symbols into a context with 1 symbol
            MaxContext->Flags = (MaxContext->Flags & 0x10) + 0x08*(p->Symbol>=0x40);
            p[0].Freq = (p[0].Freq+1) >> 1;
            MaxContext->oneState() = p[0];
            MaxContext->NumStats=0;
            FreeUnits( p, 1 );
        }
        // go all the way down
        while( MaxContext->iSuffix ) MaxContext=suff(MaxContext);
        AuxUnit = UnitsStart;
        ExpandTextArea();
        // free up 25% of memory
        do {
            PrepareTextArea();
            cutOff( MaxContext[0], 0, _MaxOrder ); // MaxContext is a tree root here, order0
            ExpandTextArea();
        } while( GetUsedMemory()>3*(SubAllocatorSize>>2) );

        GlueCount = GlueCount1 = 0;
        OrderFall = _MaxOrder;
    }

    //---   #include "ppmd_update.inc"

    PPM_CONTEXT* saved_pc;

    PPM_CONTEXT* UpdateModel( PPM_CONTEXT* MinContext ) {
        U8 Flag,  FSymbol;
        U32 ns1, ns, cf, sf, s0, FFreq;
        U32 iSuccessor, iFSuccessor;
        PPM_CONTEXT* pc;
        STATE* p = NULL;
        FSymbol     = FoundState->Symbol;
        FFreq       = FoundState->Freq;
        iFSuccessor = FoundState->iSuccessor;
        // partial update for the suffix context
        if( MinContext->iSuffix ) {
            pc = suff(MinContext);
            // is it binary?
            if( pc[0].NumStats ) {
                p = getStats(pc);
                if( p[0].Symbol!=FSymbol ) {
                    for( p++; p[0].Symbol!=FSymbol; p++ );
                    if( p[0].Freq >= p[-1].Freq ) SWAP( p[0], p[-1] ), p--;
                }
                if( p[0].Freq<MAX_FREQ-3 ) {
                    cf = 2 + (FFreq<28);
                    p[0].Freq += cf;
                    pc[0].SummFreq += cf;
                }
            } else {
                p = &(pc[0].oneState());
                p[0].Freq += (p[0].Freq<14);
            }
        }
        pc = MaxContext;
        // try increasing the order
        if( !OrderFall && iFSuccessor ) {
            FoundState->iSuccessor = CreateSuccessors( 1, p, MinContext );
            if( !FoundState->iSuccessor ) { saved_pc=pc; return 0; };
            MaxContext = getSucc(FoundState);
            return MaxContext;
        }
        *pText++ = FSymbol;
        iSuccessor = Ptr2Indx(pText);
        if( pText>=UnitsStart ) { saved_pc=pc; return 0; };

        if( iFSuccessor ) {
            if( (U8*)Indx2Ptr(iFSuccessor) < UnitsStart )
            iFSuccessor = CreateSuccessors( 0, p, MinContext );
            //else
           // PrefetchData( Indx2Ptr(iFSuccessor) );
        } else
        iFSuccessor = ReduceOrder( p, MinContext );

        if( !iFSuccessor ) { saved_pc=pc; return 0; };

        if( !--OrderFall ) {
            iSuccessor = iFSuccessor;
            pText -= (MaxContext!=MinContext);
        }
        s0 = MinContext->SummFreq - FFreq;
        ns = MinContext->NumStats;
        Flag = 0x08*(FSymbol>=0x40);
        for( pc=MaxContext; pc!=MinContext; pc=suff(pc) ) {
            ns1 = pc[0].NumStats;
            // non-binary context?
            if( ns1 ) {
                // realloc table with alphabet size is odd
                if( ns1&1 ) {
                    p = (STATE*)ExpandUnits( getStats(pc),(ns1+1)>>1 );
                    if( !p ) { saved_pc=pc; return 0; };
                    pc[0].iStats = Ptr2Indx(p);
                }
                // increase escape freq (more for larger alphabet)
                pc[0].SummFreq += QTable[ns+4] >> 3;
            } else {
                // escaped binary context
                p = (STATE*)AllocUnits(1);
                if( !p ) { saved_pc=pc; return 0; };
                p[0] = pc[0].oneState();
                pc[0].iStats = Ptr2Indx(p);
                p[0].Freq = (p[0].Freq<=MAX_FREQ/3) ? (2*p[0].Freq-1) : (MAX_FREQ-15);
                // update escape
                pc[0].SummFreq = p[0].Freq + (ns>1) + ExpEscape[QTable[BSumm>>8]]; //-V602
            }

            // inheritance
            cf = (FFreq-1)*(5 + pc[0].SummFreq);
            sf = s0 + pc[0].SummFreq;
            // this is a weighted rescaling of symbol's freq into a new context (cf/sf)
            if( cf<=3*sf ) {
                // if the new freq is too small the we increase the escape freq too
                cf = 1 + (2*cf>sf) + (2*cf>3*sf);
                pc[0].SummFreq += 4;
            } else {
                cf = 5 + (cf>5*sf) + (cf>6*sf) + (cf>8*sf) + (cf>10*sf) + (cf>12*sf);
                pc[0].SummFreq += cf;
            }
            p = getStats(pc) + (++pc[0].NumStats);
            p[0].iSuccessor = iSuccessor;
            p[0].Symbol = FSymbol;
            p[0].Freq   = cf;
            pc[0].Flags |= Flag; // flag if last added symbol was >=0x40
        }

        MaxContext = (PPM_CONTEXT*)Indx2Ptr(iFSuccessor);
        return MaxContext;
    }

    U32 CreateSuccessors( U32 Skip, STATE* p, PPM_CONTEXT* pc ) {
        U8 tmp;
        U32 cf, s0;
        STATE*  ps[MAX_O];
        STATE** pps=ps;
        U8 sym = FoundState->Symbol;
        U32 iUpBranch = FoundState->iSuccessor;
        if( !Skip ) {
            *pps++ = FoundState;
            if( !pc[0].iSuffix ) goto NO_LOOP;
        }
        if( p ) { pc = suff(pc); goto LOOP_ENTRY; }
        do {
            pc = suff(pc);
            // increment current symbol's freq in lower order contexts
            // more partial updates?
            if( pc[0].NumStats ) {
                // find sym node
                for( p=getStats(pc); p[0].Symbol!=sym; p++ );
                // increment freq if limit allows
                tmp = 2*(p[0].Freq<MAX_FREQ-1);
                p[0].Freq += tmp;
                pc[0].SummFreq += tmp;
            } else {
                // binary context
                p = &(pc[0].oneState());
                p[0].Freq += (!suff(pc)->NumStats & (p[0].Freq<16));
            }
            LOOP_ENTRY:
            if( p[0].iSuccessor!=iUpBranch ) {
                pc = getSucc(p);
                break;
            }
            *pps++ = p;
        } while ( pc[0].iSuffix );

        NO_LOOP:
        if( pps==ps ) return Ptr2Indx(pc);
        // fetch a following symbol from the text buffer
        PPM_CONTEXT ct;
        ct.NumStats = 0;
        ct.Flags = 0x10*(sym>=0x40);
        sym = *(U8*)Indx2Ptr(iUpBranch);
        ct.oneState().iSuccessor = Ptr2Indx((U8*)Indx2Ptr(iUpBranch)+1);
        ct.oneState().Symbol = sym;
        ct.Flags |= 0x08*(sym>=0x40);

        // pc is MinContext, the context used for encoding
        if( pc[0].NumStats ) {
            for( p=getStats(pc); p[0].Symbol!=sym; p++ );
            cf = p[0].Freq - 1;
            s0 = pc[0].SummFreq - pc[0].NumStats - cf;
            cf = 1 + ((2*cf<s0) ? (12*cf>s0) : 2+cf/s0);
            ct.oneState().Freq = Min<U32>( 7, cf );
        } else {
            ct.oneState().Freq = pc[0].oneState().Freq;
        }
        // attach the new node to all orders
        do {
            PPM_CONTEXT* pc1 = (PPM_CONTEXT*)AllocContext();
            if( !pc1 ) return 0;
            ((U32*)pc1)[0] = ((U32*)&ct)[0];
            ((U32*)pc1)[1] = ((U32*)&ct)[1];
            pc1->iSuffix = Ptr2Indx(pc);
            pc = pc1; pps--;
            pps[0][0].iSuccessor = Ptr2Indx(pc);
        } while( pps!=ps );
        return Ptr2Indx(pc);
    }

    U32 ReduceOrder( STATE* p, PPM_CONTEXT* pc ) {
        U8 tmp;
        STATE* p1;
        PPM_CONTEXT* pc1=pc;
        FoundState->iSuccessor = Ptr2Indx(pText);
        U8 sym = FoundState->Symbol;
        U32 iUpBranch = FoundState->iSuccessor;
        OrderFall++;
        if( p ) { pc=suff(pc); goto LOOP_ENTRY; }
        while(1) {
            if( !pc->iSuffix ) return Ptr2Indx(pc);
            pc = suff(pc);
            if( pc->NumStats ) {
                for( p=getStats(pc); p[0].Symbol!=sym; p++ );
                tmp = 2*(p->Freq<MAX_FREQ-3);
                p->Freq += tmp;
                pc->SummFreq += tmp;
            } else {
                p = &(pc->oneState());
                p->Freq += (p->Freq<11);
            }
            LOOP_ENTRY:
            if( p->iSuccessor ) break;
            p->iSuccessor = iUpBranch;
            OrderFall++;
        }
        if( p->iSuccessor<=iUpBranch ) {
            p1 = FoundState;
            FoundState = p;
            p->iSuccessor = CreateSuccessors(0,0,pc);
            FoundState = p1;
        }
        if( OrderFall==1 && pc1==MaxContext ) {
            FoundState->iSuccessor = p->iSuccessor;
            pText--;
        }
        return p->iSuccessor;
    }

    //---   #include "ppmd_proc0.inc"

    int  PrevSuccess;
    word BinSumm[25][64]; // binary SEE-contexts

    template< int ProcMode >
    void processBinSymbol( PPM_CONTEXT& q, int symbol ) {
        STATE& rs = q.oneState();
        int   i  = NS2BSIndx[suff(&q)->NumStats] + PrevSuccess + q.Flags + ((RunLength>>26) & 0x20);
        word& bs = BinSumm[QTable[rs.Freq-1]][i];
        BSumm    = bs;
        bs      -= (BSumm+64) >> PERIOD_BITS;
        int flag = ProcMode ? 0 : rs.Symbol!=symbol;
        //  rc_BProcess( BSumm+BSumm, flag );
        if( flag ) {
            CharMask[rs.Symbol] = EscCount;
            NumMasked = 0;
            PrevSuccess = 0;
            FoundState = 0;
        } else {
            bs += INTERVAL;
            rs.Freq += (rs.Freq<196);
            RunLength++;
            PrevSuccess = 1;
            FoundState = &rs;
        }
    }

    //---   #include "ppmd_proc1.inc"

    // encode in unmasked (maxorder) context
    template< int ProcMode >
    void processSymbol1( PPM_CONTEXT& q, int symbol ) {
        STATE* p = getStats(&q);
        int cnum  = q.NumStats;
        int i     = p[0].Symbol;
        int low   = 0;
        int freq  = p[0].Freq;
        int total = q.SummFreq;
        int flag;
        //  int mode;
        int count=0;
        //  rc_Arrange( total );
        if( ProcMode ) {
            //    count = rc_GetFreq( total );
            flag  = count<freq;
        } else {
            flag  = i==symbol;
        }
        if( flag ) {
            //    mode = 2;
            PrevSuccess = 0;//(2*freq>1*total);
            p[0].Freq  += 4;
            q.SummFreq += 4;
        } else {
            PrevSuccess = 0;
            for( low=freq,i=1; i<=cnum; i++ ) {
                freq = p[i].Freq;
                flag = ProcMode ? low+freq>count : p[i].Symbol==symbol;
                if( flag ) break;
                low += freq;
            }
            //    mode = 2+1+flag;
            if( flag ) {
                p[i].Freq  += 4;
                q.SummFreq += 4;
                if( p[i].Freq > p[i-1].Freq ) SWAP( p[i], p[i-1] ), i--;
                p = &p[i];
            } else {
                //if( q.iSuffix ) PrefetchData( suff(&q) );
                freq = total-low;
                NumMasked = cnum;
                for( i=0; i<=cnum; i++ ) CharMask[p[i].Symbol]=EscCount;
                p = NULL;
            }
        }
        //  rc_Process( low, freq, total );
        FoundState = p;
        if( p && (p[0].Freq>MAX_FREQ) ) FoundState=rescale(q,OrderFall,FoundState);
    }

    //---   #include "ppmd_proc2.inc"

    SEE2_CONTEXT SEE2Cont[23][32];
    SEE2_CONTEXT DummySEE2Cont;

    // encode in masked context
    template< int ProcMode >
    void processSymbol2( PPM_CONTEXT& q, int symbol ) {
        U8 px[256];
        STATE* p = getStats(&q);
        int c;
        int count=0;
        int low;
        int see_freq;
        //int freq;
        int cnum = q.NumStats;
        SEE2_CONTEXT* psee2c;
        if( cnum != 0xFF ) {
            psee2c = SEE2Cont[ QTable[cnum+3]-4 ];
            psee2c+= (q.SummFreq > 10*(cnum+1));
            psee2c+= 2*(2*cnum < suff(&q)->NumStats+NumMasked) + q.Flags;
            see_freq = psee2c->getMean()+1;
            //    if( see_freq==0 ) psee2c->Summ+=1, see_freq=1;
        } else {
            psee2c = &DummySEE2Cont;
            see_freq = 1;
        }
        int flag=0,pj,pl;
        int i,j;
        for( i=0,j=0,low=0; i<=cnum; i++ ) {
            c = p[i].Symbol;
            if( CharMask[c]!=EscCount ) {
                CharMask[c]=EscCount;
                low += p[i].Freq;
                if( ProcMode )
                px[j++] = i;
                else
                if( c==symbol ) flag=1,j=i,pl=low;
            }
        }
        int Total = see_freq + low;
        //  rc_Arrange( Total );
        if( ProcMode ) {
            //    count = rc_GetFreq( Total );
            flag = count<low;
        }
        if( flag ) {
            if( ProcMode ) {
                for( low=0, i=0; (low+=p[j=px[i]].Freq)<=count; i++ );
            } else {
                low = pl;
            }
            p+=j;
            //freq = p[0].Freq;
            if( see_freq>2 ) psee2c->Summ -= see_freq;
            psee2c->update();
            FoundState = p;
            p[0].Freq  += 4;
            q.SummFreq += 4; if( p[0].Freq > MAX_FREQ ) FoundState=rescale(q,OrderFall,FoundState);
            RunLength = InitRL;
            EscCount++;
        } else {
            low = Total;
            //freq = see_freq;
            NumMasked  = cnum;
            psee2c->Summ += Total-see_freq;
        }
        //  rc_Process( low-freq, freq, Total );
    }
    //---   #include "ppmd_procT.inc"

    struct qsym {
        word sym;
        word freq;
        word total;
        void store( U32 _sym, U32 _freq, U32 _total ) {
            sym=_sym; freq=_freq; total=_total;
        }
    };

    qsym SQ[1024];
    U32 SQ_ptr;
    U32 sqp[256]; // symbol probs
    U32 trF[256]; // binary tree, freqs
    U32 trT[256]; // binary tree, totals

    void ConvertSQ( void ) {
        U32 i,c,j,b,freq,total,prob,cnum;
        U32 cum = 0xFFFFFF00; // base coef, add 1 to each to remove zero probs
        cnum=256;
        for( i=0; i<256; i++ ) sqp[i]=0,trF[i]=0,trT[i]=0; // init for all symbols
        for( i=0; i<SQ_ptr; i++ ) {
            c = SQ[i].sym; freq = SQ[i].freq; total = SQ[i].total;
            prob = U64(U64(cum)*freq)/total;
            if( c<256 ) {
                sqp[c] = prob+1; cnum--;
            } else {
                cum = prob;
            }
        }
        // build a binary tree with ppmd probs
        for( c=0; c<256; c++ ) {
            for( i=8; i!=0; i-- ) {
                j = (256+c)>>i;
                b = (c>>(i-1))&1;
                if( b==0 ) trF[j]+=sqp[c];
                trT[j]+=sqp[c];
            }
        }
    }


    void processBinSymbol_T( PPM_CONTEXT& q, int ) {
        STATE& rs = q.oneState();
        int   i  = NS2BSIndx[suff(&q)->NumStats] + PrevSuccess + q.Flags + ((RunLength>>26) & 0x20);
        word& bs = BinSumm[QTable[rs.Freq-1]][i];
        BSumm    = bs;
        //  bs      -= (BSumm+64) >> PERIOD_BITS;
        SQ[SQ_ptr++].store(rs.Symbol,BSumm+BSumm,SCALE);
        SQ[SQ_ptr++].store(256,SCALE-BSumm-BSumm,SCALE); // escape
        CharMask[rs.Symbol] = EscCount;
        NumMasked = 0;
    }


    // encode in unmasked (maxorder) context
    void processSymbol1_T( PPM_CONTEXT& q, int ) {
        STATE* p = getStats(&q);
        int cnum  = q.NumStats;
        int low   = 0;
        int freq  = 0;
        int total = q.SummFreq;
        int i;
        for( i=0,low=0; i<=cnum; i++ ) {
            freq = p[i].Freq;
            SQ[SQ_ptr++].store(p[i].Symbol,freq,total);
            low += freq;
        }
       // if( q.iSuffix ) PrefetchData( suff(&q) );
        NumMasked = cnum;
        for( i=0; i<=cnum; i++ ) CharMask[p[i].Symbol]=EscCount;
        SQ[SQ_ptr++].store(256,total-low,total);
    }


    // encode in masked context
    void processSymbol2_T( PPM_CONTEXT& q, int ) {
        STATE* p = getStats(&q);
        int c;
        //int count;
        int low;
        int see_freq;
        //int freq;
        int cnum = q.NumStats;
        SEE2_CONTEXT* psee2c;
        if( cnum != 0xFF ) {
            psee2c = SEE2Cont[ QTable[cnum+3]-4 ];
            psee2c+= (q.SummFreq > 10*(cnum+1));
            psee2c+= 2*(2*cnum < suff(&q)->NumStats+NumMasked) + q.Flags;
            see_freq = psee2c->getMean()+1;
            //    if( see_freq==0 ) psee2c->Summ+=1, see_freq=1;
        } else {
            psee2c = &DummySEE2Cont;
            see_freq = 1;
        }
      //  int /*flag=0,*/pj,pl;
        int i;//,j;
        for( i=0,low=0; i<=cnum; i++ ) {
            c = p[i].Symbol;
            if( CharMask[c]!=EscCount ) low += p[i].Freq;
        }
        int Total = see_freq + low;
        for( i=0; i<=cnum; i++ ) {
            c = p[i].Symbol;
            if( CharMask[c]!=EscCount ) {
                SQ[SQ_ptr++].store(c,p[i].Freq,Total);
                CharMask[c]=EscCount;
            }
        }
        SQ[SQ_ptr++].store(256,see_freq,Total);
        NumMasked  = cnum;
    }

    U32 cxt; // bit context
    U32 y;   // prev bit

    U32 Init( U32 MaxOrder, U32 MMAX, U32 CutOff, U32 filesize ) {
        _MaxOrder = MaxOrder;
        _CutOff = CutOff;
        _MMAX = MMAX;
        _filesize = filesize;
        PPMD_STARTUP();
        //f_quit=0; coro_init();
        if( !StartSubAllocator( _MMAX ) ) return 1;
        StartModelRare();
        //printf( "f_DEC=%i ord=%i mem=%i cutoff=%i\n", ProcMode, _MaxOrder, _MMAX, _CutOff );
        cxt=0; y=1; // bit context
        return 0;
    }

    void Quit( void ) {
        StopSubAllocator();
    }

    //---   #include "ppmd_byte.inc"

    void ppmd_PrepareByte( void ) {
        SQ_ptr=0; NumMasked=0;
        int _OrderFall = OrderFall;
        PPM_CONTEXT* MinContext = MaxContext;
        if( MinContext->NumStats ) {
            processSymbol1_T(   MinContext[0], 0 );
        } else {
            processBinSymbol_T( MinContext[0], 0 );
        }
        while(1) {
            do {
                if( !MinContext->iSuffix ) goto Break;
                OrderFall++;
                MinContext = suff(MinContext);
            } while( MinContext->NumStats==NumMasked );
            processSymbol2_T( MinContext[0], 0 );
        }
Break:
        EscCount++; NumMasked=0; OrderFall=_OrderFall;
        ConvertSQ();
    }

    void ppmd_UpdateByte( U32 c ) {
        PPM_CONTEXT* MinContext = MaxContext;
        if( MinContext->NumStats ) {
            processSymbol1<0>(   MinContext[0], c );
        } else {
            processBinSymbol<0>( MinContext[0], c );
        }
        while( !FoundState ) {
            do {
                //      if( !MinContext->iSuffix ) { return -1; };
                OrderFall++;
                MinContext = suff(MinContext);
            } while( MinContext->NumStats==NumMasked );
            processSymbol2<0>( MinContext[0], c );
        }
        //  if( ProcMode ) c = FoundState->Symbol;
        PPM_CONTEXT* p;
        if( (OrderFall!=0) || ((U8*)getSucc(FoundState)<UnitsStart) ) {
            p = UpdateModel( MinContext );
            if( p ) MaxContext = p;
        } else {
            p = MaxContext = getSucc(FoundState);
        }
        if( p==0 ) {
            if( _CutOff ) {
                RestoreModelRare();
            } else {
                StartModelRare();
            }
        }
    }

    /*
U32 ProcessByte( U32 c ) {

ppmd_PrepareByte();

U32 i,j,p; int bit;
for( i=8,j=1; i!=0; i-- ) {
    bit = ProcMode ? 0 : (c>>(i-1))&1;
    p = U64(U64(SCALE-2)*trF[j])/trT[j]+1;
    rc_BProcess( p, bit );
    j += j + bit;
}
c = U8(j);

ppmd_UpdateByte(c);

return c;
}
*/

    U32 ppmd_Predict( U32 SCALE, U32 y ) {
        if( cxt==0 ) cxt=1; else cxt+=cxt+y;
        if( cxt>=256 ) ppmd_UpdateByte( U8(cxt) ), cxt=1;
        if( cxt==1 ) ppmd_PrepareByte();
        U32 p = U64(U64(SCALE-2)*trF[cxt])/trT[cxt]+1;
        return p;
    }

};


__attribute__((aligned(4096))) static ppmd_Model ppmd_12_256_1;
__attribute__((aligned(4096))) static ppmd_Model ppmd_6_32_1;

void ppmdModel( Mixer& m ) {
static int init_flag = 1;
if( init_flag ) {
    ppmd_12_256_1.Init(15,512,1,0);
    ppmd_6_32_1.Init(6,32,1,0);
}

m.add( stretch(4096-ppmd_12_256_1.ppmd_Predict(4096,y)) );
m.add( stretch(4096-ppmd_6_32_1.ppmd_Predict(4096,y)) );

init_flag=0;
}

#pragma pack()

template<typename T>
class IndirectContext1 {
private:
    Array<T> data;
    T *ctx;
    const uint32_t ctxMask, inputMask, inputBits, contextBits;

public:
    IndirectContext1(const int bitsPerContext, const int inputBits, const int contextBits = sizeof(T)*8) :
      data(UINT64_C(1) << bitsPerContext), ctx(&data[0]),
      ctxMask((UINT32_C(1) << bitsPerContext) - 1), 
      inputMask((UINT32_C(1) << inputBits) - 1), 
      inputBits(inputBits),
      contextBits(contextBits) {
      if (contextBits < sizeof(T) * 8) // need for leading bit -> include it
        reset(); 
    }

    void reset() {
      for (uint64_t i = 0; i < data.size(); ++i) {
        data[i] = contextBits < sizeof(T) * 8 ? 1 : 0; // 1: leading bit to indicate the number of used bits
      }
    }

    void operator+=(const uint32_t i) {
      // note: when the context is fully mature, we need to keep the leading bit in front of the contextbits as the MSB
      T leadingBit = (*ctx) & (1 << contextBits); 
      (*ctx) <<= inputBits;
      (*ctx) |= i | leadingBit;
      (*ctx) &= (1 << (contextBits + 1)) - 1;
    };

    void operator=(const uint32_t i) {
      ctx = &data[i & ctxMask];
    }

    auto operator()() -> T & {
      return *ctx;
    };
};

#include "lstm1.inc"
class lstmModel1 {
  APM apm2,apm3;
  const int horizon;
  IndirectContext1<uint16_t> iCtx;
  LSTM::ByteModel *lstm;
public:
  lstmModel1(U32 val=0):apm2{0x800u}, apm3{ 1024 },
  horizon(20),
  iCtx{ 11, 1, 9 }  { 
  srand(0xDEADBEEF);
  lstm=new LSTM::ByteModel(25, 3, horizon, 0.05);// num_cells, num_layers, horizon, learning_rate)
 }
 int inputs() {return 2+1+1+1;}
 int nets() {return (horizon<<3)+7+1+8*256;}
 int netcount() {return 1+1;}
 int p(Mixer& m){
    lstm->Perceive(y);
    int p=lstm->Predict();
     iCtx += y;
     iCtx = (bpos << 8) | lstm->expected();
    uint32_t ctx =  iCtx();
    m.add(stretch(p));
    m.add((p-2048)>>2);
    const int pr2 = apm2.p(p, (bpos<<8) |lstm->expected());
    const int pr3 = apm3.p(pr2, ctx);
    m.add(stretch(pr2)>>1);
    m.add(stretch(pr3)>>1);
    m.set((bpos<<8) | lstm->expected(), 8 * 256);
    m.set(lstm->epoch() << 3 | bpos, (horizon<<3)+7+1);
  return 0;
}
  virtual ~lstmModel1(){ /*delete lstm;*/}
};

// Language modelling


#define TAB 0x09
#define NEW_LINE 0x0A
#define CARRIAGE_RETURN 0x0D
#define SPACE 0x20

inline bool CharInArray(const char c, const char a[], const int len) {
  if (a==nullptr)
    return false;
  int i=0;
  for (; i<len && c!=a[i]; i++);
  return i<len;
}

#define MAX_WORD_SIZE 64

class Word {
public:
  U8 Letters[MAX_WORD_SIZE];
  U8 Start, End;
  U64 Hash[4], Type, Language;
  Word() : Start(0), End(0), Hash{0,0,0,0}, Type(0) {
    memset(&Letters[0], 0, sizeof(U8)*MAX_WORD_SIZE);
  }
  bool operator==(const char *s) const {
    size_t len=strlen(s);
    return ((size_t)(End-Start+(Letters[Start]!=0))==len && memcmp(&Letters[Start], s, len)==0);
  }
  bool operator!=(const char *s) const {
    return !operator==(s);
  }
  void operator+=(const char c) {
    if (End<MAX_WORD_SIZE-1) {
      End+=(Letters[End]>0);
      Letters[End]=tolower(c);
    }
  }
  U8 operator[](U8 i) const {
    return (End-Start>=i)?Letters[Start+i]:0;
  }
  U8 operator()(U8 i) const {
    return (End-Start>=i)?Letters[End-i]:0;
  }
  U32 Length() const {
    if (Letters[Start]!=0)
      return End-Start+1;
    return 0;
  }
  void GetHashes() {
    Hash[0] = 0xc01dflu, Hash[1] = ~Hash[0];
    for (int i=Start; i<=End; i++) {
      U8 l = Letters[i];
      Hash[0]^=hash(Hash[0], l, i);
      Hash[1]^=hash(Hash[1], 
        ((l&0x80)==0)?l&0x5F:
        ((l&0xC0)==0x80)?l&0x3F:
        ((l&0xE0)==0xC0)?l&0x1F:
        ((l&0xF0)==0xE0)?l&0xF:l&0x7
      );
    }
    Hash[2] = (~Hash[0])^Hash[1];
    Hash[3] = (~Hash[1])^Hash[0];
  }
  bool ChangeSuffix(const char *OldSuffix, const char *NewSuffix) {
    size_t len=strlen(OldSuffix);
    if (Length()>len && memcmp(&Letters[End-len+1], OldSuffix, len)==0) {
      size_t n=strlen(NewSuffix);
      if (n>0) {
        memcpy(&Letters[End-int(len)+1], NewSuffix, min(MAX_WORD_SIZE-1,End+int(n))-End);
        End=min(MAX_WORD_SIZE-1, End-int(len)+int(n));
      }
      else
        End-=U8(len);
      return true;
    }
    return false;
  }
  bool MatchesAny(const char* a[], const int count) {
    int i=0;
    size_t len = (size_t)Length();
    for (; i<count && (len!=strlen(a[i]) || memcmp(&Letters[Start], a[i], len)!=0); i++);
    return i<count;
  }
  bool EndsWith(const char *Suffix) const {
    size_t len=strlen(Suffix);
    return (Length()>len && memcmp(&Letters[End-len+1], Suffix, len)==0);
  }
  bool StartsWith(const char *Prefix) const {
    size_t len=strlen(Prefix);
    return (Length()>len && memcmp(&Letters[Start], Prefix, len)==0);
  }
};

class Segment {
public:
  Word FirstWord; // useful following questions
  U32 WordCount;
  U32 NumCount;
};

class Sentence : public Segment {
public:
  enum Types { // possible sentence types, excluding Imperative
    Declarative,
    Interrogative,
    Exclamative,
    Count
  };
  Types Type;
  U32 SegmentCount;
  U32 VerbIndex; // relative position of last detected verb
  U32 NounIndex; // relative position of last detected noun
  U32 CapitalIndex; // relative position of last capitalized word, excluding the initial word of this sentence
  Word lastVerb, lastNoun, lastCapital;
};

class Paragraph {
public:
  U32 SentenceCount, TypeCount[Sentence::Types::Count], TypeMask;
};

class Language {
public:
  enum Flags {
    Verb                   = (1<<0),
    Noun                   = (1<<1)
  };
  enum Ids {
    Unknown,
    English,
    French,
    German,
    Count
  };
  virtual ~Language() {};
  virtual bool IsAbbreviation(Word *W) = 0;
};

class English: public Language {
private:
  static const int NUM_ABBREV = 6;
  const char *Abbreviations[NUM_ABBREV]={ "mr","mrs","ms","dr","st","jr" };
public:
  enum Flags {
    Adjective              = (1<<2),
    Plural                 = (1<<3),
    Male                   = (1<<4),
    Female                 = (1<<5),
    Negation               = (1<<6),
    PastTense              = (1<<7)|Verb,
    PresentParticiple      = (1<<8)|Verb,
    AdjectiveSuperlative   = (1<<9)|Adjective,
    AdjectiveWithout       = (1<<10)|Adjective,
    AdjectiveFull          = (1<<11)|Adjective,
    AdverbOfManner         = (1<<12),
    SuffixNESS             = (1<<13),
    SuffixITY              = (1<<14)|Noun,
    SuffixCapable          = (1<<15),
    SuffixNCE              = (1<<16),
    SuffixNT               = (1<<17),
    SuffixION              = (1<<18),
    SuffixAL               = (1<<19)|Adjective,
    SuffixIC               = (1<<20)|Adjective,
    SuffixIVE              = (1<<21),
    SuffixOUS              = (1<<22)|Adjective,
    PrefixOver             = (1<<23),
    PrefixUnder            = (1<<24)
  };
  bool IsAbbreviation(Word *W) { return W->MatchesAny(Abbreviations, NUM_ABBREV); };
};

class French: public Language {
private:
  static const int NUM_ABBREV = 2;
  const char *Abbreviations[NUM_ABBREV]={ "m","mm" };
public:
  enum Flags {
    Adjective              = (1<<2),
    Plural                 = (1<<3)
  };
  bool IsAbbreviation(Word *W) { return W->MatchesAny(Abbreviations, NUM_ABBREV); };
};

class German : public Language {
private:
  static const int NUM_ABBREV = 3;
  const char *Abbreviations[NUM_ABBREV]={ "fr","hr","hrn" };
public:
  enum Flags {
    Adjective              = (1<<2),
    Plural                 = (1<<3),
    Female                 = (1<<4)
  };
  bool IsAbbreviation(Word *W) { return W->MatchesAny(Abbreviations, NUM_ABBREV); };
};

//////////////////////////// Stemming routines /////////////////////////

class Stemmer {
protected:
  U32 GetRegion(const Word *W, const U32 From) {
    bool hasVowel = false;
    for (int i=W->Start+From; i<=W->End; i++) {
      if (IsVowel(W->Letters[i])) {
        hasVowel = true;
        continue;
      }
      else if (hasVowel)
        return i-W->Start+1;
    }
    return W->Start+W->Length();
  }
  bool SuffixInRn(const Word *W, const U32 Rn, const char *Suffix) {
    return (W->Start!=W->End && Rn<=W->Length()-strlen(Suffix));
  }
public:
  virtual ~Stemmer() {};
  virtual bool IsVowel(const char c) = 0;
  virtual void Hash(Word *W) = 0;
  virtual bool Stem(Word *W) = 0;
};

/*
  English affix stemmer, based on the Porter2 stemmer.

  Changelog:
  (29/12/2017) v127: Initial release by Márcio Pais
  (02/01/2018) v128: Small changes to allow for compilation with MSVC
  Fix buffer overflow (thank you Jan Ondrus)
  (28/01/2018) v133: Refactoring, added processing of "non/non-" prefixes
  (04/02/2018) v135: Refactoring, added processing of gender-specific words and common words
*/

class EnglishStemmer: public Stemmer {
private:
  static const int NUM_VOWELS = 6;
  const char Vowels[NUM_VOWELS]={'a','e','i','o','u','y'};
  static const int NUM_DOUBLES = 9;
  const char Doubles[NUM_DOUBLES]={'b','d','f','g','m','n','p','r','t'};
  static const int NUM_LI_ENDINGS = 10;
  const char LiEndings[NUM_LI_ENDINGS]={'c','d','e','g','h','k','m','n','r','t'};
  static const int NUM_NON_SHORT_CONSONANTS = 3;
  const char NonShortConsonants[NUM_NON_SHORT_CONSONANTS]={'w','x','Y'};
  static const int NUM_MALE_WORDS = 9;
  const char *MaleWords[NUM_MALE_WORDS]={"he","him","his","himself","man","men","boy","husband","actor"};
  static const int NUM_FEMALE_WORDS = 8;
  const char *FemaleWords[NUM_FEMALE_WORDS]={"she","her","herself","woman","women","girl","wife","actress"};
  static const int NUM_COMMON_WORDS = 12;
  const char *CommonWords[NUM_COMMON_WORDS]={"the","be","to","of","and","in","that","you","have","with","from","but"};
  static const int NUM_SUFFIXES_STEP0 = 3;
  const char *SuffixesStep0[NUM_SUFFIXES_STEP0]={"'s'","'s","'"};
  static const int NUM_SUFFIXES_STEP1b = 6;
  const char *SuffixesStep1b[NUM_SUFFIXES_STEP1b]={"eedly","eed","ed","edly","ing","ingly"};
  const U32 TypesStep1b[NUM_SUFFIXES_STEP1b]={English::AdverbOfManner,0,English::PastTense,English::AdverbOfManner|English::PastTense,English::PresentParticiple,English::AdverbOfManner|English::PresentParticiple};
  static const int NUM_SUFFIXES_STEP2 = 22;
  const char *(SuffixesStep2[NUM_SUFFIXES_STEP2])[2]={
    {"ization", "ize"},
    {"ational", "ate"},
    {"ousness", "ous"},
    {"iveness", "ive"},
    {"fulness", "ful"},
    {"tional", "tion"},
    {"lessli", "less"},
    {"biliti", "ble"},
    {"entli", "ent"},
    {"ation", "ate"},
    {"alism", "al"},
    {"aliti", "al"},
    {"fulli", "ful"},
    {"ousli", "ous"},
    {"iviti", "ive"},
    {"enci", "ence"},
    {"anci", "ance"},
    {"abli", "able"},
    {"izer", "ize"},
    {"ator", "ate"},
    {"alli", "al"},
    {"bli", "ble"}
  };
  const U32 TypesStep2[NUM_SUFFIXES_STEP2]={
    English::SuffixION,
    English::SuffixION|English::SuffixAL,
    English::SuffixNESS,
    English::SuffixNESS,
    English::SuffixNESS,
    English::SuffixION|English::SuffixAL,
    English::AdverbOfManner,
    English::AdverbOfManner|English::SuffixITY,
    English::AdverbOfManner,
    English::SuffixION,
    0,
    English::SuffixITY,
    English::AdverbOfManner,
    English::AdverbOfManner,
    English::SuffixITY,
    0,
    0,
    English::AdverbOfManner,
    0,
    0,
    English::AdverbOfManner,
    English::AdverbOfManner
  };
  static const int NUM_SUFFIXES_STEP3 = 8;
  const char *(SuffixesStep3[NUM_SUFFIXES_STEP3])[2]={
    {"ational", "ate"},
    {"tional", "tion"},
    {"alize", "al"},
    {"icate", "ic"},
    {"iciti", "ic"},
    {"ical", "ic"},
    {"ful", ""},
    {"ness", ""}
  };
  const U32 TypesStep3[NUM_SUFFIXES_STEP3]={English::SuffixION|English::SuffixAL,English::SuffixION|English::SuffixAL,0,0,English::SuffixITY,English::SuffixAL,English::AdjectiveFull,English::SuffixNESS};
  static const int NUM_SUFFIXES_STEP4 = 20;
  const char *SuffixesStep4[NUM_SUFFIXES_STEP4]={"al","ance","ence","er","ic","able","ible","ant","ement","ment","ent","ou","ism","ate","iti","ous","ive","ize","sion","tion"};
  const U32 TypesStep4[NUM_SUFFIXES_STEP4]={
    English::SuffixAL,
    English::SuffixNCE,
    English::SuffixNCE,
    0,
    English::SuffixIC,
    English::SuffixCapable,
    English::SuffixCapable,
    English::SuffixNT,
    0,
    0,
    English::SuffixNT,
    0,
    0,
    0,
    English::SuffixITY,
    English::SuffixOUS,
    English::SuffixIVE,
    0,
    English::SuffixION,
    English::SuffixION
  };
  static const int NUM_EXCEPTION_REGION1 = 3;
  const char *ExceptionsRegion1[NUM_EXCEPTION_REGION1]={"gener","arsen","commun"};
  static const int NUM_EXCEPTIONS1 = 18;
  const char *(Exceptions1[NUM_EXCEPTIONS1])[2]={
    {"skis", "ski"},
    {"skies", "sky"},
    {"dying", "die"},
    {"lying", "lie"},
    {"tying", "tie"},
    {"idly", "idle"},
    {"gently", "gentle"},
    {"ugly", "ugli"},
    {"early", "earli"},
    {"only", "onli"},
    {"singly", "singl"},
    {"sky", "sky"},
    {"news", "news"},
    {"howe", "howe"},
    {"atlas", "atlas"},
    {"cosmos", "cosmos"},
    {"bias", "bias"},
    {"andes", "andes"}
  };
  const U32 TypesExceptions1[NUM_EXCEPTIONS1]={
    English::Noun|English::Plural,
    English::Plural,
    English::PresentParticiple,
    English::PresentParticiple,
    English::PresentParticiple,
    English::AdverbOfManner,
    English::AdverbOfManner,
    English::Adjective,
    English::Adjective|English::AdverbOfManner,
    0,
    English::AdverbOfManner,
    English::Noun,
    English::Noun,
    0,
    English::Noun,
    English::Noun,
    English::Noun,
    0
  };
  static const int NUM_EXCEPTIONS2 = 8;
  const char *Exceptions2[NUM_EXCEPTIONS2]={"inning","outing","canning","herring","earring","proceed","exceed","succeed"};
  const U32 TypesExceptions2[NUM_EXCEPTIONS2]={English::Noun,English::Noun,English::Noun,English::Noun,English::Noun,English::Verb,English::Verb,English::Verb}; 
  inline bool IsConsonant(const char c) {
    return !IsVowel(c);
  }
  inline bool IsShortConsonant(const char c) {
    return !CharInArray(c, NonShortConsonants, NUM_NON_SHORT_CONSONANTS);
  }
  inline bool IsDouble(const char c) {
    return CharInArray(c, Doubles, NUM_DOUBLES);
  }
  inline bool IsLiEnding(const char c) {
    return CharInArray(c, LiEndings, NUM_LI_ENDINGS);
  }
  U32 GetRegion1(const Word *W) {
    for (int i=0; i<NUM_EXCEPTION_REGION1; i++) {
      if (W->StartsWith(ExceptionsRegion1[i]))
        return U32(strlen(ExceptionsRegion1[i]));
    }
    return GetRegion(W, 0);
  }
  bool EndsInShortSyllable(const Word *W) {
    if (W->End==W->Start)
      return false;
    else if (W->End==W->Start+1)
      return IsVowel((*W)(1)) && IsConsonant((*W)(0));
    else
      return (IsConsonant((*W)(2)) && IsVowel((*W)(1)) && IsConsonant((*W)(0)) && IsShortConsonant((*W)(0)));
  }
  bool IsShortWord(const Word *W) {
    return (EndsInShortSyllable(W) && GetRegion1(W)==W->Length());
  }
  inline bool HasVowels(const Word *W) {
    for (int i=W->Start; i<=W->End; i++) {
      if (IsVowel(W->Letters[i]))
        return true;
    }
    return false;
  }
  bool TrimStartingApostrophe(Word *W) {
    bool r=(W->Start!=W->End && (*W)[0]=='\'');
    W->Start+=(U8)r;
    return r;
  }
  void MarkYsAsConsonants(Word *W) {
    if ((*W)[0]=='y')
      W->Letters[W->Start]='Y';
    for (int i=W->Start+1; i<=W->End; i++) {
      if (IsVowel(W->Letters[i-1]) && W->Letters[i]=='y')
        W->Letters[i]='Y';
    }
  }
  bool ProcessPrefixes(Word *W) {
    if (W->StartsWith("irr") && W->Length()>5 && ((*W)[3]=='a' || (*W)[3]=='e'))
      W->Start+=2, W->Type|=English::Negation;
    else if (W->StartsWith("over") && W->Length()>5)
      W->Start+=4, W->Type|=English::PrefixOver;
    else if (W->StartsWith("under") && W->Length()>6)
      W->Start+=5, W->Type|=English::PrefixUnder;
    else if (W->StartsWith("unn") && W->Length()>5)
      W->Start+=2, W->Type|=English::Negation;
    else if (W->StartsWith("non") && W->Length()>(U32)(5+((*W)[3]=='-')))
      W->Start+=2+((*W)[3]=='-'), W->Type|=English::Negation;
    else
      return false;
    return true;
  }
  bool ProcessSuperlatives(Word *W) {
    if (W->EndsWith("est") && W->Length()>4) {
      U8 i=W->End;
      W->End-=3;
      W->Type|=English::AdjectiveSuperlative;

      if ((*W)(0)==(*W)(1) && (*W)(0)!='r' && !(W->Length()>=4 && memcmp("sugg",&W->Letters[W->End-3],4)==0)) {
        W->End-= ( ((*W)(0)!='f' && (*W)(0)!='l' && (*W)(0)!='s') ||
                   (W->Length()>4 && (*W)(1)=='l' && ((*W)(2)=='u' || (*W)(3)=='u' || (*W)(3)=='v'))) &&
                   (!(W->Length()==3 && (*W)(1)=='d' && (*W)(2)=='o'));
        if (W->Length()==2 && ((*W)[0]!='i' || (*W)[1]!='n'))
          W->End = i, W->Type&=~English::AdjectiveSuperlative;
      }
      else {
        switch((*W)(0)) {
          case 'd': case 'k': case 'm': case 'y': break;
          case 'g': {
            if (!( W->Length()>3 && ((*W)(1)=='n' || (*W)(1)=='r') && memcmp("cong",&W->Letters[W->End-3],4)!=0 ))
              W->End = i, W->Type&=~English::AdjectiveSuperlative;
            else
              W->End+=((*W)(2)=='a');
            break;
          }
          case 'i': { W->Letters[W->End]='y'; break; }
          case 'l': {
            if (W->End==W->Start+1 || memcmp("mo",&W->Letters[W->End-2],2)==0)
              W->End = i, W->Type&=~English::AdjectiveSuperlative;
            else
              W->End+=IsConsonant((*W)(1));
            break;
          }
          case 'n': {
            if (W->Length()<3 || IsConsonant((*W)(1)) || IsConsonant((*W)(2)))
              W->End = i, W->Type&=~English::AdjectiveSuperlative;
            break;
          }
          case 'r': {
            if (W->Length()>3 && IsVowel((*W)(1)) && IsVowel((*W)(2)))
              W->End+=((*W)(2)=='u') && ((*W)(1)=='a' || (*W)(1)=='i');
            else
              W->End = i, W->Type&=~English::AdjectiveSuperlative;
            break;
          }
          case 's': { W->End++; break; }
          case 'w': {
            if (!(W->Length()>2 && IsVowel((*W)(1))))
              W->End = i, W->Type&=~English::AdjectiveSuperlative;
            break;
          }
          case 'h': {
            if (!(W->Length()>2 && IsConsonant((*W)(1))))
              W->End = i, W->Type&=~English::AdjectiveSuperlative;
            break;
          }
          default: {
            W->End+=3;
            W->Type&=~English::AdjectiveSuperlative;
          }
        }
      }
    }
    return (W->Type&English::AdjectiveSuperlative)>0;
  }
  bool Step0(Word *W) {
    for (int i=0; i<NUM_SUFFIXES_STEP0; i++) {
      if (W->EndsWith(SuffixesStep0[i])) {
        W->End-=U8(strlen(SuffixesStep0[i]));
        W->Type|=English::Plural;
        return true;
      }
    }
    return false;
  }
  bool Step1a(Word *W) {
    if (W->EndsWith("sses")) {
      W->End-=2;
      W->Type|=English::Plural;
      return true;
    }
    if (W->EndsWith("ied") || W->EndsWith("ies")) {
      W->Type|=((*W)(0)=='d')?English::PastTense:English::Plural;
      W->End-=1+(W->Length()>4);
      return true;
    }
    if (W->EndsWith("us") || W->EndsWith("ss"))
      return false;
    if ((*W)(0)=='s' && W->Length()>2) {
      for (int i=W->Start;i<=W->End-2;i++) {
        if (IsVowel(W->Letters[i])) {
          W->End--;
          W->Type|=English::Plural;
          return true;
        }
      }
    }
    if (W->EndsWith("n't") && W->Length()>4) {
      switch ((*W)(3)) {
        case 'a': {
          if ((*W)(4)=='c')
            W->End-=2;
          else
            W->ChangeSuffix("n't","ll");
          break;
        }
        case 'i': { W->ChangeSuffix("in't","m"); break; }
        case 'o': {
          if ((*W)(4)=='w')
            W->ChangeSuffix("on't","ill");
          else
            W->End-=3;
          break;
        }
        default: W->End-=3;
      }
      W->Type|=English::Negation;
      return true;
    }
    if (W->EndsWith("hood") && W->Length()>7) {
      W->End-=4;
      return true;
    }
    return false;
  }
  bool Step1b(Word *W, const U32 R1) {
    for (int i=0; i<NUM_SUFFIXES_STEP1b; i++) {
      if (W->EndsWith(SuffixesStep1b[i])) {
        switch(i) {
          case 0: case 1: {
            if (SuffixInRn(W, R1, SuffixesStep1b[i]))
              W->End-=1+i*2;
            break;
          }
          default: {
            U8 j=W->End;
            W->End-=U8(strlen(SuffixesStep1b[i]));
            if (HasVowels(W)) {
              if (W->EndsWith("at") || W->EndsWith("bl") || W->EndsWith("iz") || IsShortWord(W))
                (*W)+='e';
              else if (W->Length()>2) {
                if ((*W)(0)==(*W)(1) && IsDouble((*W)(0)))
                  W->End--;
                else if (i==2 || i==3) {
                  switch((*W)(0)) {
                    case 'c': case 's': case 'v': { W->End+=!(W->EndsWith("ss") || W->EndsWith("ias")); break; }
                    case 'd': {
                      static const char nAllowed[4] = {'a','e','i','o'};
                      W->End+=IsVowel((*W)(1)) && (!CharInArray((*W)(2), nAllowed, 4)); break;
                    }
                    case 'k': { W->End+=W->EndsWith("uak"); break; }
                    case 'l': {
                      static const char Allowed1[10] = {'b','c','d','f','g','k','p','t','y','z'};
                      static const char Allowed2[4] = {'a','i','o','u'};
                      W->End+= CharInArray((*W)(1), Allowed1, 10) ||
                                (CharInArray((*W)(1), Allowed2, 4) && IsConsonant((*W)(2)));
                      break;
                    }
                  }
                }
                else if (i>=4) {
                  switch((*W)(0)) {
                    case 'd': {
                      if (IsVowel((*W)(1)) && (*W)(2)!='a' && (*W)(2)!='e' && (*W)(2)!='o')
                        (*W)+='e';
                      break;
                    }
                    case 'g': {
                      static const char Allowed[7] = {'a','d','e','i','l','r','u'};
                      if (
                        CharInArray((*W)(1), Allowed, 7) || (
                         (*W)(1)=='n' && (
                          (*W)(2)=='e' ||
                          ((*W)(2)=='u' && (*W)(3)!='b' && (*W)(3)!='d') ||
                          ((*W)(2)=='a' && ((*W)(3)=='r' || ((*W)(3)=='h' && (*W)(4)=='c'))) ||
                          (W->EndsWith("ring") && ((*W)(4)=='c' || (*W)(4)=='f'))
                         )
                        ) 
                      )
                        (*W)+='e';
                      break;
                    }
                    case 'l': {
                      if (!((*W)(1)=='l' || (*W)(1)=='r' || (*W)(1)=='w' || (IsVowel((*W)(1)) && IsVowel((*W)(2)))))
                        (*W)+='e';
                      if (W->EndsWith("uell") && W->Length()>4 && (*W)(4)!='q')
                        W->End--;
                      break;
                    }
                    case 'r': {
                      if ((
                        ((*W)(1)=='i' && (*W)(2)!='a' && (*W)(2)!='e' && (*W)(2)!='o') ||
                        ((*W)(1)=='a' && (!((*W)(2)=='e' || (*W)(2)=='o' || ((*W)(2)=='l' && (*W)(3)=='l')))) ||
                        ((*W)(1)=='o' && (!((*W)(2)=='o' || ((*W)(2)=='t' && (*W)(3)!='s')))) ||
                        (*W)(1)=='c' || (*W)(1)=='t') && (!W->EndsWith("str"))
                      )
                        (*W)+='e';
                      break;
                    }
                    case 't': {
                      if ((*W)(1)=='o' && (*W)(2)!='g' && (*W)(2)!='l' && (*W)(2)!='i' && (*W)(2)!='o')
                        (*W)+='e';
                      break;
                    }
                    case 'u': {
                      if (!(W->Length()>3 && IsVowel((*W)(1)) && IsVowel((*W)(2))))
                        (*W)+='e';
                      break;
                    }
                    case 'z': {
                      if (W->EndsWith("izz") && W->Length()>3 && ((*W)(3)=='h' || (*W)(3)=='u'))
                        W->End--;
                      else if ((*W)(1)!='t' && (*W)(1)!='z')
                        (*W)+='e';
                      break;
                    }
                    case 'k': {
                      if (W->EndsWith("uak"))
                        (*W)+='e';
                      break;
                    }
                    case 'b': case 'c': case 's': case 'v': {
                      if (!(
                        ((*W)(0)=='b' && ((*W)(1)=='m' || (*W)(1)=='r')) ||
                        W->EndsWith("ss") || W->EndsWith("ias") || (*W)=="zinc"
                      ))
                        (*W)+='e';
                      break;
                    }
                  }
                }
              }
            }
            else {
              W->End=j;
              return false;
            }
          }
        }
        W->Type|=TypesStep1b[i];
        return true;
      }
    }
    return false;
  }
  bool Step1c(Word *W) {
    if (W->Length()>2 && tolower((*W)(0))=='y' && IsConsonant((*W)(1))) {
      W->Letters[W->End]='i';
      return true;
    }
    return false;
  }
  bool Step2(Word *W, const U32 R1) {
    for (int i=0; i<NUM_SUFFIXES_STEP2; i++) {
      if (W->EndsWith(SuffixesStep2[i][0]) && SuffixInRn(W, R1, SuffixesStep2[i][0])) {
        W->ChangeSuffix(SuffixesStep2[i][0], SuffixesStep2[i][1]);
        W->Type|=TypesStep2[i];
        return true;
      }
    }
    if (W->EndsWith("logi") && SuffixInRn(W, R1, "ogi")) {
      W->End--;
      return true;
    }
    else if (W->EndsWith("li")) {
      if (SuffixInRn(W, R1, "li") && IsLiEnding((*W)(2))) {
        W->End-=2;
        W->Type|=English::AdverbOfManner;
        return true;
      }
      else if (W->Length()>3) {
        switch((*W)(2)) {
            case 'b': {
              W->Letters[W->End]='e';
              W->Type|=English::AdverbOfManner;
              return true;              
            }
            case 'i': {
              if (W->Length()>4) {
                W->End-=2;
                W->Type|=English::AdverbOfManner;
                return true;
              }
              break;
            }
            case 'l': {
              if (W->Length()>5 && ((*W)(3)=='a' || (*W)(3)=='u')) {
                W->End-=2;
                W->Type|=English::AdverbOfManner;
                return true;
              }
              break;
            }
            case 's': {
              W->End-=2;
              W->Type|=English::AdverbOfManner;
              return true;
            }
            case 'e': case 'g': case 'm': case 'n': case 'r': case 'w': {
              if (W->Length()>(U32)(4+((*W)(2)=='r'))) {
                W->End-=2;
                W->Type|=English::AdverbOfManner;
                return true;
              }
            }
        }
      }
    }
    return false;
  }
  bool Step3(Word *W, const U32 R1, const U32 R2) {
    bool res=false;
    for (int i=0; i<NUM_SUFFIXES_STEP3; i++) {
      if (W->EndsWith(SuffixesStep3[i][0]) && SuffixInRn(W, R1, SuffixesStep3[i][0])) {
        W->ChangeSuffix(SuffixesStep3[i][0], SuffixesStep3[i][1]);
        W->Type|=TypesStep3[i];
        res=true;
        break;
      }
    }
    if (W->EndsWith("ative") && SuffixInRn(W, R2, "ative")) {
      W->End-=5;
      W->Type|=English::SuffixIVE;
      return true;
    }
    if (W->Length()>5 && W->EndsWith("less")) {
      W->End-=4;
      W->Type|=English::AdjectiveWithout;
      return true;
    }
    return res;
  }
  bool Step4(Word *W, const U32 R2) {
    bool res=false;
    for (int i=0; i<NUM_SUFFIXES_STEP4; i++) {
      if (W->EndsWith(SuffixesStep4[i]) && SuffixInRn(W, R2, SuffixesStep4[i])) {
        W->End-=U8(strlen(SuffixesStep4[i])-(i>17));
        if (i!=10 || (*W)(0)!='m')
          W->Type|=TypesStep4[i];
        if (i==0 && W->EndsWith("nti")) {
          W->End--;
          res=true;
          continue;
        }
        return true;
      }
    }
    return res;
  }
  bool Step5(Word *W, const U32 R1, const U32 R2) {
    if ((*W)(0)=='e' && (*W)!="here") {
      if (SuffixInRn(W, R2, "e"))
        W->End--;
      else if (SuffixInRn(W, R1, "e")) {
        W->End--;
        W->End+=EndsInShortSyllable(W);
      }
      else
        return false;
      return true;
    }
    else if (W->Length()>1 && (*W)(0)=='l' && SuffixInRn(W, R2, "l") && (*W)(1)=='l') {
      W->End--;
      return true;
    }
    return false;
  }
public:
  inline bool IsVowel(const char c) final {
    return CharInArray(c, Vowels, NUM_VOWELS);
  }
  inline void Hash(Word *W) final {
    W->Hash[2] = W->Hash[3] = 0xb0a710ad;
    for (int i=W->Start; i<=W->End; i++) {
      U8 l = W->Letters[i];
      W->Hash[2]=W->Hash[2]*263*32 + l;
      if (IsVowel(l))
        W->Hash[3]=W->Hash[3]*997*8 + (l/4-22);
      else if (l>='b' && l<='z')
        W->Hash[3]=W->Hash[3]*271*32 + (l-97);
      else
        W->Hash[3]=W->Hash[3]*11*32 + l;
    }
  }
  bool Stem(Word *W) {
    if (W->Length()<2) {
      Hash(W);
      return false;
    }
    bool res = TrimStartingApostrophe(W);
    res|=ProcessPrefixes(W);
    res|=ProcessSuperlatives(W);
    for (int i=0; i<NUM_EXCEPTIONS1; i++) {
      if ((*W)==Exceptions1[i][0]) {
        if (i<11) {
          size_t len=strlen(Exceptions1[i][1]);
          memcpy(&W->Letters[W->Start], Exceptions1[i][1], len);
          W->End=W->Start+U8(len-1);
        }
        Hash(W);
        W->Type|=TypesExceptions1[i];
        W->Language = Language::English;
        return (i<11);
      }
    }

    // Start of modified Porter2 Stemmer
    MarkYsAsConsonants(W);
    U32 R1=GetRegion1(W), R2=GetRegion(W,R1);
    res|=Step0(W);
    res|=Step1a(W);
    for (int i=0; i<NUM_EXCEPTIONS2; i++) {
      if ((*W)==Exceptions2[i]) {
        Hash(W);
        W->Type|=TypesExceptions2[i];
        W->Language = Language::English;
        return res;
      }
    }
    res|=Step1b(W, R1);
    res|=Step1c(W);
    res|=Step2(W, R1);
    res|=Step3(W, R1, R2);
    res|=Step4(W, R2);
    res|=Step5(W, R1, R2);

    for (U8 i=W->Start; i<=W->End; i++) {
      if (W->Letters[i]=='Y')
        W->Letters[i]='y';
    }
    if (!W->Type || W->Type==English::Plural) {
      if (W->MatchesAny(MaleWords, NUM_MALE_WORDS))
        res = true, W->Type|=English::Male;
      else if (W->MatchesAny(FemaleWords, NUM_FEMALE_WORDS))
        res = true, W->Type|=English::Female;
    }
    if (!res)
      res=W->MatchesAny(CommonWords, NUM_COMMON_WORDS);
    Hash(W);
    if (res)
      W->Language = Language::English;
    return res;
  }
};

/*
  French suffix stemmer, based on the Porter stemmer.

  Changelog:
  (28/01/2018) v133: Initial release by Márcio Pais
  (04/02/2018) v135: Added processing of common words
  (25/02/2018) v139: Added UTF8 conversion
*/

class FrenchStemmer: public Stemmer {
private:
  static const int NUM_VOWELS = 17;
  const char Vowels[NUM_VOWELS]={'a','e','i','o','u','y','\xE2','\xE0','\xEB','\xE9','\xEA','\xE8','\xEF','\xEE','\xF4','\xFB','\xF9'};
  static const int NUM_COMMON_WORDS = 10;
  const char *CommonWords[NUM_COMMON_WORDS]={"de","la","le","et","en","un","une","du","que","pas"};
  static const int NUM_EXCEPTIONS = 3;
  const char *(Exceptions[NUM_EXCEPTIONS])[2]={
    {"monument", "monument"},
    {"yeux", "oeil"},
    {"travaux", "travail"},
  };
  const U32 TypesExceptions[NUM_EXCEPTIONS]={
    French::Noun,
    French::Noun|French::Plural,
    French::Noun|French::Plural
  };
  static const int NUM_SUFFIXES_STEP1 = 39;
  const char *SuffixesStep1[NUM_SUFFIXES_STEP1]={
    "ance","iqUe","isme","able","iste","eux","ances","iqUes","ismes","ables","istes", //11
    "atrice","ateur","ation","atrices","ateurs","ations", //6
    "logie","logies", //2
    "usion","ution","usions","utions", //4
    "ence","ences", //2
    "issement","issements", //2
    "ement","ements", //2
    "it\xE9","it\xE9s", //2
    "if","ive","ifs","ives", //4
    "euse","euses", //2
    "ment","ments" //2
  };
  static const int NUM_SUFFIXES_STEP2a = 35;
  const char *SuffixesStep2a[NUM_SUFFIXES_STEP2a]={
    "issaIent", "issantes", "iraIent", "issante",
    "issants", "issions", "irions", "issais",
    "issait", "issant", "issent", "issiez", "issons",
    "irais", "irait", "irent", "iriez", "irons",
    "iront", "isses", "issez", "\xEEmes",
    "\xEEtes", "irai", "iras", "irez", "isse",
    "ies", "ira", "\xEEt", "ie", "ir", "is",
    "it", "i"
  };
  static const int NUM_SUFFIXES_STEP2b = 38;
  const char *SuffixesStep2b[NUM_SUFFIXES_STEP2b]={
    "eraIent", "assions", "erions", "assent",
    "assiez", "\xE8rent", "erais", "erait",
    "eriez", "erons", "eront", "aIent", "antes",
    "asses", "ions", "erai", "eras", "erez",
    "\xE2mes", "\xE2tes", "ante", "ants",
    "asse", "\xE9""es", "era", "iez", "ais",
    "ait", "ant", "\xE9""e", "\xE9s", "er",
    "ez", "\xE2t", "ai", "as", "\xE9", "a"
  };
  static const int NUM_SET_STEP4 = 6;
  const char SetStep4[NUM_SET_STEP4]={'a','i','o','u','\xE8','s'};
  static const int NUM_SUFFIXES_STEP4 = 7;
  const char *SuffixesStep4[NUM_SUFFIXES_STEP4]={"i\xE8re","I\xE8re","ion","ier","Ier","e","\xEB"};
  static const int NUM_SUFFIXES_STEP5 = 5;
  const char *SuffixesStep5[NUM_SUFFIXES_STEP5]={"enn","onn","ett","ell","eill"};
  inline bool IsConsonant(const char c) {
    return !IsVowel(c);
  }
  void ConvertUTF8(Word *W) {
    for (int i=W->Start; i<W->End; i++) {
      U8 c = W->Letters[i+1]+((W->Letters[i+1]<0xA0)?0x60:0x40);
      if (W->Letters[i]==0xC3 && (IsVowel(c) || (W->Letters[i+1]&0xDF)==0x87)) {
        W->Letters[i] = c;
        if (i+1<W->End)
          memmove(&W->Letters[i+1], &W->Letters[i+2], W->End-i-1);
        W->End--;
      }
    }
  }
  void MarkVowelsAsConsonants(Word *W) {
    for (int i=W->Start; i<=W->End; i++) {
      switch (W->Letters[i]) {
        case 'i': case 'u': {
          if (i>W->Start && i<W->End && (IsVowel(W->Letters[i-1]) || (W->Letters[i-1]=='q' && W->Letters[i]=='u')) && IsVowel(W->Letters[i+1]))
            W->Letters[i] = toupper(W->Letters[i]);
          break;
        }
        case 'y': {
          if ((i>W->Start && IsVowel(W->Letters[i-1])) || (i<W->End && IsVowel(W->Letters[i+1])))
            W->Letters[i] = toupper(W->Letters[i]);
        }
      }
    }
  }
  U32 GetRV(Word *W) {
    U32 len = W->Length(), res = W->Start+len;
    if (len>=3 && ((IsVowel(W->Letters[W->Start]) && IsVowel(W->Letters[W->Start+1])) || W->StartsWith("par") || W->StartsWith("col") || W->StartsWith("tap") ))
      return W->Start+3;
    else {
      for (int i=W->Start+1;i<=W->End;i++) {
        if (IsVowel(W->Letters[i]))
          return i+1;
      }
    }
    return res;
  }
  bool Step1(Word *W, const U32 RV, const U32 R1, const U32 R2, bool *ForceStep2a) {
    int i = 0;
    for (; i<11; i++) {
      if (W->EndsWith(SuffixesStep1[i]) && SuffixInRn(W, R2, SuffixesStep1[i])) {
        W->End-=U8(strlen(SuffixesStep1[i]));
        if (i==3 /*able*/)
          W->Type|=French::Adjective;
        return true;
      }
    }
    for (; i<17; i++) {
      if (W->EndsWith(SuffixesStep1[i]) && SuffixInRn(W, R2, SuffixesStep1[i])) {
        W->End-=U8(strlen(SuffixesStep1[i]));
        if (W->EndsWith("ic"))
          W->ChangeSuffix("c", "qU");
        return true;
      }
    }
    for (; i<25;i++) {
      if (W->EndsWith(SuffixesStep1[i]) && SuffixInRn(W, R2, SuffixesStep1[i])) {
        W->End-=U8(strlen(SuffixesStep1[i]))-1-(i<19)*2;
        if (i>22) {
          W->End+=2;
          W->Letters[W->End]='t';
        }
        return true;
      }
    }
    for (; i<27; i++) {
      if (W->EndsWith(SuffixesStep1[i]) && SuffixInRn(W, R1, SuffixesStep1[i]) && IsConsonant((*W)((U8)strlen(SuffixesStep1[i])))) {
        W->End-=U8(strlen(SuffixesStep1[i]));
        return true;
      }
    }
    for (; i<29; i++) {
      if (W->EndsWith(SuffixesStep1[i]) && SuffixInRn(W, RV, SuffixesStep1[i])) {
        W->End-=U8(strlen(SuffixesStep1[i]));
        if (W->EndsWith("iv") && SuffixInRn(W, R2, "iv")) {
          W->End-=2;
          if (W->EndsWith("at") && SuffixInRn(W, R2, "at"))
            W->End-=2;
        }
        else if (W->EndsWith("eus")) {
          if (SuffixInRn(W, R2, "eus"))
            W->End-=3;
          else if (SuffixInRn(W, R1, "eus"))
            W->Letters[W->End]='x';
        }
        else if ((W->EndsWith("abl") && SuffixInRn(W, R2, "abl")) || (W->EndsWith("iqU") && SuffixInRn(W, R2, "iqU")))
          W->End-=3;
        else if ((W->EndsWith("i\xE8r") && SuffixInRn(W, RV, "i\xE8r")) || (W->EndsWith("I\xE8r") && SuffixInRn(W, RV, "I\xE8r"))) {
          W->End-=2;
          W->Letters[W->End]='i';
        }
        return true;
      }
    }
    for (; i<31; i++) {
      if (W->EndsWith(SuffixesStep1[i]) && SuffixInRn(W, R2, SuffixesStep1[i])) {
        W->End-=U8(strlen(SuffixesStep1[i]));
        if (W->EndsWith("abil")) {
          if (SuffixInRn(W, R2, "abil"))
            W->End-=4;
          else
            W->End--, W->Letters[W->End]='l';
        }
        else if (W->EndsWith("ic")) {
          if (SuffixInRn(W, R2, "ic"))
            W->End-=2;
          else
            W->ChangeSuffix("c", "qU");
        }
        else if (W->EndsWith("iv") && SuffixInRn(W, R2, "iv"))
          W->End-=2;
        return true;
      }
    }
    for (; i<35; i++) {
      if (W->EndsWith(SuffixesStep1[i]) && SuffixInRn(W, R2, SuffixesStep1[i])) {
        W->End-=U8(strlen(SuffixesStep1[i]));
        if (W->EndsWith("at") && SuffixInRn(W, R2, "at")) {
          W->End-=2;
          if (W->EndsWith("ic")) {
            if (SuffixInRn(W, R2, "ic"))
              W->End-=2;
            else
              W->ChangeSuffix("c", "qU");
          }
        }
        return true;
      }
    }
    for (; i<37; i++) {
      if (W->EndsWith(SuffixesStep1[i])) {
        if (SuffixInRn(W, R2, SuffixesStep1[i])) {
          W->End-=U8(strlen(SuffixesStep1[i]));
          return true;
        }
        else if (SuffixInRn(W, R1, SuffixesStep1[i])) {
          W->ChangeSuffix(SuffixesStep1[i], "eux");
          return true;
        }
      }
    }
    for (; i<NUM_SUFFIXES_STEP1; i++) {
      if (W->EndsWith(SuffixesStep1[i]) && SuffixInRn(W, RV+1, SuffixesStep1[i]) && IsVowel((*W)((U8)strlen(SuffixesStep1[i])))) {
        W->End-=U8(strlen(SuffixesStep1[i]));
        (*ForceStep2a) = true;
        return true;
      }
    }
    if (W->EndsWith("eaux") || (*W)=="eaux") {
      W->End--;
      W->Type|=French::Plural;
      return true;
    }
    else if (W->EndsWith("aux") && SuffixInRn(W, R1, "aux")) {
      W->End--, W->Letters[W->End] = 'l';
      W->Type|=French::Plural;
      return true;
    }
    else if (W->EndsWith("amment") && SuffixInRn(W, RV, "amment")) {
      W->ChangeSuffix("amment", "ant");
      (*ForceStep2a) = true;
      return true;
    }
    else if (W->EndsWith("emment") && SuffixInRn(W, RV, "emment")) {
      W->ChangeSuffix("emment", "ent");
      (*ForceStep2a) = true;
      return true;
    }
    return false;
  }
  bool Step2a(Word *W, const U32 RV) {
    for (int i=0; i<NUM_SUFFIXES_STEP2a; i++) {
      if (W->EndsWith(SuffixesStep2a[i]) && SuffixInRn(W, RV+1, SuffixesStep2a[i]) && IsConsonant((*W)((U8)strlen(SuffixesStep2a[i])))) {
        W->End-=U8(strlen(SuffixesStep2a[i]));
        if (i==31 /*ir*/)
          W->Type|=French::Verb;
        return true;
      }
    }
    return false;
  }
  bool Step2b(Word *W, const U32 RV, const U32 R2) {
    for (int i=0; i<NUM_SUFFIXES_STEP2b; i++) {
      if (W->EndsWith(SuffixesStep2b[i]) && SuffixInRn(W, RV, SuffixesStep2b[i])) {
        switch (SuffixesStep2b[i][0]) {
          case 'a': case '\xE2': {
            W->End-=U8(strlen(SuffixesStep2b[i]));
            if (W->EndsWith("e") && SuffixInRn(W, RV, "e"))
              W->End--;
            return true;
          }
          default: {
            if (i!=14 || SuffixInRn(W, R2, SuffixesStep2b[i])) {
              W->End-=U8(strlen(SuffixesStep2b[i]));
              return true;
            }
          }
        }        
      }
    }
    return false;
  }
  void Step3(Word *W) {
    char *final = (char *)&W->Letters[W->End];
    if ((*final)=='Y')
      (*final) = 'i';
    else if ((*final)=='\xE7')
      (*final) = 'c';
  }
  bool Step4(Word *W, const U32 RV, const U32 R2) {
    bool res = false;
    if (W->Length()>=2 && W->Letters[W->End]=='s' && !CharInArray((*W)(1), SetStep4, NUM_SET_STEP4)) {
      W->End--;
      res = true;
    }
    for (int i=0; i<NUM_SUFFIXES_STEP4; i++) {
      if (W->EndsWith(SuffixesStep4[i]) && SuffixInRn(W, RV, SuffixesStep4[i])) {
        switch (i) {
          case 2: { //ion
            char prec = (*W)(3);
            if (SuffixInRn(W, R2, SuffixesStep4[i]) && SuffixInRn(W, RV+1, SuffixesStep4[i]) && (prec=='s' || prec=='t')) {
              W->End-=3;
              return true;
            }
            break;
          }
          case 5: { //e
            W->End--;
            return true;
          }
          case 6: { //\xEB
            if (W->EndsWith("gu\xEB")) {
              W->End--;
              return true;
            }
            break;
          }
          default: {
            W->ChangeSuffix(SuffixesStep4[i], "i");
            return true;
          }
        }
      }
    }
    return res;
  }
  bool Step5(Word *W) {
    for (int i=0; i<NUM_SUFFIXES_STEP5; i++) {
      if (W->EndsWith(SuffixesStep5[i])) {
        W->End--;
        return true;
      }
    }
    return false;
  }
  bool Step6(Word *W) {
    for (int i=W->End; i>=W->Start; i--) {
      if (IsVowel(W->Letters[i])) {
        if (i<W->End && (W->Letters[i]&0xFE)==0xE8) {
          W->Letters[i] = 'e';
          return true;
        }
        return false;
      }
    }
    return false;
  }
public:
  inline bool IsVowel(const char c) final {
    return CharInArray(c, Vowels, NUM_VOWELS);
  }
  inline void Hash(Word *W) final {
    W->Hash[2] = W->Hash[3] = ~0xeff1cace;
    for (int i=W->Start; i<=W->End; i++) {
      U8 l = W->Letters[i];
      W->Hash[2]=W->Hash[2]*251*32 + l;
      if (IsVowel(l))
        W->Hash[3]=W->Hash[3]*997*16 + l;
      else if (l>='b' && l<='z')
        W->Hash[3]=W->Hash[3]*271*32 + (l-97);
      else
        W->Hash[3]=W->Hash[3]*11*32 + l;
    }
  }
  bool Stem(Word *W) {
    ConvertUTF8(W);
    if (W->Length()<2) {
      Hash(W);
      return false;
    }
    for (int i=0; i<NUM_EXCEPTIONS; i++) {
      if ((*W)==Exceptions[i][0]) {
        size_t len=strlen(Exceptions[i][1]);
        memcpy(&W->Letters[W->Start], Exceptions[i][1], len);
        W->End=W->Start+U8(len-1);
        Hash(W);
        W->Type|=TypesExceptions[i];
        W->Language = Language::French;
        return true;
      }
    }
    MarkVowelsAsConsonants(W);
    U32 RV=GetRV(W), R1=GetRegion(W, 0), R2=GetRegion(W, R1);
    bool DoNextStep=false, res=Step1(W, RV, R1, R2, &DoNextStep);
    DoNextStep|=!res;
    if (DoNextStep) {
      DoNextStep = !Step2a(W, RV);
      res|=!DoNextStep;
      if (DoNextStep)
        res|=Step2b(W, RV, R2);
    }
    if (res)
      Step3(W);
    else
      res|=Step4(W, RV, R2);
    res|=Step5(W);
    res|=Step6(W);
    for (int i=W->Start; i<=W->End; i++)
      W->Letters[i] = tolower(W->Letters[i]);
    if (!res)
      res=W->MatchesAny(CommonWords, NUM_COMMON_WORDS);
    Hash(W);
    if (res)
      W->Language = Language::French;
    return res;
  }
};

/*
  German suffix stemmer, based on the Porter stemmer.

  Changelog:
  (27/02/2018) v140: Initial release by Márcio Pais
*/

class GermanStemmer : public Stemmer {
private:
  static const int NUM_VOWELS = 9;
  const char Vowels[NUM_VOWELS]={'a','e','i','o','u','y','\xE4','\xF6','\xFC'};
  static const int NUM_COMMON_WORDS = 10;
  const char *CommonWords[NUM_COMMON_WORDS]={"der","die","das","und","sie","ich","mit","sich","auf","nicht"};
  static const int NUM_ENDINGS = 10;
  const char Endings[NUM_ENDINGS]={'b','d','f','g','h','k','l','m','n','t'}; //plus 'r' for words ending in 's'
  static const int NUM_SUFFIXES_STEP1 = 6;
  const char *SuffixesStep1[NUM_SUFFIXES_STEP1]={"em","ern","er","e","en","es"};
  static const int NUM_SUFFIXES_STEP2 = 3;
  const char *SuffixesStep2[NUM_SUFFIXES_STEP2]={"en","er","est"};
  static const int NUM_SUFFIXES_STEP3 = 7;
  const char *SuffixesStep3[NUM_SUFFIXES_STEP3]={"end","ung","ik","ig","isch","lich","heit"};
  void ConvertUTF8(Word *W) {
    for (int i=W->Start; i<W->End; i++) {
      U8 c = W->Letters[i+1]+((W->Letters[i+1]<0x9F)?0x60:0x40);
      if (W->Letters[i]==0xC3 && (IsVowel(c) || c==0xDF)) {
        W->Letters[i] = c;
        if (i+1<W->End)
          memcpy(&W->Letters[i+1], &W->Letters[i+2], W->End-i-1);
        W->End--;
      }
    }
  }
  void ReplaceSharpS(Word *W) {
    for (int i=W->Start; i<=W->End; i++) {
      if (W->Letters[i]==0xDF) {
        W->Letters[i]='s';
        if (i+1<MAX_WORD_SIZE) {
          memmove(&W->Letters[i+2], &W->Letters[i+1], MAX_WORD_SIZE-i-2);
          W->Letters[i+1]='s';
          W->End+=(W->End<MAX_WORD_SIZE-1);
        }
      }
    }
  }    
  void MarkVowelsAsConsonants(Word *W) {
    for (int i=W->Start+1; i<W->End; i++) {
      U8 c = W->Letters[i];
      if ((c=='u' || c=='y') && IsVowel(W->Letters[i-1]) && IsVowel(W->Letters[i+1]))
        W->Letters[i] = toupper(c);
    }
  }
  inline bool IsValidEnding(const char c, const bool IncludeR = false) {
    return CharInArray(c, Endings, NUM_ENDINGS) || (IncludeR && c=='r');
  }
  bool Step1(Word *W, const U32 R1) {
    int i = 0;
    for (; i<3; i++) {
      if (W->EndsWith(SuffixesStep1[i]) && SuffixInRn(W, R1, SuffixesStep1[i])) {
        W->End-=U8(strlen(SuffixesStep1[i]));
        return true;
      }
    }
    for (; i<NUM_SUFFIXES_STEP1; i++) {
      if (W->EndsWith(SuffixesStep1[i]) && SuffixInRn(W, R1, SuffixesStep1[i])) {
        W->End-=U8(strlen(SuffixesStep1[i]));
        W->End-=U8(W->EndsWith("niss"));
        return true;
      }
    }
    if (W->EndsWith("s") && SuffixInRn(W, R1, "s") && IsValidEnding((*W)(1), true)) {
      W->End--;
      return true;
    }
    return false;
  }
  bool Step2(Word *W, const U32 R1) {
    for (int i=0; i<NUM_SUFFIXES_STEP2; i++) {
      if (W->EndsWith(SuffixesStep2[i]) && SuffixInRn(W, R1, SuffixesStep2[i])) {
        W->End-=U8(strlen(SuffixesStep2[i]));
        return true;
      }
    }
    if (W->EndsWith("st") && SuffixInRn(W, R1, "st") && W->Length()>5 && IsValidEnding((*W)(2))) {
      W->End-=2;
      return true;
    }
    return false;
  }
  bool Step3(Word *W, const U32 R1, const U32 R2) {
    int i = 0;
    for (; i<2; i++) {
      if (W->EndsWith(SuffixesStep3[i]) && SuffixInRn(W, R2, SuffixesStep3[i])) {
        W->End-=U8(strlen(SuffixesStep3[i]));
        if (W->EndsWith("ig") && (*W)(2)!='e' && SuffixInRn(W, R2, "ig"))
          W->End-=2;
        if (i)
          W->Type|=German::Noun;
        return true;
      }
    }
    for (; i<5; i++) {
      if (W->EndsWith(SuffixesStep3[i]) && SuffixInRn(W, R2, SuffixesStep3[i]) && (*W)((U8)strlen(SuffixesStep3[i]))!='e') {
        W->End-=U8(strlen(SuffixesStep3[i]));
        if (i>2)
          W->Type|=German::Adjective;
        return true;
      }
    }
    for (; i<NUM_SUFFIXES_STEP3; i++) {
      if (W->EndsWith(SuffixesStep3[i]) && SuffixInRn(W, R2, SuffixesStep3[i])) {
        W->End-=U8(strlen(SuffixesStep3[i]));
        if ((W->EndsWith("er") || W->EndsWith("en")) && SuffixInRn(W, R1, "e?"))
          W->End-=2;
        if (i>5)
          W->Type|=German::Noun|German::Female;
        return true;
      }
    }
    if (W->EndsWith("keit") && SuffixInRn(W, R2, "keit")) {
      W->End-=4;
      if (W->EndsWith("lich") && SuffixInRn(W, R2, "lich"))
        W->End-=4;
      else if (W->EndsWith("ig") && SuffixInRn(W, R2, "ig"))
        W->End-=2;
      W->Type|=German::Noun|German::Female;
      return true;
    }
    return false;
  }
public:
  inline bool IsVowel(const char c) final {
    return CharInArray(c, Vowels, NUM_VOWELS);
  }
  inline void Hash(Word *W) final {
    W->Hash[2] = W->Hash[3] = ~0xbea7ab1e;
    for (int i=W->Start; i<=W->End; i++) {
      U8 l = W->Letters[i];
      W->Hash[2]=W->Hash[2]*263*32 + l;
      if (IsVowel(l))
        W->Hash[3]=W->Hash[3]*997*16 + l;
      else if (l>='b' && l<='z')
        W->Hash[3]=W->Hash[3]*251*32 + (l-97);
      else
        W->Hash[3]=W->Hash[3]*11*32 + l;
    }
  }
  bool Stem(Word *W) {
    ConvertUTF8(W);
    if (W->Length()<2) {
      Hash(W);
      return false;
    }
    ReplaceSharpS(W);
    MarkVowelsAsConsonants(W);
    U32 R1=GetRegion(W, 0), R2=GetRegion(W, R1);
    R1 = min(3, R1);
    bool res = Step1(W, R1);
    res|=Step2(W, R1);
    res|=Step3(W, R1, R2);
    for (int i=W->Start; i<=W->End; i++) {
      switch (W->Letters[i]) {
        case 0xE4: { W->Letters[i] = 'a'; break; }
        case 0xF6: case 0xFC: { W->Letters[i]-=0x87; break; }
        default: W->Letters[i] = tolower(W->Letters[i]);
      }
    }
    if (!res)
      res=W->MatchesAny(CommonWords, NUM_COMMON_WORDS);
    Hash(W);
    if (res)
      W->Language = Language::German;
    return res;
  }
};

//////////////////////////// Models //////////////////////////////

// All of the models below take a Mixer as a parameter and write
// predictions to it.

//////////////////////////// TextModel ///////////////////////////

template <class T, const U32 Size> class Cache {
  static_assert(Size>1 && (Size&(Size-1))==0, "Cache size must be a power of 2 bigger than 1");
private:
  Array<T> Data;
  U32 Index;
public:
  explicit Cache() : Data(Size) { Index=0; }
  T& operator()(U32 i) {
    return Data[(Index-i)&(Size-1)];
  }
  void operator++(int) {
    Index++;
  }
  void operator--(int) {
    Index--;
  }
  T& Next() {
    Index++;
    Data[Index&(Size-1)] = T();
    return Data[Index&(Size-1)];
  }
};

/*
  Text model

  Changelog:
  (04/02/2018) v135: Initial release by Márcio Pais
  (11/02/2018) v136: Uses 16 contexts, sets 3 mixer contexts
  (15/02/2018) v138: Uses 21 contexts, sets 4 mixer contexts
  (25/02/2018) v139: Uses 26 contexts
  (27/02/2018) v140: Sets 6 mixer contexts
  (12/05/2018) v142: Sets 7 mixer contexts
  (02/12/2018) v172: Sets 8 mixer contexts
*/

const U8 AsciiGroupC0[254] ={
  0, 10,
  0, 1, 10, 10,
  0, 4, 2, 3, 10, 10, 10, 10,
  0, 0, 5, 4, 2, 2, 3, 3, 10, 10, 10, 10, 10, 10, 10, 10,
  0, 0, 0, 0, 5, 5, 9, 4, 2, 2, 2, 2, 3, 3, 3, 3, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
  0, 0, 0, 0, 0, 0, 0, 0, 5, 8, 8, 5, 9, 9, 6, 5, 2, 2, 2, 2, 2, 2, 2, 8, 3, 3, 3, 3, 3, 3, 3, 8, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 8, 8, 8, 8, 8, 5, 5, 9, 9, 9, 9, 9, 7, 8, 5, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 8, 8, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 8, 8, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10
};
const U8 AsciiGroup[128] = {
  0,  5,  5,  5,  5,  5,  5,  5,
  5,  5,  4,  5,  5,  4,  5,  5,
  5,  5,  5,  5,  5,  5,  5,  5,
  5,  5,  5,  5,  5,  5,  5,  5,
  6,  7,  8, 17, 17,  9, 17, 10,
  11, 12, 17, 17, 13, 14, 15, 16,
  1,  1,  1,  1,  1,  1,  1,  1,
  1,  1, 18, 19, 20, 23, 21, 22,
  23,  2,  2,  2,  2,  2,  2,  2,
  2,  2,  2,  2,  2,  2,  2,  2,
  2,  2,  2,  2,  2,  2,  2,  2,
  2,  2,  2, 24, 27, 25, 27, 26,
  27,  3,  3,  3,  3,  3,  3,  3,
  3,  3,  3,  3,  3,  3,  3,  3,
  3,  3,  3,  3,  3,  3,  3,  3,
  3,  3,  3, 28, 30, 29, 30, 30
};

class TextModel {
private:
  const U32 MIN_RECOGNIZED_WORDS = 4;
  ContextMap Map;
  Array<Stemmer*> Stemmers;
  Array<Language*> Languages;
  Cache<Word, 8> Words[Language::Count];
  Cache<Segment, 4> Segments;
  Cache<Sentence, 4> Sentences;
  Cache<Paragraph, 2> Paragraphs;
  Array<U32> WordPos;
  U32 BytePos[256];
  Word *cWord, *pWord; // current word, previous word
  Segment *cSegment; // current segment
  Sentence *cSentence; // current sentence
  Paragraph *cParagraph; // current paragraph
  enum Parse {
    Unknown,
    ReadingWord,
    PossibleHyphenation,
    WasAbbreviation,
    AfterComma,
    AfterQuote,
    AfterAbbreviation,
    ExpectDigit
  } State, pState;
  struct {
    U32 Count[Language::Count-1]; // number of recognized words of each language in the last 64 words seen
    U64 Mask[Language::Count-1];  // binary mask with the recognition status of the last 64 words for each language
    int Id;  // current language detected
    int pId; // detected language of the previous word
  } Lang;
  struct {
    U64 numbers[2];   // last 2 numbers seen
    U64 numHashes[2]; // hashes of the last 2 numbers seen
    U8  numLength[2]; // digit length of last 2 numbers seen
    U32 numMask;      // binary mask of the results of the arithmetic comparisons between the numbers seen
    U32 numDiff;      // log2 of the consecutive differences between the last 16 numbers seen, clipped to 2 bits per difference
    U32 lastUpper;    // distance to last uppercase letter
    U32 maskUpper;    // binary mask of uppercase letters seen (in the last 32 bytes)
    U32 lastLetter;   // distance to last letter
    U32 lastDigit;    // distance to last digit
    U32 lastPunct;    // distance to last punctuation character
    U32 lastNewLine;  // distance to last new line character
    U32 prevNewLine;  // distance to penultimate new line character
    U32 wordGap;      // distance between the last words
    U32 spaces;       // binary mask of whitespace characters seen (in the last 32 bytes)
    U32 spaceCount;   // count of whitespace characters seen (in the last 32 bytes)
    U32 commas;       // number of commas seen in this line (not this segment/sentence)
    U32 quoteLength;  // length (in words) of current quote
    U32 maskPunct;    // mask of relative position of last comma related to other punctuation
    U32 nestHash;     // hash representing current nesting state
    U32 lastNest;     // distance to last nesting character
    U64 asciiMask;
    U32 masks[5],
        wordLength[2];
    int UTF8Remaining;// remaining bytes for current UTF8-encoded Unicode code point (-1 if invalid byte found)
    U8 firstLetter;   // first letter of current word
    U8 firstChar;     // first character of current line
    U8 expectedDigit; // next expected digit of detected numerical sequence
    U8 prevPunct;     // most recent punctuation character seen
    Word TopicDescriptor; // last word before ':'
  } Info;
  U64 ParseCtx;
  void Update(Buf& buffer);
  void SetContexts(Buf& buffer);
public:
  TextModel(const U32 Size) : Map(Size, 33), Stemmers(Language::Count-1), Languages(Language::Count-1), WordPos(0x10000), State(Parse::Unknown), pState(State), Lang{ {0}, {0}, Language::Unknown, Language::Unknown }, Info{}, ParseCtx(0) {
    Stemmers[Language::English-1] = new EnglishStemmer();
    Stemmers[Language::French-1] = new FrenchStemmer();
    Stemmers[Language::German-1] = new GermanStemmer();
    Languages[Language::English-1] = new English();
    Languages[Language::French-1] = new French();
    Languages[Language::German-1] = new German();
    cWord = &Words[Lang.Id](0);
    pWord = &Words[Lang.Id](1);
    cSegment = &Segments(0);
    cSentence = &Sentences(0);
    cParagraph = &Paragraphs(0);
    memset(&BytePos[0], 0, 256*sizeof(U32));
  }
  ~TextModel() {
    for (int i=0; i<Language::Count-1; i++) {
      /*delete Stemmers[i];*/
      /*delete Languages[i];*/
    }
  }
  void Predict(Mixer& mixer, Buf& buffer) {
    if (bpos==0) {
      Update(buffer);
      SetContexts(buffer);
    }
    Map.mix(mixer);
    mixer.set(finalize64(hash((Lang.Id!=Language::Unknown)?1+Stemmers[Lang.Id-1]->IsVowel(buffer(1)):0, Info.masks[1]&0xFF, c0), 11), 2048);
    mixer.set(finalize64(hash(ilog2(Info.wordLength[0]+1), c0,
      (Info.lastDigit<Info.wordLength[0]+Info.wordGap)|
      ((Info.lastUpper<Info.lastLetter+Info.wordLength[1])<<1)|
      ((Info.lastPunct<Info.wordLength[0]+Info.wordGap)<<2)|
      ((Info.lastUpper<Info.wordLength[0])<<3)
    ), 11), 2048);
    mixer.set(finalize64(hash(Info.masks[1]&0x3FF, grp0, Info.lastUpper<Info.wordLength[0], Info.lastUpper<Info.lastLetter+Info.wordLength[1]), 12), 4096);
    mixer.set(finalize64(hash(Info.spaces&0x1FF, grp0,
      (Info.lastUpper<Info.wordLength[0])|
      ((Info.lastUpper<Info.lastLetter+Info.wordLength[1])<<1)|
      ((Info.lastPunct<Info.lastLetter)<<2)|
      ((Info.lastPunct<Info.wordLength[0]+Info.wordGap)<<3)|
      ((Info.lastPunct<Info.lastLetter+Info.wordLength[1]+Info.wordGap)<<4)
    ), 12), 4096);
    mixer.set(finalize64(hash(Info.firstLetter*(Info.wordLength[0]<4), min(6, Info.wordLength[0]), c0), 11), 2048);
    mixer.set(finalize64(hash((*pWord)[0], (*pWord)(0), min(4, Info.wordLength[0]), Info.lastPunct<Info.lastLetter), 11), 2048);
    mixer.set(finalize64(hash(min(4, Info.wordLength[0]), grp0,
      Info.lastUpper<Info.wordLength[0],
      (Info.nestHash>0)?Info.nestHash&0xFF:0x100|(Info.firstLetter*(Info.wordLength[0]>0 && Info.wordLength[0]<4))
    ), 12), 4096);
    mixer.set(finalize64(hash(grp0, Info.masks[4]&0x1F, (Info.masks[4]>>5)&0x1F), 13), 8192);
  }
};

void TextModel::Update(Buf& buffer) {
  Info.lastUpper  = min(0xFF, Info.lastUpper+1), Info.maskUpper<<=1;
  Info.lastLetter = min(0x1F, Info.lastLetter+1);
  Info.lastDigit  = min(0xFF, Info.lastDigit+1);
  Info.lastPunct  = min(0x3F, Info.lastPunct+1);
  Info.lastNewLine++, Info.prevNewLine++, Info.lastNest++;
  Info.spaceCount-=(Info.spaces>>31), Info.spaces<<=1;
  Info.masks[0]<<=2, Info.masks[1]<<=2, Info.masks[2]<<=4, Info.masks[3]<<=3;
  pState = State;  

  U8 c = buffer(1), pC=tolower(c), g = (c<0x80)?AsciiGroup[c]:31;
  if(!((g<=4) && g==(Info.asciiMask&0x1f))) //repetition is allowed for groups 0..4
      Info.asciiMask = ((Info.asciiMask<<5) | g)&((U64(1)<<60)-1); //keep last 12 groups (12x5=60 bits)
  Info.masks[4] = Info.asciiMask&((1<<30)-1);
  BytePos[c] = pos;
  if (c!=pC) {
    c = pC;
    Info.lastUpper = 0, Info.maskUpper|=1;
  }
  pC = buffer(2);
  ParseCtx = hash(State=Parse::Unknown, pWord->Hash[1], c, (ilog2(Info.lastNewLine)+1)*(Info.lastNewLine*3>Info.prevNewLine), Info.masks[1]&0xFC);

  if ((c>='a' && c<='z') || c=='\'' || c=='-' || c>0x7F) {    
    if (Info.wordLength[0]==0) {
      // check for hyphenation with "+"
      if (pC==NEW_LINE && ((Info.lastLetter==3 && buffer(3)=='+') || (Info.lastLetter==4 && buffer(3)==CARRIAGE_RETURN && buffer(4)=='+'))) {
        Info.wordLength[0] = Info.wordLength[1];
        for (int i=Language::Unknown; i<Language::Count; i++)
          Words[i]--;
        cWord = pWord, pWord = &Words[Lang.pId](1);
        *cWord = Word();
        for (U32 i=0; i<Info.wordLength[0]; i++)
          (*cWord)+=buffer(Info.wordLength[0]-i+Info.lastLetter);
        Info.wordLength[1] = (*pWord).Length();
        cSegment->WordCount--;
        cSentence->WordCount--;
      }
      else {
        Info.wordGap = Info.lastLetter;
        Info.firstLetter = c;
      }
    }
    Info.lastLetter = 0;
    Info.wordLength[0]++;
    Info.masks[0]+=(Lang.Id!=Language::Unknown)?1+Stemmers[Lang.Id-1]->IsVowel(c):1, Info.masks[1]++, Info.masks[3]+=Info.masks[0]&3;
    if (c=='\'') {
      Info.masks[2]+=12;
      if (Info.wordLength[0]==1) {
        if (Info.quoteLength==0 && pC==SPACE)
          Info.quoteLength = 1;
        else if (Info.quoteLength>0 && Info.lastPunct==1) {
          Info.quoteLength = 0;
          ParseCtx = hash(State=Parse::AfterQuote, pC);
        }
      }
    }
    (*cWord)+=c;
    cWord->GetHashes();
    ParseCtx = hash(State=Parse::ReadingWord, cWord->Hash[1]);
  }
  else {
    if (cWord->Length()>0) {
      if (Lang.Id!=Language::Unknown)
        memcpy(&Words[Language::Unknown](0), cWord, sizeof(Word));

      for (int i=Language::Count-1; i>Language::Unknown; i--) {
        Lang.Count[i-1]-=(Lang.Mask[i-1]>>63), Lang.Mask[i-1]<<=1;
        if (i!=Lang.Id)
          memcpy(&Words[i](0), cWord, sizeof(Word));
        if (Stemmers[i-1]->Stem(&Words[i](0)))
          Lang.Count[i-1]++, Lang.Mask[i-1]|=1;
      }      
      Lang.Id = Language::Unknown;
      U32 best = MIN_RECOGNIZED_WORDS;
      for (int i=Language::Count-1; i>Language::Unknown; i--) {
        if (Lang.Count[i-1]>=best) {
          best = Lang.Count[i-1] + (i==Lang.pId); //bias to prefer the previously detected language
          Lang.Id = i;
        }
        Words[i]++;
      }
      Words[Language::Unknown]++;
      Lang.pId = Lang.Id;
      pWord = &Words[Lang.Id](1), cWord = &Words[Lang.Id](0);
      *cWord = Word();
      WordPos[pWord->Hash[1]&(WordPos.size()-1)] = pos;
      if (cSegment->WordCount==0)
        memcpy(&cSegment->FirstWord, pWord, sizeof(Word));
      cSegment->WordCount++;
      if (cSentence->WordCount==0)
        memcpy(&cSentence->FirstWord, pWord, sizeof(Word));
      cSentence->WordCount++;
      Info.wordLength[1] = Info.wordLength[0], Info.wordLength[0] = 0;
      Info.quoteLength+=(Info.quoteLength>0);
      if (Info.quoteLength>0x1F)
        Info.quoteLength = 0;
      cSentence->VerbIndex++, cSentence->NounIndex++, cSentence->CapitalIndex++;
      if ((pWord->Type&Language::Verb)!=0) {
        cSentence->VerbIndex = 0;
        memcpy(&cSentence->lastVerb, pWord, sizeof(Word));
      }
      if ((pWord->Type&Language::Noun)!=0) {
        cSentence->NounIndex = 0;
        memcpy(&cSentence->lastNoun, pWord, sizeof(Word));
      }
      if (cSentence->WordCount>1 && Info.lastUpper<Info.wordLength[1]) {
        cSentence->CapitalIndex = 0;
        memcpy(&cSentence->lastCapital, pWord, sizeof(Word));
      }
    }
    bool skip = false;
    switch (c) {
      case '.': {
        if (Lang.Id!=Language::Unknown && Info.lastUpper==Info.wordLength[1] && Languages[Lang.Id-1]->IsAbbreviation(pWord)) {
          ParseCtx = hash(State=Parse::WasAbbreviation, pWord->Hash[1]);
          break;
        }
      }
      case '?': case '!': {
        cSentence->Type = (c=='.')?Sentence::Types::Declarative:(c=='?')?Sentence::Types::Interrogative:Sentence::Types::Exclamative;
        cSentence->SegmentCount++;
        cParagraph->SentenceCount++;
        cParagraph->TypeCount[cSentence->Type]++;
        cParagraph->TypeMask<<=2, cParagraph->TypeMask|=cSentence->Type;
        cSentence = &Sentences.Next();
        Info.masks[3]+=3;
        skip = true;
      }
      case ',': case ';': case ':': {
        if (c==',') {
          Info.commas++;
          ParseCtx = hash(State=Parse::AfterComma, ilog2(Info.quoteLength+1), ilog2(Info.lastNewLine), Info.lastUpper<Info.lastLetter+Info.wordLength[1]);
        }
        else if (c==':')
          memcpy(&Info.TopicDescriptor, pWord, sizeof(Word));
        if (!skip) {
          cSentence->SegmentCount++;
          Info.masks[3]+=4;
        }
        Info.lastPunct = 0, Info.prevPunct = c;
        Info.masks[0]+=3, Info.masks[1]+=2, Info.masks[2]+=15;
        cSegment = &Segments.Next();
        break;
      }
      case NEW_LINE: {
        Info.prevNewLine = Info.lastNewLine, Info.lastNewLine = 0;
        Info.commas = 0;
        if (Info.prevNewLine==1 || (Info.prevNewLine==2 && pC==CARRIAGE_RETURN))
          cParagraph = &Paragraphs.Next();
        else if ((Info.lastLetter==2 && pC=='+') || (Info.lastLetter==3 && pC==CARRIAGE_RETURN && buffer(3)=='+'))
          ParseCtx = hash(Parse::ReadingWord, pWord->Hash[1]), State=Parse::PossibleHyphenation;
      }
      case TAB: case CARRIAGE_RETURN: case SPACE: {
        Info.spaceCount++, Info.spaces|=1;
        Info.masks[1]+=3, Info.masks[3]+=5;
        if (c==SPACE && pState==Parse::WasAbbreviation) {
          ParseCtx = hash(State=Parse::AfterAbbreviation, pWord->Hash[1]);
        }
        break;
      }
      case '(' : Info.masks[2]+=1; Info.masks[3]+=6; Info.nestHash+=31; Info.lastNest=0; break;
      case '[' : Info.masks[2]+=2; Info.nestHash+=11; Info.lastNest=0; break;
      case '{' : Info.masks[2]+=3; Info.nestHash+=17; Info.lastNest=0; break;
      case '<' : Info.masks[2]+=4; Info.nestHash+=23; Info.lastNest=0; break;
      case 0xAB: Info.masks[2]+=5; break;
      case ')' : Info.masks[2]+=6; Info.nestHash-=31; Info.lastNest=0; break;
      case ']' : Info.masks[2]+=7; Info.nestHash-=11; Info.lastNest=0; break;
      case '}' : Info.masks[2]+=8; Info.nestHash-=17; Info.lastNest=0; break;
      case '>' : Info.masks[2]+=9; Info.nestHash-=23; Info.lastNest=0; break;
      case 0xBB: Info.masks[2]+=10; break;
      case '"': {
        Info.masks[2]+=11;
        // start/stop counting
        if (Info.quoteLength==0)
          Info.quoteLength = 1;
        else {
          Info.quoteLength = 0;
          ParseCtx = hash(State=Parse::AfterQuote, 0x100|pC);
        }
        break;
      }
      case '/' : case '-': case '+': case '*': case '=': case '%': Info.masks[2]+=13; break;
      case '\\': case '|': case '_': case '@': case '&': case '^': Info.masks[2]+=14; break;
    }
    if (c>='0' && c<='9') {
      Info.numbers[0] = Info.numbers[0]*10 + (c&0xF), Info.numLength[0] = min(19, Info.numLength[0]+1);
      Info.numHashes[0] = combine64(Info.numHashes[0], c);
      Info.expectedDigit = -1;
      if (Info.numLength[0]<Info.numLength[1] && (pState==Parse::ExpectDigit || ((Info.numDiff&3)==0 && Info.numLength[0]<=1))) {
        U64 ExpectedNum = Info.numbers[1]+(Info.numMask&3)-2, PlaceDivisor=1;
        for (int i=0; i<Info.numLength[1]-Info.numLength[0]; i++, PlaceDivisor*=10);
        if (ExpectedNum/PlaceDivisor==Info.numbers[0]) {
          PlaceDivisor/=10;
          Info.expectedDigit = (ExpectedNum/PlaceDivisor)%10;
          State = Parse::ExpectDigit;
        }
      }
      else {
        U8 d = buffer(Info.numLength[0]+2);
        if (Info.numLength[0]<3 && buffer(Info.numLength[0]+1)==',' && d>='0' && d<='9')
          State = Parse::ExpectDigit;
      }
      Info.lastDigit = 0;
      Info.masks[3]+=7;
    }
    else if (Info.numbers[0]>0) {
      Info.numMask<<=2, Info.numMask|=1+(Info.numbers[0]>=Info.numbers[1])+(Info.numbers[0]>Info.numbers[1]);
      Info.numDiff<<=2, Info.numDiff|=min(3,ilog2(abs((int)(Info.numbers[0]-Info.numbers[1]))));
      Info.numbers[1] = Info.numbers[0], Info.numbers[0] = 0;
      Info.numHashes[1] = Info.numHashes[0], Info.numHashes[0] = 0;
      Info.numLength[1] = Info.numLength[0], Info.numLength[0] = 0;
      cSegment->NumCount++, cSentence->NumCount++;
    }
  }
  if (Info.lastNewLine==1)
    Info.firstChar = (Lang.Id!=Language::Unknown)?c:min(c,96);
  if (Info.lastNest>512)
    Info.nestHash = 0;
  int leadingBitsSet = 0;
  while (((c>>(7-leadingBitsSet))&1)!=0) leadingBitsSet++;

  if (Info.UTF8Remaining>0 && leadingBitsSet==1)
    Info.UTF8Remaining--;
  else
    Info.UTF8Remaining = (leadingBitsSet!=1)?(c!=0xC0 && c!=0xC1 && c<0xF5)?(leadingBitsSet-(leadingBitsSet>0)):-1:0;
  Info.maskPunct = (BytePos[(unsigned char)',']>BytePos[(unsigned char)'.'])|((BytePos[(unsigned char)',']>BytePos[(unsigned char)'!'])<<1)|((BytePos[(unsigned char)',']>BytePos[(unsigned char)'?'])<<2)|((BytePos[(unsigned char)',']>BytePos[(unsigned char)':'])<<3)|((BytePos[(unsigned char)',']>BytePos[(unsigned char)';'])<<4);
}

void TextModel::SetContexts(Buf& buffer) {
  const U8 c = buffer(1), lc = tolower(c), m2 = Info.masks[2]&0xF, column = min(0xFF, Info.lastNewLine);;
  const U16 w = ((State==Parse::ReadingWord)?cWord->Hash[1]:pWord->Hash[1])&0xFFFF;
  const U32 h = ((State==Parse::ReadingWord)?cWord->Hash[1]:pWord->Hash[2])*271+c;
  U64 i = State<<6;

  Map.set(ParseCtx);
  Map.set(hash(i++, cWord->Hash[0], pWord->Hash[0],
    (Info.lastUpper<Info.wordLength[0])|
    ((Info.lastDigit<Info.wordLength[0]+Info.wordGap)<<1)
  )); 
  Map.set(hash(i++, cWord->Hash[1], Words[Lang.pId](2).Hash[1], min(10,ilog2((U32)Info.numbers[0])),
    (Info.lastUpper<Info.lastLetter+Info.wordLength[1])|
    ((Info.lastLetter>3)<<1)|
    ((Info.lastLetter>0 && Info.wordLength[1]<3)<<2)
  ));
  Map.set(hash(i++, cWord->Hash[1]&0xFFF, Info.masks[1]&0x3FF, Words[Lang.pId](3).Hash[2],
    (Info.lastDigit<Info.wordLength[0]+Info.wordGap)|
    ((Info.lastUpper<Info.lastLetter+Info.wordLength[1])<<1)|
    ((Info.spaces&0x7F)<<2)
  ));
  Map.set(hash(i++, cWord->Hash[1], pWord->Hash[3], Words[Lang.pId](2).Hash[3]));
  Map.set(hash(i++, h&0x7FFF, Words[Lang.pId](2).Hash[1]&0xFFF, Words[Lang.pId](3).Hash[1]&0xFFF));
  Map.set(hash(i++, cWord->Hash[1], c, (cSentence->VerbIndex<cSentence->WordCount)?cSentence->lastVerb.Hash[1]:0));
  Map.set(hash(i++, pWord->Hash[2], Info.masks[1]&0xFC, lc, Info.wordGap));
  Map.set(hash(i++, (Info.lastLetter==0)?cWord->Hash[1]:pWord->Hash[1], c, cSegment->FirstWord.Hash[2], min(3,ilog2(cSegment->WordCount+1))));
  Map.set(hash(i++, cWord->Hash[1], c, Segments(1).FirstWord.Hash[3]));
  Map.set(hash(i++, max(31,lc), Info.masks[1]&0xFFC, (Info.spaces&0xFE)|(Info.lastPunct<Info.lastLetter), (Info.maskUpper&0xFF)|(((0x100|Info.firstLetter)*(Info.wordLength[0]>1))<<8)));
  Map.set(hash(i++, column, min(7,ilog2(Info.lastUpper+1)), ilog2(Info.lastPunct+1)));
  Map.set(
    (column&0xF8)|(Info.masks[1]&3)|((Info.prevNewLine-Info.lastNewLine>63)<<2)|
    (min(3, Info.lastLetter)<<8)|
    (Info.firstChar<<10)|
    ((Info.commas>4)<<18)|
    ((m2>=1 && m2<=5)<<19)|
    ((m2>=6 && m2<=10)<<20)|
    ((m2==11 || m2==12)<<21)|
    ((Info.lastUpper<column)<<22)|
    ((Info.lastDigit<column)<<23)|
    ((column<Info.prevNewLine-Info.lastNewLine)<<24)
  );
  Map.set(hash(
    (2*column)/3,
    min(13, Info.lastPunct)+(Info.lastPunct>16)+(Info.lastPunct>32)+Info.maskPunct*16,
    ilog2(Info.lastUpper+1),
    ilog2(Info.prevNewLine-Info.lastNewLine),
    ((Info.masks[1]&3)==0)|
    ((m2<6)<<1)|
    ((m2<11)<<2)
  ));
  Map.set(hash(i++, column>>1, Info.spaces&0xF));
  Map.set(hash(
    Info.masks[3]&0x3F,
    min((max(Info.wordLength[0],3)-2)*(Info.wordLength[0]<8),3),
    Info.firstLetter*(Info.wordLength[0]<5),
    w&0x3FF,
    (c==buffer(2))|
    ((Info.masks[2]>0)<<1)|
    ((Info.lastPunct<Info.wordLength[0]+Info.wordGap)<<2)|
    ((Info.lastUpper<Info.wordLength[0])<<3)|
    ((Info.lastDigit<Info.wordLength[0]+Info.wordGap)<<4)|
    ((Info.lastPunct<2+Info.wordLength[0]+Info.wordGap+Info.wordLength[1])<<5)
  ));
  Map.set(hash(i++, w, c, Info.numHashes[1]));
  Map.set(hash(i++, w, c, llog(pos-WordPos[w])>>1));
  Map.set(hash(i++, w, c, Info.TopicDescriptor.Hash[1]&0x7FFF));
  Map.set(hash(i++, Info.numLength[0], c, Info.TopicDescriptor.Hash[1]&0x7FFF));
  Map.set(hash(i++, (Info.lastLetter>0)?c:0x100, Info.masks[1]&0xFFC, Info.nestHash&0x7FF));
  Map.set(hash(i++, w*17+c, Info.masks[3]&0x1FF,
    ((cSentence->VerbIndex==0 && cSentence->lastVerb.Length()>0)<<6)|
    ((Info.wordLength[1]>3)<<5)|
    ((cSegment->WordCount==0)<<4)|
    ((cSentence->SegmentCount==0 && cSentence->WordCount<2)<<3)|
    ((Info.lastPunct>=Info.lastLetter+Info.wordLength[1]+Info.wordGap)<<2)|
    ((Info.lastUpper<Info.lastLetter+Info.wordLength[1])<<1)|
    (Info.lastUpper<Info.wordLength[0]+Info.wordGap+Info.wordLength[1])
  ));
  Map.set(hash(i++, c, pWord->Hash[2], Info.firstLetter*(Info.wordLength[0]<6),
    ((Info.lastPunct<Info.wordLength[0]+Info.wordGap)<<1)|
    (Info.lastPunct>=Info.lastLetter+Info.wordLength[1]+Info.wordGap)
  ));
  Map.set(hash(i++, w*23+c, Words[Lang.pId](1+(Info.wordLength[0]==0)).Letters[Words[Lang.pId](1+(Info.wordLength[0]==0)).Start], Info.firstLetter*(Info.wordLength[0]<7)));
  Map.set(hash(i++, column, Info.spaces&7, Info.nestHash&0x7FF));
  Map.set(hash(i++, cWord->Hash[1], (Info.lastUpper<column)|((Info.lastUpper<Info.wordLength[0])<<1), min(5, Info.wordLength[0])));
  Map.set(Info.masks[4]); //last 6 groups
  Map.set(hash(U32(Info.asciiMask),U32(Info.asciiMask>>32))); //last 12 groups
  Map.set(Info.asciiMask & ((1<<20)-1)); //last 4 groups
  Map.set(Info.asciiMask & ((1<<10)-1)); //last 2 groups
  Map.set(hash((Info.asciiMask>>5) &((1<<30)-1),buffer(1)));
  Map.set(hash((Info.asciiMask>>10)&((1<<30)-1),buffer(1),buffer(2)));
  Map.set(hash((Info.asciiMask>>15)&((1<<30)-1),buffer(1),buffer(2),buffer(3)));
}

// Main model - predicts next bit probability from previous data
static int predictNext() {
  static ContextMap cm(MEM * 32, 22);
  static Mixer mixer(1500, 40160, 100);
  static APM a1(0x100), a2(0x10000), a3(0x10000);
  static U32 cxt[15], t1[0x100];
  static lstmModel1 lstm;
  static nestModel1 nest;
  static dmcModel1 dmc;
  static TextModel txt(MEM * 8);
  static U16 t2[0x10000];
  static U32 mask = 0, mask2 = 0, word0 = 0, word1 = 0;

  c0 += c0 + y;
  if (c0 >= 256) {
    buf[pos++] = c0;
    c4 = (c4 << 8) + c0 - 256;
    c0 = 1;
  }   
  grp0 = (bpos>0)?AsciiGroupC0[(1<<bpos)-2+(c0&((1<<bpos)-1))]:0;
  bpos = (bpos + 1) & 7;  
  int c1 = c4 & 0xff, c2 = (c4 & 0xff00) >> 8;

  mixer.update();
  int ismatch = ilog(matchModel(mixer));
  ppmdModel(mixer);
  lstm.p(mixer);
  nest.p(mixer);
  dmc.p(mixer);
  txt.Predict(mixer, buf);

  if (bpos == 0) {
    for (int i = 14; i > 0; --i) {
      cxt[i] = hash(cxt[i - 1], c1);
    }
    cm.set(0);
    cm.set(c1);
    cm.set(c4 & 0x0000ffff);
    cm.set(c4 & 0x00ffffff);
    cm.set(c4);
    cm.set(cxt[5]);
    cm.set(cxt[6]);
    cm.set(cxt[14]);
    cm.set(ismatch | (c4 & 0xffff0000));
    cm.set(ismatch | (c4 & 0x0000ff00));
    cm.set(ismatch | (c4 & 0x00ff0000));
    mask = (mask << 3) | (!c1 ? 0 : isalpha(c1) ? 1 : ispunct(c1) ? 2 :
      isspace(c1) ? 3 : (c1 == 255) ? 4 : (c1 < 16) ? 5 : (c1 < 64) ? 6 : 7);
    cm.set(mask);
    mask2 = (mask2 << 3) | ((mask >> 27) & 7);
    cm.set(hash(mask << 5, mask2 << 2));
    U32& ic1r = t1[c2];
    ic1r = ic1r << 8 | c1;
    U16& ic2r = t2[(buf(3) << 8) | c2];
    ic2r = ic2r << 8 | c1;
    const U32 ic1 = c1 | t1[c1] << 8;
    const U32 ic2 = ((c2 << 8) | c1) | t2[(c2 << 8) | c1] << 16;
    cm.set((ic1 >> 8) & ((1 << 16) - 1));
    cm.set((ic2 >> 16) & ((1 << 8) - 1));
    cm.set(ic1 & ((1 << 16) - 1));
    cm.set(ic2 & ((1 << 24) - 1));  
    int c = (c1 >= 'A' && c1 <= 'Z') ? c1 + 'a' - 'A' : c1;
    if ((c >= 'a' && c <= 'z') || c >= 128) word0 = hash(word0, c);
    else if (word0) word1 = word0, word0 = 0;
    cm.set(word0);
    cm.set(hash(word0, word1));
  }
  
  int o = cm.mix(mixer);
  mixer.set(c1 + 8, 264);
  mixer.set(c0, 256);
  mixer.set(o + ((c1 > 32) << 4) + ((bpos == 0) << 5) + ((c1 == c2) << 6), 128);
  mixer.set(c2, 256);
  mixer.set(ismatch, 256);  
     
  int pr0 = mixer.p();
  pr0 = (a1.p(pr0, c0) * 5
         + a2.p(pr0, c0 + (c1 << 8)) * 15
         + a3.p(pr0, hash(bpos, c1, c2) & 0xffff) * 12
         + 16) >> 5; // Probability adjusted with 3 APMs                       
  return pr0;
}

int pr(int yv) {
  y=yv;  return predictNext();
}

int load_prompt(const char * prompt, int len) {
  int p = 0;
  for (int i = 0; i < len; i++)
    for (int j = 7; j >= 0; j--)
      p = pr((prompt[i] >> j) & 1);
  return p;
}

int sample(int p) {
  if (p < 32) return 0; else if (p > 4050) return 1;
  float r = (float)rand() / RAND_MAX;
  return r < (float)p / 4096;
}

void generate(const char * prompt, int len) {
  char response[128];
  int p = load_prompt(prompt, len);
  printf("User says: %s\n", prompt);
  for (int i = 0; i < 128; i++) {
    response[i] = 0;
    for (int j = 7; j >= 0; j--) {
      int bit = sample(p);
      response[i] |= bit << j;
      p = pr(bit);
    }
  }
  response[128] = 0;
  printf("PAQ says: %s\n", response);
}

#include <sys/stat.h>

size_t file_size(const char * filename) {
  struct stat st;
  stat(filename, &st);
  return st.st_size;
}

int main() {
  FILE * training_data = fopen("result10.txt", "r");
  size_t max = file_size("result10.txt"), progress = 0;
  while (!feof(training_data)) {
    int byte = fgetc(training_data); if (byte == EOF) break;
    for (int i = 7; i >= 0; i--)
      pr((byte >> i) & 1);
    progress++;
    if (progress % 4096 == 0)
      { printf("Training: %.2f%%\n", 100.0 * progress / max); }
  }
  printf("\n");
  fclose(training_data);
  printf("Training complete\n");
  char prompt[128] = { 0 }; fgets(prompt, 128, stdin);
  int len = strlen(prompt);
  prompt[len - 1] = 0;
  generate(prompt, len - 1);
  return 0;
}
