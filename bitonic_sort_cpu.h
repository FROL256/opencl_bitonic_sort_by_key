#pragma once

struct int2
{
  int x, y;
};

typedef int2 ElemT;
typedef int  KeyT;
typedef int  ValT;

inline KeyT getKey(ElemT v) { return v.x; }
inline ValT getVal(ElemT v) { return v.y; }

inline bool compare(ElemT a, ElemT b) { return getKey(a) < getKey(b); }

struct MyCompare
{
  bool operator()(ElemT a, ElemT b) { return compare(a, b); }
};


void bitonic_sort(ElemT* a_data, int a_N);