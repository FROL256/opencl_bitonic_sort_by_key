#include "bitonic_sort_cpu.h"



inline void bitonic_pass(ElemT* theArray, int a_N, int stage, int passOfStage, int a_invertModeOn)
{
  const int kernelSize = (a_N >> 1);

  for (int j = 0; j < kernelSize; j++)
  {
    const int r     = 1 << (passOfStage);
    const int lmask = r - 1;

    const int left  = ((j >> passOfStage) << (passOfStage+1)) + (j & lmask);
    const int right = left + r;

    const ElemT a = theArray[left];
    const ElemT b = theArray[right];

    const bool cmpRes = compare(a, b);

    const ElemT minElem = cmpRes ? a : b;
    const ElemT maxElem = cmpRes ? b : a;

    const int oddEven = j >> stage;

    if ((oddEven & 1) & a_invertModeOn)
    {
      theArray[right] = minElem;
      theArray[left]  = maxElem;
    }
    else
    {
      theArray[left]  = minElem;
      theArray[right] = maxElem;
    }

  }


}

/*

// not much faster on CPU

inline void bitonic_pass1(ElemT* theArray, int a_N, int stage, int passOfStage, int a_invertModeOn)
{
  const int kernelSize = (a_N >> 1);

  for (int j = 0; j < kernelSize; j++)
  {
    const int r = 1 << (passOfStage);
    const int lmask = r - 1;

    const int left = ((j >> passOfStage) << (passOfStage + 1)) + (j & lmask);
    const int right = left + r;

    const ElemT a = theArray[left];
    const ElemT b = theArray[right];

    const int oddEven = j >> stage;
    const bool less = compare(a, b) ^ ((oddEven & 1) & a_invertModeOn);

    if (!less)
    {
      theArray[left]  = b;
      theArray[right] = a;
    }

  }

}

*/

void bitonic_sort(ElemT* a_data, int a_N)
{
  int numStages = 0;
  for (int temp = a_N; temp > 2; temp >>= 1)
    numStages++;

  // up, form bitonic sequence with half allays
  //
  for (int stage = 0; stage < numStages; stage++)
  {
    for (int passOfStage = stage; passOfStage >= 0; passOfStage--)
      bitonic_pass(a_data, a_N, stage, passOfStage, 1);
  }

  // down, finally sort it
  //
  for (int passOfStage = numStages; passOfStage >= 0; passOfStage--)
    bitonic_pass(a_data, a_N, numStages-1, passOfStage, 0);

}
