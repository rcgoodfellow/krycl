
/******************************************************************************
 * Sparse Matrix Dense Vector Multiplier Kernel
 *
 * ~ ry
 * ***************************************************************************/

void print_vec_d(global double *v, uint n)
{
    for(uint i=0; i<(n-1); ++i)
    {
        printf("%f,", v[i]);
    }
    printf("%f\n", v[n-1]);
}

void print_vec_u(global uint *v, uint n)
{
    for(uint i=0; i<(n-1); ++i)
    {
        printf("%u,", v[i]);
    }
    printf("%u\n", v[n-1]);
}

kernel
void mul_sp_dv(
        global double *A, 
        global uint *c, 
        global uint *r,
        global double *v,
        global double *Av,
        uint n,
        uint N)
{
  int tid = get_global_id(0);
  if(tid > n) return;
  uint beg = r[tid],
       end = r[tid+1];

  if(tid == 0)
  {
      printf("N = %u, n = %u\n", N, n);
      printf("%s", "A: ");
      print_vec_d(A, N);
      printf("%s", "c: ");
      print_vec_u(c, N);
      printf("%s", "r: ");
      print_vec_u(r, n+1);
  }
  barrier(CLK_GLOBAL_MEM_FENCE);

  Av[tid] = 0;
  for(uint i=beg; i<end; ++i)
  {
    Av[tid] += A[i] * v[c[i]];
  }
  barrier(CLK_GLOBAL_MEM_FENCE);
  printf("[%d] %f\n", tid, Av[tid]);
}
