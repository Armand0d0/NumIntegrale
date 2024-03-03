#if !defined(__CUDA__)
#define __CUDA__

class Cuda{

        
        public:
                Cuda();

                enum funcOP{
                        COMP,
                        MUL,
                        SUM,
                    };
                    #define COMP Cuda::COMP
                    #define MUL Cuda::MUL
                    #define SUM Cuda::SUM
                    
                    struct func{
                        funcOP opType;
                        struct func* func1;
                        struct func* func2;
                        int funcId1;
                        int funcId2;
                    
                    
                    }   ;


                //void printFunc(Cuda::func * f);
                //double eval(Cuda::func* fun, double x, double  (*F[])(double));
                double evalCu(Cuda::func* fun,double x,double  (*F[])(double) );
                double show2(func *fun[], int n,double* b, int nb,double  (*F[])(double), double goal, double dist_goal,int iter);
               double show3(Cuda::func* fun,double* b, int nb,double  (*FU[])(double) ,double goal,double dist_goal );

                        /*

                typedef struct
                {
                        int h;
                        int w;
                        float* arr;
                }Matrix;

                float SumArr(float* a,int N);
                 float*  normalize(unsigned char* data);
                 float* Square(float* a,int N);
                 float* MatrixVectorMul(Matrix,Matrix);
                 float* VecSum(float* a,float* b,int N);
                 float* VectorScalarMul(float* a,float b,int N);
                 float* Relu(float* in,int N);
                 float* SoftMax(float* a,int N);
                 float* computeLastLayerWeightsNegGradient(float* z,float* a,float* bias4,float* l,int K,int J);
                 float* computeLastLayerWeightsNegGradient2(float* z,float* a,float* bias4,float* l,int K,int J);
                 float* VecMul(float* a,float* b, int N);
                 void clean(float* f);
                 void clean(float** f,int n);*/

};


#endif

